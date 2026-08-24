"""계획은 되는데 base autonomy가 그 좌표를 하나도 안 받아줄 때의 라이브락 방지.

실측(2026-08-24): plan_route()가 2-hop route를 반환했지만 goal_publisher.publish()가
두 hop 모두 None을 반환(waypointConverter가 받아줄 지점이 근처에 없음)해서 state가
OBSERVE로 되돌아갔다. 로봇이 한 발짝도 안 움직이니 occupancy map이 그대로고, 지도가
그대로니 다음 사이클에 **완전히 동일한 route**가 다시 나왔다 - 1.2초 주기 무한루프.

plan_route()는 빈 route를 반환하지 않으므로(frontier도 있고 A*로 도달도 가능하다)
미션별 종료 처리(_on_exploration_result(route=[]))가 영원히 안 불린다. Mission 2는
MISSION2_EXPLORATION_TIME_LIMIT_SEC이 결국 구해주지만 Mission 1/3은 탈출구가 없다.

방어선이 둘이다:
  1. planner blacklist - 거부된 좌표를 mark_unpublishable()로 되먹여서 후보 풀에서 뺀다.
  2. streak 승격 - 그래도 안 풀리면 sysnav_node가 탐사 소진으로 승격한다(여기서는
     planner 쪽만 검증한다. node 쪽은 ROS 런타임이 필요하다).
"""

import unittest

from sysnav import config
from sysnav.exploration.coverage_planner import CoveragePlanner
from sysnav.exploration.viewpoint_memory import ViewpointMemory


def _open_room_planner():
    """정사각형 방 하나. 오른쪽 밖은 미탐사라 그쪽 경계가 frontier가 된다."""
    planner = CoveragePlanner()
    planner.origin_x = planner.origin_y = -6.0
    planner.grid[25:45, 25:45] = config.OCC_FREE
    planner.grid[24, 24:46] = config.OCC_OCCUPIED
    planner.grid[45, 24:46] = config.OCC_OCCUPIED
    planner.grid[24:46, 24] = config.OCC_OCCUPIED
    return planner


class UnpublishableBlacklistTest(unittest.TestCase):
    def setUp(self):
        self.planner = _open_room_planner()
        x, y = self.planner.grid_to_world(35, 30)
        self.pose = {"x": x, "y": y, "yaw": 0.0}

    def _route(self):
        return self.planner.plan_route(self.pose, ViewpointMemory())

    def test_route_exists_before_any_strike(self):
        self.assertTrue(self._route(), "방 오른쪽 frontier로 갈 경로가 있어야 한다")

    def test_one_strike_is_not_enough_to_drop_a_cell(self):
        """terrain_map은 롤링 윈도우라 지금 못 받는 지점도 가까이 가면 받아줄 수 있다.
        그래서 한 번 거부됐다고 바로 후보에서 빼지 않는다."""
        route = self._route()
        goal = route[-1]
        self.planner.mark_unpublishable(goal["x"], goal["y"])
        cell = self.planner.world_to_grid(goal["x"], goal["y"])
        self.assertEqual(self.planner._unpublishable_counts[cell], 1)
        self._route()
        self.assertEqual(self.planner.last_plan_diagnostics["rejected_by_unpublishable"], 0)

    def test_a_blacklisted_cell_is_never_targeted_again(self):
        """blacklist의 실제 계약: strike가 찬 셀은 다시 route의 목적지로 안 잡힌다.

        이게 없으면 planner는 A*로 멀쩡히 도달 가능한 같은 좌표를 영원히 다시 고른다
        (거부는 planner가 볼 수 없는 base autonomy 쪽 사정이라 스스로는 못 걸러낸다).
        """
        struck: set[tuple[int, int]] = set()
        for _ in range(15):
            route = self._route()
            if not route:
                break
            target = self.planner.world_to_grid(route[-1]["x"], route[-1]["y"])
            self.assertNotIn(
                target, struck,
                "발행 거부가 확정된 셀이 다시 목적지로 잡혔다 - 라이브락이 그대로다",
            )
            for hop in route:
                cell = self.planner.world_to_grid(hop["x"], hop["y"])
                for _ in range(config.EXPLORATION_UNPUBLISHABLE_MAX_STRIKES):
                    self.planner.mark_unpublishable(hop["x"], hop["y"])
                struck.add(cell)
        self.assertTrue(struck)
        self.assertGreater(
            self.planner.last_plan_diagnostics.get("rejected_by_unpublishable", 0), 0,
            "blacklist가 후보 풀에서 실제로 걸러내고 있어야 한다",
        )

    def test_pool_empties_when_the_reachable_area_is_small(self):
        """갈 수 있는 곳이 유한하면 blacklist만으로도 빈 route(=탐사 종료)에 수렴한다.

        넓은 공간에서는 매 사이클 새로 무작위 샘플링하므로 후보가 금방 안 마른다 -
        그 경우를 위해 sysnav_node 쪽 streak 승격(2번 방어선)이 따로 있다.
        """
        planner = CoveragePlanner()
        planner.origin_x = planner.origin_y = -6.0
        planner.grid[32:36, 30:34] = config.OCC_FREE       # 4x4 작은 방
        planner.grid[31, 29:35] = config.OCC_OCCUPIED
        planner.grid[36, 29:35] = config.OCC_OCCUPIED
        planner.grid[31:37, 29] = config.OCC_OCCUPIED
        x, y = planner.grid_to_world(33, 31)
        pose = {"x": x, "y": y, "yaw": 0.0}

        for _ in range(40):
            route = planner.plan_route(pose, ViewpointMemory())
            if not route:
                break
            for hop in route:
                for _ in range(config.EXPLORATION_UNPUBLISHABLE_MAX_STRIKES):
                    planner.mark_unpublishable(hop["x"], hop["y"])
        else:
            self.fail("발행 거부를 계속 되먹였는데도 route가 계속 나온다 - 라이브락")
        self.assertEqual(route, [])

    def test_strikes_are_cleared_on_reset(self):
        """새 질문마다 reset()이 불린다 - 지난 질문의 blacklist를 물려받으면 안 된다."""
        route = self._route()
        self.planner.mark_unpublishable(route[-1]["x"], route[-1]["y"])
        self.planner.reset({"x": 0.0, "y": 0.0})
        self.assertEqual(self.planner._unpublishable_counts, {})

    def test_out_of_map_coordinate_is_ignored(self):
        self.assertEqual(self.planner.mark_unpublishable(1e6, 1e6), 0)
        self.assertEqual(self.planner._unpublishable_counts, {})


class AnchorIsNotExemptTest(unittest.TestCase):
    """anchor의 is_near_visited 면제는 "아직 안 가본 frontier를 놓치지 않기" 위한
    것이지, base autonomy가 절대 못 받는 좌표를 계속 다시 고르기 위한 게 아니다.
    blacklist는 anchor에도 적용돼야 한다."""

    def test_blacklisted_anchor_is_dropped_from_the_pool(self):
        planner = _open_room_planner()
        x, y = planner.grid_to_world(35, 30)
        pose = {"x": x, "y": y, "yaw": 0.0}
        planner.plan_route(pose, ViewpointMemory())

        # 이번 사이클에 anchor로 잡힌 셀 전부에 strike를 채운다.
        anchors = list(planner._anchor_visit_counts.keys())
        self.assertTrue(anchors, "이 픽스처는 frontier anchor를 만들어야 한다")
        for cell in anchors:
            planner._unpublishable_counts[cell] = config.EXPLORATION_UNPUBLISHABLE_MAX_STRIKES

        planner.plan_route(pose, ViewpointMemory())
        diagnostics = planner.last_plan_diagnostics
        self.assertGreater(
            diagnostics.get("rejected_by_unpublishable", 0), 0,
            "blacklist가 anchor를 걸러내야 한다",
        )


if __name__ == "__main__":
    unittest.main()
