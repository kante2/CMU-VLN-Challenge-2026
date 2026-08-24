"""A committed Mission 3 marker must remain the active navigation target."""

import unittest
from threading import RLock

from sysnav import config
from sysnav.missions import mission3_pipe


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, _message):
        pass


class _TerrainMonitor:
    last_selection = "test"


class _Node:
    def __init__(self):
        self.state_lock = RLock()
        self.state = "MISSION3_NAVIGATE_STEP"
        self.mission3_step_index = 0
        self.mission3_subgoal_retries = 0
        self.target_goal_xy = (4.0, -2.0)
        self.target_final_theta = 0.7
        self.target_forbidden_mask = "mask"
        self.target_object_id = 12
        self.target_object_xy = (4.4, -2.1)
        self.target_marker_index = 0
        self.restarted = None
        self.marker_refreshes = 0
        self.navigation_clears = 0
        self.step_calls = 0
        self.terrain_monitor = _TerrainMonitor()
        self.approach_result = None          # None이면 완화 재시도도 실패한 것
        self.target_forbidden_mask = "mask"

    def step_target_navigation(self, _pose):
        self.step_calls += 1
        return "unreachable"

    def start_target_navigation(self, pose, goal_xy, final_theta, **kwargs):
        self.restarted = (pose, goal_xy, final_theta, kwargs)

    def refresh_goal_marker(self):
        self.marker_refreshes += 1

    def clear_target_navigation(self):
        self.navigation_clears += 1

    def approach_pose_for(self, _pose, _position, **_kwargs):
        if self.approach_result is None:
            raise AssertionError("완화 접근점이 없어야 하는 테스트에서 호출됨")
        return (*self.approach_result, 0.0)

    def get_logger(self):
        return _Logger()


class Mission3CommittedSubgoalTest(unittest.TestCase):
    def test_unreachable_subgoal_is_republished_without_returning_to_exploration(self):
        node = _Node()
        pose = {"x": 1.0, "y": 2.0, "yaw": 0.0}

        mission3_pipe._navigate_step(node, {"steps": [{}]}, pose)

        self.assertEqual(node.state, "MISSION3_NAVIGATE_STEP")
        self.assertEqual(node.restarted[1], (4.0, -2.0))
        self.assertEqual(node.restarted[2], 0.7)
        self.assertEqual(node.restarted[3]["forbidden_mask"], "mask")
        self.assertEqual(node.restarted[3]["object_id"], 12)
        self.assertEqual(node.restarted[3]["object_xy"], (4.4, -2.1))
        self.assertEqual(node.restarted[3]["marker_index"], 0)
        self.assertEqual(node.marker_refreshes, 1)

    def test_one_meter_radius_completes_only_mission3_step(self):
        node = _Node()
        node.target_goal_xy = (0.9, 0.0)
        task = {"steps": [{"is_stop": True}]}

        mission3_pipe._navigate_step(
            node, task, {"x": 0.0, "y": 0.0, "yaw": 0.0}
        )

        self.assertEqual(node.step_calls, 0)
        self.assertEqual(node.navigation_clears, 1)
        self.assertEqual(node.mission3_step_index, 1)
        self.assertEqual(node.state, "MISSION3_SELECT_STEP")


class Mission3SubgoalGiveUpTest(unittest.TestCase):
    """확정된 subgoal을 무한히 재발행하면 안 된다.

    실측 2026-08-24: 변기 접근점의 clearance가 0.42m라 base autonomy(기준 0.75m)가
    목적지 후보로 안 받았고, mission3는 "확정된 subgoal은 탐사로 덮지 않는다" 정책에
    상한이 없어서 "재발행 -> 거부 -> unreachable -> 재발행"을 0.5초 주기로 영원히 돌았다.
    한 step에 갇혀 남은 step을 통째로 버리는 것보다 넘어가는 쪽이 부분점수에서 낫다.
    """

    def _drive(self, node, times):
        for _ in range(times):
            mission3_pipe._navigate_step(node, {"steps": [{}, {}]}, {"x": 1.0, "y": 2.0, "yaw": 0.0})

    def test_republishes_up_to_the_limit_then_moves_on(self):
        node = _Node()
        node.target_object_xy = None          # 완화 재시도 대상 없음 -> 바로 포기
        self._drive(node, config.MISSION3_SUBGOAL_MAX_RETRIES - 1)
        self.assertEqual(node.mission3_step_index, 0, "상한 전에는 계속 재발행한다")
        self.assertEqual(node.state, "MISSION3_NAVIGATE_STEP")

        self._drive(node, 1)
        self.assertEqual(node.mission3_step_index, 1, "상한에서 다음 step으로 넘어간다")
        self.assertEqual(node.state, "MISSION3_SELECT_STEP")
        self.assertEqual(node.mission3_subgoal_retries, 0, "카운터는 step마다 새로 센다")

    def test_relaxed_retarget_is_tried_before_giving_up(self):
        """포기 직전에 접근 상한을 풀고 한 번 다시 잡아본다 - 성공하면 step 유지."""
        node = _Node()
        node.approach_result = (9.0, 9.0)     # 지금 목표(4,-2)와 다른 지점
        self._drive(node, config.MISSION3_SUBGOAL_MAX_RETRIES)
        self.assertEqual(node.mission3_step_index, 0, "재타겟에 성공하면 step을 안 버린다")
        self.assertEqual(node.restarted[1], (9.0, 9.0))
        self.assertEqual(node.mission3_subgoal_retries, 0)

    def test_relaxed_retarget_that_returns_the_same_point_gives_up(self):
        """같은 자리를 다시 고르면 재시도해봐야 결과가 같다."""
        node = _Node()
        node.approach_result = (4.0, -2.0)    # 현재 target_goal_xy와 동일
        self._drive(node, config.MISSION3_SUBGOAL_MAX_RETRIES)
        self.assertEqual(node.mission3_step_index, 1)


if __name__ == "__main__":
    unittest.main()
