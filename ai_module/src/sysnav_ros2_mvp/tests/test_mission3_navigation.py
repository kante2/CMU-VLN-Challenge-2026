"""Mission 3 목적지 주행 회귀 테스트.

원래 이 파일은 leg-queue 시절의 _mission3_goal_reached()를 검증했다. 그 구조는
sysnav_node.step_target_navigation()으로 대체됐지만(base autonomy의 waypointConverter가
adjDisThre=5.0 안의 goal을 계속 재타게팅해서, 우리가 hop을 잘라 보내면 로봇이 목표
반대편으로 끌려갔다), 검증하려던 동작 자체는 그대로 유효해서 새 API 기준으로 옮겼다:

  - 직전 step의 stale goal로 도착 판정이 나면 안 된다
  - 도달 판정 반경이 실제로 적용돼야 한다
  - 진전이 없다고 곧바로 목표를 버리면 안 된다(먼저 같은 goal을 재발행)
"""

import time
import unittest
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

from sysnav import config
from sysnav.sysnav_node import SysNavNode


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass


def _node(goal_xy, current_goal, *, stalled=True, best_distance_m=3.0):
    """step_target_navigation()만 돌릴 수 있는 최소 노드."""
    node = object.__new__(SysNavNode)
    node.current_goal = current_goal
    node.target_route = deque()
    node.target_goal_xy = goal_xy
    node.target_final_theta = 0.0
    node.target_forbidden_mask = None
    node.target_object_id = None
    node.target_object_xy = None
    node.target_marker_index = None
    node.state_lock = nullcontext()
    node._target_replan_count = 0
    node._target_last_replan_time = None
    node._target_retarget_count = 0
    node._target_republish_count = 0
    node._target_unreachable_reason = None
    node._trace_navigation = lambda *_args: None
    node.coverage_planner = SimpleNamespace(last_direct_path_diagnostics={"reason": "ok"})
    node.get_logger = lambda: _Logger()
    node.published = []
    node.goal_publisher = SimpleNamespace(
        publish=lambda x, y, theta: node.published.append((x, y, theta))
    )
    # 진행도 감시: stalled=True면 이미 타임아웃을 넘긴 상태로 시작한다.
    # 최단거리 기록. 테스트 pose보다 크면 "가까워졌다"로 판정돼 타이머가
    # 리셋되므로, 정지 상황을 만들 때는 그 pose의 실제 거리를 넣어야 한다.
    node._target_goal_best_distance_m = best_distance_m
    node._target_goal_last_progress_time = (
        time.monotonic() - config.TARGET_REPLAN_STUCK_TIMEOUT_SEC - 1.0
        if stalled else time.monotonic()
    )
    return node


class Mission3NavigationTest(unittest.TestCase):
    def test_arrival_uses_goal_reached_radius(self):
        """도달 판정 반경이 실제로 적용되는지."""
        inside = config.GOAL_REACHED_DISTANCE_M - 0.05
        outside = config.TARGET_ARRIVAL_FALLBACK_MAX_M + 0.5

        node = _node((0.0, 0.0), None, stalled=False)
        self.assertEqual(node.step_target_navigation({"x": inside, "y": 0.0}), "arrived")

        node = _node((0.0, 0.0), None, stalled=False)
        self.assertEqual(node.step_target_navigation({"x": outside, "y": 0.0}), "driving")

    def test_stalled_republishes_before_giving_up(self):
        """진전이 없다고 곧바로 포기하지 않고 같은 goal을 먼저 다시 쏜다.

        base autonomy가 목표를 놓쳤을 뿐인 경우가 있어서, 판정 전에 재발행하는 편이
        이득이다(재발행은 비용이 없다)."""
        goal = {"x": 3.0, "y": 0.0, "theta": 0.0, "type": "target"}
        node = _node((3.0, 0.0), goal)

        self.assertEqual(node.step_target_navigation({"x": 0.0, "y": 0.0}), "driving")
        self.assertEqual(node.published, [(3.0, 0.0, 0.0)])
        self.assertEqual(node.target_goal_xy, (3.0, 0.0))

    def test_stalled_beyond_republish_limit_is_unreachable(self):
        """재발행해도 진전이 없고 목적지가 도달 인정 범위 밖이면 포기한다."""
        goal = {"x": 3.0, "y": 0.0, "theta": 0.0, "type": "target"}
        node = _node((3.0, 0.0), goal)
        node._target_republish_count = config.TARGET_REPUBLISH_MAX_COUNT

        self.assertEqual(node.step_target_navigation({"x": 0.0, "y": 0.0}), "unreachable")

    def test_stalled_within_fallback_counts_as_arrived(self):
        """더 못 가는데 목적지 코앞이면 도달로 인정한다 - 0.43m 남기고 7분 정지하던
        실패(2026-08-11)를 막는 장치."""
        near = config.TARGET_ARRIVAL_FALLBACK_MAX_M - 0.1
        goal = {"x": 0.0, "y": 0.0, "theta": 0.0, "type": "target"}
        node = _node((0.0, 0.0), goal, best_distance_m=near)
        node._target_republish_count = config.TARGET_REPUBLISH_MAX_COUNT

        self.assertEqual(node.step_target_navigation({"x": near, "y": 0.0}), "arrived")

    def test_cleared_navigation_does_not_arrive_on_stale_goal(self):
        """목적지를 정리한 뒤에는 직전 goal로 도착 판정이 나면 안 된다.

        mission3는 step 사이에 clear_target_navigation()을 부르는데, 여기서 stale goal이
        남으면 다음 step이 시작되기도 전에 "도착"으로 넘어가버린다."""
        node = _node((0.0, 0.0), {"x": 0.0, "y": 0.0, "theta": 0.0, "type": "target"})
        node.clear_target_navigation()

        self.assertIsNone(node.current_goal)
        self.assertIsNone(node.target_goal_xy)
        self.assertEqual(node.step_target_navigation({"x": 0.0, "y": 0.0}), "driving")


if __name__ == "__main__":
    unittest.main()
