"""주행 중 waypoint가 왕복하지 않는지에 대한 회귀 테스트.

배경: choose_approach_point()는 로봇 현재 위치 기준(direction = object - robot)으로
접근 지점을 고른다. 예전에는 terrain이 "지금 목표는 못 쓴다"고 하면 주행 중에도 바로
다시 골라서, 로봇이 움직일 때마다 접근 지점이 물체 주위를 따라 돌며 waypoint가
왕복했다(A -> B -> A). 이제는 (1) 목표까지 실제로 가까워지는 동안에는 재선택 자체를
보지 않고, (2) 재선택할 때는 버린 지점을 기록해 같은 자리를 다시 못 고르게 한다.
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


def _node(*, progress_age: float, supported: bool = False,
          unsupported_age: float | None = 99.0,
          approach_point=(0.9, -2.0)):
    """progress_age: 마지막 "진전" 이후 흐른 시간(초).
    unsupported_age: "이 목표 못 쓴다"가 처음 보인 뒤 흐른 시간(초)."""
    node = object.__new__(SysNavNode)
    node.target_route = deque()
    node.target_goal_xy = (0.40, -2.28)
    node.target_final_theta = 0.0
    node.target_object_xy = (0.0, -3.10)
    node.target_marker_index = None
    node.current_goal = {"x": 0.40, "y": -2.28, "theta": 0.0, "type": "target"}
    node.sensor_lock = nullcontext()
    node.state_lock = nullcontext()
    node._target_retarget_count = 0
    node._target_republish_count = 0
    node._target_rejected_points = []
    node._target_last_replan_time = None
    node._target_goal_best_distance_m = 0.0
    node._target_goal_last_progress_time = time.monotonic() - progress_age
    node._waypoint_echo_xy = None
    node._waypoint_echo_time = None
    node._waypoint_echo_mismatch_since = None
    node._target_unsupported_since = (
        None if unsupported_age is None else time.monotonic() - unsupported_age
    )
    node._trace_navigation = lambda *_args: None
    node.get_logger = lambda: _Logger()
    node.published = []
    node._publish_target_goal = lambda x, y, theta, is_final: node.published.append((x, y))
    node.refresh_goal_marker = lambda: None
    node.chosen_with = []
    node.terrain_monitor = SimpleNamespace(
        last_selection="-",
        ready=lambda: True,
        is_waypoint_supported=lambda *_args: supported,
        choose_approach_point=lambda *args, **kwargs: (
            node.chosen_with.append(kwargs.get("rejected")) or approach_point
        ),
    )
    return node


class TargetWaypointStabilityTest(unittest.TestCase):
    def test_progress_holds_the_current_waypoint(self):
        """목표에 가까워지는 중이면 terrain이 못 쓴다고 해도 목표를 바꾸지 않는다."""
        node = _node(progress_age=1.0)

        status = node.step_target_navigation({"x": 3.0, "y": -2.28, "yaw": 0.0})

        self.assertEqual(status, "driving")
        self.assertEqual(node.target_goal_xy, (0.40, -2.28))
        self.assertEqual(node.published, [])

    def test_stalled_robot_retargets(self):
        """진전이 멈춘 뒤에야 다른 접근 지점으로 갈아탄다."""
        node = _node(progress_age=config.TARGET_RETARGET_PATIENCE_SEC + 1.0)

        status = node.step_target_navigation({"x": 3.0, "y": -2.28, "yaw": 0.0})

        self.assertEqual(status, "driving")
        self.assertEqual(node.target_goal_xy, (0.9, -2.0))
        self.assertEqual(node.published, [(0.9, -2.0)])

    def test_retarget_forbids_the_point_it_just_left(self):
        """버린 목표를 rejected에 넣어야 A -> B -> A 왕복이 끊긴다."""
        node = _node(progress_age=config.TARGET_RETARGET_PATIENCE_SEC + 1.0)

        node._retarget_if_unsupported({"x": 3.0, "y": -2.28, "yaw": 0.0})

        self.assertIn((0.40, -2.28), node.chosen_with[0])
        self.assertEqual(node._target_rejected_points, [(0.40, -2.28)])

    def test_no_alternative_keeps_the_goal_and_forgets_nothing(self):
        """대안이 없으면 목표를 그대로 두고, 그 지점을 영구 제외하지도 않는다."""
        node = _node(progress_age=config.TARGET_RETARGET_PATIENCE_SEC + 1.0)
        node.terrain_monitor.choose_approach_point = lambda *_a, **_k: None

        self.assertFalse(node._retarget_if_unsupported({"x": 3.0, "y": -2.28, "yaw": 0.0}))
        self.assertEqual(node.target_goal_xy, (0.40, -2.28))
        self.assertEqual(node._target_rejected_points, [])

    def test_supported_goal_is_never_retargeted(self):
        """terrain이 통과라고 보는 목표는 멈춰 있어도 여기서 바꾸지 않는다
        (그 경우는 converter 되먹임/stall 백스톱이 다룬다)."""
        node = _node(progress_age=config.TARGET_RETARGET_PATIENCE_SEC + 1.0, supported=True)

        self.assertFalse(node._retarget_if_unsupported({"x": 3.0, "y": -2.28, "yaw": 0.0}))
        self.assertEqual(node.published, [])

    def test_single_frame_unsupported_verdict_is_ignored(self):
        """/terrain_map은 롤링 로컬 맵이라 한 프레임의 "못 쓴다"로는 갈아타지 않는다."""
        node = _node(
            progress_age=config.TARGET_RETARGET_PATIENCE_SEC + 1.0, unsupported_age=None
        )

        self.assertFalse(node._retarget_if_unsupported({"x": 3.0, "y": -2.28, "yaw": 0.0}))
        self.assertIsNotNone(node._target_unsupported_since)  # 시작 시각만 기록
        self.assertEqual(node.published, [])

    def test_sustained_unsupported_verdict_retargets(self):
        """확인 시간 동안 계속 못 쓰는 상태면 그때 갈아탄다."""
        node = _node(
            progress_age=config.TARGET_RETARGET_PATIENCE_SEC + 1.0,
            unsupported_age=config.TERRAIN_UNSUPPORTED_CONFIRM_SEC + 0.5,
        )

        self.assertTrue(node._retarget_if_unsupported({"x": 3.0, "y": -2.28, "yaw": 0.0}))
        self.assertEqual(node.published, [(0.9, -2.0)])

    def test_supported_again_resets_the_dwell(self):
        """중간에 다시 통과로 보이면 dwell을 처음부터 다시 센다."""
        node = _node(progress_age=config.TARGET_RETARGET_PATIENCE_SEC + 1.0, supported=True)

        self.assertFalse(node._retarget_if_unsupported({"x": 3.0, "y": -2.28, "yaw": 0.0}))
        self.assertIsNone(node._target_unsupported_since)

    def test_goal_collapsing_onto_the_robot_is_not_published(self):
        """접근 지점이 로봇이 서 있는 자리로 수축하면 발행하지 않는다 - 발행해도
        로봇은 안 움직이고 waypoint만 새로 찍힌 것처럼 보인다."""
        node = _node(
            progress_age=config.TARGET_RETARGET_PATIENCE_SEC + 1.0,
            approach_point=(3.05, -2.30),
        )

        self.assertFalse(node._retarget_if_unsupported({"x": 3.0, "y": -2.28, "yaw": 0.0}))
        self.assertEqual(node.published, [])
        self.assertEqual(node.target_goal_xy, (0.40, -2.28))
        self.assertEqual(node._target_rejected_points, [])


if __name__ == "__main__":
    unittest.main()
