"""waypointConverter 되먹임(/way_point) 감시 회귀 테스트.

배경(2026-08-18 실측): 우리가 /way_point_with_heading에 (0.40,-2.28)을 찍었는데
base autonomy의 waypointConverter는 그 좌표를 버리고 로봇 바로 옆 (1.08,-0.84)를
/way_point로 내보냈다. 로봇 입장에선 이미 도착한 지점이라 움직이지 않았고, 우리
terrain 복제(terrain_monitor)는 그 목표를 "통과"로 봤기 때문에 기존 재타게팅
경로(_retarget_if_unsupported)로는 영원히 못 잡았다. 40초짜리 stall 타임아웃이
돌 때까지 기다리는 대신, 실제로 나간 /way_point를 보고 즉시 다른 접근 지점으로
옮기는 것이 이 로직이다.

시뮬레이터 컨테이너(base autonomy)는 챌린지 규정상 수정할 수 없으므로
(README Submission: 변경 허용 범위는 ai_module/ 아래뿐), 대응은 전부 이쪽에서 한다.
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


def _node(goal_xy=(0.40, -2.28), *, echo_xy=None, echo_age=0.0, mismatch_age=None):
    node = object.__new__(SysNavNode)
    node.target_route = deque()
    node.target_goal_xy = goal_xy
    node.target_final_theta = 0.0
    node.target_object_xy = (0.0, -3.10)
    node.target_marker_index = None
    node.current_goal = None
    node.sensor_lock = nullcontext()
    node.state_lock = nullcontext()
    node._target_retarget_count = 0
    node._target_rejected_points = []
    node._target_goal_last_progress_time = time.monotonic()
    node._waypoint_echo_xy = echo_xy
    node._waypoint_echo_time = None if echo_xy is None else time.monotonic() - echo_age
    node._waypoint_echo_mismatch_since = (
        None if mismatch_age is None else time.monotonic() - mismatch_age
    )
    node._trace_navigation = lambda *_args: None
    node.get_logger = lambda: _Logger()
    node.published = []
    node._publish_target_goal = lambda x, y, theta, is_final: node.published.append((x, y))
    node.refresh_goal_marker = lambda: None
    node.terrain_monitor = SimpleNamespace(
        last_selection="-",
        choose_approach_point=lambda *_args, **_kwargs: (0.9, -2.0),
    )
    return node


class WaypointConverterFeedbackTest(unittest.TestCase):
    def test_matching_echo_is_not_a_rejection(self):
        """converter가 우리 좌표를 그대로(미세 조정 범위 안) 내보내면 정상이다."""
        node = _node(echo_xy=(0.45, -2.30), mismatch_age=10.0)
        self.assertIsNone(node.converter_rejected_goal())
        self.assertIsNone(node._waypoint_echo_mismatch_since)

    def test_single_frame_mismatch_waits_for_confirmation(self):
        """한 번 어긋난 것만으로는 거부로 보지 않는다(직전 목표의 잔상 등)."""
        node = _node(echo_xy=(1.08, -0.84))
        self.assertIsNone(node.converter_rejected_goal())
        self.assertIsNotNone(node._waypoint_echo_mismatch_since)

    def test_sustained_mismatch_is_a_rejection(self):
        """확인 시간 동안 계속 어긋나 있으면 거부로 확정하고 어긋난 거리를 돌려준다."""
        node = _node(
            echo_xy=(1.08, -0.84),
            mismatch_age=config.WAYPOINT_ECHO_CONFIRM_SEC + 1.0,
        )
        deviation = node.converter_rejected_goal()
        self.assertIsNotNone(deviation)
        self.assertGreater(deviation, config.WAYPOINT_ECHO_TOLERANCE_M)

    def test_stale_echo_is_not_judged(self):
        """echo가 끊긴 상태에서는 판정하지 않는다 - converter가 멈춘 것일 수 있다."""
        node = _node(
            echo_xy=(1.08, -0.84),
            echo_age=config.WAYPOINT_ECHO_STALE_SEC + 1.0,
            mismatch_age=config.WAYPOINT_ECHO_CONFIRM_SEC + 1.0,
        )
        self.assertIsNone(node.converter_rejected_goal())

    def test_rejection_retargets_and_remembers_the_bad_point(self):
        """거부되면 그 지점을 기록하고 다른 접근 지점을 발행한다."""
        node = _node(
            echo_xy=(1.08, -0.84),
            mismatch_age=config.WAYPOINT_ECHO_CONFIRM_SEC + 1.0,
        )
        self.assertTrue(node._retarget_if_converter_rejected({"x": 1.38, "y": -0.76}))
        self.assertEqual(node._target_rejected_points, [(0.40, -2.28)])
        self.assertEqual(node.target_goal_xy, (0.9, -2.0))
        self.assertEqual(node.published, [(0.9, -2.0)])

    def test_no_alternative_leaves_the_goal_for_the_stall_check(self):
        """대체 접근 지점이 없으면 목표를 그대로 두고 기존 stall 판정에 맡긴다."""
        node = _node(
            echo_xy=(1.08, -0.84),
            mismatch_age=config.WAYPOINT_ECHO_CONFIRM_SEC + 1.0,
        )
        node.terrain_monitor.choose_approach_point = lambda *_a, **_k: None
        self.assertFalse(node._retarget_if_converter_rejected({"x": 1.38, "y": -0.76}))
        self.assertEqual(node.target_goal_xy, (0.40, -2.28))
        self.assertEqual(node.published, [])

    def test_waypoint_hops_are_left_alone(self):
        """forbidden 우회로의 중간 hop 주행 중에는 개입하지 않는다."""
        node = _node(
            echo_xy=(1.08, -0.84),
            mismatch_age=config.WAYPOINT_ECHO_CONFIRM_SEC + 1.0,
        )
        node.target_route = deque([{"x": 1.0, "y": 1.0, "theta": 0.0}])
        self.assertFalse(node._retarget_if_converter_rejected({"x": 1.38, "y": -0.76}))


if __name__ == "__main__":
    unittest.main()
