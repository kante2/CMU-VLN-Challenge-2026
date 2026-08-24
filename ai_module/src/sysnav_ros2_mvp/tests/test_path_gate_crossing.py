""""take the path between A and B"는 두 물체 사이를 실제로 가로질러야 성공한다."""

import unittest
from threading import RLock

from sysnav.missions import mission3_pipe, path_gate


class SegmentIntersectionTest(unittest.TestCase):
    GATE_A = (0.0, -1.0)
    GATE_B = (0.0, 1.0)

    def test_crossing_the_gate_is_detected(self):
        self.assertTrue(path_gate.segments_intersect((-1.0, 0.0), (1.0, 0.0), self.GATE_A, self.GATE_B))

    def test_stopping_short_of_the_gate_is_not_a_crossing(self):
        self.assertFalse(path_gate.segments_intersect((-1.0, 0.0), (-0.2, 0.0), self.GATE_A, self.GATE_B))

    def test_passing_outside_the_gate_is_not_a_crossing(self):
        # 게이트 선분을 넘어가긴 하지만 두 물체 **바깥**으로 돌아간 궤적.
        self.assertFalse(path_gate.segments_intersect((-1.0, 3.0), (1.0, 3.0), self.GATE_A, self.GATE_B))

    def test_parallel_travel_along_the_gate_is_not_a_crossing(self):
        self.assertFalse(path_gate.segments_intersect((0.5, -1.0), (0.5, 1.0), self.GATE_A, self.GATE_B))

    def test_extension_catches_a_pass_just_outside_an_endpoint(self):
        """물체 바로 옆(선분 끝단 살짝 바깥)을 스치듯 지나가는 궤적도 통과로 친다."""
        extended_a, extended_b = path_gate.extend_segment(self.GATE_A, self.GATE_B, 0.3)
        grazing = ((-1.0, 1.15), (1.0, 1.15))
        self.assertFalse(path_gate.segments_intersect(*grazing, self.GATE_A, self.GATE_B))
        self.assertTrue(path_gate.segments_intersect(*grazing, extended_a, extended_b))


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, _message):
        pass


class _Node:
    def __init__(self):
        self.state_lock = RLock()
        self.sensor_lock = RLock()
        self.state = "MISSION3_NAVIGATE_STEP"
        self.mission3_step_index = 0
        # 게이트: (0,-1)-(0,1). 중점 waypoint는 (0,0)이지만 로봇은 그 근처에 오지 않는다.
        self.mission3_gate_segment = ((0.0, -1.0), (0.0, 1.0))
        self.mission3_gate_crossed = False
        self.mission3_gate_last_xy = (-3.0, 0.9)
        self.mission3_gate_last_stamp = 0.0
        self.pose_buffer = []
        self.target_goal_xy = (0.0, 0.0)
        self.target_final_theta = 0.0
        self.target_marker_index = 0
        self.marker_refreshes = 0
        self.navigation_clears = 0
        self.step_calls = 0

    def step_target_navigation(self, _pose):
        self.step_calls += 1
        return "driving"

    def refresh_goal_marker(self):
        self.marker_refreshes += 1

    def clear_target_navigation(self):
        self.navigation_clears += 1

    def get_logger(self):
        return _Logger()


class GateCompletesStepTest(unittest.TestCase):
    def test_crossing_completes_the_step_even_far_from_the_midpoint(self):
        node = _Node()
        # 중점(0,0)에서 0.9m 떨어진 높이로 게이트를 가로지른다 - 반경 판정만으로는
        # 애매하고, 궤적 샘플도 tick 사이에 게이트를 건너뛴다.
        node.pose_buffer = [
            (1.0, {"x": -1.0, "y": 0.9}),
            (2.0, {"x": 1.0, "y": 0.9}),
        ]
        pose = {"x": 2.0, "y": 0.9, "yaw": 0.0, "stamp": 3.0}

        mission3_pipe._navigate_step(node, {"steps": [{"is_stop": False}]}, pose)

        self.assertTrue(node.mission3_gate_crossed)
        self.assertEqual(node.step_calls, 0)
        self.assertEqual(node.mission3_step_index, 1)
        self.assertEqual(node.state, "MISSION3_SELECT_STEP")

    def test_not_crossing_keeps_driving(self):
        node = _Node()
        node.target_goal_xy = (0.0, 5.0)  # 중점 반경 판정도 안 걸리는 위치
        node.pose_buffer = [(1.0, {"x": -2.0, "y": 0.9})]
        pose = {"x": -1.5, "y": 0.9, "yaw": 0.0, "stamp": 2.0}

        mission3_pipe._navigate_step(node, {"steps": [{"is_stop": False}]}, pose)

        self.assertFalse(node.mission3_gate_crossed)
        self.assertEqual(node.step_calls, 1)
        self.assertEqual(node.mission3_step_index, 0)

    def test_radius_still_completes_a_step_without_a_gate(self):
        """OR 조건 회귀 - 게이트가 없는 step은 예전 그대로 반경으로 넘어간다."""
        node = _Node()
        node.mission3_gate_segment = None
        node.target_goal_xy = (0.9, 0.0)
        pose = {"x": 0.0, "y": 0.0, "yaw": 0.0, "stamp": 1.0}

        mission3_pipe._navigate_step(node, {"steps": [{"is_stop": True}]}, pose)

        self.assertEqual(node.step_calls, 0)
        self.assertEqual(node.mission3_step_index, 1)


class GateArmingTest(unittest.TestCase):
    def test_arming_ignores_the_trajectory_of_the_previous_step(self):
        node = _Node()
        node.pose_buffer = [
            (1.0, {"x": -1.0, "y": 0.0}),   # 이전 step에서 이미 게이트를 가로질렀다
            (2.0, {"x": 1.0, "y": 0.0}),
        ]

        path_gate.arm_gate(node, ((0.0, -1.0), (0.0, 1.0)), {"x": 1.0, "y": 0.0, "stamp": 2.0})

        self.assertFalse(path_gate.update_gate_crossing(node, {"x": 1.2, "y": 0.0, "stamp": 3.0}))

    def test_a_step_without_a_gate_never_reports_a_crossing(self):
        node = _Node()
        path_gate.arm_gate(node, None, {"x": 0.0, "y": 0.0, "stamp": 1.0})

        self.assertIsNone(node.mission3_gate_segment)
        self.assertFalse(path_gate.update_gate_crossing(node, {"x": 5.0, "y": 5.0, "stamp": 2.0}))


if __name__ == "__main__":
    unittest.main()
