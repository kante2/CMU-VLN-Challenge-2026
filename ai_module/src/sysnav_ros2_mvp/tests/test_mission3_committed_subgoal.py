"""A committed Mission 3 marker must remain the active navigation target."""

import unittest
from threading import RLock

from sysnav.missions import mission3_pipe


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
        self.state = "MISSION3_NAVIGATE_STEP"
        self.mission3_step_index = 0
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

    def step_target_navigation(self, _pose):
        self.step_calls += 1
        return "unreachable"

    def start_target_navigation(self, pose, goal_xy, final_theta, **kwargs):
        self.restarted = (pose, goal_xy, final_theta, kwargs)

    def refresh_goal_marker(self):
        self.marker_refreshes += 1

    def clear_target_navigation(self):
        self.navigation_clears += 1

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


if __name__ == "__main__":
    unittest.main()
