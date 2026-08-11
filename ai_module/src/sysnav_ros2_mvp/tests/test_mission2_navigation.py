import unittest
from unittest.mock import Mock, patch

from sysnav.missions import mission2_pipe


class Mission2NavigationTest(unittest.TestCase):
    def _node(self):
        node = Mock()
        node.current_goal = {
            "x": 4.0,
            "y": 2.0,
            "theta": 0.5,
            "type": "target",
            "object_id": 7,
            "best_distance_m": 4.0,
            "last_progress_time": 10.0,
        }
        node.goal_reached.return_value = False
        return node

    @patch.object(mission2_pipe.time, "monotonic", return_value=19.0)
    def test_stuck_target_republishes_original_goal(self, _now):
        node = self._node()

        mission2_pipe._run_navigate_target(node, {"x": 0.0, "y": 2.0})

        node.goal_publisher.publish.assert_called_once_with(4.0, 2.0, 0.5)
        self.assertEqual(node.current_goal["last_progress_time"], 19.0)

    @patch.object(mission2_pipe.time, "monotonic", return_value=19.0)
    def test_meaningful_progress_resets_watchdog_without_republish(self, _now):
        node = self._node()

        mission2_pipe._run_navigate_target(node, {"x": 1.0, "y": 2.0})

        node.goal_publisher.publish.assert_not_called()
        self.assertEqual(node.current_goal["best_distance_m"], 3.0)
        self.assertEqual(node.current_goal["last_progress_time"], 19.0)


if __name__ == "__main__":
    unittest.main()
