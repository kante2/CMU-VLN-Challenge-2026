import time
import unittest
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

from sysnav.missions import mission3_pipe


class _Logger:
    def warning(self, _message):
        pass


class Mission3NavigationTest(unittest.TestCase):
    @staticmethod
    def _node(goal):
        return SimpleNamespace(
            current_goal=goal,
            mission3_step_index=0,
            mission3_leg_queue=deque([goal]),
            state="MISSION3_NAVIGATE_STEP",
            state_lock=nullcontext(),
            get_logger=lambda: _Logger(),
            _exploration_goal_unreachable=lambda _pose: True,
        )

    def test_reached_rejects_stale_exploration_goal(self):
        node = self._node({
            "x": 0.0,
            "y": 0.0,
            "type": "exploration",
            "step_index": 0,
            "activated_at_monotonic": time.monotonic() - 1.0,
        })
        self.assertFalse(mission3_pipe._mission3_goal_reached(node, {"x": 0.0, "y": 0.0}))

    def test_reached_uses_strict_step_radius(self):
        node = self._node({
            "x": 0.0,
            "y": 0.0,
            "type": "mission3_leg",
            "step_index": 0,
            "activated_at_monotonic": time.monotonic() - 1.0,
        })
        self.assertFalse(mission3_pipe._mission3_goal_reached(node, {"x": 0.60, "y": 0.0}))
        self.assertTrue(mission3_pipe._mission3_goal_reached(node, {"x": 0.40, "y": 0.0}))

    def test_unreachable_final_goal_remains_active(self):
        goal = {
            "x": 2.0,
            "y": 0.0,
            "type": "mission3_leg",
            "step_index": 0,
            "activated_at_monotonic": time.monotonic() - 1.0,
        }
        node = self._node(goal)
        published = []
        node.goal_publisher = SimpleNamespace(
            publish=lambda x, y, theta: published.append((x, y, theta))
        )
        node._exploration_goal_best_distance_m = 3.0
        node._exploration_goal_last_progress_time = 0.0
        mission3_pipe._navigate_step(node, {"steps": [{"is_stop": True}]}, {"x": 0.0, "y": 0.0})
        self.assertEqual(node.state, "MISSION3_NAVIGATE_STEP")
        self.assertIs(node.current_goal, goal)
        self.assertEqual(list(node.mission3_leg_queue), [goal])
        self.assertEqual(published, [(2.0, 0.0, 0.0)])
        self.assertEqual(node.mission3_step_index, 0)


if __name__ == "__main__":
    unittest.main()
