"""Mission 2 목적지 주행 회귀 테스트.

원래 이 파일은 mission2_pipe 안에 있던 watchdog(정지 시 같은 goal 재발행)을 직접
검증했다. 그 로직은 mission2/mission3가 공유하는 sysnav_node.step_target_navigation()
으로 옮겨졌고(판정이 미션마다 달라질 이유가 없다), mission2_pipe는 그 결과에 따라
SUCCESS/탐색복귀만 정한다. 그래서 여기서는 "outcome -> 미션 상태 전이"를 검증한다.
재발행 자체의 검증은 test_mission3_navigation.py에 있다.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from sysnav.missions import mission2_pipe


class Mission2NavigationTest(unittest.TestCase):
    @staticmethod
    def _node(outcome):
        node = Mock()
        node.state_lock = nullcontext()
        node.state = "NAVIGATE_TARGET"
        node.target_object_id = 7
        node.object_memory.get.return_value = {"category": "chair"}
        node.step_target_navigation.return_value = outcome
        return node

    def test_driving_keeps_state(self):
        """주행 중에는 상태를 건드리지 않는다."""
        node = self._node("driving")

        mission2_pipe._run_navigate_target(node, {"x": 0.0, "y": 0.0})

        self.assertEqual(node.state, "NAVIGATE_TARGET")
        node.clear_target_navigation.assert_not_called()

    def test_arrived_finishes_task(self):
        node = self._node("arrived")

        mission2_pipe._run_navigate_target(node, {"x": 0.0, "y": 0.0})

        self.assertEqual(node.state, "SUCCESS")
        node.clear_target_navigation.assert_called_once()

    def test_unreachable_returns_to_exploration(self):
        """지금 지도로 못 가면 FAILED가 아니라 탐색으로 되돌린다 - "길이 없다"는
        보통 "아직 안 뚫었다"라서, 맵이 넓어지면 갈 수 있게 되는 경우가 많다."""
        node = self._node("unreachable")

        mission2_pipe._run_navigate_target(node, {"x": 0.0, "y": 0.0})

        self.assertEqual(node.state, "PLAN_EXPLORATION")
        node.clear_target_navigation.assert_called_once()


if __name__ == "__main__":
    unittest.main()
