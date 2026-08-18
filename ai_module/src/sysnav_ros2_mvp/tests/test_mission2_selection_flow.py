"""Mission 2 - "탐색 완주 -> Scene Graph 완성 -> 선택" 회귀 테스트.

예전 구조는 target 카테고리 후보가 하나라도 잡히면 곧바로 SELECT_TARGET으로 갔다.
그 결과 "closest/farthest" 같은 최상급 문장에서 아직 못 본 물체가 정답이면 그대로
오답이 됐고, 탐색이 소진되면 FAILED로 끝나 답 자체를 못 냈다. 여기서는 그 전이가
현재 구조대로 유지되는지만 검증한다(주행 전이는 test_mission2_navigation.py).
"""

import time
import unittest
from contextlib import nullcontext
from unittest.mock import Mock

from sysnav import config
from sysnav.missions import mission2_pipe


def _node(*, elapsed_sec=0.0, recovery_points=None, pose=None):
    node = Mock()
    node.state_lock = nullcontext()
    node.sensor_lock = nullcontext()
    node.state = "OBSERVE"
    node.task_start_time = time.monotonic() - elapsed_sec
    node.mission2_select_final = False
    node.mission2_recovery_points = [] if recovery_points is None else list(recovery_points)
    node.exploration_route = []
    node.latest_pose = pose
    # selected_id=None -> object_memory.get(None)은 실제로 None을 돌려준다.
    node.object_memory.get.return_value = None
    return node


class Mission2SelectionFlowTest(unittest.TestCase):
    def test_candidates_do_not_stop_exploration(self):
        """후보가 생겨도 선택으로 넘어가지 않는다 - 탐색을 완주해야 한다."""
        node = _node()
        mission2_pipe._on_perception_result(
            node, {"candidates": [{"object_id": 1}]}, "OBSERVE"
        )
        self.assertEqual(node.state, "PLAN_EXPLORATION")
        self.assertFalse(node.mission2_select_final)

    def test_perception_while_moving_does_not_interrupt(self):
        """이동 중 관측은 메모리/그래프만 갱신하고 이동을 방해하지 않는다."""
        node = _node()
        node.state = "FOLLOW_EXPLORATION"
        mission2_pipe._on_perception_result(
            node, {"candidates": [{"object_id": 1}]}, "FOLLOW_EXPLORATION"
        )
        self.assertEqual(node.state, "FOLLOW_EXPLORATION")

    def test_exploration_exhausted_selects_instead_of_failing(self):
        """탐색 소진은 실패가 아니라 최종 선택 시점이다."""
        node = _node()
        mission2_pipe._on_exploration_result(node, {"route": []})
        self.assertEqual(node.state, "SELECT_TARGET")
        self.assertTrue(node.mission2_select_final)

    def test_deadline_stops_exploration(self):
        """10분 제한이 임박하면 프론티어가 남아도 최종 선택으로 넘어간다."""
        node = _node(elapsed_sec=config.MISSION2_SELECT_DEADLINE_SEC + 1.0)
        mission2_pipe._on_exploration_result(node, {"route": [{"x": 1.0, "y": 0.0}]})
        self.assertEqual(node.state, "SELECT_TARGET")
        self.assertTrue(node.mission2_select_final)

    def test_route_is_followed_before_the_deadline(self):
        """제한 시간 전에는 평소대로 탐색 경로를 따라간다."""
        node = _node()
        mission2_pipe._on_exploration_result(node, {"route": [{"x": 1.0, "y": 0.0}]})
        node.publish_next_exploration_goal.assert_called_once()
        self.assertFalse(node.mission2_select_final)

    def test_final_selection_without_candidate_patrols_first(self):
        """최종 선택에서 못 골랐으면 바로 실패하지 않고 recovery patrol을 돈다."""
        node = _node(pose={"x": 0.0, "y": 0.0, "theta": 0.0})
        node.mission2_select_final = True
        node.coverage_planner.plan_recovery_patrol.return_value = [
            {"x": 2.0, "y": 1.0, "theta": 0.0}
        ]
        mission2_pipe._on_selection_result(node, {"selected_id": None})
        self.assertEqual(len(node.mission2_recovery_points), 1)
        node.publish_next_exploration_goal.assert_called_once()
        self.assertNotEqual(node.state, "FAILED")

    def test_final_selection_fails_after_patrol_budget(self):
        """patrol 예산까지 소진하면 FAILED로 끝낸다."""
        exhausted = [(float(i), 0.0) for i in range(config.MISSION2_RECOVERY_PATROL_MAX_POINTS)]
        node = _node(pose={"x": 0.0, "y": 0.0, "theta": 0.0}, recovery_points=exhausted)
        node.mission2_select_final = True
        mission2_pipe._on_selection_result(node, {"selected_id": None})
        self.assertEqual(node.state, "FAILED")
        node.coverage_planner.plan_recovery_patrol.assert_not_called()

    def test_pending_before_exhaustion_returns_to_exploration(self):
        """탐색이 아직 안 끝났을 때의 pending은 예전처럼 탐색으로 되돌린다."""
        node = _node()
        mission2_pipe._on_selection_result(
            node, {"selected_id": None, "relation_pending": True}
        )
        self.assertEqual(node.state, "PLAN_EXPLORATION")

    def test_selection_job_receives_final_flag(self):
        """SELECT_TARGET에서 selection_job에 final 플래그를 그대로 넘긴다."""
        node = _node()
        node.mission2_select_final = True
        mission2_pipe.loop(node, "SELECT_TARGET", {"target": "chair"}, 3, {"x": 0.0, "y": 0.0})
        args, kwargs = node.submit_job.call_args
        self.assertIs(args[-1], True)
        self.assertEqual(kwargs["origin_state"], "SELECT_TARGET")


if __name__ == "__main__":
    unittest.main()
