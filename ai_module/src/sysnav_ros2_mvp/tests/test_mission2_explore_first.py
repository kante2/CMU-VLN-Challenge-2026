from collections import deque
import sys
import types
import unittest


# mission2_pipe의 상태 전환은 ROS 메시지 타입과 무관하다. 테스트 환경에 ROS가
# 없어도 import할 수 있도록 marker builder 의존성만 최소 stub으로 대체한다.
_module_name = "sysnav.scene_graph.scene_graph_rviz"
_previous_module = sys.modules.get(_module_name)
_stub = types.ModuleType(_module_name)
_stub.build_selected_object_marker = lambda obj, stamp: None
sys.modules[_module_name] = _stub
from sysnav.missions import mission2_pipe  # noqa: E402
if _previous_module is None:
    del sys.modules[_module_name]
else:
    sys.modules[_module_name] = _previous_module


class _Lock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))


class _ObjectMemory:
    def __init__(self, candidates):
        self.candidates = candidates

    def find_by_category(self, category):
        return list(self.candidates)


class _SceneGraph:
    def __init__(self, candidates):
        self.candidates = candidates

    def snapshot(self):
        return {"objects": list(self.candidates)}

    @staticmethod
    def resolve_relation_chain(task):
        # 이 픽스처의 task에는 관계 체인이 없어 조기 종료가 걸리지 않는다.
        # (조기 종료 자체는 tests/test_mission2_early_answer.py가 검증한다.)
        return []


class _CoveragePlanner:
    @staticmethod
    def describe_last_plan_failure():
        return "reason=no_surface_points"


class _Node:
    def __init__(self, candidates):
        self.state_lock = _Lock()
        self.state = "OBSERVE"
        self.exploration_route = deque([{"x": 1.0}])
        self.mission2_exploration_complete = False
        self.mission2_answer_object_id = None
        self.object_memory = _ObjectMemory(candidates)
        self.scene_graph = _SceneGraph(candidates)
        self.coverage_planner = _CoveragePlanner()
        self.task = {"target": "pillows"}
        self.logger = _Logger()

    def get_logger(self):
        return self.logger

    def clear_target_navigation(self):
        self.target_navigation_cleared = True


class Mission2ExploreFirstTest(unittest.TestCase):
    def test_detection_does_not_interrupt_exploration_route(self):
        node = _Node(candidates=[{"object_id": 3, "category": "pillows"}])
        node.state = "FOLLOW_EXPLORATION"
        mission2_pipe._on_perception_result(
            node, {"candidates": [{"object_id": 3}]}, origin_state="FOLLOW_EXPLORATION"
        )
        self.assertEqual(node.state, "FOLLOW_EXPLORATION")
        self.assertEqual(len(node.exploration_route), 1)

    def test_observe_always_returns_to_exploration_even_with_candidate(self):
        node = _Node(candidates=[{"object_id": 3, "category": "pillows"}])
        mission2_pipe._on_perception_result(
            node, {"candidates": [{"object_id": 3}]}, origin_state="OBSERVE"
        )
        self.assertEqual(node.state, "PLAN_EXPLORATION")
        self.assertEqual(len(node.exploration_route), 1)

    def test_full_exploration_then_selects_from_accumulated_graph(self):
        node = _Node(candidates=[{"object_id": 3, "category": "pillows"}])
        mission2_pipe._on_exploration_result(
            node, {"target": "pillows"}, {"route": []}
        )
        self.assertTrue(node.mission2_exploration_complete)
        self.assertEqual(node.state, "SELECT_TARGET")

    def test_full_exploration_without_candidate_fails(self):
        node = _Node(candidates=[])
        mission2_pipe._on_exploration_result(
            node, {"target": "pillows"}, {"route": []}
        )
        self.assertTrue(node.mission2_exploration_complete)
        self.assertEqual(node.state, "FAILED")

    def test_unreachable_target_does_not_restart_finished_exploration(self):
        node = _Node(candidates=[{"object_id": 3, "category": "pillows"}])
        node.mission2_exploration_complete = True
        mission2_pipe._give_up_target(node)
        self.assertTrue(node.target_navigation_cleared)
        self.assertEqual(node.state, "FAILED",
                         "답을 아직 못 냈으므로 도달 실패는 진짜 실패다")


if __name__ == "__main__":
    unittest.main()
