"""관계 체인이 후보 하나로 좁혀지면 탐사를 끝까지 하지 않고 바로 고른다.

예전에는 "탐사 100% 완료"가 SELECT_TARGET으로 가는 유일한 관문이었다.
실측(2026-08-23, "Find the vase on the cabinet below the picture"):
  6분 시점에 체인이 이미 GT 정답 하나로 확정됐는데도 frontier가 남아 못 갔고,
  frontier는 탐사할수록 오히려 늘었다(21 -> 128셀). 그 사이 10분 제한이 지나간다.

후보가 둘 이상이면 계속 탐사한다 - README는 정답이 유일하다고 보장하므로 여럿
남은 것은 "덜 봤다"는 신호다.
"""

import sys
import types
import unittest
from collections import deque

_module_name = "sysnav.scene_graph.scene_graph_rviz"
_previous = sys.modules.get(_module_name)
_stub = types.ModuleType(_module_name)
_stub.build_selected_object_marker = lambda obj, stamp: None
sys.modules[_module_name] = _stub
from sysnav.missions import mission2_pipe      # noqa: E402
if _previous is None:
    del sys.modules[_module_name]
else:
    sys.modules[_module_name] = _previous


class _Lock:
    def __enter__(self): return self
    def __exit__(self, *args): return False


class _Logger:
    def __init__(self): self.messages = []
    def info(self, m): self.messages.append(("info", m))
    def warning(self, m): self.messages.append(("warning", m))


class _SceneGraph:
    def __init__(self, survivors): self.survivors = survivors
    def resolve_relation_chain(self, task): return list(self.survivors)


class _Node:
    def __init__(self, survivors, task=None):
        self.state_lock = _Lock()
        self.state = "OBSERVE"
        self.exploration_route = deque()
        self.scene_graph = _SceneGraph(survivors)
        self.task = task if task is not None else {
            "target": "vase", "relation": "on", "reference_objects": ["cabinet"],
            "relation_chain": [("vase", "on", "cabinet"), ("cabinet", "under", "picture")],
        }
        self.logger = _Logger()

    def get_logger(self): return self.logger


def _observe(node, candidates=()):
    mission2_pipe._on_perception_result(
        node, {"candidates": list(candidates)}, origin_state="OBSERVE"
    )


class EarlyAnswerTest(unittest.TestCase):
    def test_single_survivor_goes_straight_to_selection(self):
        node = _Node(survivors=[16])
        _observe(node, candidates=[{"object_id": 16}])
        self.assertEqual(node.state, "SELECT_TARGET")
        self.assertIn("SETTLED", node.logger.messages[-1][1])

    def test_two_survivors_keep_exploring(self):
        """후보가 갈리면 아직 구별할 정보가 부족하다 - unknown 쪽을 더 봐야 한다."""
        node = _Node(survivors=[5, 16])
        _observe(node)
        self.assertEqual(node.state, "PLAN_EXPLORATION")

    def test_no_survivor_keeps_exploring(self):
        node = _Node(survivors=[])
        _observe(node)
        self.assertEqual(node.state, "PLAN_EXPLORATION")

    def test_task_without_a_relation_chain_never_settles_early(self):
        """관계 제약이 없으면 '하나로 좁혀졌다'가 성립하지 않는다."""
        node = _Node(survivors=[], task={"target": "vase"})
        _observe(node, candidates=[{"object_id": 3}])
        self.assertEqual(node.state, "PLAN_EXPLORATION")

    def test_driving_observations_never_change_state(self):
        """주행 중 관측은 graph만 갱신한다 - 판단은 정지 상태에서만."""
        node = _Node(survivors=[16])
        node.state = "FOLLOW_EXPLORATION"
        mission2_pipe._on_perception_result(
            node, {"candidates": [{"object_id": 16}]}, origin_state="FOLLOW_EXPLORATION"
        )
        self.assertEqual(node.state, "FOLLOW_EXPLORATION")

    def test_exhausted_exploration_still_selects_when_unsettled(self):
        """조기 종료가 안 걸려도 탐사가 끝나면 기존대로 가진 것으로 고른다."""
        node = _Node(survivors=[5, 16])
        node.mission2_exploration_complete = False
        node.coverage_planner = types.SimpleNamespace(
            describe_last_plan_failure=lambda: "reason=no_frontier"
        )
        node.scene_graph.snapshot = lambda: {"objects": [{"category": "vase"}]}
        mission2_pipe._on_exploration_result(node, {"target": "vase"}, {"route": []})
        self.assertTrue(node.mission2_exploration_complete)
        self.assertEqual(node.state, "SELECT_TARGET")


if __name__ == "__main__":
    unittest.main()
