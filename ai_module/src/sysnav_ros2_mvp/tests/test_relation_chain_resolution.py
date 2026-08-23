"""SceneGraphManager.resolve_relation_chain - 체인을 전부 만족하는 후보 추리기.

"Find the vase on the cabinet below the picture"는 두 hop이다:
  vase --on--> cabinet --under--> picture
두 hop이 모두 임계값을 넘어야 후보로 살아남는다. 실제 씬(livingroom_1)에서 vase는
여러 개고 cabinet도 여러 개라, hop 하나만 보면 엉뚱한 화병이 살아남는다.
"""

import sys
import types
import unittest

for name in ("cv2", "numpy"):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:                       # pragma: no cover
            sys.modules[name] = types.ModuleType(name)

from sysnav import config                                          # noqa: E402
from sysnav.scene_graph.scene_graph_manager import SceneGraphManager  # noqa: E402


TASK = {
    "target": "vase",
    "relation": "on",
    "reference_objects": ["cabinet"],
    "relation_chain": [("vase", "on", "cabinet"), ("cabinet", "under", "picture")],
}
HIGH = config.SCENE_GRAPH_RELATION_MIN_CONFIDENCE + 0.10
LOW = config.SCENE_GRAPH_RELATION_MIN_CONFIDENCE - 0.10


def _graph(objects, edges):
    """lock/저장 구조만 흉내 낸 최소 그래프 (ROS 초기화 없이 조회부만 검증)."""
    manager = SceneGraphManager.__new__(SceneGraphManager)
    import threading
    manager._lock = threading.RLock()
    manager._objects = {i: {"category": c} for i, c in objects.items()}
    manager._edges = {
        f"e{index}": {
            "edge_type": "object_object",
            "source": f"object_{source}",
            "targets": [f"object_{target}"],
            "relation": relation,
            "metadata": {"confidence": confidence},
        }
        for index, (source, relation, target, confidence) in enumerate(edges)
    }
    return manager


class ResolveChainTest(unittest.TestCase):
    def test_full_chain_survives(self):
        graph = _graph(
            {1: "vase", 2: "cabinet", 3: "picture"},
            [(1, "on", 2, HIGH), (2, "under", 3, HIGH)],
        )
        self.assertEqual(graph.resolve_relation_chain(TASK), [1])

    def test_broken_second_hop_drops_the_candidate(self):
        """실측 케이스: vase#5가 cabinet#1 위에 있는 건 맞지만, cabinet#1은 방
        반대편이라 어떤 picture 아래도 아니다."""
        graph = _graph(
            {1: "vase", 2: "cabinet", 3: "picture"},
            [(1, "on", 2, HIGH)],                      # 두 번째 hop 없음
        )
        self.assertEqual(graph.resolve_relation_chain(TASK), [])

    def test_low_confidence_hop_is_ignored(self):
        graph = _graph(
            {1: "vase", 2: "cabinet", 3: "picture"},
            [(1, "on", 2, HIGH), (2, "under", 3, LOW)],
        )
        self.assertEqual(graph.resolve_relation_chain(TASK), [])

    def test_only_the_candidate_with_a_complete_chain_survives(self):
        graph = _graph(
            {1: "vase", 2: "cabinet", 3: "picture",
             4: "vase", 5: "cabinet"},
            [(1, "on", 2, HIGH), (2, "under", 3, HIGH),   # 완전한 체인
             (4, "on", 5, HIGH)],                         # cabinet#5는 picture 아래가 아님
        )
        self.assertEqual(graph.resolve_relation_chain(TASK), [1])

    def test_two_complete_chains_return_both(self):
        """둘 다 성립하면 아직 못 고른다 - 호출 측이 '계속 탐사'로 처리한다."""
        graph = _graph(
            {1: "vase", 2: "cabinet", 3: "picture",
             4: "vase", 5: "cabinet", 6: "picture"},
            [(1, "on", 2, HIGH), (2, "under", 3, HIGH),
             (4, "on", 5, HIGH), (5, "under", 6, HIGH)],
        )
        self.assertEqual(graph.resolve_relation_chain(TASK), [1, 4])

    def test_wrong_category_at_a_hop_breaks_the_chain(self):
        graph = _graph(
            {1: "vase", 2: "cabinet", 3: "sofa"},
            [(1, "on", 2, HIGH), (2, "under", 3, HIGH)],   # picture가 아니라 sofa
        )
        self.assertEqual(graph.resolve_relation_chain(TASK), [])

    def test_task_without_a_chain_returns_nothing(self):
        graph = _graph({1: "vase"}, [])
        self.assertEqual(graph.resolve_relation_chain({"target": "vase"}), [])

    def test_single_hop_task(self):
        graph = _graph(
            {1: "vase", 2: "cabinet", 4: "vase"},
            [(1, "on", 2, HIGH)],
        )
        task = {"target": "vase", "relation": "on", "reference_objects": ["cabinet"]}
        self.assertEqual(graph.resolve_relation_chain(task), [1])


if __name__ == "__main__":
    unittest.main()
