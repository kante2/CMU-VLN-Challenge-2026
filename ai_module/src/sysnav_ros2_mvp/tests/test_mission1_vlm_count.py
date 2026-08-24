"""Mission 1의 개수는 viewpoint 파노라마 한 장을 VLM에게 보여 센다.

왜 바꿨나: object_memory 기반 집계(len(candidates))는 **탐지 재현율에 갇힌다**.
실측(home_building_1, kante/fix_mission_1): pillow가 GT 18개인데 최종 메모리엔 7개만
남았다. 베개 4개 중 2개만 탐지되면 병합·필터를 아무리 손봐도 답은 영원히 2다.
이미지를 직접 보고 세면 그 상한을 우회한다.

뷰는 **한 장만** 쓴다 - scene_graph.best_viewpoint_for_objects()가 그 카테고리 물체를
가장 많이 동시에 본 viewpoint를 고른다. 여러 뷰의 개수를 합치면 같은 물체가 여러 뷰에
찍혀 중복 계산되는데, 뷰를 하나로 확정하면 그 문제가 구조적으로 사라진다.

fail-open이 아니라 fail-quiet이다: VLM이 실패하면 기존 기하 기반 개수를 그대로 쓴다.
개수 미션은 0/1 채점이라 "답을 못 냄"이 최악이다.
"""

import sys
import types
import unittest

if "rclpy" not in sys.modules:
    try:
        import rclpy  # noqa: F401
    except ImportError:                                       # pragma: no cover
        package = types.ModuleType("rclpy")
        logging_module = types.ModuleType("rclpy.logging")

        class _RclpyLogger:
            def info(self, *a, **k): pass
            def warning(self, *a, **k): pass

        logging_module.get_logger = lambda name: _RclpyLogger()
        package.logging = logging_module
        sys.modules["rclpy"] = package
        sys.modules["rclpy.logging"] = logging_module

# mission1_pipe는 /numerical_response 발행에만 std_msgs를 쓴다. 아래 테스트는 개수
# 산출부만 보므로 ROS 없는 환경에서도 import되도록 최소 stub으로 대체한다.
if "std_msgs.msg" not in sys.modules:
    try:
        import std_msgs.msg  # noqa: F401
    except ImportError:                                       # pragma: no cover
        package = types.ModuleType("std_msgs")
        module = types.ModuleType("std_msgs.msg")

        class _Int32:
            def __init__(self): self.data = 0

        module.Int32 = _Int32
        package.msg = module
        sys.modules.setdefault("std_msgs", package)
        sys.modules["std_msgs.msg"] = module

from sysnav.missions import mission1_pipe                      # noqa: E402
from sysnav.scene_graph.scene_graph_manager import SceneGraphManager  # noqa: E402


class _Logger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass


class _ObjectMemory:
    def __init__(self, nodes):
        self._nodes = nodes

    def find_by_category(self, category):
        return [dict(n) for n in self._nodes if n["category"] == category]


class _SceneGraph:
    def __init__(self, viewpoint):
        self.viewpoint = viewpoint
        self.asked_ids = None

    def best_viewpoint_for_objects(self, object_ids):
        self.asked_ids = list(object_ids)
        return self.viewpoint


class _VlmCounter:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def count(self, question, target, viewpoint):
        self.calls.append((question, target, viewpoint["viewpoint_id"]))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _Node:
    def __init__(self, objects, viewpoint, vlm_result):
        self.object_memory = _ObjectMemory(objects)
        self.scene_graph = _SceneGraph(viewpoint)
        self.vlm_counter = _VlmCounter(vlm_result)

    def get_logger(self):
        return _Logger()


_VIEWPOINT = {"viewpoint_id": 7, "image_path": "/tmp/vp7.jpg",
              "visible_object_ids": [1, 2], "visible_count": 2}
_TASK = {"target": "pillow", "raw": "How many pillows are on the sofa?"}


def _objects(count):
    return [{"object_id": i, "category": "pillow"} for i in range(1, count + 1)]


class VlmCountTest(unittest.TestCase):
    def test_vlm_count_overrides_the_geometric_count(self):
        """VLM이 세면 그 값을 쓴다 - 기하 집계보다 크게 나오는 게 이 변경의 목적이다."""
        node = _Node(_objects(2), _VIEWPOINT, vlm_result=5)
        self.assertEqual(mission1_pipe._count_with_vlm(node, _TASK, 2), 5)
        self.assertEqual(node.vlm_counter.calls[0][:2], (_TASK["raw"], "pillow"))

    def test_the_question_is_passed_verbatim(self):
        """"on the sofa" 같은 제약을 우리가 재구성하면 뉘앙스가 깎인다 - 원문 그대로
        넘겨 VLM이 이미지에서 직접 보게 한다."""
        node = _Node(_objects(1), _VIEWPOINT, vlm_result=3)
        mission1_pipe._count_with_vlm(node, _TASK, 1)
        self.assertEqual(node.vlm_counter.calls[0][0], _TASK["raw"])

    def test_view_selection_uses_every_mapped_object_of_the_category(self):
        """뷰 선정은 relation/attribute 필터 전의 카테고리 전체로 한다 - 걸러진 물체도
        이미지에는 찍혀 있고, 그게 바로 VLM이 대신 세줘야 하는 대상이다."""
        node = _Node(_objects(4), _VIEWPOINT, vlm_result=4)
        mission1_pipe._count_with_vlm(node, _TASK, 1)
        self.assertEqual(node.scene_graph.asked_ids, [1, 2, 3, 4])

    def test_zero_is_a_valid_vlm_answer(self):
        """0을 None(=실패)과 헷갈리면 "하나도 없다"를 기하 집계로 덮어써 버린다."""
        node = _Node(_objects(2), _VIEWPOINT, vlm_result=0)
        self.assertEqual(mission1_pipe._count_with_vlm(node, _TASK, 2), 0)

    def test_no_viewpoint_falls_back_quietly(self):
        node = _Node(_objects(2), None, vlm_result=9)
        self.assertIsNone(mission1_pipe._count_with_vlm(node, _TASK, 2))

    def test_no_mapped_object_of_the_category_falls_back(self):
        node = _Node([], _VIEWPOINT, vlm_result=9)
        self.assertIsNone(mission1_pipe._count_with_vlm(node, _TASK, 0))

    def test_vlm_returning_none_falls_back(self):
        node = _Node(_objects(2), _VIEWPOINT, vlm_result=None)
        self.assertIsNone(mission1_pipe._count_with_vlm(node, _TASK, 2))

    def test_a_task_without_a_raw_question_falls_back(self):
        node = _Node(_objects(2), _VIEWPOINT, vlm_result=9)
        self.assertIsNone(
            mission1_pipe._count_with_vlm(node, {"target": "pillow", "raw": ""}, 2)
        )


class BestViewpointTest(unittest.TestCase):
    """scene_graph.best_viewpoint_for_objects - "가장 많이 동시에 본" 뷰 하나."""

    def _graph(self, viewpoints):
        graph = SceneGraphManager.__new__(SceneGraphManager)
        import threading
        graph._lock = threading.RLock()
        graph._viewpoints = viewpoints
        return graph

    def test_picks_the_viewpoint_that_sees_the_most(self):
        graph = self._graph({
            1: {"observed_object_ids": [1], "image_path": "/a.jpg"},
            2: {"observed_object_ids": [1, 2, 3], "image_path": "/b.jpg"},
            3: {"observed_object_ids": [2], "image_path": "/c.jpg"},
        })
        best = graph.best_viewpoint_for_objects([1, 2, 3])
        self.assertEqual(best["viewpoint_id"], 2)
        self.assertEqual(best["visible_count"], 3)
        self.assertEqual(best["image_path"], "/b.jpg")

    def test_viewpoints_without_an_image_are_unusable(self):
        """이미지가 없으면 VLM에 넣을 수 없다 - 더 많이 봤어도 못 쓴다."""
        graph = self._graph({
            1: {"observed_object_ids": [1, 2, 3], "image_path": None},
            2: {"observed_object_ids": [1], "image_path": "/b.jpg"},
        })
        best = graph.best_viewpoint_for_objects([1, 2, 3])
        self.assertEqual(best["viewpoint_id"], 2)

    def test_counts_only_the_requested_objects(self):
        """다른 카테고리 물체를 많이 본 뷰가 이기면 안 된다."""
        graph = self._graph({
            1: {"observed_object_ids": [1, 2], "image_path": "/a.jpg"},
            2: {"observed_object_ids": [8, 9, 10, 11], "image_path": "/b.jpg"},
        })
        best = graph.best_viewpoint_for_objects([1, 2])
        self.assertEqual(best["viewpoint_id"], 1)

    def test_no_overlap_returns_none(self):
        graph = self._graph({1: {"observed_object_ids": [9], "image_path": "/a.jpg"}})
        self.assertIsNone(graph.best_viewpoint_for_objects([1, 2]))

    def test_empty_request_returns_none(self):
        graph = self._graph({1: {"observed_object_ids": [1], "image_path": "/a.jpg"}})
        self.assertIsNone(graph.best_viewpoint_for_objects([]))


class MajorityCountTest(unittest.TestCase):
    """self-consistency 투표 - temperature=0.0인데도 응답이 결정적이지 않다."""

    def setUp(self):
        from sysnav.reasoning.vlm_counter import VlmCounter
        self.vote = VlmCounter._majority_count

    def test_takes_the_most_frequent(self):
        self.assertEqual(self.vote([2, 3, 3, 3, 2]), 3)

    def test_a_tie_takes_the_smaller_count(self):
        """동률을 임의로 깨면 같은 입력에 답이 흔들려 투표의 의미가 없어진다. 작은 쪽인
        이유는 파노라마의 흔한 오류가 좌우 wrap 구간 중복 계수(과다)이기 때문이다."""
        self.assertEqual(self.vote([2, 2, 3, 3]), 2)

    def test_single_sample(self):
        self.assertEqual(self.vote([4]), 4)


if __name__ == "__main__":
    unittest.main()
