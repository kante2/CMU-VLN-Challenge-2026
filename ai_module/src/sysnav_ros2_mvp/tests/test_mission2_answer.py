"""Object Reference 답안(/selected_object_marker)의 수명 규칙.

README 채점:
  **Object Reference** (/2): Marker must be published on `/selected_object_marker`,
  and is scored based on its degree of overlap with the ground truth bounding box.

점수를 정하는 것은 발행한 marker의 bbox 겹침뿐이다. 로봇이 물체 앞까지 갔는지는
채점 항목이 아니다. 여기서 고정하는 것:

  1. 답을 낸 뒤의 주행 실패는 FAILED가 아니다.
  2. 답은 한 번만 내고 끝내지 않는다 (publisher가 VOLATILE이라 유실될 수 있다).
  3. 가까이 가서 bbox가 정밀해지면 그 값으로 다시 낸다 - 그게 주행의 목적이다.
"""

import sys
import time
import types
import unittest
from collections import deque

_module_name = "sysnav.scene_graph.scene_graph_rviz"
_previous = sys.modules.get(_module_name)
_stub = types.ModuleType(_module_name)
_stub.build_selected_object_marker = lambda obj, stamp: {
    "id": obj["object_id"], "scale": tuple(obj.get("extent_3d", ())),
}
sys.modules[_module_name] = _stub
from sysnav import config                      # noqa: E402
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


class _Memory:
    def __init__(self, obj): self.obj = obj
    def get(self, object_id):
        return self.obj if self.obj and self.obj["object_id"] == object_id else None


class _Publisher:
    def __init__(self): self.sent = []
    def publish(self, marker): self.sent.append(marker)


class _Clock:
    @staticmethod
    def now(): return types.SimpleNamespace(to_msg=lambda: None)


_VASE = {"object_id": 15, "category": "vase",
         "position": (-2.37, -7.11, 0.93), "extent_3d": (0.20, 0.20, 0.30)}


class _Node:
    def __init__(self, obj=None):
        self.state_lock = _Lock()
        self.state = "NAVIGATE_TARGET"
        self.exploration_route = deque()
        self.mission2_exploration_complete = True
        self.mission2_answer_object_id = None
        self._mission2_answer_extent = None
        self._mission2_last_answer_publish = None
        self.object_memory = _Memory(obj if obj is not None else dict(_VASE))
        self.selected_object_marker_pub = _Publisher()
        self.last_response_summary = None
        self.target_object_id = 15
        self.logger = _Logger()
        self.cleared = False

    def get_logger(self): return self.logger
    def get_clock(self): return _Clock()
    def clear_target_navigation(self): self.cleared = True
    def distance_to_target(self, pose): return 0.42


class NavigationFailureKeepsTheAnswerTest(unittest.TestCase):
    def test_unreachable_after_answering_is_not_a_failure(self):
        """실측(2026-08-23): vase #15를 맞게 골라 발행하고도, 접근 지점 재선정이 5초간
        실패하자 FAILED로 처리했다. 캐비닛 위 물체라 base autonomy가 받아줄 지점이
        없었던 것뿐이고, 채점상으로는 이미 딴 점수였다."""
        node = _Node()
        mission2_pipe._publish_answer(node, dict(_VASE))
        mission2_pipe._give_up_target(node)
        self.assertEqual(node.state, "SUCCESS")
        self.assertTrue(node.cleared)
        self.assertIn("answer stands", node.logger.messages[-1][1])

    def test_unreachable_before_answering_is_still_a_failure(self):
        node = _Node()
        mission2_pipe._give_up_target(node)
        self.assertEqual(node.state, "FAILED")

    def test_unreachable_before_full_exploration_returns_to_exploring(self):
        node = _Node()
        node.mission2_exploration_complete = False
        mission2_pipe._give_up_target(node)
        self.assertEqual(node.state, "PLAN_EXPLORATION")


class AnswerRepublishTest(unittest.TestCase):
    def test_publishing_records_what_was_answered(self):
        node = _Node()
        mission2_pipe._publish_answer(node, dict(_VASE))
        self.assertEqual(node.mission2_answer_object_id, 15)
        self.assertEqual(len(node.selected_object_marker_pub.sent), 1)
        self.assertIn("vase", node.last_response_summary)

    def test_refresh_republishes_once_the_interval_has_passed(self):
        node = _Node()
        mission2_pipe._publish_answer(node, dict(_VASE))
        node._mission2_last_answer_publish = (
            time.monotonic() - config.MISSION2_ANSWER_REPUBLISH_SEC - 0.01
        )
        mission2_pipe._refresh_answer(node)
        self.assertEqual(len(node.selected_object_marker_pub.sent), 2)

    def test_refresh_is_throttled(self):
        node = _Node()
        mission2_pipe._publish_answer(node, dict(_VASE))
        for _ in range(10):
            mission2_pipe._refresh_answer(node)
        self.assertEqual(len(node.selected_object_marker_pub.sent), 1,
                         "주기 안에서는 다시 쏘지 않는다")

    def test_refresh_does_nothing_before_an_answer_exists(self):
        node = _Node()
        mission2_pipe._refresh_answer(node)
        self.assertEqual(node.selected_object_marker_pub.sent, [])

    def test_refined_bbox_is_republished_with_the_new_extent(self):
        """주행의 목적. 가까이서 다시 보면 extent_3d가 정밀해지고, 채점이 bbox 겹침이라
        그 값을 내보내야 점수가 오른다."""
        node = _Node()
        mission2_pipe._publish_answer(node, dict(_VASE))
        node.object_memory.obj["extent_3d"] = (0.24, 0.23, 0.34)   # 가까이서 재관측
        node._mission2_last_answer_publish = None
        mission2_pipe._refresh_answer(node)
        self.assertEqual(node.selected_object_marker_pub.sent[-1]["scale"],
                         (0.24, 0.23, 0.34))
        self.assertIn("refined", node.logger.messages[-1][1])

    def test_arrival_publishes_the_closest_observation(self):
        node = _Node()
        mission2_pipe._publish_answer(node, dict(_VASE))
        node.object_memory.obj["extent_3d"] = (0.25, 0.25, 0.35)
        mission2_pipe._finish_navigate_target(node, {"x": -1.4, "y": -7.4})
        self.assertEqual(node.state, "SUCCESS")
        self.assertEqual(node.selected_object_marker_pub.sent[-1]["scale"],
                         (0.25, 0.25, 0.35), "도착 시점 관측이 가장 정확하다")


if __name__ == "__main__":
    unittest.main()
