"""채점 답안 marker(`/selected_object_marker`)의 형식.

README: "Marker message on topic `/selected_object_marker`, containing **object label**
and bounding box of the selected object."

Marker 메시지에서 라벨이 들어갈 자리는 `ns`와 `text`뿐이고, 대회가 준 참조 구현
(dummy_vlm/src/dummyVLM.cpp `pubObjectMarker`)은 `ns = objLabel`만 채운다 - `text`는
아예 안 건드린다. 게다가 우리가 쓰는 CUBE 타입은 `text`를 화면에 그리지도 않는다.
비공개 평가 노드가 dummy와 같은 자리에서 라벨을 읽는다면 `ns`가 유일한 통로다.
예전에는 여기에 "selected_object"라는 고정 문자열이 들어가서 무엇을 골랐든 라벨이
같았고, 실제 카테고리는 아무도 안 보는 `text`에만 있었다.

visualization_msgs가 필요해서 ROS가 없는 호스트에서는 통째로 skip된다(컨테이너에서 돈다).
"""

import unittest

try:
    from sysnav.scene_graph.scene_graph_rviz import (
        build_selected_object_delete_marker,
        build_selected_object_marker,
        selected_object_namespace,
    )
    from visualization_msgs.msg import Marker
    _ROS = True
except ImportError:  # pragma: no cover - 호스트에는 ROS가 없다
    _ROS = False


_VASE = {"object_id": 15, "category": "vase", "position": (-2.37, -7.11, 0.93),
         "extent_3d": (0.2, 0.2, 0.3)}


@unittest.skipUnless(_ROS, "visualization_msgs 필요 (컨테이너에서 실행)")
class SelectedObjectMarkerFormatTest(unittest.TestCase):
    def test_label_goes_into_ns_like_the_reference_implementation(self):
        marker = build_selected_object_marker(dict(_VASE), None)
        self.assertEqual(marker.ns, "vase")

    def test_label_is_also_mirrored_into_text(self):
        """어느 쪽을 읽든 맞도록 둘 다 채운다. CUBE에서 text는 안 그려지므로 화면은
        그대로다."""
        self.assertEqual(build_selected_object_marker(dict(_VASE), None).text, "vase")

    def test_missing_category_falls_back_to_a_non_empty_label(self):
        """ns가 비면 구독자 쪽에서 marker 식별이 모호해진다 - 빈 문자열은 안 된다."""
        for broken in ({}, {"category": None}, {"category": "   "}):
            self.assertEqual(selected_object_namespace(broken), "unknown")

    def test_delete_marker_targets_the_same_identity(self):
        """marker의 정체성은 (ns, id)다. 지우려면 그 둘이 ADD 때와 같아야 한다."""
        added = build_selected_object_marker(dict(_VASE), None)
        deleted = build_selected_object_delete_marker("vase", None)
        self.assertEqual((deleted.ns, deleted.id), (added.ns, added.id))
        self.assertEqual(deleted.action, Marker.DELETE)


if __name__ == "__main__":
    unittest.main()
