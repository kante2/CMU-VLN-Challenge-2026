"""Scene graph의 object node들을 RViz MarkerArray(CUBE + 라벨)로 변환."""

from __future__ import annotations

from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker, MarkerArray

from sysnav import config

_DEFAULT_COLOR = (0.2, 0.5, 1.0, 0.45)
_SELECTED_COLOR = (0.1, 0.9, 0.2, 0.65)
_NAMESPACE = "sysnav_objects"


def build_object_marker_array(
    objects: list[dict],
    selected_object_id: int | None,
    stamp: Time,
) -> MarkerArray:
    array = MarkerArray()

    # 이전 사이클에서 사라진(더 이상 관측 안 되는) object의 marker를 지우기 위해
    # 매번 DELETEALL을 먼저 보내고 현재 object들만 다시 그린다.
    clear = Marker()
    clear.header.frame_id = config.OBJECT_MARKER_FRAME_ID
    clear.header.stamp = stamp
    clear.ns = _NAMESPACE
    clear.action = Marker.DELETEALL
    array.markers.append(clear)

    for obj in objects:
        object_id = int(obj["object_id"])
        position = obj.get("position", (0.0, 0.0, 0.0))
        extent = obj.get("extent_3d", (0.0, 0.0, 0.0))
        size = [value if value > 0.01 else config.OBJECT_MARKER_DEFAULT_SIZE_M for value in extent]
        is_selected = selected_object_id is not None and object_id == int(selected_object_id)
        color = _SELECTED_COLOR if is_selected else _DEFAULT_COLOR

        box = Marker()
        box.header.frame_id = config.OBJECT_MARKER_FRAME_ID
        box.header.stamp = stamp
        box.ns = _NAMESPACE
        box.id = object_id * 2
        box.type = Marker.CUBE
        box.action = Marker.ADD
        box.pose.position.x = float(position[0])
        box.pose.position.y = float(position[1])
        box.pose.position.z = float(position[2])
        box.pose.orientation.w = 1.0
        box.scale.x, box.scale.y, box.scale.z = (float(value) for value in size)
        box.color.r, box.color.g, box.color.b, box.color.a = color
        array.markers.append(box)

        label = Marker()
        label.header.frame_id = config.OBJECT_MARKER_FRAME_ID
        label.header.stamp = stamp
        label.ns = _NAMESPACE
        label.id = object_id * 2 + 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = float(position[0])
        label.pose.position.y = float(position[1])
        label.pose.position.z = float(position[2]) + float(size[2]) / 2.0 + 0.15
        label.pose.orientation.w = 1.0
        label.scale.z = 0.2
        label.color.r, label.color.g, label.color.b, label.color.a = (1.0, 1.0, 1.0, 1.0)
        label.text = f"{obj.get('category', '?')}#{object_id}"
        array.markers.append(label)

    return array
