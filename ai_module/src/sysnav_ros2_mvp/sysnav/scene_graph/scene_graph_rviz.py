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


_SELECTED_OBJECT_FALLBACK_NAMESPACE = "unknown"


def selected_object_namespace(obj: dict) -> str:
    """채점용 marker의 `ns`에 넣을 물체 라벨.

    README: "Marker message ... containing **object label** and bounding box of the
    selected object." 그런데 Marker 메시지에서 라벨을 담을 자리는 `ns`와 `text`뿐이고,
    대회가 준 참조 구현(dummy_vlm/src/dummyVLM.cpp `pubObjectMarker`)은 `ns = objLabel`
    하나만 채운다(`text`는 아예 안 건드린다). 즉 비공개 평가 노드가 라벨을 읽는다면
    `ns`에서 읽을 가능성이 가장 크다. 예전에는 여기에 "selected_object"라는 고정
    문자열이 들어가서, 무엇을 골랐든 라벨이 항상 같았다.

    빈 문자열은 안 된다 - ns가 비면 RViz/구독자 쪽에서 marker 식별이 모호해진다."""
    category = str(obj.get("category", "") or "").strip()
    return category or _SELECTED_OBJECT_FALLBACK_NAMESPACE


def build_selected_object_delete_marker(namespace: str, stamp: Time) -> Marker:
    """이전 답안 marker를 지우는 DELETE marker.

    ns에 라벨을 넣으면서 필요해졌다: marker의 정체성은 (ns, id)이므로 답을 다른
    카테고리로 바꾸면 **새 ns에 새 marker가 생기고 옛 marker는 그대로 남는다**
    (같은 ns에 다시 쓸 때는 덮어써져서 이런 문제가 없었다). dummy_vlm도 같은 이유로
    답을 바꿀 때 delObjectMarker()로 이전 ns를 DELETE한다."""
    marker = Marker()
    marker.header.frame_id = config.OBJECT_MARKER_FRAME_ID
    marker.header.stamp = stamp
    marker.ns = namespace
    marker.id = 0
    marker.type = Marker.CUBE
    marker.action = Marker.DELETE
    return marker


def build_selected_object_marker(obj: dict, stamp: Time) -> Marker:
    """`/selected_object_marker`(Marker 단수) 발행용 - Object Reference 채점 토픽.
    CLAUDE.md 확인 사항: 이 토픽은 dummy_vlm과 챌린지 visualizationTools가 이미
    Marker(단수) 타입으로 고정 구독 중이라 MarkerArray로 바꾸면 안 된다."""
    position = obj.get("position", (0.0, 0.0, 0.0))
    extent = obj.get("extent_3d", (0.0, 0.0, 0.0))
    size = [value if value > 0.01 else config.OBJECT_MARKER_DEFAULT_SIZE_M for value in extent]

    marker = Marker()
    marker.header.frame_id = config.OBJECT_MARKER_FRAME_ID
    marker.header.stamp = stamp
    # ns/text 둘 다 라벨로 채운다 - 어느 쪽을 읽든 맞도록. CUBE marker에서 text는
    # 화면에 안 그려지므로 시각적으로 달라지는 것은 없다(selected_object_namespace 주석 참고).
    marker.ns = selected_object_namespace(obj)
    marker.id = 0
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = float(position[0])
    marker.pose.position.y = float(position[1])
    marker.pose.position.z = float(position[2])
    marker.pose.orientation.w = 1.0
    marker.scale.x, marker.scale.y, marker.scale.z = (float(value) for value in size)
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = _SELECTED_COLOR
    marker.text = selected_object_namespace(obj)
    return marker
