"""RViz markers for resolved Mission 3 destinations."""

from __future__ import annotations

import colorsys

from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker, MarkerArray

from sysnav import config

_NAMESPACE = "sysnav_mission3_steps"


def build_step_marker_array(destinations: list[dict], stamp: Time) -> MarkerArray:
    """Build numbered, color-coded destination markers for resolved steps."""
    array = MarkerArray()

    clear = Marker()
    clear.header.frame_id = config.OBJECT_MARKER_FRAME_ID
    clear.header.stamp = stamp
    clear.ns = _NAMESPACE
    clear.action = Marker.DELETEALL
    array.markers.append(clear)

    for destination in destinations:
        step_index = int(destination["step_index"])
        position = destination["position"]
        red, green, blue = colorsys.hsv_to_rgb((step_index * 0.21) % 1.0, 0.8, 1.0)

        sphere = Marker()
        sphere.header.frame_id = config.OBJECT_MARKER_FRAME_ID
        sphere.header.stamp = stamp
        sphere.ns = _NAMESPACE
        sphere.id = step_index * 2
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(position[0])
        sphere.pose.position.y = float(position[1])
        sphere.pose.position.z = float(position[2]) if len(position) > 2 else 0.25
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.45
        sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = (
            red, green, blue, 0.9
        )
        array.markers.append(sphere)

        label = Marker()
        label.header.frame_id = config.OBJECT_MARKER_FRAME_ID
        label.header.stamp = stamp
        label.ns = _NAMESPACE
        label.id = step_index * 2 + 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = float(position[0])
        label.pose.position.y = float(position[1])
        label.pose.position.z = sphere.pose.position.z + 0.45
        label.pose.orientation.w = 1.0
        label.scale.z = 0.28
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = f"STEP {step_index + 1}: {destination['label']}"
        array.markers.append(label)

    return array
