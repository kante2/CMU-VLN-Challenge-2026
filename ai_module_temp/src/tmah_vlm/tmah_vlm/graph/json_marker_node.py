#!/usr/bin/env python3
"""Publish saved scene_graph_latest.json objects as RViz MarkerArray nodes."""

import json
import os

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from tmah_vlm import config


def _as_float_triplet(value, fallback):
    try:
        if len(value) < 3:
            return fallback
        return (float(value[0]), float(value[1]), float(value[2]))
    except Exception:
        return fallback


def _color_for_name(name):
    seed = sum(ord(ch) for ch in str(name))
    palette = [
        (1.0, 0.75, 0.10),
        (0.20, 0.85, 1.00),
        (0.60, 1.00, 0.35),
        (1.00, 0.45, 0.85),
        (0.85, 0.65, 1.00),
        (1.00, 0.55, 0.25),
    ]
    return palette[seed % len(palette)]


class SceneGraphJsonMarkerNode(Node):
    """Small RViz helper that visualizes already saved graph objects."""

    def __init__(self):
        super().__init__("scene_graph_json_markers")
        default_path = os.path.join(config.DEBUG_DIR, "scene_graph_latest.json")

        self.declare_parameter("graph_path", default_path)
        self.declare_parameter("topic", config.TOPIC_SCENE_GRAPH_JSON_MARKERS)
        self.declare_parameter("frame_id", config.FRAME_MAP)
        self.declare_parameter("publish_period_sec", 1.0)
        self.declare_parameter("show_edges", True)
        self.declare_parameter("show_captions", True)

        self.graph_path = self.get_parameter("graph_path").value
        self.topic = self.get_parameter("topic").value
        self.frame_id = self.get_parameter("frame_id").value
        period = float(self.get_parameter("publish_period_sec").value)

        self.pub = self.create_publisher(MarkerArray, self.topic, 5)
        self.timer = self.create_timer(max(period, 0.2), self.publish_markers)
        self._missing_logged = False

        self.get_logger().info(
            f"publishing saved scene graph markers: {self.topic} from {self.graph_path}"
        )

    def publish_markers(self):
        if not os.path.exists(self.graph_path):
            if not self._missing_logged:
                self.get_logger().warn(f"scene graph json not found yet: {self.graph_path}")
                self._missing_logged = True
            return

        try:
            with open(self.graph_path, "r", encoding="utf-8") as f:
                graph = json.load(f)
        except Exception as exc:
            self.get_logger().warn(f"failed to read scene graph json: {exc}")
            return

        objects = graph.get("objects", {})
        edges = graph.get("edges", [])
        stamp = self.get_clock().now().to_msg()

        markers = MarkerArray()
        markers.markers.append(self._delete_all(stamp))

        marker_id = 1
        if bool(self.get_parameter("show_edges").value):
            for edge in edges:
                marker = self._make_edge_marker(stamp, marker_id, objects, edge)
                marker_id += 1
                if marker is not None:
                    markers.markers.append(marker)

        for object_id, obj in objects.items():
            markers.markers.append(self._make_object_node(stamp, marker_id, object_id, obj))
            marker_id += 1
            markers.markers.append(self._make_object_label(stamp, marker_id, object_id, obj))
            marker_id += 1

        self.pub.publish(markers)

    def _delete_all(self, stamp):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = "scene_graph_json"
        marker.id = 0
        marker.action = Marker.DELETEALL
        return marker

    def _make_object_node(self, stamp, marker_id, object_id, obj):
        center = _as_float_triplet(obj.get("center", (0.0, 0.0, 0.0)), (0.0, 0.0, 0.0))
        size = _as_float_triplet(obj.get("size", (0.35, 0.35, 0.35)), (0.35, 0.35, 0.35))
        radius = max(0.18, min(max(size), 0.55))
        r, g, b = _color_for_name(obj.get("name", object_id))

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = "scene_graph_json_nodes"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2]
        marker.pose.orientation.w = 1.0
        marker.scale.x = radius
        marker.scale.y = radius
        marker.scale.z = radius
        marker.color.a = 0.92
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        return marker

    def _make_object_label(self, stamp, marker_id, object_id, obj):
        center = _as_float_triplet(obj.get("center", (0.0, 0.0, 0.0)), (0.0, 0.0, 0.0))
        size = _as_float_triplet(obj.get("size", (0.35, 0.35, 0.35)), (0.35, 0.35, 0.35))
        observations = obj.get("observations", [])
        name = str(obj.get("name", object_id))
        caption = str(obj.get("caption", "")).strip()

        lines = [f"{name}  [{len(observations)}]", object_id]
        if bool(self.get_parameter("show_captions").value) and caption:
            lines.append(caption[:80])

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = "scene_graph_json_labels"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2] + max(size[2] * 0.5, 0.35)
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.22
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.text = "\n".join(lines)
        return marker

    def _make_edge_marker(self, stamp, marker_id, objects, edge):
        source = objects.get(edge.get("source", ""))
        target = objects.get(edge.get("target", ""))
        if source is None or target is None:
            return None

        source_center = _as_float_triplet(source.get("center", (0.0, 0.0, 0.0)), (0.0, 0.0, 0.0))
        target_center = _as_float_triplet(target.get("center", (0.0, 0.0, 0.0)), (0.0, 0.0, 0.0))

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = "scene_graph_json_edges"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.025
        marker.color.a = 0.35
        marker.color.r = 0.7
        marker.color.g = 0.7
        marker.color.b = 1.0
        marker.points.append(self._point(source_center, z_offset=0.12))
        marker.points.append(self._point(target_center, z_offset=0.12))
        return marker

    @staticmethod
    def _point(center, z_offset=0.0):
        from geometry_msgs.msg import Point

        return Point(x=center[0], y=center[1], z=center[2] + z_offset)


def main(args=None):
    rclpy.init(args=args)
    node = SceneGraphJsonMarkerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
