#!/usr/bin/env python3
"""
RViz visualization for the online scene graph.

The challenge marker topic stays untouched. This file publishes a separate
MarkerArray showing every object node currently stored in the graph.
"""

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from tmah_vlm import config
from tmah_vlm.bbox3d.wireframe import wireframe_edge_points


def publish_scene_graph_markers(node):
    """Publish all object nodes in node.scene_graph as RViz markers."""
    if not hasattr(node, "scene_graph") or node.scene_graph is None:
        return
    if not hasattr(node, "scene_graph_marker_pub"):
        return

    graph = node.scene_graph
    stamp = node.get_clock().now().to_msg()

    marker_array = MarkerArray()
    marker_array.markers.append(make_delete_all_marker(stamp))

    marker_id = 1
    for edge in graph.edges:
        marker_array.markers.append(make_relation_edge_marker(stamp, marker_id, graph, edge))
        marker_id += 1

    for object_node in graph.objects.values():
        marker_array.markers.append(make_object_wireframe_marker(stamp, marker_id, object_node))
        marker_id += 1
        marker_array.markers.append(make_object_label_marker(stamp, marker_id, object_node))
        marker_id += 1
        marker_array.markers.append(make_object_center_marker(stamp, marker_id, object_node))
        marker_id += 1

    node.scene_graph_marker_pub.publish(marker_array)


def make_delete_all_marker(stamp):
    marker = Marker()
    marker.header.frame_id = config.FRAME_MAP
    marker.header.stamp = stamp
    marker.ns = "scene_graph"
    marker.id = 0
    marker.action = Marker.DELETEALL
    return marker


def make_relation_edge_marker(stamp, marker_id, graph, edge):
    marker = Marker()
    marker.header.frame_id = config.FRAME_MAP
    marker.header.stamp = stamp
    marker.ns = "scene_graph_relation"
    marker.id = marker_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.012

    marker.color.a = 0.55
    marker.color.r = 0.8
    marker.color.g = 0.55
    marker.color.b = 1.0

    source = graph.objects.get(edge.source)
    target = graph.objects.get(edge.target)
    if source is None or target is None:
        return marker

    marker.points = [
        Point(x=float(source.center[0]), y=float(source.center[1]), z=float(source.center[2]) + 0.08),
        Point(x=float(target.center[0]), y=float(target.center[1]), z=float(target.center[2]) + 0.08),
    ]
    return marker


def make_object_wireframe_marker(stamp, marker_id, object_node):
    marker = Marker()
    marker.header.frame_id = config.FRAME_MAP
    marker.header.stamp = stamp
    marker.ns = "scene_graph_bbox"
    marker.id = marker_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.025

    marker.color.a = 1.0
    marker.color.r = 0.2
    marker.color.g = 0.8
    marker.color.b = 1.0

    marker.points = [
        Point(x=float(x), y=float(y), z=float(z))
        for x, y, z in wireframe_edge_points(object_node.center, object_node.size)
    ]
    return marker


def make_object_label_marker(stamp, marker_id, object_node):
    marker = Marker()
    marker.header.frame_id = config.FRAME_MAP
    marker.header.stamp = stamp
    marker.ns = "scene_graph_label"
    marker.id = marker_id
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD

    x, y, z = object_node.center
    sx, sy, sz = object_node.size
    marker.pose.position.x = float(x)
    marker.pose.position.y = float(y)
    marker.pose.position.z = float(z) + max(float(sz) * 0.5, 0.25)
    marker.pose.orientation.w = 1.0

    marker.scale.z = 0.25
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 1.0

    marker.text = f"{object_node.name} ({len(object_node.observations)})"
    return marker


def make_object_center_marker(stamp, marker_id, object_node):
    marker = Marker()
    marker.header.frame_id = config.FRAME_MAP
    marker.header.stamp = stamp
    marker.ns = "scene_graph_center"
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD

    marker.pose.position.x = float(object_node.center[0])
    marker.pose.position.y = float(object_node.center[1])
    marker.pose.position.z = float(object_node.center[2])
    marker.pose.orientation.w = 1.0

    marker.scale.x = 0.12
    marker.scale.y = 0.12
    marker.scale.z = 0.12
    marker.color.a = 0.9
    marker.color.r = 1.0
    marker.color.g = 0.8
    marker.color.b = 0.1
    return marker
