#!/usr/bin/env python3
"""Runtime bridge from the live scene graph to SORT3D-lite reasoning."""

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

from tmah_vlm import config
from tmah_vlm.bbox3d.wireframe import wireframe_edge_points
from tmah_vlm.helper.node_helpers import get_robot_pose
from tmah_vlm.sort3d.pipeline import Sort3DLite


RELATION_WORDS = (
    " on ", " near ", " between ", " closest ", " nearest ", " furthest ",
    " farthest ", " left ", " right ", " above ", " below ", " behind ",
    " in front ",
)


def is_relation_query(question):
    text = " " + str(question or "").lower() + " "
    return any(word in text for word in RELATION_WORDS)


def build_sort3d_from_node(node):
    sort3d = Sort3DLite.from_scene_graph(getattr(node, "scene_graph", None))
    return sort3d if sort3d.objects else None


def try_sort3d_graph_fallback(node, question):
    """
    Try selecting a target from the online scene graph.

    Returns True if a waypoint/marker was published. This intentionally never
    reads GT object_list.txt, so it remains compatible with hidden evaluation.
    """
    log = node.get_logger()
    sort3d = build_sort3d_from_node(node)
    if sort3d is None:
        log.info("[SORT3D] graph fallback skipped: scene graph is empty")
        return False

    selection = sort3d.select_target(question)
    candidate_ids = selection.get("candidate_ids", [])
    if not candidate_ids:
        log.info(f"[SORT3D] graph fallback found no target: {selection}")
        return False

    target = sort3d.toolbox.get(candidate_ids[0])
    if target is None:
        log.info(f"[SORT3D] graph fallback target missing: {selection}")
        return False

    waypoint = sort3d.action_for_selection(selection, get_robot_pose(node))
    if waypoint is None:
        log.info(f"[SORT3D] graph fallback could not make waypoint: {selection}")
        return False

    publish_sort3d_result(node, target, waypoint)
    log.info(
        f"[SORT3D] graph target={target.object_id}, name={target.name}, "
        f"tool={selection.get('tool')}, waypoint=({waypoint['x']:.2f}, "
        f"{waypoint['y']:.2f}, {waypoint['heading']:.2f})"
    )
    return True


def publish_sort3d_result(node, target, waypoint):
    stamp = node.get_clock().now().to_msg()
    node.marker_pub.publish(make_sort3d_cube_marker(stamp, target))
    node.wireframe_marker_pub.publish(make_sort3d_wireframe_marker(stamp, target))

    from tmah_vlm.handlers.object_reference import publish_waypoint
    publish_waypoint(node, waypoint)


def make_sort3d_cube_marker(stamp, target):
    marker = Marker()
    marker.header.frame_id = config.FRAME_MAP
    marker.header.stamp = stamp
    marker.ns = "sort3d_selected_object"
    marker.id = 0
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = float(target.center[0])
    marker.pose.position.y = float(target.center[1])
    marker.pose.position.z = float(target.center[2])
    marker.pose.orientation.w = 1.0
    marker.scale.x = float(target.size[0])
    marker.scale.y = float(target.size[1])
    marker.scale.z = float(target.size[2])
    marker.color.a = 0.75
    marker.color.r = 1.0
    marker.color.g = 0.4
    marker.color.b = 0.05
    return marker


def make_sort3d_wireframe_marker(stamp, target):
    marker = Marker()
    marker.header.frame_id = config.FRAME_MAP
    marker.header.stamp = stamp
    marker.ns = "sort3d_selected_object"
    marker.id = 1
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.025
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.8
    marker.color.b = 0.0
    marker.points = [
        Point(x=float(x), y=float(y), z=float(z))
        for x, y, z in wireframe_edge_points(target.center, target.size)
    ]
    return marker
