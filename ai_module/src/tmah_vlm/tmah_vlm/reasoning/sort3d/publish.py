#!/usr/bin/env python3
"""SORT3D-lite가 고른 타겟을 marker/waypoint로 발행한다."""

from geometry_msgs.msg import Point, Pose2D
from visualization_msgs.msg import Marker

from tmah_vlm import config
from tmah_vlm.perception.lidar.bbox_wireframe import wireframe_edge_points


def publish_sort3d_result(node, target, waypoint):
    stamp = node.get_clock().now().to_msg()
    node.marker_pub.publish(make_sort3d_cube_marker(stamp, target))
    node.wireframe_marker_pub.publish(make_sort3d_wireframe_marker(stamp, target))
    publish_waypoint(node, waypoint)


def publish_waypoint(node, waypoint):
    # 접근 waypoint(dict)를 Pose2D로 발행한다.
    msg = Pose2D()
    msg.x = float(waypoint["x"])
    msg.y = float(waypoint["y"])
    msg.theta = float(waypoint["heading"])
    node.waypoint_pub.publish(msg)


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
