#!/usr/bin/env python3
"""instruction_following 질문 처리 (아직 stub -> 순차 waypoint 예정)."""

from geometry_msgs.msg import Pose2D


def handle(node, question: str):
    log = node.get_logger()
    log.info("[instruction] (stub) single waypoint forward")
    pose = node.get_robot_pose()
    m = Pose2D()
    m.x = pose["x"] + 1.0
    m.y = pose["y"]
    m.theta = pose["yaw"]
    node.waypoint_pub.publish(m)
