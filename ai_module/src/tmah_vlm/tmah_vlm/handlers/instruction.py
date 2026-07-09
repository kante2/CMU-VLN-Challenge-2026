#!/usr/bin/env python3
"""
Instruction handler — 그 외 모든 질문("find"/"how many"/"count"가 아닌 것)을
처리하는 Process 함수.

아직 stub이다: 그냥 로봇 앞 1m를 waypoint로 찍는다. 나중에 순차 waypoint
실행(경로 지시 따라가기)으로 확장 예정.
vlm_node.py의 dispatch_question()이 이 파일의 instruction_process()를 호출한다.
"""

from geometry_msgs.msg import Pose2D

from tmah_vlm.helper.node_helpers import get_robot_pose


# ========================================
# Process
# ========================================

def instruction_process(node, question):
    node.get_logger().info("[Instruction] (stub) single waypoint forward")
    waypoint = make_forward_waypoint(node)
    publish_waypoint(node, waypoint)


# ========================================
# Steps
# ========================================

def make_forward_waypoint(node):
    pose = get_robot_pose(node)
    return {"x": pose["x"] + 1.0, "y": pose["y"], "heading": pose["yaw"]}


# ========================================
# Publish
# ========================================

def publish_waypoint(node, waypoint):
    msg = Pose2D()
    msg.x = float(waypoint["x"])
    msg.y = float(waypoint["y"])
    msg.theta = float(waypoint["heading"])
    node.waypoint_pub.publish(msg)
