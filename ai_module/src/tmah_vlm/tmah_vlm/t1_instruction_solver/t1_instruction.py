#!/usr/bin/env python3
"""
t1 Instruction solver — "find"/"how many"/"count"가 아닌 그 외 모든 질문 처리.

아직 stub이다: 그냥 로봇 앞 1m를 waypoint로 찍는다. 나중에 순차 waypoint
실행(경로 지시 따라가기)으로 확장 예정.
main_node.py의 dispatch_question()이 이 파일의 instruction_process()를 호출한다.

process는 조건문 + 함수 호출만 나열한다. 각 스텝 함수는 ctx(make_instruction_context)를
받아 자기 필드를 채우고, 다음 함수가 그 필드를 읽어 이어서 쓴다.
"""

from geometry_msgs.msg import Pose2D

from tmah_vlm.node.context import make_instruction_context
from tmah_vlm.node.helpers import get_robot_pose


# ========================================
# Process
# ========================================

def instruction_process(node, question):
    # 명령형 질문 1건: (stub) 로봇 앞 1m waypoint를 만들어 발행한다.
    ctx = make_instruction_context(question)

    node.get_logger().info("[Instruction] (stub) single waypoint forward")
    read_robot_pose(node, ctx)         # ctx.robot_pose 채움
    plan_forward_waypoint(ctx)         # ctx.robot_pose 읽어 ctx.waypoint 채움
    publish_waypoint(node, ctx)


# ========================================
# Steps
# ========================================

def read_robot_pose(node, ctx):
    # 현재 로봇 pose(x, y, yaw) 스냅샷을 ctx에 담는다.
    ctx.robot_pose = get_robot_pose(node)


def plan_forward_waypoint(ctx):
    # ctx.robot_pose 기준 정면 1m 앞 지점을 목표 waypoint로 계산한다.
    ctx.waypoint = {
        "x": ctx.robot_pose["x"] + 1.0,
        "y": ctx.robot_pose["y"],
        "heading": ctx.robot_pose["yaw"],
    }


# ========================================
# Publish
# ========================================

def publish_waypoint(node, ctx):
    # 계산된 waypoint를 Pose2D로 발행한다.
    msg = Pose2D()
    msg.x = float(ctx.waypoint["x"])
    msg.y = float(ctx.waypoint["y"])
    msg.theta = float(ctx.waypoint["heading"])
    node.waypoint_pub.publish(msg)
