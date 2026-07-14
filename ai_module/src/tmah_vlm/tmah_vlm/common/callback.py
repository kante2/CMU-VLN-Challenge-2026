#!/usr/bin/env python3
"""
질문/로봇 pose 콜백 — 특정 센서(카메라/lidar)에 속하지 않는 공용 콜백.
카메라 콜백은 perception/camera/callback.py, lidar 콜백은 perception/lidar/callback.py에 있다.

전부 "최신값 저장"만 한다. 무거운 VLM 추론은 여기서 절대 안 돈다.
common/initialize.py의 initialize_subscribers()가 이 함수들을 구독 콜백으로 등록한다.
(node를 인자로 받는 자유 함수라 solver들의 *_process(node, ...)와 같은 패턴이다.)
"""

import math


def question_callback(node, msg):
    """
    질문 callback.

    여기서는 질문만 저장한다. 실제 VLM pipeline은 main_control_loop에서 처리한다.
    """
    question = msg.data.strip()
    if question == "":
        return

    with node.state_lock:
        node.pending_question = question

    node.get_logger().info(f"[Question] queued: {question}")


def pose_callback(node, msg):
    """로봇 위치/자세 최신값 저장."""
    position = msg.pose.pose.position
    orientation = msg.pose.pose.orientation

    node.robot["x"] = position.x
    node.robot["y"] = position.y
    node.robot["z"] = position.z
    node.robot["yaw"] = quaternion_to_yaw(
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    )


def quaternion_to_yaw(qx, qy, qz, qw):
    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(sin_yaw, cos_yaw)
