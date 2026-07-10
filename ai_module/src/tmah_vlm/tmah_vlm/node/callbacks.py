#!/usr/bin/env python3
"""
센서/질문 콜백 — 전부 "최신값 저장"만 한다. 무거운 VLM 추론은 여기서 절대 안 돈다.

initialize/setup.py의 initialize_subscribers()가 이 함수들을 구독 콜백으로 등록한다.
(node를 인자로 받는 자유 함수라 handlers/*.py의 handle(node, ...)와 같은 패턴이다.)
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


def image_callback(node, msg):
    """카메라 이미지 최신값 저장."""
    node.latest_image = msg
    node.image_count += 1


def scan_callback(node, msg):
    """PointCloud2 최신값 저장.

    image와 scan은 서로 다른 callback으로 들어오기 때문에,
    단순 latest_scan만 쓰면 이미지 시각과 다른 point cloud가 섞일 수 있다.
    따라서 최근 scan들을 buffer에 저장해두고, 질문 처리 시 image stamp와
    가장 가까운 scan을 선택한다.
    """
    node.latest_scan = msg
    node.scan_buffer.append(msg)
    node.scan_count += 1


def quaternion_to_yaw(qx, qy, qz, qw):
    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(sin_yaw, cos_yaw)
