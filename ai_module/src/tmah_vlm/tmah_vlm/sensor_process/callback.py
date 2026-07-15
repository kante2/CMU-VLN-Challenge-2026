#!/usr/bin/env python3
"""
센서(카메라/LiDAR) 구독 콜백.

전부 "최신값 저장"만 한다. 무거운 VLM 추론은 여기서 절대 안 돈다.
common/initialize.py의 initialize_subscribers()가 이 함수들을 구독 콜백으로 등록한다.
(질문/pose 콜백은 센서 무관 공용이라 common/callback.py에 있다.)
"""


def image_callback(node, msg):
    """카메라 이미지 최신값 저장."""
    node.latest_image = msg
    node.image_count += 1


def scan_callback(node, msg):
    """PointCloud2 최신값 저장.

    image와 scan은 서로 다른 callback으로 들어오기 때문에,
    단순 latest_scan만 쓰면 이미지 시각과 다른 point cloud가 섞일 수 있다.
    따라서 최근 scan들을 buffer에 저장해두고, 질문 처리 시 image stamp와
    가장 가까운 scan을 선택한다(sensor_process/scan_transform.py).
    """
    node.latest_scan = msg
    node.scan_buffer.append(msg)
    node.scan_count += 1
