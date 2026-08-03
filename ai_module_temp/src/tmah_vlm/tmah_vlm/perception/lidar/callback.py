#!/usr/bin/env python3
"""LiDAR PointCloud2 콜백. "최신값 저장"만 한다. 무거운 VLM 추론은 여기서 절대 안 돈다."""


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
