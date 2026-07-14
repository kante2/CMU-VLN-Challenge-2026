#!/usr/bin/env python3
"""카메라 이미지 콜백. "최신값 저장"만 한다. 무거운 VLM 추론은 여기서 절대 안 돈다."""


def image_callback(node, msg):
    """카메라 이미지 최신값 저장."""
    node.latest_image = msg
    node.image_count += 1
