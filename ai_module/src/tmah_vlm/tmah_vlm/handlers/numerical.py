#!/usr/bin/env python3
"""numerical 질문 처리 (아직 stub -> 검출 개수 세기로 확장 예정)."""

from std_msgs.msg import Int32

from tmah_vlm.perception.image_utils import ros_image_to_pil
from tmah_vlm.perception.query_parser import extract_target


def handle(node, question: str):
    log = node.get_logger()
    if node.detector is None or node.latest_image is None:
        log.warn("[numerical] not ready, publishing 0")
        _pub(node, 0)
        return

    # 검출 개수를 세는 간단 버전 (탐색 없이 현재 뷰만)
    pil = ros_image_to_pil(node.latest_image)
    obj = extract_target(question)["object"]
    dets = node.detector.detect(pil, obj)
    count = len(dets)
    log.info(f"[numerical] '{obj}' count={count} (current view only)")
    _pub(node, count)


def _pub(node, n):
    m = Int32()
    m.data = int(n)
    node.numerical_pub.publish(m)
