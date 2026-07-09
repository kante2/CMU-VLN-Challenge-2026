#!/usr/bin/env python3
"""
Numerical handler — "how many ..." / "count ..." 질문을 처리하는 Process 함수.

아직 stub이다: 현재 시야에서 GroundingDINO 검출 개수만 센다 (탐색/이동 없음).
vlm_node.py의 dispatch_question()이 이 파일의 numerical_process()를 호출한다.
"""

from std_msgs.msg import Int32

from tmah_vlm.perception.image_utils import ros_image_to_pil
from tmah_vlm.perception.query_parser import extract_target


# ========================================
# Process
# ========================================

def numerical_process(node, question):
    if not check_input_ready(node):
        publish_count(node, 0)
        return

    image = prepare_image(node)
    detect_prompt = parse_question(question)
    count = count_in_current_view(node, image, detect_prompt)

    node.get_logger().info(f"[Numerical] '{detect_prompt}' count={count} (current view only)")
    publish_count(node, count)


# ========================================
# Steps
# ========================================

def check_input_ready(node):
    if node.detector is None or node.latest_image is None:
        node.get_logger().warn("[Numerical] not ready, publishing 0")
        return False
    return True


def prepare_image(node):
    return ros_image_to_pil(node.latest_image)


def parse_question(question):
    return extract_target(question)["object"]


def count_in_current_view(node, image, detect_prompt):
    return len(node.detector.detect(image, detect_prompt))


# ========================================
# Publish
# ========================================

def publish_count(node, count):
    msg = Int32()
    msg.data = int(count)
    node.numerical_pub.publish(msg)
