#!/usr/bin/env python3
"""
Numerical handler — "how many ..." / "count ..." 질문을 처리하는 Process 함수.

현재 시야에서 GroundingDINO로 후보를 검출하고, 질문에 공간관계
("between the table and the wall", "near the window" 등)가 있으면
object_filter/candidate_filter.py로 deterministic하게 걸러낸 다음 개수를 센다
(탐사는 아직 안 함 — 지금 보이는 화면 기준).
vlm_node.py의 dispatch_question()이 이 파일의 numerical_process()를 호출한다.
"""

from std_msgs.msg import Int32

from tmah_vlm.perception.image_utils import ros_image_to_pil
from tmah_vlm.perception.query_parser import extract_target
from tmah_vlm.object_filter.candidate_filter import filter_candidates_by_relations
from tmah_vlm.helper.node_helpers import get_scan_points_in_map


# ========================================
# Process
# ========================================

def numerical_process(node, question):
    if not check_input_ready(node):
        publish_count(node, 0)
        return

    image, image_stamp = prepare_image(node)
    detect_prompt = parse_question(question)
    detections = detect_candidates(node, image, detect_prompt)

    scan_points_map = get_scan_points_in_map(node, "Numerical")
    candidate_indices = filter_candidates_by_relations(
        node, question, detections, image, image_stamp, scan_points_map,
    )
    count = len(candidate_indices)

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
    image_msg = node.latest_image
    return ros_image_to_pil(image_msg), image_msg.header.stamp


def parse_question(question):
    return extract_target(question)["object"]


def detect_candidates(node, image, detect_prompt):
    return node.detector.detect(image, detect_prompt)


# ========================================
# Publish
# ========================================

def publish_count(node, count):
    msg = Int32()
    msg.data = int(count)
    node.numerical_pub.publish(msg)
