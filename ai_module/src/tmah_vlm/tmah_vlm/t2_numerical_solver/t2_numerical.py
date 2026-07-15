#!/usr/bin/env python3
"""
t2 Numerical solver — "how many ..." / "count ..." 개수 세기 질문 처리.

현재 시야에서 GroundingDINO로 후보를 검출하고, 질문에 공간관계
("between the table and the wall", "near the window" 등)가 있으면
reasoning/spatial/candidate_filter.py로 deterministic하게 걸러낸 다음 개수를 센다
(탐사는 아직 안 함 — 지금 보이는 화면 기준).
main_node.py의 dispatch_question()이 이 파일의 numerical_process()를 호출한다.

process는 조건문 + 함수 호출만 나열한다. 센서 공용 스텝(카메라→box→scan)은
sensor_process/sensor_process.py, 발행은 nav_publish.py에서 가져온다.
각 스텝 함수는 ctx(make_numerical_context)를 받아 자기 필드를 채우고, 다음 함수가 그 필드를 읽어 이어서 쓴다.
"""

from tmah_vlm.question_process.context import make_numerical_context
from tmah_vlm.question_process.query_parser import extract_target
from tmah_vlm.sensor_process.sensor_process import (
    grab_camera_image,
    detect_candidate_boxes,
    load_scan_points_in_map,
)
from tmah_vlm.reasoning.spatial.candidate_filter import filter_candidates_by_relations
from tmah_vlm.nav_publish import publish_count


# ========================================
# Process
# ========================================

def numerical_process(node, question):
    # 개수 세기 질문 1건: 검출 → 공간관계 필터 → 남은 후보 개수를 발행한다.
    # [ question, robot_pose, waypoint ] -> ctx
    ctx = make_numerical_context(question)

    if not sensors_and_models_ready(node):
        publish_count(node, 0)
        return

    grab_camera_image(node, ctx)                 # ctx.image, ctx.image_stamp 채움
    extract_target_object(ctx)                   # ctx.detect_prompt 채움
    detect_candidate_boxes(node, ctx)            # ctx.detections 채움

    load_scan_points_in_map(node, ctx, "Numerical")  # ctx.scan_points_map 채움
    narrow_candidates_by_relation(node, ctx)     # ctx.candidate_indices 채움
    count_candidates(node, ctx)                  # ctx.count 채움

    publish_count(node, ctx.count)


# ========================================
# Steps
# ========================================

def sensors_and_models_ready(node):
    # 검출기(GroundingDINO)와 카메라 이미지가 준비됐는지 확인한다.
    if node.detector is None or node.latest_image is None:
        node.get_logger().warn("[Numerical] not ready, publishing 0")
        return False
    return True


def extract_target_object(ctx):
    # 질문에서 GroundingDINO에 넣을 검출어(object)를 추출한다.
    ctx.detect_prompt = extract_target(ctx.question)["object"]


def narrow_candidates_by_relation(node, ctx):
    # 질문에 공간관계가 있으면 랜드마크 기준으로 후보를 deterministic하게 좁힌다.
    ctx.candidate_indices = filter_candidates_by_relations(
        node, ctx.question, ctx.detections, ctx.image, ctx.image_stamp, ctx.scan_points_map,
    )


def count_candidates(node, ctx):
    # 좁혀진 후보 개수를 세고 로그로 남긴다.
    ctx.count = len(ctx.candidate_indices)
    node.get_logger().info(
        f"[Numerical] '{ctx.detect_prompt}' count={ctx.count} (current view only)"
    )
