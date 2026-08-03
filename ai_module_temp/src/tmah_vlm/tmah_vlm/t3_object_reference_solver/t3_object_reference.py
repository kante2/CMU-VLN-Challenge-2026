#!/usr/bin/env python3
"""
t3 Object reference solver — "find ..." 물체 지시 질문 처리.

예:
  Find the red chair near the window

main_node.py의 dispatch_question()이 이 파일의 object_reference_process()를 호출한다.
process는 조건문 + 함수 호출만 나열한다 — 함수 이름을 위에서 아래로 읽으면 그대로
파이프라인 순서가 된다. 각 스텝 함수는 ctx(make_object_ref_context)를 받아 자기 필드를
채우고, 다음 함수가 그 필드를 읽어 이어서 쓴다. 실제 로직은 각 함수 안에 있다.
"""

from tmah_vlm.context.context import make_object_ref_context
from tmah_vlm.perception.camera.image_utils import ros_image_to_pil
from tmah_vlm.perception.camera.query_parser import extract_target
from tmah_vlm.perception.camera.visualize import save_detection_image, save_3d_result_text
from tmah_vlm.geometry.projector import (
    box_to_3d,
    approach_waypoint,
    save_projection_overlay,
)
from tmah_vlm import config
from tmah_vlm.graph.runtime import record_object_observation
from tmah_vlm.common.helpers import get_robot_pose, get_scan_points_in_map
from tmah_vlm.spatial.candidate_filter import filter_candidates_by_relations
from tmah_vlm.sort3d.runtime import is_relation_query, try_sort3d_graph_fallback
from tmah_vlm.t3_object_reference_solver.publish import publish_object_result


# ========================================
# Process
# ========================================

def object_reference_process(node, question):
    # "find ..." 질문 1건: 검출 → 후보 좁히기 → 선택 → 3D 위치/크기 → 발행.
    ctx = make_object_ref_context(question)

    if not sensors_and_models_ready(node):
        return

    grab_camera_image(node, ctx)              # ctx.image, ctx.image_stamp 채움
    extract_target_object(ctx)                # ctx.detect_prompt 채움

    # 공간관계 질문은 먼저 누적된 scene graph로 풀어보고, 되면 여기서 끝낸다.
    if has_spatial_relation(ctx) and solved_by_scene_graph(node, ctx):
        return

    detect_candidate_boxes(node, ctx)         # ctx.detections 채움
    log_detected_candidates(node, ctx)
    if no_candidate_found(ctx):
        handle_no_candidate(node, ctx)
        return

    load_scan_points_in_map(node, ctx)        # ctx.scan_points_map 채움
    narrow_candidates_by_relation(node, ctx)  # ctx.candidate_indices 채움
    pick_final_candidate(node, ctx)           # ctx.selected_index, ctx.selected 채움
    segment_selected_object(node, ctx)        # ctx.segmentation_mask 채움
    estimate_target_3d_pose(node, ctx)        # ctx.result 채움
    read_robot_pose(node, ctx)                # ctx.robot_pose 채움
    compute_approach_waypoint(ctx)            # ctx.result + ctx.robot_pose 읽어 ctx.waypoint 채움

    publish_object_result(node, ctx)
    record_observation_in_graph(node, ctx)
    log_final_target(node, ctx)
    save_object_debug_outputs(node, ctx)


# ========================================
# Steps
# ========================================

def sensors_and_models_ready(node):
    # 검출기(GroundingDINO)가 로드됐고 카메라 이미지가 들어와 있는지 확인한다.
    log = node.get_logger()

    if node.detector is None:
        log.warn("[ObjectRef] detector is still loading")
        return False

    if node.latest_image is None:
        log.warn("[ObjectRef] no camera image yet")
        return False

    return True


def grab_camera_image(node, ctx):
    # 최신 ROS 이미지를 PIL로 변환하고 캡처 시각(stamp)을 같은 스냅샷에서 함께 꺼낸다.
    # (콜백이 latest_image를 계속 갱신하므로 이미지와 stamp를 따로 읽으면 어긋난다.)
    image_msg = node.latest_image
    ctx.image = ros_image_to_pil(image_msg)
    ctx.image_stamp = image_msg.header.stamp


def extract_target_object(ctx):
    # 질문에서 GroundingDINO에 넣을 검출어(object)를 추출한다.
    ctx.detect_prompt = extract_target(ctx.question)["object"]


def has_spatial_relation(ctx):
    # 질문에 "near/between/closest to ..." 같은 공간관계가 있는지 판정한다.
    return is_relation_query(ctx.question)


def solved_by_scene_graph(node, ctx):
    # 누적된 scene graph(SORT-3D)만으로 답을 낼 수 있으면 풀고 True를 반환한다.
    return try_sort3d_graph_fallback(node, ctx.question)


def detect_candidate_boxes(node, ctx):
    # GroundingDINO로 검출어에 해당하는 2D 후보 박스들을 검출한다.
    ctx.detections = node.detector.detect(ctx.image, ctx.detect_prompt)


def no_candidate_found(ctx):
    # 검출된 후보가 하나도 없는지 확인한다.
    return len(ctx.detections) == 0


def handle_no_candidate(node, ctx):
    # 후보가 없을 때: scene graph fallback을 한 번 더 시도하고, 실패하면 디버그 이미지만 남긴다.
    if try_sort3d_graph_fallback(node, ctx.question):
        return
    image_path = save_detection_image(ctx.image, ctx.detections, ctx.detect_prompt, -1)
    node.get_logger().warn(f"[ObjectRef] no candidates, debug image saved: {image_path}")


def load_scan_points_in_map(node, ctx):
    # 이미지 시각에 동기화된 point cloud를 map frame 3D 점들로 변환해 담는다.
    ctx.scan_points_map = get_scan_points_in_map(node, "ObjectRef")


def narrow_candidates_by_relation(node, ctx):
    # 질문에 공간관계가 있으면 랜드마크를 즉석 검출해 후보를 deterministic하게 좁힌다.
    # (spatial/candidate_filter.py, SORT-3D Module 3+4)
    ctx.candidate_indices = filter_candidates_by_relations(
        node, ctx.question, ctx.detections, ctx.image, ctx.image_stamp, ctx.scan_points_map,
    )


def pick_final_candidate(node, ctx):
    # 좁혀진 후보 안에서 최종 1개를 고른다.
    # 관계로 1개까지 좁혀졌으면 그대로, 여러 개면 그 안에서 Qwen이 시각적으로 선택.
    if len(ctx.candidate_indices) == 1:
        node.get_logger().info(
            f"[ObjectRef] selected by spatial relation, index={ctx.candidate_indices[0]}"
        )
        ctx.selected_index = ctx.candidate_indices[0]
    else:
        filtered = [ctx.detections[i] for i in ctx.candidate_indices]
        local_index = choose_with_qwen(node, ctx.image, filtered, ctx.question)
        ctx.selected_index = ctx.candidate_indices[local_index]

    ctx.selected = ctx.detections[ctx.selected_index]


def choose_with_qwen(node, image, detections, question):
    # Qwen2.5-VL로 후보 중 하나를 고른다. selector 미로딩/에러/범위밖이면 0번으로 안전 fallback.
    if node.selector is None:
        node.get_logger().info("[ObjectRef] selector is not ready, use #0")
        return 0

    try:
        selected_index = node.selector.choose(image, detections, question)
    except Exception as error:
        node.get_logger().error(f"[ObjectRef] selector error: {error}, use #0")
        selected_index = 0

    if selected_index < 0:
        selected_index = 0
    if selected_index >= len(detections):
        selected_index = len(detections) - 1

    return selected_index


def segment_selected_object(node, ctx):
    # 선택된 물체 1개만 SAM box-prompt로 pixel 마스크를 얻는다.
    # 미로딩/실패 시 None (projector가 None이면 자동으로 box 기반으로 대체).
    # 세그멘테이션을 실제로 썼는지 매 쿼리마다 로그로 남긴다 (핑크가 안 나오는 원인 추적용).
    log = node.get_logger()

    if node.segmenter is None:
        log.warn("[ObjectRef] segmenter not ready → 마스크 없이 box로 진행 (핑크 없음)")
        ctx.segmentation_mask = None
        return

    try:
        mask = node.segmenter.segment(ctx.image, ctx.selected.box)
    except Exception as error:
        import traceback
        log.error(f"[ObjectRef] segmentation failed: {error}\n{traceback.format_exc()}")
        ctx.segmentation_mask = None
        return

    ctx.segmentation_mask = mask
    if mask is None:
        log.warn("[ObjectRef] segmenter.segment() returned None")
    else:
        log.info(
            f"[ObjectRef] segmentation mask ok: shape={mask.shape}, "
            f"true_px={int(mask.sum())}, box={ctx.selected.box}"
        )


def estimate_target_3d_pose(node, ctx):
    # 선택 box(+마스크)를 point cloud에 재투영해 3D 좌표+크기를 계산한다.
    ctx.result = box_to_3d(
        ctx.selected.box,
        ctx.image.size,
        ctx.scan_points_map,
        node.transformer,
        ctx.detect_prompt,
        image_stamp=ctx.image_stamp,
        segmentation_mask=ctx.segmentation_mask,
    )


def read_robot_pose(node, ctx):
    # 현재 로봇 pose(x, y, yaw) 스냅샷을 ctx에 담는다.
    ctx.robot_pose = get_robot_pose(node)


def compute_approach_waypoint(ctx):
    # ctx.result(목표 3D 위치)와 ctx.robot_pose로 접근 waypoint를 계산한다.
    ctx.waypoint = approach_waypoint(ctx.result["point"], ctx.robot_pose)


def record_observation_in_graph(node, ctx):
    # 이번에 찾은 물체를 scene graph에 관측으로 누적한다(다음 관계 질문에 활용).
    record_object_observation(
        node, ctx.question, ctx.selected, ctx.result,
        image_stamp=ctx.image_stamp, image=ctx.image,
    )


# ========================================
# Debug print / save
# ========================================

def log_detected_candidates(node, ctx):
    # 검출된 후보 개수와 각 후보의 label/score/center를 로그로 남긴다.
    log = node.get_logger()
    log.info(f"[ObjectRef] detected {len(ctx.detections)} candidate(s)")

    for index, det in enumerate(ctx.detections):
        log.info(
            f"  #{index}: label={det.label}, score={det.score:.2f}, "
            f"center=({det.cx:.0f}, {det.cy:.0f})"
        )


def log_final_target(node, ctx):
    # 최종 선택 index와 map 좌표, 추정 방식을 한 줄로 요약해 로그로 남긴다.
    target_x, target_y, target_z = ctx.result["point"]
    node.get_logger().info(
        f"[ObjectRef] '{ctx.detect_prompt}' -> #{ctx.selected_index}, "
        f"map xyz=({target_x:.2f}, {target_y:.2f}, {target_z:.2f}), "
        f"method={ctx.result['method']}, matched={ctx.result['n_matched']}"
    )


def save_object_debug_outputs(node, ctx):
    # 투영 오버레이 이미지 + 검출 이미지 + 3D 결과 텍스트를 디버그로 저장한다.
    save_projection_debug(node, ctx)
    save_debug_files(node, ctx)


def save_projection_debug(node, ctx):
    # point cloud를 이미지에 재투영한 오버레이를 저장한다(설정으로 on/off).
    if not config.DEBUG_SAVE_PROJECTION_OVERLAY:
        return

    log = node.get_logger()
    try:
        overlay_path = save_projection_overlay(
            ctx.image,
            ctx.selected.box,
            ctx.scan_points_map,
            node.transformer,
            ctx.detect_prompt,
            image_stamp=ctx.image_stamp,
            segmentation_mask=ctx.segmentation_mask,
        )
        log.info(f"[ObjectRef] projection overlay saved: {overlay_path}")
    except Exception as error:
        log.warn(f"[ObjectRef] projection overlay failed: {error}")


def save_debug_files(node, ctx):
    # 검출 박스가 그려진 이미지와 3D 결과 텍스트를 저장한다.
    log = node.get_logger()
    image_path = save_detection_image(ctx.image, ctx.detections, ctx.detect_prompt, ctx.selected_index)
    text_path = save_3d_result_text(ctx.question, ctx.selected_index, ctx.result, ctx.waypoint)
    log.info(f"[ObjectRef] debug image saved: {image_path}")
    log.info(f"[ObjectRef] 3D debug text saved: {text_path}")
