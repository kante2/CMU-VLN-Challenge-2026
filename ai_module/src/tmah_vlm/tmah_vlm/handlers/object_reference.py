#!/usr/bin/env python3
"""
Object reference handler

예:
  Find the red chair near the window

이 파일은 VLM object finding pipeline을 단계별로 보여주기 위한 파일이다.
각 stage 함수는 하나의 기능만 담당한다.

Pipeline:
  Stage 0. 입력 준비 상태 확인
  Stage 1. ROS Image -> PIL Image 변환
  Stage 2. 질문에서 검출 prompt 추출
  Stage 3. GroundingDINO로 2D 후보 검출
  Stage 4. Qwen selector로 후보 1개 선택
  Stage 5. PointCloud2를 map frame으로 변환
  Stage 6. 선택 box 영역의 ray bundle depth mode로 3D 좌표 계산
  Stage 7. 접근 waypoint 계산
  Stage 8. RViz marker / waypoint publish
  Stage 9. debug 파일 저장
"""

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose2D

from tmah_vlm import config
from tmah_vlm.perception.image_utils import ros_image_to_pil
from tmah_vlm.perception.query_parser import extract_target
from tmah_vlm.perception.visualize import save_detection_image, save_3d_result_text
from tmah_vlm.grounding.projector import (
    pointcloud_to_xyz,
    box_to_3d,
    approach_waypoint,
    save_projection_overlay,
)


# ========================================
# Entry
# ========================================

def handle(node, question):
    """vlm_node.py에서 호출하는 object finding 진입점."""
    log = node.get_logger()

    if not stage0_check_input(node):
        return

    log.info("[ObjectRef] Stage 1/9: prepare image")
    image = stage1_prepare_image(node)

    log.info("[ObjectRef] Stage 2/9: parse question")
    parsed = stage2_parse_question(question)
    detect_prompt = parsed["object"]
    log.info(f"[ObjectRef] detect prompt: {detect_prompt}")

    log.info("[ObjectRef] Stage 3/9: detect 2D candidates")
    detections = stage3_detect_candidates(node, image, detect_prompt)
    print_detection_summary(node, detections)

    if len(detections) == 0:
        log.warn("[ObjectRef] no candidates")
        image_path = save_detection_image(image, detections, detect_prompt, -1)
        log.info(f"[ObjectRef] debug image saved: {image_path}")
        return

    log.info("[ObjectRef] Stage 4/9: select target candidate")
    selected_index = stage4_select_candidate(node, image, detections, question)
    selected = detections[selected_index]
    log.info(f"[ObjectRef] selected candidate: #{selected_index}")

    log.info("[ObjectRef] Stage 5/9: transform scan to map frame")
    scan_points_map = stage5_get_scan_points_in_map(node)

    log.info("[ObjectRef] Stage 6/9: estimate 3D target")
    result = stage6_estimate_3d_target(
        node,
        selected.box,
        image.size,
        scan_points_map,
        detect_prompt,
    )
    target_x, target_y, target_z = result["point"]
    log.info(
        f"[ObjectRef] target map xyz=({target_x:.2f}, {target_y:.2f}, {target_z:.2f}), "
        f"method={result['method']}, matched={result['n_matched']}"
    )

    log.info("[ObjectRef] Stage 7/9: make approach waypoint")
    waypoint = stage7_make_waypoint(node, result["point"])

    log.info("[ObjectRef] Stage 8/9: publish result")
    stage8_publish_result(node, selected, result["point"], waypoint)

    if config.DEBUG_SAVE_PROJECTION_OVERLAY:
        try:
            overlay_path = save_projection_overlay(
                image,
                selected.box,
                scan_points_map,
                node.transformer,
                detect_prompt,
            )
            log.info(f"[ObjectRef] projection overlay saved: {overlay_path}")
        except Exception as error:
            log.warn(f"[ObjectRef] projection overlay failed: {error}")

    log.info("[ObjectRef] Stage 9/9: save debug files")
    stage9_save_debug(
        image,
        detections,
        detect_prompt,
        selected_index,
        question,
        result,
        waypoint,
        log,
    )


# ========================================
# Stage functions
# ========================================

def stage0_check_input(node):
    log = node.get_logger()

    if node.detector is None:
        log.warn("[ObjectRef] detector is still loading")
        return False

    if node.latest_image is None:
        log.warn("[ObjectRef] no camera image yet")
        return False

    return True


def stage1_prepare_image(node):
    return ros_image_to_pil(node.latest_image)


def stage2_parse_question(question):
    return extract_target(question)


def stage3_detect_candidates(node, image, detect_prompt):
    return node.detector.detect(image, detect_prompt)


def stage4_select_candidate(node, image, detections, question):
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


def stage5_get_scan_points_in_map(node):
    log = node.get_logger()

    if node.latest_scan is None:
        log.warn("[ObjectRef] no point cloud yet; 3D will use fallback depth")
        return None

    scan_msg = node.latest_scan
    sync_dt = None

    if hasattr(node, "get_synced_scan_for_latest_image"):
        scan_msg, sync_dt = node.get_synced_scan_for_latest_image()

    if scan_msg is None:
        log.warn("[ObjectRef] no point cloud yet; 3D will use fallback depth")
        return None

    if sync_dt is not None:
        log.info(f"[ObjectRef] using scan closest to image stamp, dt={sync_dt:.3f}s")

    try:
        points = pointcloud_to_xyz(scan_msg)
    except Exception as error:
        log.error(f"[ObjectRef] point cloud parsing failed: {error}")
        return None

    source_frame = scan_msg.header.frame_id
    if source_frame is None or source_frame == "":
        source_frame = config.FRAME_SENSOR

    try:
        points_map = node.transformer.transform_points(
            points,
            source_frame,
            config.FRAME_MAP,
        )
        log.info(
            f"[ObjectRef] point cloud: {len(points)} points, "
            f"frame {source_frame} -> {config.FRAME_MAP}"
        )
        return points_map
    except Exception as error:
        log.error(f"[ObjectRef] point cloud TF failed: {error}")
        return None


def stage6_estimate_3d_target(node, selected_box, image_size, scan_points_map, detect_prompt):
    return box_to_3d(
        selected_box,
        image_size,
        scan_points_map,
        node.transformer,
        detect_prompt,
    )


def stage7_make_waypoint(node, target_xyz):
    robot_pose = node.get_robot_pose()
    return approach_waypoint(target_xyz, robot_pose)


def stage8_publish_result(node, selected_detection, target_xyz, waypoint):
    publish_marker(node, selected_detection, target_xyz)
    publish_waypoint(node, waypoint)


def stage9_save_debug(
    image,
    detections,
    detect_prompt,
    selected_index,
    question,
    result,
    waypoint,
    log,
):
    image_path = save_detection_image(
        image,
        detections,
        detect_prompt,
        selected_index,
    )
    text_path = save_3d_result_text(
        question,
        selected_index,
        result,
        waypoint,
    )
    log.info(f"[ObjectRef] debug image saved: {image_path}")
    log.info(f"[ObjectRef] 3D debug text saved: {text_path}")


# ========================================
# Publish
# ========================================

def publish_marker(node, det, target_xyz):
    target_x, target_y, target_z = target_xyz

    marker = Marker()
    marker.header.frame_id = config.FRAME_MAP
    marker.header.stamp = node.get_clock().now().to_msg()
    marker.ns = "selected_object"
    marker.id = 0
    marker.type = Marker.CUBE
    marker.action = Marker.ADD

    marker.pose.position.x = float(target_x)
    marker.pose.position.y = float(target_y)
    marker.pose.position.z = float(target_z)
    marker.pose.orientation.w = 1.0

    # 현재는 3D 크기를 정확히 모르는 단계라 고정 크기 박스로 표시한다.
    marker.scale.x = 0.4
    marker.scale.y = 0.4
    marker.scale.z = 0.4

    marker.color.a = 0.7
    marker.color.r = 0.1
    marker.color.g = 0.9
    marker.color.b = 0.2

    node.marker_pub.publish(marker)


def publish_waypoint(node, waypoint):
    msg = Pose2D()
    msg.x = float(waypoint["x"])
    msg.y = float(waypoint["y"])
    msg.theta = float(waypoint["heading"])
    node.waypoint_pub.publish(msg)


# ========================================
# Debug print
# ========================================

def print_detection_summary(node, detections):
    log = node.get_logger()
    log.info(f"[ObjectRef] detected {len(detections)} candidate(s)")

    for index, det in enumerate(detections):
        log.info(
            f"  #{index}: label={det.label}, score={det.score:.2f}, "
            f"center=({det.cx:.0f}, {det.cy:.0f})"
        )
