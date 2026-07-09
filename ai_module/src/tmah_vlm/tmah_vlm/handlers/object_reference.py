#!/usr/bin/env python3
"""
Object reference handler — "find ..." 질문을 처리하는 Process 함수.

예:
  Find the red chair near the window

vlm_node.py의 dispatch_question()이 이 파일의 object_reference_process()를 호출한다.
object_reference_process()는 아래 함수들을 순서대로 부르기만 한다 (실제 로직은 각 함수 안에 있음):

  check_input_ready        -> detector 로딩됐는지, 이미지 있는지 확인
  prepare_image             -> ROS Image -> PIL 변환 + 캡처 시각(stamp)
  parse_question             -> 질문에서 GroundingDINO용 검출어 추출
  detect_candidates          -> GroundingDINO로 2D 후보 검출
  select_candidate            -> Qwen selector로 후보 1개 선택
  segment_selected_object      -> 선택된 후보 1개만 SAM으로 pixel 단위 마스크 추출
                                  (segmentation/segmenter.py). 실패하면 None -> box로 대체
  get_scan_points_in_map      -> PointCloud2를 map frame으로 변환
  estimate_3d_target          -> 선택 box(+마스크)를 point cloud에 재투영해서 3D 좌표+크기 계산
                                  (grounding/projector.py가 bbox3d/estimator.py 호출)
  make_waypoint               -> 접근 waypoint 계산
  publish_result               -> RViz marker(실측 크기 반영) / waypoint publish
  save_projection_debug/save_debug_files -> 디버그 이미지·텍스트 저장
"""

from geometry_msgs.msg import Point, Pose2D
from visualization_msgs.msg import Marker

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
from tmah_vlm.bbox3d.wireframe import wireframe_edge_points
from tmah_vlm.helper.node_helpers import get_robot_pose, get_synced_scan_for_latest_image


# ========================================
# Process
# ========================================

def object_reference_process(node, question):
    """vlm_node.py에서 호출하는 object finding 진입점."""
    log = node.get_logger()

    if not check_input_ready(node):
        return

    image, image_stamp = prepare_image(node)
    detect_prompt = parse_question(question)["object"]

    detections = detect_candidates(node, image, detect_prompt)
    log_detection_summary(node, detections)

    if len(detections) == 0:
        image_path = save_detection_image(image, detections, detect_prompt, -1)
        log.warn(f"[ObjectRef] no candidates, debug image saved: {image_path}")
        return

    selected_index = select_candidate(node, image, detections, question)
    selected = detections[selected_index]

    segmentation_mask = segment_selected_object(node, image, selected.box)

    scan_points_map = get_scan_points_in_map(node)
    result = estimate_3d_target(
        node, selected.box, image.size, scan_points_map, detect_prompt, image_stamp,
        segmentation_mask,
    )

    waypoint = make_waypoint(node, result["point"])
    publish_result(node, selected, result, waypoint)

    target_x, target_y, target_z = result["point"]
    log.info(
        f"[ObjectRef] '{detect_prompt}' -> #{selected_index}, "
        f"map xyz=({target_x:.2f}, {target_y:.2f}, {target_z:.2f}), "
        f"method={result['method']}, matched={result['n_matched']}"
    )

    save_projection_debug(node, image, selected.box, scan_points_map, detect_prompt, image_stamp, segmentation_mask)
    save_debug_files(image, detections, detect_prompt, selected_index, question, result, waypoint, log)


# ========================================
# Steps
# ========================================

def check_input_ready(node):
    log = node.get_logger()

    if node.detector is None:
        log.warn("[ObjectRef] detector is still loading")
        return False

    if node.latest_image is None:
        log.warn("[ObjectRef] no camera image yet")
        return False

    return True


def prepare_image(node):
    """PIL 이미지와 함께 원본 ROS Image의 캡처 시각도 같이 반환한다.

    이 시각은 나중에 camera ray/TF lookup에 쓰인다. node.latest_image는 콜백이
    계속 갱신하므로, 이미지와 stamp를 같은 스냅샷에서 같이 꺼내야 어긋나지 않는다.
    """
    image_msg = node.latest_image
    return ros_image_to_pil(image_msg), image_msg.header.stamp


def parse_question(question):
    return extract_target(question)


def detect_candidates(node, image, detect_prompt):
    return node.detector.detect(image, detect_prompt)


def select_candidate(node, image, detections, question):
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


def segment_selected_object(node, image, box):
    """선택된 물체 1개에 대해서만 SAM box-prompt로 pixel 단위 마스크를 얻는다.

    실패/미로딩 시 None을 반환한다 — grounding/projector.py가 None이면
    자동으로 box 기반 방식으로 대체하므로 여기서 별도 처리 불필요.
    """
    if node.segmenter is None:
        return None

    try:
        return node.segmenter.segment(image, box)
    except Exception as error:
        node.get_logger().warn(f"[ObjectRef] segmentation failed: {error}")
        return None


def get_scan_points_in_map(node):
    log = node.get_logger()

    if node.latest_scan is None:
        log.warn("[ObjectRef] no point cloud yet; 3D will use fallback depth")
        return None

    scan_msg, sync_dt = get_synced_scan_for_latest_image(node)

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
            stamp=scan_msg.header.stamp,
        )
        log.info(
            f"[ObjectRef] point cloud: {len(points)} points, "
            f"frame {source_frame} -> {config.FRAME_MAP}"
        )
        return points_map
    except Exception as error:
        log.error(f"[ObjectRef] point cloud TF failed: {error}")
        return None


def estimate_3d_target(
    node, selected_box, image_size, scan_points_map, detect_prompt,
    image_stamp=None, segmentation_mask=None,
):
    return box_to_3d(
        selected_box,
        image_size,
        scan_points_map,
        node.transformer,
        detect_prompt,
        image_stamp=image_stamp,
        segmentation_mask=segmentation_mask,
    )


def make_waypoint(node, target_xyz):
    robot_pose = get_robot_pose(node)
    return approach_waypoint(target_xyz, robot_pose)


def publish_result(node, selected_detection, result, waypoint):
    publish_marker(node, selected_detection, result)
    publish_waypoint(node, waypoint)


def save_projection_debug(node, image, box, scan_points_map, detect_prompt, image_stamp, segmentation_mask=None):
    if not config.DEBUG_SAVE_PROJECTION_OVERLAY:
        return

    log = node.get_logger()
    try:
        overlay_path = save_projection_overlay(
            image,
            box,
            scan_points_map,
            node.transformer,
            detect_prompt,
            image_stamp=image_stamp,
            segmentation_mask=segmentation_mask,
        )
        log.info(f"[ObjectRef] projection overlay saved: {overlay_path}")
    except Exception as error:
        log.warn(f"[ObjectRef] projection overlay failed: {error}")


def save_debug_files(image, detections, detect_prompt, selected_index, question, result, waypoint, log):
    image_path = save_detection_image(image, detections, detect_prompt, selected_index)
    text_path = save_3d_result_text(question, selected_index, result, waypoint)
    log.info(f"[ObjectRef] debug image saved: {image_path}")
    log.info(f"[ObjectRef] 3D debug text saved: {text_path}")


# ========================================
# Publish
# ========================================

def publish_marker(node, det, result):
    """
    선택된 물체를 RViz에 표시한다.

    /selected_object_marker는 챌린지 쪽 visualizationTools/RViz가 이미
    Marker(단수) 타입으로 구독 중인 고정 규격이라(dummy_vlm과 동일) 여기서
    타입을 바꾸면 안 된다. 그래서 CUBE는 원래 토픽 그대로 두고, wireframe
    테두리는 우리 전용 디버그 토픽(TOPIC_MARKER_WIREFRAME)으로 따로 뺐다.
    """
    stamp = node.get_clock().now().to_msg()

    # bbox3d/estimator.py가 크기를 추정 못했으면(ray fallback 등) result["point"]를
    # 중심으로, config.BBOX3D_DEFAULT_SIZE_M 고정 크기로 대체한다.
    bbox_center = result.get("bbox_center") or result["point"]
    bbox_size = result.get("bbox_size") or (config.BBOX3D_DEFAULT_SIZE_M,) * 3

    node.marker_pub.publish(make_cube_marker(stamp, bbox_center, bbox_size))
    node.wireframe_marker_pub.publish(make_wireframe_marker(stamp, bbox_center, bbox_size))


def make_cube_marker(stamp, center, size):
    marker = Marker()
    marker.header.frame_id = config.FRAME_MAP
    marker.header.stamp = stamp
    marker.ns = "selected_object"
    marker.id = 0
    marker.type = Marker.CUBE
    marker.action = Marker.ADD

    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (
        float(v) for v in center
    )
    marker.pose.orientation.w = 1.0
    marker.scale.x, marker.scale.y, marker.scale.z = (float(v) for v in size)

    marker.color.a = 0.7
    marker.color.r = 0.1
    marker.color.g = 0.9
    marker.color.b = 0.2

    return marker


def make_wireframe_marker(stamp, center, size):
    marker = Marker()
    marker.header.frame_id = config.FRAME_MAP
    marker.header.stamp = stamp
    marker.ns = "selected_object"
    marker.id = 1
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD

    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.02  # LINE_LIST에서 scale.x = 선 굵기(m)

    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 1.0

    marker.points = [
        Point(x=float(x), y=float(y), z=float(z))
        for x, y, z in wireframe_edge_points(center, size)
    ]

    return marker


def publish_waypoint(node, waypoint):
    msg = Pose2D()
    msg.x = float(waypoint["x"])
    msg.y = float(waypoint["y"])
    msg.theta = float(waypoint["heading"])
    node.waypoint_pub.publish(msg)


# ========================================
# Debug print
# ========================================

def log_detection_summary(node, detections):
    log = node.get_logger()
    log.info(f"[ObjectRef] detected {len(detections)} candidate(s)")

    for index, det in enumerate(detections):
        log.info(
            f"  #{index}: label={det.label}, score={det.score:.2f}, "
            f"center=({det.cx:.0f}, {det.cy:.0f})"
        )
