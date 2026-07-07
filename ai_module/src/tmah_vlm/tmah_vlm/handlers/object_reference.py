#!/usr/bin/env python3
"""
object_reference 질문 처리 (전체 파이프라인).

질문 -> 명사추출 -> GroundingDINO 후보 -> Qwen 선택
     -> 3D 좌표화(라이다) -> marker + waypoint 발행 + 시각화
"""

import math

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose2D

from tmah_vlm.perception.image_utils import ros_image_to_pil
from tmah_vlm.perception.query_parser import extract_target
from tmah_vlm.perception.visualize import save_detection_image
from tmah_vlm.grounding.projector import (scan_to_points, box_to_3d,
                                          approach_waypoint)


def handle(node, question: str):
    """node: TmahVLM 인스턴스 (모델·발행자·상태 접근용)."""
    log = node.get_logger()

    if node.detector is None:
        log.warn("Detector loading, skip.")
        return
    if node.latest_image is None:
        log.warn("No camera image yet, skip.")
        return

    pil = ros_image_to_pil(node.latest_image)

    # 1) 검출용 명사 (색 제거)
    parsed = extract_target(question)
    obj = parsed["object"]
    log.info(f"[object_ref] detect prompt = '{obj}'")

    # 2) GroundingDINO 후보
    dets = node.detector.detect(pil, obj)
    log.info(f"  detected {len(dets)} candidate(s)")
    for i, d in enumerate(dets):
        log.info(f"    #{i} {d.label} {d.score:.2f} center=({d.cx:.0f},{d.cy:.0f})")

    if len(dets) == 0:
        log.warn("  no candidates -> nothing to publish")
        _save_viz(node, pil, dets, obj, -1)
        return

    # 3) Qwen 선택
    chosen = 0
    if node.selector is not None:
        try:
            chosen = node.selector.choose(pil, dets, question)
            log.info(f"  Qwen selected #{chosen}")
        except Exception as e:
            log.error(f"  selector error: {e} -> fallback #0")
            chosen = 0
    else:
        log.info("  selector not ready -> #0")
    chosen = max(0, min(chosen, len(dets) - 1))
    det = dets[chosen]

    # 4) 3D 좌표화
    pose = node.get_robot_pose()
    scan_pts = None
    if node.latest_scan is not None:
        try:
            scan_pts = scan_to_points(node.latest_scan)
        except Exception as e:
            log.error(f"  scan parse error: {e}")

    result = box_to_3d(det.box, pose, scan_pts)
    X, Y, Z = result["point"]
    log.info(f"  3D target ({X:.2f},{Y:.2f},{Z:.2f}) "
             f"via {result['method']} (matched={result['n_matched']})")

    # 5) marker 발행 (채점 대상!)
    _publish_marker(node, det, X, Y, Z, chosen)

    # 6) waypoint 발행 (대상 앞으로 이동)
    wp = approach_waypoint((X, Y, Z), pose)
    _publish_waypoint(node, wp)
    log.info(f"  waypoint -> ({wp['x']:.2f},{wp['y']:.2f})")

    # 7) 디버그 시각화
    _save_viz(node, pil, dets, obj, chosen)


def _publish_marker(node, det, X, Y, Z, idx):
    m = Marker()
    m.header.frame_id = node.cfg.FRAME_MAP
    m.header.stamp = node.get_clock().now().to_msg()
    m.ns = "selected_object"
    m.id = 0
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose.position.x = float(X)
    m.pose.position.y = float(Y)
    m.pose.position.z = float(Z)
    m.pose.orientation.w = 1.0
    # 박스 크기: 검출 박스 크기 기반 대략 추정 (정확한 3D 크기는 Phase 1c)
    m.scale.x = 0.4
    m.scale.y = 0.4
    m.scale.z = 0.4
    m.color.a = 0.6
    m.color.r = 0.1
    m.color.g = 0.9
    m.color.b = 0.2
    node.marker_pub.publish(m)


def _publish_waypoint(node, wp):
    m = Pose2D()
    m.x = float(wp["x"])
    m.y = float(wp["y"])
    m.theta = float(wp["heading"])
    node.waypoint_pub.publish(m)


def _save_viz(node, pil, dets, obj, chosen):
    try:
        path = save_detection_image(pil, dets, f"{obj}_sel{chosen}")
        node.get_logger().info(f"  viz -> {path}")
    except Exception as e:
        node.get_logger().error(f"  viz failed: {e}")
