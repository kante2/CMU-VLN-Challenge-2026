#!/usr/bin/env python3
"""
vlm_node.py의 main_control_loop()와 handlers/*.py가 공용으로 쓰는 node 상태 헬퍼.

node를 인자로 받는 자유 함수라 handlers/*.py의 process(node, ...)와 같은 패턴이다.
"""

import time

from tmah_vlm import config


def peek_pending_question(node):
    """아직 처리하지 않은 질문을 확인한다."""
    with node.state_lock:
        return node.pending_question


def ready_to_process(node, question):
    """현재 질문을 처리할 준비가 됐는지 확인한다."""
    lower_question = question.lower()

    if node.latest_image is None:
        return False

    if lower_question.startswith("find") and node.detector is None:
        return False

    # selector는 없으면 0번 후보를 선택하는 fallback이 있으므로 필수는 아니다.
    return True


def print_waiting_reason(node, question):
    """준비가 덜 됐을 때 너무 자주 출력하지 않도록 3초에 한 번만 로그."""
    now = time.time()
    if now - node.last_wait_log_time < 3.0:
        return
    node.last_wait_log_time = now

    reasons = []
    lower_question = question.lower()

    if node.latest_image is None:
        reasons.append("camera image")
    if lower_question.startswith("find") and node.detector is None:
        reasons.append("GroundingDINO")

    reason_text = ", ".join(reasons)
    node.get_logger().warn(f"[Pipeline] waiting for: {reason_text}")


def get_robot_pose(node):
    """handler에서 현재 로봇 pose를 읽기 위한 함수."""
    return dict(node.robot)


def get_synced_scan_for_latest_image(node):
    """latest_image의 header stamp와 가장 가까운 PointCloud2를 반환한다.

    반환:
      scan_msg, dt_sec
      dt_sec = abs(image_time - scan_time)
    """
    if node.latest_image is None:
        return node.latest_scan, None

    if len(node.scan_buffer) == 0:
        return node.latest_scan, None

    image_time = stamp_to_sec(node.latest_image.header.stamp)

    # stamp가 0이면 simulator가 header stamp를 안 넣는 경우다.
    # 이 경우에는 기존처럼 최신 scan을 사용한다.
    if image_time <= 0.0:
        return node.latest_scan, None

    best_scan = None
    best_dt = None

    for scan in list(node.scan_buffer):
        scan_time = stamp_to_sec(scan.header.stamp)
        if scan_time <= 0.0:
            continue

        dt = abs(image_time - scan_time)
        if best_dt is None or dt < best_dt:
            best_scan = scan
            best_dt = dt

    if best_scan is None:
        return node.latest_scan, None

    node.last_sync_dt = best_dt

    if best_dt > config.SYNC_WARN_TIME_DIFF_SEC:
        node.get_logger().warn(
            f"[Sync] image-scan dt is large: {best_dt:.3f}s "
            f"(recommended < {config.SYNC_WARN_TIME_DIFF_SEC:.3f}s)"
        )
    else:
        node.get_logger().info(f"[Sync] image-scan dt={best_dt:.3f}s")

    return best_scan, best_dt


def heartbeat(node):
    """현재 노드 상태 확인용 로그."""
    detector_state = "ok" if node.detector is not None else "loading"
    if not config.ENABLE_QWEN_SELECTOR:
        selector_state = "disabled"
    else:
        selector_state = "ok" if node.selector is not None else "loading"
    segmenter_state = "ok" if getattr(node, "segmenter", None) is not None else "loading"

    node.get_logger().info(
        f"[Health] img={node.image_count}, scan={node.scan_count}, "
        f"pose=({node.robot['x']:.2f}, {node.robot['y']:.2f}, "
        f"yaw={node.robot['yaw']:.2f}), "
        f"sync_dt={node.last_sync_dt if node.last_sync_dt is not None else -1:.3f}, "
        f"detector={detector_state}, selector={selector_state}, segmenter={segmenter_state}"
    )


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9
