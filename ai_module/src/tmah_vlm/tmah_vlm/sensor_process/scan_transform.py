#!/usr/bin/env python3
"""LiDAR PointCloud2를 이미지와 시간 동기화해서 map frame 3D 점들로 변환하는 헬퍼.

t3_object_reference_solver / t2_numerical_solver가 공용으로 쓰는 "sync + 좌표변환"
파이프라인. 저수준 TF는 sensor_process/coordinate_transform.py의 CoordinateTransformer
(node.transformer)가 담당하고, 여기서는 그것을 이미지 stamp에 맞춰 호출하는 조합을 한다.

node를 인자로 받는 자유 함수라 solver들의 *_process(node, ...)와 같은 패턴이다.
시간 유틸 stamp_to_sec은 좌표변환과 별개 관심사라 common/helpers.py에 두고 import한다.
"""

from tmah_vlm import config
from tmah_vlm.sensor_process.projector import pointcloud_to_xyz
from tmah_vlm.common.helpers import stamp_to_sec


def get_synced_scan_for_latest_image(node):
    """latest_image의 촬영 시각과 시간적으로 가장 가까운 PointCloud2를 골라 반환한다.

    카메라와 LiDAR는 서로 다른 주기로 들어오므로, "지금 이미지"에 맞는 scan은
    최신 scan이 아니라 이미지 stamp에 가장 가까운 scan이다(로봇이 움직이면 그 차이만큼
    투영이 밀린다). node.scan_buffer에 쌓인 최근 scan들을 순회하며 stamp 차이(dt)가
    가장 작은 것을 찾는다.

    반환:
      scan_msg, dt_sec
      dt_sec = abs(image_time - scan_time)  (동기화 실패/불가 시 None)
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

    # 버퍼를 훑어 image_time과 dt가 최소인 scan을 선택한다.
    # (list()로 복사: 순회 중 콜백이 scan_buffer에 append해도 안전하도록)
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

    # heartbeat 로그가 참고할 수 있게 마지막 동기화 오차를 노드에 기록한다.
    node.last_sync_dt = best_dt

    # dt가 너무 크면(로봇이 그만큼 움직였을 수 있어) 투영이 부정확해질 수 있으므로 경고.
    if best_dt > config.SYNC_WARN_TIME_DIFF_SEC:
        node.get_logger().warn(
            f"[Sync] image-scan dt is large: {best_dt:.3f}s "
            f"(recommended < {config.SYNC_WARN_TIME_DIFF_SEC:.3f}s)"
        )
    else:
        node.get_logger().info(f"[Sync] image-scan dt={best_dt:.3f}s")

    return best_scan, best_dt


def get_scan_points_in_map_frame(node, log_tag="Helper"):
    """image stamp와 가장 가까운 PointCloud2를 골라 map frame의 3D 점들로 변환해 반환한다.

    흐름: get_synced_scan_for_latest_image()로 동기화된 scan 선택
        → pointcloud_to_xyz()로 (N,3) xyz 배열 파싱
        → transformer.transform_points()로 sensor frame → map frame 변환.
    변환에는 scan.header.stamp를 넘겨서 "그 scan을 찍은 시각"의 TF를 쓴다
    (추론이 오래 걸려도 캡처 시점 기준으로 투영되도록 — 회전 중 밀림 방지).

    t3_object_reference_solver와 t2_numerical_solver가 공용으로 쓴다
    (원래 object_reference.py에만 있던 걸 옮김 — 둘 다 똑같은 sync+변환이 필요함).
    log_tag는 로그 접두어만 다르게 하려는 용도(예: "ObjectRef", "Numerical").
    실패 시(스캔 없음/파싱 실패/TF 실패) None을 반환한다.
    """
    log = node.get_logger()

    if node.latest_scan is None:
        log.warn(f"[{log_tag}] no point cloud yet")
        return None

    scan_msg, sync_dt = get_synced_scan_for_latest_image(node)

    if scan_msg is None:
        log.warn(f"[{log_tag}] no point cloud yet")
        return None

    if sync_dt is not None:
        log.info(f"[{log_tag}] using scan closest to image stamp, dt={sync_dt:.3f}s")

    try:
        points = pointcloud_to_xyz(scan_msg)
    except Exception as error:
        log.error(f"[{log_tag}] point cloud parsing failed: {error}")
        return None

    # 어느 frame의 점인지: 메시지에 frame_id가 있으면 그걸 쓰고,
    # 비어 있으면 기본 센서 frame으로 가정한다.
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
            f"[{log_tag}] point cloud: {len(points)} points, "
            f"frame {source_frame} -> {config.FRAME_MAP}"
        )
        return points_map
    except Exception as error:
        log.error(f"[{log_tag}] point cloud TF failed: {error}")
        return None
