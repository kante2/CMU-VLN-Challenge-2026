#!/usr/bin/env python3
"""
main_control_loop()와 solver들(t1/t2/t3)이 공용으로 쓰는 상태 조회 헬퍼.

질문 대기열/처리 준비 여부 판단은 context/helpers.py를 참고.
node를 인자로 받는 자유 함수라 solver들의 *_process(node, ...)와 같은 패턴이다.
"""

from tmah_vlm import config
from tmah_vlm.geometry.projector import pointcloud_to_xyz


def get_robot_pose(node):
    """handler에서 현재 로봇 pose(x, y, yaw)를 읽기 위한 함수.

    dict(node.robot)로 복사본을 돌려주는 이유: 원본 node.robot은 pose 콜백이 계속
    덮어쓰므로, handler가 참조를 그대로 들고 있으면 처리 도중에 값이 바뀐다.
    스냅샷을 떠서 처리 시작 시점의 pose를 고정한다.
    """
    return dict(node.robot)


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


def get_scan_points_in_map(node, log_tag="Helper"):
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


def heartbeat(node):
    """노드가 살아있고 무엇을 받고 있는지 주기적으로 찍는 상태 로그.

    타이머로 주기 호출되며, 지금까지 받은 이미지/스캔 수, 현재 로봇 pose,
    마지막 image-scan 동기화 오차(sync_dt), 그리고 각 모델(detector/selector/segmenter)의
    로드 상태(ok/loading/disabled)를 한 줄로 출력한다. 모델이 계속 loading이면
    로드 실패를, 카운트가 안 늘면 센서 토픽이 안 들어옴을 이 로그로 진단할 수 있다.
    """
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
    """ROS2 Time 메시지(sec + nanosec)를 float 초 단위로 합쳐서 반환한다.

    이미지/스캔 stamp를 하나의 실수로 만들어 dt 비교(동기화)에 쓰기 위한 헬퍼.
    """
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9
