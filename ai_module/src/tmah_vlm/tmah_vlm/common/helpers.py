#!/usr/bin/env python3
"""
main_control_loop()와 solver들(t1/t2/t3)이 공용으로 쓰는 상태 조회 헬퍼.

질문 대기열/처리 준비 여부 판단은 question_process/dispatch.py를 참고.
LiDAR scan을 map frame으로 변환하는 sync+좌표변환은 sensor_process/scan_transform.py를 참고.
node를 인자로 받는 자유 함수라 solver들의 *_process(node, ...)와 같은 패턴이다.
"""

from tmah_vlm import config


def get_robot_pose(node):
    """handler에서 현재 로봇 pose(x, y, yaw)를 읽기 위한 함수.

    dict(node.robot)로 복사본을 돌려주는 이유: 원본 node.robot은 pose 콜백이 계속
    덮어쓰므로, handler가 참조를 그대로 들고 있으면 처리 도중에 값이 바뀐다.
    스냅샷을 떠서 처리 시작 시점의 pose를 고정한다.
    """
    return dict(node.robot)


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
