""""take the path between A and B" 통과 판정 - A-B 선분을 게이트로 두고 로봇 궤적이
그 선분을 실제로 가로질렀는지 본다.

왜 필요한가: 예전에는 A-B의 **중점** 하나를 waypoint로 삼고 거기서
MISSION3_TARGET_SUCCESS_DISTANCE_M(1.0m) 안에 들어오면 step을 넘겼다. 그런데 채점은
"두 물체 사이를 지나갔는가"를 보므로, 중점 옆에 잠깐 들렀다가 되돌아가도 통과로 찍히는
반면 A-B 사이를 넓게 가로질렀는데 중점에서 1.2m 떨어져 있으면 통과로 안 찍히는 문제가
있었다. 게이트 교차는 그 제약을 직접 판정한다.

기존 반경 판정을 없애지는 않는다(OR 조건) - 좁은 문틈처럼 두 물체가 거의 붙어 있는
경우엔 게이트가 짧아 교차 검출이 예민해지므로, 그때는 기존 방식이 백업이 된다.

ROS 의존이 없는 순수 기하 함수라 tests/test_path_gate_crossing.py에서 그대로 돌아간다.
"""

from __future__ import annotations

import math

from sysnav import config


def _orientation(a, b, c) -> float:
    """a->b->c의 부호 있는 외적. >0 = 좌회전, <0 = 우회전, 0 = 일직선."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1, p2, a, b) -> bool:
    """로봇 이동 선분 p1->p2가 게이트 선분 a->b를 가로질렀는가.

    양쪽 선분이 서로를 "반대편으로 가른다"는 표준 orientation 판정. 한쪽 끝점이 다른
    선분 위에 정확히 얹히는 degenerate 케이스는 통과로 치지 않는다 - 게이트 위를
    스치듯 따라간 궤적까지 통과로 세면, 물체 옆을 지나가기만 해도 step이 넘어간다.
    """
    d1 = _orientation(a, b, p1)
    d2 = _orientation(a, b, p2)
    d3 = _orientation(p1, p2, a)
    d4 = _orientation(p1, p2, b)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def extend_segment(point_a, point_b, margin_m: float):
    """A-B를 양 끝에서 margin_m만큼 늘린 선분. 물체 바로 옆(선분 끝단)을 스치듯
    통과하는 궤적도 잡기 위한 것이다. 길이가 0이면 그대로 돌려준다."""
    ax, ay = float(point_a[0]), float(point_a[1])
    bx, by = float(point_b[0]), float(point_b[1])
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return (ax, ay), (bx, by)
    ux, uy = dx / length, dy / length
    return (ax - ux * margin_m, ay - uy * margin_m), (bx + ux * margin_m, by + uy * margin_m)


def arm_gate(node, segment, pose: dict) -> None:
    """이번 step의 게이트를 세운다. 판정은 **여기서부터** 시작한다 - 이전 step을
    주행하다 우연히 지나간 것은 이번 step의 통과가 아니기 때문이다.
    segment가 None이면 게이트 없는 step(반경 판정만)."""
    node.mission3_gate_segment = None if segment is None else (
        (float(segment[0][0]), float(segment[0][1])),
        (float(segment[1][0]), float(segment[1][1])),
    )
    node.mission3_gate_crossed = False
    node.mission3_gate_last_xy = (float(pose["x"]), float(pose["y"]))
    node.mission3_gate_last_stamp = float(pose["stamp"])


def update_gate_crossing(node, pose: dict) -> bool:
    """이번 step의 게이트를 로봇이 통과했는지 갱신하고 그 결과를 반환한다.

    궤적 소스는 이미 있는 node.pose_buffer다 - 제어 루프 tick(0.2초) 사이에도 로봇은
    움직이므로, "직전 tick 좌표 -> 지금 좌표" 한 선분만 보면 게이트를 뛰어넘은 프레임을
    통째로 놓칠 수 있다. arm_gate() 이후 새로 들어온 pose 샘플만 연속 쌍으로 이어서
    본다(이미 본 샘플을 다시 보지 않으므로 게이트를 세우기 전 궤적도 섞이지 않는다).

    게이트가 없는 step(node.mission3_gate_segment is None)이면 항상 False다.
    한 번 통과로 확정되면 계속 True를 유지한다(로봇이 지나간 뒤 되돌아와도 사실은 사실).
    """
    segment = getattr(node, "mission3_gate_segment", None)
    if segment is None:
        return False
    if node.mission3_gate_crossed:
        return True

    gate_a, gate_b = extend_segment(segment[0], segment[1], config.MISSION3_GATE_EXTENSION_M)

    last_stamp = node.mission3_gate_last_stamp
    with node.sensor_lock:
        trajectory = [
            (float(stamp), (float(sample["x"]), float(sample["y"])))
            for stamp, sample in node.pose_buffer
            if float(stamp) > last_stamp
        ]
    trajectory.append((float(pose["stamp"]), (float(pose["x"]), float(pose["y"]))))

    previous = node.mission3_gate_last_xy
    for stamp, point in trajectory:
        if segments_intersect(previous, point, gate_a, gate_b):
            node.mission3_gate_crossed = True
            break
        previous = point
        node.mission3_gate_last_stamp = max(node.mission3_gate_last_stamp, stamp)
    node.mission3_gate_last_xy = previous
    return node.mission3_gate_crossed
