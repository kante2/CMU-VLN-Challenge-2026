#!/usr/bin/env python3
"""
axis-aligned 3D box(center, size) -> 테두리(wireframe) 선분 좌표.

RViz Marker(LINE_LIST)로 그리면 반투명 CUBE보다 물체 경계가 뚜렷하게 보이고,
뒤에 있는 point cloud/mesh를 가리지 않는다. ROS 메시지 타입은 모르는
순수 geometry 유틸이고, 실제 Marker 메시지 조립은 t3_object_reference_solver/publish.py에서 한다.
"""


def bbox_corners(center, size):
    """axis-aligned box의 8개 꼭짓점을 반환한다."""
    cx, cy, cz = center
    hx, hy, hz = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)

    signs = [
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
    ]
    return [
        (cx + sx * hx, cy + sy * hy, cz + sz * hz)
        for sx, sy, sz in signs
    ]


def wireframe_edge_points(center, size):
    """
    box의 12개 모서리를 LINE_LIST용 (start, end) 점 쌍 24개로 펼쳐서 반환한다.

    RViz Marker(LINE_LIST)는 points 배열을 2개씩 끊어서 선분으로 그리므로,
    이 함수가 반환하는 리스트를 그대로 marker.points에 넣으면 된다.
    """
    corners = bbox_corners(center, size)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 아래 면
        (4, 5), (5, 6), (6, 7), (7, 4),  # 위 면
        (0, 4), (1, 5), (2, 6), (3, 7),  # 수직 기둥
    ]

    points = []
    for a, b in edges:
        points.append(corners[a])
        points.append(corners[b])
    return points
