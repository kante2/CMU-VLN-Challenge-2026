#!/usr/bin/env python3
"""
Spatial reasoning toolbox (SORT-3D 논문의 Module 4).

"창문 근처의 회색 쓰레기통" 같은 질문을 LLM에게 통째로 주고 좌표 계산까지
맡기면, "왼쪽"을 단순히 x값이 작은 물체로 착각하는 식의 실수가 난다. 그래서
LLM은 "어떤 함수를 어떤 인자로 호출할지"만 결정하고, 실제 기하 계산은 여기
정의된 결정론적 함수가 한다 (LLM이 직접 좌표 비교를 하지 않는다).

grounding/(2D<->3D 투영)과는 다른 책임이라 별도 폴더로 뺐다: 여기 함수들은
이미 3D로 확정된 좌표들끼리의 관계만 다루고, 이미지/카메라를 전혀 모른다.

입출력 규약 (이 파일의 모든 함수 공통):
  points: N x 3 numpy array 또는 그렇게 변환 가능한 리스트 (map frame 좌표들).
  reference_point / point_a / point_b / viewer_point: (x, y, z) 튜플 1개
    (랜드마크나 로봇 위치처럼 기준이 되는 점 하나).
  find_* 함수는 매칭 여부를 나타내는 length-N bool mask를 반환한다
  (geometry/projector.py의 make_box_candidate_mask 등과 같은 스타일).
  order_*/closest_to/farthest_from은 index(정수)를 반환한다.

  point들의 실제 정체(뭐가 어떤 물체인지)는 이 파일이 모른다 — 호출하는 쪽
  (예: t2_numerical_solver, t3_object_reference_solver)이 candidate 목록과
  같은 순서로 넘기고, 반환된 mask/index로 자기 목록을 다시 인덱싱해서 쓴다.

  find_near/closest_to/farthest_from/order_by_distance는 point_radii/reference_radius로
  물체 반지름(있으면)을 받아 "표면 간 거리" 기준으로 계산한다(surface_distances_to).
  안 넘기면(기본값) 중심점 거리와 동일해서 반지름 정보가 없는 호출부는 기존과 동일하게 동작.

  sort3d/reasoning/toolbox.py의 SpatialToolbox(누적 scene graph의 이름 있는 물체 대상 추론)도
  같은 near/closest/farthest/left/right 판정에 이 파일의 함수를 그대로 쓴다(2026-07-13 통합).
  find_between/find_above/find_below는 "표면에 닿아있다"는 더 타이트한 물리적 접촉 판정이
  필요해서(예: "테이블 위") toolbox.py가 자체 구현을 유지한다 — 이 파일의 find_above/find_below는
  "그냥 더 높이/근처에 있다" 정도의 느슨한 판정이라 의도적으로 다르다.
"""

import numpy as np

from tmah_vlm import config


def _as_points(points):
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def _as_point(point):
    return np.asarray(point, dtype=np.float64).reshape(3)


def distances_to(points, reference_point):
    """각 point와 reference_point 사이 유클리드 거리 배열(길이 N)."""
    points = _as_points(points)
    reference_point = _as_point(reference_point)
    return np.linalg.norm(points - reference_point, axis=1)


def surface_distances_to(points, reference_point, point_radii=None, reference_radius=0.0):
    """distances_to에서 각자의 xy 반지름을 빼서 "표면 간 거리"를 구한다.

    point_radii/reference_radius를 안 넘기면(기본 0) distances_to와 동일한 값이라,
    반지름 정보가 없는 호출부(예: 이번 프레임 raw point)는 기존 동작이 그대로 유지된다.
    반지름이 있는 호출부(예: sort3d의 Sort3DObject.radius_xy)는 물체 크기를 반영한
    "표면끼리 얼마나 가까운지"를 얻을 수 있다 — 중심점끼리의 거리보다 "근처/가장 가까움"의
    직관과 더 잘 맞는다(큰 물체는 중심이 멀어도 표면은 가까울 수 있음).
    """
    center_dist = distances_to(points, reference_point)
    if point_radii is None:
        point_radii = 0.0
    else:
        point_radii = np.asarray(point_radii, dtype=np.float64)
    return np.maximum(0.0, center_dist - point_radii - float(reference_radius))


def find_near(points, reference_point, max_distance=None, point_radii=None, reference_radius=0.0):
    """reference_point에서 max_distance 이내(표면 기준)인 point들의 mask."""
    if max_distance is None:
        max_distance = config.SPATIAL_NEAR_THRESHOLD_M
    return surface_distances_to(points, reference_point, point_radii, reference_radius) <= max_distance


def find_between(points, point_a, point_b, corridor_width=None):
    """
    point_a - point_b 선분 "사이"에 있는 point들의 mask (XY 평면 기준, 높이는 안 봄).

    선분에 투영했을 때 두 끝점 사이(0~1)에 들어오고, 선분에서
    corridor_width 이내로 가까운 point만 포함한다. 두 랜드마크를 잇는
    통로 폭을 corridor_width로 표현한 것.
    """
    if corridor_width is None:
        corridor_width = config.SPATIAL_BETWEEN_CORRIDOR_M

    a = _as_point(point_a)[:2]
    b = _as_point(point_b)[:2]
    pts = _as_points(points)[:, :2]

    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))

    if ab_len_sq < 1e-9:
        # 두 landmark가 같은 위치면 "사이"라는 개념이 성립하지 않는다.
        return np.zeros(len(pts), dtype=bool)

    ap = pts - a
    t = (ap @ ab) / ab_len_sq

    on_segment = (t >= 0.0) & (t <= 1.0)

    projection = a + np.outer(t, ab)
    perp_dist = np.linalg.norm(pts - projection, axis=1)

    return on_segment & (perp_dist <= corridor_width)


def side_of_line(points, viewer_point, reference_point):
    """
    viewer_point에서 reference_point 방향을 바라볼 때, 각 point가 그 시선의
    왼쪽(양수)/오른쪽(음수)에 있는지를 부호 있는 값으로 반환한다 (XY 평면 기준).

    map frame이 z가 위로 향하는 오른손 좌표계이므로, 시선 방향 기준 반시계
    방향(CCW)이 왼쪽이 된다. 절댓값 크기는 "얼마나 옆으로 벗어났는지"의
    대략적인 척도라 order_left_to_right 정렬에도 그대로 쓴다.
    """
    viewer = _as_point(viewer_point)[:2]
    reference = _as_point(reference_point)[:2]
    pts = _as_points(points)[:, :2]

    forward = reference - viewer
    to_point = pts - reference

    return forward[0] * to_point[:, 1] - forward[1] * to_point[:, 0]


def find_left(points, viewer_point, reference_point):
    """
    viewer_point는 지금 로봇 현재 위치를 쓰는 걸 상정한다.

    (참고용 SORT-3D 구현은 viewer_point를 따로 안 받고, "두 물체 중간점에서
    가장 가까운 자유공간(바닥) point"를 자동으로 vantage point로 잡는다 —
    로봇이 지금 어디 있든 "그 물체 쌍을 보기 자연스러운 위치"를 쓰는 셈이라
    더 정확하다. 다만 이건 floor/freespace map이 있어야 가능한데, 저희는
    아직 그 맵이 없어서 지금은 robot 위치를 viewer_point로 넘기는 방식을
    쓴다. occupancy grid 같은 바닥 맵이 생기면 이 함수를 그 방식으로
    업그레이드할 것.)
    """
    return side_of_line(points, viewer_point, reference_point) > 0.0


def find_right(points, viewer_point, reference_point):
    return side_of_line(points, viewer_point, reference_point) < 0.0


def find_above(points, reference_point, min_height_diff=None, max_horizontal_dist=None):
    """
    reference_point보다 위에 있으면서, 수평(XY)으로도 가까운 point들의 mask.

    높이만 보면 "문 위의 그림"을 물었을 때 방 반대편의 아무 높은 물체나 다
    걸린다 — reference_point 바로 위쪽인지까지 같이 확인해야 한다
    (참고한 SORT-3D 구현도 bbox 겹침/근접으로 이 수평 조건을 같이 봄).
    """
    if min_height_diff is None:
        min_height_diff = config.SPATIAL_ABOVE_BELOW_MIN_DIFF_M
    if max_horizontal_dist is None:
        max_horizontal_dist = config.SPATIAL_NEAR_THRESHOLD_M

    pts = _as_points(points)
    reference = _as_point(reference_point)

    higher = pts[:, 2] > (reference[2] + min_height_diff)
    horizontal_close = np.linalg.norm(pts[:, :2] - reference[:2], axis=1) <= max_horizontal_dist

    return higher & horizontal_close


def find_below(points, reference_point, min_height_diff=None, max_horizontal_dist=None):
    """find_above와 대칭. reference_point보다 아래 + 수평으로 가까운 point만 포함."""
    if min_height_diff is None:
        min_height_diff = config.SPATIAL_ABOVE_BELOW_MIN_DIFF_M
    if max_horizontal_dist is None:
        max_horizontal_dist = config.SPATIAL_NEAR_THRESHOLD_M

    pts = _as_points(points)
    reference = _as_point(reference_point)

    lower = pts[:, 2] < (reference[2] - min_height_diff)
    horizontal_close = np.linalg.norm(pts[:, :2] - reference[:2], axis=1) <= max_horizontal_dist

    return lower & horizontal_close


def order_by_distance(points, reference_point, descending=False, point_radii=None, reference_radius=0.0):
    """reference_point 기준 가까운 순(기본, 표면 거리 기준) index 배열."""
    order = np.argsort(surface_distances_to(points, reference_point, point_radii, reference_radius))
    return order[::-1] if descending else order


def order_left_to_right(points, viewer_point, reference_point):
    """왼쪽부터 오른쪽 순으로 정렬된 index 배열."""
    side = side_of_line(points, viewer_point, reference_point)
    return np.argsort(-side)


def closest_to(points, reference_point, point_radii=None, reference_radius=0.0):
    """가장 가까운(표면 거리 기준) point의 index. point가 없으면 None."""
    points = _as_points(points)
    if points.shape[0] == 0:
        return None
    return int(np.argmin(surface_distances_to(points, reference_point, point_radii, reference_radius)))


def farthest_from(points, reference_point, point_radii=None, reference_radius=0.0):
    """가장 먼(표면 거리 기준) point의 index. point가 없으면 None."""
    points = _as_points(points)
    if points.shape[0] == 0:
        return None
    return int(np.argmax(surface_distances_to(points, reference_point, point_radii, reference_radius)))


def largest_face_area(bbox_size):
    """
    geometry/bbox_estimator.py가 만든 (sx, sy, sz) 크기에서 가장 넓은 면의 넓이를 구한다.

    "가장 큰/작은 물체" 질문에서 부피보다 이 값이 더 사람 직관과 맞는다
    (얇고 넓은 러그 vs 작고 두꺼운 상자를 부피로 비교하면 직관과 어긋남).
    """
    sx, sy, sz = _as_point(bbox_size)
    return max(sx * sy, sy * sz, sx * sz)


def order_by_size(bbox_sizes, descending=False):
    """
    bbox_sizes: N개 물체의 (sx, sy, sz) 크기 리스트.

    largest_face_area 기준 작은 것부터(기본) 정렬된 index 배열을 반환한다.
    """
    areas = np.array([largest_face_area(size) for size in bbox_sizes], dtype=np.float64)
    order = np.argsort(areas)
    return order[::-1] if descending else order
