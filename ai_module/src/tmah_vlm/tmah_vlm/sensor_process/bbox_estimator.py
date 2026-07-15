#!/usr/bin/env python3
"""
3D bounding box 크기 추정.

지금까지는 물체 크기를 몰라서 RViz marker를 0.4m 고정 박스로 그렸다
(t3_object_reference_solver/publish.py publish_object_marker의 "현재는 3D 크기를 정확히 모르는
단계라 고정 크기 박스로 표시한다" 주석). sensor_process/projector.py가 depth
histogram으로 이미 "이 물체에 해당하는" point 묶음을 골라두므로, 그 point들의
실제 퍼짐(extent)을 쓰면 대략적인 크기를 알 수 있다.

역할 분리:
  sensor_process/projector.py: 2D box -> 어떤 point들이 물체에 해당하는지 선택
  geometry/bbox_estimator.py:     그 point들 -> map frame 3D bounding box(중심, 크기)
"""

import numpy as np

from tmah_vlm import config


def estimate_bbox_from_points(points_map):
    """
    N x 3 map-frame point들의 robust axis-aligned bounding box를 계산한다.

    percentile clipping을 쓰는 이유: min/max를 그대로 쓰면 다른 물체/배경에서
    섞여 들어온 point 하나만 튀어도 박스가 확 커진다. 양 끝 몇 %를 잘라낸
    범위를 쓰면 그런 이상치에 덜 흔들린다.

    반환: (center_xyz, size_xyz) 튜플. point가 부족하면 (None, None).
    size_xyz는 각 축의 전체 길이(half-extent 아님).
    """
    points = np.asarray(points_map, dtype=np.float64)

    if points.shape[0] < config.BBOX3D_MIN_POINTS:
        return None, None

    low = np.percentile(points, config.BBOX3D_PERCENTILE, axis=0)
    high = np.percentile(points, 100.0 - config.BBOX3D_PERCENTILE, axis=0)

    size = np.clip(high - low, config.BBOX3D_MIN_SIZE_M, config.BBOX3D_MAX_SIZE_M)
    center = (low + high) / 2.0

    return tuple(center.tolist()), tuple(size.tolist())


def estimate_object_bbox(points_camera_selected, transformer, image_stamp=None):
    """
    선택된 cluster의 camera-frame point들을 map frame으로 옮기고 bounding box를 계산한다.

    points_camera_selected: sensor_process/projector.py의 find_best_point_by_box_projection()이
      depth histogram으로 골라낸, 그 물체에 해당하는 point들 (camera frame).
    """
    if points_camera_selected is None or len(points_camera_selected) == 0:
        return None, None

    try:
        points_map = transformer.transform_points(
            points_camera_selected,
            config.FRAME_CAMERA,
            config.FRAME_MAP,
            stamp=image_stamp,
        )
    except Exception:
        return None, None

    return estimate_bbox_from_points(points_map)
