#!/usr/bin/env python3
"""
360 equirectangular panorama pixel <-> camera ray 변환.

주의할 점:
  RViz에서 확인한 sensor -> camera quaternion이 (-0.5, 0.5, -0.5, 0.5)이다.
  이 값은 일반적인 ROS camera optical frame 관계와 맞다.

따라서 여기서 만드는 ray는 camera optical frame 기준이다.
  +x : image right
  +y : image down
  +z : camera forward

그 다음 tf/coordinate_transform.py가 camera frame ray를 map frame ray로 바꾼다.
"""

import math

import numpy as np

from tmah_vlm import config


def pixel_to_ray(px, py, width=None, height=None):
    """
    파노라마 픽셀 위치를 camera optical frame 단위 방향벡터로 변환한다.

    입력:
      px, py : 이미지 픽셀 좌표
      width, height : 실제 이미지 크기. None이면 config 기본값 사용

    출력:
      numpy array [x_right, y_down, z_forward]
    """
    if width is None:
        width = config.PANO_WIDTH
    if height is None:
        height = config.PANO_HEIGHT

    h_fov = math.radians(config.PANO_H_FOV_DEG)
    v_fov = math.radians(config.PANO_V_FOV_DEG)

    # 이미지 오른쪽으로 갈수록 azimuth 양수.
    azimuth = ((px - config.PANO_FORWARD_X) / float(width)) * h_fov

    # 이미지 위쪽으로 갈수록 elevation 양수.
    elevation = -((py - height / 2.0) / float(height)) * v_fov

    cos_el = math.cos(elevation)

    z_forward = cos_el * math.cos(azimuth)
    x_right = cos_el * math.sin(azimuth)
    y_down = -math.sin(elevation)

    ray = np.array([x_right, y_down, z_forward], dtype=np.float64)
    norm = np.linalg.norm(ray)
    if norm < 1e-12:
        return ray
    return ray / norm


def ray_to_pixel(direction, width=None, height=None):
    """camera optical frame 방향벡터를 파노라마 픽셀로 다시 변환한다."""
    if width is None:
        width = config.PANO_WIDTH
    if height is None:
        height = config.PANO_HEIGHT

    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return 0.0, 0.0

    x_right, y_down, z_forward = direction / norm

    azimuth = math.atan2(x_right, z_forward)
    elevation = math.asin(max(-1.0, min(1.0, -y_down)))

    h_fov = math.radians(config.PANO_H_FOV_DEG)
    v_fov = math.radians(config.PANO_V_FOV_DEG)

    px = config.PANO_FORWARD_X + (azimuth / h_fov) * width
    py = height / 2.0 - (elevation / v_fov) * height
    return px, py
