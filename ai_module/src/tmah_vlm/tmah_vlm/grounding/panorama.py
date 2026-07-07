#!/usr/bin/env python3
"""
Equirectangular(360 파노라마) 픽셀 <-> 3D 방향(ray) 변환.

파노라마는 핀홀 카메라가 아니라, 구(sphere)를 평면에 편 것.
  - 가로 픽셀 x  -> 방위각(azimuth, 좌우 각도)
  - 세로 픽셀 y  -> 고도각(elevation, 상하 각도)

이 방향 벡터는 "센서 프레임" 기준. 로봇 전방을 +x, 좌측을 +y, 위를 +z 로 가정
(ROS REP-103 표준). 실제 시뮬 관례와 다르면 config 의 오프셋/부호로 조정.

주의: 이 시뮬 파노라마의 정확한 투영 규약(정면 위치, 상하 화각, 회전 방향)은
      실측 확인이 필요. 우선 표준 가정으로 구현하고, 검증 후 config 에서 보정.
"""

import math
import numpy as np

from tmah_vlm import config


def pixel_to_ray(px: float, py: float,
                 width: int = None, height: int = None) -> np.ndarray:
    """
    파노라마 픽셀 (px, py) -> 센서프레임 3D 단위 방향벡터 (x,y,z).
    x: 전방, y: 좌, z: 상 (ROS 표준).
    """
    width = width or config.PANO_WIDTH
    height = height or config.PANO_HEIGHT

    # 방위각: 정면(PANO_FORWARD_X)이 0, 오른쪽으로 갈수록 -각 (시계방향)
    # 이미지 x 증가 = 오른쪽 = 로봇 기준 우측 = -y 방향
    h_fov = math.radians(config.PANO_H_FOV_DEG)
    az = -((px - config.PANO_FORWARD_X) / width) * h_fov   # rad

    # 고도각: 이미지 위쪽(py 작음)이 +elevation(위)
    v_fov = math.radians(config.PANO_V_FOV_DEG)
    el = -((py - height / 2.0) / height) * v_fov           # rad

    # 구면 -> 데카르트 (전방 x, 좌 y, 상 z)
    cos_el = math.cos(el)
    x = cos_el * math.cos(az)
    y = cos_el * math.sin(az)
    z = math.sin(el)

    v = np.array([x, y, z], dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def ray_to_pixel(direction: np.ndarray,
                 width: int = None, height: int = None) -> tuple:
    """3D 방향벡터 -> 파노라마 픽셀 (역변환, 디버그/검증용)."""
    width = width or config.PANO_WIDTH
    height = height or config.PANO_HEIGHT

    x, y, z = direction / (np.linalg.norm(direction) + 1e-9)
    az = math.atan2(y, x)         # 좌우
    el = math.asin(max(-1.0, min(1.0, z)))  # 상하

    h_fov = math.radians(config.PANO_H_FOV_DEG)
    v_fov = math.radians(config.PANO_V_FOV_DEG)

    px = config.PANO_FORWARD_X - (az / h_fov) * width
    py = height / 2.0 - (el / v_fov) * height
    return (px, py)
