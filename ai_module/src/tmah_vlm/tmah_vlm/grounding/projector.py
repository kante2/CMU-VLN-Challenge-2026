#!/usr/bin/env python3
"""
검출 박스(2D) -> 3D map 좌표.

방법 (경로 A, test-time 안전 - depth 없어도 됨):
  1) 박스 중심 픽셀 -> panorama.pixel_to_ray 로 센서프레임 방향벡터
  2) /registered_scan (라이다, 이미 map 좌표) 포인트들 중,
     로봇 위치에서 그 방향과 가장 잘 맞는(각도 오차 작은) 포인트를 찾음
  3) 그 포인트의 map 좌표 = 대상의 3D 위치
  4) 못 찾으면 방향 * fallback 거리로 근사

registered_scan 이 map 좌표라고 가정 (CMU stack 관례).
센서프레임<->map 회전은 로봇 yaw 로 근사 (state_estimation).
정밀히 하려면 TF 로 변환해야 하지만, 우선 yaw 근사로 시작.
"""

import math
import numpy as np

import sensor_msgs_py.point_cloud2 as pc2

from tmah_vlm import config
from tmah_vlm.grounding.panorama import pixel_to_ray


def _yaw_from_quaternion(qx, qy, qz, qw) -> float:
    """쿼터니언 -> yaw(z축 회전) rad."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def _rotate_z(vec: np.ndarray, yaw: float) -> np.ndarray:
    """센서프레임 벡터를 yaw 만큼 회전 -> map 프레임 방향 (2D 평면 회전)."""
    c, s = math.cos(yaw), math.sin(yaw)
    x, y, z = vec
    return np.array([c * x - s * y, s * x + c * y, z], dtype=np.float64)


def scan_to_points(scan_msg) -> np.ndarray:
    """PointCloud2 -> Nx3 numpy (map 좌표 가정)."""
    pts = pc2.read_points(scan_msg, field_names=("x", "y", "z"),
                          skip_nans=True)
    arr = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float64)
    return arr


def box_to_3d(box, robot_pose, scan_points) -> dict:
    """
    box: (x1,y1,x2,y2) 파노라마 픽셀
    robot_pose: dict {x, y, z, yaw}  (map 프레임 로봇 위치/방향)
    scan_points: Nx3 (map 좌표)
    return: {"point": (X,Y,Z) map좌표, "method": "lidar"/"fallback", "n_matched": int}
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # 1) 센서프레임 방향
    ray_sensor = pixel_to_ray(cx, cy)
    # 2) map 프레임 방향 (yaw 회전)
    ray_map = _rotate_z(ray_sensor, robot_pose["yaw"])

    origin = np.array([robot_pose["x"], robot_pose["y"], robot_pose["z"]],
                      dtype=np.float64)

    # 3) 라이다 포인트 중 방향 일치하는 것 찾기
    if scan_points is not None and len(scan_points) > 0:
        vecs = scan_points - origin              # 로봇->각 포인트 벡터
        norms = np.linalg.norm(vecs, axis=1)
        valid = norms > 1e-3
        vecs_u = vecs[valid] / norms[valid][:, None]
        pts_valid = scan_points[valid]
        d_valid = norms[valid]

        # 방향 코사인 유사도
        cos_sim = vecs_u @ ray_map
        angle = np.arccos(np.clip(cos_sim, -1.0, 1.0))

        mask = angle < config.RAY_MATCH_ANGLE_RAD
        if np.any(mask):
            # 방향 맞는 포인트들 중, 각도 오차 가중해 가장 가까운 것
            cand_pts = pts_valid[mask]
            cand_ang = angle[mask]
            cand_d = d_valid[mask]
            # 각도 오차 작고 거리 가까운 것 선호
            score = cand_ang + 0.05 * cand_d
            best = np.argmin(score)
            p = cand_pts[best]
            return {"point": (float(p[0]), float(p[1]), float(p[2])),
                    "method": "lidar", "n_matched": int(mask.sum())}

    # 4) fallback: 방향 * 기본거리
    p = origin + ray_map * config.FALLBACK_DEPTH_M
    return {"point": (float(p[0]), float(p[1]), float(p[2])),
            "method": "fallback", "n_matched": 0}


def approach_waypoint(target_xyz, robot_pose) -> dict:
    """대상 3D 위치 -> 그 앞 standoff 거리에서 멈출 waypoint (map, heading 포함)."""
    tx, ty, _ = target_xyz
    rx, ry = robot_pose["x"], robot_pose["y"]
    dx, dy = tx - rx, ty - ry
    dist = math.hypot(dx, dy)
    heading = math.atan2(dy, dx)

    if dist > config.APPROACH_STANDOFF_M:
        ratio = (dist - config.APPROACH_STANDOFF_M) / dist
        wx = rx + dx * ratio
        wy = ry + dy * ratio
    else:
        wx, wy = rx, ry  # 이미 충분히 가까움
    return {"x": wx, "y": wy, "heading": heading}
