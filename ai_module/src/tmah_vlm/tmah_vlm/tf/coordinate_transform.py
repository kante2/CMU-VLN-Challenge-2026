#!/usr/bin/env python3
"""
좌표 변환 유틸.

현재 목표:
  1. TF가 살아 있으면 실제 TF를 사용한다.
  2. TF lookup이 실패하면 config.py의 fallback tree를 사용한다.
  3. camera ray, sensor point cloud를 map frame으로 변환한다.

확인된 기본 구조:
    map
     └── sensor
          └── camera

fallback 값은 config.STATIC_TF_FALLBACKS에 있다.
"""

from collections import deque

import numpy as np
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from tmah_vlm import config


def clean_frame(frame_name):
    """'/map'처럼 들어온 frame 이름을 'map' 형태로 정리한다."""
    if frame_name is None:
        return ""
    return str(frame_name).strip().lstrip("/")


def canonical_frame(frame_name):
    """
    fallback에서만 사용할 frame alias를 적용한다.

    예: PointCloud2 header가 sensor_scan이면 sensor와 같은 frame으로 취급한다.
    """
    frame = clean_frame(frame_name)
    return config.FRAME_ALIASES.get(frame, frame)


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """quaternion(x,y,z,w)을 3x3 회전행렬로 변환한다."""
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)

    quat = quat / norm
    x, y, z, w = quat

    return np.array([
        [1.0 - 2.0 * y * y - 2.0 * z * z,
         2.0 * x * y - 2.0 * z * w,
         2.0 * x * z + 2.0 * y * w],
        [2.0 * x * y + 2.0 * z * w,
         1.0 - 2.0 * x * x - 2.0 * z * z,
         2.0 * y * z - 2.0 * x * w],
        [2.0 * x * z - 2.0 * y * w,
         2.0 * y * z + 2.0 * x * w,
         1.0 - 2.0 * x * x - 2.0 * y * y],
    ], dtype=np.float64)


def make_transform_matrix(xyz, quat_xyzw):
    """
    child frame 좌표를 parent frame 좌표로 바꾸는 4x4 행렬을 만든다.

    xyz, quat는 ROS TF에서 parent 기준 child pose로 표시되는 값이다.
    """
    x, y, z = xyz
    qx, qy, qz, qw = quat_xyzw

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    matrix[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return matrix


def ros_transform_to_matrix(transform_msg):
    """geometry_msgs/Transform을 4x4 행렬로 변환한다."""
    trans = transform_msg.translation
    rot = transform_msg.rotation
    return make_transform_matrix(
        (trans.x, trans.y, trans.z),
        (rot.x, rot.y, rot.z, rot.w),
    )


class CoordinateTransformer:
    def __init__(self, node):
        self.node = node
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, node)
        self.warned_pairs = set()

    def get_matrix(self, target_frame, source_frame):
        """source_frame 좌표를 target_frame 좌표로 바꾸는 4x4 행렬을 반환한다."""
        target_frame = clean_frame(target_frame)
        source_frame = clean_frame(source_frame)

        if target_frame == source_frame:
            return np.eye(4, dtype=np.float64)

        try:
            tf_msg = self.buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
            )
            return ros_transform_to_matrix(tf_msg.transform)

        except Exception as error:
            pair = (target_frame, source_frame)
            if pair not in self.warned_pairs:
                self.warned_pairs.add(pair)
                self.node.get_logger().warn(
                    f"TF lookup failed: {target_frame} <- {source_frame}. "
                    f"Use config fallback if possible. reason={error}"
                )
            return self.get_fallback_matrix(target_frame, source_frame)

    def get_fallback_matrix(self, target_frame, source_frame):
        """config.STATIC_TF_FALLBACKS만 사용해서 변환 행렬을 찾는다."""
        target_frame = canonical_frame(target_frame)
        source_frame = canonical_frame(source_frame)

        if target_frame == source_frame:
            return np.eye(4, dtype=np.float64)

        graph = {}

        for parent, child, xyz, quat in config.STATIC_TF_FALLBACKS:
            parent = canonical_frame(parent)
            child = canonical_frame(child)

            parent_from_child = make_transform_matrix(xyz, quat)
            child_from_parent = np.linalg.inv(parent_from_child)

            graph.setdefault(child, []).append((parent, parent_from_child))
            graph.setdefault(parent, []).append((child, child_from_parent))

        queue = deque()
        queue.append((source_frame, np.eye(4, dtype=np.float64)))
        visited = set()

        while queue:
            current_frame, current_from_source = queue.popleft()

            if current_frame == target_frame:
                return current_from_source

            if current_frame in visited:
                continue
            visited.add(current_frame)

            for next_frame, next_from_current in graph.get(current_frame, []):
                if next_frame in visited:
                    continue
                next_from_source = next_from_current @ current_from_source
                queue.append((next_frame, next_from_source))

        raise RuntimeError(
            f"No fallback TF path: {target_frame} <- {source_frame}"
        )

    def transform_point(self, point_xyz, source_frame, target_frame):
        """점 1개를 source_frame에서 target_frame으로 변환한다."""
        point = np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0],
                         dtype=np.float64)
        matrix = self.get_matrix(target_frame, source_frame)
        transformed = matrix @ point
        return transformed[:3]

    def transform_points(self, points_xyz, source_frame, target_frame):
        """N x 3 점 배열을 source_frame에서 target_frame으로 변환한다."""
        if points_xyz is None or len(points_xyz) == 0:
            return np.empty((0, 3), dtype=np.float64)

        points = np.asarray(points_xyz, dtype=np.float64)
        matrix = self.get_matrix(target_frame, source_frame)
        ones = np.ones((points.shape[0], 1), dtype=np.float64)
        homogeneous = np.hstack([points, ones])
        transformed = (matrix @ homogeneous.T).T
        return transformed[:, :3]

    def transform_direction(self, direction_xyz, source_frame, target_frame):
        """
        방향 벡터를 source_frame에서 target_frame으로 변환한다.
        위치 이동 성분은 무시하고 회전만 적용한다.
        """
        matrix = self.get_matrix(target_frame, source_frame)
        rotation = matrix[:3, :3]
        direction = rotation @ np.asarray(direction_xyz, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            return direction
        return direction / norm

    def get_frame_origin(self, frame_name, target_frame):
        """frame_name 원점이 target_frame 기준 어디인지 반환한다."""
        return self.transform_point((0.0, 0.0, 0.0), frame_name, target_frame)
