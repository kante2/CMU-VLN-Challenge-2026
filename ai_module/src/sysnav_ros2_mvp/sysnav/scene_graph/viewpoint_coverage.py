"""Coverage-region construction for SysNav viewpoint nodes.

The paper defines a viewpoint coverage region C(v) as voxels fully observed within
``d_cover`` of the robot pose. This implementation approximates that region from the
360-degree LiDAR scan: valid LiDAR rays are transformed into the map frame and
voxelized from the sensor origin to each measured endpoint. The resulting voxel-key
set is compact, deterministic, and can be compared with the accumulated union of
existing viewpoint coverage regions.
"""

from __future__ import annotations

import math

import numpy as np

from sysnav import config


VoxelKey = tuple[int, int, int]


class ViewpointCoverageBuilder:
    def __init__(self) -> None:
        self.voxel_size = float(config.VIEWPOINT_COVERAGE_VOXEL_SIZE_M)
        self.coverage_distance = float(config.VIEWPOINT_COVERAGE_DISTANCE_M)
        self.max_rays = int(config.VIEWPOINT_COVERAGE_MAX_RAYS)
        self.sensor_to_base = np.asarray(config.T_SENSOR_TO_BASE, dtype=np.float64)

    def compute(self, points_sensor: np.ndarray, pose: dict) -> set[VoxelKey]:
        """Return the observed map-frame voxel keys for the current robot pose.

        A 360-degree LiDAR endpoint alone describes only a surface. To represent the
        region geometrically observed by the ray, points between the sensor origin and
        each endpoint are also voxelized. Rays are clipped to ``d_cover`` and sampled
        to keep runtime bounded.
        """
        if not isinstance(points_sensor, np.ndarray) or points_sensor.size == 0:
            return set()

        points = points_sensor.reshape(-1, 3).astype(np.float64, copy=False)
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if len(points) == 0:
            return set()

        homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
        points_base = (homogeneous @ self.sensor_to_base.T)[:, :3]
        ranges = np.linalg.norm(points_base, axis=1)
        valid = (
            np.isfinite(points_base).all(axis=1)
            & (ranges >= config.VIEWPOINT_COVERAGE_MIN_RANGE_M)
            & (ranges <= self.coverage_distance)
            & (points_base[:, 2] >= config.VIEWPOINT_COVERAGE_Z_MIN_M)
            & (points_base[:, 2] <= config.VIEWPOINT_COVERAGE_Z_MAX_M)
        )
        points_base = points_base[valid]
        if len(points_base) == 0:
            return set()

        if len(points_base) > self.max_rays:
            indices = np.linspace(0, len(points_base) - 1, self.max_rays, dtype=np.int64)
            points_base = points_base[indices]

        sensor_origin_base = self.sensor_to_base[:3, 3]
        yaw = float(pose["yaw"])
        rotation = np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        map_translation = np.array(
            [float(pose["x"]), float(pose["y"]), float(pose.get("z", 0.0))],
            dtype=np.float64,
        )
        sensor_origin_map = rotation @ sensor_origin_base + map_translation
        endpoints_map = points_base @ rotation.T + map_translation

        coverage: set[VoxelKey] = set()
        for endpoint in endpoints_map:
            ray = endpoint - sensor_origin_map
            distance = float(np.linalg.norm(ray))
            if not np.isfinite(distance) or distance <= 1e-8:
                continue
            steps = max(1, int(math.ceil(distance / self.voxel_size)))
            interpolation = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)
            samples = sensor_origin_map[None, :] + interpolation[:, None] * ray[None, :]
            keys = np.floor(samples / self.voxel_size).astype(np.int32)
            coverage.update((int(x), int(y), int(z)) for x, y, z in keys)

        return coverage
