"""Fuse panorama masks with LiDAR points to create 3D object observations."""

from __future__ import annotations

import math

import numpy as np

from sysnav import config


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(-1, 3)
    homogeneous = np.column_stack([points.astype(np.float64), np.ones(len(points))])
    return (homogeneous @ matrix.T)[:, :3]


def _base_to_map(points: np.ndarray, pose: dict) -> np.ndarray:
    yaw = float(pose["yaw"])
    rotation = np.array([
        [math.cos(yaw), -math.sin(yaw), 0.0],
        [math.sin(yaw),  math.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    translation = np.array([pose["x"], pose["y"], pose.get("z", 0.0)], dtype=np.float64)
    return points @ rotation.T + translation


class PanoramaLidarGrounder:
    def __init__(self) -> None:
        self.t_lidar_to_camera = np.asarray(config.T_LIDAR_TO_CAMERA, dtype=np.float64)
        self.t_sensor_to_base = np.asarray(config.T_SENSOR_TO_BASE, dtype=np.float64)

    def _project(self, points_sensor: np.ndarray, image_shape: tuple[int, int]) -> dict:
        height, width = image_shape
        points_camera = _transform(points_sensor, self.t_lidar_to_camera)
        x, y, z = points_camera[:, 0], points_camera[:, 1], points_camera[:, 2]
        horizontal = np.hypot(x, y)
        ranges = np.linalg.norm(points_camera, axis=1)
        valid = (
            np.isfinite(points_camera).all(axis=1)
            & (ranges >= config.GROUNDING_MIN_RANGE_M)
            & (ranges <= config.GROUNDING_MAX_RANGE_M)
            & (horizontal > 1e-6)
        )
        indices = np.flatnonzero(valid)
        if not len(indices):
            return {"indices": indices, "u": np.empty(0, int), "v": np.empty(0, int)}

        x, y, z = x[valid], y[valid], z[valid]
        horizontal = horizontal[valid]
        ranges = ranges[valid]
        yaw = np.arctan2(x, y) + math.radians(config.PANORAMA_YAW_OFFSET_DEG)
        down = np.arctan2(z, horizontal) + math.radians(config.PANORAMA_PITCH_OFFSET_DEG)
        yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
        u = np.floor(((yaw / (2.0 * math.pi)) + 0.5) * width).astype(np.int32) % width
        v = np.floor(((down / math.pi) + 0.5) * height).astype(np.int32)
        inside = (v >= 0) & (v < height)
        indices, u, v, ranges = indices[inside], u[inside], v[inside], ranges[inside]

        pixel_id = v.astype(np.int64) * width + u.astype(np.int64)
        order = np.argsort(ranges)
        _, first = np.unique(pixel_id[order], return_index=True)
        keep = order[first]
        return {"indices": indices[keep], "u": u[keep], "v": v[keep]}

    @staticmethod
    def _crop(image_rgb: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        crop = image_rgb[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2]
        if crop.size == 0:
            return crop
        return np.where(crop_mask[..., None], crop, np.full_like(crop, 127))

    # self.grounder.ground(image_rgb, points_sensor, segmented, robot_pose) -> list[dict]
    # observations = self.grounder.ground
    # SAM2가 만든 2D 객체 마스크 안에 들어오는 LiDAR 포인트만 골라서, 객체의 3D 위치와 크기를 계산하는 함수
    
    '''
    RGB 이미지
    +
    LiDAR points
    +
    SAM2 객체 mask
    +
    Robot pose
            ↓
    LiDAR point를 이미지에 투영
            ↓
    각 객체 mask 안의 LiDAR point 선택
            ↓
    Sensor frame → Base frame → Map frame
            ↓
    객체 3D 중심, 크기, point cloud 계산
            ↓
    list[dict] 반환
    '''
    def ground(
        self,
        image_rgb: np.ndarray,
        points_sensor: np.ndarray,
        segmented_objects: list[dict], # SAM2가 만든 2D 객체 mask와 bounding box 정보
        robot_pose: dict,
    ) -> list[dict]:
        # 0. error check
        if not segmented_objects or points_sensor.size == 0:
            return []
        # 1. lidar을 이미지에 투영
        # LiDAR의 3D 점을 파노라마 이미지의 2D 픽셀로 바꾼다.
        projection_lidar_to_image = self._project(points_sensor, image_rgb.shape[:2]) # LiDAR point를 이미지에 투영
        # - error check: 투영된 LiDAR point가 없으면 빈 list 반환
        if not len(projection_lidar_to_image["indices"]):
            return []

        # 2. SAM2 mask 안에 들어오는 LiDAR point만 선택
        projected_lidar_points_in_SAM2mask = points_sensor[projection_lidar_to_image["indices"]]
        # projection_lidar_to_image["indices"] : 원본 points_sensor에서 유효하게 이미지에 투영된 포인트들의 index
        
        # 3. Sensor frame → Base frame → Map frame
        points_base = _transform(projected_lidar_points_in_SAM2mask, self.t_sensor_to_base)
        points_map = _base_to_map(points_base, robot_pose)
        output_3D_list_ = [] # 각 객체의 3D 정보를 하나씩 만들어 저장할 리스트

        for segmented in segmented_objects:
            selected = segmented["mask"][
                projection_lidar_to_image["v"], projection_lidar_to_image["u"]
                # segmented된 mask에 대해, projection["u"], projection["v"]에는 각 LiDAR 포인트가 투영된 이미지 좌표
                ]
            object_points = points_map[np.flatnonzero(selected)] # np.flatnonzero(selected)는 True인 index를 반환
            object_points = object_points[np.isfinite(object_points).all(axis=1)]
            # # mask 안에 들어온 lidar point가 최소 개수보다 적으면, 해당 객체는 3D 정보 계산하지 않고 건너뜀
            if len(object_points) < config.GROUNDING_MIN_POINTS: 
                continue
            
            # mask 안에 들어온 lidar point가 최대 개수보다 많으면, 랜덤하게 최대 개수만큼 샘플링 (연산 과부화 방지)
            if len(object_points) > config.GROUNDING_MAX_OBJECT_POINTS:
                idx = np.linspace(0, len(object_points) - 1, config.GROUNDING_MAX_OBJECT_POINTS, dtype=np.int64)
                object_points = object_points[idx]

            # 객체 3D 포인트들의 x, y, z 각각에 대해 중앙값을 계산
            position = np.median(object_points, axis=0)
            minimum = np.percentile(object_points, 5.0, axis=0)
            maximum = np.percentile(object_points, 95.0, axis=0)
            # item에 기존 segmentation 정보(mask, bbox 등)와 새로 계산한 3D 정보(position, point cloud, 3D bounding box 등)를 합쳐서 복사, 
            item = {key: value for key, value in segmented.items() if key != "mask"}
            # 기존 2d에서, 추가로 계산한 3D 정보를 item에 추가
            item.update({
                "position": tuple(float(v) for v in position), # <- 3D 중심 좌표 * 
                "point_cloud": object_points.astype(np.float32),
                "bbox_3d_min": tuple(float(v) for v in minimum),
                "bbox_3d_max": tuple(float(v) for v in maximum),
                "extent_3d": tuple(float(v) for v in (maximum - minimum)),
                "crop_image": self._crop(image_rgb, segmented["mask"], segmented["bbox"]),
                "num_points": int(len(object_points)),
            })
            output_3D_list_.append(item) # 완성된 객체 검출을, output_3D_list_에 추가
        return output_3D_list_
