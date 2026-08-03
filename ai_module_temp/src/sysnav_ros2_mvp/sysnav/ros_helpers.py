"""ROS message conversion and timestamp synchronization helpers."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def message_stamp_to_sec(msg) -> float:
    return stamp_to_sec(msg.header.stamp)


def image_msg_to_rgb(msg) -> np.ndarray:
    import cv2

    height = int(msg.height)
    width = int(msg.width)
    encoding = str(msg.encoding).lower()
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    channels = {"mono8": 1, "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(encoding)
    if channels is None:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    rows = raw.reshape(height, int(msg.step))
    image = rows[:, : width * channels].reshape(height, width, channels)

    if encoding == "rgb8":
        return image.copy()
    if encoding == "bgr8":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def pointcloud2_to_xyz(msg) -> np.ndarray:
    from sensor_msgs_py import point_cloud2

    try:
        points = point_cloud2.read_points_numpy(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )
        points = np.asarray(points)
        if points.dtype.names:
            points = np.column_stack([points["x"], points["y"], points["z"]])
        points = points.reshape(-1, 3)
    except (AttributeError, TypeError, ValueError):
        points = np.asarray(
            list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
            dtype=np.float32,
        ).reshape(-1, 3)

    return points[np.isfinite(points).all(axis=1)].astype(np.float32, copy=False)


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def odometry_to_pose(msg) -> dict:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return {
        "x": float(p.x),
        "y": float(p.y),
        "z": float(p.z),
        "yaw": quaternion_to_yaw(q.x, q.y, q.z, q.w),
        "stamp": message_stamp_to_sec(msg),
    }


def closest_stamped_item(
    items: Sequence[Tuple[float, object]],
    target_stamp: float,
    tolerance: float,
) -> Optional[object]:
    if not items:
        return None
    stamp, item = min(items, key=lambda pair: abs(pair[0] - target_stamp))
    return item if abs(stamp - target_stamp) <= tolerance else None
