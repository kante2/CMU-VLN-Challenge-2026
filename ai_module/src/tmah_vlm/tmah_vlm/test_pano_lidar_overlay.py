#!/usr/bin/env python3

import os
import math
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge


TOPIC_IMAGE = "/camera/image"
TOPIC_SCAN = "/sensor_scan"


def wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class PanoScanOverlayNode(Node):
    def __init__(self):
        super().__init__("pano_scan_overlay_node")

        self.bridge = CvBridge()

        self.latest_image = None
        self.latest_image_stamp = None

        self.out_dir = "/home/docker/ai_module/debug"
        os.makedirs(self.out_dir, exist_ok=True)

        # -----------------------------
        # 중요:
        # T_lidar_to_camera 는 p_camera = T @ p_lidar 형태.
        #
        # 현재 네 TF:
        # sensor -> camera
        # translation = (0, 0, 0.85)
        # quaternion = (-0.5, 0.5, -0.5, 0.5)
        #
        # sensor frame 가 x forward, y left, z up 이고,
        # camera frame 이 optical frame, 즉 x right, y down, z forward 라고 보면
        # 아래 행렬이 sensor/lidar -> camera 변환임.
        # -----------------------------
        self.T_lidar_to_camera = np.array([
            [0.0, -1.0,  0.0, 0.0],
            [1.0,  0.0,  0.0, 0.0],
            [0.0,  0.0, -1.0, 100.0],
            [0.0,  0.0,  0.0, 1.0],
        ], dtype=np.float64)

        # 네가 이미 상대변환 행렬을 따로 가지고 있으면 위 행렬만 바꾸면 됨.
        # 단, 반드시 p_camera = T_lidar_to_camera @ p_lidar 규칙이어야 함.

        self.yaw_offset_deg = 0.0
        self.pitch_offset_deg = 0.0
        self.v_fov_deg = 180.0
        self.invert_yaw = False

        self.max_range = 30.0
        self.min_range = 0.2
        self.point_size = 2

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.sub_image = self.create_subscription(
            Image,
            TOPIC_IMAGE,
            self.image_callback,
            qos_sensor,
        )

        self.sub_scan = self.create_subscription(
            LaserScan,
            TOPIC_SCAN,
            self.scan_callback,
            qos_sensor,
        )

        self.pub_overlay = self.create_publisher(
            Image,
            "/debug/pano_scan_overlay",
            10,
        )

        self.get_logger().info("PanoScanOverlayNode started")
        self.get_logger().info(f"image topic: {TOPIC_IMAGE}")
        self.get_logger().info(f"scan topic : {TOPIC_SCAN}")
        self.get_logger().info("publishing overlay: /debug/pano_scan_overlay")

    def image_callback(self, msg: Image):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image = image
            self.latest_image_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().warn(f"image conversion failed: {e}")

    def scan_to_points_lidar(self, scan: LaserScan):
        ranges = np.array(scan.ranges, dtype=np.float64)

        angles = scan.angle_min + np.arange(len(ranges), dtype=np.float64) * scan.angle_increment

        valid = np.isfinite(ranges)
        valid &= ranges > max(scan.range_min, self.min_range)
        valid &= ranges < min(scan.range_max, self.max_range)

        ranges = ranges[valid]
        angles = angles[valid]

        # LaserScan 기준:
        # x: forward
        # y: left
        # z: up
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        z = np.zeros_like(x)

        points_lidar = np.stack([x, y, z], axis=1)
        return points_lidar

    def transform_points(self, points_lidar):
        if points_lidar.shape[0] == 0:
            return points_lidar

        ones = np.ones((points_lidar.shape[0], 1), dtype=np.float64)
        points_h = np.concatenate([points_lidar, ones], axis=1)

        points_camera_h = (self.T_lidar_to_camera @ points_h.T).T
        points_camera = points_camera_h[:, :3]

        return points_camera

    def project_points_to_pano(self, image, points_camera):
        H, W = image.shape[:2]

        if points_camera.shape[0] == 0:
            return image.copy()

        # camera optical frame:
        # x: right
        # y: down
        # z: forward
        x = points_camera[:, 0]
        y = points_camera[:, 1]
        z = points_camera[:, 2]

        dist = np.sqrt(x * x + y * y + z * z)

        valid = np.isfinite(dist)
        valid &= dist > self.min_range
        valid &= dist < self.max_range

        x = x[valid]
        y = y[valid]
        z = z[valid]
        dist = dist[valid]

        if len(dist) == 0:
            return image.copy()

        # 360도 equirectangular projection
        yaw = np.arctan2(x, z)
        pitch = np.arctan2(-y, np.sqrt(x * x + z * z))

        yaw += math.radians(self.yaw_offset_deg)
        pitch += math.radians(self.pitch_offset_deg)

        yaw = wrap_pi(yaw)

        v_fov = math.radians(self.v_fov_deg)

        visible = np.abs(pitch) <= v_fov / 2.0

        yaw = yaw[visible]
        pitch = pitch[visible]
        dist = dist[visible]

        if len(dist) == 0:
            return image.copy()

        if self.invert_yaw:
            u = (0.5 - yaw / (2.0 * np.pi)) * W
        else:
            u = (0.5 + yaw / (2.0 * np.pi)) * W

        v = (0.5 - pitch / v_fov) * H

        u = u.astype(np.int32)
        v = v.astype(np.int32)

        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        u = u[inside]
        v = v[inside]
        dist = dist[inside]

        overlay = image.copy()

        # 가까운 점은 더 붉게, 먼 점은 더 초록/어둡게
        depth_norm = np.clip(dist / self.max_range, 0.0, 1.0)
        red = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
        green = (255.0 * depth_norm).astype(np.uint8)

        for px, py, r, g in zip(u, v, red, green):
            color = (0, int(g), int(r))
            cv2.circle(overlay, (px, py), self.point_size, color, -1)

        return overlay

    def scan_callback(self, scan: LaserScan):
        if self.latest_image is None:
            return

        image = self.latest_image.copy()

        points_lidar = self.scan_to_points_lidar(scan)
        points_camera = self.transform_points(points_lidar)

        overlay = self.project_points_to_pano(image, points_camera)

        # RViz에서 볼 수 있게 publish
        msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        msg.header.stamp = scan.header.stamp
        msg.header.frame_id = "camera"
        self.pub_overlay.publish(msg)

        # 이미지 저장
        now = time.time()

        if now - self.last_save_time >= self.save_every_sec:
            self.last_save_time = now

            latest_path = os.path.join(
                self.out_dir,
                "pano_lidar_overlay_latest.png"
            )

            timestamp_path = os.path.join(
                self.out_dir,
                f"pano_lidar_overlay_{int(now * 1000)}.png"
            )

            cv2.imwrite(latest_path, overlay)
            cv2.imwrite(timestamp_path, overlay)

            self.get_logger().info(
                f"saved overlay image: {latest_path}",
                throttle_duration_sec=1.0,
            )

def main():
    rclpy.init()
    node = PanoScanOverlayNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()