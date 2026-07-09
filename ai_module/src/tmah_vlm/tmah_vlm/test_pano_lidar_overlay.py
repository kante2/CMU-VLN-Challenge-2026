#!/usr/bin/env python3

import os
import math
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
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

        # 저장 경로 고정
        self.out_dir = "/home/docker/ai_module/debug"
        os.makedirs(self.out_dir, exist_ok=True)

        # 저장 주기
        self.save_every_sec = 0.5
        self.last_save_time = 0.0

        # ---------------------------------------------------------
        # 좌표계 정의
        #
        # LiDAR frame:
        #   x_lidar: forward
        #   y_lidar: left
        #   z_lidar: up
        #
        # Camera frame:
        #   x_cam: right
        #   y_cam: forward
        #   z_cam: down
        #
        # 축 관계:
        #   x_cam = -y_lidar
        #   y_cam =  x_lidar
        #   z_cam = -z_lidar + 0.1
        #
        # Camera가 LiDAR보다 0.1m 위에 있다고 가정.
        #
        # p_camera = T_lidar_to_camera @ p_lidar
        # ---------------------------------------------------------
        self.lidar_to_camera_z_offset = 0.1

        self.T_lidar_to_camera = np.array([
            [0.0, -1.0,  0.0, 0.0],
            [1.0,  0.0,  0.0, 0.0],
            [0.0,  0.0, -1.0, self.lidar_to_camera_z_offset],
            [0.0,  0.0,  0.0, 1.0],
        ], dtype=np.float64)

        # 360도 panorama 설정
        self.yaw_offset_deg = 0.0
        self.pitch_offset_deg = 0.0
        self.v_fov_deg = 180.0
        self.invert_yaw = False

        # 시각화 설정
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
            PointCloud2,
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
        self.get_logger().info(f"saving overlay images to: {self.out_dir}")
        self.get_logger().info(f"T_lidar_to_camera:\n{self.T_lidar_to_camera}")

    def image_callback(self, msg: Image):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image = image
            self.latest_image_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().warn(f"image conversion failed: {e}")

    def scan_to_points_lidar(self, scan: PointCloud2):
        points = pc2.read_points(
            scan,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )
        points_lidar = np.array(
            [[p[0], p[1], p[2]] for p in points],
            dtype=np.float64,
        )

        if points_lidar.shape[0] == 0:
            return points_lidar

        dist = np.linalg.norm(points_lidar, axis=1)
        valid = np.isfinite(dist)
        valid &= dist > self.min_range
        valid &= dist < self.max_range

        return points_lidar[valid]

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

        # Camera frame:
        # x: right
        # y: forward
        # z: down
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

        # ---------------------------------------------------------
        # 360도 equirectangular projection
        #
        # Camera 기준:
        #   x = right
        #   y = forward
        #   z = down
        #
        # yaw:
        #   정면 y축 기준 좌우 각도
        #
        # pitch:
        #   수평면 기준 위/아래 각도
        #   z가 down이므로 위쪽은 -z 방향
        # ---------------------------------------------------------
        yaw = np.arctan2(x, y)
        pitch = np.arctan2(-z, np.sqrt(x * x + y * y))

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

        # 가까운 점은 붉게, 먼 점은 초록색 계열
        depth_norm = np.clip(dist / self.max_range, 0.0, 1.0)
        red = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
        green = (255.0 * depth_norm).astype(np.uint8)

        for px, py, r, g in zip(u, v, red, green):
            color = (0, int(g), int(r))
            cv2.circle(overlay, (px, py), self.point_size, color, -1)

        return overlay

    def save_overlay_image(self, overlay):
        now = time.time()

        if now - self.last_save_time < self.save_every_sec:
            return

        self.last_save_time = now

        latest_path = os.path.join(
            self.out_dir,
            "pano_lidar_overlay_latest.png"
        )

        timestamp_path = os.path.join(
            self.out_dir,
            (
                f"pano_lidar_overlay_{int(now * 1000)}"
                f"_vfov{int(self.v_fov_deg)}"
                f"_pitch{self.pitch_offset_deg}"
                f"_yaw{self.yaw_offset_deg}"
                f"_zoffset{self.lidar_to_camera_z_offset}.png"
            )
        )

        cv2.imwrite(latest_path, overlay)
        cv2.imwrite(timestamp_path, overlay)

        self.get_logger().info(
            f"saved overlay image: {latest_path}",
            throttle_duration_sec=1.0,
        )

    def scan_callback(self, scan: PointCloud2):
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
        self.save_overlay_image(overlay)


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
    main()# HOST_EDIT_TEST
