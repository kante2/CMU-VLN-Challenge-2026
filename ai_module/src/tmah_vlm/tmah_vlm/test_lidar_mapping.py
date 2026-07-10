#!/usr/bin/env python3
"""
LiDAR 기반 지도(map) 누적 테스트 스크립트.

로봇이 돌아다니며 들어오는 /sensor_scan(PointCloud2)을 매 프레임 TF로
map frame에 등록해서, voxel grid로 dedup하며 계속 누적한다. 누적된 점을
주기적으로 top-down(위에서 본) 이미지로 저장해서, 지도가 드리프트 없이
방/가구 모양대로 잘 쌓이는지 확인한다. test_pano_lidar_overlay.py와 같은
"먼저 스크립트로 검증 -> 잘 되면 본 파이프라인에 통합" 패턴이다.

누적 데이터 자체는 3D 그대로 유지된다(voxel key가 x,y,z 전부). PNG는 z를
버리고 색으로만 힌트를 주는 확인용 시각화라, 진짜 3D로 보려면 RViz에서:
  - /debug/lidar_map_cloud      (PointCloud2 display) -> 누적된 3D map
  - /debug/lidar_map_robot_pose (Marker display)       -> 로봇 현재 위치+heading

TF는 geometry/coordinate_transform.py의 CoordinateTransformer를 그대로 쓴다
(캡처 시각 stamp 기반 lookup까지 이미 검증된 것 재사용 — 회전 중에도 안 어긋남).

다음 단계 (여기엔 아직 없음, 이 스크립트로 지도 누적 자체가 검증된 다음 할 일):
  - 이 누적 map 위에 t3_object_reference_solver가 검출한 물체의 3D 위치를
    같이 얹어서 "탐지된 물체 그래프"를 만든다 (물체 label + 위치 + 지도 좌표).
  - 질문이 들어왔을 때 현재 시야가 아니라 이 그래프를 먼저 찾아보고,
    필요하면 그 물체 근처로 이동하게 한다.
  - 결과가 괜찮으면 initialize/callback/helper 패턴으로 본 파이프라인에 통합
    (예: mapping/ 폴더 하나 새로 만들어서).
"""

import os
import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
import sensor_msgs_py.point_cloud2 as pc2

from tmah_vlm import config
from tmah_vlm.geometry.projector import pointcloud_to_xyz
from tmah_vlm.geometry.coordinate_transform import CoordinateTransformer

# 이 테스트 스크립트만 쓰는 디버그 토픽(본 파이프라인 규격과 무관해서 자유롭게 정함).
TOPIC_MAP_CLOUD = "/debug/lidar_map_cloud"
TOPIC_ROBOT_MARKER = "/debug/lidar_map_robot_pose"


class LidarMappingNode(Node):
    def __init__(self):
        super().__init__("lidar_mapping_node")

        self.transformer = CoordinateTransformer(self)

        self.out_dir = config.DEBUG_DIR
        os.makedirs(self.out_dir, exist_ok=True)

        # voxel 해상도(m). 이 크기로 반올림해서 dedup -> 누적해도 점 개수가
        # 폭발하지 않는다. 너무 작으면 점이 계속 늘고, 너무 크면 디테일이 뭉갠다.
        self.voxel_size = 0.1

        # key: (vx, vy, vz) 정수 voxel 좌표. value 필요 없어서 set만 씀.
        self.map_voxels = set()

        self.latest_pose = None

        # 저장/발행 주기
        self.save_every_sec = 2.0
        self.last_save_time = 0.0

        # top-down 이미지 설정
        self.image_resolution_m = 0.05  # 1 pixel당 몇 m
        self.image_margin_m = 1.0

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.sub_scan = self.create_subscription(
            PointCloud2,
            config.TOPIC_SCAN,
            self.scan_callback,
            qos_sensor,
        )
        self.sub_pose = self.create_subscription(
            Odometry,
            config.TOPIC_STATE,
            self.pose_callback,
            5,
        )

        self.pub_map_cloud = self.create_publisher(PointCloud2, TOPIC_MAP_CLOUD, 1)
        self.pub_robot_marker = self.create_publisher(Marker, TOPIC_ROBOT_MARKER, 5)

        self.get_logger().info("LidarMappingNode started")
        self.get_logger().info(f"scan topic: {config.TOPIC_SCAN}")
        self.get_logger().info(f"voxel size: {self.voxel_size}m")
        self.get_logger().info(f"saving map snapshots to: {self.out_dir}")
        self.get_logger().info(f"publishing map cloud: {TOPIC_MAP_CLOUD} (RViz: PointCloud2 display)")
        self.get_logger().info(f"publishing robot pose: {TOPIC_ROBOT_MARKER} (RViz: Marker display)")

    def scan_callback(self, scan: PointCloud2):
        points_sensor = pointcloud_to_xyz(scan)
        if points_sensor.shape[0] == 0:
            return

        source_frame = scan.header.frame_id or config.FRAME_SENSOR

        try:
            # 캡처 시각(stamp) 기준으로 TF 조회 -> 로봇이 회전 중이어도 안 어긋남
            # (coordinate_transform.py에서 이미 검증한 방식 그대로 재사용).
            points_map = self.transformer.transform_points(
                points_sensor,
                source_frame,
                config.FRAME_MAP,
                stamp=scan.header.stamp,
            )
        except Exception as error:
            self.get_logger().warn(f"TF failed, skip this scan: {error}")
            return

        self.insert_points(points_map)
        self.save_map_snapshot()

    def pose_callback(self, msg: Odometry):
        """로봇 현재 pose 저장 + 즉시 RViz marker로 발행 (지도보다 자주 갱신해도 가벼움)."""
        self.latest_pose = msg.pose.pose
        self.publish_robot_marker()

    def publish_robot_marker(self):
        if self.latest_pose is None:
            return

        marker = Marker()
        marker.header.frame_id = config.FRAME_MAP
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "robot_pose"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = self.latest_pose

        # ARROW는 orientation 방향으로 화살표를 그려서 위치 + heading을 같이 보여준다.
        marker.scale.x = 0.5
        marker.scale.y = 0.12
        marker.scale.z = 0.12

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0

        self.pub_robot_marker.publish(marker)

    def publish_map_cloud(self):
        """누적된 voxel map을 PointCloud2로 발행한다 (RViz PointCloud2 display로 3D 확인)."""
        if len(self.map_voxels) == 0:
            return

        points = np.array(list(self.map_voxels), dtype=np.float64) * self.voxel_size

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = config.FRAME_MAP

        cloud_msg = pc2.create_cloud_xyz32(header, points.tolist())
        self.pub_map_cloud.publish(cloud_msg)

    def insert_points(self, points_map):
        """voxel 단위로 반올림해서 set에 넣는다. 이미 있는 voxel은 자동으로 무시된다."""
        voxel_idx = np.round(points_map / self.voxel_size).astype(np.int64)
        for vx, vy, vz in voxel_idx:
            self.map_voxels.add((int(vx), int(vy), int(vz)))

    def save_map_snapshot(self):
        now = time.time()
        if now - self.last_save_time < self.save_every_sec:
            return
        self.last_save_time = now

        if len(self.map_voxels) == 0:
            return

        image = self.render_topdown()

        latest_path = os.path.join(self.out_dir, "lidar_map_latest.png")
        cv2.imwrite(latest_path, image)

        self.publish_map_cloud()

        self.get_logger().info(
            f"saved map snapshot: {latest_path} ({len(self.map_voxels)} voxels), "
            f"published cloud: {TOPIC_MAP_CLOUD}",
            throttle_duration_sec=1.0,
        )

    def render_topdown(self):
        """누적된 voxel을 XY 평면(위에서 본 시점)에 투영해서 이미지로 그린다.

        높이(z)에 따라 빨강(낮음)~초록(높음)으로 색을 줘서 바닥/벽/천장을
        대략 구분할 수 있게 한다.
        """
        points = np.array(list(self.map_voxels), dtype=np.float64) * self.voxel_size

        min_xy = points[:, :2].min(axis=0) - self.image_margin_m
        max_xy = points[:, :2].max(axis=0) + self.image_margin_m

        size_m = max_xy - min_xy
        width = max(1, int(size_m[0] / self.image_resolution_m))
        height = max(1, int(size_m[1] / self.image_resolution_m))

        image = np.full((height, width, 3), 30, dtype=np.uint8)  # 어두운 회색 배경

        z = points[:, 2]
        z_range = max(1e-3, float(z.max() - z.min()))
        z_norm = np.clip((z - z.min()) / z_range, 0.0, 1.0)

        px = ((points[:, 0] - min_xy[0]) / self.image_resolution_m).astype(np.int32)
        py = (height - 1 - (points[:, 1] - min_xy[1]) / self.image_resolution_m).astype(np.int32)

        px = np.clip(px, 0, width - 1)
        py = np.clip(py, 0, height - 1)

        red = (255.0 * (1.0 - z_norm)).astype(np.uint8)
        green = (255.0 * z_norm).astype(np.uint8)

        image[py, px, 1] = green
        image[py, px, 2] = red

        return image


def main():
    rclpy.init()
    node = LidarMappingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
