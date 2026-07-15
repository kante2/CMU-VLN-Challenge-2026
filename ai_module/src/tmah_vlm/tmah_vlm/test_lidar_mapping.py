#!/usr/bin/env python3
"""
LiDAR 기반 지도(map) 누적 + 자율 탐색(frontier exploration) 테스트 스크립트.

두 가지를 동시에 한다:
  (1) 매핑  : /sensor_scan(PointCloud2)을 매 프레임 TF로 map frame에 등록해서
              3D voxel로 dedup 누적 (기존 기능 그대로).
  (2) 탐색  : 같은 스캔으로 2D 점유격자(free/occupied/unknown)를 ray-casting으로
              만들고, free↔unknown 경계(frontier)를 찾아 그쪽으로 waypoint를 쏴서
              로봇이 미탐사 공간을 스스로 돌아다니게 한다. 이동은 기존 base
              autonomy(waypointConverter -> localPlanner -> pathFollower)가 처리하므로
              여기선 "어디로 갈지"만 정해서 /way_point_with_heading으로 넘긴다.

이건 TARE 같은 무거운 탐사 플래너 없이, 이 테스트 스크립트 안에서 자족적으로 도는
가벼운 frontier 탐색이다. "먼저 스크립트로 검증 -> 잘 되면 본 파이프라인에 통합"
패턴 유지.

RViz로 보려면 (Fixed Frame = map):
  - /debug/lidar_map_cloud       (PointCloud2)  -> 누적된 3D map
  - /debug/lidar_map_robot_pose  (Marker)       -> 로봇 현재 위치+heading
  - /debug/explore_goal          (Marker)       -> 현재 탐색 목표점
PNG(호스트 ai_module/debug/):
  - lidar_map_latest.png     -> 3D voxel top-down (높이별 색)
  - lidar_explore_latest.png -> 2D 점유격자(회색=free / 흰=obstacle / 초록=frontier
                                / 파랑=로봇 / 빨강=목표)

TF는 sensor_process/coordinate_transform.py의 CoordinateTransformer를 그대로 쓴다
(캡처 시각 stamp 기반 lookup까지 이미 검증된 것 재사용 — 회전 중에도 안 어긋남).

주의: /way_point_with_heading은 본 파이프라인의 tmah_vlm도 발행하는 토픽이라, 이
스크립트를 tmah_vlm과 동시에 돌리면 목표가 충돌한다. 이 스크립트는 "매핑+탐색만
단독 검증"용으로, tmah_vlm이 waypoint를 안 쏘는 상태에서 돌리는 걸 전제로 한다.

다음 단계 (여기엔 아직 없음):
  - 이 누적 map 위에 t3가 검출한 물체의 3D 위치를 얹어 "물체 그래프"를 만든다.
  - 질문이 오면 현재 시야가 아니라 그 그래프를 먼저 조회, 필요하면 근처로 이동.
  - 결과가 괜찮으면 initialize/callback/helper 패턴으로 본 파이프라인에 통합.
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
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker
import sensor_msgs_py.point_cloud2 as pc2

from tmah_vlm import config
from tmah_vlm.sensor_process.projector import pointcloud_to_xyz
from tmah_vlm.sensor_process.coordinate_transform import CoordinateTransformer

# 이 테스트 스크립트만 쓰는 디버그 토픽(본 파이프라인 규격과 무관해서 자유롭게 정함).
TOPIC_MAP_CLOUD = "/debug/lidar_map_cloud"
TOPIC_ROBOT_MARKER = "/debug/lidar_map_robot_pose"
TOPIC_GOAL_MARKER = "/debug/explore_goal"

# ===== 탐색(frontier exploration) 파라미터 =====
ENABLE_EXPLORATION = True      # False면 순수 매핑 테스트로만 동작(waypoint 안 쏨).

# 2D 점유격자
GRID_RES_M = 0.2               # 격자 한 칸 크기(m). map 해상도와 별개(매핑용 voxel과 무관).
GRID_HALF_EXTENT_M = 30.0      # 원점 기준 ±범위(m). 60x60m 커버. 벗어난 점은 무시.
OBSTACLE_Z_MIN_M = -0.4        # 센서 원점 기준 이 높이대의 점만 "장애물"로 취급(바닥/천장 제외).
OBSTACLE_Z_MAX_M = 1.2
RAY_SAMPLES = 25               # 원점~장애물 사이를 몇 등분해서 free로 칠할지.
MAX_SCAN_POINTS = 2000         # ray-casting 비용 상한(초과 시 다운샘플).
ROBOT_FREE_RADIUS_M = 0.6      # 로봇 주변 이 반경은 확실히 free로 마킹(seed).

# 목표 선정/수명주기
CONTROL_PERIOD_S = 1.0         # 탐색 결정 주기.
REACH_THRESH_M = 0.8           # 목표에 이만큼 가까우면 도착으로 간주.
GOAL_TIMEOUT_S = 25.0          # 한 목표에 이 시간 넘게 못 가면 포기(blacklist).
STUCK_TIME_S = 8.0             # 이 시간 동안 목표까지 거리 개선 없으면 막힌 걸로 간주.
STUCK_EPS_M = 0.3              # "개선"으로 칠 최소 거리 감소량.
MIN_FRONTIER_CELLS = 4         # 이보다 작은 frontier 덩어리는 노이즈로 무시.
BLACKLIST_RADIUS_M = 1.5       # 실패한 목표 근처는 당분간 다시 안 고름.
MIN_EXPLORE_FREE_CELLS = 200   # free 셀이 이만큼 쌓이기 전엔 완료 판정 보류.
                               # (시작 직후 TF 준비 전 빈 격자로 '완료' 오판되는 것 방지)


class LidarMappingNode(Node):
    def __init__(self):
        super().__init__("lidar_mapping_node")

        self.transformer = CoordinateTransformer(self)

        self.out_dir = config.DEBUG_DIR
        os.makedirs(self.out_dir, exist_ok=True)

        # ----- 매핑(3D voxel) -----
        # voxel 해상도(m). 이 크기로 반올림해서 dedup -> 누적해도 점 개수가
        # 폭발하지 않는다. 너무 작으면 점이 계속 늘고, 너무 크면 디테일이 뭉갠다.
        self.voxel_size = 0.1
        # key: (vx, vy, vz) 정수 voxel 좌표. value 필요 없어서 set만 씀.
        self.map_voxels = set()

        self.latest_pose = None

        # 저장/발행 주기
        self.save_every_sec = 2.0
        self.last_save_time = 0.0

        # top-down 이미지 설정(3D voxel PNG용)
        self.image_resolution_m = 0.05  # 1 pixel당 몇 m
        self.image_margin_m = 1.0

        # ----- 탐색(2D 점유격자) -----
        self.grid_origin = np.array([-GRID_HALF_EXTENT_M, -GRID_HALF_EXTENT_M])
        self.grid_n = int(2 * GRID_HALF_EXTENT_M / GRID_RES_M)
        # free/occ 두 bool 격자. unknown = ~free & ~occ. occ가 free보다 우선.
        self.grid_free = np.zeros((self.grid_n, self.grid_n), dtype=bool)
        self.grid_occ = np.zeros((self.grid_n, self.grid_n), dtype=bool)

        # 목표 상태
        self.current_goal = None          # np.array([x, y]) or None
        self.goal_start_time = 0.0
        self.goal_best_dist = np.inf
        self.goal_last_progress_time = 0.0
        self.blacklist = []               # 실패한 목표 world xy 리스트
        self.exploration_done = False

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
        self.pub_waypoint = self.create_publisher(Pose2D, config.TOPIC_WAYPOINT, 5)
        self.pub_goal_marker = self.create_publisher(Marker, TOPIC_GOAL_MARKER, 5)

        if ENABLE_EXPLORATION:
            self.control_timer = self.create_timer(CONTROL_PERIOD_S, self.exploration_tick)

        self.get_logger().info("LidarMappingNode started")
        self.get_logger().info(f"scan topic: {config.TOPIC_SCAN}")
        self.get_logger().info(f"voxel size: {self.voxel_size}m, grid res: {GRID_RES_M}m")
        self.get_logger().info(f"exploration: {'ON' if ENABLE_EXPLORATION else 'OFF'}"
                               f" (waypoint -> {config.TOPIC_WAYPOINT})")
        self.get_logger().info(f"saving map snapshots to: {self.out_dir}")

    # ==================== 콜백 ====================
    def scan_callback(self, scan: PointCloud2):
        points_sensor = pointcloud_to_xyz(scan)
        if points_sensor.shape[0] == 0:
            return

        source_frame = scan.header.frame_id or config.FRAME_SENSOR

        try:
            # 캡처 시각(stamp) 기준으로 TF 조회 -> 로봇이 회전 중이어도 안 어긋남.
            points_map = self.transformer.transform_points(
                points_sensor,
                source_frame,
                config.FRAME_MAP,
                stamp=scan.header.stamp,
            )
            # 센서 원점(=ray의 시작점)도 같은 시각 TF로 map frame에서 구한다.
            origin_map = self.transformer.transform_points(
                np.zeros((1, 3), dtype=np.float64),
                source_frame,
                config.FRAME_MAP,
                stamp=scan.header.stamp,
            )[0]
        except Exception as error:
            self.get_logger().warn(f"TF failed, skip this scan: {error}")
            return

        self.insert_points(points_map)                 # 3D voxel 누적(기존)
        if ENABLE_EXPLORATION:
            self.update_occupancy(points_map, origin_map)  # 2D 점유격자 갱신(신규)
        self.save_map_snapshot()

    def pose_callback(self, msg: Odometry):
        """로봇 현재 pose 저장 + 즉시 RViz marker로 발행."""
        self.latest_pose = msg.pose.pose
        self.publish_robot_marker()

    # ==================== 매핑(3D voxel) — 기존 ====================
    def insert_points(self, points_map):
        """voxel 단위로 반올림해서 set에 넣는다. 이미 있는 voxel은 자동으로 무시된다."""
        voxel_idx = np.round(points_map / self.voxel_size).astype(np.int64)
        for vx, vy, vz in voxel_idx:
            self.map_voxels.add((int(vx), int(vy), int(vz)))

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
        marker.scale.x = 0.5
        marker.scale.y = 0.12
        marker.scale.z = 0.12
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        self.pub_robot_marker.publish(marker)

    def publish_map_cloud(self):
        if len(self.map_voxels) == 0:
            return
        points = np.array(list(self.map_voxels), dtype=np.float64) * self.voxel_size
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = config.FRAME_MAP
        cloud_msg = pc2.create_cloud_xyz32(header, points.tolist())
        self.pub_map_cloud.publish(cloud_msg)

    def save_map_snapshot(self):
        now = time.time()
        if now - self.last_save_time < self.save_every_sec:
            return
        self.last_save_time = now

        if len(self.map_voxels) == 0:
            return

        cv2.imwrite(os.path.join(self.out_dir, "lidar_map_latest.png"), self.render_topdown())
        if ENABLE_EXPLORATION:
            cv2.imwrite(os.path.join(self.out_dir, "lidar_explore_latest.png"), self.render_explore())
        self.publish_map_cloud()

        self.get_logger().info(
            f"map: {len(self.map_voxels)} voxels | "
            f"grid free={int(self.grid_free.sum())} occ={int(self.grid_occ.sum())}",
            throttle_duration_sec=2.0,
        )

    def render_topdown(self):
        """누적된 voxel을 XY 평면에 투영. 높이(z)별 빨강(낮음)~초록(높음)."""
        points = np.array(list(self.map_voxels), dtype=np.float64) * self.voxel_size
        min_xy = points[:, :2].min(axis=0) - self.image_margin_m
        max_xy = points[:, :2].max(axis=0) + self.image_margin_m
        size_m = max_xy - min_xy
        width = max(1, int(size_m[0] / self.image_resolution_m))
        height = max(1, int(size_m[1] / self.image_resolution_m))
        image = np.full((height, width, 3), 30, dtype=np.uint8)
        z = points[:, 2]
        z_range = max(1e-3, float(z.max() - z.min()))
        z_norm = np.clip((z - z.min()) / z_range, 0.0, 1.0)
        px = ((points[:, 0] - min_xy[0]) / self.image_resolution_m).astype(np.int32)
        py = (height - 1 - (points[:, 1] - min_xy[1]) / self.image_resolution_m).astype(np.int32)
        px = np.clip(px, 0, width - 1)
        py = np.clip(py, 0, height - 1)
        image[py, px, 1] = (255.0 * z_norm).astype(np.uint8)
        image[py, px, 2] = (255.0 * (1.0 - z_norm)).astype(np.uint8)
        return image

    # ==================== 탐색(2D 점유격자) — 신규 ====================
    def world_to_cell(self, xy):
        """world xy(...,2) -> 격자 정수 인덱스(...,2) = [ix, iy]."""
        return np.floor((np.asarray(xy) - self.grid_origin) / GRID_RES_M).astype(np.int64)

    def cell_to_world(self, cell):
        """격자 인덱스 -> 셀 중심 world xy."""
        return self.grid_origin + (np.asarray(cell, dtype=np.float64) + 0.5) * GRID_RES_M

    def _mark(self, grid, cells):
        """범위 안 셀만 True로 마킹."""
        if len(cells) == 0:
            return
        ix, iy = cells[:, 0], cells[:, 1]
        ok = (ix >= 0) & (ix < self.grid_n) & (iy >= 0) & (iy < self.grid_n)
        grid[ix[ok], iy[ok]] = True

    def update_occupancy(self, points_map, origin_map):
        """스캔으로 2D 점유격자를 갱신한다.

        센서 원점 기준 특정 높이대의 점만 '장애물'로 보고(바닥/천장 제외),
        원점->장애물 ray 위를 free로, 장애물 셀을 occupied로 칠한다. 어느 방향에
        장애물이 없으면(열린 공간) ray가 안 생겨 그쪽은 unknown으로 남고, 이게
        나중에 frontier가 되어 로봇을 그쪽으로 이끈다.
        """
        origin_xy = origin_map[:2]

        # 1) 높이대 필터로 장애물 후보만 남긴다.
        dz = points_map[:, 2] - origin_map[2]
        band = (dz > OBSTACLE_Z_MIN_M) & (dz < OBSTACLE_Z_MAX_M)
        endpoints = points_map[band][:, :2]
        if endpoints.shape[0] == 0:
            # 장애물은 없어도 로봇 주변 free seed는 찍어둔다.
            self._mark_robot_free_disk(origin_xy)
            return

        # 비용 상한: 너무 많으면 균등 다운샘플.
        if endpoints.shape[0] > MAX_SCAN_POINTS:
            sel = np.random.choice(endpoints.shape[0], MAX_SCAN_POINTS, replace=False)
            endpoints = endpoints[sel]

        # 2) ray를 등분 샘플링해서 free로 칠한다 (마지막 구간은 장애물이라 제외).
        t = np.linspace(0.0, 1.0, RAY_SAMPLES, endpoint=False)          # (K,)
        samples = origin_xy[None, None, :] + \
            t[None, :, None] * (endpoints[:, None, :] - origin_xy[None, None, :])  # (M,K,2)
        free_cells = self.world_to_cell(samples.reshape(-1, 2))
        self._mark(self.grid_free, free_cells)

        # 3) 장애물 셀을 occupied로 (free보다 우선하도록 뒤에 마킹).
        occ_cells = self.world_to_cell(endpoints)
        self._mark(self.grid_occ, occ_cells)

        # 4) 로봇 주변은 확실히 free.
        self._mark_robot_free_disk(origin_xy)

    def _mark_robot_free_disk(self, center_xy):
        r = int(ROBOT_FREE_RADIUS_M / GRID_RES_M)
        if r <= 0:
            return
        c = self.world_to_cell(center_xy)
        offs = np.arange(-r, r + 1)
        gx, gy = np.meshgrid(offs, offs)
        disk = (gx ** 2 + gy ** 2) <= r ** 2
        cells = np.stack([c[0] + gx[disk], c[1] + gy[disk]], axis=1)
        self._mark(self.grid_free, cells)

    def compute_frontier(self):
        """free 이면서 4-이웃에 unknown이 있는 셀 = frontier."""
        free_only = self.grid_free & ~self.grid_occ
        unknown = ~self.grid_free & ~self.grid_occ
        neigh = np.zeros_like(unknown)
        neigh[:-1, :] |= unknown[1:, :]
        neigh[1:, :] |= unknown[:-1, :]
        neigh[:, :-1] |= unknown[:, 1:]
        neigh[:, 1:] |= unknown[:, :-1]
        return free_only & neigh

    def select_frontier_goal(self, robot_xy):
        """frontier 덩어리 중 가장 좋은 것의 대표점을 world xy로 반환. 없으면 None.

        점수 = 덩어리 크기 / (1 + 로봇까지 거리) — 크고 가까운 곳 선호.
        blacklist(실패 지점) 근처는 제외.
        """
        frontier = self.compute_frontier()
        if not frontier.any():
            return None

        num, labels = cv2.connectedComponents(frontier.astype(np.uint8))
        best_score = -np.inf
        best_goal = None
        for lbl in range(1, num):
            pts = np.argwhere(labels == lbl)          # (P,2) = [ix, iy]
            if len(pts) < MIN_FRONTIER_CELLS:
                continue
            centroid = pts.mean(axis=0)
            # centroid에 가장 가까운 실제 frontier 셀로 스냅(목표가 격자 안 유효점이 되도록).
            target_cell = pts[np.argmin(((pts - centroid) ** 2).sum(axis=1))]
            goal_xy = self.cell_to_world(target_cell)

            if self._is_blacklisted(goal_xy):
                continue

            dist = float(np.linalg.norm(goal_xy - robot_xy))
            score = len(pts) / (1.0 + dist)
            if score > best_score:
                best_score = score
                best_goal = goal_xy
        return best_goal

    def _is_blacklisted(self, xy):
        for b in self.blacklist:
            if np.linalg.norm(xy - b) < BLACKLIST_RADIUS_M:
                return True
        return False

    def exploration_tick(self):
        """탐색 상태 기계: 목표 감시(도착/막힘/타임아웃) + 없으면 새 목표 선정."""
        if self.latest_pose is None or self.exploration_done:
            return

        # 시작 직후 TF 준비 전엔 스캔이 버려져 2D 격자가 비어 있다. 이때 frontier가
        # 0개라고 '완료'로 래치하면 로봇이 영영 목표를 못 받는다. 맵이 어느 정도
        # 찰 때까지 완료 판정을 보류한다.
        if int(self.grid_free.sum()) < MIN_EXPLORE_FREE_CELLS:
            self.get_logger().info("맵 채우는 중... (탐색 대기)", throttle_duration_sec=3.0)
            return

        robot_xy = np.array([self.latest_pose.position.x, self.latest_pose.position.y])
        now = time.time()

        # 현재 목표가 있으면 진행 상황을 본다.
        if self.current_goal is not None:
            dist = float(np.linalg.norm(self.current_goal - robot_xy))
            if dist < REACH_THRESH_M:
                self.get_logger().info(f"✅ reached goal ({self.current_goal[0]:.1f}, {self.current_goal[1]:.1f})")
                self.current_goal = None
            elif now - self.goal_start_time > GOAL_TIMEOUT_S:
                self.get_logger().warn("⏱️ goal timeout -> blacklist & 재선정")
                self.blacklist.append(self.current_goal)
                self.current_goal = None
            else:
                # 막힘 감지: 거리 개선이 STUCK_TIME_S 동안 없으면 포기.
                if dist < self.goal_best_dist - STUCK_EPS_M:
                    self.goal_best_dist = dist
                    self.goal_last_progress_time = now
                elif now - self.goal_last_progress_time > STUCK_TIME_S:
                    self.get_logger().warn("🧱 stuck -> blacklist & 재선정")
                    self.blacklist.append(self.current_goal)
                    self.current_goal = None
                else:
                    # 목표 유지 중 — localPlanner가 잊지 않게 주기적으로 재발행.
                    self.publish_waypoint(self.current_goal, robot_xy)
                    return

        # 목표가 없으면 새로 고른다.
        goal = self.select_frontier_goal(robot_xy)
        if goal is None:
            self.get_logger().info("🎉 탐색 완료: 더 이상 frontier 없음. 대기.")
            self.exploration_done = True
            return

        self.current_goal = goal
        self.goal_start_time = now
        self.goal_best_dist = float(np.linalg.norm(goal - robot_xy))
        self.goal_last_progress_time = now
        self.get_logger().info(f"🚩 new goal ({goal[0]:.1f}, {goal[1]:.1f}), dist {self.goal_best_dist:.1f}m, "
                               f"blacklist {len(self.blacklist)}")
        self.publish_waypoint(goal, robot_xy)

    def publish_waypoint(self, goal_xy, robot_xy):
        """목표점을 Pose2D(/way_point_with_heading)로 발행 + RViz 목표 마커."""
        heading = float(np.arctan2(goal_xy[1] - robot_xy[1], goal_xy[0] - robot_xy[0]))
        msg = Pose2D()
        msg.x = float(goal_xy[0])
        msg.y = float(goal_xy[1])
        msg.theta = heading
        self.pub_waypoint.publish(msg)

        marker = Marker()
        marker.header.frame_id = config.FRAME_MAP
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "explore_goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(goal_xy[0])
        marker.pose.position.y = float(goal_xy[1])
        marker.pose.position.z = 0.3
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.5
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2
        self.pub_goal_marker.publish(marker)

    def render_explore(self):
        """2D 점유격자를 top-down 이미지로. 회색=free, 흰=obstacle, 초록=frontier,
        파랑=로봇, 빨강=목표. (행=y 위로, 열=x 오른쪽으로)"""
        img = np.full((self.grid_n, self.grid_n, 3), 30, dtype=np.uint8)
        free_only = self.grid_free & ~self.grid_occ
        frontier = self.compute_frontier()

        # ix -> col, iy -> row(위가 +y가 되도록 뒤집음)
        def paint(mask, color):
            ixs, iys = np.where(mask)
            if len(ixs) == 0:
                return
            rows = self.grid_n - 1 - iys
            cols = ixs
            img[rows, cols] = color

        paint(free_only, (90, 90, 90))
        paint(self.grid_occ, (255, 255, 255))
        paint(frontier, (0, 200, 0))

        def dot(xy, color, size=2):
            c = self.world_to_cell(xy)
            if not (0 <= c[0] < self.grid_n and 0 <= c[1] < self.grid_n):
                return
            r = self.grid_n - 1 - c[1]
            cv2.circle(img, (int(c[0]), int(r)), size, color, -1)

        if self.latest_pose is not None:
            dot(np.array([self.latest_pose.position.x, self.latest_pose.position.y]), (255, 120, 0), 3)
        if self.current_goal is not None:
            dot(self.current_goal, (0, 0, 255), 3)
        return img


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
