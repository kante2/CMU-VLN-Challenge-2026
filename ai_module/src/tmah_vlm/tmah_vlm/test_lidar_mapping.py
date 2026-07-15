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
  (3) 복귀  : 더 갈 frontier가 없으면(=탐색 완료) 시작 지점(home)으로 돌아오고 끝낸다.

페이즈: EXPLORING(frontier 탐색) -> RETURNING(원점 복귀) -> DONE(대기, 맵은 유지).
복귀를 시작하면 탐색으로 되돌아가지 않는다(돌아가는 길은 이미 본 공간이라 왔다갔다 방지).

--------------------------------------------------------------------------
완료 조건 (헷갈리기 쉬운 부분이라 여기 요약. 상세는 각 함수 주석 참고)
--------------------------------------------------------------------------
"완료"는 2단계이고, 3D voxel 매핑 자체엔 완료 개념이 없다(노드가 살아있는 한 계속 누적).

  [워밍업]  grid_free >= MIN_EXPLORE_FREE_CELLS 가 될 때까지는 판정 자체를 안 한다.
            (시작 직후엔 TF가 아직 준비 안 돼 스캔이 버려지고 격자가 비어 있는데,
             그걸 'frontier 0개 = 완료'로 오판하면 로봇이 영영 목표를 못 받는다.)
        |
  [1단계] 탐색 완료  : select_frontier_goal()이 None을 NO_FRONTIER_CONFIRM번 "연속"
                       반환 -> start_return_home(). 제어주기가 1초라 약 5초.
                       (한 번 None이라고 바로 끝내면 노이즈로 조기 종료된다.)
        |
  [2단계] 최종 완료  : 원점까지 REACH_THRESH_M 안으로 도착 -> DONE.
                       (또는 RETURN_TIMEOUT_S 초과 시 포기하고 DONE.)

주의: "완료"는 "지도 100%"가 아니라 "쫓아갈 만한 미탐사 경계가 없음"이다. blacklist가
쌓이거나(실패 지점) 격자 범위(±GRID_HALF_EXTENT_M) 밖이면 갈 곳이 남아도 완료로 뜬다.

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
MIN_FRONTIER_CELLS = 4         # 우선적으로 노리는 frontier 덩어리 최소 크기.
SMALL_FRONTIER_CELLS = 2       # 큰 게 없을 때 문/좁은 통로(작은 frontier)라도 시도할 하한.
BLACKLIST_RADIUS_M = 1.5       # 실패한 목표 근처는 당분간 다시 안 고름.
MIN_EXPLORE_FREE_CELLS = 200   # free 셀이 이만큼 쌓이기 전엔 완료 판정 보류.
                               # (시작 직후 TF 준비 전 빈 격자로 '완료' 오판되는 것 방지)
NO_FRONTIER_CONFIRM = 5        # frontier가 이만큼 연속 tick 없으면 탐색 완료로 보고 복귀 시작.
RETURN_TIMEOUT_S = 90.0        # 원점 복귀를 이 시간 안에 못 하면 포기하고 대기.


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
        self.no_frontier_count = 0        # 연속 no-frontier tick 수 (탐색 완료 판정용)

        # 페이즈: EXPLORING(탐색) -> RETURNING(원점 복귀) -> DONE(대기)
        self.phase = "EXPLORING"
        self.home_xy = None               # 시작 지점(복귀 목표). 첫 pose에서 기록.

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
        """frontier = "지금까지 본 빈 공간"과 "아직 못 본 공간"이 맞닿은 셀.

        격자 각 칸은 3가지 상태 중 하나다 (free/occ 두 bool의 조합으로 표현):
          free_only : 관측했고 비어 있음    (grid_free 이고 grid_occ 아님)
          occupied  : 벽/장애물             (grid_occ)      <- 여기선 못 지나감
          unknown   : 아직 한 번도 못 봄    (free도 occ도 아님)

        frontier는 "free_only 이면서 4-이웃 중에 unknown이 있는 칸"이다. 즉 내가 서 있을
        수 있는 빈 공간의 가장자리인데 그 너머는 아직 모르는 곳 -> 저기로 가면 새 공간이
        보인다. 그래서 탐색은 "frontier로 계속 가는 것"이고, 반대로 frontier가 하나도
        없다 = 더 볼 게 없다 = 탐색 완료다.

        벽에 붙은 free 칸은 이웃이 occupied라서 frontier가 아니다(넘어갈 수 없으니 당연).
        열린 문/센서 사거리 끝처럼 unknown과 직접 맞닿은 곳만 frontier가 된다.
        """
        free_only = self.grid_free & ~self.grid_occ
        unknown = ~self.grid_free & ~self.grid_occ
        # 상/하/좌/우로 한 칸씩 밀어서 OR -> "이웃에 unknown이 하나라도 있나"를 한 번에 계산.
        neigh = np.zeros_like(unknown)
        neigh[:-1, :] |= unknown[1:, :]
        neigh[1:, :] |= unknown[:-1, :]
        neigh[:, :-1] |= unknown[:, 1:]
        neigh[:, 1:] |= unknown[:, :-1]
        return free_only & neigh

    def select_frontier_goal(self, robot_xy):
        """다음에 갈 frontier 목표를 world xy로 반환. 갈 곳이 없으면 None.

        ★ 이 함수가 None을 반환하는 것이 곧 "탐색 완료" 판정의 근거다
          (exploration_tick이 이게 NO_FRONTIER_CONFIRM번 연속되면 복귀를 시작한다).

        None이 되는 경우는 셋 중 하나 — 하나라도 아니면 탐색은 계속된다:
          1. frontier 셀이 아예 0개        : 본 빈 공간의 모든 경계가 벽이거나 이미 다 봄.
          2. 남은 덩어리가 전부 너무 작음  : SMALL_FRONTIER_CELLS 미만은 노이즈로 무시.
          3. 남은 덩어리가 전부 blacklist  : 가려다 실패(타임아웃/막힘)한 지점 근처.

        고르는 기준: 점수 = 덩어리 크기 / (1 + 로봇까지 거리) -> 크고 가까운 곳을 선호.
        단 2단계로 본다. 큰 덩어리(>=MIN_FRONTIER_CELLS)를 우선 노리되, 그런 게 하나도
        없으면 작은 덩어리(>=SMALL_FRONTIER_CELLS)라도 노린다. 문/좁은 통로의 frontier는
        몇 칸밖에 안 돼서, 큰 것만 고집하면 옆방으로 못 넘어가고 조기 종료돼 버린다.
        """
        frontier = self.compute_frontier()
        if not frontier.any():
            return None  # (1) 미탐사 경계 자체가 없음 -> 완료 후보

        num, labels = cv2.connectedComponents(frontier.astype(np.uint8))
        # 2단계 선택: 우선 큰 덩어리(>=MIN_FRONTIER_CELLS), 없으면 문/좁은통로 같은
        # 작은 덩어리(>=SMALL_FRONTIER_CELLS)라도 노린다. 점수 = 크기/(1+거리).
        best_big = (-np.inf, None)
        best_small = (-np.inf, None)
        for lbl in range(1, num):
            pts = np.argwhere(labels == lbl)          # (P,2) = [ix, iy]
            n = len(pts)
            if n < SMALL_FRONTIER_CELLS:
                continue  # (2) 너무 작은 덩어리 = 노이즈로 보고 버림
            centroid = pts.mean(axis=0)
            # centroid에 가장 가까운 실제 frontier 셀로 스냅(목표가 격자 안 유효점이 되도록).
            target_cell = pts[np.argmin(((pts - centroid) ** 2).sum(axis=1))]
            goal_xy = self.cell_to_world(target_cell)

            if self._is_blacklisted(goal_xy):
                continue  # (3) 전에 가려다 실패한 곳 -> 다시 시도해봐야 또 막힘

            dist = float(np.linalg.norm(goal_xy - robot_xy))
            score = n / (1.0 + dist)
            if n >= MIN_FRONTIER_CELLS:
                if score > best_big[0]:
                    best_big = (score, goal_xy)
            else:
                if score > best_small[0]:
                    best_small = (score, goal_xy)
        return best_big[1] if best_big[1] is not None else best_small[1]

    def _is_blacklisted(self, xy):
        for b in self.blacklist:
            if np.linalg.norm(xy - b) < BLACKLIST_RADIUS_M:
                return True
        return False

    def exploration_tick(self):
        """페이즈 상태기계: EXPLORING -> RETURNING(원점 복귀) -> DONE. 1초마다 호출.

        읽는 순서:
          1) home 기록 (첫 pose = 시작 지점 = 나중에 돌아올 곳)
          2) 페이즈 분기 (DONE이면 아무것도 안 함 / RETURNING이면 복귀만)
          3) EXPLORING: 워밍업 대기 -> 현재 목표 감시 -> 없으면 새 frontier 목표 선정
             -> 그것도 없으면(연속 N회) 탐색 완료로 보고 복귀 시작
        """
        if self.latest_pose is None:
            return

        robot_xy = np.array([self.latest_pose.position.x, self.latest_pose.position.y])
        now = time.time()

        # 시작 지점을 home으로 기록해 둔다(탐색이 끝나면 여기로 돌아온다).
        if self.home_xy is None:
            self.home_xy = robot_xy.copy()
            self.get_logger().info(f"🏠 home 기록: ({self.home_xy[0]:.1f}, {self.home_xy[1]:.1f})")

        if self.phase == "DONE":
            return
        if self.phase == "RETURNING":
            self.return_home_tick(robot_xy, now)
            return

        # ===== 이하 EXPLORING =====
        # 시작 직후 TF 준비 전엔 스캔이 버려져 2D 격자가 비어 있다. 이때 frontier가
        # 0개라고 '완료'로 판정하면 로봇이 목표를 못 받는다. 맵이 어느 정도 찰 때까지 대기.
        if int(self.grid_free.sum()) < MIN_EXPLORE_FREE_CELLS:
            self.get_logger().info("맵 채우는 중... (탐색 대기)", throttle_duration_sec=3.0)
            return

        # 현재 목표가 있으면 진행 상황을 보고, 도착/타임아웃/막힘이 아니면 그대로 유지한다.
        if self.current_goal is not None:
            dist = float(np.linalg.norm(self.current_goal - robot_xy))

            # 진행(거리 개선) 추적 — 개선되면 막힘 타이머 리셋.
            if dist < self.goal_best_dist - STUCK_EPS_M:
                self.goal_best_dist = dist
                self.goal_last_progress_time = now

            if dist < REACH_THRESH_M:
                self.get_logger().info(f"✅ reached goal ({self.current_goal[0]:.1f}, {self.current_goal[1]:.1f})")
                self.current_goal = None
            elif now - self.goal_start_time > GOAL_TIMEOUT_S:
                self.get_logger().warn("⏱️ goal timeout -> blacklist & 재선정")
                self.blacklist.append(self.current_goal)
                self.current_goal = None
            elif now - self.goal_last_progress_time > STUCK_TIME_S:
                self.get_logger().warn("🧱 stuck -> blacklist & 재선정")
                self.blacklist.append(self.current_goal)
                self.current_goal = None
            else:
                # 목표 유지 중 — localPlanner가 잊지 않게 주기적으로 재발행하고 이 tick 종료.
                self.publish_waypoint(self.current_goal, robot_xy)
                return

        # 목표가 없으면(= 방금 도착했거나 포기했으면) 새 frontier 목표를 고른다.
        goal = self.select_frontier_goal(robot_xy)

        # ★ 탐색 완료 판정 지점 ★
        # goal이 None = "지금 갈 만한 미탐사 경계가 없다". 하지만 한 번 None이라고 바로
        # 끝내면 안 된다 — 로봇이 벽을 보고 있는 순간이나 스캔 노이즈로 frontier가 잠깐
        # 사라질 수 있고, 그걸 완료로 받으면 실제론 절반도 안 훑고 끝나버린다.
        # 그래서 NO_FRONTIER_CONFIRM번 "연속"으로 None일 때만 진짜 완료로 인정한다.
        if goal is None:
            self.no_frontier_count += 1
            if self.no_frontier_count >= NO_FRONTIER_CONFIRM:
                self.get_logger().info("🎉 탐색 완료: 더 이상 frontier 없음 → 원점 복귀 시작")
                self.start_return_home(robot_xy, now)  # -> 여기서 페이즈가 RETURNING으로 바뀐다
            return

        # 목표를 찾았으면 연속 카운트를 리셋한다(완료 판정은 '연속'일 때만 유효하므로).
        self.no_frontier_count = 0
        self.current_goal = goal
        self.goal_start_time = now
        self.goal_best_dist = float(np.linalg.norm(goal - robot_xy))
        self.goal_last_progress_time = now
        self.get_logger().info(f"🚩 new goal ({goal[0]:.1f}, {goal[1]:.1f}), dist {self.goal_best_dist:.1f}m, "
                               f"blacklist {len(self.blacklist)}")
        self.publish_waypoint(goal, robot_xy)

    def start_return_home(self, robot_xy, now):
        """탐색 완료 -> 원점(시작 지점) 복귀 페이즈로 전환."""
        self.phase = "RETURNING"
        self.current_goal = self.home_xy.copy()
        self.goal_start_time = now
        self.goal_best_dist = float(np.linalg.norm(self.home_xy - robot_xy))
        self.goal_last_progress_time = now
        self.get_logger().info(
            f"🏠 복귀 목표 ({self.home_xy[0]:.1f}, {self.home_xy[1]:.1f}), "
            f"남은 거리 {self.goal_best_dist:.1f}m"
        )
        self.publish_waypoint(self.current_goal, robot_xy)

    def return_home_tick(self, robot_xy, now):
        """원점까지 이동 감시. 도착하면 DONE, 너무 오래 걸리면 포기하고 DONE.

        돌아가는 길은 이미 탐색한 공간이라 새 frontier가 거의 안 생긴다. 그래서
        복귀를 시작하면 탐색으로 되돌아가지 않고 복귀에 커밋한다(왔다갔다 방지).
        집은 반드시 가야 하므로 blacklist는 쓰지 않고, 막히면 재발행으로 재시도한다.
        """
        dist = float(np.linalg.norm(self.current_goal - robot_xy))

        if dist < self.goal_best_dist - STUCK_EPS_M:
            self.goal_best_dist = dist
            self.goal_last_progress_time = now

        if dist < REACH_THRESH_M:
            self.get_logger().info("🏁 원점 복귀 완료 — 매핑/탐색 종료. (맵은 계속 유지)")
            self.phase = "DONE"
            self.current_goal = None
            return

        if now - self.goal_start_time > RETURN_TIMEOUT_S:
            self.get_logger().warn(f"⏱️ 복귀 타임아웃 (남은 거리 {dist:.1f}m) — 포기하고 대기.")
            self.phase = "DONE"
            self.current_goal = None
            return

        if now - self.goal_last_progress_time > STUCK_TIME_S:
            # 복귀 중 막힘: 포기하지 않고 진행 타이머만 리셋해 계속 재시도한다.
            self.get_logger().warn(f"🧱 복귀 중 막힘 (남은 거리 {dist:.1f}m) — 재시도.",
                                   throttle_duration_sec=3.0)
            self.goal_last_progress_time = now

        self.publish_waypoint(self.current_goal, robot_xy)

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

        if self.home_xy is not None:
            dot(self.home_xy, (0, 255, 255), 3)          # 노랑 = home(원점)
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
