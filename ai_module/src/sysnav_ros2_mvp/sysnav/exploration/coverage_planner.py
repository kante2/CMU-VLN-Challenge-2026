"""Single-room online occupancy mapping and surface-coverage exploration planning.

In-room exploration follows the SysNav paper (Sec. IV-B-1, "In-room Exploration
Policy"): sample a pose horizon H, define the surface point set S as the
free/non-free boundary (non-free = occupied *and* unknown), and drive the robot
until every surface point has been seen from within d_cover.

Two details decide whether that actually happens:

* S must include the occupied side. With only the unknown side (the classic
  frontier) S empties out the moment the LiDAR has swept the room, and
  exploration declares victory after a couple of viewpoints even though the
  camera - which is what recognizes objects - has barely seen the place. With
  the paper's definition S becomes the wall and furniture surfaces once unknown
  space is gone, so frontier-chasing and surface-inspection are one objective at
  two stages of the same map rather than two phases.
* the covered set must persist. Coverage is `surface_point_mask & observed`, and
  update_from_scan() accumulates `observed` for cells actually seen inside the
  observe radius, so Ŝ (the still-uncovered surface) never silently resets.

Candidates are then chosen by greedy set cover over Ŝ - each pick removes what it
covers before the next - with the score decayed by distance so the robot does not
ping-pong across the room, and the picks TSP-ordered into one tour. Candidates
that scored but were not visited stay in a global horizon for the next cycle
(the paper's rolling local/global window) instead of being resampled from
scratch.
"""

from __future__ import annotations

from itertools import permutations
import heapq
import math
import threading

import cv2
import numpy as np

from sysnav import config
from sysnav.exploration.frontier_extractor import FrontierExtractor
from sysnav.exploration.viewpoint_memory import ViewpointMemory
from sysnav.exploration import visibility_path

_TSP_EXACT_MAX_N = 7
# surface_point_mask()가 8-이웃으로 free/non-free 경계를 찾는다
# (frontier_extractor의 같은 상수와 동일한 정의).
_NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _bresenham(row0: int, col0: int, row1: int, col1: int) -> list[tuple[int, int]]:
    points = []
    dx = abs(col1 - col0)
    dy = -abs(row1 - row0)
    sx = 1 if col0 < col1 else -1
    sy = 1 if row0 < row1 else -1
    error = dx + dy
    row, col = row0, col0
    while True:
        points.append((row, col))
        if row == row1 and col == col1:
            break
        e2 = 2 * error
        if e2 >= dy:
            error += dy
            col += sx
        if e2 <= dx:
            error += dx
            row += sy
    return points


class CoveragePlanner:
    def __init__(self) -> None:
        self.resolution = float(config.MAP_RESOLUTION_M)
        self.size_cells = int(round(config.MAP_SIZE_M / self.resolution))
        self.grid = np.full((self.size_cells, self.size_cells), config.OCC_UNKNOWN, dtype=np.int8)
        # OCC_OCCUPIED은 벽/가구를 구분 안 하고 z-band 안의 아무 point나 다 잡는다. room
        # segmentation은 "벽"만 경계로 써야 하는데(가구가 방을 쪼개면 안 됨), 이 셀에서
        # 지금까지 관측된 point 중 가장 높은 z를 기록해뒀다가, 충분히 높이 닿은 셀만
        # "진짜 벽"으로 보는 데 쓴다 (RoomSegmenter._structural_wall_mask 참고).
        self.max_height = np.full((self.size_cells, self.size_cells), -np.inf, dtype=np.float32)
        # "가까이서 실제로 훑어본" 셀 - frontier와 달리 지도가 known인지가 아니라
        # EXPLORATION_OBSERVE_RADIUS_M 안에서 시야가 뚫린 채 관측됐는지를 기록한다
        # (plan_floor_coverage()의 목표 함수). update_from_scan()이 LiDAR ray를
        # 따라가며 채우므로 별도 LOS 계산이 필요 없다 - ray가 지나간 셀은 정의상
        # 그 순간 로봇에서 실제로 보인 셀이다.
        self.observed = np.zeros((self.size_cells, self.size_cells), dtype=bool)
        self.origin_x: float | None = None
        self.origin_y: float | None = None
        self.frontier_extractor = FrontierExtractor()
        self.sensor_to_base = np.asarray(config.T_SENSOR_TO_BASE, dtype=np.float64)
        self._lock = threading.RLock()
        self._rng = np.random.default_rng()
        # plan_route()가 빈 route를 반환했을 때 "왜"인지 보려고 남기는 진단 정보.
        # sysnav_node.py가 "No reachable frontier remains" 로그에 같이 붙인다.
        self.last_plan_diagnostics: dict = {}
        # anchor cell(frontier 근처 보장 후보)이 plan_route() 호출마다 몇 번 연속
        # 다시 잡혔는지 - 유리창처럼 LiDAR가 절대 못 뚫는 frontier 옆에 서 있으면
        # 이 값이 계속 늘어난다 (anchor_max_revisits 참고).
        self._anchor_visit_counts: dict[tuple[int, int], int] = {}
        # 논문의 global horizon: 점수까지 매겼지만 이번 사이클에 방문하지 않은 후보를
        # 다음 사이클로 넘겨 재사용한다 (plan_route의 rolling window).
        self._global_horizon: list[tuple[int, int]] = []
        # plan_direct_path()가 실패했을 때 "왜"인지 남기는 진단 정보(cross-room
        # navigation처럼 두 점 사이 경로 하나만 구하는 호출용) - last_plan_diagnostics와
        # 같은 목적이지만 plan_route()의 후보 샘플링과는 무관한 별도 호출이라 분리했다.
        self.last_direct_path_diagnostics: dict = {}
        # plan_floor_coverage()가 왜 멈췄는지(목표 도달 / 후보 없음 / 경로 없음) 남긴다.
        self.last_floor_coverage_diagnostics: dict = {}

    def describe_last_plan_failure(self) -> str:
        return ", ".join(f"{key}={value}" for key, value in self.last_plan_diagnostics.items())

    def reset(self, robot_pose: dict | None = None) -> None:
        with self._lock:
            self.grid.fill(config.OCC_UNKNOWN)
            self.max_height.fill(-np.inf)
            self.observed.fill(False)
            self._anchor_visit_counts.clear()
            self._global_horizon.clear()
            if robot_pose is None:
                self.origin_x = None
                self.origin_y = None
            else:
                half = config.MAP_SIZE_M / 2.0
                self.origin_x = float(robot_pose["x"]) - half
                self.origin_y = float(robot_pose["y"]) - half

    def snapshot_grid(self) -> np.ndarray:
        with self._lock:
            return self.grid.copy()

    def snapshot_max_height(self) -> np.ndarray:
        with self._lock:
            return self.max_height.copy()

    def reset_observed(self) -> None:
        """observed 마스크만 비운다 - 다음 순회에서 같은 공간을 다시 훑게 만들 때
        쓴다(EXPLORATION_COVERAGE_ROUNDS). occupancy 지도는 그대로 두는 게 핵심이다:
        벽/장애물 정보는 경로 계획에 계속 필요하고 다시 그릴 이유가 없다."""
        with self._lock:
            self.observed.fill(False)

    def snapshot_observed(self) -> np.ndarray:
        with self._lock:
            return self.observed.copy()

    def floor_coverage(self) -> dict:
        """"지금까지 가까이서 훑어본 바닥 면적"의 진행도. plan_route()의 frontier
        기준(=지도가 known인가)과 달리, 실제로 관측 반경 안에서 본 free 셀만 센다."""
        with self._lock:
            free = self.grid == config.OCC_FREE
            observed_free = free & self.observed
            free_cells = int(free.sum())
            observed_cells = int(observed_free.sum())
        cell_area = self.resolution ** 2
        return {
            "free_cells": free_cells,
            "observed_cells": observed_cells,
            "ratio": (observed_cells / free_cells) if free_cells else 0.0,
            "unobserved_area_m2": (free_cells - observed_cells) * cell_area,
        }

    def surface_point_mask(self, grid: np.ndarray) -> np.ndarray:
        """논문(Sec. IV-B-1)의 surface point set S: free와 non-free(occupied +
        unknown) 사이의 경계. plan_route()가 candidate 점수(wcov)를 매길 때 쓰는 것과
        동일한 마스크이고, 디버그 시각화(exploration_visualizer.py)에도 쓴다.

        frontier_mask()와의 차이가 탐사 동작을 좌우한다: frontier는 unknown 쪽 경계만
        보므로 LiDAR가 방을 한 번 휩쓸면 0이 되어 탐사가 끝나버린다(viewpoint 두어
        곳에서 종료됐던 원인). 반면 S는 unknown이 사라지면 벽·가구 '표면'으로 남아,
        로봇이 모든 표면을 d_cover 안에서 볼 때까지 탐사가 이어진다 - 물체를 보는 건
        카메라이고 물체는 표면에 붙어 있으니 인식 목적과도 맞는다."""
        free = grid == config.OCC_FREE
        non_free = (grid == config.OCC_UNKNOWN) | (grid == config.OCC_OCCUPIED)
        padded = np.pad(non_free, 1, constant_values=False)
        adjacent = np.zeros_like(non_free, dtype=bool)
        for dr, dc in _NEIGHBORS_8:
            adjacent |= padded[1 + dr:1 + dr + grid.shape[0], 1 + dc:1 + dc + grid.shape[1]]
        return free & adjacent

    def frontier_mask(self, grid: np.ndarray) -> np.ndarray:
        """free/unknown 경계만 - "아직 안 가본 공간이 남았는지"를 뜻하는 고전적 frontier.
        surface_point_mask()가 탐사 목표를 정하는 데 쓰이는 것과 달리, 이쪽은 미탐색
        공간을 절대 놓치지 않기 위한 anchor 후보 생성과 진단용으로만 쓴다."""
        return self.frontier_extractor._mask(grid)

    def surface_coverage(self, grid: np.ndarray | None = None) -> dict:
        """S 중 관측 반경 안에서 실제로 본 비율. 논문의 Ŝ(미커버 surface)를 그대로
        보여주는 지표이고, observed 마스크가 누적되므로 런 전체에서 유지된다."""
        with self._lock:
            if grid is None:
                grid = self.grid.copy()
            observed = self.observed.copy()
        surface = self.surface_point_mask(grid)
        total = int(surface.sum())
        covered = int((surface & observed).sum())
        return {
            "total": total,
            "covered": covered,
            "uncovered": total - covered,
            "ratio": (covered / total) if total else 0.0,
        }

    def _ensure_origin(self, pose: dict) -> None:
        if self.origin_x is None:
            half = config.MAP_SIZE_M / 2.0
            self.origin_x = float(pose["x"]) - half
            self.origin_y = float(pose["y"]) - half

    def world_to_grid(self, x: float, y: float) -> tuple[int, int] | None:
        if self.origin_x is None or self.origin_y is None:
            return None
        col = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        if 0 <= row < self.size_cells and 0 <= col < self.size_cells:
            return row, col
        return None

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        if self.origin_x is None or self.origin_y is None:
            raise RuntimeError("Map origin is not initialized")
        return (
            float(self.origin_x + (col + 0.5) * self.resolution),
            float(self.origin_y + (row + 0.5) * self.resolution),
        )

    @staticmethod
    def line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        """start-end 사이 grid cell을 잇는 직선(Bresenham) - Instruction-Following의
        forbidden corridor 마스크 생성(missions/mission3_pipe.py)처럼, 두 셀 사이를
        따라가며 뭔가 칠해야 할 때 쓴다."""
        return _bresenham(start[0], start[1], end[0], end[1])

    def update_from_scan(self, points_sensor: np.ndarray, pose: dict) -> None:
        if points_sensor.size == 0:
            return
        with self._lock:
            self._ensure_origin(pose)
            robot_cell = self.world_to_grid(float(pose["x"]), float(pose["y"]))
            if robot_cell is None:
                return
            points = points_sensor.reshape(-1, 3).astype(np.float64, copy=False)
            homogeneous = np.column_stack([points, np.ones(len(points))])
            points_base = (homogeneous @ self.sensor_to_base.T)[:, :3]
            ranges = np.linalg.norm(points_base[:, :2], axis=1)
            valid = (
                np.isfinite(points_base).all(axis=1)
                & (ranges >= config.MAP_MIN_RANGE_M)
                & (ranges <= config.MAP_MAX_RANGE_M)
                & (points_base[:, 2] >= config.MAP_OBSTACLE_Z_MIN_M)
                & (points_base[:, 2] <= config.MAP_OBSTACLE_Z_MAX_M)
            )
            points_base = points_base[valid]
            if len(points_base) > config.MAP_MAX_RAYS_PER_SCAN:
                points_base = points_base[np.linspace(0, len(points_base) - 1, config.MAP_MAX_RAYS_PER_SCAN, dtype=np.int64)]

            observe_radius_cells_sq = (config.EXPLORATION_OBSERVE_RADIUS_M / self.resolution) ** 2
            yaw = float(pose["yaw"])
            x_map = math.cos(yaw) * points_base[:, 0] - math.sin(yaw) * points_base[:, 1] + float(pose["x"])
            y_map = math.sin(yaw) * points_base[:, 0] + math.cos(yaw) * points_base[:, 1] + float(pose["y"])
            endpoints = []
            for x, y, z in zip(x_map, y_map, points_base[:, 2]):
                endpoint = self.world_to_grid(float(x), float(y))
                if endpoint is None:
                    continue
                ray = _bresenham(robot_cell[0], robot_cell[1], endpoint[0], endpoint[1])
                # 이미 벽으로 확정된 셀을 만나면 ray를 거기서 멈춘다 - 실제 LiDAR는
                # 벽을 못 뚫으므로, 이 point가 그 벽 너머까지 도달했다는 건(반사 노이즈,
                # multi-path, 혹은 Unity raycast가 얇은 콜라이더를 놓친 경우 등 원인이
                # 뭐든) 신뢰할 수 없다는 뜻이다. 예전엔 "그 한 칸만 안 덮어쓰고" 계속
                # 진행해서 벽 너머 공간을 통째로 free로 잘못 마킹했었다(탐색 디버그
                # 이미지에서 벽을 뚫고 나가는 것처럼 보이는 frontier의 원인).
                blocked = False
                # ray를 따라 여기까지 왔다는 것 자체가 이 셀이 지금 로봇에서 보였다는
                # 뜻이므로, 관측 반경 안이면 observed로 표시한다. ray는 로봇에서
                # 바깥으로 순서대로 나아가니 거리는 단조 증가한다 - 반경을 한 번
                # 벗어나면 그 뒤로는 검사할 필요가 없다(셀마다 hypot을 부르면
                # 스캔당 수만 번이라 매핑이 눈에 띄게 느려진다).
                within_observe_radius = True
                for row, col in ray[:-1]:
                    if self.grid[row, col] == config.OCC_OCCUPIED:
                        blocked = True
                        break
                    self.grid[row, col] = config.OCC_FREE
                    if within_observe_radius:
                        dr = row - robot_cell[0]
                        dc = col - robot_cell[1]
                        if dr * dr + dc * dc <= observe_radius_cells_sq:
                            self.observed[row, col] = True
                        else:
                            within_observe_radius = False
                if blocked:
                    continue  # 벽 너머로 찍힌 것으로 추정되는 endpoint도 신뢰 안 함(occupied로도 안 찍음)
                endpoints.append((endpoint[0], endpoint[1], float(z)))
            for row, col, z in endpoints:
                self.grid[row, col] = config.OCC_OCCUPIED
                if z > self.max_height[row, col]:
                    self.max_height[row, col] = z
            rr, cc = robot_cell
            radius = max(1, int(round(0.35 / self.resolution)))
            row_slice = slice(max(0, rr - radius), min(self.size_cells, rr + radius + 1))
            col_slice = slice(max(0, cc - radius), min(self.size_cells, cc + radius + 1))
            # 이 박스는 반경이 0.35m(=clearance보다 넓다)라, 무조건 FREE로 덮어쓰면
            # 로봇이 벽에 붙어 지나갈 때 벽 셀까지 지워버린다. 그러면 지도에서 벽이
            # 사라져 그 자리가 traversable로 보이고, 목표/경로가 벽 안쪽에 잡혀
            # 로봇이 벽에 박거나(goals on OCCUPIED) 벽 너머 유령 공간으로 새어나가
            # traversable 그래프가 쪼개진다(도달 가능한 후보가 없어 탐사가 조기 종료).
            # 그래서 UNKNOWN만 FREE로 풀고, 이미 관측된 장애물은 건드리지 않는다.
            box = self.grid[row_slice, col_slice]
            box[box == config.OCC_UNKNOWN] = config.OCC_FREE
            # 로봇이 서 있는 자리는 당연히 관측된 것으로 본다.
            self.observed[row_slice, col_slice] = True

    @staticmethod
    def _nearest_traversable(traversable: np.ndarray, row: int, col: int, radius: int = 8) -> tuple[int, int] | None:
        rows, cols = traversable.shape
        candidates = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols and traversable[nr, nc]:
                    candidates.append((dr * dr + dc * dc, nr, nc))
        if not candidates:
            return None
        _, nr, nc = min(candidates)
        return nr, nc

    @staticmethod
    def _room_scoped_traversable(
        traversable: np.ndarray,
        robot_cell: tuple[int, int],
        room_segmentation: dict | None,
    ) -> np.ndarray:
        """room_segmentation(room_segmenter.RoomSegmenter.segment()의 결과)에서 로봇이
        지금 있는 방의 room_id를 찾아, traversable을 그 방 mask로만 제한한다. room
        segmentation이 없거나(아직 계산 전) 로봇 위치가 어느 방으로도 분류 안 됐으면
        (예: 문 한복판, watershed boundary) 원래 traversable을 그대로 반환한다."""
        if not room_segmentation:
            return traversable
        labels = room_segmentation.get("labels")
        if labels is None or labels.shape != traversable.shape:
            return traversable
        row, col = robot_cell
        if not (0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]):
            return traversable
        room_id = int(labels[row, col])
        if room_id <= 0:
            return traversable
        return traversable & (labels == room_id)

    @staticmethod
    def _astar_path(traversable: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
        """4-connected A*. 거리뿐 아니라 실제 셀 경로(벽을 피해간 경로)를 반환한다 -
        최종 목적지 하나만 던지는 대신, 이 경로를 따라 중간 waypoint를 만들기 위함."""
        if start == goal:
            return [start]
        rows, cols = traversable.shape
        queue = [(0.0, 0.0, start)]
        best = {start: 0.0}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        while queue:
            _, cost, current = heapq.heappop(queue)
            if current == goal:
                path = [current]
                while path[-1] != start:
                    path.append(came_from[path[-1]])
                path.reverse()
                return path
            if cost > best.get(current, float("inf")):
                continue
            row, col = current
            for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if not (0 <= nr < rows and 0 <= nc < cols and traversable[nr, nc]):
                    continue
                next_cost = cost + 1.0
                neighbor = (nr, nc)
                if next_cost >= best.get(neighbor, float("inf")):
                    continue
                best[neighbor] = next_cost
                came_from[neighbor] = current
                heuristic = abs(goal[0] - nr) + abs(goal[1] - nc)
                heapq.heappush(queue, (next_cost + heuristic, next_cost, neighbor))
        return None

    def _simplify_path_indices(self, path: list[tuple[int, int]], blocking: np.ndarray) -> list[int]:
        """A* 경로를 hop index 리스트로 줄인다 (string-pulling). 각 hop에서 다음 hop까지는
        - EXPLORATION_PATH_WAYPOINT_SPACING_M 이내면서
        - inflated 그리드 기준 line-of-sight가 뚫려 있는 한(즉 그 사이에 벽이 없는 한)
        최대한 멀리 건너뛴다. 코너/문틀에서는 LOS가 막히는 지점에서 자동으로 hop이 촘촘해지고,
        뚫린 직선 구간에서는 hop이 줄어든다 - 고정 간격 downsample과 달리 hop 사이 직선이
        벽을 스치듯 지나가는 경우가 없다."""
        n = len(path)
        if n <= 1:
            return [0] if path else []
        spacing_cells = max(1, int(round(config.EXPLORATION_PATH_WAYPOINT_SPACING_M / self.resolution)))
        last_idx = n - 1
        indices = [0]
        i = 0
        while i < last_idx:
            cap = min(last_idx, i + spacing_cells)
            j = cap
            while j > i + 1 and not self._line_of_sight(blocking, path[i], path[j]):
                j -= 1
            indices.append(j)
            i = j
        return indices

    def _leg_waypoints(
        self,
        path: list[tuple[int, int]],
        blocking: np.ndarray,
        final_theta: float | None,
        credited_len: int,
    ) -> list[dict]:
        """A* 경로(path)를 _simplify_path_indices로 줄인 hop들로 waypoint dict 목록을 만든다.
        마지막 hop만 candidate의 실제 score/coverage_score/theta를 지니고, 중간 hop들은
        진행 방향을 바라보며 score 0으로 그냥 지나가는 경유점이다."""
        hop_indices = self._simplify_path_indices(path, blocking)[1:] or [len(path) - 1]
        last_idx = len(path) - 1

        waypoints = []
        for i, idx in enumerate(hop_indices):
            x, y = self.grid_to_world(*path[idx])
            is_final = idx == last_idx
            if is_final and final_theta is not None:
                theta = final_theta
            else:
                look_idx = hop_indices[i + 1] if i + 1 < len(hop_indices) else idx
                nx, ny = self.grid_to_world(*path[look_idx])
                theta = math.atan2(ny - y, nx - x) if (nx, ny) != (x, y) else 0.0
            waypoints.append({
                "x": x,
                "y": y,
                "theta": theta,
                "score": float(credited_len) if is_final else 0.0,
                "coverage_score": credited_len if is_final else 0,
                "path_distance_m": idx * self.resolution,
                # 진짜 candidate(=coverage를 credit받은 지점)인지, 그냥 거쳐가는 중간
                # hop인지 구분한다. viewpoint_memory에는 진짜 candidate만 기록해야 한다 -
                # 중간 hop까지 다 "방문함"으로 찍으면 방금 지나온 복도 전체가 반경
                # VIEWPOINT_MIN_DISTANCE_M 안에서 후보 제외 대상이 되어, 몇 번만 이동해도
                # 새로 뽑는 candidate가 죄다 근처-방문 판정으로 걸러져 탐색이 조기 종료된다.
                "is_viewpoint": is_final,
            })
        return waypoints

    @staticmethod
    def _waypoints_from_world_polyline(
        points: list[tuple[float, float]], final_theta: float | None, credited_len: int
    ) -> list[dict]:
        """visibility_path.shortest_path()의 (x, y) polyline을 _leg_waypoints와
        같은 waypoint dict 포맷으로 변환한다. polyline의 각 segment는 이미
        Polygon.covers()로 충돌 없음이 검증돼 있으므로(grid LOS 재확인 불필요),
        EXPLORATION_PATH_WAYPOINT_SPACING_M 간격으로 arc-length를 따라 다시
        샘플링만 해서 hop을 만든다."""
        if len(points) < 2:
            return []
        cumulative = [0.0]
        for a, b in zip(points, points[1:]):
            cumulative.append(cumulative[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        total = cumulative[-1]
        if total < 1e-9:
            return []

        def point_at(dist: float) -> tuple[float, float]:
            for i in range(1, len(cumulative)):
                if dist <= cumulative[i] + 1e-9:
                    a, b = points[i - 1], points[i]
                    seg_len = cumulative[i] - cumulative[i - 1]
                    t = 0.0 if seg_len < 1e-9 else (dist - cumulative[i - 1]) / seg_len
                    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            return points[-1]

        spacing = max(config.EXPLORATION_PATH_WAYPOINT_SPACING_M, 1e-3)
        n_hops = max(1, int(math.ceil(total / spacing)))
        dists = [min(total, spacing * k) for k in range(1, n_hops + 1)]
        if dists[-1] < total - 1e-6:
            dists.append(total)

        waypoints = []
        for i, dist in enumerate(dists):
            x, y = point_at(dist)
            is_final = i == len(dists) - 1
            if is_final and final_theta is not None:
                theta = final_theta
            else:
                nx, ny = point_at(dists[i + 1]) if i + 1 < len(dists) else (x, y)
                theta = math.atan2(ny - y, nx - x) if (nx, ny) != (x, y) else 0.0
            waypoints.append({
                "x": x,
                "y": y,
                "theta": theta,
                "score": float(credited_len) if is_final else 0.0,
                "coverage_score": credited_len if is_final else 0,
                "path_distance_m": dist,
                "is_viewpoint": is_final,
            })
        return waypoints

    @staticmethod
    def _has_uncovered_nearby(
        cell: tuple[int, int], uncovered_mask: np.ndarray, radius_cells: float
    ) -> bool:
        """cell 주변 d_cover 안에 아직 안 덮인 surface point가 하나라도 있는지.

        후보를 "예전에 근처를 지나갔다"는 이유로 버리기 전에, 실제로 덮을 게 남았는지
        확인하는 데 쓴다 (line-of-sight까지 보지 않는 값싼 사전 판정 - 정확한 점수는
        나중에 scov에서 LOS까지 확인해서 매긴다)."""
        radius = int(math.ceil(radius_cells))
        row0 = max(0, cell[0] - radius)
        row1 = min(uncovered_mask.shape[0], cell[0] + radius + 1)
        col0 = max(0, cell[1] - radius)
        col1 = min(uncovered_mask.shape[1], cell[1] + radius + 1)
        return bool(uncovered_mask[row0:row1, col0:col1].any())

    @staticmethod
    def _line_of_sight(occupied: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> bool:
        """start/end 자체는 항상 FREE라는 전제(둘 다 traversable/surface 셀)이므로 중간 셀만 검사한다."""
        for row, col in _bresenham(start[0], start[1], end[0], end[1])[1:-1]:
            if occupied[row, col]:
                return False
        return True

    @staticmethod
    def _tour_cost(start: tuple[int, int], ordered: list[tuple[int, int]]) -> float:
        total = 0.0
        current = start
        for cell in ordered:
            total += math.hypot(cell[0] - current[0], cell[1] - current[1])
            current = cell
        return total

    def _nearest_neighbor_tour(self, start: tuple[int, int], cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        remaining = list(cells)
        tour = []
        current = start
        while remaining:
            nxt = min(remaining, key=lambda c: math.hypot(c[0] - current[0], c[1] - current[1]))
            tour.append(nxt)
            remaining.remove(nxt)
            current = nxt
        return tour

    def _two_opt(self, start: tuple[int, int], tour: list[tuple[int, int]]) -> list[tuple[int, int]]:
        improved = True
        while improved:
            improved = False
            for i in range(len(tour) - 1):
                for j in range(i + 1, len(tour)):
                    candidate = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                    if self._tour_cost(start, candidate) < self._tour_cost(start, tour):
                        tour = candidate
                        improved = True
        return tour

    def _solve_tsp(self, start: tuple[int, int], cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """선택된 candidate 집합에 대해 TSP를 풀어 방문 순서를 정한다."""
        if not cells:
            return []
        if len(cells) <= _TSP_EXACT_MAX_N:
            return min(
                (list(perm) for perm in permutations(cells)),
                key=lambda perm: self._tour_cost(start, perm),
            )
        return self._two_opt(start, self._nearest_neighbor_tour(start, cells))

    def _fail(self, diag: dict, reason: str) -> list[dict]:
        """plan_route()가 빈 route를 반환할 때, 어느 단계에서 왜 막혔는지
        self.last_plan_diagnostics에 남긴다 - "No reachable frontier remains"만
        보고는 원인을 알 수 없어서(surface_points가 없는지, candidate가 없는지,
        anchor가 A*/방문이력 때문에 걸러졌는지 등) 로그 한 줄로 바로 보이게 한다."""
        diag["reason"] = reason
        self.last_plan_diagnostics = diag
        return []

    def plan_route(
        self,
        robot_pose: dict,
        viewpoint_memory: ViewpointMemory,
        room_segmentation: dict | None = None,
    ) -> list[dict]:
        diag: dict = {}
        with self._lock:
            grid = self.grid.copy()
            origin_ready = self.origin_x is not None
        if not origin_ready:
            return self._fail(diag, "origin_not_ready")
        robot_cell = self.world_to_grid(robot_pose["x"], robot_pose["y"])
        if robot_cell is None:
            return self._fail(diag, "robot_cell_out_of_map")
        occupied = (grid == config.OCC_OCCUPIED).astype(np.uint8)
        inflation = max(1, int(round(config.ROBOT_CLEARANCE_M / self.resolution)))
        inflated = cv2.dilate(occupied, np.ones((2 * inflation + 1, 2 * inflation + 1), np.uint8)).astype(bool)
        traversable = (grid == config.OCC_FREE) & (~inflated)
        start = self._nearest_traversable(traversable, *robot_cell, radius=10)
        diag["robot_cell"] = robot_cell
        if start is None:
            return self._fail(diag, "robot_not_near_any_traversable_cell")
        diag["start_cell"] = start
        # hop-to-hop 경로 안전성 체크(_leg_waypoints)는 로봇 몸체가 실제로 못 지나가면
        # 안 되니 전체 clearance(inflated)를 그대로 써야 한다.
        nav_blocking = inflated

        # 반면 "이 candidate에서 이 surface point가 보이는가"(scov)는 몸체 clearance가
        # 아니라 순수 가시선 문제라서, inflated를 그대로 쓰면 문제가 생긴다: frontier는
        # 정의상 벽 바로 옆(= clearance 버퍼 안)에 자주 있는데, candidate는 항상 버퍼
        # 밖에 서 있어야 하니, 그 사이 직선이 "실제 장애물은 없는데" 버퍼 셀 한두 개를
        # 스치기만 해도 안 보인다고 잘못 판정돼서 coverage 점수가 0이 돼버린다(문 근처
        # frontier가 이래서 계속 후보에서 못 뽑히는 원인이 됐었다). 그래서 LOS 차단에는
        # 대각선 코너 스침만 막을 정도의 작은 margin만 쓴다.
        los_margin = max(1, int(round(config.FRONTIER_LOS_WALL_MARGIN_M / self.resolution)))
        los_blocking = cv2.dilate(occupied, np.ones((2 * los_margin + 1, 2 * los_margin + 1), np.uint8)).astype(bool)

        # 논문의 surface point set S(free/non-free 경계)와, 그중 아직 가까이서 보지
        # 못한 Ŝ. 커버 여부는 observed 마스크(update_from_scan이 관측 반경 안에서 실제로
        # 보인 셀만 기록)로 판단하므로 사이클을 넘어 누적된다 - 예전처럼 매 사이클
        # frontier를 처음부터 다시 세는 방식은 "이미 훑은 표면"을 기억하지 못했다.
        with self._lock:
            observed = self.observed.copy()
        surface_mask = self.surface_point_mask(grid)
        uncovered_mask = surface_mask & (~observed)
        surface_points = {tuple(cell) for cell in np.argwhere(uncovered_mask)}
        diag["surface_point_count"] = int(surface_mask.sum())
        diag["uncovered_surface_count"] = len(surface_points)
        if not surface_points:
            return self._fail(diag, "all_surface_points_covered")

        # anchor(미탐색 공간 보장)는 여전히 unknown 쪽 경계 기준이다 - 표면이 남았다는
        # 것과 "아직 안 가본 공간이 있다"는 것은 다른 문제라서 섞으면 안 된다.
        frontier_mask = self.frontier_mask(grid)
        diag["frontier_count"] = int(frontier_mask.sum())

        # 논문의 H_trav = {c in H | traversable(c) & m_j^r(c)=1} - candidate를 "지금 로봇이
        # 있는 방" 안으로만 제한한다. room 구분이 없으면(또는 로봇 위치가 아직 room으로
        # 분류 안 됐으면) 맵 전체에서 샘플링하던 예전 동작으로 자연스럽게 폴백한다.
        #
        # 이렇게 room으로 제한하는 이유: room 제한이 없으면 탐색이 진행될수록 traversable
        # 영역(=지금까지 본 맵 전체)이 계속 커지는데, 60개를 맵 전체에서 무작위로 뽑다 보니
        # 남은 frontier(예: 문 하나)가 작고 멀면 운 나쁘게 계속 못 뽑아서 "후보가 하나도 없다"
        # (No reachable frontier remains)로 잘못 실패하는 경우가 생겼다. 방 안으로 제한하면
        # 맵 전체가 아무리 커져도 후보 pool은 "지금 방 크기"로 고정돼서 이 문제가 없다.
        active_traversable = self._room_scoped_traversable(traversable, robot_cell, room_segmentation)
        traversable_cells = np.argwhere(active_traversable)
        diag["room_scoped_traversable_cell_count"] = len(traversable_cells)
        diag["fell_back_to_whole_map"] = False
        if len(traversable_cells) == 0:
            # 지금 방 안에 더 뽑을 candidate가 없다(방을 다 봤을 가능성) -> room 제한 없이
            # 전체 traversable에서 다시 시도한다 (cross-room 정책이 아직 없어서, 완전히
            # 포기하는 대신 예전처럼 맵 전체를 보는 걸로 안전하게 폴백).
            active_traversable = traversable
            traversable_cells = np.argwhere(traversable)
            diag["fell_back_to_whole_map"] = True
        diag["active_traversable_cell_count"] = len(traversable_cells)
        if len(traversable_cells) == 0:
            return self._fail(diag, "no_traversable_cells_anywhere")
        sample_n = min(config.EXPLORATION_CANDIDATE_SAMPLES, len(traversable_cells))
        sample_idx = self._rng.choice(len(traversable_cells), size=sample_n, replace=False)
        pool_cells = [(int(traversable_cells[idx][0]), int(traversable_cells[idx][1])) for idx in sample_idx]

        # 순수 무작위 샘플링만 쓰면, 탐색이 진행돼서 맵(혹은 방을 다 봐서 폴백한 전체 맵)이
        # 커질수록 작고 먼 frontier(예: 문 하나)를 우연히 못 뽑을 확률이 계속 올라간다 -
        # 실제로 이것 때문에 후보가 하나도 안 뽑혀서 "No reachable frontier remains"로
        # 잘못 실패하는 걸 겪었다. 그래서 무작위 샘플링과 별개로, frontier_mask의 연결
        # component마다(FrontierExtractor.extract()의 FRONTIER_MIN_CLUSTER_CELLS 최소
        # 크기 필터를 안 거친, 아무리 작아도 전부 잡는 버전) 그 근처 traversable cell을
        # "보장된" 후보로 추가한다 - 맵 크기와 무관하게, 문 하나짜리 작은 frontier도
        # 후보 풀에서 절대 안 놓친다.
        #
        # 이 anchor 탐색은 반드시 room-scope 없는 전체 traversable에서 해야 한다 (room으로
        # 제한된 active_traversable이 아니라) - 문(doorway)은 두 방 경계에 걸친 좁은 지점이라,
        # frontier 셀 바로 근처의 traversable cell들이 로봇이 있는 방이 아니라 반대편 방/
        # watershed 경계(미배정)로 라벨링되는 경우가 흔하다. anchor를 room으로 제한해버리면
        # 그런 경우 "frontier는 분명 있는데 이 방 안에서는 근처에 설 자리가 없다"고 오판해서
        # anchor를 못 찾는다 - "방을 다 보면 벽/미지 경계로 새 탐색 지점을 잘 찍던" 예전 동작이
        # room-scope를 넣으면서 깨진 지점이 바로 여기였다. random 샘플링만 방 안 커버리지를
        # 우선하도록 room-scope를 적용하고, anchor(절대 놓치면 안 되는 안전장치)는 항상 전체
        # 맵에서 찾는다.
        anchor_radius = max(1, int(round(1.0 / self.resolution)))
        anchor_cells: set[tuple[int, int]] = set()
        n_frontier_components, frontier_labels = cv2.connectedComponents(frontier_mask.astype(np.uint8), connectivity=8)
        for label in range(1, n_frontier_components):
            rows, cols = np.nonzero(frontier_labels == label)
            if len(rows) == 0:
                continue
            anchor = self._nearest_traversable(
                traversable, int(rows[0]), int(cols[0]), radius=anchor_radius
            )
            if anchor is not None:
                anchor_cells.add(anchor)

        # 이 anchor가 이번까지 몇 번 연속 다시 잡혔는지 센다. 진짜 갈 수 있는 곳이면
        # 한두 번 방문 후 그 옆 unknown 셀이 free/occupied로 풀려서 frontier가 사라지고
        # 다시는 anchor로 안 잡힌다. 계속 잡힌다는 건(예: 유리창이라 LiDAR가 절대
        # 못 뚫는 경우) 이 지점이 구조적으로 안 풀린다는 뜻이므로, 일정 횟수를 넘으면
        # is_near_visited 예외 자격을 박탈해서(아래 루프) 결국 후보에서 빠지게 한다 -
        # 안 그러면 "도착 -> 다시 같은 anchor 선택 -> 도착 -> ..." 무한 루프에 걸린다.
        stale_anchor_cells: set[tuple[int, int]] = set()
        for cell in anchor_cells:
            count = self._anchor_visit_counts.get(cell, 0) + 1
            self._anchor_visit_counts[cell] = count
            if count > config.EXPLORATION_ANCHOR_MAX_REVISITS:
                stale_anchor_cells.add(cell)
        anchor_cells -= stale_anchor_cells
        diag["stale_anchor_count"] = len(stale_anchor_cells)

        # 미커버 표면(Ŝ) 옆에 설 자리도 "보장된" 후보로 넣는다. frontier anchor가 미탐색
        # 공간을 놓치지 않게 하는 것과 같은 이유로, 남은 표면이 작을 때 무작위 샘플링이
        # 우연히 그 근처를 못 뽑아 조기 종료하는 걸 막는다.
        surface_anchor_cells: set[tuple[int, int]] = set()
        uncovered_cells = np.argwhere(uncovered_mask)
        if len(uncovered_cells):
            stride = max(1, len(uncovered_cells) // 25)
            for cell in uncovered_cells[::stride]:
                anchor = self._nearest_traversable(
                    traversable, int(cell[0]), int(cell[1]), radius=inflation + 3
                )
                if anchor is not None:
                    surface_anchor_cells.add(anchor)
        diag["surface_anchor_count"] = len(surface_anchor_cells)

        # 논문의 rolling window: 지난 사이클에 점수까지 매겼지만 방문하지 않은 후보를
        # 버리지 않고 global horizon으로 넘겨 재사용한다. 매번 처음부터 샘플링하면 이미
        # 계산해둔 좋은 후보를 잃고, 남은 영역이 작을 때 못 뽑아서 조기 종료한다.
        pool_cells = list(dict.fromkeys(
            pool_cells
            + list(anchor_cells)
            + list(stale_anchor_cells)
            + list(surface_anchor_cells)
            + list(self._global_horizon)
        ))  # 순서 유지하며 중복 제거
        diag["anchor_count"] = len(anchor_cells)
        diag["global_horizon_in"] = len(self._global_horizon)
        diag["pool_cell_count"] = len(pool_cells)

        # Scov 반경(논문의 d_cover). 반드시 "커버로 인정되는" 반경과 같아야 한다:
        # 커버 판정은 observed 마스크로 하고 그건 EXPLORATION_OBSERVE_RADIUS_M 안에서만
        # 채워지므로, 점수를 그보다 넓은 반경으로 매기면 플래너는 덮을 거라 믿고 갔는데
        # 실제로는 안 덮여서 같은 후보를 영원히 재선택한다(실측: uncovered가 83에서
        # 멈춘 채 사이클만 소모). 둘 중 좁은 쪽을 쓴다.
        cover_radius_m = min(
            config.FRONTIER_COVERAGE_RADIUS_M, config.EXPLORATION_OBSERVE_RADIUS_M
        )
        cover_radius_cells = cover_radius_m / self.resolution
        diag["cover_radius_m"] = cover_radius_m

        candidates = []
        path_from_start: dict[tuple[int, int], list[tuple[int, int]]] = {}
        rejected_by_visited = 0
        rejected_by_astar = 0
        anchors_rejected_by_astar = 0
        for cell in pool_cells:
            # is_near_visited(방문 좌표 근처면 건너뛰기)는 목표가 frontier였을 때 제자리
            # 맴돌기를 막기 위한 장치였다. 지금은 목표가 "아직 안 덮인 표면(Ŝ)"이고
            # observed 마스크가 누적되므로, 이미 훑은 곳은 gain이 0이라 애초에 선택되지
            # 않는다 - 즉 이 필터는 중복이면서, 아직 안 덮인 표면 앞 자리까지 "근처를
            # 지나간 적 있다"는 이유로 막아버린다(실측: 후보 71개 중 69개가 이 필터에
            # 걸려 같은 방을 5사이클이 아니라 32사이클에 훑었다). 그래서 실제로 덮을
            # 표면이 남은 후보는 통과시키고, gain이 없는 후보에만 적용한다.
            if cell not in anchor_cells and cell not in surface_anchor_cells:
                x, y = self.grid_to_world(*cell)
                if viewpoint_memory.is_near_visited(x, y) and not self._has_uncovered_nearby(
                    cell, uncovered_mask, cover_radius_cells
                ):
                    rejected_by_visited += 1
                    continue
            path = self._astar_path(traversable, start, cell)
            if path is None:
                rejected_by_astar += 1
                if cell in anchor_cells:
                    anchors_rejected_by_astar += 1
                continue
            path_from_start[cell] = path
            candidates.append(cell)
        diag["rejected_by_visited"] = rejected_by_visited
        diag["rejected_by_astar"] = rejected_by_astar
        diag["anchors_rejected_by_astar"] = anchors_rejected_by_astar
        diag["candidate_count"] = len(candidates)
        if not candidates:
            return self._fail(diag, "all_pool_cells_rejected")

        # Scov(c) = c에서 d_cover 이내 + line-of-sight로 보이는 surface point들
        scov: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for c in candidates:
            covered = {
                p for p in surface_points
                if math.hypot(p[0] - c[0], p[1] - c[1]) <= cover_radius_cells
                and self._line_of_sight(los_blocking, c, p)
            }
            scov[c] = covered
        diag["candidates_with_nonempty_scov"] = sum(1 for c in candidates if scov[c])

        # 후보 선택: 남은 Ŝ에 대한 greedy set cover. 하나 고를 때마다 그 후보가 덮는
        # surface point를 남은 집합에서 빼고 다시 고르므로, 같은 벽만 쳐다보는 후보가
        # 여러 개 뽑히지 않는다. 점수는 로봇(또는 직전에 고른 후보)으로부터의 거리로
        # 감쇠시킨다 - 순수 점수만 쓰면 방 양쪽 끝을 번갈아 고르며 왔다갔다한다.
        # 멈추는 기준(MIN_SCORE_DELTA)은 감쇠 전 순수 점수로 판단한다: "덮을 게 남았나"와
        # "무엇을 먼저 갈까"는 다른 문제다.
        halflife_cells = max(
            1e-6, config.EXPLORATION_DISTANCE_PENALTY_HALFLIFE_M / self.resolution
        )
        remaining = set(surface_points)
        available = list(candidates)
        picked: list[tuple[int, int]] = []
        best_credited: dict[tuple[int, int], set[tuple[int, int]]] = {}
        anchor_cell = start
        while len(picked) < config.EXPLORATION_MAX_CANDIDATES_PER_CYCLE and remaining:
            best_cell = None
            best_priority = 0.0
            best_gain: set[tuple[int, int]] = set()
            for cell in available:
                gain = scov[cell] & remaining
                if len(gain) < config.EXPLORATION_MIN_SCORE_DELTA:
                    continue
                distance = math.hypot(cell[0] - anchor_cell[0], cell[1] - anchor_cell[1])
                priority = len(gain) / (1.0 + distance / halflife_cells)
                if priority > best_priority:
                    best_priority = priority
                    best_cell = cell
                    best_gain = gain
            if best_cell is None:
                break
            picked.append(best_cell)
            best_credited[best_cell] = best_gain
            remaining -= best_gain
            available = [cell for cell in available if cell != best_cell]
            anchor_cell = best_cell

        if not picked:
            # MIN_SCORE_DELTA를 넘는 후보가 없다. 바로 포기하면 문처럼 멀어서 점수가 낮게
            # 나오는 것 말고는 갈 곳이 없는 상황(방을 거의 다 본 뒤 문틈만 남은 경우)에서
            # 탐색이 조기 종료되므로, 조금이라도 덮는 후보 중 최고점 하나는 살린다.
            fallback = max(candidates, key=lambda c: len(scov[c]), default=None)
            if fallback is not None and scov[fallback]:
                picked = [fallback]
                best_credited = {fallback: set(scov[fallback])}

        if not picked:
            self._global_horizon = []
            return self._fail(diag, "no_candidate_had_any_visible_uncovered_surface_point")

        best_ordered = self._solve_tsp(start, picked) if len(picked) > 1 else picked
        # 점수는 있었지만 이번에 방문하지 않는 후보는 다음 사이클로 넘긴다.
        leftover = [
            cell for cell in candidates
            if cell not in best_credited and scov[cell]
        ]
        self._global_horizon = leftover[: config.EXPLORATION_GLOBAL_HORIZON_MAX]
        diag["picked_count"] = len(best_ordered)
        diag["global_horizon_out"] = len(self._global_horizon)

        diag["reason"] = "ok"
        self.last_plan_diagnostics = diag

        # 각 candidate를 최종 목적지 하나로 바로 던지지 않고, A* 경로(벽을 피해가는 경로)를
        # 따라 짧은 간격의 중간 waypoint로 잘라서 순서대로 내보낸다 (벽 너머로 직선 goal을
        # 찍어서 로봇이 벽에 막히는 문제 방지).
        route: list[dict] = []
        leg_start = start
        leg_start_world = self.grid_to_world(*start)
        # Same reasoning as plan_recovery_patrol()/plan_direct_path(): grid A*'s
        # LOS-based hop simplification only "sees" collisions at
        # MAP_RESOLUTION_M, so a leg can graze clearance in a small cluttered
        # room. Try the continuous visibility-graph route first, falling back
        # to the existing (cached-or-fresh) grid A* unchanged if unavailable.
        polygon = visibility_path.build_traversable_polygon(traversable, self.grid_to_world)
        for cell in best_ordered:
            credited = best_credited.get(cell, set())
            final_theta: float | None = None
            if credited:
                rows, cols = zip(*credited)
                ux, uy = self.grid_to_world(sum(rows) / len(rows), sum(cols) / len(cols))
                x, y = self.grid_to_world(*cell)
                final_theta = math.atan2(uy - y, ux - x)

            goal_world = self.grid_to_world(*cell)
            polyline = visibility_path.shortest_path(
                polygon, leg_start_world, goal_world, simplify_tolerance=2.0 * self.resolution
            )
            if polyline is not None:
                waypoints = self._waypoints_from_world_polyline(polyline, final_theta, len(credited))
                if waypoints:
                    route.extend(waypoints)
                    leg_start = cell
                    leg_start_world = goal_world
                    continue

            path = path_from_start.get(cell) if leg_start == start else None
            if path is None:
                path = self._astar_path(traversable, leg_start, cell)
            if path is None:
                # 이 leg만 도달 불가 -> 이 candidate는 건너뛰고 다음 candidate로 이어간다.
                continue
            route.extend(self._leg_waypoints(path, nav_blocking, final_theta, len(credited)))
            leg_start = cell
            leg_start_world = goal_world
        return route

    def plan_direct_path(
        self,
        start_pose: dict,
        goal_xy: tuple[float, float],
        final_theta: float | None = None,
        forbidden_mask: np.ndarray | None = None,
    ) -> list[dict] | None:
        """start_pose에서 goal_xy까지 벽(+옵션으로 forbidden_mask)을 피해가는 A*
        경로를 waypoint 시퀀스로 만든다. plan_route()와 달리 "다음에 어디를 탐색할지"
        고르는 게 아니라 이미 정해진 두 점 사이 경로 하나만 필요할 때 쓴다 -
        Instruction-Following(missions/mission3_pipe.py)의 "avoiding the path between
        A and B" 같은 negative constraint에서, base autonomy의 point-to-point 이동만
        으로는 특정 영역을 피해가도록 강제할 수 없기 때문에 우리가 직접 우회 경로를
        계산해서 여러 waypoint로 잘라 보낸다. 실패(경로 없음/맵 미준비)하면 None."""
        diag: dict = {"goal_xy": tuple(float(v) for v in goal_xy)}
        with self._lock:
            grid = self.grid.copy()
            origin_ready = self.origin_x is not None
        if not origin_ready:
            diag["reason"] = "origin_not_ready"
            self.last_direct_path_diagnostics = diag
            return None
        start_cell = self.world_to_grid(start_pose["x"], start_pose["y"])
        goal_cell = self.world_to_grid(goal_xy[0], goal_xy[1])
        diag["start_cell"] = start_cell
        diag["goal_cell"] = goal_cell
        if start_cell is None or goal_cell is None:
            diag["reason"] = "cell_out_of_map"
            self.last_direct_path_diagnostics = diag
            return None

        occupied = (grid == config.OCC_OCCUPIED).astype(np.uint8)
        inflation = max(1, int(round(config.ROBOT_CLEARANCE_M / self.resolution)))
        inflated = cv2.dilate(occupied, np.ones((2 * inflation + 1, 2 * inflation + 1), np.uint8)).astype(bool)
        traversable = (grid == config.OCC_FREE) & (~inflated)
        if forbidden_mask is not None and forbidden_mask.shape == traversable.shape:
            traversable = traversable & (~forbidden_mask)
        diag["traversable_cell_count"] = int(traversable.sum())

        start = self._nearest_traversable(traversable, *start_cell, radius=10)
        goal = self._nearest_traversable(traversable, *goal_cell, radius=10)
        diag["start_snap"] = start
        diag["goal_snap"] = goal
        if start is None or goal is None:
            # 로봇 위치나 목표 지점(예: 방 centroid) 주변 10칸(2m) 안에 갈 수 있는
            # 셀이 하나도 없다는 뜻 - 목표 지점이 가구 한복판이거나, 그 방 자체가
            # 아직 거의 안 뚫려있을 때(=segmentation은 됐지만 내부가 대부분 unknown)
            # 발생한다.
            diag["reason"] = (
                "start_snap_failed" if start is None and goal is not None else
                "goal_snap_failed" if goal is None and start is not None else
                "both_snap_failed"
            )
            self.last_direct_path_diagnostics = diag
            return None

        # Same reasoning as plan_recovery_patrol(): grid A*'s LOS-based hop
        # simplification only ever "sees" collisions at MAP_RESOLUTION_M, so a
        # step's destination leg can graze clearance in a small cluttered
        # room. Try the continuous visibility-graph route first (forbidden_mask
        # is already baked into `traversable` above, so it's respected here
        # too), falling back to grid A* unchanged if unavailable/no path.
        polygon = visibility_path.build_traversable_polygon(traversable, self.grid_to_world)
        polyline = visibility_path.shortest_path(
            polygon,
            (float(start_pose["x"]), float(start_pose["y"])),
            (float(goal_xy[0]), float(goal_xy[1])),
            simplify_tolerance=2.0 * self.resolution,
        )
        if polyline is not None:
            waypoints = self._waypoints_from_world_polyline(polyline, final_theta, credited_len=0)
            if waypoints:
                diag["reason"] = "ok_visibility_graph"
                diag["path_len"] = len(polyline)
                self.last_direct_path_diagnostics = diag
                return waypoints

        path = self._astar_path(traversable, start, goal)
        if path is None:
            # start/goal 둘 다 traversable인 건 확인됐는데, 그 사이를 잇는 경로가
            # 없다는 뜻 - 즉 로봇이 있는 영역과 목표 지점이 지금 알려진 free space
            # 기준으로 서로 다른(연결 안 된) 덩어리에 있음. 진짜 통로가 너무 좁아서
            # clearance 부풀리기 후 끊겼거나, 그 사이 미탐색 구간이 있어서 아직
            # 연결이 안 잡힌 것.
            diag["reason"] = "astar_no_path_start_goal_disconnected"
            self.last_direct_path_diagnostics = diag
            return None
        diag["reason"] = "ok"
        diag["path_len"] = len(path)
        self.last_direct_path_diagnostics = diag
        return self._leg_waypoints(path, inflated, final_theta, credited_len=0)

    def plan_floor_coverage(self, robot_pose: dict) -> list[dict]:
        """frontier가 소진된 뒤 "아직 가까이서 안 본 바닥"을 목표로 이어서 탐사한다.

        plan_route()는 free/unknown 경계(frontier)를 목표로 하므로, LiDAR가 한 번
        휩쓸어 방 전체가 known이 되면 갈 곳이 없다고 판단해 멈춘다. 그런데 물체를
        보는 건 카메라이고, 그 시점에 카메라는 방의 일부만 봤다. 여기서는 목표를
        "observed 마스크가 아직 안 찍힌 free 셀"로 바꿔서, 그런 셀을 한 번에 가장
        많이 새로 볼 수 있는 지점을 골라 이동한다 (다른 팀 구현의 floor-coverage
        greedy set cover와 같은 발상이지만, 전체 지도를 미리 아는 오프라인이 아니라
        지금까지 만든 지도 위에서 한 지점씩 온라인으로 고른다).

        반환: plan_route()와 같은 waypoint dict 리스트. 더 볼 곳이 없거나 목표
        커버리지에 도달했으면 빈 리스트."""
        diag: dict = {}
        coverage = self.floor_coverage()
        diag["coverage_ratio"] = round(coverage["ratio"], 4)
        diag["unobserved_area_m2"] = round(coverage["unobserved_area_m2"], 2)
        if coverage["free_cells"] == 0:
            return self._fail_floor(diag, "nothing_mapped_yet")
        if coverage["ratio"] >= config.EXPLORATION_FLOOR_COVERAGE_TARGET:
            return self._fail_floor(diag, "coverage_target_reached")

        with self._lock:
            grid = self.grid.copy()
            observed = self.observed.copy()
            origin_ready = self.origin_x is not None
        if not origin_ready:
            return self._fail_floor(diag, "origin_not_ready")
        robot_cell = self.world_to_grid(robot_pose["x"], robot_pose["y"])
        if robot_cell is None:
            return self._fail_floor(diag, "robot_cell_out_of_map")

        occupied = (grid == config.OCC_OCCUPIED).astype(np.uint8)
        inflation = max(1, int(round(config.ROBOT_CLEARANCE_M / self.resolution)))
        inflated = cv2.dilate(occupied, np.ones((2 * inflation + 1, 2 * inflation + 1), np.uint8)).astype(bool)
        traversable = (grid == config.OCC_FREE) & (~inflated)
        start = self._nearest_traversable(traversable, *robot_cell, radius=10)
        if start is None:
            return self._fail_floor(diag, "robot_not_near_any_traversable_cell")

        # 관측 여부 판정은 몸체 clearance가 아니라 순수 가시선 문제라, plan_route()의
        # scov와 같은 이유로 얇은 margin만 쓴다.
        los_margin = max(1, int(round(config.FRONTIER_LOS_WALL_MARGIN_M / self.resolution)))
        los_blocking = cv2.dilate(occupied, np.ones((2 * los_margin + 1, 2 * los_margin + 1), np.uint8)).astype(bool)

        targets = [tuple(cell) for cell in np.argwhere((grid == config.OCC_FREE) & (~observed))]
        diag["unobserved_cell_count"] = len(targets)
        if not targets:
            return self._fail_floor(diag, "no_unobserved_free_cells")

        traversable_cells = np.argwhere(traversable)
        if len(traversable_cells) == 0:
            return self._fail_floor(diag, "no_traversable_cells")
        sample_n = min(config.EXPLORATION_CANDIDATE_SAMPLES, len(traversable_cells))
        sample_idx = self._rng.choice(len(traversable_cells), size=sample_n, replace=False)
        pool = [(int(traversable_cells[i][0]), int(traversable_cells[i][1])) for i in sample_idx]
        # 미관측 셀 바로 옆에 설 자리를 항상 후보에 넣어둔다 - 무작위 샘플만 쓰면
        # 남은 미관측 영역이 작을 때 우연히 근처 후보가 안 뽑혀 조기 종료된다
        # (plan_route()의 frontier anchor와 같은 안전장치).
        for cell in targets[:: max(1, len(targets) // 20)]:
            anchor = self._nearest_traversable(traversable, cell[0], cell[1], radius=inflation + 2)
            if anchor is not None:
                pool.append(anchor)
        pool = list(dict.fromkeys(pool))
        diag["pool_cell_count"] = len(pool)

        observe_radius_cells = config.EXPLORATION_OBSERVE_RADIUS_M / self.resolution
        target_array = np.asarray(targets, dtype=np.float64)
        min_gain = config.EXPLORATION_FLOOR_COVERAGE_MIN_GAIN_CELLS
        scored: list[tuple[int, tuple[int, int]]] = []
        for cell in pool:
            # 반경 안의 미관측 셀만 먼저 거리로 추려낸 뒤 LOS를 본다 (전부 LOS를
            # 계산하면 후보 수십 개 x 미관측 수천 개라 너무 느리다).
            within = np.hypot(target_array[:, 0] - cell[0], target_array[:, 1] - cell[1]) <= observe_radius_cells
            nearby = [targets[i] for i in np.nonzero(within)[0]]
            if len(nearby) < min_gain:
                continue  # 반경 안을 다 본다 해도 최소 이득에 못 미친다
            gain = sum(1 for target in nearby if self._line_of_sight(los_blocking, cell, target))
            if gain >= min_gain:
                scored.append((gain, cell))
        diag["scored_candidate_count"] = len(scored)
        diag["best_gain_cells"] = max((gain for gain, _ in scored), default=0)
        if not scored:
            return self._fail_floor(diag, "no_candidate_gains_enough_new_floor")

        # 순수 이득만으로 고르면 방 양쪽 끝을 번갈아 찍으며 왔다갔다한다 -
        # plan_route()의 후보 선택이 EXPLORATION_DISTANCE_PENALTY_HALFLIFE_M로
        # 같은 진동을 막는 것과 동일한 이유로, 여기서도 로봇으로부터의 거리로 점수를
        # 감쇠시켜 가까운 곳부터 훑게 한다 (멈추는 기준인 MIN_GAIN_CELLS 비교는 감쇠
        # 전 순수 이득으로 한다 - "볼 게 남았나"와 "뭘 먼저 갈까"는 다른 문제다).
        halflife_cells = max(1e-6, config.EXPLORATION_DISTANCE_PENALTY_HALFLIFE_M / self.resolution)

        def priority(item: tuple[int, tuple[int, int]]) -> float:
            gain, cell = item
            distance = math.hypot(cell[0] - start[0], cell[1] - start[1])
            return gain / (1.0 + distance / halflife_cells)

        # 우선순위가 높은 순서로 시도한다 - 1등 후보가 도달 불가(벽 뒤 고립 등)라고
        # 해서 탐사를 끝내면 안 된다. 실제로 갈 수 있는 첫 후보로 이어간다.
        polygon = visibility_path.build_traversable_polygon(traversable, self.grid_to_world)
        start_world = (float(robot_pose["x"]), float(robot_pose["y"]))
        for gain, cell in sorted(scored, key=priority, reverse=True):
            goal_world = self.grid_to_world(*cell)
            final_theta = math.atan2(goal_world[1] - robot_pose["y"], goal_world[0] - robot_pose["x"])

            polyline = visibility_path.shortest_path(
                polygon, start_world, goal_world, simplify_tolerance=2.0 * self.resolution
            )
            if polyline is not None:
                waypoints = self._waypoints_from_world_polyline(polyline, final_theta, gain)
                if waypoints:
                    diag["reason"] = "ok_visibility_graph"
                    diag["chosen_gain_cells"] = gain
                    self.last_floor_coverage_diagnostics = diag
                    return waypoints

            path = self._astar_path(traversable, start, cell)
            if path is None:
                continue
            diag["reason"] = "ok"
            diag["chosen_gain_cells"] = gain
            self.last_floor_coverage_diagnostics = diag
            return self._leg_waypoints(path, inflated, final_theta, gain)

        return self._fail_floor(diag, "no_reachable_candidate")

    def _fail_floor(self, diag: dict, reason: str) -> list[dict]:
        diag["reason"] = reason
        self.last_floor_coverage_diagnostics = diag
        return []

    def describe_last_floor_coverage(self) -> str:
        return ", ".join(f"{key}={value}" for key, value in self.last_floor_coverage_diagnostics.items())

    def plan_recovery_patrol(
        self,
        start_pose: dict,
        previous_points: list[tuple[float, float]],
    ) -> list[dict]:
        """Plan to a distinct known-free point after frontier exhaustion.

        Frontier score can be zero while a visual target remains unseen.  This
        fallback deliberately ignores surface-coverage gain and selects the
        reachable traversable cell farthest from the current/recent recovery
        locations, producing a bounded room patrol rather than retrying the
        same exhausted frontier computation forever.
        """
        with self._lock:
            grid = self.grid.copy()
            origin_ready = self.origin_x is not None
        if not origin_ready:
            return []
        start_cell = self.world_to_grid(start_pose["x"], start_pose["y"])
        if start_cell is None:
            return []

        occupied = (grid == config.OCC_OCCUPIED).astype(np.uint8)
        inflation = max(1, int(round(config.ROBOT_CLEARANCE_M / self.resolution)))
        inflated = cv2.dilate(
            occupied,
            np.ones((2 * inflation + 1, 2 * inflation + 1), np.uint8),
        ).astype(bool)
        traversable = (grid == config.OCC_FREE) & (~inflated)
        start = self._nearest_traversable(traversable, *start_cell, radius=10)
        if start is None:
            return []

        spacing = float(config.MISSION3_RECOVERY_PATROL_MIN_SPACING_M)
        anchors = [(float(start_pose["x"]), float(start_pose["y"])), *previous_points]
        stride = max(1, int(round(0.4 / self.resolution)))
        scored: list[tuple[float, tuple[int, int]]] = []
        for row, col in np.argwhere(traversable)[::stride]:
            cell = (int(row), int(col))
            x, y = self.grid_to_world(*cell)
            min_distance = min(math.hypot(x - ax, y - ay) for ax, ay in anchors)
            if min_distance >= spacing:
                scored.append((min_distance, cell))

        # Small cluttered rooms leave only a cell or two of inflated free
        # space between furniture - 4-connected grid A* routes there still
        # graze the clearance boundary at MAP_RESOLUTION_M quantization. Try
        # a continuous visibility-graph route first (exact geometry, no grid
        # quantization) and only fall back to grid A* if that's unavailable
        # (shapely not installed yet) or fails for this particular goal.
        polygon = visibility_path.build_traversable_polygon(traversable, self.grid_to_world)
        start_world = (float(start_pose["x"]), float(start_pose["y"]))

        for _, goal in sorted(scored, reverse=True):
            x, y = self.grid_to_world(*goal)
            theta = math.atan2(
                float(start_pose["y"]) - y,
                float(start_pose["x"]) - x,
            )
            polyline = visibility_path.shortest_path(
                polygon, start_world, (x, y), simplify_tolerance=2.0 * self.resolution
            )
            if polyline is not None:
                waypoints = self._waypoints_from_world_polyline(polyline, theta, credited_len=0)
                if waypoints:
                    return waypoints

            path = self._astar_path(traversable, start, goal)
            if path is None:
                continue
            return self._leg_waypoints(path, inflated, theta, credited_len=0)
        return []
