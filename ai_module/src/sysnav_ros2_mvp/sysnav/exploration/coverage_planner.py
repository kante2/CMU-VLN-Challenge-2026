"""Single-room online occupancy mapping and frontier coverage planning.

In-room exploration candidate selection follows the SysNav paper (Sec. IV-B-1,
"In-room Exploration Policy"): sample a pose horizon H, define a surface point
set S (the free/non-free boundary), score each candidate by the still-uncovered
surface it can see (wcov = |Scov(c) ∩ Ŝ|), pick candidates via stochastic
sampling weighted by wcov (removing what they cover from Ŝ each pick, repeating
until every remaining score falls below δ), repeat that K times, and keep the
candidate set whose TSP tour is cheapest.
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

_TSP_EXACT_MAX_N = 7


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
        self.origin_x: float | None = None
        self.origin_y: float | None = None
        self.frontier_extractor = FrontierExtractor()
        self.sensor_to_base = np.asarray(config.T_SENSOR_TO_BASE, dtype=np.float64)
        self._lock = threading.RLock()
        self._rng = np.random.default_rng()

    def reset(self, robot_pose: dict | None = None) -> None:
        with self._lock:
            self.grid.fill(config.OCC_UNKNOWN)
            if robot_pose is None:
                self.origin_x = None
                self.origin_y = None
            else:
                half = config.MAP_SIZE_M / 2.0
                self.origin_x = float(robot_pose["x"]) - half
                self.origin_y = float(robot_pose["y"]) - half

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

            yaw = float(pose["yaw"])
            x_map = math.cos(yaw) * points_base[:, 0] - math.sin(yaw) * points_base[:, 1] + float(pose["x"])
            y_map = math.sin(yaw) * points_base[:, 0] + math.cos(yaw) * points_base[:, 1] + float(pose["y"])
            endpoints = []
            for x, y in zip(x_map, y_map):
                endpoint = self.world_to_grid(float(x), float(y))
                if endpoint is None:
                    continue
                ray = _bresenham(robot_cell[0], robot_cell[1], endpoint[0], endpoint[1])
                for row, col in ray[:-1]:
                    if self.grid[row, col] != config.OCC_OCCUPIED:
                        self.grid[row, col] = config.OCC_FREE
                endpoints.append(endpoint)
            for row, col in endpoints:
                self.grid[row, col] = config.OCC_OCCUPIED
            rr, cc = robot_cell
            radius = max(1, int(round(0.35 / self.resolution)))
            self.grid[max(0, rr - radius):min(self.size_cells, rr + radius + 1), max(0, cc - radius):min(self.size_cells, cc + radius + 1)] = config.OCC_FREE

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
    def _line_of_sight(occupied: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> bool:
        """start/end 자체는 항상 FREE라는 전제(둘 다 traversable/surface 셀)이므로 중간 셀만 검사한다."""
        for row, col in _bresenham(start[0], start[1], end[0], end[1])[1:-1]:
            if occupied[row, col]:
                return False
        return True

    def _stochastic_select(
        self,
        candidates: list[tuple[int, int]],
        scov: dict[tuple[int, int], set[tuple[int, int]]],
        surface_points: set[tuple[int, int]],
    ) -> list[tuple[tuple[int, int], set[tuple[int, int]]]]:
        """wcov(c) = |Scov(c) ∩ Ŝ|에 비례한 확률로 후보를 하나씩 뽑고, 뽑을 때마다
        그 후보가 덮는 surface point를 Ŝ에서 제거해 재계산 -> 남은 점수가 전부
        δ 밑으로 떨어지면 멈춘다. 반환값은 (뽑힌 후보, 그 시점에 실제로 credit된
        surface point 집합) 리스트."""
        pool = list(candidates)
        uncovered = set(surface_points)
        selected: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []
        while pool:
            scores = np.array([len(scov[c] & uncovered) for c in pool], dtype=np.float64)
            best = float(scores.max())
            if best < config.EXPLORATION_MIN_SCORE_DELTA:
                break
            weights = scores / scores.sum()
            pick_index = int(self._rng.choice(len(pool), p=weights))
            picked = pool.pop(pick_index)
            credited = scov[picked] & uncovered
            uncovered -= credited
            selected.append((picked, credited))
        return selected

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

    def plan_route(self, robot_pose: dict, viewpoint_memory: ViewpointMemory) -> list[dict]:
        with self._lock:
            grid = self.grid.copy()
            origin_ready = self.origin_x is not None
        if not origin_ready:
            return []
        robot_cell = self.world_to_grid(robot_pose["x"], robot_pose["y"])
        if robot_cell is None:
            return []
        occupied = (grid == config.OCC_OCCUPIED).astype(np.uint8)
        inflation = max(1, int(round(config.ROBOT_CLEARANCE_M / self.resolution)))
        inflated = cv2.dilate(occupied, np.ones((2 * inflation + 1, 2 * inflation + 1), np.uint8)).astype(bool)
        traversable = (grid == config.OCC_FREE) & (~inflated)
        start = self._nearest_traversable(traversable, *robot_cell, radius=10)
        if start is None:
            return []
        # LOS는 원본(1셀 두께) occupied가 아니라 clearance만큼 팽창된 inflated를 써야 한다.
        # 안 그러면 얇은 벽의 대각선 코너를 Bresenham 선이 스치듯 통과해 벽 건너편 surface
        # point가 "보인다"고 잘못 판단하고, 그쪽으로 waypoint heading이 벽을 뚫고 잡힌다.
        blocking = inflated

        # 논문의 surface point set S: free / non-free(occupied+unknown) 경계
        surface_points = {tuple(cell) for cell in np.argwhere(self.frontier_extractor._mask(grid))}
        if not surface_points:
            return []

        # 논문의 local planning horizon H: traversable space에서 candidate pose 샘플링
        traversable_cells = np.argwhere(traversable)
        if len(traversable_cells) == 0:
            return []
        sample_n = min(config.EXPLORATION_CANDIDATE_SAMPLES, len(traversable_cells))
        sample_idx = self._rng.choice(len(traversable_cells), size=sample_n, replace=False)
        candidates = []
        path_from_start: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for idx in sample_idx:
            cell = (int(traversable_cells[idx][0]), int(traversable_cells[idx][1]))
            x, y = self.grid_to_world(*cell)
            if viewpoint_memory.is_near_visited(x, y):
                continue
            path = self._astar_path(traversable, start, cell)
            if path is None:
                continue
            path_from_start[cell] = path
            candidates.append(cell)
        if not candidates:
            return []

        # Scov(c) = c에서 d_cover 이내 + line-of-sight로 보이는 surface point들
        cover_radius_cells = config.FRONTIER_COVERAGE_RADIUS_M / self.resolution
        scov: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for c in candidates:
            covered = {
                p for p in surface_points
                if math.hypot(p[0] - c[0], p[1] - c[1]) <= cover_radius_cells
                and self._line_of_sight(blocking, c, p)
            }
            scov[c] = covered

        best_cost: float | None = None
        best_ordered: list[tuple[int, int]] = []
        best_credited: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for _ in range(config.EXPLORATION_STOCHASTIC_TRIALS):
            selected = self._stochastic_select(candidates, scov, surface_points)
            if not selected:
                continue
            picked_cells = [cell for cell, _ in selected]
            credited_by_cell = dict(selected)
            ordered = self._solve_tsp(start, picked_cells)
            cost = self._tour_cost(start, ordered)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_ordered = ordered
                best_credited = credited_by_cell

        if not best_ordered:
            # 모든 trial에서 EXPLORATION_MIN_SCORE_DELTA를 넘는 candidate가 하나도 없었다는
            # 뜻이다. 그렇다고 바로 포기하면(exploration 완전 중단) 문처럼 멀어서 wcov 점수가
            # 낮게 나오는 것 말고는 갈 곳이 없는 상황(예: 방을 거의 다 본 뒤 문틈만 남은 경우)
            # 에서 실제로는 갈 곳이 있는데도 탐색이 조기 종료된다. score가 조금이라도 있는
            # candidate 중 최고점 하나만이라도 골라서 이어간다.
            fallback = max(candidates, key=lambda c: len(scov[c]), default=None)
            if fallback is not None and scov[fallback]:
                best_ordered = [fallback]
                best_credited = {fallback: set(scov[fallback])}

        if not best_ordered:
            return []

        # 각 candidate를 최종 목적지 하나로 바로 던지지 않고, A* 경로(벽을 피해가는 경로)를
        # 따라 짧은 간격의 중간 waypoint로 잘라서 순서대로 내보낸다 (벽 너머로 직선 goal을
        # 찍어서 로봇이 벽에 막히는 문제 방지).
        route: list[dict] = []
        leg_start = start
        for cell in best_ordered:
            path = path_from_start.get(cell) if leg_start == start else None
            if path is None:
                path = self._astar_path(traversable, leg_start, cell)
            if path is None:
                # 이 leg만 도달 불가 -> 이 candidate는 건너뛰고 다음 candidate로 이어간다.
                continue
            credited = best_credited.get(cell, set())
            final_theta: float | None = None
            if credited:
                rows, cols = zip(*credited)
                ux, uy = self.grid_to_world(sum(rows) / len(rows), sum(cols) / len(cols))
                x, y = self.grid_to_world(*cell)
                final_theta = math.atan2(uy - y, ux - x)
            route.extend(self._leg_waypoints(path, blocking, final_theta, len(credited)))
            leg_start = cell
        return route
