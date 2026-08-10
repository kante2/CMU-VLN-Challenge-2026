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
        # OCC_OCCUPIED은 벽/가구를 구분 안 하고 z-band 안의 아무 point나 다 잡는다. room
        # segmentation은 "벽"만 경계로 써야 하는데(가구가 방을 쪼개면 안 됨), 이 셀에서
        # 지금까지 관측된 point 중 가장 높은 z를 기록해뒀다가, 충분히 높이 닿은 셀만
        # "진짜 벽"으로 보는 데 쓴다 (RoomSegmenter._structural_wall_mask 참고).
        self.max_height = np.full((self.size_cells, self.size_cells), -np.inf, dtype=np.float32)
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

    def describe_last_plan_failure(self) -> str:
        return ", ".join(f"{key}={value}" for key, value in self.last_plan_diagnostics.items())

    def reset(self, robot_pose: dict | None = None) -> None:
        with self._lock:
            self.grid.fill(config.OCC_UNKNOWN)
            self.max_height.fill(-np.inf)
            self._anchor_visit_counts.clear()
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

    def surface_point_mask(self, grid: np.ndarray) -> np.ndarray:
        """논문의 surface point set S(free/non-free 경계) - plan_route()가 candidate
        점수(wcov)를 매길 때 쓰는 것과 동일한 마스크. 디버그 시각화 전용으로 외부에
        노출한다 (exploration_visualizer.py)."""
        return self.frontier_extractor._mask(grid)

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
                for row, col in ray[:-1]:
                    if self.grid[row, col] == config.OCC_OCCUPIED:
                        blocked = True
                        break
                    self.grid[row, col] = config.OCC_FREE
                if blocked:
                    continue  # 벽 너머로 찍힌 것으로 추정되는 endpoint도 신뢰 안 함(occupied로도 안 찍음)
                endpoints.append((endpoint[0], endpoint[1], float(z)))
            for row, col, z in endpoints:
                self.grid[row, col] = config.OCC_OCCUPIED
                if z > self.max_height[row, col]:
                    self.max_height[row, col] = z
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
    def _line_of_sight(occupied: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> bool:
        """start/end 자체는 항상 FREE라는 전제(둘 다 traversable/surface 셀)이므로 중간 셀만 검사한다."""
        for row, col in _bresenham(start[0], start[1], end[0], end[1])[1:-1]:
            if occupied[row, col]:
                return False
        return True

    def _stochastic_select(
        self,
        start: tuple[int, int],
        candidates: list[tuple[int, int]],
        scov: dict[tuple[int, int], set[tuple[int, int]]],
        surface_points: set[tuple[int, int]],
    ) -> list[tuple[tuple[int, int], set[tuple[int, int]]]]:
        """wcov(c) = |Scov(c) ∩ Ŝ|에 비례한 확률로 후보를 하나씩 뽑고, 뽑을 때마다
        그 후보가 덮는 surface point를 Ŝ에서 제거해 재계산 -> 남은 점수가 전부
        δ 밑으로 떨어지면 멈춘다. 반환값은 (뽑힌 후보, 그 시점에 실제로 credit된
        surface point 집합) 리스트.

        EXPLORATION_MAX_CANDIDATES_PER_CYCLE로 한 번에 몇 개까지 뽑을지 제한한다 -
        frontier anchor가 방 양쪽 끝처럼 서로 먼 곳에 여러 개 생길 수 있는데, 그걸
        전부 한 cycle에 다 뽑아서 TSP로 묶어버리면 "이쪽 찍고 저쪽 찍고" 왔다갔다하는
        경로가 나온다 - 다음 cycle에서 최신 지도로 다시 고르는 게 낫다.

        뽑을 확률(weights)은 순수 wcov가 아니라 start(로봇 현재 위치)로부터의 거리로
        감쇠시킨 값을 쓴다 - 안 그러면 방 양쪽 끝의 wcov 점수가 비슷할 때 매 cycle
        50:50에 가깝게 왼쪽/오른쪽이 갈려서, cycle마다 반대쪽을 골라 "왔다갔다"하는
        것처럼 보인다. 가까운 후보를 확실히 우선해서 이 진동을 줄인다(멈추는 기준인
        delta 비교는 여전히 순수 wcov로 한다 - "커버할 게 남았는지"와 "그중 뭘 먼저
        갈지"는 다른 문제라서)."""
        pool = list(candidates)
        uncovered = set(surface_points)
        selected: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []
        max_candidates = max(1, int(config.EXPLORATION_MAX_CANDIDATES_PER_CYCLE))
        halflife_cells = max(1e-6, config.EXPLORATION_DISTANCE_PENALTY_HALFLIFE_M / self.resolution)
        while pool and len(selected) < max_candidates:
            scores = np.array([len(scov[c] & uncovered) for c in pool], dtype=np.float64)
            best = float(scores.max())
            if best < config.EXPLORATION_MIN_SCORE_DELTA:
                break
            distances = np.array(
                [math.hypot(c[0] - start[0], c[1] - start[1]) for c in pool],
                dtype=np.float64,
            )
            weighted_scores = scores / (1.0 + distances / halflife_cells)
            weights = weighted_scores / weighted_scores.sum()
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

        # 논문의 surface point set S: free / non-free(occupied+unknown) 경계
        frontier_mask = self.frontier_extractor._mask(grid)
        surface_points = {tuple(cell) for cell in np.argwhere(frontier_mask)}
        diag["surface_point_count"] = len(surface_points)
        if not surface_points:
            return self._fail(diag, "no_surface_points_anywhere_in_known_map")

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

        pool_cells = list(dict.fromkeys(pool_cells + list(anchor_cells) + list(stale_anchor_cells)))  # 순서 유지하며 중복 제거
        diag["anchor_count"] = len(anchor_cells)
        diag["pool_cell_count"] = len(pool_cells)

        candidates = []
        path_from_start: dict[tuple[int, int], list[tuple[int, int]]] = {}
        rejected_by_visited = 0
        rejected_by_astar = 0
        anchors_rejected_by_astar = 0
        for cell in pool_cells:
            # 무작위 샘플은 "최근에 간 근처는 건너뛰기"(is_near_visited)를 적용해서 제자리
            # 맴돌기를 막지만, frontier anchor는 예외다 - anchor는 "아직 안 풀린 frontier가
            # 실제로 존재"할 때만 생기므로, 예전에 그 근처에 viewpoint가 하나 찍혔다고 걸러
            # 버리면(그 viewpoint가 그 frontier를 실제로 해소 못 했는데도) 이 frontier를
            # 영원히 다시 못 뽑는다 - 문 근처를 한 번 스쳐 지나간 적만 있어도 그 문 너머로
            # 다시는 못 가는 게 바로 이 문제였다.
            if cell not in anchor_cells:
                x, y = self.grid_to_world(*cell)
                if viewpoint_memory.is_near_visited(x, y):
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
        cover_radius_cells = config.FRONTIER_COVERAGE_RADIUS_M / self.resolution
        scov: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for c in candidates:
            covered = {
                p for p in surface_points
                if math.hypot(p[0] - c[0], p[1] - c[1]) <= cover_radius_cells
                and self._line_of_sight(los_blocking, c, p)
            }
            scov[c] = covered
        diag["candidates_with_nonempty_scov"] = sum(1 for c in candidates if scov[c])

        best_cost: float | None = None
        best_ordered: list[tuple[int, int]] = []
        best_credited: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for _ in range(config.EXPLORATION_STOCHASTIC_TRIALS):
            selected = self._stochastic_select(start, candidates, scov, surface_points)
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
            return self._fail(diag, "no_candidate_had_any_visible_uncovered_surface_point")

        diag["reason"] = "ok"
        self.last_plan_diagnostics = diag

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
            route.extend(self._leg_waypoints(path, nav_blocking, final_theta, len(credited)))
            leg_start = cell
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
        with self._lock:
            grid = self.grid.copy()
            origin_ready = self.origin_x is not None
        if not origin_ready:
            return None
        start_cell = self.world_to_grid(start_pose["x"], start_pose["y"])
        goal_cell = self.world_to_grid(goal_xy[0], goal_xy[1])
        if start_cell is None or goal_cell is None:
            return None

        occupied = (grid == config.OCC_OCCUPIED).astype(np.uint8)
        inflation = max(1, int(round(config.ROBOT_CLEARANCE_M / self.resolution)))
        inflated = cv2.dilate(occupied, np.ones((2 * inflation + 1, 2 * inflation + 1), np.uint8)).astype(bool)
        traversable = (grid == config.OCC_FREE) & (~inflated)
        if forbidden_mask is not None and forbidden_mask.shape == traversable.shape:
            traversable = traversable & (~forbidden_mask)

        start = self._nearest_traversable(traversable, *start_cell, radius=10)
        goal = self._nearest_traversable(traversable, *goal_cell, radius=10)
        if start is None or goal is None:
            return None
        path = self._astar_path(traversable, start, goal)
        if path is None:
            return None
        return self._leg_waypoints(path, inflated, final_theta, credited_len=0)
