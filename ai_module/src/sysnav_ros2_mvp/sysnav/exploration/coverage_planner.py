"""Single-room online occupancy mapping and frontier coverage planning."""

from __future__ import annotations

import heapq
import math
import threading

import cv2
import numpy as np

from sysnav import config
from sysnav.exploration.frontier_extractor import FrontierExtractor
from sysnav.exploration.viewpoint_memory import ViewpointMemory


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
    def _astar_length(traversable: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> float | None:
        if start == goal:
            return 0.0
        rows, cols = traversable.shape
        queue = [(0.0, 0.0, start)]
        best = {start: 0.0}
        while queue:
            _, cost, current = heapq.heappop(queue)
            if current == goal:
                return cost
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
                heuristic = abs(goal[0] - nr) + abs(goal[1] - nc)
                heapq.heappush(queue, (next_cost + heuristic, next_cost, neighbor))
        return None

    def _unknown_coverage(self, grid: np.ndarray, row: int, col: int) -> int:
        radius = int(round(config.FRONTIER_COVERAGE_RADIUS_M / self.resolution))
        r0, r1 = max(0, row - radius), min(grid.shape[0], row + radius + 1)
        c0, c1 = max(0, col - radius), min(grid.shape[1], col + radius + 1)
        patch = grid[r0:r1, c0:c1]
        yy, xx = np.ogrid[r0:r1, c0:c1]
        circle = (yy - row) ** 2 + (xx - col) ** 2 <= radius ** 2
        return int(np.count_nonzero((patch == config.OCC_UNKNOWN) & circle))

    @staticmethod
    def _order(start: tuple[float, float], candidates: list[dict]) -> list[dict]:
        remaining = list(candidates)
        ordered = []
        current = start
        while remaining:
            index = min(range(len(remaining)), key=lambda i: math.hypot(remaining[i]["x"] - current[0], remaining[i]["y"] - current[1]))
            selected = remaining.pop(index)
            ordered.append(selected)
            current = (selected["x"], selected["y"])
        return ordered

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

        scored = []
        for frontier in self.frontier_extractor.extract(grid):
            cell = self._nearest_traversable(traversable, frontier["row"], frontier["col"])
            if cell is None:
                continue
            path_cells = self._astar_length(traversable, start, cell)
            if path_cells is None:
                continue
            x, y = self.grid_to_world(*cell)
            if viewpoint_memory.is_near_visited(x, y):
                continue
            coverage = self._unknown_coverage(grid, *cell)
            distance_m = path_cells * self.resolution
            score = coverage + config.FRONTIER_CLUSTER_WEIGHT * frontier["cluster_size"] - config.FRONTIER_DISTANCE_WEIGHT * distance_m
            ur, uc = frontier["unknown_centroid"]
            ux, uy = self.grid_to_world(int(round(ur)), int(round(uc)))
            scored.append({
                "x": x,
                "y": y,
                "theta": math.atan2(uy - y, ux - x),
                "score": float(score),
                "coverage_score": int(coverage),
                "path_distance_m": float(distance_m),
            })
        if not scored:
            return []
        top = sorted(scored, key=lambda item: item["score"], reverse=True)[:config.FRONTIER_TOP_K]
        return self._order((float(robot_pose["x"]), float(robot_pose["y"])), top)
