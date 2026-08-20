"""Room Node segmentation - SysNav paper Sec. IV-A-1 ("Scene Representation Building").

논문 원문: "we identify wall structures in the environment by fitting planar
surfaces and analyzing the distribution of point clouds along the vertical (z)
axis. By dilating the detected wall regions, we partition the initially
connected space into multiple disconnected components, each of which is
treated as an individual room node in our representation."

우리는 3D point cloud에서 벽 평면을 따로 피팅하지 않고, CoveragePlanner가 이미
쌓아둔 2D top-down occupancy grid(OCC_OCCUPIED = 벽/가구를 구분 안 한 2D 투영)를
쓴다. 다만 OCC_OCCUPIED를 그대로 "벽"으로 쓰면 소파/테이블 같은 가구도 room을
쪼개버리는 문제가 있어서(논문은 3D point cloud의 z축 분포를 분석해서 진짜 벽
구조만 뽑는다), CoveragePlanner가 셀별로 기록해둔 max_height를 같이 받아서
"충분히 높이까지 닿은" 셀만 진짜 벽으로 취급한다 (_structural_wall_mask 참고).

문(door)은 free space를 계속 이어주는 좁은 통로라서, free space의 connected
component를 그냥 보면 문틈으로 온 집이 하나로 이어져버린다. 그래서
distance-transform 기반 watershed를 쓴다: 벽에서 충분히 먼(=방 안쪽처럼 넓은)
영역을 "room core" seed로 잡고 거기서부터 자라나가게 하면, 문처럼 좁은 지점은
core가 될 만큼 벽에서 멀지 않아서 자연스럽게 두 방 사이의 경계(ridge)가 된다.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from sysnav import config


class RoomSegmenter:
    def __init__(self) -> None:
        self.resolution = float(config.MAP_RESOLUTION_M)

    def _structural_wall_mask(self, grid: np.ndarray, max_height: np.ndarray | None) -> np.ndarray:
        """OCC_OCCUPIED 중 "진짜 벽"만 남긴다. max_height가 없으면(예: 옛날 코드 호출
        경로 호환용) 예전처럼 OCC_OCCUPIED 전부를 벽으로 취급한다."""
        occupied = grid == config.OCC_OCCUPIED
        if max_height is not None:
            occupied = occupied & (max_height >= config.ROOM_WALL_MIN_HEIGHT_M)

        padding_m = config.ROOM_SEGMENTATION_WALL_PADDING_M
        if padding_m > 0:
            pad_cells = max(1, int(round(padding_m / self.resolution)))
            kernel = np.ones((2 * pad_cells + 1, 2 * pad_cells + 1), np.uint8)
            occupied = cv2.dilate(occupied.astype(np.uint8), kernel).astype(bool)
        return occupied

    def _room_anchor(self, mask: np.ndarray, distance: np.ndarray) -> tuple[float, float]:
        """Return a navigable point deep inside ``mask`` rather than its mean.

        A geometric centroid can land on furniture or even outside a concave room.  The
        maximum distance-transform cell is guaranteed to belong to the room and gives
        cross-room navigation a stable, obstacle-clear entry target.
        """
        score = np.where(mask, distance, -1.0)
        row, col = np.unravel_index(int(np.argmax(score)), score.shape)
        return float(row), float(col)

    def _detect_doorways(
        self,
        free: np.ndarray,
        watershed_boundaries: np.ndarray,
        room_labels: np.ndarray,
    ) -> tuple[list[dict], dict[int, list[int]]]:
        """Convert short watershed ridges between exactly two rooms into doors."""
        radius = max(1, int(round(config.ROOM_DOOR_NEIGHBOR_RADIUS_M / self.resolution)))
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        ridge = watershed_boundaries & (free > 0)

        doorways: list[dict] = []
        room_ids = sorted(int(room_id) for room_id in np.unique(room_labels) if int(room_id) > 0)
        adjacency: dict[int, set[int]] = {room_id: set() for room_id in room_ids}
        max_width_cells = max(1.0, config.ROOM_DOOR_MAX_WIDTH_M / self.resolution)
        expanded = {
            room_id: cv2.dilate((room_labels == room_id).astype(np.uint8), kernel).astype(bool)
            for room_id in room_ids
        }
        for index, room_a in enumerate(room_ids):
            for room_b in room_ids[index + 1:]:
                # Watershed marks the entire room/background outline as one connected
                # ridge.  Intersecting it with both expanded room masks isolates only
                # the interface shared by this room pair (normally the doorway).
                pair_ridge = (ridge & expanded[room_a] & expanded[room_b]).astype(np.uint8)
                component_count, components = cv2.connectedComponents(
                    pair_ridge, connectivity=8
                )
                for component_id in range(1, component_count):
                    component = components == component_id
                    rows, cols = np.nonzero(component)
                    if len(rows) < int(config.ROOM_DOOR_MIN_CELLS):
                        continue
                    # A long watershed line through an open room is a segmentation
                    # seam, not a doorway. Real door interfaces stay short.
                    span = math.hypot(
                        float(rows.max() - rows.min()), float(cols.max() - cols.min())
                    )
                    if span > max_width_cells:
                        continue
                    neighborhood = cv2.dilate(
                        component.astype(np.uint8), kernel
                    ).astype(bool)
                    neighbors = {
                        int(value) for value in np.unique(room_labels[neighborhood])
                        if int(value) > 0
                    }
                    if neighbors != {room_a, room_b}:
                        continue
                    adjacency[room_a].add(room_b)
                    adjacency[room_b].add(room_a)
                    doorways.append({
                        "door_id": len(doorways) + 1,
                        "room_a": room_a,
                        "room_b": room_b,
                        "centroid_row": float(rows.mean()),
                        "centroid_col": float(cols.mean()),
                        "cell_count": int(len(rows)),
                        "width_m": float(max(self.resolution, (span + 1.0) * self.resolution)),
                    })

        return doorways, {room_id: sorted(values) for room_id, values in adjacency.items()}

    def segment(self, grid: np.ndarray, max_height: np.ndarray | None = None) -> dict:
        """grid(occupancy) -> {"labels": room_id별 int32 배열(0=미배정), "rooms": [...]}.

        max_height: CoveragePlanner.snapshot_max_height() - 셀별로 관측된 point의
        최대 높이. 주면 가구를 벽에서 제외하고, 안 주면 OCC_OCCUPIED 전부를 벽으로 본다."""
        wall = self._structural_wall_mask(grid, max_height)
        # room 판정에서 "free"는 진짜 traversable free만이 아니라, 벽이 아닌 known
        # 영역 전부(가구가 있는 칸도 포함) - 가구 칸이 room 밖으로 빠지면 그 가구가
        # 있는 자리만 room이 안 뚫려서 이상하게 잘려 보인다.
        known = grid != config.OCC_UNKNOWN
        free = (known & ~wall).astype(np.uint8)
        empty_result = {
            "labels": np.zeros(grid.shape, dtype=np.int32),
            "rooms": [],
            "doorways": [],
            "adjacency": {},
        }
        if int(free.sum()) == 0:
            return empty_result

        # 벽에서 몇 셀 떨어져 있는지 - 문처럼 좁은 통로는 값이 낮고, 방 내부처럼
        # 넓은 공간은 값이 높다.
        distance = cv2.distanceTransform(free, cv2.DIST_L2, 5)

        core_radius_cells = config.ROOM_SEGMENTATION_MIN_CLEARANCE_M / self.resolution
        _, core_mask = cv2.threshold(distance, core_radius_cells, 255, cv2.THRESH_BINARY)
        core_mask = core_mask.astype(np.uint8)

        n_cores, core_labels = cv2.connectedComponentsWithStats(core_mask)[:2]
        if n_cores <= 1:
            # 아직 어디도 벽에서 충분히 멀지 않다 (탐색 초반이라 맵이 좁게만 열려있음) ->
            # room core seed가 하나도 없어서 watershed를 돌릴 수 없다.
            return empty_result

        # watershed marker 규약: 0=미정(watershed가 채움), 1=배경(non-free), 2..n=room core seed
        markers = np.zeros(grid.shape, dtype=np.int32)
        markers[free == 0] = 1
        for label in range(1, n_cores):
            markers[core_labels == label] = label + 1

        elevation = cv2.normalize(-distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elevation_bgr = cv2.cvtColor(elevation, cv2.COLOR_GRAY2BGR)
        cv2.watershed(elevation_bgr, markers)

        room_labels = np.zeros(grid.shape, dtype=np.int32)
        rooms = []
        min_cells = int(config.ROOM_SEGMENTATION_MIN_ROOM_CELLS)
        next_id = 1
        for label in range(2, n_cores + 1):
            mask = markers == label
            cell_count = int(mask.sum())
            if cell_count < min_cells:
                continue
            rows, cols = np.nonzero(mask)
            room_labels[mask] = next_id
            anchor_row, anchor_col = self._room_anchor(mask, distance)
            rooms.append({
                "room_id": next_id,
                "cell_count": cell_count,
                "area_m2": float(cell_count * self.resolution * self.resolution),
                "centroid_row": float(rows.mean()),
                "centroid_col": float(cols.mean()),
                "anchor_row": anchor_row,
                "anchor_col": anchor_col,
            })
            next_id += 1

        doorways, adjacency = self._detect_doorways(
            free,
            markers == -1,
            room_labels,
        )
        for room in rooms:
            room["neighbors"] = adjacency.get(int(room["room_id"]), [])

        return {
            "labels": room_labels,
            "rooms": rooms,
            "doorways": doorways,
            "adjacency": adjacency,
        }
