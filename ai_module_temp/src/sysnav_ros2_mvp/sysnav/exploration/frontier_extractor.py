"""Extract free-unknown frontier clusters from an occupancy grid."""

from __future__ import annotations

from collections import deque

import numpy as np

from sysnav import config

_NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class FrontierExtractor:
    def __init__(self, min_cluster_cells: int = config.FRONTIER_MIN_CLUSTER_CELLS) -> None:
        self.min_cluster_cells = min_cluster_cells

    @staticmethod
    def _mask(grid: np.ndarray) -> np.ndarray:
        free = grid == config.OCC_FREE
        unknown = grid == config.OCC_UNKNOWN
        padded = np.pad(unknown, 1, constant_values=False)
        adjacent = np.zeros_like(unknown, dtype=bool)
        for dr, dc in _NEIGHBORS_8:
            adjacent |= padded[1 + dr:1 + dr + grid.shape[0], 1 + dc:1 + dc + grid.shape[1]]
        return free & adjacent

    def extract(self, grid: np.ndarray) -> list[dict]:
        mask = self._mask(grid)
        visited = np.zeros_like(mask, dtype=bool)
        rows, cols = grid.shape
        clusters = []
        for sr, sc in np.argwhere(mask):
            sr, sc = int(sr), int(sc)
            if visited[sr, sc]:
                continue
            queue = deque([(sr, sc)])
            visited[sr, sc] = True
            cells = []
            unknown_neighbors = []
            while queue:
                row, col = queue.popleft()
                cells.append((row, col))
                for dr, dc in _NEIGHBORS_8:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if grid[nr, nc] == config.OCC_UNKNOWN:
                            unknown_neighbors.append((nr, nc))
                        if mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
            if len(cells) < self.min_cluster_cells:
                continue
            array = np.asarray(cells, dtype=np.float64)
            centroid = np.mean(array, axis=0)
            index = int(np.argmin(np.linalg.norm(array - centroid, axis=1)))
            row, col = cells[index]
            unknown_centroid = np.mean(np.asarray(unknown_neighbors), axis=0) if unknown_neighbors else np.asarray([row, col])
            clusters.append({
                "cells": cells,
                "row": row,
                "col": col,
                "cluster_size": len(cells),
                "unknown_centroid": tuple(float(v) for v in unknown_centroid),
            })
        return clusters
