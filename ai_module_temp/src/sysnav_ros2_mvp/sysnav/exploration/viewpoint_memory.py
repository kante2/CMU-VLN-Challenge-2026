"""Visited exploration-goal memory.

This helper prevents the frontier planner from selecting nearly identical goals. It is
not the paper-style Scene Graph Viewpoint store; representative Viewpoint nodes and
coverage regions are managed by ``scene_graph/scene_graph_manager.py``.
"""

from __future__ import annotations

import math
import threading
import time

from sysnav import config


class ViewpointMemory:
    def __init__(self) -> None:
        self._items: list[dict] = []
        self._lock = threading.RLock()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def add(self, x: float, y: float, yaw: float, coverage_score: float | None = None) -> None:
        with self._lock:
            self._items.append({
                "x": float(x), "y": float(y), "yaw": float(yaw),
                "coverage_score": coverage_score, "timestamp": time.time(),
            })

    def is_near_visited(self, x: float, y: float, threshold: float = config.VIEWPOINT_MIN_DISTANCE_M) -> bool:
        with self._lock:
            return any(math.hypot(item["x"] - x, item["y"] - y) < threshold for item in self._items)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [dict(item) for item in self._items]
