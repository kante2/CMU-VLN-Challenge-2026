"""Persistent room graph for hierarchical exploration.

The segmenter rebuilds watershed labels on every map update. This registry
turns those ephemeral labels into stable room nodes, preserves visit/coverage
state, and converts detected doorways into a graph used for cross-room travel.
"""

from __future__ import annotations

from collections import deque
import math
import threading
import time

import numpy as np

from sysnav import config


class RoomRegistry:
    def __init__(self) -> None:
        self._rooms: dict[int, dict] = {}
        self._next_id = 1
        self._labels: np.ndarray | None = None
        self._doorways: list[dict] = []
        self._adjacency: dict[int, set[int]] = {}
        self._active_ids: set[int] = set()
        self._revision = 0
        self._lock = threading.RLock()

    def clear(self) -> None:
        with self._lock:
            self._rooms.clear()
            self._next_id = 1
            self._labels = None
            self._doorways.clear()
            self._adjacency.clear()
            self._active_ids.clear()
            self._revision = 0

    @staticmethod
    def _new_room(room_id: int) -> dict:
        return {
            "room_id": room_id,
            "cell_count": 0,
            "area_m2": 0.0,
            "centroid_row": 0.0,
            "centroid_col": 0.0,
            "anchor_row": 0.0,
            "anchor_col": 0.0,
            "mask": None,
            "viewpoint_ids": set(),
            "representative_viewpoint_id": None,
            "representative_coverage": -1,
            "representative_image_path": None,
            "category": None,
            "classified_viewpoint_id": None,
            "last_classification_attempt_time": None,
            "object_labels": set(),
            "visited": False,
            "visit_count": 0,
            "covered": False,
            "empty_plan_streak": 0,
            "last_seen_revision": 0,
        }

    @staticmethod
    def _iou(first: np.ndarray | None, second: np.ndarray) -> float:
        if first is None or first.shape != second.shape:
            return 0.0
        union = int(np.logical_or(first, second).sum())
        if union == 0:
            return 0.0
        return float(np.logical_and(first, second).sum()) / union

    def _match_room(self, cycle_room: dict, cycle_mask: np.ndarray, available: set[int]) -> int:
        best_id = None
        best_iou = float(config.ROOM_REGISTRY_MIN_IOU)
        for room_id in available:
            score = self._iou(self._rooms[room_id]["mask"], cycle_mask)
            if score >= best_iou:
                best_iou = score
                best_id = room_id
        if best_id is not None:
            return best_id

        radius_cells = float(config.ROOM_REGISTRY_MATCH_RADIUS_M) / float(config.MAP_RESOLUTION_M)
        best_distance = radius_cells
        for room_id in available:
            room = self._rooms[room_id]
            distance = math.hypot(
                float(cycle_room["centroid_row"]) - room["centroid_row"],
                float(cycle_room["centroid_col"]) - room["centroid_col"],
            )
            if distance <= best_distance:
                best_distance = distance
                best_id = room_id
        if best_id is not None:
            return best_id

        room_id = self._next_id
        self._next_id += 1
        self._rooms[room_id] = self._new_room(room_id)
        return room_id

    @staticmethod
    def _object_position(obj: dict) -> tuple[float, float] | None:
        position = obj.get("position")
        if not position or len(position) < 2:
            return None
        return float(position[0]), float(position[1])

    def update(
        self,
        segmentation: dict,
        viewpoints: list[dict],
        world_to_grid,
        objects: list[dict] | None = None,
        robot_cell: tuple[int, int] | None = None,
    ) -> dict:
        labels = segmentation.get("labels")
        cycle_rooms = segmentation.get("rooms") or []
        if labels is None:
            return {"labels": None, "rooms": [], "doorways": [], "adjacency": {}}

        with self._lock:
            self._revision += 1
            available = set(self._rooms)
            id_map: dict[int, int] = {}
            for cycle_room in cycle_rooms:
                cycle_id = int(cycle_room["room_id"])
                cycle_mask = labels == cycle_id
                persistent_id = self._match_room(cycle_room, cycle_mask, available)
                available.discard(persistent_id)
                id_map[cycle_id] = persistent_id

                room = self._rooms[persistent_id]
                old_cells = int(room["cell_count"])
                new_cells = int(cycle_room["cell_count"])
                if room["covered"] and new_cells > max(old_cells + 50, int(old_cells * 1.25)):
                    room["covered"] = False
                    room["empty_plan_streak"] = 0
                for key in (
                    "cell_count", "area_m2", "centroid_row", "centroid_col",
                    "anchor_row", "anchor_col",
                ):
                    if key in cycle_room:
                        room[key] = cycle_room[key]
                room["mask"] = cycle_mask.copy()
                room["last_seen_revision"] = self._revision

            persistent_labels = np.zeros_like(labels, dtype=np.int32)
            for cycle_id, persistent_id in id_map.items():
                persistent_labels[labels == cycle_id] = persistent_id
            self._labels = persistent_labels
            self._active_ids = set(id_map.values())

            # Viewpoint history is cumulative; clearing it each cycle made an entered
            # room become "unvisited" again whenever no new representative was added.
            for viewpoint in viewpoints:
                pose = viewpoint.get("pose") or {}
                cell = world_to_grid(float(pose.get("x", 0.0)), float(pose.get("y", 0.0)))
                room_id = self._room_at_cell_locked(cell)
                if room_id is None:
                    continue
                room = self._rooms[room_id]
                room["viewpoint_ids"].add(int(viewpoint["viewpoint_id"]))
                room["visited"] = True
                coverage = int(viewpoint.get("coverage_voxel_count", 0))
                image_path = viewpoint.get("image_path")
                if image_path and coverage > room["representative_coverage"]:
                    room["representative_coverage"] = coverage
                    room["representative_viewpoint_id"] = int(viewpoint["viewpoint_id"])
                    room["representative_image_path"] = str(image_path)

            for obj in objects or []:
                position = self._object_position(obj)
                if position is None:
                    continue
                room_id = self._room_at_cell_locked(world_to_grid(*position))
                category = str(obj.get("category", "")).strip().lower()
                if room_id is not None and category:
                    self._rooms[room_id]["object_labels"].add(category)

            if robot_cell is not None:
                current_id = self._room_at_cell_locked(robot_cell)
                if current_id is not None:
                    room = self._rooms[current_id]
                    if not room["visited"]:
                        room["visit_count"] += 1
                    room["visited"] = True

            self._doorways = []
            self._adjacency = {room_id: set() for room_id in self._active_ids}
            for doorway in segmentation.get("doorways") or []:
                room_a = id_map.get(int(doorway["room_a"]))
                room_b = id_map.get(int(doorway["room_b"]))
                if room_a is None or room_b is None or room_a == room_b:
                    continue
                persistent_door = dict(doorway)
                persistent_door.update({
                    "door_id": len(self._doorways) + 1,
                    "room_a": room_a,
                    "room_b": room_b,
                })
                self._doorways.append(persistent_door)
                self._adjacency.setdefault(room_a, set()).add(room_b)
                self._adjacency.setdefault(room_b, set()).add(room_a)

            return self._snapshot_locked()

    def _room_at_cell_locked(self, cell: tuple[int, int] | None) -> int | None:
        if self._labels is None or cell is None:
            return None
        row, col = int(cell[0]), int(cell[1])
        rows, cols = self._labels.shape
        if not (0 <= row < rows and 0 <= col < cols):
            return None
        value = int(self._labels[row, col])
        if value > 0:
            return value
        # Door ridges are label 0. Snap to the closest room so room transitions
        # do not temporarily disable room-scoped exploration.
        radius = max(1, int(round(config.ROOM_DOOR_NEIGHBOR_RADIUS_M / config.MAP_RESOLUTION_M)))
        candidates: list[tuple[int, int]] = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    room_id = int(self._labels[nr, nc])
                    if room_id > 0:
                        candidates.append((dr * dr + dc * dc, room_id))
        return min(candidates)[1] if candidates else None

    def room_at_cell(self, cell: tuple[int, int] | None) -> int | None:
        with self._lock:
            return self._room_at_cell_locked(cell)

    def record_exploration_result(self, room_id: int | None, has_route: bool) -> bool:
        """Return True only after an empty in-room plan is confirmed repeatedly."""
        if room_id is None:
            return False
        with self._lock:
            room = self._rooms.get(int(room_id))
            if room is None:
                return False
            if has_route:
                room["empty_plan_streak"] = 0
                room["covered"] = False
                return False
            room["empty_plan_streak"] += 1
            if room["empty_plan_streak"] >= max(1, int(config.ROOM_COMPLETION_CONFIRMATIONS)):
                room["covered"] = True
            return bool(room["covered"])

    def _door_between_locked(self, room_a: int, room_b: int) -> dict | None:
        matches = [
            door for door in self._doorways
            if {int(door["room_a"]), int(door["room_b"])} == {int(room_a), int(room_b)}
        ]
        if not matches:
            return None
        return max(matches, key=lambda item: (float(item.get("width_m", 0.0)), int(item.get("cell_count", 0))))

    def _bfs_locked(self, start: int, goal: int) -> list[int] | None:
        queue = deque([[start]])
        seen = {start}
        while queue:
            path = queue.popleft()
            if path[-1] == goal:
                return path
            for neighbor in sorted(self._adjacency.get(path[-1], set())):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(path + [neighbor])
        return None

    def navigation_candidates(self, current_room_id: int | None) -> list[dict]:
        """Rooms reachable through the live doorway graph and not yet covered."""
        with self._lock:
            candidates = []
            for room_id in sorted(self._active_ids):
                if room_id == current_room_id:
                    continue
                room = self._rooms[room_id]
                if room["covered"]:
                    continue
                if current_room_id is None:
                    room_path = [room_id]
                else:
                    room_path = self._bfs_locked(int(current_room_id), room_id)
                    if not room_path:
                        continue
                doors = []
                for room_a, room_b in zip(room_path, room_path[1:]):
                    door = self._door_between_locked(room_a, room_b)
                    if door is None:
                        doors = []
                        break
                    doors.append(dict(door))
                if len(room_path) > 1 and not doors:
                    continue
                candidates.append({
                    "room_id": room_id,
                    "category": room["category"],
                    "object_labels": sorted(room["object_labels"]),
                    "centroid_row": float(room["centroid_row"]),
                    "centroid_col": float(room["centroid_col"]),
                    "anchor_row": float(room["anchor_row"]),
                    "anchor_col": float(room["anchor_col"]),
                    "image_path": room["representative_image_path"],
                    "visited": bool(room["visited"]),
                    "room_path": room_path,
                    "doorways": doors,
                })
            return candidates

    def get_room(self, room_id: int | None) -> dict | None:
        if room_id is None:
            return None
        with self._lock:
            room = self._rooms.get(int(room_id))
            return None if room is None else self._room_summary(room)

    def known_room_count(self) -> int:
        with self._lock:
            return len(self._rooms)

    def rooms_needing_classification(self) -> list[dict]:
        now = time.time()
        cooldown = float(config.ROOM_CLASSIFICATION_RETRY_COOLDOWN_SEC)
        with self._lock:
            pending = []
            for room in self._rooms.values():
                representative_id = room["representative_viewpoint_id"]
                if representative_id is None or not room["representative_image_path"]:
                    continue
                if representative_id == room["classified_viewpoint_id"]:
                    continue
                attempted_at = room["last_classification_attempt_time"]
                if attempted_at is not None and now - attempted_at < cooldown:
                    continue
                pending.append({
                    "room_id": room["room_id"],
                    "image_path": room["representative_image_path"],
                })
            return pending

    def unvisited_rooms(self) -> list[dict]:
        # Backward-compatible API used by older callers.
        with self._lock:
            return [
                self._room_summary(room) for room in self._rooms.values()
                if not room["visited"] and room["last_seen_revision"] == self._revision
            ]

    def set_category(self, room_id: int, category: str) -> None:
        with self._lock:
            room = self._rooms.get(int(room_id))
            if room is None:
                return
            room["category"] = str(category).strip().lower()
            room["classified_viewpoint_id"] = room["representative_viewpoint_id"]
            room["last_classification_attempt_time"] = None

    def mark_classification_failed(self, room_id: int) -> None:
        with self._lock:
            room = self._rooms.get(int(room_id))
            if room is not None:
                room["last_classification_attempt_time"] = time.time()

    @staticmethod
    def _room_summary(room: dict) -> dict:
        return {
            "room_id": int(room["room_id"]),
            "cell_count": int(room["cell_count"]),
            "area_m2": float(room["area_m2"]),
            "centroid_row": float(room["centroid_row"]),
            "centroid_col": float(room["centroid_col"]),
            "anchor_row": float(room["anchor_row"]),
            "anchor_col": float(room["anchor_col"]),
            "category": room["category"],
            "viewpoint_count": len(room["viewpoint_ids"]),
            "representative_viewpoint_id": room["representative_viewpoint_id"],
            "representative_image_path": room["representative_image_path"],
            "object_labels": sorted(room["object_labels"]),
            "visited": bool(room["visited"]),
            "covered": bool(room["covered"]),
            "empty_plan_streak": int(room["empty_plan_streak"]),
        }

    def _snapshot_locked(self) -> dict:
        return {
            "labels": None if self._labels is None else self._labels.copy(),
            "rooms": [self._room_summary(self._rooms[room_id]) for room_id in sorted(self._active_ids)],
            "doorways": [dict(door) for door in self._doorways],
            "adjacency": {room_id: sorted(values) for room_id, values in self._adjacency.items()},
            "revision": self._revision,
        }

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_locked()
