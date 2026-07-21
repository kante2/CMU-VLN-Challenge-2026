"""Persistent object nodes for the current single-room map."""

from __future__ import annotations

import copy
import threading
import time

import numpy as np

from sysnav import config
from sysnav.memory.object_association import find_best_match


class ObjectMemory:
    def __init__(self) -> None:
        self._nodes: dict[int, dict] = {}
        self._next_id = 1
        self._lock = threading.RLock()

    def clear(self) -> None:
        with self._lock:
            self._nodes.clear()
            self._next_id = 1

    @staticmethod
    def _copy_node(node: dict) -> dict:
        return {key: value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value) for key, value in node.items()}

    def _new_node(self, observation: dict, timestamp: float) -> dict:
        object_id = self._next_id
        self._next_id += 1
        return {
            "object_id": object_id,
            "category": str(observation["category"]).lower(),
            "position": tuple(float(v) for v in observation["position"]),
            "point_cloud": observation.get("point_cloud", np.empty((0, 3), np.float32)).copy(),
            "bbox_3d_min": tuple(observation.get("bbox_3d_min", (0, 0, 0))),
            "bbox_3d_max": tuple(observation.get("bbox_3d_max", (0, 0, 0))),
            "extent_3d": tuple(observation.get("extent_3d", (0, 0, 0))),
            "representative_image": observation.get("crop_image").copy() if isinstance(observation.get("crop_image"), np.ndarray) else None,
            "representative_confidence": float(observation.get("confidence", 0.0)),
            "confidence": float(observation.get("confidence", 0.0)),
            "observation_count": 1,
            "first_seen_time": timestamp,
            "last_seen_time": timestamp,
            "latest_bbox_2d": tuple(observation.get("bbox", (0, 0, 0, 0))),
            "num_points": int(observation.get("num_points", 0)),
        }

    @staticmethod
    def _merge_points(old_points: np.ndarray, new_points: np.ndarray) -> np.ndarray:
        arrays = [arr.reshape(-1, 3) for arr in (old_points, new_points) if isinstance(arr, np.ndarray) and arr.size]
        if not arrays:
            return np.empty((0, 3), dtype=np.float32)
        merged = np.concatenate(arrays, axis=0)
        if len(merged) > config.MEMORY_MAX_POINTS_PER_OBJECT:
            merged = merged[np.linspace(0, len(merged) - 1, config.MEMORY_MAX_POINTS_PER_OBJECT, dtype=np.int64)]
        return merged.astype(np.float32, copy=False)

    def _merge(self, node: dict, observation: dict, timestamp: float, metrics: dict) -> None:
        count = int(node["observation_count"])
        alpha = 1.0 / min(count + 1, 10)
        old_position = np.asarray(node["position"], dtype=np.float64)
        new_position = np.asarray(observation["position"], dtype=np.float64)
        node["position"] = tuple(float(v) for v in ((1 - alpha) * old_position + alpha * new_position))
        node["point_cloud"] = self._merge_points(node["point_cloud"], observation.get("point_cloud"))
        old_extent = np.asarray(node["extent_3d"], dtype=np.float64)
        new_extent = np.asarray(observation.get("extent_3d", old_extent), dtype=np.float64)
        node["extent_3d"] = tuple(float(v) for v in ((1 - alpha) * old_extent + alpha * new_extent))
        node["bbox_3d_min"] = tuple(observation.get("bbox_3d_min", node["bbox_3d_min"]))
        node["bbox_3d_max"] = tuple(observation.get("bbox_3d_max", node["bbox_3d_max"]))
        node["latest_bbox_2d"] = tuple(observation.get("bbox", node["latest_bbox_2d"]))
        node["confidence"] = max(node["confidence"], float(observation.get("confidence", 0.0)))
        node["last_seen_time"] = timestamp
        node["observation_count"] = count + 1
        node["num_points"] = len(node["point_cloud"])
        node["association_score"] = float(metrics["score"])
        if metrics.get("observation_histogram") is not None:
            node["appearance_histogram"] = metrics["observation_histogram"].copy()
        crop = observation.get("crop_image")
        confidence = float(observation.get("confidence", 0.0))
        if isinstance(crop, np.ndarray) and crop.size and confidence >= node["representative_confidence"]:
            node["representative_image"] = crop.copy()
            node["representative_confidence"] = confidence

    def update(self, observations: list[dict], timestamp: float | None = None) -> list[int]:
        timestamp = time.time() if timestamp is None else float(timestamp)
        changed = []
        with self._lock:
            for observation in observations:
                same_category = [node for node in self._nodes.values() if node["category"] == str(observation["category"]).lower()]
                match, metrics = find_best_match(same_category, observation)
                if match is None:
                    node = self._new_node(observation, timestamp)
                    self._nodes[node["object_id"]] = node
                    changed.append(node["object_id"])
                else:
                    self._merge(match, observation, timestamp, metrics)
                    changed.append(match["object_id"])
        return changed

    def find_by_category(self, category: str) -> list[dict]:
        with self._lock:
            return [self._copy_node(node) for node in self._nodes.values() if node["category"] == category.strip().lower()]

    def get(self, object_id: int) -> dict | None:
        with self._lock:
            node = self._nodes.get(int(object_id))
            return None if node is None else self._copy_node(node)

    def all_nodes(self) -> list[dict]:
        with self._lock:
            return [self._copy_node(node) for node in self._nodes.values()]
