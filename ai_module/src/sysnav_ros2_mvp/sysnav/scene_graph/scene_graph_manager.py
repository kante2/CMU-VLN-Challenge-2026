"""Thread-safe single-room structured scene graph manager.

The current package still uses one fixed Room_0, but Viewpoint construction follows
SysNav's coverage rule:

    C_prev = union of all existing viewpoint coverage regions
    add current pose as a viewpoint only when |C_t - C_prev| > omega

A viewpoint stores its pose, panorama image, coverage region, and visible objects.
Task-specific Object-Object edges are inferred on demand by retrieving previously
stored viewpoints that observe both target and reference objects.

Every graph update overwrites scene_graph_latest.json/.dot/.png in DEBUG_DIR.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from sysnav import config
from sysnav.reasoning.spatial_relation_reasoner import SpatialRelationReasoner
from sysnav.scene_graph.scene_graph_visualizer import SceneGraphVisualizer
from sysnav.scene_graph.viewpoint_coverage import ViewpointCoverageBuilder, VoxelKey


class SceneGraphManager:
    def __init__(self, debug_dir: str = config.DEBUG_DIR) -> None:
        self._lock = threading.RLock()
        self._room = {
            "room_id": int(config.SCENE_GRAPH_SINGLE_ROOM_ID),
            "name": config.SCENE_GRAPH_SINGLE_ROOM_NAME,
            "category": "single_room",
            "created_time": time.time(),
        }
        self._viewpoints: dict[int, dict] = {}
        self._objects: dict[int, dict] = {}
        self._edges: dict[str, dict] = {}
        self._accumulated_coverage: set[VoxelKey] = set()
        self._relation_checks: set[tuple[str, int]] = set()
        self._next_viewpoint_id = 1
        self._selected_object_id: int | None = None
        self._active_task: dict | None = None
        self._last_export_paths: dict | None = None
        self._last_export_error: str | None = None
        self._visualizer = SceneGraphVisualizer(debug_dir)
        self._relation_reasoner = SpatialRelationReasoner()
        self._coverage_builder = ViewpointCoverageBuilder()
        self._viewpoint_image_dir = Path(debug_dir) / "scene_graph_viewpoints"
        self._safe_export()

    def clear(self) -> None:
        with self._lock:
            self._viewpoints.clear()
            self._objects.clear()
            self._edges.clear()
            self._accumulated_coverage.clear()
            self._relation_checks.clear()
            self._next_viewpoint_id = 1
            self._selected_object_id = None
            self._active_task = None
            self._safe_export_locked()

    def start_task(self, task_id: int, task: dict) -> None:
        with self._lock:
            self._active_task = {
                "task_id": int(task_id),
                "raw": str(task.get("raw", "")),
                "target": str(task.get("target", "")),
                "relation": task.get("relation"),
                "reference_objects": list(task.get("reference_objects", [])),
            }
            self._selected_object_id = None
            self._safe_export_locked()

    def add_observation(
        self,
        image_rgb: np.ndarray,
        pose: dict,
        timestamp: float,
        observations: list[dict],
        object_ids: list[int],
        object_nodes: list[dict],
        task: dict,
        points_sensor: np.ndarray | None = None,
    ) -> dict:
        """Update objects and conditionally add a representative viewpoint.

        Object nodes are synchronized for every semantic observation. A Viewpoint node
        and its Viewpoint-Object edges are created only when the current LiDAR coverage
        contributes more than ``VIEWPOINT_NOVEL_VOXEL_THRESHOLD`` unseen voxels.
        Object-Object constraints are then evaluated from stored common viewpoints.
        """
        with self._lock:
            unique_object_ids = list(dict.fromkeys(int(value) for value in object_ids))
            self._sync_objects(object_nodes)

            coverage = self._coverage_builder.compute(
                np.empty((0, 3), dtype=np.float32) if points_sensor is None else points_sensor,
                pose,
            )
            novel_coverage = coverage.difference(self._accumulated_coverage)
            viewpoint_created = bool(coverage) and (
                not self._viewpoints
                or len(novel_coverage) > int(config.VIEWPOINT_NOVEL_VOXEL_THRESHOLD)
            )

            viewpoint_id: int | None = None
            if viewpoint_created:
                viewpoint_id = self._create_viewpoint(
                    image_rgb=image_rgb,
                    pose=pose,
                    timestamp=timestamp,
                    coverage=coverage,
                    novel_coverage=novel_coverage,
                    observations=observations,
                    object_ids=object_ids,
                    unique_object_ids=unique_object_ids,
                )
                self._accumulated_coverage.update(coverage)

            # SysNav Object-Object edges are on-demand. They are not restricted to the
            # current frame: previously stored viewpoints that observe both objects are
            # retrieved and their panorama images are reused for relation verification.
            relation_edges = self._infer_task_relations_from_common_viewpoints(task)

            paths = self._safe_export_locked()
            return {
                "viewpoint_created": viewpoint_created,
                "viewpoint_id": viewpoint_id,
                "coverage_voxel_count": len(coverage),
                "novel_voxel_count": len(novel_coverage),
                "novel_threshold": int(config.VIEWPOINT_NOVEL_VOXEL_THRESHOLD),
                "accumulated_coverage_voxel_count": len(self._accumulated_coverage),
                "observed_object_ids": unique_object_ids,
                "relation_edges": relation_edges,
                "debug_files": paths,
            }

    def mark_selected_object(self, object_id: int | None) -> None:
        with self._lock:
            self._selected_object_id = None if object_id is None else int(object_id)
            self._safe_export_locked()

    def find_matching_target_ids(self, task: dict) -> list[int]:
        relation = task.get("relation")
        references = set(str(value).lower() for value in task.get("reference_objects", []))
        if not relation or not references:
            return []

        with self._lock:
            matched = []
            for edge in self._edges.values():
                if edge["edge_type"] != "object_object" or edge["relation"] != relation:
                    continue
                target_ids = [self._parse_object_node_id(value) for value in edge["targets"]]
                target_categories = {
                    self._objects[object_id]["category"]
                    for object_id in target_ids
                    if object_id in self._objects
                }
                if not references.issubset(target_categories):
                    continue
                source_id = self._parse_object_node_id(edge["source"])
                if (
                    source_id in self._objects
                    and self._objects[source_id]["category"]
                    == str(task.get("target", "")).lower()
                ):
                    matched.append(source_id)
            return sorted(set(matched))

    def common_viewpoint_ids(self, object_ids: list[int]) -> list[int]:
        """Return representative viewpoints that observe every requested object."""
        requested = {int(value) for value in object_ids}
        if not requested:
            return []
        with self._lock:
            return [
                viewpoint_id
                for viewpoint_id, viewpoint in sorted(self._viewpoints.items())
                if requested.issubset(set(viewpoint.get("observed_object_ids", [])))
            ]

    def snapshot(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._snapshot_locked())

    @property
    def last_export_error(self) -> str | None:
        with self._lock:
            return self._last_export_error

    def _sync_objects(self, object_nodes: list[dict]) -> None:
        room_node = self._room_node_id()
        for object_node in object_nodes:
            object_id = int(object_node["object_id"])
            self._objects[object_id] = self._object_summary(object_node)
            self._upsert_edge(
                edge_id=f"room_object:{object_id}",
                edge_type="room_object",
                source=self._object_node_id(object_id),
                targets=[room_node],
                relation="lies_in",
                metadata={"room_id": self._room["room_id"]},
            )

    def _create_viewpoint(
        self,
        image_rgb: np.ndarray,
        pose: dict,
        timestamp: float,
        coverage: set[VoxelKey],
        novel_coverage: set[VoxelKey],
        observations: list[dict],
        object_ids: list[int],
        unique_object_ids: list[int],
    ) -> int:
        viewpoint_id = self._next_viewpoint_id
        self._next_viewpoint_id += 1
        image_path = self._save_viewpoint_image(viewpoint_id, image_rgb)

        viewpoint = {
            "viewpoint_id": viewpoint_id,
            "pose": {
                "x": float(pose["x"]),
                "y": float(pose["y"]),
                "z": float(pose.get("z", 0.0)),
                "yaw": float(pose["yaw"]),
            },
            "timestamp": float(timestamp),
            "image_path": image_path,
            "room_id": self._room["room_id"],
            "observed_object_ids": unique_object_ids,
            "object_observations": self._observation_summaries(observations, object_ids),
            "coverage_distance_m": float(config.VIEWPOINT_COVERAGE_DISTANCE_M),
            "coverage_voxel_size_m": float(config.VIEWPOINT_COVERAGE_VOXEL_SIZE_M),
            "coverage_voxel_count": len(coverage),
            "novel_voxel_count": len(novel_coverage),
            # C_i is kept as integer voxel coordinates, matching the paper's
            # viewpoint attribute A(v_i^v) = {p_i, C_i, I_i}.
            "coverage_region": [list(key) for key in sorted(coverage)],
        }
        self._viewpoints[viewpoint_id] = viewpoint

        room_node = self._room_node_id()
        viewpoint_node = self._viewpoint_node_id(viewpoint_id)
        self._upsert_edge(
            edge_id=f"room_viewpoint:{viewpoint_id}",
            edge_type="room_viewpoint",
            source=viewpoint_node,
            targets=[room_node],
            relation="lies_in",
            metadata={"room_id": self._room["room_id"]},
        )
        for object_id in unique_object_ids:
            if object_id not in self._objects:
                continue
            self._upsert_edge(
                edge_id=f"viewpoint_object:{viewpoint_id}:{object_id}",
                edge_type="viewpoint_object",
                source=viewpoint_node,
                targets=[self._object_node_id(object_id)],
                relation="observes",
                metadata={"timestamp": float(timestamp)},
            )
        return viewpoint_id

    @staticmethod
    def _observation_summaries(observations: list[dict], object_ids: list[int]) -> list[dict]:
        summaries: dict[int, dict] = {}
        for observation, object_id in zip(observations, object_ids):
            object_id = int(object_id)
            item = {
                "object_id": object_id,
                "category": str(observation.get("category", "")).lower(),
                "bbox": [int(value) for value in observation.get("bbox", (0, 0, 0, 0))],
                "confidence": float(observation.get("confidence", 0.0)),
            }
            previous = summaries.get(object_id)
            if previous is None or item["confidence"] >= previous["confidence"]:
                summaries[object_id] = item
        return [summaries[key] for key in sorted(summaries)]

    def _infer_task_relations_from_common_viewpoints(self, task: dict) -> list[dict]:
        relation = task.get("relation")
        references = list(task.get("reference_objects") or [])
        if not relation or not references:
            return []

        inferred: list[dict] = []
        task_signature = self._task_relation_signature(task)
        for viewpoint_id, viewpoint in sorted(self._viewpoints.items(), reverse=True):
            check_key = (task_signature, int(viewpoint_id))
            if check_key in self._relation_checks:
                continue
            visible_ids = [
                int(value)
                for value in viewpoint.get("observed_object_ids", [])
                if int(value) in self._objects
            ]
            if not self._viewpoint_can_contain_task_relation(task, visible_ids):
                self._relation_checks.add(check_key)
                continue

            image_rgb = self._load_viewpoint_image(viewpoint.get("image_path"))
            observations_by_id = {
                int(item["object_id"]): item
                for item in viewpoint.get("object_observations", [])
            }
            ordered_ids = [value for value in visible_ids if value in observations_by_id]
            observations = [
                {
                    "category": observations_by_id[object_id].get(
                        "category", self._objects[object_id]["category"]
                    ),
                    "bbox": tuple(observations_by_id[object_id].get("bbox", (0, 0, 0, 0))),
                    "confidence": float(
                        observations_by_id[object_id].get(
                            "confidence", self._objects[object_id].get("confidence", 0.0)
                        )
                    ),
                }
                for object_id in ordered_ids
            ]
            object_nodes = [self._objects[object_id] for object_id in ordered_ids]
            if not ordered_ids:
                continue

            edges = self._relation_reasoner.infer(
                task=task,
                image_rgb=image_rgb,
                viewpoint_pose=viewpoint["pose"],
                observations=observations,
                object_ids=ordered_ids,
                object_nodes=object_nodes,
            )
            for edge in edges:
                if self._add_object_relation_edge(viewpoint_id, edge):
                    inferred.append({**edge, "viewpoint_id": viewpoint_id})
            self._relation_checks.add(check_key)
        return inferred

    @staticmethod
    def _task_relation_signature(task: dict) -> str:
        references = ",".join(str(value).lower() for value in task.get("reference_objects", []))
        return "|".join(
            [
                str(task.get("target", "")).lower(),
                str(task.get("relation", "")).lower(),
                references,
                str(task.get("raw", "")).strip().lower(),
            ]
        )

    def _viewpoint_can_contain_task_relation(self, task: dict, visible_ids: list[int]) -> bool:
        categories = [self._objects[value]["category"] for value in visible_ids]
        if str(task.get("target", "")).lower() not in categories:
            return False
        required_references = [str(value).lower() for value in task.get("reference_objects", [])]
        return all(reference in categories for reference in required_references)

    @staticmethod
    def _load_viewpoint_image(image_path: str | None) -> np.ndarray:
        if image_path:
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is not None:
                return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # Geometry fallback does not require pixels. The tiny image also keeps the
        # reasoner interface stable when image export is disabled.
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def _add_object_relation_edge(self, viewpoint_id: int, edge: dict) -> bool:
        source_id = int(edge["source_object_id"])
        target_ids = [int(value) for value in edge["target_object_ids"]]
        if source_id not in self._objects or any(value not in self._objects for value in target_ids):
            return False

        relation = str(edge["relation"])
        target_part = "-".join(str(value) for value in target_ids)
        edge_id = f"object_object:{source_id}:{relation}:{target_part}"
        existing = self._edges.get(edge_id)
        evidence_ids = [] if existing is None else list(
            existing.get("metadata", {}).get("evidence_viewpoint_ids", [])
        )
        if int(viewpoint_id) not in evidence_ids:
            evidence_ids.append(int(viewpoint_id))

        self._upsert_edge(
            edge_id=edge_id,
            edge_type="object_object",
            source=self._object_node_id(source_id),
            targets=[self._object_node_id(value) for value in target_ids],
            relation=relation,
            metadata={
                "viewpoint_id": int(viewpoint_id),
                "evidence_viewpoint_ids": sorted(evidence_ids),
                "confidence": float(edge.get("confidence", 0.0)),
                "method": str(edge.get("method", "unknown")),
                "reason": str(edge.get("reason", "")),
            },
        )
        return existing is None

    def _upsert_edge(
        self,
        edge_id: str,
        edge_type: str,
        source: str,
        targets: list[str],
        relation: str,
        metadata: dict,
    ) -> None:
        existing = self._edges.get(edge_id)
        observation_count = 1 if existing is None else int(existing.get("observation_count", 1)) + 1
        self._edges[edge_id] = {
            "edge_id": edge_id,
            "edge_type": edge_type,
            "source": source,
            "target": targets[0] if len(targets) == 1 else None,
            "targets": list(targets),
            "relation": relation,
            "metadata": copy.deepcopy(metadata),
            "observation_count": observation_count,
            "updated_time": time.time(),
        }

    @staticmethod
    def _object_summary(node: dict) -> dict:
        return {
            "object_id": int(node["object_id"]),
            "category": str(node["category"]),
            "position": [float(value) for value in node["position"]],
            "extent_3d": [float(value) for value in node.get("extent_3d", (0, 0, 0))],
            "bbox_3d_min": [float(value) for value in node.get("bbox_3d_min", (0, 0, 0))],
            "bbox_3d_max": [float(value) for value in node.get("bbox_3d_max", (0, 0, 0))],
            "confidence": float(node.get("confidence", 0.0)),
            "observation_count": int(node.get("observation_count", 1)),
            "first_seen_time": float(node.get("first_seen_time", 0.0)),
            "last_seen_time": float(node.get("last_seen_time", 0.0)),
            "room_id": int(config.SCENE_GRAPH_SINGLE_ROOM_ID),
        }

    def _save_viewpoint_image(self, viewpoint_id: int, image_rgb: np.ndarray) -> str | None:
        if (
            not config.SCENE_GRAPH_SAVE_VIEWPOINT_IMAGES
            or not isinstance(image_rgb, np.ndarray)
            or not image_rgb.size
        ):
            return None
        try:
            self._viewpoint_image_dir.mkdir(parents=True, exist_ok=True)
            path = self._viewpoint_image_dir / f"viewpoint_{viewpoint_id:06d}.jpg"
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                return None
            return str(path)
        except Exception:
            return None

    def _snapshot_locked(self) -> dict:
        return {
            "schema_version": 2,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "room": copy.deepcopy(self._room),
            "active_task": copy.deepcopy(self._active_task),
            "selected_object_id": self._selected_object_id,
            "coverage": {
                "distance_m": float(config.VIEWPOINT_COVERAGE_DISTANCE_M),
                "voxel_size_m": float(config.VIEWPOINT_COVERAGE_VOXEL_SIZE_M),
                "novel_voxel_threshold": int(config.VIEWPOINT_NOVEL_VOXEL_THRESHOLD),
                "accumulated_voxel_count": len(self._accumulated_coverage),
            },
            "relation_check_count": len(self._relation_checks),
            "viewpoints": [
                copy.deepcopy(value) for _, value in sorted(self._viewpoints.items())
            ],
            "objects": [copy.deepcopy(value) for _, value in sorted(self._objects.items())],
            "edges": [copy.deepcopy(value) for _, value in sorted(self._edges.items())],
        }

    def _safe_export(self) -> dict | None:
        with self._lock:
            return self._safe_export_locked()

    def _safe_export_locked(self) -> dict | None:
        if not config.SCENE_GRAPH_EXPORT_ENABLED:
            return None
        try:
            self._last_export_paths = self._visualizer.export(self._snapshot_locked())
            self._last_export_error = None
            return copy.deepcopy(self._last_export_paths)
        except Exception as error:
            self._last_export_error = str(error)
            return None

    def _room_node_id(self) -> str:
        return f'room_{self._room["room_id"]}'

    @staticmethod
    def _viewpoint_node_id(viewpoint_id: int) -> str:
        return f"viewpoint_{int(viewpoint_id)}"

    @staticmethod
    def _object_node_id(object_id: int) -> str:
        return f"object_{int(object_id)}"

    @staticmethod
    def _parse_object_node_id(node_id: str) -> int:
        return int(node_id.split("_", 1)[1])
