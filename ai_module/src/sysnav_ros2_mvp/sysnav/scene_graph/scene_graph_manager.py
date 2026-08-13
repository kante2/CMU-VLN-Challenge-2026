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
from sysnav.task.query_parser import effective_relation_chain


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
            # Lang2LTL-2(Sec IV-C, Spatial Predicate Grounding) 스타일 보강 - 위
            # 경로는 두 물체가 "같은 viewpoint에서 동시에" 관측돼야만 검증을 시도한다.
            # 유리창처럼 LiDAR grounding 성공률이 낮은 물체가 참조 물체로 쓰이면 그
            # "동시에 보이는 순간"이 영영 안 와서 관계 검증 자체가 시작도 못 하는
            # 문제가 있었다. 이 보강 경로는 그 제약이 없다 - 두 물체가 각각 언제든
            # 한 번이라도 grounding만 됐으면(같은 프레임일 필요 없음) 이미 저장된
            # 전역 위치만으로 순수 기하 판정을 한다.
            relation_edges = relation_edges + self._infer_task_relations_globally(task, pose)

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

    def infer_relations_for_task(self, task: dict, pose: dict) -> list[dict]:
        """add_observation()이 매 perception 프레임마다 relation edge를 갱신하긴 하지만,
        그건 그 순간 perception_job에 넘어간 task(=self.task, 최상위 task) 기준이다.
        Mission 3(Instruction-Following)는 절마다 독립된 relation을 가진 step 단위
        task(mission3_pipe.py의 step["parsed"])를 selection_job에 직접 넘기는데, 최상위
        task는 target/relation이 빈 placeholder라서 그 step의 relation은 add_observation을
        아무리 거쳐도 절대 edge화되지 않는다(항상 image-verification 폴백으로만 빠짐).
        selection_job이 실제로 판정하려는 task로 이 메서드를 먼저 호출해서 그 gap을 메운다.
        같은 (task, viewpoint) 조합은 add_observation과 동일하게 _relation_checks로
        캐시되므로 반복 호출해도 비용이 크지 않다."""
        with self._lock:
            inferred = self._infer_task_relations_from_common_viewpoints(task)
            inferred = inferred + self._infer_task_relations_globally(task, pose)
            self._safe_export_locked()
            return inferred

    def find_matching_target_ids(self, task: dict) -> list[int]:
        """target 카테고리에서 시작해서 relation_chain을 hop-by-hop으로 따라가며
        실제 object-object edge가 이어지는 object만 남긴다 (예: "A closest to B near C"는
        A--nearest-->B, B--near-->C 두 edge가 각각 실제로 존재해야 A가 매칭됨).
        "between"은 3항 relation(하나의 edge가 target을 2개 가짐)이라 예전처럼 별도 처리한다."""
        chain = effective_relation_chain(task)
        if not chain:
            return []

        with self._lock:
            if len(chain) == 1 and chain[0][1] == "between":
                references = set(str(value).lower() for value in task.get("reference_objects", []))
                if len(references) < 2:
                    return []
                matched = []
                for edge in self._edges.values():
                    if edge["edge_type"] != "object_object" or edge["relation"] != "between":
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
                    if source_id in self._objects and self._objects[source_id]["category"] == chain[0][0]:
                        matched.append(source_id)
                return sorted(set(matched))

            # frontier: 현재 hop까지 도달한 "체인상의 현재 object_id" -> 그 경로가 시작된
            # 원래 target(root) object_id 집합. hop을 넘어갈 때마다 edge를 타고 이동하는
            # object가 바뀌므로(예: bowl -> knife_rack -> trash_can), root와 현재 위치를
            # 따로 추적해야 마지막에 "root(target) object"만 뽑아낼 수 있다.
            frontier: dict[int, set[int]] = {
                object_id: {object_id}
                for object_id, obj in self._objects.items()
                if obj["category"] == chain[0][0]
            }
            for _, relation, target_category in chain:
                if not frontier:
                    return []
                next_frontier: dict[int, set[int]] = {}
                for edge in self._edges.values():
                    if edge["edge_type"] != "object_object" or edge["relation"] != relation:
                        continue
                    source_id = self._parse_object_node_id(edge["source"])
                    roots = frontier.get(source_id)
                    if not roots:
                        continue
                    target_ids = [self._parse_object_node_id(value) for value in edge["targets"]]
                    for target_id in target_ids:
                        if target_id in self._objects and self._objects[target_id]["category"] == target_category:
                            next_frontier.setdefault(target_id, set()).update(roots)
                frontier = next_frontier

            matched_roots: set[int] = set()
            for roots in frontier.values():
                matched_roots.update(roots)
            return sorted(matched_roots)

    def finalize_unique_comparative_relation(self, task: dict) -> int | None:
        """Create a nearest edge when exhaustive search found one target only.

        During exploration, a single observed source is merely the closest seen
        so far and must not receive a superlative edge.  After reachable
        frontiers are exhausted, however, one source is the unique global
        candidate.  A reference node is still required so the resulting edge is
        explicit and inspectable rather than an unverified destination shortcut.
        """
        chain = effective_relation_chain(task)
        if not chain or chain[0][1] not in ("nearest", "closest"):
            return None
        source_category, relation, reference_category = chain[0]
        with self._lock:
            sources = [
                obj for obj in self._objects.values()
                if obj["category"] == source_category
            ]
            references = [
                obj for obj in self._objects.values()
                if obj["category"] == reference_category
            ]
            if len(sources) != 1 or not references:
                return None
            source = sources[0]
            reference = min(
                references,
                key=lambda obj: float(np.linalg.norm(
                    np.asarray(source["position"][:2], dtype=np.float64)
                    - np.asarray(obj["position"][:2], dtype=np.float64)
                )),
            )
            added = self._add_object_relation_edge(0, {
                "source_object_id": int(source["object_id"]),
                "target_object_ids": [int(reference["object_id"])],
                "relation": relation,
                "confidence": 1.0,
                "method": "unique_after_exploration",
                "reason": "only target instance after reachable-frontier exhaustion",
            })
            if added:
                self._safe_export_locked()
            return int(source["object_id"])

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

    def best_viewpoint_for_objects(self, object_ids: list[int]) -> dict | None:
        """요청한 물체들을 가장 많이 동시에 본 viewpoint. 없으면 None.

        common_viewpoint_ids()는 "전부 다 본" viewpoint만 돌려주므로 물체가 여러 개면
        보통 빈 목록이 된다. 개수 세기는 "한 장에 최대한 많이 담긴 뷰"가 필요해서,
        교집합이 아니라 가장 많이 겹치는 뷰를 고른다 - 뷰를 하나로 확정하면 여러 뷰의
        개수를 합칠 때 생기는 중복 계산이 원천적으로 없어진다.

        반환: {"viewpoint_id", "image_path", "visible_object_ids", "visible_count"}
        """
        requested = {int(value) for value in object_ids}
        if not requested:
            return None
        best: dict | None = None
        with self._lock:
            for viewpoint_id, viewpoint in sorted(self._viewpoints.items()):
                visible = requested & set(viewpoint.get("observed_object_ids", []))
                if not visible:
                    continue
                image_path = viewpoint.get("image_path")
                if not image_path:
                    continue  # 이미지가 없으면 VLM에 넣을 수 없다
                if best is None or len(visible) > best["visible_count"]:
                    best = {
                        "viewpoint_id": int(viewpoint_id),
                        "image_path": str(image_path),
                        "visible_object_ids": sorted(visible),
                        "visible_count": len(visible),
                    }
        return best

    def snapshot(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._snapshot_locked())

    def list_viewpoints(self) -> list[dict]:
        """RoomRegistry가 mapping cycle(0.35초)마다 부르는 가벼운 조회 - object/edge
        전체를 deepcopy하는 snapshot()과 달리 room 배정에 필요한 필드만 뽑는다."""
        with self._lock:
            return [
                {
                    "viewpoint_id": viewpoint["viewpoint_id"],
                    "pose": dict(viewpoint["pose"]),
                    "image_path": viewpoint.get("image_path"),
                    "coverage_voxel_count": viewpoint.get("coverage_voxel_count", 0),
                }
                for viewpoint in self._viewpoints.values()
            ]

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
        # effective_relation_chain()을 써야 한다 - task["relation"]/["reference_objects"]는
        # "target -> 첫 hop"만 담고 있어서, 체인의 첫 hop이 비어있는(이론상) 경우를 놓칠 수 있다.
        chain = effective_relation_chain(task)
        if not chain:
            return []

        inferred: list[dict] = []
        attempted_viewpoint_ids: list[int] = []
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
            attempted_viewpoint_ids.append(int(viewpoint_id))

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

        if not attempted_viewpoint_ids:
            # SpatialRelationReasoner.infer()가 한 번도 안 불렸다는 뜻 - 즉 sysnav_relation_check.txt
            # 에 아무 줄도 안 남는다. "왜 relation 검증이 아예 시도조차 안 됐는지"를 별도로
            # 남겨야 사용자가 "이 물체가 아직 한 프레임에 같이 안 보였구나"를 알 수 있다.
            self._relation_reasoner.log_relation_check_skipped(task, chain, self._objects)
        return inferred

    def _infer_task_relations_globally(self, task: dict, pose: dict) -> list[dict]:
        """SpatialRelationReasoner.infer_global() 참고 - "같은 viewpoint에서 동시에
        관측" 제약이 없는 보강 경로. object_id=0을 "특정 viewpoint에 묶이지 않고
        전역 위치로 판정함"을 나타내는 sentinel로 쓴다(실제 viewpoint_id는 1부터
        시작하므로 충돌 없음)."""
        inferred: list[dict] = []
        for edge in self._relation_reasoner.infer_global(task, list(self._objects.values()), pose):
            if self._add_object_relation_edge(0, edge):
                inferred.append({**edge, "viewpoint_id": None})
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
        """이 viewpoint에서 relation_chain 중 최소 한 hop이라도 검증을 시도할 만한지 본다.
        "between"(3항 relation)은 target+두 reference가 한 프레임에 다 보여야 기하 판정이
        되므로 예전처럼 전부 요구하지만, 그 외 체인은 hop마다 다른 viewpoint에서 관측됐을
        수 있으므로(예: "A closest to B near C"에서 A,B와 B,C가 서로 다른 프레임에 보일 수
        있음) hop 하나만 양쪽 다 보이면 시도할 가치가 있다고 판단한다 - 나머지 hop은
        해당 hop이 다 보이는 다른 viewpoint가 처리한다."""
        categories = {self._objects[value]["category"] for value in visible_ids}
        chain = effective_relation_chain(task)
        if not chain:
            return False
        if task.get("relation") == "between":
            if str(task.get("target", "")).lower() not in categories:
                return False
            required_references = [str(value).lower() for value in task.get("reference_objects", [])]
            return all(reference in categories for reference in required_references)
        return any(
            source_category in categories and target_category in categories
            for source_category, _, target_category in chain
        )

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
            # SysNav paper Sec. IV-A-1의 self-attribute(φ, 예: color) - object_memory에
            # 캐싱된 VLM 추론 결과(attribute_verifier.py)를 그대로 노출한다. 아직 한 번도
            # 물어본 적 없으면 빈 dict.
            "self_attributes": {
                str(key): bool(value) for key, value in (node.get("self_attributes") or {}).items()
            },
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
