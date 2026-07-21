"""On-demand Object-Object spatial relation reasoning.

SysNav does not create every possible Object-Object edge in advance. When the
instruction contains a spatial constraint, this module checks only the target and
reference objects observed from the same viewpoint. Gemini can validate the relation
from the annotated RGB image; deterministic geometry is used as a fallback.
"""

from __future__ import annotations

from itertools import combinations, product
import json
import math
import os

import cv2
import numpy as np

from sysnav import config


class SpatialRelationReasoner:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._client = None

    def infer(
        self,
        task: dict,
        image_rgb: np.ndarray,
        viewpoint_pose: dict,
        observations: list[dict],
        object_ids: list[int],
        object_nodes: list[dict],
    ) -> list[dict]:
        relation = task.get("relation")
        references = list(task.get("reference_objects") or [])
        if not relation or not references:
            return []

        records = self._build_records(observations, object_ids, object_nodes)
        candidates = self._candidate_relations(task, records)
        if not candidates:
            return []

        if config.SCENE_GRAPH_USE_GEMINI_RELATIONS and self.api_key:
            try:
                gemini_edges = self._infer_with_gemini(
                    question=task.get("raw", ""),
                    relation=relation,
                    image_rgb=image_rgb,
                    candidates=candidates,
                    records=records,
                )
                if gemini_edges:
                    return gemini_edges
            except Exception:
                # Gemini API, model, response-schema, or network errors must not stop
                # perception. The geometric check below keeps the graph operational.
                pass

        return self._infer_with_geometry(
            relation=relation,
            candidates=candidates,
            records=records,
            viewpoint_pose=viewpoint_pose,
        )

    @staticmethod
    def _build_records(
        observations: list[dict],
        object_ids: list[int],
        object_nodes: list[dict],
    ) -> dict[int, dict]:
        nodes_by_id = {int(node["object_id"]): node for node in object_nodes}
        records: dict[int, dict] = {}

        for observation, object_id in zip(observations, object_ids):
            object_id = int(object_id)
            node = nodes_by_id.get(object_id)
            if node is None:
                continue

            record = {
                "object_id": object_id,
                "category": str(node["category"]).lower(),
                "position": tuple(float(v) for v in node["position"]),
                "extent_3d": tuple(float(v) for v in node.get("extent_3d", (0, 0, 0))),
                "bbox_3d_min": tuple(float(v) for v in node.get("bbox_3d_min", (0, 0, 0))),
                "bbox_3d_max": tuple(float(v) for v in node.get("bbox_3d_max", (0, 0, 0))),
                "bbox_2d": tuple(int(v) for v in observation.get("bbox", (0, 0, 0, 0))),
                "confidence": float(observation.get("confidence", node.get("confidence", 0.0))),
            }

            # One physical object may be associated with multiple observations in a
            # frame. Keep the clearest observation for image annotation.
            previous = records.get(object_id)
            if previous is None or record["confidence"] >= previous["confidence"]:
                records[object_id] = record

        return records

    @staticmethod
    def _candidate_relations(task: dict, records: dict[int, dict]) -> list[dict]:
        target_category = str(task.get("target", "")).lower()
        reference_categories = [str(value).lower() for value in task.get("reference_objects", [])]
        relation = str(task.get("relation", ""))

        targets = [record for record in records.values() if record["category"] == target_category]
        if not targets:
            return []

        if relation == "between" and len(reference_categories) >= 2:
            first_refs = [record for record in records.values() if record["category"] == reference_categories[0]]
            second_refs = [record for record in records.values() if record["category"] == reference_categories[1]]
            output = []
            for target, first_ref, second_ref in product(targets, first_refs, second_refs):
                ids = {target["object_id"], first_ref["object_id"], second_ref["object_id"]}
                if len(ids) != 3:
                    continue
                output.append({
                    "source_object_id": target["object_id"],
                    "target_object_ids": [first_ref["object_id"], second_ref["object_id"]],
                    "relation": relation,
                })
            return output

        reference_category = reference_categories[0] if reference_categories else ""
        references = [record for record in records.values() if record["category"] == reference_category]
        return [
            {
                "source_object_id": target["object_id"],
                "target_object_ids": [reference["object_id"]],
                "relation": relation,
            }
            for target, reference in product(targets, references)
            if target["object_id"] != reference["object_id"]
        ]

    def _load_client(self) -> None:
        if self._client is not None:
            return
        from google import genai
        self._client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _annotate_image(image_rgb: np.ndarray, records: dict[int, dict]) -> np.ndarray:
        annotated = image_rgb.copy()
        for object_id, record in records.items():
            x1, y1, x2, y2 = record["bbox_2d"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"id={object_id} {record['category']}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return annotated

    @staticmethod
    def _jpeg(image_rgb: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        if not ok:
            raise RuntimeError("Spatial-relation image JPEG encoding failed")
        return encoded.tobytes()

    def _infer_with_gemini(
        self,
        question: str,
        relation: str,
        image_rgb: np.ndarray,
        candidates: list[dict],
        records: dict[int, dict],
    ) -> list[dict]:
        self._load_client()
        from google.genai import types

        annotated = self._annotate_image(image_rgb, records)
        object_summary = [
            {
                "object_id": object_id,
                "category": record["category"],
                "position_xyz": [round(v, 3) for v in record["position"]],
                "bbox_2d": list(record["bbox_2d"]),
            }
            for object_id, record in sorted(records.items())
        ]

        prompt = f"""
You validate an on-demand Object-Object edge for a mobile robot scene graph.
Instruction: {question}
Requested normalized relation: {relation}
Visible objects: {json.dumps(object_summary, ensure_ascii=False)}
Candidate checks: {json.dumps(candidates, ensure_ascii=False)}

The image is annotated with exact object IDs. For every candidate check, decide whether
its requested relation is visibly true. Return only candidates that are true. Preserve the
provided source_object_id, target_object_ids, and normalized relation exactly. Do not add
new object IDs or relations. A relation that is ambiguous must be omitted.
""".strip()

        response = self._client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[
                prompt,
                types.Part.from_bytes(data=self._jpeg(annotated), mime_type="image/jpeg"),
            ],
            config=types.GenerateContentConfig(
                temperature=config.GEMINI_TEMPERATURE,
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "relations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source_object_id": {"type": "integer"},
                                    "target_object_ids": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                    },
                                    "relation": {"type": "string"},
                                    "confidence": {"type": "number"},
                                    "reason": {"type": "string"},
                                },
                                "required": [
                                    "source_object_id",
                                    "target_object_ids",
                                    "relation",
                                    "confidence",
                                ],
                            },
                        }
                    },
                    "required": ["relations"],
                },
            ),
        )
        if not response.text:
            return []

        allowed = {
            (
                int(candidate["source_object_id"]),
                tuple(int(value) for value in candidate["target_object_ids"]),
                candidate["relation"],
            )
            for candidate in candidates
        }
        output = []
        for item in json.loads(response.text).get("relations", []):
            key = (
                int(item["source_object_id"]),
                tuple(int(value) for value in item["target_object_ids"]),
                str(item["relation"]),
            )
            confidence = float(item.get("confidence", 0.0))
            if key not in allowed or confidence < config.SCENE_GRAPH_RELATION_MIN_CONFIDENCE:
                continue
            output.append({
                "source_object_id": key[0],
                "target_object_ids": list(key[1]),
                "relation": key[2],
                "confidence": confidence,
                "method": "gemini",
                "reason": str(item.get("reason", "")),
            })
        return output

    def _infer_with_geometry(
        self,
        relation: str,
        candidates: list[dict],
        records: dict[int, dict],
        viewpoint_pose: dict,
    ) -> list[dict]:
        output = []
        for candidate in candidates:
            source = records[candidate["source_object_id"]]
            targets = [records[object_id] for object_id in candidate["target_object_ids"]]
            holds, confidence, reason = self._geometry_check(
                relation,
                source,
                targets,
                viewpoint_pose,
            )
            if not holds:
                continue
            output.append({
                **candidate,
                "confidence": confidence,
                "method": "geometry",
                "reason": reason,
            })
        return output

    @staticmethod
    def _local_xy(position: tuple[float, float, float], pose: dict) -> np.ndarray:
        dx = float(position[0]) - float(pose["x"])
        dy = float(position[1]) - float(pose["y"])
        yaw = float(pose["yaw"])
        return np.array([
            math.cos(yaw) * dx + math.sin(yaw) * dy,
            -math.sin(yaw) * dx + math.cos(yaw) * dy,
        ])

    def _geometry_check(
        self,
        relation: str,
        source: dict,
        targets: list[dict],
        pose: dict,
    ) -> tuple[bool, float, str]:
        source_position = np.asarray(source["position"], dtype=np.float64)
        target_position = np.asarray(targets[0]["position"], dtype=np.float64)
        difference = source_position - target_position
        distance_xy = float(np.linalg.norm(difference[:2]))

        source_extent = np.asarray(source["extent_3d"], dtype=np.float64)
        target_extent = np.asarray(targets[0]["extent_3d"], dtype=np.float64)
        adaptive_near = max(
            config.SCENE_GRAPH_NEAR_DISTANCE_M,
            0.55 * float(np.linalg.norm(source_extent[:2] + target_extent[:2])),
        )

        if relation == "near":
            confidence = max(0.0, 1.0 - distance_xy / max(adaptive_near, 1e-6))
            return distance_xy <= adaptive_near, confidence, f"xy_distance={distance_xy:.3f}m"

        if relation == "beside":
            height_difference = abs(float(source_position[2] - target_position[2]))
            holds = distance_xy <= adaptive_near and height_difference <= config.SCENE_GRAPH_BESIDE_Z_TOLERANCE_M
            confidence = max(0.0, 1.0 - distance_xy / max(adaptive_near, 1e-6))
            return holds, confidence, f"xy_distance={distance_xy:.3f}m, dz={height_difference:.3f}m"

        source_local = self._local_xy(source["position"], pose)
        target_local = self._local_xy(targets[0]["position"], pose)
        local_difference = source_local - target_local
        margin = config.SCENE_GRAPH_DIRECTION_MARGIN_M

        if relation == "left_of":
            return local_difference[1] > margin, min(1.0, abs(local_difference[1])), "viewpoint-local left axis"
        if relation == "right_of":
            return local_difference[1] < -margin, min(1.0, abs(local_difference[1])), "viewpoint-local right axis"
        if relation == "in_front_of":
            return local_difference[0] > margin, min(1.0, abs(local_difference[0])), "viewpoint-local forward axis"
        if relation == "behind":
            return local_difference[0] < -margin, min(1.0, abs(local_difference[0])), "viewpoint-local backward axis"

        source_min = np.asarray(source["bbox_3d_min"], dtype=np.float64)
        source_max = np.asarray(source["bbox_3d_max"], dtype=np.float64)
        target_min = np.asarray(targets[0]["bbox_3d_min"], dtype=np.float64)
        target_max = np.asarray(targets[0]["bbox_3d_max"], dtype=np.float64)

        if relation == "above":
            gap = float(source_min[2] - target_max[2])
            return gap >= -config.SCENE_GRAPH_ON_VERTICAL_TOLERANCE_M, max(0.0, 1.0 - abs(gap)), f"vertical_gap={gap:.3f}m"
        if relation == "under":
            gap = float(target_min[2] - source_max[2])
            return gap >= -config.SCENE_GRAPH_ON_VERTICAL_TOLERANCE_M, max(0.0, 1.0 - abs(gap)), f"vertical_gap={gap:.3f}m"
        if relation == "on":
            vertical_gap = float(source_min[2] - target_max[2])
            horizontal_inside = (
                target_min[0] - config.SCENE_GRAPH_ON_HORIZONTAL_MARGIN_M
                <= source_position[0]
                <= target_max[0] + config.SCENE_GRAPH_ON_HORIZONTAL_MARGIN_M
                and target_min[1] - config.SCENE_GRAPH_ON_HORIZONTAL_MARGIN_M
                <= source_position[1]
                <= target_max[1] + config.SCENE_GRAPH_ON_HORIZONTAL_MARGIN_M
            )
            holds = abs(vertical_gap) <= config.SCENE_GRAPH_ON_VERTICAL_TOLERANCE_M and horizontal_inside
            confidence = max(0.0, 1.0 - abs(vertical_gap) / max(config.SCENE_GRAPH_ON_VERTICAL_TOLERANCE_M, 1e-6))
            return holds, confidence, f"vertical_gap={vertical_gap:.3f}m, horizontal_inside={horizontal_inside}"

        if relation == "between" and len(targets) >= 2:
            first = np.asarray(targets[0]["position"][:2], dtype=np.float64)
            second = np.asarray(targets[1]["position"][:2], dtype=np.float64)
            point = source_position[:2]
            segment = second - first
            denominator = float(np.dot(segment, segment))
            if denominator <= 1e-8:
                return False, 0.0, "reference objects overlap"
            t = float(np.dot(point - first, segment) / denominator)
            projection = first + np.clip(t, 0.0, 1.0) * segment
            line_distance = float(np.linalg.norm(point - projection))
            holds = 0.10 <= t <= 0.90 and line_distance <= config.SCENE_GRAPH_BETWEEN_LINE_TOLERANCE_M
            confidence = max(0.0, 1.0 - line_distance / max(config.SCENE_GRAPH_BETWEEN_LINE_TOLERANCE_M, 1e-6))
            return holds, confidence, f"segment_t={t:.3f}, line_distance={line_distance:.3f}m"

        return False, 0.0, f"unsupported relation={relation}"
