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
import time

import cv2
import numpy as np

from sysnav import config
from sysnav.activity_log import LLM, activity
from sysnav.llm_trace import llm_trace
from sysnav.task.query_parser import effective_relation_chain


class SpatialRelationReasoner:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._client = None
        # sysnav_relation_check.txt는 매번 append라 노드를 재실행할 때마다 이전 실행
        # 기록이 계속 쌓였다. 노드가 새로 뜰 때(=SpatialRelationReasoner가 새로 생성될
        # 때) 한 번 지워서, 이번 실행 동안의 기록만 남게 한다.
        self._reset_debug_table()

    @staticmethod
    def _reset_debug_table() -> None:
        if not config.SAVE_DEBUG_IMAGES:
            return
        try:
            path = os.path.join(config.DEBUG_DIR, "sysnav_relation_check.txt")
            if os.path.exists(path):
                os.remove(path)
        except Exception as error:  # pragma: no cover - debug output must never crash startup
            print(f"[spatial_relation_reasoner] failed to reset relation debug table: {error}")

    def infer(
        self,
        task: dict,
        image_rgb: np.ndarray,
        viewpoint_pose: dict,
        observations: list[dict],
        object_ids: list[int],
        object_nodes: list[dict],
    ) -> list[dict]:
        chain = effective_relation_chain(task)
        if not chain:
            return []

        records = self._build_records(observations, object_ids, object_nodes)
        candidates = self._candidate_relations(task, records)
        if not candidates:
            return []

        if config.SCENE_GRAPH_USE_GEMINI_RELATIONS and self.api_key:
            try:
                gemini_edges = self._infer_with_gemini(
                    question=task.get("raw", ""),
                    image_rgb=image_rgb,
                    candidates=candidates,
                    records=records,
                )
                self._save_debug_table(
                    task=task, candidates=candidates,
                    records=records, accepted=gemini_edges, method="gemini",
                )
                gemini_edges = self._veto_impossible_contacts(
                    gemini_edges, records, viewpoint_pose
                )
                if gemini_edges:
                    return gemini_edges
            except Exception:
                # Gemini API, model, response-schema, or network errors must not stop
                # perception. The geometric check below keeps the graph operational.
                pass

        geometry_edges = self._infer_with_geometry(
            candidates=candidates,
            records=records,
            viewpoint_pose=viewpoint_pose,
        )
        self._save_debug_table(
            task=task, candidates=candidates,
            records=records, accepted=geometry_edges, method="geometry",
        )
        return geometry_edges

    def infer_global(self, task: dict, object_nodes: list[dict], robot_pose: dict) -> list[dict]:
        """co-observation(같은 프레임/viewpoint에 동시에 잡혀야 함) 요구 없이,
        object_memory에 이미 쌓인 전역 위치만으로 관계를 판정한다.

        Lang2LTL-2(Sec IV-C, Spatial Predicate Grounding) 방식: figure/ground를
        독립적으로(언제 어떤 프레임에서 관측됐든 상관없이) 각자 grounding한 뒤,
        순수 기하 연산(거리/각도/구간)으로 관계를 판정한다. 우리 기존 infer()는
        "같은 viewpoint에서 둘 다 보여야" 호출되는데(scene_graph_manager의
        _viewpoint_can_contain_task_relation), 유리창처럼 LiDAR grounding
        성공률이 낮은 물체가 reference로 쓰이면 "동시에 보이는 순간"이 영영 안 와서
        관계 검증 자체가 시작도 못 하는 문제가 있었다 - 이 메서드는 그 제약이 아예
        없다(따라서 반드시 필요한 "같은 프레임 이미지"도 없으므로 Gemini는 안 쓰고
        _infer_with_geometry만 돈다 - 어차피 그쪽 relation 판정 자체가 처음부터
        position/extent_3d/bbox_3d만 쓰고 이미지에 의존하지 않는다).
        """
        chain = effective_relation_chain(task)
        if not chain:
            return []
        records = self._build_records_from_nodes(object_nodes)
        candidates = self._candidate_relations(task, records)
        if not candidates:
            return []
        return self._infer_with_geometry(candidates, records, viewpoint_pose=robot_pose)

    @staticmethod
    def _build_records_from_nodes(object_nodes: list[dict]) -> dict[int, dict]:
        records: dict[int, dict] = {}
        for node in object_nodes:
            object_id = int(node["object_id"])
            records[object_id] = {
                "object_id": object_id,
                "category": str(node["category"]).lower(),
                "position": tuple(float(v) for v in node["position"]),
                "extent_3d": tuple(float(v) for v in node.get("extent_3d", (0, 0, 0))),
                "bbox_3d_min": tuple(float(v) for v in node.get("bbox_3d_min", (0, 0, 0))),
                "bbox_3d_max": tuple(float(v) for v in node.get("bbox_3d_max", (0, 0, 0))),
                "bbox_2d": tuple(int(v) for v in node.get("latest_bbox_2d", (0, 0, 0, 0))),
                "confidence": float(node.get("confidence", 0.0)),
            }
        return records

    # ------------------------------------------------------------------
    # Debug: [obj1] [obj2] [문장 속 relation] [LLM/기하 검증 결과] 표를
    # ai_module/debug/sysnav_relation_check.txt 에 계속 append한다.
    # ------------------------------------------------------------------

    @staticmethod
    def log_relation_check_skipped(
        task: dict,
        chain: list[tuple[str, str, str]],
        objects: dict[int, dict],
    ) -> None:
        """infer()가 단 한 번도 안 불렸을 때(=이 파일에 아무 줄도 안 남을 때) 왜인지
        남긴다. 가장 흔한 원인: chain의 어느 hop도 두 카테고리가 같은 프레임(viewpoint)
        에서 동시에 관측된 적이 없음 - 예를 들어 "knife rack"이 아직 한 번도 bowl이나
        trash can과 같은 화면에 안 잡혔거나, 애초에 아직 탐지된 적이 없는 카테고리."""
        if not config.SAVE_DEBUG_IMAGES:
            return
        try:
            known_categories = {obj["category"] for obj in objects.values()}
            os.makedirs(config.DEBUG_DIR, exist_ok=True)
            path = os.path.join(config.DEBUG_DIR, "sysnav_relation_check.txt")
            lines = [
                f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} | question: {task.get('raw', '')} ===",
                "SKIPPED - no stored viewpoint has co-observed both categories of any relation hop yet:",
            ]
            for source_category, relation, target_category in chain:
                missing = [
                    category for category in (source_category, target_category)
                    if category not in known_categories
                ]
                note = f" (never detected: {', '.join(missing)})" if missing else " (detected, but not together in one frame)"
                lines.append(f"  {source_category} --{relation}--> {target_category}{note}")
            lines.append("")

            with open(path, "a", encoding="utf-8") as file:
                file.write("\n".join(lines) + "\n")
        except Exception as error:  # pragma: no cover - debug output must never crash reasoning
            print(f"[spatial_relation_reasoner] failed to write relation-skip debug note: {error}")

    @staticmethod
    def _save_debug_table(
        task: dict,
        candidates: list[dict],
        records: dict[int, dict],
        accepted: list[dict],
        method: str,
    ) -> None:
        if not config.SAVE_DEBUG_IMAGES:
            return
        try:
            accepted_by_key: dict[tuple, dict] = {
                (edge["source_object_id"], tuple(edge["target_object_ids"]), edge["relation"]): edge
                for edge in accepted
            }

            rows = []
            for candidate in candidates:
                source = records.get(candidate["source_object_id"])
                if source is None:
                    continue
                obj1 = f"{source['category']}#{source['object_id']}"
                relation = candidate["relation"]
                key = (candidate["source_object_id"], tuple(candidate["target_object_ids"]), relation)
                edge = accepted_by_key.get(key)
                verdict = "TRUE" if edge is not None else "false"
                reason = edge.get("reason", "") if edge is not None else ""
                for target_id in candidate["target_object_ids"]:
                    target = records.get(target_id)
                    if target is None:
                        continue
                    obj2 = f"{target['category']}#{target['object_id']}"
                    rows.append((obj1, obj2, relation, verdict, method, reason))

            if not rows:
                return

            os.makedirs(config.DEBUG_DIR, exist_ok=True)
            path = os.path.join(config.DEBUG_DIR, "sysnav_relation_check.txt")
            lines = [
                f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} | question: {task.get('raw', '')} ===",
                f"{'obj1':<22}{'obj2':<22}{'relation(sentence)':<20}{'verdict':<8}{'method':<10}reason",
            ]
            for obj1, obj2, rel, verdict, method_used, reason in rows:
                lines.append(f"{obj1:<22}{obj2:<22}{rel:<20}{verdict:<8}{method_used:<10}{reason}")
            lines.append("")

            with open(path, "a", encoding="utf-8") as file:
                file.write("\n".join(lines) + "\n")
        except Exception as error:  # pragma: no cover - debug output must never crash reasoning
            print(f"[spatial_relation_reasoner] failed to write relation debug table: {error}")

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
    def _records_of_category(task: dict, records: dict[int, dict], category: str) -> list[dict]:
        """해당 카테고리의 record 중, 참조 속성 제약("the **black** chair")을 통과한
        것만 남긴다.

        task["reference_allowed_ids"]는 sysnav_node.selection_job이
        reasoning/attribute_filter.reference_allowed_ids()로 채워 넣는다. 키가 없는
        카테고리는 "속성 제약 없음"이라 전부 통과 - 이 필드가 아예 없는 task(mission3의
        placeholder, 손으로 만든 테스트 task)도 예전과 똑같이 동작한다."""
        matched = [record for record in records.values() if record["category"] == category]
        allowed = (task.get("reference_allowed_ids") or {}).get(category)
        if allowed is None:
            return matched
        return [record for record in matched if int(record["object_id"]) in allowed]

    @staticmethod
    def _candidate_relations(task: dict, records: dict[int, dict]) -> list[dict]:
        """문장이 "A relation1 B relation2 C"처럼 relation을 연쇄로 담고 있으면
        (effective_relation_chain), 매 hop(source_category, relation, target_category)
        마다 실제 매칭되는 object pair를 전부 후보로 만든다 - 문장에 등장한 물체
        전부가 검증 대상이 되도록 한다 (예전엔 relation != "between"일 때
        reference_objects[0]만 쓰고 나머지 reference는 조용히 버렸음)."""
        chain = effective_relation_chain(task)
        if not chain:
            return []

        if len(chain) == 1 and chain[0][1] == "between":
            source_category = chain[0][0]
            reference_categories = [str(value).lower() for value in task.get("reference_objects", [])]
            if len(reference_categories) < 2:
                return []
            sources = [record for record in records.values() if record["category"] == source_category]
            first_refs = SpatialRelationReasoner._records_of_category(
                task, records, reference_categories[0]
            )
            second_refs = SpatialRelationReasoner._records_of_category(
                task, records, reference_categories[1]
            )
            output = []
            for source, first_ref, second_ref in product(sources, first_refs, second_refs):
                ids = {source["object_id"], first_ref["object_id"], second_ref["object_id"]}
                if len(ids) != 3:
                    continue
                output.append({
                    "source_object_id": source["object_id"],
                    "target_object_ids": [first_ref["object_id"], second_ref["object_id"]],
                    "relation": "between",
                })
            return output

        output = []
        for source_category, relation, target_category in chain:
            sources = [record for record in records.values() if record["category"] == source_category]
            targets = SpatialRelationReasoner._records_of_category(task, records, target_category)
            for source, target in product(sources, targets):
                if source["object_id"] == target["object_id"]:
                    continue
                output.append({
                    "source_object_id": source["object_id"],
                    "target_object_ids": [target["object_id"]],
                    "relation": relation,
                })
        return output

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

    @staticmethod
    def _save_debug_image(annotated: np.ndarray, relation: str) -> None:
        if not config.SAVE_DEBUG_IMAGES:
            return
        try:
            os.makedirs(config.DEBUG_DIR, exist_ok=True)
            filename = f"sysnav_spatial_relation_{relation}_{time.time():.3f}.jpg"
            cv2.imwrite(
                os.path.join(config.DEBUG_DIR, filename),
                cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
            )
        except Exception as error:  # pragma: no cover - debug output must never crash reasoning
            print(f"[spatial_relation_reasoner] failed to save debug image: {error}")

    def _infer_with_gemini(
        self,
        question: str,
        image_rgb: np.ndarray,
        candidates: list[dict],
        records: dict[int, dict],
    ) -> list[dict]:
        self._load_client()
        from google.genai import types

        annotated = self._annotate_image(image_rgb, records)
        # 문장이 relation을 연쇄로 담고 있으면(예: "A closest to B near C") candidate마다
        # relation이 다를 수 있어서, 파일명 태그는 candidate들에 등장하는 모든 relation을 합쳐 만든다.
        relation_tag = "-".join(sorted({str(candidate["relation"]) for candidate in candidates})) or "relation"
        self._save_debug_image(annotated, relation_tag)
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
You validate on-demand Object-Object edges for a mobile robot scene graph.
Instruction: {question}
Visible objects: {json.dumps(object_summary, ensure_ascii=False)}
Candidate checks (each has its own normalized relation): {json.dumps(candidates, ensure_ascii=False)}

Relation glossary (physical contact relations are strict):
  - "on"       : the SOURCE object physically rests on the TARGET's top surface.
  - "supports" : the reverse - the TARGET object physically rests on the SOURCE's
                 top surface ("the table WITH a vase ON it" -> table supports vase).
                 Being visible in the same picture, or on a nearby shelf, is NOT
                 enough - the object must be sitting on that surface, touching it.
  - "above"/"under" : vertically stacked but not necessarily touching.

The image is annotated with exact object IDs. For every candidate check, decide whether
its own requested relation is visibly true between its source and target object(s). Return
only candidates that are true. Preserve the provided source_object_id, target_object_ids,
and normalized relation exactly. Do not add new object IDs or relations. A relation that is
ambiguous must be omitted.
""".strip()

        with activity.operation(LLM, "Gemini 공간관계 판정"):
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
            self._trace(question, annotated, candidates, records, [])
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
        self._trace(question, annotated, candidates, records, output)
        return output

    @staticmethod
    def _trace(
        question: str,
        annotated: np.ndarray,
        candidates: list[dict],
        records: dict[int, dict],
        accepted: list[dict],
    ) -> None:
        """모델이 본 주석 이미지 그대로와 candidate별 판정을 대시보드용으로 남긴다.
        sysnav_relation_check.txt와 같은 내용이지만, 사진과 같은 카드에 붙는다."""
        accepted_by_key = {
            (edge["source_object_id"], tuple(edge["target_object_ids"]), edge["relation"]): edge
            for edge in accepted
        }

        def name(object_id: int) -> str:
            record = records.get(int(object_id))
            return f"{record['category']}#{object_id}" if record else f"#{object_id}"

        verdicts = []
        for candidate in candidates:
            key = (
                candidate["source_object_id"],
                tuple(candidate["target_object_ids"]),
                candidate["relation"],
            )
            edge = accepted_by_key.get(key)
            targets = ", ".join(name(value) for value in candidate["target_object_ids"])
            verdicts.append((
                f"{name(candidate['source_object_id'])} {candidate['relation']} {targets}",
                edge is not None,
                str(edge.get("reason", "")) if edge else "",
            ))
        llm_trace.record(
            kind="공간관계 판정",
            question=question,
            images=[("주석 파노라마 (모델이 본 그대로)", annotated)],
            verdicts=verdicts,
            summary=f"참으로 판정된 관계 {len(accepted)} / 후보 {len(candidates)}",
        )

    # 이미지만으로는 판정이 구조적으로 불가능한 관계들. 360도 파노라마는 원근이
    # 심하게 왜곡돼서, 몇 미터 떨어진 선반 위 물건과 바닥의 스툴이 "위에 놓인" 것처럼
    # 보인다(2026-08-24 실측). 반면 "윗면에 얹혀 있다"는 3D bbox로 확실히 반증할 수
    # 있으므로, 이 관계에 한해 기하 판정에 거부권을 준다. 다른 관계(near/left_of 등)는
    # 기존대로 Gemini 판단을 그대로 쓴다 - 거기서는 이미지가 더 정확하다.
    _CONTACT_RELATIONS = ("on", "supports")

    def _veto_impossible_contacts(
        self,
        edges: list[dict],
        records: dict[int, dict],
        viewpoint_pose: dict,
    ) -> list[dict]:
        """Gemini가 통과시킨 edge 중 접촉 관계는 3D bbox로 반증되면 버린다."""
        kept = []
        for edge in edges:
            if edge.get("relation") not in self._CONTACT_RELATIONS:
                kept.append(edge)
                continue
            source = records.get(edge["source_object_id"])
            targets = [records[i] for i in edge["target_object_ids"] if i in records]
            if source is None or not targets:
                kept.append(edge)
                continue
            holds, _, reason = self._geometry_check(
                edge["relation"], source, targets, viewpoint_pose
            )
            if holds:
                kept.append(edge)
            else:
                print(
                    f"[spatial_relation_reasoner] 관계 거부(기하 반증): "
                    f"{edge['relation']} {edge['source_object_id']}->"
                    f"{edge['target_object_ids']} - {reason}"
                )
        return kept

    def _infer_with_geometry(
        self,
        candidates: list[dict],
        records: dict[int, dict],
        viewpoint_pose: dict,
    ) -> list[dict]:
        # "nearest/closest"는 경쟁하는 후보 중 argmin 하나만 참이 되는 최상급 relation이라
        # 나머지(pairwise 판정) relation과 채점 방식이 다르다. 체인이면 candidate마다 relation이
        # 다를 수 있으므로(예: "A closest to B near C") relation별로 나눠서 처리한다.
        superlatives = {"nearest": min, "closest": min, "farthest": max, "furthest": max}
        nearest_candidates = [c for c in candidates if c["relation"] in superlatives]
        other_candidates = [c for c in candidates if c["relation"] not in superlatives]

        output = []
        for relation, pick in superlatives.items():
            group = [c for c in nearest_candidates if c["relation"] == relation]
            if group:
                output.extend(self._infer_superlative_with_geometry(group, records, pick))
        for candidate in other_candidates:
            source = records[candidate["source_object_id"]]
            targets = [records[object_id] for object_id in candidate["target_object_ids"]]
            holds, confidence, reason = self._geometry_check(
                candidate["relation"],
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
    def _infer_superlative_with_geometry(
        candidates: list[dict], records: dict[int, dict], pick=min
    ) -> list[dict]:
        """"nearest/closest"는 근접 threshold(near)와 달리 최상급(argmin) relation이다:
        같은 hop(예: "bedside table nearest window")에서 경쟁하는 모든 (source, target)
        후보 쌍 중 거리가 가장 짧은 단 하나만 참이 된다. hop 식별은 (source_category,
        target_category) 조합으로 한다 - target object_id로 묶으면 reference 카테고리
        인스턴스가 2개 이상(예: 창문이 2개) 있을 때 각각 별도 그룹이 되어 버려서
        argmin이 무력화되는 버그가 있었다(각 그룹이 원소 1개짜리라 전부 "승자"가 됨)."""
        groups: dict[tuple[str, str], list[tuple[dict, float]]] = {}
        for candidate in candidates:
            source = records[candidate["source_object_id"]]
            target = records[candidate["target_object_ids"][0]]
            distance = float(np.linalg.norm(
                np.asarray(source["position"][:2], dtype=np.float64)
                - np.asarray(target["position"][:2], dtype=np.float64)
            ))
            groups.setdefault((source["category"], target["category"]), []).append((candidate, distance))

        output = []
        for scored in groups.values():
            if not scored:
                continue
            winner, winner_distance = pick(scored, key=lambda item: item[1])
            output.append({
                **winner,
                "confidence": 1.0,
                "method": "geometry",
                "reason": f"xy_distance={winner_distance:.3f}m "
                          f"({'min' if pick is min else 'max'} among {len(scored)} candidate(s))",
            })
        return output

    @staticmethod
    def _xy_gap(
        source_min: np.ndarray, source_max: np.ndarray,
        target_min: np.ndarray, target_max: np.ndarray,
    ) -> float:
        """두 XY bbox가 떨어진 거리. 겹치면 0.

        중심 사이 거리를 안 쓰는 이유: 캐비닛(폭 1.75m) 위의 그림(폭 0.90m)처럼 크기가
        많이 다르면 중심이 자연스럽게 어긋난다. 겹침을 봐야 크기 차이에 안 휘둘린다.
        """
        dx = max(0.0, float(target_min[0] - source_max[0]), float(source_min[0] - target_max[0]))
        dy = max(0.0, float(target_min[1] - source_max[1]), float(source_min[1] - target_max[1]))
        return math.hypot(dx, dy)

    @classmethod
    def _vertical_relation(
        cls, gap: float,
        source_min: np.ndarray, source_max: np.ndarray,
        target_min: np.ndarray, target_max: np.ndarray,
    ) -> tuple[bool, float, str]:
        """"A under B" / "A above B" 판정. 높이 순서 + 수평 정렬 + 높이차 상한.

        수평 정렬을 반드시 같이 봐야 한다 (2026-08-23 GT 대조로 발견): 예전에는 높이차만
        봐서 방 반대편 물체끼리 성립했다 -
          cabinet#1(2.19,-0.79) --under--> picture#8(-2.62,-6.55) conf=0.937, 수평 7.51m
        이 엉터리 edge가 정답 체인을 밀어내고 틀린 화병을 가리키게 만들었다.

        신뢰도를 "높이차가 작을수록 높게"에서 바꾼 이유도 같다. 그림은 캐비닛에 붙어
        있지 않고 0.5~1.2m 위에 걸린다. 붙어 있을수록 높다고 점수를 주면 정상 쌍
        (실측 0.604m -> 0.396)이 임계값 0.55에 못 미쳐 탈락하고, 우연히 높이가 맞은
        방 반대편 쌍이 이긴다. 그래서 수평 정렬을 주 신호로 쓰고 높이차는 상한만 본다.
        """
        margin = config.SCENE_GRAPH_VERTICAL_HORIZONTAL_MARGIN_M
        max_gap = config.SCENE_GRAPH_VERTICAL_MAX_GAP_M
        xy_gap = cls._xy_gap(source_min, source_max, target_min, target_max)
        ordered = gap >= -config.SCENE_GRAPH_ON_VERTICAL_TOLERANCE_M
        holds = ordered and xy_gap <= margin and gap <= max_gap
        # 수평 정렬이 주 신호다. 높이차는 위 gate(max_gap)에서 이미 걸렀으므로 여기서는
        # 동점을 가르는 정도로만 반영한다(최대 30% 감점) - 그림은 캐비닛 바로 위에
        # 붙지 않고 0.5~1.2m 떠 있는 게 정상이라, 이걸 크게 깎으면 정상 쌍이 임계값
        # 아래로 떨어진다(실측: 곱셈으로 하면 정답 쌍이 0.501 < 0.55로 탈락했다).
        horizontal_score = max(0.0, 1.0 - xy_gap / max(margin, 1e-6))
        vertical_score = max(0.0, 1.0 - max(0.0, gap) / max(max_gap, 1e-6))
        confidence = horizontal_score * (0.7 + 0.3 * vertical_score) if holds else 0.0
        return holds, confidence, (
            f"vertical_gap={gap:.3f}m, xy_gap={xy_gap:.3f}m (max {margin:.2f}m)"
        )

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

        if relation in ("above", "under"):
            if relation == "above":
                gap = float(source_min[2] - target_max[2])
            else:
                gap = float(target_min[2] - source_max[2])
            return self._vertical_relation(
                gap, source_min, source_max, target_min, target_max
            )
        if relation == "supports":
            # "the table WITH a vase ON it" - source(table) 위에 target(vase)이 **놓여
            # 있다**. under/above와 반드시 구분해야 한다: 그쪽은 "캐비닛 위 0.5~1.2m에
            # 걸린 그림"을 위한 판정이라 높이차 2.0m/수평 0.60m까지 허용하는데,
            # 그 느슨함으로 "on"을 판정하면 선반 위 화병과 3m 떨어진 스툴이 짝지어진다
            # (2026-08-24 실측: vase(2.19,-0.89,1.24) vs table(1.71,1.93,0.24), 수평 2.86m).
            # 접촉 관계는 "윗면에 얹혀 있다"가 물리적 사실이므로 on과 같은 기준을 쓴다.
            vertical_gap = float(target_min[2] - source_max[2])
            horizontal_inside = (
                source_min[0] - config.SCENE_GRAPH_ON_HORIZONTAL_MARGIN_M
                <= float(target_position[0])
                <= source_max[0] + config.SCENE_GRAPH_ON_HORIZONTAL_MARGIN_M
                and source_min[1] - config.SCENE_GRAPH_ON_HORIZONTAL_MARGIN_M
                <= float(target_position[1])
                <= source_max[1] + config.SCENE_GRAPH_ON_HORIZONTAL_MARGIN_M
            )
            holds = abs(vertical_gap) <= config.SCENE_GRAPH_ON_VERTICAL_TOLERANCE_M and horizontal_inside
            confidence = max(0.0, 1.0 - abs(vertical_gap) / max(config.SCENE_GRAPH_ON_VERTICAL_TOLERANCE_M, 1e-6))
            return holds, confidence, f"vertical_gap={vertical_gap:.3f}m, horizontal_inside={horizontal_inside}"

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
