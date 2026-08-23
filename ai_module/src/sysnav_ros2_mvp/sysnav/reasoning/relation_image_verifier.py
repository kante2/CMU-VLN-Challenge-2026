"""후보 자신의 이미지만으로 관계(relation) 제약을 VLM에게 직접 확인받는 폴백.

geometric/co-observation 기반 관계 검증(scene_graph_manager.find_matching_target_ids,
reasoning/spatial_relation_reasoner.py)은 참조 물체(reference, 예: "window")가
object_memory에 3D 위치로 존재해야 성립한다. 근데 유리창처럼 LiDAR 반사가 아예 없어서
(0 point) approximate grounding조차 못 만드는 물체는 그 경로로는 영원히 검증이
불가능하다.

이 모듈은 참조 물체를 3D로 grounding할 필요 없이, 후보(예: bedside table)의 대표
이미지 자체를 보고 "이 사진에 [reference]가 [relation]하게 보이는가?"를 VLM에게 직접
물어서 판정한다 - object self-attribute 추론(attribute_verifier.py)과 정확히 같은
패턴을, 속성이 아니라 관계형 predicate로 확장한 버전이다.

attribute_verifier.py와 같은 이유로 fail-closed: 실패하면 통과시키지 않는다("확인
안 했는데 확정"이 애초에 고치려던 문제이므로).
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
from rclpy.logging import get_logger

from sysnav import config
from sysnav.activity_log import LLM, activity


class RelationImageVerifier:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._client = None
        self._logger = get_logger("sysnav_relation_image_verifier")

    def _load(self) -> None:
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai
        self._client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _jpeg(image_rgb: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        if not ok:
            raise RuntimeError("Relation-image verification JPEG encoding failed")
        return encoded.tobytes()

    def verify(self, candidates: list[dict], relation: str, reference_category: str) -> set[int]:
        """candidates: object_memory 노드 리스트(representative_image 포함).
        relation/reference_category(예: "nearest"/"window")가 각 후보 자신의 대표
        이미지에서 시각적으로 참인지 VLM에게 직접 확인받는다 - reference 물체를 3D로
        grounding할 필요가 아예 없다. 반환: 통과한 object_id 집합(실패하면 빈 set).

        반드시 context_image를 써야 한다 - representative_image(attribute_verifier가
        쓰는 것)는 배경을 회색으로 지운 물체 단독 사진이라 애초에 참조 물체가 그
        사진 안에 나타날 수가 없다(항상 확인 불가로 실패하게 됨). context_image는
        같은 detection에서 배경을 안 지우고 여유를 두고 자른 사진이라 주변 맥락이
        보인다."""
        usable = [
            candidate for candidate in candidates
            if isinstance(candidate.get("context_image"), np.ndarray)
            and candidate["context_image"].size
        ]
        if not usable:
            return set()
        try:
            self._load()
            from google.genai import types

            relation_phrase = str(relation).replace("_", " ")
            contents: list[object] = [
                f"For each image below, decide whether a {reference_category} is visibly "
                f"{relation_phrase} the object shown in that same image (the object may be "
                "partially cropped/centered in the frame; the reference, if present, would "
                "appear elsewhere in the same photo). Judge each image independently, purely "
                "from what's visible in it - do not guess if it's not visible.\n"
                + json.dumps([{"object_id": int(candidate["object_id"])} for candidate in usable], ensure_ascii=False)
            ]
            for candidate in usable:
                contents.append(f"object_id={int(candidate['object_id'])} image:")
                contents.append(
                    types.Part.from_bytes(
                        data=self._jpeg(candidate["context_image"]), mime_type="image/jpeg"
                    )
                )

            with activity.operation(LLM, "Gemini 관계 이미지 검증"):
                response = self._client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema={
                            "type": "object",
                            "properties": {
                                "results": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "object_id": {"type": "integer"},
                                            "holds": {"type": "boolean"},
                                        },
                                        "required": ["object_id", "holds"],
                                    },
                                }
                            },
                            "required": ["results"],
                        },
                    ),
                )
            if not response.text:
                raise RuntimeError("Gemini가 빈 응답을 반환함")
            allowed = {int(candidate["object_id"]) for candidate in usable}
            verified = {
                int(entry["object_id"])
                for entry in json.loads(response.text).get("results", [])
                if int(entry["object_id"]) in allowed and bool(entry["holds"])
            }
            self._logger.info(f"Relation image verification ({relation} {reference_category}): {verified}")
            return verified
        except Exception as error:
            self._logger.warning(f"Relation image verification skipped (unverified, not fail-open): {error}")
            return set()

    def rank_superlative(
        self, candidates: list[dict], reference_category: str, relation: str = "nearest"
    ) -> int | None:
        """"nearest"/"closest"는 최상급(비교) relation이라 verify()처럼 후보마다
        독립적으로 yes/no만 물어보면 안 된다 - 예를 들어 bedside table이 2개 있고
        둘 다 사진에 창문이 보이면 둘 다 "yes"가 나와서 어느 게 진짜 더 가까운지
        구분이 안 된다. 이 메서드는 후보 전부를 한 번에 보여주고 VLM에게 직접
        비교시켜서 가장 가까운 후보 하나만 고른다. reference_category가 참조 물체를
        3D로 grounding 못 해서(0 point) 거리 계산 자체가 불가능할 때(즉 verify()와
        같은 상황)만 쓴다. 반환: 승자 object_id, 실패/불확실하면 None."""
        usable = [
            candidate for candidate in candidates
            if isinstance(candidate.get("context_image"), np.ndarray)
            and candidate["context_image"].size
        ]
        if len(usable) < 2:
            return None
        try:
            self._load()
            from google.genai import types

            farthest = relation in ("farthest", "furthest")
            wording = "farthest from" if farthest else "closest to"
            contents: list[object] = [
                "Each image below shows a different candidate object and its surrounding "
                f"context. Decide which single candidate is {wording} a {reference_category} "
                f"visible in its own photo. If a candidate's photo doesn't show a "
                f"{reference_category} at all, it cannot be the answer. If none of the "
                f"candidates show a {reference_category}, set reference_visible_in_any to "
                "false.\n"
                + json.dumps([{"object_id": int(candidate["object_id"])} for candidate in usable], ensure_ascii=False)
            ]
            for candidate in usable:
                contents.append(f"object_id={int(candidate['object_id'])} image:")
                contents.append(
                    types.Part.from_bytes(
                        data=self._jpeg(candidate["context_image"]), mime_type="image/jpeg"
                    )
                )

            with activity.operation(LLM, "Gemini 관계 이미지 검증"):
                response = self._client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema={
                            "type": "object",
                            "properties": {
                                "object_id": {"type": "integer"},
                                "reference_visible_in_any": {"type": "boolean"},
                            },
                            "required": ["object_id", "reference_visible_in_any"],
                        },
                    ),
                )
            if not response.text:
                raise RuntimeError("Gemini가 빈 응답을 반환함")
            result = json.loads(response.text)
            if not result.get("reference_visible_in_any"):
                self._logger.info(
                    f"Relation image nearest-ranking ({reference_category}): "
                    "reference not visible in any candidate"
                )
                return None
            allowed = {int(candidate["object_id"]) for candidate in usable}
            winner = int(result["object_id"])
            if winner not in allowed:
                raise RuntimeError(f"Gemini가 후보 밖의 object_id를 반환함: {winner}")
            self._logger.info(f"Relation image nearest-ranking ({reference_category}): winner={winner}")
            return winner
        except Exception as error:
            self._logger.warning(f"Relation image nearest-ranking skipped (not fail-open): {error}")
            return None
