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

캐싱도 attribute_verifier와 같은 패턴이다: 판정 결과는 object_memory 노드의
`relation_checks`에 적립되고(저장은 호출 쪽 책임 - 이 클래스는 object_memory를 모른다),
캐시 키에 노드의 `image_version`이 들어가서 **사진이 바뀔 때만** 다시 묻는다.

왜 필요했나: 이 폴백은 참조 물체가 끝내 grounding 안 되는 경우(유리창 등)를 위한
것인데, 그런 경우 selection_job이 relation_pending을 반환 -> PLAN_EXPLORATION ->
OBSERVE -> 다시 SELECT_TARGET으로 되돌아온다. 캐시가 없으면 같은 후보의 같은 사진을
같은 질문으로 이 사이클마다 계속 Gemini에 올린다(mission3는 step마다 새로 시작해서
더 심하다). 사진이 그대로면 답도 같을 수밖에 없으므로 순수 낭비다.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
from rclpy.logging import get_logger

from sysnav import config
from sysnav.activity_log import LLM, activity
from sysnav.llm_trace import llm_trace


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

    @staticmethod
    def _usable(candidates: list[dict]) -> list[dict]:
        """context_image가 실제로 있는 후보만. representative_image는 배경을 지운
        물체 단독 사진이라 참조 물체가 애초에 안 찍혀서 못 쓴다."""
        return [
            candidate for candidate in candidates
            if isinstance(candidate.get("context_image"), np.ndarray)
            and candidate["context_image"].size
        ]

    @staticmethod
    def _traced_images(usable: list[dict]) -> list[tuple[str, np.ndarray]]:
        """모델에 올린 그 context_image들을 대시보드용 캡션과 함께 돌려준다."""
        return [
            (
                f"{candidate.get('category', '?')}#{int(candidate['object_id'])} context",
                candidate["context_image"],
            )
            for candidate in usable
        ]

    @staticmethod
    def _version(candidate: dict) -> int:
        return int(candidate.get("image_version", 0))

    @classmethod
    def _verify_key(cls, candidate: dict, relation: str, reference_category: str) -> str:
        """후보 하나짜리 판정의 캐시 키. 그 후보의 사진이 바뀌면 키가 바뀐다."""
        return f"verify|{relation}|{reference_category}|v{cls._version(candidate)}"

    @classmethod
    def _rank_key(cls, usable: list[dict], relation: str, reference_category: str) -> str:
        """최상급 비교는 후보 **집합 전체**를 놓고 내린 판정이라, 집합이나 그중 한 장의
        사진이 바뀌면 결과가 달라질 수 있다. 그래서 참가자 전원의 (id, 사진버전)을
        키에 넣고, 참가한 모든 노드에 같은 키로 적립한다."""
        signature = ",".join(
            f"{int(candidate['object_id'])}:{cls._version(candidate)}"
            for candidate in sorted(usable, key=lambda item: int(item["object_id"]))
        )
        return f"rank|{relation}|{reference_category}|{signature}"

    @staticmethod
    def _cached(candidate: dict, key: str) -> bool | None:
        return (candidate.get("relation_checks") or {}).get(key)

    def verify(
        self, candidates: list[dict], relation: str, reference_category: str
    ) -> dict[int, dict[str, bool]]:
        """candidates: object_memory 노드 리스트(context_image 포함).
        relation/reference_category(예: "nearest"/"window")가 각 후보 자신의 사진에서
        시각적으로 참인지 VLM에게 직접 확인받는다 - reference 물체를 3D로 grounding할
        필요가 아예 없다.

        반환: `{object_id: {cache_key: bool}}` - attribute_verifier.verify()와 같은
        모양이다. 캐시에 이미 있던 판정은 VLM을 안 거치고 그대로 되돌려주고, 없던
        후보만 새로 묻는다. 호출 쪽은 값이 True인 후보를 통과시키고, 반환된 dict를
        object_memory.update_relation_checks()로 적립한다. VLM 호출이 통째로 실패하면
        새로 물어본 후보는 아예 빠진 채 돌아온다(fail-closed - 다음 기회에 재시도).

        반드시 context_image를 써야 한다 - representative_image(attribute_verifier가
        쓰는 것)는 배경을 회색으로 지운 물체 단독 사진이라 애초에 참조 물체가 그
        사진 안에 나타날 수가 없다(항상 확인 불가로 실패하게 됨). context_image는
        같은 detection에서 배경을 안 지우고 여유를 두고 자른 사진이라 주변 맥락이
        보인다."""
        results: dict[int, dict[str, bool]] = {}
        pending: list[dict] = []
        for candidate in self._usable(candidates):
            key = self._verify_key(candidate, relation, reference_category)
            cached = self._cached(candidate, key)
            if cached is None:
                pending.append(candidate)
            else:
                results[int(candidate["object_id"])] = {key: bool(cached)}
        if not pending:
            if results:
                self._logger.info(
                    f"Relation image verification ({relation} {reference_category}): "
                    f"all {len(results)} candidate(s) served from cache"
                )
            return results

        cached_count = len(results)
        usable = pending
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
            holds = {
                int(entry["object_id"]): bool(entry["holds"])
                for entry in json.loads(response.text).get("results", [])
                if int(entry["object_id"]) in allowed
            }
            for candidate in usable:
                object_id = int(candidate["object_id"])
                if object_id not in holds:
                    # 응답에 빠진 후보는 "확인 안 됨"으로 남긴다 - 캐시에도 안 넣어서
                    # 다음 기회에 다시 물어본다.
                    continue
                key = self._verify_key(candidate, relation, reference_category)
                results[object_id] = {key: holds[object_id]}
            llm_trace.record(
                kind="관계 이미지 검증",
                question=f"각 사진에 {reference_category}이(가) {relation_phrase} 보이는가",
                images=self._traced_images(usable),
                verdicts=[
                    (
                        f"{candidate.get('category', '?')}#{int(candidate['object_id'])}",
                        holds.get(int(candidate["object_id"])),
                        "",
                    )
                    for candidate in usable
                ],
                summary=(
                    f"통과 {sum(1 for value in holds.values() if value)} / "
                    f"질의 {len(usable)} (캐시 {cached_count})"
                ),
            )
            passed = sorted(object_id for object_id, value in holds.items() if value)
            self._logger.info(
                f"Relation image verification ({relation} {reference_category}): "
                f"passed={passed} (asked {len(usable)}, from cache {cached_count})"
            )
            return results
        except Exception as error:
            self._logger.warning(f"Relation image verification skipped (unverified, not fail-open): {error}")
            return results

    def rank_superlative(
        self, candidates: list[dict], reference_category: str, relation: str = "nearest"
    ) -> dict[int, dict[str, bool]]:
        """"nearest"/"closest"는 최상급(비교) relation이라 verify()처럼 후보마다
        독립적으로 yes/no만 물어보면 안 된다 - 예를 들어 bedside table이 2개 있고
        둘 다 사진에 창문이 보이면 둘 다 "yes"가 나와서 어느 게 진짜 더 가까운지
        구분이 안 된다. 이 메서드는 후보 전부를 한 번에 보여주고 VLM에게 직접
        비교시켜서 가장 가까운 후보 하나만 고른다. reference_category가 참조 물체를
        3D로 grounding 못 해서(0 point) 거리 계산 자체가 불가능할 때(즉 verify()와
        같은 상황)만 쓴다.

        반환: verify()와 같은 `{object_id: {cache_key: bool}}` - 승자만 True다.
        참가자 전원이 **같은 키**를 공유하므로, 다음 호출에서 후보 집합과 사진이
        그대로면 전원이 캐시에 걸려 VLM을 다시 안 부른다. "어느 후보 사진에도 참조
        물체가 안 보인다"는 결론도 전원 False로 캐싱한다 - 그것도 돈 주고 얻은
        판정이라 같은 사진으로 다시 물어볼 이유가 없다."""
        usable = self._usable(candidates)
        if len(usable) < 2:
            return {}
        key = self._rank_key(usable, relation, reference_category)
        cached = [self._cached(candidate, key) for candidate in usable]
        if all(value is not None for value in cached):
            self._logger.info(
                f"Relation image nearest-ranking ({reference_category}): served from cache"
            )
            return {
                int(candidate["object_id"]): {key: bool(value)}
                for candidate, value in zip(usable, cached)
            }
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
                llm_trace.record(
                    kind="관계 이미지 최상급 비교",
                    question=f"어느 후보가 {wording} a {reference_category}",
                    images=self._traced_images(usable),
                    verdicts=[
                        (f"{candidate.get('category', '?')}#{int(candidate['object_id'])}", False, "")
                        for candidate in usable
                    ],
                    summary=f"어느 후보 사진에도 {reference_category}이(가) 안 보임 - 전원 탈락",
                )
                self._logger.info(
                    f"Relation image nearest-ranking ({reference_category}): "
                    "reference not visible in any candidate"
                )
                return {int(candidate["object_id"]): {key: False} for candidate in usable}
            allowed = {int(candidate["object_id"]) for candidate in usable}
            winner = int(result["object_id"])
            if winner not in allowed:
                raise RuntimeError(f"Gemini가 후보 밖의 object_id를 반환함: {winner}")
            self._logger.info(f"Relation image nearest-ranking ({reference_category}): winner={winner}")
            llm_trace.record(
                kind="관계 이미지 최상급 비교",
                question=f"어느 후보가 {wording} a {reference_category}",
                images=self._traced_images(usable),
                verdicts=[
                    (
                        f"{candidate.get('category', '?')}#{int(candidate['object_id'])}",
                        int(candidate["object_id"]) == winner,
                        "",
                    )
                    for candidate in usable
                ],
                summary=f"승자 object_id={winner} (후보 {len(usable)})",
            )
            return {
                int(candidate["object_id"]): {key: int(candidate["object_id"]) == winner}
                for candidate in usable
            }
        except Exception as error:
            self._logger.warning(f"Relation image nearest-ranking skipped (not fail-open): {error}")
            return {}
