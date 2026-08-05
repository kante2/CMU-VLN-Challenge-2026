"""Infer/verify object self-attributes via VLM (SysNav paper Sec. IV-A-1, Object Node).

논문 원문: "object self-attributes are inferred on demand rather than predefined.
When a task specifies self-attribute constraints φ for category c_i, we retrieve the
corresponding nodes {v_j^o | c(v_j^o) = c_i} and use their RGB images to prompt the
VLM for attribute inference, appending the results φ to each node. This design
efficiently satisfies universal self-attribute constraints while avoiding the
redundancy."

즉 판단 기준은 "후보가 몇 개인지"가 아니라 "이번 task가 속성 제약(예: color=black)을
요구하는지"다 - 후보가 1개뿐이어도 속성 제약이 있으면 반드시 검증한다. 한 번 추론된
속성은 object_memory 노드에 캐싱되어(update_self_attributes) 다시 안 물어본다.

DetectionVerifier와 달리 이 모듈은 실패 시 fail-open(그냥 통과)하지 않는다 - "속성을
검증 안 했는데 확정해버리는" 게 바로 지금 고치려는 버그이므로, 검증이 안 되면
"아직 확인 안 됨"(불통과)으로 취급해서 호출 쪽이 확정을 미루게 한다.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
from rclpy.logging import get_logger

from sysnav import config


class AttributeVerifier:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._client = None
        self._logger = get_logger("sysnav_attribute_verifier")

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
            ".jpg",
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        if not ok:
            raise RuntimeError("Attribute-verification JPEG encoding failed")
        return encoded.tobytes()

    def verify(self, candidates: list[dict], attributes: list[str]) -> dict[int, dict[str, bool]]:
        """candidates: object_memory 노드 리스트(각각 object_id, self_attributes,
        representative_image 필드를 가짐). attributes: 이번 task가 요구하는 속성들
        (예: ["black"]).

        반환: {object_id: {attribute: bool, ...}} - 캐시에 이미 있던 속성은 그대로
        재사용하고, 없던 것만 VLM으로 새로 추론해서 채운다. 캐싱 자체(object_memory에
        저장)는 호출 쪽 책임이다(이 클래스는 object_memory를 모른다)."""
        cleaned_attributes = [str(value).strip().lower() for value in attributes if str(value).strip()]
        if not candidates or not cleaned_attributes:
            return {int(c["object_id"]): {} for c in candidates}

        results: dict[int, dict[str, bool]] = {}
        pending: list[dict] = []
        for candidate in candidates:
            object_id = int(candidate["object_id"])
            cached = candidate.get("self_attributes", {}) or {}
            results[object_id] = {attr: cached[attr] for attr in cleaned_attributes if attr in cached}
            missing = [attr for attr in cleaned_attributes if attr not in cached]
            if missing:
                pending.append({"candidate": candidate, "missing": missing})

        if not pending:
            return results

        inferred = self._infer(pending)
        for object_id, attribute_results in inferred.items():
            results.setdefault(object_id, {}).update(attribute_results)
        # VLM 호출이 아예 실패해서 추론을 못 받은 candidate는 "아직 확인 안 됨" 상태로
        # 남긴다(불통과로 취급되도록 결과 dict에 아예 안 넣음) - fail-open 안 함.
        return results

    def _infer(self, pending: list[dict]) -> dict[int, dict[str, bool]]:
        # on/off 스위치(ATTRIBUTE_VERIFICATION_ENABLED)는 호출 쪽(selection_job)이
        # verify() 자체를 부를지 말지로 처리한다 - 여기서 또 걸면, "기능 꺼짐"과
        # "VLM 호출 실패"가 똑같이 "불통과"로 섞여버려서 의미가 달라진다(꺼짐은 예전
        # 동작인 무필터링으로 돌아가야 하는데, 여기서 막으면 그것도 "확인 안 됨"이
        # 돼버림).
        try:
            self._load()
            from google.genai import types

            contents: list[object] = [
                "You infer physical self-attributes (e.g. color, material, size, state) of "
                "candidate objects for a mobile robot, purely from each candidate's own "
                "representative image. For every (object_id, attribute) pair below, decide "
                "whether that attribute is visibly true of that specific object. Judge each "
                "object only from its own image, not the others.\n"
                + json.dumps(
                    [
                        {"object_id": int(item["candidate"]["object_id"]), "attributes_to_check": item["missing"]}
                        for item in pending
                    ],
                    ensure_ascii=False,
                )
            ]
            for item in pending:
                candidate = item["candidate"]
                image = candidate.get("representative_image")
                contents.append(f"object_id={int(candidate['object_id'])} image:")
                if isinstance(image, np.ndarray) and image.size:
                    contents.append(types.Part.from_bytes(data=self._jpeg(image), mime_type="image/jpeg"))
                else:
                    contents.append("(no image available for this object)")

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
                                        "attribute": {"type": "string"},
                                        "value": {"type": "boolean"},
                                    },
                                    "required": ["object_id", "attribute", "value"],
                                },
                            }
                        },
                        "required": ["results"],
                    },
                ),
            )
            if not response.text:
                raise RuntimeError("Gemini가 빈 응답을 반환함")

            allowed = {
                (int(item["candidate"]["object_id"]), attr)
                for item in pending
                for attr in item["missing"]
            }
            output: dict[int, dict[str, bool]] = {}
            for entry in json.loads(response.text).get("results", []):
                key = (int(entry["object_id"]), str(entry["attribute"]).strip().lower())
                if key not in allowed:
                    continue
                output.setdefault(key[0], {})[key[1]] = bool(entry["value"])
            self._logger.info(f"Attribute inference: {output}")
            return output
        except Exception as error:
            self._logger.warning(f"Attribute inference skipped (unverified, not fail-open): {error}")
            return {}
