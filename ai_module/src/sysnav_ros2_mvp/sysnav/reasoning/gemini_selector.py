"""Gemini 2.5 Flash selector for choosing one target candidate."""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
from rclpy.logging import get_logger

from sysnav import config
from sysnav.activity_log import LLM, activity


class GeminiSelector:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._client = None
        self._logger = get_logger("sysnav_gemini_selector")
        # Mission 3의 동일 step은 unreachable/재탐사로 selection_job이 다시 호출될 수
        # 있다. 최종 검증은 step당 최대 한 번만 호출하고 그 결과(None 포함)를 재사용한다.
        self._mission3_verification_cache: dict[tuple, int | None] = {}

    def _load(self) -> None:
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Install google-genai: pip install google-genai") from exc
        self._client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=config.GEMINI_SELECTOR_TIMEOUT_MS),
        )

    @staticmethod
    def _fallback(candidates: list[dict], robot_pose: dict | None) -> int:
        if not candidates:
            raise ValueError("No candidates")
        if robot_pose is None:
            return int(max(candidates, key=lambda x: float(x.get("confidence", 0.0)))["object_id"])
        robot_xy = np.array([robot_pose["x"], robot_pose["y"]], dtype=np.float64)
        def score(item: dict) -> float:
            distance = float(np.linalg.norm(np.asarray(item["position"][:2]) - robot_xy))
            return float(item.get("confidence", 0.0)) - 0.02 * distance
        return int(max(candidates, key=score)["object_id"])

    @staticmethod
    def _jpeg(image: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return encoded.tobytes()

    @staticmethod
    def _summary(item: dict) -> dict:
        return {
            "object_id": int(item["object_id"]),
            "category": item["category"],
            "position_xyz": [round(float(v), 3) for v in item["position"]],
            "extent_xyz": [round(float(v), 3) for v in item.get("extent_3d", (0, 0, 0))],
            "confidence": round(float(item.get("confidence", 0.0)), 3),
            "observation_count": int(item.get("observation_count", 1)),
        }

    def select(
        self,
        question: str,
        candidates: list[dict],
        context_objects: list[dict] | None = None,
        robot_pose: dict | None = None,
        final_verification: bool = False,
        verification_key: tuple | None = None,
    ) -> int | None:
        if not candidates:
            raise ValueError("No target candidates")
        if final_verification and verification_key in self._mission3_verification_cache:
            return self._mission3_verification_cache[verification_key]
        if len(candidates) == 1 and not final_verification:
            return int(candidates[0]["object_id"])

        valid_ids = {int(item["object_id"]) for item in candidates}
        try:
            self._load()
            from google.genai import types
            verification_instruction = """
This is the final Mission 3 subgoal sanity check. First verify that each displayed
bounding-box crop really depicts the requested target category, rather than a visually
similar object. Then verify that the chosen candidate is consistent with spatial words
in the instruction (for example closest to, farthest from, between, or near). The supplied
3D positions are the authoritative source for metric distance; use images to validate
object identity and reference-object correspondence, not to invent distances. If no
candidate is visibly credible, set accepted=false.
""".strip() if final_verification else ""
            prompt = f"""
You are a target-object selector for a mobile robot.
User instruction: {question}
Target candidates: {json.dumps([self._summary(x) for x in candidates], ensure_ascii=False)}
Other scene objects: {json.dumps([self._summary(x) for x in (context_objects or []) if int(x['object_id']) not in valid_ids], ensure_ascii=False)}
Choose exactly one target candidate. Use visual attributes and supplied 3D context.
Return JSON only and never output an object_id outside the target candidate list.
{verification_instruction}
""".strip()
            contents: list[object] = [prompt]
            for item in candidates:
                contents.append(f"candidate object_id={int(item['object_id'])} isolated crop:")
                image = item.get("representative_image")
                if isinstance(image, np.ndarray) and image.size:
                    contents.append(types.Part.from_bytes(data=self._jpeg(image), mime_type="image/jpeg"))
                if final_verification:
                    context_image = item.get("context_image")
                    if isinstance(context_image, np.ndarray) and context_image.size:
                        contents.append(
                            f"candidate object_id={int(item['object_id'])} context crop:"
                        )
                        contents.append(
                            types.Part.from_bytes(
                                data=self._jpeg(context_image), mime_type="image/jpeg"
                            )
                        )
            with activity.operation(LLM, "Gemini 대상 선택"):
                response = self._client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=config.GEMINI_TEMPERATURE,
                        response_mime_type="application/json",
                        response_schema={
                            "type": "object",
                            "properties": {
                                "object_id": {"type": "integer"},
                                "accepted": {"type": "boolean"},
                                "reason": {"type": "string"},
                            },
                            "required": ["object_id", "accepted"],
                        },
                    ),
                )
            if not response.text:
                raise RuntimeError("Empty Gemini response")
            parsed = json.loads(response.text)
            selected_id = int(parsed["object_id"])
            if selected_id not in valid_ids:
                raise RuntimeError(f"Invalid Gemini object_id: {selected_id}")
            if final_verification and not bool(parsed.get("accepted")):
                self._logger.warning(
                    "Mission 3 final subgoal rejected by Gemini: "
                    f"{parsed.get('reason', 'no reason')}"
                )
                if verification_key is not None:
                    self._mission3_verification_cache[verification_key] = None
                return None
            if final_verification:
                self._logger.info(
                    f"Mission 3 final subgoal verified: object_id={selected_id}, "
                    f"reason={parsed.get('reason', '-') }"
                )
                if verification_key is not None:
                    self._mission3_verification_cache[verification_key] = selected_id
            return selected_id
        except Exception as error:
            selected_id = self._fallback(candidates, robot_pose)
            if final_verification:
                self._logger.warning(
                    "Mission 3 final subgoal verification unavailable; fail-open with "
                    f"Scene Graph candidate object_id={selected_id}: {error}"
                )
                if verification_key is not None:
                    self._mission3_verification_cache[verification_key] = selected_id
            return selected_id
