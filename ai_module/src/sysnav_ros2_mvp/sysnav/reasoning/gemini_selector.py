"""Gemini 2.5 Flash selector for choosing one target candidate."""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from sysnav import config


class GeminiSelector:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._client = None

    def _load(self) -> None:
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install google-genai: pip install google-genai") from exc
        self._client = genai.Client(api_key=self.api_key)

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
    ) -> int:
        if not candidates:
            raise ValueError("No target candidates")
        if len(candidates) == 1:
            return int(candidates[0]["object_id"])

        valid_ids = {int(item["object_id"]) for item in candidates}
        try:
            self._load()
            from google.genai import types
            prompt = f"""
You are a target-object selector for a mobile robot.
User instruction: {question}
Target candidates: {json.dumps([self._summary(x) for x in candidates], ensure_ascii=False)}
Other scene objects: {json.dumps([self._summary(x) for x in (context_objects or []) if int(x['object_id']) not in valid_ids], ensure_ascii=False)}
Choose exactly one target candidate. Use visual attributes and supplied 3D context.
Return JSON only and never output an object_id outside the target candidate list.
""".strip()
            contents: list[object] = [prompt]
            for item in candidates:
                contents.append(f"candidate object_id={int(item['object_id'])}")
                image = item.get("representative_image")
                if isinstance(image, np.ndarray) and image.size:
                    contents.append(types.Part.from_bytes(data=self._jpeg(image), mime_type="image/jpeg"))
            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=config.GEMINI_TEMPERATURE,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {"object_id": {"type": "integer"}, "reason": {"type": "string"}},
                        "required": ["object_id"],
                    },
                ),
            )
            if not response.text:
                raise RuntimeError("Empty Gemini response")
            selected_id = int(json.loads(response.text)["object_id"])
            if selected_id not in valid_ids:
                raise RuntimeError(f"Invalid Gemini object_id: {selected_id}")
            return selected_id
        except Exception:
            return self._fallback(candidates, robot_pose)
