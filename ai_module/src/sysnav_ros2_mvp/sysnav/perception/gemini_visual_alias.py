"""Image-aware alias fallback for canonical categories missed by YOLO-World."""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
from rclpy.logging import get_logger

from sysnav import config
from sysnav.task.prompt_bank import normalize_dynamic_aliases, update_alias_cache


class GeminiVisualAliasFallback:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._client = None
        self._attempted: set[str] = set()
        self._logger = get_logger("sysnav_visual_alias")

    def reset(self) -> None:
        self._attempted.clear()

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
            raise RuntimeError("Visual-alias JPEG encoding failed")
        return encoded.tobytes()

    def suggest(
        self,
        image_rgb: np.ndarray,
        missing_categories: list[str],
        existing_prompts: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        if not config.GEMINI_VISUAL_ALIAS_FALLBACK_ENABLED:
            return {}
        pending = [category for category in missing_categories if category not in self._attempted]
        self._attempted.update(pending)
        if not pending:
            return {}

        try:
            self._load()
            from google.genai import types
            prompt = f"""
You help an open-vocabulary object detector recover objects it missed.
Missing canonical categories: {json.dumps(pending, ensure_ascii=False)}
Already-tried detector phrases: {json.dumps(existing_prompts, ensure_ascii=False)}

Inspect this exact image. For each missing category that may be visible, return up
to {config.GEMINI_VISUAL_ALIAS_MAX_ALIASES} short English visual noun phrases that
describe the same physical object as it appears here. Do not return relations,
sentences, coordinates, broad room labels, or a different object category.
Return an empty aliases list when the object is not plausibly visible.
""".strip()
            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=self._jpeg(image_rgb), mime_type="image/jpeg"),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "categories": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "category": {"type": "string"},
                                        "aliases": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": ["category", "aliases"],
                                },
                            }
                        },
                        "required": ["categories"],
                    },
                ),
            )
            if not response.text:
                return {}
            allowed = set(pending)
            normalized = normalize_dynamic_aliases(json.loads(response.text).get("categories", []))
            normalized = {
                category: aliases[:config.GEMINI_VISUAL_ALIAS_MAX_ALIASES]
                for category, aliases in normalized.items()
                if category in allowed
            }
            if normalized:
                try:
                    update_alias_cache(normalized)
                except OSError as error:
                    self._logger.warning(f"Could not cache visual aliases: {error}")
                self._logger.info(f"VLM visual aliases: {normalized}")
            return normalized
        except Exception as error:
            self._logger.warning(f"VLM visual-alias fallback skipped: {error}")
            return {}
