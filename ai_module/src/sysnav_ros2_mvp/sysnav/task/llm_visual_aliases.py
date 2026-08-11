"""Expand canonical task categories into conservative YOLO-World prompts."""

from __future__ import annotations

import json
import os
from typing import Any

from rclpy.logging import get_logger

from sysnav import config

# Empirically checked on the challenge frame.  These are also the deterministic
# fallback when Gemini is unavailable, so a transient API failure does not
# remove a known-good detector prompt.
_SAFE_ALIASES: dict[str, list[str]] = {
    "painting": ["wall picture"],
}

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "categories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "canonical": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["canonical", "aliases"],
            },
        }
    },
    "required": ["categories"],
}

_PROMPT = """
Create conservative visual detector aliases for YOLO-World.

Instruction: {question}
Canonical object categories: {categories}

For every canonical category, return zero to {max_aliases} short English noun
phrases that describe the SAME physical object and are likely common image
labels. Never add color, size, location, spatial relations, scene context,
actions, or a broader/different object. Never use another canonical category as
an alias. Use at most {max_words} words. If no safe synonym exists, return an
empty aliases list. Example: painting may use "wall picture"; do not use
"framed picture" or "hanging picture" because those can describe a TV.
""".strip()


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def normalize_alias_payload(
    canonical_categories: list[str], payload: dict[str, Any]
) -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    """Validate aliases and return prompts, prompt->canonical map and debug map."""
    canonicals = list(dict.fromkeys(_clean(value) for value in canonical_categories if _clean(value)))
    canonical_set = set(canonicals)
    proposed: dict[str, list[str]] = {canonical: [] for canonical in canonicals}

    for item in list(payload.get("categories") or []):
        if not isinstance(item, dict):
            continue
        canonical = _clean(item.get("canonical"))
        if canonical not in canonical_set:
            continue
        for raw_alias in list(item.get("aliases") or []):
            alias = _clean(raw_alias)
            if (
                not alias
                or alias == canonical
                or alias in canonical_set
                or len(alias.split()) > config.LLM_VISUAL_ALIAS_MAX_WORDS
                or alias in proposed[canonical]
            ):
                continue
            proposed[canonical].append(alias)

    prompts: list[str] = []
    canonical_by_prompt: dict[str, str] = {}
    accepted: dict[str, list[str]] = {}
    for canonical in canonicals:
        aliases = list(_SAFE_ALIASES.get(canonical, [])) + proposed[canonical]
        aliases = list(dict.fromkeys(_clean(value) for value in aliases if _clean(value)))
        aliases = aliases[: config.LLM_VISUAL_ALIAS_MAX_PER_CATEGORY]
        accepted[canonical] = aliases
        for prompt in [canonical, *aliases]:
            if prompt not in canonical_by_prompt:
                prompts.append(prompt)
                canonical_by_prompt[prompt] = canonical
    return prompts, canonical_by_prompt, accepted


class LLMVisualAliasExpander:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._client = None
        self._cache: dict[tuple[str, tuple[str, ...]], tuple[list[str], dict[str, str], dict]] = {}
        self._logger = get_logger("sysnav_llm_visual_aliases")

    def _load(self) -> None:
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai
        self._client = genai.Client(api_key=self.api_key)

    def expand(
        self, question: str, canonical_categories: list[str]
    ) -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
        canonicals = list(dict.fromkeys(_clean(value) for value in canonical_categories if _clean(value)))
        cache_key = (_clean(question), tuple(canonicals))
        if cache_key in self._cache:
            return self._cache[cache_key]

        payload: dict[str, Any] = {"categories": []}
        if config.LLM_VISUAL_ALIASES_ENABLED:
            try:
                self._load()
                from google.genai import types
                response = self._client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=_PROMPT.format(
                        question=question,
                        categories=json.dumps(canonicals, ensure_ascii=False),
                        max_aliases=config.LLM_VISUAL_ALIAS_MAX_PER_CATEGORY,
                        max_words=config.LLM_VISUAL_ALIAS_MAX_WORDS,
                    ),
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=_RESPONSE_SCHEMA,
                    ),
                )
                if not response.text:
                    raise RuntimeError("Gemini returned an empty response")
                payload = json.loads(response.text)
            except Exception as error:
                self._logger.warning(f"Visual alias generation failed; using safe aliases: {error}")

        result = normalize_alias_payload(canonicals, payload)
        self._cache[cache_key] = result
        aliases = result[2]
        self._logger.info(f"YOLO visual aliases: {aliases}")
        return result
