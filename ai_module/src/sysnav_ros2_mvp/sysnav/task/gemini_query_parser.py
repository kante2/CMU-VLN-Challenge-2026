"""Gemini-backed instruction parser with deterministic fallback."""

from __future__ import annotations

import json
import os
from typing import Any

from rclpy.logging import get_logger

from sysnav import config
from sysnav.task.prompt_bank import (
    enrich_task_prompts,
    load_alias_cache,
    normalize_dynamic_aliases,
    update_alias_cache,
)
from sysnav.task.query_parser import extract_target


_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "attributes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "relation": {"type": "string"},
                    "references": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["relation", "references"],
            },
        },
        "prompt_aliases": {
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
        },
    },
    "required": ["target", "attributes", "constraints", "prompt_aliases"],
}


def _clean_phrase(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _unique(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        cleaned = _clean_phrase(value)
        if cleaned and cleaned not in output:
            output.append(cleaned)
    return output


def _normalize_relation(value: Any) -> str:
    relation = _clean_phrase(value).replace(" ", "_")
    return {
        "closest_to": "nearest",
        "nearest_to": "nearest",
        "next_to": "beside",
    }.get(relation, relation)


def normalize_gemini_result(
    question: str,
    payload: dict[str, Any],
    dynamic_aliases: dict[str, list[str]] | None = None,
) -> dict:
    """Validate Gemini output and add fields expected by the existing pipeline."""
    target = _clean_phrase(payload.get("target"))
    if not target:
        raise ValueError("Gemini parser returned an empty target")

    attributes = _unique(list(payload.get("attributes") or []))
    constraints = []
    all_references: list[str] = []
    for item in list(payload.get("constraints") or []):
        if not isinstance(item, dict):
            continue
        relation = _normalize_relation(item.get("relation"))
        references = _unique(list(item.get("references") or []))
        if not relation or not references:
            continue
        constraints.append({"relation": relation, "references": references})
        all_references.extend(references)

    all_references = _unique(all_references)
    primary = constraints[0] if constraints else None
    relation_chain: list[tuple[str, str, str]] = []
    source = target
    for constraint in constraints:
        references = constraint["references"]
        if not references:
            continue
        if constraint["relation"] == "between":
            relation_chain.extend(
                (source, constraint["relation"], reference)
                for reference in references
            )
        else:
            relation_chain.append((source, constraint["relation"], references[0]))
        source = references[0]
    return enrich_task_prompts({
        "raw": question.strip(),
        "target": target,
        "attributes": attributes,
        # Compatibility fields for existing scene-graph and selection code.
        "relation": None if primary is None else primary["relation"],
        "reference_objects": [] if primary is None else list(primary["references"]),
        "relation_chain": relation_chain,
        "constraints": constraints,
        "detection_prompts": _unique([target, *all_references]),
        "parser": "gemini",
    }, dynamic_aliases)


class GeminiQueryParser:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._client = None
        self._logger = get_logger("sysnav_query_parser")

    def _load(self) -> None:
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install google-genai: pip install google-genai") from exc
        self._client = genai.Client(api_key=self.api_key)

    def parse(self, question: str) -> dict:
        if not config.GEMINI_QUERY_PARSER_ENABLED:
            return self._fallback(question, "disabled by configuration")

        try:
            self._load()
            from google.genai import types

            prompt = f"""
Parse a mobile-robot visual navigation instruction into structured JSON.

Instruction: {question}

Rules:
- target must be only the object category to find, never a full relation phrase.
- attributes contains only visual properties of the target.
- constraints preserves every spatial or comparative condition in sentence order.
- Each constraint has a canonical snake_case relation and concrete reference
  object categories. Examples: near, beside, left_of, right_of, in_front_of,
  behind, on, under, between, closest_to, farthest_from.
- Keep multiword categories intact, for example "trash can" and "knife rack".
- Do not invent objects or conditions not stated in the instruction.
- prompt_aliases contains one entry for the target and every reference category.
- Each alias must be a short English visual noun phrase for the same physical
  object, never an instruction or a spatial relation. Return at most 4 aliases
  per category. Include common names such as "knife block" for "knife rack".
""".strip()
            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=_RESPONSE_SCHEMA,
                ),
            )
            if not response.text:
                raise RuntimeError("Gemini parser returned an empty response")
            payload = json.loads(response.text)
            generated_aliases = normalize_dynamic_aliases(
                payload.get("prompt_aliases", [])
            )
            cached_aliases = load_alias_cache()
            aliases = dict(cached_aliases)
            aliases.update(generated_aliases)
            if generated_aliases:
                try:
                    aliases = update_alias_cache(generated_aliases)
                except OSError as error:
                    self._logger.warning(f"Could not update prompt alias cache: {error}")
            parsed = normalize_gemini_result(question, payload, aliases)
            self._logger.info(
                f"Gemini parse: target={parsed['target']}, "
                f"attributes={parsed['attributes']}, "
                f"constraints={parsed['constraints']}, "
                f"prompts={parsed['detection_prompts']}"
            )
            return parsed
        except Exception as error:
            return self._fallback(question, str(error))

    def _fallback(self, question: str, reason: str) -> dict:
        parsed = extract_target(question)
        parsed["constraints"] = (
            [{
                "relation": parsed["relation"],
                "references": list(parsed["reference_objects"]),
            }]
            if parsed.get("relation") and parsed.get("reference_objects")
            else []
        )
        parsed["parser"] = "rules"
        parsed["detection_prompts"] = _unique([
            parsed["target"],
            *[
                reference
                for constraint in parsed["constraints"]
                for reference in constraint["references"]
            ],
        ])
        self._logger.warning(f"Gemini parsing failed; using rule fallback: {reason}")
        return enrich_task_prompts(parsed, load_alias_cache())
