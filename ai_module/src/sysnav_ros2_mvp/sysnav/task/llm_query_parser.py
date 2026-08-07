"""LLM 기반 문장 파서 - SysNav 논문 Sec. III(Problem Formulation)의 정의를 따른다.

논문 정의: 로봇은 target object category c_tgt와 semantic constraint 집합
Φ = {φ_1, ..., φ_K}로 구성된 목표 G = (c_tgt, Φ)를 받는다. 각 φ_k(·)는 물체
속성(color/state 등) 또는 다른 물체와의 공간 관계를 인코딩하며("chair, red" 또는
"person, sitting on a couch"), 로봇은 {c(o_i) = c_tgt} ∧ {∀φ_k ∈ Φ, φ_k(o_i) = 1}을
만족하는 물체 o_i를 찾아야 한다.

이 c_tgt/Φ 추출은 논문 Sec. IV-A-2 "VLM Reasoning"(VLM Query 모듈)이 담당한다고
서술돼 있을 뿐 구체적 프롬프트/스키마는 논문에 없어서, 그 역할을 우리가 직접
설계해서 구현한 것이 이 모듈이다: target=c_tgt, attributes=속성류 φ_k,
constraints=공간관계류 φ_k로 나눠 뽑는다.

동의어/시각 alias 생성은 이 모듈 책임이 아니다(그건 오검출 대응이라 별개 관심사이고,
실제 카메라 프레임을 보고 추측하게 시키면 할루시네이션 위험이 크다고 판단해서 뺐다).
API 실패/키 없음/빈 응답 등 어떤 이유로든 항상 정규식 기반 task/query_parser.py로
자동 폴백한다.
"""

from __future__ import annotations

import json
import os
from typing import Any

from rclpy.logging import get_logger

from sysnav import config
from sysnav.task.query_parser import extract_target

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "attributes": {"type": "array", "items": {"type": "string"}},
        "constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "relation": {"type": "string"},
                    "references": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["relation", "references"],
            },
        },
    },
    "required": ["target", "attributes", "constraints"],
}

_RELATION_ALIASES = {
    "closest_to": "nearest",
    "nearest_to": "nearest",
    "next_to": "beside",
}

_PROMPT_TEMPLATE = """
You parse a mobile-robot object-navigation instruction into a formal goal
G = (target, Φ), where target is an object category and Φ is a set of semantic
constraints the target object instance must satisfy: {{c(o) = target}} AND
{{every constraint in Φ holds for o}}.

Each element of Φ is either:
  - a visual attribute of the target itself (color, material, size, state) -> put
    it in "attributes".
  - a spatial or comparative relation to another object category -> put it in
    "constraints", in the order it appears in the sentence.

Instruction: {question}

Rules:
- target is only the object category to find (a short noun phrase), never a full
  relation clause.
- If the instruction is a counting question ("How many X are Y?", "Count the
  number of X that Y."), target is still just X (the object category being
  counted) - never include "how many"/"count"/"the number of" or the verb
  "is"/"are"/"was"/"were" in target.
- Each constraint has a canonical snake_case relation (near, beside, left_of,
  right_of, in_front_of, behind, on, under, above, between, nearest) and concrete
  reference object categories.
- Keep multiword categories intact (e.g. "trash can", "knife rack").
- Do not invent objects or constraints that are not stated in the instruction.
""".strip()


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _unique(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        cleaned = _clean(value)
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


def _normalize_relation(value: Any) -> str:
    relation = _clean(value).replace(" ", "_")
    return _RELATION_ALIASES.get(relation, relation)


def normalize_llm_result(question: str, payload: dict[str, Any]) -> dict:
    """Gemini 응답을 검증하고, 기존 파이프라인이 기대하는 필드를 채워서
    query_parser.extract_target()과 동일한 스키마의 dict로 반환한다 - 하위 코드
    (scene_graph, selection_job 등)가 어느 파서를 썼는지 몰라도 그대로 동작한다."""
    target = _clean(payload.get("target"))
    if not target:
        raise ValueError("target이 비어있음")

    attributes = _unique(list(payload.get("attributes") or []))

    constraints: list[dict] = []
    for item in list(payload.get("constraints") or []):
        if not isinstance(item, dict):
            continue
        relation = _normalize_relation(item.get("relation"))
        references = _unique(list(item.get("references") or []))
        if relation and references:
            constraints.append({"relation": relation, "references": references})

    # 논문의 Φ = {φ_1, ..., φ_K} 중 공간관계류를 문장에 나온 순서대로 체인으로 잇는다:
    # 첫 constraint의 source는 target, 그 다음부터는 이전 constraint의 첫 reference가
    # source가 된다. "between"은 3항이라 두 reference를 같은 source에 매단다.
    relation_chain: list[tuple[str, str, str]] = []
    source = target
    for constraint in constraints:
        references = constraint["references"]
        if constraint["relation"] == "between":
            relation_chain.extend((source, "between", reference) for reference in references)
        else:
            relation_chain.append((source, constraint["relation"], references[0]))
            source = references[0]

    primary = constraints[0] if constraints else None
    all_references = _unique([
        reference for constraint in constraints for reference in constraint["references"]
    ])

    return {
        "raw": question.strip(),
        "target": target,
        "attributes": attributes,
        # relation/reference_objects: 체인의 첫 hop만 - 기존 단일-relation 코드 경로와
        # 호환용. relation_chain이 전체 체인을 담는다.
        "relation": None if primary is None else primary["relation"],
        "reference_objects": [] if primary is None else list(primary["references"]),
        "relation_chain": relation_chain,
        "detection_prompts": _unique([target, *all_references]),
        "parser": "llm",
    }


class LLMQueryParser:
    """문장 -> (target, attributes, relation_chain). SysNav paper의 VLM Query
    모듈 중 instruction parsing 역할만 담당한다 - 물체 후보 선택은
    reasoning/gemini_selector.py, 공간관계 검증은
    reasoning/spatial_relation_reasoner.py가 별도로 한다."""

    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._client = None
        self._logger = get_logger("sysnav_llm_query_parser")

    def _load(self) -> None:
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai
        self._client = genai.Client(api_key=self.api_key)

    def parse(self, question: str) -> dict:
        if not config.LLM_QUERY_PARSER_ENABLED:
            return self._fallback(question, "config로 비활성화됨")

        try:
            self._load()
            from google.genai import types

            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=_PROMPT_TEMPLATE.format(question=question),
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=_RESPONSE_SCHEMA,
                ),
            )
            if not response.text:
                raise RuntimeError("Gemini가 빈 응답을 반환함")
            parsed = normalize_llm_result(question, json.loads(response.text))
            self._logger.info(
                f"LLM parse: target={parsed['target']}, attributes={parsed['attributes']}, "
                f"relation_chain={parsed['relation_chain']}"
            )
            return parsed
        except Exception as error:
            return self._fallback(question, str(error))

    def _fallback(self, question: str, reason: str) -> dict:
        self._logger.warning(f"LLM 파싱 실패, 규칙 기반으로 폴백: {reason}")
        parsed = extract_target(question)
        parsed["parser"] = "rules"
        return parsed
