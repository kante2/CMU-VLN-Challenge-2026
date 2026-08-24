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
from sysnav.activity_log import LLM, activity
from sysnav.task.query_parser import extract_target, merge_reference_attributes, singularize

# 모든 object reference는 {category, attributes}다 - 예전엔 그냥 문자열이라
# "the black chair"가 통째로 카테고리가 됐고, 그 문자열이 그대로 YOLO-World 프롬프트로
# 들어갔다. open-vocab 검출기는 색 형용사를 구분 못 해서 흰 식탁의자를 0.79로 잡았고,
# object_memory 카테고리 이름까지 "black chair"가 되어 scene graph도 오염됐다.
# 이제 검출은 category("chair")만 쓰고, 색/모양 판정은 reasoning/attribute_verifier.py가 한다.
_OBJECT_REF_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "attributes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["category", "attributes"],
}

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "target": _OBJECT_REF_SCHEMA,
        "constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "relation": {"type": "string"},
                    "references": {"type": "array", "items": _OBJECT_REF_SCHEMA},
                },
                "required": ["relation", "references"],
            },
        },
    },
    "required": ["target", "constraints"],
}

_RELATION_ALIASES = {
    "closest_to": "nearest",
    "nearest_to": "nearest",
    "farthest_from": "farthest",
    "furthest_from": "farthest",
    "farthest": "farthest",
    "furthest": "farthest",
    "next_to": "beside",
}

_PROMPT_TEMPLATE = """
You parse a mobile-robot object-navigation instruction into a formal goal
G = (target, Φ), where target is an object category and Φ is a set of semantic
constraints the target object instance must satisfy: {{c(o) = target}} AND
{{every constraint in Φ holds for o}}.

Each element of Φ is either:
  - a visual attribute of the target itself (color, material, size, state) -> put
    it in the target's "attributes".
  - a spatial or comparative relation to another object -> put it in
    "constraints", in the order it appears in the sentence.

Instruction: {question}

Rules:
- Every object reference (the target and every constraint reference) is an object
  {{"category": ..., "attributes": [...]}}.
- "category" is the bare noun phrase with EVERY adjective removed. Colors,
  materials, sizes, shapes and states (black, white, wooden, metal, round,
  square, tall, small, open, closed, ...) always go into "attributes", never
  into "category". An open-vocabulary detector cannot tell colors apart, so a
  category that still carries an adjective silently matches the wrong object.
- Do NOT split multiword nouns: "trash can", "knife rack", "coffee table" are
  single categories. Only adjectives are separated, never the noun phrase itself.
- Write "category" in the singular ("tables" -> "table").
- Examples:
    "the black chair"   -> {{"category": "chair", "attributes": ["black"]}}
    "the round tables"  -> {{"category": "table", "attributes": ["round"]}}
    "the trash can"     -> {{"category": "trash can", "attributes": []}}
- target is only the object to find, never a full relation clause.
- If the instruction is a counting question ("How many X are Y?", "Count the
  number of X that Y."), target is still just X (the object category being
  counted) - never include "how many"/"count"/"the number of" or the verb
  "is"/"are"/"was"/"were" in target.
- Each constraint has a canonical snake_case relation (near, beside, left_of,
  right_of, in_front_of, behind, on, under, above, between, nearest, farthest) and concrete
  reference objects.
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


def _category(value: Any) -> str:
    """카테고리 표기를 규칙 파서와 통일한다 - 마지막 토큰만 단수화("round tables"의
    "tables" -> "table"). object_memory/scene_graph는 카테고리 문자열을 그대로 키로
    쓰므로, 두 파서가 다른 표기를 내면 같은 물체가 두 카테고리로 갈라진다."""
    tokens = _clean(value).split()
    if not tokens:
        return ""
    tokens[-1] = singularize(tokens[-1])
    return " ".join(tokens)


def _object_ref(value: Any) -> dict:
    """{category, attributes} 하나를 정규화한다.

    문자열도 받아준다 - 스키마를 바꾸기 전 형식으로 답하는 모델/캐시가 있어도
    조용히 깨지지 않게 하려는 것이다(그 경우 attributes는 비고, 카테고리에 형용사가
    남을 수 있지만 최소한 파이프라인은 돈다)."""
    if isinstance(value, dict):
        return {
            "category": _category(value.get("category")),
            "attributes": _unique(list(value.get("attributes") or [])),
        }
    return {"category": _category(value), "attributes": []}


def normalize_llm_result(question: str, payload: dict[str, Any]) -> dict:
    """Gemini 응답을 검증하고, 기존 파이프라인이 기대하는 필드를 채워서
    query_parser.extract_target()과 동일한 스키마의 dict로 반환한다 - 하위 코드
    (scene_graph, selection_job 등)가 어느 파서를 썼는지 몰라도 그대로 동작한다."""
    target_ref = _object_ref(payload.get("target"))
    target = target_ref["category"]
    if not target:
        raise ValueError("target이 비어있음")

    # 스키마 변경 전에는 attributes가 top-level이었다 - 옛 형식 응답도 살려준다.
    attributes = target_ref["attributes"] or _unique(list(payload.get("attributes") or []))

    constraints: list[dict] = []
    for item in list(payload.get("constraints") or []):
        if not isinstance(item, dict):
            continue
        relation = _normalize_relation(item.get("relation"))
        reference_refs: list[dict] = []
        for reference in list(item.get("references") or []):
            normalized = _object_ref(reference)
            if normalized["category"] and normalized["category"] not in [
                existing["category"] for existing in reference_refs
            ]:
                reference_refs.append(normalized)
        if relation and reference_refs:
            constraints.append({
                "relation": relation,
                "references": [reference["category"] for reference in reference_refs],
                "reference_refs": reference_refs,
            })

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
        # 형용사가 제거된 순수 카테고리만 나간다 - 이게 그대로 YOLO-World set_classes()로
        # 들어간다(perception/perception_pipeline.py).
        "detection_prompts": _unique([target, *all_references]),
        # 참조 물체의 속성. reasoning/attribute_filter.py가 이걸 읽어 관계 판정/좌표
        # 선택 대상을 실제로 좁힌다(예: "closest to the black chair"의 chair 후보).
        "reference_attributes": merge_reference_attributes([
            (reference["category"], reference["attributes"])
            for constraint in constraints
            for reference in constraint["reference_refs"]
        ]),
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

            with activity.operation(LLM, "Gemini 질문 파싱"):
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
