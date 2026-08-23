"""LLM 기반 Instruction-Following 절 분해 폴백.

missions/mission3_pipe.py의 규칙 기반 _split_clauses()는 알려진 트리거 단어
(go to/near/between, stop at/by, take the path, avoid(ing) the path, pass by,
생략된 "and then to"/"and finally, to")로 questions.json의 실제 instruction_following
30문장 전부를 정확히 쪼개는 걸 검증했다. 근데 이 트리거 목록에 없는 새 표현(예:
"walk over to", "head to", "arrive at")이 들어오면 규칙 기반은 절을 하나도 못 찾고
빈 리스트를 반환하고, 그러면 mission3_pipe.parse_instruction()의 steps도 비어서
question_callback이 태스크 자체를 시작하지 않는다(그 질문은 자동으로 0점).

이 모듈은 그 경우에만(_split_clauses()가 완전히 빈 결과를 냈을 때만) 호출되는
폴백이다. _split_clauses()가 만드는 것과 동일한 raw_steps 형식
({"kind": "destination", "text": ...} 등)으로 변환해서 반환하므로,
mission3_pipe.parse_instruction()의 절 소비 로직은 그대로 재사용된다.
"""

from __future__ import annotations

import json
import os

from rclpy.logging import get_logger

from sysnav import config
from sysnav.activity_log import LLM, activity

_PROMPT_TEMPLATE = """
You split a multi-step robot navigation instruction into an ordered list of clauses,
in the order they should be executed.

Each clause has a "kind":
  - "destination": a place the robot must actually stop at (e.g. "go to X", "stop at
    Y", "arrive at Y"). Give "text" = the noun phrase describing the place (e.g.
    "the chair closest to the TV").
  - "destination_between": a stop that is the point between two named references
    (e.g. "go between the bench and the bed"). Give "refs" = [A, B].
  - "positive_path_near" / "positive_path_between": a place/corridor the robot must
    pass through or near before the next destination, but does not stop at (e.g.
    "take the path near Z" -> ref=Z; "take the path between A and B" -> refs=[A, B]).
  - "negative_path_near" / "negative_path_between": an area the robot must avoid
    passing through anywhere in its route (same ref shape as positive_path).

Instruction: {question}

Rules:
- Preserve the exact order clauses appear in the instruction.
- For plain object-category noun phrases, keep them intact (e.g. "the chair closest
  to the TV"), including their own attributes/relations - do not split those further.
- Do not invent clauses that are not stated in the instruction.
""".strip()

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "clauses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": [
                            "destination", "destination_between",
                            "positive_path_near", "positive_path_between",
                            "negative_path_near", "negative_path_between",
                        ],
                    },
                    "text": {"type": "string"},
                    "ref": {"type": "string"},
                    "refs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["kind"],
            },
        }
    },
    "required": ["clauses"],
}


def _convert(clause: dict) -> dict | None:
    kind = clause.get("kind")
    if kind == "destination":
        text = str(clause.get("text", "")).strip()
        return {"kind": "destination", "text": text} if text else None
    if kind == "destination_between":
        refs = [str(value).strip() for value in (clause.get("refs") or []) if str(value).strip()]
        if len(refs) != 2:
            return None
        return {"kind": "destination", "between_ref": {"kind": "between", "refs": refs}}
    if kind in ("positive_path_near", "negative_path_near"):
        ref = str(clause.get("ref", "")).strip()
        if not ref:
            return None
        path_kind = "positive_path" if kind.startswith("positive") else "negative_path"
        return {"kind": path_kind, "ref": {"kind": "near", "refs": [ref]}}
    if kind in ("positive_path_between", "negative_path_between"):
        refs = [str(value).strip() for value in (clause.get("refs") or []) if str(value).strip()]
        if len(refs) != 2:
            return None
        path_kind = "positive_path" if kind.startswith("positive") else "negative_path"
        return {"kind": path_kind, "ref": {"kind": "between", "refs": refs}}
    return None


class LLMInstructionSplitter:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._client = None
        self._logger = get_logger("sysnav_llm_instruction_splitter")

    def _load(self) -> None:
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai
        self._client = genai.Client(api_key=self.api_key)

    def split(self, question: str) -> list[dict]:
        """_split_clauses()와 동일한 raw_steps 형식으로 반환. 실패하면(무엇이든)
        빈 리스트 - 호출 쪽이 이미 "steps 없음 = 파싱 실패"로 처리하므로 별도
        예외 처리가 필요 없다."""
        try:
            self._load()
            from google.genai import types

            with activity.operation(LLM, "Gemini 지시문 절 분해"):
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
            clauses = json.loads(response.text).get("clauses") or []
            raw_steps = [step for step in (_convert(clause) for clause in clauses) if step]
            self._logger.info(f"LLM instruction split (rule-based fallback triggered): {raw_steps}")
            return raw_steps
        except Exception as error:
            self._logger.warning(f"LLM instruction split failed: {error}")
            return []
