#!/usr/bin/env python3
"""
질문 텍스트에서 "찾을 대상"을 뽑는 아주 단순한 파서 (Phase 1a 용).

지금은 규칙 기반의 최소 구현입니다. 나중에 LLM 이나 더 똑똑한 파싱으로
교체할 예정 (그때 이 함수 시그니처만 유지하면 나머지는 안 바뀜).

예)
  "Find the teal pillow on the sofa farthest from the window"
    -> 대략 "pillow" (핵심 명사)
지금은 완벽하지 않아도 됩니다. 검출이 되는지 보는 게 목적.
"""

import re

# object_reference 질문 앞에 흔히 붙는 군더더기
_LEADING = re.compile(r"^\s*(find|locate|show me|where is|point to)\s+", re.I)
_ARTICLES = re.compile(r"\b(the|a|an)\b", re.I)

# 공간관계 이후는 버림 (핵심 명사만 남기려는 단순 휴리스틱)
_SPATIAL_CUT = re.compile(
    r"\b(on|in|at|near|next to|farthest|closest|between|above|below|behind|"
    r"under|beside|that|which|with)\b", re.I)


def extract_target(question: str) -> str:
    """
    질문 -> GroundingDINO 프롬프트로 쓸 대상 문구.
    지금은 대충 핵심 명사구만 남긴다.
    """
    q = question.strip()
    q = _LEADING.sub("", q)              # "Find " 제거

    # 공간관계 단어가 처음 나오는 지점에서 자른다
    m = _SPATIAL_CUT.search(q)
    if m:
        q = q[:m.start()]

    q = _ARTICLES.sub("", q)             # 관사 제거
    q = re.sub(r"\s+", " ", q).strip(" .,")

    # 너무 짧아지면 원래 질문에서 마지막 명사라도
    if not q:
        q = question.strip()

    return q.lower()
