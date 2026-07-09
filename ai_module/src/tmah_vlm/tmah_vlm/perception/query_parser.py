#!/usr/bin/env python3
"""
질문 -> GroundingDINO 검출용 "물체명" 추출 (최소 버전).

이전엔 색/공간관계까지 정규식으로 다 파싱했지만, 이제 그건 Qwen(selector)이
원본 질문을 통째로 이해해서 처리한다. 그래서 여기선 "무엇을 검출할지"의
핵심 명사구만 뽑는다. 규칙은 얇게 유지.

반환:
  {
    "object": "pillow",           # GroundingDINO 프롬프트 (색/관계 제거)
    "raw": "<원본 질문>",          # Qwen 에 그대로 넘길 원문
  }
"""

import re

_LEADING = re.compile(
    r"^\s*(find|locate|show me|where is|point to|identify|how many)\s+", re.I)
_ARTICLES = re.compile(r"\b(the|a|an)\b", re.I)

# 공간관계/수식어가 시작되는 지점에서 자른다 (핵심 명사만 남기기)
_CUT = re.compile(
    r"\b(on|in|at|near|next to|farthest|furthest|closest|nearest|between|"
    r"above|below|behind|under|beside|that|which|with|to the|on the)\b", re.I)

# 색 형용사 (검출 프롬프트에서 제거 -> 오검출 방지)
_COLORS = re.compile(
    r"\b(red|orange|yellow|green|blue|teal|cyan|purple|violet|pink|brown|"
    r"black|white|gray|grey|beige|tan|gold|silver|maroon|navy)\b", re.I)

# Challenge labels often include material/container modifiers that make
# open-vocabulary detection brittle. Keep the full question for selection, but
# ask the detector for the simpler visual object class.
_DETECT_ALIASES = {
    "potted plant": "plant",
    "paper cup": "cup",
    "computer monitor": "monitor",
    "computer monitors": "monitor",
    "file cabinet": "cabinet",
    "projector screen": "screen",
}


def _normalize_detector_prompt(obj: str) -> str:
    obj = obj.lower()
    if obj in _DETECT_ALIASES:
        return _DETECT_ALIASES[obj]

    for phrase, alias in _DETECT_ALIASES.items():
        if phrase in obj:
            return alias

    return obj


def extract_target(question: str) -> dict:
    q = question.strip()
    raw = q

    q_body = _LEADING.sub("", q)

    # 공간관계 전까지만
    m = _CUT.search(q_body)
    if m:
        q_body = q_body[:m.start()]

    q_body = _ARTICLES.sub("", q_body)
    # 검출 프롬프트에서 색 제거 (형태만 남김)
    obj = _COLORS.sub("", q_body)
    obj = re.sub(r"\s+", " ", obj).strip(" .,")

    if not obj:
        # 색 제거로 비면, 색 포함본이라도 사용
        obj = re.sub(r"\s+", " ", q_body).strip(" .,")
    if not obj:
        obj = question.strip()

    return {"object": _normalize_detector_prompt(obj), "raw": raw}
