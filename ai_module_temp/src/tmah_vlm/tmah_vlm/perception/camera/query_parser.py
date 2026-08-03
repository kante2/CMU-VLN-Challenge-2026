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

extract_target()가 진입점이고, 아래 정리 단계들을 순서대로 부른다:
  strip_leading_command -> cut_at_spatial_relation -> remove_articles
  -> remove_colors -> normalize_detector_prompt
"""

import re

# "Find / Locate / How many ..." 같은 앞머리 명령어
_LEADING_COMMAND = re.compile(
    r"^\s*(find|locate|show me|where is|point to|identify|how many)\s+", re.I)

# 관사 the/a/an
_ARTICLE = re.compile(r"\b(the|a|an)\b", re.I)

# 공간관계/수식어가 시작되는 지점 (여기서부터 잘라 핵심 명사만 남김)
_SPATIAL_RELATION = re.compile(
    r"\b(on|in|at|near|next to|farthest|furthest|closest|nearest|between|"
    r"above|below|behind|under|beside|that|which|with|to the|on the)\b", re.I)

# 색 형용사 (검출 프롬프트에서 제거 -> 오검출 방지)
_COLOR_WORD = re.compile(
    r"\b(red|orange|yellow|green|blue|teal|cyan|purple|violet|pink|brown|"
    r"black|white|gray|grey|beige|tan|gold|silver|maroon|navy)\b", re.I)

# 챌린지 라벨엔 재질/용기 수식어가 붙어 open-vocab 검출이 불안정해지는 경우가 있다.
# 선택(selection)에는 원문을 그대로 쓰되, 검출기에는 더 단순한 시각적 물체 클래스를 준다.
_DETECT_ALIASES = {
    "potted plant": "plant",
    "paper cup": "cup",
    "computer monitor": "monitor",
    "computer monitors": "monitor",
    "file cabinet": "cabinet",
    "projector screen": "screen",
}


# ========================================
# 진입점
# ========================================

def extract_target(question: str) -> dict:
    raw = question.strip()

    # "Find the red pillow on the sofa" 를 단계적으로 깎아 핵심 명사만 남긴다.
    body = strip_leading_command(raw)      # -> "the red pillow on the sofa"
    body = cut_at_spatial_relation(body)   # -> "the red pillow "
    body = remove_articles(body)           # -> " red pillow "

    # 검출 프롬프트는 색을 뺀 형태만 사용한다.
    object_name = tidy(remove_colors(body))       # -> "pillow"
    # 색을 빼면 비는 경우("red" 하나뿐 등)엔 색 포함본을, 그것도 비면 원본 질문을 쓴다.
    if not object_name:
        object_name = tidy(body)
    if not object_name:
        object_name = raw

    return {"object": normalize_detector_prompt(object_name), "raw": raw}


# ========================================
# 정리 단계 (각 함수는 문자열 하나를 받아 다듬은 문자열을 돌려준다)
# ========================================

def strip_leading_command(text: str) -> str:
    # 앞머리 명령어("find ", "how many " 등)를 떼어낸다.
    return _LEADING_COMMAND.sub("", text)


def cut_at_spatial_relation(text: str) -> str:
    # 공간관계 단어("on/near/between ...")가 처음 나오는 지점 앞까지만 남긴다.
    match = _SPATIAL_RELATION.search(text)
    if match:
        return text[:match.start()]
    return text


def remove_articles(text: str) -> str:
    # 관사(the/a/an)를 제거한다.
    return _ARTICLE.sub("", text)


def remove_colors(text: str) -> str:
    # 색 형용사를 제거한다 (검출 프롬프트에 색이 있으면 오검출이 잘 남).
    return _COLOR_WORD.sub("", text)


def tidy(text: str) -> str:
    # 연속 공백을 하나로 합치고 양끝 공백/구두점을 다듬는다.
    return re.sub(r"\s+", " ", text).strip(" .,")


def normalize_detector_prompt(object_name: str) -> str:
    # 재질/용기 수식어가 붙은 라벨을 검출기가 잘 잡는 단순 클래스명으로 바꾼다.
    object_name = object_name.lower()

    if object_name in _DETECT_ALIASES:
        return _DETECT_ALIASES[object_name]

    for phrase, alias in _DETECT_ALIASES.items():
        if phrase in object_name:
            return alias

    return object_name
