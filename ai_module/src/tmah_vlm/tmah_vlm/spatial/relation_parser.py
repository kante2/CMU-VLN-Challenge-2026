#!/usr/bin/env python3
"""
Object Filtering (SORT-3D 논문의 Module 3) — 질문에서 공간관계절을 뽑는다.

perception/query_parser.py는 "GroundingDINO가 뭘 검출해야 하는가"(타겟 클래스
명사)만 뽑는 원래 역할을 그대로 유지한다. 이 파일은 그와 별개로, "near the
window", "between the table and the wall", "closest to the fridge" 같은
공간관계절을 구조화해서 뽑는 것만 전담한다 — VLM이 관계/거리를 직접 판단하면
부정확하므로(spatial/relations.py의 deterministic 함수가 대신 계산),
여기서는 "무슨 관계, 어떤 랜드마크"만 텍스트에서 추출한다.

반환 (extract_relations):
  [
    {"type": "between", "refs": ["table", "wall"]},
    {"type": "closest", "refs": ["fridge"]},
  ]

type은 spatial/relations.py의 find_*/closest_to 등과 1:1 대응한다:
  between, near, closest, farthest, left, right, above, below, on.

같은 문장에 관계가 여러 개 나올 수 있다(예: "on the kitchen island that is
closest to the fridge") — 등장 순서대로 전부 반환한다.
"""

import re

_ARTICLES = re.compile(r"\b(the|a|an)\b", re.I)

# relation 절 하나를 다음 relation 키워드/that/which/문장부호/끝에서 끊는다.
_STOP = (
    r"(?=\s+\b(?:between|near|next\s+to|close\s+to|closest\s+to|nearest\s+to|"
    r"farthest\s+from|furthest\s+from|above|over|below|under|beneath|on)\b"
    r"|\s+\bthat\b|\s+\bwhich\b|[,.?!]|$)"
)

# 하드하게 박힌
_RELATION_PATTERNS = [
    ("between", re.compile(r"\bbetween\s+(.+?)\s+and\s+(.+?)" + _STOP, re.I)),
    ("closest", re.compile(r"\b(?:closest|nearest)\s+to\s+(.+?)" + _STOP, re.I)),
    ("farthest", re.compile(r"\b(?:farthest|furthest)\s+from\s+(.+?)" + _STOP, re.I)),
    ("left", re.compile(r"\b(?:to\s+the\s+left\s+of|left\s+of)\s+(.+?)" + _STOP, re.I)),
    ("right", re.compile(r"\b(?:to\s+the\s+right\s+of|right\s+of)\s+(.+?)" + _STOP, re.I)),
    ("above", re.compile(r"\b(?:above|over)\s+(.+?)" + _STOP, re.I)),
    ("below", re.compile(r"\b(?:below|under|beneath)\s+(.+?)" + _STOP, re.I)),
    ("on", re.compile(r"\bon\s+(.+?)" + _STOP, re.I)),
    ("near", re.compile(r"\b(?:near|next\s+to|close\s+to)\s+(.+?)" + _STOP, re.I)),
]


def _clean_landmark(phrase):
    phrase = _ARTICLES.sub("", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip(" .,?!")
    return phrase.lower()


def extract_relations(text):
    """질문 본문에서 공간관계절(들)을 등장 순서대로 뽑는다."""
    found = []

    for rel_type, pattern in _RELATION_PATTERNS:
        for m in pattern.finditer(text):
            refs = [_clean_landmark(g) for g in m.groups() if g]
            refs = [r for r in refs if r]
            if refs:
                found.append((m.start(), {"type": rel_type, "refs": refs}))

    found.sort(key=lambda item: item[0])
    return [relation for _, relation in found]


def all_referenced_landmarks(relations):
    """relations 리스트에서 등장한 랜드마크 이름을 중복 없이 순서대로 모은다.

    handlers가 이 이름들로 (지금은 즉석 검출, 나중엔 semantic map 조회를 통해)
    각 랜드마크의 3D 위치를 찾아야 한다.
    """
    seen = []
    for relation in relations:
        for ref in relation["refs"]:
            if ref not in seen:
                seen.append(ref)
    return seen
