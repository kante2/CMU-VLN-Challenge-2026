"""문장 -> 미션 타입(Numerical / Object Reference / Instruction-Following) 분류.

챌린지 questions.json의 세 미션은 응답 형식(Int32 / Marker / Pose2D 시퀀스)과
상태머신 흐름이 완전히 다르므로(MISSION_1/2/3_*_CLAUDE.txt 참고), 목표 물체를
파싱하기 전에 먼저 어느 파이프라인으로 보낼지 정해야 한다.

규칙 기반으로 충분하다 - LLM 없이도 세 유형이 표현 패턴으로 뚜렷이 구분된다
(questions.json 전체 15개 씬 x (numerical 1 + object_reference 2 + instruction 2)
문장을 전수 확인해서 검증함, task/mission_classifier_selftest에서 재검증 가능).
"""

from __future__ import annotations

import re

MISSION_NUMERICAL = "numerical"
MISSION_OBJECT_REFERENCE = "object_reference"
MISSION_INSTRUCTION_FOLLOWING = "instruction_following"

# "How many ..."/"Count the number of ..." - 문장 맨 앞에서만 인정한다(중간에 우연히
# 등장하는 경우는 없다고 questions.json 전수 확인함).
_NUMERICAL_PATTERN = re.compile(r"^\s*(how many|count)\b", re.IGNORECASE)

# instruction-following은 "이동 동사"가 있는 문장이다 - object_reference는 항상
# "Find the X" 또는 "The X..." 명사구로 끝나고 이동 동사가 없다. questions.json의
# instruction_following 30문장 전수 확인 결과 등장하는 이동 동사 전부를 포함한다.
_INSTRUCTION_PATTERN = re.compile(
    r"\b(go (to|near|between)|take the path|avoid(ing)? the path|"
    r"stop (at|by)|pass by|first,)\b",
    re.IGNORECASE,
)


def classify_mission(question: str) -> str:
    text = (question or "").strip()
    if _NUMERICAL_PATTERN.match(text):
        return MISSION_NUMERICAL
    if _INSTRUCTION_PATTERN.search(text):
        return MISSION_INSTRUCTION_FOLLOWING
    return MISSION_OBJECT_REFERENCE
