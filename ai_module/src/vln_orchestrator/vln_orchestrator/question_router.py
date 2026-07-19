#!/usr/bin/env python3
"""Question classification for the CMU VLN Challenge 2026.

The base `dummyVLM.cpp` routes purely on the prefixes "Find"/"How many". That is
INSUFFICIENT for the real question set in questions/questions.json, which also
contains:
  - numerical questions starting with "Count ..." (e.g. chinese_room Q1)
  - object-reference questions starting with "The ..." (e.g. japanese_room Q2/Q3,
    loft Q2) with no leading "Find"

Both would be mis-routed to instruction-following by the naive prefix check. This
module classifies robustly and is validated against all 75 training questions
(see test/test_question_router.py). Pure-python, no ROS dependency, so it can be
unit-tested off-robot.
"""
from __future__ import annotations

import re
from enum import Enum


class QType(str, Enum):
    NUMERICAL = "numerical"
    OBJECT_REFERENCE = "object_reference"
    INSTRUCTION_FOLLOWING = "instruction_following"


# Numerical: asks for a count. Strong, unambiguous signals.
_NUMERICAL_PATTERNS = [
    r"\bhow many\b",
    r"\bcount\b",
    r"\bnumber of\b",
]

# Instruction-following: imperative navigation. The command tells the robot to
# move along a path / through/around landmarks, often multi-step.
_INSTRUCTION_PATTERNS = [
    r"\bgo to\b",
    r"\bgo near\b",
    r"\bgo between\b",
    r"\btake the path\b",
    r"\bavoid(ing)?\b",
    r"\bstop at\b",
    r"\bstop by\b",
    r"\bpass by\b",
    r"\bpass the\b",
    r"^\s*first,",
    r"^\s*go\b",
    r"^\s*take\b",
    r"\bthen\b.*\b(go|stop|take|to)\b",
]

# Object-reference: locate/identify a single object. Often "Find ..." but the
# dataset also phrases these as a bare noun phrase "The <obj> that is closest ...".
_OBJECT_REF_PATTERNS = [
    r"^\s*find\b",
    r"^\s*locate\b",
    r"^\s*identify\b",
    r"^\s*the\b",  # bare definite noun phrase, e.g. "The red pillow closest to ..."
]

_NUM_RE = [re.compile(p, re.IGNORECASE) for p in _NUMERICAL_PATTERNS]
_INS_RE = [re.compile(p, re.IGNORECASE) for p in _INSTRUCTION_PATTERNS]
_OBJ_RE = [re.compile(p, re.IGNORECASE) for p in _OBJECT_REF_PATTERNS]


def classify(question: str) -> QType:
    """Classify a challenge question into one of the three answer types.

    Precedence: numerical > instruction-following > object-reference. Numerical is
    checked first (its cues are unambiguous), then instruction-following (the
    presence of motion verbs/path language), and object-reference is the default
    for "find/locate" or bare definite noun phrases.
    """
    q = question.strip()

    if any(r.search(q) for r in _NUM_RE):
        return QType.NUMERICAL

    if any(r.search(q) for r in _INS_RE):
        return QType.INSTRUCTION_FOLLOWING

    if any(r.match(q) for r in _OBJ_RE):
        return QType.OBJECT_REFERENCE

    # Fallback: a bare noun phrase with a spatial relation but no motion verb is
    # almost certainly an object-reference. Default there rather than misfiring a
    # navigation command.
    return QType.OBJECT_REFERENCE
