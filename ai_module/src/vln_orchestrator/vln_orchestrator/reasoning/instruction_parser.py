#!/usr/bin/env python3
"""Parse an instruction-following command into an ORDERED list of sub-goals.

The official /6 scorer rewards visiting the right landmarks in the right order,
passing through required "via" regions, and avoiding forbidden regions. So we
decompose the command into ordered SubGoals of four kinds:

  GOTO   navigate to/near a landmark            ("go to X", "go near X", "to X")
  STOP   terminal landmark (final waypoint)     ("stop at X", "stop by X")
  VIA    pass through a region between/near      ("take the path between A and B",
         landmarks                                "go between A and B", "pass by X")
  AVOID  forbidden region                        ("avoid[ing] the path between A and B")

This is pure logic (no ROS / VLM) so it can be unit-tested off-robot against all
30 instruction-following training questions. A VLM refinement can override this
later, but the rule-based parser is a strong, debuggable default.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class GoalKind(str, Enum):
    GOTO = "goto"
    STOP = "stop"
    VIA = "via"
    AVOID = "avoid"


@dataclass
class SubGoal:
    kind: GoalKind
    landmark: str                 # the object/region phrase
    near: bool = False            # "go near"/"path near" vs "go to"/"between"


@dataclass
class ParsedInstruction:
    subgoals: list[SubGoal]
    raw: str

    @property
    def avoid_regions(self) -> list[SubGoal]:
        return [g for g in self.subgoals if g.kind == GoalKind.AVOID]

    @property
    def ordered_waypoints(self) -> list[SubGoal]:
        """GOTO/STOP/VIA in order — the things the robot must visit/traverse."""
        return [g for g in self.subgoals if g.kind != GoalKind.AVOID]


# split the command into action clauses. We split on sequence connectors, but
# only on an "and" that introduces a new action verb (so "between A and B" is
# preserved). Order matters: try multi-word connectors first.
_SPLIT = re.compile(
    r"""
      \s*,?\s*and\s+finally\s*,?\s+              # "and finally,"
    | \s*,?\s*and\s+then\s+                       # "and then"
    | \s*,?\s*then\s*,?\s+                        # "then", "then,", ", then"
    | \s*,?\s+and\s+(?=(?:go|stop|take|pass|avoid)\b)   # ", and go/stop/..."
    | \s*,\s+(?=(?:go|stop|take|pass|avoid|avoiding)\b) # ", pass by", ", go", ", avoiding"
    """,
    re.IGNORECASE | re.VERBOSE,
)

_LEAD_FIRST = re.compile(r"^\s*first\s*,?\s*", re.IGNORECASE)

# verb prefixes -> (kind, near, regex). Checked in order; first match wins.
_CLAUSE_PATTERNS = [
    (GoalKind.AVOID, False, re.compile(r"^(?:avoiding|avoid)\s+(?:the\s+path\s+)?(.*)$", re.I)),
    (GoalKind.VIA,   False, re.compile(r"^take\s+the\s+path\s+between\s+(.*)$", re.I)),
    (GoalKind.VIA,   True,  re.compile(r"^take\s+the\s+path\s+near\s+(.*)$", re.I)),
    (GoalKind.VIA,   False, re.compile(r"^take\s+the\s+path\s+(.*)$", re.I)),
    (GoalKind.VIA,   False, re.compile(r"^go\s+between\s+(.*)$", re.I)),
    (GoalKind.VIA,   True,  re.compile(r"^pass\s+by\s+(.*)$", re.I)),
    (GoalKind.VIA,   False, re.compile(r"^pass\s+(?:the\s+)?(.*)$", re.I)),
    (GoalKind.STOP,  False, re.compile(r"^stop\s+at\s+(.*)$", re.I)),
    (GoalKind.STOP,  False, re.compile(r"^stop\s+by\s+(.*)$", re.I)),
    (GoalKind.GOTO,  True,  re.compile(r"^go\s+near\s+(.*)$", re.I)),
    (GoalKind.GOTO,  False, re.compile(r"^go\s+to\s+(.*)$", re.I)),
    (GoalKind.GOTO,  False, re.compile(r"^to\s+(.*)$", re.I)),          # "then to the X"
    (GoalKind.GOTO,  False, re.compile(r"^go\s+(.*)$", re.I)),          # bare "go ..."
]


def _clean_landmark(text: str) -> str:
    text = text.strip().strip(".,").strip()
    text = re.sub(r"^the\s+", "", text, flags=re.I)
    return text


def _classify_clause(clause: str) -> SubGoal | None:
    c = clause.strip().strip(".,").strip()
    if not c:
        return None
    for kind, near, rx in _CLAUSE_PATTERNS:
        m = rx.match(c)
        if m:
            return SubGoal(kind=kind, landmark=_clean_landmark(m.group(1)), near=near)
    # no verb matched: treat as a GOTO landmark (continuation phrase)
    return SubGoal(kind=GoalKind.GOTO, landmark=_clean_landmark(c), near=False)


def _split_via_destination(g: SubGoal) -> list[SubGoal]:
    """"take the path near the wardrobe doors to the flowers ..." -> VIA + GOTO."""
    if g.kind != GoalKind.VIA:
        return [g]
    m = re.match(r"^(.*?)\s+to\s+(?:the\s+)?(.+)$", g.landmark, re.I)
    if m and m.group(1).strip():
        return [
            SubGoal(GoalKind.VIA, _clean_landmark(m.group(1)), near=g.near),
            SubGoal(GoalKind.GOTO, _clean_landmark(m.group(2)), near=False),
        ]
    return [g]


def parse_instruction(command: str) -> ParsedInstruction:
    text = _LEAD_FIRST.sub("", command.strip())
    clauses = _SPLIT.split(text)
    subgoals: list[SubGoal] = []
    for clause in clauses:
        g = _classify_clause(clause)
        if g is None:
            continue
        subgoals.extend(_split_via_destination(g))
    return ParsedInstruction(subgoals=subgoals, raw=command)
