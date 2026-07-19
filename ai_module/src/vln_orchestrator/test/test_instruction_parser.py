#!/usr/bin/env python3
"""Validate parse_instruction() on all 30 instruction-following training commands.

Asserts structural properties (we don't have a labelled gold parse, but the
command surface gives strong invariants):
  - >= 2 ordered subgoals each, no empty landmarks
  - AVOID detected in exactly the 3 commands containing "avoid"/"avoiding"
  - VIA detected in every command with "take the path"/"go between"/"pass by"
  - terminal STOP recovered when the command says "stop at/by"
  - " and " inside a landmark ("between the TV and the door") is NOT split

Run: python3 ai_module/src/vln_orchestrator/test/test_instruction_parser.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))
from vln_orchestrator.reasoning.instruction_parser import (  # noqa: E402
    GoalKind, parse_instruction,
)

REPO = PKG.parents[2]
QUESTIONS = json.loads((REPO / "questions" / "questions.json").read_text())
CMDS = [q for e in QUESTIONS for q in e["questions"]["instruction_following"]]


def main() -> int:
    assert len(CMDS) == 30, f"expected 30 commands, got {len(CMDS)}"
    fails: list[str] = []

    for i, cmd in enumerate(CMDS):
        p = parse_instruction(cmd)
        kinds = [g.kind for g in p.subgoals]

        def fail(msg):
            fails.append(f"[{i}] {msg}\n     {cmd!r}\n     parsed={[(g.kind.value, g.landmark) for g in p.subgoals]}")

        if len(p.subgoals) < 2:
            fail(f"only {len(p.subgoals)} subgoal(s)")
        if any(not g.landmark for g in p.subgoals):
            fail("empty landmark present")

        has_avoid_word = ("avoid" in cmd.lower())
        if has_avoid_word and GoalKind.AVOID not in kinds:
            fail("AVOID word present but no AVOID subgoal")
        if not has_avoid_word and GoalKind.AVOID in kinds:
            fail("AVOID subgoal but no avoid word")

        if any(s in cmd.lower() for s in ("take the path", "go between", "pass by")):
            if GoalKind.VIA not in kinds:
                fail("expected a VIA subgoal")

        if ("stop at" in cmd.lower()) or ("stop by" in cmd.lower()):
            non_avoid = [g for g in p.subgoals if g.kind != GoalKind.AVOID]
            if not non_avoid or non_avoid[-1].kind != GoalKind.STOP:
                fail("terminal STOP not recovered")

    # explicit non-split check: command #15 (index 14) keeps "...and the door"
    p15 = parse_instruction(CMDS[14])
    stops = [g for g in p15.subgoals if g.kind == GoalKind.STOP]
    if not stops or "door" not in stops[-1].landmark:
        fails.append(f"[14] landmark split lost 'door': "
                     f"{[(g.kind.value, g.landmark) for g in p15.subgoals]}")

    print(f"instruction parser: {30 - len({int(f.split(']')[0][1:]) for f in fails})}"
          f"/30 commands clean")
    if fails:
        print("\nFAILURES:")
        for f in fails:
            print("  " + f)
        return 1

    # dump a few parses for eyeballing
    print("\nsample parses:")
    for i in (1, 3, 7, 13, 23):
        p = parse_instruction(CMDS[i])
        print(f"  [{i}] {CMDS[i]}")
        for g in p.subgoals:
            print(f"        {g.kind.value:5s} near={g.near!s:5s} {g.landmark!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
