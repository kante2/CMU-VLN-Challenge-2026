#!/usr/bin/env python3
"""Validate heuristic_decompose() target extraction on the training questions.

We check that the union of (attributes + target_object) covers the expected
target phrase's words. This tolerates the attribute/compound-noun split
(e.g. "potted plant" -> attributes=["potted"], target="plant").

Run: python3 ai_module/src/vln_orchestrator/test/test_decomposition.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))
from vln_orchestrator.reasoning.decomposition import heuristic_decompose  # noqa: E402

REPO = PKG.parents[2]
QUESTIONS = json.loads((REPO / "questions" / "questions.json").read_text())

# expected target phrase per object-reference question, in JSON order
EXPECTED_OBJREF = [
    # arabic_room
    "pillow", "wall lamp",
    # chinese_room
    "bowl", "pillow",
    # home_building_1
    "clock", "bowl",
    # home_building_2
    "lamp", "speaker",
    # hotel_room_1
    "bedside table", "picture",
    # hotel_room_2
    "flowers", "picture",
    # japanese_room
    "lantern", "red pillow",
    # livingroom_1
    "vase", "pillow",
    # livingroom_2
    "stool", "pillow",
    # livingroom_3
    "potted plant", "vase",
    # livingroom_4
    "picture", "fossil decoration",
    # loft
    "blue chair", "potted plant",
    # office_1
    "potted plant", "paper cup",
    # office_2
    "computer monitor", "box",
    # studio
    "vase", "beer bottle",
]


def main() -> int:
    objref_qs = [q for e in QUESTIONS for q in e["questions"]["object_reference"]]
    assert len(objref_qs) == len(EXPECTED_OBJREF) == 30, "objref count mismatch"

    fails = []
    for q, expected in zip(objref_qs, EXPECTED_OBJREF):
        d = heuristic_decompose(q)
        union = set((" ".join(d.attributes) + " " + d.target_object).split())
        if not set(expected.split()).issubset(union):
            fails.append((q, expected, d.attributes, d.target_object))

    n = len(objref_qs)
    print(f"object-reference target extraction: {n - len(fails)}/{n} covered")
    for q, exp, attrs, tgt in fails:
        print(f"  MISS expected={exp!r} got attrs={attrs} target={tgt!r}\n      {q!r}")

    # numerical: target should be non-empty for every numerical question
    num_qs = [e["questions"]["numerical"][0] for e in QUESTIONS]
    empty = [q for q in num_qs if not heuristic_decompose(q).target_object]
    print(f"numerical target non-empty: {len(num_qs) - len(empty)}/{len(num_qs)}")
    for q in empty:
        print(f"  EMPTY target for: {q!r}")

    # relation extraction: "How many X are <rel> Y?" must recover the relation
    # and a clean anchor (the leading copula + '?' used to swallow them).
    rel_checks = [
        ("How many sofas are below a window?", "below", "window"),
        ("How many pillows are on the sofa?", "on", "sofa"),
        ("How many pictures are above the bed?", "above", "bed"),
    ]
    rel_fail = 0
    for q, rel, anc in rel_checks:
        d = heuristic_decompose(q)
        if d.spatial_relation != rel or anc not in d.anchor_object:
            rel_fail += 1
            print(f"  REL MISS {q!r} -> rel={d.spatial_relation!r} anchor={d.anchor_object!r}")
    print(f"numerical relation extraction: {len(rel_checks) - rel_fail}/{len(rel_checks)}")

    # identity adjective stays on the noun (not dropped as a color attribute)
    d = heuristic_decompose("How many potted plants are on a table?")
    id_ok = "potted" not in d.attributes
    print(f"identity-adjective kept on noun: {id_ok}")

    return 1 if (fails or empty or rel_fail or not id_ok) else 0


if __name__ == "__main__":
    raise SystemExit(main())
