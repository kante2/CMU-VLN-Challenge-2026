#!/usr/bin/env python3
"""Validate question_router.classify() against ALL 75 training questions.

The ground-truth type of each question is given by which key it sits under in
questions/questions.json (numerical / object_reference / instruction_following).
We require 100% accuracy on the training set.

Run: python3 ai_module/src/vln_orchestrator/test/test_question_router.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# allow running directly without installing the package
PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))
from vln_orchestrator.question_router import QType, classify  # noqa: E402

REPO = PKG.parents[2]  # ai_module/src/vln_orchestrator -> repo root
QUESTIONS = json.loads((REPO / "questions" / "questions.json").read_text())

KEY_TO_QTYPE = {
    "numerical": QType.NUMERICAL,
    "object_reference": QType.OBJECT_REFERENCE,
    "instruction_following": QType.INSTRUCTION_FOLLOWING,
}


def main() -> int:
    total = 0
    wrong = []
    for entry in QUESTIONS:
        scene = entry["scene"]
        for key, expected in KEY_TO_QTYPE.items():
            for q in entry["questions"].get(key, []):
                total += 1
                got = classify(q)
                if got != expected:
                    wrong.append((scene, expected.value, got.value, q))

    print(f"Classified {total} training questions; {total - len(wrong)} correct.")
    if wrong:
        print(f"\n{len(wrong)} MISCLASSIFIED:")
        for scene, exp, got, q in wrong:
            print(f"  [{scene}] expected={exp} got={got}\n      {q!r}")
        return 1
    print("All training questions classified correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
