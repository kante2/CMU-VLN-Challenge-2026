#!/usr/bin/env python3
"""Numerical handler (/1) — answer "how many / count" questions.

Wired so far:
  1. Decompose the question into target/attributes/spatial/anchor/region
     (reasoning.decomposition — heuristic now, VLM when a key is present).
  2. Counting logic (reasoning.counting) filters semantic-map instances by
     class + attributes + spatial relation and returns the count.

Pending perception on the Jazzy box (documented hook below):
  - Source the live 3D instances from the semantic map (SysNav semantic_mapping)
    and the VLM attribute check; feed them into count_matching().

Scored 0/1 by exact match, so calibration of the count matters more than recall.
"""
from __future__ import annotations

from vln_orchestrator.handlers.base import BaseHandler
from vln_orchestrator.reasoning.counting import count_matching, matching_instances
from vln_orchestrator.reasoning.decomposition import (
    Decomposition,
    heuristic_decompose,
    vlm_decompose,
)


class NumericalHandler(BaseHandler):
    def __init__(self, node) -> None:
        super().__init__(node)
        self._client = None  # lazy VLM client; only built if a key is present

    def _decompose(self, question: str) -> Decomposition:
        from vln_orchestrator.reasoning.vlm_client import answer_key_available
        if answer_key_available():
            try:
                if self._client is None:
                    from vln_orchestrator.reasoning.vlm_client import VLMClient
                    self._client = VLMClient()
                decomp = vlm_decompose(question, self._client)
                self.log.info("decompose: used VLM answer path.")
                return decomp
            except Exception as e:
                self.log.warn(f"VLM decompose failed ({e}); using heuristic.")
        return heuristic_decompose(question)

    def handle(self, question: str) -> None:
        decomp = self._decompose(question)
        self.log.info(
            f"decomposed -> target={decomp.target_object!r} "
            f"attrs={decomp.attributes} rel={decomp.spatial_relation!r} "
            f"anchor={decomp.anchor_object!r}"
        )

        sm = getattr(self.node, "semantic_map", None)
        if sm is not None and len(sm):
            count = self._count(decomp, sm.all_instances())
            self.node.publish_numerical(count)
            return
        self.log.warn("NumericalHandler: no semantic map; using fallback.")
        self.fallback(question)

    def _count(self, decomp: Decomposition, instances) -> int:
        """Geometric count, refined by per-candidate VLM verification when a key is
        available AND the question carries a visual/spatial constraint (color or a
        relation) the geometry only approximates. Falls back to the geometric count
        if no client, no constraint, or any verification error."""
        cands = matching_instances(decomp, instances)
        need_vlm = bool(decomp.attributes) or bool(decomp.spatial_relation)
        if self._client is None or not need_vlm or not cands:
            return len(cands)
        from vln_orchestrator.reasoning.verification import (
            count_by_verification,
            verify_candidate,
        )

        def is_match(inst) -> bool:
            img = self._load_image(getattr(inst, "image_path", ""))
            if img is None:
                return True   # no crop to check -> trust the geometric candidate
            try:
                return verify_candidate(decomp, img, self._client).is_target
            except Exception as e:
                self.log.warn(f"count verify failed id={inst.id}: {e}")
                return True
        try:
            n = count_by_verification(cands, is_match)
            self.log.info(f"count: VLM-verified {n}/{len(cands)} candidates.")
            return n
        except Exception as e:
            self.log.warn(f"VLM count failed ({e}); geometric count.")
            return len(cands)

    def _load_image(self, path: str):
        if not path:
            return None
        try:
            import numpy as np
            return np.load(path)
        except Exception:
            return None

    def fallback(self, question: str) -> None:
        # A neutral non-zero guess. The training-set count distribution skews to
        # small integers (1-8, mode ~2); 2 is the safest single guess. Replaced by
        # the real count_matching() output as soon as the instance source lands.
        self.node.publish_numerical(2)
