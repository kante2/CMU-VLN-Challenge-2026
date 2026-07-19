#!/usr/bin/env python3
"""Object-reference handler (/2) — locate one unique object, publish its 3D bbox.

Wired so far:
  1. Decompose the question into target/attributes/spatial/anchor/region
     (reasoning.decomposition — heuristic now, VLM when a key is present).

Pending perception on the Jazzy box (documented hooks below):
  2. Explore; build semantic map of candidate instances (SysNav semantic_mapping).
  3. Verify candidates with reasoning.verification.verify_candidate (VLM + image),
     select the single best match.
  4. publish_object_marker(bbox, label) — CUBE; its center doubles as a waypoint.

Scored by 3D bbox IoU with GT (0-2).
"""
from __future__ import annotations

from vln_orchestrator.handlers.base import BaseHandler
from vln_orchestrator.reasoning.decomposition import (
    Decomposition,
    heuristic_decompose,
    vlm_decompose,
)


class ObjectReferenceHandler(BaseHandler):
    def __init__(self, node) -> None:
        super().__init__(node)
        self._client = None  # lazy VLM client; only built if a key is present

    def _decompose(self, question: str) -> Decomposition:
        """VLM decomposition when an API key is available, else heuristic."""
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
            cands = sm.candidates(decomp)
            best = self._select(cands, decomp)
            if best is not None:
                self.node.publish_object_marker(
                    best.bbox,
                    label=decomp.target_object or best.label,
                    obj_id=best.id,
                )
                return
            self.log.warn("ObjectReferenceHandler: no candidate matched; fallback.")
        else:
            self.log.warn("ObjectReferenceHandler: no semantic map; fallback box.")
        self.fallback(question)

    def _select(self, candidates, decomp):
        """Pick the candidate matching the target. When a VLM client is available
        and disambiguation is needed (attributes specified or >1 candidate), verify
        each candidate's saved crop with the VLM; otherwise take the best geometric
        candidate."""
        if not candidates:
            return None
        need_vlm = bool(decomp.attributes) or len(candidates) > 1
        if self._client is None or not need_vlm:
            return candidates[0]

        from vln_orchestrator.reasoning.verification import (
            select_by_verification,
            verify_candidate,
        )

        def is_match(inst) -> bool:
            img = self._load_image(inst.image_path)
            if img is None:
                return False
            try:
                return verify_candidate(decomp, img, self._client).is_target
            except Exception as e:
                self.log.warn(f"verify failed for id={inst.id}: {e}")
                return False

        return select_by_verification(candidates, is_match)

    def _load_image(self, path: str):
        """Load a candidate's best crop (SysNav saves these as .npy BGR arrays)."""
        if not path:
            return None
        try:
            import numpy as np
            return np.load(path)
        except Exception as e:
            self.log.warn(f"could not load candidate image {path!r}: {e}")
            return None

    def fallback(self, question: str) -> None:
        # Always emit a marker (non-empty answer). Box at current vehicle pose.
        bbox = {
            "cx": self.node.vehicle_x, "cy": self.node.vehicle_y, "cz": 0.5,
            "l": 0.3, "w": 0.3, "h": 0.3, "heading": 0.0,
        }
        label = heuristic_decompose(question).target_object or "unknown_object"
        self.node.publish_object_marker(bbox, label=label)
