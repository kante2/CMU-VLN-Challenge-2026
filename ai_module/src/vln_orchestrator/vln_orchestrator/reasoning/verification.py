#!/usr/bin/env python3
"""VLM verification: does a candidate object match the decomposed target?

Prompt adapted from SysNav vlm_node.process_target_object_query /
process_target_object_spatial_query (BSD-3, Haokun Zhu). Runtime-only (needs a
close-up image of the candidate from the semantic map + a VLM), so this is a thin
interface the object-reference / numerical handlers call once perception is wired
on the Jazzy box. No offline unit test (requires images + an API key).
"""
from __future__ import annotations

from dataclasses import dataclass

from vln_orchestrator.reasoning.decomposition import Decomposition

_VERIFY_PROMPT = """
You are given an instruction describing a target object (type, attributes,
spatial/room conditions) and a close-up image of a candidate object, optionally
with surrounding-viewpoint images. Decide whether the candidate matches the
target. Consider attribute, spatial, and room conditions. If any condition is not
satisfied or cannot be clearly determined, return is_target=false. Return strictly
valid JSON: {"is_target": bool, "reason": str}.
"""


@dataclass
class VerificationResult:
    is_target: bool
    reason: str


def select_by_verification(candidates, is_match, max_checks: int = 6):
    """Pick the first candidate that `is_match(candidate)` accepts, checking at
    most `max_checks` (latency cap for the 10-min budget). If none verify, fall
    back to the best geometric candidate (candidates[0]).

    `is_match` is a callable candidate -> bool. Kept separate from the VLM call so
    the selection policy is unit-testable with a mock (no API key / images).
    """
    cands = list(candidates)
    if not cands:
        return None
    for inst in cands[:max_checks]:
        if is_match(inst):
            return inst
    return cands[0]


def count_by_verification(candidates, is_match, max_checks: int = 12) -> int:
    """Count how many candidates `is_match` accepts (VLM-verified count), checking
    at most `max_checks` (latency cap for the 10-min budget). `is_match` is a
    callable candidate -> bool; kept separate from the VLM call so the count policy
    is unit-testable with a mock. Used by the numerical handler to adjudicate
    color/spatial constraints visually instead of by brittle geometry."""
    return sum(1 for inst in list(candidates)[:max_checks] if is_match(inst))


def verify_candidate(
    decomp: Decomposition,
    candidate_image_bgr,
    client,
    surrounding_images_bgr: list | None = None,
) -> VerificationResult:
    """Return whether `candidate_image_bgr` matches the decomposed target.

    `client` is a VLMClient. Images are OpenCV BGR arrays. Surrounding viewpoints
    help adjudicate spatial relations (mirrors SysNav's spatial query).
    """
    from pydantic import BaseModel
    from vln_orchestrator.reasoning.vlm_client import encode_image_jpg

    class _Schema(BaseModel):
        reason: str
        is_target: bool

    content = [
        client.text(
            f"Instruction: {decomp.as_instruction()} "
            f"{{target_object:{decomp.target_object}, "
            f"attributes:{decomp.attributes or 'None'}, "
            f"spatial:{decomp.spatial_relation or 'None'}, "
            f"anchor:{decomp.anchor_object or 'None'}, "
            f"region:{decomp.region or 'None'}}} "
            "Determine whether the object in the image matches this target."
        ),
        client.image_b64(encode_image_jpg(candidate_image_bgr)),
    ]
    for img in (surrounding_images_bgr or []):
        content.append(client.image_b64(encode_image_jpg(img)))

    parsed = client.parse(system=_VERIFY_PROMPT, content=content,
                          response_format=_Schema, lite=False)
    return VerificationResult(is_target=parsed.is_target, reason=parsed.reason)
