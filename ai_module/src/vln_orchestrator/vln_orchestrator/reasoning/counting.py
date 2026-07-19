#!/usr/bin/env python3
"""Counting logic for the numerical (/1) question type.

Given the decomposed question and the set of 3D object instances from the
semantic map, return how many instances of the target class satisfy the
attribute and spatial-relation constraints.

The geometry/filter core is pure (numpy-only) and unit-testable off-robot with
synthetic instance lists. Only the *source* of instances (the live semantic map)
needs the Jazzy box. Scored 0/1 by exact match, so we bias toward the constraint
the question states rather than over-counting: an unconstrained "how many sofas"
returns every sofa; a constrained "sofas below a window" returns only those that
pass the relation against a window anchor.
"""
from __future__ import annotations

from vln_orchestrator.reasoning.decomposition import Decomposition
from vln_orchestrator.reasoning.spatial import (
    Instance,
    between,
    binary_predicate,
    distance,
    footprint_iou,
    is_between_relation,
    is_binary_relation,
    is_superlative_relation,
    superlative_selector,
)

# Duplicate-instance thresholds. The semantic map can carry the same physical
# object as several un-merged instances (same class, seen from different
# viewpoints) -> raw counts over-report (e.g. 21 chairs where there are 8). Two
# same-class instances are treated as one if their footprints overlap enough OR
# their centers are nearly coincident (covers degenerate/empty-cloud boxes).
DEDUP_IOU = 0.3
DEDUP_CENTER_M = 0.25


def _singular(word: str) -> str:
    w = word.lower().strip()
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if w.endswith("ses") or w.endswith("xes") or w.endswith("zes"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def label_matches(label: str, target: str) -> bool:
    """Loose noun match tolerant of plurals and multi-word labels.

    "sofas" matches "sofa", "potted plant" matches "plant", "office chair"
    matches "chair". Empty target matches nothing.
    """
    if not target:
        return False
    lab = {_singular(t) for t in label.lower().split()}
    tgt = {_singular(t) for t in target.lower().split()}
    # match if every target word appears (singularized) in the label, or vice
    # versa for short compound labels.
    return tgt.issubset(lab) or lab.issubset(tgt) or bool(lab & tgt)


def filter_by_label(instances: list[Instance], target: str) -> list[Instance]:
    return [o for o in instances if label_matches(o.label, target)]


def dedup_instances(
    instances: list[Instance],
    iou_thresh: float = DEDUP_IOU,
    center_thresh: float = DEDUP_CENTER_M,
) -> list[Instance]:
    """Collapse duplicate detections of the same physical object.

    Greedy NMS within each (singularized) label group: an instance is a duplicate
    of an already-kept one if their footprint IoU exceeds `iou_thresh` or their
    centers are within `center_thresh` metres. Distinct adjacent objects (e.g.
    dining chairs ~0.5 m apart with low overlap) are preserved; only near-
    coincident re-observations are merged. Order-stable.
    """
    kept: list[Instance] = []
    for o in instances:
        dup = False
        for r in kept:
            if not label_matches(o.label, r.label):
                continue
            if (footprint_iou(o.bbox, r.bbox) > iou_thresh
                    or distance(o.bbox, r.bbox) < center_thresh):
                dup = True
                break
        if not dup:
            kept.append(o)
    return kept


def filter_by_attributes(
    instances: list[Instance], attributes: list[str]
) -> list[Instance]:
    """Best-effort attribute filter.

    Attributes (color/size/material) usually need a VLM/visual check, which is
    not available offline. So we only filter instances that carry recorded
    attributes; instances with none are kept (the perception/VLM stage refines
    them later). This keeps recall safe and avoids dropping valid objects just
    because attributes were not annotated.
    """
    if not attributes:
        return instances
    wanted = {a.lower() for a in attributes}
    out = []
    for o in instances:
        if not o.attributes:
            out.append(o)                          # unknown -> keep
        elif wanted & {a.lower() for a in o.attributes}:
            out.append(o)
    return out


def matching_instances(
    decomp: Decomposition, instances: list[Instance]
) -> list[Instance]:
    """The deduped instances of the target class satisfying the decomposed
    constraints (the set `count_matching` counts). Exposed so the numerical
    handler can VLM-verify each candidate visually before counting (color +
    spatial truth the geometry only approximates).

    Resolution order:
      1. target class + attributes
      2. spatial relation vs. anchor instances, if any:
         - binary ("below/on/near ...") -> keep targets passing it vs ANY anchor.
         - "between" -> keep targets between SOME pair of anchors.
         - superlative ("closest/farthest ...") -> ONE object (the best match).
    """
    targets = dedup_instances(
        filter_by_attributes(
            filter_by_label(instances, decomp.target_object), decomp.attributes
        )
    )
    if not targets:
        return []

    relation = decomp.spatial_relation
    if not relation or not decomp.anchor_object:
        return targets

    anchors = filter_by_label(instances, decomp.anchor_object)
    if not anchors:
        # constraint references an anchor we never found; fall back to the
        # unconstrained class set rather than asserting empty.
        return targets

    if is_superlative_relation(relation):
        selector = superlative_selector(relation)
        best = selector(targets, anchors[0].bbox)
        return [best] if best is not None else []

    if is_between_relation(relation):
        return [
            t for t in targets
            if any(
                between(t.bbox, anchors[i].bbox, anchors[j].bbox)
                for i in range(len(anchors))
                for j in range(i + 1, len(anchors))
            )
        ]

    if is_binary_relation(relation):
        pred = binary_predicate(relation)
        return [t for t in targets if any(pred(t.bbox, a.bbox) for a in anchors)]

    # unknown relation phrase -> don't guess a filter; return class set.
    return targets


def count_matching(decomp: Decomposition, instances: list[Instance]) -> int:
    """Count instances of the target class satisfying the decomposed constraints
    (geometric). `len(matching_instances(...))`."""
    return len(matching_instances(decomp, instances))
