#!/usr/bin/env python3
"""Geometric spatial-relation predicates over 3D object instances.

Pure standard-library geometry (no numpy), so it is unit-testable off-robot
with synthetic boxes AND imposes zero third-party dependencies at runtime in
the ROS container. Both the numerical counting filter (reasoning.counting) and
the object-reference candidate ranking call these to adjudicate relations like
"under the window", "on the table", "between the two columns", "farthest from
the columns".

Box convention matches the Marker fields in dummyVLM.cpp and eval_harness.scoring:
a box is a dict with center (cx, cy, cz) and full extents (l, w, h) in meters,
l along x, w along y, h along z. Heading is ignored (axis-aligned approximation),
consistent with the scoring proxy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Instance:
    """One detected object instance from the semantic map."""
    label: str
    bbox: dict                                   # cx,cy,cz,l,w,h[,heading]
    id: int = -1
    attributes: list[str] = field(default_factory=list)
    confidence: float = 1.0
    image_path: str = ""                         # best crop (.npy) for VLM verify


# --------------------------------------------------------------------------- #
# Low-level geometry on box dicts
# --------------------------------------------------------------------------- #
def center3(b: dict) -> tuple[float, float, float]:
    return (float(b["cx"]), float(b["cy"]), float(b["cz"]))


def center2(b: dict) -> tuple[float, float]:
    return (float(b["cx"]), float(b["cy"]))


def bounds(b: dict):
    """(min_xyz, max_xyz) corners of the axis-aligned box."""
    cx, cy, cz = center3(b)
    hl, hw, hh = b["l"] / 2.0, b["w"] / 2.0, b["h"] / 2.0
    return (cx - hl, cy - hw, cz - hh), (cx + hl, cy + hw, cz + hh)


def distance(a: dict, b: dict, planar: bool = True) -> float:
    """Center-to-center distance; planar (xy) by default."""
    ax, ay, az = center3(a)
    bx, by, bz = center3(b)
    if planar:
        return math.hypot(ax - bx, ay - by)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def xy_overlap(a: dict, b: dict) -> bool:
    """Do the two boxes overlap in the horizontal (xy) footprint?"""
    a_min, a_max = bounds(a)
    b_min, b_max = bounds(b)
    return (a_min[0] <= b_max[0] and a_max[0] >= b_min[0]
            and a_min[1] <= b_max[1] and a_max[1] >= b_min[1])


def footprint_iou(a: dict, b: dict) -> float:
    """Axis-aligned IoU of the two boxes' horizontal (xy) footprints, in [0,1].

    Heading is ignored (cheap approximation; good enough for duplicate detection).
    Used to spot the same physical object mapped as several un-merged instances.
    """
    a_min, a_max = bounds(a)
    b_min, b_max = bounds(b)
    ix = max(0.0, min(a_max[0], b_max[0]) - max(a_min[0], b_min[0]))
    iy = max(0.0, min(a_max[1], b_max[1]) - max(a_min[1], b_min[1]))
    inter = ix * iy
    area_a = (a_max[0] - a_min[0]) * (a_max[1] - a_min[1])
    area_b = (b_max[0] - b_min[0]) * (b_max[1] - b_min[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# Binary relations: relation(target, anchor) -> bool
# --------------------------------------------------------------------------- #
NEAR_THRESH_M = 1.5          # default planar radius for "near"
ON_GAP_M = 0.25              # vertical tolerance for "resting on top of"
# geometry tolerances (NOT scene tuning): real furniture boxes mean an object
# "on" a sofa rests on the seat (inside the sofa envelope, not above its top),
# and a wall picture "above" a bed sits beside, not over, the bed footprint.
ON_VERTICAL_GAP = 0.3        # slack on the resting-within-envelope band
VERTICAL_PROXIMITY_M = 1.5   # planar reach for above/below when footprints miss


def near(t: dict, a: dict, thresh: float = NEAR_THRESH_M) -> bool:
    return distance(t, a, planar=True) <= thresh


def _horiz_near(t: dict, a: dict, thresh: float = VERTICAL_PROXIMITY_M) -> bool:
    """Footprints overlap OR centers are within `thresh` (xy). Lets wall-mounted
    objects count as above/below nearby furniture whose footprint they miss."""
    return xy_overlap(t, a) or distance(t, a, planar=True) <= thresh


def below(t: dict, a: dict) -> bool:
    """target below anchor: target center under anchor's center, horizontally near
    (so "below the window", "under the picture")."""
    return center3(t)[2] < center3(a)[2] and _horiz_near(t, a)


def above(t: dict, a: dict) -> bool:
    return center3(t)[2] > center3(a)[2] and _horiz_near(t, a)


# "under" is the same relation as "below" for our purposes
under = below


def on(t: dict, a: dict, gap: float = ON_VERTICAL_GAP) -> bool:
    """target rests on/within anchor: target horizontal CENTER inside the anchor
    footprint AND its bottom within the anchor's vertical envelope (+gap). Handles
    objects resting on a seat/shelf inside the bbox, e.g. a pillow on a sofa or a
    monitor on a table — not just things perched on the very top edge."""
    a_min, a_max = bounds(a)
    t_bottom = bounds(t)[0][2]
    within_envelope = a_min[2] - gap <= t_bottom <= a_max[2] + gap
    # footprint overlap (not center-strict): a pillow whose centre sits at the
    # sofa's seat edge still counts as on it.
    return xy_overlap(t, a) and within_envelope


def between(t: dict, a: dict, b: dict, tol: float = 1.0) -> bool:
    """target lies between anchors a and b in the xy plane: its perpendicular
    distance to the segment a-b is within tol AND its projection falls inside
    the segment (e.g. "between the two columns")."""
    px, py = center2(a)
    qx, qy = center2(b)
    xx, xy = center2(t)
    sx, sy = qx - px, qy - py
    seg_len2 = sx * sx + sy * sy
    if seg_len2 == 0.0:
        return distance(t, a, planar=True) <= tol
    s = ((xx - px) * sx + (xy - py) * sy) / seg_len2     # projection parameter
    if not (0.0 <= s <= 1.0):
        return False
    proj_x, proj_y = px + s * sx, py + s * sy
    return math.hypot(xx - proj_x, xy - proj_y) <= tol


# --------------------------------------------------------------------------- #
# Superlatives: pick one target from candidates relative to anchor(s)
# --------------------------------------------------------------------------- #
def closest_to(candidates: list[Instance], anchor: dict) -> Instance | None:
    if not candidates:
        return None
    return min(candidates, key=lambda c: distance(c.bbox, anchor, planar=True))


def farthest_from(candidates: list[Instance], anchor: dict) -> Instance | None:
    if not candidates:
        return None
    return max(candidates, key=lambda c: distance(c.bbox, anchor, planar=True))


# --------------------------------------------------------------------------- #
# Relation phrase dispatch
# --------------------------------------------------------------------------- #
# Maps a normalized relation phrase to a binary predicate (target, anchor)->bool.
_BINARY = {
    "near": near,
    "next to": near,
    "by": near,
    "beside": near,
    "below": below,
    "under": under,
    "underneath": under,
    "beneath": under,
    "above": above,
    "over": above,
    "on": on,
    "on top of": on,
}

# superlative phrases -> selector(candidates, anchor)->Instance
_SUPERLATIVE = {
    "closest to": closest_to,
    "nearest to": closest_to,
    "nearest": closest_to,
    "closest": closest_to,
    "farthest from": farthest_from,
    "furthest from": farthest_from,
    "farthest": farthest_from,
    "furthest": farthest_from,
}


def normalize_relation(phrase: str) -> str:
    return (phrase or "").strip().lower().rstrip(",")


def is_binary_relation(phrase: str) -> bool:
    return normalize_relation(phrase) in _BINARY


def is_superlative_relation(phrase: str) -> bool:
    return normalize_relation(phrase) in _SUPERLATIVE


def is_between_relation(phrase: str) -> bool:
    return normalize_relation(phrase) == "between"


def binary_predicate(phrase: str):
    return _BINARY.get(normalize_relation(phrase))


def superlative_selector(phrase: str):
    return _SUPERLATIVE.get(normalize_relation(phrase))
