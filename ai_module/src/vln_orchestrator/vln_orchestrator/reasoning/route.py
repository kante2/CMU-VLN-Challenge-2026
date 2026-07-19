#!/usr/bin/env python3
"""Build the ordered (x, y) waypoint route for an instruction, honouring AVOID
regions by inserting detour waypoints around them.

Pure geometry on a located point list — unit-testable off-robot (no ROS). The
handler localises landmarks via the semantic map, then calls `build_route`. The
base autonomy does the fine obstacle avoidance between waypoints; this layer only
ensures the ORDERED path bends away from forbidden regions (the challenge penalises
passing through an avoid region), which a straight landmark-to-landmark line ignores.
"""
from __future__ import annotations

import math
import re

# leading prepositions/qualifiers on a region phrase ("the path between ...",
# "near the cabinet") that hide the actual landmark noun(s).
_REGION_LEAD = re.compile(
    r"^\s*(the\s+path\s+)?(between|near|around|by|through|past|along)\s+",
    re.IGNORECASE)


def _centroid(points):
    pts = [p for p in points if p]
    if not pts:
        return None
    return (sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts))


def _pt(inst):
    return (inst.bbox["cx"], inst.bbox["cy"]) if inst is not None else None


def locate_region(sm, phrase: str):
    """(x, y) centre of a via/avoid REGION phrase, or None. Handles "between A and
    B" (midpoint of A and B), "two X" (centroid of all X) and a plain landmark.
    Strips the leading relation word that would otherwise break landmark lookup."""
    p = _REGION_LEAD.sub("", phrase).strip()
    if " and " in p:                                   # "the chair and the screen"
        a, b = p.split(" and ", 1)
        return _centroid([_pt(sm.locate(a)), _pt(sm.locate(b))])
    m = re.match(r"^(?:the\s+)?two\s+(.+)$", p, re.IGNORECASE)
    if m:                                              # "two columns" -> all columns
        return _centroid([_pt(i) for i in sm.instances_of(m.group(1))])
    return _pt(sm.locate(p))

# how close (m) a segment may pass an avoid centre before we detour around it
AVOID_RADIUS_M = 1.2
# how far (m) past the avoid centre the detour waypoint is pushed
DETOUR_OFFSET_M = 1.8


def _seg_dist_and_proj(p, a, b):
    """Min distance from point p to segment a-b, and the projection parameter s."""
    ax, ay = a; bx, by = b; px, py = p
    sx, sy = bx - ax, by - ay
    L2 = sx * sx + sy * sy
    if L2 == 0.0:
        return math.hypot(px - ax, py - ay), 0.0
    s = max(0.0, min(1.0, ((px - ax) * sx + (py - ay) * sy) / L2))
    qx, qy = ax + s * sx, ay + s * sy
    return math.hypot(px - qx, py - qy), s


def _detour_point(a, b, c, offset):
    """A waypoint beside the a-b segment, pushed away from avoid centre c so the
    path rounds it. Offset is perpendicular to the segment, on the far side of c."""
    ax, ay = a; bx, by = b; cx, cy = c
    sx, sy = bx - ax, by - ay
    L = math.hypot(sx, sy) or 1.0
    nx, ny = -sy / L, sx / L                       # unit normal to the segment
    # closest point on the segment to c, then step away from c past it
    _, s = _seg_dist_and_proj(c, a, b)
    mx, my = ax + s * sx, ay + s * sy
    # choose the normal direction pointing AWAY from c
    if (mx - cx) * nx + (my - cy) * ny < 0:
        nx, ny = -nx, -ny
    return (cx + nx * offset, cy + ny * offset)


def build_route(waypoints, avoid_centers,
                avoid_radius=AVOID_RADIUS_M, offset=DETOUR_OFFSET_M):
    """Insert a detour waypoint into any ordered segment that passes within
    `avoid_radius` of an avoid centre. `waypoints` and `avoid_centers` are (x, y).
    Returns the ordered route. No avoid regions -> unchanged."""
    if not avoid_centers or len(waypoints) < 2:
        return list(waypoints)
    route = [waypoints[0]]
    for b in waypoints[1:]:
        a = route[-1]
        # detour for the nearest threatening avoid centre on this segment
        threat = None
        best = avoid_radius
        for c in avoid_centers:
            d, s = _seg_dist_and_proj(c, a, b)
            if d < best and 0.0 < s < 1.0:         # actually crosses near it
                best, threat = d, c
        if threat is not None:
            route.append(_detour_point(a, b, threat, offset))
        route.append(b)
    return route
