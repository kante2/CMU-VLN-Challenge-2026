#!/usr/bin/env python3
"""Unit tests for reasoning.route — avoid-detour + region locating.

Pure geometry / synthetic semantic map; no ROS, no VLA-3D.
Run: python3 ai_module/src/vln_orchestrator/test/test_route.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from vln_orchestrator.reasoning.route import build_route, locate_region  # noqa: E402
from vln_orchestrator.reasoning.spatial import Instance  # noqa: E402
from vln_orchestrator.perception.semantic_map_adapter import SemanticMap  # noqa: E402


def box(cx, cy):
    return {"cx": cx, "cy": cy, "cz": 0.5, "l": 0.4, "w": 0.4, "h": 0.4, "heading": 0.0}


def test_detour_inserted():
    wps = [(0.0, 0.0), (4.0, 0.0)]
    # avoid region sitting on the straight segment -> must detour around it
    route = build_route(wps, [(2.0, 0.1)])
    assert len(route) == 3, route
    # the inserted point is pushed off the line (away from the avoid centre)
    assert abs(route[1][1]) > 1.0
    print("✓ detour inserted around avoid region on the path")


def test_no_detour_when_clear():
    wps = [(0.0, 0.0), (4.0, 0.0)]
    # avoid region far from the segment -> unchanged
    assert build_route(wps, [(2.0, 9.0)]) == wps
    assert build_route(wps, []) == wps                 # no avoid regions
    print("✓ no detour when path is clear")


def test_locate_region():
    sm = SemanticMap()
    sm._instances = [
        Instance("chair", box(0, 0), id=1),
        Instance("screen", box(2, 0), id=2),
        Instance("column", box(-3, 0), id=3),
        Instance("column", box(3, 0), id=4),
    ]
    # "between A and B" -> midpoint
    c = locate_region(sm, "the path between the chair and the screen")
    assert c is not None and abs(c[0] - 1.0) < 1e-6 and abs(c[1]) < 1e-6, c
    # "two X" -> centroid of all X
    c = locate_region(sm, "the two columns")
    assert c is not None and abs(c[0]) < 1e-6, c
    # plain landmark with a leading preposition
    c = locate_region(sm, "near the chair")
    assert c is not None and abs(c[0]) < 1e-6 and abs(c[1]) < 1e-6, c
    print("✓ locate_region (between / two / leading-prep)")


if __name__ == "__main__":
    test_detour_inserted()
    test_no_detour_when_clear()
    test_locate_region()
    print("\nAll route tests passed.")
