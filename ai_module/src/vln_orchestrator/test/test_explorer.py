#!/usr/bin/env python3
"""Unit tests for exploration.ExplorationController — frontier selection + the
done criterion on synthetic (x, y) terrain/visited inputs. No ROS needed.

Run: python3 ai_module/src/vln_orchestrator/test/test_explorer.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from vln_orchestrator.exploration.explorer import ExplorationController  # noqa: E402


def test_no_terrain_not_covered():
    e = ExplorationController()
    # no terrain received yet -> we can't conclude coverage
    assert e.is_covered(0.0, 0.0) is False
    assert e.next_goal(0.0, 0.0) is None
    print("✓ no terrain -> not covered (won't answer prematurely)")


def test_nearest_frontier():
    e = ExplorationController(frontier_clearance=2.0, max_step=100.0)
    e.set_terrain([(3.0, 0.0), (10.0, 0.0), (-8.0, 0.0)])
    e.mark_visited(0.0, 0.0)            # only the origin visited
    g = e.next_goal(0.0, 0.0)
    # all three are frontiers; nearest to the current pose is (3,0)
    assert g is not None and abs(g[0] - 3.0) < 1e-6 and abs(g[1]) < 1e-6
    print("✓ picks nearest unexplored frontier (systematic local coverage)")


def test_blocked_frontier_excluded():
    e = ExplorationController(frontier_clearance=2.0, max_step=100.0, blocked_clearance=3.0)
    e.set_terrain([(10.0, 0.0), (0.0, 10.0)])
    e.mark_visited(0.0, 0.0)
    assert e.next_goal(0.0, 0.0) == (10.0, 0.0)   # farthest-ish, tie broken by max()
    e.mark_blocked(10.0, 0.0)                     # unreachable -> exclude it
    assert e.next_goal(0.0, 0.0) == (0.0, 10.0)   # re-routes to the other frontier
    print("✓ blocked frontier excluded -> re-routes")


def test_clamp_to_max_step():
    e = ExplorationController(frontier_clearance=2.0, max_step=4.0)
    e.set_terrain([(10.0, 0.0)])
    e.mark_visited(0.0, 0.0)
    g = e.next_goal(0.0, 0.0)
    # far frontier clamped to 4 m along the line
    assert g is not None and abs(g[0] - 4.0) < 1e-6
    print("✓ clamps far goal to max_step")


def test_covered_when_all_visited():
    e = ExplorationController(frontier_clearance=2.0)
    pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    e.set_terrain(pts)
    for x, y in pts:                    # visit on top of every terrain point
        e.mark_visited(x, y)
    assert e.is_covered(0.5, 0.5) is True
    assert e.next_goal(0.5, 0.5) is None
    print("✓ covered when every terrain point is near a visited pose")


def test_frontier_appears_with_new_terrain():
    e = ExplorationController(frontier_clearance=2.0, max_step=100.0)
    e.set_terrain([(1.0, 0.0)])         # only nearby terrain
    e.mark_visited(0.0, 0.0)
    assert e.is_covered(0.0, 0.0) is True   # nothing new to reach
    e.set_terrain([(1.0, 0.0), (9.0, 0.0)])  # terrain extends as robot moves
    assert e.is_covered(0.0, 0.0) is False
    assert e.next_goal(0.0, 0.0) == (9.0, 0.0)
    print("✓ new terrain reveals new frontier")


def test_visited_dedup():
    e = ExplorationController(visited_spacing=0.5)
    e.mark_visited(0.0, 0.0)
    e.mark_visited(0.1, 0.0)            # within spacing -> not added
    e.mark_visited(1.0, 0.0)            # beyond spacing -> added
    assert len(e.visited) == 2
    print("✓ visited poses deduped by spacing")


if __name__ == "__main__":
    test_no_terrain_not_covered()
    test_nearest_frontier()
    test_blocked_frontier_excluded()
    test_clamp_to_max_step()
    test_covered_when_all_visited()
    test_frontier_appears_with_new_terrain()
    test_visited_dedup()
    print("\nAll exploration tests passed.")
