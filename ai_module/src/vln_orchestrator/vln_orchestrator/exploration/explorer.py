#!/usr/bin/env python3
"""Lightweight frontier-style exploration controller (pure Python, ROS-free).

The challenge requires the AI module to explore the unknown scene by sending
waypoints; the base system navigates to them and snaps out-of-traversable
waypoints onto traversable ground. The robot's camera is a 360-degree panorama,
so a single pose sees all directions — coverage just needs the robot to VISIT
enough of the traversable area (no in-place rotation required).

Goal selection picks the MOST unexplored traversable point (farthest from every
visited pose), clamped to a max step, so the robot crosses the room instead of
lingering near its start. Goals that turn out unreachable are marked "blocked"
(by the node's stuck timeout) and excluded so we re-route. When no frontier
remains, the reachable area is covered.

Pure geometry on (x, y) tuples -> unit-testable off-robot.
"""
from __future__ import annotations

import math


class ExplorationController:
    def __init__(self, frontier_clearance: float = 2.0, max_step: float = 4.0,
                 visited_spacing: float = 0.5, blocked_clearance: float = 3.0) -> None:
        # a terrain point is an unexplored frontier if it is at least this far
        # from every visited pose
        self.frontier_clearance = frontier_clearance
        # cap a single goal's distance from the current pose (drive incrementally)
        self.max_step = max_step
        # dedupe visited poses closer than this to keep the set small
        self.visited_spacing = visited_spacing
        # exclude frontiers within this radius of a blocked (unreachable) goal
        self.blocked_clearance = blocked_clearance
        self.visited: list[tuple[float, float]] = []
        self.blocked: list[tuple[float, float]] = []
        self.terrain: list[tuple[float, float]] = []

    def set_terrain(self, points) -> None:
        """Latest traversable (x, y) points (from /terrain_map)."""
        self.terrain = [(float(x), float(y)) for x, y in points]

    def mark_visited(self, x: float, y: float) -> None:
        x, y = float(x), float(y)
        if not self.visited or math.hypot(
            x - self.visited[-1][0], y - self.visited[-1][1]
        ) > self.visited_spacing:
            self.visited.append((x, y))

    def mark_blocked(self, x: float, y: float) -> None:
        """Record an unreachable goal so nearby frontiers are skipped."""
        self.blocked.append((float(x), float(y)))

    def _min_dist(self, p, pts) -> float:
        if not pts:
            return float("inf")
        return min(math.hypot(p[0] - qx, p[1] - qy) for qx, qy in pts)

    def _frontiers(self):
        return [
            p for p in self.terrain
            if self._min_dist(p, self.visited) >= self.frontier_clearance
            and self._min_dist(p, self.blocked) >= self.blocked_clearance
        ]

    def next_goal(self, cur_x: float, cur_y: float):
        """Head to the NEAREST unexplored frontier (classic frontier exploration:
        covers the area systematically from the start rather than beelining to a
        far point and wandering out of the room). Clamped to max_step; None if no
        frontier remains (covered)."""
        frontiers = self._frontiers()
        if not frontiers:
            return None
        gx, gy = min(frontiers, key=lambda p: math.hypot(p[0] - cur_x, p[1] - cur_y))
        d = math.hypot(gx - cur_x, gy - cur_y)
        if d > self.max_step:
            s = self.max_step / d
            gx, gy = cur_x + (gx - cur_x) * s, cur_y + (gy - cur_y) * s
        return (gx, gy)

    def is_covered(self, cur_x: float, cur_y: float) -> bool:
        """Covered = we have terrain data and no unexplored frontier remains.
        Returns False when no terrain has been received yet (can't conclude)."""
        if not self.terrain:
            return False
        return not self._frontiers()
