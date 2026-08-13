"""Continuous-geometry shortest-path routing over the traversable region, via
a visibility graph - used as a drop-in improvement over 4-connected grid A*
for recovery patrol (coverage_planner.py's plan_recovery_patrol()).

Why: _astar_path()'s grid search plus _simplify_path_indices()'s Bresenham
line-of-sight check only ever "see" collisions at MAP_RESOLUTION_M
quantization. A hop that looks LOS-clear against the (already
ROBOT_CLEARANCE_M-inflated) occupancy grid can still shave right along - or
slip a hair across - the true clearance boundary once a small cluttered room
leaves only a cell or two of inflated free space to route through. Tracing
the exact polygon boundary of that inflated free space (via cv2 contours) and
routing through a visibility graph checked with real-valued
Polygon.covers(LineString) removes that quantization slop entirely, instead
of merely penalizing it.

shapely is optional at import time: if it isn't installed yet (image not
rebuilt), every function here returns None and callers fall back to the
existing grid A* path unchanged.
"""

from __future__ import annotations

import heapq
import math
from typing import Callable

import cv2
import numpy as np

try:
    from shapely.geometry import LineString, Point, Polygon
    from shapely.geometry.base import BaseGeometry
    from shapely.ops import unary_union

    _SHAPELY_AVAILABLE = True
except ImportError:  # pragma: no cover - only true before requirements are installed
    _SHAPELY_AVAILABLE = False
    BaseGeometry = object  # type: ignore[assignment,misc]


def shapely_available() -> bool:
    return _SHAPELY_AVAILABLE


def build_traversable_polygon(
    traversable: np.ndarray, grid_to_world: Callable[[int, int], tuple[float, float]]
) -> "BaseGeometry | None":
    """traversable(bool grid, [row, col]) -> a shapely polygon in world
    coordinates (holes = untraversable islands, e.g. furniture fully
    surrounded by free space), via cv2 contour tracing."""
    if not _SHAPELY_AVAILABLE:
        return None
    mask = traversable.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return None
    hierarchy = hierarchy[0]

    def to_world(contour) -> list[tuple[float, float]]:
        # cv2 contour points are (x=col, y=row); grid_to_world expects (row, col).
        return [grid_to_world(int(pt[0][1]), int(pt[0][0])) for pt in contour]

    polygons = []
    for i, contour in enumerate(contours):
        if hierarchy[i][3] != -1 or len(contour) < 3:
            continue  # holes are attached to their parent contour below
        holes = []
        child = hierarchy[i][2]
        while child != -1:
            if len(contours[child]) >= 3:
                holes.append(to_world(contours[child]))
            child = hierarchy[child][0]
        try:
            poly = Polygon(to_world(contour), holes)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)
        except Exception:
            continue
    if not polygons:
        return None
    return unary_union(polygons)


def _region_containing(polygon: "BaseGeometry", point: tuple[float, float]) -> "BaseGeometry | None":
    """MultiPolygon-safe lookup of the single connected blob that actually
    holds `point` - start/goal must route within one blob, not jump gaps."""
    geoms = polygon.geoms if hasattr(polygon, "geoms") else [polygon]
    p = Point(point)
    for geom in geoms:
        if geom.covers(p):
            return geom
    return None


def shortest_path(
    polygon: "BaseGeometry | None",
    start: tuple[float, float],
    goal: tuple[float, float],
    simplify_tolerance: float,
) -> list[tuple[float, float]] | None:
    """Shortest collision-free polyline from start to goal via a visibility
    graph over polygon's (simplified) vertices. None if shapely is
    unavailable, start/goal aren't in the same traversable blob, or no path
    exists - callers should fall back to grid A* in that case."""
    if not _SHAPELY_AVAILABLE or polygon is None:
        return None
    region = _region_containing(polygon, start)
    if region is None or not region.covers(Point(goal)):
        return None

    if region.covers(LineString([start, goal])):
        return [start, goal]

    simplified = region.simplify(simplify_tolerance, preserve_topology=True)
    rings = [simplified.exterior, *simplified.interiors]
    nodes: list[tuple[float, float]] = [start, goal]
    for ring in rings:
        nodes.extend(list(ring.coords)[:-1])

    # De-dup near-identical vertices (simplify can leave duplicates at seams).
    deduped: list[tuple[float, float]] = []
    for node in nodes:
        if not any(math.hypot(node[0] - o[0], node[1] - o[1]) < 1e-6 for o in deduped):
            deduped.append(node)
    nodes = deduped
    start_idx, goal_idx = 0, 1

    n = len(nodes)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = nodes[i], nodes[j]
            # Re-validate against the ORIGINAL (unsimplified) region, since
            # simplification can shift the boundary slightly - never trust a
            # collision check against the simplified copy alone.
            if not region.covers(LineString([a, b])):
                continue
            dist = math.hypot(a[0] - b[0], a[1] - b[1])
            adjacency[i].append((j, dist))
            adjacency[j].append((i, dist))

    best = {start_idx: 0.0}
    came_from: dict[int, int] = {}
    queue: list[tuple[float, int]] = [(0.0, start_idx)]
    visited: set[int] = set()
    while queue:
        cost, node = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        if node == goal_idx:
            break
        for neighbor, weight in adjacency[node]:
            new_cost = cost + weight
            if new_cost < best.get(neighbor, float("inf")):
                best[neighbor] = new_cost
                came_from[neighbor] = node
                heapq.heappush(queue, (new_cost, neighbor))

    if goal_idx not in best:
        return None
    path_idx = [goal_idx]
    while path_idx[-1] != start_idx:
        path_idx.append(came_from[path_idx[-1]])
    path_idx.reverse()
    return [nodes[i] for i in path_idx]
