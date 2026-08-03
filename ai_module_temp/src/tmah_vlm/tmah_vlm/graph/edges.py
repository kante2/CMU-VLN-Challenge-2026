#!/usr/bin/env python3
"""
Spatial relation edges between object nodes.

These edges are geometric and intentionally lightweight. They are computed from
the current map-frame object centers and axis-aligned 3D bbox sizes.
"""

from dataclasses import dataclass
import math


@dataclass
class RelationEdge:
    source: str
    target: str
    relation: str
    score: float
    distance_m: float
    delta: tuple

    def to_dict(self):
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "score": float(self.score),
            "distance_m": float(self.distance_m),
            "delta": list(self.delta),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            source=str(data.get("source", "")),
            target=str(data.get("target", "")),
            relation=str(data.get("relation", "")),
            score=float(data.get("score", 0.0)),
            distance_m=float(data.get("distance_m", 0.0)),
            delta=tuple(data.get("delta", (0.0, 0.0, 0.0))),
        )


def center_delta(source, target):
    return tuple(float(b) - float(a) for a, b in zip(source.center, target.center))


def object_radius_xy(obj):
    sx, sy, _ = obj.size
    return 0.5 * math.hypot(float(sx), float(sy))


def xy_distance(delta):
    return math.hypot(float(delta[0]), float(delta[1]))


def xyz_distance(delta):
    return math.sqrt(sum(float(v) * float(v) for v in delta))


def near_threshold(source, target):
    return max(1.0, object_radius_xy(source) + object_radius_xy(target) + 0.8)


def relation_score(value, threshold):
    if threshold <= 1e-6:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(value) / float(threshold)))


def add_directed(edges, source, target, relation, score, distance, delta):
    edges.append(RelationEdge(
        source=source.object_id,
        target=target.object_id,
        relation=relation,
        score=score,
        distance_m=distance,
        delta=delta,
    ))


def pair_relations(source, target):
    """
    Compute directed object-object relations for one unordered object pair.

    Coordinate convention:
      x: map x. Smaller x is treated as left_of, larger x as right_of.
      y: map y. Larger y is treated as in_front_of, smaller y as behind.
      z: map z. Larger z is above, smaller z is below.
    """
    delta = center_delta(source, target)
    distance_xy = xy_distance(delta)
    distance_xyz = xyz_distance(delta)
    edges = []

    threshold = near_threshold(source, target)
    if distance_xy <= threshold:
        score = relation_score(distance_xy, threshold)
        add_directed(edges, source, target, "near", score, distance_xyz, delta)
        add_directed(edges, target, source, "near", score, distance_xyz, tuple(-v for v in delta))

    dx, dy, dz = delta
    sx, sy, sz = source.size
    tx, ty, tz = target.size

    lateral_margin = max(0.25, 0.5 * (float(sx) + float(tx)))
    depth_margin = max(0.25, 0.5 * (float(sy) + float(ty)))
    vertical_margin = max(0.20, 0.5 * (float(sz) + float(tz)))

    if abs(dx) > lateral_margin:
        score = relation_score(abs(dy), max(1.0, threshold))
        if dx > 0:
            add_directed(edges, target, source, "left_of", score, distance_xyz, tuple(-v for v in delta))
            add_directed(edges, source, target, "right_of", score, distance_xyz, delta)
        else:
            add_directed(edges, source, target, "left_of", score, distance_xyz, delta)
            add_directed(edges, target, source, "right_of", score, distance_xyz, tuple(-v for v in delta))

    if abs(dy) > depth_margin:
        score = relation_score(abs(dx), max(1.0, threshold))
        if dy > 0:
            add_directed(edges, target, source, "behind", score, distance_xyz, tuple(-v for v in delta))
            add_directed(edges, source, target, "in_front_of", score, distance_xyz, delta)
        else:
            add_directed(edges, source, target, "behind", score, distance_xyz, delta)
            add_directed(edges, target, source, "in_front_of", score, distance_xyz, tuple(-v for v in delta))

    if abs(dz) > vertical_margin:
        score = relation_score(distance_xy, max(1.0, threshold))
        if dz > 0:
            add_directed(edges, target, source, "below", score, distance_xyz, tuple(-v for v in delta))
            add_directed(edges, source, target, "above", score, distance_xyz, delta)
        else:
            add_directed(edges, source, target, "below", score, distance_xyz, delta)
            add_directed(edges, target, source, "above", score, distance_xyz, tuple(-v for v in delta))

    return edges


def compute_relation_edges(objects):
    object_list = list(objects.values())
    edges = []

    for i in range(len(object_list)):
        for j in range(i + 1, len(object_list)):
            edges.extend(pair_relations(object_list[i], object_list[j]))

    edges.sort(key=lambda edge: (edge.source, edge.target, edge.relation))
    return edges
