#!/usr/bin/env python3
"""Rule-based captions that mimic SORT3D's object-attribute field cheaply."""


SUPPORT_SURFACE_NAMES = {
    "table", "desk", "shelf", "file cabinet", "cabinet", "bench", "counter",
    "stand", "box",
}


def size_words(obj):
    volume = obj.volume
    height = obj.size[2]
    words = []
    if volume > 4.0:
        words.append("large")
    elif volume < 0.03:
        words.append("small")

    if height > 1.5:
        words.append("tall")
    elif height < 0.2:
        words.append("thin")
    return words


def find_support_object(obj, objects, xy_threshold=0.45, z_gap_threshold=0.35):
    best = None
    best_score = None
    for other in objects:
        if other.object_id == obj.object_id:
            continue
        if not any(name in other.name for name in SUPPORT_SURFACE_NAMES):
            continue

        xy_gap = obj.surface_distance_xy(other)
        z_gap = obj.z_min - other.z_max
        if xy_gap > xy_threshold:
            continue
        if z_gap < -0.15 or z_gap > z_gap_threshold:
            continue

        score = xy_gap + abs(z_gap) * 0.5
        if best_score is None or score < best_score:
            best = other
            best_score = score
    return best


def nearby_objects(obj, objects, max_count=3, max_distance=1.0, exclude_ids=None):
    exclude_ids = set(exclude_ids or [])
    pairs = []
    for other in objects:
        if other.object_id == obj.object_id:
            continue
        if other.object_id in exclude_ids:
            continue
        distance = obj.surface_distance_xy(other)
        if distance <= max_distance:
            pairs.append((distance, other))
    pairs.sort(key=lambda item: item[0])
    return [other for _, other in pairs[:max_count]]


def generate_caption(obj, objects):
    """Generate a compact caption from geometry and nearby context."""
    descriptors = size_words(obj)
    base = " ".join(descriptors + [obj.name]).strip()
    if not base:
        base = "object"

    clauses = [f"a {base}"]

    support = find_support_object(obj, objects)
    if support is not None:
        clauses.append(f"on or above the {support.name}")

    exclude_ids = {support.object_id} if support is not None else set()
    nearby = nearby_objects(obj, objects, exclude_ids=exclude_ids)
    nearby_names = []
    for other in nearby:
        if other.name not in nearby_names:
            nearby_names.append(other.name)
    if nearby_names:
        clauses.append("near " + ", ".join(nearby_names[:3]))

    return " ".join(clauses)


def attach_rule_captions(objects):
    for obj in objects:
        if not obj.caption:
            obj.caption = generate_caption(obj, objects)
    return objects
