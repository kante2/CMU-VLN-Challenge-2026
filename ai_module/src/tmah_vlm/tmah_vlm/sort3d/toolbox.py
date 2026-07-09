#!/usr/bin/env python3
"""SORT3D-style heuristic spatial toolbox."""

import math

from tmah_vlm.sort3d.filters import objects_named


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class SpatialToolbox:
    """Heuristic spatial functions over Sort3DObject instances."""

    def __init__(self, objects, near_threshold_m=1.0, vertical_overlap_m=0.2):
        self.objects = list(objects)
        self.object_by_id = {obj.object_id: obj for obj in self.objects}
        self.near_threshold_m = float(near_threshold_m)
        self.vertical_overlap_m = float(vertical_overlap_m)

    def get(self, object_id):
        return self.object_by_id.get(str(object_id))

    def named(self, name):
        return objects_named(self.objects, name)

    def ids(self, objects):
        return [obj.object_id for obj in objects]

    def find_near(self, target_name, anchor):
        anchors = self._resolve(anchor)
        targets = self.named(target_name)
        out = []
        for target in targets:
            if any(target.object_id != a.object_id and target.surface_distance_xy(a) <= self.near_threshold_m for a in anchors):
                out.append(target)
        return self.ids(out)

    def find_between(self, target_name, anchor1, anchor2, width_tolerance_m=1.2):
        first = self._resolve(anchor1)
        second = self._resolve(anchor2)
        targets = self.named(target_name)
        out = []
        for target in targets:
            if any(self._is_between(target, a, b, width_tolerance_m) for a in first for b in second):
                out.append(target)
        return self.ids(out)

    def find_above(self, target_name, anchor):
        anchors = self._resolve(anchor)
        targets = self.named(target_name)
        out = []
        for target in targets:
            if any(self._xy_overlaps(target, a) and target.z_min >= a.z_max - self.vertical_overlap_m for a in anchors):
                out.append(target)
        return self.ids(out)

    def find_below(self, target_name, anchor):
        anchors = self._resolve(anchor)
        targets = self.named(target_name)
        out = []
        for target in targets:
            if any(self._xy_overlaps(target, a) and target.z_max <= a.z_min + self.vertical_overlap_m for a in anchors):
                out.append(target)
        return self.ids(out)

    def find_left(self, target_name, anchor, reference_yaw=0.0):
        return self._find_lateral(target_name, anchor, reference_yaw, want_left=True)

    def find_right(self, target_name, anchor, reference_yaw=0.0):
        return self._find_lateral(target_name, anchor, reference_yaw, want_left=False)

    def find_in_front_of(self, target_name, anchor, reference_yaw=0.0):
        return self._find_forward(target_name, anchor, reference_yaw, want_front=True)

    def find_behind(self, target_name, anchor, reference_yaw=0.0):
        return self._find_forward(target_name, anchor, reference_yaw, want_front=False)

    def order_smallest_to_largest(self, target_name):
        return self.ids(sorted(self.named(target_name), key=lambda obj: obj.volume))

    def order_bottom_to_top(self, target_name):
        return self.ids(sorted(self.named(target_name), key=lambda obj: obj.z))

    def closest_to(self, target_name, anchor):
        anchors = self._resolve(anchor)
        targets = self.named(target_name)
        if not anchors or not targets:
            return []
        targets.sort(key=lambda obj: min(obj.surface_distance_xy(a) for a in anchors))
        return self.ids(targets)

    def furthest_from(self, target_name, anchor):
        anchors = self._resolve(anchor)
        targets = self.named(target_name)
        if not anchors or not targets:
            return []
        targets.sort(key=lambda obj: max(obj.surface_distance_xy(a) for a in anchors), reverse=True)
        return self.ids(targets)

    def _resolve(self, value):
        resolved = []
        for item in _as_list(value):
            if hasattr(item, "object_id"):
                resolved.append(item)
            elif str(item) in self.object_by_id:
                resolved.append(self.object_by_id[str(item)])
            else:
                resolved.extend(self.named(str(item)))
        return resolved

    def _is_between(self, target, anchor1, anchor2, width_tolerance_m):
        ax = anchor1.x
        ay = anchor1.y
        bx = anchor2.x
        by = anchor2.y
        vx = bx - ax
        vy = by - ay
        length_sq = vx * vx + vy * vy
        if length_sq <= 1e-6:
            return False
        tx = target.x - ax
        ty = target.y - ay
        projection = (tx * vx + ty * vy) / length_sq
        if projection < 0.0 or projection > 1.0:
            return False
        closest_x = ax + projection * vx
        closest_y = ay + projection * vy
        lateral = math.hypot(target.x - closest_x, target.y - closest_y)
        return lateral <= width_tolerance_m

    def _xy_overlaps(self, a, b):
        return a.surface_distance_xy(b) <= 0.15

    def _find_lateral(self, target_name, anchor, reference_yaw, want_left):
        anchors = self._resolve(anchor)
        targets = self.named(target_name)
        left_x = -math.sin(reference_yaw)
        left_y = math.cos(reference_yaw)
        out = []
        for target in targets:
            for a in anchors:
                lateral = (target.x - a.x) * left_x + (target.y - a.y) * left_y
                if (want_left and lateral > 0.0) or ((not want_left) and lateral < 0.0):
                    out.append(target)
                    break
        return self.ids(out)

    def _find_forward(self, target_name, anchor, reference_yaw, want_front):
        anchors = self._resolve(anchor)
        targets = self.named(target_name)
        forward_x = math.cos(reference_yaw)
        forward_y = math.sin(reference_yaw)
        out = []
        for target in targets:
            for a in anchors:
                forward = (target.x - a.x) * forward_x + (target.y - a.y) * forward_y
                if (want_front and forward > 0.0) or ((not want_front) and forward < 0.0):
                    out.append(target)
                    break
        return self.ids(out)
