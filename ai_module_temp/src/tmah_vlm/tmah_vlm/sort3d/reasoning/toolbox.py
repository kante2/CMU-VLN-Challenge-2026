#!/usr/bin/env python3
"""
SORT3D-style heuristic spatial toolbox — 누적 scene graph의 이름 있는 Sort3DObject 대상 추론.

near/closest/farthest/left/right 판정은 spatial/relations.py의 함수를 그대로 쓴다
(2026-07-13 통합 — 예전엔 이 파일이 같은 판정을 별도로 재구현하고 있었고, 특히 left/right는
reference_yaw 기본값(0.0)이 실제로 넘겨진 적이 없어서 로봇 방향과 무관하게 고정된 map frame
방향을 "왼쪽"으로 판정하는 버그가 있었다 — 이제 spatial/relations.py와 동일하게 viewer_point
기준으로 판정한다).

find_between/find_above/find_below/find_in_front_of/find_behind는 여기서 자체 구현을 유지한다.
"표면에 닿아있다"(예: "테이블 위") 같은 물리적 접촉 판정이 필요해서 spatial/relations.py의
느슨한 above/below 판정과는 의도적으로 다르다.
"""

import math

import numpy as np

from tmah_vlm.spatial import relations as spatial_relations
from tmah_vlm.sort3d.reasoning.filters import objects_named


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
        if not anchors or not targets:
            return []

        anchor_ids = {a.object_id for a in anchors}
        points = [t.center for t in targets]
        radii = [t.radius_xy for t in targets]

        keep = np.zeros(len(targets), dtype=bool)
        for a in anchors:
            keep |= spatial_relations.find_near(
                points, a.center,
                max_distance=self.near_threshold_m,
                point_radii=radii, reference_radius=a.radius_xy,
            )

        out = [t for t, k in zip(targets, keep) if k and t.object_id not in anchor_ids]
        return self.ids(out)

    def closest_to(self, target_name, anchor):
        anchors = self._resolve(anchor)
        targets = self.named(target_name)
        if not anchors or not targets:
            return []

        points = [t.center for t in targets]
        radii = [t.radius_xy for t in targets]
        best = None
        for a in anchors:
            dist = spatial_relations.surface_distances_to(
                points, a.center, point_radii=radii, reference_radius=a.radius_xy,
            )
            best = dist if best is None else np.minimum(best, dist)

        order = np.argsort(best)
        return self.ids([targets[i] for i in order])

    def furthest_from(self, target_name, anchor):
        anchors = self._resolve(anchor)
        targets = self.named(target_name)
        if not anchors or not targets:
            return []

        points = [t.center for t in targets]
        radii = [t.radius_xy for t in targets]
        worst = None
        for a in anchors:
            dist = spatial_relations.surface_distances_to(
                points, a.center, point_radii=radii, reference_radius=a.radius_xy,
            )
            worst = dist if worst is None else np.maximum(worst, dist)

        order = np.argsort(-worst)
        return self.ids([targets[i] for i in order])

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

    def find_left(self, target_name, anchor, viewer_point):
        return self._find_lateral(target_name, anchor, viewer_point, want_left=True)

    def find_right(self, target_name, anchor, viewer_point):
        return self._find_lateral(target_name, anchor, viewer_point, want_left=False)

    def find_in_front_of(self, target_name, anchor, viewer_yaw=0.0):
        return self._find_forward(target_name, anchor, viewer_yaw, want_front=True)

    def find_behind(self, target_name, anchor, viewer_yaw=0.0):
        return self._find_forward(target_name, anchor, viewer_yaw, want_front=False)

    def order_smallest_to_largest(self, target_name):
        return self.ids(sorted(self.named(target_name), key=lambda obj: obj.volume))

    def order_bottom_to_top(self, target_name):
        return self.ids(sorted(self.named(target_name), key=lambda obj: obj.z))

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

    def _find_lateral(self, target_name, anchor, viewer_point, want_left):
        """viewer_point(로봇 실제 위치) -> anchor 방향 기준 왼쪽/오른쪽 판정.

        spatial/relations.py의 side_of_line과 동일한 기준(2026-07-13 통합 전에는
        reference_yaw 기본값 0.0이 실제로 안 넘겨져서 로봇 방향과 무관하게 고정된
        map frame 방향을 "왼쪽"으로 잘못 판정하던 버그가 있었다).
        """
        anchors = self._resolve(anchor)
        targets = self.named(target_name)
        if not anchors or not targets:
            return []

        points = [t.center for t in targets]
        seen = set()
        out = []
        for a in anchors:
            side = spatial_relations.side_of_line(points, viewer_point, a.center)
            for target, s in zip(targets, side):
                if target.object_id in seen:
                    continue
                if (want_left and s > 0.0) or ((not want_left) and s < 0.0):
                    out.append(target)
                    seen.add(target.object_id)
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
