"""Guard rails for the object map's merge and filter steps.

These lock in the properties the tuning of 2026-08-13 relied on, so a later change
to ObjectMemory or to the thresholds cannot quietly undo them. They are written
against synthetic nodes rather than a recorded scene, so they stay meaningful
without shipping a large snapshot - use tests/check_object_memory.py against a
real run for the scene-level numbers.

Run:
    cd ai_module/src/sysnav_ros2_mvp
    PYTHONPATH=. python3 -m pytest tests/test_object_memory_quality.py -v
"""

from __future__ import annotations

import unittest

import numpy as np

from sysnav import config
from sysnav.memory.object_memory import ObjectMemory, filter_reliable


def make_node(object_id: int, category: str, position, extent=(0.4, 0.4, 0.4),
              observations: int = 1, confidence: float = 0.9) -> dict:
    half = np.asarray(extent, dtype=np.float64) / 2.0
    centre = np.asarray(position, dtype=np.float64)
    return {
        "object_id": object_id,
        "category": category,
        "position": tuple(float(v) for v in centre),
        "extent_3d": tuple(float(v) for v in extent),
        "bbox_3d_min": tuple(float(v) for v in (centre - half)),
        "bbox_3d_max": tuple(float(v) for v in (centre + half)),
        "point_cloud": np.empty((0, 3), np.float32),
        "num_points": 0,
        "observation_count": observations,
        "confidence": confidence,
        "representative_confidence": confidence,
        "representative_image": None,
        "context_image": None,
        "first_seen_time": 0.0,
        "last_seen_time": 0.0,
        "latest_bbox_2d": (0, 0, 1, 1),
        "self_attributes": {},
    }


def memory_with(nodes: list[dict]) -> ObjectMemory:
    memory = ObjectMemory()
    with memory._lock:
        for node in nodes:
            memory._nodes[node["object_id"]] = node
    return memory


class MergeTest(unittest.TestCase):
    def test_merges_when_one_box_sits_inside_another(self):
        """같은 물체를 크게/작게 잡은 두 관측은 bbox가 크게 겹친다.

        실측(home_building_1): 병합된 8쌍이 전부 이 겹침 조건으로 잡혔고, 겹침 비율은
        1.0/0.90/0.88처럼 뚜렷하게 높았다. 거리 조건으로 잡힌 쌍은 0이었다."""
        memory = memory_with([
            make_node(1, "sofa", (0.0, 0.0, 0.5), extent=(2.0, 1.0, 0.6), observations=8),
            make_node(2, "sofa", (0.3, 0.0, 0.5), extent=(0.6, 0.5, 0.4), observations=4),
        ])
        merged = memory.merge_duplicates()
        self.assertEqual(merged, 1, "a box contained in another was not merged")
        self.assertEqual(len(memory.all_nodes()), 1)

    def test_opposite_ends_of_a_long_object_still_do_not_merge(self):
        """알려진 한계를 명시해 둔다: 2.7m 소파의 양 끝을 각각 잡으면 겹침이 0.11,
        centroid 거리가 1.66m로 두 조건 모두 못 넘어 별개 물체로 남는다. 이걸 잡으려면
        겹침 임계를 크게 낮춰야 하는데, 그러면 나란히 붙은 서로 다른 물체까지 합쳐져
        개수 세기가 과소 집계된다. 지금은 의도적으로 병합하지 않는다."""
        memory = memory_with([
            make_node(1, "sofa", (6.65, 3.40, 0.88), extent=(2.70, 1.02, 0.54), observations=8),
            make_node(2, "sofa", (5.08, 2.89, 0.71), extent=(1.30, 0.90, 0.71), observations=4),
        ])
        self.assertEqual(memory.merge_duplicates(), 0)

    def test_keeps_distinct_neighbours_apart(self):
        """나란히 놓였지만 겹치지 않는 서로 다른 물체는 합치면 안 된다 - 개수 세기에서
        과소 집계로 직결된다(pillow가 이미 GT 18개 대비 과소 탐지 상태다)."""
        memory = memory_with([
            make_node(1, "pillow", (0.0, 0.0, 0.8), extent=(0.4, 0.4, 0.3)),
            make_node(2, "pillow", (1.2, 0.0, 0.8), extent=(0.4, 0.4, 0.3)),
        ])
        self.assertEqual(memory.merge_duplicates(), 0)
        self.assertEqual(len(memory.all_nodes()), 2)

    def test_never_merges_across_categories(self):
        memory = memory_with([
            make_node(1, "sofa", (0.0, 0.0, 0.5), extent=(1.0, 1.0, 0.6)),
            make_node(2, "chair", (0.1, 0.0, 0.5), extent=(1.0, 1.0, 0.6)),
        ])
        self.assertEqual(memory.merge_duplicates(), 0)

    def test_merge_sums_observations(self):
        """관측 횟수 합산이 filter 임계(obs>=4)를 정당화한 근거다 - 병합 없이 같은
        임계를 걸면 정탐을 잃는다."""
        memory = memory_with([
            make_node(1, "sofa", (0.0, 0.0, 0.5), extent=(2.0, 1.0, 0.6), observations=3),
            make_node(2, "sofa", (0.6, 0.0, 0.5), extent=(2.0, 1.0, 0.6), observations=2),
        ])
        memory.merge_duplicates()
        survivor = memory.all_nodes()[0]
        self.assertEqual(survivor["observation_count"], 5)
        # 합집합 bbox라 크기가 두 관측을 모두 담아야 한다.
        self.assertGreaterEqual(survivor["extent_3d"][0], 2.0)


class FilterTest(unittest.TestCase):
    def test_drops_flickering_detections(self):
        """오탐은 특정 각도에서만 나타나 관측 횟수가 적다(실측 중앙 2 vs 정탐 17)."""
        nodes = [
            make_node(1, "sofa", (0.0, 0.0, 0.5), observations=17),
            make_node(2, "sofa", (5.0, 5.0, 0.5), observations=1),
        ]
        kept, dropped = filter_reliable(nodes)
        self.assertEqual(dropped, 1)
        self.assertEqual([n["object_id"] for n in kept], [1])

    def test_returns_everything_rather_than_nothing(self):
        """전부 걸러지면 원본을 돌려준다 - 탐사가 짧게 끝나 진짜 물체도 관측이 적을 수
        있고, 그때 무응답이 되는 게 약한 후보로 답하는 것보다 나쁘다."""
        nodes = [make_node(1, "sofa", (0.0, 0.0, 0.5), observations=1)]
        kept, dropped = filter_reliable(nodes)
        self.assertEqual(kept, nodes)
        self.assertEqual(dropped, 0)

    def test_empty_input_stays_empty(self):
        self.assertEqual(filter_reliable([]), ([], 0))

    def test_thresholds_are_the_measured_ones(self):
        """임계값이 조용히 바뀌면 위 성질들의 근거가 무효가 되므로 함께 고정한다."""
        self.assertEqual(config.OBJECT_MIN_OBSERVATIONS, 4)
        self.assertEqual(config.OBJECT_MIN_CONFIDENCE, 0.0)


if __name__ == "__main__":
    unittest.main()
