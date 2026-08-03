import unittest

import numpy as np

from sysnav.perception.lidar_grounding import (
    PanoramaLidarGrounder,
    count_bbox_hits,
    supplement_mask_hits_from_bbox,
)


class ProvisionalGroundingTest(unittest.TestCase):
    def test_counts_projected_points_inside_bbox(self):
        u = np.array([9, 10, 19, 20, 15])
        v = np.array([15, 10, 19, 15, 20])
        self.assertEqual(count_bbox_hits(u, v, (10, 10, 20, 20)), 2)

    def test_supplements_sparse_mask_with_nearby_depth_consistent_bbox_points(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[5, 5] = True
        u = np.array([5, 6, 7, 8])
        v = np.array([5, 5, 5, 5])
        points = np.array([
            [2.0, 0.0, 0.0],
            [2.1, 0.0, 0.0],
            [1.9, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])
        selected, added = supplement_mask_hits_from_bbox(
            mask, u, v, (4, 4, 10, 8), points, mask[v, u]
        )
        self.assertEqual(added, 2)
        self.assertEqual(int(np.count_nonzero(selected)), 3)
        self.assertFalse(selected[3])

    def test_uses_bbox_fallback_without_any_mask_hit(self):
        mask = np.zeros((12, 12), dtype=bool)
        u = np.array([5, 6, 7])
        v = np.array([5, 5, 5])
        points = np.ones((3, 3), dtype=np.float32)
        selected, added = supplement_mask_hits_from_bbox(
            mask, u, v, (4, 4, 10, 8), points, mask[v, u]
        )
        self.assertEqual(added, 3)
        self.assertTrue(np.all(selected))

    def test_empty_mask_bbox_fallback_still_requires_three_hits(self):
        mask = np.zeros((12, 12), dtype=bool)
        u = np.array([5, 6])
        v = np.array([5, 5])
        points = np.ones((2, 3), dtype=np.float32)
        selected, added = supplement_mask_hits_from_bbox(
            mask, u, v, (4, 4, 10, 8), points, mask[v, u]
        )
        self.assertEqual(added, 0)
        self.assertFalse(np.any(selected))

    def test_promotes_three_points_observed_across_two_frames(self):
        grounder = PanoramaLidarGrounder()
        grounder._frame_id = 1
        promoted, point_count, frame_count = grounder._accumulate_provisional(
            "bowl",
            np.array([[1.0, 2.0, 0.8], [1.02, 2.0, 0.8]], dtype=np.float32),
            now=1.0,
        )
        self.assertIsNone(promoted)
        self.assertEqual((point_count, frame_count), (2, 1))

        grounder._frame_id = 2
        promoted, point_count, frame_count = grounder._accumulate_provisional(
            "bowl",
            np.array([[1.01, 2.01, 0.8]], dtype=np.float32),
            now=2.0,
        )
        self.assertIsNotNone(promoted)
        self.assertEqual(len(promoted), 3)
        self.assertEqual((point_count, frame_count), (3, 2))

    def test_does_not_promote_duplicate_detections_in_one_frame(self):
        grounder = PanoramaLidarGrounder()
        grounder._frame_id = 1
        first = grounder._accumulate_provisional(
            "bowl",
            np.array([[1.0, 2.0, 0.8], [1.02, 2.0, 0.8]], dtype=np.float32),
            now=1.0,
        )
        second = grounder._accumulate_provisional(
            "bowl",
            np.array([[1.01, 2.01, 0.8]], dtype=np.float32),
            now=1.0,
        )
        self.assertIsNone(first[0])
        self.assertIsNone(second[0])


if __name__ == "__main__":
    unittest.main()
