import unittest

import numpy as np

from sysnav.perception.lidar_grounding import PanoramaLidarGrounder


class ProvisionalGroundingTest(unittest.TestCase):
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
