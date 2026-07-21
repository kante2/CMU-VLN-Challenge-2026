import unittest

import numpy as np

from sysnav.scene_graph.viewpoint_coverage import ViewpointCoverageBuilder


class ViewpointCoverageBuilderTest(unittest.TestCase):
    def test_same_pose_produces_same_voxels(self):
        builder = ViewpointCoverageBuilder()
        points = np.array(
            [
                [2.0, 0.0, 0.2],
                [0.0, 2.0, 0.5],
                [-2.0, 0.0, 0.8],
                [0.0, -2.0, 1.0],
            ],
            dtype=np.float32,
        )
        pose = {"x": 1.0, "y": 2.0, "z": 0.0, "yaw": 0.0}
        self.assertEqual(builder.compute(points, pose), builder.compute(points, pose))

    def test_translation_changes_map_voxel_keys(self):
        builder = ViewpointCoverageBuilder()
        points = np.array([[2.0, 0.0, 0.2], [0.0, 2.0, 0.5]], dtype=np.float32)
        first = builder.compute(points, {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0})
        second = builder.compute(points, {"x": 5.0, "y": 0.0, "z": 0.0, "yaw": 0.0})
        self.assertTrue(first)
        self.assertTrue(second)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
