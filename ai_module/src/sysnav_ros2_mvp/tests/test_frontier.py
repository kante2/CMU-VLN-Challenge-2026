import unittest

import numpy as np

from sysnav import config
from sysnav.exploration.coverage_planner import CoveragePlanner
from sysnav.exploration.frontier_extractor import FrontierExtractor


class FrontierTest(unittest.TestCase):
    def test_frontier(self):
        grid = np.full((30, 30), config.OCC_UNKNOWN, dtype=np.int8)
        grid[10:20, 10:20] = config.OCC_FREE
        clusters = FrontierExtractor(min_cluster_cells=3).extract(grid)
        self.assertGreaterEqual(len(clusters), 1)

    def test_recovery_patrol_works_without_frontier_gain(self):
        planner = CoveragePlanner()
        pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        planner.reset(pose)
        center = planner.world_to_grid(0.0, 0.0)
        self.assertIsNotNone(center)
        row, col = center
        planner.grid[row - 12:row + 13, col - 12:col + 13] = config.OCC_FREE

        route = planner.plan_recovery_patrol(pose, previous_points=[])

        self.assertTrue(route)
        endpoint = route[-1]
        self.assertGreaterEqual(
            np.hypot(endpoint["x"] - pose["x"], endpoint["y"] - pose["y"]),
            config.MISSION3_RECOVERY_PATROL_MIN_SPACING_M,
        )


if __name__ == "__main__":
    unittest.main()
