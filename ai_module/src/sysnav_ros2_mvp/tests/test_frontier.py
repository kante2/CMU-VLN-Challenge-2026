import unittest

import numpy as np

from sysnav import config
from sysnav.exploration.frontier_extractor import FrontierExtractor


class FrontierTest(unittest.TestCase):
    def test_frontier(self):
        grid = np.full((30, 30), config.OCC_UNKNOWN, dtype=np.int8)
        grid[10:20, 10:20] = config.OCC_FREE
        clusters = FrontierExtractor(min_cluster_cells=3).extract(grid)
        self.assertGreaterEqual(len(clusters), 1)


if __name__ == "__main__":
    unittest.main()
