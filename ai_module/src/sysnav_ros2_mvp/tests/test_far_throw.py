"""교착 탈출: commandable 지점이 없을 때 5m 밖으로 던지기.

terrain_map은 noDecayDis=1.75m 롤링이라 로봇 반경 1.75m 밖에는 base autonomy가 받아줄
지점이 구조적으로 없다(실측: 2.0m 밖 0점). waypointConverter는 adjDisThre(5m) **밖**
목표만 손대지 않으므로, 그 밖으로 던지는 게 유일한 탈출구다.

실측(2026-08-23): 1.5m/2.5m 요청은 갈아끼워져 로봇이 0.00m 움직였고, 8.0m 요청은
그대로 통과해 2.57m 전진했다.
"""

import math
import unittest

import numpy as np

from sysnav import config
from sysnav.exploration.coverage_planner import CoveragePlanner


def _planner_with_corridor(wall_x=None):
    planner = CoveragePlanner()
    planner.origin_x = planner.origin_y = -30.0
    for x in np.arange(-1.0, 12.0, 0.1):
        for y in np.arange(-2.0, 2.0, 0.1):
            cell = planner.world_to_grid(float(x), float(y))
            if cell is not None:
                planner.grid[cell] = config.OCC_FREE
    if wall_x is not None:
        for y in np.arange(-2.0, 2.0, 0.1):
            cell = planner.world_to_grid(wall_x, float(y))
            if cell is not None:
                planner.grid[cell] = config.OCC_OCCUPIED
    return planner


class ClearDistanceTest(unittest.TestCase):
    def test_open_corridor_reports_full_range(self):
        planner = _planner_with_corridor(None)
        distance = planner.clear_distance_along((0.0, 0.0), (1.0, 0.0), 6.0)
        self.assertGreaterEqual(distance, 5.5)

    def test_wall_limits_the_distance(self):
        planner = _planner_with_corridor(wall_x=3.0)
        distance = planner.clear_distance_along((0.0, 0.0), (1.0, 0.0), 6.0)
        self.assertLess(distance, 3.1)
        self.assertGreater(distance, 2.0)

    def test_unknown_space_does_not_block(self):
        """탐색은 미지 영역으로 가는 게 목적이라 UNKNOWN은 막지 않는다
        (A*는 반대로 unknown을 통과 불가로 본다 - 기준이 다르다)."""
        planner = CoveragePlanner()
        planner.origin_x = planner.origin_y = -30.0
        distance = planner.clear_distance_along((0.0, 0.0), (1.0, 0.0), 6.0)
        self.assertGreaterEqual(distance, 5.5)


class FarThrowGeometryTest(unittest.TestCase):
    """던진 좌표가 실제로 adjDisThre 밖이어야 waypointConverter가 손을 안 댄다."""

    def test_thrown_target_clears_adj_dis_thre(self):
        robot = (3.58, 2.63)
        goal = (4.5, 2.9)
        dx, dy = goal[0] - robot[0], goal[1] - robot[1]
        norm = math.hypot(dx, dy)
        distance = config.TERRAIN_ADJ_DIS_M + config.FAR_THROW_MARGIN_M
        target = (robot[0] + dx / norm * distance, robot[1] + dy / norm * distance)
        self.assertGreater(math.dist(robot, target), config.TERRAIN_ADJ_DIS_M)

    def test_direction_is_preserved(self):
        robot = (0.0, 0.0)
        goal = (1.0, 1.0)
        dx, dy = goal[0] - robot[0], goal[1] - robot[1]
        norm = math.hypot(dx, dy)
        distance = config.TERRAIN_ADJ_DIS_M + config.FAR_THROW_MARGIN_M
        target = (dx / norm * distance, dy / norm * distance)
        self.assertAlmostEqual(
            math.atan2(*reversed(target)), math.atan2(dy, dx), places=6
        )

    def test_margin_keeps_it_out_of_snap_range_while_approaching(self):
        """5m 판정은 매 순간 다시 이뤄진다 - 여유가 0이면 한 발짝만 가도 스냅이 걸린다."""
        self.assertGreater(config.FAR_THROW_MARGIN_M, 0.0)


if __name__ == "__main__":
    unittest.main()
