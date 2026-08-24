"""가려진 공간은 UNKNOWN으로 남아야 한다 (update_from_scan의 폐색 처리).

센서는 바닥 위 약 0.75m라 소파(0.8m)·테이블(0.75m) 같은 가구와 거의 같은 높이다.
z 범위를 하나만 쓰면 가구 **위로** 스쳐 지나간 거의 수평인 광선까지 free 판정에 쓰여,
가구 뒤 가려진 바닥이 free로 칠해진다. 그러면 거기에 frontier가 안 생겨 탐색이
"다 봤다"로 조기 종료된다.

이 파일은 가구 뒤가 UNKNOWN으로 남는지(폐색)를 고정한다.
"""

import math
import unittest

import numpy as np

from sysnav import config
from sysnav.exploration.coverage_planner import CoveragePlanner

SENSOR_H = 0.75          # 바닥 위 센서 높이 (state_estimation z)


def _scan(robot, walls, furniture, elevations=32, az_step_deg=0.5):
    """3D LiDAR 스캔을 흉내낸다. 반환은 센서 좌표계 (x, y, z).

    거리 방향 전진을 벡터화했다 - 광선마다 파이썬 루프를 돌면 이 파일 하나가
    30초를 넘어간다. 수직 FOV(-28~+33도)는 실제 센서 값(local_planner.launch의
    minDyObsVFOV/maxDyObsVFOV)을 따른다."""
    walls_a = np.asarray(walls, dtype=np.float64).reshape(-1, 4)
    furn_a = np.asarray(furniture, dtype=np.float64).reshape(-1, 5) if furniture else np.zeros((0, 5))
    steps = np.arange(0.1, 20.0, 0.04)
    points = []
    for elevation in np.radians(np.linspace(-28, 33, elevations)):
        slope = math.tan(elevation)
        z_abs = SENSOR_H + steps * slope
        for azimuth in np.arange(0, 2 * math.pi, math.radians(az_step_deg)):
            xs = robot[0] + steps * math.cos(azimuth)
            ys = robot[1] + steps * math.sin(azimuth)
            blocked = (z_abs <= 0.0) | (z_abs >= 2.7)
            for x0, y0, x1, y1 in walls_a:
                blocked |= (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1) & (z_abs <= 2.6)
            for x0, y0, x1, y1, height in furn_a:
                blocked |= (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1) & (z_abs <= height)
            index = int(np.argmax(blocked)) if blocked.any() else -1
            if index >= 0:
                distance = steps[index]
                points.append((distance * math.cos(azimuth),
                               distance * math.sin(azimuth),
                               distance * slope))
    return np.array(points, dtype=np.float32)


def _planner():
    planner = CoveragePlanner()
    planner.origin_x = planner.origin_y = -30.0
    return planner


class OcclusionTest(unittest.TestCase):
    #        벽으로 둘러싼 6x5m 방, 소파(0.80m)가 로봇 정면을 가린다.
    WALLS = [(0, 0, 6, 0.1), (0, 4.9, 6, 5), (0, 0, 0.1, 5), (5.9, 0, 6, 5)]
    SOFA = (2.0, 0.3, 4.2, 1.1, 0.80)
    ROBOT = (0.6, 0.6)

    def setUp(self):
        self.planner = _planner()
        scan = _scan(self.ROBOT, self.WALLS, [self.SOFA])
        self.planner.update_from_scan(scan, {"x": self.ROBOT[0], "y": self.ROBOT[1], "yaw": 0.0})

    def _state(self, x, y):
        return self.planner.grid[self.planner.world_to_grid(x, y)]

    def test_space_behind_furniture_stays_unknown(self):
        """소파 뒤(로봇에서 소파 너머)는 관측된 적이 없으므로 UNKNOWN."""
        for x in (4.6, 5.0, 5.4):
            with self.subTest(x=x):
                self.assertEqual(self._state(x, 0.7), config.OCC_UNKNOWN)

    def test_space_before_furniture_is_free(self):
        self.assertEqual(self._state(1.5, 0.7), config.OCC_FREE)

    def test_furniture_itself_is_occupied(self):
        near_face = [self._state(x, 0.7) for x in (2.1, 2.3)]
        self.assertIn(config.OCC_OCCUPIED, near_face)

    def test_unknown_region_produces_frontier(self):
        """가려진 공간이 UNKNOWN이어야 그 경계에 frontier가 생기고 탐색이 이어진다."""
        self.assertGreater(self.planner.map_stats()["frontier_cells"], 0)


class ScanCostTest(unittest.TestCase):
    def test_single_scan_stays_well_under_the_update_budget(self):
        """스캔 한 번의 갱신 비용이 매핑 주기 안에 들어오는지
        (MAP_UPDATE_INTERVAL_SEC = 0.35s)."""
        import time
        planner = _planner()
        scan = _scan((3.0, 2.5), OcclusionTest.WALLS, [OcclusionTest.SOFA])
        started = time.perf_counter()
        planner.update_from_scan(scan, {"x": 3.0, "y": 2.5, "yaw": 0.0})
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, config.MAP_UPDATE_INTERVAL_SEC / 2)


if __name__ == "__main__":
    unittest.main()
