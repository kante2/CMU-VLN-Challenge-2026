"""커버리지(논문의 covered set)는 카메라 인식이 실제로 돈 프레임에서만 늘어야 한다.

배경: occupancy grid는 경로계획/방 분할 때문에 스캔마다 갱신해야 하지만, "이 표면을
봤다"는 회계를 스캔마다 하면 인식이 PERCEPTION_WHILE_MOVING_INTERVAL_SEC(1.5초)마다만
도는 사이에 지나친 표면까지 덮은 것으로 계산된다. 물체를 보는 건 카메라이므로 그건
커버리지가 아니다. update_from_scan(credit_coverage=...)이 그 두 회계를 나눈다.
"""

import math
import unittest

import numpy as np

from sysnav import config
from sysnav.exploration.coverage_planner import CoveragePlanner


POSE = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}


def ring_scan(radius: float = 2.0, count: int = 180) -> np.ndarray:
    """반경 radius의 벽에 부딪힌 360도 스캔 (센서 프레임, T_SENSOR_TO_BASE = I)."""
    angles = np.linspace(-math.pi, math.pi, count, endpoint=False)
    return np.column_stack([
        radius * np.cos(angles), radius * np.sin(angles), np.full(count, 0.3)
    ]).astype(np.float64)


class CoverageCreditTest(unittest.TestCase):
    def setUp(self):
        self.planner = CoveragePlanner()
        self.planner.reset(POSE)

    def test_mapping_only_scan_updates_the_grid_but_not_coverage(self):
        self.planner.update_from_scan(ring_scan(), POSE, credit_coverage=False)

        self.assertGreater(int((self.planner.grid == config.OCC_FREE).sum()), 0)
        self.assertEqual(int(self.planner.observed.sum()), 0)

    def test_perception_frame_credits_coverage(self):
        self.planner.update_from_scan(ring_scan(), POSE, credit_coverage=True)

        self.assertGreater(int(self.planner.observed.sum()), 0)

    def test_uncredited_scans_leave_the_surface_uncovered(self):
        """지도는 다 찼는데 인식이 안 돌았으면 Ŝ는 그대로 남아야 한다 - 그래야 탐사가
        "아직 카메라가 안 본 표면"을 계속 목표로 삼는다."""
        self.planner.update_from_scan(ring_scan(), POSE, credit_coverage=False)

        report = self.planner.surface_coverage()

        self.assertGreater(report["total"], 0)
        self.assertEqual(report["covered"], 0)

    def test_credited_scan_covers_surface_within_the_observe_radius(self):
        self.planner.update_from_scan(ring_scan(), POSE, credit_coverage=True)

        report = self.planner.surface_coverage()

        self.assertGreater(report["covered"], 0)

    def test_default_credits_coverage_for_simulation_harnesses(self):
        """오프라인 하니스는 "관측 한 번 = 인식 한 번"이라 기본값이 True다."""
        self.planner.update_from_scan(ring_scan(), POSE)

        self.assertGreater(int(self.planner.observed.sum()), 0)


if __name__ == "__main__":
    unittest.main()
