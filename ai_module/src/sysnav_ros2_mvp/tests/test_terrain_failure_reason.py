"""접근/스냅 실패 사유를 커버리지와 클리어런스로 구분한다.

실측(2026-08-23): 6분간 RETARGET_FAIL 1039회, SNAP_FAIL 1125회가 나왔는데 메시지가
전부 "no supported point" / "no point with 0.75m clearance" 한 문장이라, 아래 둘 중
무엇이 문제인지 알 수 없어 접근 지점 로직을 고칠 근거가 없었다.

  커버리지 탈락  - 그 자리에 travArea 점이 아예 없다 ("아직 안 가봤다")
  클리어런스 탈락 - 점은 있는데 장애물에서 0.75m를 못 벌린다 ("접근 자체가 불가")

전자는 더 탐사하면 풀리고, 후자는 아무리 탐사해도 안 풀린다. 대응이 정반대라
로그가 반드시 구분해야 한다.
"""

import sys
import types
import unittest

import numpy as np

# terrain_monitor는 /terrain_map 파싱에만 sensor_msgs_py를 쓴다. 여기서는 순수 기하
# 판정만 보므로 ROS 없는 환경에서도 import되도록 stub으로 대체한다
# (tests/test_terrain_snap.py와 같은 패턴).
_stub_name = "sensor_msgs_py.point_cloud2"
if _stub_name not in sys.modules:
    package = types.ModuleType("sensor_msgs_py")
    module = types.ModuleType(_stub_name)
    module.read_points = lambda *args, **kwargs: []
    package.point_cloud2 = module
    sys.modules.setdefault("sensor_msgs_py", package)
    sys.modules[_stub_name] = module

from sysnav import config  # noqa: E402
from sysnav.navigation.terrain_monitor import TerrainMonitor  # noqa: E402


def _grid(x0, x1, y0, y1, step=0.05):
    xs = np.arange(x0, x1, step)
    ys = np.arange(y0, y1, step)
    return np.array([(float(x), float(y)) for x in xs for y in ys], dtype=np.float64)


class SupportDetailTest(unittest.TestCase):
    def test_no_travarea_nearby_reports_no_coverage(self):
        trav = _grid(5.0, 6.0, 5.0, 6.0)          # 목표에서 멀리 떨어진 곳에만 있음
        ok, clearance = TerrainMonitor._support_detail(trav, np.empty((0, 2)),
                                                       np.array([0.0, 0.0]))
        self.assertFalse(ok)
        self.assertIsNone(clearance, "커버리지 탈락은 clearance=None으로 구분된다")

    def test_open_floor_passes(self):
        trav = _grid(-1.0, 1.0, -1.0, 1.0)
        ok, clearance = TerrainMonitor._support_detail(trav, np.empty((0, 2)),
                                                       np.array([0.0, 0.0]))
        self.assertTrue(ok)

    def test_obstacle_too_close_reports_the_best_clearance(self):
        # 사방을 클리어런스보다 가깝게 둘러싼다. 벽 하나만 두면 반대쪽 점이 여유를
        # 확보해 통과하므로, 임계값이 바뀌어도 유효하도록 임계값에서 거리를 정한다.
        wall = config.TERRAIN_CLEARANCE_M - 0.15
        trav = _grid(-0.2, 0.2, -0.2, 0.2)
        obstacle = np.vstack([
            _grid(-2.0, 2.0, wall, wall + 0.1), _grid(-2.0, 2.0, -wall - 0.1, -wall),
            _grid(wall, wall + 0.1, -2.0, 2.0), _grid(-wall - 0.1, -wall, -2.0, 2.0),
        ])
        ok, clearance = TerrainMonitor._support_detail(trav, obstacle,
                                                       np.array([0.0, 0.0]))
        self.assertFalse(ok)
        self.assertIsNotNone(clearance, "커버리지는 통과했으므로 숫자가 나와야 한다")
        self.assertLess(clearance, config.TERRAIN_CLEARANCE_M)
        self.assertGreater(clearance, 0.0)

    def test_supported_matches_the_detail_verdict(self):
        trav = _grid(-1.0, 1.0, -1.0, 1.0)
        obstacle = _grid(0.30, 0.40, -1.0, 1.0)
        self.assertEqual(
            TerrainMonitor._supported(trav, obstacle, np.array([0.0, 0.0])),
            TerrainMonitor._support_detail(trav, obstacle, np.array([0.0, 0.0]))[0],
        )


class ApproachFailureMessageTest(unittest.TestCase):
    def _monitor(self, trav, obstacle):
        monitor = TerrainMonitor()
        monitor._trav = trav
        monitor._obstacle = obstacle
        monitor._updated_time = float("inf")      # ready() 통과용
        monitor.ready = lambda: True
        return monitor

    def test_unobserved_area_says_so(self):
        monitor = self._monitor(_grid(20.0, 21.0, 20.0, 21.0), np.empty((0, 2)))
        self.assertIsNone(monitor.choose_approach_point((0.0, 0.0), (3.0, 0.0)))
        self.assertIn("unobserved", monitor.last_selection)

    def test_enclosed_object_reports_the_clearance_shortfall(self):
        """캐비닛 위 화병처럼 사방이 막힌 물체. 아무리 탐사해도 안 풀린다."""
        trav = _grid(-4.0, 4.0, -4.0, 4.0, step=0.10)
        obstacle = _grid(-4.0, 4.0, -4.0, 4.0, step=0.10)   # 온통 장애물
        monitor = self._monitor(trav, obstacle)
        self.assertIsNone(monitor.choose_approach_point((0.0, 0.0), (3.0, 0.0)))
        self.assertIn("best clearance", monitor.last_selection)
        self.assertIn("unobserved", monitor.last_selection)


class SnapFailureMessageTest(unittest.TestCase):
    def test_snap_failure_reports_the_best_margin(self):
        monitor = TerrainMonitor()
        monitor._trav = _grid(-0.5, 0.5, -0.5, 0.5)
        monitor._obstacle = _grid(-0.5, 0.5, -0.5, 0.5)     # 후보마다 장애물이 붙어있음
        monitor._updated_time = float("inf")
        monitor.ready = lambda: True
        self.assertIsNone(monitor.nearest_commandable(0.0, 0.0, (0.0, 0.0)))
        self.assertIn("best", monitor.last_selection)

    def test_open_floor_snaps_and_reports_the_distance(self):
        monitor = TerrainMonitor()
        monitor._trav = _grid(-2.0, 2.0, -2.0, 2.0, step=0.10)
        monitor._obstacle = np.empty((0, 2))
        monitor._updated_time = float("inf")
        monitor.ready = lambda: True
        self.assertIsNotNone(monitor.nearest_commandable(0.3, 0.3, (0.0, 0.0)))
        self.assertIn("snap=", monitor.last_selection)


if __name__ == "__main__":
    unittest.main()
