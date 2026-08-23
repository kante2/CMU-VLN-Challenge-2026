"""TerrainMonitor.nearest_commandable - 발행 직전 스냅(Layer 1)의 판정 로직.

base autonomy(waypointConverter)는 우리 좌표를 그대로 쓰지 않고 자기 travArea 점으로
갈아끼운다. 실측에서 요청의 93~97%가 0.3m 넘게 밀렸고 중앙값이 2.1~2.5m였다. 여기서는
"우리가 미리 옮겨 보내면 그 갈아끼우기가 안 일어난다"의 판정부를 검증한다.
"""

import sys
import time
import types
import unittest

import numpy as np

# terrain_monitor는 /terrain_map 파싱에만 sensor_msgs_py를 쓴다. 아래 테스트는 순수
# 기하 판정만 보므로, ROS 없는 환경에서도 import되도록 최소 stub으로 대체한다
# (tests/test_mission2_explore_first.py와 같은 패턴).
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


def _monitor(trav: np.ndarray, obstacle: np.ndarray) -> TerrainMonitor:
    """update()는 ROS 메시지를 받으므로, 테스트에서는 내부 상태를 직접 채운다."""
    monitor = TerrainMonitor()
    monitor._trav = np.asarray(trav, dtype=np.float64).reshape(-1, 2)
    monitor._obstacle = np.asarray(obstacle, dtype=np.float64).reshape(-1, 2)
    monitor._updated_time = time.monotonic()
    return monitor


def _grid(x0: float, x1: float, y0: float, y1: float, step: float = 0.1) -> np.ndarray:
    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    mesh = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)
    return mesh.reshape(-1, 2)


class NearestCommandableTest(unittest.TestCase):
    def test_point_with_enough_clearance_is_left_alone(self):
        """이미 base autonomy가 받아줄 지점이면 옮기지 않는다 - 옮기면 우리가 계산한
        관측 위치가 어긋난다."""
        trav = _grid(-3.0, 3.0, -3.0, 3.0)
        obstacle = np.array([[5.0, 5.0]])  # 멀리 - 어디든 clearance 충분
        monitor = _monitor(trav, obstacle)

        result = monitor.nearest_commandable(1.0, 1.0, robot_xy=(0.0, 0.0))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], 1.0, places=6)
        self.assertAlmostEqual(result[1], 1.0, places=6)

    def test_point_too_close_to_obstacle_is_moved_away(self):
        """벽에서 TERRAIN_CLEARANCE_M 안쪽을 요청하면 여유 있는 지점으로 옮긴다."""
        trav = _grid(-3.0, 3.0, -3.0, 3.0)
        obstacle = np.array([[2.0, y] for y in np.arange(-3.0, 3.01, 0.1)])  # x=2 세로벽
        monitor = _monitor(trav, obstacle)

        requested = (1.9, 0.0)  # 벽에서 0.1m
        result = monitor.nearest_commandable(*requested, robot_xy=(0.0, 0.0))
        self.assertIsNotNone(result)
        clearance = float(np.min(np.linalg.norm(obstacle - np.array(result), axis=1)))
        self.assertGreaterEqual(clearance, config.TERRAIN_CLEARANCE_M)
        # 필요한 만큼만 옮겨야 한다 - 반대편으로 던지면 안 된다.
        self.assertLessEqual(float(np.linalg.norm(np.array(result) - np.array(requested))),
                             config.TERRAIN_SNAP_MAX_M)

    def test_returns_none_when_nothing_commandable_is_near(self):
        """근처에 받아줄 지점이 없으면 조용히 엉뚱한 곳으로 보내지 않고 실패를 알린다."""
        # 관측된 좁은 패치를 사방에서 클리어런스보다 가깝게 둘러싼다. 한쪽에만 벽을
        # 두면 반대쪽 점이 여유를 확보해 통과하므로, 임계값이 바뀌어도 유효하도록
        # 거리를 config에서 끌어온다.
        wall = config.TERRAIN_CLEARANCE_M - 0.15
        trav = _grid(-0.2, 0.2, -0.2, 0.2)          # 좁은 통로만 관측됨
        obstacle = np.vstack([
            _grid(-2.0, 2.0, wall, wall + 0.1), _grid(-2.0, 2.0, -wall - 0.1, -wall),
            _grid(wall, wall + 0.1, -2.0, 2.0), _grid(-wall - 0.1, -wall, -2.0, 2.0),
        ])
        monitor = _monitor(trav, obstacle)

        self.assertIsNone(monitor.nearest_commandable(0.0, 0.0, robot_xy=(0.0, 0.0)))

    def test_returns_none_when_goal_has_no_observed_terrain(self):
        """아직 관측 안 된(=travArea 점이 없는) 영역은 우리 grid가 free로 보고 있어도
        waypointConverter의 후보가 될 수 없다."""
        trav = _grid(-1.0, 1.0, -1.0, 1.0)
        obstacle = np.array([[9.0, 9.0]])
        monitor = _monitor(trav, obstacle)

        self.assertIsNone(monitor.nearest_commandable(6.0, 6.0, robot_xy=(0.0, 0.0)))

    def test_candidates_outside_search_radius_are_rejected(self):
        """waypointConverter는 차량에서 searchDisThre 안의 travArea 점만 후보로 본다."""
        trav = _grid(-1.0, 1.0, -1.0, 1.0)
        obstacle = np.array([[50.0, 50.0]])
        monitor = _monitor(trav, obstacle)

        far_robot = (config.TERRAIN_SEARCH_DIS_M + 10.0, 0.0)
        self.assertIsNone(monitor.nearest_commandable(0.0, 0.0, robot_xy=far_robot))
        self.assertIsNotNone(monitor.nearest_commandable(0.0, 0.0, robot_xy=(0.0, 0.0)))

    def test_stale_terrain_does_not_snap(self):
        """지형 데이터가 오래되면 판정하지 않는다 - 옛 지형으로 목표를 옮기면
        없느니만 못하다. 호출 측은 원본을 그대로 발행한다."""
        monitor = _monitor(_grid(-1.0, 1.0, -1.0, 1.0), np.array([[9.0, 9.0]]))
        monitor._updated_time = time.monotonic() - config.TERRAIN_STALE_SEC - 1.0
        self.assertIsNone(monitor.nearest_commandable(0.0, 0.0, robot_xy=(0.0, 0.0)))


class HasCommandablePointsTest(unittest.TestCase):
    """"후보가 아예 없음"과 "후보는 있는데 목표 근처엔 없음"을 가르는 판정.

    waypointConverter는 후보가 하나도 없으면(`if (minInd >= 0)`가 거짓) 우리 좌표를
    그대로 쓴다. 그래서 전자는 원본을 발행해야 하고, 후자만 발행을 막아야 한다.
    """

    def test_false_when_everything_is_too_close_to_obstacles(self):
        """사방이 막힌 좁은 공간 - 어느 travArea 점도 클리어런스를 못 낸다."""
        wall = config.TERRAIN_CLEARANCE_M - 0.15
        trav = _grid(-0.3, 0.3, -0.3, 0.3)
        # 패치를 클리어런스보다 가깝게 사방으로 둘러싼다 (임계값이 바뀌어도 성립).
        obstacle = np.vstack([
            _grid(-2.0, 2.0, wall, wall + 0.1), _grid(-2.0, 2.0, -wall - 0.1, -wall),
            _grid(wall, wall + 0.1, -2.0, 2.0), _grid(-wall - 0.1, -wall, -2.0, 2.0),
        ])
        self.assertFalse(_monitor(trav, obstacle).has_commandable_points((0.0, 0.0)))

    def test_true_when_some_point_has_clearance(self):
        trav = _grid(-3.0, 3.0, -3.0, 3.0)
        obstacle = np.array([[9.0, 9.0]])
        self.assertTrue(_monitor(trav, obstacle).has_commandable_points((0.0, 0.0)))

    def test_false_when_candidates_are_outside_search_radius(self):
        trav = _grid(-1.0, 1.0, -1.0, 1.0)
        obstacle = np.array([[50.0, 50.0]])
        far = (config.TERRAIN_SEARCH_DIS_M + 10.0, 0.0)
        self.assertFalse(_monitor(trav, obstacle).has_commandable_points(far))

    def test_false_when_terrain_not_ready(self):
        monitor = _monitor(_grid(-1.0, 1.0, -1.0, 1.0), np.array([[9.0, 9.0]]))
        monitor._updated_time = time.monotonic() - config.TERRAIN_STALE_SEC - 1.0
        self.assertFalse(monitor.has_commandable_points((0.0, 0.0)))


class SnapContractTest(unittest.TestCase):
    def test_snapped_point_would_win_waypoint_converter_cost(self):
        """waypointConverter의 비용식 cost(q) = |q-목표| + 0.5*|q-차량| 에서, 우리가
        commandable 지점 p를 목표로 찍으면 p가 반드시 최소가 된다는 성질을 확인한다.
        이게 성립하기 때문에 Layer 1 스냅이 통한다 (config.TERRAIN_SNAP_MAX_M 주석)."""
        rng = np.random.default_rng(0)
        vehicle = np.array([0.0, 0.0])
        for _ in range(200):
            p = rng.uniform(-4.0, 4.0, size=2)          # 우리가 찍은 commandable 지점
            q = rng.uniform(-4.0, 4.0, size=2)          # 경쟁 후보
            cost_p = 0.0 + 0.5 * float(np.linalg.norm(p - vehicle))
            cost_q = (float(np.linalg.norm(q - p))
                      + 0.5 * float(np.linalg.norm(q - vehicle)))
            self.assertLessEqual(cost_p, cost_q + 1e-9)


if __name__ == "__main__":
    unittest.main()
