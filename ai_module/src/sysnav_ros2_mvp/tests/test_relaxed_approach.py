"""명령 가능한 지점이 희박한 씬에서 접근점을 못 찾아 굳는 문제.

실측 2026-08-24 (tools/probe_waypoint_push.py, 로봇 (2.19,-0.53) / 목표 (2.84,1.43)):
    travArea 1066점 중 clearance >= 0.75m 통과: 7.1% (76점)
    목표 좌표 자체의 clearance: 0.42m
    waypointConverter가 스냅할 지점: (1.89,-0.33) = 로봇에서 0.36m -> 갈 거리가 없음

Mission 3의 링 샘플링은 MISSION3_OBJECT_APPROACH_MAX_M(0.9m) 때문에 링 하나 x 7각도 =
후보 7개만 본다. 통과 비율이 7%인 씬에서 그 7개가 걸릴 리 없으니 거의 항상 실패하고,
terrain을 아예 안 보는 고정 standoff로 폴백해 명령 불가한 좌표를 잡았다. 그리고 mission3는
확정된 subgoal을 절대 안 버리므로 "재발행 -> 거부 -> unreachable -> 재발행"을 영원히 돌았다.

두 가지를 고정한다:
  1. allow_relaxed=True면 링 대신 commandable set을 직접 훑어 물체에 가장 가까운
     통과 지점을 결정론적으로 찾는다.
  2. 기본값(allow_relaxed=False)에서는 mission 상한이 그대로 지켜진다 - 그 상한은
     탐색 범위가 아니라 "물체 0.9m 안에 서야 go to를 수행한 것"이라는 의미 규칙이다.
"""

import sys
import time
import types
import unittest

import numpy as np

_stub_name = "sensor_msgs_py.point_cloud2"
if _stub_name not in sys.modules:
    package = types.ModuleType("sensor_msgs_py")
    module = types.ModuleType(_stub_name)
    module.read_points = lambda *args, **kwargs: []
    package.point_cloud2 = module
    sys.modules.setdefault("sensor_msgs_py", package)
    sys.modules[_stub_name] = module

from sysnav import config                                      # noqa: E402
from sysnav.navigation.terrain_monitor import TerrainMonitor   # noqa: E402


def _monitor(trav, obstacle) -> TerrainMonitor:
    monitor = TerrainMonitor()
    monitor._trav = np.asarray(trav, dtype=np.float64).reshape(-1, 2)
    monitor._obstacle = np.asarray(obstacle, dtype=np.float64).reshape(-1, 2)
    monitor._updated_time = time.monotonic()
    return monitor


class RelaxedApproachTest(unittest.TestCase):
    """물체(0,0) 주변이 좁아서 0.9m 안에는 통과 지점이 없고, 1.8m에만 있는 상황."""

    OBJECT = (0.0, 0.0)
    ROBOT = (-3.0, 0.0)

    def setUp(self):
        # 물체 (0,0)이 폭 1.2m 좁은 통로 안에 있다 - 통로 안 어느 점도 0.75m
        # 클리어런스를 못 만든다(화장실 같은 자리의 축소판). 통로를 벗어난 -1.8m,
        # -2.4m 지점만 통과한다.
        wall_x = np.arange(-1.0, 0.45, 0.1)
        obstacle = np.concatenate([
            np.stack([wall_x, np.full_like(wall_x, 0.6)], axis=1),
            np.stack([wall_x, np.full_like(wall_x, -0.6)], axis=1),
        ])
        trav = np.array([[-0.9, 0.0], [-1.2, 0.0], [-1.8, 0.0], [-2.4, 0.0]])
        self.monitor = _monitor(trav, obstacle)

    def test_strict_mode_keeps_the_mission_limit(self):
        """평상시에는 0.9m 의미 규칙을 그대로 지킨다 - 못 찾으면 None."""
        chosen = self.monitor.choose_approach_point(
            self.OBJECT, self.ROBOT,
            max_distance_m=config.MISSION3_OBJECT_APPROACH_MAX_M,
        )
        self.assertIsNone(chosen)

    def test_relaxed_mode_finds_the_closest_commandable_point(self):
        """상한을 풀면 통과 지점 중 물체에 가장 가까운 것을 고른다."""
        chosen = self.monitor.choose_approach_point(
            self.OBJECT, self.ROBOT,
            max_distance_m=config.MISSION3_OBJECT_APPROACH_MAX_M,
            allow_relaxed=True,
        )
        self.assertIsNotNone(chosen, "완화 모드는 갈 수 있는 지점을 찾아야 한다")
        self.assertEqual(tuple(round(v, 2) for v in chosen), (-1.8, 0.0))
        self.assertIn("relaxed", self.monitor.last_selection)

    def test_relaxed_mode_never_picks_a_point_beyond_the_fallback_cap(self):
        """무제한으로 넓히면 벽 너머 다른 방의 점이 뽑힐 수 있다 - 상한이 있어야 한다."""
        far = config.TERRAIN_APPROACH_FALLBACK_MAX_M + 1.0
        monitor = _monitor(
            np.array([[-far, 0.0]]),
            np.array([[0.0, 0.0]]),
        )
        chosen = monitor.choose_approach_point(
            self.OBJECT, (-far - 1.0, 0.0),
            max_distance_m=config.MISSION3_OBJECT_APPROACH_MAX_M,
            allow_relaxed=True,
        )
        self.assertIsNone(chosen)

    def test_relaxed_mode_requires_progress_toward_the_object(self):
        """지금 로봇보다 물체에서 더 먼 지점은 "접근점"이 아니다."""
        monitor = _monitor(
            np.array([[-2.5, 0.0]]),      # 로봇(-1.0)보다 물체에서 멀다
            np.array([[0.0, 0.0]]),
        )
        chosen = monitor.choose_approach_point(
            self.OBJECT, (-1.0, 0.0),
            max_distance_m=config.MISSION3_OBJECT_APPROACH_MAX_M,
            allow_relaxed=True,
        )
        self.assertIsNone(chosen)

    def test_relaxed_mode_still_honours_the_clearance_rule(self):
        """완화되는 것은 **물체까지의 거리**뿐이다. base autonomy의 0.75m 클리어런스는
        저쪽 하드 필터라 우리가 완화할 수 있는 값이 아니다."""
        monitor = _monitor(
            np.array([[-1.5, 0.0]]),
            np.array([[-1.5, 0.5]]),      # 후보에서 0.5m -> 0.75m 미달
        )
        chosen = monitor.choose_approach_point(
            self.OBJECT, self.ROBOT,
            max_distance_m=config.MISSION3_OBJECT_APPROACH_MAX_M,
            allow_relaxed=True,
        )
        self.assertIsNone(chosen)


class CommandableRatioTest(unittest.TestCase):
    """"이 씬에서 애초에 목적지로 찍을 수 있는 곳이 몇 %인가" 진단값."""

    def test_counts_only_points_that_clear_the_threshold(self):
        monitor = _monitor(
            np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]),
            np.array([[0.0, 0.5]]),       # 첫 점만 0.5m -> 탈락
        )
        commandable, total = monitor.commandable_ratio()
        self.assertEqual((commandable, total), (2, 3))

    def test_no_obstacles_means_everything_is_commandable(self):
        monitor = _monitor(np.array([[0.0, 0.0], [1.0, 0.0]]), np.empty((0, 2)))
        self.assertEqual(monitor.commandable_ratio(), (2, 2))

    def test_empty_terrain_is_reported_as_zero(self):
        monitor = _monitor(np.empty((0, 2)), np.empty((0, 2)))
        self.assertEqual(monitor.commandable_ratio(), (0, 0))


if __name__ == "__main__":
    unittest.main()
