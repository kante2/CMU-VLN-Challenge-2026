""""아깝게 모자란 clearance"는 발행 불가가 아니다.

실측 2026-08-25: 목표 근처 후보 1164개의 최선 clearance가 0.74m였고 기준은 0.75m라
nearest_commandable()이 전멸 -> resolve()가 None -> 아무것도 발행 안 함 -> 로봇이 한
발짝도 못 움직임 -> 지도가 그대로라 다음 사이클에 완전히 같은 route -> 5회 반복 후
탐사 소진 -> 0:40에 FAILED(10분 예산 중 9분 20초 남음).

TERRAIN_CLEARANCE_M은 waypointConverter의 obstacleDisThre 복제본이지만 우리는
/terrain_map을 우리 방식으로 다시 계산하므로 저쪽과 cm 단위로 일치하지 않는다.
1cm 차이로 전멸시킬 근거가 없다 - 그때는 원본을 그대로 보내 저쪽이 판단하게 둔다.
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

if "rclpy" not in sys.modules:
    try:
        import rclpy  # noqa: F401
    except ImportError:                                       # pragma: no cover
        rclpy_pkg = types.ModuleType("rclpy")
        logging_module = types.ModuleType("rclpy.logging")

        class _StubLogger:
            def info(self, *a, **k): pass
            def warning(self, *a, **k): pass
            def error(self, *a, **k): pass

        logging_module.get_logger = lambda name: _StubLogger()
        rclpy_pkg.logging = logging_module
        sys.modules["rclpy"] = rclpy_pkg
        sys.modules["rclpy.logging"] = logging_module

# goal_publisher는 메시지 타입 때문에 ROS 패키지를 import한다. resolve()는 순수 기하
# 판정이라 타입 자체는 안 쓰이므로 최소 stub으로 대체한다.
for _pkg, _names in (("geometry_msgs.msg", ("Point", "Pose2D")),
                     ("visualization_msgs.msg", ("Marker", "MarkerArray"))):
    if _pkg not in sys.modules:
        try:
            __import__(_pkg)
        except ImportError:                                   # pragma: no cover
            _parent, _, _child = _pkg.rpartition(".")
            _module = types.ModuleType(_pkg)
            for _name in _names:
                setattr(_module, _name, type(_name, (), {"__init__": lambda self: None}))
            sys.modules.setdefault(_parent, types.ModuleType(_parent))
            setattr(sys.modules[_parent], _child, _module)
            sys.modules[_pkg] = _module

from sysnav import config                                          # noqa: E402
from sysnav.navigation.terrain_monitor import TerrainMonitor        # noqa: E402
from sysnav.navigation.goal_publisher import GoalPublisher          # noqa: E402


def _monitor(trav, obstacle) -> TerrainMonitor:
    monitor = TerrainMonitor()
    monitor._trav = np.asarray(trav, dtype=np.float64).reshape(-1, 2)
    monitor._obstacle = np.asarray(obstacle, dtype=np.float64).reshape(-1, 2)
    monitor._updated_time = time.monotonic()
    return monitor


def _corridor(clearance: float) -> TerrainMonitor:
    """모든 travArea 점이 장애물에서 정확히 `clearance`만큼 떨어진 좁은 통로."""
    trav = np.array([[x, 0.0] for x in np.arange(-2.0, 2.01, 0.1)])
    obstacle = np.array([[x, clearance] for x in np.arange(-2.0, 2.01, 0.1)])
    return _monitor(trav, obstacle)


class _Logger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass


class _Node:
    """goal_publisher가 쓰는 최소 인터페이스만."""

    def __init__(self, monitor, robot_xy=(0.0, 0.0)):
        self.terrain_monitor = monitor
        self.latest_pose = {"x": robot_xy[0], "y": robot_xy[1], "yaw": 0.0}
        import threading
        self.sensor_lock = threading.RLock()
        self.traces = []

    def _trace_navigation(self, kind, detail):
        self.traces.append((kind, detail))

    def get_logger(self):
        return _Logger()

    def create_publisher(self, *args, **kwargs):
        class _Pub:
            def publish(self, _msg): pass
        return _Pub()


def _publisher(monitor, robot_xy=(0.0, 0.0)):
    node = _Node(monitor, robot_xy)
    publisher = GoalPublisher.__new__(GoalPublisher)
    publisher._node = node
    return publisher, node


class BestClearanceIsRecordedTest(unittest.TestCase):
    def test_it_is_filled_when_the_filter_rejects_everything(self):
        monitor = _corridor(0.74)
        self.assertIsNone(monitor.nearest_commandable(1.0, 0.0, robot_xy=(0.0, 0.0)))
        self.assertAlmostEqual(monitor.last_best_clearance_m, 0.74, places=2)

    def test_it_is_filled_on_success_too(self):
        monitor = _corridor(1.20)
        self.assertIsNotNone(monitor.nearest_commandable(1.0, 0.0, robot_xy=(0.0, 0.0)))
        self.assertAlmostEqual(monitor.last_best_clearance_m, 1.20, places=2)

    def test_it_is_cleared_when_no_candidate_was_measured(self):
        """직전 호출의 값이 남아있으면 호출 측이 오판한다."""
        monitor = _corridor(0.74)
        monitor.nearest_commandable(1.0, 0.0, robot_xy=(0.0, 0.0))
        self.assertIsNotNone(monitor.last_best_clearance_m)
        # 목표가 travArea에서 멀어 후보 자체가 없는 호출
        self.assertIsNone(monitor.nearest_commandable(50.0, 50.0, robot_xy=(0.0, 0.0)))
        self.assertIsNone(monitor.last_best_clearance_m)


class NearMissPassthruTest(unittest.TestCase):
    def test_a_one_centimetre_miss_is_published_as_is(self):
        """보고된 케이스. 0.74m vs 기준 0.75m면 원본을 그대로 내보낸다."""
        publisher, node = _publisher(_corridor(0.74))
        resolved = publisher.resolve(1.0, 0.0, label="exploration viewpoint", trace_failure=True)
        self.assertIsNotNone(resolved, "발행을 막으면 로봇은 확실히 0m 움직인다")
        effective, snapped = resolved
        self.assertEqual(effective, (1.0, 0.0))     # 원본 그대로
        self.assertIsNone(snapped)                  # 스냅한 게 아니다
        self.assertIn("PASSTHRU_NEAR_MISS", [kind for kind, _ in node.traces])

    def test_a_large_shortfall_is_still_rejected(self):
        """near-miss가 아무거나 통과시키면 안 된다 - 0.20m는 애초에 접근 불가한 자리다."""
        publisher, node = _publisher(_corridor(0.20))
        resolved = publisher.resolve(1.0, 0.0, label="exploration viewpoint", trace_failure=True)
        self.assertNotIn("PASSTHRU_NEAR_MISS", [kind for kind, _ in node.traces])

    def test_the_boundary_follows_the_config_value(self):
        """임계를 config 대신 하드코딩하지 않았는지."""
        margin = config.TERRAIN_CLEARANCE_NEAR_MISS_M
        just_inside = config.TERRAIN_CLEARANCE_M - margin + 0.01
        just_outside = config.TERRAIN_CLEARANCE_M - margin - 0.01
        for clearance, expected in ((just_inside, True), (just_outside, False)):
            with self.subTest(clearance=round(clearance, 3)):
                publisher, node = _publisher(_corridor(clearance))
                publisher.resolve(1.0, 0.0, label="goal", trace_failure=True)
                fired = "PASSTHRU_NEAR_MISS" in [kind for kind, _ in node.traces]
                self.assertEqual(fired, expected)

    def test_a_successful_snap_never_reaches_the_near_miss_branch(self):
        publisher, node = _publisher(_corridor(1.20))
        resolved = publisher.resolve(1.0, 0.0, label="goal", trace_failure=True)
        self.assertIsNotNone(resolved)
        self.assertNotIn("PASSTHRU_NEAR_MISS", [kind for kind, _ in node.traces])


if __name__ == "__main__":
    unittest.main()
