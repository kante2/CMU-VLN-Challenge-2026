#!/usr/bin/env python3
"""waypointConverter가 우리 좌표를 어디로 옮기는지 read-only로 재현한다.

/terrain_map과 /state_estimation만 구독하고 **아무것도 발행하지 않는다** - 로봇을 움직이지
않고도 "이 좌표를 보내면 실제로는 어디로 갈까"를 알 수 있다. waypointConverter.cpp의
poseHandler / waypointAdj 분기를 그대로 옮겨왔다:

    travArea    = terrain_map 중 intensity <  obstacleHeightThre(0.05)
    obstacleArea= terrain_map 중 intensity >= obstacleHeightThre
    (둘 다 terrainVoxelSize(0.05) voxel로 downsample)

    후보 = travArea 점 중 차량에서 searchDisThre(5.0m) 이내이고,
           모든 obstacleArea 점에서 obstacleDisThre(0.75m) 이상 떨어진 것
    선택 = argmin( |p - 요청좌표| + vehicleDisWeight(0.5) * |p - 차량| )

또 하나 같이 본다: waypointConverter는 도달 판정을 **원래 좌표가 아니라 스냅된 좌표**
기준으로 한다(`dis = |vehicle - waypointX2|`, `dis < waypointXYRadius(0.3)`). 그래서 스냅이
로봇 근처로 떨어지면 우리 목표는 몇 m 밖인데도 즉시 "Waypoint reached"가 뜬다. 각 probe에
대해 그 조건도 같이 표시한다.

실행 (system 컨테이너 안):
    python3 /path/to/probe_waypoint_push.py                # 로봇 주변 링을 훑는다
    python3 /path/to/probe_waypoint_push.py 2.84 1.43      # 특정 좌표 하나만 본다
    python3 /path/to/probe_waypoint_push.py 2.84,1.43 3.1,0.8

좌표를 주면 링 스캔 대신 그 좌표들만 조사한다. sysnav 로그의
`target goal (X, Y) is not commandable`에 찍힌 좌표를 그대로 넣어서, waypointConverter가
그 좌표로 실제 무엇을 하는지(후보를 찾는가 / 어디로 스냅하는가 / 즉시 "도착" 처리되는가)를
로봇을 움직이지 않고 확인하는 용도다.
"""

from __future__ import annotations

import math
import statistics
import sys

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

# waypoint_converter.launch / terrain_analysis.launch 파라미터의 복제본.
OBSTACLE_HEIGHT_THRE = 0.05
OBSTACLE_DIS_THRE = 0.75
SEARCH_DIS_THRE = 5.0
VEHICLE_DIS_WEIGHT = 0.5
WAYPOINT_XY_RADIUS = 0.3
TERRAIN_VOXEL_SIZE = 0.05

PROBE_RADII_M = (1.0, 1.5, 2.0, 2.5, 3.0)
PROBE_ANGLES_DEG = tuple(range(0, 360, 30))


def voxel_downsample(points: np.ndarray, leaf: float) -> np.ndarray:
    if len(points) == 0:
        return points
    keys = np.floor(points / leaf).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(index)]


class WaypointPushProbe(Node):
    def __init__(self) -> None:
        super().__init__("waypoint_push_probe")
        self.pose: tuple[float, float, float] | None = None
        self.trav: np.ndarray | None = None
        self.obstacle: np.ndarray | None = None
        self.create_subscription(Odometry, "/state_estimation", self._on_pose,
                                 qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "/terrain_map", self._on_terrain,
                                 qos_profile_sensor_data)

    def _on_pose(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self.pose = (p.x, p.y, p.z)

    def _on_terrain(self, msg: PointCloud2) -> None:
        raw = point_cloud2.read_points_numpy(
            msg, field_names=("x", "y", "z", "intensity"), skip_nans=True
        )
        if raw.size == 0:
            return
        free = raw[raw[:, 3] < OBSTACLE_HEIGHT_THRE][:, :3]
        blocked = raw[raw[:, 3] >= OBSTACLE_HEIGHT_THRE][:, :3]
        self.trav = voxel_downsample(free, TERRAIN_VOXEL_SIZE)
        self.obstacle = voxel_downsample(blocked, TERRAIN_VOXEL_SIZE)

    def ready(self) -> bool:
        return self.pose is not None and self.trav is not None and self.obstacle is not None

    # ------------------------------------------------------------------
    # waypointConverter.cpp 의 waypointAdj 분기 재현
    # ------------------------------------------------------------------

    def snap(self, requested: tuple[float, float]) -> dict:
        vx, vy, _ = self.pose
        trav_xy = self.trav[:, :2]
        obstacle_xy = self.obstacle[:, :2]

        from_vehicle = np.hypot(trav_xy[:, 0] - vx, trav_xy[:, 1] - vy)
        in_range = from_vehicle <= SEARCH_DIS_THRE
        candidates = trav_xy[in_range]
        if len(candidates) == 0:
            return {"actual": None, "reason": "travArea 점이 반경 안에 없음"}

        # 모든 obstacleArea 점에서 obstacleDisThre 이상 떨어졌는지 (원본의 이중 루프)
        clear = np.full(len(candidates), True)
        if len(obstacle_xy):
            for start in range(0, len(candidates), 2048):
                chunk = candidates[start:start + 2048]
                d = np.hypot(chunk[:, 0, None] - obstacle_xy[None, :, 0],
                             chunk[:, 1, None] - obstacle_xy[None, :, 1])
                clear[start:start + 2048] = d.min(axis=1) >= OBSTACLE_DIS_THRE

        usable = candidates[clear]
        if len(usable) == 0:
            return {"actual": None, "reason": "obstacleDisThre를 만족하는 점이 하나도 없음",
                    "n_candidates": int(in_range.sum())}

        cost = (np.hypot(usable[:, 0] - requested[0], usable[:, 1] - requested[1])
                + VEHICLE_DIS_WEIGHT * np.hypot(usable[:, 0] - vx, usable[:, 1] - vy))
        best = usable[int(np.argmin(cost))]
        return {"actual": (float(best[0]), float(best[1])),
                "n_candidates": int(in_range.sum()),
                "n_usable": int(len(usable))}

    def clearance_at(self, point: tuple[float, float]) -> float:
        if self.obstacle is None or not len(self.obstacle):
            return float("inf")
        return float(np.min(np.hypot(self.obstacle[:, 0] - point[0],
                                     self.obstacle[:, 1] - point[1])))

    # ------------------------------------------------------------------

    def run(self, probes: list[tuple[float, float]] | None = None) -> str:
        vx, vy, _ = self.pose
        lines = ["=" * 96,
                 "waypointConverter가 우리 좌표를 옮기는 정도 (read-only 재현, 로봇 안 움직임)",
                 f"로봇 위치 ({vx:.2f}, {vy:.2f}) | travArea {len(self.trav)}점 / obstacleArea {len(self.obstacle)}점",
                 f"규칙: obstacleDisThre={OBSTACLE_DIS_THRE}m, searchDisThre={SEARCH_DIS_THRE}m, "
                 f"vehicleDisWeight={VEHICLE_DIS_WEIGHT}, waypointXYRadius={WAYPOINT_XY_RADIUS}m",
                 "=" * 96, "",
                 " 거리  각도    요청좌표          요청지점    실제 도착지        밀림    즉시'도착'?",
                 "                                 clearance",
                 "-" * 96]

        # 좌표를 명시적으로 받았으면 그것만, 아니면 로봇 주변 링을 훑는다.
        if probes:
            targets = [
                (math.dist((vx, vy), point),
                 math.degrees(math.atan2(point[1] - vy, point[0] - vx)),
                 point)
                for point in probes
            ]
        else:
            targets = [
                (radius, float(angle),
                 (vx + radius * math.cos(math.radians(angle)),
                  vy + radius * math.sin(math.radians(angle))))
                for radius in PROBE_RADII_M
                for angle in PROBE_ANGLES_DEG
            ]

        pushes, instant_reached, unusable = [], 0, 0
        for radius, angle, requested in targets:
            clearance = self.clearance_at(requested)
            result = self.snap(requested)
            actual = result.get("actual")
            if actual is None:
                unusable += 1
                lines.append("%4.1fm %4.0f°  (%6.2f,%6.2f)   %5.2fm     %-18s   -       -"
                             % (radius, angle, *requested, clearance, result["reason"][:18]))
                continue
            push = math.dist(requested, actual)
            pushes.append(push)
            # 스냅 지점이 로봇에서 waypointXYRadius 안이면 그 즉시 "도착"으로 처리된다.
            reached_now = math.dist((vx, vy), actual) < WAYPOINT_XY_RADIUS
            instant_reached += reached_now
            lines.append("%4.1fm %4.0f°  (%6.2f,%6.2f)   %5.2fm     (%6.2f,%6.2f)   %5.2fm   %s"
                         % (radius, angle, *requested, clearance, *actual, push,
                            "예 ← 즉시 도착 처리" if reached_now else ""))

        total = len(pushes) + unusable
        lines += ["", "요약", "-" * 50]
        if pushes:
            lines += [
                "probe 지점 수            : %d" % total,
                "밀림 중앙값              : %.2f m" % statistics.median(pushes),
                "밀림 평균 / 최대         : %.2f m / %.2f m" % (statistics.fmean(pushes), max(pushes)),
                "0.30m 넘게 밀린 비율     : %.0f%% (%d/%d)"
                % (100 * sum(p > 0.30 for p in pushes) / len(pushes), sum(p > 0.30 for p in pushes), len(pushes)),
                "1.00m 넘게 밀린 비율     : %.0f%% (%d/%d)"
                % (100 * sum(p > 1.00 for p in pushes) / len(pushes), sum(p > 1.00 for p in pushes), len(pushes)),
                "스냅이 로봇 0.3m 안 (즉시 '도착' 처리) : %d/%d" % (instant_reached, len(pushes)),
            ]
        if unusable:
            lines.append("유효 지점 자체가 없던 probe : %d/%d" % (unusable, total))

        free_ratio_note = self.clearance_summary()
        lines += ["", free_ratio_note]
        return "\n".join(lines)

    def clearance_summary(self) -> str:
        """travArea 점들이 obstacleDisThre를 얼마나 통과하는지 - "이 방에서 애초에
        목표로 찍을 수 있는 지점이 몇 %인가"에 해당한다."""
        trav_xy = self.trav[:, :2]
        obstacle_xy = self.obstacle[:, :2]
        if not len(trav_xy) or not len(obstacle_xy):
            return "clearance 요약: 데이터 부족"
        clear = np.empty(len(trav_xy), dtype=float)
        for start in range(0, len(trav_xy), 2048):
            chunk = trav_xy[start:start + 2048]
            d = np.hypot(chunk[:, 0, None] - obstacle_xy[None, :, 0],
                         chunk[:, 1, None] - obstacle_xy[None, :, 1])
            clear[start:start + 2048] = d.min(axis=1)
        parts = ["travArea %d점 중 목표로 삼을 수 있는 비율:" % len(trav_xy)]
        for threshold in (0.20, 0.25, 0.50, OBSTACLE_DIS_THRE):
            ratio = 100.0 * float((clear >= threshold).mean())
            tag = "  <- base autonomy 기준" if threshold == OBSTACLE_DIS_THRE else ""
            parts.append("   clearance >= %.2fm : %5.1f%%%s" % (threshold, ratio, tag))
        return "\n".join(parts)


def parse_probes(argv: list[str]) -> list[tuple[float, float]]:
    """"2.84 1.43" / "2.84,1.43" / 여러 좌표 섞어쓰기를 전부 받는다."""
    numbers: list[float] = []
    for token in argv:
        for piece in token.replace(",", " ").split():
            numbers.append(float(piece))
    if len(numbers) % 2:
        raise SystemExit("좌표는 x y 쌍으로 주세요 (예: 2.84 1.43)")
    return [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers), 2)]


def main() -> int:
    probes = parse_probes(sys.argv[1:])
    rclpy.init()
    node = WaypointPushProbe()
    deadline = node.get_clock().now().nanoseconds + int(20e9)
    while rclpy.ok() and not node.ready():
        rclpy.spin_once(node, timeout_sec=0.2)
        if node.get_clock().now().nanoseconds > deadline:
            print("terrain_map / state_estimation 을 20초 안에 못 받았습니다.")
            node.destroy_node()
            rclpy.shutdown()
            return 1
    # 최신 terrain 몇 프레임을 더 받아 안정된 스냅샷을 쓴다.
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.2)
    print(node.run(probes))
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
