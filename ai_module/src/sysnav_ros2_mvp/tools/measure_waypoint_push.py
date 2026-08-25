#!/usr/bin/env python3
"""base autonomy가 우리 waypoint를 얼마나 밀어내는지 실측한다.

우리가 /way_point_with_heading(Pose2D)로 보낸 좌표를 waypointConverter가 그대로 쓰지
않고, obstacleDisThre(0.75m) 안에 장애물이 없는 travArea 점으로 갈아끼운 뒤
/way_point(PointStamped)로 내보낸다(waypointConverter.cpp의 waypointAdj 분기).
이 스크립트는 두 토픽을 같이 듣고 요청↔실제의 차이를 요청 건별로 정리한다.

주의 - 그냥 두 토픽의 차이를 다 세면 안 된다. waypointConverter는 목표에 도달한 뒤
(waypointReached) /way_point를 "차량 앞 waypointProjDis(0.5m)" 지점으로 계속 재발행한다.
그 구간을 포함하면 실제로는 밀어낸 적 없는데도 큰 값이 잔뜩 잡힌다. 그래서
  - 요청 직후 MEASURE_WINDOW_SEC 안의 값만 쓰고
  - 실제 좌표가 로봇에서 waypointProjDis 근처(±PROJ_TOLERANCE_M)면 projection으로 보고 제외
한다.

실행 (sysnav 컨테이너 안):
    python3 /home/docker/ai_module/src/sysnav_ros2_mvp/tools/measure_waypoint_push.py
Ctrl-C로 끝내면 요약이 나오고 ai_module/debug/waypoint_push_report.txt로도 저장된다.


⚠️  개발 전용. 이 스크립트는 /way_point를 구독하는데, 그 토픽은 README의
System Outputs 표(테스트 때 사용 가능한 토픽)에 없다. 규정은 개발 중에는
시뮬레이터가 주는 무엇이든 써도 된다고 하지만("During training/development,
you are free to use whatever information the system simulator provides"),
테스트 때는 안 된다. launch에 물리지 말고 손으로만 돌릴 것.
sysnav 런타임에서는 /way_point 구독을 제거했다(tests/test_allowed_topics.py).
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PointStamped, Pose2D
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

# waypointConverter/terrainAnalysis launch 파라미터의 복제본. 그쪽이 바뀌면 같이 고칠 것.
OBSTACLE_DIS_THRE = 0.75
OBSTACLE_HEIGHT_THRE = 0.05
WAYPOINT_PROJ_DIS = 0.5

MEASURE_WINDOW_SEC = 3.0
PROJ_TOLERANCE_M = 0.15
REPORT_PATH = Path("/home/docker/ai_module/debug/waypoint_push_report.txt")


class WaypointPushMeasurer(Node):
    def __init__(self) -> None:
        super().__init__("waypoint_push_measurer")
        self.requests: list[dict] = []
        self.pose: tuple[float, float] | None = None
        self.terrain_free_ratio: float | None = None
        self.terrain_counts: tuple[int, int] | None = None

        self.create_subscription(Pose2D, "/way_point_with_heading", self._on_request, 10)
        self.create_subscription(PointStamped, "/way_point", self._on_actual, 10)
        self.create_subscription(Odometry, "/state_estimation", self._on_pose,
                                 qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "/terrain_map", self._on_terrain,
                                 qos_profile_sensor_data)
        self.get_logger().info(
            "listening: /way_point_with_heading (ours) vs /way_point (base autonomy). Ctrl-C to stop."
        )

    def _on_pose(self, msg: Odometry) -> None:
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _on_terrain(self, msg: PointCloud2) -> None:
        """terrain_map을 waypointConverter와 같은 규칙으로 갈라 본다: intensity(지면 위
        높이)가 OBSTACLE_HEIGHT_THRE 미만이면 travArea, 이상이면 obstacleArea."""
        free = obstacle = 0
        for _, _, _, intensity in point_cloud2.read_points(
            msg, field_names=("x", "y", "z", "intensity"), skip_nans=True
        ):
            if intensity < OBSTACLE_HEIGHT_THRE:
                free += 1
            else:
                obstacle += 1
        total = free + obstacle
        if total:
            self.terrain_counts = (free, obstacle)
            self.terrain_free_ratio = free / total

    def _on_request(self, msg: Pose2D) -> None:
        self.requests.append({
            "time": time.monotonic(),
            "requested": (msg.x, msg.y),
            "robot": self.pose,
            "terrain": self.terrain_counts,
            "samples": [],
        })
        self.get_logger().info(f"request #{len(self.requests)}: ({msg.x:.2f}, {msg.y:.2f})")

    def _on_actual(self, msg: PointStamped) -> None:
        if not self.requests:
            return
        entry = self.requests[-1]
        elapsed = time.monotonic() - entry["time"]
        if elapsed > MEASURE_WINDOW_SEC:
            return
        actual = (msg.point.x, msg.point.y)
        # 도달 후 projection 모드(차량 앞 0.5m 재발행)는 "밀어냄"이 아니므로 제외한다.
        if self.pose is not None:
            from_robot = math.dist(self.pose, actual)
            if abs(from_robot - WAYPOINT_PROJ_DIS) <= PROJ_TOLERANCE_M:
                return
        entry["samples"].append(math.dist(entry["requested"], actual))
        entry["actual"] = actual

    # ------------------------------------------------------------------

    def report(self) -> str:
        lines = ["=" * 78,
                 "base autonomy가 우리 waypoint를 밀어낸 거리 (obstacleDisThre=%.2fm)" % OBSTACLE_DIS_THRE,
                 "=" * 78, ""]
        measured = [e for e in self.requests if e["samples"]]
        if not measured:
            lines.append("측정된 요청이 없습니다. sysnav가 waypoint를 발행했는지,")
            lines.append("waypointConverter가 떠 있는지(ros2 topic hz /way_point) 확인하세요.")
            return "\n".join(lines)

        lines.append(" #   requested          actual             밀림     로봇거리  terrain(free/obs)")
        lines.append(" " + "-" * 76)
        pushes = []
        for index, entry in enumerate(measured, 1):
            push = statistics.median(entry["samples"])
            pushes.append(push)
            actual = entry.get("actual", (float("nan"), float("nan")))
            robot_distance = (math.dist(entry["robot"], entry["requested"])
                              if entry["robot"] else float("nan"))
            terrain = entry["terrain"]
            terrain_text = f"{terrain[0]}/{terrain[1]}" if terrain else "-"
            lines.append(
                "%2d   (%6.2f,%6.2f)   (%6.2f,%6.2f)   %5.2fm   %5.2fm   %s"
                % (index, *entry["requested"], *actual, push, robot_distance, terrain_text)
            )

        lines += ["", "요약", "-" * 40,
                  "요청 수                : %d" % len(measured),
                  "밀림 중앙값            : %.2f m" % statistics.median(pushes),
                  "밀림 평균              : %.2f m" % statistics.fmean(pushes),
                  "밀림 최대              : %.2f m" % max(pushes),
                  "0.30m 초과 비율        : %.0f%% (%d/%d)"
                  % (100 * sum(p > 0.30 for p in pushes) / len(pushes),
                     sum(p > 0.30 for p in pushes), len(pushes)),
                  "0.75m 초과 비율        : %.0f%% (%d/%d)"
                  % (100 * sum(p > OBSTACLE_DIS_THRE for p in pushes) / len(pushes),
                     sum(p > OBSTACLE_DIS_THRE for p in pushes), len(pushes))]
        if self.terrain_free_ratio is not None:
            lines.append("terrain free 비율(최근): %.0f%%" % (100 * self.terrain_free_ratio))
        lines += ["",
                  "해석: 밀림이 0에 가까우면 우리 좌표가 그대로 채택된 것이고, 크면",
                  "waypointConverter가 obstacleDisThre 조건 때문에 다른 지점으로 갈아끼운 것이다.",
                  "우리 planner의 ROBOT_CLEARANCE_M(0.20m)과 obstacleDisThre(0.75m) 차이가",
                  "그대로 드러나는 값이다."]
        return "\n".join(lines)


def main() -> int:
    rclpy.init()
    node = WaypointPushMeasurer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        report = node.report()
        print("\n" + report)
        try:
            REPORT_PATH.write_text(report, encoding="utf-8")
            print(f"\nsaved: {REPORT_PATH}")
        except OSError as error:
            print(f"\ncould not save report: {error}")
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
