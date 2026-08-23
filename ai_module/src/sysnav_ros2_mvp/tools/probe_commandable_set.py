#!/usr/bin/env python3
""""base autonomy가 실제로 받아주는 지점"만 써도 탐색이 되는지 판정한다 (read-only).

probe_waypoint_push.py가 "우리 좌표가 얼마나 밀리는지"를 쟀다면, 이건 그 다음 질문에
답한다: 밀리지 않는 지점(= commandable set)만 골라서 목표로 삼기로 하면,

  1) 그 집합이 몇 개나 되고 얼마나 흩어져 있는가 (localPlanner가 사이를 이어줄 수 있는가)
  2) 그 지점들에서 방 전체가 보이는가 (탐색 커버리지가 유지되는가)

commandable set 정의 - waypointConverter가 후보로 쓰는 조건 그대로:
    관측된 travArea 점(intensity < obstacleHeightThre) 이면서
    모든 obstacleArea 점에서 obstacleDisThre(0.75m) 이상 떨어진 것

아무것도 발행하지 않는다. 로봇은 움직이지 않는다.
"""

from __future__ import annotations

import sys

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

OBSTACLE_HEIGHT_THRE = 0.05
OBSTACLE_DIS_THRE = 0.75
TERRAIN_VOXEL_SIZE = 0.05
# localPlanner가 목표 사이를 알아서 이어줄 수 있다고 보는 거리(adjacentRange 3.5m).
# 이보다 가까운 commandable 점들끼리는 "이어져 있다"고 본다.
BRIDGE_DIS_M = 3.5
# 한 지점에서 관측 가능하다고 볼 반경 (config.FRONTIER_COVERAGE_RADIUS_M와 맞춤).
COVERAGE_RADIUS_M = 3.0
GRID_M = 0.20  # 커버리지 계산용 격자 = 우리 planner 해상도


def voxel_downsample(points: np.ndarray, leaf: float) -> np.ndarray:
    if len(points) == 0:
        return points
    keys = np.floor(points / leaf).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(index)]


def min_distance_to(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """points 각각에서 targets 중 가장 가까운 것까지의 거리 (메모리 절약 위해 청크 처리)."""
    out = np.empty(len(points), dtype=float)
    for start in range(0, len(points), 2048):
        chunk = points[start:start + 2048]
        d = np.hypot(chunk[:, 0, None] - targets[None, :, 0],
                     chunk[:, 1, None] - targets[None, :, 1])
        out[start:start + 2048] = d.min(axis=1)
    return out


class CommandableSetProbe(Node):
    def __init__(self) -> None:
        super().__init__("commandable_set_probe")
        self.pose = None
        self.trav = None
        self.obstacle = None
        self.create_subscription(Odometry, "/state_estimation", self._on_pose,
                                 qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "/terrain_map", self._on_terrain,
                                 qos_profile_sensor_data)

    def _on_pose(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self.pose = (p.x, p.y)

    def _on_terrain(self, msg: PointCloud2) -> None:
        raw = point_cloud2.read_points_numpy(
            msg, field_names=("x", "y", "z", "intensity"), skip_nans=True
        )
        if raw.size == 0:
            return
        self.trav = voxel_downsample(raw[raw[:, 3] < OBSTACLE_HEIGHT_THRE][:, :3], TERRAIN_VOXEL_SIZE)
        self.obstacle = voxel_downsample(raw[raw[:, 3] >= OBSTACLE_HEIGHT_THRE][:, :3], TERRAIN_VOXEL_SIZE)

    def ready(self) -> bool:
        return self.pose is not None and self.trav is not None and self.obstacle is not None

    # ------------------------------------------------------------------

    def run(self) -> str:
        trav_xy = self.trav[:, :2]
        obstacle_xy = self.obstacle[:, :2]
        lines = ["=" * 88,
                 "base autonomy가 받아주는 지점(commandable set)만으로 탐색이 되는가",
                 f"로봇 ({self.pose[0]:.2f}, {self.pose[1]:.2f}) | "
                 f"travArea {len(trav_xy)}점 / obstacleArea {len(obstacle_xy)}점",
                 "=" * 88, ""]

        if not len(obstacle_xy):
            return "\n".join(lines + ["obstacleArea가 비어 판정 불가"])

        clearance = min_distance_to(trav_xy, obstacle_xy)

        lines.append("[1] 클리어런스 기준별 후보 수")
        for threshold in (0.20, 0.25, 0.50, 0.60, OBSTACLE_DIS_THRE):
            count = int((clearance >= threshold).sum())
            tag = "   <- base autonomy 기준" if threshold == OBSTACLE_DIS_THRE else ""
            lines.append("    >= %.2fm : %5d점 (%5.1f%%)%s"
                         % (threshold, count, 100.0 * count / len(trav_xy), tag))

        commandable = trav_xy[clearance >= OBSTACLE_DIS_THRE]
        lines.append("")
        if len(commandable) == 0:
            return "\n".join(lines + ["commandable 지점이 0개 - 이 지점에서는 어떤 목표도 그대로 수용되지 않는다."])

        # [2] 흩어진 정도: BRIDGE_DIS_M 안에서 서로 이어지는 덩어리 수
        lines.append("[2] commandable 지점이 이어져 있는가 (localPlanner가 %.1fm까지 이어준다고 가정)"
                     % BRIDGE_DIS_M)
        clusters = self._cluster(commandable, BRIDGE_DIS_M)
        sizes = sorted((len(c) for c in clusters), reverse=True)
        lines.append("    덩어리 %d개, 크기 %s" % (len(clusters), sizes[:8]))
        lines.append("    최대 덩어리가 전체의 %.0f%%" % (100.0 * sizes[0] / len(commandable)))

        # [3] 커버리지: commandable 지점들에서 travArea 전체가 보이는가
        lines.append("")
        lines.append("[3] commandable 지점에서 반경 %.1fm 안에 들어오는 travArea 비율" % COVERAGE_RADIUS_M)
        covered = min_distance_to(trav_xy, commandable) <= COVERAGE_RADIUS_M
        lines.append("    %5.1f%%  (%d / %d 점)"
                     % (100.0 * covered.mean(), int(covered.sum()), len(trav_xy)))
        uncovered = trav_xy[~covered]
        if len(uncovered):
            far = min_distance_to(uncovered, commandable)
            lines.append("    못 덮는 영역: %d점, commandable에서 최대 %.2fm 떨어짐"
                         % (len(uncovered), far.max()))

        # [4] 우리 planner 기준(0.20m)으로 서 있을 수 있는 곳 대비 얼마나 줄어드는가
        ours = trav_xy[clearance >= 0.20]
        lines += ["",
                  "[4] 우리 planner(0.20m) 대비",
                  "    우리 기준 후보 %d점 -> base autonomy 수용 %d점 (%.1f배 감소)"
                  % (len(ours), len(commandable), len(ours) / max(1, len(commandable)))]

        lines += ["", "판정", "-" * 40]
        if len(clusters) == 1:
            lines.append("  연결성: OK - commandable 지점이 하나로 이어져 있어 순서대로 방문 가능")
        else:
            lines.append("  연결성: 덩어리 %d개로 나뉨 - 사이를 localPlanner가 이어주는지 확인 필요"
                         % len(clusters))
        ratio = 100.0 * covered.mean()
        if ratio >= 90:
            lines.append("  커버리지: OK (%.0f%%) - commandable 지점만으로도 방을 거의 다 관측 가능" % ratio)
        elif ratio >= 70:
            lines.append("  커버리지: 부분적 (%.0f%%) - 일부 사각지대 발생" % ratio)
        else:
            lines.append("  커버리지: 부족 (%.0f%%) - commandable 지점만으로는 방을 못 덮는다" % ratio)
        return "\n".join(lines)

    @staticmethod
    def _cluster(points: np.ndarray, link_distance: float) -> list[np.ndarray]:
        """link_distance 안에서 서로 이어지는 점들을 한 덩어리로 묶는다 (단순 BFS)."""
        remaining = set(range(len(points)))
        clusters = []
        while remaining:
            seed = remaining.pop()
            group = [seed]
            queue = [seed]
            while queue:
                current = queue.pop()
                near = [i for i in remaining
                        if np.hypot(*(points[i] - points[current])) <= link_distance]
                for i in near:
                    remaining.discard(i)
                    group.append(i)
                    queue.append(i)
            clusters.append(points[group])
        return clusters


def main() -> int:
    rclpy.init()
    node = CommandableSetProbe()
    deadline = node.get_clock().now().nanoseconds + int(20e9)
    while rclpy.ok() and not node.ready():
        rclpy.spin_once(node, timeout_sec=0.2)
        if node.get_clock().now().nanoseconds > deadline:
            print("terrain_map / state_estimation 을 20초 안에 못 받았습니다.")
            node.destroy_node(); rclpy.shutdown()
            return 1
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.2)
    print(node.run())
    node.destroy_node(); rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
