"""Publish target or exploration goals as geometry_msgs/Pose2D."""

from __future__ import annotations

import math

from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker, MarkerArray

from sysnav import config

_MISSION3_GOAL_NAMESPACE = "mission3_goals"
_MISSION3_GOAL_COLOR = (1.0, 0.55, 0.0, 0.9)  # 주황 - object/scene graph marker와 구분됨


class GoalPublisher:
    def __init__(self, node) -> None:
        self._node = node
        self.publisher = node.create_publisher(Pose2D, config.TOPIC_WAYPOINT, 10)
        self.goal_marker_pub = node.create_publisher(
            MarkerArray, config.TOPIC_MISSION3_GOAL_MARKERS, 10
        )
        self._goal_markers: list[Marker] = []

    def publish(self, x: float, y: float, theta: float) -> Pose2D:
        message = Pose2D()
        message.x, message.y, message.theta = float(x), float(y), float(theta)
        self.publisher.publish(message)
        return message

    def reset_step_markers(self) -> None:
        """새 mission3 task 시작 시 이전 task의 goal1/2/3 마커를 지운다."""
        self._goal_markers = []
        clear = Marker()
        clear.header.frame_id = config.OBJECT_MARKER_FRAME_ID
        clear.header.stamp = self._node.get_clock().now().to_msg()
        clear.ns = _MISSION3_GOAL_NAMESPACE
        clear.action = Marker.DELETEALL
        self.goal_marker_pub.publish(MarkerArray(markers=[clear]))

    def add_step_goal_marker(self, step_index: int, x: float, y: float, label: str) -> None:
        """mission3 step(0-based index)이 실제로 향하는 최종 goal 좌표를 goal{N} 라벨로
        추가한다 - "success는 떴는데 실제로는 물체 앞까지 안 갔다"를 RViz에서 눈으로 바로
        비교하기 위함(로봇 실제 이동 경로 vs 여기 찍힌 goal 위치). 이전 step의 마커는
        지우지 않고 계속 쌓아서 한 task의 goal1/2/3을 한눈에 같이 볼 수 있게 한다."""
        stamp = self._node.get_clock().now().to_msg()

        sphere = Marker()
        sphere.header.frame_id = config.OBJECT_MARKER_FRAME_ID
        sphere.header.stamp = stamp
        sphere.ns = _MISSION3_GOAL_NAMESPACE
        sphere.id = step_index * 2
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(x)
        sphere.pose.position.y = float(y)
        sphere.pose.position.z = 0.2
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.25
        sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = _MISSION3_GOAL_COLOR

        text = Marker()
        text.header.frame_id = config.OBJECT_MARKER_FRAME_ID
        text.header.stamp = stamp
        text.ns = _MISSION3_GOAL_NAMESPACE
        text.id = step_index * 2 + 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(x)
        text.pose.position.y = float(y)
        text.pose.position.z = 0.55
        text.pose.orientation.w = 1.0
        text.scale.z = 0.25
        text.color.r, text.color.g, text.color.b, text.color.a = (1.0, 1.0, 1.0, 1.0)
        text.text = label

        self._goal_markers = [
            marker for marker in self._goal_markers
            if marker.id not in (sphere.id, text.id)
        ]
        self._goal_markers.extend([sphere, text])
        self.goal_marker_pub.publish(MarkerArray(markers=list(self._goal_markers)))

    @staticmethod
    def object_approach_pose(robot_pose: dict, object_position, standoff: float = config.TARGET_STANDOFF_DISTANCE_M) -> tuple[float, float, float]:
        ox, oy = float(object_position[0]), float(object_position[1])
        rx, ry = float(robot_pose["x"]), float(robot_pose["y"])
        dx, dy = ox - rx, oy - ry
        distance = math.hypot(dx, dy)
        if distance <= 1e-6:
            return rx, ry, float(robot_pose["yaw"])
        usable = min(standoff, max(0.0, distance - 0.15))
        gx, gy = ox - usable * dx / distance, oy - usable * dy / distance
        return gx, gy, math.atan2(oy - gy, ox - gx)
