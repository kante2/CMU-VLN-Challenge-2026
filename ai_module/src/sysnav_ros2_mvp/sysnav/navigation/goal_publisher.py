"""Publish target or exploration goals as geometry_msgs/Pose2D."""

from __future__ import annotations

import math

from geometry_msgs.msg import Pose2D

from sysnav import config


class GoalPublisher:
    def __init__(self, node) -> None:
        self.publisher = node.create_publisher(Pose2D, config.TOPIC_WAYPOINT, 10)

    def publish(self, x: float, y: float, theta: float) -> Pose2D:
        message = Pose2D()
        message.x, message.y, message.theta = float(x), float(y), float(theta)
        self.publisher.publish(message)
        return message

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
