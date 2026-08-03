#!/usr/bin/env python3
"""Convert selected SORT3D objects into navigation-friendly waypoints."""

import math


def go_near(obj, robot_pose=None, standoff_m=0.8):
    """Return a Pose2D-like dict near an object center."""
    rx = 0.0 if robot_pose is None else float(robot_pose.get("x", 0.0))
    ry = 0.0 if robot_pose is None else float(robot_pose.get("y", 0.0))
    dx = obj.x - rx
    dy = obj.y - ry
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        heading = 0.0
        wx = obj.x
        wy = obj.y
    else:
        heading = math.atan2(dy, dx)
        wx = obj.x - math.cos(heading) * standoff_m
        wy = obj.y - math.sin(heading) * standoff_m
    return {"x": wx, "y": wy, "heading": heading, "target_id": obj.object_id}


def go_between(obj1, obj2, robot_pose=None):
    x = (obj1.x + obj2.x) * 0.5
    y = (obj1.y + obj2.y) * 0.5
    heading = math.atan2(obj2.y - obj1.y, obj2.x - obj1.x)
    return {"x": x, "y": y, "heading": heading, "target_ids": [obj1.object_id, obj2.object_id]}
