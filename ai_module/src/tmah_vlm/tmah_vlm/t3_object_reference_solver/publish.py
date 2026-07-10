#!/usr/bin/env python3
"""
t3 Object reference solver — 선택된 물체를 RViz marker / waypoint로 발행하는 부분.

t3_object_reference.py의 process가 publish_object_result(node, ctx) 하나만 부르고,
그 안에서 CUBE marker + wireframe marker + waypoint 발행을 순서대로 처리한다.
"""

from geometry_msgs.msg import Point, Pose2D
from visualization_msgs.msg import Marker

from tmah_vlm import config
from tmah_vlm.geometry.bbox_wireframe import wireframe_edge_points


# ========================================
# Publish (entry)
# ========================================

def publish_object_result(node, ctx):
    # 선택 물체의 3D marker(CUBE + wireframe)와 접근 waypoint를 모두 발행한다.
    publish_object_marker(node, ctx)
    publish_waypoint(node, ctx)


# ========================================
# Marker
# ========================================

def publish_object_marker(node, ctx):
    """
    선택된 물체를 RViz에 표시한다.

    /selected_object_marker는 챌린지 쪽 visualizationTools/RViz가 이미
    Marker(단수) 타입으로 구독 중인 고정 규격이라(dummy_vlm과 동일) 여기서
    타입을 바꾸면 안 된다. 그래서 CUBE는 원래 토픽 그대로 두고, wireframe
    테두리는 우리 전용 디버그 토픽(TOPIC_MARKER_WIREFRAME)으로 따로 뺐다.
    """
    stamp = node.get_clock().now().to_msg()

    # geometry/bbox_estimator.py가 크기를 추정 못했으면(ray fallback 등) result["point"]를
    # 중심으로, config.BBOX3D_DEFAULT_SIZE_M 고정 크기로 대체한다.
    bbox_center = ctx.result.get("bbox_center") or ctx.result["point"]
    bbox_size = ctx.result.get("bbox_size") or (config.BBOX3D_DEFAULT_SIZE_M,) * 3

    node.marker_pub.publish(make_cube_marker(stamp, bbox_center, bbox_size))
    node.wireframe_marker_pub.publish(make_wireframe_marker(stamp, bbox_center, bbox_size))


def make_cube_marker(stamp, center, size):
    # 실측 크기(center/size)를 반영한 초록 반투명 CUBE marker를 만든다.
    marker = Marker()
    marker.header.frame_id = config.FRAME_MAP
    marker.header.stamp = stamp
    marker.ns = "selected_object"
    marker.id = 0
    marker.type = Marker.CUBE
    marker.action = Marker.ADD

    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (
        float(v) for v in center
    )
    marker.pose.orientation.w = 1.0
    marker.scale.x, marker.scale.y, marker.scale.z = (float(v) for v in size)

    marker.color.a = 0.7
    marker.color.r = 0.1
    marker.color.g = 0.9
    marker.color.b = 0.2

    return marker


def make_wireframe_marker(stamp, center, size):
    # center/size로부터 12개 모서리(LINE_LIST) 흰색 wireframe marker를 만든다.
    marker = Marker()
    marker.header.frame_id = config.FRAME_MAP
    marker.header.stamp = stamp
    marker.ns = "selected_object"
    marker.id = 1
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD

    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.02  # LINE_LIST에서 scale.x = 선 굵기(m)

    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 1.0

    marker.points = [
        Point(x=float(x), y=float(y), z=float(z))
        for x, y, z in wireframe_edge_points(center, size)
    ]

    return marker


# ========================================
# Waypoint
# ========================================

def publish_waypoint(node, ctx):
    # 접근 waypoint를 Pose2D로 발행하고 구독자 수까지 로그로 남긴다.
    msg = Pose2D()
    msg.x = float(ctx.waypoint["x"])
    msg.y = float(ctx.waypoint["y"])
    msg.theta = float(ctx.waypoint["heading"])
    node.waypoint_pub.publish(msg)
    node.get_logger().info(
        f"[ObjectRef] waypoint published: topic={config.TOPIC_WAYPOINT}, "
        f"x={msg.x:.2f}, y={msg.y:.2f}, theta={msg.theta:.2f}, "
        f"subscribers={node.waypoint_pub.get_subscription_count()}"
    )
