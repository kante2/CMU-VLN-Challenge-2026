#!/usr/bin/env python3
"""
nav / challenge 로 나가는 발행을 한곳에 모은 파일.

각 solver(t1/t2/t3)는 자기 결과를 여기 함수로 발행한다:
  - publish_waypoint(node, ctx)        : 접근/이동 목표 Pose2D  (t1, t3)
  - publish_object_result(node, ctx)   : 선택 물체 marker(CUBE+wireframe) + waypoint (t3)
  - publish_count(node, count)         : 개수 Int32            (t2)

marker/waypoint 토픽과 색/크기 상수는 config.py에 있다.
/selected_object_marker는 챌린지 visualizationTools/RViz가 Marker(단수) 타입으로
구독 중인 고정 규격이라 타입을 바꾸면 안 된다(→ wireframe은 별도 디버그 토픽으로 분리).

scene graph 전체 marker 발행은 추론 디버그 시각화라 여기 두지 않고
reasoning/graph/visualizer.py에 있다.
"""

from geometry_msgs.msg import Point, Pose2D
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker

from tmah_vlm import config
from tmah_vlm.sensor_process.bbox_wireframe import wireframe_edge_points


# ========================================
# Waypoint (t1, t3)
# ========================================

def publish_waypoint(node, ctx):
    # 계산된 접근/이동 waypoint를 Pose2D로 발행하고 구독자 수까지 로그로 남긴다.
    msg = Pose2D()
    msg.x = float(ctx.waypoint["x"])
    msg.y = float(ctx.waypoint["y"])
    msg.theta = float(ctx.waypoint["heading"])
    node.waypoint_pub.publish(msg)
    node.get_logger().info(
        f"[Nav] waypoint published: topic={config.TOPIC_WAYPOINT}, "
        f"x={msg.x:.2f}, y={msg.y:.2f}, theta={msg.theta:.2f}, "
        f"subscribers={node.waypoint_pub.get_subscription_count()}"
    )


# ========================================
# Object result: marker + waypoint (t3)
# ========================================

def publish_object_result(node, ctx):
    # 선택 물체의 3D marker(CUBE + wireframe)와 접근 waypoint를 모두 발행한다.
    publish_object_marker(node, ctx)
    publish_waypoint(node, ctx)


def publish_object_marker(node, ctx):
    """
    선택된 물체를 RViz에 표시한다.

    /selected_object_marker는 챌린지 쪽 visualizationTools/RViz가 이미
    Marker(단수) 타입으로 구독 중인 고정 규격이라(dummy_vlm과 동일) 여기서
    타입을 바꾸면 안 된다. 그래서 CUBE는 원래 토픽 그대로 두고, wireframe
    테두리는 우리 전용 디버그 토픽(TOPIC_MARKER_WIREFRAME)으로 따로 뺐다.
    """
    stamp = node.get_clock().now().to_msg()

    # sensor_process/bbox_estimator.py가 크기를 추정 못했으면(ray fallback 등) result["point"]를
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

    r, g, b, a = config.MARKER_CUBE_RGBA
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = r, g, b, a

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
    marker.scale.x = config.MARKER_WIREFRAME_LINE_WIDTH_M  # LINE_LIST에서 scale.x = 선 굵기(m)

    r, g, b, a = config.MARKER_WIREFRAME_RGBA
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = r, g, b, a

    marker.points = [
        Point(x=float(x), y=float(y), z=float(z))
        for x, y, z in wireframe_edge_points(center, size)
    ]

    return marker


# ========================================
# Count (t2)
# ========================================

def publish_count(node, count):
    # 개수를 Int32로 발행한다.
    msg = Int32()
    msg.data = int(count)
    node.numerical_pub.publish(msg)
