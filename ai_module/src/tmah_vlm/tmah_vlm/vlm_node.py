#!/usr/bin/env python3
"""
TMAH VLM Node — CMU VLN Challenge 2026
======================================

이 노드는 dummy_vlm 을 대체하는 뼈대(skeleton)입니다.
지금은 '하드코딩 답 + 센서 구독 로그' 상태이고,
파이프라인이 도는 것을 확인한 뒤 각 함수 안쪽 로직을
실제 인식/추론(VLM, GroundingDINO, API 등)으로 채우면 됩니다.

--- 인터페이스 계약 (base 시스템과의 약속) ---
[구독 / Inputs]
  /challenge_question   std_msgs/String        질문 텍스트
  /camera/image         sensor_msgs/Image      360 RGB (test-time 허용)
  /registered_scan      sensor_msgs/PointCloud2 라이다 (map 좌표, test-time 허용)
  /state_estimation     nav_msgs/Odometry      로봇 pose
  # (개발용) /camera/semantic_image 도 있지만 test-time엔 안 나옴 → 의존 금지

[발행 / Outputs]
  /way_point_with_heading   geometry_msgs/Pose2D   이동 목표 (instruction/object)
  /selected_object_marker   visualization_msgs/Marker  선택 객체 박스 (object)
  /numerical_response       std_msgs/Int32         개수 답 (numerical)
"""

import math

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Int32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker


class TmahVLM(Node):
    def __init__(self):
        super().__init__("tmah_vlm")

        # --- 내부 상태 ---
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.latest_image = None          # 가장 최근 카메라 프레임 저장
        self.latest_scan = None           # 가장 최근 포인트클라우드 저장
        self.image_count = 0              # 센서 수신 확인용 카운터
        self.scan_count = 0
        self.busy = False                 # 질문 처리 중 중복 방지

        # --- 구독 (Inputs) ---
        self.create_subscription(String, "/challenge_question",
                                 self.question_cb, 5)
        self.create_subscription(Odometry, "/state_estimation",
                                 self.pose_cb, 5)
        self.create_subscription(Image, "/camera/image",
                                 self.image_cb, 5)
        self.create_subscription(PointCloud2, "/registered_scan",
                                 self.scan_cb, 5)

        # --- 발행 (Outputs) ---
        self.waypoint_pub = self.create_publisher(
            Pose2D, "/way_point_with_heading", 5)
        self.marker_pub = self.create_publisher(
            Marker, "/selected_object_marker", 5)
        self.numerical_pub = self.create_publisher(
            Int32, "/numerical_response", 5)

        # 센서가 실제로 들어오는지 3초마다 로그 (파이프라인 헬스체크)
        self.create_timer(3.0, self.heartbeat)

        self.get_logger().info("TMAH VLM node started. Awaiting question...")

    # ================= 콜백 =================
    def pose_cb(self, msg: Odometry):
        self.vehicle_x = msg.pose.pose.position.x
        self.vehicle_y = msg.pose.pose.position.y

    def image_cb(self, msg: Image):
        self.latest_image = msg
        self.image_count += 1

    def scan_cb(self, msg: PointCloud2):
        self.latest_scan = msg
        self.scan_count += 1

    def heartbeat(self):
        self.get_logger().info(
            f"[health] images={self.image_count} scans={self.scan_count} "
            f"pose=({self.vehicle_x:.2f}, {self.vehicle_y:.2f})")

    def question_cb(self, msg: String):
        if self.busy:
            self.get_logger().warn("Still processing previous question, ignoring.")
            return
        question = msg.data.strip()
        self.get_logger().info(f"Received question: {question}")
        self.busy = True
        try:
            self.dispatch(question)
        finally:
            self.busy = False
            self.get_logger().info("Awaiting question...")

    # ================= 디스패처 =================
    # dummy 의 분류 로직을 그대로 옮김: 첫 단어로 질문 타입 판별
    def dispatch(self, question: str):
        q = question.lower()
        if q.startswith("find"):
            self.handle_object_reference(question)
        elif q.startswith("how many") or q.startswith("count"):
            self.handle_numerical(question)
        else:
            self.handle_instruction(question)

    # ============ 질문 타입별 처리 (여기를 채워나감) ============
    def handle_object_reference(self, question: str):
        """
        TODO: 실제 구현
          1) self.latest_image 로 후보 객체 검출 (GroundingDINO / VLM / API)
          2) 공간관계 추론으로 정답 하나 선택
          3) 3D 좌표/크기 추정 (registered_scan 활용)
          4) publish_marker(...) + publish_waypoint(...)
        지금은 하드코딩 값으로 파이프라인만 검증.
        """
        self.get_logger().info("[object_reference] (stub) publishing dummy marker + waypoint")
        # object_list.txt 의 예시 sofa 값 재사용
        self.publish_marker(obj_id=0, label="sofa",
                            x=3.37, y=-2.09, z=0.50,
                            l=2.86, w=1.20, h=1.02, heading=0.0)
        self.publish_waypoint(3.37, -2.09, 0.0)

    def handle_numerical(self, question: str):
        """
        TODO: 실제 구현
          탐색하며 이미지 수집 → 대상 객체 검출/카운트 → 정수 publish
        """
        answer = 1  # stub
        self.get_logger().info(f"[numerical] (stub) publishing {answer}")
        self.publish_numerical(answer)

    def handle_instruction(self, question: str):
        """
        TODO: 실제 구현
          질문을 landmark 시퀀스로 파싱 → 각 landmark 좌표화 →
          순차 waypoint (도달 확인하며 다음으로)
        지금은 현재 위치 살짝 앞으로 한 점만.
        """
        self.get_logger().info("[instruction] (stub) publishing single waypoint")
        self.publish_waypoint(self.vehicle_x + 1.0, self.vehicle_y, 0.0)

    # ================= 발행 헬퍼 =================
    def publish_waypoint(self, x: float, y: float, heading: float = 0.0):
        msg = Pose2D()
        msg.x = float(x)
        msg.y = float(y)
        msg.theta = float(heading)  # 올해는 heading 무시해도 됨
        self.waypoint_pub.publish(msg)
        self.get_logger().info(f"  -> waypoint ({x:.2f}, {y:.2f})")

    def publish_numerical(self, value: int):
        msg = Int32()
        msg.data = int(value)
        self.numerical_pub.publish(msg)

    def publish_marker(self, obj_id, label, x, y, z, l, w, h, heading):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = label
        m.id = int(obj_id)
        m.action = Marker.ADD
        m.type = Marker.CUBE
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        # heading(yaw) -> quaternion
        m.pose.orientation.z = math.sin(heading / 2.0)
        m.pose.orientation.w = math.cos(heading / 2.0)
        m.scale.x = float(l)
        m.scale.y = float(w)
        m.scale.z = float(h)
        m.color.a = 0.5
        m.color.r = 0.0
        m.color.g = 0.0
        m.color.b = 1.0
        self.marker_pub.publish(m)
        self.get_logger().info(f"  -> marker '{label}' at ({x:.2f}, {y:.2f})")


def main(args=None):
    rclpy.init(args=args)
    node = TmahVLM()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
