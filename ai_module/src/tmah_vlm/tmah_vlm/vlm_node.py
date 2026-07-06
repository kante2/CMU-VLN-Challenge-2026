#!/usr/bin/env python3
"""
TMAH VLM Node — CMU VLN Challenge 2026  (Phase 1a)
==================================================

Phase 1a: object_reference 질문에 대해
  현재 카메라 뷰 -> GroundingDINO 2D 검출 -> 박스 그려 파일 저장 + 로그
까지 수행합니다. (3D 좌표화/waypoint 는 Phase 1b 에서)

numerical / instruction 은 아직 stub 유지.
"""

import threading

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Int32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker

# perception 모듈
from tmah_vlm.perception.image_utils import ros_image_to_pil
from tmah_vlm.perception.query_parser import extract_target
from tmah_vlm.perception.visualize import save_detection_image


class TmahVLM(Node):
    def __init__(self):
        super().__init__("tmah_vlm")

        # --- 내부 상태 ---
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.latest_image = None
        self.latest_scan = None
        self.image_count = 0
        self.scan_count = 0
        self.busy = False

        # --- 검출기: 무거우니 백그라운드에서 로드 ---
        self.detector = None
        self.get_logger().info("Loading GroundingDINO in background...")
        threading.Thread(target=self._load_detector, daemon=True).start()

        # --- 구독 ---
        self.create_subscription(String, "/challenge_question",
                                 self.question_cb, 5)
        self.create_subscription(Odometry, "/state_estimation",
                                 self.pose_cb, 5)
        self.create_subscription(Image, "/camera/image",
                                 self.image_cb, 5)
        self.create_subscription(PointCloud2, "/registered_scan",
                                 self.scan_cb, 5)

        # --- 발행 ---
        self.waypoint_pub = self.create_publisher(
            Pose2D, "/way_point_with_heading", 5)
        self.marker_pub = self.create_publisher(
            Marker, "/selected_object_marker", 5)
        self.numerical_pub = self.create_publisher(
            Int32, "/numerical_response", 5)

        self.create_timer(3.0, self.heartbeat)
        self.get_logger().info("TMAH VLM node started. Awaiting question...")

    def _load_detector(self):
        try:
            from tmah_vlm.perception.detector import GroundingDINODetector
            self.detector = GroundingDINODetector()
            self.get_logger().info("GroundingDINO loaded. Ready to detect.")
        except Exception as e:
            self.get_logger().error(f"Failed to load detector: {e}")

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
        ready = "ready" if self.detector is not None else "loading"
        self.get_logger().info(
            f"[health] images={self.image_count} scans={self.scan_count} "
            f"pose=({self.vehicle_x:.2f}, {self.vehicle_y:.2f}) detector={ready}")

    def question_cb(self, msg: String):
        if self.busy:
            self.get_logger().warn("Busy, ignoring question.")
            return
        question = msg.data.strip()
        self.get_logger().info(f"Received question: {question}")
        self.busy = True
        try:
            self.dispatch(question)
        except Exception as e:
            self.get_logger().error(f"Error handling question: {e}")
        finally:
            self.busy = False
            self.get_logger().info("Awaiting question...")

    # ================= 디스패처 =================
    def dispatch(self, question: str):
        q = question.lower()
        if q.startswith("find"):
            self.handle_object_reference(question)
        elif q.startswith("how many") or q.startswith("count"):
            self.handle_numerical(question)
        else:
            self.handle_instruction(question)

    # ============ object_reference: Phase 1a (검출 + 시각화) ============
    def handle_object_reference(self, question: str):
        if self.detector is None:
            self.get_logger().warn("Detector still loading, skipping.")
            return
        if self.latest_image is None:
            self.get_logger().warn("No camera image yet, skipping.")
            return

        # 1) 현재 카메라 뷰
        pil = ros_image_to_pil(self.latest_image)

        # 2) 질문에서 대상 명사 추출
        target = extract_target(question)
        self.get_logger().info(f"[object_reference] target prompt = '{target}'")

        # 3) GroundingDINO 검출
        dets = self.detector.detect(pil, target)

        # 4) 로그로 요약
        self.get_logger().info(f"  detected {len(dets)} object(s):")
        for i, d in enumerate(dets):
            self.get_logger().info(
                f"    #{i+1} {d.label} score={d.score:.2f} "
                f"box=({d.box[0]:.0f},{d.box[1]:.0f},{d.box[2]:.0f},{d.box[3]:.0f}) "
                f"center=({d.cx:.0f},{d.cy:.0f})")

        # 5) 박스 그린 이미지 저장
        try:
            path = save_detection_image(pil, dets, target)
            self.get_logger().info(f"  saved visualization -> {path}")
        except Exception as e:
            self.get_logger().error(f"  failed to save visualization: {e}")

        # 6) (Phase 1b 예정) 3D 좌표화 + marker/waypoint 발행
        #    지금은 2D 검출 확인만. 아직 발행 안 함.

    # ============ 아직 stub ============
    def handle_numerical(self, question: str):
        self.get_logger().info("[numerical] (stub) publishing 1")
        self.publish_numerical(1)

    def handle_instruction(self, question: str):
        self.get_logger().info("[instruction] (stub) single waypoint")
        self.publish_waypoint(self.vehicle_x + 1.0, self.vehicle_y, 0.0)

    # ================= 발행 헬퍼 =================
    def publish_waypoint(self, x, y, heading=0.0):
        m = Pose2D()
        m.x, m.y, m.theta = float(x), float(y), float(heading)
        self.waypoint_pub.publish(m)

    def publish_numerical(self, value):
        m = Int32()
        m.data = int(value)
        self.numerical_pub.publish(m)


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
