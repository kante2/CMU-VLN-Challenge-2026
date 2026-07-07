#!/usr/bin/env python3
"""
TMAH VLM Node — CMU VLN Challenge 2026  (하이브리드: GroundingDINO + Qwen2.5-VL)

object_reference 흐름:
  질문 -> query_parser(명사) -> GroundingDINO(후보 검출)
       -> Qwen selector(후보 중 정답 선택) -> 시각화 저장
  (3D 좌표화 + marker/waypoint 발행은 Phase 1b 예정)
"""

import threading

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Int32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker

from tmah_vlm.perception.image_utils import ros_image_to_pil
from tmah_vlm.perception.query_parser import extract_target
from tmah_vlm.perception.visualize import save_detection_image


class TmahVLM(Node):
    def __init__(self):
        super().__init__("tmah_vlm")

        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.latest_image = None
        self.latest_scan = None
        self.image_count = 0
        self.scan_count = 0
        self.busy = False

        # 모델(무거움)은 백그라운드 로드
        self.detector = None
        self.selector = None
        self.get_logger().info("Loading models in background (GroundingDINO + Qwen)...")
        threading.Thread(target=self._load_models, daemon=True).start()

        # 구독
        self.create_subscription(String, "/challenge_question", self.question_cb, 5)
        self.create_subscription(Odometry, "/state_estimation", self.pose_cb, 5)
        self.create_subscription(Image, "/camera/image", self.image_cb, 5)
        self.create_subscription(PointCloud2, "/registered_scan", self.scan_cb, 5)

        # 발행
        self.waypoint_pub = self.create_publisher(Pose2D, "/way_point_with_heading", 5)
        self.marker_pub = self.create_publisher(Marker, "/selected_object_marker", 5)
        self.numerical_pub = self.create_publisher(Int32, "/numerical_response", 5)

        self.create_timer(3.0, self.heartbeat)
        self.get_logger().info("TMAH VLM node started. Awaiting question...")

    def _load_models(self):
        try:
            from tmah_vlm.perception.detector import GroundingDINODetector
            self.detector = GroundingDINODetector()
            self.get_logger().info("GroundingDINO loaded.")
        except Exception as e:
            self.get_logger().error(f"Failed to load detector: {e}")
        try:
            from tmah_vlm.reasoning.selector import QwenSelector
            self.selector = QwenSelector()
            self.get_logger().info("Qwen2.5-VL selector loaded.")
        except Exception as e:
            self.get_logger().error(f"Failed to load selector: {e}")
        self.get_logger().info("Model loading finished.")

    # 콜백
    def pose_cb(self, msg):
        self.vehicle_x = msg.pose.pose.position.x
        self.vehicle_y = msg.pose.pose.position.y

    def image_cb(self, msg):
        self.latest_image = msg
        self.image_count += 1

    def scan_cb(self, msg):
        self.latest_scan = msg
        self.scan_count += 1

    def heartbeat(self):
        d = "ok" if self.detector is not None else "loading"
        s = "ok" if self.selector is not None else "loading"
        self.get_logger().info(
            f"[health] images={self.image_count} scans={self.scan_count} "
            f"pose=({self.vehicle_x:.2f},{self.vehicle_y:.2f}) "
            f"detector={d} selector={s}")

    def question_cb(self, msg):
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

    def dispatch(self, question):
        q = question.lower()
        if q.startswith("find"):
            self.handle_object_reference(question)
        elif q.startswith("how many") or q.startswith("count"):
            self.handle_numerical(question)
        else:
            self.handle_instruction(question)

    # object_reference: 하이브리드
    def handle_object_reference(self, question):
        if self.detector is None:
            self.get_logger().warn("Detector still loading, skipping.")
            return
        if self.latest_image is None:
            self.get_logger().warn("No camera image yet, skipping.")
            return

        pil = ros_image_to_pil(self.latest_image)

        # 1) 검출용 명사
        parsed = extract_target(question)
        obj = parsed["object"]
        self.get_logger().info(f"[object_reference] detect prompt = '{obj}'")

        # 2) GroundingDINO 후보 검출
        dets = self.detector.detect(pil, obj)
        self.get_logger().info(f"  GroundingDINO found {len(dets)} candidate(s)")
        for i, d in enumerate(dets):
            self.get_logger().info(
                f"    #{i} {d.label} score={d.score:.2f} center=({d.cx:.0f},{d.cy:.0f})")

        # 3) Qwen selector: 후보 중 정답 선택
        chosen_idx = -1
        if self.selector is not None and len(dets) > 0:
            try:
                chosen_idx = self.selector.choose(pil, dets, question)
                self.get_logger().info(f"  Qwen selected -> #{chosen_idx}")
            except Exception as e:
                self.get_logger().error(f"  selector error: {e}")
                chosen_idx = 0  # fallback: 최고 점수
        elif len(dets) > 0:
            chosen_idx = 0
            self.get_logger().info("  selector not ready -> fallback #0")

        # 4) 시각화 저장 (선택된 것 표시)
        try:
            path = save_detection_image(pil, dets, f"{obj}_sel{chosen_idx}")
            self.get_logger().info(f"  saved visualization -> {path}")
        except Exception as e:
            self.get_logger().error(f"  viz save failed: {e}")

        # 5) (Phase 1b) 선택된 박스 -> 3D 좌표화 -> marker/waypoint
        #    지금은 선택까지만.

    # stub
    def handle_numerical(self, question):
        self.get_logger().info("[numerical] (stub) publishing 1")
        m = Int32(); m.data = 1
        self.numerical_pub.publish(m)

    def handle_instruction(self, question):
        self.get_logger().info("[instruction] (stub) single waypoint")
        m = Pose2D(); m.x = self.vehicle_x + 1.0; m.y = self.vehicle_y; m.theta = 0.0
        self.waypoint_pub.publish(m)


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
