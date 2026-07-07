#!/usr/bin/env python3
"""
TMAH VLM Node — CMU VLN Challenge 2026

지휘자(orchestrator) 역할만:
  - ROS 구독/발행 세팅
  - 모델(GroundingDINO + Qwen) 백그라운드 로드
  - 질문 오면 타입 분류 -> 해당 handler 에 위임

실제 로직은 handlers/ 안에 기능별로 분리:
  handlers/object_reference.py  (검출->선택->3D->발행)
  handlers/numerical.py
  handlers/instruction.py
"""

import math
import threading

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Int32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker

from tmah_vlm import config
from tmah_vlm.handlers import object_reference, numerical, instruction


class TmahVLM(Node):
    def __init__(self):
        super().__init__("tmah_vlm")
        self.cfg = config

        # 상태
        self.robot = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        self.latest_image = None
        self.latest_scan = None
        self.image_count = 0
        self.scan_count = 0
        self.busy = False

        # 모델 (백그라운드 로드)
        self.detector = None
        self.selector = None
        self.get_logger().info("Loading models in background...")
        threading.Thread(target=self._load_models, daemon=True).start()

        # 구독
        self.create_subscription(String, config.TOPIC_QUESTION, self.question_cb, 5)
        self.create_subscription(Odometry, config.TOPIC_STATE, self.pose_cb, 5)
        self.create_subscription(Image, config.TOPIC_IMAGE, self.image_cb, 5)
        self.create_subscription(PointCloud2, config.TOPIC_SCAN, self.scan_cb, 5)

        # 발행
        self.waypoint_pub = self.create_publisher(Pose2D, config.TOPIC_WAYPOINT, 5)
        self.marker_pub = self.create_publisher(Marker, config.TOPIC_MARKER, 5)
        self.numerical_pub = self.create_publisher(Int32, config.TOPIC_NUMERICAL, 5)

        self.create_timer(3.0, self.heartbeat)
        self.get_logger().info("TMAH VLM node started. Awaiting question...")

    def _load_models(self):
        try:
            from tmah_vlm.perception.detector import GroundingDINODetector
            self.detector = GroundingDINODetector(
                box_threshold=config.BOX_THRESHOLD,
                text_threshold=config.TEXT_THRESHOLD)
            self.get_logger().info("GroundingDINO loaded.")
        except Exception as e:
            self.get_logger().error(f"detector load fail: {e}")
        try:
            from tmah_vlm.reasoning.selector import QwenSelector
            self.selector = QwenSelector()
            self.get_logger().info("Qwen selector loaded.")
        except Exception as e:
            self.get_logger().error(f"selector load fail: {e}")
        self.get_logger().info("Model loading finished.")

    # ---- 콜백 ----
    def pose_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot["x"] = p.x
        self.robot["y"] = p.y
        self.robot["z"] = p.z
        # yaw
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot["yaw"] = math.atan2(siny, cosy)

    def image_cb(self, msg):
        self.latest_image = msg
        self.image_count += 1

    def scan_cb(self, msg):
        self.latest_scan = msg
        self.scan_count += 1

    def get_robot_pose(self):
        return dict(self.robot)

    def heartbeat(self):
        d = "ok" if self.detector is not None else "loading"
        s = "ok" if self.selector is not None else "loading"
        self.get_logger().info(
            f"[health] img={self.image_count} scan={self.scan_count} "
            f"pose=({self.robot['x']:.2f},{self.robot['y']:.2f},"
            f"yaw={self.robot['yaw']:.2f}) det={d} sel={s}")

    def question_cb(self, msg):
        if self.busy:
            self.get_logger().warn("Busy, ignoring.")
            return
        question = msg.data.strip()
        self.get_logger().info(f"Received question: {question}")
        self.busy = True
        try:
            self.dispatch(question)
        except Exception as e:
            self.get_logger().error(f"handler error: {e}")
        finally:
            self.busy = False
            self.get_logger().info("Awaiting question...")

    def dispatch(self, question):
        q = question.lower()
        if q.startswith("find"):
            object_reference.handle(self, question)
        elif q.startswith("how many") or q.startswith("count"):
            numerical.handle(self, question)
        else:
            instruction.handle(self, question)


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
