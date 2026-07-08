#!/usr/bin/env python3
"""
TMAH VLM Node

이 파일은 C++ main node처럼 전체 구조가 보이도록 정리한 파일이다.

구조:
  1. Import
  2. Node class
     - initialize_state()
     - initialize_modules()
     - initialize_subscribers()
     - initialize_publishers()
     - initialize_timers()
  3. Callback
     - callback에서는 최신 센서값/질문만 저장한다.
     - 무거운 VLM 추론은 callback 안에서 바로 돌리지 않는다.
  4. Main control loop
     - pending question이 있으면 pipeline으로 넘긴다.
  5. Helper
  6. main()

현재 pipeline:
  challenge_question
    -> main_control_loop()
    -> dispatch_question()
    -> handlers/object_reference.py
       -> perception / reasoning / grounding / tf 기능을 단계적으로 사용
"""

import math
import threading
import time
from collections import deque

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Int32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker

from tmah_vlm import config
from tmah_vlm.handlers import object_reference, numerical, instruction
from tmah_vlm.tf.coordinate_transform import CoordinateTransformer


# ========================================
# Node
# ========================================

class TmahVLM(Node):
    def __init__(self):
        super().__init__("tmah_vlm")

        self.cfg = config

        self.initialize_state()
        self.initialize_modules()
        self.initialize_subscribers()
        self.initialize_publishers()
        self.initialize_timers()

        self.get_logger().info("TMAH VLM node started")
        self.get_logger().info("Waiting for /challenge_question ...")

    # ========================================
    # Initialize
    # ========================================

    def initialize_state(self):
        """노드가 계속 들고 있어야 하는 최신 상태값."""
        self.robot = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "yaw": 0.0,
        }

        self.latest_image = None
        self.latest_scan = None
        self.scan_buffer = deque(maxlen=config.SYNC_SCAN_BUFFER_SIZE)
        self.last_sync_dt = None

        self.image_count = 0
        self.scan_count = 0

        self.pending_question = None
        self.busy = False
        self.state_lock = threading.Lock()
        self.last_wait_log_time = 0.0

    def initialize_modules(self):
        """TF 변환기와 VLM/Detector 모델을 준비한다."""
        self.transformer = CoordinateTransformer(self)

        self.detector = None
        self.selector = None

        self.get_logger().info("Loading models in background...")
        model_thread = threading.Thread(target=self.load_models, daemon=True)
        model_thread.start()

    def initialize_subscribers(self):
        """ROS subscriber 목록."""
        self.question_sub = self.create_subscription(
            String,
            config.TOPIC_QUESTION,
            self.question_callback,
            5,
        )
        self.pose_sub = self.create_subscription(
            Odometry,
            config.TOPIC_STATE,
            self.pose_callback,
            5,
        )
        self.image_sub = self.create_subscription(
            Image,
            config.TOPIC_IMAGE,
            self.image_callback,
            5,
        )
        self.scan_sub = self.create_subscription(
            PointCloud2,
            config.TOPIC_SCAN,
            self.scan_callback,
            5,
        )

    def initialize_publishers(self):
        """ROS publisher 목록."""
        self.waypoint_pub = self.create_publisher(
            Pose2D,
            config.TOPIC_WAYPOINT,
            5,
        )
        self.marker_pub = self.create_publisher(
            Marker,
            config.TOPIC_MARKER,
            5,
        )
        self.numerical_pub = self.create_publisher(
            Int32,
            config.TOPIC_NUMERICAL,
            5,
        )

    def initialize_timers(self):
        """주기적으로 돌아가는 loop."""
        self.main_timer = self.create_timer(0.2, self.main_control_loop)
        self.health_timer = self.create_timer(3.0, self.heartbeat)

    # ========================================
    # Callback
    # ========================================

    def question_callback(self, msg):
        """
        질문 callback.

        여기서는 질문만 저장한다.
        실제 VLM pipeline은 main_control_loop에서 처리한다.
        """
        question = msg.data.strip()
        if question == "":
            return

        with self.state_lock:
            self.pending_question = question

        self.get_logger().info(f"[Question] queued: {question}")

    def pose_callback(self, msg):
        """로봇 위치/자세 최신값 저장."""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        self.robot["x"] = position.x
        self.robot["y"] = position.y
        self.robot["z"] = position.z
        self.robot["yaw"] = quaternion_to_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )

    def image_callback(self, msg):
        """카메라 이미지 최신값 저장."""
        self.latest_image = msg
        self.image_count += 1

    def scan_callback(self, msg):
        """PointCloud2 최신값 저장.

        image와 scan은 서로 다른 callback으로 들어오기 때문에,
        단순 latest_scan만 쓰면 이미지 시각과 다른 point cloud가 섞일 수 있다.
        따라서 최근 scan들을 buffer에 저장해두고, 질문 처리 시 image stamp와
        가장 가까운 scan을 선택한다.
        """
        self.latest_scan = msg
        self.scan_buffer.append(msg)
        self.scan_count += 1

    # ========================================
    # Main control loop
    # ========================================

    def main_control_loop(self):
        """
        C++ 코드의 mainControlLoop 역할.

        센서 callback은 계속 최신값을 갱신하고,
        여기서 pending question이 있을 때만 pipeline을 한 번 실행한다.
        """
        if self.busy:
            return

        question = self.peek_pending_question()
        if question is None:
            return

        if not self.ready_to_process(question):
            self.print_waiting_reason(question)
            return

        self.busy = True

        try:
            self.get_logger().info("========================================")
            self.get_logger().info(f"[Pipeline] start: {question}")
            self.dispatch_question(question)
            self.get_logger().info("[Pipeline] finished")

            with self.state_lock:
                if self.pending_question == question:
                    self.pending_question = None

        except Exception as error:
            self.get_logger().error(f"[Pipeline] failed: {error}")

            with self.state_lock:
                if self.pending_question == question:
                    self.pending_question = None

        finally:
            self.busy = False
            self.get_logger().info("Waiting for /challenge_question ...")

    def dispatch_question(self, question):
        """질문 종류에 따라 handler를 선택한다."""
        lower_question = question.lower()

        if lower_question.startswith("find"):
            object_reference.handle(self, question)
        elif lower_question.startswith("how many") or lower_question.startswith("count"):
            numerical.handle(self, question)
        else:
            instruction.handle(self, question)

    # ========================================
    # Helper
    # ========================================

    def load_models(self):
        """GroundingDINO와 Qwen selector를 로드한다."""
        try:
            from tmah_vlm.perception.detector import GroundingDINODetector
            self.detector = GroundingDINODetector(
                box_threshold=config.BOX_THRESHOLD,
                text_threshold=config.TEXT_THRESHOLD,
            )
            self.get_logger().info("GroundingDINO loaded")
        except Exception as error:
            self.get_logger().error(f"GroundingDINO load failed: {error}")

        try:
            from tmah_vlm.reasoning.selector import QwenSelector
            self.selector = QwenSelector()
            self.get_logger().info("Qwen selector loaded")
        except Exception as error:
            self.get_logger().error(f"Qwen selector load failed: {error}")

        self.get_logger().info("Model loading finished")

    def peek_pending_question(self):
        """아직 처리하지 않은 질문을 확인한다."""
        with self.state_lock:
            return self.pending_question

    def ready_to_process(self, question):
        """현재 질문을 처리할 준비가 됐는지 확인한다."""
        lower_question = question.lower()

        if self.latest_image is None:
            return False

        if lower_question.startswith("find") and self.detector is None:
            return False

        # selector는 없으면 0번 후보를 선택하는 fallback이 있으므로 필수는 아니다.
        return True

    def print_waiting_reason(self, question):
        """준비가 덜 됐을 때 너무 자주 출력하지 않도록 3초에 한 번만 로그."""
        now = time.time()
        if now - self.last_wait_log_time < 3.0:
            return
        self.last_wait_log_time = now

        reasons = []
        lower_question = question.lower()

        if self.latest_image is None:
            reasons.append("camera image")
        if lower_question.startswith("find") and self.detector is None:
            reasons.append("GroundingDINO")

        reason_text = ", ".join(reasons)
        self.get_logger().warn(f"[Pipeline] waiting for: {reason_text}")

    def get_robot_pose(self):
        """handler에서 현재 로봇 pose를 읽기 위한 함수."""
        return dict(self.robot)

    def get_synced_scan_for_latest_image(self):
        """latest_image의 header stamp와 가장 가까운 PointCloud2를 반환한다.

        반환:
          scan_msg, dt_sec
          dt_sec = abs(image_time - scan_time)
        """
        if self.latest_image is None:
            return self.latest_scan, None

        if len(self.scan_buffer) == 0:
            return self.latest_scan, None

        image_time = stamp_to_sec(self.latest_image.header.stamp)

        # stamp가 0이면 simulator가 header stamp를 안 넣는 경우다.
        # 이 경우에는 기존처럼 최신 scan을 사용한다.
        if image_time <= 0.0:
            return self.latest_scan, None

        best_scan = None
        best_dt = None

        for scan in list(self.scan_buffer):
            scan_time = stamp_to_sec(scan.header.stamp)
            if scan_time <= 0.0:
                continue

            dt = abs(image_time - scan_time)
            if best_dt is None or dt < best_dt:
                best_scan = scan
                best_dt = dt

        if best_scan is None:
            return self.latest_scan, None

        self.last_sync_dt = best_dt

        if best_dt > config.SYNC_WARN_TIME_DIFF_SEC:
            self.get_logger().warn(
                f"[Sync] image-scan dt is large: {best_dt:.3f}s "
                f"(recommended < {config.SYNC_WARN_TIME_DIFF_SEC:.3f}s)"
            )
        else:
            self.get_logger().info(f"[Sync] image-scan dt={best_dt:.3f}s")

        return best_scan, best_dt

    def heartbeat(self):
        """현재 노드 상태 확인용 로그."""
        detector_state = "ok" if self.detector is not None else "loading"
        selector_state = "ok" if self.selector is not None else "loading"

        self.get_logger().info(
            f"[Health] img={self.image_count}, scan={self.scan_count}, "
            f"pose=({self.robot['x']:.2f}, {self.robot['y']:.2f}, "
            f"yaw={self.robot['yaw']:.2f}), "
            f"sync_dt={self.last_sync_dt if self.last_sync_dt is not None else -1:.3f}, "
            f"detector={detector_state}, selector={selector_state}"
        )


# ========================================
# Utility
# ========================================

def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quaternion_to_yaw(qx, qy, qz, qw):
    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(sin_yaw, cos_yaw)


# ========================================
# main
# ========================================

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
