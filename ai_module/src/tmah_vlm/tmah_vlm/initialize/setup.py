#!/usr/bin/env python3
"""
TmahVLM 노드 초기화 — TmahVLM.__init__()이 순서대로 호출하는 initialize_* 함수 모음.

  initialize_state       -> 계속 들고 있어야 하는 최신 상태값 초기화
  initialize_modules      -> TF 변환기 준비 + GroundingDINO/Qwen 백그라운드 로딩(load_models)
  initialize_subscribers  -> ROS subscriber 등록 (콜백 로직은 callback/sensor_callbacks.py)
  initialize_publishers   -> ROS publisher 등록
  initialize_timers       -> 주기 실행 timer 등록

node를 인자로 받는 자유 함수라 handlers/*.py의 process(node, ...)와 같은 패턴이다.
"""

import threading
from collections import deque

from std_msgs.msg import String, Int32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker

from tmah_vlm import config
from tmah_vlm.tf.coordinate_transform import CoordinateTransformer
from tmah_vlm.callback.sensor_callbacks import (
    question_callback,
    pose_callback,
    image_callback,
    scan_callback,
)
from tmah_vlm.helper.node_helpers import heartbeat


def initialize_state(node):
    """노드가 계속 들고 있어야 하는 최신 상태값."""
    node.robot = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "yaw": 0.0,
    }

    node.latest_image = None
    node.latest_scan = None
    node.scan_buffer = deque(maxlen=config.SYNC_SCAN_BUFFER_SIZE)
    node.last_sync_dt = None

    node.image_count = 0
    node.scan_count = 0

    node.pending_question = None
    node.busy = False
    node.state_lock = threading.Lock()
    node.last_wait_log_time = 0.0


def initialize_modules(node):
    """TF 변환기와 VLM/Detector/Segmenter 모델을 준비한다."""
    node.transformer = CoordinateTransformer(node)

    node.detector = None
    node.selector = None
    node.segmenter = None

    node.get_logger().info("Loading models in background...")
    model_thread = threading.Thread(target=load_models, args=(node,), daemon=True)
    model_thread.start()


def load_models(node):
    """GroundingDINO, Qwen selector, SAM segmenter를 로드한다 (백그라운드 스레드)."""
    try:
        from tmah_vlm.perception.detector import GroundingDINODetector
        node.detector = GroundingDINODetector(
            box_threshold=config.BOX_THRESHOLD,
            text_threshold=config.TEXT_THRESHOLD,
        )
        node.get_logger().info("GroundingDINO loaded")
    except Exception as error:
        node.get_logger().error(f"GroundingDINO load failed: {error}")

    try:
        from tmah_vlm.reasoning.selector import QwenSelector
        node.selector = QwenSelector()
        node.get_logger().info("Qwen selector loaded")
    except Exception as error:
        node.get_logger().error(f"Qwen selector load failed: {error}")

    try:
        from tmah_vlm.segmentation.segmenter import SAMSegmenter
        node.segmenter = SAMSegmenter(model_id=config.SEGMENTATION_MODEL_ID)
        node.get_logger().info("SAM segmenter loaded")
    except Exception as error:
        node.get_logger().error(f"SAM segmenter load failed: {error}")

    node.get_logger().info("Model loading finished")


def initialize_subscribers(node):
    """ROS subscriber 목록. 실제 콜백 로직은 callback/sensor_callbacks.py에 있다."""
    node.question_sub = node.create_subscription(
        String,
        config.TOPIC_QUESTION,
        lambda msg: question_callback(node, msg),
        5,
    )
    node.pose_sub = node.create_subscription(
        Odometry,
        config.TOPIC_STATE,
        lambda msg: pose_callback(node, msg),
        5,
    )
    node.image_sub = node.create_subscription(
        Image,
        config.TOPIC_IMAGE,
        lambda msg: image_callback(node, msg),
        5,
    )
    node.scan_sub = node.create_subscription(
        PointCloud2,
        config.TOPIC_SCAN,
        lambda msg: scan_callback(node, msg),
        5,
    )


def initialize_publishers(node):
    """ROS publisher 목록."""
    node.waypoint_pub = node.create_publisher(
        Pose2D,
        config.TOPIC_WAYPOINT,
        5,
    )

    # 챌린지 visualizationTools/RViz가 Marker(단수)로 구독 중이라 타입 고정.
    node.marker_pub = node.create_publisher(
        Marker,
        config.TOPIC_MARKER,
        5,
    )
    # bbox wireframe은 우리 전용 디버그 토픽이라 별도로 뺌.
    node.wireframe_marker_pub = node.create_publisher(
        Marker,
        config.TOPIC_MARKER_WIREFRAME,
        5,
    )

    node.numerical_pub = node.create_publisher(
        Int32,
        config.TOPIC_NUMERICAL,
        5,
    )


def initialize_timers(node):
    """주기적으로 돌아가는 loop."""
    node.main_timer = node.create_timer(0.2, node.main_control_loop)
    node.health_timer = node.create_timer(3.0, lambda: heartbeat(node))
