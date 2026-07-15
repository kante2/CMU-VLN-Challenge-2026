#!/usr/bin/env python3
"""
TmahVLM 노드 초기화 — TmahVLM.__init__()이 순서대로 호출하는 initialize_* 함수 모음.

  initialize_state       -> 계속 들고 있어야 하는 최신 상태값 초기화
  initialize_modules      -> TF 변환기 준비 + GroundingDINO/Qwen 백그라운드 로딩(load_models)
  initialize_subscribers  -> ROS subscriber 등록 (콜백 로직은 common/callback.py,
                             sensor_process/callback.py)
  initialize_publishers   -> ROS publisher 등록
  initialize_timers       -> 주기 실행 timer 등록

node를 인자로 받는 자유 함수라 solver들의 *_process(node, ...)와 같은 패턴이다.
"""

import threading
from collections import deque

from std_msgs.msg import String, Int32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker, MarkerArray

from tmah_vlm import config
from tmah_vlm.sensor_process.coordinate_transform import CoordinateTransformer
from tmah_vlm.common.callback import question_callback, pose_callback
from tmah_vlm.sensor_process.callback import image_callback
from tmah_vlm.sensor_process.callback import scan_callback
from tmah_vlm.common.helpers import heartbeat
from tmah_vlm.reasoning.graph.visualizer import publish_scene_graph_markers


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
    node.state_lock = threading.Lock() # node.state_lock = threading.Lock()
    # 여러 스레드가 같은 데이터를 동시에 건드릴 때 생기는 경합(race condition)을 막는다.
    #  한 번에 한 스레드만 락을 잡을 수 있고, 나머지는 락이 풀릴 때까지 기다립니다.
    node.last_wait_log_time = 0.0
    node.scene_graph = None


def initialize_modules(node):
    """TF 변환기와 VLM/Detector/Segmenter 모델을 준비한다."""
    node.transformer = CoordinateTransformer(node)

    node.detector = None
    node.selector = None
    node.segmenter = None
    node.vlm_captioner = None
    node.vlm_captioner_failed = False

    node.get_logger().info("Loading models in background...")
    model_thread = threading.Thread(target=load_models, args=(node,), daemon=True)
    model_thread.start()


def load_models(node):
    """GroundingDINO, Qwen selector, SAM segmenter를 로드한다 (백그라운드 스레드)."""
    try:
        from tmah_vlm.sensor_process.detector import GroundingDINODetector
        node.detector = GroundingDINODetector(
            box_threshold=config.BOX_THRESHOLD,
            text_threshold=config.TEXT_THRESHOLD,
        )
        node.get_logger().info("GroundingDINO loaded")
    except Exception as error:
        node.get_logger().error(f"GroundingDINO load failed: {error}")

    if config.ENABLE_QWEN_SELECTOR:
        try:
            from tmah_vlm.sensor_process.selector import QwenSelector
            node.selector = QwenSelector()
            node.get_logger().info("Qwen selector loaded")
        except Exception as error:
            node.get_logger().error(f"Qwen selector load failed: {error}")
    else:
        node.get_logger().info("Qwen selector disabled; using first detection candidate")

    try:
        from tmah_vlm.sensor_process.segmenter import SAMSegmenter
        node.segmenter = SAMSegmenter(
            model_id=config.SEGMENTATION_MODEL_ID,
            device=config.SEGMENTATION_DEVICE,
        )
        node.get_logger().info(f"SAM segmenter loaded (device={config.SEGMENTATION_DEVICE})")
    except Exception as error:
        node.get_logger().error(f"SAM segmenter load failed: {error}")

    node.get_logger().info("Model loading finished")


def initialize_subscribers(node):
    """ROS subscriber 목록. 실제 콜백 로직은 common/callback.py,
    sensor_process/callback.py에 있다."""
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
    node.scene_graph_marker_pub = node.create_publisher(
        MarkerArray,
        config.TOPIC_SCENE_GRAPH_MARKERS,
        5,
    )

    node.numerical_pub = node.create_publisher(
        Int32,
        config.TOPIC_NUMERICAL,
        5,
    )


def initialize_timers(node):
    """주기적으로 돌아가는 loop. main_control_loop 타이머는 main_node.py에서 등록한다."""
    node.health_timer = node.create_timer(
        config.HEALTH_TIMER_PERIOD_SEC,
        lambda: heartbeat(node),
    )
    node.scene_graph_marker_timer = node.create_timer(
        config.SCENE_GRAPH_MARKER_PERIOD_SEC,
        lambda: publish_scene_graph_markers(node),
    )
