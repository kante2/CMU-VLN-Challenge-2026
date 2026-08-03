"""SysNav single-room MVP configuration."""

from __future__ import annotations

import os

TOPIC_QUESTION = "/challenge_question"
TOPIC_STATE = "/state_estimation"
TOPIC_IMAGE = "/camera/image"
TOPIC_SCAN = "/sensor_scan"
TOPIC_WAYPOINT = "/way_point_with_heading"
TOPIC_OBJECT_MARKERS = "/sysnav/object_markers"

OBJECT_MARKER_FRAME_ID = "map"
OBJECT_MARKER_DEFAULT_SIZE_M = 0.3

CONTROL_PERIOD_SEC = 0.20
PERCEPTION_WHILE_MOVING_INTERVAL_SEC = 1.50
SENSOR_SYNC_TOLERANCE_SEC = 0.30
SCAN_BUFFER_SIZE = 40
POSE_BUFFER_SIZE = 100
GOAL_REACHED_DISTANCE_M = 0.55
TARGET_GOAL_REACHED_DISTANCE_M = 1.50
# exploration goal까지 거리가 이 이상 줄지 않은 채 이 시간(초)이 지나면 도달 불가로 보고 포기한다.
# (벽 너머 등 실제로는 갈 수 없는 waypoint에 로봇이 영원히 박혀있는 것을 막기 위한 안전장치)
EXPLORATION_STUCK_TIMEOUT_SEC = 8.0
EXPLORATION_STUCK_PROGRESS_M = 0.10
TARGET_STATUS_LOG_INTERVAL_SEC = 1.0
TARGET_STUCK_TIMEOUT_SEC = 8.0
TARGET_STUCK_PROGRESS_M = 0.10
TARGET_STANDOFF_DISTANCE_M = 0.90
KEEP_MEMORY_BETWEEN_TASKS = True

YOLO_WORLD_WEIGHTS = os.getenv("YOLO_WORLD_WEIGHTS", "yolov8x-worldv2.pt")
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "0")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.20"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.50"))
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "1280"))
YOLO_MAX_DETECTIONS = int(os.getenv("YOLO_MAX_DETECTIONS", "30"))

SAM2_CHECKPOINT = os.getenv("SAM2_CHECKPOINT", "")
SAM2_MODEL_CFG = os.getenv("SAM2_MODEL_CFG", "configs/sam2.1/sam2.1_hiera_t.yaml")
SAM2_DEVICE = os.getenv("SAM2_DEVICE", "cuda")
SAM2_MIN_MASK_AREA_PX = 80

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_TEMPERATURE = 0.0
GEMINI_QUERY_PARSER_ENABLED = os.getenv(
    "GEMINI_QUERY_PARSER_ENABLED", "1"
) not in ("0", "false", "False")
GEMINI_VISUAL_ALIAS_FALLBACK_ENABLED = os.getenv(
    "GEMINI_VISUAL_ALIAS_FALLBACK_ENABLED", "1"
) not in ("0", "false", "False")
GEMINI_VISUAL_ALIAS_MAX_ALIASES = max(
    1, int(os.getenv("GEMINI_VISUAL_ALIAS_MAX_ALIASES", "3"))
)

SAVE_DEBUG_IMAGES = os.getenv("SYSNAV_SAVE_DEBUG_IMAGES", "1") not in ("0", "false", "False")
DEBUG_DIR = os.getenv("SYSNAV_DEBUG_DIR", "/home/docker/ai_module/debug")

# p_camera = T_LIDAR_TO_CAMERA @ p_lidar
# Default convention: lidar(x forward, y left, z up), camera(x right, y forward, z down)
T_LIDAR_TO_CAMERA = [
    [0.0, -1.0, 0.0, 0.0],
    [1.0,  0.0, 0.0, 0.0],
    [0.0,  0.0, -1.0, 0.10],
    [0.0,  0.0, 0.0, 1.0],
]
T_SENSOR_TO_BASE = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
PANORAMA_YAW_OFFSET_DEG = float(os.getenv("PANORAMA_YAW_OFFSET_DEG", "0.0"))
PANORAMA_PITCH_OFFSET_DEG = float(os.getenv("PANORAMA_PITCH_OFFSET_DEG", "0.0"))
GROUNDING_MIN_RANGE_M = 0.30
GROUNDING_MAX_RANGE_M = 30.0
GROUNDING_MIN_POINTS = 3
GROUNDING_PROVISIONAL_MIN_POINTS = 1
GROUNDING_PROVISIONAL_MIN_FRAMES = 2
GROUNDING_PROVISIONAL_TIMEOUT_SEC = 5.0
GROUNDING_PROVISIONAL_ASSOCIATION_DISTANCE_M = 0.75
GROUNDING_BBOX_FALLBACK_MAX_MASK_DISTANCE_PX = 5.0
GROUNDING_BBOX_FALLBACK_DEPTH_TOLERANCE_M = 0.30
GROUNDING_MAX_OBJECT_POINTS = 2048

ASSOCIATION_MAX_DISTANCE_M = 1.20
ASSOCIATION_DISTANCE_SIGMA_M = 0.60
ASSOCIATION_THRESHOLD = 0.58
ASSOCIATION_WEIGHT_DISTANCE = 0.55
ASSOCIATION_WEIGHT_SHAPE = 0.20
ASSOCIATION_WEIGHT_APPEARANCE = 0.25
MEMORY_MAX_POINTS_PER_OBJECT = 4096

MAP_RESOLUTION_M = 0.20
MAP_SIZE_M = 60.0
MAP_MAX_RAYS_PER_SCAN = 1200
MAP_MIN_RANGE_M = 0.40
MAP_MAX_RANGE_M = 25.0
MAP_OBSTACLE_Z_MIN_M = -0.40
MAP_OBSTACLE_Z_MAX_M = 1.60
MAP_UPDATE_INTERVAL_SEC = 0.35
OCC_UNKNOWN = -1
OCC_FREE = 0
OCC_OCCUPIED = 100
ROBOT_CLEARANCE_M = 0.45
FRONTIER_MIN_CLUSTER_CELLS = 5
FRONTIER_COVERAGE_RADIUS_M = 3.0  # 논문의 d_cover

# In-room exploration policy (SysNav paper Sec. IV-B-1): stochastic candidate selection
EXPLORATION_CANDIDATE_SAMPLES = 60  # |H|, 한 사이클에 샘플링할 pose 후보 수
EXPLORATION_MIN_SCORE_DELTA = 3     # δ, wcov가 이 밑으로 떨어지면 후보 뽑기를 멈춤
EXPLORATION_STOCHASTIC_TRIALS = 4   # K, stochastic sampling을 반복해서 TSP 비용 최소인 것을 채택
# candidate까지 A*로 구한 경로를 이 간격(m)으로 잘라 중간 waypoint를 만든다. 최종 목적지 하나만
# 찍어서 보내면 그 사이에 벽이 있을 때 base autonomy가 돌아가지 못하고 벽에 막힐 수 있어서,
# 내부 occupancy grid가 이미 계산해둔 (벽을 피해가는) A* 경로를 따라 짧게 여러 번 나눠 보낸다.
EXPLORATION_PATH_WAYPOINT_SPACING_M = 1.5

VIEWPOINT_MIN_DISTANCE_M = 1.0

# ---------------------------------------------------------------------------
# Single-room scene graph
# ---------------------------------------------------------------------------

SCENE_GRAPH_EXPORT_ENABLED = os.getenv("SYSNAV_SCENE_GRAPH_EXPORT", "1") not in ("0", "false", "False")
SCENE_GRAPH_SAVE_VIEWPOINT_IMAGES = os.getenv("SYSNAV_SCENE_GRAPH_SAVE_IMAGES", "1") not in ("0", "false", "False")
SCENE_GRAPH_USE_GEMINI_RELATIONS = os.getenv("SYSNAV_SCENE_GRAPH_USE_GEMINI", "1") not in ("0", "false", "False")
SCENE_GRAPH_SINGLE_ROOM_ID = int(os.getenv("SYSNAV_SINGLE_ROOM_ID", "0"))
SCENE_GRAPH_SINGLE_ROOM_NAME = os.getenv("SYSNAV_SINGLE_ROOM_NAME", "Room_0")
SCENE_GRAPH_RELATION_MIN_CONFIDENCE = float(os.getenv("SYSNAV_RELATION_MIN_CONFIDENCE", "0.55"))

# Geometric fallback thresholds for on-demand Object-Object edges.
SCENE_GRAPH_NEAR_DISTANCE_M = float(os.getenv("SYSNAV_NEAR_DISTANCE_M", "1.20"))
SCENE_GRAPH_BESIDE_Z_TOLERANCE_M = float(os.getenv("SYSNAV_BESIDE_Z_TOLERANCE_M", "0.80"))
SCENE_GRAPH_DIRECTION_MARGIN_M = float(os.getenv("SYSNAV_DIRECTION_MARGIN_M", "0.20"))
SCENE_GRAPH_ON_VERTICAL_TOLERANCE_M = float(os.getenv("SYSNAV_ON_VERTICAL_TOLERANCE_M", "0.25"))
SCENE_GRAPH_ON_HORIZONTAL_MARGIN_M = float(os.getenv("SYSNAV_ON_HORIZONTAL_MARGIN_M", "0.20"))
SCENE_GRAPH_BETWEEN_LINE_TOLERANCE_M = float(os.getenv("SYSNAV_BETWEEN_LINE_TOLERANCE_M", "0.70"))

# ---------------------------------------------------------------------------
# SysNav paper-style Viewpoint coverage
# ---------------------------------------------------------------------------
# C(v): map-frame 3D voxel keys observed by 360-degree LiDAR rays within d_cover.
# A new representative Viewpoint is added only when |C_t - C_prev| > omega.
VIEWPOINT_COVERAGE_DISTANCE_M = float(os.getenv("SYSNAV_VIEWPOINT_COVERAGE_DISTANCE_M", "4.0"))
VIEWPOINT_COVERAGE_VOXEL_SIZE_M = float(os.getenv("SYSNAV_VIEWPOINT_COVERAGE_VOXEL_SIZE_M", "0.40"))
VIEWPOINT_NOVEL_VOXEL_THRESHOLD = int(os.getenv("SYSNAV_VIEWPOINT_NOVEL_VOXEL_THRESHOLD", "120"))
VIEWPOINT_COVERAGE_MAX_RAYS = int(os.getenv("SYSNAV_VIEWPOINT_COVERAGE_MAX_RAYS", "1600"))
VIEWPOINT_COVERAGE_MIN_RANGE_M = float(os.getenv("SYSNAV_VIEWPOINT_COVERAGE_MIN_RANGE_M", "0.20"))
VIEWPOINT_COVERAGE_Z_MIN_M = float(os.getenv("SYSNAV_VIEWPOINT_COVERAGE_Z_MIN_M", "-0.60"))
VIEWPOINT_COVERAGE_Z_MAX_M = float(os.getenv("SYSNAV_VIEWPOINT_COVERAGE_Z_MAX_M", "2.50"))
