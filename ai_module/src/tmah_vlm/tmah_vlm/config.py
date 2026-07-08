#!/usr/bin/env python3
"""
TMAH VLM 설정 파일.

토픽명, 프레임명, 임계값을 한곳에 모아둔다.
대회 환경에서 이름이 다르면 이 파일만 먼저 수정한다.
"""

# -----------------------------------------------------------------------------
# ROS topic
# -----------------------------------------------------------------------------
TOPIC_QUESTION = "/challenge_question"
TOPIC_STATE = "/state_estimation"
TOPIC_IMAGE = "/camera/image"

# RViz에서 확인한 PointCloud2 topic.
# 만약 실제 실행에서 /registered_scan을 쓰면 여기만 바꾸면 된다.
TOPIC_SCAN = "/sensor_scan"

TOPIC_WAYPOINT = "/way_point_with_heading"
TOPIC_MARKER = "/selected_object_marker"
TOPIC_NUMERICAL = "/numerical_response"

# -----------------------------------------------------------------------------
# Coordinate frame
# -----------------------------------------------------------------------------
FRAME_MAP = "map"
FRAME_SENSOR = "sensor"
FRAME_CAMERA = "camera"

# PointCloud2 header frame이 sensor_scan/velodyne처럼 들어와도
# fallback 변환에서는 sensor와 같은 위치로 취급한다.
FRAME_ALIASES = {
    "sensor_scan": "sensor",
    "sensor_at_scan": "sensor",   # 추가
    "velodyne": "sensor",
    "velodyne_link": "sensor",
    "lidar": "sensor",
    "lidar_link": "sensor",
}

# RViz에서 확인한 TF 구조:
# map -> sensor -> camera
# 값은 TF lookup이 실패했을 때만 쓰는 fallback이다.
# 형식: (parent, child, (x,y,z), (qx,qy,qz,qw))
STATIC_TF_FALLBACKS = [
    ("map", "sensor", (0.0, 0.0, 0.75), (0.0, 0.0, 0.0, 1.0)),
    ("sensor", "camera", (0.0, 0.0, 0.85), (-0.5, 0.5, -0.5, 0.5)),
]

# -----------------------------------------------------------------------------
# 360 panorama camera
# -----------------------------------------------------------------------------
PANO_WIDTH = 1920
PANO_HEIGHT = 640
PANO_H_FOV_DEG = 360.0
PANO_V_FOV_DEG = 120.0

# 파노라마 이미지에서 로봇 정면이 위치한 x 픽셀.
# 보통 중앙이 정면이다. 실제로 어긋나면 여기만 조정한다.
PANO_FORWARD_X = PANO_WIDTH / 2.0

# -----------------------------------------------------------------------------
# GroundingDINO
# -----------------------------------------------------------------------------
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
PANO_YAW_OFFSET_DEG = 0.0
PANO_PITCH_OFFSET_DEG = 0.0

# -----------------------------------------------------------------------------
# 2D detection -> 3D target matching
# -----------------------------------------------------------------------------
RAY_MATCH_ANGLE_RAD = 0.15
FALLBACK_DEPTH_M = 2.0
APPROACH_STANDOFF_M = 0.8

# Box projection 기반 3D target 추정 설정.
# 기존 ray-only 방식은 앞 물체가 있으면 가까운 점을 잘못 고를 수 있으므로,
# point cloud를 이미지에 다시 투영한 뒤 selected 2D box 내부 점을 우선 사용한다.
BBOX_INNER_SCALE = 0.30
BBOX_MIN_POINTS = 3
BBOX_MIN_CLUSTER_POINTS = 3
BBOX_DEPTH_CLUSTER_GAP_M = 0.35
BBOX_MIN_DEPTH_M = 0.25
BBOX_MAX_DEPTH_M = 15.0

# -----------------------------------------------------------------------------
# Debug
# -----------------------------------------------------------------------------
DEBUG_DIR = "/home/docker/ai_module/debug"


# Object category별 depth cluster 선택 정책.
# TV/picture/window처럼 벽/뒤쪽 표면에 붙은 물체는 앞 선반/테이블 점이 box 안에 섞여도
# 더 뒤쪽 depth cluster를 선택하는 것이 안전하다.
DEPTH_POLICY_FAR_OBJECTS = [
    "tv", "television", "screen", "monitor", "display",
    "picture", "painting", "poster", "photo", "frame",
    "window", "door", "sign", "board", "clock", "mirror",
]

# far-depth 정책에서 box 중심과 너무 멀리 떨어진 cluster는 배경/잡음으로 본다.
BBOX_CLUSTER_MAX_CENTER_ERROR = 0.85
BBOX_FAR_CENTER_PENALTY = 0.6

# Stage5: bbox ray bundle depth-mode policy.
# 선택된 2D box 안의 point depth histogram에서 가장 많은 거리 bin을 object depth로 사용한다.
BBOX_DEPTH_MODE_BIN_M = 0.15

# depth bin 선택 시 "포인트 개수"를 그대로 쓰지 않고,
# bbox 중심에 가까운 ray일수록 더 큰 가중치를 준다.
# 작을수록 box 중심부 ray만 강하게 따른다. 권장: 0.25 ~ 0.45
BBOX_DEPTH_MODE_CENTER_SIGMA = 0.25

# 최고 점수 bin의 몇 % 이상을 후보로 남길지.
# 낮을수록 포인트 수가 조금 적어도 중심 ray와 맞는 depth bin이 살아남는다.
BBOX_DEPTH_MODE_COUNT_KEEP_RATIO = 0.45

# bin 대표 위치가 box 중심에서 멀면 감점한다.
BBOX_DEPTH_MODE_CENTER_PENALTY = 0.35

# count/center가 비슷할 때 tie-break.
# tv/picture/window 계열은 먼 depth를, 일반 물체는 가까운 depth를 약하게 선호한다.
BBOX_DEPTH_MODE_DEPTH_TIE_WEIGHT = 0.02

# -----------------------------------------------------------------------------
# Image / PointCloud synchronization debug
# -----------------------------------------------------------------------------
# latest image와 latest scan을 그냥 쓰지 않고, image stamp와 가장 가까운 scan을 고른다.
SYNC_SCAN_BUFFER_SIZE = 30

# image와 scan timestamp 차이가 이 값보다 크면 warning을 출력한다.
SYNC_WARN_TIME_DIFF_SEC = 0.15

# selected bbox와 point cloud projection이 맞는지 확인하기 위한 overlay 이미지 저장 여부.
DEBUG_SAVE_PROJECTION_OVERLAY = True

# overlay에서 너무 많은 point를 그리면 느려지므로 샘플링한다.
DEBUG_PROJECTION_MAX_POINTS = 12000
