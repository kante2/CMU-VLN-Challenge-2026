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

# 챌린지 쪽 visualizationTools/RViz가 이미 이 토픽을 Marker(단수) 타입으로 구독하고
# 있어서(dummy_vlm도 동일하게 씀) 여기는 절대 MarkerArray로 바꾸면 안 된다.
# 타입이 다르면 에러 없이 그냥 연결이 안 된다.
TOPIC_MARKER = "/selected_object_marker"

# 3D bbox wireframe은 challenge 쪽 계약이 없는 우리 전용 디버그 토픽이라 자유롭게 사용.
TOPIC_MARKER_WIREFRAME = "/selected_object_marker_wireframe"

# 누적된 online scene graph 전체를 RViz에서 보기 위한 디버그 MarkerArray 토픽.
TOPIC_SCENE_GRAPH_MARKERS = "/scene_graph_markers"

# 저장된 scene_graph_latest.json을 다시 읽어서 RViz에 띄우는 전용 토픽.
# live tmah_vlm graph publisher와 충돌하지 않게 별도 토픽으로 둔다.
TOPIC_SCENE_GRAPH_JSON_MARKERS = "/scene_graph_json_markers"

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
    ("sensor", "camera", (0.0, 0.0, 0.1), (-0.5, 0.5, -0.5, 0.5)),
]

# -----------------------------------------------------------------------------
# 360 panorama camera
# -----------------------------------------------------------------------------
PANO_WIDTH = 1920
PANO_HEIGHT = 640
PANO_H_FOV_DEG = 360.0

# test_pano_lidar_overlay.py로 LiDAR point cloud를 카메라 이미지에 직접 겹쳐서
# 실측 검증함: 120도로는 point band가 실제 벽/천장 경계선과 어긋났고,
# 180도로 맞추니 경계선을 따라감. (2026-07-09)
PANO_V_FOV_DEG = 180.0

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

# Qwen selector is optional. On 8GB GPUs it can consume the remaining CUDA
# memory after GroundingDINO loads, causing detection itself to fail with
# CUBLAS/CUDA OOM. If selector=loading never resolves in the heartbeat (GPU
# OOM), set this back to False; detection falls back to candidate #0 when no
# selector is loaded.
ENABLE_QWEN_SELECTOR = True

# -----------------------------------------------------------------------------
# Segmentation (SAM) — box 안 배경 point 오염을 줄이기 위한 pixel 단위 마스크.
# 선택된 후보 1개에만 돌린다 (매 후보마다 돌리지 않음).
# -----------------------------------------------------------------------------
SEGMENTATION_MODEL_ID = "facebook/sam-vit-base"

# SAM을 어느 device에서 돌릴지. GroundingDINO + Qwen2.5-VL이 이미 GPU를 거의 다
# 써서(7.5GB GPU 기준 ~350MB만 남음) SAM 추론이 CUDA OOM으로 매번 실패했다.
# 세그멘테이션은 쿼리당 1회라 CPU로 돌려도 몇 초면 되므로 기본을 "cpu"로 둔다.
# GPU 여유가 충분한 환경이면 "cuda"로 바꿔도 된다.
SEGMENTATION_DEVICE = "cpu"

# -----------------------------------------------------------------------------
# Spatial reasoning toolbox (spatial/relations.py, SORT-3D Module 4)
# 기본값. LLM이 좌표 계산을 직접 하면 실수하므로, 미리 정의된 함수가 대신 계산한다.
# -----------------------------------------------------------------------------
SPATIAL_NEAR_THRESHOLD_M = 1.5          # find_near 기본 반경
SPATIAL_BETWEEN_CORRIDOR_M = 1.0        # find_between 기본 통로 폭(선분에서 이 거리 이내)
SPATIAL_ABOVE_BELOW_MIN_DIFF_M = 0.1    # find_above/find_below 최소 높이차

# -----------------------------------------------------------------------------
# Object crop captioning (SORT3D-style)
# -----------------------------------------------------------------------------
# Hidden evaluation에서는 object_list.txt를 쓰지 않으므로, 관측된 object crop에서
# caption을 만들어 scene graph에 저장한다. GPU OOM을 피하기 위해 CPU lazy-load가 기본.
ENABLE_VLM_CAPTIONER = True
CAPTION_MODEL_ID = "microsoft/Florence-2-base"
CAPTION_DEVICE = "cpu"
CAPTION_MAX_NEW_TOKENS = 64
CAPTION_CROP_MARGIN_PX = 16

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
# 3D bounding box 크기 추정 (bbox3d/)
# -----------------------------------------------------------------------------
# 선택된 depth cluster의 실제 point 퍼짐으로 물체 크기를 추정한다.
# point가 너무 적으면(노이즈 위험) 추정을 포기하고 BBOX3D_DEFAULT_SIZE_M 고정 박스로 표시한다.
BBOX3D_MIN_POINTS = 8

# 양 끝 percentile%를 잘라내고 그 사이 범위를 크기로 쓴다.
# min/max를 그대로 쓰면 point 하나가 튀어도 박스가 확 커진다.
BBOX3D_PERCENTILE = 5.0

# 추정 크기의 하한/상한. 너무 작으면(포인트가 한 곳에 몰림) 안 보이고,
# 너무 크면(배경 point 섞임) 비현실적이라 이 범위로 clip한다.
BBOX3D_MIN_SIZE_M = 0.15
BBOX3D_MAX_SIZE_M = 2.0

# 크기 추정을 못 했을 때(fallback ray 경로 등) 쓰는 고정 크기.
BBOX3D_DEFAULT_SIZE_M = 0.4

# -----------------------------------------------------------------------------
# Online HOV-SG style scene graph (graph/)
# -----------------------------------------------------------------------------
# 같은 room 안에서 label이 호환되고 중심 거리가 이 값보다 가까우면 같은 object node로
# 누적한다. 너무 작으면 같은 물체가 여러 노드로 쪼개지고, 너무 크면 다른 물체가 합쳐진다.
SCENE_GRAPH_MERGE_DISTANCE_M = 0.75

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
