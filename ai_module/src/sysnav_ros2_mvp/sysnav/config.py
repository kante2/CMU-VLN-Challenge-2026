"""SysNav single-room MVP configuration."""

from __future__ import annotations

import os

TOPIC_QUESTION = "/challenge_question"
TOPIC_STATE = "/state_estimation"
TOPIC_IMAGE = "/camera/image"
TOPIC_SCAN = "/sensor_scan"
TOPIC_WAYPOINT = "/way_point_with_heading"
TOPIC_OBJECT_MARKERS = "/sysnav/object_markers"
# mission3(Instruction-Following)가 각 step에서 실제로 찍은 goal 좌표를 디버그용으로
# 남긴다 - "success는 떴는데 실제로는 물체 앞까지 안 갔다"류 문제를 RViz에서 눈으로
# 바로 확인하기 위함(goal_reached()의 도달 판정 반경 자체는 로봇 실제 위치와 별개).
TOPIC_MISSION3_GOAL_MARKERS = "/sysnav/mission3_goal_markers"
# 채점 대상 토픽(README) - Object Reference: 이 두 개는 절대 이름/타입을 바꾸지 말 것
# (Marker 단수, MarkerArray 아님 - CLAUDE.md의 하드-런 규칙).
TOPIC_SELECTED_OBJECT_MARKER = "/selected_object_marker"
# 채점 대상 토픽(README) - Numerical.
TOPIC_NUMERICAL_RESPONSE = "/numerical_response"

OBJECT_MARKER_FRAME_ID = "map"
OBJECT_MARKER_DEFAULT_SIZE_M = 0.3

CONTROL_PERIOD_SEC = 0.20
# 디버깅용 미션 상태 대시보드(mission_dashboard.py) - ai_module/debug/mission_status_latest.html을
# 이 주기(초)마다 다시 써서 덮어쓴다. README의 질문당 10분 제한(재탐색+답변 합산) 표시에도 쓴다.
MISSION_DASHBOARD_REFRESH_SEC = 1.0
MISSION_TIME_LIMIT_SEC = 600.0
PERCEPTION_WHILE_MOVING_INTERVAL_SEC = 1.50
SENSOR_SYNC_TOLERANCE_SEC = 0.30
SCAN_BUFFER_SIZE = 40
POSE_BUFFER_SIZE = 100
# base autonomy(waypointConverter.cpp)의 자체 도착 판정 반경(waypointXYRadius)이 0.3m라
# 그보다 타이트하면 base autonomy는 이미 "도착"으로 보고 정지했는데 우리만 계속 기다리다
# stuck-timeout으로 SKIP되는 근본적 불일치가 생긴다(2026-08-10 실측: 0.15m일 때 0.31~0.34m
# 남기고 반복 SKIP). 0.3m보다 살짝 여유를 둔다.
GOAL_REACHED_DISTANCE_M = 0.35
# exploration goal까지 거리가 이 이상 줄지 않은 채 이 시간(초)이 지나면 도달 불가로 보고 포기한다.
# (벽 너머 등 실제로는 갈 수 없는 waypoint에 로봇이 영원히 박혀있는 것을 막기 위한 안전장치)
EXPLORATION_STUCK_TIMEOUT_SEC = 8.0
EXPLORATION_STUCK_PROGRESS_M = 0.10
# mission3(Instruction-Following) leg 이동은 exploration frontier hopping과 성격이 다르다 -
# 후보가 수십 개라 하나가 느리면 빨리 포기하고 다음으로 넘어가는 게 이득인 exploration과 달리,
# mission3는 goal이 1~3개뿐이고 그중 is_stop=True는 채점 대상 정지점이라 하나하나가 중요하다.
# EXPLORATION_STUCK_TIMEOUT_SEC(8초)를 그대로 썼더니 회전-후-직진 구간이나 우회 경로 때문에
# 8초 창 안에 10cm 진전이 안 잡혀서 실제로 접근 중인데도 도착 직전에 "도달 불가"로 오판하고
# 건너뛰는 사례가 실측됨(2026-08-10, 3 step 전부 정확히 8.0초 만에 SKIP, robot_pose는 거의
# 안 움직인 채 그대로 SUCCESS 처리). mission3 leg에는 훨씬 여유 있는 timeout을 따로 쓴다.
MISSION3_LEG_STUCK_TIMEOUT_SEC = 30.0
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

# 문장 -> (target, attributes, relation_chain) 파싱을 LLM(Gemini)이 하도록 켤지.
# SysNav 논문 Sec. III의 G=(c_tgt, Φ) 정의를 그대로 따르는 task/llm_query_parser.py가
# 이걸 담당한다 - 실패(키 없음/에러/빈 응답 등 무엇이든)하면 항상 정규식 기반
# task/query_parser.extract_target()로 자동 폴백한다.
LLM_QUERY_PARSER_ENABLED = os.getenv(
    "LLM_QUERY_PARSER_ENABLED", "1"
) not in ("0", "false", "False")

# YOLO-World가 애매한 confidence(YOLO_CONFIDENCE는 넘었지만 이 값 밑)로 탐지한 것만
# Gemini에게 한 번 더 물어서 진짜 맞는지 확인한다 (예: 침대를 0.29로 sofa라고 오검출하는
# 경우). 한 프레임의 애매한 detection을 전부 모아서 Gemini 호출 1번으로 묶어 검증한다 -
# 박스마다 따로 부르면 지연이 누적된다. API 에러/키 없음 등으로 검증 자체가 안 되면
# fail-open(그냥 통과)한다 - 검증 실패가 원래 있던 탐지를 막으면 안 되므로.
DETECTION_VERIFICATION_ENABLED = os.getenv(
    "DETECTION_VERIFICATION_ENABLED", "1"
) not in ("0", "false", "False")
DETECTION_VERIFICATION_CONFIDENCE_THRESHOLD = float(
    os.getenv("DETECTION_VERIFICATION_CONFIDENCE_THRESHOLD", "0.35")
)

# SysNav paper Sec. IV-A-1 (Object Node self-attribute): 문장이 속성 제약(예: "black"
# chair)을 요구하면, 후보가 몇 개든(1개여도!) 그 카테고리 후보 전부를 VLM한테 이미지로
# 보여주고 속성이 맞는지 확인한다. 끄면 예전처럼 속성 검증 없이 카테고리만 보고 진행한다
# (reasoning/attribute_verifier.py).
ATTRIBUTE_VERIFICATION_ENABLED = os.getenv(
    "ATTRIBUTE_VERIFICATION_ENABLED", "1"
) not in ("0", "false", "False")

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
GROUNDING_MIN_POINTS = 5
GROUNDING_MAX_OBJECT_POINTS = 2048
# SAM2 mask 안에 들어온 LiDAR point를 로봇 기준 거리로 1차원 클러스터링할 때, 이
# 값(m) 이상 거리 차이가 나면 다른 물체(다른 깊이)로 취급해서 군집을 나눈다.
# 목표 물체 앞/뒤로 다른 물체가 mask 경계에 살짝 겹쳐 들어왔을 때, 가장 점이 많이
# 뭉친 구간(=진짜 그 물체 표면)만 남기고 나머지는 버린다.
DEPTH_CLUSTER_GAP_M = 0.3
# 유리창처럼 LiDAR 반사가 잘 안 되는 물체는 GROUNDING_MIN_POINTS(정밀 bbox에 필요한
# 최소 point 수)를 영영 못 채워서 3D grounding 자체가 항상 버려졌다 - relation 판정
# (nearest/near/between)은 정밀한 bbox 없이 대략적인 위치만으로도 충분하므로, 이보다
# 적은(그러나 0은 아닌) point만 있어도 "approximate" 등급으로 위치를 만든다. 특정
# 카테고리를 하드코딩하는 게 아니라 point 개수 기준의 일반 규칙이라 어떤 물체든
# 이 상황이면 동일하게 적용된다. object_memory에 들어간 뒤 나중에 더 잘 grounding된
# 관측이 들어오면 지수이동평균(_merge())으로 자연스럽게 정밀한 위치로 수렴한다.
GROUNDING_MIN_POINTS_APPROXIMATE = 1
GROUNDING_APPROXIMATE_DEFAULT_SIZE_M = 0.3
# crop_image(attribute_verifier/gemini_selector용)는 배경을 회색으로 지운 물체
# 단독 이미지라 relation 판정("이 물체 주변에 참조 물체가 보이는가",
# reasoning/relation_image_verifier.py)엔 못 쓴다 - 배경 자체가 없으니까. bbox
# 한 변 길이의 이 비율만큼 여유를 두고, 배경은 안 지운 채 잘라내는
# context_image를 따로 만든다.
CONTEXT_CROP_MARGIN_RATIO = 1.0

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
# 이 값 x2가 "통과 가능한 최소 문/통로 폭"이다 (양쪽에서 벽을 이만큼씩 부풀리므로).
# 원래 0.45 -> 0.30으로 낮췄는데도 문 통과가 잘 안 돼서 더 낮춤(최소 통과 폭 0.4m).
# 실제 로봇 폭을 몰라 정확한 값은 아니니, 나중에 실측되면 갱신할 것 - 로봇이 문틀에
# 부딪히면 다시 올려야 한다.
ROBOT_CLEARANCE_M = 0.20

# Room Node segmentation (SysNav paper Sec. IV-A-1, "Scene Representation Building").
# 논문은 3D point cloud에서 벽 평면을 피팅해서 방을 나누지만, 우리는 이미 2D top-down
# occupancy grid(OCC_OCCUPIED = 벽의 2D 투영)가 있어서 그걸로 대신한다: 벽에서
# ROOM_SEGMENTATION_MIN_CLEARANCE_M 이상 떨어진 "넓은" 영역을 방의 core로 잡고,
# distance-transform 기반 watershed로 거기서부터 채워나간다 - 문처럼 좁은 통로는
# 이 값보다 거리가 짧아서 자연스럽게 두 방 사이의 경계(ridge)가 된다. 문 폭의
# 절반보다는 크고, 가장 좁은 방의 절반 폭보다는 작아야 한다.
ROOM_SEGMENTATION_MIN_CLEARANCE_M = 0.75
ROOM_SEGMENTATION_MIN_ROOM_CELLS = 40
# OCC_OCCUPIED은 벽/가구를 구분 안 하므로, room 경계로는 "충분히 높이까지 닿은" 셀만
# 진짜 벽으로 본다 (소파/테이블/의자 등은 보통 이 아래, 실제 벽은 MAP_OBSTACLE_Z_MAX_M
# 안에서도 이 위까지 계속 point가 찍힘). CoveragePlanner.max_height 기준.
ROOM_WALL_MIN_HEIGHT_M = 1.30
# 벽(위 기준으로 판정된 것) 주변에 이만큼 더 두껍게 padding을 줘서 distance-transform을
# 계산한다 - 문처럼 좁은 통로가 코어 임계값보다 확실히 더 낮게 나오도록 여유를 더 준다.
# (room segmentation 전용 값이라 실제 주행 가능 여부(ROBOT_CLEARANCE_M)에는 영향 없음 -
# 방을 나누는 그림/판정만 더 정확해질 뿐, 원래도 두꺼워서 문 통과가 막힌 원인은 아니었지만
# 요청대로 줄여둠.)
ROOM_SEGMENTATION_WALL_PADDING_M = 0.05

# Room Node identity persistence (rooms/room_registry.py). RoomSegmenter.segment()는
# 매 mapping cycle마다 watershed를 처음부터 다시 돌려 room_id를 1..N으로 새로 매기므로
# (사이클 간 정체성이 없음), RoomRegistry가 centroid 근접 매칭으로 "같은 물리적 방"을
# 안정적인 room_id에 이어붙인다. 이 반경(m)보다 centroid가 더 멀어지면 다른 방으로
# 취급해 새 id를 발급한다.
ROOM_REGISTRY_MATCH_RADIUS_M = 2.0

# Room category 추론(VLM) - SysNav paper Sec. IV-A-1의 room attribute c_i. Object
# self-attribute와 같은 on-demand 패턴: room의 "best view"(가장 coverage_voxel_count가
# 큰 viewpoint, 논문 각주의 "visible voxels 최대화")가 바뀔 때만 재추론하고, 그 외엔
# 캐시를 재사용한다.
ROOM_CLASSIFICATION_ENABLED = os.getenv("SYSNAV_ROOM_CLASSIFICATION_ENABLED", "1") not in ("0", "false", "False")
# VLM 호출이 실패(키 없음 등)했을 때 매 mapping cycle(0.35초 간격)마다 재시도하면 로그가
# 스팸이 되므로, 실패한 room은 이 시간(초)만큼 재시도를 쉰다.
ROOM_CLASSIFICATION_RETRY_COOLDOWN_SEC = 15.0

# frontier anchor(is_near_visited 예외 대상, 문 통과 문제 때문에 도입)가 plan_route()
# 호출마다 계속 다시 잡히면(=그 옆 unknown 셀이 영영 안 풀림, 예: 유리창이라 LiDAR가
# 못 뚫는 경우) 이 횟수를 넘는 순간부터 예외 자격을 박탈해서 결국 후보에서 빠지게 한다.
# 안 그러면 "도착 -> 같은 지점 재선택 -> 도착 -> ..." 무한 루프에 걸린다(실측으로 확인됨).
EXPLORATION_ANCHOR_MAX_REVISITS = 5

FRONTIER_MIN_CLUSTER_CELLS = 5
FRONTIER_COVERAGE_RADIUS_M = 3.0  # 논문의 d_cover
# candidate에서 surface point가 "보이는지"(LOS) 판정할 때 쓰는 벽 margin. ROBOT_CLEARANCE_M
# (몸체가 실제로 지나갈 수 있는지)과는 다른 목적이라 따로 둔다 - frontier는 정의상 벽
# 바로 옆에 있는 경우가 많은데, candidate는 항상 clearance 밖에 서야 해서 그 둘 사이
# 직선이 clearance 버퍼 셀을 스치기만 해도 "안 보인다"고 오판되어 coverage 점수가
# 0이 되어버리는 문제가 있었다. 대각선 코너 스침만 막을 정도의 작은 값으로 둔다.
FRONTIER_LOS_WALL_MARGIN_M = 0.15

# In-room exploration policy (SysNav paper Sec. IV-B-1): stochastic candidate selection
EXPLORATION_CANDIDATE_SAMPLES = 60  # |H|, 한 사이클에 샘플링할 pose 후보 수
EXPLORATION_MIN_SCORE_DELTA = 3     # δ, wcov가 이 밑으로 떨어지면 후보 뽑기를 멈춤
EXPLORATION_STOCHASTIC_TRIALS = 4   # K, stochastic sampling을 반복해서 TSP 비용 최소인 것을 채택
# 한 plan_route() 사이클에서 최대 몇 개의 candidate를 한 번에 뽑아 TSP로 묶을지. frontier
# anchor(모든 남은 frontier마다 보장되는 후보)까지 생기면서, 방 양쪽 끝처럼 서로 먼 두 곳이
# 한 번에 다 뽑혀서 "이쪽 찍고 저쪽 찍고" 왔다갔다 하는 경로가 나오는 문제가 있었다. 1로
# 두면 매 사이클 가장 좋은 후보 하나만 골라서 그리로 갔다가, 도착하면 최신 지도로 다시
# 고른다 - 논문의 "batch로 여러 개 묶어서 효율화"보다는 덜 효율적일 수 있지만 왔다갔다
# 하는 걸 원천적으로 막는다.
EXPLORATION_MAX_CANDIDATES_PER_CYCLE = 1
# candidate를 뽑는 확률 가중치를 순수 wcov가 아니라 로봇 현재 위치로부터의 거리로
# 감쇠시킨다 (weight = wcov / (1 + distance/halflife)). 이 값(m)만큼 떨어지면 가중치가
# 절반이 된다. MAX_CANDIDATES_PER_CYCLE=1로도, 방 양쪽 끝의 wcov 점수가 비슷하면 매
# cycle 거의 50:50으로 반대쪽이 뽑혀서 여전히 왔다갔다하는 것처럼 보이는 문제가 있었다 -
# 가까운 후보를 확실히 우선해서 이 진동을 줄인다. 값을 키우면 거리 영향이 약해진다.
EXPLORATION_DISTANCE_PENALTY_HALFLIFE_M = 3.0
# candidate까지 A*로 구한 경로를 이 간격(m)으로 잘라 중간 waypoint를 만든다. 최종 목적지 하나만
# 찍어서 보내면 그 사이에 벽이 있을 때 base autonomy가 돌아가지 못하고 벽에 막힐 수 있어서,
# 내부 occupancy grid가 이미 계산해둔 (벽을 피해가는) A* 경로를 따라 짧게 여러 번 나눠 보낸다.
# (원래 1.5m라 waypoint(보라색 점)가 너무 자주/가깝게 찍혀서 3.0m로 늘림 - 뚫린 직선
# 구간에서는 여전히 한 번에 더 길게 건너뛴다, string-pulling이라 코너/문에서는 자동으로
# 촘촘해짐 - _simplify_path_indices 참고.)
EXPLORATION_PATH_WAYPOINT_SPACING_M = 3.0

VIEWPOINT_MIN_DISTANCE_M = 1.0

# Instruction-Following (missions/mission3_pipe.py) - "avoiding the path between A
# and B"/"avoid the path near Z" 같은 negative constraint를 A-B 선분(또는 Z 한 점)
# 주변 이 반경(m)만큼 non-traversable로 마킹해서 plan_direct_path()가 우회하게 한다.
# 실제 문/복도 폭을 모르므로 대략적인 근사치 - 너무 크면 우회 자체가 불가능해질 수 있다.
INSTRUCTION_FORBIDDEN_RADIUS_M = 0.8

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
