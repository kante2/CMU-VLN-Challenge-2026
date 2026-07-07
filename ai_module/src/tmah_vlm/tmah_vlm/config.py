#!/usr/bin/env python3
"""
설정 한곳 모음. 토픽명/프레임/임계값 하드코딩 제거.
평가 환경에서 토픽이나 프레임 이름이 다르면 여기만 고치면 됨.
"""

# --- 토픽 (구독) ---
TOPIC_QUESTION = "/challenge_question"
TOPIC_STATE = "/state_estimation"
TOPIC_IMAGE = "/camera/image"
TOPIC_SCAN = "/registered_scan"

# --- 토픽 (발행) ---
TOPIC_WAYPOINT = "/way_point_with_heading"
TOPIC_MARKER = "/selected_object_marker"
TOPIC_NUMERICAL = "/numerical_response"

# --- 좌표계 프레임 ---
# CMU autonomy stack 표준 관례. 실제와 다르면 tf_monitor 로 확인 후 수정.
FRAME_MAP = "map"
FRAME_SENSOR = "sensor"      # 라이다/카메라 센서 프레임 (state_estimation 기준)

# --- 카메라 (360 파노라마 equirectangular) ---
PANO_WIDTH = 1920
PANO_HEIGHT = 640
# 파노라마 좌우 = 360도 방위각, 상하 = 수직 화각.
# 일반적 equirectangular 는 상하 180도지만, 이 시뮬은 상하 화각이 제한적일 수 있음.
# 실측으로 조정 필요. 우선 표준값.
PANO_H_FOV_DEG = 360.0
PANO_V_FOV_DEG = 180.0
# 파노라마에서 정면(로봇 전방)이 이미지의 어느 x 픽셀인지 (0=왼쪽끝).
# 보통 중앙(width/2)이 전방. 실측으로 오프셋 조정.
PANO_FORWARD_X = PANO_WIDTH / 2.0

# --- GroundingDINO 검출 ---
# 검출은 넓게(재현율↑) -> selector 가 고름. 너무 높이면 진짜 후보 놓침.
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25

# --- 3D 좌표화 ---
# 검출 박스 광선 방향에서, 라이다 포인트가 이 각도(rad) 이내면 매칭.
RAY_MATCH_ANGLE_RAD = 0.15     # 약 8.6도
# 매칭된 포인트가 없을 때 대체 거리(m) — 광선 방향으로 이만큼 앞에 있다고 가정.
FALLBACK_DEPTH_M = 2.0

# --- waypoint ---
# 대상 앞 이 거리(m)에서 멈추도록 (대상에 너무 붙지 않게).
APPROACH_STANDOFF_M = 0.8

# --- 디버그 ---
DEBUG_DIR = "/home/docker/ai_module/debug"
