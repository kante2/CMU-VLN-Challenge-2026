"""SysNav single-room MVP configuration."""

from __future__ import annotations

import os

TOPIC_QUESTION = "/challenge_question"
TOPIC_STATE = "/state_estimation"
TOPIC_IMAGE = "/camera/image"
TOPIC_SCAN = "/sensor_scan"
TOPIC_WAYPOINT = "/way_point_with_heading"
# base autonomy(terrainAnalysis)의 지형 분석 결과. README의 System Outputs 표에 있는
# 테스트 때도 사용 허용된 토픽이다. waypointConverter가 우리 waypoint를 받아들일지
# 판정할 때 쓰는 것과 동일한 데이터라, 우리도 같은 걸 보고 목표를 찍는다
# (navigation/terrain_monitor.py).
TOPIC_TERRAIN_MAP = "/terrain_map"
TOPIC_OBJECT_MARKERS = "/sysnav/object_markers"
# mission3(Instruction-Following)가 각 step에서 실제로 찍은 goal 좌표를 디버그용으로
# 남긴다 - "success는 떴는데 실제로는 물체 앞까지 안 갔다"류 문제를 RViz에서 눈으로
# 바로 확인하기 위함(goal_reached()의 도달 판정 반경 자체는 로봇 실제 위치와 별개).
TOPIC_MISSION3_GOAL_MARKERS = "/sysnav/mission3_goal_markers"
# base autonomy(waypointConverter)는 우리가 보낸 좌표를 그대로 쓰지 않는다 -
# obstacleDisThre(0.75m) 안에 장애물이 있는 지점은 후보에서 빼고, 남은 travArea 점 중
# 하나로 목표를 갈아끼운다(waypointConverter.cpp의 waypointAdj 분기). 우리가 "원한" 좌표는
# Pose2D(header 없음)라 RViz에 못 띄우므로, 같은 좌표를 Marker로 한 번 더 내보내서
# base autonomy가 실제로 향하는 /way_point와 나란히 비교할 수 있게 한다.
TOPIC_REQUESTED_WAYPOINT = "/sysnav/requested_waypoint"
# waypointConverter가 최종 확정한 목표(PointStamped). 우리가 구독해서 위 요청 좌표와의
# 차이를 계산/기록한다 - 읽기 전용이며 base autonomy 동작에는 영향을 주지 않는다.
TOPIC_ACTUAL_WAYPOINT = "/way_point"
# 우리 planner가 실제로 보고 있는 지도/경로를 RViz에서 base autonomy 것과 겹쳐 보기
# 위한 발행 전용 토픽들. PNG(exploration_debug_latest.png)로도 보지만, 그건 별도 창이라
# /registered_scan, /terrain_map, /way_point와 같은 좌표계에 겹쳐볼 수가 없다.
# occupancy grid의 셀 값(OCC_UNKNOWN/-1, OCC_FREE/0, OCC_OCCUPIED/100)은 이미
# nav_msgs/OccupancyGrid 규약과 같아서 그대로 내보내면 된다.
TOPIC_SYSNAV_OCCUPANCY = "/sysnav/occupancy_grid"
TOPIC_SYSNAV_PATH = "/sysnav/planned_path"
TOPIC_SYSNAV_FRONTIER = "/sysnav/frontier"
# 지도 발행 주기(초). 300x300 int8 = 90KB라 1Hz면 부담이 없다.
MAP_PUBLISH_INTERVAL_SEC = 1.0
# 요청 좌표와 실제 목표가 이만큼 넘게 벌어지면 "밀려났다"고 보고 로그/추적에 남긴다.
WAYPOINT_DISPLACEMENT_WARN_M = 0.30
# /way_point에는 "밀림"이 아닌 값이 섞여 들어온다. 그대로 세면 통계가 망가진다:
#   1. waypointConverter가 목표에 도달하면(waypointReached) /way_point를 "차량 앞
#      waypointProjDis(0.5m)" 지점으로 계속 재발행한다 - 이건 밀어낸 게 아니다.
#   2. 첫 waypoint를 받기 전 초기값 (0,0)이 나온다 (실측 trace에서 389건 중 99건).
# 그래서 (1)은 로봇에서 0.5m±tolerance면 제외하고, (2)는 정확히 (0,0)이면 제외하며,
# 우리가 발행한 직후 WAYPOINT_MEASURE_WINDOW_SEC 안의 값만 센다.
WAYPOINT_PROJ_DIS_M = 0.5
WAYPOINT_PROJ_TOLERANCE_M = 0.15
WAYPOINT_MEASURE_WINDOW_SEC = 2.0
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
# 남기고 반복 SKIP). 즉 0.3m가 하한이다.
#
# 0.35 -> 0.5 (2026-08-12): base autonomy 도착 반경과의 불일치를 줄이기 위해
# 조정했다. 주의: 이 값은 탐색 waypoint에도 쓰이며, 최종 target 성공 조건은 아래
# TARGET_SUCCESS_DISTANCE_M가 별도로 고정한다.
GOAL_REACHED_DISTANCE_M = 0.5
# 최종 target marker에 대한 미션 성공 반경. 탐색 waypoint 허용 반경과 분리해,
# 향후 탐색 튜닝이 최종 미션 성공 조건을 느슨하게 바꾸지 못하게 한다.
TARGET_SUCCESS_DISTANCE_M = 0.5

# Object Reference 답안(/selected_object_marker) 재발행 주기.
#
# 왜 한 번으로 부족한가: 이 publisher는 기본 QoS(RELIABLE/VOLATILE)라, 평가 노드가
# 그 순간 구독 중이 아니면(discovery가 아직 안 붙었거나 늦게 뜬 경우) 메시지를 영영
# 못 받는다. 그러면 답이 맞아도 0점이다. 실측(2026-08-23): 선택 확정 시점에 딱 한 번만
# 발행하고 있었다.
#
# QoS를 TRANSIENT_LOCAL(latched)로 바꾸는 게 더 깔끔하지만, 이 토픽은 dummy_vlm과
# 챌린지 visualizationTools가 고정 규격으로 구독 중이라 CLAUDE.md가 변경을 경고한다
# (타입을 바꿨다가 두 구독자가 조용히 끊긴 전례가 있다). 그래서 재발행으로 푼다.
#
# 주행이 끝나 SUCCESS로 정착하면 재발행도 멈춘다 - 같은 답을 시간 제한까지 계속 쏘면
# "언제 답을 냈는가"가 흐려져 조기 완료 보너스(README Timing)에 불리할 수 있다.
MISSION2_ANSWER_REPUBLISH_SEC = 1.0
# exploration goal까지 거리가 이 이상 줄지 않은 채 이 시간(초)이 지나면 도달 불가로 보고 포기한다.
# (벽 너머 등 실제로는 갈 수 없는 waypoint에 로봇이 영원히 박혀있는 것을 막기 위한 안전장치)
EXPLORATION_STUCK_TIMEOUT_SEC = 8.0
EXPLORATION_STUCK_PROGRESS_M = 0.10
TARGET_STANDOFF_DISTANCE_M = 0.90

# ---------------------------------------------------------------------------
# Terrain 기반 접근 지점 판정 (navigation/terrain_monitor.py)
#
# 아래 두 값은 base autonomy waypointConverter 파라미터의 복제본이다. 그쪽이 바뀌면
# 여기도 같이 바꿔야 우리 판정이 의미를 갖는다 (2026-08-12 라이브 확인값):
#   obstacleHeightThre = 0.05   -> TERRAIN_OBSTACLE_INTENSITY
#   obstacleDisThre    = 0.75   -> TERRAIN_CLEARANCE_M
# ---------------------------------------------------------------------------
TERRAIN_OBSTACLE_INTENSITY = 0.05
# 0.75 -> 0.50 (2026-08-23, 측정 목적의 의도적 불일치).
#
# 주의: 이 값은 더 이상 waypointConverter(obstacleDisThre=0.75)의 복제본이 아니다.
# 낮추면 waypointConverter가 거부할 지점을 우리가 통과시키게 되고, 그러면 base
# autonomy가 목표를 자기 후보로 갈아끼운다(= PUSHED 이벤트가 늘어난다). 그게 이번
# 변경의 목적이다 - 실측(6분간 SNAP_FAIL 1125회, 전부 "no point with 0.75m clearance")
# 에서 0.75가 이 씬에서 사실상 만족 불가능한지, 아니면 아깝게 떨어지는지 모르기 때문에,
# 문턱을 낮춰 통과시킨 뒤 **실제로 얼마나 밀려나는지**를 PUSHED 수치로 재본다.
#
# 판단 기준:
#   PUSHED 변위가 작다(< 0.3m)  -> 0.75가 과했다. 낮춘 값을 유지할 만하다.
#   PUSHED 변위가 크다(> 1m)    -> waypointConverter가 결국 거부하는 자리였다.
#                                  0.75로 되돌리고 목표 선정 자체를 바꿔야 한다.
# 되돌리려면 SYSNAV_TERRAIN_CLEARANCE_M=0.75.
TERRAIN_CLEARANCE_M = float(os.getenv("SYSNAV_TERRAIN_CLEARANCE_M", "0.50"))

# waypointConverter 파라미터의 복제본 (waypoint_converter.launch, 2026-08-21 확인).
#   adjDisThre    = 5.0 -> TERRAIN_ADJ_DIS_M
#   searchDisThre = 5.0 -> TERRAIN_SEARCH_DIS_M
# 목표가 로봇에서 TERRAIN_ADJ_DIS_M **밖**이면 waypointConverter는 좌표를 손대지 않고
# 그대로 쓴다(`if (dis < adjDisThre && waypointTravAdj) waypointAdj = true;`). 즉 스냅은
# 5m 안에서만 일어나므로, 우리도 그 안에서만 미리 맞춰주면 된다.
TERRAIN_ADJ_DIS_M = 5.0
# waypointConverter가 후보로 보는 travArea 점의 범위(차량 기준). 이 밖의 점은 우리가
# 아무리 정확히 찍어도 후보에 안 들어간다.
TERRAIN_SEARCH_DIS_M = 5.0

# 요청 좌표를 commandable 지점으로 옮길 때 허용할 최대 거리. 이보다 멀면 "옮긴다"가
# 아니라 "다른 데로 보낸다"가 되므로 스냅하지 않고 실패로 보고한다.
#
# 왜 스냅이 통하는가(waypointConverter의 비용식으로 증명됨): 후보 비용은
#   cost(q) = |q - 우리목표| + vehicleDisWeight(0.5) * |q - 차량|
# 우리가 목표를 commandable 지점 p에 정확히 찍으면 cost(p) = 0.5*|p - 차량|이다.
# 다른 후보 q가 이를 이기려면 |q - 차량| > |p - 차량|이어야 하는데, 그러면
# cost(q) >= 0.5*|q - 차량| > cost(p)라 모순이다. 따라서 p가 반드시 선택된다 -
# 우리가 commandable 지점을 찍는 한 base autonomy는 그 좌표를 그대로 쓴다.
#
# 값 선택 근거 (2026-08-21 라이브 실측, 실제로 목표 삼을 만한 지점 120개에서 가장 가까운
# commandable 지점까지의 거리): 중앙값 0.68m, 90퍼센타일 1.29m, 최대 2.12m.
#   반경 0.75m -> 54% / 1.00m -> 73% / 1.50m -> 95% / 2.00m -> 99%
# 1.0m면 요청의 1/4이 "스냅 대상 없음"으로 빠지는데, 그때는 원본이 그대로 나가서 결국
# base autonomy가 1.5~2.5m 던져버린다. 우리가 1.5m 안에서 통제해 옮기는 편이 낫다.
# (근본 해법은 애초에 commandable 지점만 후보로 고르는 것 - Layer 2.)
TERRAIN_SNAP_MAX_M = 1.50

# ---------------------------------------------------------------------------
# Far throw - "받아줄 지점이 로봇 코앞에만 있는" 교착 탈출
#
# 실측(2026-08-21): commandable 지점이 로봇 반경 1.75m 안에만 존재하고 2.0m 밖에는
# 하나도 없는 상태가 반복됐다(terrainAnalysis의 noDecayDis=1.75/decayTime=1.0 때문).
# 그 상태에서는 어떤 목표를 줘도 스냅이 로봇 발밑으로 떨어져 로봇이 서 있는다.
#
# waypointConverter는 목표가 adjDisThre(5m) 밖이면 손대지 않고 그대로 넘긴다
# (`if (dis < adjDisThre && waypointTravAdj) waypointAdj = true;`). 그래서 목표 방향으로
# 5m를 넘겨 던지면 localPlanner가 자체 회피로 그쪽으로 몰고 간다. 도착은 못 하지만
# 로봇이 움직이기 시작하고, 움직이면 terrain이 갱신돼 다음 사이클엔 정상 목표가 잡힌다.
# 기본 OFF (2026-08-23). 전제가 틀린 것이 실측으로 드러났다.
#
# 전제: "adjDisThre(5m) 밖으로 던지면 waypointConverter가 목표를 안 건드린다."
# 실제: waypointConverter.cpp의 waypointAdj는 **매 pose 콜백(10Hz)마다 현재 거리로
#   다시 판정**되고, 한 번 켜지면 새 /way_point_with_heading이 올 때까지 안 꺼지는
#   래치다. 로봇이 1~2초만 움직여 5m 안으로 들어오면 보호가 사라지고, 그때는 목표가
#   벽 너머 무의미한 점이라 던지기 전보다 나쁘다.
#
# 실측(2026-08-23 06:31~06:32, 7.webm + sysnav_navigation_trace.txt):
#   06:31:44  로봇(0.65,-7.96)  요청 목표 1.55m 앞  -> 6.0m 밖으로 던짐 (3.9배)
#   06:31:59  로봇(-1.72,-6.72) 요청 목표 0.81m 앞  -> 6.0m 밖으로 던짐 (7.4배)
#   06:32:07  waypointConverter "Waypoint reached."   <- 도착한 적 없는데 도착 판정
#   06:32:09  actual이 로봇+0.50m(=waypointProjDis)   <- 제자리 회전 모드
# 0.81m 앞을 보러 가려다 6m 밖 벽 너머로 던져지고, 2초 뒤 로봇이 멈춰 돌기만 했다.
# 같은 6분간 FAR_THROW 35회 / SNAP_FAIL 1125회 / SNAP_NO_PROGRESS 516회.
#
# 되살리려면 SYSNAV_FAR_THROW_ENABLED=1. 다만 위 래치 문제를 먼저 풀어야 한다.
FAR_THROW_ENABLED = os.getenv("SYSNAV_FAR_THROW_ENABLED", "0") not in ("0", "false", "False")
# adjDisThre 위로 얹을 여유. 5.0 정확히 쓰면 부동소수 경계에서 갈린다.
FAR_THROW_MARGIN_M = 1.0
# 던지는 방향이 이만큼도 안 뚫려 있으면 던지지 않는다 - 벽을 코앞에 두고 던지면
# localPlanner가 갈 길을 못 찾아 제자리 회전만 한다.
FAR_THROW_MIN_CLEAR_M = 1.5

# --- 5m 밖 던지기 (교착 탈출) ---
#
# terrain_map은 noDecayDis=1.75m 롤링이라 로봇 반경 1.75m 밖에는 commandable 지점이
# **구조적으로** 존재하지 않는다(실측: 2.0m 밖 0점). 그래서 frontier를 아무리 잘 찾아도
# 그쪽으로 목표를 못 보내고, 안 움직이니 terrain도 안 자라는 교착에 빠진다.
#
# 탈출구는 waypointConverter의 이 줄이다:
#     if (dis < adjDisThre && waypointTravAdj) waypointAdj = true;
# 목표가 adjDisThre(5m) **밖**이면 좌표를 손대지 않고 그대로 localPlanner에 넘긴다.
# localPlanner는 자체 장애물 회피(경로 라이브러리 + terrain_map)로 그쪽으로 몰고 간다.
#
# 실측 (2026-08-23, 로봇 (3.58,2.63)에서 +x 방향):
#     1.5m 요청 -> 갈아끼워짐 -> 로봇 이동 0.00m
#     2.5m 요청 -> 갈아끼워짐 -> 로봇 이동 0.00m
#     8.0m 요청 -> 그대로 통과 -> 로봇 이동 2.57m
TERRAIN_SUPPORT_RADIUS_M = 0.35

# 접근 지점 탐색 범위. 물체에서 이만큼 떨어진 지점부터 훑는다. 하한이 1.0인 이유:
# 물체 자체가 obstacleArea에 들어가므로 클리어런스 0.75m + 물체 반지름을 감안하면
# 그보다 가까운 지점은 구조적으로 통과할 수 없다.
TERRAIN_APPROACH_MIN_M = 1.00
TERRAIN_APPROACH_MAX_M = 2.20
TERRAIN_APPROACH_STEP_M = 0.20
# 로봇->물체 방향 기준 각도 오프셋(도). 정면 접근이 막혀도 옆에서 되는 경우가 많다.
TERRAIN_APPROACH_ANGLES_DEG = (0.0, 20.0, -20.0, 40.0, -40.0, 60.0, -60.0)

# terrain 데이터가 이보다 오래되면 판정하지 않는다(보류).
TERRAIN_STALE_SEC = 3.0

# 확정된 목적지로 가는 주행(mission2 NAVIGATE_TARGET)의 경로 재계획 설정.
# 탐색(EXPLORATION_*)과 값을 공유하지 않고 따로 두는 이유: 탐색 쪽 값은 "이 후보는
# 포기하고 다음 후보로 넘어간다"는 판단용이라 짧아도 되지만(8초), 확정된 목적지는
# 포기할 대상이 아니라 끝까지 가야 하는 곳이라 훨씬 보수적이어야 한다.
#
# forbidden_mask("avoid the path between A and B")가 있을 때만 쓰는 hop 간격.
# 평소 목적지 주행은 hop 없이 목표 하나만 보낸다 - hop을 강제하면 localPlanner의
# pathCropByGoal 때문에 그쪽 시야가 "목표거리+0.5m"로 잘린다(start_target_navigation 주석).
#
# 값은 terrain_map 유효 반경(noDecayDis=1.75m) 안으로 잡는다. 그래야 각 hop이 발행
# 직전 검증(Layer 1)에서 실제로 판정 가능한 범위에 들어온다.
TARGET_PATH_WAYPOINT_SPACING_M = 1.5

# 목표에 가까워지지 못한 채 이 시간이 지나면 "base autonomy가 갈 수 있는 만큼 갔다"로
# 보고 도달 판정으로 넘어간다.
#
# 30 -> 10 -> 20 (2026-08-12).
#
# 10초로 줄였더니 목표 6개가 전부 이 폴백으로 끝났고(정상 도달 0건), 도달 인정 거리가
# 0.41~1.08m였다. 처음엔 "로봇이 물리적으로 더 못 붙는다"고 봤으나 RViz에서는 로봇이
# goal에 거의 올라타 있었다 - 즉 우리가 접근을 중간에 끊고 있었을 가능성이 크다.
#
# 판정 기준이 "역대 최단거리가 EXPLORATION_STUCK_PROGRESS_M(10cm) 이상 줄었나"인데,
# base autonomy는 목표에 가까워질수록 감속하므로 마지막 구간에서 10초 안에 10cm를
# 못 줄이는 일이 생긴다. 그러면 아직 다가가는 중인데도 정지로 판정해버린다.
# 시간을 늘려 그 조기 종료를 막는다.
TARGET_REPLAN_STUCK_TIMEOUT_SEC = 20.0

# 진전 없이(=hop 도착 없이) 연속으로 재계획한 횟수 상한. 넘으면 지금 지도로는 못 가는
# 것으로 보고 탐사 재계획으로 넘긴다.
#
# 주의: hop 도착으로 인한 재계획은 여기에 안 센다(그건 정상 진행이다). 1.5m마다 한 번씩
# 도착하므로 10m만 가도 6~7번인데 그걸 세면 정상 주행 중에 상한을 넘어버린다.
TARGET_REPLAN_MAX_COUNT = 3

# 정체(stalled) 폴백도 최종 target marker에서 0.5m를 넘으면 성공으로 인정하지 않는다.
# 이전 값 0.7m 때문에 로그처럼 0.53m에서도 SUCCESS가 발생했다. 기존 이름은 다른
# 호출부/설정과의 호환을 위해 유지하지만 성공 반경과 동일하게 고정한다.
TARGET_ARRIVAL_FALLBACK_MAX_M = TARGET_SUCCESS_DISTANCE_M

# 목표를 terrain 기준으로 재선택(choose_approach_point)하는 게 이 시간 동안 연속으로
# 실패하면 "지금 지도로는 이 물체에 접근할 수 없다"로 확정한다.
#
# 없을 때 무슨 일이 벌어졌나(2026-08-21 실측): 목표가 unsupported -> 재선택 시도 ->
# 49개 후보 전부 실패 -> "driving" 반환 -> 다음 tick에 또 같은 일. 그동안 로봇은
# 직전에 명령받은(이미 못 쓰게 된) 목표 쪽으로 계속 움직이니 진행도 감시가 리셋되어
# stalled 판정도 안 났다. trace에 RETARGET_FAIL이 642회 쌓였다.
#
# TARGET_REPLAN_STUCK_TIMEOUT_SEC(20초)보다 짧게 둔다 - 저쪽은 "가까워지지 못했나"를
# 보는 백스톱이고, 이쪽은 "애초에 목표를 찍을 데가 없다"는 확정적 판정이라 더 빨리
# 결론 내도 된다.
TARGET_RETARGET_GIVEUP_SEC = 5.0

# 재계획 최소 간격(초). hop이 막혔다는 판정은 control_loop(0.2초)마다 나올 수 있어서,
# 간격 제한이 없으면 "막힘 판정 -> 재계획 -> 같은 경로 -> 또 막힘 판정"이 0.2초마다
# 반복되어 1초도 안 되어 상한을 소진한다.
TARGET_REPLAN_MIN_INTERVAL_SEC = 1.0

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
# 같은 물체를 매 프레임 다시 Gemini에 묻지 않기 위한 캐시.
#
# 검증 대상은 이미 질문 관련 카테고리로 좁혀져 있다(detection_prompts = target +
# reference_objects). 그런데도 비용이 큰 이유는 **같은 박스를 반복해서 묻기** 때문이다.
# 실측(2026-08-23 06:31:24~06:32:09): 45초 중 Gemini 검출 재확인 7회에 약 30초.
# perception이 주행 중 PERCEPTION_WHILE_MOVING_INTERVAL_SEC(1.5초)마다 도는데, 로봇이
# 느리거나 멈춰 있으면 연속 프레임의 박스가 거의 같다.
#
# 캐시 키는 (카테고리, 양자화한 2D bbox)다. 같은 입력이면 같은 답이므로 품질 손실이
# 없다. 로봇이 빠르게 움직이면 박스가 어긋나 적중률이 떨어진다 - 그때는 원래대로
# 매번 묻는다.
DETECTION_VERIFICATION_CACHE_TTL_SEC = float(
    os.getenv("DETECTION_VERIFICATION_CACHE_TTL_SEC", "30.0")
)
# bbox 양자화 폭(px). 파노라마가 1920px이라 48px는 가로 2.5% 수준이다.
DETECTION_VERIFICATION_CACHE_BBOX_QUANT_PX = 48

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
# 지도에 반영할 point의 z 범위(센서 기준). 센서는 바닥 위 약 0.75m에 있으므로 바닥
# point(z 약 -0.75)는 여기서 제외된다. 이 범위의 point가 max_height에 기록되고,
# max_height는 RoomSegmenter가 "진짜 벽"을 가릴 때 쓴다(ROOM_WALL_MIN_HEIGHT_M=1.30).
# 그래서 이 상한을 낮추면 벽이 하나도 안 잡혀 방 분할이 통째로 깨진다.
MAP_OBSTACLE_Z_MIN_M = -0.40
MAP_OBSTACLE_Z_MAX_M = 1.60

# free/occupied(폐색) 판정에만 쓰는 더 얇은 z 범위. 위 범위와 **분리해야 한다**.
#
# 왜: 센서(0.75m)가 소파(0.8m)·테이블(0.75m) 같은 가구와 거의 같은 높이라, 위 범위
# (+1.60 = 절대 2.35m)를 그대로 쓰면 가구 **위로** 스쳐 지나간 거의 수평인 광선까지
# free 판정에 쓰인다. 그 광선은 가구 뒤 바닥 높이를 확인한 적이 없는데도 그 셀을
# free로 칠한다. 실측(6x5m 가구 4개 방, 스캔 1회): 시야가 실제 닿은 건 44%인데 73%를
# "봤다"고 판단했고, 가구 뒤에 생겨야 할 frontier가 통째로 사라져 탐색이 조기 종료됐다.
#
# 상한 값 선택 근거 (실측, 6x5m 방 + 소파 0.80m, 센서 0.75m, 스캔 3회):
#   +1.60(기존) : free 66.9m² frontier 58  - 소파 뒤 전부 FREE (폐색 실패)
#   +0.20       : free 66.9m² frontier 58  - 여전히 FREE. 소파 위 0.15m 틈으로 광선이 샘
#   +0.10       : free 63.1m² frontier 80  - 소파 뒤 UNKNOWN (정상)
#   +0.05       : 동일
# 소파 윗면이 센서 기준 +0.05m라, 상한이 그보다 충분히 낮아야 "가구를 넘어간 광선"이
# 배제된다. 0.05는 0.10과 결과가 같으면서 기하 변화에 여유가 더 있어 이쪽을 쓴다.
# 벽 검출(220셀)과 방 분할 결과는 기존과 동일해서 영향이 없다.
MAP_FREE_Z_MIN_M = -0.40
MAP_FREE_Z_MAX_M = 0.05

# 폐색 판정 시 방위각별로 "가장 가까운 장애물"만 남길 때 쓰는 각도 분해능(bin 수).
# 25m에서 셀(0.20m) 하나를 구분하려면 0.46도면 충분하므로 1440(0.25도)로 둔다.
MAP_AZIMUTH_BINS = 1440

MAP_UPDATE_INTERVAL_SEC = 0.35
OCC_UNKNOWN = -1
OCC_FREE = 0
OCC_OCCUPIED = 100
# 이 값 x2가 "통과 가능한 최소 문/통로 폭"이다 (양쪽에서 벽을 이만큼씩 부풀리므로).
# 원래 0.45 -> 0.30으로 낮췄는데도 문 통과가 잘 안 돼서 더 낮춤(최소 통과 폭 0.4m).
# 실제 로봇 폭을 몰라 정확한 값은 아니니, 나중에 실측되면 갱신할 것 - 로봇이 문틀에
# 부딪히면 다시 올려야 한다.
ROBOT_CLEARANCE_M = 0.10

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

# Watershed는 unknown/벽을 배경 marker로 잡기 때문에, 경계 주변에 "어느 방에도 속하지
# 않는 띠"가 남는다(실측 0.3~1.1m). frontier는 정의상 free/unknown 경계 = 바로 그 띠
# 한복판에 있어서, room mask를 그대로 쓰면 방 안 frontier가 0개가 되어 탐색이 "이 방
# 다 봤다"로 조기 종료된다(CoveragePlanner._active_room_mask 참고). 이 반경 안에 있고
# 다른 어떤 방보다 이 방에 더 가까운 미라벨 셀은 이 방에 속한 것으로 본다.
ROOM_FRONTIER_ASSIGN_RADIUS_M = float(
    os.getenv("SYSNAV_ROOM_FRONTIER_ASSIGN_RADIUS_M", "1.5")
)

# Watershed가 만든 room ridge 중 실제로 두 room을 잇는 짧은 구간을 doorway로
# 등록한다. ridge 주변 이 반경 안에서 정확히 두 room label이 보여야 하며, ridge의
# 길이가 MAX_WIDTH보다 길면 넓은 공간을 잘못 둘로 나눈 경계로 보고 버린다.
ROOM_DOOR_NEIGHBOR_RADIUS_M = float(os.getenv("SYSNAV_ROOM_DOOR_NEIGHBOR_RADIUS_M", "0.60"))
ROOM_DOOR_MIN_CELLS = int(os.getenv("SYSNAV_ROOM_DOOR_MIN_CELLS", "1"))
ROOM_DOOR_MAX_WIDTH_M = float(os.getenv("SYSNAV_ROOM_DOOR_MAX_WIDTH_M", "2.40"))

# 방 identity는 centroid만으로 이어붙이면 인접 방의 크기가 바뀔 때 ID가 서로
# 뒤바뀔 수 있다. 직전 mask와의 IoU가 이 값 이상이면 overlap을 우선 사용하고,
# overlap이 부족한 신규/초기 room만 centroid 반경으로 폴백한다.
ROOM_REGISTRY_MIN_IOU = float(os.getenv("SYSNAV_ROOM_REGISTRY_MIN_IOU", "0.12"))

# Room Node identity persistence (rooms/room_registry.py). RoomSegmenter.segment()는
# 매 mapping cycle마다 watershed를 처음부터 다시 돌려 room_id를 1..N으로 새로 매기므로
# (사이클 간 정체성이 없음), RoomRegistry가 centroid 근접 매칭으로 "같은 물리적 방"을
# 안정적인 room_id에 이어붙인다. 이 반경(m)보다 centroid가 더 멀어지면 다른 방으로
# 취급해 새 id를 발급한다.
ROOM_REGISTRY_MATCH_RADIUS_M = 2.0

# 빈 in-room route가 한 번 나온 것만으로 room 완료를 선언하면 stochastic sampling의
# 일시 실패에도 다음 방으로 넘어간다. 연속 N번 비어야 covered로 확정한다.
ROOM_COMPLETION_CONFIRMATIONS = int(os.getenv("SYSNAV_ROOM_COMPLETION_CONFIRMATIONS", "2"))
# 방 안에 남은 surface cell이 이 값 이하이면 다음 방 선택을 worker에서 미리 계산한다.
# VLM이 현재 방을 더 유망하다고 고르면 기존 in-room route를 그대로 계속 수행한다.
ROOM_EARLY_STOP_SURFACE_CELLS = int(os.getenv("SYSNAV_ROOM_EARLY_STOP_SURFACE_CELLS", "18"))
ROOM_EARLY_STOP_ENABLED = os.getenv("SYSNAV_ROOM_EARLY_STOP_ENABLED", "1") not in ("0", "false", "False")
# doorway를 지난 뒤 다음 room 쪽으로 이 거리만큼 들어간 점을 room-entry waypoint로
# 사용한다. 문 중심에만 도착하면 room label 0 경계에 계속 남는 문제를 막는다.
ROOM_ENTRY_DEPTH_M = float(os.getenv("SYSNAV_ROOM_ENTRY_DEPTH_M", "1.20"))

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
# 0이 되어버리는 문제가 있었다.
#
# 0.15였을 때 이 의도가 실제로는 전혀 반영되지 않았다: 셀 크기가 0.20m인데 dilate 반경을
# `max(1, round(margin / 0.20))`으로 잡아서 0.15든 0.20이든 똑같이 1셀이 됐고, 결국 LOS
# 마스크가 ROBOT_CLEARANCE_M(0.20 -> 1셀) 팽창 마스크와 완전히 동일해졌다. 그래서 frontier
# 0.4m 앞에 선 candidate조차 "안 보인다"로 떨어져(실측 anchor 7개 중 5개) 탐색이
# no_candidate_had_any_visible_uncovered_surface_point로 죽었다.
#
# 가시성은 몸체 통과 여부가 아니라 광선 문제다 - 벽에서 0.2m 옆을 스치는 LiDAR 광선은
# 실제로 그 너머를 본다. 0으로 두면 팽창 없이 진짜 장애물 셀만 막는다(_bresenham이
# 셀 단위로 훑으므로 곧은 벽은 그대로 차단된다). 대각선 코너를 스쳐 지나가는 과대평가가
# 남지만, "가서 봤더니 생각보다 덜 보였다" 정도라 탐색이 통째로 멈추는 것보다 훨씬 낫다.
FRONTIER_LOS_WALL_MARGIN_M = float(os.getenv("SYSNAV_FRONTIER_LOS_WALL_MARGIN_M", "0.0"))

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
# 탐색 경로를 hop으로 자르는 간격. cross_room_navigator(방 사이 이동 경로)와
# coverage_planner의 leg waypoint 생성이 쓴다.
#
# terrain_map은 decayTime=1.0s / noDecayDis=1.75m 롤링이라 로봇에서 1.75m 밖은 1초면
# 사라진다. hop 간격이 그보다 크면 다음 목표가 "travArea 점이 없는" 구간에 떨어져
# waypointConverter가 로봇 발밑으로 덤프한다. 그래서 유효 반경 안으로 줄여둔다
# (3.0 -> 1.5, 2026-08-22).
EXPLORATION_PATH_WAYPOINT_SPACING_M = 1.5

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
# "A under B" / "A above B"의 수평 정렬 판정 여유. 두 물체의 XY bbox를 이만큼 부풀려
# 겹치면 "위/아래"로 본다.
#
# 왜 필요한가 (2026-08-23, GT 대조로 발견): under/above가 **높이차만** 보고 수평
# 거리를 전혀 안 봤다. 그래서 방 반대편 물체끼리 "아래에 있다"가 성립했다:
#   cabinet#1(2.19,-0.79) --under--> picture#8(-2.62,-6.55)  conf=0.937  수평 7.51m
# 이 엉터리 edge가 "vase on cabinet below picture"의 체인을 완성해서, GT 정답
# (vase#23)이 아니라 방 반대편 화병을 가리키게 만들었다.
#
# 0.60m인 이유: 캐비닛은 넓고(GT livingroom_1 cabinet#2는 1.75m) 그 위 그림은 좁아서
# (picture#88은 0.90m) 중심이 어긋나고, LiDAR extent도 부정확하다. GT 정답 쌍의 실제
# 수평 중심거리는 0.21m, 우리 검출로는 0.49m였다. 오검출 쌍은 4.9m 이상이라 넉넉히 갈린다.
SCENE_GRAPH_VERTICAL_HORIZONTAL_MARGIN_M = float(
    os.getenv("SYSNAV_VERTICAL_HORIZONTAL_MARGIN_M", "0.60")
)
# under/above로 인정할 최대 높이차. 이보다 벌어지면 "위/아래"라기보다 그냥 다른 층이다.
# GT 정답 쌍(cabinet#2 -> picture#88)의 높이차가 1.19m라 그보다 여유를 둔다.
SCENE_GRAPH_VERTICAL_MAX_GAP_M = float(os.getenv("SYSNAV_VERTICAL_MAX_GAP_M", "2.00"))
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
