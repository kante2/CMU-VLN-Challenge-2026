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
# 하나로 목표를 갈아끼운다(waypointConverter.cpp의 waypointAdj 분기). 그래서 우리가
# /terrain_map으로 먼저 같은 판정을 해서 옮겨 보낸다. 원래 찍으려던 좌표와 실제로 발행한
# 좌표는 Pose2D(header 없음)라 RViz에 못 띄우므로, 둘을 Marker로 한 번 더 내보내서
# "우리 목표가 발행 전에 얼마나 옮겨졌나"를 눈으로 확인한다.
TOPIC_REQUESTED_WAYPOINT = "/sysnav/requested_waypoint"
# 우리 planner가 실제로 보고 있는 지도/경로를 RViz에서 base autonomy 것과 겹쳐 보기
# 위한 발행 전용 토픽들. PNG(exploration_debug_latest.png)로도 보지만, 그건 별도 창이라
# /registered_scan, /terrain_map과 같은 좌표계에 겹쳐볼 수가 없다.
# occupancy grid의 셀 값(OCC_UNKNOWN/-1, OCC_FREE/0, OCC_OCCUPIED/100)은 이미
# nav_msgs/OccupancyGrid 규약과 같아서 그대로 내보내면 된다.
TOPIC_SYSNAV_OCCUPANCY = "/sysnav/occupancy_grid"
TOPIC_SYSNAV_PATH = "/sysnav/planned_path"
TOPIC_SYSNAV_FRONTIER = "/sysnav/frontier"
# 지도 발행 주기(초). 300x300 int8 = 90KB라 1Hz면 부담이 없다.
MAP_PUBLISH_INTERVAL_SEC = 1.0
# 원래 찍으려던 좌표에서 이만큼 넘게 옮겨서 발행하면 "밀려났다"고 보고 로그/추적에 남긴다.
#
# 이력: 예전엔 base autonomy가 발행하는 /way_point를 구독해서 "실제로 얼마나 밀렸나"를
# 사후에 쟀다. 그런데 README의 System Outputs 표(테스트 때 사용 가능한 토픽 6개)에
# /way_point는 없고, 표 아래에 "these are the only ones allowed to be used during test
# time"이라고 명시돼 있어 규정 위반이었다. 지금은 발행 **전에** 우리가 /terrain_map으로
# 직접 계산한 스냅 거리(goal_publisher.last_snap_distance_m)를 같은 용도로 쓴다 -
# /terrain_map은 표에 있는 허용 토픽이고, waypointConverter가 판정에 쓰는 것과 같은
# 데이터라 예측이 사후 측정과 사실상 같은 값을 준다(navigation/terrain_monitor.py).
WAYPOINT_DISPLACEMENT_WARN_M = 0.30
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
# Mission 2는 전체 제한 전에 반드시 marker를 제출할 시간을 확보한다.
MISSION2_EXPLORATION_TIME_LIMIT_SEC = float(
    os.getenv("MISSION2_EXPLORATION_TIME_LIMIT_SEC", "420.0")
)
# Mission 1도 같은 이유로 집계 시각을 보장한다. Mission 1은 "관계의 참조 물체를 다
# 볼 때까지 집계하지 않는다"는 게이트가 있어서(missions/mission1_pipe.py), 참조 물체가
# 끝내 안 보이면 답을 영영 못 낼 수 있다. 무응답은 0점이라 최악이므로, 이 시간이 지나면
# 가진 근거만으로 집계해서 발행한다.
MISSION1_EXPLORATION_TIME_LIMIT_SEC = float(
    os.getenv("MISSION1_EXPLORATION_TIME_LIMIT_SEC", "420.0")
)
# 참조 물체를 기다리는 동안 같은 로그를 매 사이클 찍지 않기 위한 간격(초).
MISSION1_MISSING_LOG_INTERVAL_SEC = 10.0
# 관계 체인이 확정된 뒤, 집계 대상 개수가 이 횟수만큼 연속으로 변하지 않으면 탐색을
# 끝내고 답을 낸다.
#
# 왜 "확정되면 즉시"가 아닌가: "table with a vase" 하나를 검증한 순간에 세면, 그 테이블
# 주위를 아직 반밖에 안 돌아서 의자가 3개만 잡혀 있을 수 있다(실측 2026-08-24: GT 8개인데
# 대시보드에 3개). 관계가 확정됐다는 것과 대상을 다 봤다는 것은 다른 얘기다.
# 반대로 개수가 여러 관측에 걸쳐 그대로면 더 볼 게 없다는 뜻이므로 거기서 끊는다.
MISSION1_SETTLED_STABLE_OBSERVATIONS = int(
    os.getenv("MISSION1_SETTLED_STABLE_OBSERVATIONS", "3")
)
PERCEPTION_WHILE_MOVING_INTERVAL_SEC = 1.50

# ---------------------------------------------------------------------------
# 같은 질문의 반복 발행 처리
#
# 대회 채점 환경은 /challenge_question을 한 번만 쏘지 않고 **계속 발행**한다
# (`ros2 topic pub`에서 --once를 뺀 형태 = 1Hz 반복). 예전 question_callback은 들어오는
# 메시지마다 무조건 새 task로 받아서, 매 초 Gemini 파싱을 다시 돌리고 task_id를 올리고
# object_memory/scene_graph/coverage_planner를 통째로 초기화했다. 그러면 로봇은 영원히
# 첫 관측 상태를 벗어나지 못한다.
#
# 그래서 "지금 처리 중인 문장과 같은 문장"은 무시한다. 문장이 실제로 바뀌면 그때만
# 새 task로 받는다.
# ---------------------------------------------------------------------------
# 파싱에 실패한 문장은 계속 막아두면 영영 못 받으므로 이 간격 뒤에는 다시 시도한다.
# (1Hz 그대로 재시도하면 Gemini 호출로 시간 예산을 태운다.)
QUESTION_REPARSE_RETRY_SEC = float(os.getenv("QUESTION_REPARSE_RETRY_SEC", "30.0"))
# 중복 문장을 몇 개 버렸는지 알려주는 로그 간격(초). 매번 찍으면 로그가 도배된다.
QUESTION_DUPLICATE_LOG_INTERVAL_SEC = 30.0
SENSOR_SYNC_TOLERANCE_SEC = 0.30
# perception job을 못 던지고 대기하는 동안 그 이유를 다시 찍기까지의 간격(초).
# 진단 전용 - sysnav_node._log_sensor_wait() 참고.
SENSOR_WAIT_LOG_INTERVAL_SEC = 5.0

# 카메라 파이프라인이 밀려서 image stamp가 scan보다 한참 뒤처질 때의 구제책.
#
# 실측(2026-08-24): /camera/image는 압축 스트림을 sim_image_repub이 3.7MB 생이미지로
# 다시 뿌리는 구조라, 시뮬레이터가 느려지면 image stamp가 scan 버퍼 전체보다 오래돼서
# ±0.30s 동기화가 **영원히** 실패한다. 그러면 perception job을 한 번도 못 던지고
# OBSERVE에서 무한 대기한다(로봇이 한 발짝도 안 움직임).
#
# 그렇다고 아무 때나 짝지으면 안 된다 - 이미지와 스캔이 다른 시점이면 LiDAR grounding이
# 엉뚱한 3D 좌표를 만든다. 하지만 **로봇이 그 사이에 사실상 안 움직였다면** 두 센서가
# 보는 장면이 같으므로 짝지어도 안전하다. 그래서 "얼마나 오래됐나"가 아니라
# "그 사이 얼마나 움직였나"로 허용 여부를 정한다.
SENSOR_SYNC_FALLBACK_MAX_SEC = float(os.getenv("SENSOR_SYNC_FALLBACK_MAX_SEC", "5.0"))
SENSOR_SYNC_FALLBACK_MAX_MOVE_M = 0.15
SENSOR_SYNC_FALLBACK_MAX_YAW_RAD = 0.15
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
# Mission 3는 물체/경유점 주변의 실제 trajectory 달성이 채점 대상이다. 가구 가까이의
# subgoal은 base autonomy의 장애물 안전거리 때문에 0.5m 안까지 못 들어가는 경우가
# 있으므로, Mission 3의 step 완료 판정에만 1m 반경을 사용한다. Mission 1/2와 탐사
# waypoint의 도착 반경은 위 값을 그대로 쓴다.
MISSION3_TARGET_SUCCESS_DISTANCE_M = 1.0
# Mission 3의 "go to/near <object>" subgoal은 선택된 물체 앞에 있어야 한다. terrain
# clearance를 만족시키려고 2m 이상 떨어진 점을 고르면 관계 대상은 맞아도 instruction을
# 수행했다고 보기 어려우므로, 모든 Mission 3 물체 접근점과 실제-waypoint 동기화에
# 동일한 최대 거리를 적용한다.
MISSION3_OBJECT_APPROACH_MAX_M = 0.9
# "take the path between A and B"의 성공 조건은 두 물체 사이를 실제로 **가로지르는**
# 것이다(중점 근처에 도달하는 것과 다르다 - 중점 옆에 서 있다가 돌아가도 반경 판정은
# 통과해버린다). 그래서 A-B 선분을 게이트로 두고 로봇 궤적이 그 선분과 교차했는지 본다
# (missions/path_gate.py). 물체 바로 옆을 스치듯 통과하는 경우까지 잡으려고 선분을 양쪽
# 끝에서 이만큼 연장한 뒤 교차를 판정한다.
MISSION3_GATE_EXTENSION_M = 0.3

# "take the path between A and B"에서 A 후보 x B 후보를 **짝으로** 평가할 때 쓰는 값들.
#
# 왜: 예전엔 두 참조를 각각 독립적으로 "로봇에 가장 가까운 것" argmin으로 골랐다
# (detection confidence를 아예 안 읽었고, 두 물체가 실제로 통과 가능한 게이트를 이루는지도
# 안 봤다). 실측 2026-08-25: "take the path between the sofa and the round tables"에서
# 신뢰도 0.58짜리 오검출(벽 옆 화분받침을 table로 검출)이 로봇에 더 가깝다는 이유만으로
# 신뢰도 0.85짜리 진짜 원형 테이블을 이겼다. 게다가 로봇 pose에 의존해서 unreachable
# 재시도마다 목표가 떠돌았다. 이제 쌍 단위로 채점하고 pose는 쓰지 않는다.
#
# GAP: 두 물체 중심 사이 거리의 허용 범위. 하한은 "로봇이 들어갈 틈도 안 되는" 짝을,
# 상한은 "같은 방에 있을 뿐 통로가 아닌" 짝을 걸러낸다(로봇 거리 항을 없앤 대신 이
# 상한이 국소성을 담당한다).
MISSION3_BETWEEN_MIN_GAP_M = float(os.getenv("SYSNAV_MISSION3_BETWEEN_MIN_GAP_M", "0.4"))
MISSION3_BETWEEN_MAX_GAP_M = float(os.getenv("SYSNAV_MISSION3_BETWEEN_MAX_GAP_M", "6.0"))
# 게이트 통과 판정용 장애물 팽창 반경. ROBOT_CLEARANCE_M(현재 0.0)을 그대로 쓰면 1셀
# (0.2m)밖에 안 부풀어서 "로봇이 못 지나갈 만큼 좁은 틈"을 통과 가능으로 본다.
MISSION3_BETWEEN_GATE_CLEARANCE_M = float(os.getenv("SYSNAV_MISSION3_BETWEEN_GATE_CLEARANCE_M", "0.30"))
# A-B 선분 위 셀 중 통과 가능해야 하는 최소 비율.
MISSION3_BETWEEN_MIN_CLEAR_FRACTION = float(os.getenv("SYSNAV_MISSION3_BETWEEN_MIN_CLEAR_FRACTION", "0.6"))
# 선분을 물체 몸통 밖에서 시작시키기 위해 양 끝에서 잘라낼 반경의 상한. 물체 중심 셀은
# 당연히 occupied라 안 자르면 모든 짝이 똑같이 감점돼 통과가능성 판정이 무의미해진다.
MISSION3_BETWEEN_OBJECT_RADIUS_MAX_M = float(os.getenv("SYSNAV_MISSION3_BETWEEN_OBJECT_RADIUS_MAX_M", "1.0"))
# 쌍 점수 가중치. confidence를 지배항으로 둔다 - 통과 가능한 게이트들 중에서는
# "신뢰도 높은 쪽이 이긴다"가 되도록. 통과가능성은 점수 항이 아니라 필터(tier)다.
MISSION3_BETWEEN_CONFIDENCE_WEIGHT = float(os.getenv("SYSNAV_MISSION3_BETWEEN_CONFIDENCE_WEIGHT", "0.65"))
MISSION3_BETWEEN_GAP_WEIGHT = float(os.getenv("SYSNAV_MISSION3_BETWEEN_GAP_WEIGHT", "0.35"))
# 쌍 채점 결과를 로그에 몇 줄까지 찍을 것인가(디버깅용).
MISSION3_BETWEEN_PAIR_LOG_TOP_N = int(os.getenv("SYSNAV_MISSION3_BETWEEN_PAIR_LOG_TOP_N", "5"))

# 확정된 subgoal을 몇 번 연속 "도달 불가"로 받고도 계속 재발행할 것인가.
#
# mission3는 한 번 marker까지 만든 subgoal을 탐사 목표로 덮어쓰지 않는다(채점이 subgoal
# 순서와 실제 궤적을 보므로 RViz의 goalN과 로봇이 향하는 곳이 달라지면 안 된다). 그런데
# 그 정책에 상한이 없어서, 목적지가 base autonomy에게 명령 불가한 자리이면 "재발행 ->
# 발행 거부 -> unreachable -> 재발행"을 영원히 돌았다(실측 2026-08-24: 변기 앞에서 0.5초
# 주기로 무한 정지). 이 횟수를 넘으면 그 step은 "여기까지가 최선"으로 인정하고 다음
# step으로 넘어간다 - 한 step에 갇혀 남은 step을 통째로 버리는 것보다 부분점수가 낫다.
# 0.5초 주기 x 20 = 약 10초.
MISSION3_SUBGOAL_MAX_RETRIES = int(os.getenv("SYSNAV_MISSION3_SUBGOAL_MAX_RETRIES", "20"))

# Mission 2에서 최종 target까지의 주행이 "도달 불가"로 끝났을 때 몇 번까지 다시
# 시도할 것인가.
#
# 실측 2026-08-24: toilet을 옳게 고르고도 접근점이 base autonomy에게 명령 불가라
# TARGET SELECTED 7ms 뒤에 바로 주행을 접었다(로그: "target goal (2.24, 1.93) is not
# commandable" -> "ANSWER SUBMITTED (navigation incomplete)"). 답 자체는 이미 냈으니
# FAILED는 아니지만, 물체에 가까이 가서 extent_3d를 정밀하게 만드는 이득(그게 곧
# bbox 겹침 점수다)을 통째로 못 받았다. 남은 시간이 6분이었는데도 그랬다.
#
# 재시도 1회 = (a) 접근 상한을 푼 재선정, 실패하면 (b) 탐사로 돌아가 지도를 넓히고
# 다시 선택. mission3와 달리 매 tick 도는 게 아니라 "주행이 끝난 뒤"에만 세므로
# 횟수는 작게 잡는다 - (b)는 selection job(LLM 호출 가능)을 한 번씩 더 쓴다.
MISSION2_TARGET_MAX_RETRIES = int(os.getenv("SYSNAV_MISSION2_TARGET_MAX_RETRIES", "3"))

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
# waypointConverter의 obstacleDisThre와 반드시 같은 값이어야 한다. 측정 목적으로
# 0.50m까지 낮췄을 때 요청 (2.13,-3.19)이 실제 (1.82,-4.48)로 1.32m 밀렸고, 로봇은
# 실제 목표에 도착했지만 Mission 3 marker까지 1.4m가 남아 무한 재발행했다. 따라서
# 기본값을 base autonomy와 동일한 0.75m로 유지한다.
TERRAIN_CLEARANCE_M = float(os.getenv("SYSNAV_TERRAIN_CLEARANCE_M", "0.75"))
# clearance가 이 값 안쪽으로 **아깝게** 모자랄 때는 "발행 불가"로 보지 않고 원본을
# 그대로 내보낸다 (navigation/goal_publisher.py의 near-miss PASSTHRU).
#
# 왜: TERRAIN_CLEARANCE_M은 waypointConverter의 obstacleDisThre 복제본이지만, 우리는
# /terrain_map을 우리 방식으로 다시 계산하므로 저쪽과 cm 단위로 일치하지 않는다.
# 실측 2026-08-25: 후보 1164개의 최선이 0.74m라 전멸 -> 아무것도 발행 못 함 -> 로봇이
# 한 발짝도 못 움직임 -> 지도가 그대로라 다음 사이클에 완전히 같은 route -> 0:40에
# FAILED(9분 20초 남음). 안 보내면 확실히 0m지만, 보내면 저쪽이 자기 기준으로 판단해서
# 최소한 움직인다. 크게 모자란 경우(0.20m 등)는 여전히 발행하지 않는다.
TERRAIN_CLEARANCE_NEAR_MISS_M = float(
    os.getenv("SYSNAV_TERRAIN_CLEARANCE_NEAR_MISS_M", "0.10")
)

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
# 1.5 -> 1.0 (2026-08-24). 1.5를 고른 근거는 "여기서 못 찾으면 원본이 그대로 나가서
# base autonomy가 1.5~2.5m 던져버린다"였는데, PREDICT_CONVERTER_FALLBACK이 생기면서
# 그 전제가 사라졌다 - 이제 못 찾으면 원본 발행이 아니라 converter 비용식 argmin으로
# 넘어간다(아래 PREDICT_CONVERTER_FALLBACK_ENABLED 주석).
#
# 두 규칙의 차이가 곧 이 값을 줄이는 이유다:
#   - 여기(nearest_commandable)는 |q - 우리목표|만 본다. 비용식의 0.5*|q - 차량| 항을
#     빼먹은 근사라, 목표가 멀수록 converter의 실제 선택과 어긋난다. 게다가 전진 가드가
#     SNAP_NO_PROGRESS 하나뿐이라 목표 옆/뒤 지점으로도 1.5m까지 옮길 수 있었다
#     (실측 PUSHED: 요청 (-3.07,0.01) -> 실제 (-0.63,-0.02), 반대 방향 2.44m).
#   - 예측 폴백은 비용식을 그대로 쓰고 gain >= PREDICT_CONVERTER_MIN_GAIN_M도 요구한다.
# 즉 근사 규칙의 관할을 좁혀 가드 있는 규칙으로 더 많이 보내는 것이 이 변경의 목적이다.
# 0.75m까지 더 줄이면 위 실측 분포상 54%만 스냅되어 폴백 의존이 과해지므로 1.0에서 멈춘다.
TERRAIN_SNAP_MAX_M = float(os.getenv("SYSNAV_TERRAIN_SNAP_MAX_M", "1.00"))

# waypointConverter의 후보 비용식 cost(c) = dist(c, our_goal) + 0.5 * dist(c, robot)에서
# 로봇 거리 항의 가중치. 저쪽 소스 값(0.5)을 그대로 복제한 것이라 바꿀 일은 없고,
# 예측식(terrain_monitor.predict_converter_choice)이 무엇을 흉내내는지 드러내려고 상수로 뺐다.
TERRAIN_CONVERTER_ROBOT_COST_WEIGHT = 0.5

# "목표 근처에 통과 지점이 없다"고 발행을 거부하는 대신, waypointConverter가 고를 점을
# 우리가 예측해서 그 점으로 보낼 것인가.
#
# 왜: 벽에 붙은 가구 앞은 벽과 물체가 양쪽에서 clearance를 깎아 0.75m 기준을 만족하는
# 점이 구조적으로 없다(실측 2026-08-24 mission3 step 1: 목표 1.5m 안 travArea 349점,
# 최대 clearance 0.67m). 지금은 그럴 때 아무 명령도 안 보내서 로봇이 한 발짝도 못 가고
# step/target을 통째로 포기했다. 저쪽은 후보가 있으면 어차피 argmin을 고르므로, 그 점이
# **전진이 되는 경우에 한해** 우리가 먼저 그 점을 찍어주는 편이 항상 낫다.
#
# 끄려면 SYSNAV_PREDICT_CONVERTER_FALLBACK=0. 끄면 예전처럼 SNAP_FAIL로 거부한다.
PREDICT_CONVERTER_FALLBACK_ENABLED = os.getenv(
    "SYSNAV_PREDICT_CONVERTER_FALLBACK", "1"
) not in ("0", "false", "False")
# 예측 지점을 채택할 최소 전진량: "로봇->목표 거리"가 이만큼은 줄어야 한다. 이 값이
# 없으면 목표 옆이나 뒤쪽 점을 골라 제자리에서 왕복한다(SNAP_NO_PROGRESS와 같은 문제).
PREDICT_CONVERTER_MIN_GAIN_M = 0.30

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

# 위 링 샘플링이 전부 실패했을 때, commandable set을 **직접** 훑어 물체에 가장 가까운
# 지점을 고르는 폴백의 상한(물체로부터의 거리).
#
# 왜 필요한가 (실측 2026-08-24, probe_waypoint_push.py):
#   이 씬은 travArea 1066점 중 clearance >= 0.75m를 통과하는 점이 **7.1%(76점)뿐**이다.
#   그런데 Mission 3의 링 샘플링은 MISSION3_OBJECT_APPROACH_MAX_M(0.9m) 때문에 링 하나 x
#   7각도 = 후보 7개만 본다. 76/1066 확률에서 링 7점이 걸릴 리가 없어서 거의 항상 실패하고,
#   terrain을 아예 안 보는 고정 standoff로 폴백했다. 그 좌표(clearance 0.42m)를 발행하려다
#   막혀서 mission3가 같은 subgoal을 무한 재발행했다(변기 앞에서 로봇이 영원히 정지).
#   링 반경/각도 운에 맡기는 대신 통과 지점 집합을 직접 훑으면 "base autonomy가 허용하는
#   가장 가까운 standoff"가 결정론적으로 나온다.
#
# 상한을 두는 이유: 무제한이면 벽 너머 다른 방의 점이 기하학적으로 더 가깝다는 이유로
# 뽑힐 수 있다(travArea는 연결성을 모른다). 이 안에서 못 찾으면 못 가는 게 맞다.
TERRAIN_APPROACH_FALLBACK_MAX_M = float(
    os.getenv("SYSNAV_TERRAIN_APPROACH_FALLBACK_MAX_M", "3.0")
)

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

# YOLO-World의 open-vocabulary 검출을 COCO 사전학습 YOLO12s로 보강한다. 질문의
# prompt 중 COCO class로 정확히 매핑되는 것이 있을 때만 두 번째 모델을 실행한다.
YOLO12_COCO_ENABLED = os.getenv("YOLO12_COCO_ENABLED", "1") not in ("0", "false", "False")
YOLO12_WEIGHTS = os.getenv("YOLO12_WEIGHTS", "yolo12s.pt")
YOLO12_CONFIDENCE = float(os.getenv("YOLO12_CONFIDENCE", "0.05"))
YOLO12_BOOK_CONFIDENCE = float(os.getenv("YOLO12_BOOK_CONFIDENCE", "0.05"))
YOLO12_DEFAULT_CLASS_CONFIDENCE = float(
    os.getenv("YOLO12_DEFAULT_CLASS_CONFIDENCE", "0.20")
)
YOLO_ENSEMBLE_MERGE_IOU = float(os.getenv("YOLO_ENSEMBLE_MERGE_IOU", "0.50"))

SAM2_CHECKPOINT = os.getenv("SAM2_CHECKPOINT", "")
SAM2_MODEL_CFG = os.getenv("SAM2_MODEL_CFG", "configs/sam2.1/sam2.1_hiera_t.yaml")
SAM2_DEVICE = os.getenv("SAM2_DEVICE", "cuda")
SAM2_MIN_MASK_AREA_PX = 80

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# Numerical 개수 세기(reasoning/vlm_counter.py) 전용 모델. 다른 호출과 분리한 이유:
# 이 호출은 질문당 1회뿐이라 상위 모델을 써도 예산에 거의 영향이 없는 반면, 파노라마
# 한 장에서 제약을 만족하는 물체를 빠짐없이 세는 건 이 시스템에서 가장 어려운 추론이다.
# 반대로 GEMINI_MODEL을 통째로 올리면 perception/detection_verifier.py까지 느려져서
# 탐사 시간을 깎아먹는다. 이 모델 호출이 실패하면(모델명 오타, 권한/쿼터 없음 등)
# GEMINI_MODEL로 한 번 더 시도한다.
#
# 기본값이 flash인 이유(kante/fix_mission_1 실측): loft viewpoint_000010
# ("How many black pillows are on the sofa?") 5회 투표에서 flash [2,3,3,3,3] -> 3, 6.6초 /
# pro [2,3,2,2,2] -> 2, 66.2초. 정확도 이득이 확인되지 않았고, pro는 부하가 걸리면
# 응답이 극단적으로 느려져(95.9초) MISSION1_FINALIZE_COUNT가 몇 분간 멈춘 적이 있다.
GEMINI_COUNTING_MODEL = os.getenv("GEMINI_COUNTING_MODEL", "gemini-3.5-flash")
# VLM 개수 세기 호출 하나의 상한(초). 없으면 모델이 느려졌을 때 finalize가 무한정
# 매달린다. 끊기면 폴백 모델 -> 그것도 실패하면 기하 기반 개수(fail-quiet).
GEMINI_COUNTING_TIMEOUT_SEC = float(os.getenv("GEMINI_COUNTING_TIMEOUT_SEC", "45"))

# Numerical 미션의 개수를, 대상 물체를 가장 많이 담은 viewpoint 파노라마 한 장으로
# VLM에게 직접 세게 한다(reasoning/vlm_counter.py). object_memory 기반 집계는 **탐지
# 재현율에 갇힌다** - 실측(home_building_1)에서 pillow가 GT 18개인데 메모리엔 7개만
# 남았다. 베개 4개 중 2개만 탐지되면 병합·필터를 아무리 손봐도 답은 영원히 2다.
# 뷰를 한 장으로 확정하는 이유는 여러 뷰의 개수를 합치면 같은 물체가 중복 계산되기
# 때문이다. 실패하면 기존 기하 기반 개수를 그대로 쓴다 - 개수 미션은 0/1 채점이라
# "답을 못 냄"이 최악이다.
NUMERICAL_VLM_COUNT_ENABLED = os.getenv(
    "SYSNAV_NUMERICAL_VLM_COUNT", "1"
) not in ("0", "false", "False")
# 같은 이미지를 이 횟수만큼 병렬로 물어보고 최빈 개수를 채택한다(self-consistency).
# temperature=0.0인데도 API 응답이 결정적이지 않아 1회 호출은 사실상 동전던지기다 -
# 실측: temp 0.0에서 4회가 [3,3,3,2], temp 0.7에서 [2,2,2,3]. 0/1 채점에서 이 흔들림은
# 그대로 점수 손실이라 다수결로 고정한다. 병렬이라 지연은 1회와 비슷하고(4회 ~10s),
# 질문당 한 번뿐이라 예산 영향도 거의 없다. 1로 두면 단일 호출로 동작한다.
NUMERICAL_VLM_COUNT_SAMPLES = int(os.getenv("SYSNAV_NUMERICAL_VLM_COUNT_SAMPLES", "5"))
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
# 검증 대상은 이미 질문 관련 카테고리로 좁혀져 있다(mission3는 현재 step까지). 그런데도
# 비용이 큰 이유는 **같은 물체를 반복해서 묻기** 때문이다. perception이 주행 중
# PERCEPTION_WHILE_MOVING_INTERVAL_SEC(1.5초)마다 도는데, 같은 물체가 계속 보인다.
# 실측(2026-08-23): 45초 중 Gemini 재확인 7회에 약 30초.
#
# 캐시 키는 (카테고리, 양자화한 **3D 위치**)다. 예전엔 2D bbox였는데, 파노라마에서
# 박스는 로봇이 조금만 움직여도 양자화 폭을 넘게 밀려서 사실상 매번 miss였다
# (실측 2026-08-25: 33초 동안 같은 picture를 5번 질의, TTL 30초 안인데도 전부 재질의).
# 3D 위치는 로봇 위치와 무관하므로 물체당 한 번으로 수렴한다. 같은 물체면 같은 답이라
# 품질 손실도 없다.
DETECTION_VERIFICATION_CACHE_TTL_SEC = float(
    os.getenv("DETECTION_VERIFICATION_CACHE_TTL_SEC", "30.0")
)
# 캐시 키의 3D 위치 양자화 폭(m). 검증은 LiDAR grounding **뒤에** 돌기 때문에 map
# 프레임 좌표를 키로 쓸 수 있고, 그래서 로봇이 움직여도 같은 물체는 한 번만 묻는다.
# 0.5m인 이유: 같은 물체의 위치 추정은 관측마다 이 정도 흔들리고(EMA로 수렴하기 전),
# 서로 다른 가구가 0.5m 안에 겹쳐 잡히는 일은 드물다.
DETECTION_VERIFICATION_CACHE_POSITION_QUANT_M = float(
    os.getenv("DETECTION_VERIFICATION_CACHE_POSITION_QUANT_M", "0.5")
)
# 3D 위치가 아직 없는 detection에 쓰는 예전 키의 bbox 양자화 폭(px). 파노라마가
# 1920px이라 48px는 가로 2.5% 수준이다.
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

# 최종 후보 선택 요청이 API/network 문제로 무기한 멈추지 않게 한다. google-genai의
# HttpOptions.timeout 단위는 millisecond이며, timeout이면 GeminiSelector가 로컬
# confidence/distance fallback으로 즉시 하나를 확정한다.
GEMINI_SELECTOR_TIMEOUT_MS = int(os.getenv("GEMINI_SELECTOR_TIMEOUT_MS", "30000"))

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
PANORAMA_V_FOV_DEG = float(os.getenv("PANORAMA_V_FOV_DEG", "120.0"))
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
# point(z 약 -0.75)는 여기서 제외된다.
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
ROBOT_CLEARANCE_M = 0.0 # 0.10 -> 0.0 ? ***

# frontier anchor(is_near_visited 예외 대상, 문 통과 문제 때문에 도입)가 plan_route()
# 호출마다 계속 다시 잡히면(=그 옆 unknown 셀이 영영 안 풀림, 예: 유리창이라 LiDAR가
# 못 뚫는 경우) 이 횟수를 넘는 순간부터 예외 자격을 박탈해서 결국 후보에서 빠지게 한다.
# 안 그러면 "도착 -> 같은 지점 재선택 -> 도착 -> ..." 무한 루프에 걸린다(실측으로 확인됨).
EXPLORATION_ANCHOR_MAX_REVISITS = 5

# base autonomy(waypointConverter)가 받아줄 지점이 없어서 **발행조차 못 한** 좌표의
# 셀이 이 횟수만큼 거부되면 후보 풀에서 뺀다(CoveragePlanner.mark_unpublishable).
# A*로는 도달 가능해서 planner 혼자서는 절대 못 걸러내는 좌표들이라, 이게 없으면
# "계획 -> 전 hop 발행 거부 -> OBSERVE -> 로봇이 안 움직여 지도 동일 -> 같은 계획"의
# 무한루프가 된다(실측 2026-08-24, 1.2초 주기).
# 1이 아니라 2인 이유: terrain_map은 로봇 주변 롤링 윈도우(decayTime 1.0s /
# noDecayDis 1.75m)라 지금 멀어서 못 받는 지점도 가까이 가면 받아줄 수 있다.
EXPLORATION_UNPUBLISHABLE_MAX_STRIKES = int(
    os.getenv("SYSNAV_EXPLORATION_UNPUBLISHABLE_MAX_STRIKES", "2")
)
# 위 blacklist가 후보 풀을 비우기 전에도 라이브락이 길어질 수 있으므로(발행 거부가
# 매번 다른 셀에서 나는 경우), "route는 나왔는데 전 hop이 발행 거부"가 이 횟수만큼
# 연속되면 탐사 소진으로 승격해 미션별 종료 처리로 넘긴다. Mission 2는
# MISSION2_EXPLORATION_TIME_LIMIT_SEC이 결국 구해주지만 Mission 1/3은 탈출구가 없다.
EXPLORATION_UNPUBLISHABLE_ROUTE_LIMIT = int(
    os.getenv("SYSNAV_EXPLORATION_UNPUBLISHABLE_ROUTE_LIMIT", "5")
)
# 위 횟수 상한과 **함께** 걸리는 시간 하한. 둘 다 넘겨야 소진으로 본다.
#
# 왜: 경로 계획이 0.2초라 5회가 1~2초 만에 소진된다. 그 사이 로봇은 한 번도 안 움직였고
# 지도도 그대로니 5번 다 같은 결과가 나오는 게 당연하다 - 라이브락을 감지한 게 아니라
# 같은 계산을 5번 반복한 것뿐이다(실측 2026-08-25: 0:40에 FAILED, 9분 20초 남음).
# 시간을 같이 걸면 그동안 terrain이 갱신되거나 로봇이 조금이라도 움직일 기회가 생긴다.
EXPLORATION_UNPUBLISHABLE_MIN_SEC = float(
    os.getenv("SYSNAV_EXPLORATION_UNPUBLISHABLE_MIN_SEC", "20.0")
)

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

# Exploration policy (SysNav paper Sec. IV-B-1): stochastic candidate selection
EXPLORATION_CANDIDATE_SAMPLES = 60  # |H|, 한 사이클에 샘플링할 pose 후보 수
EXPLORATION_MIN_SCORE_DELTA = 3     # δ, wcov가 이 밑으로 떨어지면 후보 뽑기를 멈춤
EXPLORATION_STOCHASTIC_TRIALS = 4   # K, stochastic sampling을 반복해서 TSP 비용 최소인 것을 채택
# 한 plan_route() 사이클에서 최대 몇 개의 candidate를 한 번에 뽑아 TSP로 묶을지. frontier
# anchor(모든 남은 frontier마다 보장되는 후보)까지 생기면서, 맵 양쪽 끝처럼 서로 먼 두 곳이
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
# 탐색 경로를 hop으로 자르는 간격. coverage_planner의 leg waypoint 생성이 쓴다.
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

# 참조 물체를 3D로 못 잡았을 때 이미지로 관계를 판정하는 후보 수 상한
# (missions/mission3_pipe._resolve_reference_by_image).
#
# 왜: rank_superlative()는 후보 전부를 이미지로 붙여 한 번에 보낸다. 좁은 방(후보 3~4개)
# 에서는 문제가 없지만 넓은 맵에서 picture가 30개 쌓이면 (1) 응답이 느려지고 (2) 그보다
# 심각하게 VLM 정확도가 무너지며(30장 중 "가장 문에 가까운 것"은 사람도 못 고른다)
# (3) _rank_key가 후보 집합 전체를 키에 넣어서 후보 하나만 늘어도 전원이 cache miss가
# 된다. 자르는 건 최종 선택이 아니라 예선이라 품질 손실이 작다.
RELATION_IMAGE_MAX_CANDIDATES = int(
    os.getenv("SYSNAV_RELATION_IMAGE_MAX_CANDIDATES", "8")
)

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
