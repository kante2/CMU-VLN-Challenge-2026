"""ROS2 orchestration node for frontier-coverage SysNav exploration.

Callbacks only cache messages. Heavy perception, Gemini and exploration jobs run in
worker threads and are coordinated by a timer-driven state machine.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import math
import os
import threading
import time

from geometry_msgs.msg import Point, PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Int32, String
from visualization_msgs.msg import Marker, MarkerArray

from sysnav import config
from sysnav.activity_log import JOB, NAV, PERCEPTION, STATE, WARN, activity
from sysnav.llm_trace import llm_trace
from sysnav.exploration.coverage_planner import CoveragePlanner
from sysnav.exploration.exploration_visualizer import export_exploration_debug
from sysnav.exploration.viewpoint_memory import ViewpointMemory
from sysnav.mission_dashboard import export_mission_dashboard
from sysnav.missions import mission1_pipe, mission2_pipe, mission3_pipe
from sysnav.memory.object_memory import ObjectMemory
from sysnav.navigation.goal_publisher import GoalPublisher
from sysnav.navigation.terrain_monitor import TerrainMonitor
from sysnav.perception.perception_pipeline import PerceptionPipeline
from sysnav.reasoning.attribute_filter import filter_by_attributes, reference_allowed_ids
from sysnav.reasoning.attribute_verifier import AttributeVerifier
from sysnav.reasoning.gemini_selector import GeminiSelector
from sysnav.reasoning.relation_image_verifier import RelationImageVerifier
from sysnav.reasoning.vlm_counter import VlmCounter
from sysnav.scene_graph.scene_graph_manager import SceneGraphManager
from sysnav.scene_graph.scene_graph_rviz import build_object_marker_array
from sysnav.ros_helpers import (
    closest_stamped_item,
    image_msg_to_rgb,
    message_stamp_to_sec,
    odometry_to_pose,
    pointcloud2_to_xyz,
)
from sysnav.task.llm_instruction_splitter import LLMInstructionSplitter
from sysnav.task.llm_query_parser import LLMQueryParser
from sysnav.task.mission_classifier import (
    MISSION_INSTRUCTION_FOLLOWING,
    MISSION_NUMERICAL,
    MISSION_OBJECT_REFERENCE,
    classify_mission,
)
from sysnav.task.query_parser import effective_relation_chain

# state 이름 -> 처리할 mission pipe 모듈. 미션에 없는 state로 잘못 분기되지 않도록
# question_callback에서 항상 task["mission_type"]을 이 dict의 키 중 하나로 채운다.
def normalize_question(text: str) -> str:
    """질문 문자열 비교용 정규화. 앞뒤 공백과 연속 공백만 접는다.

    같은 문장이 1Hz로 반복 발행될 때 중복 판정에 쓰인다(_claim_question). 발행 쪽
    포맷팅 차이(줄바꿈/두 칸 띄어쓰기)로 같은 질문이 다른 질문으로 보이면, 그때마다
    task가 새로 만들어져 지도와 메모리가 초기화된다."""
    return " ".join(str(text or "").strip().split())


_MISSION_PIPES = {
    MISSION_OBJECT_REFERENCE: mission2_pipe,
    MISSION_NUMERICAL: mission1_pipe,
    MISSION_INSTRUCTION_FOLLOWING: mission3_pipe,
}

'''
ThreadPoolExecutor 문법 -  시간이 오래걸리는 함수를 별도 작업 스레드에서 실행하도록 맡기는 도구,
즉 해당 도구로 무거운 작업을 다른 스레드로 넘기고, 메인 스레드는 계속 움직일수 있다.

< 문법 >
future = executor.submit(heavy_work)
print("다른 작업 수행")

max_workers=1 - 동시에 작업을 처리할 스레드를 하나만 만든다
thread_name_prefix - 스레디 이름 앞에 붙는 이름 / 어떤 스레드인지 알아보기 쉽게 만드는 옵션

< 문법 2 >
future = executor.submit(function, argument1, argument2)
function(argument1, argument2)를 Worker 스레드에서 실행하라는 의미
이를 통해 worker 스레드에 넘길 수 있다.

< 문법 3>
future - submit() 의 반환값은 실제 작업 결과가 아니라, future객체이다.
future객체를 통해서, 미래에 완료될 작업 결과를 나타내는 객체이다.
future.done() 반환값이 true, false임에 따라서 스레드의 결과를 확인할 수 있다.

future.result() 로 작업 결과를 가져올 수 있다.

-------------------------------------------------------------------------------

submit_job()
    → Worker에게 작업을 맡김

Worker Thread
    → perception / selection / exploration 실행

consume_future()
    → 완료 여부 확인
    → 결과 또는 오류 회수
    → 다음 state 결정
    → 필요하면 waypoint 발행

'''


_JOB_LABELS = {
    "perception": "인식(YOLO+SAM+LiDAR)",
    "selection": "대상 선택(Gemini)",
    "exploration": "탐색 경로 계획",
    "count": "개수 확정(VLM 카운트 + 기하)",
}
_NAV_LABELS = {
    "APPROACH": "접근 지점 선정", "GOAL": "목표 확정", "SNAP": "목표 보정(스냅)",
    "SNAP_FAIL": "목표 발행 불가", "PASSTHRU": "목표 그대로 전달",
    "SNAP_NO_PROGRESS": "스냅이 전진을 삼킴(건너뜀)",
    "FAR_THROW": "5m 밖으로 던짐(교착 탈출)", "FAR_THROW_SKIP": "던지기 불가(앞이 막힘)",
    "PUSHED": "base autonomy가 목표를 옮김", "RETARGET": "접근 지점 재선정",
    "RETARGET_FAIL": "접근 지점 재선정 실패", "UNREACHABLE": "도달 불가로 판단",
    "PREDICT_FALLBACK": "converter 예측 지점으로 전진",
    "PREDICT_REJECT": "예측 지점이 전진 아님(발행 안 함)",
}
# 이 이벤트들은 "왜 멈췄나"의 직접 원인이라 경고색으로 띄운다.
_NAV_PROBLEM_EVENTS = {
    "SNAP_FAIL", "RETARGET_FAIL", "UNREACHABLE", "PUSHED", "SNAP_NO_PROGRESS",
    "PREDICT_REJECT",
}


class SysNavNode(Node):
    def __init__(self) -> None:
        super().__init__("sysnav_node")

        # 지난 실행이 debug/llm_trace/에 남긴 이미지를 지운다(레코드는 메모리라 새
        # 프로세스에선 이미 비어있는데, 파일만 남으면 폴더가 계속 커진다).
        llm_trace.reset()

        self.callback_group = ReentrantCallbackGroup()
        # control_timer 전용 - MultiThreadedExecutor + ReentrantCallbackGroup 조합에서는
        # 이전 control_loop() 호출이 아직 안 끝났는데 다음 타이머 틱이 동시에 또 실행될 수
        # 있다(실측으로 확인됨 - 같은 좌표의 "SKIP" 로그 2줄이 0.2초가 아니라 0.3ms 간격으로
        # 찍힘). self.current_goal/self.exploration_route는 어떤 락도 안 걸려있어서 두
        # 실행이 동시에 건드리면 레이스가 나고, 한쪽이 current_goal을 None으로 만든 직후
        # 다른 쪽이 그걸 다시 읽다가 TypeError로 죽는 문제가 있었다. control_timer만 별도
        # MutuallyExclusiveCallbackGroup에 둬서 자기 자신과는 절대 안 겹치게 한다(센서
        # 콜백들은 원래대로 Reentrant라 control_loop와 동시에 돌아도 무방 - 락으로 보호된
        # 버퍼 쓰기뿐이라).
        self.control_callback_group = MutuallyExclusiveCallbackGroup()
        self.sensor_lock = threading.RLock()
        self.state_lock = threading.RLock()

        self.latest_image: Image | None = None
        self.latest_pose: dict | None = None
        self.scan_buffer = deque(maxlen=config.SCAN_BUFFER_SIZE)
        self.pose_buffer = deque(maxlen=config.POSE_BUFFER_SIZE)

        self.task_id = 0
        self.task: dict | None = None
        self.state = "IDLE"
        # 반복 발행되는 같은 질문을 걸러내기 위한 상태(config의 "같은 질문의 반복 발행
        # 처리" 주석 참고). _accepted_ok는 "이 문장을 실제로 task로 받는 데 성공했는가"다 -
        # 파싱에 실패한 문장은 재시도 간격 뒤에 다시 받아야 하므로 구분이 필요하다.
        self._accepted_question: str | None = None
        self._accepted_question_at = 0.0
        self._accepted_question_ok = False
        self._duplicate_question_count = 0
        self._last_duplicate_log = 0.0
        self.last_processed_image_stamp = -1.0
        self.last_perception_wall_time = 0.0
        # perception job을 못 던지는 이유(진단 전용). OBSERVE에서 아무 로그도 없이
        # 영원히 대기하는 상황을 밖에서 판별할 방법이 없어서 넣었다.
        self._snapshot_block_reason: str | None = None
        self._last_sensor_wait_log = 0.0

        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sysnav_worker")
        self.map_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sysnav_map")
        self.active_future: Future | None = None
        self.active_kind: str | None = None
        self.active_task_id: int | None = None
        self.active_origin_state: str | None = None
        self.mapping_future: Future | None = None
        self.last_map_submit_time = 0.0

        self.perception = PerceptionPipeline()
        self.query_parser = LLMQueryParser()
        self.instruction_splitter = LLMInstructionSplitter()
        self.object_memory = ObjectMemory()
        # Viewpoint/Object node와 edge를 관리한다. Viewpoint는 매 프레임이 아니라
        # novel LiDAR voxel coverage가 충분할 때만 생성하며 debug graph를 갱신한다.
        self.scene_graph = SceneGraphManager(debug_dir=config.DEBUG_DIR)
        self.selector = GeminiSelector()
        self.attribute_verifier = AttributeVerifier()
        self.relation_image_verifier = RelationImageVerifier()
        # Numerical 전용 - viewpoint 파노라마 한 장을 VLM에게 보여 직접 세게 한다
        # (reasoning/vlm_counter.py). 다른 미션은 안 쓴다.
        self.vlm_counter = VlmCounter()
        self.coverage_planner = CoveragePlanner()
        self.viewpoint_memory = ViewpointMemory()
        self.goal_publisher = GoalPublisher(self)
        self.terrain_monitor = TerrainMonitor()
        self.object_marker_pub = self.create_publisher(MarkerArray, config.TOPIC_OBJECT_MARKERS, 10)
        # 채점 대상 토픽(README) - Object Reference/Numerical. 절대 이름/타입을 바꾸지
        # 말 것 (Marker 단수, MarkerArray 아님 - CLAUDE.md 하드-룰).
        self.selected_object_marker_pub = self.create_publisher(
            Marker, config.TOPIC_SELECTED_OBJECT_MARKER, 10
        )
        self.numerical_response_pub = self.create_publisher(
            Int32, config.TOPIC_NUMERICAL_RESPONSE, 10
        )
        # RViz에서 base autonomy 데이터와 겹쳐 보기 위한 발행 전용 토픽들.
        # 주행 로직에는 전혀 관여하지 않는다(진단 목적).
        self.occupancy_pub = self.create_publisher(
            OccupancyGrid, config.TOPIC_SYSNAV_OCCUPANCY, 1
        )
        self.planned_path_pub = self.create_publisher(Path, config.TOPIC_SYSNAV_PATH, 1)
        self.frontier_pub = self.create_publisher(Marker, config.TOPIC_SYSNAV_FRONTIER, 1)

        self.current_goal: dict | None = None
        self.exploration_route = deque()
        # "route는 나왔는데 그 hop을 하나도 발행 못 했다"가 몇 번 연속됐는가.
        # 로봇이 못 움직이면 지도가 안 바뀌고, 지도가 안 바뀌면 같은 route가 다시
        # 나와서 PLAN_EXPLORATION <-> OBSERVE를 무한히 돈다 - publish_next_exploration_goal()
        # 주석 참고. 하나라도 발행되면 0으로 리셋한다.
        self._unpublishable_route_streak = 0
        self._exploration_goal_best_distance_m: float | None = None
        self._exploration_goal_last_progress_time: float | None = None

        # 확정된 목적지로 가는 주행(mission2 NAVIGATE_TARGET) 전용 상태.
        # exploration_route/publish_next_exploration_goal()을 재사용하지 않는 이유:
        # 그쪽은 route를 소진하면 state를 OBSERVE로 강제해서 미션 흐름과 충돌한다.
        self.target_route = deque()
        self.target_goal_xy: tuple[float, float] | None = None
        self.target_final_theta: float | None = None
        # mission3의 "avoiding the path between A and B" 제약(missions/mission3_pipe.py가
        # 만든 mission3_forbidden_mask)을 경로 계획에 그대로 넘기기 위한 보관 필드.
        # 재계획 때도 같은 제약이 유지돼야 하므로 목적지와 함께 들고 있는다.
        self.target_forbidden_mask = None
        self.target_object_id: int | None = None
        self.target_object_xy: tuple[float, float] | None = None
        self.target_marker_index: int | None = None
        self._target_replan_count = 0
        self._target_last_replan_time: float | None = None
        self._target_goal_best_distance_m: float | None = None
        self._target_goal_last_progress_time: float | None = None
        # 진단용(동작 영향 없음)
        self._target_retarget_count = 0
        self._target_unreachable_reason: str | None = None

        # Mission 3(Instruction-Following, missions/mission3_pipe.py) 전용 상태 -
        # 여러 목적지를 순서대로 처리해야 해서 mission2/1과 달리 진행 인덱스가
        # 필요하다. 다른 미션에서는 안 쓰이므로 매 새 질문마다 리셋만 하면 무해하다.
        self.mission3_step_index = 0
        self.mission3_forbidden_mask = None
        # 더 볼 frontier가 없는가. mission3는 "금지구역 참조 물체"와 "step에 필요한
        # 물체들"을 찾을 때까지 탐사를 먼저 하는데, 영원히 기다릴 수는 없으므로 이
        # 플래그가 서면 증거가 부족해도 진행한다 - mission3_pipe._select_step 주석 참고.
        self.mission3_exploration_exhausted = False
        # "take the path between A and B"의 게이트(A-B 선분). 중점 반경 도달과 별개로
        # 로봇이 실제로 그 사이를 가로질렀는지 판정한다 - missions/path_gate.py.
        # step이 바뀔 때마다 arm_gate()가 다시 세운다(이전 step 궤적은 안 센다).
        self.mission3_gate_segment = None
        self.mission3_gate_crossed = False
        self.mission3_gate_last_xy = None
        self.mission3_gate_last_stamp = 0.0
        # 확정된 subgoal을 연속 몇 번 "도달 불가"로 받았는가(mission3_pipe._navigate_step).
        # MISSION3_SUBGOAL_MAX_RETRIES를 넘으면 그 step을 포기하고 다음으로 넘어간다.
        self.mission3_subgoal_retries = 0
        # Mission 2는 탐사 중 발견 즉시 이동하지 않고, 모든 frontier를 소진한
        # 뒤 누적 Scene Graph에서 한 번만 최종 target을 고른다.
        self.mission2_exploration_complete = False
        self.mission2_exploration_deadline_reached = False
        # 채점 대상 답안(/selected_object_marker)을 이미 낸 물체. None이 아니면 "이번
        # 질문의 점수는 확보됐다"는 뜻이다 - 이후 주행이 실패해도 답을 취소하지 않는다
        # (README 채점: Object Reference는 marker bbox 겹침만 본다. 궤적 항목 없음).
        self.mission2_answer_object_id: int | None = None
        self._mission2_answer_extent: tuple | None = None
        self._mission2_last_answer_publish: float | None = None
        # 최종 target 주행이 "도달 불가"로 끝난 횟수(mission2_pipe._give_up_target).
        # config.MISSION2_TARGET_MAX_RETRIES를 넘으면 답안을 낸 채로 마무리한다.
        self.mission2_target_retries = 0

        # 디버깅용 미션 상태 대시보드(mission_dashboard.py)용 상태.
        self.task_start_time: float | None = None
        self.last_response_summary: str | None = None
        self._last_dashboard_write_time = 0.0
        # base autonomy가 우리 목표를 옮긴 정도(actual_waypoint_callback이 채운다).
        self.last_actual_waypoint_xy: tuple[float, float] | None = None
        self.last_waypoint_displacement_m: float | None = None
        self._last_traced_displacement_m: float | None = None
        # 우리가 마지막으로 waypoint를 발행한 시각. 이 직후 값만 밀림 측정에 쓴다.
        self._last_goal_publish_time: float | None = None
        # 직전 target goal 발행이 "받아줄 지점 없음"으로 막혔는가. 막힌 채로 두면
        # 로봇에게 아무 명령도 안 가서 그냥 서 있게 되므로, 그 상태를 감지해
        # stuck timeout(20초)을 기다리지 않고 바로 unreachable로 넘긴다.
        self._target_publish_blocked = False
        # 접근 지점 재선택이 연속으로 실패하기 시작한 시각(성공하면 None으로 되돌린다).
        self._target_retarget_fail_since: float | None = None
        # 활동 로그용 - 마지막으로 기록한 상태(전이 감지에 쓴다).
        self._logged_state = "IDLE"
        self.active_started_at: float | None = None

        self.question_sub = self.create_subscription(
            String,
            config.TOPIC_QUESTION,
            self.question_callback,
            10,
            callback_group=self.callback_group,
        )
        self.state_sub = self.create_subscription(
            Odometry,
            config.TOPIC_STATE,
            self.state_callback,
            qos_profile_sensor_data,
            callback_group=self.callback_group,
        )
        self.image_sub = self.create_subscription(
            Image,
            config.TOPIC_IMAGE,
            self.image_callback,
            qos_profile_sensor_data,
            callback_group=self.callback_group,
        )
        self.scan_sub = self.create_subscription(
            PointCloud2,
            config.TOPIC_SCAN,
            self.scan_callback,
            qos_profile_sensor_data,
            callback_group=self.callback_group,
        )
        self.terrain_sub = self.create_subscription(
            PointCloud2,
            config.TOPIC_TERRAIN_MAP,
            self.terrain_callback,
            qos_profile_sensor_data,
            callback_group=self.callback_group,
        )
        # base autonomy가 우리 좌표를 어디로 옮겼는지 읽기만 한다(발행 없음).
        self.actual_waypoint_sub = self.create_subscription(
            PointStamped,
            config.TOPIC_ACTUAL_WAYPOINT,
            self.actual_waypoint_callback,
            10,
            callback_group=self.callback_group,
        )
        self.map_publish_timer = self.create_timer(
            config.MAP_PUBLISH_INTERVAL_SEC,
            self._publish_map_topics,
            callback_group=self.callback_group,
        )
        self.control_timer = self.create_timer(
            config.CONTROL_PERIOD_SEC,
            self.control_loop,
            callback_group=self.control_callback_group,
        )
        self.get_logger().info("SysNav frontier-coverage planner started")

    # ------------------------------------------------------------------
    # ROS callbacks
    '''
    self.state_lock: # 읽는 도중 콜백으로 덮어쓰지 않도록 lock을 걸어준다.
    '''
    # ------------------------------------------------------------------

    def _claim_question(self, question: str) -> bool:
        """이 문장을 새 task로 처리해야 하면 True, 무시해야 하면 False.

        채점 환경은 /challenge_question을 1Hz로 계속 발행한다(--once 없이). 그대로
        받으면 매 초 Gemini 파싱을 다시 돌리고 task_id를 올리며 object_memory와
        scene_graph, coverage_planner를 통째로 초기화해서 로봇이 첫 관측 상태를
        영원히 못 벗어난다. 그래서 "지금 처리 중인 문장과 같은 문장"은 여기서 끊는다.

        파싱 **전에** 선점하는 이유: 구독 콜백이 ReentrantCallbackGroup이라, 파싱이
        2~14초 걸리는 동안 들어온 중복 메시지가 다른 executor 스레드에서 동시에
        파싱을 시작해버린다. 선점을 파싱 뒤로 미루면 중복 차단 자체가 무의미해진다.

        파싱에 실패한 문장은 영영 막아두면 복구가 안 되므로
        QUESTION_REPARSE_RETRY_SEC 뒤에는 다시 받아준다.
        """
        now = time.monotonic()
        dropped = 0
        with self.state_lock:
            already_handled = question == self._accepted_question and (
                self._accepted_question_ok
                or now - self._accepted_question_at < config.QUESTION_REPARSE_RETRY_SEC
            )
            if already_handled:
                self._duplicate_question_count += 1
                if now - self._last_duplicate_log < config.QUESTION_DUPLICATE_LOG_INTERVAL_SEC:
                    return False
                self._last_duplicate_log = now
                dropped = self._duplicate_question_count
            else:
                self._accepted_question = question
                self._accepted_question_at = now
                self._accepted_question_ok = False
        if already_handled:
            self.get_logger().info(
                f"🔁 같은 질문이 계속 들어옴 - 누적 {dropped}건 무시 "
                f"(진행 중인 Task #{self.task_id} 유지)"
            )
            return False
        return True

    def question_callback(self, msg: String) -> None:
        # 채점 환경은 같은 문장을 1Hz로 계속 발행한다. 파싱보다 **먼저** 걸러야 한다 -
        # Gemini 파싱은 2~14초가 걸리는데, 그 사이 들어온 중복 메시지들이 다른 executor
        # 스레드에서 동시에 파싱을 시작해버리기 때문이다(구독 콜백이 Reentrant 그룹).
        # 그래서 파싱 전에 문장을 "선점"해두고, 같은 문장은 여기서 바로 돌려보낸다.
        question = normalize_question(msg.data)
        if not question:
            return
        if not self._claim_question(question):
            return

        # 문장이 세 미션(Numerical/Object Reference/Instruction-Following) 중 어디로
        # 가야 하는지부터 정한다 - 응답 형식/상태머신이 미션마다 완전히 다르다
        # (MISSION_1/2/3_*_CLAUDE.txt 참고).
        mission_type = classify_mission(question)
        if mission_type == MISSION_INSTRUCTION_FOLLOWING:
            # 다단계 목적지 + 경로 제약 문장이라 단일 target G=(c_tgt,Φ) 파서로는
            # 못 담는다 - 절 단위로 쪼개서 목적지 절마다 같은 LLMQueryParser를 재사용.
            parsed = mission3_pipe.parse_instruction(self, question)
            is_valid = bool(parsed.get("steps"))
        else:
            # SysNav paper Sec. III의 G=(c_tgt, Φ) 파싱을 LLM이 하고, 실패하면 항상
            # 규칙 기반 query_parser.extract_target()로 자동 폴백한다.
            parsed = self.query_parser.parse(question)
            is_valid = bool(parsed.get("target"))
        parsed["mission_type"] = mission_type

        if not is_valid:
            # 선점은 유지한 채 실패로 남긴다 - _claim_question()이
            # QUESTION_REPARSE_RETRY_SEC 뒤에 같은 문장을 다시 받아준다(그 전까지는
            # 1Hz 반복 발행이 그대로 Gemini 호출로 이어지지 않도록 막는다).
            self.get_logger().error(f"Could not parse question ({mission_type}): {question}")
            return

        with self.state_lock: # 읽는 도중 콜백으로 덮어쓰지 않도록 lock을 걸어준다.
            self.task_id += 1
            self.task = parsed
            self.state = "OBSERVE"
            self.current_goal = None
            self.exploration_route.clear()
            self._unpublishable_route_streak = 0
            self.clear_target_navigation()
            self.last_processed_image_stamp = -1.0
            self.mission3_step_index = 0
            self.mission3_forbidden_mask = None
            self.mission3_exploration_exhausted = False
            self.mission3_gate_segment = None
            self.mission3_gate_crossed = False
            self.mission3_gate_last_xy = None
            self.mission3_gate_last_stamp = 0.0
            self.mission3_subgoal_retries = 0
            self.mission2_exploration_complete = False
            self.mission2_exploration_deadline_reached = False
            self.mission2_answer_object_id = None
            self._mission2_answer_extent = None
            self._mission2_last_answer_publish = None
            self.mission2_target_retries = 0
            self.task_start_time = time.monotonic()
            self.last_response_summary = None
            # 여기까지 왔으면 이 문장은 실제 task가 됐다 - 이후 같은 문장의 반복
            # 발행은 _claim_question()이 전부 무시한다(재시도 간격도 적용 안 됨).
            self._accepted_question_ok = True
            self._duplicate_question_count = 0

        if not config.KEEP_MEMORY_BETWEEN_TASKS:
            self.object_memory.clear()
            self.scene_graph.clear()
            self.viewpoint_memory.clear()
            with self.sensor_lock:
                pose = None if self.latest_pose is None else dict(self.latest_pose)
            self.coverage_planner.reset(pose)
        self.scene_graph.start_task(self.task_id, parsed)
        self.publish_object_markers()
        self.goal_publisher.reset_step_markers()

        if mission_type == MISSION_INSTRUCTION_FOLLOWING:
            self.get_logger().info(
                f"📩 NEW QUESTION [{mission_type}] - Task #{self.task_id}: \"{question}\" -> "
                f"steps={parsed['steps']}"
            )
        else:
            self.get_logger().info(
                f"📩 NEW QUESTION [{mission_type}] - Task #{self.task_id}: \"{question}\" -> "
                f"target={parsed['target']}, attributes={parsed['attributes']}, "
                f"relation={parsed['relation']}, references={parsed['reference_objects']}, "
                f"prompts={parsed['detection_prompts']}, parser={parsed.get('parser', 'rules')}"
            )

    def state_callback(self, msg: Odometry) -> None:
        pose = odometry_to_pose(msg)
        with self.sensor_lock:
            self.latest_pose = pose
            self.pose_buffer.append((pose["stamp"], pose))

    def image_callback(self, msg: Image) -> None:
        with self.sensor_lock:
            self.latest_image = msg

    def terrain_callback(self, msg: PointCloud2) -> None:
        """base autonomy의 지형 분석 결과를 그대로 받아둔다. 목표를 어디에 찍을지
        판정하는 데만 쓰고 주행 제어에는 관여하지 않는다."""
        try:
            self.terrain_monitor.update(msg)
        except Exception as error:
            self.get_logger().warning(f"terrain_map parse failed: {error}")

    def actual_waypoint_callback(self, msg: PointStamped) -> None:
        """base autonomy(waypointConverter)가 최종 확정한 목표를 받아, 우리가 요청한
        좌표와 얼마나 떨어졌는지 기록한다. Mission 3의 확정 subgoal 주행 중에는 실제
        좌표를 현재 목표와 marker에도 반영한다.

        waypointConverter는 우리 Pose2D를 그대로 쓰지 않고 obstacleDisThre(0.75m) 조건을
        만족하는 travArea 점으로 갈아끼운다. 그래서 "우리 planner는 A로 가라고 했는데
        로봇은 B로 갔다"가 조용히 일어나는데, 지금까지는 그걸 볼 방법이 없었다. 여기서
        차이를 계산해 RViz 마커와 navigation trace에 남긴다(읽기 전용, 주행에 영향 없음).
        """
        actual_xy = (float(msg.point.x), float(msg.point.y))
        requested_xy = self.goal_publisher.last_requested_xy
        self.last_actual_waypoint_xy = actual_xy
        if requested_xy is None or not self._is_measurable_waypoint(actual_xy):
            return

        displacement = math.hypot(actual_xy[0] - requested_xy[0], actual_xy[1] - requested_xy[1])
        self.last_waypoint_displacement_m = displacement
        self.goal_publisher.publish_requested_marker(actual_xy=actual_xy)

        # Mission 3 채점은 실제 trajectory를 보는데, 예전에는 marker/도착 판정은 요청
        # 좌표 A를 계속 사용하고 로봇은 waypointConverter가 확정한 B로 움직였다. 그러면
        # B에 정상 도착해도 A까지 1m 이상 남아 같은 subgoal을 무한 재발행한다.
        #
        # 도착 뒤 vehicle 앞 0.5m를 내보내는 projection 메시지는 위
        # _is_measurable_waypoint()에서 이미 제외했다. 또한 탐사 waypoint가 Mission 3
        # marker를 옮기지 않도록 확정 target 주행 상태에서만 동기화한다.
        task = self.task
        sync_mission3_target = (
            task is not None
            and task.get("mission_type") == MISSION_INSTRUCTION_FOLLOWING
            and self.state == "MISSION3_NAVIGATE_STEP"
            and self.current_goal is not None
            and self.current_goal.get("type") == "target"
        )
        if sync_mission3_target:
            # 물체 subgoal은 실제 waypoint가 물체 앞의 의미 있는 범위 안에 있을 때만
            # 동기화한다. waypointConverter가 다른 통과점으로 2~3m 밀어낸 좌표까지
            # goal marker로 채택하면 "go to pillow" marker가 pillow와 무관한 곳에 찍힌다.
            object_xy = self.target_object_xy
            if object_xy is not None:
                actual_object_distance = math.hypot(
                    actual_xy[0] - object_xy[0], actual_xy[1] - object_xy[1]
                )
                sync_mission3_target = (
                    actual_object_distance <= config.MISSION3_OBJECT_APPROACH_MAX_M
                )
        if sync_mission3_target:
            with self.state_lock:
                self.target_goal_xy = actual_xy
                self.current_goal["x"] = actual_xy[0]
                self.current_goal["y"] = actual_xy[1]
            self.refresh_goal_marker()

        # 같은 목표에 대해 매 프레임(10Hz) 같은 내용을 남기면 trace가 쓸모없어지므로,
        # 임계값을 넘고 직전에 남긴 값과 뚜렷이 달라졌을 때만 기록한다.
        if displacement < config.WAYPOINT_DISPLACEMENT_WARN_M:
            return
        previous = self._last_traced_displacement_m
        if previous is not None and abs(displacement - previous) < 0.10:
            return
        self._last_traced_displacement_m = displacement
        self._trace_navigation(
            "PUSHED",
            f"requested=({requested_xy[0]:.2f},{requested_xy[1]:.2f}) "
            f"actual=({actual_xy[0]:.2f},{actual_xy[1]:.2f}) "
            f"displacement={displacement:.2f}m label={self.goal_publisher.last_requested_label}"
        )

    def _is_measurable_waypoint(self, actual_xy: tuple[float, float]) -> bool:
        """이 /way_point 값이 "밀림"을 재는 데 쓸 수 있는가 (config.WAYPOINT_PROJ_DIS_M
        주석 참고). 도달 후 projection 값과 초기값 (0,0)을 걸러낸다."""
        if actual_xy == (0.0, 0.0):
            return False
        published_at = self._last_goal_publish_time
        if published_at is None:
            return False
        if time.monotonic() - published_at > config.WAYPOINT_MEASURE_WINDOW_SEC:
            return False
        with self.sensor_lock:
            pose = None if self.latest_pose is None else dict(self.latest_pose)
        if pose is not None:
            from_robot = math.hypot(actual_xy[0] - pose["x"], actual_xy[1] - pose["y"])
            if abs(from_robot - config.WAYPOINT_PROJ_DIS_M) <= config.WAYPOINT_PROJ_TOLERANCE_M:
                return False
        return True

    def scan_callback(self, msg: PointCloud2) -> None:
        stamp = message_stamp_to_sec(msg) # ROS 메시지에는 촬영 시간이 존재, 이를 추출
        with self.sensor_lock:
            self.scan_buffer.append((stamp, msg))
            pose = closest_stamped_item(
                list(self.pose_buffer),
                stamp,
                config.SENSOR_SYNC_TOLERANCE_SEC,
            )
            if pose is None and self.latest_pose is not None:
                pose = dict(self.latest_pose)

        now = time.monotonic()
        if (
            pose is not None
            and now - self.last_map_submit_time >= config.MAP_UPDATE_INTERVAL_SEC
            and (self.mapping_future is None or self.mapping_future.done())
        ):
            self.last_map_submit_time = now
            self.mapping_future = self.map_worker.submit(self.mapping_job, msg, dict(pose))

    # ------------------------------------------------------------------
    # Worker jobs
    # ------------------------------------------------------------------

    def mapping_job(self, scan_msg: PointCloud2, pose: dict) -> None:
        self.coverage_planner.update_from_scan(pointcloud2_to_xyz(scan_msg), pose)
        #-> Occupancy Map
        # frontier는 이 occupancy map을 통해서 찾게 된다.
        self._update_exploration_debug(pose)

    # surface point(S, plan_route()가 candidate 점수 매길 때 쓰는 것과 동일한 frontier
    # 마스크)를 exploration_debug_latest.png로 시각화 - "지금 frontier를 제대로 잡고
    # 있는지" RViz 없이 바로 확인할 수 있게 한다.
    def _update_exploration_debug(self, pose: dict) -> None:
        grid = self.coverage_planner.snapshot_grid()
        surface_mask = self.coverage_planner.surface_point_mask(grid)
        robot_cell = self.coverage_planner.world_to_grid(pose["x"], pose["y"])
        export_exploration_debug(grid, surface_mask, robot_cell)
    '''
    NumPy XYZ 배열
    ↓
    로봇 pose를 이용해 map 좌표로 변환
    ↓
    Occupancy Grid 갱신
    '''

    def stamp_age_sec(self, stamp: float) -> float:
        """ROS stamp가 지금으로부터 몇 초 전인지. 진단 로그용."""
        return max(0.0, self.get_clock().now().nanoseconds * 1e-9 - float(stamp))

    def _log_sensor_wait(self, state: str) -> None:
        """센서가 안 맞아 perception job을 못 던지는 동안 이유를 주기적으로 남긴다.
        진단 전용이라 주행 로직은 건드리지 않는다 - 예전에는 완전 무음이라 "질문은
        접수됐는데 그 뒤로 아무 일도 안 일어남" 상태를 밖에서 구분할 수 없었다."""
        now = time.monotonic()
        if now - self._last_sensor_wait_log < config.SENSOR_WAIT_LOG_INTERVAL_SEC:
            return
        self._last_sensor_wait_log = now
        reason = self._snapshot_block_reason or "(이유 미기록)"
        self.get_logger().warning(f"⏳ {state} 대기 - perception 못 던짐: {reason}")
        activity.add(WARN, f"{state} 대기 - perception 못 던짐", reason)

    def sensor_snapshot(self):
        with self.sensor_lock: 
            # 이 블록 안에서 센서 데이터를 읽는 동안 다른 callback이 같은 센서 변수에 접근하는 것을 잠시 막는다.
            # 블록이 끝나면 lock은 자동으로 해제된다.
            if self.latest_image is None or self.latest_pose is None:
                self._snapshot_block_reason = (
                    f"latest_image={'None' if self.latest_image is None else 'ok'}, "
                    f"latest_pose={'None' if self.latest_pose is None else 'ok'} "
                    f"(scan_buffer={len(self.scan_buffer)}, pose_buffer={len(self.pose_buffer)})"
                )
                return None
            image_msg = self.latest_image
            image_stamp = message_stamp_to_sec(image_msg)
            scan_msg = closest_stamped_item(
                list(self.scan_buffer), # ->  scan buffer 은 deque로 되어있다. 이를 일반 Python list로 복사해서 함수에 전달
                image_stamp,
                config.SENSOR_SYNC_TOLERANCE_SEC, # <- 동기화 허용 오차 시간
            )
            pose = closest_stamped_item(
                list(self.pose_buffer),
                image_stamp,
                config.SENSOR_SYNC_TOLERANCE_SEC,
            )
            if scan_msg is None:
                stamps = [stamp for stamp, _ in self.scan_buffer]
                nearest = min((stamp - image_stamp for stamp in stamps), key=abs, default=None)
                self._snapshot_block_reason = (
                    "scan_buffer 비어있음" if not stamps else
                    f"image stamp {image_stamp:.3f}에서 ±{config.SENSOR_SYNC_TOLERANCE_SEC}s 안에 "
                    f"scan 없음 (scan {len(stamps)}개, 가장 가까운 것 {nearest:+.3f}s)"
                )
                return None
            self._snapshot_block_reason = None
            if pose is None:
                pose = dict(self.latest_pose) # 참조를 하여 POSE를 넘기는 이유는 callback을 통해 데이터가 변형될 수 있기 때문.
            return image_msg, scan_msg, dict(pose), image_stamp # 4개의 튜플 형태로 반환된다.
    '''
    최신 이미지
    ↓ 이미지 촬영 시간 확인
    가장 가까운 LiDAR 검색
        ↓
    가장 가까운 robot pose 검색
    ↓
    Image + LiDAR + Pose + Timestamp 반환
    '''

    def perception_job(
        self,
        task_id: int, # 현재 처리중인 질문 번호 / worker가 어느 질문인지 확인하기 위함.
        task: dict, # 질문을 파싱한 결과 / target, attributes, relation, reference_objects 
        # task[detection_prompts]: YOLO-World가 검출해야 하는 객체 목록 /  task[target]: 최종적으로 찾으려는 목표 객체 종류
        image_msg: Image, # ROS 
        scan_msg: PointCloud2,
        pose: dict,
        image_stamp: float,
    ) -> dict:
        image_rgb = image_msg_to_rgb(image_msg) # ROS image -> numpy
        points_sensor = pointcloud2_to_xyz(scan_msg) # 동일 LiDAR를 인식과 Viewpoint coverage 계산에 공용 사용
        observations = self.perception.process( # 실제 객체 인식 파이프라인 실행
            image_rgb=image_rgb,
            points_sensor=points_sensor, # pointcloud -> numpy
            prompts=list(task["detection_prompts"]), #  YOLO-World가 검출해야 하는 객체 목록 -> prompts
            robot_pose=pose, # LiDAR의 객체 point를 map 좌표로 변환
        )
        '''
        < self.perception.process 내부 구조 >
        YOLO-World
            ↓
        2D Bounding Box

        SAM2
            ↓
        Object Mask

        LiDAR Grounding
            ↓
        3D Object Observation
        '''
        # update()의 반환값은 observations 순서에 대응하는 실제 object_id 목록이다.
        memory_before = self.object_memory.count()
        observed_object_ids = self.object_memory.update(observations, timestamp=image_stamp)
        memory_after = self.object_memory.count()
        activity.add(
            PERCEPTION,
            f"⑤ 메모리 반영 - 신규 {memory_after - memory_before}개, "
            f"기존과 병합 {len(set(observed_object_ids)) - (memory_after - memory_before)}개",
            f"누적 물체 {memory_after}개",
        )
        observed_object_nodes = [
            node
            for object_id in dict.fromkeys(observed_object_ids)
            if (node := self.object_memory.get(object_id)) is not None
        ]

        # 논문 로직대로 현재 LiDAR coverage C_t를 기존 Viewpoint coverage 합집합과 비교한다.
        # |C_t - C_prev|가 임계값보다 클 때만 대표 Viewpoint Node와 panorama를 저장한다.
        # Object-Object 관계는 현재 프레임에 한정하지 않고, 두 객체를 함께 관측한
        # 기존 Viewpoint들의 저장 이미지를 검색하여 on-demand로 검증한다.
        graph_update = self.scene_graph.add_observation(
            image_rgb=image_rgb,
            points_sensor=points_sensor,
            pose=pose,
            timestamp=image_stamp,
            observations=observations,
            object_ids=observed_object_ids,
            object_nodes=observed_object_nodes,
            task=task,
        )
        relation_edges = graph_update.get("relation_edges") or []
        activity.add(
            PERCEPTION,
            "⑥ Scene Graph 갱신"
            + (f" - viewpoint {graph_update.get('viewpoint_id')} 신규 추가"
               if graph_update.get("viewpoint_created") else " - viewpoint 추가 없음"),
            f"새 관계 edge {len(relation_edges)}개"
            + (f" ({', '.join(str(e.get('relation')) for e in relation_edges[:4])})"
               if relation_edges else "")
            + f" | novel_voxels={graph_update.get('novel_voxel_count', 0)}",
        )
        return {
            "task_id": task_id,
            "image_stamp": image_stamp,
            "candidates": self.object_memory.find_by_category(task["target"]),
            "scene_graph": graph_update,
        }
    '''
    동기화된 이미지·LiDAR·로봇 pose를 이용해 객체를 3D로 인식하고, Object Memory를 갱신한 뒤 목표 객체 후보들을 반환하는 작업

    Image + LiDAR + Pose + Task
                ↓
        Perception Pipeline
    YOLO-World → SAM2 (segment anything model) → 3D Grounding
                ↓
        3D Object Observations
                ↓
        Object Memory Update
                ↓
    질문의 Target category 후보 반환

    '''

    # task_id, # 현재 처리중인 질문 번호 / worker가 어느 질문인지 확인하기 위함.
    # task # 질문을 query_parser.py에서 분석한 결과
    # 
    def selection_job(
        self,
        task_id: int,
        task: dict,
        pose: dict,
        mission3_step_index: int | None = None,
    ) -> dict:
        # Mission 2의 최종 후보는 누적 Scene Graph에 실제 Object Node로 들어간 것만
        # 사용한다. ObjectMemory는 이미지/point cloud 원본을 가져오는 backing store이고,
        # 후보 집합 자체의 source of truth는 Scene Graph다.
        graph_snapshot = self.scene_graph.snapshot()
        graph_candidate_ids = {
            int(obj["object_id"])
            for obj in graph_snapshot.get("objects", [])
            if str(obj.get("category", "")).lower() == str(task["target"]).lower()
        }
        candidates = [
            candidate for candidate in self.object_memory.find_by_category(task["target"])
            if int(candidate["object_id"]) in graph_candidate_ids
        ]
        unfiltered_candidates = list(candidates)

        # 문장에 spatial constraint가 있고 Scene Graph에 검증된 Object-Object edge가
        # 존재하면, 해당 edge의 source object만 우선 후보로 사용한다. mission3는 절마다
        # 독립된 relation을 가진 step task를 여기로 직접 넘기는데, add_observation은
        # 최상위 placeholder task(relation 없음) 기준으로만 edge를 갱신하므로 이 task의
        # relation은 그쪽에서 절대 잡히지 않는다 - 여기서 먼저 직접 시도해서 채운다
        # (mission1/2는 이미 add_observation이 같은 task로 채워놨을 것이므로 대부분
        # _relation_checks 캐시에 걸려 사실상 공짜다).
        if effective_relation_chain(task):
            # 참조 물체에 속성 제약이 붙어 있으면("closest to the black chair") 관계
            # 판정 전에 그 카테고리 후보를 먼저 걸러 둔다 - 안 그러면 nearest의 argmin이
            # 검은 의자가 아닌 의자까지 포함해서 돌아 엉뚱한 lamp가 답이 된다.
            # 화이트리스트는 task에 실어 spatial_relation_reasoner가 읽는다.
            task["reference_allowed_ids"] = reference_allowed_ids(self, task)
            self.scene_graph.infer_relations_for_task(task, pose)
        relation_candidate_ids = set(self.scene_graph.find_matching_target_ids(task))
        if relation_candidate_ids:
            candidates = [
                candidate
                for candidate in candidates
                if int(candidate["object_id"]) in relation_candidate_ids
            ]
        elif effective_relation_chain(task):
            # 문장에 relation 제약(예: "knife rack 근처의")이 있는데 geometric/
            # co-observation 경로로는 아직 하나도 검증 안 됨 - 보통 참조 물체를
            # 아직 못 봤거나(전역 위치 없음), 유리창처럼 LiDAR grounding이 구조적으로
            # 실패해서(approximate 등급조차 못 만듦) 3D 위치 자체가 없기 때문.
            image_verified_ids: set[int] = set()
            if candidates:
                # 두 경로 모두 `{object_id: {cache_key: bool}}`를 돌려준다 -
                # attribute_verifier와 같은 on-demand 캐싱 패턴이라, 같은 사진에 대한
                # 같은 질문은 두 번 Gemini에 안 간다(relation_image_verifier 모듈
                # 주석 참고). 캐시 적립은 여기서(=object_memory를 아는 쪽에서) 한다.
                image_checks: dict[int, dict[str, bool]] = {}
                # relation-chain tuple은 (source_category, relation, target_category)다.
                # 첫 원소를 relation으로 잘못 쓰면 "near books" 대신
                # "potted plant books"처럼 무의미한 이미지 검증 prompt가 만들어진다.
                _, first_relation, first_reference = effective_relation_chain(task)[0]
                superlative = first_relation in ("nearest", "closest", "farthest", "furthest")
                if superlative and len(candidates) > 1:
                    # 최상급(비교) relation은 후보마다 독립적으로 yes/no만 물으면 안 된다 -
                    # bedside table이 2개 있고 둘 다 사진에 창문이 보이면 verify()는 둘 다
                    # 통과시켜버려서 어느 게 진짜 가까운지(먼지) 못 가린다. 후보 전부를
                    # 한 번에 놓고 VLM이 직접 비교해서 하나만 고르게 한다.
                    image_checks = self.relation_image_verifier.rank_superlative(
                        candidates, first_reference, first_relation
                    )
                else:
                    # 참조 물체를 3D로 잡을 필요 없이, 후보 자신의 사진만으로 "이 사진에
                    # 참조 물체가 보이는가"를 VLM에게 직접 확인받는다 (attribute_verifier와
                    # 같은 on-demand 이미지 판정 패턴).
                    image_checks = self.relation_image_verifier.verify(
                        candidates, first_relation, first_reference
                    )
                for object_id, checks in image_checks.items():
                    self.object_memory.update_relation_checks(object_id, checks)
                image_verified_ids = {
                    object_id for object_id, checks in image_checks.items()
                    if any(checks.values())
                }
            if image_verified_ids:
                candidates = [
                    candidate for candidate in candidates
                    if int(candidate["object_id"]) in image_verified_ids
                ]
            else:
                # 후보가 아직 없거나, 이미지 확인도 실패(또는 검증 안 됨) - 확정하지
                # 않고 계속 탐색해서 후보/참조 물체를 더 찾아보거나 다른 각도에서
                # 다시 시도한다.
                if not self.mission2_exploration_deadline_reached:
                    return {"task_id": task_id, "selected_id": None, "relation_pending": True}
                candidates = list(unfiltered_candidates)
                self.get_logger().warning(
                    "Mission 2 deadline fallback: relation remains unverified; "
                    "selecting from category candidates"
                )

        # SysNav paper Sec. IV-A-1 (self-attribute): 문장에 속성 제약(예: "black" chair)이
        # 있으면, 후보가 1개뿐이어도 반드시 VLM으로 확인한다 - "후보가 하나뿐이면 그냥
        # 확정"하던 예전 GeminiSelector 지름길이 색을 전혀 안 보고 넘어가버리는 원인이었다.
        attributes = list(task.get("attributes") or [])
        if attributes and config.ATTRIBUTE_VERIFICATION_ENABLED and candidates:
            candidates = filter_by_attributes(self, candidates, attributes)
            if not candidates:
                # 속성이 확인된 후보가 하나도 없다(전부 불일치했거나 아직 검증 자체가
                # 안 됨) - 확정하지 않고 계속 탐색해서 진짜 맞는 물체를 더 찾아본다.
                if not self.mission2_exploration_deadline_reached:
                    return {"task_id": task_id, "selected_id": None, "attribute_pending": True}
                candidates = list(unfiltered_candidates)
                self.get_logger().warning(
                    "Mission 2 deadline fallback: attributes remain unverified; "
                    "selecting from category candidates"
                )

        # GeminiSelector()
        selected_id = self.selector.select(
            question=task["raw"], # Gemini가 원본 문장을 그대로 이해하도록 전달
            candidates=candidates,
            # 전체 object node 가져오기 - Object Memory에 저장된 모든 객체를 가져
            context_objects=self.object_memory.all_nodes(),
            robot_pose=pose,
            final_verification=mission3_step_index is not None,
            # 같은 step/같은 관측 증거에서는 LLM을 다시 호출하지 않는다. 탐사 후
            # observation_count가 늘거나 후보 구성이 바뀌면 새 증거이므로 한 번 재검증한다.
            verification_key=(
                task_id,
                mission3_step_index,
                tuple(sorted(
                    (int(item["object_id"]), int(item.get("observation_count", 1)))
                    for item in candidates
                )),
            ) if mission3_step_index is not None else None,
        )
        if selected_id is None:
            return {
                "task_id": task_id,
                "selected_id": None,
                "relation_pending": False,
                "verification_pending": True,
            }
        return {
            "task_id": task_id, # 현재 처리중인 질문 번호 / worker가 어느 질문인지 확인하기 위함.
            "selected_id": selected_id,
            "relation_pending": False,
            "verification_pending": False,
            } # task (질의문장) 에 대해 선택된 object_id 반환
    '''
    Object Memory에 저장된 목표 후보들 중에서, 질문에 가장 맞는 객체 하나의 object_id를 고르는 작업

    Object Memory
        ↓
    Target category 객체만 검색
        ↓
    Gemini에 질문 + 후보 이미지 + 3D 정보 전달
        ↓
    가장 적절한 Object ID 선택
        ↓
    selected_id 반환

    < 핵심 2가지 >
    task["target"]: Object Memory에서 어떤 종류의 객체를 후보로 가져올지 결정
    task["raw"]: Gemini가 원본 문장을 그대로 이해하도록 전달
    '''

    def exploration_job(self, task_id: int, pose: dict) -> dict:
        route = self.coverage_planner.plan_route(
            pose,
            self.viewpoint_memory,
            # 탐사 경로도 금지구역을 피해야 한다. 채점은 목적지 주행이 아니라 실제
            # 주행 궤적 전체를 보므로, 탐사 중에 지나가면 그대로 감점이다.
            forbidden_mask=self.mission3_forbidden_mask,
        )
        return {
            "task_id": task_id,
            "route": route,
            "diagnostics": dict(self.coverage_planner.last_plan_diagnostics),
        }

    # ------------------------------------------------------------------
    # Future management
    # ------------------------------------------------------------------

    def submit_job(self, kind: str, function, *args, origin_state: str) -> None:
        if self.active_future is not None:
            return
        self.active_future = self.worker.submit(function, *args)
        self.active_kind = kind
        self.active_task_id = self.task_id
        self.active_origin_state = origin_state
        self.active_started_at = time.monotonic()
        activity.add(JOB, f"{_JOB_LABELS.get(kind, kind)} 작업 시작", f"state={origin_state}")

    # Worker Thread에 맡겨둔 작업이 끝났는지 확인하고, 완료된 결과를 받아 상태 머신에 반영하는 함수
    '''
        < Worker: >
    YOLO → SAM2 실행 중

        < Control loop: >
    consume_future() 호출
            ↓
    future.done() == False
            ↓
    return
    '''
    def consume_future(self) -> None:
        # 1. 작업이 완료되었는 지 확인
        # - 실행중인 작업이 없는 경우
        # - 작업이 아직 끝나지 않은 경우
        if self.active_future is None or not self.active_future.done():
            return
        '''
        worker에게 맞긴 경우, 아직 결과가 없을 수 있다. -> 그래서 실제 결과 대신 Future 객체를 먼저 받는다.
        future가 가리키는 Future 객체의 내부 상태가 Worker 실행 상황에 따라 갱신

        future객체에서는, 
            Future
            ├── 작업이 대기 중인가?
            ├── 실행 중인가?
            ├── 끝났는가?
            ├── 반환값은 무엇인가?
            └── 예외가 발생했는가?
        '''
        
        # 2.완료된 작업을 지역변수로 복사
        future = self.active_future
        kind = self.active_kind
        expected_task_id = self.active_task_id
        origin_state = self.active_origin_state

        # 3. activate 작업 상태 초기화
        self.active_future = None
        self.active_kind = None
        self.active_task_id = None
        self.active_origin_state = None

        # 4. worker 결과 가져오기
        try:
            elapsed = (
                "-" if self.active_started_at is None
                else f"{time.monotonic() - self.active_started_at:.1f}초"
            )
            result = future.result() # WORKER가 반환한 값을 .result() 을 통해서 가져온다.
            activity.add(JOB, f"{_JOB_LABELS.get(kind, kind)} 작업 완료", elapsed)
        #  worker에서 예외 발생시,
        except Exception as error: # Worker 함수 안에서 오류가 발생하면 future.result()를 호출할 때 그 예외가 다시 발생
            self.get_logger().error(f"⚠️ {kind} job failed: {error}")
            # ---------------- 작업 종류별 오류 복구 -----------------------
            with self.state_lock:
                if kind == "perception":
                    # - 초기 관측 중 실패        -> 인식에 실패했으니 탐색 계획 단계
                    # - 탐색 이동 중, 재관측 실패 -> 현재 탐색 계속
                    self.state = "FOLLOW_EXPLORATION" if origin_state == "FOLLOW_EXPLORATION" else "PLAN_EXPLORATION"
                elif kind == "selection":
                    # - Gemini 후보 선택이 실패했다면 목표 객체를 확정하지 않고 다시 탐색
                    self.state = "PLAN_EXPLORATION" # 탐색 이동중 재관측 실패
                else:
                    # - exploration 실패시, 다음 waypoint가 없으면 PLAN_EXPLORATION으로 돌아가서 새로운 waypoint를 찾는다.
                    self.state = "FAILED"
            return
        
        # 오래된 질문인지 확인
        #  비동기 작업 중 새 질문이 들어온 경우, 이전 질문의 결과를 버리는 안전장치
        if expected_task_id != self.task_id or result.get("task_id") != self.task_id:
            return

        with self.state_lock:
            task = None if self.task is None else dict(self.task)

        # perception 결과 반영(스탬프/scene graph 로깅/marker publish)은 세 미션
        # 공통이다 - "이 결과로 어떤 state로 갈지"만 미션마다 다르므로 그 판단만
        # missions/*.py에 위임한다.
        if kind == "perception":
            self.last_processed_image_stamp = float(result["image_stamp"])
            self.last_perception_wall_time = time.monotonic()
            graph_update = result.get("scene_graph")
            if graph_update and graph_update.get("debug_files"):
                if graph_update.get("viewpoint_created"):
                    self.get_logger().info(
                        f"Viewpoint {graph_update['viewpoint_id']} added: "
                        f"novel_voxels={graph_update['novel_voxel_count']}"
                    )
                else:
                    self.get_logger().debug(
                        "Viewpoint skipped: "
                        f"novel_voxels={graph_update.get('novel_voxel_count', 0)} "
                        f"<= threshold={graph_update.get('novel_threshold', 0)}"
                    )
                self.get_logger().debug(
                    f"Scene graph updated: {graph_update['debug_files']['json']}"
                )
            elif self.scene_graph.last_export_error:
                self.get_logger().warning(
                    f"Scene graph export failed: {self.scene_graph.last_export_error}"
                )
            self.publish_object_markers()

        if kind == "exploration":
            route = result.get("route") or []
            diagnostics = result.get("diagnostics") or {}
            if not route:
                # 지도 원점/로봇 cell이 아직 준비되지 않은 일시적 실패는 전체 탐사
                # 종료로 해석하지 않는다.
                transient_reasons = {
                    "origin_not_ready",
                    "robot_cell_out_of_map",
                    "robot_not_near_any_traversable_cell",
                    "no_traversable_cells_anywhere",
                }
                if diagnostics.get("reason") in transient_reasons:
                    with self.state_lock:
                        self.state = "PLAN_EXPLORATION"
                    return

        mission_pipe = _MISSION_PIPES.get(
            (task or {}).get("mission_type", MISSION_OBJECT_REFERENCE), mission2_pipe
        )
        mission_pipe.on_job_result(self, task, kind, result, origin_state)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def control_loop(self) -> None:
        self.consume_future()

        with self.state_lock:
            state = self.state
            task = None if self.task is None else dict(self.task)
            task_id = self.task_id
        # 상태 대입이 미션 파이프 여기저기에 흩어져 있어서, 대입부마다 로그를 넣는 대신
        # 여기서 변화를 관찰한다 - 이러면 전이를 하나도 안 놓친다.
        if state != self._logged_state:
            activity.add(WARN if state == "FAILED" else STATE,
                         f"상태 {self._logged_state} → {state}")
            self._logged_state = state

        self._update_mission_dashboard(state, task, task_id)

        if task is None or state in {"IDLE", "SUCCESS", "FAILED"}:
            return
        if self.active_future is not None:
            return

        # 실행 중인 worker가 없는 경계에서만 deadline 전환해 exploration 결과와
        # SELECT_TARGET 상태가 서로 덮어쓰는 race를 피한다.
        if (
            task.get("mission_type") == MISSION_OBJECT_REFERENCE
            and mission2_pipe.maybe_force_selection_at_deadline(self, state)
        ):
            return
        # Mission 1도 같은 이유로 탈출구가 필요하다 - 참조 물체를 못 찾으면
        # mission1_pipe의 게이트가 집계를 계속 미루므로, 예산을 넘기면 강제로 센다.
        if (
            task.get("mission_type") == MISSION_NUMERICAL
            and mission1_pipe.maybe_force_count_at_deadline(self, state)
        ):
            return

        with self.sensor_lock:
            pose = None if self.latest_pose is None else dict(self.latest_pose)
        if pose is None:
            return

        # OBSERVE/PLAN_EXPLORATION/FOLLOW_EXPLORATION은 세 미션이 공유하는 인프라
        # (perception/exploration job 제출, 이동 중 stuck 감지)라 여기서 그대로
        # 처리한다. 그 외 state(SELECT_TARGET/NAVIGATE_TARGET, MISSION1_*,
        # MISSION3_*)는 미션마다 의미가 달라서 해당 missions/*.py의 loop()에 위임한다.
        if state == "FOLLOW_EXPLORATION":
            if self.goal_reached(pose):
                goal = self.current_goal or {}
                self.get_logger().info(
                    f"🚩 ARRIVED - exploration waypoint reached: "
                    f"is_viewpoint={goal.get('is_viewpoint')}, "
                    f"coverage={goal.get('coverage_score', 0)}, "
                    f"robot_pose=({pose['x']:.2f}, {pose['y']:.2f}), "
                    f"remaining_in_route={len(self.exploration_route)}"
                )
                # 경로의 중간 hop까지 전부 "방문한 viewpoint"로 기록하면 안 된다 - 진짜
                # candidate(마지막 hop, is_viewpoint=True)만 기록해야 근처-방문 판정이
                # 지나온 복도 전체를 덮어버려 탐색이 조기 종료되는 걸 막을 수 있다.
                if self.current_goal is not None and self.current_goal.get("is_viewpoint"):
                    self.viewpoint_memory.add(
                        self.current_goal["x"],
                        self.current_goal["y"],
                        self.current_goal["theta"],
                        self.current_goal.get("coverage_score"),
                    )
                self.publish_next_exploration_goal()
                return

            if self._exploration_goal_unreachable(pose):
                self.get_logger().warning(
                    f"⏭️ SKIP - exploration goal unreachable (no progress for "
                    f"{config.EXPLORATION_STUCK_TIMEOUT_SEC:.0f}s), skipping "
                    f"({self.current_goal['x']:.2f}, {self.current_goal['y']:.2f})"
                )
                # 도달 실패한 후보는 재선택하지 않도록 방문 처리한다.
                self.viewpoint_memory.add(
                    self.current_goal["x"], self.current_goal["y"],
                    self.current_goal["theta"], self.current_goal.get("coverage_score"),
                )
                self.publish_next_exploration_goal()
                return

            if time.monotonic() - self.last_perception_wall_time >= config.PERCEPTION_WHILE_MOVING_INTERVAL_SEC:
                snapshot = self.sensor_snapshot()
                if snapshot is not None:
                    image_msg, scan_msg, synced_pose, image_stamp = snapshot
                    if image_stamp > self.last_processed_image_stamp:
                        self.submit_job(
                            "perception",
                            self.perception_job,
                            task_id,
                            task,
                            image_msg,
                            scan_msg,
                            synced_pose,
                            image_stamp,
                            origin_state="FOLLOW_EXPLORATION",
                        )
            return

        if state == "OBSERVE":
            snapshot = self.sensor_snapshot()
            if snapshot is None:
                self._log_sensor_wait("OBSERVE")
                return
            image_msg, scan_msg, synced_pose, image_stamp = snapshot
            if image_stamp <= self.last_processed_image_stamp:
                # 카메라 프레임이 갱신되지 않고 있다. latest_image가 이미 처리한
                # 프레임 그대로면 여기서 영원히 되돌아가는데, 예전엔 아무 로그도
                # 없어서 "OBSERVE인데 아무 일도 안 일어남"으로만 보였다.
                self._snapshot_block_reason = (
                    f"새 카메라 프레임 없음 - latest_image stamp {image_stamp:.3f}는 "
                    f"이미 처리함(last_processed={self.last_processed_image_stamp:.3f}, "
                    f"{self.stamp_age_sec(image_stamp):.1f}초 전 프레임)"
                )
                self._log_sensor_wait("OBSERVE")
                return
            self.submit_job(
                "perception",
                self.perception_job,
                task_id,
                task,
                image_msg,
                scan_msg,
                synced_pose,
                image_stamp,
                origin_state="OBSERVE",
            )
            return

        if state == "PLAN_EXPLORATION":
            self.submit_job(
                "exploration",
                self.exploration_job,
                task_id,
                pose,
                origin_state=state,
            )
            return

        mission_pipe = _MISSION_PIPES.get(
            task.get("mission_type", MISSION_OBJECT_REFERENCE), mission2_pipe
        )
        mission_pipe.loop(self, state, task, task_id, pose)

    def _finish_exploration_as_exhausted(self) -> None:
        """탐사를 "더 갈 곳 없음"으로 확정하고 미션별 종료 처리로 넘긴다.

        plan_route()는 빈 route를 반환하지 않았다 - frontier도 있고 A*로 도달도
        가능하니 planner 입장에선 갈 곳이 있는 게 맞다. 하지만 그 좌표를 base
        autonomy가 하나도 받아주지 않으면 로봇은 한 발짝도 못 움직이고, 움직이지
        않으니 지도도 안 바뀌어서 다음 사이클에 똑같은 route가 다시 나온다. 실질적으로
        "더 갈 수 있는 곳이 없다"와 같은 상황이므로 여기서 빈 route와 동일하게 취급한다.

        Mission 2는 MISSION2_EXPLORATION_TIME_LIMIT_SEC이 결국 구해주지만 Mission 1/3은
        탈출구가 없어서 10분을 통째로 이 루프에 쓴다(실측 2026-08-24).
        """
        with self.state_lock:
            task = None if self.task is None else dict(self.task)
            task_id = self.task_id
        self._unpublishable_route_streak = 0
        self.exploration_route.clear()
        if task is None:
            # 처리 중인 질문이 없다 - 넘길 곳이 없으므로 그냥 대기 상태로 둔다.
            with self.state_lock:
                self.state = "OBSERVE"
            return
        self.get_logger().warning(
            "🧭 EXPLORATION LIVELOCK - planner keeps producing routes but base autonomy "
            f"accepts none of their hops ({config.EXPLORATION_UNPUBLISHABLE_ROUTE_LIMIT} "
            "consecutive routes); treating exploration as exhausted"
        )
        mission_pipe = _MISSION_PIPES.get(
            (task or {}).get("mission_type", MISSION_OBJECT_REFERENCE), mission2_pipe
        )
        mission_pipe.on_job_result(
            self, task, "exploration", {"task_id": task_id, "route": []}, "PLAN_EXPLORATION"
        )

    # state == "FOLLOW_EXPLORATION" -> publish next exploration goal
    def publish_next_exploration_goal(self) -> None:
        # base autonomy가 받아줄 수 없는 hop은 건너뛰고 다음 후보를 바로 시도한다.
        # 예전엔 그런 좌표도 그대로 발행했는데, waypointConverter가 그걸 로봇 발밑에
        # 떨어뜨려서 로봇이 서 있고 stuck timeout(8초)을 통째로 버린 뒤에야 다음으로
        # 넘어갔다 (goal_publisher.publish() docstring의 실측 참고).
        skipped = 0
        goal = None
        published = None
        while self.exploration_route:
            candidate = self.exploration_route.popleft()
            published = self.goal_publisher.publish(
                candidate["x"], candidate["y"], candidate["theta"],
                label="exploration viewpoint" if candidate.get("is_viewpoint") else "exploration hop",
            )
            if published is not None:
                goal = candidate
                break
            # 거부된 좌표는 planner에 되먹인다. 안 그러면 A*로는 멀쩡히 도달 가능한
            # 좌표라 다음 사이클에도 똑같이 뽑힌다.
            self.coverage_planner.mark_unpublishable(candidate["x"], candidate["y"])
            skipped += 1

        if goal is None:
            self.current_goal = None
            if skipped:
                self._unpublishable_route_streak += 1
                self.get_logger().info(
                    f"🧭 exploration route exhausted - {skipped} hop(s) had no point "
                    f"base autonomy would accept "
                    f"(streak {self._unpublishable_route_streak}/"
                    f"{config.EXPLORATION_UNPUBLISHABLE_ROUTE_LIMIT}); re-observing"
                )
                if self._unpublishable_route_streak >= config.EXPLORATION_UNPUBLISHABLE_ROUTE_LIMIT:
                    self._finish_exploration_as_exhausted()
                    return
            with self.state_lock:
                self.state = "OBSERVE"
            return
        self._unpublishable_route_streak = 0
        if skipped:
            self.get_logger().info(
                f"🧭 skipped {skipped} unreachable exploration hop(s) before this goal"
            )
        # 발행된 좌표(스냅 반영)를 그대로 current_goal로 쓴다 - 원본을 저장하면 로봇이
        # 갈 수 없는 좌표를 기준으로 도착 판정을 하게 되어 영원히 도착하지 않는다.
        goal = {**goal, "x": float(published.x), "y": float(published.y)}
        self.current_goal = {**goal, "type": "exploration"}
        self._exploration_goal_best_distance_m = None
        self._exploration_goal_last_progress_time = time.monotonic()
        with self.state_lock:
            self.state = "FOLLOW_EXPLORATION"

        # "보라색 waypoint는 찍히는데 로봇이 안 움직인다" 같은 증상을 로그 하나로 바로
        # 진단하기 위해, 목표가 로봇 현재 위치 대비 실제로 얼마나 먼지 같이 남긴다.
        with self.sensor_lock:
            robot_pose = None if self.latest_pose is None else dict(self.latest_pose)
        if robot_pose is not None:
            distance_m = math.hypot(goal["x"] - robot_pose["x"], goal["y"] - robot_pose["y"])
            distance_note = f", robot=({robot_pose['x']:.2f}, {robot_pose['y']:.2f}), dist={distance_m:.2f}m"
        else:
            distance_note = ", robot pose unknown"

        remaining = len(self.exploration_route)
        self.get_logger().info(
            f"➡️ DEPARTING - exploration goal=({goal['x']:.2f}, {goal['y']:.2f}, {goal['theta']:.2f}), "
            f"is_viewpoint={goal.get('is_viewpoint')}, coverage={goal.get('coverage_score', 0)}, "
            f"remaining_in_route={remaining}{distance_note}"
        )

    # ------------------------------------------------------------------
    # Target navigation (확정된 목적지로 가는 주행 + 경로 재계획)
    #
    # 탐색(FOLLOW_EXPLORATION)과 달리, 여기서 다루는 목적지는 "포기하고 다음 후보로
    # 넘어갈" 대상이 아니라 끝까지 가야 하는 곳이다. 그래서 한 번 계산한 경로를 끝까지
    # 고집하지 않고, 아래 세 시점에 최신 지도로 A*를 다시 돌린다:
    #   1. hop 도착 (아직 목적지가 아니면 다음 구간을 새로 계산)
    #   2. 주행 중 hop line-of-sight 차단 (지도가 "이 길 막혔다"고 알려준 즉시)
    #   3. TARGET_REPLAN_STUCK_TIMEOUT_SEC 동안 진전 없음 (1,2가 못 잡는 경우의 백스톱)
    # A*가 경로를 못 찾을 때만 목적지를 포기하고 탐사 재계획으로 넘긴다.
    # ------------------------------------------------------------------

    def clear_target_navigation(self) -> None:
        self._target_publish_blocked = False
        self._target_retarget_fail_since = None
        self.target_route.clear()
        self.target_goal_xy = None
        self.target_final_theta = None
        self.target_forbidden_mask = None
        self.target_object_id = None
        self.target_object_xy = None
        self.target_marker_index = None
        # current_goal도 같이 지운다 - 안 지우면 도착/포기 후에도 직전 hop이 남아서,
        # 다음 step을 resolve하는 동안(mission3) 그 stale goal로 goal_reached()가
        # 참이 되어 "도착했다"고 잘못 판정될 수 있다.
        if self.current_goal is not None and self.current_goal.get("type") == "target":
            self.current_goal = None
        self._target_replan_count = 0
        self._target_last_replan_time = None
        self._target_goal_best_distance_m = None
        self._target_goal_last_progress_time = None
        self._target_retarget_count = 0

    def start_target_navigation(
        self,
        pose: dict,
        goal_xy: tuple[float, float],
        final_theta: float,
        forbidden_mask=None,
        object_id: int | None = None,
        object_xy: tuple[float, float] | None = None,
        marker_index: int | None = None,
    ) -> None:
        """확정된 목적지로 가는 주행을 시작한다.

        기본은 목적지 하나만 발행하고 base autonomy에 맡긴다. localPlanner + pathFollower가
        이미 실시간 지형 기반 로컬 플래너로 돌고 있어서, 우리가 A*로 짧은 hop을 강제하면
        도움이 아니라 그 판단과 싸우게 된다(5c12691에서 소스 확인 후 내린 결론).

        예외는 forbidden_mask("avoid the path between A and B")뿐이다. 이 제약은 base
        autonomy가 알 방법이 없으므로 우리가 우회 경로를 만들어 hop으로 내보내야 한다.

        object_xy: 목표 물체 좌표. 주행 중 terrain이 갱신되어 지금 목표가 못 쓰게 되면
          이걸로 접근 지점을 다시 고른다(step_target_navigation 참고).
        marker_index: RViz goal 마커의 id(mission3의 step 인덱스). 주행 중 목표를 다시
          고르면 이 id로 마커도 새 위치에 다시 그린다 - 안 그러면 마커는 옛 위치에
          남고 실제 목표만 옮겨져서, RViz로 보는 거리와 로그의 dist_to_goal이 어긋난다.
        """
        self.clear_target_navigation()
        self.target_goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
        self.target_final_theta = float(final_theta)
        self.target_forbidden_mask = forbidden_mask
        self.target_object_id = object_id
        self.target_object_xy = None if object_xy is None else (
            float(object_xy[0]), float(object_xy[1])
        )
        self.target_marker_index = marker_index

        # 기본은 목적지 하나만 발행하고 base autonomy에 맡긴다.
        #
        # 한때 "항상 A* hop으로 잘라 보내기"로 바꿨다가 되돌렸다(2026-08-22). 근거였던
        # "우리 좌표가 발밑으로 덤프된다"는 실측은 맞지만, 그건 목표가 5m 안일 때만
        # 일어나는 일이고(waypointConverter: `if (dis < adjDisThre)`) hop을 써도 마지막
        # 구간에서 똑같이 겪는다 - 즉 hop으로는 안 풀린다(그건 Layer 1 스냅이 담당).
        #
        # 반면 hop을 강제하면 localPlanner의 판단 범위를 실제로 깎는다: pathCropByGoal이
        # true라 "목표까지 거리 + goalClearRange(0.5m)" 밖의 장애물은 아예 보지 않는다.
        # 1.5m hop을 주면 시야가 adjacentRange(3.5m) -> 2.0m로 줄어든다. 게다가 우리 A*는
        # 격자 1셀(0.20m) 클리어런스라 localPlanner의 차체(0.5x0.5m)보다 낙관적이어서,
        # 우리가 "지나갈 수 있다"고 본 틈 앞에서 로봇이 멈추는 불일치만 늘어난다.
        #
        # 예외는 forbidden_mask("avoid the path between A and B")뿐이다. 이 제약은 base
        # autonomy가 알 방법이 없으므로 우리가 우회 경로를 만들어 hop으로 내보내야 한다
        # (README 채점: "passes through areas it is forbidden to go through" 감점 항목).
        if forbidden_mask is not None:
            path, elapsed_ms = self._plan_target_path(pose)
            if path:
                self.target_route = deque(path)
                self.get_logger().info(
                    f"🧭 TARGET PATH planned (forbidden constraint) - {len(path)} hops to "
                    f"({self.target_goal_xy[0]:.2f}, {self.target_goal_xy[1]:.2f}), {elapsed_ms:.1f}ms"
                )
                self.publish_next_target_hop()
                return
            diag = self.coverage_planner.last_direct_path_diagnostics
            self.get_logger().warning(
                f"🧭 TARGET PATH unavailable ({diag.get('reason')}, {elapsed_ms:.1f}ms) - "
                f"publishing goal directly; the forbidden-area constraint cannot be enforced"
            )

        self._trace_navigation(
            "GOAL",
            f"({self.target_goal_xy[0]:.2f},{self.target_goal_xy[1]:.2f}) "
            f"terrain[{self.terrain_monitor.describe()}]",
        )
        if not self._publish_target_goal(
            self.target_goal_xy[0], self.target_goal_xy[1], self.target_final_theta,
            is_final=True,
        ):
            # 목적지 근처에 base autonomy가 받아줄 지점이 없다. 억지로 발행하지 않고
            # 진행도 감시(target_progress_stalled)가 unreachable로 넘기도록 둔다.
            self.get_logger().warning(
                f"🧭 target goal ({self.target_goal_xy[0]:.2f}, {self.target_goal_xy[1]:.2f}) "
                f"is not commandable - {self.terrain_monitor.last_selection}"
            )

    def publish_next_target_hop(self) -> None:
        """target_route에서 다음 hop을 꺼내 발행한다. state는 건드리지 않는다
        (미션별 state는 missions/*.py가 관리한다)."""
        if not self.target_route:
            return
        # 경로의 **가장 먼 hop부터** 거꾸로 훑어 지금 발행 가능한 첫 hop을 쓴다.
        # 한 번에 최대한 멀리 보내야 명령 횟수가 줄고, terrain 유효 반경(1.75m) 밖
        # hop은 어차피 발행 판정에서 걸러진다. 직선거리로 "목표에 가장 가까운 점"을
        # 고르지 않는 이유: 그건 벽 뒤에 목표가 있을 때 벽 앞에서 갇히는 local minimum
        # 문제가 있다. A* 경로 위에서만 고르면 경로가 이미 벽을 우회하므로 안 갇힌다.
        hops = list(self.target_route)
        for index in range(len(hops) - 1, -1, -1):
            hop = hops[index]
            if not self.goal_publisher.can_publish(hop["x"], hop["y"]):
                continue
            if self._publish_target_goal(
                hop["x"], hop["y"], hop["theta"], is_final=(index == len(hops) - 1)
            ):
                for _ in range(index + 1):
                    self.target_route.popleft()
                if index:
                    self.get_logger().info(
                        f"🧭 advanced {index + 1} hop(s) at once "
                        f"({len(self.target_route)} left)"
                    )
                return
        # 남은 hop이 전부 발행 불가 - 억지로 보내지 않고, 진행도 감시(백스톱)가
        # unreachable로 판정해 재계획하도록 둔다.
        self.get_logger().warning(
            f"🧭 none of {len(hops)} remaining target hop(s) are commandable right now"
        )

    def _publish_target_goal(self, x: float, y: float, theta: float, is_final: bool) -> bool:
        """목표 hop을 발행한다. base autonomy가 받아줄 지점이 근처에 없어서 발행하지
        못하면 False를 돌려준다 - 호출 측은 다음 hop으로 넘어가거나 재계획해야 한다.
        받아줄 수 없는 좌표를 억지로 발행하면 로봇 발밑에 목표가 찍혀 그대로 멈춘다
        (goal_publisher.publish() docstring 참고)."""
        # publish()가 base autonomy가 받아줄 지점으로 옮길 수 있으므로, 도착 판정은
        # 원본이 아니라 실제 발행된 좌표를 기준으로 해야 한다(publish() docstring 참고).
        published = self.goal_publisher.publish(
            x, y, theta, label="target goal" if is_final else "target hop"
        )
        if published is None:
            self._target_publish_blocked = True
            return False
        self._target_publish_blocked = False
        x, y = float(published.x), float(published.y)
        self.current_goal = {
            "x": float(x),
            "y": float(y),
            "theta": float(theta),
            "type": "target",
            "is_final_hop": bool(is_final),
            # 어느 물체로 가는 중인지(도착 로그와 mission2의 SUCCESS 처리가 쓴다).
            # current_goal에서 이어받지 않고 target_object_id를 쓰는 이유: 그러면
            # 호출 쪽이 start_target_navigation() 전에 current_goal에 object_id만 든
            # 반쪽짜리 dict를 심어둬야 하는데, 그 사이에 goal_reached()가 불리면
            # x/y가 없어서 KeyError가 난다.
            "object_id": self.target_object_id,
        }
        # 진행도 감시(백스톱)는 hop이 바뀔 때마다 리셋한다 - 새 hop은 새로 재는 것이 맞다.
        self._target_goal_best_distance_m = None
        self._target_goal_last_progress_time = time.monotonic()
        self.get_logger().info(
            f"➡️ TARGET HOP - goal=({x:.2f}, {y:.2f}, {theta:.2f}), "
            f"is_final={is_final}, remaining_hops={len(self.target_route)}, "
            f"replans={self._target_replan_count}"
        )
        return True

    def target_destination_reached(self, pose: dict) -> bool:
        """RViz에 표시한 최종 marker의 0.5m 성공 반경 안에 들어왔는지."""
        return self.distance_to_target(pose) <= config.TARGET_SUCCESS_DISTANCE_M

    def approach_pose_for(
        self,
        pose: dict,
        object_position,
        max_distance_m: float | None = None,
        allow_relaxed: bool = False,
    ) -> tuple[float, float, float]:
        """물체로 접근할 (x, y, theta)를 정한다. mission2/3가 공용으로 쓴다.

        1순위는 /terrain_map 기준으로 base autonomy가 받아들일 지점(TerrainMonitor).
        terrain 데이터가 없거나 통과 지점을 못 찾으면 기존 고정 standoff 방식으로
        폴백한다 - terrain 판정 실패가 주행 자체를 막으면 안 된다.
        """
        object_xy = (float(object_position[0]), float(object_position[1]))
        robot_xy = (float(pose["x"]), float(pose["y"]))
        chosen = self.terrain_monitor.choose_approach_point(
            object_xy, robot_xy, max_distance_m=max_distance_m,
            allow_relaxed=allow_relaxed,
        )
        if chosen is not None:
            theta = math.atan2(object_xy[1] - chosen[1], object_xy[0] - chosen[0])
            self._trace_navigation(
                "APPROACH",
                f"terrain ({chosen[0]:.2f},{chosen[1]:.2f}) "
                f"obj=({object_xy[0]:.2f},{object_xy[1]:.2f}) "
                f"{self.terrain_monitor.last_selection}",
            )
            return chosen[0], chosen[1], theta

        x, y, theta = self.goal_publisher.object_approach_pose(pose, object_position)
        self._trace_navigation(
            "APPROACH",
            f"fallback ({x:.2f},{y:.2f}) obj=({object_xy[0]:.2f},{object_xy[1]:.2f}) "
            f"{self.terrain_monitor.last_selection}",
        )
        return x, y, theta

    def target_progress_stalled(self, pose: dict) -> bool:
        """백스톱 트리거: TARGET_REPLAN_STUCK_TIMEOUT_SEC 동안 목표에 가까워지지 못함."""
        distance, best, last_progress = self._track_goal_progress(
            pose,
            self._target_goal_best_distance_m,
            self._target_goal_last_progress_time,
        )
        self._target_goal_best_distance_m = best
        self._target_goal_last_progress_time = last_progress
        if distance is None or last_progress is None:
            return False
        return (
            time.monotonic() - last_progress >= config.TARGET_REPLAN_STUCK_TIMEOUT_SEC
        )

    def replan_target_route(self, pose: dict, reason: str) -> str:
        """최신 지도로 목적지까지의 경로를 다시 계산하고, 무슨 일이 있었는지 문자열로
        돌려준다. bool이 아닌 이유: 호출한 미션이 "재계획됨"과 "더 갈 곳이 없음"을
        구분해야 하는데(전자는 계속 주행, 후자는 도달 인정), bool로는 그 둘이 같아진다.

          "replanned" - 새 경로로 첫 hop까지 발행함
          "unchanged" - 계산은 됐는데 지금 향하던 hop 그대로다. hop에 이미 도착한
                        상태에서 이 값이 나오면 "지금 지도로 여기보다 더 가까이는
                        못 간다"는 뜻이다.
          "cooldown"  - 최소 간격 안이라 이번엔 건너뜀 (실패 아님, 보류)
          "failed"    - A*가 경로를 못 찾음
          "limit"     - hop 도착 없이 연속 재계획 상한 초과
        """
        if self.target_goal_xy is None:
            return "failed"

        now = time.monotonic()
        # 같은 트리거가 control_loop(0.2초)마다 반복해서 걸릴 수 있으므로 최소 간격을 둔다.
        if (
            self._target_last_replan_time is not None
            and now - self._target_last_replan_time < config.TARGET_REPLAN_MIN_INTERVAL_SEC
        ):
            return "cooldown"

        # hop 도착으로 인한 재계획은 정상 진행이므로 상한에 세지 않고 오히려 카운터를
        # 리셋한다 - 한 hop이라도 실제로 도착했다는 건 경로가 통하고 있다는 뜻이다.
        if reason.endswith("arrival"):
            self._target_replan_count = 0
        elif self._target_replan_count >= config.TARGET_REPLAN_MAX_COUNT:
            self.get_logger().warning(
                f"🧭 TARGET REPLAN limit reached ({self._target_replan_count} consecutive "
                f"replans without arriving at a hop) - giving up on "
                f"goal=({self.target_goal_xy[0]:.2f}, {self.target_goal_xy[1]:.2f})"
            )
            return "limit"
        else:
            self._target_replan_count += 1

        self._target_last_replan_time = now
        path, elapsed_ms = self._plan_target_path(pose)
        if not path:
            diag = self.coverage_planner.last_direct_path_diagnostics
            self.get_logger().warning(
                f"🧭 TARGET REPLAN failed (trigger={reason}, reason={diag.get('reason')}, "
                f"attempt={self._target_replan_count}, {elapsed_ms:.1f}ms)"
            )
            return "failed"

        # 새 경로가 지금 향하던 hop과 사실상 같으면 재발행하지 않는다 - 같은 목표를
        # 반복해서 다시 쏘면 base autonomy의 추종이 오히려 흔들린다.
        if self.current_goal is not None and self._same_hop(path[0], self.current_goal):
            self.target_route = deque(path[1:])
            self.get_logger().info(
                f"🧭 TARGET REPLAN (trigger={reason}) - path unchanged at current hop, "
                f"hops={len(self.target_route) + 1}, {elapsed_ms:.1f}ms"
            )
            return "unchanged"

        self.target_route = deque(path)
        self.get_logger().info(
            f"🧭 TARGET REPLAN (trigger={reason}, attempt={self._target_replan_count}) - "
            f"new path with {len(path)} hops, {elapsed_ms:.1f}ms"
        )
        self.publish_next_target_hop()
        return "replanned"

    def _plan_target_path(self, pose: dict) -> tuple[list[dict] | None, float]:
        """plan_direct_path 호출 + 소요 시간(ms) 측정.

        소요 시간을 재는 이유: 이 A*는 worker가 아니라 control_loop 스레드에서 도는데,
        목적지가 멀면(맵이 넓고 이미 많이 탐색된 상태) 탐색 노드가 늘어 control 주기
        (CONTROL_PERIOD_SEC=0.2초)를 잡아먹을 수 있다. 실제 맵에서 몇 ms가 나오는지
        로그로 남겨두면 worker 스레드로 옮길 필요가 있는지 실측으로 판단할 수 있다.
        """
        started = time.perf_counter()
        path = self.coverage_planner.plan_direct_path(
            pose,
            self.target_goal_xy,
            final_theta=self.target_final_theta,
            forbidden_mask=self.target_forbidden_mask,
            max_hop_spacing_m=config.TARGET_PATH_WAYPOINT_SPACING_M,
        )
        return path, (time.perf_counter() - started) * 1000.0

    def target_arrival_acceptable(self, pose: dict) -> bool:
        """정체 폴백에서도 최종 marker의 엄격한 성공 반경을 만족하는지."""
        return self.distance_to_target(pose) <= config.TARGET_ARRIVAL_FALLBACK_MAX_M

    def step_target_navigation(self, pose: dict) -> str:
        """목적지 주행 1 tick. 미션(mission2/mission3)이 매 control_loop마다 호출한다.

        반환:
          "driving"     - 계속 가는 중 (필요하면 이 안에서 경로를 다시 계산했다)
          "arrived"     - 최종 target marker의 TARGET_SUCCESS_DISTANCE_M 안에 도달.
                          정체 폴백도 이 반경을 완화하지 않는다.
          "unreachable" - 지금 지도로는 갈 방법이 없음. 미션이 탐사로 되돌릴지 결정한다.

        미션별로 도착 후 할 일만 다르고(mission2는 SUCCESS, mission3는 다음 step) 판단
        자체는 같아서 여기 한 곳에 둔다.
        """
        # 1. 목적지 도달 -> 끝.
        if self.target_destination_reached(pose):
            return "arrived"

        # 2. forbidden 경로일 때만 존재하는 중간 hop. 도착했으면 다음 hop을 낸다.
        if self.target_route and self.goal_reached(pose):
            self.publish_next_target_hop()
            return "driving"

        # 3. 주행 중 terrain이 갱신되어 지금 목표를 base autonomy가 못 받아들이게
        #    됐으면 접근 지점을 다시 고른다. 판정 근거가 우리 occupancy grid가 아니라
        #    /terrain_map인 게 핵심이다 - 로봇을 실제로 움직이는 쪽과 같은 데이터를
        #    봐야 판정이 의미가 있다.
        if self._retarget_if_unsupported(pose):
            return "driving"

        # 3.5. 재선택까지 했는데도 목표를 발행하지 못한 상태다. 이때는 로봇에게 아무
        #      명령도 가 있지 않으므로 기다려봐야 움직이지 않는다 - stuck timeout
        #      (20초)을 헛되이 소진하지 말고 바로 미션에 알린다. mission3는 탐사로
        #      되돌아가 지도를 넓히고(그 사이 terrain이 자라 다시 가능해질 수 있다),
        #      mission2는 자체 정책대로 처리한다.
        retarget_stuck = (
            self._target_retarget_fail_since is not None
            and time.monotonic() - self._target_retarget_fail_since
            >= config.TARGET_RETARGET_GIVEUP_SEC
        )
        # 재선정이 계속 실패해도 **로봇이 목표에 가까워지고 있으면** 포기하지 않는다.
        # goal_publisher의 예측 폴백(PREDICT_FALLBACK)이 생긴 뒤로는 "접근점 재선정은
        # 실패하지만 예측 지점을 향해 실제로 전진 중"인 상태가 정상 경로가 됐다.
        # 그때도 5초에 잘라버리면 폴백이 벌어준 전진을 그대로 버린다. 전진이 멈춘
        # 뒤에야 이 판정을 쓰고, 완전한 정지는 아래 20초 백스톱이 따로 잡는다.
        progressing = (
            self._target_goal_last_progress_time is not None
            and time.monotonic() - self._target_goal_last_progress_time
            < config.TARGET_RETARGET_GIVEUP_SEC
        )
        if self._target_publish_blocked or (retarget_stuck and not progressing):
            self._trace_navigation(
                "UNREACHABLE",
                f"trigger={'not_commandable' if self._target_publish_blocked else 'retarget_exhausted'} "
                f"goal=({self.target_goal_xy[0]:.2f},{self.target_goal_xy[1]:.2f}) "
                f"{self.terrain_monitor.last_selection}",
            )
            return "unreachable"

        # 4. 최후 백스톱. base autonomy가 스스로 멈춘 것이므로 "더 못 간다"는 판단은
        #    우리 지도가 아니라 로봇의 실제 거동에서 온다.
        if self.target_progress_stalled(pose):
            return self._accept_or_reject_arrival(
                pose, "stalled", "progress_stalled", "-"
            )

        return "driving"

    def _retarget_if_unsupported(self, pose: dict) -> bool:
        """지금 목표가 terrain 기준으로 못 쓰게 됐으면 접근 지점을 다시 골라 발행한다.

        재선택하려면 물체 좌표가 있어야 하므로 target_object_xy가 없으면(경유점 등)
        아무것도 하지 않는다. forbidden 경로(hop 주행 중)도 건너뛴다 - 그건 우리가
        만든 우회로라 여기서 덮으면 제약이 깨진다.
        """
        if self.target_object_xy is None or self.target_route or self.target_goal_xy is None:
            return False
        if not self.terrain_monitor.ready():
            return False
        if self.terrain_monitor.is_waypoint_supported(*self.target_goal_xy):
            return False

        now = time.monotonic()
        if (
            self._target_last_replan_time is not None
            and now - self._target_last_replan_time < config.TARGET_REPLAN_MIN_INTERVAL_SEC
        ):
            return False
        self._target_last_replan_time = now

        robot_xy = (float(pose["x"]), float(pose["y"]))
        chosen = self.terrain_monitor.choose_approach_point(self.target_object_xy, robot_xy)
        if chosen is None:
            if self._target_retarget_fail_since is None:
                self._target_retarget_fail_since = now
            self._trace_navigation(
                "RETARGET_FAIL",
                f"goal=({self.target_goal_xy[0]:.2f},{self.target_goal_xy[1]:.2f}) "
                f"{self.terrain_monitor.last_selection}",
            )
            return False

        theta = math.atan2(
            self.target_object_xy[1] - chosen[1], self.target_object_xy[0] - chosen[0]
        )
        self._trace_navigation(
            "RETARGET",
            f"({self.target_goal_xy[0]:.2f},{self.target_goal_xy[1]:.2f}) -> "
            f"({chosen[0]:.2f},{chosen[1]:.2f}) {self.terrain_monitor.last_selection}",
        )
        if not self._publish_target_goal(chosen[0], chosen[1], theta, is_final=True):
            if self._target_retarget_fail_since is None:
                self._target_retarget_fail_since = now
            # retarget이 고른 지점조차 발행이 안 됐다 - 목표를 바꾸지 않고 실패로 알린다.
            # 여기서 True를 돌려주면 호출 측이 "새 목표로 가는 중"이라고 착각한다.
            self._trace_navigation(
                "RETARGET_FAIL",
                f"chosen=({chosen[0]:.2f},{chosen[1]:.2f}) not commandable - "
                f"{self.terrain_monitor.last_selection}",
            )
            return False
        self.target_goal_xy = chosen
        self.target_final_theta = theta
        self._target_retarget_count += 1
        self._target_retarget_fail_since = None
        self.refresh_goal_marker()
        return True

    def refresh_goal_marker(self) -> None:
        """RViz goal 마커를 현재 목표 위치로 다시 그린다.

        add_step_goal_marker()는 마커 id를 step 인덱스로만 결정하므로, 같은 인덱스로
        다시 부르면 그 자리의 마커가 새 좌표로 교체된다. 목표를 옮기고 이걸 안 부르면
        마커만 옛 위치에 남아서, 눈으로 보는 거리와 로그의 dist_to_goal이 어긋난다.
        """
        if self.target_marker_index is None or self.target_goal_xy is None:
            return
        self.goal_publisher.add_step_goal_marker(
            self.target_marker_index,
            self.target_goal_xy[0],
            self.target_goal_xy[1],
            label=f"goal{self.target_marker_index + 1}",
            # "take the path between A and B"의 게이트 선분도 같은 토픽에 그린다 -
            # 통과 전 노랑/통과 후 초록이라 "정말 그 사이로 지나갔나"를 RViz에서 바로
            # 본다. 그리는 곳을 여기 한 곳으로 모아둔 이유는 위 docstring 참고.
            gate_segment=self.mission3_gate_segment,
            gate_crossed=self.mission3_gate_crossed,
        )

    def _publish_map_topics(self) -> None:
        """우리 planner가 보고 있는 지도/경로/frontier를 RViz로 내보낸다.

        base autonomy의 /registered_scan, /terrain_map, /way_point와 같은 map 프레임이라
        RViz에서 그대로 겹쳐진다 - "우리는 free로 보는데 terrain엔 점이 없는" 구간을
        눈으로 바로 대조하려고 만든 것이다. 발행 전용이라 주행에는 영향이 없다."""
        stamp = self.get_clock().now().to_msg()
        try:
            self._publish_occupancy(stamp)
            self._publish_planned_path(stamp)
            self._publish_frontier(stamp)
        except Exception as error:      # 진단용이라 실패가 주행을 막으면 안 된다
            self.get_logger().debug(f"map topic publish skipped: {error}")

    def _publish_occupancy(self, stamp) -> None:
        snapshot = self.coverage_planner.occupancy_snapshot()
        if snapshot is None:
            return
        grid, origin_x, origin_y, resolution = snapshot
        message = OccupancyGrid()
        message.header.stamp = stamp
        message.header.frame_id = config.OBJECT_MARKER_FRAME_ID
        message.info.resolution = float(resolution)
        message.info.width = int(grid.shape[1])
        message.info.height = int(grid.shape[0])
        message.info.origin.position.x = origin_x
        message.info.origin.position.y = origin_y
        message.info.origin.orientation.w = 1.0
        # OCC_* 값이 OccupancyGrid 규약(-1/0/100)과 같고, grid[row=y][col=x]에 원점이
        # 좌하단이라 축 배치도 그대로 맞는다. 변환 없이 평탄화만 하면 된다.
        # grid는 이미 int8이고 값도 -1/0/100이라 형변환 없이 평탄화만 하면 된다.
        message.data = grid.ravel().tolist()
        self.occupancy_pub.publish(message)

    def _publish_planned_path(self, stamp) -> None:
        """지금 따라가는 A* 경로. 목적지 주행이면 target_route, 탐색이면
        exploration_route를 쓴다. 로봇 현재 위치를 맨 앞에 붙여야 RViz에서 경로가
        로봇에서 이어져 보인다."""
        hops = list(self.target_route) or list(self.exploration_route)
        message = Path()
        message.header.stamp = stamp
        message.header.frame_id = config.OBJECT_MARKER_FRAME_ID
        with self.sensor_lock:
            pose = None if self.latest_pose is None else dict(self.latest_pose)
        goal = self.current_goal
        if not hops and goal:
            hops = [goal]
        points = ([(pose["x"], pose["y"])] if pose else []) + [
            (hop["x"], hop["y"]) for hop in hops
        ]
        for x, y in points:
            item = PoseStamped()
            item.header = message.header
            item.pose.position.x = float(x)
            item.pose.position.y = float(y)
            item.pose.orientation.w = 1.0
            message.poses.append(item)
        self.planned_path_pub.publish(message)

    def _publish_frontier(self, stamp) -> None:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = config.OBJECT_MARKER_FRAME_ID
        marker.ns = "sysnav_frontier"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = float(config.MAP_RESOLUTION_M)
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (1.0, 0.85, 0.0, 1.0)
        for x, y in self.coverage_planner.frontier_points_world():
            point = Point()
            point.x, point.y, point.z = float(x), float(y), 0.05
            marker.points.append(point)
        self.frontier_pub.publish(marker)

    def _trace_navigation(self, event: str, detail: str) -> None:
        """debug/sysnav_navigation_trace.txt에 한 줄 append (진단 전용, 동작 영향 없음).

        같은 내용을 활동 로그에도 넣는다 - 대시보드에서 "왜 멈췄는지"를 보려면 상태
        전이와 주행 판단이 한 타임라인에 섞여 있어야 한다."""
        activity.add(WARN if event in _NAV_PROBLEM_EVENTS else NAV,
                     _NAV_LABELS.get(event, event), detail)
        if not config.SAVE_DEBUG_IMAGES:
            return
        try:
            os.makedirs(config.DEBUG_DIR, exist_ok=True)
            with open(os.path.join(config.DEBUG_DIR, "sysnav_navigation_trace.txt"), "a") as file:
                file.write(f"{time.strftime('%H:%M:%S')} {event:12s} {detail}\n")
        except Exception:
            pass

    def _note_unreachable(self, pose: dict, trigger: str, status: str) -> str:
        """unreachable 사유를 한 줄로 모은다 - trigger/재계획 결과/A* 실패 사유가
        원래 서로 다른 줄에 흩어져 있어서 이어붙여야만 원인을 알 수 있었다."""
        diag = self.coverage_planner.last_direct_path_diagnostics
        self._target_unreachable_reason = (
            f"trigger={trigger} replan={status} planner={diag.get('reason')} "
            f"dist_to_goal={self.distance_to_target(pose):.2f}m "
            f"retargets={self._target_retarget_count} replans={self._target_replan_count}"
        )
        self._trace_navigation("UNREACHABLE", self._target_unreachable_reason)
        self.get_logger().warning(f"🧭 TARGET unreachable - {self._target_unreachable_reason}")
        return "unreachable"

    def _accept_or_reject_arrival(
        self, pose: dict, why: str, trigger: str = "?", status: str = "?"
    ) -> str:
        distance = self.distance_to_target(pose)
        if not self.target_arrival_acceptable(pose):
            return self._note_unreachable(pose, trigger, status)
        self._trace_navigation(
            "ARRIVED",
            f"{why} dist_to_goal={distance:.2f}m retargets={self._target_retarget_count}",
        )
        self.get_logger().info(
            f"🚩 ARRIVAL accepted at closest reachable point ({why}) - "
            f"{distance:.2f}m from goal (strict limit "
            f"{config.TARGET_SUCCESS_DISTANCE_M:.2f}m)"
        )
        return "arrived"

    def distance_to_target(self, pose: dict) -> float:
        if self.target_goal_xy is None:
            return float("inf")
        return math.hypot(
            self.target_goal_xy[0] - float(pose["x"]),
            self.target_goal_xy[1] - float(pose["y"]),
        )

    @staticmethod
    def _same_hop(hop: dict, goal: dict, tolerance_m: float = 0.10) -> bool:
        return math.hypot(
            float(hop["x"]) - float(goal["x"]), float(hop["y"]) - float(goal["y"])
        ) <= tolerance_m

    def publish_object_markers(self) -> None:
        snapshot = self.scene_graph.snapshot()
        markers = build_object_marker_array(
            snapshot["objects"],
            snapshot.get("selected_object_id"),
            self.get_clock().now().to_msg(),
        )
        self.object_marker_pub.publish(markers)

    # 디버깅용 mission_status_latest.html 갱신 - exploration_debug_latest.png와 같은
    # "항상 최신 상태 하나만 남기는" 패턴. MISSION_DASHBOARD_REFRESH_SEC로 스로틀링해서
    # control_loop(0.2초 주기)마다 디스크에 쓰지 않게 한다.
    def _update_mission_dashboard(self, state: str, task: dict | None, task_id: int) -> None:
        """대시보드 HTML을 다시 쓴다. **진단 전용이라 절대 주행을 막으면 안 된다.**

        control_loop(타이머 콜백)에서 불리는데, 여기서 예외가 나면 rclpy executor가
        그대로 노드를 죽인다. 실측(2026-08-23): avoid 절이 없는 Mission 3 질문에서
        mission_dashboard._mission_detail_rows의 forbidden_desc가 미할당이라
        UnboundLocalError가 났고, 그것 하나로 프로세스가 종료됐다(exit code 1).
        표시가 깨지는 것과 로봇이 멈추는 것은 심각도가 완전히 다르다.
        """
        now = time.monotonic()
        if now - self._last_dashboard_write_time < config.MISSION_DASHBOARD_REFRESH_SEC:
            return
        self._last_dashboard_write_time = now
        try:
            self._render_mission_dashboard(state, task, task_id)
        except Exception as error:
            self.get_logger().warning(
                f"mission dashboard skipped ({type(error).__name__}: {error})"
            )

    def _render_mission_dashboard(self, state: str, task: dict | None, task_id: int) -> None:
        now = time.monotonic()
        with self.sensor_lock:
            pose = None if self.latest_pose is None else dict(self.latest_pose)

        elapsed = None if self.task_start_time is None else now - self.task_start_time
        candidate_count = None
        if task and task.get("target"):
            candidate_count = len(self.object_memory.find_by_category(task["target"]))

        export_mission_dashboard({
            "task_id": task_id,
            "state": state,
            "task": task,
            "mission_type": (task or {}).get("mission_type"),
            "pose": pose,
            "elapsed_sec": elapsed,
            "current_goal": dict(self.current_goal) if self.current_goal else None,
            # 목적지 주행 진행 상황 - "로봇이 멈춰 있는데 왜 안 끝나는지"를 대시보드만
            # 보고 판단할 수 있게 남긴다(현재 hop이 아니라 진짜 목적지까지의 거리와
            # 남은 hop 수가 있어야 도착 판정 문제인지 경로 문제인지 구분된다).
            "target_goal_xy": self.target_goal_xy,
            "target_distance_m": None if pose is None else self.distance_to_target(pose),
            "target_hops_remaining": len(self.target_route),
            "target_replans": self._target_replan_count,
            # 목표까지의 거리를 대시보드에서 바로 읽기 위한 값들. 셋을 나란히 놔야
            # "왜 아직 도착이 아닌지"가 구분된다: 목적지(=접근 지점)는 도착 반경 안인데
            # 물체는 아직 먼지, 중간 hop을 도는 중인지, 아니면 아예 전진이 멈췄는지.
            "target_object_xy": self.target_object_xy,
            "target_object_distance_m": None if (pose is None or self.target_object_xy is None) else math.hypot(
                self.target_object_xy[0] - float(pose["x"]),
                self.target_object_xy[1] - float(pose["y"]),
            ),
            # 미션마다 도착 반경이 다르다(mission3는 1.0m, 그 외 0.5m) - 거리만 보고
            # "왜 아직인가"를 판단하려면 그 기준도 같이 보여야 한다.
            "target_success_radius_m": (
                config.MISSION3_TARGET_SUCCESS_DISTANCE_M
                if (task or {}).get("mission_type") == MISSION_INSTRUCTION_FOLLOWING
                else config.TARGET_SUCCESS_DISTANCE_M
            ),
            # 역대 최단거리와 그 뒤 경과 시간 - 값이 안 줄어든 채 시간만 흐르면
            # 로봇이 목표를 향해 실제로는 못 가고 있다는 뜻이다(stuck 판정과 같은 값).
            "target_best_distance_m": self._target_goal_best_distance_m,
            "target_no_progress_sec": None if self._target_goal_last_progress_time is None
            else now - self._target_goal_last_progress_time,
            "mission3_step_index": self.mission3_step_index,
            "mission3_forbidden_active": self.mission3_forbidden_mask is not None,
            "mission2_exploration_complete": self.mission2_exploration_complete,
            "last_response_summary": self.last_response_summary,
            "candidate_count": candidate_count,
            # base autonomy가 우리 목표를 얼마나 옮겼는지 - "우리 planner는 갈 수 있다고
            # 보는데 로봇이 안 간다"를 대시보드만 보고 구분하기 위한 값.
            "requested_waypoint_xy": self.goal_publisher.last_requested_xy,
            "actual_waypoint_xy": self.last_actual_waypoint_xy,
            "waypoint_displacement_m": self.last_waypoint_displacement_m,
            # 발행 직전 스냅(Layer 1)이 좌표를 옮긴 거리. 이게 작동하면 위
            # waypoint_displacement_m이 0에 수렴해야 한다.
            "waypoint_snap_m": self.goal_publisher.last_snap_distance_m,
            # 지도 현황 - "지금 얼마나 만들었고 아직 볼 곳이 남았는지"를 대시보드에서
            # 바로 보기 위한 것. 전부 개수 계산이라 1초 주기로 불러도 부담 없다.
            "map_stats": self.coverage_planner.map_stats(),
            "graph_counts": self.scene_graph.counts(),
            "terrain_summary": self.terrain_monitor.describe(),
            "object_memory_count": self.object_memory.count(),
            "viewpoint_memory_count": self.viewpoint_memory.count(),
        })

    def goal_reached(self, pose: dict) -> bool:
        if self.current_goal is None:
            return False
        return math.hypot(
            float(self.current_goal["x"]) - float(pose["x"]),
            float(self.current_goal["y"]) - float(pose["y"]),
        ) <= config.GOAL_REACHED_DISTANCE_M

    # exploration goal(x, y)이 벽 너머 등 실제로 도달 불가능한 지점일 때, goal_reached가 영원히
    # False로 남아 로봇이 계속 벽에 박혀있는 것을 막기 위한 진행도 기반 stuck 감지.
    # 목표까지 거리가 최근 EXPLORATION_STUCK_TIMEOUT_SEC 안에 EXPLORATION_STUCK_PROGRESS_M 이상
    # 줄어들지 않으면 도달 불가로 판단한다.
    def _track_goal_progress(
        self,
        pose: dict,
        best_distance_m: float | None,
        last_progress_time: float | None,
    ) -> tuple[float | None, float | None, float | None]:
        """current_goal까지의 "역대 최단거리"를 갱신하고 (거리, 최단거리, 마지막 갱신시각)을
        돌려준다.

        직전 대비가 아니라 역대 최단거리 대비로 재는 이유: 제자리에서 흔들리거나 벽을
        따라 좌우로 움직이는 건 진전이 아닌데, 직전 대비로 재면 그것도 진전으로 잡힌다.
        exploration/target 두 감시가 임계값만 다르고 로직은 같아서 여기로 모았다.
        """
        if self.current_goal is None:
            return None, best_distance_m, last_progress_time
        distance = math.hypot(
            float(self.current_goal["x"]) - float(pose["x"]),
            float(self.current_goal["y"]) - float(pose["y"]),
        )
        now = time.monotonic()
        if (
            best_distance_m is None
            or distance <= best_distance_m - config.EXPLORATION_STUCK_PROGRESS_M
        ):
            return distance, distance, now
        return distance, best_distance_m, last_progress_time

    def _exploration_goal_unreachable(
        self, pose: dict, timeout_sec: float = config.EXPLORATION_STUCK_TIMEOUT_SEC
    ) -> bool:
        distance, best, last_progress = self._track_goal_progress(
            pose,
            self._exploration_goal_best_distance_m,
            self._exploration_goal_last_progress_time,
        )
        self._exploration_goal_best_distance_m = best
        self._exploration_goal_last_progress_time = last_progress
        if distance is None or last_progress is None:
            return False
        return time.monotonic() - last_progress >= timeout_sec

    def destroy_node(self):
        self.worker.shutdown(wait=False, cancel_futures=True)
        self.map_worker.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()
