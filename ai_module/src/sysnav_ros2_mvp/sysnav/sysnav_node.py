"""ROS2 orchestration node for the single-room SysNav MVP.

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

from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Int32, String
from visualization_msgs.msg import Marker, MarkerArray

from sysnav import config
from sysnav.exploration.coverage_planner import CoveragePlanner
from sysnav.exploration.exploration_visualizer import export_exploration_debug
from sysnav.exploration.viewpoint_memory import ViewpointMemory
from sysnav.mission_dashboard import export_mission_dashboard
from sysnav.missions import mission1_pipe, mission2_pipe, mission3_pipe
from sysnav.missions.mission3_rviz import build_step_marker_array
from sysnav.rooms import cross_room_navigator
from sysnav.rooms.room_registry import RoomRegistry
from sysnav.rooms.room_segmenter import RoomSegmenter
from sysnav.rooms.room_visualizer import export_room_segmentation
from sysnav.memory.object_memory import ObjectMemory, filter_reliable
from sysnav.navigation.goal_publisher import GoalPublisher
from sysnav.navigation.terrain_monitor import TerrainMonitor
from sysnav.perception.perception_pipeline import PerceptionPipeline
from sysnav.reasoning.attribute_verifier import AttributeVerifier
from sysnav.reasoning.gemini_selector import GeminiSelector
from sysnav.reasoning.relation_image_verifier import RelationImageVerifier
from sysnav.reasoning.room_classifier import RoomClassifier
from sysnav.reasoning.room_relevance_selector import RoomRelevanceSelector
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
from sysnav.task.llm_visual_aliases import LLMVisualAliasExpander
from sysnav.task.mission_classifier import (
    MISSION_INSTRUCTION_FOLLOWING,
    MISSION_NUMERICAL,
    MISSION_OBJECT_REFERENCE,
    classify_mission,
)
from sysnav.task.query_parser import effective_relation_chain, requires_comparative_ranking

# state 이름 -> 처리할 mission pipe 모듈. 미션에 없는 state로 잘못 분기되지 않도록
# question_callback에서 항상 task["mission_type"]을 이 dict의 키 중 하나로 채운다.
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


class SysNavNode(Node):
    def __init__(self) -> None:
        super().__init__("sysnav_node")

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
        self.last_processed_image_stamp = -1.0
        self.last_perception_wall_time = 0.0

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
        self.visual_alias_expander = LLMVisualAliasExpander()
        self.object_memory = ObjectMemory()
        # Room/Viewpoint/Object node와 edge를 관리한다. Viewpoint는 매 프레임이 아니라
        # novel LiDAR voxel coverage가 충분할 때만 생성하며 debug graph를 갱신한다.
        self.scene_graph = SceneGraphManager(debug_dir=config.DEBUG_DIR)
        self.selector = GeminiSelector()
        self.attribute_verifier = AttributeVerifier()
        self.relation_image_verifier = RelationImageVerifier()
        self.vlm_counter = VlmCounter()
        self.coverage_planner = CoveragePlanner()
        self.room_segmenter = RoomSegmenter()
        self.room_registry = RoomRegistry()
        self.room_classifier = RoomClassifier()
        self.room_relevance_selector = RoomRelevanceSelector()
        self.viewpoint_memory = ViewpointMemory()
        self.goal_publisher = GoalPublisher(self)
        self.terrain_monitor = TerrainMonitor()
        self.object_marker_pub = self.create_publisher(MarkerArray, config.TOPIC_OBJECT_MARKERS, 10)
        marker_qos = QoSProfile(depth=1)
        marker_qos.reliability = ReliabilityPolicy.RELIABLE
        marker_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.mission3_step_marker_pub = self.create_publisher(
            MarkerArray, config.TOPIC_MISSION3_STEP_MARKERS, marker_qos
        )
        # 채점 대상 토픽(README) - Object Reference/Numerical. 절대 이름/타입을 바꾸지
        # 말 것 (Marker 단수, MarkerArray 아님 - CLAUDE.md 하드-룰).
        self.selected_object_marker_pub = self.create_publisher(
            Marker, config.TOPIC_SELECTED_OBJECT_MARKER, 10
        )
        self.numerical_response_pub = self.create_publisher(
            Int32, config.TOPIC_NUMERICAL_RESPONSE, 10
        )

        self.current_goal: dict | None = None
        self.exploration_route = deque()
        self._latest_room_segmentation: dict | None = None
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
        self._target_republish_count = 0
        self._target_unreachable_reason: str | None = None

        # Mission 3(Instruction-Following, missions/mission3_pipe.py) 전용 상태 -
        # 여러 목적지를 순서대로 처리해야 해서 mission2/1과 달리 진행 인덱스가
        # 필요하다. 다른 미션에서는 안 쓰이므로 매 새 질문마다 리셋만 하면 무해하다.
        self.mission3_step_index = 0
        self.mission3_forbidden_mask = None
        self.mission3_step_destinations: list[dict] = []
        self.mission3_recovery_points: dict[int, list[tuple[float, float]]] = {}

        # Cross-room navigation(rooms/cross_room_navigator.py) - 이번 task 안에서
        # 이미 시도해본(가거나, 갔는데 경로가 안 됐던) room_id. 새 질문마다 리셋.
        self._cross_room_attempted_ids: set[int] = set()

        # 디버깅용 미션 상태 대시보드(mission_dashboard.py)용 상태.
        self.task_start_time: float | None = None
        self.last_response_summary: str | None = None
        self._last_dashboard_write_time = 0.0

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
        self.control_timer = self.create_timer(
            config.CONTROL_PERIOD_SEC,
            self.control_loop,
            callback_group=self.control_callback_group,
        )
        self.get_logger().info("SysNav single-room MVP started")

    # ------------------------------------------------------------------
    # ROS callbacks
    '''
    self.state_lock: # 읽는 도중 콜백으로 덮어쓰지 않도록 lock을 걸어준다.
    '''
    # ------------------------------------------------------------------

    def question_callback(self, msg: String) -> None:
        # 문장이 세 미션(Numerical/Object Reference/Instruction-Following) 중 어디로
        # 가야 하는지부터 정한다 - 응답 형식/상태머신이 미션마다 완전히 다르다
        # (MISSION_1/2/3_*_CLAUDE.txt 참고).
        mission_type = classify_mission(msg.data)
        if mission_type == MISSION_INSTRUCTION_FOLLOWING:
            # 다단계 목적지 + 경로 제약 문장이라 단일 target G=(c_tgt,Φ) 파서로는
            # 못 담는다 - 절 단위로 쪼개서 목적지 절마다 같은 LLMQueryParser를 재사용.
            parsed = mission3_pipe.parse_instruction(self, msg.data)
            is_valid = bool(parsed.get("steps"))
        else:
            # SysNav paper Sec. III의 G=(c_tgt, Φ) 파싱을 LLM이 하고, 실패하면 항상
            # 규칙 기반 query_parser.extract_target()로 자동 폴백한다.
            parsed = self.query_parser.parse(msg.data)
            is_valid = bool(parsed.get("target"))
        parsed["mission_type"] = mission_type

        if not is_valid:
            self.get_logger().error(f"Could not parse question ({mission_type}): {msg.data}")
            return

        # Expand only detector vocabulary. All downstream task constraints keep
        # their canonical categories, and perception maps alias hits back before
        # object memory sees them.
        expanded_prompts, canonical_by_prompt, visual_aliases = (
            self.visual_alias_expander.expand(
                msg.data, list(parsed.get("detection_prompts") or [])
            )
        )
        parsed["detection_prompts"] = expanded_prompts
        parsed["canonical_by_detection_prompt"] = canonical_by_prompt
        parsed["visual_aliases"] = visual_aliases

        with self.state_lock: # 읽는 도중 콜백으로 덮어쓰지 않도록 lock을 걸어준다.
            self.task_id += 1
            self.task = parsed
            self.state = "OBSERVE"
            self.current_goal = None
            self.exploration_route.clear()
            self.clear_target_navigation()
            self.last_processed_image_stamp = -1.0
            self.mission3_step_index = 0
            self.mission3_forbidden_mask = None
            self.mission3_step_destinations.clear()
            self.mission3_recovery_points.clear()
            self._cross_room_attempted_ids = set()
            self.task_start_time = time.monotonic()
            self.last_response_summary = None

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
                f"📩 NEW QUESTION [{mission_type}] - Task #{self.task_id}: \"{msg.data}\" -> "
                f"steps={parsed['steps']}"
            )
        else:
            self.get_logger().info(
                f"📩 NEW QUESTION [{mission_type}] - Task #{self.task_id}: \"{msg.data}\" -> "
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
        self._update_room_segmentation()
        self._update_exploration_debug(pose)

    # Room Node segmentation (SysNav paper Sec. IV-A-1) - 매핑이 갱신될 때마다 같이
    # 갱신하고, room_segmentation_latest.png를 scene_graph_latest.png와 같은 패턴으로
    # (append가 아니라 매번 통째로 다시 그려서) 덮어쓴다. exploration용
    # self._latest_room_segmentation(room_scoped sampling에 쓰임, 사이클마다 room_id가
    # 바뀌어도 무방 - 그 사이클 안에서만 일관되면 됨)과, 시각화/분류용 RoomRegistry(사이클
    # 간에도 room_id가 안정적으로 유지되어야 category를 이어붙일 수 있음)는 서로 다른
    # 목적이라 별도로 관리한다.
    def _update_room_segmentation(self) -> None:
        grid = self.coverage_planner.snapshot_grid()
        max_height = self.coverage_planner.snapshot_max_height()
        result = self.room_segmenter.segment(grid, max_height=max_height)
        self._latest_room_segmentation = result

        viewpoints = self.scene_graph.list_viewpoints()
        registry_result = self.room_registry.update(
            segmentation=result,
            viewpoints=viewpoints,
            world_to_grid=self.coverage_planner.world_to_grid,
        )
        self._classify_pending_rooms()
        export_room_segmentation(grid, registry_result)

    def _classify_pending_rooms(self) -> None:
        if not config.ROOM_CLASSIFICATION_ENABLED:
            return
        pending = self.room_registry.rooms_needing_classification()
        if not pending:
            return
        categories = self.room_classifier.classify_many(pending)
        for room in pending:
            room_id = room["room_id"]
            if room_id in categories:
                self.room_registry.set_category(room_id, categories[room_id])
            else:
                self.room_registry.mark_classification_failed(room_id)

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

    def sensor_snapshot(self):
        with self.sensor_lock: 
            # 이 블록 안에서 센서 데이터를 읽는 동안 다른 callback이 같은 센서 변수에 접근하는 것을 잠시 막는다.
            # 블록이 끝나면 lock은 자동으로 해제된다.
            if self.latest_image is None or self.latest_pose is None:
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
                return None
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
            canonical_by_prompt=dict(task.get("canonical_by_detection_prompt") or {}),
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
        observed_object_ids = self.object_memory.update(observations, timestamp=image_stamp)
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
    def selection_job(self, task_id: int, task: dict, pose: dict) -> dict:
        # 후보를 고르기 전에 갈라진 노드를 합친다 - 같은 물체가 여러 개로 남아 있으면
        # "nearest" 같은 비교 판정이 같은 물체끼리 경쟁하는 꼴이 된다.
        merged = self.object_memory.merge_duplicates()
        if merged:
            self.get_logger().info(f"🧹 MERGED {merged} duplicate object node(s)")

        # 목표 객체 후보 검색
        candidates = self.object_memory.find_by_category(task["target"]) # Object Memory에서 어떤 종류의 객체를 후보로 가져올지 결정
        candidates, dropped = filter_reliable(candidates)
        if dropped:
            self.get_logger().info(f"🧹 DROPPED {dropped} low-confidence candidate(s)")

        # 문장에 spatial constraint가 있고 Scene Graph에 검증된 Object-Object edge가
        # 존재하면, 해당 edge의 source object만 우선 후보로 사용한다. mission3는 절마다
        # 독립된 relation을 가진 step task를 여기로 직접 넘기는데, add_observation은
        # 최상위 placeholder task(relation 없음) 기준으로만 edge를 갱신하므로 이 task의
        # relation은 그쪽에서 절대 잡히지 않는다 - 여기서 먼저 직접 시도해서 채운다
        # (mission1/2는 이미 add_observation이 같은 task로 채워놨을 것이므로 대부분
        # _relation_checks 캐시에 걸려 사실상 공짜다).
        if effective_relation_chain(task):
            self.scene_graph.infer_relations_for_task(task, pose)
        relation_candidate_ids = set(self.scene_graph.find_matching_target_ids(task))
        if relation_candidate_ids:
            candidates = [
                candidate
                for candidate in candidates
                if int(candidate["object_id"]) in relation_candidate_ids
            ]
        elif requires_comparative_ranking(task) and len(candidates) < 2:
            # closest/nearest is an argmin.  A lone object is only the nearest
            # seen so far, so it cannot become a destination until exploration
            # exhaustion explicitly creates the unique-candidate relation edge.
            return {"task_id": task_id, "selected_id": None, "relation_pending": True}
        elif effective_relation_chain(task):
            # 문장에 relation 제약(예: "knife rack 근처의")이 있는데 geometric/
            # co-observation 경로로는 아직 하나도 검증 안 됨 - 보통 참조 물체를
            # 아직 못 봤거나(전역 위치 없음), 유리창처럼 LiDAR grounding이 구조적으로
            # 실패해서(approximate 등급조차 못 만듦) 3D 위치 자체가 없기 때문.
            image_verified_ids: set[int] = set()
            relation_chain = effective_relation_chain(task)
            # A crop-based fallback can validate one local relation, but it
            # cannot prove a nested chain. For A->B->C, every graph edge must
            # exist; otherwise selecting A would silently ignore B->C.
            if len(relation_chain) > 1:
                return {"task_id": task_id, "selected_id": None, "relation_pending": True}
            if candidates:
                _, first_relation, first_reference = relation_chain[0]
                if first_relation in ("nearest", "closest") and len(candidates) > 1:
                    # 참조 물체가 3D로 잡혀있으면(대부분의 경우) 단순 유클리드 거리
                    # 비교로 확정한다 - 결정적이고 VLM 호출도 필요 없다. 이미지 기반
                    # rank_nearest()는 "참조 물체가 후보 사진에 우연히 같이 찍혀야만"
                    # 성립하는 훨씬 약한 조건이라, 참조 물체가 대부분의 후보에서 안
                    # 보이면(카메라 FOV 밖) reference_visible_in_any=false로 매번
                    # 실패해서 relation이 영원히 확정 안 되는 문제가 있었다 - 유리창
                    # 처럼 LiDAR 반사가 안 돼 3D 위치 자체가 없는 경우에만 이 이미지
                    # 비교로 폴백한다.
                    references = self.object_memory.find_by_category(first_reference)
                    if references:
                        winner = min(
                            candidates,
                            key=lambda candidate: min(
                                math.hypot(
                                    candidate["position"][0] - reference["position"][0],
                                    candidate["position"][1] - reference["position"][1],
                                )
                                for reference in references
                            ),
                        )
                        image_verified_ids = {int(winner["object_id"])}
                    else:
                        # "nearest"는 최상급(비교) relation이라 후보마다 독립적으로
                        # yes/no만 물으면 안 된다 - bedside table이 2개 있고 둘 다
                        # 사진에 창문이 보이면 verify()는 둘 다 통과시켜버려서 어느 게
                        # 진짜 가까운지 못 가린다. 후보 전부를 한 번에 놓고 VLM이
                        # 직접 비교해서 하나만 고르게 한다.
                        winner_id = self.relation_image_verifier.rank_nearest(candidates, first_reference)
                        if winner_id is not None:
                            image_verified_ids = {winner_id}
                else:
                    # 참조 물체를 3D로 잡을 필요 없이, 후보 자신의 사진만으로 "이 사진에
                    # 참조 물체가 보이는가"를 VLM에게 직접 확인받는다 (attribute_verifier와
                    # 같은 on-demand 이미지 판정 패턴).
                    image_verified_ids = self.relation_image_verifier.verify(
                        candidates, first_relation, first_reference
                    )
            if image_verified_ids:
                candidates = [
                    candidate for candidate in candidates
                    if int(candidate["object_id"]) in image_verified_ids
                ]
            else:
                # 후보가 아직 없거나, 이미지 확인도 실패(또는 검증 안 됨) - 확정하지
                # 않고 계속 탐색해서 후보/참조 물체를 더 찾아보거나 다른 각도에서
                # 다시 시도한다.
                return {"task_id": task_id, "selected_id": None, "relation_pending": True}

        # SysNav paper Sec. IV-A-1 (self-attribute): 문장에 속성 제약(예: "black" chair)이
        # 있으면, 후보가 1개뿐이어도 반드시 VLM으로 확인한다 - "후보가 하나뿐이면 그냥
        # 확정"하던 예전 GeminiSelector 지름길이 색을 전혀 안 보고 넘어가버리는 원인이었다.
        attributes = list(task.get("attributes") or [])
        if attributes and config.ATTRIBUTE_VERIFICATION_ENABLED and candidates:
            attribute_results = self.attribute_verifier.verify(candidates, attributes)
            for candidate in candidates:
                newly_checked = attribute_results.get(int(candidate["object_id"]), {})
                if newly_checked:
                    self.object_memory.update_self_attributes(int(candidate["object_id"]), newly_checked)
            candidates = [
                candidate for candidate in candidates
                if all(
                    attribute_results.get(int(candidate["object_id"]), {}).get(attribute, False)
                    for attribute in attributes
                )
            ]
            if not candidates:
                # 속성이 확인된 후보가 하나도 없다(전부 불일치했거나 아직 검증 자체가
                # 안 됨) - 확정하지 않고 계속 탐색해서 진짜 맞는 물체를 더 찾아본다.
                return {"task_id": task_id, "selected_id": None, "attribute_pending": True}

        # GeminiSelector()
        selected_id = self.selector.select(
            question=task["raw"], # Gemini가 원본 문장을 그대로 이해하도록 전달
            candidates=candidates,
            # 전체 object node 가져오기 - Object Memory에 저장된 모든 객체를 가져
            context_objects=self.object_memory.all_nodes(),
            robot_pose=pose,
        )
        return {
            "task_id": task_id, # 현재 처리중인 질문 번호 / worker가 어느 질문인지 확인하기 위함.
            "selected_id": selected_id,
            "relation_pending": False,
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
        return {
            "task_id": task_id,
            "route": self.coverage_planner.plan_route(
                pose, self.viewpoint_memory, room_segmentation=self._latest_room_segmentation
            ),
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
            result = future.result() # WORKER가 반환한 값을 .result() 을 통해서 가져온다.
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

        # Cross-room navigation(SysNav paper Sec. IV-B-2, room-query navigation mode) -
        # exploration job이 "이 방(또는 지금 알려진 전체 영역)엔 더 볼 게 없다"는
        # 빈 route를 반환하면, 미션별 최종 처리(카운트 확정/FAILED)로 바로 넘기기
        # 전에 아직 안 들어가본 방이 있는지부터 확인한다. 있으면 거기로 가는 job을
        # 새로 제출하고 이번 사이클엔 미션 쪽에 알리지 않는다 - 안 가본 방이 남아있는데
        # 성급하게 끝내면 안 되니까(특히 Numerical의 카운트 정확도에 직결).
        if kind == "cross_room_select":
            self._on_cross_room_select_result(task, expected_task_id, result)
            return
        if kind == "exploration" and not result.get("route"):
            if self._try_start_cross_room_navigation(task, expected_task_id):
                return

        mission_pipe = _MISSION_PIPES.get(
            (task or {}).get("mission_type", MISSION_OBJECT_REFERENCE), mission2_pipe
        )
        mission_pipe.on_job_result(self, task, kind, result, origin_state)

    def _try_start_cross_room_navigation(self, task: dict | None, task_id: int) -> bool:
        with self.sensor_lock:
            pose = None if self.latest_pose is None else dict(self.latest_pose)
        if pose is None or task is None:
            return False
        unvisited = self.room_registry.unvisited_rooms()
        candidates = [
            room for room in unvisited
            if room["room_id"] not in self._cross_room_attempted_ids
        ]
        # "cross-room이 왜 아무것도 안 했는지"를 로그 없이는 확인할 방법이 없었다 -
        # known_room_count가 1이면 애초에 room segmentation이 이 씬에서 방을 하나로만
        # 봤다는 뜻(문 통과를 한 번도 못 했거나, 다른 방이 core 임계값을 못 넘었거나).
        self.get_logger().info(
            f"🚪 CROSS-ROOM check - known_rooms={self.room_registry.known_room_count()}, "
            f"unvisited={len(unvisited)}, "
            f"already_attempted_this_task={len(self._cross_room_attempted_ids)}, "
            f"usable_candidates={len(candidates)}"
        )
        if not candidates:
            return False
        self.submit_job(
            "cross_room_select",
            cross_room_navigator.select_job,
            self, task_id, task, pose, candidates,
            origin_state="PLAN_EXPLORATION",
        )
        return True

    def _on_cross_room_select_result(self, task: dict | None, task_id: int, result: dict) -> None:
        for room_id in result.get("failed_room_ids", []):
            self._cross_room_attempted_ids.add(int(room_id))
        room_id = result.get("room_id")
        path = result.get("path")
        if room_id is None or not path:
            # 안 가본 방이 있었지만 전부 경로를 못 찾음(또는 애초에 없었음) - 원래
            # exploration이 비어있던 상황으로 돌려서 미션별 최종 처리로 넘긴다.
            mission_pipe = _MISSION_PIPES.get(
                (task or {}).get("mission_type", MISSION_OBJECT_REFERENCE), mission2_pipe
            )
            mission_pipe.on_job_result(
                self, task, "exploration", {"task_id": task_id, "route": []}, "PLAN_EXPLORATION"
            )
            return
        self._cross_room_attempted_ids.add(int(room_id))
        self.get_logger().info(f"🚪 CROSS-ROOM - heading to unvisited room_id={room_id}")
        self.exploration_route = deque(path)
        self.publish_next_exploration_goal()

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def control_loop(self) -> None:
        self.consume_future()

        with self.state_lock:
            state = self.state
            task = None if self.task is None else dict(self.task)
            task_id = self.task_id

        self._update_mission_dashboard(state, task, task_id)

        if task is None or state in {"IDLE", "SUCCESS", "FAILED"}:
            return
        if self.active_future is not None:
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
                # 도달 실패한 지점도 방문한 것으로 취급해서 같은/근처 후보를 다시 뽑지 않게 한다.
                self.viewpoint_memory.add(
                    self.current_goal["x"],
                    self.current_goal["y"],
                    self.current_goal["theta"],
                    self.current_goal.get("coverage_score"),
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
                return
            image_msg, scan_msg, synced_pose, image_stamp = snapshot
            if image_stamp <= self.last_processed_image_stamp:
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

    # state == "FOLLOW_EXPLORATION" -> publish next exploration goal
    def publish_next_exploration_goal(self) -> None:
        if not self.exploration_route:
            self.current_goal = None
            with self.state_lock:
                self.state = "OBSERVE"
            return
        goal = self.exploration_route.popleft()
        self.goal_publisher.publish(goal["x"], goal["y"], goal["theta"])
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
        self._target_republish_count = 0

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
        self._publish_target_goal(
            self.target_goal_xy[0], self.target_goal_xy[1], self.target_final_theta,
            is_final=True,
        )

    def publish_next_target_hop(self) -> None:
        """target_route에서 다음 hop을 꺼내 발행한다. state는 건드리지 않는다
        (미션별 state는 missions/*.py가 관리한다)."""
        if not self.target_route:
            return
        hop = self.target_route.popleft()
        self._publish_target_goal(
            hop["x"], hop["y"], hop["theta"], is_final=not self.target_route
        )

    def _publish_target_goal(self, x: float, y: float, theta: float, is_final: bool) -> None:
        self.goal_publisher.publish(x, y, theta)
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

    def target_destination_reached(self, pose: dict) -> bool:
        """최종 목적지(마지막 hop이 아니라 목적지 좌표 자체)에 도달했는지."""
        if self.target_goal_xy is None:
            return False
        return math.hypot(
            self.target_goal_xy[0] - float(pose["x"]),
            self.target_goal_xy[1] - float(pose["y"]),
        ) <= config.GOAL_REACHED_DISTANCE_M

    def approach_pose_for(self, pose: dict, object_position) -> tuple[float, float, float]:
        """물체로 접근할 (x, y, theta)를 정한다. mission2/3가 공용으로 쓴다.

        1순위는 /terrain_map 기준으로 base autonomy가 받아들일 지점(TerrainMonitor).
        terrain 데이터가 없거나 통과 지점을 못 찾으면 기존 고정 standoff 방식으로
        폴백한다 - terrain 판정 실패가 주행 자체를 막으면 안 된다.
        """
        object_xy = (float(object_position[0]), float(object_position[1]))
        robot_xy = (float(pose["x"]), float(pose["y"]))
        chosen = self.terrain_monitor.choose_approach_point(object_xy, robot_xy)
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
        """목적지 판정 반경 밖이지만 "여기가 갈 수 있는 최선"일 때 도달로 인정할지.

        목적지가 가구 옆이면 경로 계획이 목표를 통행 가능한 셀로 최대 2m 스냅하므로
        GOAL_REACHED_DISTANCE_M(0.35m) 안으로는 애초에 들어갈 수 없다. 이때 도착을
        고집하면 로봇은 멈춰 있는데 미션만 영원히 안 끝난다. 다만 아무 데서나 끝났다고
        하면 안 되므로 TARGET_ARRIVAL_FALLBACK_MAX_M 안일 때만 인정한다.
        """
        if self.target_goal_xy is None:
            return False
        return math.hypot(
            self.target_goal_xy[0] - float(pose["x"]),
            self.target_goal_xy[1] - float(pose["y"]),
        ) <= config.TARGET_ARRIVAL_FALLBACK_MAX_M

    def step_target_navigation(self, pose: dict) -> str:
        """목적지 주행 1 tick. 미션(mission2/mission3)이 매 control_loop마다 호출한다.

        반환:
          "driving"     - 계속 가는 중 (필요하면 이 안에서 경로를 다시 계산했다)
          "arrived"     - 목적지 도달. 목적지 판정 반경 안이거나, "지금 지도로는 여기가
                          최선"이 확인되고 TARGET_ARRIVAL_FALLBACK_MAX_M 안인 경우.
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

        # 4. 최후 백스톱. base autonomy가 스스로 멈춘 것이므로 "더 못 간다"는 판단은
        #    우리 지도가 아니라 로봇의 실제 거동에서 온다.
        if self.target_progress_stalled(pose):
            # 포기하기 전에 같은 goal을 한 번 다시 쏴본다. base autonomy가 어떤 이유로든
            # (waypointConverter의 재타게팅, 메시지 유실 등) 목표를 놓았을 수 있는데,
            # 재발행은 공짜이고 실패해도 잃는 게 없다. 그래도 진전이 없으면 그때 판정한다.
            if self._target_republish_count < config.TARGET_REPUBLISH_MAX_COUNT:
                self._target_republish_count += 1
                goal = self.current_goal or {}
                self._trace_navigation(
                    "REPUBLISH",
                    f"#{self._target_republish_count} goal=({goal.get('x', 0.0):.2f},"
                    f"{goal.get('y', 0.0):.2f}) dist_to_goal={self.distance_to_target(pose):.2f}m",
                )
                self.get_logger().warning(
                    f"♻️ TARGET GOAL STILL ACTIVE - no recent progress; re-publishing "
                    f"({goal.get('x', 0.0):.2f}, {goal.get('y', 0.0):.2f})"
                )
                self.goal_publisher.publish(
                    goal.get("x", 0.0), goal.get("y", 0.0), goal.get("theta", 0.0)
                )
                # 진행도 감시만 새로 시작한다. 최단거리 기록은 유지해서 pose 노이즈가
                # 진전으로 둔갑하지 않게 한다.
                self._target_goal_last_progress_time = time.monotonic()
                return "driving"
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
        self.target_goal_xy = chosen
        self.target_final_theta = theta
        self._target_retarget_count += 1
        self._publish_target_goal(chosen[0], chosen[1], theta, is_final=True)
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
        )

    def _trace_navigation(self, event: str, detail: str) -> None:
        """debug/sysnav_navigation_trace.txt에 한 줄 append (진단 전용, 동작 영향 없음)."""
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
            f"{distance:.2f}m from goal (limit {config.TARGET_ARRIVAL_FALLBACK_MAX_M:.2f}m)"
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

    def publish_mission3_step_destination(self, step_index: int, point, label: str) -> None:
        """Remember and publish a resolved instruction-following destination."""
        destination = {
            "step_index": int(step_index),
            "position": tuple(float(value) for value in point),
            "label": str(label),
        }
        self.mission3_step_destinations = [
            item for item in self.mission3_step_destinations
            if item["step_index"] != destination["step_index"]
        ]
        self.mission3_step_destinations.append(destination)
        self.mission3_step_destinations.sort(key=lambda item: item["step_index"])
        self.publish_mission3_step_markers()

    def publish_mission3_step_markers(self) -> None:
        markers = build_step_marker_array(
            self.mission3_step_destinations,
            self.get_clock().now().to_msg(),
        )
        self.mission3_step_marker_pub.publish(markers)

    # 디버깅용 mission_status_latest.html 갱신 - room_segmentation_latest.png와 같은
    # "항상 최신 상태 하나만 남기는" 패턴. MISSION_DASHBOARD_REFRESH_SEC로 스로틀링해서
    # control_loop(0.2초 주기)마다 디스크에 쓰지 않게 한다.
    def _update_mission_dashboard(self, state: str, task: dict | None, task_id: int) -> None:
        now = time.monotonic()
        if now - self._last_dashboard_write_time < config.MISSION_DASHBOARD_REFRESH_SEC:
            return
        self._last_dashboard_write_time = now

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
            "mission3_step_index": self.mission3_step_index,
            "mission3_forbidden_active": self.mission3_forbidden_mask is not None,
            "last_response_summary": self.last_response_summary,
            "candidate_count": candidate_count,
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
