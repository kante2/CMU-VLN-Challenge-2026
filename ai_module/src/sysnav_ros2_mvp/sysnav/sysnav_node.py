"""ROS2 orchestration node for the single-room SysNav MVP.

Callbacks only cache messages. Heavy perception, Gemini and exploration jobs run in
worker threads and are coordinated by a timer-driven state machine.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import math
import threading
import time

from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Int32, String
from visualization_msgs.msg import Marker, MarkerArray

from sysnav import config
from sysnav.exploration.coverage_planner import CoveragePlanner
from sysnav.exploration.exploration_visualizer import export_exploration_debug
from sysnav.exploration.viewpoint_memory import ViewpointMemory
from sysnav.missions import mission1_pipe, mission2_pipe, mission3_pipe
from sysnav.rooms.room_registry import RoomRegistry
from sysnav.rooms.room_segmenter import RoomSegmenter
from sysnav.rooms.room_visualizer import export_room_segmentation
from sysnav.memory.object_memory import ObjectMemory
from sysnav.navigation.goal_publisher import GoalPublisher
from sysnav.perception.perception_pipeline import PerceptionPipeline
from sysnav.reasoning.attribute_verifier import AttributeVerifier
from sysnav.reasoning.gemini_selector import GeminiSelector
from sysnav.reasoning.room_classifier import RoomClassifier
from sysnav.scene_graph.scene_graph_manager import SceneGraphManager
from sysnav.scene_graph.scene_graph_rviz import build_object_marker_array
from sysnav.ros_helpers import (
    closest_stamped_item,
    image_msg_to_rgb,
    message_stamp_to_sec,
    odometry_to_pose,
    pointcloud2_to_xyz,
)
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
        self.object_memory = ObjectMemory()
        # Room/Viewpoint/Object node와 edge를 관리한다. Viewpoint는 매 프레임이 아니라
        # novel LiDAR voxel coverage가 충분할 때만 생성하며 debug graph를 갱신한다.
        self.scene_graph = SceneGraphManager(debug_dir=config.DEBUG_DIR)
        self.selector = GeminiSelector()
        self.attribute_verifier = AttributeVerifier()
        self.coverage_planner = CoveragePlanner()
        self.room_segmenter = RoomSegmenter()
        self.room_registry = RoomRegistry()
        self.room_classifier = RoomClassifier()
        self.viewpoint_memory = ViewpointMemory()
        self.goal_publisher = GoalPublisher(self)
        self.object_marker_pub = self.create_publisher(MarkerArray, config.TOPIC_OBJECT_MARKERS, 10)
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

        # Mission 3(Instruction-Following, missions/mission3_pipe.py) 전용 상태 -
        # 여러 목적지를 순서대로 처리해야 해서 mission2/1과 달리 진행 인덱스가
        # 필요하다. 다른 미션에서는 안 쓰이므로 매 새 질문마다 리셋만 하면 무해하다.
        self.mission3_step_index = 0
        self.mission3_leg_queue = deque()
        self.mission3_forbidden_mask = None

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
        self.control_timer = self.create_timer(
            config.CONTROL_PERIOD_SEC,
            self.control_loop,
            callback_group=self.callback_group,
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

        with self.state_lock: # 읽는 도중 콜백으로 덮어쓰지 않도록 lock을 걸어준다.
            self.task_id += 1
            self.task = parsed
            self.state = "OBSERVE"
            self.current_goal = None
            self.exploration_route.clear()
            self.last_processed_image_stamp = -1.0
            self.mission3_step_index = 0
            self.mission3_leg_queue.clear()
            self.mission3_forbidden_mask = None

        if not config.KEEP_MEMORY_BETWEEN_TASKS:
            self.object_memory.clear()
            self.scene_graph.clear()
            self.viewpoint_memory.clear()
            with self.sensor_lock:
                pose = None if self.latest_pose is None else dict(self.latest_pose)
            self.coverage_planner.reset(pose)
        self.scene_graph.start_task(self.task_id, parsed)
        self.publish_object_markers()

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
        # 목표 객체 후보 검색
        candidates = self.object_memory.find_by_category(task["target"]) # Object Memory에서 어떤 종류의 객체를 후보로 가져올지 결정

        # 문장에 spatial constraint가 있고 Scene Graph에 검증된 Object-Object edge가
        # 존재하면, 해당 edge의 source object만 우선 후보로 사용한다.
        relation_candidate_ids = set(self.scene_graph.find_matching_target_ids(task))
        if relation_candidate_ids:
            candidates = [
                candidate
                for candidate in candidates
                if int(candidate["object_id"]) in relation_candidate_ids
            ]
        elif effective_relation_chain(task):
            # 문장에 relation 제약(예: "knife rack 근처의")이 있는데 아직 그 제약을
            # 만족하는 candidate가 하나도 검증 안 됐다 - 보통 참조 물체(knife rack)를
            # 아직 못 봤기 때문. 검증 안 된 채로 아무 bowl이나 확정지으면 오답으로
            # navigation을 끝내버리게 되므로, 여기서 확정하지 않고 계속 탐색하게 한다.
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

    def publish_object_markers(self) -> None:
        snapshot = self.scene_graph.snapshot()
        markers = build_object_marker_array(
            snapshot["objects"],
            snapshot.get("selected_object_id"),
            self.get_clock().now().to_msg(),
        )
        self.object_marker_pub.publish(markers)

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
    def _exploration_goal_unreachable(self, pose: dict) -> bool:
        if self.current_goal is None:
            return False
        distance = math.hypot(
            float(self.current_goal["x"]) - float(pose["x"]),
            float(self.current_goal["y"]) - float(pose["y"]),
        )
        now = time.monotonic()
        if (
            self._exploration_goal_best_distance_m is None
            or distance <= self._exploration_goal_best_distance_m - config.EXPLORATION_STUCK_PROGRESS_M
        ):
            self._exploration_goal_best_distance_m = distance
            self._exploration_goal_last_progress_time = now
            return False
        assert self._exploration_goal_last_progress_time is not None
        return now - self._exploration_goal_last_progress_time >= config.EXPLORATION_STUCK_TIMEOUT_SEC

    def destroy_node(self):
        self.worker.shutdown(wait=False, cancel_futures=True)
        self.map_worker.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()
