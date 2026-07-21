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
from std_msgs.msg import String

from sysnav import config
from sysnav.exploration.coverage_planner import CoveragePlanner
from sysnav.exploration.viewpoint_memory import ViewpointMemory
from sysnav.memory.object_memory import ObjectMemory
from sysnav.navigation.goal_publisher import GoalPublisher
from sysnav.perception.perception_pipeline import PerceptionPipeline
from sysnav.reasoning.gemini_selector import GeminiSelector
from sysnav.ros_helpers import (
    closest_stamped_item,
    image_msg_to_rgb,
    message_stamp_to_sec,
    odometry_to_pose,
    pointcloud2_to_xyz,
)
from sysnav.task.query_parser import extract_target

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
        self.object_memory = ObjectMemory()
        self.selector = GeminiSelector()
        self.coverage_planner = CoveragePlanner()
        self.viewpoint_memory = ViewpointMemory()
        self.goal_publisher = GoalPublisher(self)

        self.current_goal: dict | None = None
        self.exploration_route = deque()

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
        parsed = extract_target(msg.data)
        if not parsed["target"]:
            self.get_logger().error(f"Could not extract target object: {msg.data}")
            return

        with self.state_lock: # 읽는 도중 콜백으로 덮어쓰지 않도록 lock을 걸어준다.
            self.task_id += 1
            self.task = parsed
            self.state = "OBSERVE"
            self.current_goal = None
            self.exploration_route.clear()
            self.last_processed_image_stamp = -1.0

        if not config.KEEP_MEMORY_BETWEEN_TASKS:
            self.object_memory.clear()
        self.viewpoint_memory.clear()

        with self.sensor_lock:
            pose = None if self.latest_pose is None else dict(self.latest_pose)
        self.coverage_planner.reset(pose)

        self.get_logger().info(
            f"Task #{self.task_id}: target={parsed['target']}, "
            f"attributes={parsed['attributes']}, relation={parsed['relation']}, "
            f"references={parsed['reference_objects']}"
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
        observations = self.perception.process( # 실제 객체 인식 파이프라인 실행
            image_rgb=image_msg_to_rgb(image_msg), # ROS image -> numpy
            points_sensor=pointcloud2_to_xyz(scan_msg), # pointcloud -> numpy
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
        self.object_memory.update(observations, timestamp=image_stamp)
        return {
            "task_id": task_id,
            "image_stamp": image_stamp,
            "candidates": self.object_memory.find_by_category(task["target"]),
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

    def selection_job(self, task_id: int, task: dict, pose: dict) -> dict:
        candidates = self.object_memory.find_by_category(task["target"])
        selected_id = self.selector.select(
            question=task["raw"],
            candidates=candidates,
            context_objects=self.object_memory.all_nodes(),
            robot_pose=pose,
        )
        return {"task_id": task_id, "selected_id": selected_id}

    def exploration_job(self, task_id: int, pose: dict) -> dict:
        return {
            "task_id": task_id,
            "route": self.coverage_planner.plan_route(pose, self.viewpoint_memory),
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

    def consume_future(self) -> None:
        if self.active_future is None or not self.active_future.done():
            return

        future = self.active_future
        kind = self.active_kind
        expected_task_id = self.active_task_id
        origin_state = self.active_origin_state
        self.active_future = None
        self.active_kind = None
        self.active_task_id = None
        self.active_origin_state = None

        try:
            result = future.result()
        except Exception as error:
            self.get_logger().error(f"{kind} job failed: {error}")
            with self.state_lock:
                if kind == "perception":
                    self.state = "FOLLOW_EXPLORATION" if origin_state == "FOLLOW_EXPLORATION" else "PLAN_EXPLORATION"
                elif kind == "selection":
                    self.state = "PLAN_EXPLORATION"
                else:
                    self.state = "FAILED"
            return

        if expected_task_id != self.task_id or result.get("task_id") != self.task_id:
            return

        if kind == "perception":
            self.last_processed_image_stamp = float(result["image_stamp"])
            self.last_perception_wall_time = time.monotonic()
            if result["candidates"]:
                with self.state_lock:
                    self.state = "SELECT_TARGET"
                    self.exploration_route.clear()
            elif origin_state == "OBSERVE":
                with self.state_lock:
                    self.state = "PLAN_EXPLORATION"

        elif kind == "selection":
            selected = self.object_memory.get(result["selected_id"])
            with self.sensor_lock:
                pose = None if self.latest_pose is None else dict(self.latest_pose)
            if selected is None or pose is None:
                with self.state_lock:
                    self.state = "PLAN_EXPLORATION"
                return
            x, y, theta = self.goal_publisher.object_approach_pose(pose, selected["position"])
            self.goal_publisher.publish(x, y, theta)
            self.current_goal = {
                "x": x,
                "y": y,
                "theta": theta,
                "type": "target",
                "object_id": selected["object_id"],
            }
            with self.state_lock:
                self.state = "NAVIGATE_TARGET"
            self.get_logger().info(
                f"Selected object_id={selected['object_id']}, "
                f"goal=({x:.2f}, {y:.2f}, {theta:.2f})"
            )

        elif kind == "exploration":
            route = result["route"]
            if not route:
                with self.state_lock:
                    self.state = "FAILED"
                self.get_logger().warning("No reachable frontier remains")
                return
            self.exploration_route = deque(route)
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

        if task is None or state in {"IDLE", "SUCCESS", "FAILED"}:
            return
        if self.active_future is not None:
            return

        with self.sensor_lock:
            pose = None if self.latest_pose is None else dict(self.latest_pose)
        if pose is None:
            return

        if state == "NAVIGATE_TARGET":
            if self.goal_reached(pose):
                with self.state_lock:
                    self.state = "SUCCESS"
                self.get_logger().info("Target navigation completed")
            return

        if state == "FOLLOW_EXPLORATION":
            if self.goal_reached(pose):
                if self.current_goal is not None:
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

        if state == "SELECT_TARGET":
            self.submit_job(
                "selection",
                self.selection_job,
                task_id,
                task,
                pose,
                origin_state=state,
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

    def publish_next_exploration_goal(self) -> None:
        if not self.exploration_route:
            self.current_goal = None
            with self.state_lock:
                self.state = "OBSERVE"
            return
        goal = self.exploration_route.popleft()
        self.goal_publisher.publish(goal["x"], goal["y"], goal["theta"])
        self.current_goal = {**goal, "type": "exploration"}
        with self.state_lock:
            self.state = "FOLLOW_EXPLORATION"
        self.get_logger().info(
            f"Exploration goal=({goal['x']:.2f}, {goal['y']:.2f}, {goal['theta']:.2f}), "
            f"coverage={goal.get('coverage_score', 0)}"
        )

    def goal_reached(self, pose: dict) -> bool:
        if self.current_goal is None:
            return False
        return math.hypot(
            float(self.current_goal["x"]) - float(pose["x"]),
            float(self.current_goal["y"]) - float(pose["y"]),
        ) <= config.GOAL_REACHED_DISTANCE_M

    def destroy_node(self):
        self.worker.shutdown(wait=False, cancel_futures=True)
        self.map_worker.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()
