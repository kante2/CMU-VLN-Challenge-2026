#!/usr/bin/env python3
"""
TMAH VLM Node — 전체 파이프라인의 진입점. 처음 보는 사람은 이 파일부터 읽으면 된다.

실행 순서 (질문 하나가 처리되는 흐름):
  1. question_callback / pose_callback (common/callback.py),
     image_callback (perception/camera/callback.py), scan_callback (perception/lidar/callback.py)
     — 전부 "최신값 저장"만 한다. 무거운 VLM 추론은 여기서 절대 안 돈다.
  2. main_control_loop()            (0.2초 timer)
     pending question이 있고, 이미지/모델이 준비됐을 때만 파이프라인을 1회 실행한다.
  3. dispatch_question()
     질문 첫 단어를 보고 아래 solver 중 하나로 분기한다.
  4. t*_*_solver/ 가 실제 검출 -> 선택 -> 3D 위치 -> 발행까지 처리한다.
     (예: t3_object_reference.py의 process는 함수 이름 자체가 순서를 나타내는
     flat call sequence라서, 그 파일만 봐도 무슨 일이 일어나는지 순서대로 보인다.
     각 스텝 함수는 solver별 context 구조체(context/context.py)를 받아
     자기 필드를 채우고, 다음 함수가 그 필드를 읽어 이어서 쓴다.)

폴더 구조 (도메인별로 묶여 있음):
  common/      특정 센서에 안 속하는 노드 뼈대·수명주기. initialize(초기화/모델로딩/구독·발행 등록),
               callback(question_callback·pose_callback), helpers(get_robot_pose 등 공용 상태 조회)
  context/     solver별 작업변수 ctx 생성 함수(context.py) + 질문 대기열·처리 준비 판단(helpers.py)
  t1_instruction_solver/     그 외 질문(instruction_process, 아직 stub)
  t2_numerical_solver/       "how many/count"(numerical_process, 아직 stub)
  t3_object_reference_solver/ "find"(object_reference_process) + 발행(publish.py)
  perception/
    camera/    2D 인식: ROS Image 변환(image_utils), GroundingDINO 검출(detector),
               검출어 추출(query_parser), 디버그 저장(visualize), SAM 마스크(segmenter),
               Qwen 시각 선택(selector), image_callback
    lidar/     3D bbox 크기/wireframe(bbox_estimator, bbox_wireframe), scan_callback
  geometry/    3D 기하: 좌표 변환(coordinate_transform), 2D->3D 재투영(projector)
               — camera/lidar 둘 다 쓰는 공통 기하코드라 perception 밑으로 안 내림
  spatial/     공간관계 파싱·필터: relations, candidate_filter, relation_parser
  graph/       누적 scene graph 구축/렌더/시각화 (관계 질문의 근거)
  sort3d/      SORT3D-lite 추론 (scene graph 기반 관계 질문 fallback)
  config.py    토픽명/프레임명/임계값을 전부 여기 모아둠. 환경이 바뀌면 여기부터 확인.

이 파일 자체는 이제 node 로직을 직접 담지 않고 "조립"만 한다:
  1. Import
  2. Node class: __init__은 common/initialize.py의 함수들을 순서대로 호출
  3. Main control loop: main_control_loop() -> dispatch_question()
     (context/helpers.py의 함수를 불러 쓴다)
  4. main(): 노드 생성 + main_control_loop 타이머(0.2초) 등록 + spin
"""

import rclpy
from rclpy.node import Node

from tmah_vlm import config
from tmah_vlm.t1_instruction_solver.t1_instruction import instruction_process
from tmah_vlm.t2_numerical_solver.t2_numerical import numerical_process
from tmah_vlm.t3_object_reference_solver.t3_object_reference import object_reference_process
from tmah_vlm.common.initialize import (
    initialize_state,
    initialize_modules,
    initialize_subscribers,
    initialize_publishers,
    initialize_timers,
)
from tmah_vlm.context.helpers import (
    peek_pending_question,
    ready_to_process,
    print_waiting_reason,
)


# ========================================
# Node
# ========================================

class TmahVLM(Node): # <- TmahVLM클래스는 ROS2 Node(상속받을 클래스) 를 상속받아 TmahVLM 객체는 ROS2 노드 기능을 사용할 수 있다.
    def __init__(self): # <- self는 현재 만들어진 객체 자기 자신
        super().__init__("tmah_vlm")

        self.cfg = config

        # initialize -> state, modules, subscribers, publishers, timers
        initialize_state(self)
        initialize_modules(self)
        initialize_subscribers(self)
        initialize_publishers(self)
        initialize_timers(self)

        self.get_logger().info("TMAH VLM node started")
        self.get_logger().info("Waiting for /challenge_question ...")

    # ========================================
    # Main control loop
    # ========================================

    def main_control_loop(self):
        """
        C++ 코드의 mainControlLoop 역할. 0.2초마다 불린다.

        센서 callback은 계속 최신값만 갱신하고, 실제 pipeline은 여기서 시작된다:
          1. 이미 처리 중이면 스킵
          2. 처리할 질문이 없으면 스킵
          3. 이미지/모델이 아직 준비 안 됐으면 대기 (이유를 로그로)
          4. dispatch_question()으로 질문 유형별 handler 실행
          5. 처리 끝난 질문을 pending에서 제거
        """

        # ========================================
        # 
        if self.busy: # 설명 필요
            return
        # 아직 처리하지 않은 질문을 확인
        question = peek_pending_question(self)
        if question is None:
            return
        # 현재 질문을 처리할 준비가 됐는지(필요한 입력/모델이 다 준비됐는지) 확인
        if not ready_to_process(self, question):
            print_waiting_reason(self, question)
            return

        self.busy = True

        try:
            self.get_logger().info("========================================")
            self.get_logger().info(f"[Pipeline] start: {question}")

            self.dispatch_question(question)

            self.get_logger().info("[Pipeline] finished")

            with self.state_lock:
                if self.pending_question == question:
                    self.pending_question = None

        except Exception as error:
            self.get_logger().error(f"[Pipeline] failed: {error}")

            with self.state_lock:
                if self.pending_question == question:
                    self.pending_question = None

        finally:
            self.busy = False
            self.get_logger().info("Waiting for /challenge_question ...")

    def dispatch_question(self, question):
        """
        질문 첫 단어 -> solver 매핑.

          "find ..."                    -> t3_object_reference_solver
          "how many ..." / "count ..."  -> t2_numerical_solver   (stub)
          그 외                          -> t1_instruction_solver (stub)
        """
        lower_question = question.lower()

        # 일부 질문에 하드하게 된 부분이 있어서 수정할 필요(0709확인) -> **
        if lower_question.startswith("find"):
            object_reference_process(self, question)
        elif lower_question.startswith("how many") or lower_question.startswith("count"):
            numerical_process(self, question)
        else:
            instruction_process(self, question)


# ========================================
# main
# ========================================

def main(args=None):
    rclpy.init(args=args)
    node = TmahVLM() # <- TmahVLM 노드를 생성
    node.create_timer(0.2, node.main_control_loop)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
