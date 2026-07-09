#!/usr/bin/env python3
"""
TMAH VLM Node — 전체 파이프라인의 진입점. 처음 보는 사람은 이 파일부터 읽으면 된다.

실행 순서 (질문 하나가 처리되는 흐름):
  1. question_callback / image_callback / scan_callback / pose_callback
     (callback/sensor_callbacks.py) — 전부 "최신값 저장"만 한다.
     무거운 VLM 추론은 여기서 절대 안 돈다.
  2. main_control_loop()            (0.2초 timer)
     pending question이 있고, 이미지/모델이 준비됐을 때만 파이프라인을 1회 실행한다.
  3. dispatch_question()
     질문 첫 단어를 보고 아래 handler 중 하나로 분기한다.
  4. handlers/*.py 가 실제 검출 -> 선택 -> 3D 위치 -> 발행까지 처리한다.
     (예: object_reference.py는 함수 이름 자체가 순서를 나타내는 flat call
     sequence라서, 그 파일만 봐도 무슨 일이 일어나는지 순서대로 보인다.)

폴더 구조 (기능별로 분리돼 있음):
  initialize/  TmahVLM.__init__()이 부르는 initialize_state/modules/
               subscribers/publishers/timers, 모델 백그라운드 로딩(load_models)
  callback/    센서/질문 콜백. 최신값 저장만 하고 무거운 처리 없음
  helper/      main_control_loop와 handlers/*.py가 공용으로 쓰는 node 상태 조회
               (pending question 확인, robot pose, image-scan 시각 동기화, heartbeat)
  handlers/    질문 유형별 진입점. object_reference.py(find) / numerical.py(count,
               아직 stub) / instruction.py(그 외, 아직 stub)
  perception/  센서 원본 -> 쓸 수 있는 형태. ROS Image 변환(image_utils),
               GroundingDINO 2D 검출(detector), 질문에서 검출어 추출(query_parser),
               디버그 이미지 저장(visualize)
  grounding/   2D 검출 결과를 3D 위치로 연결. panorama pixel <-> camera ray,
               point cloud를 이미지에 재투영해서 detection box와 매칭(projector)
  bbox3d/      선택된 물체에 해당하는 point들로 3D bounding box(중심/크기) 추정
               (estimator) — RViz marker 크기를 실측값으로 표시하는 데 씀
  reasoning/   Qwen2.5-VL로 후보 중 하나를 선택(selector) — 색/공간관계 같은
               "이미지 보고 판단해야 하는" 부분은 여기서 처리
  tf/          map/sensor/camera 좌표 변환. 실시간 TF 우선, 실패 시
               config.STATIC_TF_FALLBACKS로 대체(coordinate_transform)
  config.py    토픽명/프레임명/임계값을 전부 여기 모아둠. 환경이 바뀌면 여기부터 확인.

이 파일 자체는 이제 initialize/callback/helper 로직을 직접 담지 않고 "조립"만 한다:
  1. Import
  2. Node class: __init__은 initialize/setup.py의 함수들을 순서대로 호출
  3. Main control loop: main_control_loop() -> dispatch_question()
     (helper/node_helpers.py의 함수를 불러 쓴다)
  4. main()
"""

import rclpy
from rclpy.node import Node

from tmah_vlm import config
from tmah_vlm.handlers import object_reference, numerical, instruction
from tmah_vlm.initialize.setup import (
    initialize_state,
    initialize_modules,
    initialize_subscribers,
    initialize_publishers,
    initialize_timers,
)
from tmah_vlm.helper.node_helpers import (
    peek_pending_question,
    ready_to_process,
    print_waiting_reason,
)


# ========================================
# Node
# ========================================

class TmahVLM(Node):
    def __init__(self):
        super().__init__("tmah_vlm")

        self.cfg = config

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
        if self.busy:
            return

        question = peek_pending_question(self)
        if question is None:
            return

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
        질문 첫 단어 -> handler 매핑.

          "find ..."                    -> handlers/object_reference.py
          "how many ..." / "count ..."  -> handlers/numerical.py   (stub)
          그 외                          -> handlers/instruction.py (stub)
        """
        lower_question = question.lower()

        if lower_question.startswith("find"):
            object_reference.object_reference_process(self, question)
        elif lower_question.startswith("how many") or lower_question.startswith("count"):
            numerical.numerical_process(self, question)
        else:
            instruction.instruction_process(self, question)


# ========================================
# main
# ========================================

def main(args=None):
    rclpy.init(args=args)
    node = TmahVLM()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
