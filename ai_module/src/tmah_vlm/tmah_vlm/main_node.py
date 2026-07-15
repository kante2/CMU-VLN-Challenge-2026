#!/usr/bin/env python3
"""
TMAH VLM Node — 전체 파이프라인의 진입점. 처음 보는 사람은 이 파일부터 읽으면 된다.

실행 순서 (질문 하나가 처리되는 흐름):
  1. question_callback / pose_callback (common/callback.py),
     image_callback / scan_callback (sensor_process/callback.py)
     — 전부 "최신값 저장"만 한다. 무거운 VLM 추론은 여기서 절대 안 돈다.
  2. main_control_loop()            (config.MAIN_LOOP_PERIOD_SEC timer)
     pending question이 있고, 이미지/모델이 준비됐을 때만 파이프라인을 1회 실행한다.
  3. dispatch_question()            (question_process/dispatch.py)
     질문 첫 단어를 보고 아래 solver 중 하나로 분기한다(문장 → 미션 선택).
  4. t*_*_solver/ 가 실제 검출 -> 선택 -> 3D 위치 -> 발행까지 처리한다.
     (예: t3_object_reference.py의 process는 함수 이름 자체가 순서를 나타내는
     flat call sequence라서, 그 파일만 봐도 무슨 일이 일어나는지 순서대로 보인다.
     각 스텝 함수는 solver별 context 구조체(question_process/context.py)를 받아
     자기 필드를 채우고, 다음 함수가 그 필드를 읽어 이어서 쓴다.)

폴더 구조 (기능 단위로 묶여 있음):
  common/          센서 무관 노드 뼈대·수명주기. initialize(초기화/모델로딩/구독·발행 등록),
                   callback(question_callback·pose_callback), helpers(get_robot_pose 등 공용 상태 조회)
  question_process/ 문장 → 미션 선택 + 구조체 생성. dispatch(분기+처리준비 판단),
                   query_parser(질문→검출어), context(solver별 ctx 생성)
  t1_instruction_solver/     그 외 질문(instruction_process, 아직 stub)
  t2_numerical_solver/       "how many/count"(numerical_process)
  t3_object_reference_solver/ "find"(object_reference_process)
  sensor_process/  카메라→2Dbox→segmentation→LiDAR 좌표변환→ray→3D 위치.
                   sensor_process(공용 센서 스텝 흐름파일), detector/segmenter/selector,
                   image_utils, visualize, coordinate_transform, projector, scan_transform,
                   bbox_estimator/bbox_wireframe, callback(image·scan)
  nav_publish.py   nav/challenge로 나가는 발행 전부(waypoint/marker/count)
  reasoning/       누적 관측 기반 공간/관계 추론. spatial(관계 파싱·필터),
                   graph(scene graph 구축/시각화), sort3d(SORT3D-lite fallback)
  config.py        토픽명/프레임명/임계값/상수를 전부 여기 모아둠. 환경이 바뀌면 여기부터 확인.

이 파일 자체는 이제 node 로직을 직접 담지 않고 "조립"만 한다:
  1. Import
  2. Node class: __init__은 common/initialize.py의 함수들을 순서대로 호출
  3. Main control loop: main_control_loop() -> dispatch_question()
     (question_process/dispatch.py의 함수를 불러 쓴다)
  4. main(): 노드 생성 + main_control_loop 타이머 등록 + spin
"""

import rclpy
from rclpy.node import Node

from tmah_vlm import config
from tmah_vlm.common.initialize import (
    initialize_state,
    initialize_modules,
    initialize_subscribers,
    initialize_publishers,
    initialize_timers,
)
from tmah_vlm.question_process.dispatch import (
    # dispatch_question,
    peek_pending_question,
    ready_to_process,
    print_waiting_reason,
)
# 문장 → 미션 분기를 main_control_loop 안에 인라인으로 두기로 해서, solver 진입점을
# 여기서 직접 import 한다 (예전엔 dispatch.py가 이 세 개를 import 했다).
from tmah_vlm.t1_instruction_solver.t1_instruction import instruction_process
from tmah_vlm.t2_numerical_solver.t2_numerical import numerical_process
from tmah_vlm.t3_object_reference_solver.t3_object_reference import object_reference_process


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
        # Main control loop logic
        # 1. 대기
        if self.busy: # 설명 필요
            return
        # 아직 처리하지 않은 질문을 확인
        # 질문 콜백(common/callback.py)이 `node.pending_question`에 넣어둔 값을
        # main_control_loop이 매 주기마다 이 함수로 들여다본다.
        # 즉, 읽기만 하고 clear하지 않음(peek). 준비가 안 됐으면 다음 tick에서 다시 볼 수 있다.
        question = peek_pending_question(self)
        if question is None:
            return
        # 현재 질문을 처리할 준비가 됐는지(필요한 입력/모델이 다 준비됐는지) 확인
        if not ready_to_process(self, question):
            print_waiting_reason(self, question)
            return

        self.busy = True

        # 2. 실 처리
        try:
            self.get_logger().info("========================================")
            self.get_logger().info(f"[Pipeline] start: {question}")

            # question processing dispatch
            # question은 질문 문장자체(문자열)에 해당
            
            """
            질문 첫 단어 -> solver 매핑.t1~t3_solver 중 하나로 분기한다.

            "find ..."                    -> t3_object_reference_solver
            "how many ..." / "count ..."  -> t2_numerical_solver   (stub)
            그 외                          -> t1_instruction_solver (stub)
            """   

            # # Dispatch: 문장 → 미션 선택
            # dispatch_question(self, question)
            lower_question = question.lower()

            # 일부 질문에 하드하게 된 부분이 있어서 수정할 필요가 있어보임,
            if lower_question.startswith("find"):
                object_reference_process(self, question)
                # t3 object_reference_solver -저 물체를 찾아서 정확한 3D 위치를 짚기
            elif lower_question.startswith("how many") or lower_question.startswith("count"):
                numerical_process(self, question)
                # t2 numerical_solver -개수를 세라
            else:
                instruction_process(self, question)
                # t1 instruction_solver -그 외 질문(아직 stub)


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


# ========================================
# main
# ========================================

def main(args=None):
    rclpy.init(args=args)
    node = TmahVLM() # <- TmahVLM 노드를 생성
    node.create_timer(config.MAIN_LOOP_PERIOD_SEC, node.main_control_loop)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


'''
1. t3

pipeline: find ... 질문 처리 흐름
1.카메라 최신 프레임 스냅샷
2.질문에서 대상 단어 추출 (chair)
3.(공간관계 있으면) 누적 scene graph로 먼저 풀어보고 되면 조기 종료
4.GroundingDINO로 2D 후보 박스 검출
5.공간관계로 후보 좁히기 (near the window)
6.Qwen 비전 모델로 최종 후보 하나 선택
7.SAM으로 세그멘테이션 마스크
8.LiDAR ray로 3D 위치/크기 추정
9.접근 waypoint 계산 → marker(CUBE + wireframe) + waypoint 발행


2. t2. 

pipeline: how many ... / count ... 질문 처리 흐름
1.카메라 이미지 + GroundingDINO 검출 (t3의 앞부분 공유)
2.공간관계 있으면 deterministic 필터로 후보 걸러냄
3.남은 후보 개수 세기 → count 발행
→ t3의 부분집합. 단, 탐사는 안 하고 지금 보이는 화면 기준으로만 셈


3. t1.

로봇 현재 pose 읽기
로봇 앞 1m 지점을 waypoint로 계산 → 발행

'''