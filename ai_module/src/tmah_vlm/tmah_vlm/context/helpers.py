#!/usr/bin/env python3
"""
main_control_loop()이 쓰는 "처리할 질문이 있는지 / 처리할 준비가 됐는지" 판단 헬퍼.

node를 인자로 받는 자유 함수라 solver들의 *_process(node, ...)와 같은 패턴이다.
ctx 팩토리 함수(make_*_context)는 context/context.py에 있다.
"""

import time


def peek_pending_question(node):
    """아직 처리하지 않은 질문을 확인한다.

    질문 콜백(common/callback.py)이 `node.pending_question`에 넣어둔 값을
    main_control_loop이 매 tick마다 이 함수로 들여다본다.
    값을 소비(clear)하지 않고 읽기만(peek) 하므로, 아직 처리할 준비가 안 됐으면 다음 tick에서 다시 볼 수 있다.
    콜백 스레드와 제어 루프가 동시에 접근하므로 state_lock으로 보호한다.
    """
    with node.state_lock:  #<--
        return node.pending_question


def ready_to_process(node, question):
    """현재 질문을 처리할 준비가 됐는지(필요한 입력/모델이 다 준비됐는지) 확인한다.

    준비가 안 됐으면 False를 반환해서 main_control_loop이 이번 질문을 처리하지 않고
    다음 tick으로 미루게 한다. 판정 기준:
      - 카메라 이미지는 무조건 있어야 한다(모든 처리의 기본 입력).
      - "find ..." 형태의 질문은 GroundingDINO(detector)가 로드돼 있어야 한다.
    """
    lower_question = question.lower()

    if node.latest_image is None:
        return False

    if lower_question.startswith("find") and node.detector is None:
        return False

    # selector는 없으면 0번 후보를 선택하는 fallback이 있으므로 필수는 아니다.
    return True


def print_waiting_reason(node, question):
    """준비가 덜 됐을 때(ready_to_process가 False) 무엇을 기다리는지 로그로 알려준다.

    제어 루프는 매 tick마다 돌기 때문에 그대로 찍으면 로그가 도배된다. 그래서
    node.last_wait_log_time으로 마지막 출력 시각을 기억해 3초에 한 번만 출력한다.
    ready_to_process와 같은 조건을 검사해서 부족한 항목만 골라 문자열로 합쳐 warn한다.
    """
    now = time.time()
    if now - node.last_wait_log_time < 3.0:
        return
    node.last_wait_log_time = now

    reasons = []
    lower_question = question.lower()

    if node.latest_image is None:
        reasons.append("camera image")
    if lower_question.startswith("find") and node.detector is None:
        reasons.append("GroundingDINO")

    reason_text = ", ".join(reasons)
    node.get_logger().warn(f"[Pipeline] waiting for: {reason_text}")
