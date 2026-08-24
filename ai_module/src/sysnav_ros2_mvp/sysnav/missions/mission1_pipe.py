"""Mission 1 - Numerical (MISSION_1numerical_CLAUDE.txt).

Object Reference와 문장 파싱(target/attributes/relation_chain)은 동일한
LLMQueryParser를 재사용하지만, 종료 조건이 근본적으로 다르다: candidate를
찾아도 멈추지 않고 탐색이 완전히 끝날 때까지(plan_route()가 빈 route를 반환할
때까지) 계속 돌다가, 그 시점에 최종 후보 개수를 세어 `/numerical_response`
(Int32)로 발행한다.

state 흐름: OBSERVE/PLAN_EXPLORATION/FOLLOW_EXPLORATION(공용, sysnav_node.py) ->
(exploration exhausted) -> MISSION1_FINALIZE_COUNT -> SUCCESS.

탐색 종료 조건: coverage_planner.plan_route()가 빈 route를 반환하는 시점
(_on_exploration_result 참고) - Object Reference는 같은 상황을 FAILED로 보지만,
Numerical은 "더 볼 곳이 없다 = 최종 개수를 확정할 시점"으로 정상 종료 처리한다.

개수를 정하는 방법이 두 겹이다:
  1. 기하 기반 - object_memory 노드를 카테고리/relation/attribute로 걸러 len()을 센다.
     탐지 재현율이 그대로 상한이 된다(못 본 물체는 못 센다).
  2. VLM 기반(reasoning/vlm_counter.py, 기본 활성) - 대상 물체를 가장 많이 담은
     viewpoint 파노라마 한 장을 Gemini에게 보여 직접 세게 한다. 1의 상한을 넘기 위한
     것이고, 실패하면 조용히 1의 값을 쓴다(fail-quiet).
"""


from __future__ import annotations

import time
from collections import deque

from std_msgs.msg import Int32

from sysnav import config
from sysnav.reasoning.attribute_filter import filter_by_attributes
from sysnav.task.query_parser import effective_relation_chain


def loop(node, state: str, task: dict, task_id: int, pose: dict) -> None:
    if state == "MISSION1_FINALIZE_COUNT":
        node.submit_job("count", count_job, node, task_id, task, origin_state=state)


def maybe_force_count_at_deadline(node, state: str) -> bool:
    """시간 예산을 넘기면 참조 물체를 못 찾았어도 지금 가진 근거로 집계한다.

    아래 _on_exploration_result의 게이트("참조 카테고리를 다 볼 때까지 집계 금지")는
    참조 물체가 끝내 안 보이면 영원히 안 끝난다. 무응답은 0점이라 최악이므로
    (count_job docstring 참고) 여기서 반드시 탈출구를 만든다 - mission2의
    maybe_force_selection_at_deadline과 같은 취지/같은 호출 위치다."""
    if state not in {"OBSERVE", "PLAN_EXPLORATION", "FOLLOW_EXPLORATION"}:
        return False
    started = node.task_start_time
    if started is None or time.monotonic() - started < config.MISSION1_EXPLORATION_TIME_LIMIT_SEC:
        return False
    node.exploration_route.clear()
    node.current_goal = None
    with node.state_lock:
        node.state = "MISSION1_FINALIZE_COUNT"
    node.get_logger().warning(
        f"⏰ MISSION 1 EXPLORATION DEADLINE - "
        f"{config.MISSION1_EXPLORATION_TIME_LIMIT_SEC:.0f}s elapsed; "
        "counting with the evidence gathered so far"
    )
    return True


def _missing_categories(node, task: dict) -> list[str]:
    """이 질문을 집계하려면 무엇이 관측돼 있어야 하는가.

    detection_prompts는 target + 관계 체인의 모든 참조 물체의 합집합이다
    ("chairs near the table with a vase on it"이면 chair/table/vase 셋 다). 이 중
    하나라도 못 봤으면 관계 판정 자체가 성립할 수 없다 - 예를 들어 vase를 한 번도
    못 본 상태에서 세면 "vase가 있는 table"이 아니라 "아무 table" 옆 chair를 세게 된다.
    Mission 3의 같은 이름 함수(missions/mission3_pipe.py)와 동일한 판정이다."""
    return [
        category for category in (task.get("detection_prompts") or [])
        if not node.object_memory.find_by_category(category)
    ]


def _log_missing_categories(node, missing: list[str]) -> None:
    now = time.monotonic()
    last = getattr(node, "_mission1_last_missing_log", 0.0)
    if now - last < config.MISSION1_MISSING_LOG_INTERVAL_SEC:
        return
    node._mission1_last_missing_log = now
    node.get_logger().info(
        f"🔍 아직 못 본 참조 물체가 있어 집계를 보류한다: {', '.join(missing)} "
        "- 재관측하며 기다린다"
    )


def on_job_result(node, task: dict, kind: str, result: dict, origin_state: str) -> None:
    if kind == "perception":
        _on_perception_result(node, origin_state)
    elif kind == "exploration":
        _on_exploration_result(node, task, result)
    elif kind == "count":
        _on_count_result(node, result)


def _count_settled(node, task: dict | None) -> bool:
    """관계 체인이 확정됐고 집계 대상 개수도 더 안 변하면 True.

    Mission 1은 원래 "frontier가 하나도 안 남을 때까지" 돌았다. 그런데 "the table with
    a vase on it"처럼 참조가 유일하게 확정되는 질문은, 그 테이블 주변을 다 본 순간부터
    남은 탐색이 답을 바꾸지 못한다 - 그때부터는 시간만 쓴다(실측 2026-08-24: 관계가
    잡힌 뒤에도 계속 OBSERVE/탐색을 반복).

    다만 "확정 즉시 종료"는 위험하다. 관계가 확정된 순간엔 아직 테이블 반대편을 못 봐서
    의자가 3개만 잡혀 있을 수 있다(GT 8개). 그래서 개수가
    MISSION1_SETTLED_STABLE_OBSERVATIONS번 연속 그대로일 때만 끊는다.
    """
    if not task:
        return False
    if not effective_relation_chain(task):
        # 관계가 없는 질문("How many chairs?")은 방 전체가 대상이라 조기 종료 근거가
        # 없다 - 예전대로 탐색을 끝까지 돈다.
        return False
    matched = node.scene_graph.find_matching_target_ids(task)
    if not matched:
        node._mission1_stable_streak = 0
        node._mission1_last_count = None
        return False

    count = len(matched)
    if count == getattr(node, "_mission1_last_count", None):
        node._mission1_stable_streak = getattr(node, "_mission1_stable_streak", 0) + 1
    else:
        node._mission1_last_count = count
        node._mission1_stable_streak = 1

    needed = config.MISSION1_SETTLED_STABLE_OBSERVATIONS
    if node._mission1_stable_streak < needed:
        node.get_logger().info(
            f"🔢 관계 확정됨(대상 {count}개) - 개수 안정화 확인 중 "
            f"{node._mission1_stable_streak}/{needed}"
        )
        return False
    node.get_logger().info(
        f"🏁 관계 체인이 확정되고 대상 개수({count})가 {needed}회 연속 동일 - "
        "탐색을 끝내고 집계한다"
    )
    return True


def _on_perception_result(node, origin_state: str) -> None:
    # candidate를 찾아도 절대 멈추지 않는다 - 전체 탐색이 끝나야 정확한 개수를 셀 수
    # 있다(Object Reference와의 핵심 차이). OBSERVE에서 온 관측만 다음 탐색을
    # 계획하고, FOLLOW_EXPLORATION 중 관측(perception-while-moving)은 object_memory만
    # 갱신하고 이동 자체는 방해하지 않는다.
    if origin_state == "OBSERVE":
        with node.state_lock:
            task = None if node.task is None else dict(node.task)
        if _count_settled(node, task):
            node.exploration_route.clear()
            node.current_goal = None
            with node.state_lock:
                node.state = "MISSION1_FINALIZE_COUNT"
            return
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"


def _on_exploration_result(node, task: dict | None, result: dict) -> None:
    route = result["route"]
    if not route:
        # frontier가 없다고 바로 세면 안 된다 - 관계의 참조 물체를 아직 하나도 못
        # 봤다면 그 제약을 적용할 수가 없어서, 제약 없는 숫자(예: "아무 table 옆
        # chair")를 답으로 내보내게 된다. frontier가 없어도 제자리 재관측으로 잡히는
        # 물체가 있으므로, 데드라인(maybe_force_count_at_deadline)까지는 OBSERVE로
        # 되돌아가 계속 본다.
        missing = _missing_categories(node, task or {})
        if missing:
            _log_missing_categories(node, missing)
            with node.state_lock:
                node.state = "OBSERVE"
            return
        # 더 이상 탐색할 곳이 없다 - Object Reference와 달리 이건 "실패"가 아니라
        # "다 봤다"는 정상 종료 신호다.
        with node.state_lock:
            node.state = "MISSION1_FINALIZE_COUNT"
        node.get_logger().info(
            "🔍 EXPLORATION EXHAUSTED - no more frontiers, finalizing count "
            f"({node.coverage_planner.describe_last_plan_failure()})"
        )
        return
    node.exploration_route = deque(route)
    node.publish_next_exploration_goal()


def count_job(node, task_id: int, task: dict) -> dict:
    """탐색이 끝난 뒤 최종 개수를 센다 - Object Reference의 selection_job과 같은
    필터링(relation -> attribute)을 쓰되, 마지막에 GeminiSelector로 하나만 고르는
    대신 통과한 candidate 전부의 개수를 반환한다."""
    candidates = node.object_memory.find_by_category(task["target"])

    relation_required = bool(effective_relation_chain(task))
    relation_candidate_ids = set(node.scene_graph.find_matching_target_ids(task))
    if relation_candidate_ids:
        candidates = [
            candidate for candidate in candidates
            if int(candidate["object_id"]) in relation_candidate_ids
        ]
    elif relation_required:
        # exploration이 이미 끝난 뒤의 최종 집계 시점이다 - relation 제약을 만족하는
        # candidate가 하나도 검증 안 됐다면(Object Reference처럼 더 탐색해서 기다릴
        # 곳이 없으므로) 0으로 확정한다.
        candidates = []

    candidates = filter_by_attributes(node, candidates, task.get("attributes"))
    geometric_count = len(candidates)

    # object_memory 기반 집계는 **탐지 재현율에 갇힌다** - 못 본 물체는 셀 수가 없다
    # (실측 home_building_1: pillow GT 18개인데 메모리엔 7개). 물체를 가장 많이 담은
    # viewpoint 한 장을 VLM에게 보여 직접 세게 해서 그 상한을 넘어본다. 실패하면 기하
    # 기반 개수를 그대로 쓴다(0/1 채점이라 무응답이 최악이다).
    #
    # 단, 관계 제약이 있는데 그걸 만족하는 candidate가 **하나도 검증되지 않은** 경우는
    # 예외다. 그 상태의 기하 집계는 0인데, VLM은 관계 필터를 거치지 않고(질문 원문만
    # 넘긴다) 파노라마 한 장에서 센 숫자를 돌려주므로 그 0을 덮어써버린다. 즉 "vase를
    # 한 번도 못 본 채 vase 위 table 옆 chair 수"를 자신 있게 발행하게 된다. 근거가
    # 없을 때는 VLM으로 메우지 않고 기하 결과를 그대로 낸다.
    if config.NUMERICAL_VLM_COUNT_ENABLED and not (relation_required and not relation_candidate_ids):
        vlm_count = _count_with_vlm(node, task, geometric_count)
        if vlm_count is not None:
            return {"task_id": task_id, "count": vlm_count}
    elif relation_required and not relation_candidate_ids:
        node.get_logger().warning(
            "🔢 VLM count skipped - 관계 제약을 만족하는 candidate가 하나도 검증되지 "
            f"않음(chain={effective_relation_chain(task)}); 기하 집계 {geometric_count}를 그대로 사용"
        )

    return {"task_id": task_id, "count": geometric_count}


def _count_with_vlm(node, task: dict, geometric_count: int) -> int | None:
    """대상 물체를 가장 많이 담은 viewpoint 이미지로 VLM에게 세게 한다. 실패 시 None.

    뷰 선정은 relation/attribute 필터 **전의** "그 카테고리 전체"로 한다 - 관계 edge가
    아직 없어서 걸러진 물체도 이미지에는 찍혀 있고, 그게 바로 VLM이 대신 세줘야 하는
    대상이다. 제약(관계/속성) 판정은 질문 원문을 그대로 넘겨 VLM이 이미지에서 직접 본다.
    """
    target = str(task.get("target", "")).strip()
    question = str(task.get("raw", "")).strip()
    if not target or not question:
        return None

    category_ids = [
        int(item["object_id"]) for item in node.object_memory.find_by_category(target)
    ]
    if not category_ids:
        return None
    viewpoint = node.scene_graph.best_viewpoint_for_objects(category_ids)
    if viewpoint is None:
        node.get_logger().info("VLM count skipped: no viewpoint image observes this category")
        return None

    count = node.vlm_counter.count(question, target, viewpoint)
    if count is None:
        return None
    node.get_logger().info(
        f"🔢 COUNT - geometric={geometric_count}, vlm={count} "
        f"(viewpoint {viewpoint['viewpoint_id']} saw {viewpoint['visible_count']}/"
        f"{len(category_ids)} of the mapped {target}); using vlm"
    )
    return count


def _on_count_result(node, result: dict) -> None:
    count = int(result["count"])
    message = Int32()
    message.data = count
    node.numerical_response_pub.publish(message)
    node.last_response_summary = f"/numerical_response = {count}"
    with node.state_lock:
        node.state = "SUCCESS"
    node.get_logger().info(f"🚩🏁 COUNT PUBLISHED (task SUCCESS): count={count}")
