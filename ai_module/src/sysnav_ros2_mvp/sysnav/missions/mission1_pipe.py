"""Mission 1 - Numerical (MISSION_1numerical_CLAUDE.txt).

Object Reference와 문장 파싱(target/attributes/relation_chain)은 동일한
LLMQueryParser를 재사용하지만, 종료 조건이 근본적으로 다르다: candidate를
찾아도 멈추지 않고 탐색이 완전히 끝날 때까지(plan_route()가 빈 route를 반환할
때까지) 계속 돌다가, 그 시점에 최종 후보 개수를 세어 `/numerical_response`
(Int32)로 발행한다.

state 흐름: OBSERVE/PLAN_EXPLORATION/FOLLOW_EXPLORATION(공용, sysnav_node.py) ->
(exploration exhausted) -> MISSION1_FINALIZE_COUNT -> SUCCESS.
"""

from __future__ import annotations

from collections import deque

from std_msgs.msg import Int32

from sysnav import config
from sysnav.task.query_parser import effective_relation_chain


def loop(node, state: str, task: dict, task_id: int, pose: dict) -> None:
    if state == "MISSION1_FINALIZE_COUNT":
        node.submit_job("count", count_job, node, task_id, task, origin_state=state)


def on_job_result(node, task: dict, kind: str, result: dict, origin_state: str) -> None:
    if kind == "perception":
        _on_perception_result(node, origin_state)
    elif kind == "exploration":
        _on_exploration_result(node, result)
    elif kind == "count":
        _on_count_result(node, result)


def _on_perception_result(node, origin_state: str) -> None:
    # candidate를 찾아도 절대 멈추지 않는다 - 전체 탐색이 끝나야 정확한 개수를 셀 수
    # 있다(Object Reference와의 핵심 차이). OBSERVE에서 온 관측만 다음 탐색을
    # 계획하고, FOLLOW_EXPLORATION 중 관측(perception-while-moving)은 object_memory만
    # 갱신하고 이동 자체는 방해하지 않는다.
    if origin_state == "OBSERVE":
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"


def _on_exploration_result(node, result: dict) -> None:
    route = result["route"]
    if not route:
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

    relation_candidate_ids = set(node.scene_graph.find_matching_target_ids(task))
    if relation_candidate_ids:
        candidates = [
            candidate for candidate in candidates
            if int(candidate["object_id"]) in relation_candidate_ids
        ]
    elif effective_relation_chain(task):
        # exploration이 이미 끝난 뒤의 최종 집계 시점이다 - relation 제약을 만족하는
        # candidate가 하나도 검증 안 됐다면(Object Reference처럼 더 탐색해서 기다릴
        # 곳이 없으므로) 0으로 확정한다.
        candidates = []

    attributes = list(task.get("attributes") or [])
    if attributes and config.ATTRIBUTE_VERIFICATION_ENABLED and candidates:
        attribute_results = node.attribute_verifier.verify(candidates, attributes)
        for candidate in candidates:
            newly_checked = attribute_results.get(int(candidate["object_id"]), {})
            if newly_checked:
                node.object_memory.update_self_attributes(int(candidate["object_id"]), newly_checked)
        candidates = [
            candidate for candidate in candidates
            if all(
                attribute_results.get(int(candidate["object_id"]), {}).get(attribute, False)
                for attribute in attributes
            )
        ]

    return {"task_id": task_id, "count": len(candidates)}


def _on_count_result(node, result: dict) -> None:
    count = int(result["count"])
    message = Int32()
    message.data = count
    node.numerical_response_pub.publish(message)
    with node.state_lock:
        node.state = "SUCCESS"
    node.get_logger().info(f"🚩🏁 COUNT PUBLISHED (task SUCCESS): count={count}")
