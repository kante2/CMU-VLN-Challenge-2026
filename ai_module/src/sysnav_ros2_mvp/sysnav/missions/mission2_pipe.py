"""Mission 2 - Object Reference (MISSION_2object_reference_CLAUDE.txt).

먼저 모든 frontier를 탐사하며 Scene Graph를 누적한다. 탐사가 완전히 끝난 뒤
Graph의 object/relation/viewpoint 정보를 사용해 문장이 가리키는 유일한 물체를 고르고,
bbox marker(`/selected_object_marker`)와 navigation waypoint를 발행한다.

state 흐름: OBSERVE/PLAN_EXPLORATION/FOLLOW_EXPLORATION(공용, sysnav_node.py) ->
SELECT_TARGET -> NAVIGATE_TARGET -> SUCCESS.

채점 규정과의 대응 (README "Question Types and Initial Scoring"):

  **Object Reference** (/2): Marker must be published on `/selected_object_marker`,
  and is scored based on its degree of overlap with the ground truth bounding box.

즉 점수를 정하는 것은 **발행한 marker의 bbox 겹침 하나뿐**이고, 로봇이 물체 앞까지
갔는지는 채점에 안 들어간다. 그래서 이 파이프라인은 세 가지를 지킨다:

  1. 선택이 확정되면 즉시 marker를 낸다 - 그 순간 점수가 확보된다.
  2. 그 뒤에도 주행한다. 무의미해서가 아니라, 가까이서 다시 보면 object_memory의
     extent_3d가 정밀해져(bbox 겹침이 올라가) **점수가 오르기 때문**이다. 좋아진
     bbox는 다시 발행해야 의미가 있다.
  3. 주행이 실패해도 FAILED로 뒤집지 않는다. 이미 낸 답은 유효하다.
     (실측 2026-08-23: vase #15를 맞게 골라 발행하고도, 접근 지점 재선정이 5초간
     실패하자 FAILED로 처리했다. 채점상으로는 이미 딴 점수였다.)
"""

from __future__ import annotations

import time
from collections import deque

from sysnav import config
from sysnav.scene_graph.scene_graph_rviz import build_selected_object_marker


def loop(node, state: str, task: dict, task_id: int, pose: dict) -> None:
    if state == "SELECT_TARGET":
        node.submit_job("selection", node.selection_job, task_id, task, pose, origin_state=state)
        return
    if state == "NAVIGATE_TARGET":
        _run_navigate_target(node, pose)


def maybe_force_selection_at_deadline(node, state: str) -> bool:
    """Stop Mission 2 exploration after its budget and select current best evidence."""
    if state not in {"OBSERVE", "PLAN_EXPLORATION", "FOLLOW_EXPLORATION"}:
        return False
    started = node.task_start_time
    if started is None or time.monotonic() - started < config.MISSION2_EXPLORATION_TIME_LIMIT_SEC:
        return False
    node.mission2_exploration_deadline_reached = True
    node.exploration_route.clear()
    node.current_goal = None
    with node.state_lock:
        node.state = "SELECT_TARGET"
    node.get_logger().warning(
        f"⏰ MISSION 2 EXPLORATION DEADLINE - "
        f"{config.MISSION2_EXPLORATION_TIME_LIMIT_SEC:.0f}s elapsed; "
        "selecting best available Scene Graph candidate"
    )
    return True


def on_job_result(node, task: dict, kind: str, result: dict, origin_state: str) -> None:
    if kind == "perception":
        _on_perception_result(node, result, origin_state)
    elif kind == "selection":
        _on_selection_result(node, result)
    elif kind == "exploration":
        _on_exploration_result(node, task, result)


def _on_perception_result(node, result: dict, origin_state: str) -> None:
    # 주행 중(FOLLOW_EXPLORATION) 관측은 graph만 갱신하고 흐름을 건드리지 않는다.
    # 판단은 정지 상태(OBSERVE)에서만 한다 - 관측이 가장 안정적인 순간이다.
    if origin_state != "OBSERVE":
        return

    # 관계 체인이 후보 하나로 확정됐으면 탐사를 더 하지 않고 바로 고른다.
    if _answer_is_settled(node):
        return

    if result.get("candidates"):
        node.get_logger().info(
            f"Scene Graph target candidates accumulated: {len(result['candidates'])}; "
            "relation chain not settled yet - continuing exploration"
        )
    with node.state_lock:
        node.state = "PLAN_EXPLORATION"


def _answer_is_settled(node) -> bool:
    """관계 체인이 후보 **하나**로 좁혀졌으면 SELECT_TARGET으로 보내고 True.

    예전에는 "탐사 100% 완료"가 선택의 유일한 관문이었다. 실측(2026-08-23): 6분
    시점에 체인이 이미 GT 정답 하나로 확정됐는데도 frontier가 남아 SELECT_TARGET에
    못 갔고, frontier는 탐사할수록 오히려 늘어(21 -> 128셀) 끝이 안 보였다. 그 사이
    10분 제한이 지나간다.

    후보가 둘 이상이면 아직 구별할 정보가 부족하다는 뜻이므로 계속 탐사한다 - README는
    정답이 유일하다고 보장하므로, 여럿 남은 것은 "덜 봤다"는 신호다.

    체인이 없는 질문(관계 없이 카테고리만 찾는 경우)은 이 조기 종료를 쓰지 않는다.
    비교할 제약이 없어 "하나로 좁혀졌다"가 성립하지 않기 때문이다.
    """
    task = node.task
    if not task:
        return False
    survivors = node.scene_graph.resolve_relation_chain(task)
    if len(survivors) != 1:
        return False
    node.get_logger().info(
        f"🧠 RELATION CHAIN SETTLED on object #{survivors[0]} without finishing "
        f"exploration - selecting now (README Timing: finishing early earns bonus)"
    )
    with node.state_lock:
        node.state = "SELECT_TARGET"
    return True


def _on_selection_result(node, result: dict) -> None:
    if result.get("relation_pending"):
        # 전체 탐사 전이라면 보류할 수 있지만, Mission 2의 정상 흐름에서는 전체
        # 탐사 뒤 한 번만 선택하므로 이 시점의 evidence 부족은 최종 실패다.
        if node.mission2_exploration_complete:
            _fail_after_full_exploration(
                node, "no Scene Graph object satisfies the relation constraint"
            )
            return
        node.get_logger().info("Selection deferred: relation evidence is still incomplete")
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"
        return
    if result.get("attribute_pending"):
        # 문장의 속성 제약(예: "black" chair)을 만족하는 후보가 없다.
        if node.mission2_exploration_complete:
            _fail_after_full_exploration(
                node, "no Scene Graph object satisfies the attribute constraint"
            )
            return
        node.get_logger().info("Selection deferred: attribute evidence is still incomplete")
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"
        return

    selected = node.object_memory.get(result["selected_id"])
    with node.sensor_lock:
        pose = None if node.latest_pose is None else dict(node.latest_pose)
    if selected is None or pose is None:
        if node.mission2_exploration_complete:
            _fail_after_full_exploration(node, "Scene Graph final selection returned no object")
            return
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"
        return

    # 접근 지점은 /terrain_map 기준으로 고른다 - base autonomy의 waypointConverter가
    # 우리 좌표를 자기 traversable area로 스냅해버리므로, 그쪽이 받아들일 지점을
    # 처음부터 찍어야 엉뚱한 데로 끌려가지 않는다 (navigation/terrain_monitor.py).
    x, y, theta = node.approach_pose_for(pose, selected["position"])
    node.start_target_navigation(
        pose, (x, y), theta,
        object_id=selected["object_id"],
        object_xy=selected["position"][:2],
    )
    # 선택된 target object를 Scene Graph에 표시하고 debug PNG/JSON/DOT을 갱신한다.
    node.scene_graph.mark_selected_object(selected["object_id"])
    node.publish_object_markers()

    # README: "The center point of the bounding box marker will be used as a
    # waypoint to navigate the robot system" - marker와 navigation waypoint를
    # 같은 시점(선택 확정 시점)에 함께 낸다. 이 토픽이 실제 채점 대상이다.
    _publish_answer(node, selected)

    with node.state_lock:
        node.state = "NAVIGATE_TARGET"
    node.get_logger().info(
        f"🎯 TARGET SELECTED - object_id={selected['object_id']} "
        f"category={selected['category']}, heading to "
        f"goal=({x:.2f}, {y:.2f}, {theta:.2f})"
    )


def _publish_answer(node, selected: dict) -> None:
    """채점 대상 답안을 `/selected_object_marker`로 낸다.

    선택 확정 시 한 번, 그리고 주행 중 bbox 추정이 좋아질 때마다 다시 불린다.
    marker는 매번 object_memory의 **현재** 값으로 새로 만든다 - 캐시해두면 가까이
    가서 얻은 정밀한 extent_3d가 반영되지 않아 주행이 점수로 이어지지 않는다.
    """
    marker = build_selected_object_marker(selected, node.get_clock().now().to_msg())
    node.selected_object_marker_pub.publish(marker)
    node.mission2_answer_object_id = selected["object_id"]
    node._mission2_answer_extent = tuple(selected.get("extent_3d", ()) or ())
    node._mission2_last_answer_publish = time.monotonic()
    node.last_response_summary = (
        f"/selected_object_marker = {selected['category']} #{selected['object_id']}"
    )


def _refresh_answer(node) -> None:
    """주행 중 답안을 다시 낸다.

    두 가지를 동시에 해결한다:
      * 유실 방지 - publisher가 VOLATILE이라 평가 노드가 첫 발행을 놓쳤을 수 있다
        (config.MISSION2_ANSWER_REPUBLISH_SEC 주석 참고).
      * 점수 개선 - 물체에 가까워지면 object_memory.extent_3d가 갱신된다. 그 값이
        곧 marker의 scale이고 채점은 bbox 겹침이므로, 갱신분을 내보내야 점수에 반영된다.
    """
    object_id = node.mission2_answer_object_id
    if object_id is None:
        return
    last = node._mission2_last_answer_publish
    if last is not None and time.monotonic() - last < config.MISSION2_ANSWER_REPUBLISH_SEC:
        return
    selected = node.object_memory.get(object_id)
    if selected is None:
        return
    before = node._mission2_answer_extent
    _publish_answer(node, selected)
    after = node._mission2_answer_extent
    if before and after and before != after:
        node.get_logger().info(
            f"📦 answer bbox refined - extent {tuple(round(v, 2) for v in before)} -> "
            f"{tuple(round(v, 2) for v in after)} (republished)"
        )


def _on_exploration_result(node, task: dict, result: dict) -> None:
    route = result["route"]
    if not route:
        node.mission2_exploration_complete = True
        graph_candidates = [
            obj for obj in node.scene_graph.snapshot().get("objects", [])
            if str(obj.get("category", "")).lower() == str(task["target"]).lower()
        ]
        if not graph_candidates:
            _fail_after_full_exploration(
                node,
                "full exploration completed but target category is absent from Scene Graph",
            )
            return
        with node.state_lock:
            node.state = "SELECT_TARGET"
        node.get_logger().info(
            f"🧠 FULL EXPLORATION COMPLETE - selecting final target from Scene Graph "
            f"(category={task['target']}, candidates={len(graph_candidates)}, "
            f"planner={node.coverage_planner.describe_last_plan_failure()})"
        )
        return
    node.exploration_route = deque(route)
    node.publish_next_exploration_goal()


def _fail_after_full_exploration(node, reason: str) -> None:
    with node.state_lock:
        node.state = "FAILED"
    node.get_logger().warning(f"❌ TASK FAILED AFTER FULL EXPLORATION - {reason}")


def _run_navigate_target(node, pose: dict) -> None:
    """확정된 목적지로 가는 주행. 한 번 계산한 경로를 끝까지 고집하지 않고, 아래 순서로
    판단해서 필요하면 최신 지도로 A*를 다시 돌린다 (sysnav_node.py의 target navigation
    섹션 주석 참고).

    예전에는 `if not goal_reached(pose): return`이 전부였다 - 접근 포즈에 도달하지
    못하면 SUCCESS도 FAILED도 아닌 채로 영원히 대기했고, 주행 중 지도가 갱신돼서 그
    경로가 막힌 게 드러나도 반영할 방법이 없었다.
    """
    # 주행하는 이유가 곧 이것이다 - 가까워질수록 bbox가 정밀해지고, 그 개선분은
    # 다시 발행해야 점수가 된다. 주행 성패와 무관하게 매 tick 돌린다.
    _refresh_answer(node)

    outcome = node.step_target_navigation(pose)
    if outcome == "arrived":
        _finish_navigate_target(node, pose)
    elif outcome == "unreachable":
        _give_up_target(node)


def _finish_navigate_target(node, pose: dict) -> None:
    object_id = node.target_object_id
    obj = None if object_id is None else node.object_memory.get(object_id)
    category = obj["category"] if obj else "?"
    goal_distance = node.distance_to_target(pose)
    # 물체 앞에서 얻은 관측이 가장 정확하다. 여기서 한 번 더 내야 그 값이 채점에 쓰인다.
    if obj is not None:
        node._mission2_last_answer_publish = None      # 주기 제한 무시하고 즉시
        _refresh_answer(node)
    node.clear_target_navigation()
    with node.state_lock:
        node.state = "SUCCESS"
    node.get_logger().info(
        f"🚩🏁 ARRIVED - FINAL TARGET REACHED (task SUCCESS): "
        f"object_id={object_id} category={category}, "
        f"robot_pose=({pose['x']:.2f}, {pose['y']:.2f}), "
        f"goal_distance={goal_distance:.2f}m"
    )


def _give_up_target(node) -> None:
    """step_target_navigation()이 "지금 지도로는 갈 방법이 없다"고 판단했을 때 불린다.

    답안(marker)을 이미 냈다면 **FAILED가 아니다.** Object Reference의 점수는 marker
    bbox 겹침만으로 정해지고 로봇 위치는 채점에 안 들어간다(모듈 docstring 참고).
    물체가 캐비닛 위처럼 접근 불가한 자리에 있으면 base autonomy가 받아줄 지점이
    아예 없을 수 있는데, 그건 답이 틀렸다는 뜻이 아니다.

    아직 답을 못 냈고 탐사도 안 끝났으면 예전처럼 탐사로 되돌린다.
    """
    node.clear_target_navigation()
    if node.mission2_answer_object_id is not None:
        _settle_answered(node, "could not reach the object, but the answer stands")
        return
    if node.mission2_exploration_complete:
        _fail_after_full_exploration(node, "selected target remained unreachable after replanning")
        return
    with node.state_lock:
        node.state = "PLAN_EXPLORATION"
    node.get_logger().warning("🧭 back to exploration to open up the map")


def _settle_answered(node, reason: str) -> None:
    """답안을 낸 채로 이번 질문을 마무리한다.

    상태를 SUCCESS로 두면 control_loop이 더 이상 돌지 않아 재발행도 멈춘다. 그게
    의도다 - 같은 답을 10분 끝까지 계속 쏘면 "언제 답을 냈는가"가 흐려져 조기 완료
    보너스(README Timing)에 불리할 수 있다.
    """
    with node.state_lock:
        node.state = "SUCCESS"
    node.get_logger().warning(
        f"🏁 ANSWER SUBMITTED (navigation incomplete) - {reason}; "
        f"response={node.last_response_summary}"
    )
