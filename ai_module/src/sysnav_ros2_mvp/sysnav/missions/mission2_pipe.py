"""Mission 2 - Object Reference (MISSION_2object_reference_CLAUDE.txt).

먼저 모든 room/frontier를 탐사하며 Scene Graph를 누적한다. 탐사가 완전히 끝난 뒤
Graph의 object/relation/viewpoint 정보를 사용해 문장이 가리키는 유일한 물체를 고르고,
bbox marker(`/selected_object_marker`)와 navigation waypoint를 발행한다.

state 흐름: OBSERVE/PLAN_EXPLORATION/FOLLOW_EXPLORATION(공용, sysnav_node.py) ->
SELECT_TARGET -> NAVIGATE_TARGET -> SUCCESS.
"""

from __future__ import annotations

from collections import deque

from sysnav.scene_graph.scene_graph_rviz import build_selected_object_marker


def loop(node, state: str, task: dict, task_id: int, pose: dict) -> None:
    if state == "SELECT_TARGET":
        node.submit_job("selection", node.selection_job, task_id, task, pose, origin_state=state)
        return
    if state == "NAVIGATE_TARGET":
        _run_navigate_target(node, pose)


def on_job_result(node, task: dict, kind: str, result: dict, origin_state: str) -> None:
    if kind == "perception":
        _on_perception_result(node, result, origin_state)
    elif kind == "selection":
        _on_selection_result(node, result)
    elif kind == "exploration":
        _on_exploration_result(node, task, result)


def _on_perception_result(node, result: dict, origin_state: str) -> None:
    # Mission 2는 탐사 도중 target이 보여도 route를 끊거나 SELECT_TARGET으로 가지
    # 않는다. 여러 방을 모두 본 뒤의 Scene Graph가 있어야 relation/attribute 및
    # 동종 객체 비교가 안정적이기 때문이다. FOLLOW 중 관측은 graph만 갱신하고,
    # waypoint 도착 후 OBSERVE 관측만 다음 탐사 계획을 시작한다.
    if result.get("candidates") and origin_state == "OBSERVE":
        node.get_logger().info(
            f"Scene Graph target candidates accumulated: {len(result['candidates'])}; "
            "continuing full exploration"
        )
    if origin_state == "OBSERVE":
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"


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
    marker = build_selected_object_marker(selected, node.get_clock().now().to_msg())
    node.selected_object_marker_pub.publish(marker)
    node.last_response_summary = (
        f"/selected_object_marker = {selected['category']} #{selected['object_id']}"
    )

    with node.state_lock:
        node.state = "NAVIGATE_TARGET"
    node.get_logger().info(
        f"🎯 TARGET SELECTED - object_id={selected['object_id']} "
        f"category={selected['category']}, heading to "
        f"goal=({x:.2f}, {y:.2f}, {theta:.2f})"
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

    Mission 2는 target 선택 전에 이미 전체 탐사를 끝냈으므로 다시 탐사로 보내 같은
    target을 무한 재선택하지 않는다. 이전/호환 호출처럼 아직 탐사 완료 전인 경우만
    PLAN_EXPLORATION으로 되돌린다.
    """
    node.clear_target_navigation()
    if node.mission2_exploration_complete:
        _fail_after_full_exploration(node, "selected target remained unreachable after replanning")
        return
    with node.state_lock:
        node.state = "PLAN_EXPLORATION"
    node.get_logger().warning("🧭 back to exploration to open up the map")
