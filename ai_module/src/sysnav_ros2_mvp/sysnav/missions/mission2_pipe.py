"""Mission 2 - Object Reference (MISSION_2object_reference_CLAUDE.txt).

문장 하나가 가리키는 유일한 물체를 찾아 bbox marker(`/selected_object_marker`)와
navigation waypoint(`/way_point_with_heading`)를 낸다. sysnav_node.py의 기존
object_reference 파이프라인(리팩터 이전 코드)을 그대로 옮긴 것 - 동작은 바뀌지
않았고, 이번에 빠져 있던 `/selected_object_marker` 발행만 추가됐다.

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
        _on_exploration_result(node, result)


def _on_perception_result(node, result: dict, origin_state: str) -> None:
    if result["candidates"]:
        with node.state_lock:
            node.state = "SELECT_TARGET"
            node.exploration_route.clear()
    elif origin_state == "OBSERVE":
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"


def _on_selection_result(node, result: dict) -> None:
    if result.get("relation_pending"):
        # 문장의 relation 제약(예: "knife rack 근처")이 아직 검증 안 된 candidate뿐이다.
        # 확정하지 않고 계속 탐색해서 참조 물체를 더 찾아본다.
        node.get_logger().info(
            "Selection deferred: relation constraint not verified for any candidate yet, "
            "continuing exploration"
        )
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"
        return
    if result.get("attribute_pending"):
        # 문장의 속성 제약(예: "black" chair)을 만족하는 candidate가 검증되지 않았다
        # (불일치했거나 아직 확인 자체가 안 됨) - 확정하지 않고 계속 탐색해서 진짜
        # 속성이 맞는 물체를 더 찾아본다.
        node.get_logger().info(
            "Selection deferred: attribute constraint not verified for any "
            "candidate yet, continuing exploration"
        )
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"
        return

    selected = node.object_memory.get(result["selected_id"])
    with node.sensor_lock:
        pose = None if node.latest_pose is None else dict(node.latest_pose)
    if selected is None or pose is None:
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"
        return

    x, y, theta = node.goal_publisher.object_approach_pose(pose, selected["position"])
    node.goal_publisher.publish(x, y, theta)
    node.current_goal = {
        "x": x,
        "y": y,
        "theta": theta,
        "type": "target",
        "object_id": selected["object_id"],
    }
    # 선택된 target object를 Scene Graph에 표시하고 debug PNG/JSON/DOT을 갱신한다.
    node.scene_graph.mark_selected_object(selected["object_id"])
    node.publish_object_markers()

    # README: "The center point of the bounding box marker will be used as a
    # waypoint to navigate the robot system" - marker와 navigation waypoint를
    # 같은 시점(선택 확정 시점)에 함께 낸다. 이 토픽이 실제 채점 대상이다.
    marker = build_selected_object_marker(selected, node.get_clock().now().to_msg())
    node.selected_object_marker_pub.publish(marker)

    with node.state_lock:
        node.state = "NAVIGATE_TARGET"
    node.get_logger().info(
        f"🎯 TARGET SELECTED - object_id={selected['object_id']} "
        f"category={selected['category']}, heading to "
        f"goal=({x:.2f}, {y:.2f}, {theta:.2f})"
    )


def _on_exploration_result(node, result: dict) -> None:
    route = result["route"]
    if not route:
        with node.state_lock:
            node.state = "FAILED"
        node.get_logger().warning(
            f"❌ TASK FAILED - no reachable frontier remains "
            f"({node.coverage_planner.describe_last_plan_failure()})"
        )
        return
    node.exploration_route = deque(route)
    node.publish_next_exploration_goal()


def _run_navigate_target(node, pose: dict) -> None:
    if not node.goal_reached(pose):
        return
    object_id = node.current_goal.get("object_id") if node.current_goal else None
    obj = None if object_id is None else node.object_memory.get(object_id)
    category = obj["category"] if obj else "?"
    with node.state_lock:
        node.state = "SUCCESS"
    node.get_logger().info(
        f"🚩🏁 ARRIVED - FINAL TARGET REACHED (task SUCCESS): "
        f"object_id={object_id} category={category}, "
        f"robot_pose=({pose['x']:.2f}, {pose['y']:.2f})"
    )
