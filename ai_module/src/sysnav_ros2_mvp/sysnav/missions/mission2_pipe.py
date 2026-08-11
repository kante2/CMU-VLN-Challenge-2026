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
    # 목적지 좌표 하나를 그대로 던지지 않고, 현재 지도로 A* 경로를 만들어 hop 단위로
    # 이동한다 - hop에 도착할 때마다(그리고 주행 중 hop이 막히면 즉시) 최신 지도로
    # 경로를 다시 계산하기 위해서다. 경로를 못 만들면 start_target_navigation()이
    # 알아서 기존 동작(목적지 직접 발행)으로 폴백한다.
    node.start_target_navigation(
        pose, (x, y), theta, object_id=selected["object_id"],
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
    node.clear_target_navigation()
    with node.state_lock:
        node.state = "SUCCESS"
    node.get_logger().info(
        f"🚩🏁 ARRIVED - FINAL TARGET REACHED (task SUCCESS): "
        f"object_id={object_id} category={category}, "
        f"robot_pose=({pose['x']:.2f}, {pose['y']:.2f})"
    )


def _give_up_target(node) -> None:
    """step_target_navigation()이 "지금 지도로는 갈 방법이 없다"고 판단했을 때 불린다.

    바로 FAILED로 끝내지 않고 탐사로 되돌리는 이유: "지금 아는 지도로 길이 없다"는
    "갈 수 없다"가 아니라 "아직 안 뚫었다"인 경우가 많다. 맵을 더 넓힌 뒤 다시 이
    물체가 선택되면 그때는 경로가 나올 수 있다.
    """
    node.clear_target_navigation()
    with node.state_lock:
        node.state = "PLAN_EXPLORATION"
    node.get_logger().warning("🧭 back to exploration to open up the map")
