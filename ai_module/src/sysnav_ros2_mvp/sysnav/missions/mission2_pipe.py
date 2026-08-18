"""Mission 2 - Object Reference (MISSION_2object_reference_CLAUDE.txt).

문장 하나가 가리키는 유일한 물체를 찾아 bbox marker(`/selected_object_marker`)와
navigation waypoint(`/way_point_with_heading`)를 낸다.

**탐색 완주 -> Scene Graph 완성 -> 선택** 구조다(2026-08-18 변경). 후보를 하나
찾자마자 고르던 예전 greedy 구조는 "지금까지 본 것 중 최선"을 답으로 확정해버려서,
"closest/farthest" 같은 최상급 문장에서 아직 못 본 더 가까운 물체가 있으면 그대로
오답이 됐다. 이제 Mission 1(Numerical)과 같은 종료 조건을 쓴다 - candidate가
생겨도 멈추지 않고 coverage_planner.plan_route()가 빈 route를 반환할 때까지
탐색하면서 Scene Graph를 채우고, 그 시점에 완성된 그래프로 한 번에 고른다.

두 가지 안전장치가 붙어 있다:
  - 10분 제한(config.MISSION2_SELECT_DEADLINE_SEC): 탐색이 안 끝나도 이 시간을
    넘으면 지금까지의 그래프로 최종 선택을 강행한다. 채점이 0/1이라 무응답이 최악이다.
  - recovery patrol(config.MISSION2_RECOVERY_PATROL_MAX_POINTS): 프론티어가 소진됐는데도
    후보를 못 고르면(그림/창문처럼 카메라로만 보이는 물체) mission3와 같은 방식으로
    이미 아는 통행 가능 지점을 더 돌아본 뒤 다시 선택한다.

최종 선택(`selection_job(final=True)`)은 relation/attribute가 검증 안 됐다고 미루지
않고 반드시 하나를 고른다 - 더 탐색해서 기다릴 여지가 이미 없기 때문이다.

state 흐름: OBSERVE/PLAN_EXPLORATION/FOLLOW_EXPLORATION(공용, sysnav_node.py) ->
(exploration exhausted 또는 deadline) -> SELECT_TARGET -> NAVIGATE_TARGET -> SUCCESS.
"""

from __future__ import annotations

import time
from collections import deque

from sysnav import config
from sysnav.scene_graph.scene_graph_rviz import build_selected_object_marker


def loop(node, state: str, task: dict, task_id: int, pose: dict) -> None:
    if state == "SELECT_TARGET":
        node.submit_job(
            "selection",
            node.selection_job,
            task_id,
            task,
            pose,
            node.mission2_select_final,
            origin_state=state,
        )
        return
    if state == "NAVIGATE_TARGET":
        _run_navigate_target(node, pose)


def _deadline_reached(node) -> bool:
    """10분 제한이 임박했는지 - 탐색을 더 못 기다리는 시점인지 판단한다."""
    if node.task_start_time is None:
        return False
    return (time.monotonic() - node.task_start_time) >= config.MISSION2_SELECT_DEADLINE_SEC


def _enter_final_selection(node, reason: str) -> None:
    """탐색을 끝내고(또는 포기하고) 완성된 Scene Graph로 최종 선택에 들어간다."""
    node.mission2_select_final = True
    node.exploration_route.clear()
    node.current_goal = None
    with node.state_lock:
        node.state = "SELECT_TARGET"
    node.get_logger().info(f"🔍 FINAL SELECTION - {reason}")


def on_job_result(node, task: dict, kind: str, result: dict, origin_state: str) -> None:
    if kind == "perception":
        _on_perception_result(node, result, origin_state)
    elif kind == "selection":
        _on_selection_result(node, result)
    elif kind == "exploration":
        _on_exploration_result(node, result)


def _on_perception_result(node, result: dict, origin_state: str) -> None:
    # candidate를 찾아도 멈추지 않는다 - 탐색을 완주해야 Scene Graph가 완성되고,
    # 그래야 "closest/farthest" 같은 비교 판정이 전체 후보를 놓고 이뤄진다
    # (Mission 1과 같은 종료 조건). 시간 제한이 임박한 경우만 예외다.
    if _deadline_reached(node) and result["candidates"]:
        _enter_final_selection(
            node,
            f"time limit approaching ({config.MISSION2_SELECT_DEADLINE_SEC:.0f}s), "
            f"selecting from {len(result['candidates'])} candidate(s) found so far",
        )
        return
    # FOLLOW_EXPLORATION 중 관측(perception-while-moving)은 object_memory와 Scene
    # Graph만 갱신하고 이동 자체는 방해하지 않는다.
    if origin_state == "OBSERVE":
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"


def _on_selection_result(node, result: dict) -> None:
    # *_pending은 "더 탐색하면 확정할 수 있다"는 뜻이라 최종 선택(final=True)에서는
    # 애초에 나오지 않는다. 시간 제한 직전에 강행한 선택 등 예외 경로를 위해
    # 남겨두되, 이미 탐색이 끝난 뒤라면 탐색으로 되돌리지 않고 아래 recovery로 간다.
    if result.get("relation_pending") or result.get("attribute_pending"):
        constraint = "relation" if result.get("relation_pending") else "attribute"
        if not node.mission2_select_final:
            node.get_logger().info(
                f"Selection deferred: {constraint} constraint not verified for any "
                "candidate yet, continuing exploration"
            )
            with node.state_lock:
                node.state = "PLAN_EXPLORATION"
            return
        _recover_or_fail(node, f"{constraint} constraint unverified at final selection")
        return

    selected = node.object_memory.get(result["selected_id"])
    with node.sensor_lock:
        pose = None if node.latest_pose is None else dict(node.latest_pose)
    if selected is None or pose is None:
        # 최종 선택인데도 고를 게 없다(카테고리 후보 자체가 없거나 pose 미준비).
        if node.mission2_select_final:
            _recover_or_fail(node, "no candidate available at final selection")
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


def _on_exploration_result(node, result: dict) -> None:
    route = result["route"]
    if not route:
        # 더 볼 곳이 없다 - Mission 1과 마찬가지로 이건 "실패"가 아니라 "다 봤다,
        # 이제 완성된 Scene Graph로 고를 시점"이라는 신호다. (예전엔 여기서 바로
        # FAILED였고, 그래서 유일 후보인 최상급 문장이 답 없이 끝났다.)
        _enter_final_selection(
            node,
            "exploration exhausted, selecting from the completed scene graph "
            f"({node.coverage_planner.describe_last_plan_failure()})",
        )
        return
    if _deadline_reached(node):
        _enter_final_selection(
            node,
            f"time limit approaching ({config.MISSION2_SELECT_DEADLINE_SEC:.0f}s), "
            "stopping exploration",
        )
        return
    node.exploration_route = deque(route)
    node.publish_next_exploration_goal()


def _recover_or_fail(node, reason: str) -> None:
    """최종 선택에서 아무것도 못 골랐을 때 - 이미 아는 통행 가능 지점을 더 돌아보고
    (mission3의 recovery patrol과 같은 방식), 그것도 소진되면 FAILED로 끝낸다."""
    with node.sensor_lock:
        pose = None if node.latest_pose is None else dict(node.latest_pose)
    if (
        pose is not None
        and not _deadline_reached(node)
        and len(node.mission2_recovery_points) < config.MISSION2_RECOVERY_PATROL_MAX_POINTS
    ):
        recovery_route = node.coverage_planner.plan_recovery_patrol(
            pose, node.mission2_recovery_points
        )
        if recovery_route:
            endpoint = recovery_route[-1]
            node.mission2_recovery_points.append((float(endpoint["x"]), float(endpoint["y"])))
            node.exploration_route = deque(recovery_route)
            node.get_logger().info(
                f"🧭 RECOVERY PATROL {len(node.mission2_recovery_points)}/"
                f"{config.MISSION2_RECOVERY_PATROL_MAX_POINTS} - {reason}, "
                f"checking viewpoint ({endpoint['x']:.2f}, {endpoint['y']:.2f})"
            )
            node.publish_next_exploration_goal()
            return
    with node.state_lock:
        node.state = "FAILED"
    node.get_logger().warning(
        f"❌ TASK FAILED - {reason}; frontier and recovery patrol exhausted "
        f"({node.coverage_planner.describe_last_plan_failure()})"
    )


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
