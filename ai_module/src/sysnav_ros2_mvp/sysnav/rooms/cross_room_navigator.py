"""Cross-room navigation job (SysNav paper Sec. IV-B-2, "Room-query navigation
mode") - in-room exploration이 소진됐을 때(plan_route()가 빈 route) 호출된다.

아직 안 들어가본 방(RoomRegistry.unvisited_rooms())을 "물체가 있을법한 방 먼저,
카테고리를 모르는 방은 가까운 방부터" 순서로 정렬한 뒤, 그 순서대로
plan_direct_path()가 실제로 되는 첫 방을 골라 거기까지 가는 waypoint 경로를
반환한다. node를 인자로 받는 자유 함수 - sysnav_node.submit_job()의 worker
스레드에서 실행되어 VLM 호출(관련도 순위 판단)이 메인/ROS 스레드를 막지 않는다.
"""

from __future__ import annotations

import math


def select_job(node, task_id: int, task: dict, pose: dict, candidates: list[dict]) -> dict:
    """candidates: RoomRegistry.unvisited_rooms() 결과(이미 attempted 필터링된 상태).
    반환: {"task_id", "room_id", "path"} - 어디로도 못 가면 room_id/path 둘 다 None."""
    categorized = [room for room in candidates if room.get("category")]
    uncategorized = [room for room in candidates if not room.get("category")]

    ordered_categorized = categorized
    if categorized:
        ranked_ids = node.room_relevance_selector.rank(
            task.get("raw", ""),
            [{"room_id": room["room_id"], "category": room["category"]} for room in categorized],
        )
        rank_of = {room_id: index for index, room_id in enumerate(ranked_ids)}
        ordered_categorized = sorted(
            categorized, key=lambda room: rank_of.get(room["room_id"], len(rank_of))
        )

    def _distance(room: dict) -> float:
        wx, wy = node.coverage_planner.grid_to_world(room["centroid_row"], room["centroid_col"])
        return math.hypot(wx - pose["x"], wy - pose["y"])

    ordered_uncategorized = sorted(uncategorized, key=_distance)

    # 논문 취지: category를 아는 방(관련도 판단 가능)을 먼저 시도하고, 그 안에서는
    # 관련도 순, category를 모르는 방은 그 뒤에 거리순으로 붙인다.
    tried_room_ids: list[int] = []
    for room in ordered_categorized + ordered_uncategorized:
        wx, wy = node.coverage_planner.grid_to_world(room["centroid_row"], room["centroid_col"])
        path = node.coverage_planner.plan_direct_path(pose, (wx, wy))
        if path:
            return {
                "task_id": task_id, "room_id": room["room_id"], "path": path,
                "failed_room_ids": tried_room_ids,
            }
        # 경로를 못 찾은 방도 "시도해봤음"으로 남겨서 호출 쪽이 다음 사이클에 또
        # 헛수고로 재시도하지 않게 한다 (A*로 못 가는 방이 갑자기 되지는 않음 - 맵이
        # 더 갱신되기 전까지는).
        tried_room_ids.append(room["room_id"])

    return {"task_id": task_id, "room_id": None, "path": None, "failed_room_ids": tried_room_ids}
