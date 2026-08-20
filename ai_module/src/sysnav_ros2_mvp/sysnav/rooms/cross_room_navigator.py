"""Door-graph navigation between room-scoped coverage phases."""

from __future__ import annotations

import math

from sysnav import config


def _world(node, row: float, col: float) -> tuple[float, float]:
    return node.coverage_planner.grid_to_world(row, col)


def _distance(node, pose: dict, room: dict) -> float:
    x, y = _world(node, room["anchor_row"], room["anchor_col"])
    return math.hypot(x - float(pose["x"]), y - float(pose["y"]))


def _ranking_payload(room: dict, distance_m: float) -> dict:
    return {
        "room_id": int(room["room_id"]),
        "category": room.get("category") or "unknown",
        "objects": list(room.get("object_labels") or []),
        "distance_m": round(float(distance_m), 2),
        "visited": bool(room.get("visited", False)),
        "image_path": room.get("image_path") or room.get("representative_image_path"),
    }


def _plan_room_route(node, pose: dict, candidate: dict) -> list[dict] | None:
    """Plan to each door, step inside the next room, then reach its anchor."""
    route: list[dict] = []
    virtual_pose = dict(pose)
    room_path = list(candidate.get("room_path") or [candidate["room_id"]])
    doorways = list(candidate.get("doorways") or [])

    def append_segment(goal_xy: tuple[float, float], entering_room_id: int | None = None) -> bool:
        nonlocal virtual_pose
        segment = node.coverage_planner.plan_direct_path(
            virtual_pose,
            goal_xy,
            max_hop_spacing_m=config.EXPLORATION_PATH_WAYPOINT_SPACING_M,
        )
        if not segment:
            return False
        for waypoint in segment:
            waypoint["is_viewpoint"] = False
            waypoint["navigation_mode"] = "cross_room"
            if entering_room_id is not None:
                waypoint["entering_room_id"] = int(entering_room_id)
        route.extend(segment)
        virtual_pose = {
            **virtual_pose,
            "x": float(segment[-1]["x"]),
            "y": float(segment[-1]["y"]),
        }
        return True

    for index, doorway in enumerate(doorways):
        next_room_id = int(room_path[index + 1])
        door_xy = _world(node, doorway["centroid_row"], doorway["centroid_col"])
        if not append_segment(door_xy):
            return None

        next_room = node.room_registry.get_room(next_room_id)
        if next_room is None:
            return None
        anchor_xy = _world(node, next_room["anchor_row"], next_room["anchor_col"])
        dx, dy = anchor_xy[0] - door_xy[0], anchor_xy[1] - door_xy[1]
        norm = math.hypot(dx, dy)
        if norm > 1e-6:
            depth = min(float(config.ROOM_ENTRY_DEPTH_M), max(0.4, norm * 0.7))
            entry_xy = (door_xy[0] + dx / norm * depth, door_xy[1] + dy / norm * depth)
            if not append_segment(entry_xy, entering_room_id=next_room_id):
                return None

    target_xy = _world(node, candidate["anchor_row"], candidate["anchor_col"])
    if math.hypot(target_xy[0] - virtual_pose["x"], target_xy[1] - virtual_pose["y"]) > 0.35:
        if not append_segment(target_xy, entering_room_id=int(candidate["room_id"])):
            return None
    if route:
        route[-1]["room_entry"] = True
        route[-1]["target_room_id"] = int(candidate["room_id"])
    return route or None


def select_job(
    node,
    task_id: int,
    task: dict,
    pose: dict,
    candidates: list[dict],
    current_room: dict | None = None,
    early_stop: bool = False,
) -> dict:
    """Choose a room semantically, then realize it through the doorway graph."""
    candidates = sorted(candidates, key=lambda room: _distance(node, pose, room))
    candidate_by_id = {int(room["room_id"]): room for room in candidates}
    payloads = [_ranking_payload(room, _distance(node, pose, room)) for room in candidates]

    if early_stop:
        # Put the current room first. If VLM is unavailable, rank() preserves this
        # fallback order and exploration safely continues instead of leaving early.
        if current_room is None:
            return {"task_id": task_id, "room_id": None, "path": None, "deferred": True}
        current_payload = _ranking_payload(current_room, 0.0)
        ranked_ids = node.room_relevance_selector.rank(
            task.get("raw", ""), [current_payload] + payloads
        )
        if not ranked_ids or int(ranked_ids[0]) == int(current_room["room_id"]):
            return {
                "task_id": task_id, "room_id": None, "path": None,
                "deferred": True, "failed_room_ids": [], "early_stop": True,
            }
        ordered_ids = [room_id for room_id in ranked_ids if room_id in candidate_by_id]
    else:
        ranked_ids = node.room_relevance_selector.rank(task.get("raw", ""), payloads)
        ordered_ids = [room_id for room_id in ranked_ids if room_id in candidate_by_id]
        ordered_ids += [room_id for room_id in candidate_by_id if room_id not in ordered_ids]

    failed: list[int] = []
    for room_id in ordered_ids:
        room = candidate_by_id[int(room_id)]
        path = _plan_room_route(node, pose, room)
        if path:
            return {
                "task_id": task_id,
                "room_id": int(room_id),
                "path": path,
                "failed_room_ids": failed,
                "room_path": list(room.get("room_path") or []),
                "early_stop": early_stop,
                "deferred": False,
            }
        diag = node.coverage_planner.last_direct_path_diagnostics
        node.get_logger().warning(
            f"Door-graph route failed: room_id={room_id} "
            f"room_path={room.get('room_path')} reason={diag.get('reason')}"
        )
        failed.append(int(room_id))

    return {
        "task_id": task_id,
        "room_id": None,
        "path": None,
        "failed_room_ids": failed,
        "early_stop": early_stop,
        "deferred": bool(early_stop),
    }
