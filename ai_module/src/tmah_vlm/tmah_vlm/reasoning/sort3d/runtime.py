#!/usr/bin/env python3
"""Runtime bridge from the live scene graph to SORT3D-lite reasoning.

marker/waypoint 발행은 sort3d/publish.py로 분리했다.
"""

from tmah_vlm.common.helpers import get_robot_pose
from tmah_vlm.reasoning.sort3d.pipeline import Sort3DLite
from tmah_vlm.reasoning.sort3d.publish import publish_sort3d_result


RELATION_WORDS = (
    " on ", " near ", " between ", " closest ", " nearest ", " furthest ",
    " farthest ", " left ", " right ", " above ", " below ", " behind ",
    " in front ",
)


def is_relation_query(question):
    text = " " + str(question or "").lower() + " "
    return any(word in text for word in RELATION_WORDS)


def build_sort3d_from_node(node):
    sort3d = Sort3DLite.from_scene_graph(getattr(node, "scene_graph", None))
    return sort3d if sort3d.objects else None


def try_sort3d_graph_fallback(node, question):
    """
    Try selecting a target from the online scene graph.

    Returns True if a waypoint/marker was published. This intentionally never
    reads GT object_list.txt, so it remains compatible with hidden evaluation.
    """
    log = node.get_logger()
    sort3d = build_sort3d_from_node(node)
    if sort3d is None:
        log.info("[SORT3D] graph fallback skipped: scene graph is empty")
        return False

    robot_pose = get_robot_pose(node)
    selection = sort3d.select_target(question, robot_pose)
    candidate_ids = selection.get("candidate_ids", [])
    if not candidate_ids:
        log.info(f"[SORT3D] graph fallback found no target: {selection}")
        return False

    target = sort3d.toolbox.get(candidate_ids[0])
    if target is None:
        log.info(f"[SORT3D] graph fallback target missing: {selection}")
        return False

    waypoint = sort3d.action_for_selection(selection, robot_pose)
    if waypoint is None:
        log.info(f"[SORT3D] graph fallback could not make waypoint: {selection}")
        return False

    publish_sort3d_result(node, target, waypoint)
    log.info(
        f"[SORT3D] graph target={target.object_id}, name={target.name}, "
        f"tool={selection.get('tool')}, waypoint=({waypoint['x']:.2f}, "
        f"{waypoint['y']:.2f}, {waypoint['heading']:.2f})"
    )
    return True
