"""Mission 3 - Instruction-Following (MISSION_3instruction_following_CLAUDE.txt).

문장을 절 단위로 쪼갠다:
  - destination 절("go to X"/"go near X"/"stop at X"/"stop by X"/"go between A and B") ->
    순서대로 실제 도달해야 하는 정지점(is_stop=True). 채점 대상("순서/제약 달성").
  - positive path 절("take the path between A and B"/"take the path near Z"/"pass by Z") ->
    반드시 지나가야 하지만 별도로 "정지"하지는 않는 경유 waypoint(is_stop=False).
    base autonomy가 point-to-point로 이동하는 것만으로 "이 지점을 지나갔다"는 제약을
    충분히 만족시킬 수 있어서, destination과 동일하게 순서대로 waypoint 큐에 넣는다.
  - negative path 절("avoiding the path between A and B"/"avoid the path near Z") ->
    문장 내 어디에 있든(선행 destination 뒤에 붙는 경우가 많음) README의 "passes
    through areas it is forbidden to go through"는 전체 경로에 대한 제약이라고
    해석해서, 이후 모든 leg의 경로 계획에 전역으로 적용한다(leg 하나에 한정 안 함).

목적지/경유지 텍스트는 object_reference와 동일한 LLMQueryParser로 파싱해서 재사용한다
(각 절이 그 자체로 하나의 G=(c_tgt,Φ))). "between A and B"/"near Z" 형태의 기하학적
참조("이 지점")는 category 후보의 위치를 직접 쓰는 별도의 가벼운(= VLM 호출 없는)
resolve 경로를 쓴다 - object_reference처럼 "유일한 정답"을 정밀하게 골라야 하는
문제가 아니라 대략의 통과 지점만 있으면 되기 때문이다.

state 흐름: OBSERVE/PLAN_EXPLORATION/FOLLOW_EXPLORATION(공용, sysnav_node.py) ->
MISSION3_SELECT_STEP -> MISSION3_NAVIGATE_STEP -> (다음 step) ... -> SUCCESS.
"""

from __future__ import annotations

import math
import re
from collections import deque

import numpy as np

from sysnav import config

# ---------------------------------------------------------------------------
# 문장 -> 절(clause) 분리. questions.json의 instruction_following 30문장 전수
# 검증됨 (go to/near/between, stop at/by, take the path, avoid(ing) the path,
# pass by, 생략된 "and (then/finally,) to X" 형태까지 포함).
# ---------------------------------------------------------------------------

_TRIGGER_RE = re.compile(
    r"(?P<neg>avoid(?:ing)? the path)"
    r"|(?P<pos>take the path|pass by)"
    r"|(?P<dest_between>go between)"
    r"|(?P<dest>go to|go near|stop at|stop by)"
    r"|(?P<dest_to>and\s+then\s+to|and\s+finally,\s*to)",
    re.IGNORECASE,
)


def _clean_clause(text: str) -> str:
    text = text.strip()
    text = re.sub(
        r"^(first,|then,?|,?\s*and\s+then,?|,?\s*and)\s*", "", text, flags=re.IGNORECASE
    ).strip()
    text = re.sub(
        r"\s*,?\s*(and\s+then|and|then)\s*,?\s*$", "", text, flags=re.IGNORECASE
    ).strip()
    return text.rstrip(" ,.").strip()


def _parse_between_argument(text: str, strip_leading_between: bool) -> tuple[dict | None, str | None]:
    """"between A and B[, to W]" 또는 "the two X[, to W]" 형태를 해석한다.
    반환: (ref_spec, fused_destination_text|None)."""
    text = text.strip()
    if strip_leading_between:
        m = re.match(r"between\s+(.+)", text, re.IGNORECASE)
        if not m:
            return None, None
        text = m.group(1)

    m2 = re.match(r"(.+?)\s+and\s+(.+)", text, re.IGNORECASE)
    if m2:
        ref_a, tail = m2.group(1).strip(), m2.group(2).strip()
        fused_destination = None
        if " to " in tail:
            ref_b, _, fused_destination = tail.partition(" to ")
            ref_b, fused_destination = ref_b.strip(), fused_destination.strip()
        else:
            ref_b = tail
        return {"kind": "between", "refs": [ref_a, ref_b]}, fused_destination

    # "the two columns" 처럼 명시적 "and"가 없는 collective 형태.
    collective = text.strip()
    fused_destination = None
    if " to " in collective:
        collective, _, fused_destination = collective.partition(" to ")
        collective, fused_destination = collective.strip(), fused_destination.strip()
    if not collective:
        return None, None
    return {"kind": "between_collective", "refs": [collective]}, fused_destination


def _split_path_argument(text: str) -> tuple[dict | None, str | None]:
    text = text.strip()
    if re.match(r"between\b", text, re.IGNORECASE):
        return _parse_between_argument(text, strip_leading_between=True)
    m = re.match(r"near\s+(.+)", text, re.IGNORECASE)
    if m:
        ref = m.group(1).strip()
        fused_destination = None
        if " to " in ref:
            ref, _, fused_destination = ref.partition(" to ")
            ref, fused_destination = ref.strip(), fused_destination.strip()
        return {"kind": "near", "refs": [ref]}, fused_destination
    if " to " in text:  # bare "pass by Z to W"
        ref, _, fused_destination = text.partition(" to ")
        return {"kind": "near", "refs": [ref.strip()]}, fused_destination.strip()
    return ({"kind": "near", "refs": [text]} if text else None), None


def _split_clauses(question: str) -> list[dict]:
    matches = list(_TRIGGER_RE.finditer(question))
    raw_steps: list[dict] = []
    for i, m in enumerate(matches):
        if m.group("neg"):
            kind = "negative_path"
        elif m.group("pos"):
            kind = "positive_path"
        elif m.group("dest_between"):
            kind = "destination_between"
        elif m.group("dest_to"):
            kind = "destination"  # 생략된 "and then to X"/"and finally, to X"
        else:
            kind = "destination"
        end = matches[i + 1].start() if i + 1 < len(matches) else len(question)
        arg = _clean_clause(question[m.end():end])

        if kind == "destination":
            if arg:
                raw_steps.append({"kind": "destination", "text": arg})
        elif kind == "destination_between":
            ref, fused = _parse_between_argument(arg, strip_leading_between=False)
            if ref:
                raw_steps.append({"kind": "destination", "between_ref": ref})
            if fused:
                raw_steps.append({"kind": "destination", "text": fused})
        else:
            ref, fused = _split_path_argument(arg)
            if ref:
                raw_steps.append({"kind": kind, "ref": ref})
            if fused:
                raw_steps.append({"kind": "destination", "text": fused})
    return raw_steps


def _resolve_ref_spec(node, ref: dict) -> tuple[str, list[dict]]:
    kind = ref["kind"]
    refs = ref["refs"]
    if kind == "between":
        return "between", [node.query_parser.parse(refs[0]), node.query_parser.parse(refs[1])]
    if kind == "between_collective":
        text = re.sub(r"^(the\s+)?two\s+", "the ", refs[0], flags=re.IGNORECASE)
        return "between_collective", [node.query_parser.parse(text)]
    return "near", [node.query_parser.parse(refs[0])]


def parse_instruction(node, question: str) -> dict:
    """question -> mission3용 top-level task dict. sysnav_node.question_callback이
    (LLMQueryParser.parse() 대신) 이 함수를 호출한다."""
    steps: list[dict] = []
    forbidden: list[dict] = []
    prompt_categories: list[str] = []

    raw_steps = _split_clauses(question)
    splitter = "rules"
    if not raw_steps:
        # 규칙 기반이 트리거 단어를 하나도 못 찾음(questions.json 30문장 밖의 새
        # 표현) - LLM에게 같은 절 분해를 대신 시킨다. 이것도 실패하면 raw_steps가
        # 빈 채로 남고, steps도 비어서 question_callback이 "파싱 실패"로 처리한다.
        raw_steps = node.instruction_splitter.split(question)
        splitter = "llm" if raw_steps else "rules"  # 대시보드 표시용(mission_dashboard.py)

    for raw in raw_steps:
        if raw["kind"] == "destination":
            if "text" in raw:
                parsed = node.query_parser.parse(raw["text"])
                steps.append({"is_stop": True, "resolve": "category", "parsed": parsed})
                prompt_categories.extend(parsed.get("detection_prompts") or [])
            else:
                point_mode, point_refs = _resolve_ref_spec(node, raw["between_ref"])
                steps.append({
                    "is_stop": True, "resolve": "point",
                    "point_mode": point_mode, "point_refs": point_refs,
                })
                for ref in point_refs:
                    prompt_categories.extend(ref.get("detection_prompts") or [])
        elif raw["kind"] == "positive_path":
            point_mode, point_refs = _resolve_ref_spec(node, raw["ref"])
            steps.append({
                "is_stop": False, "resolve": "point",
                "point_mode": point_mode, "point_refs": point_refs,
            })
            for ref in point_refs:
                prompt_categories.extend(ref.get("detection_prompts") or [])
        elif raw["kind"] == "negative_path":
            point_mode, point_refs = _resolve_ref_spec(node, raw["ref"])
            forbidden.append({"point_mode": point_mode, "point_refs": point_refs})
            for ref in point_refs:
                prompt_categories.extend(ref.get("detection_prompts") or [])

    return {
        "raw": question.strip(),
        "steps": steps,
        "global_forbidden": forbidden,
        # 절 분해를 규칙 기반/LLM 폴백 중 무엇으로 했는지 - mission_dashboard.py 표시용
        # (다른 미션의 parsed["parser"]와 같은 목적, LLMQueryParser.parse() 참고).
        "parser": splitter,
        # perception_job/scene_graph가 기대하는 top-level task 필드 - mission3는
        # "하나의 target"이 없으므로 빈 placeholder. detection_prompts만 실질적으로
        # 쓰인다(YOLO-World가 찾아야 할, 모든 절에 등장한 카테고리의 합집합).
        "target": "",
        "attributes": [],
        "relation": None,
        "reference_objects": [],
        "relation_chain": [],
        "detection_prompts": sorted(set(prompt_categories)),
    }


# ---------------------------------------------------------------------------
# 기하학적 참조("between"/"near") resolve - VLM 호출 없이 이미 관측된
# object_memory 후보의 위치만으로 근사한다 (정밀한 유일-정답 선택이 필요한
# object_reference와 달리, 통과 지점 근사치면 충분하다).
# ---------------------------------------------------------------------------

def _dist2(position, pose: dict) -> float:
    return (position[0] - pose["x"]) ** 2 + (position[1] - pose["y"]) ** 2


def _resolve_single_category_point(node, ref_parsed: dict, pose: dict):
    candidates = node.object_memory.find_by_category(ref_parsed["target"])
    if not candidates:
        return None
    nearest = min(candidates, key=lambda c: _dist2(c["position"], pose))
    return nearest["position"]


def _resolve_forbidden_segment(node, point_mode: str, point_refs: list[dict], pose: dict):
    """(point_a, point_b) 반환 - "near"는 같은 점을 두 번(원형 마스크용)."""
    if point_mode == "near":
        p = _resolve_single_category_point(node, point_refs[0], pose)
        return (p, p) if p is not None else None
    if point_mode == "between":
        pa = _resolve_single_category_point(node, point_refs[0], pose)
        pb = _resolve_single_category_point(node, point_refs[1], pose)
        return (pa, pb) if pa is not None and pb is not None else None
    if point_mode == "between_collective":
        candidates = node.object_memory.find_by_category(point_refs[0]["target"])
        if len(candidates) < 2:
            return None
        nearest_two = sorted(candidates, key=lambda c: _dist2(c["position"], pose))[:2]
        return (nearest_two[0]["position"], nearest_two[1]["position"])
    return None


def _resolve_point_ref(node, point_mode: str, point_refs: list[dict], pose: dict):
    segment = _resolve_forbidden_segment(node, point_mode, point_refs, pose)
    if segment is None:
        return None
    pa, pb = segment
    return tuple((a + b) / 2.0 for a, b in zip(pa, pb))


def _build_forbidden_mask(node, point_a, point_b) -> np.ndarray | None:
    planner = node.coverage_planner
    grid = planner.snapshot_grid()
    cell_a = planner.world_to_grid(point_a[0], point_a[1])
    cell_b = planner.world_to_grid(point_b[0], point_b[1])
    if cell_a is None or cell_b is None:
        return None
    radius_cells = max(1, int(round(config.INSTRUCTION_FORBIDDEN_RADIUS_M / planner.resolution)))
    mask = np.zeros(grid.shape, dtype=bool)
    for row, col in planner.line_cells(cell_a, cell_b):
        r0, r1 = max(0, row - radius_cells), min(grid.shape[0], row + radius_cells + 1)
        c0, c1 = max(0, col - radius_cells), min(grid.shape[1], col + radius_cells + 1)
        mask[r0:r1, c0:c1] = True
    return mask


def _try_resolve_forbidden(node, task: dict, pose: dict) -> None:
    if node.mission3_forbidden_mask is not None:
        return
    for forbidden in task.get("global_forbidden", []):
        segment = _resolve_forbidden_segment(
            node, forbidden["point_mode"], forbidden["point_refs"], pose
        )
        if segment is None:
            continue
        mask = _build_forbidden_mask(node, *segment)
        if mask is not None:
            node.mission3_forbidden_mask = mask
            node.get_logger().info(
                "🚧 FORBIDDEN REGION resolved - remaining legs will route around it"
            )
        return


# ---------------------------------------------------------------------------
# state machine
# ---------------------------------------------------------------------------

def loop(node, state: str, task: dict, task_id: int, pose: dict) -> None:
    if state == "MISSION3_SELECT_STEP":
        _select_step(node, task, task_id, pose)
    elif state == "MISSION3_NAVIGATE_STEP":
        _navigate_step(node, task, pose)


def on_job_result(node, task: dict, kind: str, result: dict, origin_state: str) -> None:
    if kind == "perception":
        _on_perception_result(node, origin_state)
    elif kind == "selection":
        _on_selection_result(node, result)
    elif kind == "exploration":
        _on_exploration_result(node, result)


def _on_perception_result(node, origin_state: str) -> None:
    # mission1과 같은 이유로 FOLLOW_EXPLORATION 중 관측은 이동을 방해하지 않는다.
    # OBSERVE에서 온 관측만 "이제 이번 step을 resolve할 수 있는지" 재시도한다.
    if origin_state == "OBSERVE":
        with node.state_lock:
            node.state = "MISSION3_SELECT_STEP"


def _on_selection_result(node, result: dict) -> None:
    if result.get("relation_pending") or result.get("attribute_pending"):
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
    _start_navigate_to_point(node, pose, selected["position"], is_object_target=True)


def _on_exploration_result(node, result: dict) -> None:
    route = result["route"]
    if not route:
        with node.state_lock:
            node.state = "FAILED"
        node.get_logger().warning(
            f"❌ TASK FAILED - mission3 step {node.mission3_step_index + 1} unresolved, "
            f"no reachable frontier remains ({node.coverage_planner.describe_last_plan_failure()})"
        )
        return
    node.exploration_route = deque(route)
    node.publish_next_exploration_goal()


def _select_step(node, task: dict, task_id: int, pose: dict) -> None:
    steps = task["steps"]
    if node.mission3_step_index >= len(steps):
        node.last_response_summary = f"{len(steps)}/{len(steps)} steps completed"
        with node.state_lock:
            node.state = "SUCCESS"
        node.get_logger().info("🚩🏁 ALL STEPS COMPLETE (task SUCCESS)")
        return

    _try_resolve_forbidden(node, task, pose)

    step = steps[node.mission3_step_index]
    if step["resolve"] == "category":
        node.submit_job(
            "selection", node.selection_job, task_id, step["parsed"], pose,
            origin_state="MISSION3_SELECT_STEP",
        )
        return

    point = _resolve_point_ref(node, step["point_mode"], step["point_refs"], pose)
    if point is None:
        # 참조 카테고리를 아직 못 봤다 - 계속 탐색한다.
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"
        return
    _start_navigate_to_point(node, pose, point, is_object_target=False)


def _start_navigate_to_point(node, pose: dict, point, is_object_target: bool) -> None:
    """mission2와 동일하게 node.start_target_navigation()으로 이동한다 - 목적지 좌표
    하나를 던지고 마는 대신, 현재 지도로 A* 경로를 만들어 hop 단위로 가면서 hop에
    도착할 때마다(그리고 주행 중 hop이 막히면 즉시) 경로를 다시 계산한다.

    이력: 한때 우리 쪽 "도달 불가" 판단(stuck-timeout/known-free gating)을 넣었다가
    실제로 갈 수 있는 목표를 중간에 포기하는 문제만 키워서 전부 걷어냈었다(2026-08-11,
    ad323c7/a6a7fa7). 그런데 그 상태에서는 반대로, 목적지가 가구 옆이라 도착 판정
    반경(0.35m) 안에 절대 못 들어가는 경우 로봇은 멈춰 있는데 step이 영원히 넘어가지
    않았다(실측: 0.43m 남기고 7분 정지, step 0/2). 지금 방식은 그때와 달리 "포기"가
    아니라 "재계획"이 기본이고, 포기는 A*가 경로 없음을 반환할 때만 한다.

    global_forbidden("avoiding the path between A and B")도 여기서 경로 계획에 넘긴다 -
    base autonomy의 point-to-point 이동으로는 특정 영역을 피하도록 강제할 수 없어서,
    이 제약은 우리가 직접 우회 경로를 만들어야만 지켜진다.
    """
    object_xy = None
    if is_object_target:
        # 접근 지점은 /terrain_map 기준으로 고른다 (navigation/terrain_monitor.py) -
        # base autonomy가 받아들일 지점을 처음부터 찍어야 엉뚱한 데로 끌려가지 않는다.
        x, y, _ = node.approach_pose_for(pose, point)
        object_xy = (float(point[0]), float(point[1]))
    else:
        # "take the path between A and B" 같은 경유점은 그 좌표 자체가 제약이라
        # terrain 기준으로 옮기면 안 된다.
        x, y = float(point[0]), float(point[1])
    # approach_pose_for()가 돌려주는 theta("도착해서 물체를 바라볼 방향") 대신 이동
    # 방향을 쓴다. README의 "go near"/"stop at"은 위치 도달만 요구하니 굳이 물체를
    # 바라보라고 강제할 이유가 없다.
    theta = math.atan2(y - pose["y"], x - pose["x"])

    node.get_logger().info(
        f"🧭 mission3 step {node.mission3_step_index + 1} -> goal=({x:.2f}, {y:.2f}, {theta:.2f})"
    )
    # goal을 실제로 발행한 뒤에 state를 옮긴다(mission2와 같은 순서) - 먼저 옮기면
    # start_target_navigation()이 예외로 죽었을 때 goal 없이 NAVIGATE_STEP에 들어가서
    # 직전 step의 stale goal로 도착 판정이 날 수 있다.
    #
    # marker_index를 넘기면 주행 중 목표가 다시 잡힐 때 RViz 마커도 따라 옮겨진다
    # (node.refresh_goal_marker()). 마커를 여기서 직접 그리지 않는 이유도 그것이다 -
    # 그리는 곳이 두 군데면 한쪽만 갱신되어 어긋난다.
    node.start_target_navigation(
        pose, (x, y), theta,
        forbidden_mask=node.mission3_forbidden_mask,
        object_xy=object_xy,
        marker_index=node.mission3_step_index,
    )
    node.refresh_goal_marker()
    with node.state_lock:
        node.state = "MISSION3_NAVIGATE_STEP"


def _navigate_step(node, task: dict, pose: dict) -> None:
    outcome = node.step_target_navigation(pose)
    if outcome == "driving":
        return
    if outcome == "unreachable":
        # 지금 지도로는 이 step의 목적지까지 갈 길이 없다 - 맵을 더 뚫고 다시 시도한다.
        # step_index는 올리지 않으므로 탐색 후 같은 step을 다시 resolve한다.
        node.clear_target_navigation()
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"
        node.get_logger().warning(
            f"🧭 mission3 step {node.mission3_step_index + 1} unreachable for now - "
            f"back to exploration to open up the map"
        )
        return

    steps = task["steps"]
    step = steps[node.mission3_step_index]
    node.get_logger().info(
        f"🚩 ARRIVED - mission3 step {node.mission3_step_index + 1}/{len(steps)} "
        f"({'stop' if step['is_stop'] else 'waypoint'}), "
        f"robot_pose=({pose['x']:.2f}, {pose['y']:.2f})"
    )
    node.clear_target_navigation()
    node.mission3_step_index += 1
    with node.state_lock:
        node.state = "MISSION3_SELECT_STEP"
