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
from sysnav.missions import path_gate
from sysnav.reasoning.attribute_filter import filter_by_attributes

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


def _category_candidates(node, ref_parsed: dict) -> list[dict]:
    """참조 카테고리 후보 중 속성 제약("the **round** tables")까지 만족하는 것만.

    파서가 "round tables"를 category="table" + attributes=["round"]로 쪼개주므로
    (task/llm_query_parser.py), 검출은 table 전부를 잡고 round 여부는 여기서 VLM으로
    가린다. 예전엔 find_by_category만 써서 형용사가 좌표 선택에 전혀 반영되지 않았다."""
    candidates = node.object_memory.find_by_category(ref_parsed["target"])
    if not candidates:
        return []
    return filter_by_attributes(node, candidates, ref_parsed.get("attributes"))


def _resolve_single_category_point(node, ref_parsed: dict, pose: dict):
    candidates = _category_candidates(node, ref_parsed)
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
        candidates = _category_candidates(node, point_refs[0])
        if len(candidates) < 2:
            return None
        nearest_two = sorted(candidates, key=lambda c: _dist2(c["position"], pose))[:2]
        return (nearest_two[0]["position"], nearest_two[1]["position"])
    return None


def _resolve_point_ref(node, point_mode: str, point_refs: list[dict], pose: dict):
    """(waypoint, segment) 반환. waypoint는 두 참조 물체의 중점(near면 그 물체 자체),
    segment는 게이트 판정/시각화에 쓰는 A-B 선분 원본이다 - 예전엔 중점만 돌려주고
    선분을 버려서 "실제로 그 사이를 지나갔는가"를 아무도 확인할 수 없었다."""
    segment = _resolve_forbidden_segment(node, point_mode, point_refs, pose)
    if segment is None:
        return None, None
    pa, pb = segment
    return tuple((a + b) / 2.0 for a, b in zip(pa, pb)), segment


def _gate_segment(step: dict, segment):
    """이 step에서 게이트 통과 판정을 쓸 것인가.

    "take the path between A and B"/"pass by"(is_stop=False)만 대상이다.
    "go between A and B"(is_stop=True)는 그 사이에 **정지**하는 것이 요구사항이라
    가로지르기만 해서는 안 되므로 기존 반경 판정을 그대로 쓴다(선분은 시각화만 된다)."""
    if segment is None or step.get("is_stop"):
        return None
    if step.get("point_mode") not in ("between", "between_collective"):
        return None
    return segment


def _uses_object_approach(point_mode: str) -> bool:
    """near(object)는 물체 중심 대신 terrain 기반 접근점을 사용한다."""
    return point_mode == "near"


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


def _step_categories(step: dict) -> list[str]:
    """이 step을 판정하려면 무엇이 관측돼 있어야 하는가.

    target뿐 아니라 관계의 참조 물체도 포함한다("the cup near the TV remote"라면
    cup과 tv remote 둘 다). detection_prompts가 이미 그 합집합이다."""
    if step.get("resolve") == "category":
        return list(step["parsed"].get("detection_prompts") or [])
    prompts: list[str] = []
    for ref in step.get("point_refs", []):
        prompts.extend(ref.get("detection_prompts") or [])
    return prompts


def _missing_categories(node, step: dict) -> list[str]:
    return [
        category for category in _step_categories(step)
        if not node.object_memory.find_by_category(category)
    ]


def _describe_forbidden(task: dict) -> str:
    parts = []
    for forbidden in task.get("global_forbidden", []):
        refs = ", ".join(str(ref.get("target", "?")) for ref in forbidden["point_refs"])
        parts.append(f"{forbidden['point_mode']}({refs})")
    return " + ".join(parts) or "-"


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
    if (
        result.get("relation_pending")
        or result.get("attribute_pending")
        or result.get("verification_pending")
    ):
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
        # 금지구역 참조 물체를 찾으려고 탐사 중이었다면, 여기서 실패로 끝내지 않는다.
        # 더 볼 곳이 없다는 뜻이므로 "못 찾았다"로 확정하고 step 진행으로 넘어간다 -
        # 제약은 미검증으로 남지만, 아무것도 못 하는 것보다 부분점수가 낫다.
        if not node.mission3_exploration_exhausted:
            # 더 볼 곳이 없다 - 증거가 부족해도 여기서 실패로 끝내지 않고 진행한다.
            # 부분점수가 있으므로 아무것도 못 하는 것보다 낫다.
            node.mission3_exploration_exhausted = True
            with node.state_lock:
                node.state = "MISSION3_SELECT_STEP"
            if node.mission3_forbidden_mask is None and node.task and node.task.get("global_forbidden"):
                node.get_logger().warning(
                    "🚧 forbidden-area reference not found after full exploration - "
                    "continuing WITHOUT the avoidance constraint (it cannot be enforced)"
                )
            else:
                node.get_logger().warning(
                    "🔍 exploration exhausted - deciding with whatever evidence exists"
                )
            return
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
        # 제약을 못 지킨 채 끝났으면 그대로 적어둔다. 예전엔 step 카운트만 보고
        # SUCCESS를 찍어서, 금지구역을 한 번도 적용 못 한 실행도 초록불로 보였다
        # (2026-08-22: "avoid the path near the cabinet"인데 cabinet 미발견 -> 마스크
        # 없음 -> 그대로 SUCCESS). 채점은 그걸 감점하므로 로그가 거짓말을 하면 안 된다.
        unenforced = bool(task.get("global_forbidden")) and node.mission3_forbidden_mask is None
        node.last_response_summary = (
            f"{len(steps)}/{len(steps)} steps completed"
            + (" (avoidance constraint NOT enforced)" if unenforced else "")
        )
        with node.state_lock:
            node.state = "SUCCESS"
        if unenforced:
            node.get_logger().warning(
                "🚩 ALL STEPS COMPLETE but the avoidance constraint was never enforced "
                f"({_describe_forbidden(task)} was never located) - the followed path may "
                "pass through a forbidden area"
            )
        else:
            node.get_logger().info("🚩🏁 ALL STEPS COMPLETE (task SUCCESS)")
        return

    _try_resolve_forbidden(node, task, pose)

    # 금지구역("avoiding the path near/between X")이 문장에 있는데 아직 그 참조 물체를
    # 못 찾았으면, step을 확정하기 전에 **먼저 찾는다**.
    #
    # 왜: 마스크가 없으면 탐사도 목적지 주행도 그 구역을 그대로 지나간다. README 채점은
    # 실제 주행 궤적 전체를 보고 "passes through areas it is forbidden to go through"를
    # 감점하므로, 못 찾은 채로 돌아다니는 것 자체가 손해다. 실제로 한 번은 cabinet을
    # 끝내 못 찾아 마스크 없이 주행하고도 SUCCESS가 떴다(2026-08-22).
    #
    # 단 영원히 기다리면 안 된다 - 탐사가 소진되면 forbidden_search_exhausted가 서고,
    # 그때는 제약 미검증 상태임을 남긴 채 step 진행으로 넘어간다(부분점수라도 받는 게
    # 아무것도 못 하는 것보다 낫다).
    if (
        task.get("global_forbidden")
        and node.mission3_forbidden_mask is None
        and not node.mission3_exploration_exhausted
    ):
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"
        node.get_logger().info(
            "🚧 exploring first to locate the forbidden-area reference "
            f"({_describe_forbidden(task)}) before resolving steps"
        )
        return

    step = steps[node.mission3_step_index]

    # 이 step이 가리키는 물체들(target + 관계의 참조 물체)이 아직 다 안 보이면, 판정을
    # 시도하지 말고 계속 돌아다니며 지도를 그린다.
    #
    # 왜: selection_job은 참조 물체를 3D로 못 잡았을 때 "후보 사진에 그게 보이나"를
    # Gemini에게 묻는 이미지 폴백으로 넘어간다. 그건 참조 물체가 구조적으로 grounding
    # 안 되는 경우(유리창 등)를 위한 것인데, 단지 "아직 안 가봐서 못 본" 경우에도 발동해
    # 성급하게 확정해버린다 - 실측(2026-08-22): tv remote를 한 번도 못 봤는데 컵 2개만
    # 보고 2:32에 SUCCESS. 게다가 매 시도마다 Gemini 호출이 나가 시간 예산을 깎는다.
    missing = _missing_categories(node, step)
    if missing and not node.mission3_exploration_exhausted:
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"
        node.get_logger().info(
            f"🔍 step {node.mission3_step_index + 1}: still looking for "
            f"{', '.join(missing)} - continuing exploration before deciding"
        )
        return

    if step["resolve"] == "category":
        node.submit_job(
            "selection", node.selection_job, task_id, step["parsed"], pose,
            node.mission3_step_index,
            origin_state="MISSION3_SELECT_STEP",
        )
        return

    point, segment = _resolve_point_ref(node, step["point_mode"], step["point_refs"], pose)
    if point is None:
        # 참조 카테고리를 아직 못 봤다 - 계속 탐색한다.
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"
        return
    # near(TV) 같은 경유점은 검출된 물체의 3D 중심 자체가 아니라, 매 실행의
    # /terrain_map에서 동적으로 고른 접근 가능 지점으로 가야 한다. 반면 between(A,B)는
    # 두 물체 사이 좌표를 통과하는 것이 제약의 의미이므로 그 좌표를 유지한다.
    _start_navigate_to_point(
        node, pose, point,
        is_object_target=_uses_object_approach(step["point_mode"]),
        gate_segment=_gate_segment(step, segment),
    )


def _start_navigate_to_point(
    node, pose: dict, point, is_object_target: bool, gate_segment=None
) -> None:
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
        # Mission 3의 semantic subgoal은 물체 앞이어야 하므로, 일반 navigation의 2.2m
        # 탐색 범위 대신 모든 go-to/near object에 0.9m 상한을 적용한다.
        x, y, _ = node.approach_pose_for(
            pose, point, max_distance_m=config.MISSION3_OBJECT_APPROACH_MAX_M
        )
        object_xy = (float(point[0]), float(point[1]))
    else:
        # "take the path between A and B" 같은 경유점은 그 좌표 자체가 제약이라
        # terrain 기준으로 옮기면 안 된다.
        x, y = float(point[0]), float(point[1])
    # approach_pose_for()가 돌려주는 theta("도착해서 물체를 바라볼 방향") 대신 이동
    # 방향을 쓴다. README의 "go near"/"stop at"은 위치 도달만 요구하니 굳이 물체를
    # 바라보라고 강제할 이유가 없다.
    theta = math.atan2(y - pose["y"], x - pose["x"])

    # 게이트는 goal 발행 **전에** 세운다 - 판정 시작 시점을 "이 step이 시작한 순간"으로
    # 고정해서, 직전 step을 주행하다 우연히 지나간 궤적이 통과로 잡히지 않게 한다.
    path_gate.arm_gate(node, gate_segment, pose)

    node.get_logger().info(
        f"🧭 mission3 step {node.mission3_step_index + 1} -> goal=({x:.2f}, {y:.2f}, {theta:.2f})"
        + ("" if gate_segment is None else
           f", gate=({gate_segment[0][0]:.2f},{gate_segment[0][1]:.2f})-"
           f"({gate_segment[1][0]:.2f},{gate_segment[1][1]:.2f})")
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
    # Mission 3 전용 1m 도착 반경. 공용 step_target_navigation()은 Mission 2도 쓰므로
    # 그 안의 0.5m 기준을 바꾸지 않고 여기서만 먼저 판정한다.
    goal_xy = node.target_goal_xy
    within_radius = (
        goal_xy is not None
        and math.hypot(
            float(goal_xy[0]) - float(pose["x"]),
            float(goal_xy[1]) - float(pose["y"]),
        ) <= config.MISSION3_TARGET_SUCCESS_DISTANCE_M
    )
    # "take the path between A and B"는 두 물체 사이를 실제로 가로지르는 것이 제약이다.
    # 중점 반경 판정과 OR로 묶는다 - 두 물체가 거의 붙어 있어 게이트가 짧으면 교차
    # 판정이 예민해지므로, 그때는 기존 반경 판정이 백업이 된다.
    gate_crossed = path_gate.update_gate_crossing(node, pose)
    if gate_crossed:
        node.refresh_goal_marker()  # 통과한 게이트는 RViz에서 초록으로 바뀐다
    mission3_reached = gate_crossed or within_radius
    outcome = "arrived" if mission3_reached else node.step_target_navigation(pose)
    if outcome == "driving":
        return
    if outcome == "unreachable":
        # marker까지 만든 확정 subgoal은 탐사 목표로 덮어쓰지 않는다. 예전에는 여기서
        # PLAN_EXPLORATION으로 돌아가 /way_point_with_heading에 전혀 다른 frontier를
        # 발행했기 때문에, RViz에는 goalN이 남아 있는데 로봇은 다른 곳으로 향했다.
        #
        # 채점은 subgoal의 순서와 실제 trajectory를 보므로, 같은 최종 좌표/금지 마스크/
        # 물체 접근 정보를 보존한 채 주행기를 재시작한다. 도착하기 전에는 step index도
        # 올리지 않는다. 결과적으로 한번 marker가 생긴 step은 다음 marker로 넘어가기
        # 전에 반드시 계속 같은 subgoal을 명령한다.
        goal_xy = node.target_goal_xy
        final_theta = node.target_final_theta
        forbidden_mask = node.target_forbidden_mask
        object_id = node.target_object_id
        object_xy = node.target_object_xy
        marker_index = node.target_marker_index
        if goal_xy is None or final_theta is None:
            node.get_logger().error(
                "🧭 mission3 subgoal retry failed - committed marker has no target coordinate"
            )
            return
        node.start_target_navigation(
            pose,
            goal_xy,
            final_theta,
            forbidden_mask=forbidden_mask,
            object_id=object_id,
            object_xy=object_xy,
            marker_index=marker_index,
        )
        node.refresh_goal_marker()
        with node.state_lock:
            node.state = "MISSION3_NAVIGATE_STEP"
        node.get_logger().warning(
            f"🧭 mission3 step {node.mission3_step_index + 1} not reached - "
            "re-publishing the committed subgoal (exploration will not override it)"
        )
        return

    steps = task["steps"]
    step = steps[node.mission3_step_index]
    node.get_logger().info(
        f"🚩 ARRIVED - mission3 step {node.mission3_step_index + 1}/{len(steps)} "
        f"({'stop' if step['is_stop'] else 'waypoint'}), "
        f"via={'gate' if gate_crossed else 'radius'}, "
        f"robot_pose=({pose['x']:.2f}, {pose['y']:.2f})"
    )
    node.clear_target_navigation()
    node.mission3_step_index += 1
    with node.state_lock:
        node.state = "MISSION3_SELECT_STEP"
