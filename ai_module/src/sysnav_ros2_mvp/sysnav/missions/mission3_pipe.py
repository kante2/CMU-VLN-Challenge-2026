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

import itertools
import math
import re
from collections import deque

import numpy as np

from sysnav import config
from sysnav.missions import path_gate
from sysnav.reasoning.attribute_filter import filter_by_attributes
from sysnav.task.query_parser import effective_relation_chain, merge_reference_attributes

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
    _REF_LOG_SEEN.clear()  # 새 질문이면 참조 후보 진단 로그를 다시 한 번 찍는다.

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
    passed = filter_by_attributes(node, candidates, ref_parsed.get("attributes"))
    _log_ref_candidates(node, ref_parsed, candidates, passed)
    return passed


# 후보 구성이 바뀔 때만 진단 로그를 찍기 위한 캐시. _select_step이 0.5초마다 도는데
# 매번 찍으면 로그가 폭발한다. parse_instruction()이 질문마다 비운다.
_REF_LOG_SEEN: dict[str, tuple] = {}


def _log_ref_candidates(node, ref_parsed: dict, before: list[dict], after: list[dict]) -> None:
    """속성 필터가 참조 후보를 어떻게 갈랐는지 한 블록으로 찍는다.

    왜 필요한가: "the round tables"에서 "round"는 분명히 파싱되고(task/query_parser.py의
    _ATTRIBUTES에 있다) filter_by_attributes도 분명히 호출되는데, 실측에서 원형이 아닌
    화분받침이 살아남아 진짜 원형 테이블 대신 선택됐다. 원인 후보가 5개(속성이 빈 채로
    파싱됨 / 검증 기능 꺼짐 / VLM이 정말 round=True로 판정 / fail-closed가 정답을 떨어뜨림 /
    카테고리 문자열 불일치)인데 로그가 없어서 어느 것인지 구분할 수 없었다. 아래 3항목이
    그 5개를 전부 구분해준다 - 추측 대신 데이터로 정하기 위한 것이다."""
    target = str(ref_parsed.get("target", ""))
    attributes = [str(a) for a in (ref_parsed.get("attributes") or [])]
    passed_ids = {int(c["object_id"]) for c in after}
    signature = (
        tuple(sorted((int(c["object_id"]), round(float(c.get("confidence", 0.0)), 3)) for c in before)),
        tuple(sorted(passed_ids)),
        tuple(attributes),
    )
    if _REF_LOG_SEEN.get(target) == signature:
        return
    _REF_LOG_SEEN[target] = signature

    # 이 함수는 순수 진단용이다. object_memory가 조회 API를 다 갖추지 않은 경우에도
    # (테스트의 축약 fake 등) 절대 resolve 경로를 무너뜨리면 안 되므로 선택적으로 읽는다.
    memory_get = getattr(node.object_memory, "get", None)
    all_nodes = getattr(node.object_memory, "all_nodes", None)

    verification = "on" if config.ATTRIBUTE_VERIFICATION_ENABLED else "off"
    lines = [
        f"🔎 ref '{target}' attributes={attributes} verification={verification} "
        f"-> {len(after)}/{len(before)} passed"
    ]
    for candidate in sorted(before, key=lambda c: -float(c.get("confidence", 0.0))):
        object_id = int(candidate["object_id"])
        # 필터가 방금 캐싱한 판정까지 보려면 memory에서 다시 읽어야 한다(candidate는
        # filter 호출 **전**에 뜬 복사본이라 self_attributes가 비어 있을 수 있다).
        stored = (memory_get(object_id) or {}) if memory_get else {}
        cached = stored.get("self_attributes") or candidate.get("self_attributes") or {}
        if object_id in passed_ids:
            verdict = "PASS"
        else:
            failed = [a for a in attributes if not cached.get(a, False)]
            unverified = [a for a in failed if a not in cached]
            verdict = f"DROP({'unverified: ' if unverified else ''}{','.join(failed) or '?'})"
        position = candidate.get("position", (0.0, 0.0, 0.0))
        extent = candidate.get("extent_3d", (0.0, 0.0, 0.0))
        has_image = "yes" if stored.get("representative_image") is not None else "no"
        lines.append(
            f"     #{object_id} conf={float(candidate.get('confidence', 0.0)):.2f} "
            f"pos=({position[0]:.2f},{position[1]:.2f}) "
            f"extent=({extent[0]:.2f},{extent[1]:.2f}) "
            f"obs={candidate.get('observation_count', 0)} image={has_image} "
            f"attrs={cached} {verdict}"
        )
    # find_by_category는 정확한 소문자 문자열 매칭이라 "coffee table"로 저장된 노드는
    # find_by_category("table")에 안 잡힌다. 그 경우를 즉시 드러낸다.
    census: dict[str, int] = {}
    for stored in (all_nodes() if all_nodes else []):
        category = str(stored.get("category", ""))
        if target and target in category and category != target:
            census[category] = census.get(category, 0) + 1
    if census:
        related = ", ".join(f"{name}({count})" for name, count in sorted(census.items()))
        lines.append(f"     related categories not matched by find_by_category('{target}'): {related}")
    node.get_logger().info("\n".join(lines))


# ---------------------------------------------------------------------------
# "between A and B" 참조 쌍 선택.
#
# 예전엔 A와 B를 **각각 독립적으로** "로봇에 가장 가까운 것"으로 골랐다. 그래서
# (1) detection confidence가 전혀 반영되지 않았고(실측: 신뢰도 0.58 오검출이 0.85 정답을
# 이김), (2) 두 물체가 실제로 통과 가능한 게이트를 이루는지 아무도 확인하지 않았으며,
# (3) live pose에 의존해서 unreachable 재시도마다 목표가 떠돌았다.
#
# 이제 A 후보 x B 후보 전체를 **짝으로** 채점한다. 로봇 위치는 쓰지 않는다.
# ---------------------------------------------------------------------------

def _object_radius(candidate: dict) -> float:
    """extent_3d에서 XY 반경. 게이트 선분을 물체 몸통 밖에서 시작시키기 위한 것."""
    extent = candidate.get("extent_3d") or (0.0, 0.0, 0.0)
    try:
        radius = 0.5 * max(abs(float(extent[0])), abs(float(extent[1])))
    except (TypeError, ValueError, IndexError):
        return 0.0
    if not math.isfinite(radius) or radius <= 0.0:
        return 0.0
    return min(radius, config.MISSION3_BETWEEN_OBJECT_RADIUS_MAX_M)


def _pair_clearance(planner, traversable, position_a, position_b, radius_a, radius_b):
    """(통과 가능 셀 비율, 중점 통과 가능 여부). grid origin이 아직 없으면 None.

    선분을 양 끝에서 각 물체 반경만큼 잘라낸 구간만 샘플링한다 - 물체 중심 셀은 당연히
    occupied라, 안 자르면 모든 짝이 똑같이 감점돼 이 항목이 무의미해진다."""
    ax, ay = float(position_a[0]), float(position_a[1])
    bx, by = float(position_b[0]), float(position_b[1])
    gap = math.hypot(bx - ax, by - ay)
    if gap < 1e-6:
        return 0.0, False
    ux, uy = (bx - ax) / gap, (by - ay) / gap
    # 양쪽 반경을 합쳐도 gap을 넘지 않도록 줄인다(물체끼리 겹치는 경우).
    trim_a, trim_b = radius_a, radius_b
    if trim_a + trim_b >= gap:
        scale = 0.45 * gap / max(trim_a + trim_b, 1e-6)
        trim_a, trim_b = trim_a * scale, trim_b * scale
    start = (ax + ux * trim_a, ay + uy * trim_a)
    end = (bx - ux * trim_b, by - uy * trim_b)

    cell_start = planner.world_to_grid(start[0], start[1])
    cell_end = planner.world_to_grid(end[0], end[1])
    if cell_start is None or cell_end is None:
        return None
    cells = planner.line_cells(cell_start, cell_end)
    if not cells:
        return 0.0, False
    rows, cols = traversable.shape
    clear = sum(
        1 for row, col in cells
        if 0 <= row < rows and 0 <= col < cols and bool(traversable[row, col])
    )
    midpoint_cell = planner.world_to_grid((ax + bx) / 2.0, (ay + by) / 2.0)
    midpoint_clear = (
        midpoint_cell is not None
        and 0 <= midpoint_cell[0] < rows
        and 0 <= midpoint_cell[1] < cols
        and bool(traversable[midpoint_cell])
    )
    return clear / len(cells), midpoint_clear


def _pair_score(candidate_a: dict, candidate_b: dict) -> dict:
    """쌍 점수. 로봇 위치를 쓰지 않는다.

    confidence는 min(a, b)이다 - 게이트는 약한 쪽 기둥만큼만 믿을 수 있다.
    통과가능성은 여기 안 들어간다: _select_object_pair의 tier 필터가 담당한다. 그래야
    "통과 가능한 게이트들 중에서는 신뢰도 높은 쪽이 이긴다"가 성립한다."""
    position_a = candidate_a["position"]
    position_b = candidate_b["position"]
    gap = math.hypot(
        float(position_a[0]) - float(position_b[0]),
        float(position_a[1]) - float(position_b[1]),
    )
    confidence = min(
        float(candidate_a.get("confidence", 0.0)), float(candidate_b.get("confidence", 0.0))
    )
    confidence = min(max(confidence, 0.0), 1.0)
    min_gap = config.MISSION3_BETWEEN_MIN_GAP_M
    max_gap = config.MISSION3_BETWEEN_MAX_GAP_M
    gap_score = 1.0 - (gap - min_gap) / max(max_gap - min_gap, 1e-6)
    gap_score = min(max(gap_score, 0.0), 1.0)
    weight_confidence = config.MISSION3_BETWEEN_CONFIDENCE_WEIGHT
    weight_gap = config.MISSION3_BETWEEN_GAP_WEIGHT
    total = max(weight_confidence + weight_gap, 1e-6)
    return {
        "score": (weight_confidence * confidence + weight_gap * gap_score) / total,
        "confidence": confidence,
        "gap": gap,
        "gap_score": gap_score,
    }


def _pair_iter(candidates_a: list[dict], candidates_b: list[dict], collective: bool):
    """평가할 (a, b) 짝. collective면 같은 리스트의 i<j 조합(자기 자신과 짝 짓지 않음),
    아니면 교차곱에서 object_id가 같은 짝만 제외한다."""
    if collective:
        return list(itertools.combinations(candidates_a, 2))
    return [
        (a, b)
        for a, b in itertools.product(candidates_a, candidates_b)
        if int(a["object_id"]) != int(b["object_id"])
    ]


def _select_object_pair(node, candidates_a, candidates_b, collective: bool, label: str):
    """A 후보 x B 후보 전체를 짝으로 채점해 최고점 하나를 고른다. 로봇 위치는 안 쓴다.

    tier 사다리를 쓰는 이유: hard reject가 후보를 전부 날리면 _select_step이 영원히
    PLAN_EXPLORATION으로 돌아간다(exploration_exhausted 뒤에도 point step엔 탈출구가
    없다). 런 초반엔 통로가 UNKNOWN이고 unknown은 통과 불가로 취급되므로 순진한
    hard filter는 실제로 전부 떨어뜨린다. 또 이번 실측처럼 정답 테이블이 소파에 붙어
    있으면 strict에서 떨어질 수 있는데, 그때 unverified가 회수하고 confidence+gap이
    올바르게 고른다. 후보가 2개 미만일 때만 None이다."""
    pairs = _pair_iter(candidates_a, candidates_b, collective)
    if not pairs:
        return None

    planner = getattr(node, "coverage_planner", None)
    grid = planner.snapshot_grid() if planner is not None else None

    def clearance_for(inflation_m):
        if planner is None or grid is None:
            return lambda a, b, ra, rb: None
        traversable = planner.traversable_mask(grid, inflation_m)
        return lambda a, b, ra, rb: _pair_clearance(planner, traversable, a, b, ra, rb)

    tiers = (
        ("strict", config.MISSION3_BETWEEN_GATE_CLEARANCE_M, True, True),
        ("passable", config.ROBOT_CLEARANCE_M, True, False),
        ("unverified", None, False, False),
    )
    min_gap = config.MISSION3_BETWEEN_MIN_GAP_M
    max_gap = config.MISSION3_BETWEEN_MAX_GAP_M
    min_fraction = config.MISSION3_BETWEEN_MIN_CLEAR_FRACTION

    for tier_name, inflation_m, check_geometry, check_fraction in tiers:
        measure = clearance_for(inflation_m) if check_geometry else None
        scored = []
        for candidate_a, candidate_b in pairs:
            metrics = _pair_score(candidate_a, candidate_b)
            clearance = None
            if measure is not None:
                result = measure(
                    candidate_a["position"], candidate_b["position"],
                    _object_radius(candidate_a), _object_radius(candidate_b),
                )
                if result is not None:
                    clearance, midpoint_clear = result
                    if not midpoint_clear:
                        continue
                    if check_fraction and clearance < min_fraction:
                        continue
            if check_geometry and not (min_gap <= metrics["gap"] <= max_gap):
                continue
            metrics["clearance"] = clearance
            scored.append((candidate_a, candidate_b, metrics))
        if scored:
            scored.sort(
                key=lambda item: (
                    -item[2]["score"], int(item[0]["object_id"]), int(item[1]["object_id"])
                )
            )
            _log_pair_selection(node, label, tier_name, len(pairs), scored)
            return scored[0][0], scored[0][1]
    return None


def _log_pair_selection(node, label: str, tier: str, total_pairs: int, scored: list) -> None:
    best_a, best_b, _ = scored[0]
    lines = [
        f"🚪 gate {label} [{tier}]: {total_pairs} pair(s), {len(scored)} passed -> "
        f"{best_a['category']}#{best_a['object_id']} + {best_b['category']}#{best_b['object_id']}"
    ]
    for index, (candidate_a, candidate_b, metrics) in enumerate(
        scored[: max(1, config.MISSION3_BETWEEN_PAIR_LOG_TOP_N)]
    ):
        clearance = metrics.get("clearance")
        clearance_text = "n/a" if clearance is None else f"{clearance:.2f}"
        lines.append(
            f"     {candidate_a['category']}#{candidate_a['object_id']}"
            f"+{candidate_b['category']}#{candidate_b['object_id']}  "
            f"score={metrics['score']:.2f} conf={metrics['confidence']:.2f} "
            f"gap={metrics['gap']:.2f}m clear={clearance_text}"
            f"{'   PICK' if index == 0 else ''}"
        )
    message = "\n".join(lines)
    # strict가 아닌 tier로 이겼다는 건 게이트가 기하학적으로 검증되지 않았다는 뜻이라
    # 런 로그에서 눈에 띄어야 한다.
    if tier == "strict":
        node.get_logger().info(message)
    else:
        node.get_logger().warning(message)


def _resolve_single_category_point(node, ref_parsed: dict, pose: dict):
    candidates = _category_candidates(node, ref_parsed)
    if not candidates:
        return None
    nearest = min(candidates, key=lambda c: _dist2(c["position"], pose))
    return nearest["position"]


def _resolve_forbidden_segment(node, point_mode: str, point_refs: list[dict], pose: dict):
    """(point_a, point_b) 반환 - "near"는 같은 점을 두 번(원형 마스크용).

    between 계열은 pose를 쓰지 않는다 - 두 참조를 각각 "로봇 최근접"으로 뽑던 것을
    _select_object_pair의 쌍 채점으로 바꿨기 때문이다(신뢰도/게이트 폭/통과가능성을
    같이 본다). pose 인자는 "near" 분기와 호출부 시그니처 때문에 남아 있다."""
    if point_mode == "near":
        p = _resolve_single_category_point(node, point_refs[0], pose)
        return (p, p) if p is not None else None
    if point_mode == "between":
        candidates_a = _category_candidates(node, point_refs[0])
        candidates_b = _category_candidates(node, point_refs[1])
        if not candidates_a or not candidates_b:
            return None
        label = f"'{point_refs[0]['target']}' x '{point_refs[1]['target']}'"
        pair = _select_object_pair(node, candidates_a, candidates_b, False, label)
        return (pair[0]["position"], pair[1]["position"]) if pair else None
    if point_mode == "between_collective":
        candidates = _category_candidates(node, point_refs[0])
        if len(candidates) < 2:
            return None
        label = f"'{point_refs[0]['target']}' (collective)"
        pair = _select_object_pair(node, candidates, candidates, True, label)
        return (pair[0]["position"], pair[1]["position"]) if pair else None
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


def _resolve_step_point(node, step: dict, pose: dict):
    """step의 좌표를 정한다. between 계열은 한 번 확정하면 step에 얼려서 재사용한다.

    왜: _select_step은 navigation이 시작되기 전까지 매 tick 돌고, unreachable 재시도나
    탐사 왕복 뒤에도 다시 돈다. 매번 새로 풀면 그 사이 object_memory의 position이 EMA로
    갱신되는 것만으로도 goal이 흔들리고, RViz의 goalN과 로봇이 실제로 향하는 곳이
    어긋난다. (near는 매 실행의 terrain에서 접근점을 다시 고르는 것이 의도된 동작이라
    얼리지 않는다.) step dict는 질문마다 parse_instruction이 새로 만들므로 자동으로
    초기화된다."""
    if step.get("point_mode") not in ("between", "between_collective"):
        return _resolve_point_ref(node, step["point_mode"], step["point_refs"], pose)

    frozen = step.get("resolved_segment")
    if frozen is None:
        segment = _resolve_forbidden_segment(
            node, step["point_mode"], step["point_refs"], pose
        )
        if segment is None:
            return None, None
        frozen = step["resolved_segment"] = segment
    pa, pb = frozen
    return tuple((a + b) / 2.0 for a, b in zip(pa, pb)), frozen


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


def _required_attributes(step: dict) -> dict[str, list[str]]:
    """이 step에서 카테고리별로 요구되는 속성(category -> attributes).

    _step_categories()와 짝을 이룬다: 저쪽이 "무엇이 보여야 하는가"라면 이쪽은
    "그중 어떤 것이어야 하는가"다. 파서가 이미 target의 attributes와 참조 물체의
    reference_attributes를 따로 뽑아주므로 여기서는 하나의 맵으로 합치기만 한다
    (같은 카테고리가 양쪽에 나오면 합집합 - merge_reference_attributes와 동일 규칙)."""
    pairs: list[tuple[str, list[str]]] = []
    if step.get("resolve") == "category":
        parsed = step["parsed"]
        pairs.append((parsed.get("target", ""), parsed.get("attributes") or []))
        pairs.extend((category, attributes) for category, attributes
                     in (parsed.get("reference_attributes") or {}).items())
    else:
        pairs.extend((ref.get("target", ""), ref.get("attributes") or [])
                     for ref in step.get("point_refs", []))
    return merge_reference_attributes(pairs)


# 기하로 근사할 때 "가장 먼 것"을 골라야 하는 relation. 나머지는 전부 "가장 가까운
# 것"으로 근사한다 - near/nearest/beside는 그게 정확히 맞고, on/under/above도 XY로는
# 가장 가까운 후보가 정답인 경우가 대부분이다. left_of/behind처럼 방향이 있는 relation은
# 근사가 약하지만, mission3는 "아무 데도 안 가는 것"보다 "대충 맞는 데로 가는 것"이
# 항상 낫다(부분점수 + 궤적 채점).
_FARTHEST_RELATIONS = {"farthest"}


def _xy_gap2(position_a, position_b) -> float:
    return (position_a[0] - position_b[0]) ** 2 + (position_a[1] - position_b[1]) ** 2


def _best_effort_step_target(node, step: dict, pose: dict) -> tuple[tuple | None, str]:
    """VLM 확정이 안 될 때, 지금 관측된 것만으로 이 step의 목적지를 기하로 정한다.

    반환: (position 또는 None, 무슨 근거로 골랐는지 설명하는 문자열).

    참조 물체에는 속성 필터를 걸지 않는다 - filter_by_attributes는 fail-closed라
    아직 검증 안 된 참조를 통째로 날려버려서, 정작 여기(폴백)에서는 참조가 하나도
    안 남게 된다. target 후보에만 건다.
    """
    parsed = step["parsed"]
    candidates = _category_candidates(node, parsed)
    if not candidates:
        candidates = node.object_memory.find_by_category(parsed["target"])
    if not candidates:
        return None, f"no {parsed['target']} observed yet"

    chain = effective_relation_chain(parsed)
    if not chain:
        nearest = min(candidates, key=lambda c: _dist2(c["position"], pose))
        return nearest["position"], "nearest candidate to the robot (no relation in this step)"

    _, relation, reference_category = chain[0]
    references = node.object_memory.find_by_category(reference_category)
    # 참조 물체의 속성 제약("closest to the **black** chair")을 먼저 건다. 통과하는
    # 것이 하나도 없을 때만 생 후보로 degrade하고, 그 사실을 basis에 남긴다 - 예전엔
    # 무조건 생 후보를 써서 "black을 무시했다"는 것이 로그 어디에도 안 남았다.
    reference_attributes = _required_attributes(step).get(
        str(reference_category).strip().lower()
    )
    attribute_note = ""
    if reference_attributes and references:
        narrowed = filter_by_attributes(node, references, reference_attributes)
        if narrowed:
            references = narrowed
        else:
            attribute_note = (
                f" - '{' '.join(reference_attributes)}' NOT applied "
                f"(no {reference_category} passed it)"
            )
    reference_positions = [reference["position"] for reference in references]
    if not reference_positions:
        nearest = min(candidates, key=lambda c: _dist2(c["position"], pose))
        return (
            nearest["position"],
            f"nearest candidate to the robot - '{relation} {reference_category}' NOT applied "
            f"({reference_category} has no 3D position)",
        )

    def gap(candidate) -> float:
        return min(_xy_gap2(candidate["position"], position) for position in reference_positions)

    picked = (max if relation in _FARTHEST_RELATIONS else min)(candidates, key=gap)
    return (
        picked["position"],
        f"geometric '{relation} {reference_category}' over {len(candidates)} candidate(s) "
        f"and {len(reference_positions)} reference(s), gap={math.sqrt(gap(picked)):.2f}m"
        f"{attribute_note}",
    )


def _missing_categories(node, step: dict) -> list[str]:
    """이 step을 판정하려면 무엇이 아직 관측 안 됐는가(로그용 설명 문자열 리스트).

    카테고리 존재만 보면 안 된다: "go near the lamp closest to the **black** chair"에서
    흰 식탁의자만 4개 보이는 상태를 "chair 관측됨"으로 처리하면, 검은 의자를 한 번도
    못 봤는데도 이 가드를 그냥 통과해 _best_effort_step_target이 흰 의자를 기준으로
    lamp를 골라버린다(실측 2026-08-25: 그래서 엉뚱한 방향으로 주행했다).

    "아직 못 본 참조"와 "봤지만 속성이 안 맞는 참조"를 구분하는 것이 요점이다 -
    전자는 더 탐사해야 하고, 폴백은 후자에만 쓰여야 한다. 속성 제약이 붙은 카테고리는
    **그 속성을 만족하는 인스턴스가 하나라도** 있어야 관측된 것으로 친다.
    (판정 결과는 object_memory에 캐싱되므로 매 tick 불러도 VLM 재호출은 없다.)"""
    required = _required_attributes(step)
    missing: list[str] = []
    for category in _step_categories(step):
        candidates = node.object_memory.find_by_category(category)
        if not candidates:
            missing.append(str(category))
            continue
        attributes = required.get(str(category).strip().lower())
        if attributes and not filter_by_attributes(node, candidates, attributes):
            missing.append(f"{' '.join(attributes)} {category}")
    return missing


def active_categories(node, task: dict) -> list[str] | None:
    """지금 이 순간 판정에 실제로 필요한 카테고리(= 현재 step + 전역 금지구역 참조).

    None을 돌려주면 "제한 없음"이다(mission1/2는 이 함수 자체가 없어 그렇게 동작한다).

    왜: task["detection_prompts"]는 **모든 step의 합집합**이라, step 3("stop at the
    cabinet with a picture above it")을 하는 중에도 step 1/2의 lamp·chair 저신뢰
    검출까지 매 프레임 Gemini 재확인(perception/detection_verifier.py)에 딸려 들어갔다
    (실측 2026-08-25: 이미 끝난 step의 물체를 계속 재검증). 호출 비용이자 그대로
    주행 지연이다 - 지금 step이 안 쓰는 카테고리는 물어볼 이유가 없다.

    금지구역 참조는 특정 step이 아니라 전체 경로에 걸리는 제약이라 항상 포함한다."""
    steps = task.get("steps") or []
    index = getattr(node, "mission3_step_index", 0)
    categories: list[str] = []
    if 0 <= index < len(steps):
        categories.extend(_step_categories(steps[index]))
    for forbidden in task.get("global_forbidden") or []:
        for ref in forbidden.get("point_refs") or []:
            categories.extend(ref.get("detection_prompts") or [])
    # step을 아직 못 정했거나(파싱 실패) 카테고리가 하나도 안 나오면 예전처럼 전부 검증한다.
    return list(dict.fromkeys(categories)) or None


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
        _resolve_pending_step(node, result)
        return
    selected = node.object_memory.get(result["selected_id"])
    with node.sensor_lock:
        pose = None if node.latest_pose is None else dict(node.latest_pose)
    if selected is None or pose is None:
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"
        return
    _start_navigate_to_point(node, pose, selected["position"], is_object_target=True)


def _resolve_pending_step(node, result: dict) -> None:
    """selection_job이 "아직 확정 못 하겠다"(relation/attribute/verification pending)고
    돌려줬을 때 무엇을 할지 정한다.

    예전에는 무조건 PLAN_EXPLORATION으로 되돌렸다. 그런데 selection_job의 유일한
    탈출구는 `mission2_exploration_deadline_reached`인데 그 플래그는 Mission 2 전용이라
    Mission 3에서는 절대 서지 않는다 - 즉 관계가 끝내 검증 안 되면(참조 물체가 유리창
    이라 3D grounding이 안 되거나, Gemini final_verification이 계속 거절하거나)
    SELECT_STEP -> selection -> pending -> PLAN_EXPLORATION 을 탐사가 소진될 때까지
    돌다가 결국 FAILED로 끝났다. **타겟도 참조 물체도 이미 눈앞에 있는데도** 그랬다.

    Mission 3의 채점은 subgoal 순서와 실제 주행 궤적이고 부분점수가 있다. "정확히
    맞을 때까지 안 움직인다"보다 "지금 아는 것으로 가장 그럴듯한 곳에 바로 간다"가
    항상 낫다. 그래서 이 step에 필요한 카테고리가 **전부 관측돼 있으면** 더 기다리지
    않고 기하로 목적지를 정해 바로 goal을 찍는다. 아직 못 본 물체가 있을 때만
    예전처럼 탐사로 돌아간다.
    """
    with node.state_lock:
        task = None if node.task is None else dict(node.task)
    with node.sensor_lock:
        pose = None if node.latest_pose is None else dict(node.latest_pose)
    steps = (task or {}).get("steps") or []
    if pose is None or node.mission3_step_index >= len(steps):
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"
        return
    step = steps[node.mission3_step_index]

    missing = _missing_categories(node, step)
    if missing and not node.mission3_exploration_exhausted:
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"
        node.get_logger().info(
            f"🔍 step {node.mission3_step_index + 1}: selection deferred, still looking for "
            f"{', '.join(missing)}"
        )
        return

    position, basis = _best_effort_step_target(node, step, pose)
    if position is None:
        with node.state_lock:
            node.state = "PLAN_EXPLORATION"
        node.get_logger().info(
            f"🔍 step {node.mission3_step_index + 1}: selection deferred ({basis})"
        )
        return

    reason = next(
        key for key in ("relation_pending", "attribute_pending", "verification_pending")
        if result.get(key)
    )
    node.get_logger().warning(
        f"🎯 step {node.mission3_step_index + 1}: VLM could not confirm ({reason}) but every "
        f"referenced object is already observed - committing now by {basis}"
    )
    _start_navigate_to_point(node, pose, position, is_object_target=True)


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

    point, segment = _resolve_step_point(node, step, pose)
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
    #
    # 새 subgoal이므로 재발행 카운터를 0으로 되돌린다. **재발행 경로에서는 절대 리셋하면
    # 안 된다** - 그러면 카운터가 영원히 0이라 MISSION3_SUBGOAL_MAX_RETRIES 상한이
    # 무의미해지고, 막으려던 무한루프가 그대로 돈다.
    node.mission3_subgoal_retries = 0
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
        node.mission3_subgoal_retries += 1
        if node.mission3_subgoal_retries >= config.MISSION3_SUBGOAL_MAX_RETRIES:
            # 포기하기 전에 딱 한 번, 물체 접근 상한(0.9m)을 풀고 "base autonomy가
            # 허용하는 가장 가까운 지점"을 찾아본다. 그 상한은 의미 규칙이라 평소엔
            # 지켜야 하지만, 여기까지 왔다는 건 지키면 이 step을 통째로 잃는다는 뜻이다.
            if _retry_with_relaxed_approach(node, pose):
                return
            _give_up_step(node, task)
            return
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
    node.mission3_subgoal_retries = 0
    with node.state_lock:
        node.state = "MISSION3_SELECT_STEP"


def _retry_with_relaxed_approach(node, pose: dict) -> bool:
    """MISSION3_OBJECT_APPROACH_MAX_M을 풀고 접근점을 다시 고른다. 성공하면 True.

    실측 2026-08-24(probe_waypoint_push.py): 이 씬은 travArea의 **7.1%**만 base
    autonomy가 목적지로 받아준다(clearance >= 0.75m). 그런데 Mission 3의 링 샘플링은
    0.9m 링 하나 x 7각도 = 후보 7개만 보므로 거의 항상 실패하고, terrain을 안 보는
    고정 standoff로 폴백해 명령 불가한 좌표를 잡았다. 여기서는 링이 아니라 통과 지점
    집합을 직접 훑으므로, 갈 수 있는 곳이 있으면 결정론적으로 찾아낸다.

    물체가 아닌 좌표를 향하던 step(between/near point)은 다시 고를 물체가 없으므로
    False를 돌려 그대로 포기한다.
    """
    object_xy = node.target_object_xy
    if object_xy is None:
        return False
    x, y, theta = node.approach_pose_for(
        pose, (object_xy[0], object_xy[1], 0.0),
        max_distance_m=config.MISSION3_OBJECT_APPROACH_MAX_M,
        allow_relaxed=True,
    )
    if node.target_goal_xy is not None and math.hypot(
        x - node.target_goal_xy[0], y - node.target_goal_xy[1]
    ) < config.GOAL_REACHED_DISTANCE_M:
        # 같은 자리를 다시 고른 것이면 재시도해봐야 결과가 같다.
        return False
    node.start_target_navigation(
        pose, (x, y), theta,
        forbidden_mask=node.target_forbidden_mask,
        object_id=node.target_object_id,
        object_xy=object_xy,
        marker_index=node.target_marker_index,
    )
    node.mission3_subgoal_retries = 0
    node.refresh_goal_marker()
    with node.state_lock:
        node.state = "MISSION3_NAVIGATE_STEP"
    node.get_logger().warning(
        f"🚧 step {node.mission3_step_index + 1}: the {config.MISSION3_OBJECT_APPROACH_MAX_M:.1f}m "
        f"approach limit is unreachable here - relaxing it and retargeting to "
        f"({x:.2f}, {y:.2f}) [{node.terrain_monitor.last_selection}]"
    )
    return True


def _give_up_step(node, task: dict) -> None:
    """확정된 subgoal을 계속 발행하지 못했다 - 그 step을 포기하고 다음으로 넘어간다.

    mission3는 marker까지 만든 subgoal을 탐사 목표로 덮어쓰지 않는데(채점이 subgoal
    순서와 궤적을 보므로), 그 정책에 상한이 없어서 목적지가 base autonomy에게 명령
    불가한 자리이면 "재발행 -> 거부 -> unreachable -> 재발행"을 영원히 돌았다.

    실측 2026-08-24: 변기 접근점의 clearance가 0.42m라 waypointConverter가 후보로
    안 받았고(기준 0.75m), 스냅하면 로봇에서 0.36m 지점으로 떨어져 갈 거리가 없었다.
    이 씬은 travArea의 7.1%만 명령 가능해서 그런 자리가 드물지 않다.

    한 step에 갇혀 남은 step을 통째로 버리는 것보다, 여기까지를 최선으로 인정하고
    다음 step으로 가는 쪽이 부분점수에서 항상 낫다. 다만 그 사실을 로그에 분명히
    남긴다 - "지나갔다"와 "못 갔지만 넘어갔다"가 구분돼야 한다.
    """
    steps = task["steps"]
    index = node.mission3_step_index
    node.clear_target_navigation()
    node.mission3_step_index += 1
    node.mission3_subgoal_retries = 0
    with node.state_lock:
        node.state = "MISSION3_SELECT_STEP"
    node.get_logger().warning(
        f"🚧 step {index + 1}/{len(steps)} GIVEN UP - the committed subgoal was not "
        f"commandable for {config.MISSION3_SUBGOAL_MAX_RETRIES} consecutive attempts "
        f"({node.terrain_monitor.last_selection}); moving on to keep the remaining steps"
    )
