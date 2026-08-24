"""속성(φ) 제약으로 후보를 거르는 공용 경로.

SysNav paper Sec. IV-A-1의 on-demand self-attribute 판정(reasoning/attribute_verifier.py)을
"후보 리스트 -> 통과한 후보 리스트"로 감싼 것이다. 원래 이 블록은 sysnav_node.selection_job과
missions/mission1_pipe.py에 같은 코드로 두 벌 있었는데, mission3의 좌표 해석과 관계 판정에서도
같은 필터가 필요해져서 한 곳으로 모았다.

중요: 여기서도 fail-open 하지 않는다. VLM 호출이 실패하거나 아직 판정이 안 된 후보는
"통과 안 함"으로 취급해서, 호출 쪽이 확정을 미루고 계속 탐색하도록 만든다 - "속성을 확인
안 했는데 확정해버리는" 것이 원래 고치려던 버그다(attribute_verifier.py 모듈 주석 참고).
"""

from __future__ import annotations

from sysnav import config


def filter_by_attributes(node, candidates: list[dict], attributes: list[str] | None) -> list[dict]:
    """attributes를 모두 만족하는 candidate만 반환한다.

    검증 결과는 object_memory에 캐싱되므로(update_self_attributes) 같은 물체를 다시
    물어보지 않는다. 요구 속성이 없거나 기능이 꺼져 있으면 원본을 그대로 돌려준다.
    """
    required = [str(value).strip().lower() for value in (attributes or []) if str(value).strip()]
    if not required or not candidates or not config.ATTRIBUTE_VERIFICATION_ENABLED:
        return list(candidates)

    results = node.attribute_verifier.verify(candidates, required)
    for candidate in candidates:
        newly_checked = results.get(int(candidate["object_id"]), {})
        if newly_checked:
            node.object_memory.update_self_attributes(int(candidate["object_id"]), newly_checked)
    return [
        candidate for candidate in candidates
        if all(
            results.get(int(candidate["object_id"]), {}).get(attribute, False)
            for attribute in required
        )
    ]


def reference_allowed_ids(node, task: dict) -> dict[str, set[int]]:
    """task["reference_attributes"](category -> 요구 속성)를 "그 카테고리에서 허용되는
    object_id 집합"으로 바꾼다.

    왜 필요한가: 속성 검증은 지금까지 **target 후보에만** 걸렸다. "the lamp closest to
    the black chair"는 target=lamp라 attributes가 비어 있어 검증이 아예 안 돌았고,
    nearest의 argmin이 **모든** chair를 대상으로 돌아가 흰 의자가 답이 되곤 했다.
    reasoning/spatial_relation_reasoner.py가 이 화이트리스트로 참조 후보를 한 번 더 거른다.

    속성 요구가 없는 카테고리는 아예 키를 만들지 않는다 - 없는 키는 "제한 없음"으로
    해석되므로 기존 동작이 그대로 유지된다.
    """
    allowed: dict[str, set[int]] = {}
    for category, attributes in (task.get("reference_attributes") or {}).items():
        if not attributes:
            continue
        candidates = node.object_memory.find_by_category(category)
        if not candidates:
            continue
        passed = filter_by_attributes(node, candidates, attributes)
        allowed[str(category).strip().lower()] = {
            int(candidate["object_id"]) for candidate in passed
        }
    return allowed
