"""Thin query parser for YOLO-World prompts."""

from __future__ import annotations

import re

_LEADING_COMMAND = re.compile(
    r"^\s*(?:please\s+)?(?:find|locate|search for|look for|go to|navigate to|"
    r"take me to|where is|where are|identify|show me)\s+",
    re.IGNORECASE,
)
_ARTICLE = re.compile(r"\b(?:a|an|the|this|that|these|those)\b", re.IGNORECASE)
_RELATIONS = [
    ("in front of", "in_front_of"), ("to the left of", "left_of"),
    ("to the right of", "right_of"), ("left of", "left_of"),
    ("right of", "right_of"), ("next to", "beside"), ("beside", "beside"),
    # "nearest"(최상급/argmin)는 "near"(단순 근접 threshold)와 별개 relation이다 -
    # "near"만 있으면 두 relation 다 같은 문자열을 매칭해버려서 leftmost-match가 항상
    # "near" 하나로만 잡힌다. 반드시 "near"보다 먼저 (문자열 자체가 아니라 리스트에)
    # 있을 필요는 없다 - _find_leftmost_relation이 위치(start index) 기준으로 고르므로.
    ("closest to", "nearest"), ("nearest to", "nearest"), ("close to", "nearest"),
    ("near", "near"), ("between", "between"), ("on top of", "on"),
    ("on", "on"), ("under", "under"), ("below", "under"),
    ("above", "above"), ("behind", "behind"),
]
_ATTRIBUTES = {
    "white", "black", "red", "blue", "green", "yellow", "orange", "purple",
    "pink", "brown", "gray", "grey", "silver", "gold", "wooden", "wood",
    "metal", "metallic", "plastic", "glass", "leather", "fabric", "cloth",
    "small", "large", "big", "tiny", "tall", "short", "open", "closed",
    "round", "square",
}
_IRREGULAR = {"chairs": "chair", "tables": "table", "pillows": "pillow", "boxes": "box", "shelves": "shelf"}


def _clean(text: str) -> str:
    text = _ARTICLE.sub(" ", text)
    text = re.sub(r"[?!.,;:]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _singularize(word: str) -> str:
    lower = word.lower()
    if lower in _IRREGULAR:
        return _IRREGULAR[lower]
    if lower.endswith("ies") and len(lower) > 3:
        return lower[:-3] + "y"
    if lower.endswith("s") and not lower.endswith(("ss", "us")):
        return lower[:-1]
    return lower


def _split_attributes(phrase: str) -> tuple[str, list[str]]:
    tokens = phrase.lower().split()
    attributes = [token for token in tokens if token in _ATTRIBUTES]
    object_tokens = [token for token in tokens if token not in _ATTRIBUTES]
    if not object_tokens and tokens:
        object_tokens = tokens[-1:]
    if object_tokens:
        object_tokens[-1] = _singularize(object_tokens[-1])
    return " ".join(object_tokens).strip(), attributes


def _find_leftmost_relation(text: str) -> tuple[str, int, int] | None:
    """문장에 등장하는 모든 relation phrase 중, 실제로 문장에서 가장 먼저(왼쪽)
    나오는 것을 고른다. 예전엔 _RELATIONS 리스트 순서상 먼저 나오는 relation을
    골랐는데, 그러면 "closest to ... near ..."처럼 relation이 여러 개인 문장에서
    "closest to"가 먼저인데도 리스트에서 "near"가 앞이면 near가 잘못 선택됐다."""
    lowered = text.lower()
    best: tuple[str, int, int] | None = None
    for relation_text, canonical in _RELATIONS:
        match = re.search(rf"\b{re.escape(relation_text)}\b", lowered)
        if match and (best is None or match.start() < best[1]):
            best = (canonical, match.start(), match.end())
    return best


def _extract_relation_chain(phrase: str) -> list[dict]:
    """왼쪽부터 순서대로 (object, attributes, 다음 object와의 relation)을 뽑는다.

    "bowl closest to knife rack near trash can" ->
      [{"object":"bowl", "relation":"nearest"},
       {"object":"knife rack", "relation":"near"},
       {"object":"trash can", "relation":None}]

    relation이 하나뿐인 기존 문장(예: "chair near window")도 그대로 길이 2인
    체인으로 표현되므로 동작이 바뀌지 않는다. "between"은 3항 relation이라
    "X and Y" 두 reference를 한 번에 떼어내고 체인을 끝낸다(기존 동작 유지).
    """
    remaining = phrase
    chain: list[dict] = []
    while True:
        found = _find_leftmost_relation(remaining)
        if found is None:
            obj, attributes = _split_attributes(_clean(remaining))
            if obj:
                chain.append({"object": obj, "attributes": attributes, "relation": None})
            break

        canonical, start, end = found
        obj, attributes = _split_attributes(_clean(remaining[:start]))
        if not obj:
            # relation 앞에 물체가 없으면(문장 파싱 실패) 더 진행하지 않는다.
            break
        chain.append({"object": obj, "attributes": attributes, "relation": canonical})

        if canonical == "between":
            rest = remaining[end:]
            for part in re.split(r"\band\b|,", rest, flags=re.IGNORECASE):
                ref, ref_attributes = _split_attributes(_clean(part))
                if ref:
                    chain.append({"object": ref, "attributes": ref_attributes, "relation": None})
            break

        remaining = remaining[end:]

    return chain


def extract_target(question: str) -> dict:
    from sysnav.task.prompt_bank import enrich_task_prompts, load_alias_cache

    raw = question.strip()
    normalized = _LEADING_COMMAND.sub("", raw).strip()

    chain = _extract_relation_chain(normalized)

    target_object = chain[0]["object"] if chain else ""
    attributes = chain[0]["attributes"] if chain else []
    relation_name = chain[0]["relation"] if chain else None
    reference_objects = [node["object"] for node in chain[1:] if node["object"]]

    # 문장에 relation이 연쇄로 여러 개 들어온 경우(예: "A closest to B near C")를
    # (obj1, relation, obj2) triple들로도 남겨둔다. 지금 relation/reference_objects는
    # 여전히 "target -> 첫 relation"만 담아 기존 로직(spatial_relation_reasoner 등)과
    # 호환되고, relation_chain은 디버그/향후 다단계 relation 검증에 쓴다.
    relation_chain = [
        (chain[i]["object"], chain[i]["relation"], chain[i + 1]["object"])
        for i in range(len(chain) - 1)
        if chain[i]["relation"]
    ]

    prompts = []
    for node in chain:
        if node["object"] and node["object"] not in prompts:
            prompts.append(node["object"])

    return enrich_task_prompts({
        "raw": raw,
        "target": target_object,
        "attributes": attributes,
        "relation": relation_name,
        "reference_objects": reference_objects,
        "relation_chain": relation_chain,
        "detection_prompts": prompts,
    }, load_alias_cache())


def effective_relation_chain(task: dict) -> list[tuple[str, str, str]]:
    """relation_chain을 신뢰할 수 있는 (source_category, relation, target_category)
    triple 리스트로 정규화한다.

    extract_target()이 만든 task는 항상 relation_chain을 갖고 있어 그대로 쓰지만,
    relation_chain 없이 relation/reference_objects[0]만 있는(예: 예전 방식으로
    손으로 만든) task dict를 위해 단일 hop 체인으로 대체 생성하는 하위 호환
    fallback도 겸한다.
    """
    chain = task.get("relation_chain")
    if chain:
        return list(chain)
    relation = task.get("relation")
    references = task.get("reference_objects") or []
    if not relation or not references:
        return []
    target = str(task.get("target", "")).lower()
    return [(target, str(relation), str(references[0]).lower())]
