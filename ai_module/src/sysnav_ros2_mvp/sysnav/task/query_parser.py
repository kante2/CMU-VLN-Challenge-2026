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


def extract_target(question: str) -> dict:
    raw = question.strip()
    normalized = _LEADING_COMMAND.sub("", raw).strip()
    target_phrase = normalized
    reference_phrase = ""
    relation_name = None

    lowered = normalized.lower()
    for relation_text, canonical in _RELATIONS:
        match = re.search(rf"\b{re.escape(relation_text)}\b", lowered)
        if match:
            target_phrase = normalized[:match.start()]
            reference_phrase = normalized[match.end():]
            relation_name = canonical
            break

    target_object, attributes = _split_attributes(_clean(target_phrase))
    reference_objects = []
    if reference_phrase:
        parts = re.split(r"\band\b|,", reference_phrase, flags=re.IGNORECASE) if relation_name == "between" else [reference_phrase]
        for part in parts:
            ref, _ = _split_attributes(_clean(part))
            if ref:
                reference_objects.append(ref)

    prompts = []
    for value in [target_object, *reference_objects]:
        if value and value not in prompts:
            prompts.append(value)

    return {
        "raw": raw,
        "target": target_object,
        "attributes": attributes,
        "relation": relation_name,
        "reference_objects": reference_objects,
        "detection_prompts": prompts,
    }
