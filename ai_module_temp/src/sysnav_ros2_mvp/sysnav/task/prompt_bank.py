"""Generalized YOLO-World prompt expansion and category normalization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import os
from pathlib import Path
import tempfile
import threading


# Synonyms are semantically equivalent names. Visual aliases describe the same
# physical object by its likely appearance. Broad fallbacks are deliberately
# excluded from the first-pass detector prompts to avoid false positives.
OBJECT_ONTOLOGY: dict[str, dict[str, tuple[str, ...]]] = {
    "tv cabinet": {
        "synonyms": (
            "tv cabinet",
            "television cabinet",
            "tv stand",
            "television stand",
            "media console",
            "media cabinet",
        ),
        "visual_aliases": ("cabinet under a television",),
        "fallbacks": ("cabinet",),
    },
    "trash can": {
        "synonyms": (
            "trash can",
            "garbage can",
            "waste bin",
            "garbage bin",
            "rubbish bin",
            "waste basket",
        ),
        "visual_aliases": (),
        "fallbacks": ("waste container",),
    },
    "sofa": {
        "synonyms": ("sofa", "couch", "settee"),
        "visual_aliases": (),
        "fallbacks": ("seating furniture",),
    },
    "knife rack": {
        "synonyms": ("knife rack", "knife holder"),
        "visual_aliases": (
            "knife block",
            "kitchen knife holder",
            "magnetic knife strip",
            "wall-mounted knife rack",
            "set of kitchen knives",
        ),
        "fallbacks": ("kitchen utensil holder",),
    },
}

MAX_GENERAL_PROMPTS_PER_CATEGORY = max(
    1, int(os.getenv("YOLO_MAX_PROMPTS_PER_CATEGORY", "7"))
)
MAX_ATTRIBUTE_PROMPTS = max(0, int(os.getenv("YOLO_MAX_ATTRIBUTE_PROMPTS", "2")))
ALIAS_CACHE_PATH = Path(os.getenv(
    "SYSNAV_PROMPT_ALIAS_CACHE",
    "/home/docker/ai_module/debug/prompt_alias_cache.json",
))
_CACHE_LOCK = threading.Lock()
_FORBIDDEN_RELATION_PHRASES = (
    " closest to ", " farthest from ", " next to ", " near ", " beside ",
    " left of ", " right of ", " in front of ", " behind ", " under ",
    " above ", " between ",
)


def _clean(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _static_prompts(canonical: str, *, include_fallbacks: bool = False) -> tuple[str, ...]:
    entry = OBJECT_ONTOLOGY.get(canonical)
    if entry is None:
        return (canonical,) if canonical else ()
    values = [*entry["synonyms"], *entry["visual_aliases"]]
    if include_fallbacks:
        values.extend(entry["fallbacks"])
    return tuple(dict.fromkeys(_clean(value) for value in values if _clean(value)))


# Backward-compatible flattened view for diagnostics and existing imports.
PROMPT_BANK = {
    canonical: _static_prompts(canonical)
    for canonical in OBJECT_ONTOLOGY
}


def _alias_to_canonical(
    dynamic_aliases: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, str]:
    aliases = {
        alias: canonical
        for canonical, prompts in PROMPT_BANK.items()
        for alias in prompts
    }
    for raw_canonical, raw_prompts in (dynamic_aliases or {}).items():
        canonical = _clean(raw_canonical)
        if not canonical:
            continue
        canonical = aliases.get(canonical, canonical)
        aliases.setdefault(canonical, canonical)
        for prompt in raw_prompts:
            cleaned = _clean(prompt)
            if cleaned:
                aliases.setdefault(cleaned, canonical)
    return aliases


def canonical_category(
    value: object,
    dynamic_aliases: Mapping[str, Iterable[str]] | None = None,
) -> str:
    """Return the ontology category for a known synonym."""
    cleaned = _clean(value)
    return _alias_to_canonical(dynamic_aliases).get(cleaned, cleaned)


def _valid_generated_alias(value: object) -> str:
    """Accept only short, visible noun phrases from runtime LLM generation."""
    cleaned = _clean(value)
    padded = f" {cleaned} "
    if not cleaned or len(cleaned.split()) > 6:
        return ""
    if any(phrase in padded for phrase in _FORBIDDEN_RELATION_PHRASES):
        return ""
    return cleaned


def normalize_dynamic_aliases(payload: object) -> dict[str, list[str]]:
    """Validate Gemini/cache alias data and return a stable category mapping."""
    normalized: dict[str, list[str]] = {}
    if isinstance(payload, Mapping):
        items = payload.items()
    elif isinstance(payload, list):
        items = (
            (item.get("category"), item.get("aliases", []))
            for item in payload
            if isinstance(item, Mapping)
        )
    else:
        return normalized

    for raw_category, raw_aliases in items:
        category = _valid_generated_alias(raw_category)
        if not category or not isinstance(raw_aliases, (list, tuple)):
            continue
        category = {
            alias: canonical
            for canonical, prompts in PROMPT_BANK.items()
            for alias in prompts
        }.get(category, category)
        aliases = [category]
        aliases.extend(
            alias
            for value in raw_aliases
            if (alias := _valid_generated_alias(value))
        )
        normalized[category] = list(dict.fromkeys(aliases))[
            :MAX_GENERAL_PROMPTS_PER_CATEGORY
        ]
    return normalized


def load_alias_cache(path: Path = ALIAS_CACHE_PATH) -> dict[str, list[str]]:
    try:
        with _CACHE_LOCK:
            return normalize_dynamic_aliases(json.loads(path.read_text(encoding="utf-8")))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return {}


def update_alias_cache(
    generated_aliases: object,
    path: Path = ALIAS_CACHE_PATH,
) -> dict[str, list[str]]:
    """Merge validated aliases into an atomic JSON cache and return all entries."""
    generated = normalize_dynamic_aliases(generated_aliases)
    if not generated:
        return load_alias_cache(path)

    with _CACHE_LOCK:
        try:
            existing = normalize_dynamic_aliases(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (FileNotFoundError, OSError, ValueError, TypeError):
            existing = {}
        for category, aliases in generated.items():
            existing[category] = list(dict.fromkeys([
                *existing.get(category, []),
                *aliases,
            ]))[:MAX_GENERAL_PROMPTS_PER_CATEGORY]

        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            dir=str(path.parent),
            text=True,
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(existing, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.chmod(temporary_name, 0o664)
            os.replace(temporary_name, path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise
        return existing


def category_prompts(
    category: object,
    dynamic_aliases: Mapping[str, Iterable[str]] | None = None,
) -> tuple[str, ...]:
    """Return bounded, unique first-pass prompts for one category."""
    canonical = canonical_category(category, dynamic_aliases)
    prompts = list(_static_prompts(canonical))
    for value in (dynamic_aliases or {}).get(canonical, ()):
        alias = _valid_generated_alias(value)
        if alias:
            prompts.append(alias)
    if not prompts and canonical:
        prompts.append(canonical)
    return tuple(dict.fromkeys(prompts))[:MAX_GENERAL_PROMPTS_PER_CATEGORY]


def build_detection_prompts(
    target: object,
    attributes: Iterable[object] = (),
    reference_objects: Iterable[object] = (),
    dynamic_aliases: Mapping[str, Iterable[str]] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Build bounded general and target-attribute prompts with canonical mappings."""
    canonical_target = canonical_category(target, dynamic_aliases)
    cleaned_attributes = list(dict.fromkeys(
        cleaned
        for value in attributes
        if (cleaned := _clean(value))
    ))
    prompts: list[str] = []
    prompt_categories: dict[str, str] = {}

    def add(prompt: str, canonical: str) -> None:
        cleaned_prompt = _clean(prompt)
        if cleaned_prompt and cleaned_prompt not in prompt_categories:
            prompts.append(cleaned_prompt)
            prompt_categories[cleaned_prompt] = canonical

    target_aliases = category_prompts(canonical_target, dynamic_aliases)
    for alias in target_aliases:
        add(alias, canonical_target)

    if cleaned_attributes:
        attribute_phrase = " ".join(cleaned_attributes)
        for alias in target_aliases[:MAX_ATTRIBUTE_PROMPTS]:
            add(f"{attribute_phrase} {alias}", canonical_target)

    for reference in reference_objects:
        canonical_reference = canonical_category(reference, dynamic_aliases)
        for alias in category_prompts(canonical_reference, dynamic_aliases):
            add(alias, canonical_reference)
    return prompts, prompt_categories


def enrich_task_prompts(
    task: dict,
    dynamic_aliases: Mapping[str, Iterable[str]] | None = None,
) -> dict:
    """Normalize task categories and attach expanded detector prompt metadata."""
    dynamic_aliases = normalize_dynamic_aliases(dynamic_aliases or {})
    enriched = dict(task)
    parsed_detection_categories = [
        canonical_category(value, dynamic_aliases)
        for value in task.get("detection_prompts", [])
        if canonical_category(value, dynamic_aliases)
    ]
    enriched["target"] = canonical_category(enriched.get("target"), dynamic_aliases)
    enriched["reference_objects"] = [
        canonical_category(value, dynamic_aliases)
        for value in enriched.get("reference_objects", [])
        if canonical_category(value, dynamic_aliases)
    ]
    enriched["relation_chain"] = [
        (
            canonical_category(source, dynamic_aliases),
            str(relation),
            canonical_category(reference, dynamic_aliases),
        )
        for source, relation, reference in enriched.get("relation_chain", [])
        if canonical_category(source, dynamic_aliases)
        and relation
        and canonical_category(reference, dynamic_aliases)
    ]

    constraints = []
    for constraint in enriched.get("constraints", []):
        normalized = dict(constraint)
        normalized["references"] = [
            canonical_category(value, dynamic_aliases)
            for value in constraint.get("references", [])
            if canonical_category(value, dynamic_aliases)
        ]
        constraints.append(normalized)
    enriched["constraints"] = constraints

    prompts, categories = build_detection_prompts(
        enriched["target"],
        enriched.get("attributes", []),
        (
            parsed_detection_categories[1:]
            if parsed_detection_categories
            else [
                reference
                for constraint in constraints
                for reference in constraint["references"]
            ]
        ),
        dynamic_aliases,
    )
    enriched["detection_prompts"] = prompts
    enriched["prompt_categories"] = categories
    return enriched
