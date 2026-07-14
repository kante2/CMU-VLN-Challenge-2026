#!/usr/bin/env python3
"""Instruction/object filtering before spatial reasoning."""

import re

from tmah_vlm.sort3d.data.objects import normalize_text


STOPWORDS = {
    "find", "locate", "go", "to", "the", "a", "an", "and", "then", "stop",
    "at", "on", "in", "near", "between", "closest", "nearest", "furthest",
    "farthest", "left", "right", "front", "behind", "above", "below", "of",
    "from", "with", "take", "path", "object", "room",
}

ALIASES = {
    "tv": {"tv", "television", "screen", "monitor"},
    "television": {"tv", "television", "screen", "monitor"},
    "computer": {"computer", "computer monitor", "monitor"},
    "monitor": {"computer", "computer monitor", "monitor"},
    "potted plant": {"potted plant", "plant"},
    "plant": {"potted plant", "plant"},
    "paper cup": {"paper cup", "cup"},
    "cup": {"paper cup", "cup"},
    "file cabinet": {"file cabinet", "cabinet"},
    "cabinet": {"file cabinet", "cabinet"},
}


def tokenize(text):
    return [tok for tok in re.split(r"[^a-z0-9]+", normalize_text(text)) if tok]


def expanded_terms(term):
    term = normalize_text(term)
    terms = {term}
    terms.update(ALIASES.get(term, set()))
    for key, values in ALIASES.items():
        if term in values:
            terms.add(key)
            terms.update(values)
    return {normalize_text(item) for item in terms if item}


def extract_relevant_terms(instruction, object_names=None):
    text = normalize_text(instruction)
    terms = []

    for phrase in sorted(object_names or [], key=len, reverse=True):
        phrase = normalize_text(phrase)
        if phrase and phrase in text and phrase not in terms:
            terms.append(phrase)

    for token in tokenize(text):
        if token not in STOPWORDS and len(token) > 1 and token not in terms:
            terms.append(token)
    return terms


def object_matches(obj, term):
    blob = obj.text_blob()
    return any(candidate in blob for candidate in expanded_terms(term))


def filter_relevant_objects(objects, instruction):
    names = sorted({obj.name for obj in objects})
    terms = extract_relevant_terms(instruction, names)
    relevant = []
    for obj in objects:
        if any(object_matches(obj, term) for term in terms):
            relevant.append(obj)
    return relevant, terms


def objects_named(objects, name):
    return [obj for obj in objects if object_matches(obj, name)]
