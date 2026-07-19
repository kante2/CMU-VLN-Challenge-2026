#!/usr/bin/env python3
"""Decompose an object-reference / numerical question into structured grounding
constraints.

Two paths:
  - VLMDecomposer.decompose(): authoritative, uses the VLM with a structured
    schema (prompt adapted from SysNav vlm_node instruction decomposition, BSD-3).
  - heuristic_decompose(): ROS/VLM-free rule-based extraction. Used as a fast
    offline fallback AND as the thing we can unit-test on all 30 object-reference
    training questions without an API key.

Fields mirror SysNav's decomposition so the downstream verification prompts line
up: target object, its attributes, a spatial relation to an anchor, a region
(room) condition, and the anchor's attributes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Decomposition:
    target_object: str                       # head noun, e.g. "pillow"
    attributes: list[str] = field(default_factory=list)      # e.g. ["red"]
    spatial_relation: str = ""               # e.g. "closest to", "between", "on"
    anchor_object: str = ""                  # e.g. "book on the stool"
    region: str = ""                         # e.g. "in the student lounge"
    raw: str = ""

    def as_instruction(self) -> str:
        attrs = " ".join(self.attributes)
        return f"Find the {attrs} {self.target_object} {self.spatial_relation} " \
               f"{self.anchor_object} {self.region}".strip()


# --------------------------------------------------------------------------- #
# Heuristic (offline) decomposition
# --------------------------------------------------------------------------- #
# words that, once seen, end the target noun phrase (they begin a relation/clause)
_RELATION_WORDS = {
    "on", "in", "between", "near", "closest", "farthest", "furthest", "that",
    "which", "below", "above", "under", "with", "to", "from", "next", "by",
    "behind", "beside", "over", "is", "are", "closest,", "of",
}
# Color/size attributes that lead the noun phrase. These need a VISUAL (VLM)
# check, so they are extracted as attributes and deferred to verification, not
# geometric filtering. IDENTITY/material adjectives (potted, framed, folding,
# wall, wooden, metal, glass, plastic) are deliberately NOT here: they belong to
# the object's class ("potted plant", "wall lamp") and must stay attached to the
# noun so label matching works — treating them as droppable attributes zeroed
# valid counts.
_ATTRIBUTES = {
    "red", "blue", "green", "yellow", "black", "white", "orange", "brown",
    "gray", "grey", "silver", "gold", "golden", "purple", "pink", "teal",
    "small", "large", "big", "tall", "short", "round", "square",
}
_LEADING = re.compile(r"^\s*(find|locate|identify)\s+", re.IGNORECASE)
_DETERMINER = re.compile(r"^\s*(the|a|an)\s+", re.IGNORECASE)
# numerical lead-ins, e.g. "How many red pillows are ...", "Count the number of ..."
_NUM_LEAD = re.compile(
    r"^\s*(how many|count the number of|count)\s+", re.IGNORECASE
)


def _strip_leads(q: str) -> str:
    q = q.strip().rstrip(".")
    q = _NUM_LEAD.sub("", q)
    q = _LEADING.sub("", q)
    q = _DETERMINER.sub("", q)
    return q


def heuristic_decompose(question: str) -> Decomposition:
    """Rule-based extraction of target object + leading attributes + a coarse
    spatial relation. Deterministic, no dependencies — good enough for routing,
    fallback answers, and as a sanity baseline; the VLM path supersedes it when
    a key is available."""
    body = _strip_leads(question)
    tokens = body.split()

    attributes: list[str] = []
    i = 0
    # consume leading adjectives
    while i < len(tokens) and tokens[i].lower().strip(",") in _ATTRIBUTES:
        attributes.append(tokens[i].lower().strip(","))
        i += 1

    # head noun phrase: take tokens until a relation word
    head: list[str] = []
    while i < len(tokens):
        w = tokens[i].lower().strip(",")
        if w in _RELATION_WORDS and head:
            break
        head.append(tokens[i].strip(","))
        i += 1
        # most targets are 1-2 word nouns ("wall lamp", "potted plant",
        # "computer monitor"); stop after 2 to avoid swallowing the relation
        if len(head) >= 2:
            break

    target = " ".join(head).lower() if head else (tokens[0].lower() if tokens else "")

    # remainder -> spatial relation + anchor (coarse)
    rest = " ".join(tokens[i:]).strip().rstrip("?").strip()
    # drop a leading copula/aux (+ optional that/which) so the relation regex sees
    # "below a window" not "are below a window" -- numerical Qs ("How many X are
    # <rel> Y?") otherwise lost the relation and fell back to a raw class count.
    rest = re.sub(r"^(that\s+|which\s+)?(are|is|was|were)\s+", "", rest,
                  flags=re.IGNORECASE).strip()
    relation, anchor = "", ""
    m = re.match(
        r"^(closest to|farthest from|furthest from|between|near|on|in|below|"
        r"above|under|next to|behind|beside|over|that is .*?(closest|farthest|"
        r"furthest).*?)\b\s*(.*)$",
        rest, re.IGNORECASE,
    )
    if m:
        relation = m.group(1).lower()
        anchor = (m.group(3) or "").strip()
    else:
        anchor = rest
    anchor = _DETERMINER.sub("", anchor).rstrip("?").strip()

    return Decomposition(
        target_object=target,
        attributes=attributes,
        spatial_relation=relation,
        anchor_object=anchor,
        region="",
        raw=question,
    )


# --------------------------------------------------------------------------- #
# VLM (authoritative) decomposition
# --------------------------------------------------------------------------- #
_DECOMP_PROMPT = """
You decompose a navigation/grounding instruction about an indoor scene into
structured fields. Return strictly valid JSON.

Keys:
  target_object         the main object to find/count (singular noun)
  attributes            list of adjectives describing the target (color/size/material); [] if none
  spatial_relation      the spatial relation phrase, e.g. "closest to", "between", "on"; "" if none
  anchor_object         the reference object(s) for the relation; "" if none
  region                room/area condition, e.g. "in the kitchen"; "" if none

Example:
  Instruction: "Find the red pillow on the sofa closest to the window."
  {"target_object":"pillow","attributes":["red"],"spatial_relation":"on",
   "anchor_object":"sofa closest to the window","region":""}
"""


def vlm_decompose(question: str, client) -> Decomposition:
    """Authoritative decomposition via the VLM. `client` is a VLMClient."""
    from pydantic import BaseModel

    class _Schema(BaseModel):
        target_object: str
        attributes: list[str]
        spatial_relation: str
        anchor_object: str
        region: str

    parsed = client.parse(
        system=_DECOMP_PROMPT,
        content=[client.text(f"Instruction: {question}")],
        response_format=_Schema,
        lite=True,
    )
    return Decomposition(
        target_object=parsed.target_object.lower(),
        attributes=[a.lower() for a in parsed.attributes],
        spatial_relation=parsed.spatial_relation.lower(),
        anchor_object=parsed.anchor_object.lower(),
        region=parsed.region.lower(),
        raw=question,
    )
