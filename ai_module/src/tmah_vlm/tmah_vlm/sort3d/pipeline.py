#!/usr/bin/env python3
"""High-level SORT3D-lite pipeline for tmah_vlm."""

import json
import re

from tmah_vlm.sort3d.actions import go_between, go_near
from tmah_vlm.sort3d.captioner import attach_rule_captions
from tmah_vlm.sort3d.filters import filter_relevant_objects
from tmah_vlm.sort3d.object_list import load_object_list
from tmah_vlm.sort3d.objects import Sort3DObject, normalize_text
from tmah_vlm.sort3d.toolbox import SpatialToolbox


class Sort3DLite:
    """
    Object-centric map + caption + spatial toolbox.

    This does not call an LLM. It gives us the same data shape and deterministic
    spatial functions that an LLM planner can call later.
    """

    def __init__(self, objects=None):
        self.objects = attach_rule_captions(list(objects or []))
        self.toolbox = SpatialToolbox(self.objects)

    @classmethod
    def from_object_list(cls, path):
        return cls(load_object_list(path))

    @classmethod
    def from_scene_graph_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_scene_graph_dict(data)

    @classmethod
    def from_scene_graph(cls, scene_graph):
        """Build from the live tmah_vlm.graph.scene_graph.SceneGraph object."""
        if scene_graph is None:
            return cls([])
        if hasattr(scene_graph, "to_dict"):
            return cls.from_scene_graph_dict(scene_graph.to_dict())
        if isinstance(scene_graph, dict):
            return cls.from_scene_graph_dict(scene_graph)
        return cls([])

    @classmethod
    def from_scene_graph_dict(cls, data):
        objects = [
            Sort3DObject.from_graph_node(object_id, item)
            for object_id, item in data.get("objects", {}).items()
        ]
        return cls(objects)

    def to_object_dict(self):
        return {obj.object_id: obj.to_sort3d_dict() for obj in self.objects}

    def describe_objects(self):
        return [obj.to_dict() for obj in self.objects]

    def relevant_objects(self, instruction):
        return filter_relevant_objects(self.objects, instruction)

    def select_target(self, instruction):
        """
        Deterministic first-pass selector for common challenge phrases.

        Returns a dict with candidate ids and the tool that was used. This is a
        fallback and a scaffold for plugging in an LLM tool-calling planner.
        """
        text = normalize_text(instruction)
        relevant, terms = self.relevant_objects(text)

        relation_args = self._parse_relation_args(text, terms)

        if "between" in text and len(terms) >= 3:
            target, anchor1, anchor2 = relation_args if len(relation_args) >= 3 else terms[:3]
            ids = self.toolbox.find_between(target, anchor1, anchor2)
            return self._result("find_between", ids, terms, relevant)

        closest_match = re.search(r"(closest|nearest)(?: .*?)? to (?:the )?([a-z0-9 ]+)", text)
        furthest_match = re.search(r"(furthest|farthest)(?: .*?)? from (?:the )?([a-z0-9 ]+)", text)
        target = relation_args[0] if relation_args else (terms[0] if terms else "")

        if closest_match and target:
            anchor = self._best_anchor_term(closest_match.group(2), terms)
            ids = self.toolbox.closest_to(target, anchor)
            return self._result("closest_to", ids[:1], terms, relevant)

        if furthest_match and target:
            anchor = self._best_anchor_term(furthest_match.group(2), terms)
            ids = self.toolbox.furthest_from(target, anchor)
            return self._result("furthest_from", ids[:1], terms, relevant)

        for relation, method in [
            ("near", self.toolbox.find_near),
            ("on", self.toolbox.find_above),
            ("above", self.toolbox.find_above),
            ("below", self.toolbox.find_below),
            ("left of", self.toolbox.find_left),
            ("right of", self.toolbox.find_right),
            ("behind", self.toolbox.find_behind),
        ]:
            if relation in text and len(terms) >= 2:
                target, anchor = relation_args[:2] if len(relation_args) >= 2 else terms[:2]
                ids = method(target, anchor)
                return self._result(method.__name__, ids, terms, relevant)

        ids = [obj.object_id for obj in relevant if terms and terms[0] in obj.text_blob()]
        if not ids and relevant:
            ids = [relevant[0].object_id]
        return self._result("lexical_filter", ids[:1], terms, relevant)

    def action_for_selection(self, selection, robot_pose=None):
        ids = list(selection.get("candidate_ids", []))
        if not ids:
            return None
        if selection.get("tool") == "find_between" and len(ids) >= 2:
            return go_between(self.toolbox.get(ids[0]), self.toolbox.get(ids[1]), robot_pose)
        obj = self.toolbox.get(ids[0])
        if obj is None:
            return None
        return go_near(obj, robot_pose)

    def _result(self, tool, ids, terms, relevant):
        return {
            "tool": tool,
            "candidate_ids": list(ids),
            "terms": list(terms),
            "relevant_ids": [obj.object_id for obj in relevant],
        }

    def _best_anchor_term(self, text, terms):
        text = normalize_text(text)
        for term in terms:
            if term in text:
                return term
        return terms[1] if len(terms) > 1 else text

    def _parse_relation_args(self, text, terms):
        """Infer target/anchor phrase order from common referring expressions."""
        relation_words = [
            "between", "near", "on", "above", "below", "left of", "right of",
            "behind", "closest", "nearest", "furthest", "farthest",
        ]
        matches = []
        for term in terms:
            start = text.find(term)
            if start >= 0:
                matches.append((start, term))
        matches.sort(key=lambda item: item[0])
        if not matches:
            return []

        relation_positions = [
            text.find(word) for word in relation_words if text.find(word) >= 0
        ]
        if not relation_positions:
            return [term for _, term in matches]

        relation_pos = min(relation_positions)
        before = [term for start, term in matches if start < relation_pos]
        after = [term for start, term in matches if start > relation_pos]

        ordered = []
        if before:
            ordered.append(before[-1])
        ordered.extend(after)
        for _, term in matches:
            if term not in ordered:
                ordered.append(term)
        return ordered
