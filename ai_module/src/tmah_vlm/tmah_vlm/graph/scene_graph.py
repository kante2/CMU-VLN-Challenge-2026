#!/usr/bin/env python3
"""
Online hierarchical open-vocabulary scene graph.

This is a challenge-sized adaptation of HOV-SG's graph idea. HOV-SG builds a
large offline graph from RGB-D sequences using SAM and CLIP. Here we reuse the
live tmah_vlm perception result as the object observation source and keep the
graph update cheap enough to run after each successful "find ..." query.
"""

import json
import math
import os
import re
from datetime import datetime

from tmah_vlm.graph.edges import RelationEdge, compute_relation_edges
from tmah_vlm.graph.nodes import FloorNode, ObjectNode, ObjectObservation, RoomNode


def normalize_label(text):
    """Normalize open-vocabulary labels for matching and merge decisions."""
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = [tok for tok in text.split() if tok not in {"the", "a", "an"}]
    return " ".join(tokens).strip()


def label_tokens(text):
    return set(normalize_label(text).split())


def euclidean_distance(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


class SceneGraph:
    """
    Building -> floor -> room -> object graph.

    The graph starts with one default floor and room because the current online
    pipeline does not yet segment rooms. Object nodes are merged when a new
    observation has a compatible label and nearby 3D center.
    """

    def __init__(self, merge_distance_m=0.75):
        self.merge_distance_m = float(merge_distance_m)
        self.building_id = "building"
        self.floors = {}
        self.rooms = {}
        self.objects = {}
        self.edges = []
        self._next_object_index = 0
        self.ensure_default_hierarchy()

    def ensure_default_hierarchy(self):
        if "floor_0" not in self.floors:
            self.floors["floor_0"] = FloorNode("floor_0", "floor_0", [])
        if "room_0_0" not in self.rooms:
            self.rooms["room_0_0"] = RoomNode("room_0_0", "floor_0", "unknown_room", [])
        if "room_0_0" not in self.floors["floor_0"].room_ids:
            self.floors["floor_0"].room_ids.append("room_0_0")

    def add_observation(self, observation, floor_id="floor_0", room_id="room_0_0"):
        """
        Insert or update an object observation.

        Returns the ObjectNode that owns the observation.
        """
        self.ensure_default_hierarchy()
        if floor_id not in self.floors:
            self.floors[floor_id] = FloorNode(floor_id, floor_id, [])
        if room_id not in self.rooms:
            self.rooms[room_id] = RoomNode(room_id, floor_id, "unknown_room", [])
        if room_id not in self.floors[floor_id].room_ids:
            self.floors[floor_id].room_ids.append(room_id)

        object_node = self.find_merge_target(observation, room_id)
        if object_node is None:
            object_id = self.allocate_object_id(floor_id, room_id)
            object_node = ObjectNode(
                object_id=object_id,
                room_id=room_id,
                name=observation.label,
                center=observation.bbox_center,
                size=observation.bbox_size,
            )
            self.objects[object_id] = object_node
            self.rooms[room_id].object_ids.append(object_id)

        object_node.add_observation(observation)
        self.recompute_edges()
        self.update_captions()
        return object_node

    def recompute_edges(self):
        self.edges = compute_relation_edges(self.objects)
        return self.edges

    def update_captions(self):
        """
        Persist SORT3D-lite captions on graph object nodes.

        The captioner is rule-based and uses only object labels/geometry/nearby
        context, so it is safe for hidden evaluation and does not require
        object_list.txt or an external VLM.
        """
        try:
            from tmah_vlm.sort3d.captioner import attach_rule_captions
            from tmah_vlm.sort3d.objects import Sort3DObject
        except Exception:
            return

        sort_objects = [
            Sort3DObject(
                object_id=object_id,
                name=node.name,
                center=node.center,
                size=node.size,
                caption=node.caption,
                source="scene_graph",
            )
            for object_id, node in self.objects.items()
        ]
        attach_rule_captions(sort_objects)
        captions = {obj.object_id: obj.caption for obj in sort_objects}
        for object_id, caption in captions.items():
            if object_id in self.objects:
                node = self.objects[object_id]
                if node.caption_source != "vlm":
                    node.caption = caption
                    node.caption_source = "rule"

    def find_merge_target(self, observation, room_id):
        obs_tokens = label_tokens(observation.label)
        best_node = None
        best_distance = None

        for object_id in self.rooms.get(room_id, RoomNode(room_id, "", "")).object_ids:
            node = self.objects.get(object_id)
            if node is None:
                continue

            node_tokens = label_tokens(node.name)
            if obs_tokens and node_tokens and obs_tokens.isdisjoint(node_tokens):
                continue

            distance = euclidean_distance(node.center, observation.bbox_center)
            if distance > self.merge_distance_m:
                continue

            if best_distance is None or distance < best_distance:
                best_node = node
                best_distance = distance

        return best_node

    def allocate_object_id(self, floor_id, room_id):
        while True:
            object_id = f"{floor_id}_{room_id}_{self._next_object_index}"
            self._next_object_index += 1
            if object_id not in self.objects:
                return object_id

    def query_objects(self, text, top_k=5):
        """
        Lexical open-vocabulary query over object labels/questions.

        This is intentionally lightweight. Later, CLIP text embeddings can fill
        ObjectNode.embedding and replace this scorer without changing callers.
        """
        query = normalize_label(text)
        query_tokens = label_tokens(query)
        scored = []

        for node in self.objects.values():
            names = [node.name]
            names.extend(obs.question for obs in node.observations[-3:])
            node_tokens = set()
            for name in names:
                node_tokens.update(label_tokens(name))

            if not node_tokens:
                score = 0.0
            elif query_tokens:
                score = len(query_tokens & node_tokens) / float(len(query_tokens | node_tokens))
            else:
                score = 0.0

            scored.append((score, node))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    def to_dict(self):
        return {
            "schema": "tmah_hovsg_light_v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "building_id": self.building_id,
            "merge_distance_m": self.merge_distance_m,
            "floors": {key: node.to_dict() for key, node in self.floors.items()},
            "rooms": {key: node.to_dict() for key, node in self.rooms.items()},
            "objects": {key: node.to_dict() for key, node in self.objects.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "next_object_index": self._next_object_index,
        }

    @classmethod
    def from_dict(cls, data):
        graph = cls(merge_distance_m=float(data.get("merge_distance_m", 0.75)))
        graph.building_id = str(data.get("building_id", "building"))
        graph.floors = {
            key: FloorNode.from_dict(value)
            for key, value in data.get("floors", {}).items()
        }
        graph.rooms = {
            key: RoomNode.from_dict(value)
            for key, value in data.get("rooms", {}).items()
        }
        graph.objects = {
            key: ObjectNode.from_dict(value)
            for key, value in data.get("objects", {}).items()
        }
        graph.edges = [
            RelationEdge.from_dict(item)
            for item in data.get("edges", [])
        ]
        if not graph.edges:
            graph.recompute_edges()
        graph.update_captions()
        graph._next_object_index = int(data.get("next_object_index", len(graph.objects)))
        graph.ensure_default_hierarchy()
        return graph

    def save_json(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save_hovsg_layout(self, root_dir):
        """
        Save compact JSON files in a HOV-SG-like folder layout.

        HOV-SG stores each node as .json plus .ply. We do not own full object
        point clouds in this lightweight online path, so JSON is saved now and
        .ply export can be added when point clusters are retained.
        """
        graph_dir = os.path.join(root_dir, "graph")
        for folder in ("floors", "rooms", "objects", "edges"):
            os.makedirs(os.path.join(graph_dir, folder), exist_ok=True)

        for node in self.floors.values():
            self._write_node_json(os.path.join(graph_dir, "floors", node.floor_id + ".json"), node)
        for node in self.rooms.values():
            self._write_node_json(os.path.join(graph_dir, "rooms", node.room_id + ".json"), node)
        for node in self.objects.values():
            self._write_node_json(os.path.join(graph_dir, "objects", node.object_id + ".json"), node)
        self._write_json(
            os.path.join(graph_dir, "edges", "relations.json"),
            [edge.to_dict() for edge in self.edges],
        )

        return graph_dir

    @staticmethod
    def _write_node_json(path, node):
        SceneGraph._write_json(path, node.to_dict())

    @staticmethod
    def _write_json(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
