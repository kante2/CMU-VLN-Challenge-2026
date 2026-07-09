#!/usr/bin/env python3
"""
HOV-SG inspired graph node data structures.

The original HOV-SG stores floors, rooms, and objects as point clouds plus JSON
metadata. For the challenge node we keep the same hierarchy but store compact
runtime metadata that can be updated online after each "find ..." query.
"""

from dataclasses import dataclass, field


@dataclass
class ObjectObservation:
    """One grounded 2D detection and its estimated 3D result."""

    label: str
    score: float
    question: str
    box_2d: tuple
    point: tuple
    bbox_center: tuple
    bbox_size: tuple
    method: str
    matched_points: int
    stamp_sec: float

    def to_dict(self):
        return {
            "label": self.label,
            "score": self.score,
            "question": self.question,
            "box_2d": list(self.box_2d),
            "point": list(self.point),
            "bbox_center": list(self.bbox_center),
            "bbox_size": list(self.bbox_size),
            "method": self.method,
            "matched_points": self.matched_points,
            "stamp_sec": self.stamp_sec,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            label=str(data.get("label", "")),
            score=float(data.get("score", 0.0)),
            question=str(data.get("question", "")),
            box_2d=tuple(data.get("box_2d", (0.0, 0.0, 0.0, 0.0))),
            point=tuple(data.get("point", (0.0, 0.0, 0.0))),
            bbox_center=tuple(data.get("bbox_center", data.get("point", (0.0, 0.0, 0.0)))),
            bbox_size=tuple(data.get("bbox_size", (0.4, 0.4, 0.4))),
            method=str(data.get("method", "")),
            matched_points=int(data.get("matched_points", 0)),
            stamp_sec=float(data.get("stamp_sec", 0.0)),
        )


@dataclass
class ObjectNode:
    """Object node analogous to HOV-SG's object-level scene graph node."""

    object_id: str
    room_id: str
    name: str
    center: tuple
    size: tuple
    caption: str = ""
    caption_source: str = "rule"
    embedding: list = field(default_factory=list)
    observations: list = field(default_factory=list)

    def add_observation(self, observation):
        self.observations.append(observation)
        self.name = observation.label or self.name
        self.center = observation.bbox_center
        self.size = observation.bbox_size

    @property
    def last_observation(self):
        if not self.observations:
            return None
        return self.observations[-1]

    def to_dict(self):
        return {
            "object_id": self.object_id,
            "room_id": self.room_id,
            "name": self.name,
            "caption": self.caption,
            "caption_source": self.caption_source,
            "center": list(self.center),
            "size": list(self.size),
            "embedding": list(self.embedding),
            "observations": [obs.to_dict() for obs in self.observations],
        }

    @classmethod
    def from_dict(cls, data):
        node = cls(
            object_id=str(data.get("object_id", "")),
            room_id=str(data.get("room_id", "")),
            name=str(data.get("name", "")),
            center=tuple(data.get("center", (0.0, 0.0, 0.0))),
            size=tuple(data.get("size", (0.4, 0.4, 0.4))),
            caption=str(data.get("caption", "")),
            caption_source=str(data.get("caption_source", "rule")),
            embedding=list(data.get("embedding", [])),
        )
        node.observations = [
            ObjectObservation.from_dict(item)
            for item in data.get("observations", [])
        ]
        return node


@dataclass
class RoomNode:
    """Room node. Initially all online detections go into room_0."""

    room_id: str
    floor_id: str
    name: str = "unknown_room"
    object_ids: list = field(default_factory=list)

    def to_dict(self):
        return {
            "room_id": self.room_id,
            "floor_id": self.floor_id,
            "name": self.name,
            "object_ids": list(self.object_ids),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            room_id=str(data.get("room_id", "")),
            floor_id=str(data.get("floor_id", "")),
            name=str(data.get("name", "unknown_room")),
            object_ids=list(data.get("object_ids", [])),
        )


@dataclass
class FloorNode:
    """Floor node. Initially all online detections go into floor_0."""

    floor_id: str
    name: str = "floor_0"
    room_ids: list = field(default_factory=list)

    def to_dict(self):
        return {
            "floor_id": self.floor_id,
            "name": self.name,
            "room_ids": list(self.room_ids),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            floor_id=str(data.get("floor_id", "")),
            name=str(data.get("name", "floor_0")),
            room_ids=list(data.get("room_ids", [])),
        )
