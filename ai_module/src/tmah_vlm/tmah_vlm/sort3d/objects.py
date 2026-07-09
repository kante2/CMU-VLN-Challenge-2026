#!/usr/bin/env python3
"""Shared object representation for the local SORT3D adaptation."""

from dataclasses import dataclass, field
import math


def as_float_tuple(values, length, default=0.0):
    values = list(values or [])
    out = []
    for index in range(length):
        try:
            out.append(float(values[index]))
        except Exception:
            out.append(float(default))
    return tuple(out)


def normalize_text(text):
    return " ".join(str(text or "").lower().replace("_", " ").split())


@dataclass
class Sort3DObject:
    """Object instance passed to captioning/filtering/spatial tools."""

    object_id: str
    name: str
    center: tuple
    size: tuple
    heading: float = 0.0
    caption: str = ""
    source: str = "unknown"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.object_id = str(self.object_id)
        self.name = normalize_text(self.name)
        self.center = as_float_tuple(self.center, 3)
        self.size = as_float_tuple(self.size, 3, default=0.4)
        self.heading = float(self.heading or 0.0)
        self.caption = str(self.caption or "")

    @property
    def x(self):
        return self.center[0]

    @property
    def y(self):
        return self.center[1]

    @property
    def z(self):
        return self.center[2]

    @property
    def volume(self):
        return max(self.size[0], 0.0) * max(self.size[1], 0.0) * max(self.size[2], 0.0)

    @property
    def radius_xy(self):
        return 0.5 * math.hypot(self.size[0], self.size[1])

    @property
    def z_min(self):
        return self.z - self.size[2] * 0.5

    @property
    def z_max(self):
        return self.z + self.size[2] * 0.5

    def distance_xy(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def surface_distance_xy(self, other):
        return max(0.0, self.distance_xy(other) - self.radius_xy - other.radius_xy)

    def text_blob(self):
        return normalize_text(f"{self.name} {self.caption}")

    def to_dict(self):
        return {
            "id": self.object_id,
            "name": self.name,
            "caption": self.caption,
            "cx": self.center[0],
            "cy": self.center[1],
            "cz": self.center[2],
            "center": list(self.center),
            "size": list(self.size),
            "heading": self.heading,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    def to_sort3d_dict(self):
        """Shape compatible with SORT3D's language-planner object_dict."""
        return {
            "name": self.name,
            "caption": self.caption,
            "centroid": list(self.center),
            "dimensions": list(self.size),
            "heading": self.heading,
        }

    @classmethod
    def from_graph_node(cls, object_id, data):
        return cls(
            object_id=object_id,
            name=data.get("name", ""),
            center=data.get("center", (0.0, 0.0, 0.0)),
            size=data.get("size", (0.4, 0.4, 0.4)),
            caption=data.get("caption", ""),
            source="scene_graph",
            metadata={"room_id": data.get("room_id", "")},
        )
