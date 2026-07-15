#!/usr/bin/env python3
"""Load Unity/object_list.txt ground-truth semantics into Sort3DObject."""

import re

from tmah_vlm.reasoning.sort3d.data.objects import Sort3DObject


_OBJECT_LINE = re.compile(
    r'^\s*(?P<id>\S+)\s+'
    r'(?P<x>[-+0-9.eE]+)\s+(?P<y>[-+0-9.eE]+)\s+(?P<z>[-+0-9.eE]+)\s+'
    r'(?P<sx>[-+0-9.eE]+)\s+(?P<sy>[-+0-9.eE]+)\s+(?P<sz>[-+0-9.eE]+)\s+'
    r'(?P<heading>[-+0-9.eE]+)\s+"(?P<label>.*)"\s*$'
)


def parse_object_list_line(line):
    match = _OBJECT_LINE.match(line)
    if match is None:
        return None

    data = match.groupdict()
    return Sort3DObject(
        object_id=data["id"],
        name=data["label"],
        center=(data["x"], data["y"], data["z"]),
        size=(data["sx"], data["sy"], data["sz"]),
        heading=float(data["heading"]),
        source="object_list",
    )


def load_object_list(path):
    objects = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = parse_object_list_line(line)
            if obj is not None:
                objects.append(obj)
    return objects
