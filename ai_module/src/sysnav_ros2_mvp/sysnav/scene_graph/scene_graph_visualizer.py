"""Export the current scene graph as JSON, Graphviz DOT, and a PNG overview."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np


class SceneGraphVisualizer:
    def __init__(self, debug_dir: str) -> None:
        self.debug_dir = Path(debug_dir)
        self.json_path = self.debug_dir / "scene_graph_latest.json"
        self.dot_path = self.debug_dir / "scene_graph_latest.dot"
        self.png_path = self.debug_dir / "scene_graph_latest.png"

    def export(self, graph: dict) -> dict:
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_text(self.json_path, json.dumps(graph, ensure_ascii=False, indent=2))
        self._atomic_text(self.dot_path, self._to_dot(graph))
        self._write_png(graph)
        return {
            "json": str(self.json_path),
            "dot": str(self.dot_path),
            "png": str(self.png_path),
        }

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(content)
            temp_path = handle.name
        os.replace(temp_path, path)
        os.chmod(path, 0o644)

    @staticmethod
    def _node_name(node_type: str, numeric_id: int) -> str:
        return f"{node_type}_{numeric_id}"

    def _to_dot(self, graph: dict) -> str:
        lines = [
            "digraph scene_graph {",
            '  graph [rankdir=TB, bgcolor="white", splines=true, overlap=false];',
            '  node [fontname="Arial", style="rounded,filled"];',
            '  edge [fontname="Arial", fontsize=10];',
        ]

        room = graph["room"]
        room_name = self._node_name("room", room["room_id"])
        lines.append(
            f'  {room_name} [label="{room["name"]}", shape=box, fillcolor="#d8e6f3"];'
        )

        for viewpoint in graph["viewpoints"]:
            name = self._node_name("viewpoint", viewpoint["viewpoint_id"])
            label = f'Viewpoint {viewpoint["viewpoint_id"]}\\n({viewpoint["pose"]["x"]:.2f}, {viewpoint["pose"]["y"]:.2f})'
            lines.append(f'  {name} [label="{label}", shape=ellipse, fillcolor="#f2eadf"];')

        selected_id = graph.get("selected_object_id")
        for obj in graph["objects"]:
            name = self._node_name("object", obj["object_id"])
            label = f'{obj["category"]} #{obj["object_id"]}\\n({obj["position"][0]:.2f}, {obj["position"][1]:.2f})'
            fill = "#f5d6d6" if obj["object_id"] == selected_id else "#e9edf2"
            lines.append(f'  {name} [label="{label}", shape=box, fillcolor="{fill}"];')

        for edge in graph["edges"]:
            source = edge["source"]
            targets = edge.get("targets", [edge.get("target")])
            for target in targets:
                if target is None:
                    continue
                color = "#b23a48" if edge["edge_type"] == "object_object" else "#52616b"
                style = "bold" if edge["edge_type"] == "object_object" else "solid"
                lines.append(
                    f'  {source} -> {target} [label="{edge["relation"]}", color="{color}", style="{style}"];'
                )

        lines.append("}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _positions(graph: dict, width: int, height: int) -> dict[str, tuple[int, int]]:
        positions: dict[str, tuple[int, int]] = {}
        room = graph["room"]
        positions[f'room_{room["room_id"]}'] = (width // 2, 90)

        viewpoints = graph["viewpoints"]
        for index, viewpoint in enumerate(viewpoints):
            x = int((index + 1) * width / (len(viewpoints) + 1)) if viewpoints else width // 2
            positions[f'viewpoint_{viewpoint["viewpoint_id"]}'] = (x, 330)

        objects = graph["objects"]
        columns = max(1, min(6, len(objects)))
        for index, obj in enumerate(objects):
            row = index // columns
            col = index % columns
            x = int((col + 1) * width / (columns + 1))
            y = 650 + row * 170
            positions[f'object_{obj["object_id"]}'] = (x, min(height - 80, y))
        return positions

    @staticmethod
    def _arrow(canvas: np.ndarray, start: tuple[int, int], end: tuple[int, int], color, thickness: int = 2) -> None:
        cv2.arrowedLine(canvas, start, end, color, thickness, cv2.LINE_AA, tipLength=0.025)

    @staticmethod
    def _label(canvas: np.ndarray, text: str, point: tuple[int, int], color) -> None:
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x = max(3, point[0] - text_width // 2)
        y = max(text_height + 3, point[1])
        cv2.rectangle(canvas, (x - 3, y - text_height - 3), (x + text_width + 3, y + baseline + 2), (255, 255, 255), -1)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def _write_png(self, graph: dict) -> None:
        object_rows = max(1, (len(graph["objects"]) + 5) // 6)
        width = max(1400, 240 * max(1, len(graph["viewpoints"]), min(6, len(graph["objects"]))))
        height = max(900, 720 + object_rows * 170)
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        positions = self._positions(graph, width, height)

        edge_colors = {
            "room_viewpoint": (105, 91, 63),
            "room_object": (130, 130, 130),
            "viewpoint_object": (160, 150, 130),
            "object_object": (72, 58, 178),
        }
        for edge in graph["edges"]:
            source_position = positions.get(edge["source"])
            targets = edge.get("targets", [edge.get("target")])
            for target in targets:
                target_position = positions.get(target)
                if source_position is None or target_position is None:
                    continue
                color = edge_colors.get(edge["edge_type"], (100, 100, 100))
                thickness = 3 if edge["edge_type"] == "object_object" else 1
                self._arrow(canvas, source_position, target_position, color, thickness)
                midpoint = (
                    (source_position[0] + target_position[0]) // 2,
                    (source_position[1] + target_position[1]) // 2,
                )
                self._label(canvas, edge["relation"], midpoint, color)

        room = graph["room"]
        room_position = positions[f'room_{room["room_id"]}']
        cv2.rectangle(
            canvas,
            (room_position[0] - 120, room_position[1] - 45),
            (room_position[0] + 120, room_position[1] + 45),
            (110, 70, 25),
            2,
        )
        self._label(canvas, room["name"], (room_position[0], room_position[1] + 5), (55, 45, 30))

        for viewpoint in graph["viewpoints"]:
            position = positions[f'viewpoint_{viewpoint["viewpoint_id"]}']
            cv2.circle(canvas, position, 55, (95, 90, 70), 2, cv2.LINE_AA)
            self._label(canvas, f'VP {viewpoint["viewpoint_id"]}', (position[0], position[1] + 5), (60, 55, 40))
            self._label(
                canvas,
                f'C={viewpoint.get("coverage_voxel_count", 0)} / novel={viewpoint.get("novel_voxel_count", 0)}',
                (position[0], position[1] + 78),
                (80, 80, 80),
            )
            self._label(
                canvas,
                f'({viewpoint["pose"]["x"]:.1f}, {viewpoint["pose"]["y"]:.1f})',
                (position[0], position[1] + 102),
                (80, 80, 80),
            )

        selected_id = graph.get("selected_object_id")
        for obj in graph["objects"]:
            position = positions[f'object_{obj["object_id"]}']
            color = (50, 50, 190) if obj["object_id"] == selected_id else (90, 90, 90)
            cv2.rectangle(
                canvas,
                (position[0] - 95, position[1] - 42),
                (position[0] + 95, position[1] + 42),
                color,
                3 if obj["object_id"] == selected_id else 2,
            )
            self._label(canvas, f'{obj["category"]} #{obj["object_id"]}', (position[0], position[1]), color)
            self._label(
                canvas,
                f'({obj["position"][0]:.1f}, {obj["position"][1]:.1f}, {obj["position"][2]:.1f})',
                (position[0], position[1] + 66),
                (80, 80, 80),
            )

        cv2.putText(
            canvas,
            f'Updated: {graph["updated_at"]} | viewpoints={len(graph["viewpoints"])} '
            f'objects={len(graph["objects"])} edges={len(graph["edges"])} '
            f'accumulated_voxels={graph.get("coverage", {}).get("accumulated_voxel_count", 0)}',
            (25, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

        temporary_path = self.png_path.with_suffix(".tmp.png")
        if not cv2.imwrite(str(temporary_path), canvas):
            raise RuntimeError(f"Could not write scene graph image: {temporary_path}")
        os.replace(temporary_path, self.png_path)
        os.chmod(self.png_path, 0o644)
