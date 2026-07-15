#!/usr/bin/env python3
"""
Render scene_graph_latest.json to a simple top-down JPG.

This renderer intentionally depends only on Pillow so it can run in the current
tmah_vlm container without pulling in matplotlib or HOV-SG visualization deps.
"""

import argparse
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont

from tmah_vlm.reasoning.graph.scene_graph import SceneGraph


PALETTE = [
    (64, 190, 255),
    (255, 180, 64),
    (120, 220, 120),
    (220, 120, 255),
    (255, 100, 120),
    (120, 220, 220),
]


def load_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def object_bounds(objects):
    xs = []
    ys = []
    for obj in objects:
        cx, cy, _ = obj["center"]
        sx, sy, _ = obj["size"]
        xs.extend([cx - sx / 2.0, cx + sx / 2.0])
        ys.extend([cy - sy / 2.0, cy + sy / 2.0])

    if not xs:
        return (-1.0, 1.0, -1.0, 1.0)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad = max(1.0, 0.15 * max(max_x - min_x, max_y - min_y, 1.0))
    return min_x - pad, max_x + pad, min_y - pad, max_y + pad


class TopDownProjector:
    def __init__(self, bounds, width, height, margin):
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.width = width
        self.height = height
        self.margin = margin
        span_x = max(self.max_x - self.min_x, 1e-6)
        span_y = max(self.max_y - self.min_y, 1e-6)
        self.scale = min(
            (width - margin * 2) / span_x,
            (height - margin * 2) / span_y,
        )

    def xy(self, x, y):
        px = self.margin + (float(x) - self.min_x) * self.scale
        py = self.height - self.margin - (float(y) - self.min_y) * self.scale
        return px, py


def draw_grid(draw, projector, font):
    min_x = math.floor(projector.min_x)
    max_x = math.ceil(projector.max_x)
    min_y = math.floor(projector.min_y)
    max_y = math.ceil(projector.max_y)

    for x in range(min_x, max_x + 1):
        p0 = projector.xy(x, projector.min_y)
        p1 = projector.xy(x, projector.max_y)
        color = (55, 60, 68) if x != 0 else (110, 120, 135)
        draw.line([p0, p1], fill=color, width=2 if x == 0 else 1)
        draw.text((p0[0] + 3, projector.height - projector.margin + 6), str(x), fill=(160, 165, 175), font=font)

    for y in range(min_y, max_y + 1):
        p0 = projector.xy(projector.min_x, y)
        p1 = projector.xy(projector.max_x, y)
        color = (55, 60, 68) if y != 0 else (110, 120, 135)
        draw.line([p0, p1], fill=color, width=2 if y == 0 else 1)
        draw.text((projector.margin - 34, p0[1] - 8), str(y), fill=(160, 165, 175), font=font)


def draw_object(draw, projector, obj, index, font, small_font):
    color = PALETTE[index % len(PALETTE)]
    cx, cy, cz = obj["center"]
    sx, sy, sz = obj["size"]

    x0, y0 = projector.xy(cx - sx / 2.0, cy - sy / 2.0)
    x1, y1 = projector.xy(cx + sx / 2.0, cy + sy / 2.0)
    left, right = sorted([x0, x1])
    top, bottom = sorted([y0, y1])

    draw.rectangle([left, top, right, bottom], outline=color, width=4)
    center_px = projector.xy(cx, cy)
    r = 6
    draw.ellipse(
        [center_px[0] - r, center_px[1] - r, center_px[0] + r, center_px[1] + r],
        fill=(255, 220, 60),
        outline=(20, 20, 20),
        width=2,
    )

    observations = obj.get("observations", [])
    score = observations[-1].get("score", 0.0) if observations else 0.0
    label = f"{obj['name']} ({len(observations)})"
    detail = f"{obj['object_id']}  z={cz:.2f}m  score={score:.2f}"

    label_x = right + 10
    label_y = top - 4
    bbox = draw.textbbox((label_x, label_y), label, font=font)
    detail_bbox = draw.textbbox((label_x, label_y + 22), detail, font=small_font)
    bg = [
        min(bbox[0], detail_bbox[0]) - 5,
        min(bbox[1], detail_bbox[1]) - 3,
        max(bbox[2], detail_bbox[2]) + 5,
        max(bbox[3], detail_bbox[3]) + 3,
    ]
    draw.rectangle(bg, fill=(18, 22, 28), outline=color, width=1)
    draw.text((label_x, label_y), label, fill=(245, 248, 252), font=font)
    draw.text((label_x, label_y + 22), detail, fill=(180, 190, 200), font=small_font)


def draw_edge(draw, projector, graph, edge, font):
    objects = graph.get("objects", {})
    source = objects.get(edge.get("source"))
    target = objects.get(edge.get("target"))
    if source is None or target is None:
        return

    sxy = projector.xy(source["center"][0], source["center"][1])
    txy = projector.xy(target["center"][0], target["center"][1])

    relation = edge.get("relation", "")
    if relation == "near":
        color = (120, 120, 150)
        width = 1
    else:
        color = (180, 120, 255)
        width = 2

    draw.line([sxy, txy], fill=color, width=width)

    # Label only the more informative directional/vertical relations. "near"
    # edges become too noisy once many objects are present.
    if relation != "near":
        mx = (sxy[0] + txy[0]) / 2.0
        my = (sxy[1] + txy[1]) / 2.0
        draw.text((mx + 4, my + 4), relation, fill=color, font=font)


def render_scene_graph(input_path, output_path, width=1400, height=900):
    with open(input_path, "r", encoding="utf-8") as f:
        graph = json.load(f)
    if "edges" not in graph:
        graph = SceneGraph.from_dict(graph).to_dict()

    objects = list(graph.get("objects", {}).values())
    bounds = object_bounds(objects)
    projector = TopDownProjector(bounds, width, height, margin=90)

    image = Image.new("RGB", (width, height), (28, 32, 38))
    draw = ImageDraw.Draw(image)
    title_font = load_font(26)
    font = load_font(18)
    small_font = load_font(13)

    draw_grid(draw, projector, small_font)

    for edge in graph.get("edges", []):
        draw_edge(draw, projector, graph, edge, small_font)

    for index, obj in enumerate(objects):
        draw_object(draw, projector, obj, index, font, small_font)

    title = "TMAH Online HOV-SG Scene Graph - top-down XY"
    subtitle = (
        f"objects={len(objects)}  edges={len(graph.get('edges', []))}  schema={graph.get('schema', '')}  "
        f"created={graph.get('created_at', '')}"
    )
    draw.rectangle([0, 0, width, 64], fill=(12, 15, 20))
    draw.text((24, 12), title, fill=(245, 248, 252), font=title_font)
    draw.text((24, 42), subtitle, fill=(170, 180, 192), font=small_font)
    draw.text((width - 215, height - 36), "x/y units: meters", fill=(170, 180, 192), font=small_font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path, quality=92)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to scene_graph_latest.json")
    parser.add_argument("output", help="Output jpg path")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    path = render_scene_graph(args.input, args.output, args.width, args.height)
    print(path)


if __name__ == "__main__":
    main()
