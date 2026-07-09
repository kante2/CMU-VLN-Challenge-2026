#!/usr/bin/env python3
"""
디버그 이미지 저장 유틸.

원본 카메라 이미지 위에 GroundingDINO 후보 박스를 그리고,
VLM이 선택한 후보는 SELECTED로 표시한다.
"""

import os
from datetime import datetime

from PIL import Image as PILImage, ImageDraw, ImageFont

from tmah_vlm import config


def load_font(size):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            size,
        )
    except Exception:
        return ImageFont.load_default()


def safe_filename(text, max_len=40):
    return "".join(c if c.isalnum() else "_" for c in text)[:max_len]


def draw_detections(image, detections, prompt="", selected_index=-1):
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    font = load_font(16)

    palette = [
        (255, 80, 80),
        (80, 200, 120),
        (80, 140, 255),
        (255, 190, 60),
        (200, 100, 255),
        (60, 220, 220),
    ]

    for index, det in enumerate(detections):
        x1, y1, x2, y2 = det.box
        color = palette[index % len(palette)]
        width = 5 if index == selected_index else 3

        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        caption = f"#{index} {det.label} {det.score:.2f}"
        if index == selected_index:
            caption = "SELECTED " + caption

        tx = x1
        ty = max(24, y1 - 22)

        try:
            bbox = draw.textbbox((tx, ty), caption, font=font)
            draw.rectangle(bbox, fill=color)
        except Exception:
            pass

        draw.text((tx, ty), caption, fill=(0, 0, 0), font=font)

    header = f"prompt='{prompt}'  detections={len(detections)}  selected={selected_index}"
    draw.rectangle([0, 0, img.width, 24], fill=(0, 0, 0))
    draw.text((5, 4), header, fill=(255, 255, 255), font=font)

    return img


def save_detection_image(image, detections, prompt, selected_index=-1, out_dir=None):
    if out_dir is None:
        out_dir = config.DEBUG_DIR

    os.makedirs(out_dir, exist_ok=True)
    annotated = draw_detections(image, detections, prompt, selected_index)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"det_{timestamp}_{safe_filename(prompt)}.jpg"
    path = os.path.join(out_dir, filename)
    annotated.save(path, quality=90)
    return path


def save_3d_result_text(question, selected_index, result, waypoint, out_dir=None):
    if out_dir is None:
        out_dir = config.DEBUG_DIR

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"target3d_{timestamp}.txt")

    lines = []
    lines.append(f"question: {question}")
    lines.append(f"selected_index: {selected_index}")
    lines.append(f"target_map_xyz: {result['point']}")
    lines.append(f"camera_origin_map_xyz: {result['origin']}")
    lines.append(f"ray_map_xyz: {result['ray']}")
    lines.append(f"method: {result['method']}")
    lines.append(f"matched_points: {result['n_matched']}")
    lines.append(f"target_name: {result.get('target_name', '')}")
    lines.append(f"cluster_policy: {result.get('cluster_policy', '')}")
    lines.append(f"cluster_depth_m: {result.get('cluster_depth_m', '')}")
    lines.append(f"cluster_error: {result.get('cluster_error', '')}")
    lines.append(f"cluster_count: {result.get('cluster_count', '')}")
    lines.append(f"bbox_center: {result.get('bbox_center', '')}")
    lines.append(f"bbox_size: {result.get('bbox_size', '')}")
    lines.append(f"waypoint: {waypoint}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return path
