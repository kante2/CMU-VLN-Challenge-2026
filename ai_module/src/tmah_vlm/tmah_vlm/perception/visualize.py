#!/usr/bin/env python3
"""
검출 결과를 이미지에 그려서 파일로 저장하는 시각화 유틸.

GroundingDINO 가 뱉은 박스들을 원본 이미지 위에 그려 저장합니다.
Phase 1a 목적: "지금 카메라 뷰에서 뭐가 검출되는지" 눈으로 확인.
"""

import os
from datetime import datetime
from typing import List

from PIL import Image as PILImage, ImageDraw, ImageFont

# detector.Detection 과 동일 필드를 가진 객체를 받는다고 가정
# (label, score, box=(x1,y1,x2,y2))


def draw_detections(image: PILImage.Image,
                    detections: List,
                    prompt: str = "") -> PILImage.Image:
    """원본 PIL 이미지에 검출 박스를 그린 새 이미지를 반환."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    # 기본 폰트 (컨테이너에 특정 폰트 없을 수 있어 fallback)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    palette = [
        (255, 80, 80), (80, 200, 120), (80, 140, 255),
        (255, 190, 60), (200, 100, 255), (60, 220, 220),
    ]

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.box
        color = palette[i % len(palette)]
        # 박스
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # 라벨 텍스트
        caption = f"{det.label} {det.score:.2f}"
        tx, ty = x1, max(0, y1 - 20)
        # 텍스트 배경
        try:
            bbox = draw.textbbox((tx, ty), caption, font=font)
            draw.rectangle(bbox, fill=color)
        except Exception:
            pass
        draw.text((tx, ty), caption, fill=(0, 0, 0), font=font)

    # 상단에 프롬프트/개수 표시
    header = f"prompt='{prompt}'  detections={len(detections)}"
    draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    draw.text((5, 3), header, fill=(255, 255, 255), font=font)

    return img


def save_detection_image(image: PILImage.Image,
                         detections: List,
                         prompt: str,
                         out_dir: str = "/home/docker/ai_module/debug") -> str:
    """박스 그린 이미지를 타임스탬프 파일명으로 저장하고 경로 반환."""
    os.makedirs(out_dir, exist_ok=True)
    annotated = draw_detections(image, detections, prompt)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
    filename = f"det_{ts}_{safe_prompt}.jpg"
    path = os.path.join(out_dir, filename)
    annotated.save(path, quality=90)
    return path
