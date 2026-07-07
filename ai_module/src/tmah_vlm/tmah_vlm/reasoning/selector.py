#!/usr/bin/env python3
"""
Qwen2.5-VL 기반 선택기 (selector).

GroundingDINO 가 찾은 후보 박스들 중에서, 원본 질문(색/공간관계 포함)에
가장 맞는 것 하나를 VLM 이 고른다. 규칙(색 평균, 거리 계산)을 손코딩하지 않고
VLM 의 이해력에 맡긴다.

부담 절감 전략:
  - 전체 파노라마 대신, 후보 박스를 crop 해서 번호 라벨을 붙인
    "몽타주(montage)" 한 장을 만들어 VLM 에 보여준다.
  - "몇 번이 정답?" 을 물어 번호만 받는다. -> 작은 이미지, 짧은 출력 -> 8GB 여유.

주 진입점:
  sel = QwenSelector()                       # 최초 1회 로드 (무거움)
  idx = sel.choose(full_image, detections, question)
  # idx: detections 리스트에서 정답의 인덱스 (없으면 -1)
"""

import re
from typing import List, Optional

import torch
from PIL import Image as PILImage, ImageDraw, ImageFont


class QwenSelector:
    def __init__(self,
                 model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 device: str = None,
                 max_new_tokens: int = 64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        from transformers import (Qwen2_5_VLForConditionalGeneration,
                                  AutoProcessor)
        # 메모리 절약: 시각 토큰 상한을 낮게 (crop 몽타주면 충분)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=768 * 28 * 28,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=self.device,
        )
        self.model.eval()

    # ---------- 후보 crop 몽타주 만들기 ----------
    @staticmethod
    def _make_montage(image: PILImage.Image, detections: List,
                      pad: int = 8, cell_w: int = 220) -> PILImage.Image:
        """각 후보를 crop 해서 번호 붙여 가로로 이어붙인 한 장."""
        crops = []
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(v) for v in det.box]
            # 약간 여유 두고 crop
            x1 = max(0, x1 - 10); y1 = max(0, y1 - 10)
            x2 = min(image.width, x2 + 10); y2 = min(image.height, y2 + 10)
            crop = image.crop((x1, y1, x2, y2))
            # 셀 폭 통일 (비율 유지)
            if crop.width > 0:
                ratio = cell_w / crop.width
                crop = crop.resize((cell_w, max(1, int(crop.height * ratio))))
            # 번호 라벨
            labeled = PILImage.new("RGB", (crop.width, crop.height + 28),
                                   (30, 30, 30))
            labeled.paste(crop, (0, 28))
            d = ImageDraw.Draw(labeled)
            d.text((5, 3), f"#{i}", fill=(255, 255, 0), font=font)
            crops.append(labeled)

        if not crops:
            return image

        total_w = sum(c.width for c in crops) + pad * (len(crops) + 1)
        max_h = max(c.height for c in crops) + pad * 2
        montage = PILImage.new("RGB", (total_w, max_h), (0, 0, 0))
        x = pad
        for c in crops:
            montage.paste(c, (x, pad))
            x += c.width + pad
        return montage

    # ---------- 선택 ----------
    @torch.no_grad()
    def choose(self, image: PILImage.Image, detections: List,
               question: str) -> int:
        n = len(detections)
        if n == 0:
            return -1
        if n == 1:
            return 0

        montage = self._make_montage(image, detections)

        prompt = (
            f"The image shows {n} candidate objects, each labeled with a number "
            f"from #0 to #{n-1}. Question: \"{question}\". "
            f"Which single numbered candidate best answers the question? "
            f"Reply with ONLY the number.")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": montage},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[montage],
                                padding=True, return_tensors="pt").to(self.device)

        gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True)[0]

        return self._parse_index(out, n)

    @staticmethod
    def _parse_index(text: str, n: int) -> int:
        m = re.search(r"\d+", text)
        if not m:
            return 0  # 파싱 실패시 최고 점수(0번)로 fallback
        idx = int(m.group())
        if 0 <= idx < n:
            return idx
        return 0
