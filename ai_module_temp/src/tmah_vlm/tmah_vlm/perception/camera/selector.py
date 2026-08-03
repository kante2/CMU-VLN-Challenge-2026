#!/usr/bin/env python3
"""
Qwen2-VL 기반 후보 선택기.

GroundingDINO가 만든 후보 박스 중에서 원본 질문에 가장 맞는 후보 번호를 고른다.
문법을 단순하게 하기 위해 staticmethod, torch decorator를 쓰지 않는다.

Qwen2.5-VL-3B-Instruct(~6.3GB)를 쓰다가 GroundingDINO+SAM이랑 같이 8GB GPU에
못 들어가서(OOM) Qwen2-VL-2B-Instruct(~4GB)로 낮췄다. 프롬프트/처리 방식은
2.5-VL과 거의 동일해서(같은 계열) 로직은 안 바뀌고 모델 클래스/ID만 다르다.
"""

import re

import torch
from PIL import Image as PILImage, ImageDraw, ImageFont


class QwenSelector:
    def __init__(self,
                 model_id="Qwen/Qwen2-VL-2B-Instruct",
                 device=None,
                 max_new_tokens=64):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.max_new_tokens = max_new_tokens

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=768 * 28 * 28,
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=self.device,
        )
        self.model.eval()

    def load_font(self, size):
        try:
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                size,
            )
        except Exception:
            return ImageFont.load_default()

    def make_montage(self, image, detections, pad=8, cell_w=220):
        """후보 crop들을 번호와 함께 한 장의 이미지로 이어붙인다."""
        crops = []
        font = self.load_font(20)

        for index, det in enumerate(detections):
            x1, y1, x2, y2 = [int(v) for v in det.box]
            x1 = max(0, x1 - 10)
            y1 = max(0, y1 - 10)
            x2 = min(image.width, x2 + 10)
            y2 = min(image.height, y2 + 10)

            crop = image.crop((x1, y1, x2, y2))
            if crop.width > 0:
                ratio = cell_w / float(crop.width)
                new_height = max(1, int(crop.height * ratio))
                crop = crop.resize((cell_w, new_height))

            labeled = PILImage.new("RGB", (crop.width, crop.height + 28),
                                   (30, 30, 30))
            labeled.paste(crop, (0, 28))

            draw = ImageDraw.Draw(labeled)
            draw.text((5, 3), f"#{index}", fill=(255, 255, 0), font=font)
            crops.append(labeled)

        if len(crops) == 0:
            return image

        total_width = sum(crop.width for crop in crops) + pad * (len(crops) + 1)
        max_height = max(crop.height for crop in crops) + pad * 2
        montage = PILImage.new("RGB", (total_width, max_height), (0, 0, 0))

        x = pad
        for crop in crops:
            montage.paste(crop, (x, pad))
            x += crop.width + pad

        return montage

    def parse_index(self, text, num_candidates):
        match = re.search(r"\d+", text)
        if match is None:
            return 0

        index = int(match.group())
        if 0 <= index < num_candidates:
            return index
        return 0

    def choose(self, image, detections, question):
        """Process: 몽타주 만들기 -> prompt 구성 -> 모델 추론 -> 답 파싱."""
        num_candidates = len(detections)
        if num_candidates == 0:
            return -1
        if num_candidates == 1:
            return 0

        montage = self.make_montage(image, detections)
        prompt = self.build_prompt(num_candidates, question)
        output_text = self.run_model(montage, prompt)

        return self.parse_index(output_text, num_candidates)

    def build_prompt(self, num_candidates, question):
        return (
            f"The image shows {num_candidates} candidate objects, "
            f"each labeled with a number from #0 to #{num_candidates - 1}. "
            f"Question: \"{question}\". "
            f"Which single numbered candidate best answers the question? "
            f"Reply with ONLY the number."
        )

    def run_model(self, montage, prompt):
        """montage 이미지 + prompt를 Qwen에 넣고 생성된 답변 텍스트를 반환한다."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": montage},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=[montage],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]

        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
