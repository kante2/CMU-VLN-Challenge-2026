#!/usr/bin/env python3
"""
GroundingDINO 기반 zero-shot 객체 검출기 (transformers 5.x 호환).

사용 예:
    det = GroundingDINODetector()
    boxes = det.detect(pil_image, "pillow")
"""

from dataclasses import dataclass
from typing import List

import torch
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


@dataclass
class Detection:
    label: str
    score: float
    box: tuple            # (x1, y1, x2, y2)
    cx: float = 0.0
    cy: float = 0.0


class GroundingDINODetector:
    def __init__(self,
                 model_id: str = "IDEA-Research/grounding-dino-tiny",
                 device: str = None,
                 box_threshold: float = 0.35,
                 text_threshold: float = 0.25):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id).to(self.device)
        self.model.eval()

    @staticmethod
    def _normalize_prompt(text: str) -> str:
        t = text.strip().lower()
        if not t.endswith("."):
            t += "."
        return t

    @torch.no_grad()
    def detect(self, image: PILImage.Image, prompt: str) -> List[Detection]:
        text = self._normalize_prompt(prompt)

        inputs = self.processor(images=image, text=text,
                                return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # transformers 5.x: box_threshold -> threshold
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],  # (height, width)
        )[0]

        # 라벨은 버전에 따라 text_labels(문자열) 또는 labels(정수)
        labels = results.get("text_labels", results.get("labels", []))

        dets: List[Detection] = []
        for score, label, box in zip(results["scores"], labels, results["boxes"]):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            dets.append(Detection(
                label=str(label),
                score=float(score),
                box=(x1, y1, x2, y2),
                cx=(x1 + x2) / 2.0,
                cy=(y1 + y2) / 2.0,
            ))

        dets.sort(key=lambda d: d.score, reverse=True)
        return dets
