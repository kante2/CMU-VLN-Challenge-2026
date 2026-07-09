#!/usr/bin/env python3
"""
GroundingDINO 기반 zero-shot 객체 검출기.

문법을 단순하게 하기 위해 dataclass, staticmethod, torch decorator를 쓰지 않는다.
Detection은 일반 class로 만든다.
"""

import torch
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class Detection:
    def __init__(self, label, score, box):
        self.label = str(label)
        self.score = float(score)
        self.box = tuple(float(v) for v in box)

        x1, y1, x2, y2 = self.box
        self.cx = (x1 + x2) / 2.0
        self.cy = (y1 + y2) / 2.0


class GroundingDINODetector:
    def __init__(self,
                 model_id="IDEA-Research/grounding-dino-tiny",
                 device=None,
                 box_threshold=0.45,
                 text_threshold=0.25):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(self.device)
        self.model.eval()

    def normalize_prompt(self, text):
        prompt = text.strip().lower()
        if not prompt.endswith("."):
            prompt += "."
        return prompt

    def detect(self, image, prompt):
        """Process: prompt 정규화 -> 모델 추론 -> Detection list 조립."""
        text = self.normalize_prompt(prompt)
        results = self.run_model(image, text)
        return self.build_detections(results)

    def run_model(self, image, text):
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

    def build_detections(self, results):
        """모델 raw 출력을 score 내림차순 Detection list로 정리한다."""
        labels = results.get("text_labels", results.get("labels", []))

        detections = [
            Detection(label=label, score=float(score), box=box.tolist())
            for score, label, box in zip(results["scores"], labels, results["boxes"])
        ]

        detections.sort(key=lambda det: det.score, reverse=True)
        return detections
