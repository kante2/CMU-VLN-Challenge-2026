"""YOLOv8x-WorldV2 adapter."""

from __future__ import annotations

import threading
from typing import Iterable

import numpy as np

from sysnav import config


class YoloWorldDetector:
    def __init__(self, weights: str = config.YOLO_WORLD_WEIGHTS, device: str = config.YOLO_DEVICE) -> None:
        self.weights = weights
        self.device = device
        self._model = None
        self._classes: tuple[str, ...] = ()
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLOWorld
            from ultralytics.utils.torch_utils import select_device
        except ImportError as exc:
            raise RuntimeError("Install ultralytics: pip install ultralytics") from exc
        self._model = YOLOWorld(self.weights)
        # Move to the inference device before the first set_classes() call, otherwise
        # the CLIP text model it builds and caches gets its weights migrated to GPU by
        # a later predict(device=...) call while its own `.device` attribute (used to
        # place tokenized text) stays stale at CPU, causing an index_select device
        # mismatch the next time set_classes() runs with new prompts.
        self._model.to(select_device(self.device))

    def detect(self, image_rgb: np.ndarray, prompts: Iterable[str]) -> list[dict]:
        prompt_list = []
        for prompt in prompts:
            value = str(prompt).strip().lower()
            if value and value not in prompt_list:
                prompt_list.append(value)
        if not prompt_list:
            return []

        with self._lock:
            self._load()
            if tuple(prompt_list) != self._classes:
                self._model.set_classes(prompt_list)
                self._classes = tuple(prompt_list)
            results = self._model.predict(
                source=image_rgb,
                conf=config.YOLO_CONFIDENCE,
                iou=config.YOLO_IOU,
                imgsz=config.YOLO_IMAGE_SIZE,
                max_det=config.YOLO_MAX_DETECTIONS,
                device=self.device,
                verbose=False,
            )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        result = results[0]
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        names = result.names
        height, width = image_rgb.shape[:2]
        detections = []

        for box, confidence, class_id in zip(xyxy, confidences, class_ids):
            x1, y1, x2, y2 = box.tolist()
            bbox = (
                int(max(0, min(width - 1, round(x1)))),
                int(max(0, min(height - 1, round(y1)))),
                int(max(1, min(width, round(x2)))),
                int(max(1, min(height, round(y2)))),
            )
            category = str(names[class_id] if not isinstance(names, dict) else names.get(class_id, prompt_list[class_id]))
            detections.append({"category": category.lower(), "confidence": float(confidence), "bbox": bbox})
        return detections
