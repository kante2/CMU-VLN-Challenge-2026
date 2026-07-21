"""SAM2 bbox-prompt segmentation adapter."""

from __future__ import annotations

import os
import threading

import numpy as np

from sysnav import config


class Sam2Segmenter:
    def __init__(self) -> None:
        self._predictor = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._predictor is not None:
            return
        if not config.SAM2_CHECKPOINT:
            raise RuntimeError("Set SAM2_CHECKPOINT environment variable.")
        if not os.path.exists(config.SAM2_CHECKPOINT):
            raise FileNotFoundError(config.SAM2_CHECKPOINT)
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError("Install the official facebookresearch/sam2 package.") from exc
        model = build_sam2(config.SAM2_MODEL_CFG, config.SAM2_CHECKPOINT, device=config.SAM2_DEVICE)
        self._predictor = SAM2ImagePredictor(model)

    def segment(self, image_rgb: np.ndarray, detections: list[dict]) -> list[dict]:
        if not detections:
            return []
        with self._lock:
            self._load()
            self._predictor.set_image(image_rgb)
            output = []
            for detection in detections:
                masks, scores, _ = self._predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.asarray(detection["bbox"], dtype=np.float32),
                    multimask_output=False,
                )
                if masks is None or len(masks) == 0:
                    continue
                mask = np.asarray(masks[0], dtype=bool)
                if int(mask.sum()) < config.SAM2_MIN_MASK_AREA_PX:
                    continue
                item = dict(detection)
                item["mask"] = mask
                item["segmentation_score"] = float(scores[0]) if len(scores) else 0.0
                output.append(item)
            return output
