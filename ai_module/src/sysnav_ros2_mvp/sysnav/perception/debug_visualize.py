"""Save detection/segmentation/grounding results as overlay images under config.DEBUG_DIR."""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

from sysnav import config

_MASK_COLOR_BGR = np.array([255, 0, 255], dtype=np.float32)  # magenta
_BOX_COLOR_BGR = (0, 255, 0)


def save_debug_image(
    image_rgb: np.ndarray,
    segmented: list[dict],
    grounded: list[dict] | None = None,
    tag: str = "",
) -> None:
    if not config.SAVE_DEBUG_IMAGES:
        return
    try:
        position_by_bbox = {}
        if grounded:
            for obj in grounded:
                position_by_bbox[tuple(obj["bbox"])] = obj.get("position")

        overlay = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
        for det in segmented:
            bbox = tuple(det["bbox"])
            x1, y1, x2, y2 = bbox
            mask = det.get("mask")
            if mask is not None:
                blended = 0.5 * overlay[mask].astype(np.float32) + 0.5 * _MASK_COLOR_BGR
                overlay[mask] = blended.astype(np.uint8)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), _BOX_COLOR_BGR, 2)

            label = f"{det['category']} {det['confidence']:.2f}"
            position = position_by_bbox.get(bbox)
            if position is not None:
                label += f" ({position[0]:.2f},{position[1]:.2f},{position[2]:.2f})"
            cv2.putText(
                overlay, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BOX_COLOR_BGR, 1, cv2.LINE_AA,
            )

        os.makedirs(config.DEBUG_DIR, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        filename = f"sysnav_detect_{time.time():.3f}{suffix}.jpg"
        cv2.imwrite(os.path.join(config.DEBUG_DIR, filename), overlay)
    except Exception as error:  # pragma: no cover - debug output must never crash perception
        print(f"[sysnav debug_visualize] failed to save debug image: {error}")
