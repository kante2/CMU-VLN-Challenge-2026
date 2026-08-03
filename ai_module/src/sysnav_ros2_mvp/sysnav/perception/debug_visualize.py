"""Save detection/segmentation/grounding results as overlay images under config.DEBUG_DIR."""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

from sysnav import config

_MASK_COLOR_BGR = np.array([255, 0, 255], dtype=np.float32)  # magenta
_BOX_COLOR_BGR = (0, 255, 0)


def save_lidar_projection_image(
    image_rgb: np.ndarray,
    projection: dict | None,
) -> str | None:
    """Save all projected LiDAR pixels and detection-mask hits as an overlay."""
    if not config.SAVE_DEBUG_IMAGES or not projection:
        return None
    try:
        overlay = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
        all_u = np.asarray(projection.get("u", []), dtype=np.int32)
        all_v = np.asarray(projection.get("v", []), dtype=np.int32)
        # Cyan: every valid LiDAR point projected onto the panorama.
        for u, v in zip(all_u.tolist(), all_v.tolist()):
            cv2.circle(overlay, (u, v), 1, (255, 255, 0), -1, cv2.LINE_AA)

        # Red: projected points that fall inside each SAM detection mask.
        for hit in projection.get("hits", []):
            hit_u = np.asarray(hit.get("u", []), dtype=np.int32)
            hit_v = np.asarray(hit.get("v", []), dtype=np.int32)
            for u, v in zip(hit_u.tolist(), hit_v.tolist()):
                cv2.circle(overlay, (u, v), 4, (0, 0, 255), -1, cv2.LINE_AA)
            x1, y1, x2, y2 = [int(value) for value in hit.get("bbox", (0, 0, 0, 0))]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{hit.get('category', '')} lidar_hits={len(hit_u)}"
            cv2.putText(
                overlay,
                label,
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            overlay,
            f"cyan=all projected LiDAR ({len(all_u)}), red=detection-mask hits",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        os.makedirs(config.DEBUG_DIR, exist_ok=True)
        filename = f"sysnav_lidar_projection_{time.time():.3f}.jpg"
        path = os.path.join(config.DEBUG_DIR, filename)
        cv2.imwrite(path, overlay)
        return path
    except Exception as error:  # pragma: no cover
        print(f"[sysnav debug_visualize] failed to save LiDAR projection: {error}")
        return None


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
            lidar_points = int(det.get("grounding_num_points", 0))
            label += f" lidar={lidar_points}"
            provisional_points = int(det.get("provisional_num_points", 0))
            provisional_frames = int(det.get("provisional_frame_count", 0))
            if provisional_points:
                label += f" accum={provisional_points}/{provisional_frames}f"
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
