"""Heuristic object association using category, distance, size and appearance."""

from __future__ import annotations

import math

import cv2
import numpy as np

from sysnav import config


def _histogram(image: np.ndarray | None) -> np.ndarray | None:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten().astype(np.float32)


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator <= 1e-8 else float(np.clip(np.dot(a, b) / denominator, 0.0, 1.0))


def association_metrics(existing: dict, observation: dict) -> dict:
    if existing["category"].lower() != observation["category"].lower():
        return {"allowed": False, "score": 0.0, "distance": float("inf")}

    old_position = np.asarray(existing["position"], dtype=np.float64)
    new_position = np.asarray(observation["position"], dtype=np.float64)
    distance = float(np.linalg.norm(old_position - new_position))
    if distance > config.ASSOCIATION_MAX_DISTANCE_M:
        return {"allowed": False, "score": 0.0, "distance": distance}

    sigma = max(config.ASSOCIATION_DISTANCE_SIGMA_M, 1e-6)
    distance_score = math.exp(-0.5 * (distance / sigma) ** 2)
    old_extent = np.asarray(existing.get("extent_3d", (0, 0, 0)), dtype=np.float64)
    new_extent = np.asarray(observation.get("extent_3d", (0, 0, 0)), dtype=np.float64)
    scale = np.maximum(np.maximum(old_extent, new_extent), 0.10)
    shape_score = float(np.clip(1.0 - np.mean(np.abs(old_extent - new_extent) / scale), 0.0, 1.0))
    new_hist = _histogram(observation.get("crop_image"))
    old_hist = existing.get("appearance_histogram")
    if old_hist is None:
        old_hist = _histogram(existing.get("representative_image"))
    appearance_score = _cosine(old_hist, new_hist)
    score = (
        config.ASSOCIATION_WEIGHT_DISTANCE * distance_score
        + config.ASSOCIATION_WEIGHT_SHAPE * shape_score
        + config.ASSOCIATION_WEIGHT_APPEARANCE * appearance_score
    )
    return {
        "allowed": True,
        "score": float(score),
        "distance": distance,
        "distance_score": float(distance_score),
        "shape_score": shape_score,
        "appearance_score": appearance_score,
        "observation_histogram": new_hist,
    }


def find_best_match(existing_nodes: list[dict], observation: dict) -> tuple[dict | None, dict | None]:
    best_node = None
    best_metrics = None
    for node in existing_nodes:
        metrics = association_metrics(node, observation)
        if metrics.get("allowed") and (best_metrics is None or metrics["score"] > best_metrics["score"]):
            best_node, best_metrics = node, metrics
    if best_metrics is None or best_metrics["score"] < config.ASSOCIATION_THRESHOLD:
        return None, best_metrics
    return best_node, best_metrics
