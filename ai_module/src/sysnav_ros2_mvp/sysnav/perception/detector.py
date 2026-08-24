"""YOLO-World detector with selective YOLO12s COCO-class augmentation."""

from __future__ import annotations

import threading
from typing import Iterable

import numpy as np

from sysnav import config


# COCO names accepted by Ultralytics YOLO12. Values cover common plural/spelling
# variants emitted by the query parser; detections are mapped back to the original
# prompt so Scene Graph category matching remains exact.
_COCO_ALIASES = {
    "person": "person", "people": "person",
    "bicycle": "bicycle", "bike": "bicycle", "car": "car", "motorcycle": "motorcycle",
    "airplane": "airplane", "bus": "bus", "train": "train", "truck": "truck", "boat": "boat",
    "traffic light": "traffic light", "fire hydrant": "fire hydrant", "stop sign": "stop sign",
    "parking meter": "parking meter", "bench": "bench", "bird": "bird", "cat": "cat",
    "dog": "dog", "horse": "horse", "sheep": "sheep", "cow": "cow", "elephant": "elephant",
    "bear": "bear", "zebra": "zebra", "giraffe": "giraffe", "backpack": "backpack",
    "umbrella": "umbrella", "handbag": "handbag", "tie": "tie", "suitcase": "suitcase",
    "frisbee": "frisbee", "skis": "skis", "snowboard": "snowboard", "sports ball": "sports ball",
    "kite": "kite", "baseball bat": "baseball bat", "baseball glove": "baseball glove",
    "skateboard": "skateboard", "surfboard": "surfboard", "tennis racket": "tennis racket",
    "bottle": "bottle", "bottles": "bottle", "wine glass": "wine glass", "cup": "cup",
    "cups": "cup", "fork": "fork", "knife": "knife", "spoon": "spoon", "bowl": "bowl",
    "banana": "banana", "apple": "apple", "sandwich": "sandwich", "orange": "orange",
    "broccoli": "broccoli", "carrot": "carrot", "hot dog": "hot dog", "pizza": "pizza",
    "donut": "donut", "cake": "cake", "chair": "chair", "chairs": "chair", "couch": "couch",
    "sofa": "couch", "potted plant": "potted plant", "potted plants": "potted plant",
    "bed": "bed", "table": "dining table", "tables": "dining table",
    "dining table": "dining table", "toilet": "toilet", "tv": "tv",
    "television": "tv", "laptop": "laptop", "mouse": "mouse", "remote": "remote",
    "keyboard": "keyboard", "cell phone": "cell phone", "microwave": "microwave", "oven": "oven",
    "toaster": "toaster", "sink": "sink", "refrigerator": "refrigerator", "book": "book",
    "books": "book", "clock": "clock", "vase": "vase", "scissors": "scissors",
    "teddy bear": "teddy bear", "hair drier": "hair drier", "toothbrush": "toothbrush",
}


# YOLO-World(CLIP text encoder)가 같은 물체를 훨씬 잘 잡는 **동의 명사구**들.
# 카테고리 하나를 프롬프트 여러 개로 넓혀 쏘고, 결과는 원래 카테고리로 되돌린다.
#
# 왜 필요한가(2026-08-24 실측, viewpoint 42프레임): "the round tables"를 파서가
# category="table" + attributes=["round"]로 쪼개 "table"만 프롬프트로 줬더니 거실
# 원형 테이블을 42프레임 중 5프레임에서만 잡았다(best conf 0.55). 같은 물체에
# "coffee table"을 주면 22/42(best 0.91), "side table"은 median 0.49로 잘 잡는다.
# "table"이 CLIP 임베딩에서 너무 넓은 단어라 특정 가구에 잘 안 붙는 것이다.
#
# 주의: 형용사를 붙이는 것과는 정반대의 이야기다. "round table"은 3/42로 "table"보다
# **더 나빴다** - open-vocab 검출기는 형용사를 구분 못 하므로 색/모양은 여전히
# reasoning/attribute_verifier.py가 판정해야 한다. 여기 넣는 것은 형용사가 아니라
# 물체 종류가 같은 별개의 명사구여야 한다.
#
# 실측으로 확인한 것만 넣는다(다른 카테고리도 같은 방식으로 재보고 추가할 것).
_PROMPT_ALIASES = {
    "table": ("coffee table", "dining table", "side table"),
}


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union else 0.0


def _merge_detections(detections: list[dict]) -> list[dict]:
    """Class-aware NMS across both models, retaining the stronger overlapping box."""
    kept: list[dict] = []
    for detection in sorted(detections, key=lambda item: float(item["confidence"]), reverse=True):
        duplicate = any(
            detection["category"] == other["category"]
            and _bbox_iou(detection["bbox"], other["bbox"]) >= config.YOLO_ENSEMBLE_MERGE_IOU
            for other in kept
        )
        if not duplicate:
            kept.append(detection)
    return kept[:config.YOLO_MAX_DETECTIONS]


class YoloWorldDetector:
    def __init__(self, weights: str = config.YOLO_WORLD_WEIGHTS, device: str = config.YOLO_DEVICE) -> None:
        self.weights = weights
        self.device = device
        self._model = None
        self._yolo12_model = None
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

    def _load_yolo12(self) -> None:
        if self._yolo12_model is not None:
            return
        try:
            from ultralytics import YOLO
            from ultralytics.utils.torch_utils import select_device
        except ImportError as exc:
            raise RuntimeError("Install ultralytics: pip install ultralytics") from exc
        self._yolo12_model = YOLO(config.YOLO12_WEIGHTS)
        self._yolo12_model.to(select_device(self.device))

    @staticmethod
    def _expand_prompts(prompt_list: list[str]) -> tuple[list[str], dict[str, str]]:
        """(실제로 쏠 프롬프트 목록, 별칭 -> 원래 카테고리) 반환.

        이미 질문에 나온 프롬프트는 절대 다른 카테고리로 접지 않는다 - "coffee table을
        찾아줘"라고 물었으면 그건 그 자체로 카테고리이지 table의 별칭이 아니다."""
        expanded = list(prompt_list)
        alias_to_category: dict[str, str] = {}
        for prompt in prompt_list:
            for alias in _PROMPT_ALIASES.get(prompt, ()):
                if alias in prompt_list or alias in alias_to_category:
                    continue
                expanded.append(alias)
                alias_to_category[alias] = prompt
        return expanded, alias_to_category

    @staticmethod
    def _coco_prompts(prompt_list: list[str]) -> dict[str, str]:
        output: dict[str, str] = {}
        for prompt in prompt_list:
            coco_name = _COCO_ALIASES.get(prompt)
            if coco_name is not None and coco_name not in output:
                output[coco_name] = prompt
        return output

    def detect(self, image_rgb: np.ndarray, prompts: Iterable[str]) -> list[dict]:
        prompt_list = []
        for prompt in prompts:
            value = str(prompt).strip().lower()
            if value and value not in prompt_list:
                prompt_list.append(value)
        if not prompt_list:
            return []

        coco_prompts = self._coco_prompts(prompt_list)
        # 별칭까지 넓혀서 쏘고(_PROMPT_ALIASES), 결과 라벨은 원래 카테고리로 되돌린다.
        # 되돌린 뒤 _merge_detections()가 카테고리 단위 NMS를 하므로, 같은 테이블이
        # "coffee table"과 "side table"로 두 번 잡혀도 하나로 합쳐진다.
        query_list, alias_to_category = self._expand_prompts(prompt_list)
        with self._lock:
            self._load()
            if tuple(query_list) != self._classes:
                self._model.set_classes(query_list)
                self._classes = tuple(query_list)
            results = self._model.predict(
                source=image_rgb,
                conf=config.YOLO_CONFIDENCE,
                iou=config.YOLO_IOU,
                imgsz=config.YOLO_IMAGE_SIZE,
                max_det=config.YOLO_MAX_DETECTIONS,
                device=self.device,
                verbose=False,
            )

            yolo12_results = []
            if config.YOLO12_COCO_ENABLED and coco_prompts:
                self._load_yolo12()
                names = self._yolo12_model.names
                name_items = names.items() if isinstance(names, dict) else enumerate(names)
                coco_class_ids = [
                    int(class_id)
                    for class_id, class_name in name_items
                    if str(class_name).lower() in coco_prompts
                ]
                yolo12_results = self._yolo12_model.predict(
                    source=image_rgb,
                    conf=config.YOLO12_CONFIDENCE,
                    iou=config.YOLO_IOU,
                    imgsz=config.YOLO_IMAGE_SIZE,
                    max_det=config.YOLO_MAX_DETECTIONS,
                    classes=coco_class_ids,
                    device=self.device,
                    verbose=False,
                )

        height, width = image_rgb.shape[:2]
        detections = []

        def append_result(result, source: str) -> None:
            if result is None or result.boxes is None or len(result.boxes) == 0:
                return
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confidences = result.boxes.conf.detach().cpu().numpy()
            class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
            names = result.names
            for box, confidence, class_id in zip(xyxy, confidences, class_ids):
                model_category = str(
                    names[class_id] if not isinstance(names, dict) else names.get(class_id, "")
                ).lower()
                if source == "yolo12s":
                    category = coco_prompts.get(model_category)
                    if category is None:
                        continue
                    threshold = (
                        config.YOLO12_BOOK_CONFIDENCE
                        if model_category == "book"
                        else config.YOLO12_DEFAULT_CLASS_CONFIDENCE
                    )
                    if float(confidence) < threshold:
                        continue
                else:
                    category = alias_to_category.get(model_category, model_category)
                x1, y1, x2, y2 = box.tolist()
                bbox = (
                    int(max(0, min(width - 1, round(x1)))),
                    int(max(0, min(height - 1, round(y1)))),
                    int(max(1, min(width, round(x2)))),
                    int(max(1, min(height, round(y2)))),
                )
                detections.append({
                    "category": category,
                    "confidence": float(confidence),
                    "bbox": bbox,
                    "detector": source,
                })

        if results:
            append_result(results[0], "yolo_world")
        if yolo12_results:
            append_result(yolo12_results[0], "yolo12s")
        return _merge_detections(detections)
