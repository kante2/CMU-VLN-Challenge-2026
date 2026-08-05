"""Verify YOLO-World's low-confidence detections with a single Gemini VLM pass.

YOLO-World가 애매한 confidence로 카테고리를 잘못 붙이는 경우(예: 침대를 0.29로
"sofa"라고 오검출)를 줄이기 위한 2차 검증. 한 프레임에서 애매한 detection 전부를
모아 이미지 하나 + 후보 리스트로 Gemini에 한 번만 물어본다(박스마다 따로 부르면
지연이 누적되므로). 검증 자체가 실패(API 에러, 키 없음 등)하면 원래 탐지를 막지
않도록 fail-open(전부 통과)한다.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
from rclpy.logging import get_logger

from sysnav import config


class DetectionVerifier:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._client = None
        self._logger = get_logger("sysnav_detection_verifier")

    def _load(self) -> None:
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai
        self._client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _jpeg(image_rgb: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        if not ok:
            raise RuntimeError("Detection-verification JPEG encoding failed")
        return encoded.tobytes()

    @staticmethod
    def _annotate(image_rgb: np.ndarray, detections: list[dict]) -> np.ndarray:
        annotated = image_rgb.copy()
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(
                annotated,
                f"#{index} {detection['category']}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def verify(self, image_rgb: np.ndarray, detections: list[dict]) -> list[bool]:
        """detections: [{"category": str, "bbox": (x1,y1,x2,y2), ...}, ...].
        반환: 같은 길이의 bool 리스트 (True=진짜 그 카테고리가 맞다고 확인됨).
        검증을 안 하거나 못 하면(비활성/키 없음/에러) 전부 True(통과)를 반환한다."""
        if not detections:
            return []
        if not config.DETECTION_VERIFICATION_ENABLED or not self.api_key:
            return [True] * len(detections)

        try:
            self._load()
            from google.genai import types

            annotated = self._annotate(image_rgb, detections)
            candidates = [
                {"index": index, "category": detection["category"]}
                for index, detection in enumerate(detections)
            ]
            prompt = f"""
You double-check low-confidence object detections for a mobile robot's perception
system. The image is annotated with numbered magenta boxes, one per candidate below.
Candidates: {json.dumps(candidates, ensure_ascii=False)}

For each numbered box, decide whether the stated category is really what's visible
inside that specific box. Set confirmed=true only when the label clearly matches.
If the box actually shows a different or generic object, or you are unsure, set
confirmed=false. Judge each box only by its own contents, not the rest of the scene.
""".strip()
            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=self._jpeg(annotated), mime_type="image/jpeg"),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "index": {"type": "integer"},
                                        "confirmed": {"type": "boolean"},
                                    },
                                    "required": ["index", "confirmed"],
                                },
                            }
                        },
                        "required": ["results"],
                    },
                ),
            )
            if not response.text:
                return [True] * len(detections)

            confirmed_by_index = {
                int(item["index"]): bool(item["confirmed"])
                for item in json.loads(response.text).get("results", [])
                if "index" in item
            }
            results = [confirmed_by_index.get(index, True) for index in range(len(detections))]
            rejected = [
                f"{detections[index]['category']}#{index}"
                for index, ok in enumerate(results)
                if not ok
            ]
            if rejected:
                self._logger.info(f"Detection verification rejected: {rejected}")
            return results
        except Exception as error:
            self._logger.warning(f"Detection verification skipped (fail-open): {error}")
            return [True] * len(detections)
