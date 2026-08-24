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
import time

import cv2
import numpy as np
from rclpy.logging import get_logger

from sysnav import config
from sysnav.activity_log import LLM, activity
from sysnav.llm_trace import llm_trace


class DetectionVerifier:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._client = None
        self._logger = get_logger("sysnav_detection_verifier")
        # (카테고리, 양자화 bbox) -> (판정, 기록 시각). config.DETECTION_VERIFICATION_
        # CACHE_TTL_SEC 주석 참고 - 같은 박스를 매 프레임 다시 묻는 것을 막는다.
        self._cache: dict[tuple, tuple[bool, float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _cache_key(detection: dict) -> tuple:
        quant = max(1, int(config.DETECTION_VERIFICATION_CACHE_BBOX_QUANT_PX))
        x1, y1, x2, y2 = detection["bbox"]
        return (
            str(detection["category"]).lower(),
            int(x1) // quant, int(y1) // quant, int(x2) // quant, int(y2) // quant,
        )

    def _cached(self, detection: dict) -> bool | None:
        entry = self._cache.get(self._cache_key(detection))
        if entry is None:
            return None
        verdict, stored_at = entry
        if time.monotonic() - stored_at > config.DETECTION_VERIFICATION_CACHE_TTL_SEC:
            return None
        return verdict

    def _remember(self, detection: dict, verdict: bool) -> None:
        self._cache[self._cache_key(detection)] = (verdict, time.monotonic())

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

        # 이미 같은 박스를 물어봤으면 그 답을 재사용한다. 같은 입력이니 같은 답이고,
        # 프레임마다 반복되는 왕복(실측 45초 중 30초)을 그대로 없앤다.
        #
        # 이 블록도 반드시 try 안이어야 한다 - 밖에 두면 detection 형태가 예상과 다를 때
        # (예: bbox 키 없음) 예외가 그대로 올라가 perception job을 죽인다. 이 클래스의
        # 계약은 "검증이 실패해도 원래 탐지를 막지 않는다"(fail-open)이다.
        # results는 항상 detections와 같은 길이를 유지한다 - 호출 측이 인덱스로 짝지으므로
        # 중간에 예외가 나도 길이가 달라지면 안 된다.
        results: list[bool | None] = [None] * len(detections)
        pending: list[int] = []
        try:
            for index, detection in enumerate(detections):
                cached = self._cached(detection)
                results[index] = cached
                if cached is None:
                    pending.append(index)
            self.cache_hits += len(detections) - len(pending)
            self.cache_misses += len(pending)
            if not pending:
                return [bool(value) for value in results]

            asked = [detections[index] for index in pending]

            self._load()
            from google.genai import types

            annotated = self._annotate(image_rgb, asked)
            candidates = [
                {"index": position, "category": detection["category"]}
                for position, detection in enumerate(asked)
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
            with activity.operation(LLM, "Gemini 검출 재확인"):
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
                return [True if value is None else value for value in results]

            confirmed_by_position = {
                int(item["index"]): bool(item["confirmed"])
                for item in json.loads(response.text).get("results", [])
                if "index" in item
            }
            for position, index in enumerate(pending):
                verdict = confirmed_by_position.get(position, True)
                results[index] = verdict
                self._remember(detections[index], verdict)
            rejected = [
                f"{detections[index]['category']}#{index}"
                for index, ok in enumerate(results)
                if not ok
            ]
            llm_trace.record(
                kind="검출 재확인",
                question="주석 박스 안의 물체가 정말 그 카테고리인가",
                images=[("주석 이미지 (모델이 본 그대로)", annotated)],
                verdicts=[
                    (
                        f"{detection['category']} (box {position})",
                        confirmed_by_position.get(position),
                        "",
                    )
                    for position, detection in enumerate(asked)
                ],
                summary=f"확인 {sum(1 for v in confirmed_by_position.values() if v)} / 질의 {len(asked)}",
            )
            if rejected:
                self._logger.info(f"Detection verification rejected: {rejected}")
            return [True if value is None else value for value in results]
        except Exception as error:
            # fail-open. 실패는 캐시에 남기지 않는다 - 다음 프레임엔 다시 시도해야 한다.
            self._logger.warning(f"Detection verification skipped (fail-open): {error}")
            return [True if value is None else value for value in results]
