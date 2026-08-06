"""Infer Room Node category via VLM (SysNav paper Sec. IV-A-1, Room Node).

논문 원문: 각 Room Node는 attribute A(v_r) = {m_i^r, c_i^r, I_i^r}를 갖는다 - c_i는
room category(예: kitchen, bedroom), I_i는 "the image maximizing the room's visible
voxels"(그 방에서 가장 많이 관측한 viewpoint). 이 모듈은 그 I_i 하나(또는 이번 사이클에
여러 방이 동시에 새 대표 이미지를 얻었으면 그 여러 장)를 받아 각 room의 category를
추론한다.

AttributeVerifier와 마찬가지로 fail-closed다: VLM 호출이 실패하면 아무 것도 확정하지
않는다(빈 dict 반환) - "방을 잘못 분류해놓고 그걸 근거로 room-level 판단을 하는" 것이
"미분류 상태로 남겨두는" 것보다 나쁘다. 실패한 room의 재시도 쿨다운은 호출 쪽
(RoomRegistry.mark_classification_failed)이 관리한다.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
from rclpy.logging import get_logger

from sysnav import config


class RoomClassifier:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._client = None
        self._logger = get_logger("sysnav_room_classifier")

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
            raise RuntimeError("Room-classification JPEG encoding failed")
        return encoded.tobytes()

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray | None:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return None
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def classify_many(self, rooms: list[dict]) -> dict[int, str]:
        """rooms: [{"room_id": int, "image_path": str}, ...]. 반환: {room_id: category}.
        이미지를 못 읽었거나 VLM이 이 room에 대해 답하지 않으면 그 room_id는 결과에서
        빠진다(호출 쪽이 재시도 쿨다운을 걸도록)."""
        items = []
        for room in rooms:
            image = self._load_image(room["image_path"])
            if image is None or not image.size:
                continue
            items.append({"room_id": int(room["room_id"]), "image": image})
        if not items:
            return {}

        try:
            self._load()
            from google.genai import types

            contents: list[object] = [
                "You are helping a mobile robot understand an indoor floor plan. Each "
                "image below is the single best (most-informative) camera view captured "
                "so far inside one room. For every room_id, infer a short room category "
                "label (1-3 words, e.g. \"kitchen\", \"bedroom\", \"bathroom\", \"living "
                "room\", \"office\", \"hallway\", \"laundry room\") purely from that room's "
                "own image. If the image is ambiguous or does not clearly show a "
                "recognizable room type, answer \"unknown\" for that room_id rather than "
                "guessing. Judge each room only from its own image, not the others.\n"
                + json.dumps([{"room_id": item["room_id"]} for item in items], ensure_ascii=False)
            ]
            for item in items:
                contents.append(f"room_id={item['room_id']} image:")
                contents.append(types.Part.from_bytes(data=self._jpeg(item["image"]), mime_type="image/jpeg"))

            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=contents,
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
                                        "room_id": {"type": "integer"},
                                        "category": {"type": "string"},
                                    },
                                    "required": ["room_id", "category"],
                                },
                            }
                        },
                        "required": ["results"],
                    },
                ),
            )
            if not response.text:
                raise RuntimeError("Gemini가 빈 응답을 반환함")

            allowed = {item["room_id"] for item in items}
            output: dict[int, str] = {}
            for entry in json.loads(response.text).get("results", []):
                room_id = int(entry["room_id"])
                if room_id not in allowed:
                    continue
                output[room_id] = str(entry["category"]).strip().lower()
            self._logger.info(f"Room classification: {output}")
            return output
        except Exception as error:
            self._logger.warning(f"Room classification skipped (unclassified, not fail-open): {error}")
            return {}
