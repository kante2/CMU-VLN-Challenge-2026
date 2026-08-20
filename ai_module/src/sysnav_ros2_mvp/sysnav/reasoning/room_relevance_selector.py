"""Cross-room 이동 시 방 우선순위 판단 (SysNav paper Sec. IV-B-2,
"Room-query navigation mode").

논문 원문: in-room exploration이 target을 못 찾고 끝나면, 아직 안 본 방들의 속성
{A(v_k^r) | v_k^r ∈ R_uncov}과 로봇 궤적, task goal을 VLM에게 줘서 다음에 갈 방을
고르게 한다. 여기서는 방 category, 이미 발견한 object category,
대표 viewpoint 이미지, 거리, 방문 여부와 task 문장을 함께 제공한다.

DetectionVerifier처럼 fail-open이다: 실패하면 입력 순서(호출 쪽이 보통 거리순으로
정렬해서 넘김)를 그대로 쓴다 - "똑똑하게 순서를 못 정한 것"이 "그 방으로 아예
못 가는 것"보다 훨씬 낫다 (cross_room_navigator.py가 순서대로 plan_direct_path를
시도하다 첫 성공한 방으로 감).
"""

from __future__ import annotations

import json
import os

from rclpy.logging import get_logger

from sysnav import config


class RoomRelevanceSelector:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._client = None
        self._logger = get_logger("sysnav_room_relevance")

    def _load(self) -> None:
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai
        self._client = genai.Client(api_key=self.api_key)

    def rank(self, task_description: str, rooms: list[dict]) -> list[int]:
        """Rank candidate rooms using category, objects, distance and best view."""
        fallback = [int(room["room_id"]) for room in rooms]
        if not rooms:
            return []
        try:
            self._load()
            from google.genai import types

            room_text = [
                {key: value for key, value in room.items() if key != "image_path"}
                for room in rooms
            ]
            prompt = (
                "A mobile robot is searching for something and has run out of places "
                "to look in its current area. Rank the room_ids from most useful to "
                "least useful for completing the task. Use room category, detected "
                "objects, travel distance, visited state, and the matching room image "
                "when supplied. Prefer semantic likelihood over a small distance saving. "
                "Every room_id must appear exactly once.\n"
                f"Task: {task_description}\n"
                "Rooms: " + json.dumps(room_text, ensure_ascii=False)
            )
            contents: list[object] = [prompt]
            for room in rooms:
                image_path = room.get("image_path")
                if not image_path:
                    continue
                try:
                    with open(image_path, "rb") as image_file:
                        image_bytes = image_file.read()
                    if image_bytes:
                        contents.append(f"Best view for room_id={int(room['room_id'])}:")
                        contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
                except OSError:
                    continue
            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "ranked_room_ids": {"type": "array", "items": {"type": "integer"}},
                        },
                        "required": ["ranked_room_ids"],
                    },
                ),
            )
            if not response.text:
                raise RuntimeError("Gemini가 빈 응답을 반환함")
            ranked = [int(value) for value in json.loads(response.text)["ranked_room_ids"]]
            valid_ids = {int(room["room_id"]) for room in rooms}
            ranked = [room_id for room_id in ranked if room_id in valid_ids]
            missing = [room_id for room_id in fallback if room_id not in ranked]
            result = ranked + missing
            self._logger.info(f"Room relevance ranking: {result}")
            return result
        except Exception as error:
            self._logger.warning(f"Room relevance ranking skipped (distance-order fallback): {error}")
            return fallback
