"""Cross-room 이동 시 방 우선순위 판단 (SysNav paper Sec. IV-B-2,
"Room-query navigation mode").

논문 원문: in-room exploration이 target을 못 찾고 끝나면, 아직 안 본 방들의 속성
{A(v_k^r) | v_k^r ∈ R_uncov}과 로봇 궤적, task goal을 VLM에게 줘서 다음에 갈 방을
고르게 한다. 여기서는 "아직 안 본 방들의 category"와 task 문장만 주고 관련도 순위를
매기게 한다 (전체 궤적까지는 안 줌 - 방 우선순위 판단엔 지금 뭘 찾는지가 핵심이라
군더더기 없이 이 정도면 충분하다고 판단).

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
        """rooms: [{"room_id": int, "category": str}, ...] (category가 있는 방만
        호출 쪽이 넘김). 반환: room_id를 관련도 높은 순서로 정렬한 리스트."""
        fallback = [int(room["room_id"]) for room in rooms]
        if not rooms:
            return []
        try:
            self._load()
            from google.genai import types

            prompt = (
                "A mobile robot is searching for something and has run out of places "
                "to look in its current area. Below is the robot's task and a list of "
                "rooms it has not entered yet (only each room's category is known, "
                "e.g. \"kitchen\", \"bedroom\"). Rank the room_ids from most likely to "
                "least likely to contain what the task is looking for. Every room_id "
                "must appear exactly once in the output.\n"
                f"Task: {task_description}\n"
                "Rooms: " + json.dumps(rooms, ensure_ascii=False)
            )
            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
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
