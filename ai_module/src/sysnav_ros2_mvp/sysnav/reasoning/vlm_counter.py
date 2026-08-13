"""Numerical 미션의 개수를 viewpoint 파노라마 한 장으로 VLM에게 직접 세게 한다.

왜 필요한가: count_job은 object_memory에서 후보를 세므로 **탐지 재현율에 갇힌다**.
실측(home_building_1): pillow가 GT 18개인데 최종 메모리엔 7개만 남았다. 베개 4개 중
2개만 탐지되면 답은 영원히 2다 - 병합·필터를 아무리 손봐도 못 본 물체를 셀 수는 없다.
VLM이 이미지를 직접 보고 세면 그 상한을 우회한다.

뷰는 한 장만 쓴다: scene_graph.best_viewpoint_for_objects()가 "그 카테고리 물체를 가장
많이 동시에 본" viewpoint를 고른다. 여러 뷰의 개수를 합치면 같은 베개가 여러 뷰에
찍혀 중복 계산되는데, 뷰를 하나로 확정하면 그 문제가 구조적으로 사라진다.

숫자 대신 **항목 목록**을 받는다 - VLM은 5~6개를 넘으면 총합을 자주 틀리지만 하나씩
나열하는 건 비교적 안정적이고, 무엇을 셌는지 로그로 검사할 수 있다(관계 제약이 걸린
정답은 우리에게 GT가 없어서 사람이 눈으로 확인하는 것이 유일한 검증 수단이다).

attribute_verifier.py와 달리 fail-open이 아니라 **fail-quiet**이다: 실패하면 None을
돌려주고 호출 쪽이 기존 기하 기반 개수를 그대로 쓴다. 개수 미션은 0/1 채점이라
"답을 못 냄"이 최악이므로, VLM이 죽어도 기존 경로가 답을 내야 한다.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
from rclpy.logging import get_logger

from sysnav import config


class VlmCounter:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._client = None
        self._logger = get_logger("sysnav_vlm_counter")
        self.last_items: list[str] = []
        self.last_viewpoint_id: int | None = None

    def _load(self) -> None:
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai
        self._client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _jpeg(image_bgr: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("VLM count JPEG encoding failed")
        return encoded.tobytes()

    @staticmethod
    def _load_viewpoint_image(image_path: str) -> np.ndarray | None:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        return image if image is not None and image.size else None

    def count(self, question: str, target: str, viewpoint: dict) -> int | None:
        """viewpoint 파노라마에서 question이 요구하는 물체 개수를 센다.

        question 원문을 그대로 넣는다 - "on the sofa under the pictures" 같은 제약을
        우리가 재구성해 전달하면 뉘앙스가 깎인다. 우리 기하 판정(on = 수직 0.25m /
        수평 0.20m 허용)이 grounding 오차에 민감한 것과 달리, VLM은 이미지 공간에서
        같은 제약을 훨씬 쉽게 본다.

        반환: 개수, 또는 판정 불가 시 None(호출 쪽이 기존 개수를 유지).
        """
        self.last_items = []
        self.last_viewpoint_id = viewpoint.get("viewpoint_id")
        image = self._load_viewpoint_image(viewpoint["image_path"])
        if image is None:
            self._logger.warning(
                f"VLM count skipped: cannot read {viewpoint.get('image_path')}"
            )
            return None

        try:
            self._load()
            from google.genai import types

            prompt = (
                "You are counting objects in a 360-degree panorama taken by a robot "
                "indoors. The left and right edges of the image wrap around, so an "
                "object split across both edges is ONE object.\n\n"
                f"Question: {question}\n\n"
                f"List every {target} in this image that satisfies the question, one "
                "entry per object, with a short description of where it is so a human "
                "can check your work. Count only what is actually visible - do not "
                "infer objects that are hidden or that you merely expect to be there, "
                "and do not list the same physical object twice.\n"
                "If none satisfy it, return an empty list."
            )
            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=self._jpeg(image), mime_type="image/jpeg"),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"where": {"type": "string"}},
                                    "required": ["where"],
                                },
                            }
                        },
                        "required": ["items"],
                    },
                ),
            )
            if not response.text:
                raise RuntimeError("Gemini가 빈 응답을 반환함")
            items = json.loads(response.text).get("items", [])
            self.last_items = [str(item.get("where", "?")) for item in items]
            count = len(self.last_items)
            self._logger.info(
                f"VLM count on viewpoint {self.last_viewpoint_id}: {count} x {target} "
                f"-> {self.last_items}"
            )
            return count
        except Exception as error:
            # fail-quiet: 기존 기하 기반 개수를 그대로 쓰게 둔다.
            self._logger.warning(f"VLM count unavailable, keeping geometric count: {error}")
            return None
