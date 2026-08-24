"""Numerical 미션의 개수를 viewpoint 파노라마 한 장으로 VLM에게 직접 세게 한다.

왜 필요한가: count_job은 object_memory에서 후보를 세므로 **탐지 재현율에 갇힌다**.
실측(home_building_1): pillow가 GT 18개인데 최종 메모리엔 7개만 남았다. 베개 4개 중
2개만 탐지되면 답은 영원히 2다 - 병합·필터를 아무리 손봐도 못 본 물체를 셀 수는 없다.
VLM이 이미지를 직접 보고 세면 그 상한을 우회한다.

뷰는 한 장만 쓴다: scene_graph.best_viewpoint_for_objects()가 "그 카테고리 물체를 가장
많이 동시에 본" viewpoint를 고른다. 여러 뷰의 개수를 합치면 같은 베개가 여러 뷰에
찍혀 중복 계산되는데, 뷰를 하나로 확정하면 그 문제가 구조적으로 사라진다.

같은 이미지를 config.NUMERICAL_VLM_COUNT_SAMPLES회 병렬로 물어보고 **최빈 개수**를
채택한다(self-consistency). temperature=0.0인데도 응답이 결정적이지 않아 1회 호출은
사실상 동전던지기다 - config.py의 해당 항목 주석에 실측 기록이 있다.

숫자 대신 **항목 목록**을 받는다 - VLM은 5~6개를 넘으면 총합을 자주 틀리지만 하나씩
나열하는 건 비교적 안정적이고, 무엇을 셌는지 로그로 검사할 수 있다(관계 제약이 걸린
정답은 우리에게 GT가 없어서 사람이 눈으로 확인하는 것이 유일한 검증 수단이다).

모델은 config.GEMINI_COUNTING_MODEL을 쓴다 - 다른 호출들이 쓰는 GEMINI_MODEL과 분리한
이유는 config.py의 해당 항목 주석 참고(요약: 이 호출은 질문당 1회뿐이라 상위 모델을
써도 예산에 거의 영향이 없는 반면, 파노라마 한 장에서 제약을 만족하는 물체를 빠짐없이
세는 건 이 시스템에서 가장 어려운 추론이다). 그 모델을 못 쓰면 GEMINI_MODEL로 폴백한다.

attribute_verifier.py와 달리 fail-open이 아니라 **fail-quiet**이다: 실패하면 None을
돌려주고 호출 쪽이 기존 기하 기반 개수를 그대로 쓴다. 개수 미션은 0/1 채점이라
"답을 못 냄"이 최악이므로, VLM이 죽어도 기존 경로가 답을 내야 한다.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time

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
        self.last_model: str | None = None

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
            # 상위 모델을 먼저 쓰고, 그 모델을 못 쓰면(모델명/권한/쿼터) 기본 모델로
            # 한 번 더 시도한다 - 둘 다 실패해야 fail-quiet으로 떨어진다.
            models = [config.GEMINI_COUNTING_MODEL]
            if config.GEMINI_MODEL not in models:
                models.append(config.GEMINI_MODEL)
            self._logger.info(
                f"VLM count starting: {config.NUMERICAL_VLM_COUNT_SAMPLES} parallel "
                f"sample(s) on {models[0]} (timeout {config.GEMINI_COUNTING_TIMEOUT_SEC:.0f}s each), "
                f"viewpoint {self.last_viewpoint_id}"
            )
            started = time.monotonic()
            samples = self._sample(types, prompt, image, models)
            if not samples:
                raise RuntimeError("모든 VLM count 샘플이 실패함")

            count = self._majority_count([len(items) for items, _ in samples])
            # 사람이 로그로 검증할 수 있게, 채택한 개수와 같은 샘플의 항목 목록을 남긴다.
            self.last_items, self.last_model = next(
                (items, model) for items, model in samples if len(items) == count
            )
            self._logger.info(
                f"VLM count on viewpoint {self.last_viewpoint_id} "
                f"[{self.last_model}]: {count} x {target} "
                f"(samples={[len(items) for items, _ in samples]}, "
                f"{time.monotonic() - started:.1f}s) -> {self.last_items}"
            )
            return count
        except Exception as error:
            # fail-quiet: 기존 기하 기반 개수를 그대로 쓰게 둔다.
            self._logger.warning(f"VLM count unavailable, keeping geometric count: {error}")
            return None

    def _sample(
        self, types, prompt: str, image: np.ndarray, models: list[str]
    ) -> list[tuple[list[str], str]]:
        """같은 질문을 config.NUMERICAL_VLM_COUNT_SAMPLES회 **병렬로** 물어보고 성공한
        응답만 (항목 목록, 모델명)으로 모아 돌려준다.

        병렬인 이유: 개수 세기는 탐사가 끝난 뒤 질문당 한 번뿐이라 호출 수 자체는 부담이
        아니지만, 순차로 돌리면 지연이 그대로 N배가 된다. 일부 샘플이 실패해도 성공한
        것만으로 투표한다 - 전부 실패해야 count()의 fail-quiet으로 떨어진다."""
        sample_count = max(1, int(config.NUMERICAL_VLM_COUNT_SAMPLES))
        if sample_count == 1:
            response, model = self._generate(types, prompt, image, models)
            return [(self._items(response), model)]

        results: list[tuple[list[str], str]] = []
        with ThreadPoolExecutor(max_workers=sample_count) as executor:
            futures = [
                executor.submit(self._generate, types, prompt, image, models)
                for _ in range(sample_count)
            ]
            for future in futures:
                try:
                    response, model = future.result()
                except Exception as error:
                    self._logger.warning(f"VLM count sample failed: {error}")
                    continue
                results.append((self._items(response), model))
        return results

    @staticmethod
    def _items(response) -> list[str]:
        return [
            str(item.get("where", "?"))
            for item in json.loads(response.text).get("items", [])
        ]

    @staticmethod
    def _majority_count(counts: list[int]) -> int:
        """최빈 개수. 동률이면 **작은 쪽**을 택한다.

        동률을 임의로(예: dict 순서로) 깨면 같은 입력에 답이 흔들려서, 투표로 없애려던
        문제가 그대로 되돌아온다. 작은 쪽으로 정한 이유는 파노라마에서 흔한 오류가
        좌우 wrap 구간의 같은 물체를 둘로 세는 과다 계수이기 때문이다.
        SYSNAV_NUMERICAL_VLM_COUNT_SAMPLES 기본값이 홀수라 동률 자체가 드물다."""
        frequency = Counter(counts)
        return min(frequency, key=lambda value: (-frequency[value], value))

    def _generate(self, types, prompt: str, image: np.ndarray, models: list[str]):
        """models를 순서대로 시도하고 (응답, 쓴 모델명)을 반환한다. 전부 실패하면
        마지막 예외를 그대로 올려서 count()의 fail-quiet 처리로 넘긴다."""
        jpeg = self._jpeg(image)
        last_error: Exception | None = None
        for model in models:
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        # 모델이 느려졌을 때 finalize가 무한정 매달리지 않도록 상한을 둔다
                        # (SDK는 밀리초를 받는다). 끊기면 다음 모델 -> fail-quiet 순.
                        http_options=types.HttpOptions(
                            timeout=int(config.GEMINI_COUNTING_TIMEOUT_SEC * 1000)
                        ),
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
            except Exception as error:
                last_error = error
                self._logger.warning(f"VLM count model {model} unavailable: {error}")
                continue
            if not response.text:
                last_error = RuntimeError(f"{model}이(가) 빈 응답을 반환함")
                self._logger.warning(str(last_error))
                continue
            return response, model
        raise last_error if last_error else RuntimeError("no counting model configured")
