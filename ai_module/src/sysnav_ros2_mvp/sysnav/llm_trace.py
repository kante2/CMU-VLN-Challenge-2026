"""LLM이 "어떤 이미지를 보고 무슨 판단을 했는지"를 남기는 링 버퍼.

activity_log.py는 "Gemini 대상 선택 시작/완료 3.1초"처럼 **호출이 있었다는 사실**만
남긴다. 그래서 대시보드만 보면 그 3.1초 동안 모델이 무슨 사진을 봤고 왜 그 답을
냈는지는 알 수 없었다(공간관계 판정만 예외적으로 debug/*.jpg +
sysnav_relation_check.txt로 따로 남고 있었고, 나머지 이미지 질의 - 대상 선택/속성
검증/관계 이미지 검증/검출 재확인 - 는 아무 흔적도 안 남았다).

이 모듈은 각 이미지 질의마다
  (1) **실제로 업로드한 이미지 그대로**를 config.DEBUG_DIR/llm_trace/ 에 JPEG로 저장하고
  (2) 그 질의의 판정 결과(항목별 verdict + 모델이 준 reason)를
한 레코드로 묶어 보관한다. mission_dashboard.py가 이걸 읽어 썸네일과 판정을 나란히
그린다 - 이미지는 대시보드와 같은 폴더 밑이라 상대경로로 걸리고, 서버 없이
file:// 로 열어도 그대로 보인다(_map_images_panel과 같은 방식).

디스크는 링 버퍼로 제한한다: 레코드가 밀려나면 그 레코드가 저장했던 파일도 같이
지운다. 미션 한 판이 몇십 분씩 돌아도 llm_trace/ 폴더 크기가 무한정 늘지 않는다.

스레드 안전: 기록은 perception/selection worker 스레드 양쪽에서 들어오고, 읽기는
대시보드를 그리는 타이머 콜백 스레드에서 일어난다.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from sysnav import config

_CAPACITY = 12          # 보관할 질의 개수(= 대시보드에 뜨는 카드 수)
_MAX_IMAGES = 8         # 한 질의에서 저장할 이미지 상한(후보가 많아도 디스크 폭주 방지)
_THUMB_MAX_WIDTH = 720  # 원본 그대로 두면 파노라마 한 장이 수백 KB라 폭이 이만큼이면 충분
_SUBDIR = "llm_trace"


class LLMTrace:
    def __init__(self, capacity: int = _CAPACITY) -> None:
        self._lock = threading.Lock()
        self._records: deque[dict] = deque()
        self._capacity = capacity
        self._sequence = 0

    # ------------------------------------------------------------------

    @property
    def directory(self) -> Path:
        return Path(config.DEBUG_DIR) / _SUBDIR

    def _save_image(self, image_rgb: np.ndarray, name: str) -> str | None:
        """저장 성공하면 config.DEBUG_DIR 기준 상대경로(대시보드 <img src>용)."""
        if not isinstance(image_rgb, np.ndarray) or not image_rgb.size:
            return None
        image = image_rgb
        if image.ndim == 3 and image.shape[1] > _THUMB_MAX_WIDTH:
            scale = _THUMB_MAX_WIDTH / float(image.shape[1])
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        path = self.directory / name
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 80]):
            return None
        os.chmod(path, 0o644)   # 컨테이너(uid 1001)가 쓴 파일을 호스트 브라우저가 읽는다
        return f"{_SUBDIR}/{name}"

    def record(
        self,
        kind: str,
        question: str = "",
        images: list[tuple[str, np.ndarray]] | None = None,
        verdicts: list[tuple[str, bool | None, str]] | None = None,
        summary: str = "",
    ) -> None:
        """kind: 질의 종류("대상 선택" 등). images: (캡션, RGB 이미지) - 실제로 모델에
        올린 그 이미지를 그대로 넘긴다. verdicts: (항목 라벨, 판정, 근거) - 판정이
        None이면 "확인 안 됨". summary: 한 줄 결론(선택된 id, 실패 사유 등).

        기록 실패가 추론을 죽이면 안 되므로 어떤 예외도 밖으로 내보내지 않는다."""
        if not config.SAVE_DEBUG_IMAGES:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with self._lock:
                self._sequence += 1
                sequence = self._sequence
            saved: list[dict] = []
            for index, (caption, image) in enumerate((images or [])[:_MAX_IMAGES]):
                relative = self._save_image(image, f"{sequence:05d}_{index}.jpg")
                if relative:
                    saved.append({"caption": caption, "path": relative})
            record = {
                "time": time.time(),
                "kind": kind,
                "question": question,
                "summary": summary,
                "images": saved,
                "verdicts": [
                    {"label": label, "verdict": verdict, "reason": reason}
                    for label, verdict, reason in (verdicts or [])
                ],
            }
            evicted: list[dict] = []
            with self._lock:
                self._records.append(record)
                while len(self._records) > self._capacity:
                    evicted.append(self._records.popleft())
            for old in evicted:
                for item in old["images"]:
                    try:
                        (Path(config.DEBUG_DIR) / item["path"]).unlink()
                    except OSError:
                        pass
        except Exception as error:  # pragma: no cover - 디버그 기록은 절대 추론을 죽이면 안 된다
            print(f"[llm_trace] failed to record {kind}: {error}")

    def recent(self, limit: int = _CAPACITY) -> list[dict]:
        """최신이 먼저 오도록 뒤집어서 돌려준다(대시보드가 위에서부터 읽는다)."""
        with self._lock:
            records = list(self._records)
        return list(reversed(records[-limit:]))

    def reset(self) -> None:
        """노드가 새로 뜰 때 이전 실행이 남긴 이미지를 지운다 - 안 그러면 llm_trace/
        폴더에 지난 실행 파일들이 계속 남는다(레코드는 메모리라 같이 안 지워진다)."""
        if not config.SAVE_DEBUG_IMAGES:
            return
        with self._lock:
            self._records.clear()
            self._sequence = 0
        try:
            for path in self.directory.glob("*.jpg"):
                path.unlink()
        except OSError as error:
            print(f"[llm_trace] failed to reset trace directory: {error}")


# 모듈 전역 인스턴스 - activity_log.activity와 같은 이유로 전역이다(LLM 호출부까지
# 인자로 배선하지 않는다).
llm_trace = LLMTrace()
