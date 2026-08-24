"""로봇이 "지금 무엇을 하고 있는지"를 시간순으로 남기는 링 버퍼.

mission_dashboard.py가 이걸 읽어 실시간 로그 패널을 그린다. 기존 디버그 수단으로는
"왜 멈췄는지"를 알기 어려웠다:

  - `sysnav_navigation_trace.txt`는 주행 이벤트만 있고 상태 전이/LLM 질의가 없다
  - ROS 로그는 터미널에 흘러가버리고 대시보드에서 못 본다
  - 대시보드는 "현재 상태" 스냅샷만 있어서 그 상태에 어떻게 도달했는지가 안 보인다

그래서 상태 전이 / 작업(job) 시작·종료 / LLM 질의 / 주행 판단을 한 곳에 모은다.

스레드 안전: 이벤트는 ROS 콜백 스레드와 worker 스레드 양쪽에서 들어온다.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager

# 카테고리 - 대시보드에서 색으로 구분한다.
STATE = "state"   # 상태 머신 전이
JOB = "job"       # 백그라운드 작업(perception/selection/exploration ...)
LLM = "llm"       # Gemini 등 외부 모델 질의
NAV = "nav"       # 주행 판단(목표 발행/스냅/포기 등)
# OBSERVE 한 사이클 안에서 실제로 무슨 일이 일어나는지(검출 -> 재확인 -> 분할 ->
# 3D 확정 -> 메모리/그래프 반영). 예전에는 "인식 작업 시작/완료 18.1초"만 보여서
# 그 18초 동안 어디서 시간을 쓰는지, 무엇이 몇 개 잡혔는지 알 수 없었다.
PERCEPTION = "percep"
WARN = "warn"     # 문제 상황

_CAPACITY = 300


class ActivityLog:
    def __init__(self, capacity: int = _CAPACITY) -> None:
        self._lock = threading.Lock()
        self._events: deque[dict] = deque(maxlen=capacity)
        self._inflight: dict[int, dict] = {}
        self._next_id = 0

    # ------------------------------------------------------------------

    def add(self, category: str, message: str, detail: str = "") -> None:
        with self._lock:
            self._events.append({
                "time": time.time(),
                "monotonic": time.monotonic(),
                "category": category,
                "message": message,
                "detail": detail,
            })

    @contextmanager
    def operation(self, category: str, label: str):
        """오래 걸리는 작업을 감싼다. 진행 중에는 in-flight로 노출되고(대시보드의
        "지금 하는 일"), 끝나면 소요 시간과 함께 이벤트로 남는다.

        예외가 나도 반드시 in-flight에서 빠져야 한다 - 안 그러면 대시보드가 영원히
        "질의 중"으로 남는다. 그래서 finally에서 정리한다."""
        with self._lock:
            handle = self._next_id
            self._next_id += 1
            self._inflight[handle] = {
                "category": category, "label": label, "started": time.monotonic(),
            }
        started = time.monotonic()
        self.add(category, f"{label} 시작")
        failure = None
        try:
            yield
        except BaseException as error:            # 실패도 기록해야 원인이 보인다
            failure = error
            raise
        finally:
            with self._lock:
                self._inflight.pop(handle, None)
            elapsed = time.monotonic() - started
            if failure is None:
                self.add(category, f"{label} 완료", f"{elapsed:.1f}초")
            else:
                self.add(WARN, f"{label} 실패",
                         f"{elapsed:.1f}초 - {type(failure).__name__}: {failure}")

    # ------------------------------------------------------------------

    def inflight(self) -> list[dict]:
        """지금 진행 중인 장기 작업들 (오래된 것부터)."""
        now = time.monotonic()
        with self._lock:
            items = list(self._inflight.values())
        items.sort(key=lambda item: item["started"])
        return [
            {"category": item["category"], "label": item["label"],
             "elapsed_sec": now - item["started"]}
            for item in items
        ]

    def recent(self, limit: int = 40) -> list[dict]:
        """최신이 먼저 오도록 뒤집어서 돌려준다 (대시보드가 위에서부터 읽는다)."""
        with self._lock:
            events = list(self._events)
        return list(reversed(events[-limit:]))

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._inflight.clear()


# 모듈 전역 인스턴스. 노드/추론 모듈 어디서나 import해서 쓴다 - 인자로 넘기려면
# LLM 호출부까지 배선을 다 바꿔야 해서, 로깅 같은 횡단 관심사는 전역으로 둔다.
activity = ActivityLog()
