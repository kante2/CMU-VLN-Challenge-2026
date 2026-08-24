"""LLM 판단 기록(llm_trace)과 그 대시보드 패널.

활동 로그는 "Gemini 대상 선택 완료 3.1초"까지만 남아서, 모델이 무슨 사진을 보고 왜
그 답을 냈는지는 대시보드에서 볼 수 없었다. llm_trace는 실제로 올린 이미지를
DEBUG_DIR/llm_trace/에 저장하고 항목별 판정과 묶어 보관한다.

여기서 고정하는 성질:
  - 디스크가 무한정 늘지 않는다(링 버퍼에서 밀려난 레코드의 파일은 같이 지워진다)
  - 기록 실패가 추론을 죽이지 않는다(예외를 밖으로 안 던진다)
  - 패널 렌더링이 어떤 레코드 모양에서도 안 죽고, 판정 3상태를 구분해 보여준다
"""

import tempfile
import unittest

import numpy as np

from sysnav import config
from sysnav.llm_trace import LLMTrace
from sysnav.mission_dashboard import _llm_trace_card, _verdict_chip


def _image():
    return np.zeros((8, 8, 3), dtype=np.uint8)


class LLMTraceTest(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self._previous = config.DEBUG_DIR
        config.DEBUG_DIR = self._directory.name
        self.trace = LLMTrace(capacity=2)

    def tearDown(self):
        config.DEBUG_DIR = self._previous
        self._directory.cleanup()

    def test_records_image_and_verdicts(self):
        self.trace.record(
            kind="대상 선택", question="the picture closest to the TV",
            images=[("picture#6 isolated", _image())],
            verdicts=[("picture#6", True, "matches")], summary="선택 object_id=6",
        )
        record = self.trace.recent()[0]
        self.assertEqual(record["kind"], "대상 선택")
        self.assertEqual(record["verdicts"][0]["reason"], "matches")
        # 경로는 DEBUG_DIR 기준 상대경로여야 한다 - 대시보드가 <img src>에 그대로 쓴다.
        self.assertTrue(record["images"][0]["path"].startswith("llm_trace/"))
        self.assertTrue((self.trace.directory / "00001_0.jpg").exists())

    def test_evicted_records_delete_their_images(self):
        for _ in range(4):
            self.trace.record(kind="속성 검증", images=[("c", _image())])
        self.assertEqual(len(self.trace.recent()), 2)
        self.assertEqual(len(list(self.trace.directory.glob("*.jpg"))), 2)

    def test_reset_clears_previous_run(self):
        self.trace.record(kind="검출 재확인", images=[("c", _image())])
        self.trace.reset()
        self.assertEqual(self.trace.recent(), [])
        self.assertEqual(list(self.trace.directory.glob("*.jpg")), [])

    def test_bad_image_does_not_raise(self):
        """기록은 어디까지나 디버그 수단이다 - 여기서 예외가 나면 추론이 죽는다."""
        self.trace.record(kind="속성 검증", images=[("c", "not an image")])
        self.assertEqual(self.trace.recent()[0]["images"], [])


class LLMTracePanelTest(unittest.TestCase):
    def test_verdict_chip_distinguishes_three_states(self):
        # "거짓"과 "아직 확인 안 됨"은 의미가 전혀 다르다(fail-closed 경로의 핵심).
        self.assertIn("TRUE", _verdict_chip(True))
        self.assertIn("false", _verdict_chip(False))
        self.assertIn("미확인", _verdict_chip(None))

    def test_card_renders_without_images(self):
        card = _llm_trace_card({
            "time": 0, "kind": "질문 파싱", "question": "", "summary": "",
            "images": [], "verdicts": [],
        })
        self.assertIn("이미지 없음", card)

    def test_card_escapes_untrusted_text(self):
        card = _llm_trace_card({
            "time": 0, "kind": "대상 선택", "question": "<script>", "summary": "",
            "images": [], "verdicts": [{"label": "<b>", "verdict": True, "reason": "<i>"}],
        })
        self.assertNotIn("<script>", card)
        self.assertIn("&lt;script&gt;", card)


if __name__ == "__main__":
    unittest.main()
