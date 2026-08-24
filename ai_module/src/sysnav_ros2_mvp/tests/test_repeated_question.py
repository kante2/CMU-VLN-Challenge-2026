"""채점 환경은 /challenge_question을 1Hz로 계속 발행한다(`--once` 없이).

그대로 받으면 매 초 Gemini 파싱을 다시 돌리고 task_id를 올리며 object_memory /
scene_graph / coverage_planner를 통째로 초기화해서, 로봇이 첫 관측 상태를 영원히
벗어나지 못한다. _claim_question()이 파싱 **전에** 그 중복을 끊는다.

파싱 전에 끊는 것이 핵심이다: 구독 콜백이 ReentrantCallbackGroup이라, 파싱이
2~14초 걸리는 동안 들어온 중복 메시지가 다른 executor 스레드에서 동시에 파싱을
시작해버린다.
"""

import sys
import threading
import time
import types
import unittest

if "rclpy" not in sys.modules:
    try:
        import rclpy  # noqa: F401
    except ImportError:                                       # pragma: no cover
        package = types.ModuleType("rclpy")
        logging_module = types.ModuleType("rclpy.logging")

        class _RclpyLogger:
            def info(self, *a, **k): pass
            def warning(self, *a, **k): pass
            def error(self, *a, **k): pass

        logging_module.get_logger = lambda name: _RclpyLogger()
        package.logging = logging_module
        sys.modules["rclpy"] = package
        sys.modules["rclpy.logging"] = logging_module

from sysnav import config                                      # noqa: E402


class _Logger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass


class _FakeNode:
    """SysNavNode에서 _claim_question이 실제로 쓰는 필드만 가진 최소 대역.

    SysNavNode 자체는 rclpy.Node를 상속해서 ROS 없이는 생성할 수 없으므로,
    메서드만 떼어 붙여 순수 로직을 검증한다.
    """

    from sysnav.sysnav_node import SysNavNode as _Real  # noqa: E402
    _claim_question = _Real._claim_question

    def __init__(self):
        self.state_lock = threading.RLock()
        self.task_id = 0
        self._accepted_question = None
        self._accepted_question_at = 0.0
        self._accepted_question_ok = False
        self._duplicate_question_count = 0
        self._last_duplicate_log = 0.0

    def get_logger(self):
        return _Logger()

    def accept(self):
        """파싱까지 성공해 task가 된 상태를 흉내낸다."""
        self._accepted_question_ok = True
        self._duplicate_question_count = 0


_Q = "How many pictures are above the bed?"


class RepeatedQuestionTest(unittest.TestCase):
    def test_first_message_is_claimed(self):
        node = _FakeNode()
        self.assertTrue(node._claim_question(_Q))

    def test_repeats_of_an_accepted_question_are_dropped(self):
        """1Hz로 600번 들어와도 task는 한 번만 만들어져야 한다."""
        node = _FakeNode()
        self.assertTrue(node._claim_question(_Q))
        node.accept()
        claimed = sum(1 for _ in range(600) if node._claim_question(_Q))
        self.assertEqual(claimed, 0)
        self.assertEqual(node._duplicate_question_count, 600)

    def test_whitespace_only_differences_are_still_duplicates(self):
        """발행 쪽 포맷팅 차이로 같은 질문이 새 task가 되면 지도가 초기화된다."""
        from sysnav.sysnav_node import normalize_question
        node = _FakeNode()
        node._claim_question(normalize_question(_Q))
        node.accept()
        messy = "  How many pictures   are above the bed?\n"
        self.assertEqual(normalize_question(messy), _Q)
        self.assertFalse(node._claim_question(normalize_question(messy)))

    def test_a_genuinely_new_question_is_claimed(self):
        node = _FakeNode()
        node._claim_question(_Q)
        node.accept()
        self.assertTrue(node._claim_question("Find the table"))

    def test_a_failed_parse_is_retried_after_the_backoff(self):
        """파싱 실패 문장을 영영 막으면 복구가 안 된다 - 간격 뒤엔 다시 받아준다."""
        node = _FakeNode()
        self.assertTrue(node._claim_question(_Q))       # 선점만 하고 accept() 안 함
        self.assertFalse(node._claim_question(_Q))      # 간격 전에는 막힘
        node._accepted_question_at -= config.QUESTION_REPARSE_RETRY_SEC + 1.0
        self.assertTrue(node._claim_question(_Q))       # 간격 뒤에는 재시도

    def test_an_accepted_question_is_never_retried(self):
        """정상 접수된 문장은 재시도 간격과 무관하게 계속 무시한다."""
        node = _FakeNode()
        node._claim_question(_Q)
        node.accept()
        node._accepted_question_at -= config.QUESTION_REPARSE_RETRY_SEC * 10
        self.assertFalse(node._claim_question(_Q))

    def test_concurrent_duplicates_claim_exactly_once(self):
        """파싱이 느린 동안 여러 executor 스레드가 동시에 들어와도 한 번만 통과."""
        node = _FakeNode()
        results = []
        lock = threading.Lock()

        def worker():
            ok = node._claim_question(_Q)
            with lock:
                results.append(ok)

        threads = [threading.Thread(target=worker) for _ in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sum(results), 1)


if __name__ == "__main__":
    unittest.main()
