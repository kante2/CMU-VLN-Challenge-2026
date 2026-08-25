"""탐사 소진 판정은 횟수만이 아니라 **시간**도 봐야 한다.

경로 계획이 0.2초라 EXPLORATION_UNPUBLISHABLE_ROUTE_LIMIT(5)이 1~2초 만에 소진된다.
그 사이 로봇은 한 번도 안 움직였고 지도도 그대로라 5번 다 같은 결과가 나오는 게 당연하다 -
라이브락을 감지한 게 아니라 같은 계산을 5번 반복한 것뿐이다. 실측 2026-08-25: 0:40에
FAILED가 떴고 10분 예산 중 9분 20초가 남아 있었다.

sysnav_node는 rclpy에 강하게 묶여 있어 import하지 않고, 같은 판정 계약을 최소 구현으로
고정한다(tests/test_goal_publish_skip.py와 같은 패턴).
"""

import unittest

from sysnav import config


class _Loop:
    """publish_next_exploration_goal()의 스트릭/시간 판정 부분과 동일한 계약."""

    def __init__(self):
        self.streak = 0
        self.started: float | None = None
        self.now = 0.0
        self.exhausted = False

    def all_hops_rejected(self):
        self.streak += 1
        if self.started is None:
            self.started = self.now
        stuck_for = self.now - self.started
        if (
            self.streak >= config.EXPLORATION_UNPUBLISHABLE_ROUTE_LIMIT
            and stuck_for >= config.EXPLORATION_UNPUBLISHABLE_MIN_SEC
        ):
            self.exhausted = True
            self.streak = 0
            self.started = None

    def published(self):
        self.streak = 0
        self.started = None


class ExhaustionTimingTest(unittest.TestCase):
    def setUp(self):
        self.loop = _Loop()

    def test_the_count_alone_does_not_exhaust(self):
        """보고된 버그. 0.2초짜리 계획 20번은 4초일 뿐이다."""
        for _ in range(config.EXPLORATION_UNPUBLISHABLE_ROUTE_LIMIT * 4):
            self.loop.now += 0.2
            self.loop.all_hops_rejected()
        self.assertFalse(self.loop.exhausted)

    def test_the_time_alone_does_not_exhaust(self):
        """오래 걸렸어도 거부가 한 번뿐이면 라이브락이 아니다."""
        self.loop.now += config.EXPLORATION_UNPUBLISHABLE_MIN_SEC * 3
        self.loop.all_hops_rejected()
        self.assertFalse(self.loop.exhausted)

    def test_both_conditions_exhaust(self):
        for _ in range(config.EXPLORATION_UNPUBLISHABLE_ROUTE_LIMIT):
            self.loop.now += config.EXPLORATION_UNPUBLISHABLE_MIN_SEC
            self.loop.all_hops_rejected()
        self.assertTrue(self.loop.exhausted)

    def test_a_successful_publish_resets_both(self):
        """중간에 한 번이라도 움직였으면 그때부터 다시 재야 한다."""
        for _ in range(config.EXPLORATION_UNPUBLISHABLE_ROUTE_LIMIT - 1):
            self.loop.now += config.EXPLORATION_UNPUBLISHABLE_MIN_SEC
            self.loop.all_hops_rejected()
        self.loop.published()
        self.assertEqual(self.loop.streak, 0)
        self.assertIsNone(self.loop.started)

        self.loop.now += config.EXPLORATION_UNPUBLISHABLE_MIN_SEC * 10
        self.loop.all_hops_rejected()
        self.assertFalse(self.loop.exhausted, "리셋 뒤엔 스트릭을 처음부터 다시 쌓아야 한다")

    def test_the_node_reads_both_config_values(self):
        """실제 노드 코드가 두 조건을 AND로 묶었는지 - 소스에서 확인한다."""
        import pathlib
        source = (pathlib.Path(__file__).resolve().parents[1]
                  / "sysnav" / "sysnav_node.py").read_text()
        body = source.split("def publish_next_exploration_goal", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("EXPLORATION_UNPUBLISHABLE_ROUTE_LIMIT", body)
        self.assertIn("EXPLORATION_UNPUBLISHABLE_MIN_SEC", body)
        self.assertIn("_unpublishable_streak_started", body)

    def test_every_streak_reset_also_clears_the_timestamp(self):
        """한 곳이라도 빠뜨리면 옛 시각이 남아 즉시 소진 처리된다."""
        import pathlib
        source = (pathlib.Path(__file__).resolve().parents[1]
                  / "sysnav" / "sysnav_node.py").read_text()
        import re
        # __init__은 타입 주석이 붙으므로(`: float | None = None`) 두 형태를 모두 센다.
        resets = len(re.findall(r"self\._unpublishable_route_streak\s*(?::[^=]+)?=\s*0", source))
        clears = len(re.findall(r"self\._unpublishable_streak_started\s*(?::[^=]+)?=\s*None", source))
        self.assertEqual(clears, resets, "streak을 0으로 되돌리는 곳마다 시작 시각도 지워야 한다")


if __name__ == "__main__":
    unittest.main()
