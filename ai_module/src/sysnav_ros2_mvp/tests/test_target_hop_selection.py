"""A* 경로 hop 중 "지금 발행 가능한 가장 먼 hop"을 고르는 규칙.

목적지를 한 번에 던지면 waypointConverter가 로봇 발밑으로 덤프해버린다(실측 80%).
그래서 A* 경로를 1.5m hop으로 잘라 순차 발행하는데, 매번 hop 하나씩만 가면 명령이
과하게 잦다. terrain이 허용하는 만큼 멀리 보내되, **직선거리로 목표에 가장 가까운
점**을 고르면 벽 뒤에 목표가 있을 때 벽 앞에서 갇히므로(local minimum) 반드시 A*
경로 위에서만 골라야 한다.

sysnav_node는 rclpy에 묶여 있어 여기서는 같은 선택 규칙을 최소 구현으로 고정한다.
"""

import unittest
from collections import deque


def _pick_farthest_publishable(route: deque, can_publish, publish) -> int | None:
    """sysnav_node.publish_next_target_hop()과 동일한 계약.
    반환: 사용한 hop의 인덱스(없으면 None). route에서 그 hop까지를 소비한다."""
    hops = list(route)
    for index in range(len(hops) - 1, -1, -1):
        hop = hops[index]
        if not can_publish(hop):
            continue
        if publish(hop, index == len(hops) - 1):
            for _ in range(index + 1):
                route.popleft()
            return index
    return None


def _hop(x):
    return {"x": float(x), "y": 0.0, "theta": 0.0}


class FarthestPublishableHopTest(unittest.TestCase):
    def setUp(self):
        self.published = []
        self.finals = []

    def _publish(self, hop, is_final):
        self.published.append(hop["x"])
        self.finals.append(is_final)
        return True

    def test_picks_the_farthest_hop_that_can_be_published(self):
        route = deque([_hop(1.5), _hop(3.0), _hop(4.5), _hop(6.0)])
        ok = lambda hop: hop["x"] <= 3.0
        index = _pick_farthest_publishable(route, ok, self._publish)
        self.assertEqual(index, 1)
        self.assertEqual(self.published, [3.0], "가장 먼 발행 가능 hop을 써야 한다")
        self.assertEqual(len(route), 2, "쓴 hop까지는 경로에서 소비돼야 한다")

    def test_marks_is_final_only_for_the_last_hop(self):
        route = deque([_hop(1.5), _hop(3.0)])
        _pick_farthest_publishable(route, lambda hop: hop["x"] <= 3.0, self._publish)
        self.assertEqual(self.published, [3.0])
        self.assertEqual(self.finals, [True], "마지막 hop이면 is_final=True")

        self.published.clear()
        self.finals.clear()
        route = deque([_hop(1.5), _hop(3.0), _hop(4.5)])
        _pick_farthest_publishable(route, lambda hop: hop["x"] <= 3.0, self._publish)
        self.assertEqual(self.finals, [False], "뒤에 hop이 남으면 is_final=False")

    def test_returns_none_when_no_hop_is_publishable(self):
        route = deque([_hop(1.5), _hop(3.0)])
        index = _pick_farthest_publishable(route, lambda hop: False, self._publish)
        self.assertIsNone(index)
        self.assertEqual(self.published, [], "발행 불가면 아무것도 안 내보낸다")
        self.assertEqual(len(route), 2, "경로를 소비하지 않아야 재계획이 가능하다")

    def test_near_hop_is_used_when_only_it_is_publishable(self):
        """terrain이 로봇 주변만 유효할 때의 일반적인 상황."""
        route = deque([_hop(1.5), _hop(3.0), _hop(4.5)])
        index = _pick_farthest_publishable(
            route, lambda hop: hop["x"] <= 1.5, self._publish
        )
        self.assertEqual(index, 0)
        self.assertEqual(self.published, [1.5])
        self.assertEqual(len(route), 2)


if __name__ == "__main__":
    unittest.main()
