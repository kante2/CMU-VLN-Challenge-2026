"""발행 불가한 목표를 만났을 때의 흐름 (Layer 1 - fail-open 제거).

base autonomy가 받아줄 지점이 근처에 없으면 GoalPublisher.publish()가 None을 돌려주고,
호출 측은 **다음 후보로 넘어가야 한다**. 예전처럼 원본을 그대로 발행하면 waypointConverter가
그 목표를 로봇 발밑에 떨어뜨려서(실측: 로봇에서 0.31m) 로봇이 서 있고, stuck timeout
8~20초를 통째로 버린 뒤에야 다음으로 넘어갔다.

sysnav_node는 rclpy에 강하게 묶여 있어 여기서는 import하지 않고, 같은 제어 흐름을
가진 최소 구현으로 계약만 고정한다.
"""

import unittest
from collections import deque


class _Publisher:
    """publish()가 어떤 좌표에 대해 None을 돌려주는지 흉내낸다."""

    def __init__(self, unpublishable: set[tuple[float, float]]):
        self.unpublishable = unpublishable
        self.published: list[tuple[float, float]] = []

    def publish(self, x, y, theta, label="goal"):
        if (x, y) in self.unpublishable:
            return None
        self.published.append((x, y))
        return type("Pose2D", (), {"x": x, "y": y, "theta": theta})()


def _next_goal(route: deque, publisher: _Publisher):
    """sysnav_node.publish_next_exploration_goal()의 hop 선택 흐름과 동일한 계약:
    발행 못 한 hop은 건너뛰고 다음을 시도하며, 전부 실패하면 (None, skipped)."""
    skipped = 0
    while route:
        candidate = route.popleft()
        published = publisher.publish(candidate["x"], candidate["y"], candidate["theta"])
        if published is not None:
            return {**candidate, "x": published.x, "y": published.y}, skipped
        skipped += 1
    return None, skipped


def _hop(x, y):
    return {"x": x, "y": y, "theta": 0.0}


class ExplorationSkipTest(unittest.TestCase):
    def test_unpublishable_hop_is_skipped_not_sent(self):
        """발행 불가 좌표는 아예 안 내보낸다 - 내보내면 로봇이 멈춘다."""
        publisher = _Publisher(unpublishable={(1.0, 1.0), (2.0, 2.0)})
        route = deque([_hop(1.0, 1.0), _hop(2.0, 2.0), _hop(3.0, 3.0)])

        goal, skipped = _next_goal(route, publisher)

        self.assertEqual(skipped, 2)
        self.assertEqual((goal["x"], goal["y"]), (3.0, 3.0))
        self.assertEqual(publisher.published, [(3.0, 3.0)])

    def test_route_of_all_unpublishable_hops_yields_no_goal(self):
        """전부 실패하면 아무것도 발행하지 않고 호출 측에 알린다(재관측/재계획)."""
        publisher = _Publisher(unpublishable={(1.0, 1.0), (2.0, 2.0)})
        route = deque([_hop(1.0, 1.0), _hop(2.0, 2.0)])

        goal, skipped = _next_goal(route, publisher)

        self.assertIsNone(goal)
        self.assertEqual(skipped, 2)
        self.assertEqual(publisher.published, [])

    def test_first_publishable_hop_is_used_as_is(self):
        publisher = _Publisher(unpublishable=set())
        route = deque([_hop(1.0, 1.0), _hop(2.0, 2.0)])

        goal, skipped = _next_goal(route, publisher)

        self.assertEqual(skipped, 0)
        self.assertEqual((goal["x"], goal["y"]), (1.0, 1.0))
        self.assertEqual(len(route), 1, "쓰지 않은 hop은 route에 남아 있어야 한다")

    def test_goal_carries_published_coordinates_not_requested(self):
        """스냅으로 좌표가 옮겨지면 current_goal도 그 좌표여야 도착 판정이 맞는다."""
        class _Snapping(_Publisher):
            def publish(self, x, y, theta, label="goal"):
                self.published.append((x + 0.4, y))
                return type("Pose2D", (), {"x": x + 0.4, "y": y, "theta": theta})()

        publisher = _Snapping(unpublishable=set())
        goal, _ = _next_goal(deque([_hop(1.0, 1.0)]), publisher)
        self.assertEqual((goal["x"], goal["y"]), (1.4, 1.0))


if __name__ == "__main__":
    unittest.main()
