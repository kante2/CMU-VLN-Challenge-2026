"""스냅이 전진을 통째로 삼킨 목표는 발행하지 않는다.

TERRAIN_SNAP_MAX_M(1.5m)이 GOAL_REACHED_DISTANCE_M(0.5m)보다 훨씬 커서, 1.5m 앞 hop이
로봇 발밑으로 옮겨질 수 있다. 그대로 발행하면 즉시 "도착" 처리 -> 재계획 -> 같은 방향 ->
또 되끌림으로 제자리에서 돈다.

실측(2026-08-22 실행 로그): 요청 (4.50,1.50)이 (3.22,1.58)로 되끌렸는데 로봇(3.27,1.84)
에서 0.26m라 도착 반경 안이었다. 탐색 goal 329건 중 40%가 1m 넘게 되끌리고 있었다.
"""

import math
import unittest

from sysnav import config


def _snap_gives_progress(robot, requested, snapped) -> bool:
    """goal_publisher.resolve()의 판정과 동일한 계약:
    원래는 도착 반경 밖이었는데 스냅 후 안으로 들어오면 전진이 없다."""
    before = math.dist(robot, requested)
    after = math.dist(robot, snapped)
    return not (before > config.GOAL_REACHED_DISTANCE_M >= after)


class SnapProgressTest(unittest.TestCase):
    def test_real_stall_case_is_rejected(self):
        """실제로 제자리 순환을 만들었던 좌표 그대로."""
        robot = (3.27, 1.84)
        self.assertFalse(_snap_gives_progress(robot, (4.50, 1.50), (3.22, 1.58)))

    def test_small_snap_still_publishes(self):
        """되끌림이 작아 여전히 갈 거리가 남으면 정상 발행."""
        robot = (3.27, 1.84)
        self.assertTrue(_snap_gives_progress(robot, (4.30, 2.90), (3.23, 2.72)))

    def test_goal_that_was_already_close_is_not_rejected(self):
        """원래부터 도착 반경 안인 목표(마지막 접근 등)는 거르면 안 된다 -
        거르면 물체 앞 최종 접근이 영원히 발행되지 않는다."""
        robot = (3.27, 1.84)
        near = (3.30, 1.90)                      # 0.07m
        self.assertLess(math.dist(robot, near), config.GOAL_REACHED_DISTANCE_M)
        self.assertTrue(_snap_gives_progress(robot, near, near))

    def test_snap_landing_exactly_at_arrival_radius_is_rejected(self):
        robot = (0.0, 0.0)
        self.assertFalse(_snap_gives_progress(robot, (2.0, 0.0),
                                              (config.GOAL_REACHED_DISTANCE_M, 0.0)))

    def test_parameters_still_make_the_guard_necessary(self):
        """스냅 상한이 도착 반경보다 크면 이 방어가 필요하다. 두 값을 나중에 조정하면
        이 테스트가 전제를 다시 확인해준다."""
        self.assertGreater(config.TERRAIN_SNAP_MAX_M, config.GOAL_REACHED_DISTANCE_M)


if __name__ == "__main__":
    unittest.main()
