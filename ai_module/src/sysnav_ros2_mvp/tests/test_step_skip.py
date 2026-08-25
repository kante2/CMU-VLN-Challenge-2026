"""풀 수 없는 step은 FAILED가 아니라 skip하고 다음으로 간다.

실측 2026-08-25: step 1·2를 실제로 주행하고도 step 3(curtain)에서 0:40에 FAILED가 떴다.
10분 예산 중 9분 20초가 남아 있었고, 이미 간 2개 step까지 빨간불이 됐다. 채점은 실제
궤적을 보고 부분점수가 있으므로 남은 step이라도 시도하는 쪽이 항상 낫다 - 이 파일의
다른 폴백(_give_up_step, forbidden mask 미발견, _best_effort_step_target)과 같은 원칙이다.

다만 "지나갔다"와 "못 갔지만 넘어갔다"는 화면과 요약에서 구분돼야 한다.
"""

import threading
import unittest

from sysnav.mission_dashboard import _plan_list_html, _mission_detail_rows
from sysnav.missions import mission3_pipe


def _step(is_stop=True, **extra):
    step = {
        "is_stop": is_stop,
        "resolve": "category",
        "parsed": {"target": "curtain", "attributes": [], "relation_chain": []},
    }
    step.update(extra)
    return step


class _Logger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def info(self, message):
        self.infos.append(message)

    def warning(self, message):
        self.warnings.append(message)

    def error(self, _message):
        pass


class _Planner:
    @staticmethod
    def describe_last_plan_failure():
        return "no frontier"


class _Node:
    def __init__(self, task, index=0, state="PLAN_EXPLORATION"):
        self.state_lock = threading.RLock()
        self.task = task
        self.mission3_step_index = index
        self.mission3_subgoal_retries = 3
        self.mission3_exploration_exhausted = True
        self.mission3_forbidden_mask = None
        self.state = state
        self.coverage_planner = _Planner()
        self.logger = _Logger()
        self.cleared = 0

    def clear_target_navigation(self):
        self.cleared += 1

    def get_logger(self):
        return self.logger


class SkipInsteadOfFailTest(unittest.TestCase):
    def test_a_remaining_step_is_skipped_not_failed(self):
        """보고된 버그. 남은 step이 있으면 FAILED로 끝내지 않는다."""
        task = {"steps": [_step(), _step(), _step()], "global_forbidden": []}
        node = _Node(task, index=2)
        mission3_pipe._on_exploration_result(node, {"route": []})
        self.assertNotEqual(node.state, "FAILED")
        self.assertEqual(node.state, "MISSION3_SELECT_STEP")
        self.assertEqual(node.mission3_step_index, 3)
        self.assertTrue(task["steps"][2]["skipped"])
        self.assertIn("SKIPPED", task["steps"][2]["basis"])

    def test_failing_only_when_no_step_remains(self):
        task = {"steps": [_step()], "global_forbidden": []}
        node = _Node(task, index=1)                      # 이미 마지막 step을 지났다
        mission3_pipe._on_exploration_result(node, {"route": []})
        self.assertEqual(node.state, "FAILED")

    def test_the_first_exhaustion_still_goes_back_to_select(self):
        """탐사 소진이 처음이면 예전대로 '지금 아는 것으로 결정'으로 돌아간다."""
        task = {"steps": [_step()], "global_forbidden": []}
        node = _Node(task, index=0)
        node.mission3_exploration_exhausted = False
        mission3_pipe._on_exploration_result(node, {"route": []})
        self.assertEqual(node.state, "MISSION3_SELECT_STEP")
        self.assertNotIn("skipped", task["steps"][0])


class TerminalSummaryTest(unittest.TestCase):
    def _finish(self, steps):
        task = {"steps": steps, "global_forbidden": []}
        node = _Node(task, index=len(steps), state="MISSION3_SELECT_STEP")
        mission3_pipe._select_step(node, task, task_id=1, pose={"x": 0, "y": 0, "yaw": 0})
        return node

    def test_reaching_some_steps_is_success_with_the_count(self):
        node = self._finish([_step(), _step(skipped=True, basis="SKIPPED - x")])
        self.assertEqual(node.state, "SUCCESS")
        self.assertIn("1/2 steps reached", node.last_response_summary)
        self.assertIn("1 skipped", node.last_response_summary)

    def test_reaching_nothing_is_a_failure(self):
        node = self._finish([_step(skipped=True, basis="SKIPPED - x")])
        self.assertEqual(node.state, "FAILED")
        self.assertIn("0/1 steps reached", node.last_response_summary)

    def test_a_clean_run_says_so(self):
        node = self._finish([_step(), _step()])
        self.assertEqual(node.state, "SUCCESS")
        self.assertIn("2/2 steps reached", node.last_response_summary)
        self.assertNotIn("skipped", node.last_response_summary)
        self.assertTrue(any("ALL STEPS COMPLETE" in m for m in node.logger.infos))

    def test_a_skipped_run_does_not_log_the_clean_success_line(self):
        node = self._finish([_step(), _step(skipped=True, basis="SKIPPED - x")])
        self.assertFalse(any("ALL STEPS COMPLETE (task SUCCESS)" in m for m in node.logger.infos))

    def test_a_skipped_step_is_not_counted_as_an_unverified_commit(self):
        """skip은 '확인 없이 커밋'이 아니라 '커밋 자체를 못 함'이다 - 이중 계산 금지."""
        task = {"steps": [_step(verified=False, skipped=True, basis="SKIPPED - x")]}
        self.assertEqual(mission3_pipe._unverified_step_count(task), 0)
        self.assertEqual(mission3_pipe._skipped_step_count(task), 1)


class DashboardTest(unittest.TestCase):
    def test_a_skipped_step_is_not_a_check_or_a_tilde(self):
        html = _plan_list_html([_step(skipped=True, basis="SKIPPED - no frontier")], 1)
        self.assertIn("✗", html)
        self.assertNotIn("✓", html)
        self.assertNotIn("≈", html)
        self.assertIn("SKIPPED", html)

    def test_progress_excludes_skipped_steps(self):
        """인덱스는 올라가므로 그것까지 done으로 세면 화면이 거짓말을 한다."""
        snapshot = {
            "mission_type": "instruction_following",
            "task": {"steps": [_step(), _step(skipped=True), _step()], "parser": "rules"},
            "mission3_step_index": 3,
        }
        html = _mission_detail_rows(snapshot)
        self.assertIn("2 / 3 steps reached", html)
        self.assertIn("1 skipped", html)

    def test_progress_without_skips_is_plain(self):
        snapshot = {
            "mission_type": "instruction_following",
            "task": {"steps": [_step(), _step()], "parser": "rules"},
            "mission3_step_index": 2,
        }
        html = _mission_detail_rows(snapshot)
        self.assertIn("2 / 2 steps reached", html)
        self.assertNotIn("skipped", html)


if __name__ == "__main__":
    unittest.main()
