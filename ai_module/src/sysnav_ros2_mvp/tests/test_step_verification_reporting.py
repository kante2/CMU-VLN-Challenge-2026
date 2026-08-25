"""SUCCESS가 "정답"처럼 보이면 안 된다 - step을 무엇으로 확정했는지 남긴다.

mission3의 SUCCESS 조건은 mission3_step_index >= len(steps), 즉 "step 수만큼 도착했다"
뿐이다. 그래서 관계 판정이 0/4로 전부 false여도 _best_effort_step_target이 기하로
찍어 커밋하고 그대로 3/3 SUCCESS 초록불이 떴다(실측 2026-08-25: "stop at the cabinet
with a picture above it"). 채점은 실제 궤적을 보므로 그건 점수가 아닌데 화면만 성공이었다.

forbidden mask를 한 번도 못 걸었을 때 이미 같은 이유로 경고를 남기고 있다 -
같은 원칙을 step 확정 근거에도 적용한다.
"""

import unittest

from sysnav.mission_dashboard import _plan_list_html
from sysnav.missions import mission3_pipe


def _step(verified=None, basis=None, is_stop=True):
    step = {"is_stop": is_stop, "resolve": "category",
            "parsed": {"target": "cabinet", "attributes": [], "relation_chain": []}}
    if verified is not None or basis is not None:
        step["verified"] = verified
        step["basis"] = basis
    return step


class UnverifiedStepCountTest(unittest.TestCase):
    def test_a_geometrically_guessed_step_is_counted(self):
        task = {"steps": [_step(verified=False, basis="relation_pending -> geometric")]}
        self.assertEqual(mission3_pipe._unverified_step_count(task), 1)

    def test_a_vlm_confirmed_step_is_not_counted(self):
        task = {"steps": [_step(verified=True, basis="selected cabinet#5")]}
        self.assertEqual(mission3_pipe._unverified_step_count(task), 0)

    def test_a_coordinate_step_is_not_counted(self):
        """between/near는 애초에 VLM 판정 대상이 아니라 "찍었다"로 세면 안 된다."""
        task = {"steps": [_step(verified=None, basis="geometric between")]}
        self.assertEqual(mission3_pipe._unverified_step_count(task), 0)

    def test_an_untouched_step_is_not_counted(self):
        self.assertEqual(mission3_pipe._unverified_step_count({"steps": [_step()]}), 0)


class _Node:
    def __init__(self, task, index):
        import threading
        self.state_lock = threading.RLock()
        self.task = task
        self.mission3_step_index = index


class MarkStepBasisTest(unittest.TestCase):
    def test_the_basis_lands_on_the_current_step_only(self):
        task = {"steps": [_step(), _step(), _step()]}
        mission3_pipe._mark_step_basis(_Node(task, 1), verified=False, basis="geometric under picture")
        self.assertNotIn("verified", task["steps"][0])
        self.assertIs(task["steps"][1]["verified"], False)
        self.assertEqual(task["steps"][1]["basis"], "geometric under picture")
        self.assertNotIn("verified", task["steps"][2])

    def test_an_out_of_range_index_is_ignored(self):
        task = {"steps": [_step()]}
        mission3_pipe._mark_step_basis(_Node(task, 5), verified=False, basis="x")
        self.assertNotIn("verified", task["steps"][0])

    def test_no_task_is_ignored(self):
        mission3_pipe._mark_step_basis(_Node(None, 0), verified=True, basis="x")  # 예외 없이 통과


class PlanRenderingTest(unittest.TestCase):
    def test_a_guessed_step_is_not_a_plain_green_check(self):
        """보고된 문제. 확정과 추측이 화면상 같으면 안 된다."""
        steps = [_step(verified=False, basis="relation_pending -> geometric 'under picture'")]
        html = _plan_list_html(steps, current_index=1)
        self.assertIn("≈", html)
        self.assertNotIn("✓", html)
        self.assertIn("under picture", html)      # 근거가 화면에 보인다

    def test_a_confirmed_step_stays_a_green_check(self):
        steps = [_step(verified=True, basis="selected cabinet#5")]
        html = _plan_list_html(steps, current_index=1)
        self.assertIn("✓", html)
        self.assertNotIn("≈", html)

    def test_the_current_and_pending_steps_are_unchanged(self):
        html = _plan_list_html([_step(), _step()], current_index=0)
        self.assertIn("▶", html)
        self.assertIn("○", html)

    def test_the_basis_is_html_escaped(self):
        steps = [_step(verified=False, basis="<script>alert(1)</script>")]
        html = _plan_list_html(steps, current_index=1)
        self.assertNotIn("<script>", html)


if __name__ == "__main__":
    unittest.main()
