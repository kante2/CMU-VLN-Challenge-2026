"""대시보드 렌더링은 어떤 task 모양에서도 예외를 던지지 않는다.

실측(2026-08-23): "Go to the potted plant closest to the pyramid candle holder and
stop at the vase between the TV and the door." - avoid 절이 없어 global_forbidden이
비었는데, _mission_detail_rows가 forbidden_desc를 if 밖에서 참조해 UnboundLocalError가
났다. control_loop 타이머 콜백에서 터진 예외라 rclpy executor가 노드를 통째로 죽였다
(exit code 1). 즉 **avoid 없는 Mission 3 질문은 전부 즉사**였다.

표시가 깨지는 것과 로봇이 멈추는 것은 심각도가 다르다. 여기서는 렌더링 자체가 어떤
입력에도 안 죽는 것을 고정한다(호출부의 예외 격리는 sysnav_node에 따로 있다).
"""

import unittest

from sysnav.mission_dashboard import _mission_detail_rows


def _snapshot(**overrides):
    base = {"mission_type": "instruction_following", "task": {}, "mission3_step_index": 0}
    base.update(overrides)
    return base


_STEP = {"is_stop": True, "resolve": "category",
         "parsed": {"target": "vase", "attributes": [], "relation": "between",
                    "reference_objects": ["tv", "door"]}}


class InstructionFollowingRowsTest(unittest.TestCase):
    def test_no_avoid_clause_renders(self):
        """이번 크래시의 정확한 재현 - global_forbidden이 없는 Mission 3."""
        html = _mission_detail_rows(_snapshot(task={
            "steps": [_STEP], "parser": "llm", "global_forbidden": [],
        }))
        self.assertIn("Progress", html)
        self.assertNotIn("Forbidden constraint", html,
                         "제약이 없으면 그 행 자체를 그리지 않는다")

    def test_missing_global_forbidden_key_renders(self):
        html = _mission_detail_rows(_snapshot(task={"steps": [_STEP], "parser": "llm"}))
        self.assertIn("Progress", html)

    def test_avoid_clause_renders_the_row(self):
        html = _mission_detail_rows(_snapshot(
            task={"steps": [_STEP], "parser": "llm",
                  "global_forbidden": [{"relation": "near", "reference_objects": ["cabinet"]}]},
            mission3_forbidden_active=True,
        ))
        self.assertIn("Forbidden constraint", html)
        self.assertIn("active", html)

    def test_unresolved_avoid_clause_is_highlighted(self):
        html = _mission_detail_rows(_snapshot(
            task={"steps": [_STEP], "parser": "llm",
                  "global_forbidden": [{"relation": "near", "reference_objects": ["cabinet"]}]},
            mission3_forbidden_active=False,
        ))
        self.assertIn("not yet resolved", html)
        self.assertIn("#b91c1c", html, "미적용 제약은 빨갛게 눈에 띄어야 한다")

    def test_empty_task_renders(self):
        self.assertIsInstance(_mission_detail_rows(_snapshot(task={})), str)


class OtherMissionRowsTest(unittest.TestCase):
    def test_object_reference(self):
        html = _mission_detail_rows({"mission_type": "object_reference",
                                     "task": {"target": "vase"}})
        self.assertIn("vase", html)

    def test_numerical(self):
        html = _mission_detail_rows({"mission_type": "numerical",
                                     "task": {"target": "chair"}, "candidate_count": 3})
        self.assertIn("chair", html)

    def test_unknown_mission_type(self):
        self.assertIsInstance(_mission_detail_rows({"mission_type": None, "task": {}}), str)


if __name__ == "__main__":
    unittest.main()
