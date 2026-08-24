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

from sysnav.mission_dashboard import _mission_detail_rows, _target_panel


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


class TargetDistancePanelTest(unittest.TestCase):
    """목표가 정해진 뒤 "어디로, 얼마나 남았나"를 대시보드에서 바로 읽을 수 있어야 한다.

    예전에는 좌표 한 줄(dest=..., dist=...)뿐이라, 목적지는 도착 반경 안인데 물체는
    아직 먼 상황과 중간 hop을 도는 상황이 구분되지 않았다.
    """

    _POSE = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    _FULL = {
        "pose": _POSE,
        "target_goal_xy": (3.0, 4.0),          # 로봇에서 5m
        "target_distance_m": 5.0,
        "target_object_xy": (6.0, 8.0),        # 로봇에서 10m
        "target_object_distance_m": 10.0,
        "target_success_radius_m": 1.0,
        "target_hops_remaining": 2,
        "target_replans": 3,
        "target_best_distance_m": 4.8,
        "target_no_progress_sec": 7.0,
        "current_goal": {"x": 1.0, "y": 0.0, "type": "target"},
    }

    def test_no_goal_yet(self):
        self.assertIn("아직 목표가 정해지지", _target_panel({"pose": self._POSE}))

    def test_shows_coordinate_and_distance_for_each_goal(self):
        panel = _target_panel(self._FULL)
        self.assertIn("(3.00, 4.00)", panel)
        self.assertIn("5.00 m", panel)
        self.assertIn("(6.00, 8.00)", panel)
        self.assertIn("10.00 m", panel)
        # 현재 발행 goal까지의 거리는 pose로 직접 계산한다(스냅샷에 없는 값).
        self.assertIn("(1.00, 0.00)", panel)
        self.assertIn("1.00 m", panel)

    def test_remaining_distance_uses_mission_success_radius(self):
        self.assertIn("도착까지 4.00 m", _target_panel(self._FULL))
        arrived = _target_panel({**self._FULL, "target_distance_m": 0.7})
        self.assertNotIn("도착까지", arrived)

    def test_stall_is_visible(self):
        panel = _target_panel(self._FULL)
        self.assertIn("7초째 정체", panel)
        self.assertIn("재계획 3회", panel)

    def test_missing_optional_fields_do_not_raise(self):
        # 목표만 있고 나머지가 아직 안 채워진 tick에서도 렌더링은 죽으면 안 된다.
        panel = _target_panel({"target_goal_xy": (1.0, 2.0)})
        self.assertIn("(1.00, 2.00)", panel)
