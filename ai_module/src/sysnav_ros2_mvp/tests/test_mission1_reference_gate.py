"""Mission 1은 관계의 참조 물체를 다 보기 전에는 개수를 세지 않는다.

왜 필요한가 (실측 2026-08-24): "How many chairs are near the table with a vase on
it?"에서 LLM 파서가 vase를 통째로 흘려서 prompts=['chair','table']로 나갔고, 그래서
vase는 애초에 검출 대상이 아니었다. 파서를 고쳐 vase가 prompts에 들어가도, 집계 쪽에
게이트가 없으면 "frontier가 없다"는 이유만으로 vase를 한 번도 못 본 채 집계에 들어가
**아무 table 옆 chair**를 세게 된다.

게이트만 두면 참조 물체를 끝내 못 찾을 때 답을 영영 못 내므로
(무응답 = 0점), maybe_force_count_at_deadline()이 시간 예산에서 탈출구를 만든다.
"""

import sys
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

        logging_module.get_logger = lambda name: _RclpyLogger()
        package.logging = logging_module
        sys.modules["rclpy"] = package
        sys.modules["rclpy.logging"] = logging_module

if "std_msgs.msg" not in sys.modules:
    try:
        import std_msgs.msg  # noqa: F401
    except ImportError:                                       # pragma: no cover
        package = types.ModuleType("std_msgs")
        module = types.ModuleType("std_msgs.msg")

        class _Int32:
            def __init__(self): self.data = 0

        module.Int32 = _Int32
        package.msg = module
        sys.modules.setdefault("std_msgs", package)
        sys.modules["std_msgs.msg"] = module

import threading                                               # noqa: E402
import time                                                    # noqa: E402

from sysnav import config                                      # noqa: E402
from sysnav.missions import mission1_pipe                      # noqa: E402


# "How many chairs are near the table with a vase on it?"의 (고쳐진) 파싱 결과.
_TASK = {
    "target": "chair",
    "raw": "How many chairs are near the table with a vase on it?",
    "attributes": [],
    "relation": "near",
    "reference_objects": ["table"],
    "relation_chain": [("chair", "near", "table"), ("table", "under", "vase")],
    "detection_prompts": ["chair", "table", "vase"],
}


class _Logger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass


class _ObjectMemory:
    def __init__(self, categories_seen):
        self._seen = categories_seen

    def find_by_category(self, category):
        return [{"object_id": 1, "category": category}] if category in self._seen else []


class _SceneGraph:
    def __init__(self, matching_ids):
        self._matching_ids = matching_ids

    def find_matching_target_ids(self, task):
        return list(self._matching_ids)

    def best_viewpoint_for_objects(self, object_ids):
        return None


class _Node:
    def __init__(self, categories_seen, matching_ids=(), started_ago=0.0):
        self.object_memory = _ObjectMemory(categories_seen)
        self.scene_graph = _SceneGraph(matching_ids)
        self.state = "PLAN_EXPLORATION"
        self.state_lock = threading.Lock()
        self.exploration_route = []
        self.current_goal = None
        self.task_start_time = time.monotonic() - started_ago
        self.coverage_planner = types.SimpleNamespace(
            describe_last_plan_failure=lambda: "no frontier"
        )
        self.vlm_counter = None

    def get_logger(self):
        return _Logger()


class MissingReferenceGateTest(unittest.TestCase):
    def test_missing_categories_lists_unseen_prompts(self):
        node = _Node(categories_seen={"chair", "table"})
        self.assertEqual(mission1_pipe._missing_categories(node, _TASK), ["vase"])

    def test_exploration_exhausted_does_not_finalize_while_a_reference_is_unseen(self):
        """vase를 못 본 상태에서 frontier가 없다고 집계로 넘어가면, 제약이 빠진
        숫자("아무 table 옆 chair")를 답으로 발행하게 된다."""
        node = _Node(categories_seen={"chair", "table"})
        mission1_pipe._on_exploration_result(node, _TASK, {"route": []})
        self.assertEqual(node.state, "OBSERVE")

    def test_exploration_exhausted_finalizes_once_every_reference_is_seen(self):
        node = _Node(categories_seen={"chair", "table", "vase"})
        mission1_pipe._on_exploration_result(node, _TASK, {"route": []})
        self.assertEqual(node.state, "MISSION1_FINALIZE_COUNT")

    def test_gate_is_inert_for_questions_without_relations(self):
        """관계가 없는 질문("How many chairs?")은 예전처럼 바로 집계로 간다."""
        node = _Node(categories_seen={"chair"})
        task = {"target": "chair", "raw": "How many chairs are there?",
                "detection_prompts": ["chair"], "relation_chain": []}
        mission1_pipe._on_exploration_result(node, task, {"route": []})
        self.assertEqual(node.state, "MISSION1_FINALIZE_COUNT")


class DeadlineEscapeTest(unittest.TestCase):
    def test_deadline_forces_the_count_even_with_a_reference_missing(self):
        """게이트가 영원히 잡아두면 무응답(0점)이 된다 - 예산을 넘기면 탈출한다."""
        node = _Node(categories_seen={"chair"},
                     started_ago=config.MISSION1_EXPLORATION_TIME_LIMIT_SEC + 1.0)
        self.assertTrue(mission1_pipe.maybe_force_count_at_deadline(node, "OBSERVE"))
        self.assertEqual(node.state, "MISSION1_FINALIZE_COUNT")

    def test_no_deadline_before_the_budget_elapses(self):
        node = _Node(categories_seen={"chair"}, started_ago=1.0)
        self.assertFalse(mission1_pipe.maybe_force_count_at_deadline(node, "OBSERVE"))
        self.assertEqual(node.state, "PLAN_EXPLORATION")

    def test_deadline_only_applies_while_exploring(self):
        """이미 집계/종료 상태면 건드리지 않는다."""
        node = _Node(categories_seen={"chair"},
                     started_ago=config.MISSION1_EXPLORATION_TIME_LIMIT_SEC + 1.0)
        self.assertFalse(
            mission1_pipe.maybe_force_count_at_deadline(node, "MISSION1_FINALIZE_COUNT")
        )


class VlmFallbackSuppressionTest(unittest.TestCase):
    def test_vlm_is_not_used_when_no_relation_candidate_was_verified(self):
        """VLM 집계는 관계 필터를 안 거친다(질문 원문만 넘긴다). 관계가 하나도 검증
        안 된 상태에서 쓰면 근거 없는 숫자로 기하 결과 0을 덮어써버린다."""
        node = _Node(categories_seen={"chair", "table"}, matching_ids=())
        node.vlm_counter = types.SimpleNamespace(
            count=lambda *a, **k: (_ for _ in ()).throw(AssertionError("VLM이 불리면 안 된다"))
        )
        result = mission1_pipe.count_job(node, 1, _TASK)
        self.assertEqual(result["count"], 0)

    def test_vlm_still_runs_once_the_relation_is_verified(self):
        node = _Node(categories_seen={"chair", "table", "vase"}, matching_ids=(1,))
        node.vlm_counter = types.SimpleNamespace(count=lambda *a, **k: 4)
        original = config.NUMERICAL_VLM_COUNT_ENABLED
        config.NUMERICAL_VLM_COUNT_ENABLED = True
        try:
            result = mission1_pipe.count_job(node, 1, _TASK)
        finally:
            config.NUMERICAL_VLM_COUNT_ENABLED = original
        # best_viewpoint_for_objects()가 None이라 VLM 경로는 조용히 포기하고
        # 기하 집계(관계 검증된 chair 1개)를 그대로 쓴다 - fail-quiet 동작 확인.
        self.assertEqual(result["count"], 1)


if __name__ == "__main__":
    unittest.main()


class EarlySettleTest(unittest.TestCase):
    """관계가 확정되고 개수가 안정되면 탐색을 끝낸다 - 단, '확정 즉시'는 아니다."""

    def _node(self, matching_ids):
        node = _Node(categories_seen={"chair", "table", "vase"}, matching_ids=matching_ids)
        node.task = _TASK
        return node

    def test_does_not_settle_on_the_first_verification(self):
        """관계가 잡힌 순간엔 테이블 반대편을 아직 못 봤을 수 있다(GT 8인데 3)."""
        node = self._node((1, 2, 3))
        mission1_pipe._on_perception_result(node, "OBSERVE")
        self.assertEqual(node.state, "PLAN_EXPLORATION")

    def test_settles_after_the_count_stops_changing(self):
        node = self._node((1, 2, 3))
        for _ in range(config.MISSION1_SETTLED_STABLE_OBSERVATIONS):
            mission1_pipe._on_perception_result(node, "OBSERVE")
        self.assertEqual(node.state, "MISSION1_FINALIZE_COUNT")

    def test_a_changing_count_resets_the_streak(self):
        """의자가 계속 새로 보이는 동안은 끊지 않는다."""
        node = self._node((1, 2, 3))
        mission1_pipe._on_perception_result(node, "OBSERVE")
        mission1_pipe._on_perception_result(node, "OBSERVE")
        node.scene_graph._matching_ids = (1, 2, 3, 4)   # 새 의자 발견
        mission1_pipe._on_perception_result(node, "OBSERVE")
        self.assertEqual(node.state, "PLAN_EXPLORATION")

    def test_never_settles_early_without_a_relation(self):
        """'How many chairs?'는 방 전체가 대상이라 조기 종료 근거가 없다."""
        node = _Node(categories_seen={"chair"}, matching_ids=(1, 2))
        node.task = {"target": "chair", "raw": "How many chairs are there?",
                     "detection_prompts": ["chair"], "relation_chain": []}
        for _ in range(config.MISSION1_SETTLED_STABLE_OBSERVATIONS + 2):
            mission1_pipe._on_perception_result(node, "OBSERVE")
        self.assertEqual(node.state, "PLAN_EXPLORATION")

    def test_moving_observations_do_not_settle(self):
        """주행 중 관측은 흐름을 건드리지 않는다(기존 동작 유지)."""
        node = self._node((1, 2, 3))
        node.state = "FOLLOW_EXPLORATION"
        for _ in range(config.MISSION1_SETTLED_STABLE_OBSERVATIONS + 1):
            mission1_pipe._on_perception_result(node, "FOLLOW_EXPLORATION")
        self.assertEqual(node.state, "FOLLOW_EXPLORATION")
