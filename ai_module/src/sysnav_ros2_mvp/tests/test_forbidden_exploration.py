"""탐사 경로도 금지구역("avoiding the path near/between X")을 피해야 한다.

예전엔 forbidden_mask가 목적지 주행(plan_direct_path)에만 적용됐다. 그런데 README의
Instruction-Following 채점은 목적지 주행이 아니라 **로봇이 실제로 따라간 궤적 전체**를
보고, "passes through areas it is forbidden to go through"를 감점한다. 탐사 중에
지나가면 그대로 감점이다.
"""

import unittest

import numpy as np

from sysnav import config
from sysnav.exploration.coverage_planner import CoveragePlanner
from sysnav.exploration.viewpoint_memory import ViewpointMemory


def _corridor_planner():
    """가로로 긴 복도. 왼쪽 끝에 로봇, 오른쪽 끝에 미탐사(=frontier)."""
    planner = CoveragePlanner()
    planner.origin_x = planner.origin_y = -6.0
    planner.grid[28:36, 20:80] = config.OCC_FREE
    planner.grid[27, 19:81] = config.OCC_OCCUPIED
    planner.grid[36, 19:81] = config.OCC_OCCUPIED
    planner.grid[27:37, 19] = config.OCC_OCCUPIED
    return planner


def _pose(planner, row, col):
    x, y = planner.grid_to_world(row, col)
    return {"x": x, "y": y, "yaw": 0.0}


class ForbiddenExplorationTest(unittest.TestCase):
    def setUp(self):
        self.planner = _corridor_planner()
        self.pose = _pose(self.planner, 32, 25)

    def _route(self, mask):
        return self.planner.plan_route(
            self.pose, ViewpointMemory(), forbidden_mask=mask
        )

    def test_route_exists_without_the_constraint(self):
        self.assertTrue(self._route(None), "제약이 없으면 복도 끝으로 갈 경로가 있어야 한다")

    def test_forbidden_band_blocks_the_corridor(self):
        """복도를 가로지르는 금지 띠를 두면 그 너머 후보로는 경로가 안 나와야 한다."""
        mask = np.zeros(self.planner.grid.shape, dtype=bool)
        mask[27:37, 45:50] = True                      # 복도를 가로막는 띠
        route = self._route(mask)
        for hop in route:
            cell = self.planner.world_to_grid(hop["x"], hop["y"])
            self.assertIsNotNone(cell)
            self.assertFalse(mask[cell], "금지구역 안을 waypoint로 내면 안 된다")
            self.assertLess(cell[1], 50, "금지 띠 너머로 넘어가면 안 된다")

    def test_diagnostics_report_the_constraint(self):
        mask = np.zeros(self.planner.grid.shape, dtype=bool)
        mask[27:37, 45:50] = True
        self._route(mask)
        diagnostics = self.planner.last_plan_diagnostics
        self.assertTrue(diagnostics.get("forbidden_active"))
        self.assertGreater(diagnostics.get("forbidden_cell_count", 0), 0)

    def test_no_mask_is_reported_as_inactive(self):
        self._route(None)
        self.assertFalse(self.planner.last_plan_diagnostics.get("forbidden_active"))

    def test_mask_with_wrong_shape_is_ignored(self):
        """방어: 격자 크기가 다른 마스크가 들어와도 죽지 않고 무시한다."""
        route = self._route(np.zeros((5, 5), dtype=bool))
        self.assertTrue(route)
        self.assertFalse(self.planner.last_plan_diagnostics.get("forbidden_active"))


if __name__ == "__main__":
    unittest.main()


class _Lock:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _Logger:
    def __init__(self): self.messages = []
    def info(self, m): self.messages.append(("info", m))
    def warning(self, m): self.messages.append(("warning", m))


class _Node:
    """mission3_pipe가 쓰는 최소 인터페이스만 갖춘 가짜 노드."""
    class _Memory:
        """관측된 카테고리 집합을 흉내낸다. _try_resolve_forbidden과 증거 대기가
        둘 다 find_by_category를 쓴다."""
        def __init__(self, seen): self.seen = set(seen)
        def find_by_category(self, category):
            return [{"position": (0.0, 0.0, 0.0)}] if category in self.seen else []

    def __init__(self, forbidden_mask=None, exhausted=False, step_index=0, seen=()):
        self.object_memory = self._Memory(seen)
        self.state = "MISSION3_SELECT_STEP"
        self.state_lock = _Lock()
        self.mission3_forbidden_mask = forbidden_mask
        self.mission3_exploration_exhausted = exhausted
        self.mission3_step_index = step_index
        self.last_response_summary = None
        self._logger = _Logger()
        self.exploration_route = None
        self.task = _TASK
        self.coverage_planner = type("P", (), {"describe_last_plan_failure": staticmethod(lambda: "-")})()
    def get_logger(self): return self._logger
    def selection_job(self, *a, **k): return {}
    def submit_job(self, *a, **k): self.state = "SUBMITTED"


_TASK = {
    "steps": [{"is_stop": True, "resolve": "category",
               "parsed": {"target": "cup", "detection_prompts": ["cup"]}}],
    "global_forbidden": [{"point_mode": "near", "point_refs": [{"target": "cabinet"}]}],
}


class ForbiddenFirstTest(unittest.TestCase):
    """avoid 제약이 있으면 참조 물체를 찾을 때까지 탐색을 먼저 한다."""

    def test_step_resolution_waits_until_the_reference_is_found(self):
        from sysnav.missions import mission3_pipe
        node = _Node(forbidden_mask=None)
        mission3_pipe._select_step(node, _TASK, 1, {"x": 0.0, "y": 0.0, "yaw": 0.0})
        self.assertEqual(node.state, "PLAN_EXPLORATION")

    def test_step_resolution_proceeds_once_the_mask_exists(self):
        from sysnav.missions import mission3_pipe
        node = _Node(forbidden_mask=np.zeros((4, 4), dtype=bool), seen={"cup"})
        mission3_pipe._select_step(node, _TASK, 1, {"x": 0.0, "y": 0.0, "yaw": 0.0})
        self.assertEqual(node.state, "SUBMITTED", "마스크와 증거가 모두 있으면 확정하러 간다")

    def test_exhausted_search_does_not_block_forever(self):
        from sysnav.missions import mission3_pipe
        node = _Node(forbidden_mask=None, exhausted=True)
        mission3_pipe._select_step(node, _TASK, 1, {"x": 0.0, "y": 0.0, "yaw": 0.0})
        self.assertEqual(node.state, "SUBMITTED", "못 찾아도 결국 진행해야 한다")

    def test_empty_route_marks_exhausted_instead_of_failing(self):
        from sysnav.missions import mission3_pipe
        node = _Node(forbidden_mask=None)
        mission3_pipe._on_exploration_result(node, {"route": []})
        self.assertTrue(node.mission3_exploration_exhausted)
        self.assertEqual(node.state, "MISSION3_SELECT_STEP")
        self.assertNotEqual(node.state, "FAILED")

    def test_success_records_that_the_constraint_was_not_enforced(self):
        from sysnav.missions import mission3_pipe
        node = _Node(forbidden_mask=None, exhausted=True, step_index=1)
        mission3_pipe._select_step(node, _TASK, 1, {"x": 0.0, "y": 0.0, "yaw": 0.0})
        self.assertEqual(node.state, "SUCCESS")
        self.assertIn("NOT enforced", node.last_response_summary)
        self.assertTrue(any(level == "warning" for level, _ in node.get_logger().messages))

    def test_success_is_clean_when_the_constraint_was_enforced(self):
        from sysnav.missions import mission3_pipe
        node = _Node(forbidden_mask=np.zeros((4, 4), dtype=bool), step_index=1)
        mission3_pipe._select_step(node, _TASK, 1, {"x": 0.0, "y": 0.0, "yaw": 0.0})
        self.assertEqual(node.state, "SUCCESS")
        self.assertNotIn("NOT enforced", node.last_response_summary)


_TASK_NO_FORBIDDEN = {
    "steps": [{"is_stop": True, "resolve": "category",
               "parsed": {"target": "cup",
                          "detection_prompts": ["cup", "tv remote"]}}],
    "global_forbidden": [],
}


class EvidenceBeforeDecidingTest(unittest.TestCase):
    """target과 관계 참조 물체가 다 보이기 전에는 판정을 시도하지 않는다.

    실측(2026-08-22): "the cup near the TV remote"에서 tv remote를 한 번도 못 봤는데
    컵 2개만 보고 2:32에 SUCCESS가 났다. selection_job의 이미지 폴백이 "아직 안 가봐서
    못 본" 경우에도 발동해 성급하게 확정한 것이다.
    """

    def _node(self, seen, exhausted=False):
        node = _Node(exhausted=exhausted, seen=seen)
        node.task = _TASK_NO_FORBIDDEN
        return node

    def _run(self, node):
        from sysnav.missions import mission3_pipe
        mission3_pipe._select_step(node, _TASK_NO_FORBIDDEN, 1, {"x": 0.0, "y": 0.0, "yaw": 0.0})

    def test_waits_while_the_relation_reference_is_unseen(self):
        node = self._node(seen={"cup"})          # tv remote 미관측
        self._run(node)
        self.assertEqual(node.state, "PLAN_EXPLORATION")
        self.assertIn("tv remote", node.get_logger().messages[-1][1])

    def test_waits_while_the_target_is_unseen(self):
        node = self._node(seen={"tv remote"})
        self._run(node)
        self.assertEqual(node.state, "PLAN_EXPLORATION")

    def test_decides_once_every_category_is_observed(self):
        node = self._node(seen={"cup", "tv remote"})
        self._run(node)
        self.assertEqual(node.state, "SUBMITTED")

    def test_exhausted_exploration_decides_with_what_it_has(self):
        node = self._node(seen={"cup"}, exhausted=True)
        self._run(node)
        self.assertEqual(node.state, "SUBMITTED", "더 볼 곳이 없으면 부족해도 결정한다")
