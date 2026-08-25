""""take the path between A and B"의 참조 물체 **쌍** 선택.

예전엔 A와 B를 각각 독립적으로 "로봇에 가장 가까운 것"으로 골랐다. 그래서
(1) detection confidence가 전혀 반영되지 않았고 - 실측 2026-08-25: "take the path
between the sofa and the round tables"에서 신뢰도 0.58짜리 오검출(벽 옆 화분받침을
table로 검출)이 로봇에 더 가깝다는 이유만으로 신뢰도 0.85짜리 진짜 원형 테이블을 이겼다 -
(2) 두 물체가 실제로 통과 가능한 게이트를 이루는지 아무도 확인하지 않았으며,
(3) live pose에 의존해서 unreachable 재시도마다 목표가 떠돌았다.
"""

import unittest

from sysnav import config
from sysnav.exploration.coverage_planner import CoveragePlanner
from sysnav.missions import mission3_pipe


def _obj(object_id, category, position, confidence, extent=(0.4, 0.4, 0.4)):
    return {
        "object_id": object_id,
        "category": category,
        "position": (float(position[0]), float(position[1]), 0.0),
        "confidence": float(confidence),
        "extent_3d": extent,
        "observation_count": 3,
        "self_attributes": {},
        "representative_image": None,
    }


class _Memory:
    def __init__(self, objects):
        self._objects = {int(o["object_id"]): o for o in objects}

    def find_by_category(self, category):
        wanted = str(category).strip().lower()
        return [dict(o) for o in self._objects.values() if o["category"] == wanted]

    def get(self, object_id):
        found = self._objects.get(int(object_id))
        return dict(found) if found else None

    def all_nodes(self):
        return [dict(o) for o in self._objects.values()]

    def add(self, obj):
        self._objects[int(obj["object_id"])] = obj


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, _message):
        pass


class _Node:
    def __init__(self, objects, planner):
        self.object_memory = _Memory(objects)
        self.coverage_planner = planner

    def get_logger(self):
        return _Logger()


def _open_room():
    """x in [-5, 5], y in [-2, 2]가 전부 FREE인 방. 나머지는 UNKNOWN."""
    planner = CoveragePlanner()
    planner.origin_x = planner.origin_y = -10.0
    planner.grid[40:61, 25:76] = config.OCC_FREE
    return planner


def _cell(planner, x, y):
    cell = planner.world_to_grid(x, y)
    assert cell is not None, (x, y)
    return cell


def _ref(target, attributes=None):
    return {"target": target, "attributes": attributes or []}


def _resolve(node, refs, pose=None):
    mode = "between_collective" if len(refs) == 1 else "between"
    return mission3_pipe._resolve_forbidden_segment(
        node, mode, refs, pose or {"x": 0.0, "y": 0.0, "yaw": 0.0}
    )


def _xs(segment):
    return sorted(round(point[0], 3) for point in segment)


class BetweenPairSelectionTest(unittest.TestCase):
    def setUp(self):
        mission3_pipe._REF_LOG_SEEN.clear()
        self.planner = _open_room()

    def test_confidence_picks_the_better_detection(self):
        """보고된 버그. gap이 동일하면 신뢰도 높은 쪽이 이겨야 한다."""
        node = _Node([
            _obj(1, "sofa", (0.0, 0.0), 0.92),
            _obj(2, "table", (2.0, 0.0), 0.85),   # 진짜 원형 테이블
            _obj(3, "table", (-2.0, 0.0), 0.58),  # 화분받침 오검출
        ], self.planner)
        # 로봇을 오검출 쪽에 두어도(예전 argmin이 지던 자리) 결과가 바뀌면 안 된다.
        segment = _resolve(node, [_ref("sofa"), _ref("table")], {"x": -4.0, "y": 0.0})
        self.assertEqual(_xs(segment), [0.0, 2.0])

    def test_a_partner_behind_a_wall_is_rejected(self):
        """신뢰도가 더 높아도 사이에 벽이 있으면 게이트가 아니다."""
        row_lo, col = _cell(self.planner, -1.0, -2.0)[0], _cell(self.planner, -1.0, 0.0)[1]
        self.planner.grid[row_lo:row_lo + 21, col] = config.OCC_OCCUPIED
        node = _Node([
            _obj(1, "sofa", (0.0, 0.0), 0.92),
            _obj(2, "table", (2.0, 0.0), 0.85),
            _obj(3, "table", (-2.0, 0.0), 0.95),  # 신뢰도는 최고지만 벽 너머
        ], self.planner)
        segment = _resolve(node, [_ref("sofa"), _ref("table")])
        self.assertEqual(_xs(segment), [0.0, 2.0])

    def test_the_pair_does_not_depend_on_the_robot_pose(self):
        """쌍이 로봇 위치와 무관해야 재시도 중에 goal이 안 떠돈다."""
        node = _Node([
            _obj(1, "sofa", (0.0, 0.0), 0.90),
            _obj(2, "table", (2.0, 0.0), 0.85),
            _obj(3, "table", (-2.0, 0.0), 0.85),
        ], self.planner)
        left = _resolve(node, [_ref("sofa"), _ref("table")], {"x": -4.5, "y": 0.0})
        right = _resolve(node, [_ref("sofa"), _ref("table")], {"x": 4.5, "y": 0.0})
        self.assertEqual(left, right)

    def test_the_pair_is_frozen_on_the_step(self):
        """한 번 확정한 뒤 memory가 바뀌어도 같은 좌표를 유지한다."""
        node = _Node([
            _obj(1, "sofa", (0.0, 0.0), 0.90),
            _obj(2, "table", (2.0, 0.0), 0.85),
        ], self.planner)
        step = {"point_mode": "between", "point_refs": [_ref("sofa"), _ref("table")]}
        pose = {"x": 0.0, "y": -1.0, "yaw": 0.0}
        first_point, first_segment = mission3_pipe._resolve_step_point(node, step, pose)
        self.assertIsNotNone(step.get("resolved_segment"))

        node.object_memory.add(_obj(4, "table", (0.6, 0.0), 0.99))  # 더 좋은 후보 등장
        second_point, second_segment = mission3_pipe._resolve_step_point(node, step, pose)
        self.assertEqual(first_segment, second_segment)
        self.assertEqual(first_point, second_point)

    def test_collective_never_pairs_an_object_with_itself(self):
        planner = self.planner
        single = _Node([_obj(1, "column", (0.0, 0.0), 0.9)], planner)
        self.assertIsNone(_resolve(single, [_ref("column")]))

        triple = _Node([
            _obj(1, "column", (0.0, 0.0), 0.9),
            _obj(2, "column", (1.0, 0.0), 0.9),
            _obj(3, "column", (4.0, 0.0), 0.9),
        ], planner)
        segment = _resolve(triple, [_ref("column")])
        self.assertNotEqual(segment[0], segment[1])

    def test_collective_picks_the_narrowest_traversable_gap(self):
        node = _Node([
            _obj(1, "column", (0.0, 0.0), 0.9),
            _obj(2, "column", (1.0, 0.0), 0.9),
            _obj(3, "column", (4.0, 0.0), 0.9),
        ], self.planner)
        self.assertEqual(_xs(_resolve(node, [_ref("column")])), [0.0, 1.0])

    def test_an_untraversable_midpoint_is_rejected(self):
        row, col = _cell(self.planner, 1.0, 0.0)
        self.planner.grid[row - 1:row + 2, col - 1:col + 2] = config.OCC_OCCUPIED
        node = _Node([
            _obj(1, "sofa", (0.0, 0.0), 0.95),
            _obj(2, "table", (2.0, 0.0), 0.95),   # 중점(1,0)이 막혀 있다
            _obj(3, "table", (-2.0, 0.0), 0.80),
        ], self.planner)
        self.assertEqual(_xs(_resolve(node, [_ref("sofa"), _ref("table")])), [-2.0, 0.0])

    def test_a_gap_wider_than_the_limit_loses(self):
        node = _Node([
            _obj(1, "sofa", (0.0, 0.0), 0.90),
            _obj(2, "table", (2.0, 0.0), 0.70),
            _obj(3, "table", (30.0, 0.0), 0.99),  # 신뢰도 최고지만 30m 떨어짐
        ], self.planner)
        self.assertEqual(_xs(_resolve(node, [_ref("sofa"), _ref("table")])), [0.0, 2.0])

    def test_an_unmapped_grid_still_resolves_a_pair(self):
        """지도가 없거나 전부 UNKNOWN이어도 쌍은 나와야 한다.

        여기서 None을 돌려주면 _select_step이 영원히 PLAN_EXPLORATION으로 돌아간다
        (exploration_exhausted 뒤에도 point step엔 탈출구가 없다)."""
        objects = [
            _obj(1, "sofa", (0.0, 0.0), 0.92),
            _obj(2, "table", (2.0, 0.0), 0.85),
        ]
        blank = CoveragePlanner()
        blank.origin_x = blank.origin_y = -10.0   # grid 전체가 UNKNOWN
        self.assertIsNotNone(_resolve(_Node(objects, blank), [_ref("sofa"), _ref("table")]))

        no_origin = CoveragePlanner()              # origin 자체가 없음
        self.assertIsNotNone(_resolve(_Node(objects, no_origin), [_ref("sofa"), _ref("table")]))

    def test_near_is_unchanged(self):
        """near는 여전히 로봇 최근접 인스턴스를 두 번 돌려준다."""
        node = _Node([
            _obj(1, "tv", (3.0, 0.0), 0.50),
            _obj(2, "tv", (-3.0, 0.0), 0.99),
        ], self.planner)
        segment = mission3_pipe._resolve_forbidden_segment(
            node, "near", [_ref("tv")], {"x": 2.5, "y": 0.0, "yaw": 0.0}
        )
        self.assertEqual(segment[0], segment[1])
        self.assertAlmostEqual(segment[0][0], 3.0)

    def test_forbidden_segment_still_builds_a_mask(self):
        """negative path("avoiding the path between A and B") 경로가 그대로 동작한다."""
        node = _Node([
            _obj(1, "sofa", (0.0, 0.0), 0.92),
            _obj(2, "table", (2.0, 0.0), 0.85),
        ], self.planner)
        segment = _resolve(node, [_ref("sofa"), _ref("table")])
        mask = mission3_pipe._build_forbidden_mask(node, *segment)
        self.assertIsNotNone(mask)
        self.assertTrue(mask.any())

    def test_zero_confidence_weight_lets_geometry_decide(self):
        """가중치를 config 대신 하드코딩하지 않았는지 확인."""
        node = _Node([
            _obj(1, "sofa", (0.0, 0.0), 0.90),
            _obj(2, "table", (3.0, 0.0), 0.95),   # 신뢰도 높지만 멀다
            _obj(3, "table", (-1.0, 0.0), 0.60),  # 신뢰도 낮지만 가깝다
        ], self.planner)
        self.assertEqual(_xs(_resolve(node, [_ref("sofa"), _ref("table")])), [0.0, 3.0])

        original = config.MISSION3_BETWEEN_CONFIDENCE_WEIGHT
        try:
            config.MISSION3_BETWEEN_CONFIDENCE_WEIGHT = 0.0
            mission3_pipe._REF_LOG_SEEN.clear()
            self.assertEqual(_xs(_resolve(node, [_ref("sofa"), _ref("table")])), [-1.0, 0.0])
        finally:
            config.MISSION3_BETWEEN_CONFIDENCE_WEIGHT = original


class TraversableMaskTest(unittest.TestCase):
    def test_traversable_mask_honours_the_requested_clearance(self):
        """ROBOT_CLEARANCE_M이 0.0이라 기본값으로는 좁은 틈을 못 걸러낸다."""
        planner = _open_room()
        row, col = _cell(planner, 0.0, 0.0)
        planner.grid[row, col] = config.OCC_OCCUPIED
        grid = planner.snapshot_grid()

        tight = planner.traversable_mask(grid, 0.0)
        wide = planner.traversable_mask(grid, 0.6)
        self.assertGreater(int(tight.sum()), int(wide.sum()))
        # 장애물에서 2셀(0.4m) 떨어진 자리: 기본 clearance에선 통과, 0.6m에선 막힌다.
        self.assertTrue(bool(tight[row, col + 2]))
        self.assertFalse(bool(wide[row, col + 2]))


if __name__ == "__main__":
    unittest.main()
