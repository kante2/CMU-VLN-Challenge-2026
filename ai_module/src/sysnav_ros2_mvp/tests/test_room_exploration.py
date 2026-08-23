import unittest

import cv2
import numpy as np

from sysnav import config
from sysnav.exploration.coverage_planner import CoveragePlanner
from sysnav.exploration.viewpoint_memory import ViewpointMemory
from sysnav.rooms import cross_room_navigator
from sysnav.rooms.room_registry import RoomRegistry
from sysnav.rooms.room_segmenter import RoomSegmenter


def _two_room_map():
    grid = np.full((90, 120), config.OCC_UNKNOWN, dtype=np.int8)
    grid[10:80, 10:110] = config.OCC_FREE
    grid[10, 10:110] = config.OCC_OCCUPIED
    grid[79, 10:110] = config.OCC_OCCUPIED
    grid[10:80, 10] = config.OCC_OCCUPIED
    grid[10:80, 109] = config.OCC_OCCUPIED
    grid[10:80, 59] = config.OCC_OCCUPIED
    grid[36:44, 59] = config.OCC_FREE
    max_height = np.zeros(grid.shape, dtype=np.float32)
    max_height[grid == config.OCC_OCCUPIED] = 2.0
    return grid, max_height


class RoomSegmenterTest(unittest.TestCase):
    def test_detects_rooms_door_and_adjacency(self):
        grid, max_height = _two_room_map()
        result = RoomSegmenter().segment(grid, max_height=max_height)

        self.assertEqual(len(result["rooms"]), 2)
        self.assertEqual(len(result["doorways"]), 1)
        door = result["doorways"][0]
        self.assertEqual({door["room_a"], door["room_b"]}, {1, 2})
        self.assertEqual(result["adjacency"], {1: [2], 2: [1]})
        self.assertLessEqual(door["width_m"], config.ROOM_DOOR_MAX_WIDTH_M)


class RoomRegistryTest(unittest.TestCase):
    @staticmethod
    def _segmentation(left_id, right_id):
        labels = np.zeros((30, 40), dtype=np.int32)
        labels[2:28, 2:19] = left_id
        labels[2:28, 21:38] = right_id
        rooms = [
            {
                "room_id": left_id, "cell_count": 442, "area_m2": 9.945,
                "centroid_row": 14.5, "centroid_col": 10.0,
                "anchor_row": 14.0, "anchor_col": 8.0,
            },
            {
                "room_id": right_id, "cell_count": 442, "area_m2": 9.945,
                "centroid_row": 14.5, "centroid_col": 29.0,
                "anchor_row": 14.0, "anchor_col": 31.0,
            },
        ]
        return {
            "labels": labels,
            "rooms": rooms,
            "doorways": [{
                "door_id": 1, "room_a": left_id, "room_b": right_id,
                "centroid_row": 14.5, "centroid_col": 20.0,
                "cell_count": 2, "width_m": 0.9,
            }],
            "adjacency": {left_id: [right_id], right_id: [left_id]},
        }

    def test_keeps_ids_history_and_builds_door_path(self):
        registry = RoomRegistry()
        world_to_grid = lambda x, y: (int(y), int(x))
        first = registry.update(
            self._segmentation(1, 2),
            viewpoints=[{
                "viewpoint_id": 7, "pose": {"x": 8.0, "y": 14.0},
                "image_path": "/tmp/left.jpg", "coverage_voxel_count": 80,
            }],
            objects=[{"category": "chair", "position": [8.0, 14.0, 0.0]}],
            world_to_grid=world_to_grid,
            robot_cell=(14, 8),
        )
        left_id = next(room["room_id"] for room in first["rooms"] if room["centroid_col"] < 20)
        right_id = next(room["room_id"] for room in first["rooms"] if room["centroid_col"] > 20)

        # Ephemeral watershed labels swap on the next cycle; persistent IDs must not.
        second = registry.update(
            self._segmentation(8, 7), viewpoints=[], objects=[],
            world_to_grid=world_to_grid, robot_cell=(14, 8),
        )
        left = registry.get_room(left_id)
        self.assertEqual(left["viewpoint_count"], 1)
        self.assertEqual(left["representative_viewpoint_id"], 7)
        self.assertEqual(left["object_labels"], ["chair"])
        self.assertEqual(int(second["labels"][14, 8]), left_id)
        self.assertEqual(int(second["labels"][14, 31]), right_id)

        self.assertFalse(registry.record_exploration_result(left_id, has_route=False))
        self.assertTrue(registry.record_exploration_result(left_id, has_route=False))
        candidates = registry.navigation_candidates(left_id)
        self.assertEqual([room["room_id"] for room in candidates], [right_id])
        self.assertEqual(candidates[0]["room_path"], [left_id, right_id])
        self.assertEqual(len(candidates[0]["doorways"]), 1)


def _walled_room(planner, row_range, col_range):
    """planner.grid에 사방이 벽인 known-free 방 하나를 그린다."""
    r0, r1 = row_range
    c0, c1 = col_range
    planner.grid[r0:r1, c0:c1] = config.OCC_FREE
    planner.grid[r0 - 1, c0 - 1:c1 + 1] = config.OCC_OCCUPIED
    planner.grid[r1, c0 - 1:c1 + 1] = config.OCC_OCCUPIED
    planner.grid[r0 - 1:r1 + 1, c0 - 1] = config.OCC_OCCUPIED
    planner.grid[r0 - 1:r1 + 1, c1] = config.OCC_OCCUPIED


def _room_with_doorway_gap():
    """벽 한쪽에 좁은 문이 뚫려 있고 그 너머가 UNKNOWN인 방.

    문 주변은 clearance가 낮아 watershed가 배경으로 흡수하므로 room label이 안 붙는다 -
    실제 주행에서 frontier가 방 mask 밖으로 전부 떨어져 나간 상황과 같은 구조다.
    """
    planner = CoveragePlanner()
    planner.origin_x, planner.origin_y = -6.0, -6.0
    _walled_room(planner, (20, 60), (20, 70))
    planner.grid[38:42, 70] = config.OCC_FREE  # 문(0.8m). 너머는 UNKNOWN = frontier
    planner.max_height[planner.grid == config.OCC_OCCUPIED] = 2.0
    segmentation = RoomSegmenter().segment(planner.grid, max_height=planner.max_height)
    robot_cell = (40, 45)
    x, y = planner.grid_to_world(*robot_cell)
    return planner, segmentation, robot_cell, {"x": x, "y": y, "yaw": 0.0}


class CoverageScopeTest(unittest.TestCase):
    def test_unlabeled_band_is_assigned_to_the_nearest_room(self):
        """watershed가 남긴 미라벨 띠는 "더 가까운 방" 것으로 친다. 그래야 벽 옆에 붙어
        있는 frontier가 살아남는다. 단, 옆방에 더 가까운 절반은 넘겨주지 않는다."""
        labels = np.zeros((8, 12), dtype=np.int32)
        labels[:, :5] = 3
        labels[:, 7:] = 9
        scoped = CoveragePlanner._active_room_mask(labels, 3)
        # col 5는 room 3에 1셀, room 9에 2셀 -> room 3 것. col 6은 그 반대.
        self.assertTrue(scoped[:, :6].all())
        self.assertFalse(scoped[:, 6:].any())

    def test_frontier_in_unlabeled_wall_band_is_not_dropped(self):
        """회귀: 예전처럼 room mask를 1셀만 dilate하면 문 옆 frontier가 전부 걸러져서
        plan_route()가 "이 방 다 봤다"며 빈 route를 반환했다(mission3는 그걸 곧바로
        FAILED로 해석해 태스크가 죽었다)."""
        planner, segmentation, robot_cell, pose = _room_with_doorway_gap()
        labels = segmentation["labels"]
        room_id = CoveragePlanner._active_room_id(robot_cell, segmentation)
        self.assertIsNotNone(room_id)

        frontier = planner.frontier_extractor._mask(planner.grid)
        self.assertTrue(frontier.any(), "테스트 맵에 frontier가 있어야 한다")

        legacy_margin = cv2.dilate(
            (labels == room_id).astype(np.uint8), np.ones((3, 3), np.uint8)
        ).astype(bool)
        self.assertEqual(int((frontier & legacy_margin).sum()), 0)  # 예전 동작 = 전멸

        current = CoveragePlanner._active_room_mask(labels, room_id)
        self.assertEqual(int((frontier & current).sum()), int(frontier.sum()))

        route = planner.plan_route(pose, ViewpointMemory(), room_segmentation=segmentation)
        self.assertTrue(route)
        self.assertEqual(planner.last_plan_diagnostics["reason"], "ok")
        self.assertFalse(planner.last_plan_diagnostics["fell_back_to_whole_map"])

    def test_other_rooms_frontier_stays_out_of_scope(self):
        """미라벨 띠를 넓게 흡수해도 room scoping의 원래 목적(옆방 frontier를 쫓아가지
        않기)은 유지돼야 한다."""
        planner = CoveragePlanner()
        planner.origin_x, planner.origin_y = -6.0, -6.0
        _walled_room(planner, (20, 60), (20, 110))
        planner.grid[20:60, 65] = config.OCC_OCCUPIED  # 가운데 칸막이
        planner.grid[38:42, 65] = config.OCC_FREE      # 두 방을 잇는 문
        planner.grid[38:42, 19] = config.OCC_FREE      # 왼쪽 방 frontier
        planner.grid[38:42, 110] = config.OCC_FREE     # 오른쪽 방 frontier
        planner.max_height[planner.grid == config.OCC_OCCUPIED] = 2.0
        segmentation = RoomSegmenter().segment(planner.grid, max_height=planner.max_height)
        self.assertEqual(len(segmentation["rooms"]), 2)

        left_id = CoveragePlanner._active_room_id((40, 35), segmentation)
        scoped = planner.frontier_extractor._mask(planner.grid) & (
            CoveragePlanner._active_room_mask(segmentation["labels"], left_id)
        )
        columns = np.nonzero(scoped)[1]
        self.assertTrue(columns.size)
        self.assertTrue((columns < 65).all(), "오른쪽 방 frontier가 섞이면 안 된다")

    def test_falls_back_to_whole_map_when_room_scope_has_no_frontier(self):
        """방 판정이 어긋나 방 안 frontier가 0개가 되어도, 전역에 남아 있으면 실패하지
        않고 room scope를 풀고 계속한다 - 방 판정 오류 하나가 탐색을 영구 정지시키면 안 된다."""
        planner, segmentation, robot_cell, pose = _room_with_doorway_gap()
        # 로봇 주변 아주 좁은 패치만 방으로 표시 -> frontier(문 근처)는 반경 밖이 된다.
        labels = np.zeros(planner.grid.shape, dtype=np.int32)
        labels[robot_cell[0] - 2:robot_cell[0] + 3, robot_cell[1] - 2:robot_cell[1] + 3] = 1
        segmentation = {**segmentation, "labels": labels}

        route = planner.plan_route(pose, ViewpointMemory(), room_segmentation=segmentation)
        diagnostics = planner.last_plan_diagnostics
        self.assertTrue(route)
        self.assertEqual(diagnostics["reason"], "ok")
        self.assertTrue(diagnostics["fell_back_to_whole_map"])
        # scope를 푼 사이클은 "이 방 다 봤다"고 주장하면 안 된다.
        self.assertIsNone(diagnostics["active_room_id"])
        self.assertFalse(diagnostics["room_scope_active"])


class FrontierVisibilityTest(unittest.TestCase):
    """candidate가 surface point를 "본다"고 판정하는 LOS 검사 (plan_route의 scov)."""

    @staticmethod
    def _room_with_wall_hugging_frontier():
        """벽에 난 1셀짜리 틈 너머가 UNKNOWN인 방 - frontier가 벽에 딱 붙어 있다.

        frontier는 정의상 free/unknown 경계라 늘 이런 위치에 생긴다. LOS 마스크가 벽을
        한 셀이라도 부풀리면 그 frontier로 가는 직선이 전부 막힌 것으로 오판된다.
        """
        planner = CoveragePlanner()
        planner.origin_x, planner.origin_y = -6.0, -6.0
        _walled_room(planner, (20, 40), (20, 60))
        planner.grid[40, 40] = config.OCC_FREE
        planner.max_height[planner.grid == config.OCC_OCCUPIED] = 2.0
        x, y = planner.grid_to_world(30, 40)
        return planner, {"x": x, "y": y, "yaw": 0.0}

    def test_los_margin_is_smaller_than_robot_clearance(self):
        """"몸체가 지나갈 수 있는가"와 "보이는가"는 다른 질문인데, 둘 다 같은 셀 수로
        반올림되면 판정이 완전히 같아진다. 0.15m는 셀이 0.20m라 max(1, ...) 때문에
        clearance와 똑같이 1셀이 되어 이 구분이 사라져 있었다."""
        resolution = CoveragePlanner().resolution
        los_cells = int(round(config.FRONTIER_LOS_WALL_MARGIN_M / resolution))
        clearance_cells = max(1, int(round(config.ROBOT_CLEARANCE_M / resolution)))
        self.assertLess(los_cells, clearance_cells)

    def test_wall_hugging_frontier_is_visible_from_the_room(self):
        """회귀: 벽에 붙은 frontier가 모든 후보에게 "안 보인다"로 판정돼서 plan_route가
        no_candidate_had_any_visible_uncovered_surface_point로 죽었다."""
        planner, pose = self._room_with_wall_hugging_frontier()
        route = planner.plan_route(pose, ViewpointMemory(), room_segmentation=None)
        diagnostics = planner.last_plan_diagnostics
        self.assertEqual(diagnostics["surface_point_count"], 1)
        self.assertTrue(route)
        self.assertEqual(diagnostics["reason"], "ok")
        self.assertGreater(diagnostics["candidates_with_nonempty_scov"], 0)

    def test_walls_still_block_line_of_sight(self):
        """margin을 0으로 낮춘 게 "벽을 통과해서 본다"가 되면 안 된다."""
        occupied = np.zeros((20, 20), dtype=bool)
        occupied[:, 10] = True  # 세로 벽
        self.assertFalse(CoveragePlanner._line_of_sight(occupied, (5, 5), (5, 15)))
        self.assertTrue(CoveragePlanner._line_of_sight(occupied, (5, 5), (15, 5)))


class TargetArrivalPolicyTest(unittest.TestCase):
    def test_stalled_fallback_cannot_relax_final_success_radius(self):
        self.assertEqual(config.TARGET_SUCCESS_DISTANCE_M, 0.5)
        self.assertEqual(
            config.TARGET_ARRIVAL_FALLBACK_MAX_M,
            config.TARGET_SUCCESS_DISTANCE_M,
        )


class _FakePlanner:
    last_direct_path_diagnostics = {}

    @staticmethod
    def grid_to_world(row, col):
        return float(col), float(row)

    @staticmethod
    def plan_direct_path(pose, goal_xy, max_hop_spacing_m):
        return [{"x": goal_xy[0], "y": goal_xy[1], "theta": 0.0}]


class _FakeRegistry:
    rooms = {
        2: {"anchor_row": 12.0, "anchor_col": 22.0},
        3: {"anchor_row": 12.0, "anchor_col": 34.0},
    }

    def get_room(self, room_id):
        return self.rooms.get(room_id)


class _FakeSelector:
    @staticmethod
    def rank(query, rooms):
        return [3]


class _FakeLogger:
    def warning(self, message):
        pass


class _FakeNode:
    coverage_planner = _FakePlanner()
    room_registry = _FakeRegistry()
    room_relevance_selector = _FakeSelector()

    @staticmethod
    def get_logger():
        return _FakeLogger()


class CrossRoomNavigatorTest(unittest.TestCase):
    def test_route_follows_every_door_and_steps_into_rooms(self):
        candidate = {
            "room_id": 3, "category": "kitchen", "object_labels": ["sink"],
            "anchor_row": 12.0, "anchor_col": 34.0, "visited": False,
            "room_path": [1, 2, 3],
            "doorways": [
                {"centroid_row": 10.0, "centroid_col": 15.0},
                {"centroid_row": 11.0, "centroid_col": 28.0},
            ],
        }
        result = cross_room_navigator.select_job(
            _FakeNode(), 4, {"raw": "find the sink"},
            {"x": 5.0, "y": 10.0}, [candidate],
        )

        self.assertEqual(result["room_id"], 3)
        self.assertEqual(result["room_path"], [1, 2, 3])
        self.assertEqual(result["path"][-1]["target_room_id"], 3)
        entering = {waypoint.get("entering_room_id") for waypoint in result["path"]}
        self.assertTrue({2, 3}.issubset(entering))
        self.assertTrue(all(
            waypoint["navigation_mode"] == "cross_room" for waypoint in result["path"]
        ))


if __name__ == "__main__":
    unittest.main()
