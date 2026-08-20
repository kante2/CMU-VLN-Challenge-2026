import unittest

import numpy as np

from sysnav import config
from sysnav.exploration.coverage_planner import CoveragePlanner
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


class CoverageScopeTest(unittest.TestCase):
    def test_traversability_is_limited_to_current_room(self):
        traversable = np.ones((8, 12), dtype=bool)
        labels = np.zeros((8, 12), dtype=np.int32)
        labels[:, :5] = 3
        labels[:, 7:] = 9
        scoped = CoveragePlanner._room_scoped_traversable(
            traversable, (4, 2), {"labels": labels}
        )
        self.assertTrue(scoped[:, :5].all())
        self.assertFalse(scoped[:, 5:].any())


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
