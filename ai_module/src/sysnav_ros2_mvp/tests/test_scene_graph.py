import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sysnav import config
from sysnav.scene_graph.scene_graph_manager import SceneGraphManager


class SceneGraphManagerTest(unittest.TestCase):
    def setUp(self):
        self.previous_gemini_setting = config.SCENE_GRAPH_USE_GEMINI_RELATIONS
        self.previous_threshold = config.VIEWPOINT_NOVEL_VOXEL_THRESHOLD
        config.SCENE_GRAPH_USE_GEMINI_RELATIONS = False
        config.VIEWPOINT_NOVEL_VOXEL_THRESHOLD = 30

    def tearDown(self):
        config.SCENE_GRAPH_USE_GEMINI_RELATIONS = self.previous_gemini_setting
        config.VIEWPOINT_NOVEL_VOXEL_THRESHOLD = self.previous_threshold

    @staticmethod
    def _scan(radius: float = 3.0, count: int = 360) -> np.ndarray:
        angles = np.linspace(-np.pi, np.pi, count, endpoint=False)
        z = 0.3 + 0.5 * np.sin(angles * 3.0)
        return np.column_stack(
            [radius * np.cos(angles), radius * np.sin(angles), z]
        ).astype(np.float32)

    @staticmethod
    def _nodes():
        table = {
            "object_id": 1,
            "category": "table",
            "position": (2.0, 0.0, 0.40),
            "extent_3d": (1.2, 0.8, 0.70),
            "bbox_3d_min": (1.4, -0.4, 0.05),
            "bbox_3d_max": (2.6, 0.4, 0.75),
            "confidence": 0.90,
            "observation_count": 1,
            "first_seen_time": 10.0,
            "last_seen_time": 10.0,
        }
        cup = {
            "object_id": 2,
            "category": "cup",
            "position": (2.0, 0.0, 0.90),
            "extent_3d": (0.12, 0.12, 0.22),
            "bbox_3d_min": (1.94, -0.06, 0.78),
            "bbox_3d_max": (2.06, 0.06, 1.00),
            "confidence": 0.92,
            "observation_count": 1,
            "first_seen_time": 10.0,
            "last_seen_time": 10.0,
        }
        return table, cup

    @staticmethod
    def _observations():
        return [
            {"category": "table", "bbox": (50, 180, 500, 430), "confidence": 0.90},
            {"category": "cup", "bbox": (270, 100, 350, 210), "confidence": 0.92},
        ]

    def test_builds_nodes_edges_and_debug_files(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SceneGraphManager(debug_dir=directory)
            task = {
                "raw": "Find the cup on the table",
                "target": "cup",
                "relation": "on",
                "reference_objects": ["table"],
            }
            manager.start_task(1, task)
            table, cup = self._nodes()
            result = manager.add_observation(
                image_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
                points_sensor=self._scan(),
                pose={"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                timestamp=10.0,
                observations=self._observations(),
                object_ids=[1, 2],
                object_nodes=[table, cup],
                task=task,
            )

            graph = manager.snapshot()
            self.assertTrue(result["viewpoint_created"])
            self.assertGreater(graph["viewpoints"][0]["coverage_voxel_count"], 0)
            self.assertTrue(graph["viewpoints"][0]["coverage_region"])
            self.assertEqual(len(graph["viewpoints"]), 1)
            self.assertEqual(len(graph["objects"]), 2)
            edge_types = {edge["edge_type"] for edge in graph["edges"]}
            self.assertEqual(
                edge_types,
                {"room_viewpoint", "room_object", "viewpoint_object", "object_object"},
            )
            self.assertEqual(manager.find_matching_target_ids(task), [2])
            self.assertTrue(result["relation_edges"])
            self.assertEqual(manager.common_viewpoint_ids([1, 2]), [1])

            for filename in (
                "scene_graph_latest.json",
                "scene_graph_latest.dot",
                "scene_graph_latest.png",
            ):
                self.assertTrue((Path(directory) / filename).is_file())

            saved = json.loads(
                (Path(directory) / "scene_graph_latest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["schema_version"], 2)
            self.assertEqual(saved["room"]["name"], config.SCENE_GRAPH_SINGLE_ROOM_NAME)
            self.assertEqual(len(saved["objects"]), 2)

    def test_repeated_coverage_does_not_create_duplicate_viewpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SceneGraphManager(debug_dir=directory)
            task = {
                "raw": "Find the cup",
                "target": "cup",
                "relation": None,
                "reference_objects": [],
            }
            table, cup = self._nodes()
            first = manager.add_observation(
                image_rgb=np.zeros((120, 240, 3), dtype=np.uint8),
                points_sensor=self._scan(),
                pose={"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                timestamp=10.0,
                observations=self._observations(),
                object_ids=[1, 2],
                object_nodes=[table, cup],
                task=task,
            )
            second = manager.add_observation(
                image_rgb=np.zeros((120, 240, 3), dtype=np.uint8),
                points_sensor=self._scan(),
                pose={"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                timestamp=11.0,
                observations=self._observations(),
                object_ids=[1, 2],
                object_nodes=[table, cup],
                task=task,
            )

            self.assertTrue(first["viewpoint_created"])
            self.assertFalse(second["viewpoint_created"])
            self.assertEqual(second["novel_voxel_count"], 0)
            self.assertEqual(len(manager.snapshot()["viewpoints"]), 1)

    def test_new_coverage_creates_second_viewpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SceneGraphManager(debug_dir=directory)
            task = {"raw": "Find the cup", "target": "cup", "relation": None, "reference_objects": []}
            table, cup = self._nodes()
            manager.add_observation(
                image_rgb=np.zeros((120, 240, 3), dtype=np.uint8),
                points_sensor=self._scan(),
                pose={"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                timestamp=10.0,
                observations=self._observations(),
                object_ids=[1, 2],
                object_nodes=[table, cup],
                task=task,
            )
            result = manager.add_observation(
                image_rgb=np.zeros((120, 240, 3), dtype=np.uint8),
                points_sensor=self._scan(),
                pose={"x": 4.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                timestamp=12.0,
                observations=self._observations(),
                object_ids=[1, 2],
                object_nodes=[table, cup],
                task=task,
            )
            self.assertTrue(result["viewpoint_created"])
            self.assertGreater(result["novel_voxel_count"], config.VIEWPOINT_NOVEL_VOXEL_THRESHOLD)
            self.assertEqual(len(manager.snapshot()["viewpoints"]), 2)

    def test_new_relation_query_reuses_past_common_viewpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SceneGraphManager(debug_dir=directory)
            table, cup = self._nodes()
            category_task = {
                "raw": "Find the cup",
                "target": "cup",
                "relation": None,
                "reference_objects": [],
            }
            manager.add_observation(
                image_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
                points_sensor=self._scan(),
                pose={"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                timestamp=10.0,
                observations=self._observations(),
                object_ids=[1, 2],
                object_nodes=[table, cup],
                task=category_task,
            )

            relation_task = {
                "raw": "Find the cup on the table",
                "target": "cup",
                "relation": "on",
                "reference_objects": ["table"],
            }
            manager.start_task(2, relation_task)
            result = manager.add_observation(
                image_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
                points_sensor=self._scan(),
                pose={"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                timestamp=11.0,
                observations=[],
                object_ids=[],
                object_nodes=[table, cup],
                task=relation_task,
            )

            self.assertFalse(result["viewpoint_created"])
            self.assertTrue(result["relation_edges"])
            self.assertEqual(result["relation_edges"][0]["viewpoint_id"], 1)
            self.assertEqual(manager.find_matching_target_ids(relation_task), [2])
            self.assertEqual(len(manager.snapshot()["viewpoints"]), 1)

    def test_selected_object_is_written_to_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SceneGraphManager(debug_dir=directory)
            manager.mark_selected_object(7)
            saved = json.loads(
                (Path(directory) / "scene_graph_latest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["selected_object_id"], 7)


if __name__ == "__main__":
    unittest.main()
