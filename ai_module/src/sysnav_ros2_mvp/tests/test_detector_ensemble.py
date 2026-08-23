"""Selective YOLO12 COCO routing and cross-model duplicate suppression."""

import unittest

from sysnav.perception.detector import YoloWorldDetector, _bbox_iou, _merge_detections


class CocoPromptRoutingTest(unittest.TestCase):
    def test_books_plural_maps_to_coco_book_and_keeps_prompt_name(self):
        self.assertEqual(
            YoloWorldDetector._coco_prompts(["potted plant", "books", "cabinet"]),
            {"potted plant": "potted plant", "book": "books"},
        )

    def test_non_coco_prompt_does_not_trigger_auxiliary_detection(self):
        self.assertEqual(YoloWorldDetector._coco_prompts(["knife rack", "cabinet"]), {})

    def test_table_alias_runs_coco_dining_table_and_keeps_query_category(self):
        self.assertEqual(
            YoloWorldDetector._coco_prompts(["table"]),
            {"dining table": "table"},
        )

    def test_tables_plural_maps_to_coco_dining_table(self):
        self.assertEqual(
            YoloWorldDetector._coco_prompts(["tables"]),
            {"dining table": "tables"},
        )


class MergeTest(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(_bbox_iou((0, 0, 100, 100), (0, 0, 100, 100)), 1.0)
        self.assertEqual(_bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_overlapping_same_category_keeps_higher_confidence(self):
        detections = [
            {"category": "books", "confidence": 0.30, "bbox": (10, 10, 50, 50)},
            {"category": "books", "confidence": 0.60, "bbox": (11, 11, 51, 51)},
        ]
        self.assertEqual(_merge_detections(detections), [detections[1]])

    def test_different_categories_are_not_merged(self):
        detections = [
            {"category": "book", "confidence": 0.30, "bbox": (10, 10, 50, 50)},
            {"category": "bottle", "confidence": 0.60, "bbox": (10, 10, 50, 50)},
        ]
        self.assertEqual(len(_merge_detections(detections)), 2)


if __name__ == "__main__":
    unittest.main()
