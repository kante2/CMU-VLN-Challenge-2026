import unittest

from sysnav.perception.perception_pipeline import PerceptionPipeline
from sysnav.task.llm_visual_aliases import normalize_alias_payload


class VisualAliasTests(unittest.TestCase):
    def test_aliases_are_filtered_and_safe_alias_is_always_present(self):
        prompts, canonical_by_prompt, aliases = normalize_alias_payload(
            ["painting", "tv"],
            {
                "categories": [
                    {
                        "canonical": "painting",
                        "aliases": [
                            "wall picture",
                            "TV",
                            "picture closest to the television",
                        ],
                    },
                    {"canonical": "unknown", "aliases": ["wall decor"]},
                ]
            },
        )

        self.assertEqual(prompts, ["painting", "wall picture", "tv"])
        self.assertEqual(canonical_by_prompt["wall picture"], "painting")
        self.assertEqual(aliases["painting"], ["wall picture"])

    def test_alias_detections_are_canonicalized_and_deduplicated(self):
        detections = [
            {"category": "painting", "confidence": 0.60, "bbox": [10, 10, 50, 50]},
            {"category": "wall picture", "confidence": 0.84, "bbox": [11, 10, 51, 50]},
            {"category": "tv", "confidence": 0.90, "bbox": [100, 10, 150, 60]},
        ]

        result = PerceptionPipeline._canonicalize_and_deduplicate(
            detections,
            {"painting": "painting", "wall picture": "painting", "tv": "tv"},
        )

        self.assertEqual(len(result), 2)
        painting = next(item for item in result if item["category"] == "painting")
        self.assertEqual(painting["detected_as"], "wall picture")
        self.assertAlmostEqual(painting["confidence"], 0.84)


if __name__ == "__main__":
    unittest.main()
