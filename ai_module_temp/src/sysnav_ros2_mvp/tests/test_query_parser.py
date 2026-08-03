import unittest

from sysnav.task.gemini_query_parser import normalize_gemini_result
from sysnav.task.query_parser import extract_target


class QueryParserTest(unittest.TestCase):
    def test_attribute(self):
        result = extract_target("Find the white chair.")
        self.assertEqual(result["target"], "chair")
        self.assertEqual(result["attributes"], ["white"])
        self.assertEqual(result["detection_prompts"], ["chair", "white chair"])
        self.assertEqual(result["prompt_categories"]["white chair"], "chair")

    def test_relation(self):
        result = extract_target("Find the chair beside the table.")
        self.assertEqual(result["target"], "chair")
        self.assertEqual(result["relation"], "beside")
        self.assertEqual(result["reference_objects"], ["table"])
        self.assertEqual(result["detection_prompts"], ["chair", "table"])

    def test_between(self):
        result = extract_target("Find the pillow between the sofa and the table.")
        self.assertEqual(result["target"], "pillow")
        self.assertEqual(result["reference_objects"], ["sofa", "table"])

    def test_aliases_and_canonical_mapping(self):
        result = extract_target(
            "Find the bowl closest to the knife rack near the trash can."
        )
        self.assertEqual(result["target"], "bowl")
        self.assertEqual(
            result["relation_chain"],
            [
                ("bowl", "nearest", "knife rack"),
                ("knife rack", "near", "trash can"),
            ],
        )
        self.assertIn("knife block", result["detection_prompts"])
        self.assertIn("garbage bin", result["detection_prompts"])
        self.assertEqual(result["prompt_categories"]["knife block"], "knife rack")
        self.assertEqual(result["prompt_categories"]["garbage bin"], "trash can")

    def test_gemini_structure_preserves_relation_chain(self):
        result = normalize_gemini_result(
            "find the bowl closest to the knife rack near the trash can",
            {
                "target": "bowl",
                "attributes": [],
                "constraints": [
                    {"relation": "closest_to", "references": ["knife rack"]},
                    {"relation": "near", "references": ["trash can"]},
                ],
            },
            {
                "bowl": ["bowl", "dish"],
                "knife rack": ["knife rack", "knife block"],
                "trash can": ["trash can", "garbage bin"],
            },
        )
        self.assertEqual(result["relation"], "nearest")
        self.assertEqual(
            result["relation_chain"],
            [
                ("bowl", "nearest", "knife rack"),
                ("knife rack", "near", "trash can"),
            ],
        )
        self.assertEqual(result["prompt_categories"]["knife block"], "knife rack")


if __name__ == "__main__":
    unittest.main()
