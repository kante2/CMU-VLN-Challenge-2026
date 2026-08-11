import unittest

from sysnav.task.llm_query_parser import _PROMPT_TEMPLATE
from sysnav.task.query_parser import extract_target, requires_comparative_ranking


class QueryParserTest(unittest.TestCase):
    def test_llm_prompt_json_example_survives_question_formatting(self):
        rendered = _PROMPT_TEMPLATE.format(question="Find the pillow.")
        self.assertIn('{"relation":"on", "references":["chair"]}', rendered)

    def test_nested_relative_clause_is_kept_as_two_hop_chain(self):
        parsed = extract_target(
            "Find the pillow on the chair that is closest to the TV."
        )
        self.assertEqual(parsed["target"], "pillow")
        self.assertEqual(
            parsed["relation_chain"],
            [("pillow", "on", "chair"), ("chair", "nearest", "tv")],
        )
        self.assertEqual(parsed["detection_prompts"], ["pillow", "chair", "tv"])

    def test_attribute(self):
        result = extract_target("Find the white chair.")
        self.assertEqual(result["target"], "chair")
        self.assertEqual(result["attributes"], ["white"])
        self.assertEqual(result["detection_prompts"], ["chair"])

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

    def test_closest_requires_comparing_target_instances(self):
        result = extract_target("the bedside table closest to the window")
        self.assertTrue(requires_comparative_ranking(result))

    def test_later_closest_does_not_rank_target_instances(self):
        result = extract_target("the bowl on the table closest to the window")
        self.assertFalse(requires_comparative_ranking(result))


if __name__ == "__main__":
    unittest.main()
