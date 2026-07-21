import unittest

from sysnav.task.query_parser import extract_target


class QueryParserTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
