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


class SuperlativeRelationTest(unittest.TestCase):
    """farthest/furthest는 nearest의 반대쪽 최상급인데 _RELATIONS에 없었다.

    그래서 관계 매칭이 통째로 실패해 "table farthest from the columns" 전체가 target이
    되고, 그 문자열이 그대로 YOLO 프롬프트로 나갔다(questions.json 75문항 중 9문항).
    검출과 관계 판정을 동시에 잃는 경로였다.
    """

    def test_farthest_is_parsed_as_a_relation(self):
        for text in (
            "Find the table farthest from the columns.",
            "the bedside table furthest from the window",
            "the beer bottle furthest from the couch",
        ):
            with self.subTest(text=text):
                parsed = extract_target(text)
                self.assertEqual(parsed["relation"], "farthest")
                # 프롬프트는 검출 가능한 명사여야 한다 - 관계구가 섞이면 안 된다.
                for prompt in parsed["detection_prompts"]:
                    self.assertNotIn("farthest", prompt)
                    self.assertNotIn("furthest", prompt)

    def test_farthest_target_and_reference_are_separated(self):
        parsed = extract_target("Find the table farthest from the columns.")
        self.assertEqual(parsed["target"], "table")
        self.assertEqual(parsed["reference_objects"], ["column"])
        self.assertEqual(parsed["detection_prompts"], ["table", "column"])

    def test_nearest_still_works(self):
        parsed = extract_target("Find the picture closest to the bed.")
        self.assertEqual(parsed["relation"], "nearest")
        self.assertEqual(parsed["target"], "picture")
        self.assertEqual(parsed["reference_objects"], ["bed"])


class NestedBetweenReferenceTest(unittest.TestCase):
    def test_relative_clause_in_a_between_reference_is_trimmed(self):
        """"between A and B that is closest to A"에서 B 쪽 관계절을 떼지 않으면
        참조 카테고리가 "stone decoration closest to vase"가 되어 scene graph
        카테고리와도 안 맞고 YOLO 프롬프트로도 못 쓴다."""
        parsed = extract_target(
            "The lantern between the vase and the stone decoration that is closest to the vase."
        )
        self.assertEqual(parsed["relation"], "between")
        self.assertEqual(parsed["reference_objects"], ["vase", "stone decoration"])

    def test_plain_between_is_unchanged(self):
        parsed = extract_target("Find the wall lamp that is between a door frame and a window.")
        self.assertEqual(parsed["reference_objects"], ["door frame", "window"])
