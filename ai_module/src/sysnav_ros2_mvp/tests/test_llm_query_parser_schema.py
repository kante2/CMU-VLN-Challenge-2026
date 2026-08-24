""""black chair"가 통째로 YOLO 프롬프트가 되던 문제 - 파서가 {category, attributes}로
쪼개는지 검증한다.

normalize_llm_result()는 순수 함수라 Gemini 호출 없이 payload를 손으로 만들어 돌린다.
"""

import unittest

from sysnav.task.llm_query_parser import normalize_llm_result


class LLMQueryParserSchemaTest(unittest.TestCase):
    def test_color_adjective_is_split_out_of_the_reference_category(self):
        parsed = normalize_llm_result(
            "go near the lamp closest to the black chair",
            {
                "target": {"category": "lamp", "attributes": []},
                "constraints": [
                    {
                        "relation": "closest_to",
                        "references": [{"category": "chair", "attributes": ["black"]}],
                    }
                ],
            },
        )

        self.assertEqual(parsed["target"], "lamp")
        self.assertEqual(parsed["reference_objects"], ["chair"])
        self.assertEqual(parsed["relation"], "nearest")
        self.assertEqual(parsed["reference_attributes"], {"chair": ["black"]})
        # 검출기에 나가는 프롬프트에는 형용사가 하나도 없어야 한다.
        self.assertEqual(parsed["detection_prompts"], ["lamp", "chair"])

    def test_plural_and_shape_adjective_are_normalized(self):
        parsed = normalize_llm_result(
            "take the path between the sofa and the round tables",
            {
                "target": {"category": "sofa", "attributes": []},
                "constraints": [
                    {
                        "relation": "between",
                        "references": [
                            {"category": "sofa", "attributes": []},
                            {"category": "tables", "attributes": ["round"]},
                        ],
                    }
                ],
            },
        )

        # "tables" -> "table": object_memory/scene_graph가 카테고리 문자열을 키로 쓰므로
        # 규칙 파서(query_parser._singularize)와 표기가 같아야 한다.
        self.assertIn("table", parsed["detection_prompts"])
        self.assertNotIn("tables", parsed["detection_prompts"])
        self.assertEqual(parsed["reference_attributes"], {"table": ["round"]})
        self.assertEqual(
            parsed["relation_chain"],
            [("sofa", "between", "sofa"), ("sofa", "between", "table")],
        )

    def test_multiword_nouns_are_not_split(self):
        parsed = normalize_llm_result(
            "find the bowl near the trash can",
            {
                "target": {"category": "bowl", "attributes": []},
                "constraints": [
                    {"relation": "near", "references": [{"category": "trash can", "attributes": []}]}
                ],
            },
        )

        self.assertEqual(parsed["detection_prompts"], ["bowl", "trash can"])
        # 속성 요구가 없으면 화이트리스트 키를 만들지 않는다(= 제한 없음).
        self.assertEqual(parsed["reference_attributes"], {})

    def test_target_attributes_are_kept(self):
        parsed = normalize_llm_result(
            "find the white pillow",
            {"target": {"category": "pillows", "attributes": ["white"]}, "constraints": []},
        )

        self.assertEqual(parsed["target"], "pillow")
        self.assertEqual(parsed["attributes"], ["white"])

    def test_old_string_payload_still_parses(self):
        """스키마를 바꾸기 전 형식으로 답하는 모델이 있어도 파이프라인이 안 깨져야 한다."""
        parsed = normalize_llm_result(
            "find the chair near the window",
            {
                "target": "chair",
                "attributes": ["black"],
                "constraints": [{"relation": "near", "references": ["window"]}],
            },
        )

        self.assertEqual(parsed["target"], "chair")
        self.assertEqual(parsed["attributes"], ["black"])
        self.assertEqual(parsed["reference_objects"], ["window"])


if __name__ == "__main__":
    unittest.main()
