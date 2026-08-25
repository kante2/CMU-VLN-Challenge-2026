""""the cabinet with a picture above **it**"의 대명사를 선행사로 되돌린다.

questions.json 75문장 중 대명사가 나오는 18문장은 예외 없이 "X with Y on/above it"
꼴이고, it/them은 **같은 명사구의 head noun**을 가리킨다(앞 절이 아니다). 그런데 그
head noun은 이미 "cabinet with picture"처럼 카테고리 문자열에 통째로 fuse돼 있어서,
선행사로 되돌리면 relation이 자기 자신을 가리킨다 - 정보가 없으므로 relation째로 버린다.

예전엔 "it"이 그대로 참조 카테고리이자 YOLO 프롬프트가 됐다. find_by_category("it")은
영원히 비어 있어서 mission3의 _missing_categories가 그 step을 절대 확정하지 못하고
탐사만 반복했다(실측 2026-08-25: step 3이 FOLLOW_EXPLORATION에서 안 빠져나옴).
"""

import sys
import types
import unittest

# llm_query_parser는 로깅에만 rclpy를 쓴다. ROS 없는 환경에서도 파싱 로직을 검증할 수
# 있도록 최소 stub으로 대체한다 (tests/test_detection_verify_cache.py와 같은 패턴).
if "rclpy" not in sys.modules:
    try:
        import rclpy  # noqa: F401
    except ImportError:                                       # pragma: no cover
        package = types.ModuleType("rclpy")
        logging_module = types.ModuleType("rclpy.logging")

        class _StubLogger:
            def info(self, *args, **kwargs): pass
            def warning(self, *args, **kwargs): pass
            def error(self, *args, **kwargs): pass

        logging_module.get_logger = lambda name: _StubLogger()
        package.logging = logging_module
        sys.modules["rclpy"] = package
        sys.modules["rclpy.logging"] = logging_module

from sysnav.task.llm_query_parser import normalize_llm_result   # noqa: E402
from sysnav.task.query_parser import extract_target             # noqa: E402


class RuleParserPronounTest(unittest.TestCase):
    def _parse(self, text):
        return extract_target(text)

    def test_the_reported_clause_drops_the_self_relation(self):
        parsed = self._parse("the cabinet with a picture above it")
        self.assertEqual(parsed["target"], "cabinet with picture")
        self.assertIsNone(parsed["relation"])
        self.assertEqual(parsed["reference_objects"], [])
        self.assertEqual(parsed["detection_prompts"], ["cabinet with picture"])

    def test_no_pronoun_survives_as_a_detection_prompt(self):
        """questions.json에 실제로 있는 대명사 문장 전부."""
        cases = [
            "the tea table with the elephant figurine on it",
            "the coffee table with the kettle on it",
            "the nightstand with a clock on it",
            "the small table with a vase on it",
            "the table with the flowers on it",
            "the cabinet with a picture above it",
            "Count the number of chairs with pillows on them.",
            "Find the computer monitor closest to the cabinet with a phone on it.",
            "Find the potted plant between a vase and the cabinet with a TV on it.",
        ]
        for text in cases:
            with self.subTest(text=text):
                prompts = self._parse(text)["detection_prompts"]
                self.assertNotIn("it", prompts)
                self.assertNotIn("them", prompts)

    def test_a_real_relation_before_the_pronoun_is_kept(self):
        """대명사만 떨어져 나가고 앞의 진짜 relation은 그대로 남아야 한다."""
        parsed = self._parse("the lamp on the nightstand that has the photo on it")
        self.assertEqual(parsed["target"], "lamp")
        self.assertEqual(parsed["relation"], "on")
        self.assertEqual(parsed["reference_objects"], ["nightstand has photo"])
        self.assertEqual(parsed["relation_chain"], [("lamp", "on", "nightstand has photo")])

    def test_a_sentence_without_pronouns_is_unchanged(self):
        parsed = self._parse("the pillow closest to the book on the stool")
        self.assertEqual(parsed["target"], "pillow")
        self.assertEqual(parsed["reference_objects"], ["book", "stool"])

    def test_between_references_are_unaffected(self):
        parsed = self._parse("the wall lamp that is between a door frame and a window")
        self.assertEqual(parsed["relation"], "between")
        self.assertEqual(parsed["reference_objects"], ["door frame", "window"])


class LlmParserPronounTest(unittest.TestCase):
    def test_a_pronoun_reference_from_the_llm_is_dropped(self):
        """LLM이 대명사를 참조 카테고리로 내놓아도 규칙 파서와 같게 동작한다."""
        parsed = normalize_llm_result(
            "stop at the cabinet with a picture above it",
            {
                "target": {"category": "cabinet with picture", "attributes": []},
                "constraints": [
                    {"relation": "above", "references": [{"category": "it", "attributes": []}]}
                ],
            },
        )
        self.assertEqual(parsed["detection_prompts"], ["cabinet with picture"])
        self.assertIsNone(parsed["relation"])
        self.assertEqual(parsed["relation_chain"], [])

    def test_a_mixed_reference_list_keeps_the_real_category(self):
        parsed = normalize_llm_result(
            "the monitor closest to the cabinet with a phone on it",
            {
                "target": {"category": "computer monitor", "attributes": []},
                "constraints": [
                    {
                        "relation": "closest to",
                        "references": [
                            {"category": "cabinet with phone", "attributes": []},
                            {"category": "it", "attributes": []},
                        ],
                    }
                ],
            },
        )
        self.assertEqual(parsed["reference_objects"], ["cabinet with phone"])
        self.assertNotIn("it", parsed["detection_prompts"])

    def test_a_normal_reference_is_untouched(self):
        parsed = normalize_llm_result(
            "the lamp closest to the black chair",
            {
                "target": {"category": "lamp", "attributes": []},
                "constraints": [
                    {"relation": "closest to",
                     "references": [{"category": "chair", "attributes": ["black"]}]}
                ],
            },
        )
        self.assertEqual(parsed["reference_objects"], ["chair"])
        self.assertEqual(parsed["reference_attributes"], {"chair": ["black"]})


if __name__ == "__main__":
    unittest.main()
