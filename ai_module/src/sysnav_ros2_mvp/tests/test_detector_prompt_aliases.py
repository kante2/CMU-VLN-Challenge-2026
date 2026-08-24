"""넓은 카테고리("table")를 동의 명사구로 넓혀 쏘고, 결과는 원래 카테고리로 되돌린다."""

import unittest

from sysnav.perception.detector import YoloWorldDetector


class PromptAliasTest(unittest.TestCase):
    def test_table_is_widened_and_mapped_back(self):
        query, alias_to_category = YoloWorldDetector._expand_prompts(["sofa", "table"])

        # 원래 프롬프트는 그대로 남고, 별칭이 뒤에 붙는다.
        self.assertEqual(query[:2], ["sofa", "table"])
        self.assertIn("coffee table", query)
        self.assertEqual(alias_to_category["coffee table"], "table")
        self.assertEqual(alias_to_category["side table"], "table")
        # 별칭이 아닌 프롬프트는 접히지 않는다.
        self.assertNotIn("sofa", alias_to_category)
        self.assertNotIn("table", alias_to_category)

    def test_an_explicitly_asked_alias_stays_its_own_category(self):
        """"coffee table을 찾아줘"라고 물으면 그건 table의 별칭이 아니라 그 자체 카테고리다."""
        query, alias_to_category = YoloWorldDetector._expand_prompts(["coffee table", "table"])

        self.assertNotIn("coffee table", alias_to_category)
        self.assertEqual(query.count("coffee table"), 1)

    def test_categories_without_aliases_are_untouched(self):
        query, alias_to_category = YoloWorldDetector._expand_prompts(["lamp", "chair"])

        self.assertEqual(query, ["lamp", "chair"])
        self.assertEqual(alias_to_category, {})


if __name__ == "__main__":
    unittest.main()
