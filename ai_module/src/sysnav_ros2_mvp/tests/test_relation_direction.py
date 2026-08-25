"""관계 이미지 검증이 관계를 **뒤집어** 묻던 버그.

relation chain은 (source, relation, reference)이고 "source가 reference에 대해
relation이다"로 읽는다 - "the cabinet with a picture above it"은
(cabinet, under, picture) = "캐비닛이 그림 아래에 있다"로 파싱된다.

그런데 프롬프트가 "a {reference} is {relation} the object shown"이라 실제로는
"picture가 cabinet **아래** 보이는가"를 물었다 - 정확히 반대다. 실측 2026-08-25:
정답 쌍인데 Gemini가 전부 false를 냈고, 그 질문에 대해서는 그게 맞는 답이었다.
near/between은 대칭이라 우연히 무사했지만 on/under/above/behind/in_front_of/
supports는 전부 뒤집혀 있었다.
"""

import sys
import types
import unittest

# relation_image_verifier는 로깅/이미지에만 rclpy·cv2를 쓴다.
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

for _name in ("cv2",):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except ImportError:                                   # pragma: no cover
            sys.modules[_name] = types.ModuleType(_name)

from sysnav.reasoning.relation_image_verifier import RelationImageVerifier  # noqa: E402


class PromptDirectionTest(unittest.TestCase):
    """프롬프트 문자열을 직접 조립해 방향을 검사한다 - Gemini 호출은 하지 않는다."""

    @staticmethod
    def _sentence(relation: str, reference: str) -> str:
        return (
            f"the object shown in that image is visibly "
            f"{RelationImageVerifier._phrase(relation)} a {reference}"
        )

    def test_the_candidate_is_the_subject_not_the_reference(self):
        """보고된 버그. 주어가 후보(cabinet)여야 한다."""
        sentence = self._sentence("under", "picture")
        self.assertEqual(sentence, "the object shown in that image is visibly under a picture")
        # 예전 문장은 "a picture is visibly under the object shown"이었다.
        self.assertNotIn("a picture is visibly under", sentence)

    def test_asymmetric_relations_keep_their_direction(self):
        for relation, expected in (
            ("under", "under a picture"),
            ("above", "above a picture"),
            ("on", "on a picture"),
            ("behind", "behind a picture"),
            ("in_front_of", "in front of a picture"),
        ):
            with self.subTest(relation=relation):
                self.assertTrue(self._sentence(relation, "picture").endswith(expected))

    def test_relations_that_need_rewording_are_readable(self):
        self.assertIn("resting on top", self._sentence("supports", "vase"))
        self.assertIn("the closest one to", self._sentence("nearest", "window"))
        self.assertIn("the farthest one from", self._sentence("farthest", "column"))

    def test_the_source_prompt_actually_uses_this_wording(self):
        """프롬프트 본문이 바뀌면(다시 뒤집히면) 여기서 잡는다."""
        import inspect
        body = inspect.getsource(RelationImageVerifier.verify)
        self.assertIn('f"For each image below, decide whether the object shown in that image is "', body)
        self.assertIn('f"visibly {relation_phrase} a {reference_category} (the object may be "', body)


class CacheInvalidationTest(unittest.TestCase):
    def test_the_cache_key_was_bumped_so_old_verdicts_are_discarded(self):
        """이전 키로 적립된 판정은 전부 '뒤집힌 질문'의 답이라 재사용하면 안 된다."""
        candidate = {"object_id": 7, "image_version": 3}
        key = RelationImageVerifier._verify_key(candidate, "under", "picture")
        self.assertTrue(key.startswith("verify2|"), key)
        self.assertFalse(key.startswith("verify|"), key)

    def test_the_key_still_tracks_the_image_version(self):
        older = RelationImageVerifier._verify_key({"object_id": 7, "image_version": 3}, "under", "picture")
        newer = RelationImageVerifier._verify_key({"object_id": 7, "image_version": 4}, "under", "picture")
        self.assertNotEqual(older, newer)


if __name__ == "__main__":
    unittest.main()
