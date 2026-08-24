"""이미지 기반 relation 판정은 같은 사진에 같은 질문을 두 번 묻지 않는다.

이 폴백(reasoning/relation_image_verifier.py)은 참조 물체가 끝내 3D grounding이
안 되는 경우(유리창처럼 LiDAR 반사가 없는 물체)를 위한 것이다. 그런데 바로 그
경우에 selection_job이 relation_pending을 반환하고 -> PLAN_EXPLORATION -> OBSERVE ->
다시 SELECT_TARGET으로 되돌아온다. 캐시가 없으면 이 사이클마다 같은 후보의 같은
사진을 같은 질문으로 Gemini에 다시 올렸다(mission3는 step마다 새로 시작해서 더 심함).

attribute_verifier와 같은 패턴으로 고쳤다: 판정 결과는 object_memory 노드의
`relation_checks`에 적립되고, 캐시 키에 노드의 `image_version`이 들어가서 **사진이
교체될 때만** 다시 묻는다.
"""

import sys
import types
import unittest

import numpy as np

# relation_image_verifier는 로깅에만 rclpy를 쓴다 (test_detection_verify_cache.py와
# 같은 패턴 - ROS 없는 환경에서도 캐시 로직을 검증할 수 있게 stub으로 대체).
if "rclpy" not in sys.modules:
    try:
        import rclpy  # noqa: F401
    except ImportError:                                       # pragma: no cover
        package = types.ModuleType("rclpy")
        logging_module = types.ModuleType("rclpy.logging")

        class _Logger:
            def info(self, *args, **kwargs): pass
            def warning(self, *args, **kwargs): pass

        logging_module.get_logger = lambda name: _Logger()
        package.logging = logging_module
        sys.modules["rclpy"] = package
        sys.modules["rclpy.logging"] = logging_module

from sysnav.reasoning.relation_image_verifier import RelationImageVerifier  # noqa: E402


class _FakeVerifier(RelationImageVerifier):
    """Gemini 왕복만 가짜로 바꾼다. 캐시 경로는 실제 코드를 그대로 탄다.

    google.genai가 없는 환경이라 verify()/rank_superlative() 안의 `from google.genai
    import types`도 stub이 필요하다.
    """

    def __init__(self, response_text: str):
        super().__init__()
        self.api_key = "test-key"
        self.response_text = response_text
        self.asked: list[list[int]] = []      # 실제로 Gemini에 물어본 object_id 묶음
        self._install_genai_stub()

    def _install_genai_stub(self) -> None:
        verifier = self

        class _Part:
            @staticmethod
            def from_bytes(data, mime_type):
                return ("image", len(data))

        class _Types:
            Part = _Part

            @staticmethod
            def GenerateContentConfig(**kwargs):
                return kwargs

        class _Models:
            @staticmethod
            def generate_content(model, contents, config):
                asked = [
                    int(str(item).split("=")[1].split(" ")[0])
                    for item in contents
                    if isinstance(item, str) and item.startswith("object_id=")
                ]
                verifier.asked.append(asked)
                return types.SimpleNamespace(text=verifier.response_text)

        self._client = types.SimpleNamespace(models=_Models())
        genai_types = types.ModuleType("google.genai.types")
        genai_types.Part = _Types.Part
        genai_types.GenerateContentConfig = _Types.GenerateContentConfig
        genai_module = types.ModuleType("google.genai")
        genai_module.types = genai_types
        google_module = sys.modules.get("google") or types.ModuleType("google")
        google_module.genai = genai_module
        sys.modules.setdefault("google", google_module)
        sys.modules["google.genai"] = genai_module
        sys.modules["google.genai.types"] = genai_types

    def _load(self):
        return None

    @staticmethod
    def _jpeg(image_rgb):
        return b"jpeg"


def _candidate(object_id: int, image_version: int = 0, relation_checks=None) -> dict:
    return {
        "object_id": object_id,
        "context_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "image_version": image_version,
        "relation_checks": dict(relation_checks or {}),
    }


def _apply(candidates: list[dict], results: dict) -> None:
    """object_memory.update_relation_checks()가 하는 일을 테스트에서 흉내낸다."""
    by_id = {int(c["object_id"]): c for c in candidates}
    for object_id, checks in results.items():
        by_id[object_id]["relation_checks"].update(checks)


class VerifyCacheTest(unittest.TestCase):
    RESPONSE = '{"results": [{"object_id": 1, "holds": true}, {"object_id": 2, "holds": false}]}'

    def setUp(self):
        self.verifier = _FakeVerifier(self.RESPONSE)
        self.candidates = [_candidate(1), _candidate(2)]

    def _verify(self):
        results = self.verifier.verify(self.candidates, "near", "window")
        _apply(self.candidates, results)
        return {object_id for object_id, checks in results.items() if any(checks.values())}

    def test_first_call_asks_and_returns_the_passing_id(self):
        self.assertEqual(self._verify(), {1})
        self.assertEqual(self.verifier.asked, [[1, 2]])

    def test_repeated_calls_never_ask_again(self):
        self._verify()
        for _ in range(5):
            self.assertEqual(self._verify(), {1}, "캐시 결과가 첫 판정과 같아야 한다")
        self.assertEqual(len(self.verifier.asked), 1, "같은 사진은 한 번만 물어본다")

    def test_negative_verdict_is_cached_too(self):
        """False도 돈 주고 얻은 판정이다 - 다시 묻지 않는다."""
        self._verify()
        self._verify()
        self.assertEqual(len(self.verifier.asked), 1)
        self.assertFalse(any(self.candidates[1]["relation_checks"].values()))

    def test_new_photo_invalidates_that_candidate_only(self):
        self._verify()
        self.candidates[1]["image_version"] += 1       # 더 좋은 사진으로 교체됨
        self._verify()
        self.assertEqual(
            self.verifier.asked[-1], [2],
            "사진이 바뀐 후보만 다시 물어봐야 한다",
        )

    def test_a_different_question_is_asked_separately(self):
        self._verify()
        self.verifier.verify(self.candidates, "near", "door")   # 다른 참조 물체
        self.assertEqual(len(self.verifier.asked), 2)

    def test_vlm_failure_is_not_cached(self):
        """실패는 fail-closed로 남기고 캐시하지 않는다 - 다음 기회에 재시도해야 한다."""
        self.verifier.response_text = ""                        # 빈 응답 -> 예외
        self.assertEqual(self._verify(), set())
        self.assertEqual(self.candidates[0]["relation_checks"], {})
        self.verifier.response_text = self.RESPONSE
        self.assertEqual(self._verify(), {1})

    def test_candidates_without_context_image_are_ignored(self):
        candidate = _candidate(3)
        candidate["context_image"] = None
        self.assertEqual(self.verifier.verify([candidate], "near", "window"), {})
        self.assertEqual(self.verifier.asked, [])


class RankSuperlativeCacheTest(unittest.TestCase):
    RESPONSE = '{"object_id": 2, "reference_visible_in_any": true}'

    def setUp(self):
        self.verifier = _FakeVerifier(self.RESPONSE)
        self.candidates = [_candidate(1), _candidate(2)]

    def _rank(self):
        results = self.verifier.rank_superlative(self.candidates, "window", "nearest")
        _apply(self.candidates, results)
        return {object_id for object_id, checks in results.items() if any(checks.values())}

    def test_winner_is_returned_and_cached(self):
        self.assertEqual(self._rank(), {2})
        for _ in range(4):
            self.assertEqual(self._rank(), {2})
        self.assertEqual(len(self.verifier.asked), 1, "같은 후보 집합은 한 번만 비교한다")

    def test_reference_visible_in_none_is_cached_as_all_false(self):
        self.verifier.response_text = '{"object_id": 0, "reference_visible_in_any": false}'
        self.assertEqual(self._rank(), set())
        self.assertEqual(self._rank(), set())
        self.assertEqual(len(self.verifier.asked), 1)

    def test_a_changed_candidate_set_is_compared_again(self):
        """최상급은 집합 전체를 놓고 내린 판정이라, 후보가 늘면 다시 비교해야 한다."""
        self._rank()
        self.candidates.append(_candidate(3))
        self._rank()
        self.assertEqual(len(self.verifier.asked), 2)

    def test_a_new_photo_in_the_set_invalidates_the_comparison(self):
        self._rank()
        self.candidates[0]["image_version"] += 1
        self._rank()
        self.assertEqual(len(self.verifier.asked), 2)

    def test_fewer_than_two_candidates_is_not_a_comparison(self):
        self.assertEqual(self.verifier.rank_superlative([_candidate(1)], "window"), {})
        self.assertEqual(self.verifier.asked, [])


class ObjectMemoryImageVersionTest(unittest.TestCase):
    """캐시 무효화 신호(image_version)와 적립 경로(update_relation_checks)."""

    def setUp(self):
        from sysnav.memory.object_memory import ObjectMemory
        self.memory = ObjectMemory()

    @staticmethod
    def _observation(confidence: float, fill: int):
        image = np.full((4, 4, 3), fill, dtype=np.uint8)
        return {
            "category": "vase",
            "position": (1.0, 2.0, 0.5),
            "point_cloud": np.zeros((3, 3), dtype=np.float32),
            "bbox": (10, 10, 20, 20),
            "confidence": confidence,
            "crop_image": image,
            "context_image": image,
        }

    def test_version_starts_at_zero_and_bumps_when_the_photo_is_replaced(self):
        object_id = self.memory.update([self._observation(0.5, 10)])[0]
        self.assertEqual(self.memory.get(object_id)["image_version"], 0)

        self.memory.update([self._observation(0.9, 20)])          # 더 좋은 사진
        self.assertEqual(self.memory.get(object_id)["image_version"], 1)

    def test_a_worse_photo_does_not_bump_the_version(self):
        """사진이 안 바뀌면 판정도 그대로다 - 캐시를 헛되이 버리면 안 된다."""
        object_id = self.memory.update([self._observation(0.9, 10)])[0]
        self.memory.update([self._observation(0.2, 20)])          # 더 나쁜 사진
        self.assertEqual(self.memory.get(object_id)["image_version"], 0)

    def test_equal_confidence_still_bumps(self):
        """교체 조건이 `>=`라 confidence가 같아도 사진은 바뀐다 - confidence를 버전
        대신 쓰면 안 되는 이유다."""
        object_id = self.memory.update([self._observation(0.9, 10)])[0]
        self.memory.update([self._observation(0.9, 20)])
        self.assertEqual(self.memory.get(object_id)["image_version"], 1)

    def test_relation_checks_accumulate(self):
        object_id = self.memory.update([self._observation(0.9, 10)])[0]
        self.assertEqual(self.memory.get(object_id)["relation_checks"], {})
        self.memory.update_relation_checks(object_id, {"verify|near|window|v0": True})
        self.memory.update_relation_checks(object_id, {"verify|near|door|v0": False})
        self.assertEqual(
            self.memory.get(object_id)["relation_checks"],
            {"verify|near|window|v0": True, "verify|near|door|v0": False},
        )

    def test_unknown_object_id_is_ignored(self):
        self.memory.update_relation_checks(999, {"k": True})      # 예외 없이 무시


if __name__ == "__main__":
    unittest.main()
