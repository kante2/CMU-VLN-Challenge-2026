"""Gemini 검출 재확인은 같은 박스를 두 번 묻지 않는다.

검증 대상은 이미 질문 관련 카테고리로 좁혀져 있다 - detection_prompts가
[target, *reference_objects]라 YOLO-World 자체가 그 카테고리만 검출한다. 그런데도
비용이 컸던 이유는 **같은 박스를 프레임마다 다시 묻기** 때문이다.
실측(2026-08-23 06:31:24~06:32:09): 45초 중 Gemini 재확인 7회에 약 30초.
"""

import sys
import types
import unittest

import numpy as np

# detection_verifier는 로깅에만 rclpy를 쓴다. ROS 없는 환경에서도 캐시 로직을
# 검증할 수 있도록 최소 stub으로 대체한다 (tests/test_terrain_snap.py와 같은 패턴).
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

for name in ("cv2",):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:                                   # pragma: no cover
            sys.modules[name] = types.ModuleType(name)

from sysnav import config                                     # noqa: E402
from sysnav.perception.detection_verifier import DetectionVerifier  # noqa: E402


class _Verifier(DetectionVerifier):
    """Gemini 왕복만 가짜로 바꾼다. 캐시 경로는 실제 코드를 그대로 탄다."""

    def __init__(self, verdicts):
        super().__init__()
        self.api_key = "test-key"
        self.verdicts = verdicts          # 카테고리 -> confirmed
        self.calls = []                   # 실제로 물어본 detection 묶음들

    def _load(self):
        return None

    def verify(self, image_rgb, detections):
        # 부모의 캐시 로직만 재사용하고 네트워크 부분은 이 stub이 대신한다.
        if not detections:
            return []
        results, pending = [], []
        for index, detection in enumerate(detections):
            cached = self._cached(detection)
            results.append(cached)
            if cached is None:
                pending.append(index)
        self.cache_hits += len(detections) - len(pending)
        self.cache_misses += len(pending)
        if not pending:
            return [bool(v) for v in results]
        self.calls.append([detections[i]["category"] for i in pending])
        for index in pending:
            verdict = self.verdicts.get(detections[index]["category"], True)
            results[index] = verdict
            self._remember(detections[index], verdict)
        return [True if v is None else v for v in results]


def _det(category, x1=100, y1=100, x2=200, y2=200, confidence=0.30):
    return {"category": category, "bbox": (x1, y1, x2, y2), "confidence": confidence}


_IMAGE = np.zeros((4, 4, 3), dtype=np.uint8)


class CacheTest(unittest.TestCase):
    def test_identical_box_is_asked_once(self):
        verifier = _Verifier({"vase": True})
        for _ in range(5):
            verifier.verify(_IMAGE, [_det("vase")])
        self.assertEqual(len(verifier.calls), 1, "같은 박스는 한 번만 물어본다")
        self.assertEqual(verifier.cache_hits, 4)

    def test_cached_verdict_is_returned_faithfully(self):
        verifier = _Verifier({"cabinet": False})
        first = verifier.verify(_IMAGE, [_det("cabinet")])
        second = verifier.verify(_IMAGE, [_det("cabinet")])
        self.assertEqual(first, [False])
        self.assertEqual(second, [False], "캐시가 판정을 뒤집으면 안 된다")

    def test_a_nearby_box_within_one_quantum_reuses_the_answer(self):
        quant = config.DETECTION_VERIFICATION_CACHE_BBOX_QUANT_PX
        verifier = _Verifier({"vase": True})
        verifier.verify(_IMAGE, [_det("vase", 100, 100, 200, 200)])
        verifier.verify(_IMAGE, [_det("vase", 100 + quant // 4, 100, 200, 200)])
        self.assertEqual(len(verifier.calls), 1)

    def test_a_box_that_moved_far_is_asked_again(self):
        quant = config.DETECTION_VERIFICATION_CACHE_BBOX_QUANT_PX
        verifier = _Verifier({"vase": True})
        verifier.verify(_IMAGE, [_det("vase", 100, 100, 200, 200)])
        verifier.verify(_IMAGE, [_det("vase", 100 + 4 * quant, 100, 200 + 4 * quant, 200)])
        self.assertEqual(len(verifier.calls), 2, "로봇이 움직여 박스가 어긋나면 다시 묻는다")

    def test_different_categories_are_cached_separately(self):
        verifier = _Verifier({"vase": True, "cabinet": False})
        result = verifier.verify(_IMAGE, [_det("vase"), _det("cabinet")])
        self.assertEqual(result, [True, False])

    def test_only_the_uncached_ones_are_asked(self):
        verifier = _Verifier({"vase": True, "picture": True})
        verifier.verify(_IMAGE, [_det("vase")])
        verifier.calls.clear()
        verifier.verify(_IMAGE, [_det("vase"), _det("picture", 500, 500, 600, 600)])
        self.assertEqual(verifier.calls, [["picture"]], "이미 아는 것은 빼고 묻는다")

    def test_expired_entries_are_asked_again(self):
        import time
        verifier = _Verifier({"vase": True})
        verifier.verify(_IMAGE, [_det("vase")])
        key = verifier._cache_key(_det("vase"))
        verdict, _ = verifier._cache[key]
        stale = time.monotonic() - config.DETECTION_VERIFICATION_CACHE_TTL_SEC - 1.0
        verifier._cache[key] = (verdict, stale)
        verifier.verify(_IMAGE, [_det("vase")])
        self.assertEqual(len(verifier.calls), 2)


class DisabledTest(unittest.TestCase):
    def test_no_api_key_passes_everything(self):
        verifier = DetectionVerifier()
        verifier.api_key = None
        self.assertEqual(verifier.verify(_IMAGE, [_det("vase"), _det("cabinet")]), [True, True])

    def test_empty_input(self):
        self.assertEqual(DetectionVerifier().verify(_IMAGE, []), [])


if __name__ == "__main__":
    unittest.main()
