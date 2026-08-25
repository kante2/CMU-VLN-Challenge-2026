"""검출 재확인 캐시를 2D bbox가 아니라 **3D 위치**로 키한다.

예전 키는 (카테고리, 48px로 양자화한 파노라마 bbox)였다. 그런데 파노라마에서 박스는
로봇이 조금만 움직여도 48px보다 크게 밀리기 때문에, 같은 물체를 계속 다시 물었다 -
실측 2026-08-25: 06:33:23 / :28 / :39 / :49 / :56, 33초 동안 같은 picture를 5번 질의.
TTL(30초) 안이었는데도 전부 cache miss였다.

3D 위치는 로봇이 어디서 보든 같으므로 물체당 한 번으로 수렴한다. 이걸 쓰려면 검증이
LiDAR grounding **뒤에** 돌아야 해서 perception_pipeline의 순서도 같이 바꿨다.
"""

import sys
import types
import unittest

# detection_verifier는 로깅/이미지에만 rclpy·cv2를 쓴다.
if "rclpy" not in sys.modules:
    try:
        import rclpy  # noqa: F401
    except ImportError:                                       # pragma: no cover
        package = types.ModuleType("rclpy")
        logging_module = types.ModuleType("rclpy.logging")

        class _StubLogger:
            def info(self, *args, **kwargs): pass
            def warning(self, *args, **kwargs): pass

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

from sysnav import config                                            # noqa: E402
from sysnav.perception.detection_verifier import DetectionVerifier   # noqa: E402


def _det(category="picture", position=(1.0, 2.0, 0.5), bbox=(100, 100, 200, 200), confidence=0.30):
    detection = {"category": category, "bbox": bbox, "confidence": confidence}
    if position is not None:
        detection["position"] = position
    return detection


class PositionKeyTest(unittest.TestCase):
    def setUp(self):
        self.verifier = DetectionVerifier()

    def _key(self, **kwargs):
        return self.verifier._cache_key(_det(**kwargs))

    def test_the_same_object_seen_from_a_different_pose_is_one_key(self):
        """보고된 낭비. 로봇이 움직여 bbox가 완전히 달라져도 같은 물체면 같은 키다."""
        near = self._key(bbox=(100, 100, 200, 200))
        far = self._key(bbox=(1400, 260, 1600, 420))   # 파노라마 반대편으로 밀림
        self.assertEqual(near, far)

    def test_two_objects_apart_are_different_keys(self):
        self.assertNotEqual(self._key(position=(1.0, 2.0, 0.5)),
                            self._key(position=(4.0, 2.0, 0.5)))

    def test_jitter_within_one_quantum_stays_one_key(self):
        """같은 물체의 위치 추정은 관측마다 조금씩 흔들린다 - 그걸로 다시 물으면 안 된다."""
        quant = config.DETECTION_VERIFICATION_CACHE_POSITION_QUANT_M
        base = (1.0 + quant / 4.0, 2.0, 0.5)
        self.assertEqual(self._key(position=(1.0, 2.0, 0.5)), self._key(position=base))

    def test_categories_are_still_separated(self):
        self.assertNotEqual(self._key(category="picture"), self._key(category="door"))

    def test_height_is_part_of_the_key(self):
        """같은 XY라도 높이가 다르면 다른 물체다(캐비닛 vs 그 위 그림)."""
        self.assertNotEqual(self._key(position=(1.0, 2.0, 0.4)),
                            self._key(position=(1.0, 2.0, 1.9)))

    def test_a_detection_without_a_position_falls_back_to_the_bbox_key(self):
        """3D 위치가 아직 없는 호출(옛 순서)도 키가 없어 죽지는 않아야 한다."""
        without = self._key(position=None, bbox=(100, 100, 200, 200))
        moved = self._key(position=None, bbox=(1400, 260, 1600, 420))
        self.assertNotEqual(without, moved)          # 예전 동작 그대로
        self.assertNotEqual(without, self._key())    # 위치 키와도 섞이지 않는다


class RepeatedObservationTest(unittest.TestCase):
    """실측 시나리오 재현: 같은 물체가 프레임마다 다른 박스로 잡힌다."""

    FRAMES = [
        (100, 100, 210, 240),
        (280, 120, 395, 255),
        (610, 140, 720, 270),
        (980, 130, 1090, 262),
        (1400, 110, 1512, 250),
    ]

    def _misses(self, with_position: bool) -> int:
        verifier = DetectionVerifier()
        misses = 0
        for bbox in self.FRAMES:
            detection = _det(position=(1.0, 2.0, 0.5) if with_position else None, bbox=bbox)
            if verifier._cached(detection) is None:
                misses += 1
                verifier._remember(detection, True)
        return misses

    def test_the_bbox_key_asks_every_frame(self):
        self.assertEqual(self._misses(with_position=False), 5)

    def test_the_position_key_asks_once(self):
        self.assertEqual(self._misses(with_position=True), 1)


class PipelineOrderTest(unittest.TestCase):
    def test_verification_runs_after_grounding(self):
        """위치 키를 쓰려면 grounding이 먼저여야 한다. 순서가 되돌아가면 여기서 잡는다."""
        import inspect
        from sysnav.perception.perception_pipeline import PerceptionPipeline
        body = inspect.getsource(PerceptionPipeline.process)
        ground_at = body.index("self.grounder.ground(")
        verify_at = body.index("self._verify_low_confidence(")
        self.assertLess(ground_at, verify_at)
        # 검증 대상이 detections가 아니라 grounded여야 한다.
        self.assertIn("self._verify_low_confidence(\n            image_rgb, grounded, verify_categories\n        )", body)


if __name__ == "__main__":
    unittest.main()
