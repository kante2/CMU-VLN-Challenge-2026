"""검출 재확인(Gemini)을 **지금 step에 필요한 카테고리**로만 좁힌다.

실측 2026-08-25: "First, go near the lamp closest to the black chair, then take the
path between the sofa and the round tables, and stop at the cabinet with a picture
above it."에서 step 3을 주행하는 중에도 이미 끝난 step 1/2의 lamp·chair 저신뢰 검출이
매 프레임 Gemini 재확인에 딸려 들어갔다. task["detection_prompts"]가 모든 step의
합집합이라 그렇다 - 호출 비용이자 그대로 주행 지연이다.
"""

import sys
import types
import unittest

import numpy as np

# perception_pipeline은 로깅과 무거운 모델 모듈을 import한다. 검증 범위 로직만
# 확인하면 되므로 ROS 없는 환경에서도 돌도록 최소 stub으로 대체한다
# (tests/test_detection_verify_cache.py와 같은 패턴).
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

for _name in ("cv2", "torch", "ultralytics"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except ImportError:                                   # pragma: no cover
            sys.modules[_name] = types.ModuleType(_name)

from sysnav import config                                     # noqa: E402
from sysnav.missions import mission3_pipe                     # noqa: E402
from sysnav.perception.perception_pipeline import PerceptionPipeline  # noqa: E402


def _det(category, confidence, bbox=(0, 0, 10, 10)):
    return {"category": category, "confidence": confidence, "bbox": bbox, "detector": "yolo"}


class _RecordingVerifier:
    """실제 Gemini 대신 무엇을 물어봤는지 기록하고 전부 통과시킨다."""

    def __init__(self):
        self.asked = []
        self.calls = 0

    def verify(self, _image, detections):
        self.calls += 1
        self.asked.extend(str(d["category"]) for d in detections)
        return [True] * len(detections)


class _RejectingVerifier(_RecordingVerifier):
    def verify(self, image, detections):
        super().verify(image, detections)
        return [False] * len(detections)


class _Node:
    def __init__(self, step_index):
        self.mission3_step_index = step_index


def _task():
    class _Parser:
        @staticmethod
        def parse(text):
            from sysnav.task.query_parser import extract_target
            return extract_target(text)

    node = type("N", (), {"query_parser": _Parser(), "instruction_splitter": None})()
    return mission3_pipe.parse_instruction(
        node,
        "First, go near the lamp closest to the black chair, then take the path between "
        "the sofa and the round tables, and stop at the cabinet with a picture above it.",
    )


class ActiveCategoriesTest(unittest.TestCase):
    def setUp(self):
        self.task = _task()

    def test_the_whole_task_still_drives_yolo(self):
        """YOLO 프롬프트(합집합)는 그대로다 - 좁히는 건 재확인뿐이다."""
        prompts = set(self.task["detection_prompts"])
        self.assertLessEqual({"lamp", "chair", "sofa", "table"}, prompts)

    def test_step_1_asks_about_lamp_and_chair_only(self):
        active = set(mission3_pipe.active_categories(_Node(0), self.task))
        self.assertEqual(active, {"lamp", "chair"})

    def test_step_2_asks_about_sofa_and_table_only(self):
        active = set(mission3_pipe.active_categories(_Node(1), self.task))
        self.assertEqual(active, {"sofa", "table"})

    def test_step_3_no_longer_asks_about_finished_steps(self):
        """보고된 낭비. step 3에서 lamp/chair는 더 이상 질의 대상이 아니다."""
        active = set(mission3_pipe.active_categories(_Node(2), self.task))
        self.assertNotIn("lamp", active)
        self.assertNotIn("chair", active)
        self.assertNotIn("sofa", active)
        self.assertIn("cabinet with picture", active)

    def test_an_out_of_range_step_falls_back_to_no_restriction(self):
        self.assertIsNone(mission3_pipe.active_categories(_Node(99), self.task))


class VerificationScopeTest(unittest.TestCase):
    IMAGE = np.zeros((4, 4, 3), dtype=np.uint8)

    def setUp(self):
        self.pipeline = PerceptionPipeline.__new__(PerceptionPipeline)
        self.verifier = _RecordingVerifier()
        self.pipeline.detection_verifier = self.verifier
        low = config.DETECTION_VERIFICATION_CONFIDENCE_THRESHOLD - 0.05
        high = config.DETECTION_VERIFICATION_CONFIDENCE_THRESHOLD + 0.05
        self.detections = [
            _det("cabinet", low, (0, 0, 10, 10)),
            _det("lamp", low, (20, 0, 30, 10)),
            _det("chair", low, (40, 0, 50, 10)),
            _det("lamp", high, (60, 0, 70, 10)),
        ]

    def test_only_the_active_category_is_sent_to_the_vlm(self):
        survivors, skipped = self.pipeline._verify_low_confidence(
            self.IMAGE, self.detections, {"cabinet"}
        )
        self.assertEqual(self.verifier.asked, ["cabinet"])
        self.assertEqual(self.verifier.calls, 1)
        self.assertEqual(skipped, 2)
        categories = [d["category"] for d in survivors]
        # 무관한 저신뢰 lamp/chair는 묻지도, 통과시키지도 않는다.
        self.assertEqual(categories.count("chair"), 0)
        self.assertEqual(categories.count("lamp"), 1)     # 고신뢰 lamp는 남는다

    def test_high_confidence_detections_are_never_dropped(self):
        """뒤 step에서 쓸 물체도 지나가면서 그대로 memory에 쌓여야 한다."""
        survivors, _ = self.pipeline._verify_low_confidence(
            self.IMAGE, [_det("sofa", 0.9), _det("table", 0.95)], {"cabinet"}
        )
        self.assertEqual(len(survivors), 2)
        self.assertEqual(self.verifier.calls, 0)

    def test_no_restriction_keeps_the_previous_behaviour(self):
        survivors, skipped = self.pipeline._verify_low_confidence(
            self.IMAGE, self.detections, None
        )
        self.assertEqual(sorted(self.verifier.asked), ["cabinet", "chair", "lamp"])
        self.assertEqual(skipped, 0)
        self.assertEqual(len(survivors), 4)

    def test_a_rejected_detection_is_still_dropped(self):
        self.pipeline.detection_verifier = _RejectingVerifier()
        survivors, skipped = self.pipeline._verify_low_confidence(
            self.IMAGE, self.detections, {"cabinet"}
        )
        self.assertEqual(skipped, 2)
        self.assertEqual([d["category"] for d in survivors], ["lamp"])  # 고신뢰 lamp만

    def test_nothing_is_asked_when_no_detection_is_uncertain(self):
        survivors, skipped = self.pipeline._verify_low_confidence(
            self.IMAGE, [_det("cabinet", 0.9)], {"cabinet"}
        )
        self.assertEqual(self.verifier.calls, 0)
        self.assertEqual((len(survivors), skipped), (1, 0))


if __name__ == "__main__":
    unittest.main()
