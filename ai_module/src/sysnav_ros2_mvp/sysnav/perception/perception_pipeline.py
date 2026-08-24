"""Sequential perception pipeline: detector -> segmenter -> LiDAR grounding."""

from __future__ import annotations

import numpy as np
from rclpy.logging import get_logger

from sysnav import config
from sysnav.activity_log import PERCEPTION, activity
from sysnav.perception.debug_visualize import save_debug_image
from sysnav.perception.detection_verifier import DetectionVerifier
from sysnav.perception.detector import YoloWorldDetector
from sysnav.perception.segmenter import Sam2Segmenter
from sysnav.perception.lidar_grounding import PanoramaLidarGrounder


class PerceptionPipeline:
    def __init__(self) -> None:
        self.detector = YoloWorldDetector()
        self.segmenter = Sam2Segmenter()
        self.grounder = PanoramaLidarGrounder()
        self.detection_verifier = DetectionVerifier()
        self._logger = get_logger("sysnav_perception")

    @staticmethod
    def _summarize(detections: list[dict]) -> str:
        """카테고리별 개수 + 최고 confidence. 대시보드 한 줄에 들어가야 하므로
        _format_detections(전체 나열)와 달리 압축한다."""
        if not detections:
            return "없음"
        best: dict[str, float] = {}
        counts: dict[str, int] = {}
        for detection in detections:
            category = str(detection.get("category", "?"))
            confidence = float(detection.get("confidence", 0.0))
            counts[category] = counts.get(category, 0) + 1
            best[category] = max(best.get(category, 0.0), confidence)
        return ", ".join(
            f"{category} {counts[category]}개(최고 {best[category]:.2f})"
            for category in sorted(counts, key=lambda c: -counts[c])
        )

    @staticmethod
    def _format_detections(detections: list[dict]) -> str:
        if not detections:
            return "none"
        return ", ".join(
            f"{detection['category']}={float(detection.get('confidence', 0.0)):.2f}"
            f"[{detection.get('detector', '?')}]"
            for detection in detections
        )

    def _verify_low_confidence(self, image_rgb: np.ndarray, detections: list[dict]) -> list[dict]:
        # confidence가 DETECTION_VERIFICATION_CONFIDENCE_THRESHOLD 밑인 것만 골라 Gemini
        # 한 번(프레임당 1회 호출로 묶어서)에 검증하고, 확인 안 된 것만 걸러낸다. 그 이상
        # confidence인 detection은 건드리지 않고 그대로 통과시킨다.
        if not config.DETECTION_VERIFICATION_ENABLED:
            return detections
        uncertain_indices = [
            index for index, detection in enumerate(detections)
            if float(detection.get("confidence", 1.0)) < config.DETECTION_VERIFICATION_CONFIDENCE_THRESHOLD
        ]
        if not uncertain_indices:
            return detections
        uncertain_detections = [detections[index] for index in uncertain_indices]
        confirmed = self.detection_verifier.verify(image_rgb, uncertain_detections)
        rejected_indices = {
            uncertain_indices[position]
            for position, ok in enumerate(confirmed)
            if not ok
        }
        if not rejected_indices:
            return detections
        return [detection for index, detection in enumerate(detections) if index not in rejected_indices]

    def process(
        self,
        image_rgb: np.ndarray,
        points_sensor: np.ndarray,
        prompts: list[str],
        robot_pose: dict,
    ) -> list[dict]:
        detections = self.detector.detect(image_rgb, prompts)
        self._logger.info(
            f"[Perception] YOLO-World detected: {self._format_detections(detections)} "
            f"(prompts={prompts})"
        )
        activity.add(
            PERCEPTION, f"① YOLO 검출 {len(detections)}개",
            f"{self._summarize(detections)} | prompts={', '.join(prompts)}",
        )
        if not detections:
            activity.add(PERCEPTION, "① YOLO 검출 0개 - 이 프레임은 여기서 끝",
                         f"prompts={', '.join(prompts)}")
            return []

        verified = self._verify_low_confidence(image_rgb, detections)
        if len(verified) != len(detections):
            self._logger.info(
                f"[Perception] after confidence verification: {self._format_detections(verified)}"
            )
        rejected = len(detections) - len(verified)
        activity.add(
            PERCEPTION,
            f"② 검출 재확인 - {rejected}개 기각, {len(verified)}개 통과",
            f"conf<{config.DETECTION_VERIFICATION_CONFIDENCE_THRESHOLD:.2f}인 것만 Gemini에 물어봄"
            if rejected or len(verified) != len(detections)
            else f"conf<{config.DETECTION_VERIFICATION_CONFIDENCE_THRESHOLD:.2f}인 검출이 없어 건너뜀",
        )
        detections = verified
        if not detections:
            return []

        segmented = self.segmenter.segment(image_rgb, detections)
        self._logger.info(
            f"[Perception] SAM2 segmented {len(segmented)}/{len(detections)} detections"
        )
        activity.add(PERCEPTION, f"③ SAM2 분할 {len(segmented)}/{len(detections)}")
        if not segmented:
            return []
        grounded = self.grounder.ground(image_rgb, points_sensor, segmented, robot_pose)
        self._logger.info(
            f"[Perception] LiDAR-grounded to 3D: "
            f"{[(item['category'], tuple(round(v, 2) for v in item['position']), item.get('grounding_quality', '?'), item.get('num_points', 0)) for item in grounded]}"
        )
        precise = sum(1 for item in grounded if item.get("grounding_quality") == "precise")
        dropped = len(segmented) - len(grounded)
        activity.add(
            PERCEPTION,
            f"④ LiDAR 3D 위치 확정 {len(grounded)}/{len(segmented)}",
            f"precise {precise} / approximate {len(grounded) - precise}"
            + (f" | {dropped}개는 대응 point가 없어 탈락" if dropped else "")
            + (f" | {self._summarize(grounded)}" if grounded else ""),
        )
        save_debug_image(image_rgb, segmented, grounded) # ai_module/debug에 저장
        return grounded

'''
observations = self.perception.process( # 실제 객체 인식 파이프라인 실행
image_rgb=image_msg_to_rgb(image_msg), # ROS image -> numpy
points_sensor=pointcloud2_to_xyz(scan_msg), # pointcloud -> numpy
prompts=list(task["detection_prompts"]), #  YOLO-World가 검출해야 하는 객체 목록 -> prompts
robot_pose=pose, # LiDAR의 객체 point를 map 좌표로 변환
)
< self.perception.process 내부 구조 >
YOLO-World
↓
2D Bounding Box

SAM2
↓
Object Mask

LiDAR Grounding - ground() 함수
↓
3D Object Observation
'''
