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

    def _verify_low_confidence(
        self,
        image_rgb: np.ndarray,
        detections: list[dict],
        verify_categories: set[str] | None = None,
    ) -> tuple[list[dict], int]:
        """(살아남은 detection, Gemini에 안 묻고 버린 개수).

        confidence가 DETECTION_VERIFICATION_CONFIDENCE_THRESHOLD 밑인 것만 골라 Gemini
        한 번(프레임당 1회 호출로 묶어서)에 검증하고, 확인 안 된 것만 걸러낸다. 그 이상
        confidence인 detection은 건드리지 않고 그대로 통과시킨다.

        verify_categories는 "지금 이 판정에 실제로 필요한 카테고리"다(None이면 제한 없음).
        여기 없는 카테고리의 저신뢰 detection은 **묻지 않고 버린다**:
          - 묻지 않는 이유 - mission3에서 task["detection_prompts"]는 모든 step의
            합집합이라, step 3을 하는 중에 이미 끝난 step 1/2의 lamp·chair까지 매
            프레임 재확인에 딸려 들어갔다. 호출 비용이자 그대로 주행 지연이다.
          - 그렇다고 통과시키면 안 되는 이유 - 검증을 건너뛴 저신뢰 detection이 memory에
            쌓이면, 나중에 그 카테고리가 정말 필요해진 step에서 검증 없이 들어온 쓰레기가
            후보로 올라온다(이 검증기가 원래 막으려던 바로 그 오검출이다).
        고신뢰(임계값 이상) detection은 카테고리와 무관하게 예전처럼 다 통과하므로,
        뒤 step에서 쓸 물체도 지나가면서 그대로 memory에 쌓인다."""
        if not config.DETECTION_VERIFICATION_ENABLED:
            return detections, 0
        uncertain_indices = [
            index for index, detection in enumerate(detections)
            if float(detection.get("confidence", 1.0)) < config.DETECTION_VERIFICATION_CONFIDENCE_THRESHOLD
        ]
        if not uncertain_indices:
            return detections, 0
        if verify_categories is None:
            ask_indices, skipped_indices = uncertain_indices, []
        else:
            ask_indices, skipped_indices = [], []
            for index in uncertain_indices:
                category = str(detections[index].get("category", "")).strip().lower()
                (ask_indices if category in verify_categories else skipped_indices).append(index)

        dropped_indices = set(skipped_indices)
        if ask_indices:
            confirmed = self.detection_verifier.verify(
                image_rgb, [detections[index] for index in ask_indices]
            )
            dropped_indices.update(
                ask_indices[position] for position, ok in enumerate(confirmed) if not ok
            )
        if not dropped_indices:
            return detections, 0
        survivors = [
            detection for index, detection in enumerate(detections)
            if index not in dropped_indices
        ]
        return survivors, len(skipped_indices)

    def process(
        self,
        image_rgb: np.ndarray,
        points_sensor: np.ndarray,
        prompts: list[str],
        robot_pose: dict,
        verify_categories: set[str] | None = None,
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

        segmented = self.segmenter.segment(image_rgb, detections)
        self._logger.info(
            f"[Perception] SAM2 segmented {len(segmented)}/{len(detections)} detections"
        )
        activity.add(PERCEPTION, f"② SAM2 분할 {len(segmented)}/{len(detections)}")
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
            f"③ LiDAR 3D 위치 확정 {len(grounded)}/{len(segmented)}",
            f"precise {precise} / approximate {len(grounded) - precise}"
            + (f" | {dropped}개는 대응 point가 없어 탈락" if dropped else "")
            + (f" | {self._summarize(grounded)}" if grounded else ""),
        )

        # 검출 재확인은 여기서 한다 - YOLO 직후가 아니라 **3D 위치가 정해진 뒤**다.
        #
        # 왜 옮겼나: 캐시 키를 map 프레임 3D 위치로 쓰려면 위치가 있어야 한다. 예전
        # 키(2D bbox)는 로봇이 조금만 움직여도 어긋나서 같은 물체를 매 프레임 다시
        # 물었다(실측 2026-08-25: 33초에 같은 picture 5회). 덤으로, grounding에서
        # 탈락한 detection(대응 LiDAR point 없음)은 어차피 memory에 못 들어가므로
        # 그것들에 Gemini를 쓰던 것도 사라진다. SAM2+grounding은 로컬이라 프레임당
        # 0.4초 수준이고 Gemini는 건당 수 초라, 순서를 바꾸는 쪽이 항상 싸다.
        verified, skipped = self._verify_low_confidence(
            image_rgb, grounded, verify_categories
        )
        rejected = len(grounded) - len(verified) - skipped
        if len(verified) != len(grounded):
            self._logger.info(
                f"[Perception] after confidence verification: {self._format_detections(verified)}"
            )
        scope = (
            "" if verify_categories is None
            else f" | 지금 step에 필요한 {', '.join(sorted(verify_categories))}만 질의"
        )
        activity.add(
            PERCEPTION,
            f"④ 검출 재확인 - {rejected}개 기각, {len(verified)}개 통과"
            + (f", {skipped}개 무관해서 미질의" if skipped else ""),
            (f"conf<{config.DETECTION_VERIFICATION_CONFIDENCE_THRESHOLD:.2f}인 것만 Gemini에 물어봄"
             if rejected or skipped or len(verified) != len(grounded)
             else f"conf<{config.DETECTION_VERIFICATION_CONFIDENCE_THRESHOLD:.2f}인 검출이 없어 건너뜀")
            + scope,
        )
        grounded = verified
        if not grounded:
            return []
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
