"""Sequential perception pipeline: detector -> segmenter -> LiDAR grounding."""

from __future__ import annotations

import numpy as np

from sysnav.perception.debug_visualize import save_debug_image, save_lidar_projection_image
from sysnav.perception.detector import YoloWorldDetector
from sysnav.perception.segmenter import Sam2Segmenter
from sysnav.perception.lidar_grounding import PanoramaLidarGrounder
from sysnav.perception.gemini_visual_alias import GeminiVisualAliasFallback


class PerceptionPipeline:
    def __init__(self) -> None:
        self.detector = YoloWorldDetector()
        self.segmenter = Sam2Segmenter()
        self.grounder = PanoramaLidarGrounder()
        self.visual_alias = GeminiVisualAliasFallback()

    def begin_task(self) -> None:
        self.visual_alias.reset()
        self.grounder.reset()

    def process(
        self,
        image_rgb: np.ndarray,
        points_sensor: np.ndarray,
        prompts: list[str],
        robot_pose: dict,
        prompt_categories: dict[str, str] | None = None,
    ) -> list[dict]:
        prompt_categories = dict(prompt_categories or {})
        detections = self.detector.detect(
            image_rgb,
            prompts,
            prompt_categories=prompt_categories,
        )
        required_categories = set(prompt_categories.values())
        detected_categories = {str(item["category"]).lower() for item in detections}
        missing_categories = sorted(required_categories - detected_categories)
        if missing_categories:
            existing_prompts = {
                category: [
                    prompt for prompt, canonical in prompt_categories.items()
                    if canonical == category
                ]
                for category in missing_categories
            }
            additional = self.visual_alias.suggest(
                image_rgb,
                missing_categories,
                existing_prompts,
            )
            retry_prompts = list(prompts)
            for category, aliases in additional.items():
                for alias in aliases:
                    cleaned = str(alias).strip().lower()
                    if cleaned and cleaned not in prompt_categories:
                        retry_prompts.append(cleaned)
                        prompt_categories[cleaned] = category
            if len(retry_prompts) > len(prompts):
                detections = self.detector.detect(
                    image_rgb,
                    retry_prompts,
                    prompt_categories=prompt_categories,
                )
        if not detections:
            return []
        segmented = self.segmenter.segment(image_rgb, detections)
        if not segmented:
            return []
        grounded = self.grounder.ground(image_rgb, points_sensor, segmented, robot_pose)
        save_debug_image(image_rgb, segmented, grounded) # ai_module/debug에 저장
        save_lidar_projection_image(image_rgb, self.grounder.last_projection_debug)
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
