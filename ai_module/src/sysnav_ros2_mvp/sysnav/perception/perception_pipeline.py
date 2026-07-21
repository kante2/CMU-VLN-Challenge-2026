"""Sequential perception pipeline: detector -> segmenter -> LiDAR grounding."""

from __future__ import annotations

import numpy as np

from sysnav.perception.debug_visualize import save_debug_image
from sysnav.perception.detector import YoloWorldDetector
from sysnav.perception.segmenter import Sam2Segmenter
from sysnav.perception.lidar_grounding import PanoramaLidarGrounder


class PerceptionPipeline:
    def __init__(self) -> None:
        self.detector = YoloWorldDetector()
        self.segmenter = Sam2Segmenter()
        self.grounder = PanoramaLidarGrounder()

    def process(
        self,
        image_rgb: np.ndarray,
        points_sensor: np.ndarray,
        prompts: list[str],
        robot_pose: dict,
    ) -> list[dict]:
        detections = self.detector.detect(image_rgb, prompts)
        if not detections:
            return []
        segmented = self.segmenter.segment(image_rgb, detections)
        if not segmented:
            return []
        grounded = self.grounder.ground(image_rgb, points_sensor, segmented, robot_pose)
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