#!/usr/bin/env python3
"""
센서 파이프라인 흐름파일 — 카메라 → 2D box → segmentation → LiDAR 좌표변환 → ray → 3D 위치.

t2/t3 solver가 공용으로 쓰는 "센서 스텝" 함수를 파이프라인 순서로 모아둔다.
각 함수는 ctx를 받아 자기 필드 하나를 채우고, 다음 함수가 그 필드를 읽어 이어서 쓴다.
위에서 아래로 읽으면 그대로 센서→3D 흐름이다:

  grab_camera_image       -> ctx.image, ctx.image_stamp   최신 카메라 프레임 스냅샷 (image_utils)
  detect_candidate_boxes  -> ctx.detections               GroundingDINO 2D 후보 박스 (detector)
  load_scan_points_in_map -> ctx.scan_points_map          동기화 LiDAR → map frame 3D 점
                                                          (scan_transform + coordinate_transform)
  segment_selected_object -> ctx.segmentation_mask        선택 물체 SAM 마스크 (segmenter)
  estimate_target_3d_pose -> ctx.result                   box+마스크를 점군에 재투영한 3D 위치/크기
                                                          (projector: seg에 맞는 LiDAR ray → 3D)

물체 후보 선택(Qwen)·공간관계 필터는 solver 로직이라 여기 두지 않는다.
저수준 구현은 같은 폴더의 detector/segmenter/projector/scan_transform/coordinate_transform 등에 있다.
node를 인자로 받는 자유 함수 + ctx(출력-인자) 패턴.
"""

from tmah_vlm.sensor_process.image_utils import grab_camera_image  # re-export (흐름 시작점)
from tmah_vlm.sensor_process.projector import box_to_3d
from tmah_vlm.sensor_process.scan_transform import get_scan_points_in_map_frame

__all__ = [
    "grab_camera_image",
    "detect_candidate_boxes",
    "load_scan_points_in_map",
    "segment_selected_object",
    "estimate_target_3d_pose",
]


def detect_candidate_boxes(node, ctx):
    # GroundingDINO로 검출어(ctx.detect_prompt)에 해당하는 2D 후보 박스들을 검출한다.
    ctx.detections = node.detector.detect(ctx.image, ctx.detect_prompt)


def load_scan_points_in_map(node, ctx, log_tag="Sensor"):
    # 이미지 시각에 동기화된 point cloud를 map frame 3D 점들로 변환해 담는다.
    ctx.scan_points_map = get_scan_points_in_map_frame(node, log_tag)


def segment_selected_object(node, ctx):
    # 선택된 물체 1개만 SAM box-prompt로 pixel 마스크를 얻는다.
    # 미로딩/실패 시 None (projector가 None이면 자동으로 box 기반으로 대체).
    # 세그멘테이션을 실제로 썼는지 매 쿼리마다 로그로 남긴다 (핑크가 안 나오는 원인 추적용).
    log = node.get_logger()

    if node.segmenter is None:
        log.warn("[Sensor] segmenter not ready → 마스크 없이 box로 진행 (핑크 없음)")
        ctx.segmentation_mask = None
        return

    try:
        mask = node.segmenter.segment(ctx.image, ctx.selected.box)
    except Exception as error:
        import traceback
        log.error(f"[Sensor] segmentation failed: {error}\n{traceback.format_exc()}")
        ctx.segmentation_mask = None
        return

    ctx.segmentation_mask = mask
    if mask is None:
        log.warn("[Sensor] segmenter.segment() returned None")
    else:
        log.info(
            f"[Sensor] segmentation mask ok: shape={mask.shape}, "
            f"true_px={int(mask.sum())}, box={ctx.selected.box}"
        )


def estimate_target_3d_pose(node, ctx):
    # 선택 box(+마스크)를 point cloud에 재투영해 3D 좌표+크기를 계산한다.
    # (마스크가 있으면 그 실루엣에 맞는 LiDAR ray만 골라 3D 위치를 뽑는다.)
    ctx.result = box_to_3d(
        ctx.selected.box,
        ctx.image.size,
        ctx.scan_points_map,
        node.transformer,
        ctx.detect_prompt,
        image_stamp=ctx.image_stamp,
        segmentation_mask=ctx.segmentation_mask,
    )
