#!/usr/bin/env python3
"""
SAM(Segment Anything) 기반 box-prompted segmentation.

GroundingDINO/Qwen이 고른 2D box는 사각형이라 물체 실루엣 밖(배경, 모서리)
pixel까지 포함한다. 그 상태로 point cloud를 box 안 pixel과 매칭하면 물체
뒤쪽/옆쪽 배경 point가 섞여서 3D 위치가 오검출될 수 있다. box를 prompt로
SAM을 돌려서 물체 실루엣만 정확히 담은 pixel mask를 얻으면, 그 마스크
안에 투영되는 point만 써서 더 정확한 3D 위치를 얻을 수 있다
(grounding/projector.py의 make_mask_candidate_mask()가 이 mask를 사용).
"""

import numpy as np
import torch
from transformers import SamModel, SamProcessor


class SAMSegmenter:
    def __init__(self, model_id="facebook/sam-vit-base", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def segment(self, image, box):
        """PIL image + box(x1,y1,x2,y2) -> image 해상도의 HxW bool mask."""
        masks, scores = self.run_model(image, box)
        return self.build_mask(masks, scores)

    def run_model(self, image, box):
        input_boxes = [[[float(v) for v in box]]]

        inputs = self.processor(
            image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        scores = outputs.iou_scores[0, 0]

        return masks[0], scores

    def build_mask(self, masks, scores):
        """SAM은 box 하나당 마스크 후보 3개를 주므로 iou score가 가장 높은 것만 쓴다."""
        best_index = int(torch.argmax(scores))
        return masks[0, best_index].numpy().astype(bool)
