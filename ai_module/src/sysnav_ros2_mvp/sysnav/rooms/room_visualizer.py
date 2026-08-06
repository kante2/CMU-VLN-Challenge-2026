"""Room segmentation 결과를 top-down PNG로 시각화.

scene_graph_latest.png와 같은 패턴이다 - 매번 현재 상태 전체를 새로 그려서
atomic하게 덮어쓴다. append가 아니라 통째로 다시 그리는 거라, 노드를 재시작해서
매핑이 처음부터 다시 쌓이면 이전 실행의 그림은 첫 갱신 때 바로 지워지고 최신
상태만 남는다 (sysnav_relation_check.txt처럼 append하며 쌓이는 파일이 아님).
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from sysnav import config

_ROOM_PALETTE = [
    (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
    (171, 71, 188), (0, 172, 193), (255, 112, 67), (124, 179, 66),
    (94, 53, 177), (0, 121, 107),
]


def _room_color(room_id: int) -> tuple[int, int, int]:
    return _ROOM_PALETTE[(room_id - 1) % len(_ROOM_PALETTE)]


def _category_text(room: dict) -> str:
    # SysNav paper Sec. IV-A-1의 room category(c_i) - RoomRegistry가 대표 viewpoint
    # 기준으로 on-demand 추론해 캐싱한 값. 아직 대표 viewpoint가 없으면(방금 새로
    # 생긴 room) "no viewpoint yet", 대표는 있는데 VLM 추론이 아직 안 됐으면(요청
    # 중이거나 다음 사이클에 재시도 대기 중) "classifying...".
    category = room.get("category")
    if category:
        return category
    if room.get("representative_viewpoint_id") is not None:
        return "classifying..."
    return "no viewpoint yet"


def export_room_segmentation(
    grid: np.ndarray,
    result: dict,
    scale: int = 4,
) -> str | None:
    """room_segmentation_latest.png를 갱신한다. 저장된 경로(실패/그릴 게 없으면 None).

    result: RoomRegistry.update()의 반환값 - {"labels": persistent room_id 배열,
    "rooms": [{"room_id", "cell_count", "centroid_row", "centroid_col", "category",
    "viewpoint_count", "representative_viewpoint_id"}, ...]}."""
    if not config.SAVE_DEBUG_IMAGES:
        return None

    mapped = grid != config.OCC_UNKNOWN
    if not mapped.any():
        return None

    rows, cols = np.nonzero(mapped)
    margin = 5
    r0 = max(0, int(rows.min()) - margin)
    r1 = min(grid.shape[0], int(rows.max()) + 1 + margin)
    c0 = max(0, int(cols.min()) - margin)
    c1 = min(grid.shape[1], int(cols.max()) + 1 + margin)

    cropped_grid = grid[r0:r1, c0:c1]
    cropped_labels = result["labels"][r0:r1, c0:c1]

    height, width = cropped_grid.shape
    canvas = np.full((height, width, 3), 60, dtype=np.uint8)  # unknown = 어두운 회색
    canvas[cropped_grid == config.OCC_FREE] = (200, 200, 200)  # 방 미배정 free = 밝은 회색
    for room in result["rooms"]:
        canvas[cropped_labels == room["room_id"]] = _room_color(room["room_id"])
    canvas[cropped_grid == config.OCC_OCCUPIED] = (20, 20, 20)  # 벽은 항상 진하게 덮어그림

    canvas = cv2.resize(canvas, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)

    for room in result["rooms"]:
        cy = int((room["centroid_row"] - r0) * scale)
        cx = int((room["centroid_col"] - c0) * scale)
        title = f'Room {room["room_id"]}: {_category_text(room)}'
        detail = f'{room["cell_count"]} cells, {room.get("viewpoint_count", 0)} vp'
        for offset, (text, thickness) in enumerate([(title, 2), (detail, 1)]):
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)
            origin = (max(0, cx - text_width // 2), max(text_height, cy) + offset * (text_height + 6))
            cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"rooms={len(result['rooms'])}",
        (10, canvas.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    out_path = Path(config.DEBUG_DIR) / "room_segmentation_latest.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(".tmp.png")
    if not cv2.imwrite(str(temp_path), canvas):
        return None
    os.replace(temp_path, out_path)
    os.chmod(out_path, 0o644)
    return str(out_path)
