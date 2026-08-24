"""탐색 상태(surface point / frontier)를 top-down PNG로 시각화.

scene_graph_visualizer.py와 같은 패턴 - 매번 현재 상태 전체를 새로 그려서 atomic하게
덮어쓴다 (append 아님, 재시작하면 다음 갱신 때 바로 최신 상태로 교체됨).

논문의 surface point set S(free/non-free 경계, coverage_planner.plan_route()가
candidate 점수를 매길 때 쓰는 것과 완전히 동일한 마스크)를 노란 점으로 찍어서,
frontier 탐지가 실제로 어디를 "아직 탐사 안 된 경계"로 보고 있는지 눈으로
바로 확인할 수 있게 한다.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from sysnav import config


def export_exploration_debug(
    grid: np.ndarray,
    surface_mask: np.ndarray,
    robot_cell: tuple[int, int] | None = None,
    scale: int = 4,
) -> str | None:
    """exploration_debug_latest.png를 갱신한다. 저장된 경로(실패/그릴 게 없으면 None)."""
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
    cropped_surface = surface_mask[r0:r1, c0:c1]

    height, width = cropped_grid.shape
    canvas = np.full((height, width, 3), 60, dtype=np.uint8)  # unknown = 어두운 회색
    canvas[cropped_grid == config.OCC_FREE] = (150, 150, 150)  # free = 회색
    canvas[cropped_grid == config.OCC_OCCUPIED] = (20, 20, 20)  # 벽/장애물 = 진하게

    canvas = cv2.resize(canvas, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)

    # surface point(frontier)를 노란 점으로 찍는다 - plan_route()의 scov 계산과
    # 완전히 같은 마스크라, 여기 노랗게 보이는 곳이 곧 candidate가 "새로 본다"고
    # 점수 매길 수 있는 경계다.
    surface_rows, surface_cols = np.nonzero(cropped_surface)
    for row, col in zip(surface_rows.tolist(), surface_cols.tolist()):
        cy, cx = row * scale + scale // 2, col * scale + scale // 2
        cv2.circle(canvas, (cx, cy), max(1, scale // 2), (0, 230, 255), -1)  # BGR: 노랑

    if robot_cell is not None and r0 <= robot_cell[0] < r1 and c0 <= robot_cell[1] < c1:
        ry = int((robot_cell[0] - r0) * scale + scale / 2)
        rx = int((robot_cell[1] - c0) * scale + scale / 2)
        cv2.drawMarker(canvas, (rx, ry), (255, 60, 60), cv2.MARKER_TRIANGLE_UP, scale * 3, 2)

    cv2.putText(
        canvas,
        f"surface points={int(cropped_surface.sum())}",
        (10, canvas.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    out_path = Path(config.DEBUG_DIR) / "exploration_debug_latest.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(".tmp.png")
    if not cv2.imwrite(str(temp_path), canvas):
        return None
    os.replace(temp_path, out_path)
    os.chmod(out_path, 0o644)
    return str(out_path)
