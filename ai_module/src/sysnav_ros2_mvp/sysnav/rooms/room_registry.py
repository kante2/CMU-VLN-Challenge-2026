"""Persistent Room Node identity (SysNav paper Sec. IV-A-1, "Room Node").

RoomSegmenter.segment()는 mapping cycle마다 현재 occupancy grid로 watershed를 처음부터
다시 돌려서 room_id를 1..N으로 새로 매긴다 (cv2.connectedComponentsWithStats 기반) -
"이번 사이클의 room 2"와 "다음 사이클의 room 2"가 같은 물리적 방이라는 보장이 없다.
논문의 Room Node는 attribute A(v_r) = {m_i^r, c_i^r, I_i^r}를 시간에 걸쳐 유지하는
node(category, representative image)라서, 이 정체성이 사이클 간에 안정적이지 않으면
"이 방의 카테고리"라는 개념 자체가 성립하지 않는다.

RoomRegistry가 그 다리를 놓는다: 매 사이클의 room들을 centroid 근접 매칭으로 기존
persistent room에 이어붙이고(멀면 새 room으로 취급), 그 persistent room에 매 사이클
새로 들어오는 viewpoint를 배정한다.

Representative image 선택은 논문 각주 그대로 - "the image maximizing the room's
visible voxels": 방에 배정된 viewpoint 중 coverage_voxel_count가 가장 큰 것.
카테고리(VLM 추론)는 object self-attribute와 동일한 on-demand 패턴 - representative
viewpoint가 바뀔 때만 재추론 대상이 되고, 그 외엔 캐시를 그대로 쓴다. "방을 다 봤는지"는
게이팅 조건이 아니다 - 논문도 그런 완료 조건을 두지 않는다(room-based navigation의
early-stop/room-query 의사결정 시점마다 그때까지 확보된 최선의 정보로 판단한다).
"""

from __future__ import annotations

import math
import time

import numpy as np

from sysnav import config


class RoomRegistry:
    def __init__(self) -> None:
        self._rooms: dict[int, dict] = {}
        self._next_id = 1

    def _new_room(self, room_id: int) -> dict:
        return {
            "room_id": room_id,
            "cell_count": 0,
            "centroid_row": 0.0,
            "centroid_col": 0.0,
            "viewpoint_ids": set(),
            "representative_viewpoint_id": None,
            "representative_coverage": -1,
            "representative_image_path": None,
            "category": None,
            "classified_viewpoint_id": None,
            "last_classification_attempt_time": None,
        }

    def update(self, segmentation: dict, viewpoints: list[dict], world_to_grid) -> dict:
        """segmentation: RoomSegmenter.segment() 결과(사이클마다 새로 매겨진 room_id).
        viewpoints: SceneGraphManager.snapshot()["viewpoints"] (pose/image_path/
        coverage_voxel_count를 가진 dict 리스트). world_to_grid: CoveragePlanner.world_to_grid.

        반환: {"labels": persistent room_id로 relabel된 배열, "rooms": [이번 사이클에
        실제로 매칭된 persistent room들의 요약]}. category/representative_* 등 시간축
        상태는 self._rooms에 계속 누적되고, 매칭 안 된(사라진 것처럼 보이는) 기존
        room은 지우지 않고 그대로 둔다(다음 사이클에 다시 나타날 수 있음)."""
        labels = segmentation.get("labels")
        cycle_rooms = segmentation.get("rooms") or []
        if labels is None:
            return {"labels": None, "rooms": []}
        if not cycle_rooms:
            return {"labels": np.zeros_like(labels), "rooms": []}

        resolution = float(config.MAP_RESOLUTION_M)
        radius_cells = float(config.ROOM_REGISTRY_MATCH_RADIUS_M) / resolution

        available = set(self._rooms.keys())
        id_map: dict[int, int] = {}
        for cycle_room in cycle_rooms:
            cycle_id = int(cycle_room["room_id"])
            crow = float(cycle_room["centroid_row"])
            ccol = float(cycle_room["centroid_col"])
            best_id = None
            best_dist = radius_cells
            for persistent_id in available:
                room = self._rooms[persistent_id]
                dist = math.hypot(crow - room["centroid_row"], ccol - room["centroid_col"])
                if dist <= best_dist:
                    best_dist = dist
                    best_id = persistent_id
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self._rooms[best_id] = self._new_room(best_id)
            else:
                available.discard(best_id)
            id_map[cycle_id] = best_id
            room = self._rooms[best_id]
            room["cell_count"] = int(cycle_room["cell_count"])
            room["centroid_row"] = crow
            room["centroid_col"] = ccol

        persistent_labels = np.zeros_like(labels)
        for cycle_id, persistent_id in id_map.items():
            persistent_labels[labels == cycle_id] = persistent_id

        matched_ids = set(id_map.values())
        for persistent_id in matched_ids:
            self._rooms[persistent_id]["viewpoint_ids"] = set()

        rows, cols = labels.shape
        for viewpoint in viewpoints:
            pose = viewpoint.get("pose") or {}
            cell = world_to_grid(float(pose.get("x", 0.0)), float(pose.get("y", 0.0)))
            if cell is None:
                continue
            row, col = cell
            if not (0 <= row < rows and 0 <= col < cols):
                continue
            cycle_id = int(labels[row, col])
            if cycle_id <= 0 or cycle_id not in id_map:
                continue
            persistent_id = id_map[cycle_id]
            room = self._rooms[persistent_id]
            room["viewpoint_ids"].add(int(viewpoint["viewpoint_id"]))

            coverage = int(viewpoint.get("coverage_voxel_count", 0))
            image_path = viewpoint.get("image_path")
            if image_path and coverage > room["representative_coverage"]:
                room["representative_coverage"] = coverage
                room["representative_viewpoint_id"] = int(viewpoint["viewpoint_id"])
                room["representative_image_path"] = str(image_path)

        rooms_out = []
        for persistent_id in sorted(matched_ids):
            room = self._rooms[persistent_id]
            rooms_out.append({
                "room_id": persistent_id,
                "cell_count": room["cell_count"],
                "centroid_row": room["centroid_row"],
                "centroid_col": room["centroid_col"],
                "category": room["category"],
                "viewpoint_count": len(room["viewpoint_ids"]),
                "representative_viewpoint_id": room["representative_viewpoint_id"],
            })
        return {"labels": persistent_labels, "rooms": rooms_out}

    def rooms_needing_classification(self) -> list[dict]:
        """대표 viewpoint가 있는데 아직 그 viewpoint 기준으로 분류를 안 했거나(또는
        대표가 바뀌었거나) 실패 쿨다운이 지난 room만 골라 [{"room_id", "image_path"}]로
        반환한다."""
        now = time.time()
        cooldown = float(config.ROOM_CLASSIFICATION_RETRY_COOLDOWN_SEC)
        pending = []
        for room in self._rooms.values():
            representative_id = room["representative_viewpoint_id"]
            if representative_id is None or not room["representative_image_path"]:
                continue
            if representative_id == room["classified_viewpoint_id"]:
                continue
            attempted_at = room["last_classification_attempt_time"]
            if attempted_at is not None and (now - attempted_at) < cooldown:
                continue
            pending.append({
                "room_id": room["room_id"],
                "image_path": room["representative_image_path"],
            })
        return pending

    def unvisited_rooms(self) -> list[dict]:
        """SysNav paper Sec. IV-B-2의 room-query navigation mode용 후보 - 기하학적으로는
        분할됐지만(문 너머로 살짝 스캔만 됨) 로봇이 실제로 들어가서 viewpoint를 남긴 적은
        없는 방들. category는 대표 viewpoint가 아직 없거나 분류 전이면 None(호출 쪽이
        "물체 있을법한 방" 우선순위에서 이런 방은 거리순으로만 취급하면 됨)."""
        return [
            {
                "room_id": room["room_id"],
                "category": room["category"],
                "centroid_row": room["centroid_row"],
                "centroid_col": room["centroid_col"],
            }
            for room in self._rooms.values()
            if not room["viewpoint_ids"]
        ]

    def set_category(self, room_id: int, category: str) -> None:
        room = self._rooms.get(int(room_id))
        if room is None:
            return
        room["category"] = str(category).strip().lower()
        room["classified_viewpoint_id"] = room["representative_viewpoint_id"]
        # 성공 시에는 cooldown을 안 건다 - classified_viewpoint_id ==
        # representative_viewpoint_id 자체가 "다시 물어볼 필요 없음"을 이미 보장하고,
        # 나중에 대표 viewpoint가 진짜로 바뀌면 즉시(쿨다운 없이) 재분류돼야 한다.
        # cooldown은 오직 "VLM 호출 자체가 실패"(키 없음 등)해서 매 사이클 재시도로
        # 스팸이 나는 경우를 막기 위한 것이다 (mark_classification_failed만 갱신).
        room["last_classification_attempt_time"] = None

    def mark_classification_failed(self, room_id: int) -> None:
        room = self._rooms.get(int(room_id))
        if room is None:
            return
        room["last_classification_attempt_time"] = time.time()
