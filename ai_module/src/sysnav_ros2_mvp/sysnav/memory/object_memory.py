"""Persistent object nodes for the current single-room map."""

from __future__ import annotations

'''
여러 프레임에서 관측한 객체들을 실제 객체 단위로 누적·관리하는 저장소

->
로봇이 같은 의자를 여러 위치에서 반복해서 보면 perception은 매번 새로운 observation을 만드는데,
이를 같은 물체이면 하나로 병합하는 구조이다.


'''
import copy
import threading
import time

import numpy as np

from sysnav import config
from sysnav.memory.object_association import find_best_match


class ObjectMemory:
    def __init__(self) -> None:
        self._nodes: dict[int, dict] = {} 
        # 객체 노드들을 저장하는 dictionary
        # Key는 object_id, Value는 객체 정보
        self._next_id = 1
        # 여러 스레드가 동시에 Object Memory를 읽거나 수정할 때 데이터가 꼬이지 않도록 막기 위함.
        self._lock = threading.RLock()

    # object memory 를 초기화하는 함수
    def clear(self) -> None:
        with self._lock:
            self._nodes.clear()
            self._next_id = 1

    @staticmethod
    # 내부 객체를 외부에 반환할 때 복사본을 만드는 함수
    # 외부 코드가 원본을 직접 수정하면 Object Memory 내용이 의도치 않게 변할수 있어, 이를 방지하기 위해 COPY 를 사용
    # 즉, 외부에서 수정해도 원본에는 영향 없음
    def _copy_node(node: dict) -> dict:
        return {key: value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value) for key, value in node.items()}

    # 기존 객체와 매칭되지 않은 observation을 새로운 Object Node로 만드는 함수
    def _new_node(self, observation: dict, timestamp: float) -> dict:
        object_id = self._next_id # id발급
        self._next_id += 1
        return {
            # 기본 식별 정보
            "object_id": object_id,
            "category": str(observation["category"]).lower(),
            # 대표 3d위치
            "position": tuple(float(v) for v in observation["position"]),
            "point_cloud": observation.get("point_cloud", np.empty((0, 3), np.float32)).copy(),
            "bbox_3d_min": tuple(observation.get("bbox_3d_min", (0, 0, 0))),
            "bbox_3d_max": tuple(observation.get("bbox_3d_max", (0, 0, 0))),
            "extent_3d": tuple(observation.get("extent_3d", (0, 0, 0))),
            # 대표 이미지
            "representative_image": observation.get("crop_image").copy() if isinstance(observation.get("crop_image"), np.ndarray) else None,
            "representative_confidence": float(observation.get("confidence", 0.0)),
            # 대표 이미지의 confidence
            "confidence": float(observation.get("confidence", 0.0)),
            "observation_count": 1,
            "first_seen_time": timestamp,
            "last_seen_time": timestamp,
            "latest_bbox_2d": tuple(observation.get("bbox", (0, 0, 0, 0))),
            "num_points": int(observation.get("num_points", 0)),
        }

    # 기존 객체의 pointcloud와, 새 observation의 pointcloud를 합치는 함수
    @staticmethod
    def _merge_points(old_points: np.ndarray, new_points: np.ndarray) -> np.ndarray:
        # 유효한 배열만 선택
        # (1) old_points가 정상 배열이면 추가
        # (2) new_points가 정상 배열이면 추가
        # (3) None이나 빈 배열이면 제외
        arrays = [arr.reshape(-1, 3) for arr in (old_points, new_points) if isinstance(arr, np.ndarray) and arr.size]
        if not arrays:
            return np.empty((0, 3), dtype=np.float32)
        # 결합
        merged = np.concatenate(arrays, axis=0)
        # 최대 포인트 수 제한
        #  객체를 계속 관측하면 포인트가 무한히 늘어나므로 최대 4096개만 저장
        #  균등 샘플링
        if len(merged) > config.MEMORY_MAX_POINTS_PER_OBJECT:
            merged = merged[np.linspace(0, len(merged) - 1, config.MEMORY_MAX_POINTS_PER_OBJECT, dtype=np.int64)]
        return merged.astype(np.float32, copy=False)

    # 새 observation이 기존 객체와 같은 물체로 판단됐을 때 기존 node를 갱신
    '''
    기존 Object Node
            +
    새 Observation
            ↓
    병합된 Object Node
    '''
    def _merge(self, node: dict, observation: dict, timestamp: float, metrics: dict) -> None:
        # 위치 갱신 가중치
        # 관측 횟수에 따라 새 관측이 반영되는 비율을 정하여, 초기에는 새 위치를 많이 반영 -> 관측이 누적될수록 기존 위치를 안정적으로 유지
        count = int(node["observation_count"])
        alpha = 1.0 / min(count + 1, 10)
        old_position = np.asarray(node["position"], dtype=np.float64)
        new_position = np.asarray(observation["position"], dtype=np.float64)
        node["position"] = tuple(float(v) for v in ((1 - alpha) * old_position + alpha * new_position))
        # Point cloud 병합 - 이전 포인트와 현재 포인트를 합친다.
        node["point_cloud"] = self._merge_points(node["point_cloud"], observation.get("point_cloud"))
        # 3D 크기 병합
        old_extent = np.asarray(node["extent_3d"], dtype=np.float64)
        new_extent = np.asarray(observation.get("extent_3d", old_extent), dtype=np.float64)
        node["extent_3d"] = tuple(float(v) for v in ((1 - alpha) * old_extent + alpha * new_extent))
        # bounding box 갱신
        node["bbox_3d_min"] = tuple(observation.get("bbox_3d_min", node["bbox_3d_min"]))
        node["bbox_3d_max"] = tuple(observation.get("bbox_3d_max", node["bbox_3d_max"]))
        node["latest_bbox_2d"] = tuple(observation.get("bbox", node["latest_bbox_2d"]))
        # confidence갱신
        node["confidence"] = max(node["confidence"], float(observation.get("confidence", 0.0)))
        node["last_seen_time"] = timestamp
        node["observation_count"] = count + 1
        node["num_points"] = len(node["point_cloud"])
        # 이번 observation과 기존 객체가 얼마나 유사했는지 저장
        node["association_score"] = float(metrics["score"])
        if metrics.get("observation_histogram") is not None:
            node["appearance_histogram"] = metrics["observation_histogram"].copy()
        crop = observation.get("crop_image")
        confidence = float(observation.get("confidence", 0.0))
        # 새로 들어온 객체 crop 이미지가 존재하고
        # +
        # 기존 대표 이미지보다 detection confidence가 높거나 같으면
        #         ↓
        # 새 crop 이미지를 대표 이미지로 교체
        if isinstance(crop, np.ndarray) and crop.size and confidence >= node["representative_confidence"]:
            node["representative_image"] = crop.copy()
            node["representative_confidence"] = confidence

    def update(self, observations: list[dict], timestamp: float | None = None) -> list[int]:
        timestamp = time.time() if timestamp is None else float(timestamp)
        changed_object_id_list = []
        with self._lock: # 전체 수정 과정이 끝날 때까지 다른 스레드가 메모리를 동시에 수정하지 못하게 잠근 상태 ------------------------------------
            for observation in observations:
                # 같은 카테고리만 먼저 추린다.
                same_category = [node for node in self._nodes.values() if node["category"] == str(observation["category"]).lower()]
                # 최적 매칭 검색
                match, metrics = find_best_match(same_category, observation)
                # 매칭되지 않으면 새 객체 
                if match is None:
                    node = self._new_node(observation, timestamp)
                    self._nodes[node["object_id"]] = node
                    changed_object_id_list.append(node["object_id"])
                else:
                    self._merge(match, observation, timestamp, metrics)
                    changed_object_id_list.append(match["object_id"])
        return changed_object_id_list # 새로 생성되거나 갱신된 객체 ID를 반환

    # 특정 카테고리의 객체만 반환한다.
    def find_by_category(self, category: str) -> list[dict]:
        with self._lock:
            return [self._copy_node(node) for node in self._nodes.values() if node["category"] == category.strip().lower()]
    # 객체 ID 하나로 객체를 조회
    def get(self, object_id: int) -> dict | None:
        with self._lock:
            node = self._nodes.get(int(object_id))
            return None if node is None else self._copy_node(node)
    # 저장된 모든 객체를 리스트로 반환
    def all_nodes(self) -> list[dict]:
        with self._lock:
            return [self._copy_node(node) for node in self._nodes.values()]
