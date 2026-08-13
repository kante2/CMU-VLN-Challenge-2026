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
            # 대표 이미지 - 배경을 지운 물체 단독 사진(속성 판정용). 반면 context_image는
            # 배경을 안 지운 채 여유를 두고 자른 사진(relation_image_verifier.py처럼
            # "주변에 참조 물체가 보이는가"를 판단할 때 씀 - representative_image엔
            # 배경 자체가 없어서 못 씀).
            "representative_image": observation.get("crop_image").copy() if isinstance(observation.get("crop_image"), np.ndarray) else None,
            "context_image": observation.get("context_image").copy() if isinstance(observation.get("context_image"), np.ndarray) else None,
            "representative_confidence": float(observation.get("confidence", 0.0)),
            # 대표 이미지의 confidence
            "confidence": float(observation.get("confidence", 0.0)),
            "observation_count": 1,
            "first_seen_time": timestamp,
            "last_seen_time": timestamp,
            "latest_bbox_2d": tuple(observation.get("bbox", (0, 0, 0, 0))),
            "num_points": int(observation.get("num_points", 0)),
            # SysNav paper Sec. IV-A-1의 self-attribute(φ, 예: color) - "on demand로
            # 추론하고 node에 append"한다는 설계 그대로, 처음엔 비어있다가 task가 실제로
            # 속성을 요구할 때만 VLM으로 채워지고 계속 캐싱된다 (reasoning/attribute_verifier.py).
            "self_attributes": {},
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
        context = observation.get("context_image")
        if isinstance(context, np.ndarray) and context.size and confidence >= node["representative_confidence"]:
            node["context_image"] = context.copy()

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

    # VLM으로 새로 추론한 self-attribute(φ) 결과를 node에 캐싱한다 (attribute_verifier.py가
    # 호출) - 논문의 "appending the results φ to each node"를 그대로 구현. 기존 캐시는
    # 유지하고 새로 들어온 것만 덮어써서, 나중에 다른 속성 질문이 와도 이미 확인된 속성은
    # 또 안 물어봐도 된다.
    def update_self_attributes(self, object_id: int, attributes: dict[str, bool]) -> None:
        with self._lock:
            node = self._nodes.get(int(object_id))
            if node is None:
                return
            node["self_attributes"] = {**node.get("self_attributes", {}), **attributes}
    # 저장된 모든 객체를 리스트로 반환
    def all_nodes(self) -> list[dict]:
        with self._lock:
            return [self._copy_node(node) for node in self._nodes.values()]

    def merge_duplicates(self) -> int:
        """같은 물체가 여러 노드로 갈라진 것을 하나로 합친다. 합친 개수를 반환.

        update()의 association은 관측이 들어오는 그 순간에만 판정하므로, 각도가 달라
        모양/외형 점수가 낮게 나오면 ASSOCIATION_THRESHOLD를 못 넘고 새 노드가 생긴다.
        그렇게 갈라진 노드는 이후로도 스스로 합쳐지지 않아서 계속 쌓인다 - 실측:
        home_building_1에서 sofa가 GT 4개인데 26개, picture가 GT 2개인데 15개로
        등록됐다(개수 세기 미션에서는 이게 곧 오답이다).

        판정은 gfchen01/semantic_mapping_with_360_camera_and_3d_lidar의 규칙을 따른다:
        같은 카테고리끼리 centroid 거리가 "두 물체 half-extent의 평균 크기 * 비율"보다
        가까우면 같은 물체로 본다. 큰 소파는 넉넉하게, 작은 물체는 촘촘하게 병합되도록
        크기에 적응하고, 아주 작은 물체를 위해 절대 하한을 함께 둔다.
        """
        with self._lock:
            merged_total = 0
            changed = True
            while changed:
                changed = False
                for source_id, target_id in self._find_duplicate_pair():
                    source, target = self._nodes[source_id], self._nodes[target_id]
                    # 관측이 많은 쪽을 남긴다 - 그쪽 위치가 더 수렴해 있다.
                    if int(target["observation_count"]) > int(source["observation_count"]):
                        source, target = target, source
                    self._absorb(source, target)
                    del self._nodes[int(target["object_id"])]
                    merged_total += 1
                    changed = True
                    break  # 노드가 사라졌으니 목록을 다시 훑는다
            return merged_total

    def _find_duplicate_pair(self) -> list[tuple[int, int]]:
        nodes = list(self._nodes.values())
        for i, first in enumerate(nodes):
            for second in nodes[i + 1:]:
                if first["category"] != second["category"]:
                    continue
                distance = float(np.linalg.norm(
                    np.asarray(first["position"], dtype=np.float64)
                    - np.asarray(second["position"], dtype=np.float64)
                ))
                half_first = np.asarray(first["extent_3d"], dtype=np.float64) / 2.0
                half_second = np.asarray(second["extent_3d"], dtype=np.float64) / 2.0
                size_threshold = float(np.linalg.norm((half_first + half_second) / 2.0)) \
                    * config.OBJECT_MERGE_SIZE_RATIO
                threshold = max(config.OBJECT_MERGE_MIN_DISTANCE_M, size_threshold)
                # 거리 조건만으로는 긴 물체의 양 끝을 각각 잡은 경우를 못 합친다 -
                # 실측: 2.7m 소파의 두 관측이 centroid 1.66m 떨어져 병합되지 않았다.
                # 그래서 bbox가 충분히 겹치면 거리와 무관하게 같은 물체로 본다.
                if distance < threshold or self._box_overlap_ratio(first, second) > \
                        config.OBJECT_MERGE_OVERLAP_RATIO:
                    return [(int(first["object_id"]), int(second["object_id"]))]
        return []

    @staticmethod
    def _box_overlap_ratio(first: dict, second: dict) -> float:
        """두 축정렬 bbox의 교집합 부피 / 작은 쪽 부피. 한쪽이 다른 쪽에 거의 들어가
        있으면 1에 가까워지므로, 크기가 다른 두 관측이 같은 물체인지 판단하기 좋다."""
        low = np.maximum(
            np.asarray(first["bbox_3d_min"], dtype=np.float64),
            np.asarray(second["bbox_3d_min"], dtype=np.float64),
        )
        high = np.minimum(
            np.asarray(first["bbox_3d_max"], dtype=np.float64),
            np.asarray(second["bbox_3d_max"], dtype=np.float64),
        )
        overlap = float(np.prod(np.clip(high - low, 0.0, None)))
        volumes = [
            float(np.prod(np.clip(
                np.asarray(node["bbox_3d_max"], dtype=np.float64)
                - np.asarray(node["bbox_3d_min"], dtype=np.float64), 0.0, None)))
            for node in (first, second)
        ]
        smaller = min(volumes)
        return overlap / smaller if smaller > 1e-9 else 0.0

    def _absorb(self, keep: dict, drop: dict) -> None:
        """drop 노드를 keep 노드로 흡수한다. 관측 횟수로 가중평균해서, 관측이 많은 쪽
        위치가 덜 흔들리게 한다."""
        keep_count = max(1, int(keep["observation_count"]))
        drop_count = max(1, int(drop["observation_count"]))
        total = keep_count + drop_count
        keep_position = np.asarray(keep["position"], dtype=np.float64)
        drop_position = np.asarray(drop["position"], dtype=np.float64)
        keep["position"] = tuple(
            float(v) for v in (keep_position * keep_count + drop_position * drop_count) / total
        )
        # 크기는 두 관측을 모두 담도록 bbox를 합집합으로 넓힌다.
        minimum = np.minimum(
            np.asarray(keep["bbox_3d_min"], dtype=np.float64),
            np.asarray(drop["bbox_3d_min"], dtype=np.float64),
        )
        maximum = np.maximum(
            np.asarray(keep["bbox_3d_max"], dtype=np.float64),
            np.asarray(drop["bbox_3d_max"], dtype=np.float64),
        )
        keep["bbox_3d_min"] = tuple(float(v) for v in minimum)
        keep["bbox_3d_max"] = tuple(float(v) for v in maximum)
        keep["extent_3d"] = tuple(float(v) for v in (maximum - minimum))
        keep["point_cloud"] = self._merge_points(keep["point_cloud"], drop["point_cloud"])
        keep["num_points"] = len(keep["point_cloud"])
        keep["observation_count"] = total
        keep["confidence"] = max(float(keep["confidence"]), float(drop["confidence"]))
        keep["last_seen_time"] = max(keep["last_seen_time"], drop["last_seen_time"])
        keep["self_attributes"] = {
            **drop.get("self_attributes", {}), **keep.get("self_attributes", {})
        }
        # 대표 이미지는 representative_confidence가 높은 쪽을 남긴다(노드의 실제 필드명은
        # crop_image가 아니라 representative_image다).
        if float(drop.get("representative_confidence", 0.0)) > float(
            keep.get("representative_confidence", 0.0)
        ):
            if isinstance(drop.get("representative_image"), np.ndarray):
                keep["representative_image"] = drop["representative_image"]
                keep["representative_confidence"] = drop.get("representative_confidence", 0.0)
            if isinstance(drop.get("context_image"), np.ndarray):
                keep["context_image"] = drop["context_image"]


def filter_reliable(candidates: list[dict]) -> tuple[list[dict], int]:
    """판단 시점에 신뢰도 낮은 후보를 걸러낸다. 반환: (남긴 후보, 걸러낸 개수).

    오탐은 특정 각도/프레임에서만 나타나므로 관측 횟수가 적고 confidence도 낮다
    (GT 대조 실측: 정탐 obs 중앙 17 / conf 0.89, 오탐 obs 중앙 2 / conf 0.49).
    두 조건을 함께 걸면 오탐 23개 중 14개가 사라지면서 GT 적중은 하나도 잃지 않았다.

    다만 전부 걸러지면 원본을 그대로 돌려준다 - 탐사가 짧게 끝나 진짜 물체도 관측이
    적을 수 있고, 그때 "답이 없다"고 하는 것보다 약한 후보로라도 답하는 게 낫다.
    object_memory 자체는 건드리지 않으므로, 관측이 더 쌓이면 다음 판단에서 통과한다.
    """
    kept = [
        candidate for candidate in candidates
        if int(candidate.get("observation_count", 1)) >= config.OBJECT_MIN_OBSERVATIONS
        and float(candidate.get("confidence", 0.0)) >= config.OBJECT_MIN_CONFIDENCE
    ]
    if candidates and not kept:
        return candidates, 0
    return kept, len(candidates) - len(kept)
