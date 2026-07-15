#!/usr/bin/env python3
"""
Object Filtering (SORT-3D Module 3) 중 "후보를 실제로 좁히는" 부분.

relation_parser.py가 질문에서 관계절(between/near/closest/...)을 텍스트로
뽑아내면, 여기서는 그 관계를 실제로 적용해서 candidate 후보 목록을 좁힌다:
  1. 각 후보의 대략적인 3D 위치 계산 (box 기반, SAM 없이 — 필터링용이라
     정밀도보다 속도가 중요하다. 최종 선택된 후보 1개만 handlers 쪽에서
     SAM으로 정밀하게 다시 계산한다)
  2. 관계절에 나온 랜드마크(table, window 등)를 그 자리에서 검출+위치 계산
     (아직 영속적인 semantic map이 없어서 즉석으로 처리한다 — 나중에
     SORT-3D Module 1(instance-level mapping)이 생기면 여기를 map 조회로
     바꾸면 된다)
  3. spatial/relations.py의 결정론적 함수로 후보 필터링/정렬
     (VLM이 거리/방향을 직접 판단하면 부정확하므로)

t3_object_reference_solver와 t2_numerical_solver가 공용으로 쓴다.
"""

from tmah_vlm.sensor_process.projector import box_to_3d
from tmah_vlm.reasoning.spatial import relations as spatial_relations
from tmah_vlm.reasoning.spatial.relation_parser import extract_relations, all_referenced_landmarks
from tmah_vlm.common.helpers import get_robot_pose


def localize_candidate_points(node, detections, image_size, scan_points_map, image_stamp):
    """각 후보 box의 대략적인 3D 위치를 계산한다 (SAM 없이, box 기반이라 빠름).

    실패한 후보는 None으로 남긴다 (호출하는 쪽에서 걸러서 써야 함).
    """
    points = []
    for det in detections:
        try:
            result = box_to_3d(
                det.box, image_size, scan_points_map, node.transformer,
                image_stamp=image_stamp,
            )
            points.append(result["point"])
        except Exception:
            points.append(None)
    return points


def localize_landmark(node, image, image_stamp, scan_points_map, landmark_name):
    """랜드마크 이름 하나를 그 자리에서 검출+3D 위치 계산한다.

    아직 영속적 semantic map이 없어서 즉석으로 처리한다 (모듈 1이 생기면 대체할 부분).
    """
    if node.detector is None:
        return None

    try:
        landmark_detections = node.detector.detect(image, landmark_name)
    except Exception as error:
        node.get_logger().warn(f"[ObjectFilter] landmark detect failed ({landmark_name}): {error}")
        return None

    if len(landmark_detections) == 0:
        return None

    best = landmark_detections[0]  # detect()가 score 내림차순으로 정렬해둔다.

    try:
        result = box_to_3d(
            best.box, image.size, scan_points_map, node.transformer,
            target_name=landmark_name, image_stamp=image_stamp,
        )
        return result["point"]
    except Exception as error:
        node.get_logger().warn(f"[ObjectFilter] landmark localize failed ({landmark_name}): {error}")
        return None


def _apply_relation_mask(relation, candidate_points, landmark_positions, viewer_point):
    """relation 하나를 candidate_points에 적용해서 bool mask를 반환한다.

    판정 불가(랜드마크를 못 찾음, refs 개수가 안 맞음 등)면 None을 반환하고,
    호출하는 쪽은 그 relation을 무시하고 다음 relation으로 넘어간다.
    """
    rel_type = relation["type"]
    refs = relation["refs"]

    if rel_type == "between":
        if len(refs) != 2:
            return None
        a = landmark_positions.get(refs[0])
        b = landmark_positions.get(refs[1])
        if a is None or b is None:
            return None
        return spatial_relations.find_between(candidate_points, a, b)

    if len(refs) != 1:
        return None

    landmark = landmark_positions.get(refs[0])
    if landmark is None:
        return None

    if rel_type == "near":
        return spatial_relations.find_near(candidate_points, landmark)
    if rel_type == "on":
        # "표면 위에 얹혀있다" 근사: 살짝만 위, 수평으로는 가까워야 함.
        return spatial_relations.find_above(candidate_points, landmark, min_height_diff=0.0)
    if rel_type == "above":
        return spatial_relations.find_above(candidate_points, landmark)
    if rel_type == "below":
        return spatial_relations.find_below(candidate_points, landmark)
    if rel_type == "left":
        return spatial_relations.find_left(candidate_points, viewer_point, landmark)
    if rel_type == "right":
        return spatial_relations.find_right(candidate_points, viewer_point, landmark)

    return None


def filter_candidates_by_relations(node, question, detections, image, image_stamp, scan_points_map):
    """
    질문의 공간관계절로 후보를 deterministic하게 좁힌다.

    반환: 남은 후보들의 index 리스트 (원본 detections 기준).
      - 관계가 아예 없으면 전체 index를 그대로 반환한다 (필터링 안 함,
        기존 동작과 동일).
      - 필터링했는데 하나도 안 남으면(랜드마크를 못 찾았거나 조건에 맞는
        후보가 없으면) 전체 index로 되돌린다 — 판정 실패로 후보가 아예
        사라지는 것보단, 기존처럼 전체 후보에서 고르는 게 안전하다.
      - "closest"/"farthest"가 있으면 그 기준으로 1개만 남긴 리스트를 반환한다.
    """
    relations = extract_relations(question)
    all_indices = list(range(len(detections)))
    if not relations:
        return all_indices

    candidate_points = localize_candidate_points(node, detections, image.size, scan_points_map, image_stamp)

    landmark_names = all_referenced_landmarks(relations)
    landmark_positions = {
        name: localize_landmark(node, image, image_stamp, scan_points_map, name)
        for name in landmark_names
    }

    robot_pose = get_robot_pose(node)
    viewer_point = (robot_pose["x"], robot_pose["y"], robot_pose["z"])

    remaining = [i for i in all_indices if candidate_points[i] is not None]
    ranking_relation = None

    for relation in relations:
        if relation["type"] in ("closest", "farthest"):
            ranking_relation = relation
            continue

        if not remaining:
            break

        mask = _apply_relation_mask(
            relation,
            [candidate_points[i] for i in remaining],
            landmark_positions,
            viewer_point,
        )
        if mask is None:
            continue

        remaining = [idx for idx, keep in zip(remaining, mask) if keep]

    if not remaining:
        node.get_logger().warn(
            "[ObjectFilter] spatial relation filter matched nothing, fallback to all candidates"
        )
        return all_indices

    if ranking_relation is not None:
        landmark = landmark_positions.get(ranking_relation["refs"][0])
        if landmark is not None:
            pts = [candidate_points[i] for i in remaining]
            picker = (
                spatial_relations.closest_to
                if ranking_relation["type"] == "closest"
                else spatial_relations.farthest_from
            )
            best = picker(pts, landmark)
            if best is not None:
                return [remaining[best]]

    return remaining
