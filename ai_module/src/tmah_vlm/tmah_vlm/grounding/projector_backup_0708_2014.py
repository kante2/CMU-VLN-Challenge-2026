#!/usr/bin/env python3
"""
2D 검출 박스와 PointCloud2를 이용해 3D target을 찾는 코드.

핵심 수정:
  기존 방식은 box 중심 ray와 가까운 LiDAR 점을 고르는 방식이었다.
  이 방식은 TV 뒤쪽을 보고 있어도, ray 근처에 있는 앞 선반/테이블 점이
  더 가까우면 그 점을 target으로 잡는 문제가 있다.

  따라서 먼저 PointCloud를 camera frame으로 다시 투영하고,
  선택된 2D detection box 안에 실제로 들어오는 점들만 후보로 사용한다.
  그 뒤 depth cluster를 만들고, box 중심과 가장 잘 맞는 cluster의 대표점을
  3D target으로 사용한다.

역할 분리:
  - panorama.py: 이미지 픽셀 <-> camera ray
  - tf/coordinate_transform.py: camera/sensor/map 좌표 변환
  - projector.py: 2D box 기반 3D point 선택
"""

import math

import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

from tmah_vlm import config
from tmah_vlm.grounding.panorama import pixel_to_ray, ray_to_pixel


def pointcloud_to_xyz(scan_msg):
    """PointCloud2 메시지를 N x 3 numpy 배열로 바꾼다."""
    points = pc2.read_points(
        scan_msg,
        field_names=("x", "y", "z"),
        skip_nans=True,
    )
    return np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float64)


def get_box_center(box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy


def shrink_box(box, scale):
    """
    detection box를 중심 기준으로 조금 줄인다.

    목적:
      box 가장자리에는 배경/앞 물체가 섞이기 쉽기 때문에,
      우선 box 안쪽의 point를 더 신뢰한다.
    """
    x1, y1, x2, y2 = box
    cx, cy = get_box_center(box)
    half_w = (x2 - x1) * 0.5 * scale
    half_h = (y2 - y1) * 0.5 * scale
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def box_to_camera_ray(box, image_width, image_height):
    """검출 박스 중심을 camera frame ray로 변환."""
    cx, cy = get_box_center(box)
    return pixel_to_ray(cx, cy, image_width, image_height)


def project_camera_points_to_image(points_camera, image_width, image_height):
    """
    camera frame point들을 panorama 이미지 픽셀로 투영한다.

    반환:
      pixels: N x 2, 각 point의 (u, v)
      ranges: N, camera 원점에서 point까지 거리
    """
    if points_camera is None or len(points_camera) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

    points = np.asarray(points_camera, dtype=np.float64)
    ranges = np.linalg.norm(points, axis=1)

    pixels = []
    for point in points:
        px, py = ray_to_pixel(point, image_width, image_height)
        pixels.append((px, py))

    return np.asarray(pixels, dtype=np.float64), ranges


def make_box_candidate_mask(pixels, box, image_width):
    """
    투영된 pixel이 box 내부에 들어오는지 확인한다.

    360도 panorama에서는 box가 이미지 좌우 경계를 걸칠 수 있으므로,
    x 좌표는 약간의 wrap-around도 허용한다.
    """
    x1, y1, x2, y2 = box
    px = pixels[:, 0]
    py = pixels[:, 1]

    y_mask = (py >= y1) & (py <= y2)

    # 일반적인 경우
    x_mask_normal = (px >= x1) & (px <= x2)

    # box가 좌우 경계 근처에 있을 때를 위한 보조 처리
    px_left_wrap = px + image_width
    px_right_wrap = px - image_width
    x_mask_wrap = ((px_left_wrap >= x1) & (px_left_wrap <= x2)) | \
                  ((px_right_wrap >= x1) & (px_right_wrap <= x2))

    return y_mask & (x_mask_normal | x_mask_wrap)


def normalized_pixel_error(pixels, box):
    """
    box 중심과 후보 point 투영점 사이의 정규화된 거리.

    0에 가까울수록 detection box 중심과 잘 맞는다.
    """
    x1, y1, x2, y2 = box
    cx, cy = get_box_center(box)
    half_w = max((x2 - x1) * 0.5, 1.0)
    half_h = max((y2 - y1) * 0.5, 1.0)

    dx = (pixels[:, 0] - cx) / half_w
    dy = (pixels[:, 1] - cy) / half_h
    return np.sqrt(dx * dx + dy * dy)


def split_depth_clusters(ranges):
    """
    거리값을 기준으로 depth layer를 나눈다.

    예:
      앞 선반 cluster와 뒤 TV/벽 cluster가 거리 차이를 가지면,
      서로 다른 cluster로 분리된다.
    """
    if len(ranges) == 0:
        return []

    order = np.argsort(ranges)
    sorted_ranges = ranges[order]

    clusters = []
    start = 0

    for i in range(1, len(sorted_ranges)):
        gap = sorted_ranges[i] - sorted_ranges[i - 1]
        if gap > config.BBOX_DEPTH_CLUSTER_GAP_M:
            clusters.append(order[start:i])
            start = i

    clusters.append(order[start:])
    return clusters


def target_prefers_far_depth(target_name):
    """
    TV/picture/window처럼 뒤쪽 평면에 붙어 있는 물체인지 판단한다.

    이런 물체는 detection box 안에 앞 선반/테이블 point가 겹쳐 들어와도
    가까운 cluster가 아니라 뒤쪽 cluster를 target으로 잡는 편이 안전하다.
    """
    if target_name is None:
        return False

    name = str(target_name).lower()

    for keyword in config.DEPTH_POLICY_FAR_OBJECTS:
        if keyword in name:
            return True

    return False


def summarize_depth_clusters(clusters, pixel_error, ranges):
    """cluster별 중심 오차와 depth 통계를 만든다."""
    summaries = []

    for cluster in clusters:
        count = len(cluster)

        if count < config.BBOX_MIN_CLUSTER_POINTS:
            continue

        summary = {
            "cluster": cluster,
            "count": count,
            "median_error": float(np.median(pixel_error[cluster])),
            "median_depth": float(np.median(ranges[cluster])),
        }
        summaries.append(summary)

    return summaries


def center_ray_weights(pixel_error):
    """
    bbox 중심 ray에 가까운 point일수록 큰 가중치를 준다.

    pixel_error는 normalized_pixel_error() 결과다.
      0.0 = box 중심
      1.0 = box 반폭/반높이 근처

    sigma가 작을수록 중심부 ray만 강하게 본다.
    """
    sigma = max(float(config.BBOX_DEPTH_MODE_CENTER_SIGMA), 1e-3)
    return np.exp(-0.5 * (pixel_error / sigma) ** 2)


def summarize_depth_bins(candidate_ranges, pixel_error):
    """
    bbox 안으로 들어온 point들의 거리 분포를 histogram/bin으로 요약한다.

    핵심:
      단순 point 개수가 아니라, bbox 중심부 ray에 가까운 point에
      더 높은 weight를 준 weighted_count를 같이 계산한다.
    """
    if len(candidate_ranges) == 0:
        return []

    bin_width = max(float(config.BBOX_DEPTH_MODE_BIN_M), 1e-3)
    min_depth = float(np.min(candidate_ranges))
    max_depth = float(np.max(candidate_ranges))
    weights = center_ray_weights(pixel_error)

    if max_depth <= min_depth:
        return [{
            "indices": np.arange(len(candidate_ranges), dtype=np.int64),
            "count": len(candidate_ranges),
            "weighted_count": float(np.sum(weights)),
            "median_depth": float(np.median(candidate_ranges)),
            "median_error": float(np.median(pixel_error)),
            "min_error": float(np.min(pixel_error)),
            "depth_min": float(min_depth),
            "depth_max": float(max_depth),
        }]

    # range를 일정 간격 bin으로 나눈다.
    bin_ids = np.floor((candidate_ranges - min_depth) / bin_width).astype(np.int64)
    summaries = []

    for bin_id in sorted(set(bin_ids.tolist())):
        indices = np.where(bin_ids == bin_id)[0]
        if len(indices) < config.BBOX_MIN_CLUSTER_POINTS:
            continue

        depths = candidate_ranges[indices]
        errors = pixel_error[indices]
        bin_weights = weights[indices]

        summaries.append({
            "indices": indices,
            "count": int(len(indices)),
            "weighted_count": float(np.sum(bin_weights)),
            "median_depth": float(np.median(depths)),
            "median_error": float(np.median(errors)),
            "min_error": float(np.min(errors)),
            "depth_min": float(np.min(depths)),
            "depth_max": float(np.max(depths)),
        })

    return summaries


def choose_depth_mode_bin(candidate_ranges, pixel_error, target_name=None):
    """
    bbox ray bundle 안에서 가장 지배적인 거리 bin을 선택한다.

    핵심 정책:
      - bbox 영역 안에 투영된 point들의 depth histogram을 만든다.
      - point 개수가 가장 많은 bin을 object depth로 본다.
      - 개수가 비슷한 bin이 있으면 box 중심과 더 잘 맞는 bin을 고른다.

    이 방식은 특정 point 하나를 고르는 것이 아니라,
    "박스 영역에 해당하는 여러 ray들의 거리 분포"를 보는 방식이다.
    """
    summaries = summarize_depth_bins(candidate_ranges, pixel_error)

    if len(summaries) == 0:
        if len(candidate_ranges) == 0:
            return None, {"policy": "no_depth_mode"}

        best_index = int(np.argmin(pixel_error))
        return np.array([best_index], dtype=np.int64), {
            "policy": "single_point_fallback",
            "cluster_depth": float(candidate_ranges[best_index]),
            "cluster_error": float(pixel_error[best_index]),
            "cluster_count": 1,
            "depth_bin_width": float(config.BBOX_DEPTH_MODE_BIN_M),
        }

    # 1순위: bbox 중심 ray에 가까운 point들의 weighted_count가 큰 depth bin.
    # 2순위: weighted_count가 거의 비슷하면 box 중심에 더 가까운 bin.
    # 3순위: 그래도 비슷하면 일반 물체는 가까운 bin, 벽면 물체는 먼 bin.
    max_count = max(item["count"] for item in summaries)
    max_weighted_count = max(item["weighted_count"] for item in summaries)
    count_keep_ratio = float(config.BBOX_DEPTH_MODE_COUNT_KEEP_RATIO)
    candidates = [
        item for item in summaries
        if item["weighted_count"] >= max_weighted_count * count_keep_ratio
    ]

    if len(candidates) == 0:
        candidates = summaries

    use_far_depth = target_prefers_far_depth(target_name)

    best_item = None
    best_score = None

    for item in candidates:
        count_score = float(item["weighted_count"])
        center_error = min(item["median_error"], item["min_error"])
        center_penalty = float(config.BBOX_DEPTH_MODE_CENTER_PENALTY) * center_error

        if use_far_depth:
            depth_tie_break = float(config.BBOX_DEPTH_MODE_DEPTH_TIE_WEIGHT) * item["median_depth"]
        else:
            depth_tie_break = -float(config.BBOX_DEPTH_MODE_DEPTH_TIE_WEIGHT) * item["median_depth"]

        # score가 클수록 좋다.
        score = count_score - center_penalty + depth_tie_break

        if best_score is None or score > best_score:
            best_score = score
            best_item = item

    info = {
        "policy": "bbox_ray_depth_mode",
        "cluster_depth": best_item["median_depth"],
        "cluster_error": best_item["median_error"],
        "cluster_min_error": best_item["min_error"],
        "cluster_count": best_item["count"],
        "cluster_weighted_count": best_item["weighted_count"],
        "depth_bin_width": float(config.BBOX_DEPTH_MODE_BIN_M),
        "depth_bin_min": best_item["depth_min"],
        "depth_bin_max": best_item["depth_max"],
        "max_bin_count": int(max_count),
        "max_weighted_bin_count": float(max_weighted_count),
        "num_depth_bins": int(len(summaries)),
    }
    return best_item["indices"], info


def make_target_from_center_ray_depth(box, image_size, depth_m, transformer):
    """
    선택된 depth를 bbox center ray에 태워 최종 target을 만든다.

    point들의 3D 평균을 바로 쓰면 앞 선반/테이블 쪽으로 끌릴 수 있다.
    그래서 bbox 내부 포인트들은 depth 추정용으로만 쓰고,
    최종 방향은 detection bbox 중심 ray를 사용한다.
    """
    image_width, image_height = image_size
    ray_camera = box_to_camera_ray(box, image_width, image_height)
    target_camera = ray_camera * float(depth_m)
    target_map = transformer.transform_point(
        target_camera,
        config.FRAME_CAMERA,
        config.FRAME_MAP,
    )
    return target_map


def find_best_point_by_box_projection(origin_map, points_map, transformer, box, image_size, target_name=None):
    """
    selected 2D box를 ray bundle로 보고, 그 안의 거리 분포 mode로 3D target을 정한다.

    이전 방식:
      bbox 안 point들의 3D 평균 또는 중심 ray 근처의 한 점을 선택.
      -> 앞 선반/테이블 point에 끌릴 수 있음.

    현재 방식:
      1. PointCloud를 camera image로 투영
      2. selected bbox 안에 들어온 point만 수집
      3. bbox 영역의 depth histogram을 만들고 가장 많은 depth bin 선택
      4. 선택된 bin의 median depth를 bbox center ray에 적용
      5. 그 점을 map frame target으로 변환
    """
    if points_map is None or len(points_map) == 0:
        return None, "no_scan", 0, {}

    image_width, image_height = image_size

    try:
        points_camera = transformer.transform_points(
            points_map,
            config.FRAME_MAP,
            config.FRAME_CAMERA,
        )
    except Exception:
        return None, "camera_projection_tf_failed", 0, {}

    pixels, ranges = project_camera_points_to_image(
        points_camera,
        image_width,
        image_height,
    )

    if len(pixels) == 0:
        return None, "empty_projection", 0, {}

    valid_range = (ranges >= config.BBOX_MIN_DEPTH_M) & \
                  (ranges <= config.BBOX_MAX_DEPTH_M)

    pixels = pixels[valid_range]
    ranges = ranges[valid_range]

    if len(pixels) == 0:
        return None, "no_valid_depth", 0, {}

    # 1차: box를 줄인 내부 영역으로 depth mode를 본다.
    inner_box = shrink_box(box, config.BBOX_INNER_SCALE)
    mask = make_box_candidate_mask(pixels, inner_box, image_width)
    method = "bbox_ray_bundle_inner"

    # 2차: 내부 영역에 점이 부족하면 전체 box를 사용한다.
    if int(np.sum(mask)) < config.BBOX_MIN_POINTS:
        mask = make_box_candidate_mask(pixels, box, image_width)
        method = "bbox_ray_bundle_full"

    matched_count = int(np.sum(mask))

    if matched_count == 0:
        return None, "no_bbox_projected_points", 0, {}

    candidate_pixels_xy = pixels[mask]
    candidate_ranges = ranges[mask]
    pixel_error = normalized_pixel_error(candidate_pixels_xy, box)

    selected_indices, depth_info = choose_depth_mode_bin(
        candidate_ranges,
        pixel_error,
        target_name,
    )

    if selected_indices is None or len(selected_indices) == 0:
        return None, "no_depth_mode_bin", matched_count, {}

    selected_depths = candidate_ranges[selected_indices]
    depth_m = float(np.median(selected_depths))

    point_map = make_target_from_center_ray_depth(
        box,
        image_size,
        depth_m,
        transformer,
    )

    method_name = method + "_depth_mode"
    depth_info["selected_depth_m"] = depth_m
    return point_map, method_name, matched_count, depth_info


def find_best_point_on_ray(origin_map, ray_map, points_map):
    """
    map frame에서 ray 방향과 가장 잘 맞는 point cloud 점을 찾는다.

    이 함수는 fallback용이다.
    가능한 경우에는 find_best_point_by_box_projection()을 먼저 사용한다.
    """
    if points_map is None or len(points_map) == 0:
        fallback = origin_map + ray_map * config.FALLBACK_DEPTH_M
        return fallback, "fallback_no_scan", 0

    vectors = points_map - origin_map
    distances = np.linalg.norm(vectors, axis=1)
    valid = distances > 1e-3

    if not np.any(valid):
        fallback = origin_map + ray_map * config.FALLBACK_DEPTH_M
        return fallback, "fallback_empty_scan", 0

    vectors = vectors[valid]
    points = points_map[valid]
    distances = distances[valid]

    unit_vectors = vectors / distances[:, None]
    cos_sim = unit_vectors @ ray_map
    angles = np.arccos(np.clip(cos_sim, -1.0, 1.0))

    matched = angles < config.RAY_MATCH_ANGLE_RAD
    matched_count = int(np.sum(matched))

    if matched_count == 0:
        fallback = origin_map + ray_map * config.FALLBACK_DEPTH_M
        return fallback, "fallback_no_ray_match", 0

    candidate_points = points[matched]
    candidate_angles = angles[matched]
    candidate_distances = distances[matched]

    # fallback에서도 너무 가까운 점만 무조건 고르지 않도록 각도 오차를 더 크게 본다.
    score = candidate_angles + 0.02 * candidate_distances
    best_index = int(np.argmin(score))
    return candidate_points[best_index], "fallback_lidar_ray_match", matched_count


def box_to_3d(box, image_size, scan_points_map, transformer, target_name=None):
    """
    2D box -> 3D map 좌표.

    image_size: PIL image.size, 즉 (width, height)
    scan_points_map: map frame으로 변환된 point cloud
    transformer: CoordinateTransformer 인스턴스
    """
    image_width, image_height = image_size

    ray_camera = box_to_camera_ray(box, image_width, image_height)
    ray_map = transformer.transform_direction(
        ray_camera,
        config.FRAME_CAMERA,
        config.FRAME_MAP,
    )
    origin_map = transformer.get_frame_origin(
        config.FRAME_CAMERA,
        config.FRAME_MAP,
    )

    # 1순위: point cloud를 image box로 다시 투영해서 선택한다.
    point_map, method, matched_count, cluster_info = find_best_point_by_box_projection(
        origin_map,
        scan_points_map,
        transformer,
        box,
        image_size,
        target_name,
    )

    # 2순위: box 내부 투영점이 없을 때만 기존 ray 방식 사용.
    if point_map is None:
        point_map, method, matched_count = find_best_point_on_ray(
            origin_map,
            ray_map,
            scan_points_map,
        )
        cluster_info = {"policy": "ray_fallback"}

    return {
        "point": (float(point_map[0]), float(point_map[1]), float(point_map[2])),
        "origin": (float(origin_map[0]), float(origin_map[1]), float(origin_map[2])),
        "ray": (float(ray_map[0]), float(ray_map[1]), float(ray_map[2])),
        "method": method,
        "n_matched": matched_count,
        "target_name": str(target_name),
        "cluster_policy": cluster_info.get("policy", "unknown"),
        "cluster_depth_m": cluster_info.get("cluster_depth", -1.0),
        "cluster_error": cluster_info.get("cluster_error", -1.0),
        "cluster_count": cluster_info.get("cluster_count", 0),
        "cluster_weighted_count": cluster_info.get("cluster_weighted_count", -1.0),
        "cluster_min_error": cluster_info.get("cluster_min_error", -1.0),
    }


def approach_waypoint(target_xyz, robot_pose):
    """target 앞 APPROACH_STANDOFF_M 지점으로 이동할 waypoint 계산."""
    tx, ty, _ = target_xyz
    rx = robot_pose["x"]
    ry = robot_pose["y"]

    dx = tx - rx
    dy = ty - ry
    distance = math.hypot(dx, dy)
    heading = math.atan2(dy, dx)

    if distance > config.APPROACH_STANDOFF_M:
        ratio = (distance - config.APPROACH_STANDOFF_M) / distance
        wx = rx + dx * ratio
        wy = ry + dy * ratio
    else:
        wx = rx
        wy = ry

    return {"x": wx, "y": wy, "heading": heading}

# ========================================
# Debug: point cloud projection overlay
# ========================================

def save_projection_overlay(image, box, scan_points_map, transformer, target_name="object"):
    """
    현재 PointCloud가 camera image 위에 어디로 투영되는지 저장한다.

    이 이미지를 보면 다음을 바로 판단할 수 있다.
      - PointCloud 투영이 실제 물체 위치와 맞는지
      - bbox 안으로 들어오는 point가 TV가 아니라 앞 선반 쪽인지
      - PANO_FORWARD_X / camera TF / image-scan sync가 틀어졌는지
    """
    import os
    from datetime import datetime
    from PIL import ImageDraw

    os.makedirs(config.DEBUG_DIR, exist_ok=True)

    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size

    x1, y1, x2, y2 = box
    inner_box = shrink_box(box, config.BBOX_INNER_SCALE)
    ix1, iy1, ix2, iy2 = inner_box

    # selected bbox와 inner bbox를 먼저 그린다.
    draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=4)
    draw.rectangle([ix1, iy1, ix2, iy2], outline=(80, 220, 80), width=3)

    if scan_points_map is None or len(scan_points_map) == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(config.DEBUG_DIR, f"proj_{timestamp}_{target_name}_no_scan.jpg")
        img.save(path, quality=90)
        return path

    points_camera = transformer.transform_points(
        scan_points_map,
        config.FRAME_MAP,
        config.FRAME_CAMERA,
    )
    pixels, ranges = project_camera_points_to_image(points_camera, width, height)

    valid = (ranges >= config.BBOX_MIN_DEPTH_M) & (ranges <= config.BBOX_MAX_DEPTH_M)
    pixels = pixels[valid]
    ranges = ranges[valid]

    if len(pixels) == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(config.DEBUG_DIR, f"proj_{timestamp}_{target_name}_no_valid.jpg")
        img.save(path, quality=90)
        return path

    max_points = int(config.DEBUG_PROJECTION_MAX_POINTS)
    if len(pixels) > max_points:
        step = max(1, len(pixels) // max_points)
        pixels = pixels[::step]
        ranges = ranges[::step]

    inside_full = make_box_candidate_mask(pixels, box, width)
    inside_inner = make_box_candidate_mask(pixels, inner_box, width)

    # depth별 대략적인 점 크기. 가까운 점은 조금 크게 보이게 한다.
    for idx in range(len(pixels)):
        u = float(pixels[idx, 0])
        v = float(pixels[idx, 1])

        if u < 0 or u >= width or v < 0 or v >= height:
            continue

        depth = float(ranges[idx])
        radius = 2 if depth < 5.0 else 1

        if inside_inner[idx]:
            color = (255, 240, 40)   # inner bbox 안 point: 노랑
        elif inside_full[idx]:
            color = (255, 120, 40)   # bbox 안 point: 주황
        else:
            color = (120, 180, 255)  # bbox 밖 point: 파랑

        draw.ellipse([u-radius, v-radius, u+radius, v+radius], fill=color)

    header = (
        f"projection overlay | target={target_name} | "
        f"yellow=inner, orange=box, blue=outside | "
        f"points={len(pixels)}"
    )
    draw.rectangle([0, 0, width, 24], fill=(0, 0, 0))
    draw.text((5, 4), header, fill=(255, 255, 255))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() else "_" for c in str(target_name))[:40]
    path = os.path.join(config.DEBUG_DIR, f"proj_{timestamp}_{safe_name}.jpg")
    img.save(path, quality=90)
    return path
