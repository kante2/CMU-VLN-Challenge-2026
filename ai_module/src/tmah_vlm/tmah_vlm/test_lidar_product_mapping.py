#!/usr/bin/env python3
"""
물체 매핑 테스트 — test_lidar_mapping.py(기하 지도 + 탐색 + 복귀) 위에 "물체"를 얹은 버전.

test_lidar_mapping.py가 만드는 건 "빈 공간/벽"만 있는 기하 지도라, 질문에 답하려면
"어떤 물체가 어디 있는지"가 더 필요하다. 이 스크립트는 로봇이 탐색하며 돌아다니는 동안
보이는 물체를 검출해서 그 3D 위치를 지도 위에 누적한다.

  LidarMappingNode (test_lidar_mapping.py)  : 스캔 누적 + frontier 탐색 + 원점 복귀
        └─ ObjectMappingNode (이 파일)      : 그 위에 물체 검출 -> 3D 위치 -> 물체 맵 누적

main_node.py의 t3와 다른 점: t3는 "질문에서 뽑은 대상 하나"만 찾지만, 여기선 질문 없이
미리 정한 TARGET_CLASSES를 전부 훑는다(탐색하면서 물체 맵을 미리 만들어 두는 게 목적).

--------------------------------------------------------------------------
계산량을 어떻게 줄였나 (전부 보면 감당이 안 되므로)
--------------------------------------------------------------------------
1. 클래스 고정      : TARGET_CLASSES 10개만. GroundingDINO는 zero-shot이라
                      "chair . table . ..." 한 프롬프트로 한 번에 다 검출한다.
                      -> 클래스가 늘어도 추론 횟수는 1회로 같다.
2. 검출 주기 throttle: 매 프레임(10Hz)이 아니라 DETECT_PERIOD_SEC(2초)에 1회.
                      매핑은 계속 10Hz로 돌고 검출만 띄엄띄엄 한다.
3. SAM 생략         : box_to_3d(segmentation_mask=None)이면 box 투영으로 대체된다.
                      SAM은 물체 1개당 1회라 10개면 수십 초(CPU) -> 절대 못 씀.
                      대신 3D 위치가 조금 거칠어지는데, 여러 번 관측해 평균내서 보완한다.
4. 워커 스레드      : 검출(수백 ms)을 타이머에서 돌리면 그동안 매핑/탐색 콜백이 멈춘다.
                      별도 스레드에서 돌리고 결과만 락 걸고 합친다.

--------------------------------------------------------------------------
물체 맵 누적 방식
--------------------------------------------------------------------------
검출 1건 -> box_to_3d로 map frame 3D 좌표. 같은 라벨이면서 MERGE_DIST_M 안에 있는
기존 물체가 있으면 "같은 물체"로 보고 관측을 합친다. 없으면 새 물체로 등록.
MIN_OBSERVATIONS번 이상 본 것만 "확정"으로 인정해서 발행한다(스쳐 지나간 오검출 필터).

같은 물체라도 볼 때마다 위치가 조금씩 다르게 잡히는데, 이걸 4단계로 보정한다:
  1. 관측 가중치   : 가깝고 / 검출 확신 높고 / LiDAR 점 많은 관측일수록 더 신뢰
                     (observation_weight) — 전부 동등 평균하면 나쁜 관측이 좋은 걸 오염시킨다.
  2. 가중 중앙값   : 최근 MAX_OBS_KEPT개 관측의 축별 가중 중앙값으로 대표 위치를 정한다.
                     평균과 달리 크게 튄 관측 하나에 안 끌려간다 (MappedObject.recompute).
  3. 재병합       : 위치가 크게 튀어 같은 물체가 2개로 갈라지면 merge_or_add로는 영영 못
                     합친다("새 검출 vs 기존"만 비교하므로). maintenance_tick이 주기적으로
                     "기존끼리" 다시 비교해 붙인다.
  4. 미확정 정리   : 1회만 보이고 오래 재관측 없는 건 오검출로 보고 버린다.

RViz (Fixed Frame = map):
  - /debug/object_map_markers   (MarkerArray) -> 물체 위치 CUBE + 라벨 텍스트
  - /debug/lidar_map_cloud      (PointCloud2) -> 누적 3D 기하 지도  (상속)
  - /debug/explore_goal         (Marker)      -> 현재 탐색 목표      (상속)
PNG(호스트 ai_module/debug/):
  - lidar_explore_latest.png -> 2D 격자 + 물체 위치(클래스별 색 점) 오버레이

실행: python3 src/tmah_vlm/tmah_vlm/test_lidar_product_mapping.py
주의: 이 파일은 tmah_vlm.test_lidar_mapping을 import하므로, 설치본이 최신이어야 한다
      (test_lidar_mapping.py를 고쳤다면 colcon build 후 실행할 것).
      tmah_vlm 본 노드(tmah_vlm.launch)와 동시에 켜지 말 것 — /way_point_with_heading 충돌.
"""

import threading
import time

import numpy as np
import cv2

import rclpy
from sensor_msgs.msg import Image as RosImage
from visualization_msgs.msg import Marker, MarkerArray

from tmah_vlm import config
from tmah_vlm.test_lidar_mapping import LidarMappingNode
from tmah_vlm.sensor_process.sensor_process import (
    grab_camera_image,
    detect_candidate_boxes,
    load_scan_points_in_map,
)
from tmah_vlm.sensor_process.projector import box_to_3d

from collections import deque
from types import SimpleNamespace


TOPIC_OBJECT_MARKERS = "/debug/object_map_markers"

# ===== 검출 대상 클래스 (여기 10개만 훑는다) =====
# GroundingDINO는 zero-shot이라 자유 문장이 되지만, 다 넣으면 오검출도 늘고 후처리가
# 커진다. 환경(실내)에서 의미 있고 LiDAR로 크기가 잡히는 것 위주로 골랐다.
TARGET_CLASSES = [
    "chair",
    "table",
    "sofa",
    "bed",
    "refrigerator",
    "tv monitor",
    "potted plant",
    "trash can",
    "cabinet",
    "door",
]
# GroundingDINO 멀티클래스 프롬프트 형식: "a . b . c ."
DETECT_PROMPT = " . ".join(TARGET_CLASSES) + " ."

# ===== 검출/누적 파라미터 =====
DETECT_PERIOD_SEC = 2.0        # 검출 주기(초). 낮추면 촘촘하지만 GPU 부하↑.
DETECT_MAX_DIST_M = 8.0        # 이보다 먼 검출은 버림(LiDAR가 성글어 3D가 부정확).
MIN_MATCHED_POINTS = 3         # box에 투영된 LiDAR 점이 이보다 적으면 위치를 못 믿음.
MERGE_DIST_M = 0.8             # 같은 라벨이 이 거리 안이면 같은 물체로 병합.
MIN_OBSERVATIONS = 2           # 이만큼 관측된 물체만 "확정"으로 보고 발행.
MARKER_PUBLISH_PERIOD_SEC = 1.0

# ----- 위치 정합/보정 -----
MAX_OBS_KEPT = 12              # 물체당 보관할 최근 관측 수(가중 중앙값 계산용).
MATCH_SATURATE = 30.0          # LiDAR 투영점이 이 정도면 3D 신뢰도 만점으로 본다.
MAINTENANCE_PERIOD_SEC = 5.0   # 재병합 + 미확정 정리 주기.
STALE_UNCONFIRMED_SEC = 30.0   # 미확정(관측 1회) 물체가 이 시간 재관측 없으면 삭제.


def weighted_median_1d(values, weights):
    """가중 중앙값. 평균과 달리 크게 튄 값 하나가 결과를 못 끌고 간다.

    값을 정렬해 놓고 가중치를 누적해서, 전체 가중치의 절반을 처음 넘는 지점의 값을 쓴다.
    (관측 대부분이 몰려 있으면 그 근처가 뽑히고, 멀리 튄 소수는 무시된다.)
    """
    order = np.argsort(values)
    v = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(np.median(v))
    idx = int(np.searchsorted(cw, cw[-1] / 2.0))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def observation_weight(score, dist_m, n_matched):
    """이 관측을 얼마나 믿을지 = 가중치.

    같은 물체라도 관측 품질이 천차만별이라 전부 동등하게 평균내면 나쁜 관측이 좋은 관측을
    오염시킨다. 세 가지를 곱해서 신뢰도를 만든다:
      - score      : 검출기가 확신할수록 ↑
      - 거리       : 가까울수록 ↑ (멀면 LiDAR가 성글어 3D가 부정확)
      - n_matched  : box에 투영된 LiDAR 점이 많을수록 ↑ (적으면 위치 추정이 취약)
    """
    match_factor = min(float(n_matched), MATCH_SATURATE) / MATCH_SATURATE
    return float(score) * match_factor / (1.0 + float(dist_m))


def class_color_bgr(label):
    """라벨 문자열 -> 고정 색(BGR). 같은 클래스는 항상 같은 색이 되도록 해시를 쓴다."""
    h = (hash(label) % 180)
    hsv = np.uint8([[[h, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


class MappedObject:
    """물체 맵의 원소 하나. 같은 물체의 여러 관측을 모아 "가중 중앙값"으로 위치를 정한다.

    단순 누적 평균을 쓰지 않는 이유:
      - 평균은 크게 튄 관측 하나에 통째로 끌려간다(특히 관측이 2~3개일 때 치명적).
      - 관측 품질이 제각각인데 평균은 전부 동등 취급한다.
    그래서 최근 MAX_OBS_KEPT개의 관측을 (좌표, 가중치)로 들고 있다가, 매번 가중 중앙값으로
    대표 위치를 다시 계산한다. 멀리서 잘못 잡힌 관측은 가중치도 낮고 중앙값에서도 밀려난다.
    """

    def __init__(self, label, xyz, score, size, weight):
        self.label = label
        # 관측 하나 = (xyz, weight, size, 관측시각)
        self.observations = deque(maxlen=MAX_OBS_KEPT)
        self.count = 0
        self.best_score = 0.0
        self.size = size
        self.xyz = np.asarray(xyz, dtype=np.float64)
        self.last_seen = time.time()
        self.add(xyz, score, size, weight)

    def add(self, xyz, score, size, weight):
        self.observations.append(
            (np.asarray(xyz, dtype=np.float64), float(weight), size, time.time())
        )
        self.count += 1
        self.best_score = max(self.best_score, float(score))
        self.last_seen = time.time()
        self.recompute()

    def absorb(self, other):
        """다른 물체(같은 놈이 갈라져 등록된 것)를 흡수한다."""
        for obs in other.observations:
            self.observations.append(obs)
        self.count += other.count
        self.best_score = max(self.best_score, other.best_score)
        self.last_seen = max(self.last_seen, other.last_seen)
        self.recompute()

    def recompute(self):
        """보관 중인 관측들로 대표 위치/크기를 다시 계산한다."""
        pts = np.array([o[0] for o in self.observations], dtype=np.float64)
        ws = np.array([o[1] for o in self.observations], dtype=np.float64)
        if len(pts) == 0:
            return
        if ws.sum() <= 0:
            ws = np.ones(len(ws))

        # 축별 가중 중앙값 (x, y, z 각각 독립적으로)
        self.xyz = np.array([weighted_median_1d(pts[:, k], ws) for k in range(3)])

        # 크기도 관측마다 흔들리므로 중앙값으로. (크기를 못 뽑은 관측은 제외)
        sizes = [o[2] for o in self.observations if o[2] is not None]
        if sizes:
            self.size = tuple(np.median(np.array(sizes, dtype=np.float64), axis=0))


class ObjectMappingNode(LidarMappingNode):
    def __init__(self):
        super().__init__()   # 기하 매핑 + 탐색 + 복귀는 그대로 상속

        # ----- 물체 검출에 필요한 추가 상태 -----
        # 기존 sensor_process 함수들이 node.latest_image / latest_scan / scan_buffer /
        # transformer / detector 를 쓰기 때문에 이름을 그대로 맞춰준다.
        self.latest_image = None
        self.latest_scan = None
        self.scan_buffer = deque(maxlen=config.SYNC_SCAN_BUFFER_SIZE)
        self.detector = None

        self.objects = []                     # list[MappedObject] — 물체 맵
        self.objects_lock = threading.Lock()  # 워커 스레드 <-> 메인 스레드 공유

        self.create_subscription(RosImage, config.TOPIC_IMAGE, self.image_callback, 5)
        self.pub_object_markers = self.create_publisher(MarkerArray, TOPIC_OBJECT_MARKERS, 5)

        # 마커 발행은 메인 스레드 타이머에서 (rclpy 발행을 워커 스레드에서 하지 않으려고).
        self.create_timer(MARKER_PUBLISH_PERIOD_SEC, self.publish_object_markers)
        # 물체 맵 정비(재병합 + 미확정 정리)는 가벼워서 타이머로 충분하다.
        self.create_timer(MAINTENANCE_PERIOD_SEC, self.maintenance_tick)

        # 모델 로딩과 검출은 둘 다 무거워서 백그라운드로 뺀다.
        threading.Thread(target=self.load_detector, daemon=True).start()
        threading.Thread(target=self.detect_worker, daemon=True).start()

        self.get_logger().info(f"[ObjMap] 검출 대상 {len(TARGET_CLASSES)}종: {TARGET_CLASSES}")
        self.get_logger().info(f"[ObjMap] prompt: {DETECT_PROMPT}")
        self.get_logger().info(f"[ObjMap] markers -> {TOPIC_OBJECT_MARKERS}")

    # ==================== 로딩 / 콜백 ====================
    def load_detector(self):
        try:
            from tmah_vlm.sensor_process.detector import GroundingDINODetector
            self.detector = GroundingDINODetector(
                box_threshold=config.BOX_THRESHOLD,
                text_threshold=config.TEXT_THRESHOLD,
            )
            self.get_logger().info("[ObjMap] GroundingDINO loaded")
        except Exception as error:
            self.get_logger().error(f"[ObjMap] GroundingDINO load failed: {error}")

    def image_callback(self, msg):
        # 콜백은 최신값 저장만 (무거운 추론은 워커에서).
        self.latest_image = msg

    def scan_callback(self, scan):
        # 부모의 매핑 처리에 더해, 물체 3D 계산용으로 scan을 버퍼에 남긴다.
        # (get_synced_scan_for_latest_image가 이미지 stamp에 가장 가까운 scan을 고를 때 씀)
        self.latest_scan = scan
        self.scan_buffer.append(scan)
        super().scan_callback(scan)

    # ==================== 검출 워커 ====================
    def detect_worker(self):
        """DETECT_PERIOD_SEC마다 한 번씩 검출을 돌린다 (별도 스레드).

        타이머(=메인 스레드)에서 돌리면 GroundingDINO 추론 수백 ms 동안 스캔 콜백과
        탐색 tick이 멈춰서 지도에 구멍이 나고 목표 발행이 밀린다. 그래서 스레드로 뺐다.
        """
        while rclpy.ok():
            time.sleep(DETECT_PERIOD_SEC)
            try:
                self.detect_once()
            except Exception as error:
                import traceback
                self.get_logger().error(f"[ObjMap] detect failed: {error}\n{traceback.format_exc()}")

    def detect_once(self):
        """카메라 1프레임 -> 10클래스 검출 -> 각 box의 3D 위치 -> 물체 맵에 누적."""
        if self.detector is None or self.latest_image is None or self.latest_scan is None:
            return

        # 기존 sensor_process 스텝을 그대로 재사용한다 (ctx 출력-인자 패턴).
        ctx = SimpleNamespace(
            image=None, image_stamp=None,
            detect_prompt=DETECT_PROMPT,
            detections=None, scan_points_map=None,
        )
        grab_camera_image(self, ctx)          # ctx.image, ctx.image_stamp
        detect_candidate_boxes(self, ctx)     # ctx.detections  (10클래스 한 번에)
        if not ctx.detections:
            return

        load_scan_points_in_map(self, ctx, log_tag="ObjMap")   # ctx.scan_points_map
        if ctx.scan_points_map is None:
            return

        robot_xy = np.array([self.robot_x(), self.robot_y()])
        added, merged, skipped = 0, 0, 0

        for det in ctx.detections:
            # SAM 생략 — segmentation_mask=None이면 box 투영으로 3D를 뽑는다(계산량 절감).
            # box_to_3d는 점을 하나도 못 찾으면 None을 주는 게 아니라 예외를 던질 수 있다.
            # 물체 하나 실패로 나머지 검출까지 날리지 않도록 건별로 감싼다.
            try:
                result = box_to_3d(
                    det.box,
                    ctx.image.size,
                    ctx.scan_points_map,
                    self.transformer,
                    det.label,
                    image_stamp=ctx.image_stamp,
                    segmentation_mask=None,
                )
            except Exception as error:
                self.get_logger().warn(
                    f"[ObjMap] '{det.label}' 3D 추정 실패 → 건너뜀: {error}",
                    throttle_duration_sec=5.0,
                )
                skipped += 1
                continue

            if result is None:
                skipped += 1
                continue

            # box에 투영된 LiDAR 점이 너무 적으면 위치를 신뢰할 수 없다(ray fallback 등).
            if int(result.get("n_matched", 0)) < MIN_MATCHED_POINTS:
                skipped += 1
                continue

            xyz = np.array(result["point"], dtype=np.float64)
            dist = float(np.linalg.norm(xyz[:2] - robot_xy))

            # 너무 먼 검출은 LiDAR가 성글어 3D가 부정확 -> 가까이 갔을 때 다시 잡으면 된다.
            if dist > DETECT_MAX_DIST_M:
                skipped += 1
                continue

            # 이 관측을 얼마나 믿을지: 가깝고/확신 높고/LiDAR 점 많을수록 큰 가중치.
            weight = observation_weight(det.score, dist, result.get("n_matched", 0))

            if self.merge_or_add(det.label, xyz, det.score, result.get("bbox_size"), weight):
                added += 1
            else:
                merged += 1

        if added or merged:
            with self.objects_lock:
                confirmed = sum(1 for o in self.objects if o.count >= MIN_OBSERVATIONS)
                total = len(self.objects)
            self.get_logger().info(
                f"[ObjMap] det={len(ctx.detections)} new={added} merged={merged} "
                f"skip={skipped} | 물체맵: 확정 {confirmed} / 전체 {total}"
            )

    def merge_or_add(self, label, xyz, score, size, weight):
        """같은 라벨이면서 MERGE_DIST_M 안의 기존 물체가 있으면 병합, 없으면 새로 등록.

        반환: True면 새로 추가, False면 기존 물체에 병합.
        """
        with self.objects_lock:
            best = None
            best_dist = MERGE_DIST_M
            for obj in self.objects:
                if obj.label != label:
                    continue
                d = float(np.linalg.norm(obj.xyz - xyz))
                if d < best_dist:
                    best_dist = d
                    best = obj

            if best is not None:
                best.add(xyz, score, size, weight)
                return False

            self.objects.append(MappedObject(label, xyz, score, size, weight))
            return True

    # ==================== 물체 맵 정비 ====================
    def maintenance_tick(self):
        """주기적으로 물체 맵을 정비한다: 갈라진 물체 재병합 + 오래된 미확정 정리.

        merge_or_add는 "새 검출 vs 기존"만 비교하기 때문에, 위치가 한 번 크게 튀어
        같은 물체가 2개로 등록되면 그 뒤로는 영영 안 합쳐진다(각자 따로 갱신될 뿐).
        그래서 여기서 "기존 물체끼리" 다시 비교해 붙여준다 — split 복구 장치.
        """
        remerged = self.remerge_objects()
        removed = self.drop_stale_unconfirmed()
        if remerged or removed:
            with self.objects_lock:
                confirmed = sum(1 for o in self.objects if o.count >= MIN_OBSERVATIONS)
                total = len(self.objects)
            self.get_logger().info(
                f"[ObjMap] 정비: 재병합 {remerged}건, 미확정 정리 {removed}건 "
                f"| 물체맵: 확정 {confirmed} / 전체 {total}"
            )

    def remerge_objects(self):
        """같은 라벨 + MERGE_DIST_M 안의 기존 물체끼리 합친다. 합친 횟수 반환."""
        merged_count = 0
        with self.objects_lock:
            changed = True
            while changed:
                changed = False
                for i in range(len(self.objects)):
                    for j in range(i + 1, len(self.objects)):
                        a, b = self.objects[i], self.objects[j]
                        if a.label != b.label:
                            continue
                        if float(np.linalg.norm(a.xyz - b.xyz)) < MERGE_DIST_M:
                            a.absorb(b)          # 관측을 합쳐서 위치를 다시 계산
                            self.objects.pop(j)
                            merged_count += 1
                            changed = True
                            break
                    if changed:
                        break   # 리스트가 바뀌었으니 처음부터 다시 훑는다
        return merged_count

    def drop_stale_unconfirmed(self):
        """한 번만 보이고 오래 재관측이 없는 물체를 버린다(스쳐 지나간 오검출).

        확정된(MIN_OBSERVATIONS 이상) 물체는 지우지 않는다 — 지금 안 보이는 건
        그냥 시야 밖일 뿐이지 없어진 게 아니다.
        """
        now = time.time()
        with self.objects_lock:
            before = len(self.objects)
            self.objects = [
                o for o in self.objects
                if o.count >= MIN_OBSERVATIONS or (now - o.last_seen) < STALE_UNCONFIRMED_SEC
            ]
            return before - len(self.objects)

    # ==================== 로봇 위치 헬퍼 ====================
    def robot_x(self):
        return self.latest_pose.position.x if self.latest_pose is not None else 0.0

    def robot_y(self):
        return self.latest_pose.position.y if self.latest_pose is not None else 0.0

    # ==================== 발행 / 시각화 ====================
    def publish_object_markers(self):
        """확정된 물체를 CUBE + 라벨 텍스트로 발행 (메인 스레드 타이머)."""
        with self.objects_lock:
            objs = [o for o in self.objects if o.count >= MIN_OBSERVATIONS]
            snapshot = [(o.label, o.xyz.copy(), o.size, o.count, o.best_score) for o in objs]

        array = MarkerArray()

        # 병합으로 물체가 없어질 수 있어서, 매번 전부 지우고 다시 그린다(디버그 시각화라 OK).
        clear = Marker()
        clear.header.frame_id = config.FRAME_MAP
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)

        for i, (label, xyz, size, count, score) in enumerate(snapshot):
            color = class_color_bgr(label)
            r, g, b = color[2] / 255.0, color[1] / 255.0, color[0] / 255.0

            cube = Marker()
            cube.header.frame_id = config.FRAME_MAP
            cube.header.stamp = self.get_clock().now().to_msg()
            cube.ns = "object"
            cube.id = i
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose.position.x = float(xyz[0])
            cube.pose.position.y = float(xyz[1])
            cube.pose.position.z = float(xyz[2])
            cube.pose.orientation.w = 1.0
            # 크기 추정이 실패했으면 고정 크기로 표시 (본 파이프라인과 같은 규칙).
            if size is not None:
                cube.scale.x, cube.scale.y, cube.scale.z = (float(s) for s in size)
            else:
                d = config.BBOX3D_DEFAULT_SIZE_M
                cube.scale.x = cube.scale.y = cube.scale.z = float(d)
            cube.color.a = 0.55
            cube.color.r, cube.color.g, cube.color.b = r, g, b
            array.markers.append(cube)

            text = Marker()
            text.header.frame_id = config.FRAME_MAP
            text.header.stamp = self.get_clock().now().to_msg()
            text.ns = "object_label"
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(xyz[0])
            text.pose.position.y = float(xyz[1])
            text.pose.position.z = float(xyz[2]) + 0.4
            text.pose.orientation.w = 1.0
            text.scale.z = 0.25
            text.color.a = 1.0
            text.color.r, text.color.g, text.color.b = r, g, b
            text.text = f"{label} (x{count})"
            array.markers.append(text)

        self.pub_object_markers.publish(array)

    def render_explore(self):
        """부모의 2D 격자 그림 위에 물체 위치를 클래스별 색 점으로 얹는다."""
        img = super().render_explore()

        with self.objects_lock:
            snapshot = [(o.label, o.xyz.copy(), o.count) for o in self.objects]

        for label, xyz, count in snapshot:
            if count < MIN_OBSERVATIONS:
                continue
            c = self.world_to_cell(xyz[:2])
            if not (0 <= c[0] < self.grid_n and 0 <= c[1] < self.grid_n):
                continue
            row = self.grid_n - 1 - c[1]
            cv2.circle(img, (int(c[0]), int(row)), 3, class_color_bgr(label), -1)
        return img


def main():
    rclpy.init()
    node = ObjectMappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
