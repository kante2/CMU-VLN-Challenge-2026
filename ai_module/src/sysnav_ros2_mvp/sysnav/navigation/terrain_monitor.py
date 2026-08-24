"""base autonomy가 waypoint를 받아들일지 우리가 미리 판정한다 (/terrain_map 기반).

왜 필요한가: 우리가 /way_point_with_heading에 발행한 좌표는 로봇에게 그대로 가지
않는다. base autonomy의 waypointConverter가 중간에서 바꾼다 (소스와 라이브 파라미터로
확인, 2026-08-12):

  - 목표가 로봇에서 adjDisThre=5.0m 안이면 우리 좌표를 버리고 자기 travArea에서
    후보를 고른다. 우리 목표는 접근 주행이라 거의 항상 5m 안이다.
  - 후보 비용 = dist(후보, 우리목표) + 0.5 * dist(후보, 로봇)
  - 하드 필터: 후보가 obstacleArea 점에서 obstacleDisThre=0.75m 안이면 무조건 탈락
  - travArea/obstacleArea는 /terrain_map을 intensity(=지면 대비 높이) 0.05 기준으로
    나눈 것이다.

통과 후보가 하나도 없으면 우리 좌표가 그대로 쓰이지만(무해), 통과 후보가 "로봇 뒤쪽"
에만 있으면 그쪽이 argmin으로 뽑혀서 로봇이 목표 반대 방향으로 간다. 목표 근처에
통과 후보가 하나라도 있으면 dist(후보, 우리목표)가 0에 가까워 그쪽이 이긴다.

그래서 이 모듈은 "우리 목표 근처에 통과 후보가 존재하는가"를 판정하고, 없으면 존재
하는 지점으로 접근 지점을 옮겨준다. 즉 waypointConverter와 같은 데이터(/terrain_map)를
같은 기준으로 보고, 그쪽이 받아들일 좌표를 처음부터 찍는 것이다.

/terrain_map은 README의 System Outputs 표에 있는 테스트 때도 사용 허용된 토픽이고,
frame_id가 map이라 우리 좌표계와 그대로 맞는다(변환 불필요).
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

from sysnav import config


class TerrainMonitor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._trav = np.empty((0, 2), dtype=np.float64)
        self._obstacle = np.empty((0, 2), dtype=np.float64)
        self._updated_time: float | None = None
        # 마지막 판정 결과 요약(진단 로그용).
        self.last_selection: str = "-"

    # ------------------------------------------------------------------
    # 수신
    # ------------------------------------------------------------------

    def update(self, cloud_msg) -> None:
        """/terrain_map 콜백. waypointConverter의 terrainMapHandler와 같은 기준으로
        travArea/obstacleArea를 나눈다."""
        points = np.array(
            [
                (float(x), float(y), float(intensity))
                for x, y, intensity in pc2.read_points(
                    cloud_msg, ("x", "y", "intensity"), skip_nans=True
                )
            ],
            dtype=np.float64,
        )
        if points.size == 0:
            return
        obstacle_mask = points[:, 2] >= config.TERRAIN_OBSTACLE_INTENSITY
        with self._lock:
            self._trav = points[~obstacle_mask][:, :2]
            self._obstacle = points[obstacle_mask][:, :2]
            self._updated_time = time.monotonic()

    def ready(self) -> bool:
        """terrain 데이터가 있고 충분히 최신인지. 아니면 판정을 아예 하지 않는다 -
        오래된 지형으로 목표를 옮기면 없느니만 못하다."""
        with self._lock:
            if self._updated_time is None or len(self._trav) == 0:
                return False
            return time.monotonic() - self._updated_time <= config.TERRAIN_STALE_SEC

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._trav.copy(), self._obstacle.copy()

    # ------------------------------------------------------------------
    # 판정
    # ------------------------------------------------------------------

    @staticmethod
    def _support_detail(
        trav: np.ndarray, obstacle: np.ndarray, point: np.ndarray
    ) -> tuple[bool, float | None]:
        """(통과 여부, 이 point 근처에서 얻을 수 있는 최선의 클리어런스).

        두 조건을 모두 봐야 한다:
          1. 커버리지 - point 근처에 travArea 점이 있어야 한다. 로봇이 아직 그 근처에
             가본 적 없으면 우리 occupancy grid가 멀리서 free로 "보고 있어도" 이쪽은
             비어 있다.
          2. 클리어런스 - 그 점이 obstacleArea에서 TERRAIN_CLEARANCE_M 이상 떨어져야
             한다. waypointConverter의 하드 필터와 같은 조건.

        두 번째 값을 같이 돌려주는 이유: 둘 중 **어디서** 떨어졌는지 로그가 구분하지
        못해 접근 지점 로직을 고칠 근거가 없었다(2026-08-23, RETARGET_FAIL 1039회가
        전부 "no supported point" 한 문장이었다). None이면 커버리지에서 탈락(travArea
        점 자체가 없음), 숫자면 커버리지는 통과했고 그게 달성 가능한 최대 클리어런스다.
        """
        if len(trav) == 0:
            return False, None
        near = trav[
            np.linalg.norm(trav - point, axis=1) <= config.TERRAIN_SUPPORT_RADIUS_M
        ]
        if len(near) == 0:
            return False, None
        if len(obstacle) == 0:
            return True, float("inf")
        # near의 각 점에 대해 가장 가까운 obstacle까지 거리가 클리어런스 이상이면 통과.
        distances = np.linalg.norm(
            near[:, None, :] - obstacle[None, :, :], axis=2
        ).min(axis=1)
        best = float(distances.max())
        return best >= config.TERRAIN_CLEARANCE_M, best

    @classmethod
    def _supported(cls, trav: np.ndarray, obstacle: np.ndarray, point: np.ndarray) -> bool:
        return cls._support_detail(trav, obstacle, point)[0]

    @staticmethod
    def _min_distance_to_obstacles(points: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
        """points 각각에서 가장 가까운 obstacle까지의 거리. 전체 행렬을 한 번에 만들면
        (후보 수 x 장애물 수) 크기가 되어 커지므로 청크로 나눠 계산한다."""
        result = np.empty(len(points), dtype=np.float64)
        for start in range(0, len(points), 1024):
            chunk = points[start:start + 1024]
            distances = np.linalg.norm(chunk[:, None, :] - obstacle[None, :, :], axis=2)
            result[start:start + 1024] = distances.min(axis=1)
        return result

    def has_commandable_points(self, robot_xy) -> bool:
        """차량 주변(searchDisThre 안)에 waypointConverter가 고를 수 있는 후보가
        하나라도 있는가.

        이걸 따로 봐야 하는 이유: waypointConverter는 후보가 하나도 없으면
        (`if (minInd >= 0)`가 거짓) 목표를 갈아끼우지 않고 **우리 좌표를 그대로**
        내보낸다. 즉 "아무것도 통과 못 하는 좁은 방"에서는 우리가 뭘 찍든 그대로
        전달된다 - 이때는 발행을 막으면 안 되고 원본을 보내야 한다.

        반대로 후보가 하나라도 있으면 그중 argmin이 뽑히므로, 우리 목표 근처에
        후보가 없으면 엉뚱한 곳(대개 로봇 발밑)으로 끌려간다.
        """
        if not self.ready() or robot_xy is None:
            return False
        trav, obstacle = self.snapshot()
        if len(trav) == 0:
            return False
        robot = np.asarray(robot_xy, dtype=np.float64)[:2]
        near = trav[np.linalg.norm(trav - robot, axis=1) <= config.TERRAIN_SEARCH_DIS_M]
        if len(near) == 0:
            return False
        if len(obstacle) == 0:
            return True
        return bool(
            np.any(self._min_distance_to_obstacles(near, obstacle) >= config.TERRAIN_CLEARANCE_M)
        )

    def nearest_commandable(
        self, x: float, y: float, robot_xy=None
    ) -> tuple[float, float] | None:
        """(x, y)에 가장 가까운 "base autonomy가 그대로 받아주는" 지점을 돌려준다.

        commandable = 관측된 travArea 점이면서 obstacleArea에서 TERRAIN_CLEARANCE_M 이상
        떨어진 점. waypointConverter가 후보로 삼는 조건 그대로다. 여기에 맞춰 찍으면
        스냅이 일어나지 않는다(근거는 config.TERRAIN_SNAP_MAX_M 주석의 비용식 증명).

        반환 None인 경우:
          - terrain 데이터가 없거나 오래됨 (판정 불가 -> 호출 측은 원본을 그대로 쓴다)
          - TERRAIN_SNAP_MAX_M 안에 commandable 지점이 없음 (이 목표는 지금 실행 불가)
        """
        if not self.ready():
            self.last_selection = "terrain not ready"
            return None

        trav, obstacle = self.snapshot()
        if len(trav) == 0:
            self.last_selection = "no travArea points"
            return None

        goal = np.array([float(x), float(y)], dtype=np.float64)
        to_goal = np.linalg.norm(trav - goal, axis=1)
        within = to_goal <= config.TERRAIN_SNAP_MAX_M
        candidates, distances = trav[within], to_goal[within]
        if len(candidates) == 0:
            self.last_selection = (
                f"no travArea point within {config.TERRAIN_SNAP_MAX_M:.2f}m of goal"
            )
            return None

        # waypointConverter는 차량에서 searchDisThre 안의 travArea 점만 후보로 본다.
        # 그 밖의 점을 찍으면 우리 좌표가 아니라 엉뚱한 점이 뽑히므로 여기서도 제외한다.
        if robot_xy is not None:
            robot = np.asarray(robot_xy, dtype=np.float64)[:2]
            reachable = np.linalg.norm(candidates - robot, axis=1) <= config.TERRAIN_SEARCH_DIS_M
            candidates, distances = candidates[reachable], distances[reachable]
            if len(candidates) == 0:
                self.last_selection = "commandable points are outside searchDisThre"
                return None

        if len(obstacle):
            margins = self._min_distance_to_obstacles(candidates, obstacle)
            clear = margins >= config.TERRAIN_CLEARANCE_M
            if not clear.any():
                # 최선값을 같이 남긴다 - 0.70m면 아깝게 떨어진 것이고 0.20m면 애초에
                # 접근 불가한 자리다. 이 둘은 대응이 다르다.
                self.last_selection = (
                    f"no point with {config.TERRAIN_CLEARANCE_M:.2f}m clearance near goal "
                    f"(best {float(margins.max()):.2f}m of {len(candidates)} candidate(s))"
                )
                return None
            candidates, distances = candidates[clear], distances[clear]

        best = int(np.argmin(distances))
        self.last_selection = f"snap={distances[best]:.2f}m from {len(candidates)} candidate(s)"
        return float(candidates[best][0]), float(candidates[best][1])

    def is_waypoint_supported(self, x: float, y: float) -> bool:
        if not self.ready():
            return True  # 판정 불가 = 보류. 데이터 없다고 목표를 막으면 안 된다.
        trav, obstacle = self.snapshot()
        return self._supported(trav, obstacle, np.array([x, y], dtype=np.float64))

    def choose_approach_point(
        self,
        object_xy,
        robot_xy,
        max_distance_m: float | None = None,
        allow_relaxed: bool = False,
    ) -> tuple[float, float] | None:
        """물체로 접근할 지점을 고른다. 물체에서 가까운 순서로 후보를 훑어 첫 통과점을
        반환한다 - "go near X"이므로 통과하는 것 중 가장 가까운 지점이 좋다.

        로봇->물체 방향을 기준으로 각도를 벌려가며 찾는 이유: 정면 접근이 막혀도
        (벽에 붙은 물체, 앞을 막은 가구) 옆에서 접근하면 통과하는 경우가 많다.

        allow_relaxed=True는 **최후의 수단**이다: 링 샘플링이 전부 실패했을 때
        max_distance_m 상한을 풀고 commandable set을 직접 훑어 물체에 가장 가까운
        통과 지점을 고른다(TERRAIN_APPROACH_FALLBACK_MAX_M까지).

        기본값이 False인 이유: Mission 3의 MISSION3_OBJECT_APPROACH_MAX_M(0.9m)은 탐색
        범위가 아니라 **의미 규칙**이다 - 물체에서 2m 떨어져 서 놓고 "go to X를 했다"고
        할 수 없다. 그래서 평상시에는 그 상한을 반드시 지키고, "여기서 더 버티면 그
        step을 통째로 잃는다"가 확정된 순간에만 호출 측이 명시적으로 풀어준다.

        반환 None = 통과 지점을 못 찾음. 호출 측은 기존 방식(고정 standoff)으로
        폴백해야 한다. terrain 판정 실패가 주행 자체를 막으면 안 된다.
        """
        if not self.ready():
            self.last_selection = "terrain not ready"
            return None

        object_xy = np.asarray(object_xy, dtype=np.float64)[:2]
        robot_xy = np.asarray(robot_xy, dtype=np.float64)[:2]
        direction = object_xy - robot_xy
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            self.last_selection = "robot is on the object"
            return None
        direction = direction / norm

        trav, obstacle = self.snapshot()
        tried = 0
        # 실패 사유 집계 - 커버리지(travArea 점 없음)와 클리어런스(있는데 장애물에
        # 너무 붙음)는 고쳐야 할 곳이 완전히 다르다. 전자는 "더 가서 봐야 한다",
        # 후자는 "이 물체는 접근 자체가 불가하다".
        no_coverage = 0
        best_clearance: float | None = None
        max_distance = (
            config.TERRAIN_APPROACH_MAX_M
            if max_distance_m is None
            else min(float(max_distance_m), config.TERRAIN_APPROACH_MAX_M)
        )
        # mission별 상한이 공용 최소값보다 작을 수 있다(Mission 3: 0.9m). 이 경우
        # 루프를 건너뛰지 말고 그 상한 거리의 ring을 한 번 검사한다.
        distance = min(config.TERRAIN_APPROACH_MIN_M, max_distance)
        while distance <= max_distance + 1e-9:
            for angle_deg in config.TERRAIN_APPROACH_ANGLES_DEG:
                angle = math.radians(angle_deg)
                rotated = np.array([
                    direction[0] * math.cos(angle) - direction[1] * math.sin(angle),
                    direction[0] * math.sin(angle) + direction[1] * math.cos(angle),
                ])
                candidate = object_xy - distance * rotated
                tried += 1
                ok, clearance = self._support_detail(trav, obstacle, candidate)
                if ok:
                    self.last_selection = (
                        f"d={distance:.2f}m angle={angle_deg:+.0f}deg tried={tried}"
                    )
                    return float(candidate[0]), float(candidate[1])
                if clearance is None:
                    no_coverage += 1
                elif best_clearance is None or clearance > best_clearance:
                    best_clearance = clearance
            distance += config.TERRAIN_APPROACH_STEP_M

        # 링 샘플링이 전부 실패했다 - 호출 측이 허락했을 때만, 반경/각도 운에 맡기는
        # 대신 통과 지점 집합을 직접 훑는다(TERRAIN_APPROACH_FALLBACK_MAX_M 주석 참고).
        direct = (
            self._nearest_commandable_to_object(trav, obstacle, object_xy, robot_xy)
            if allow_relaxed else None
        )
        if direct is not None:
            point, gap = direct
            self.last_selection = (
                f"relaxed commandable-set fallback: {gap:.2f}m from object "
                f"(ring x{tried} all failed, mission limit {max_distance:.2f}m waived)"
            )
            return float(point[0]), float(point[1])

        if best_clearance is None:
            verdict = f"all {tried} unobserved (no travArea point within " \
                      f"{config.TERRAIN_SUPPORT_RADIUS_M:.2f}m)"
        else:
            verdict = (
                f"{no_coverage}/{tried} unobserved, best clearance "
                f"{best_clearance:.2f}m < {config.TERRAIN_CLEARANCE_M:.2f}m"
            )
        self.last_selection = (
            f"no supported point ({verdict}, trav={len(trav)}, obstacle={len(obstacle)})"
        )
        return None

    @classmethod
    def _nearest_commandable_to_object(
        cls, trav: np.ndarray, obstacle: np.ndarray, object_xy, robot_xy
    ) -> tuple[np.ndarray, float] | None:
        """waypointConverter가 후보로 인정하는 점들 중 물체에 가장 가까운 것.

        후보 조건은 waypointConverter와 동일하게 맞춘다:
          - 관측된 travArea 점
          - 차량에서 searchDisThre 안 (그 밖은 저쪽이 후보로 안 봄)
          - 모든 obstacleArea 점에서 TERRAIN_CLEARANCE_M 이상
        여기에 두 가지를 더 건다:
          - 물체에서 TERRAIN_APPROACH_FALLBACK_MAX_M 안 (벽 너머 오검출 방지)
          - 지금 로봇보다 물체에 더 가까울 것 (전진이 없는 "접근점"은 의미가 없다)

        반환: (좌표, 물체까지 거리) 또는 None.
        """
        if len(trav) == 0:
            return None
        object_xy = np.asarray(object_xy, dtype=np.float64)[:2]
        robot_xy = np.asarray(robot_xy, dtype=np.float64)[:2]

        candidates = trav[
            np.linalg.norm(trav - robot_xy, axis=1) <= config.TERRAIN_SEARCH_DIS_M
        ]
        if len(candidates) == 0:
            return None
        to_object = np.linalg.norm(candidates - object_xy, axis=1)
        robot_gap = float(np.linalg.norm(robot_xy - object_xy))
        usable = (to_object <= config.TERRAIN_APPROACH_FALLBACK_MAX_M) & (to_object < robot_gap)
        candidates, to_object = candidates[usable], to_object[usable]
        if len(candidates) == 0:
            return None
        if len(obstacle):
            clear = cls._min_distance_to_obstacles(candidates, obstacle) >= config.TERRAIN_CLEARANCE_M
            candidates, to_object = candidates[clear], to_object[clear]
            if len(candidates) == 0:
                return None
        best = int(np.argmin(to_object))
        return candidates[best], float(to_object[best])

    def commandable_ratio(self) -> tuple[int, int]:
        """(commandable 점 수, travArea 점 수). "이 씬에서 애초에 목적지로 찍을 수 있는
        곳이 얼마나 되나"를 대시보드/로그로 보기 위한 진단용 - 7%면 링 샘플링이 거의
        항상 실패한다는 뜻이다(2026-08-24 실측)."""
        if not self.ready():
            return 0, 0
        trav, obstacle = self.snapshot()
        if len(trav) == 0:
            return 0, 0
        if len(obstacle) == 0:
            return len(trav), len(trav)
        clear = self._min_distance_to_obstacles(trav, obstacle) >= config.TERRAIN_CLEARANCE_M
        return int(clear.sum()), len(trav)

    def describe(self) -> str:
        with self._lock:
            age = (
                "-" if self._updated_time is None
                else f"{time.monotonic() - self._updated_time:.1f}s"
            )
            return f"trav={len(self._trav)} obstacle={len(self._obstacle)} age={age}"
