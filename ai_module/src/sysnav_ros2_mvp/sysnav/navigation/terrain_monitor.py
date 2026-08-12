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
    def _supported(trav: np.ndarray, obstacle: np.ndarray, point: np.ndarray) -> bool:
        """point 근처에 "waypointConverter가 고를 수 있는 후보"가 존재하는가.

        두 조건을 모두 봐야 한다:
          1. 커버리지 - point 근처에 travArea 점이 있어야 한다. 로봇이 아직 그 근처에
             가본 적 없으면 우리 occupancy grid가 멀리서 free로 "보고 있어도" 이쪽은
             비어 있다.
          2. 클리어런스 - 그 점이 obstacleArea에서 TERRAIN_CLEARANCE_M 이상 떨어져야
             한다. waypointConverter의 하드 필터와 같은 조건.
        """
        if len(trav) == 0:
            return False
        near = trav[
            np.linalg.norm(trav - point, axis=1) <= config.TERRAIN_SUPPORT_RADIUS_M
        ]
        if len(near) == 0:
            return False
        if len(obstacle) == 0:
            return True
        # near의 각 점에 대해 가장 가까운 obstacle까지 거리가 클리어런스 이상이면 통과.
        distances = np.linalg.norm(
            near[:, None, :] - obstacle[None, :, :], axis=2
        ).min(axis=1)
        return bool(np.any(distances >= config.TERRAIN_CLEARANCE_M))

    def is_waypoint_supported(self, x: float, y: float) -> bool:
        if not self.ready():
            return True  # 판정 불가 = 보류. 데이터 없다고 목표를 막으면 안 된다.
        trav, obstacle = self.snapshot()
        return self._supported(trav, obstacle, np.array([x, y], dtype=np.float64))

    def choose_approach_point(
        self, object_xy, robot_xy
    ) -> tuple[float, float] | None:
        """물체로 접근할 지점을 고른다. 물체에서 가까운 순서로 후보를 훑어 첫 통과점을
        반환한다 - "go near X"이므로 통과하는 것 중 가장 가까운 지점이 좋다.

        로봇->물체 방향을 기준으로 각도를 벌려가며 찾는 이유: 정면 접근이 막혀도
        (벽에 붙은 물체, 앞을 막은 가구) 옆에서 접근하면 통과하는 경우가 많다.

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
        distance = config.TERRAIN_APPROACH_MIN_M
        while distance <= config.TERRAIN_APPROACH_MAX_M + 1e-9:
            for angle_deg in config.TERRAIN_APPROACH_ANGLES_DEG:
                angle = math.radians(angle_deg)
                rotated = np.array([
                    direction[0] * math.cos(angle) - direction[1] * math.sin(angle),
                    direction[0] * math.sin(angle) + direction[1] * math.cos(angle),
                ])
                candidate = object_xy - distance * rotated
                tried += 1
                if self._supported(trav, obstacle, candidate):
                    self.last_selection = (
                        f"d={distance:.2f}m angle={angle_deg:+.0f}deg tried={tried}"
                    )
                    return float(candidate[0]), float(candidate[1])
            distance += config.TERRAIN_APPROACH_STEP_M

        self.last_selection = (
            f"no supported point (tried={tried}, trav={len(trav)}, obstacle={len(obstacle)})"
        )
        return None

    def describe(self) -> str:
        with self._lock:
            age = (
                "-" if self._updated_time is None
                else f"{time.monotonic() - self._updated_time:.1f}s"
            )
            return f"trav={len(self._trav)} obstacle={len(self._obstacle)} age={age}"
