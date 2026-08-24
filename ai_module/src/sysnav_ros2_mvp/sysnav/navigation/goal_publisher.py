"""Publish target or exploration goals as geometry_msgs/Pose2D."""

from __future__ import annotations

import math
import time

from geometry_msgs.msg import Point, Pose2D
from visualization_msgs.msg import Marker, MarkerArray

from sysnav import config

_MISSION3_GOAL_NAMESPACE = "mission3_goals"
_MISSION3_GOAL_COLOR = (1.0, 0.55, 0.0, 0.9)  # 주황 - object/scene graph marker와 구분됨
# "take the path between A and B"의 게이트 선분. goal 구(주황)와 헷갈리지 않게 노랑을
# 쓰고, 실제로 통과한 순간 초록으로 바뀐다.
_MISSION3_GATE_COLOR = (1.0, 0.95, 0.2, 0.9)
_MISSION3_GATE_CROSSED_COLOR = (0.1, 0.9, 0.2, 0.9)
# step 하나가 쓰는 마커 id 칸 수: goal sphere / goal text / gate line / gate endpoints.
_MARKERS_PER_STEP = 4

_REQUESTED_NAMESPACE = "requested_waypoint"
_REQUESTED_COLOR = (0.0, 0.9, 1.0, 0.9)  # 청록 - base autonomy의 /way_point와 구분됨
_DISPLACEMENT_COLOR = (1.0, 0.2, 0.2, 0.9)  # 빨강 - 밀려난 거리를 잇는 선


class GoalPublisher:
    def __init__(self, node) -> None:
        self._node = node
        self.publisher = node.create_publisher(Pose2D, config.TOPIC_WAYPOINT, 10)
        self.goal_marker_pub = node.create_publisher(
            MarkerArray, config.TOPIC_MISSION3_GOAL_MARKERS, 10
        )
        # 우리가 "원한" 좌표 시각화 (config.TOPIC_REQUESTED_WAYPOINT 주석 참고).
        self.requested_marker_pub = node.create_publisher(
            MarkerArray, config.TOPIC_REQUESTED_WAYPOINT, 10
        )
        self._goal_markers: list[Marker] = []
        # 마지막으로 요청한 좌표와 라벨. sysnav_node의 /way_point 콜백이 이걸 기준으로
        # base autonomy가 얼마나 밀어냈는지 계산한다.
        self.last_requested_xy: tuple[float, float] | None = None
        self.last_requested_label: str = "goal"
        # 발행 직전 스냅이 좌표를 옮긴 거리(None = 스냅 안 함/못 함). 대시보드 표시용.
        self.last_snap_distance_m: float | None = None

    def _robot_xy(self) -> tuple[float, float] | None:
        with self._node.sensor_lock:
            pose = self._node.latest_pose
            if pose is None:
                return None
            return float(pose["x"]), float(pose["y"])

    def resolve(
        self, x: float, y: float, label: str = "goal", trace_failure: bool = False
    ) -> tuple[tuple[float, float], tuple[float, float] | None] | None:
        """publish()가 **실제로 내보낼 좌표**를 계산만 한다(발행 없음).

        경로 위 여러 hop 중 "지금 발행 가능한 가장 먼 hop"을 고르려면, 실제로 쏘지 않고
        판정만 해봐야 한다. publish()와 판정이 갈리면 조용히 어긋나므로 로직을 여기
        한 곳에 두고 publish()도 이걸 쓴다.

        반환: (실제 발행할 좌표, 스냅된 좌표 또는 None) / 발행 불가면 None.
        """
        requested = (float(x), float(y))
        robot_xy = self._robot_xy()
        trace = getattr(self._node, "_trace_navigation", None) if trace_failure else None
        monitor = self._node.terrain_monitor

        # 목표가 adjDisThre 밖이면 waypointConverter가 손대지 않는다 - 스냅 불필요.
        needs_snap = (
            robot_xy is None
            or math.dist(robot_xy, requested) < config.TERRAIN_ADJ_DIS_M
        )
        # terrain이 없거나 오래됐으면 판정 자체를 못 한다 - 원본을 그대로 보낸다
        # (데이터가 없다고 주행을 막으면 안 된다).
        if not needs_snap or not monitor.ready():
            return requested, None

        snapped = monitor.nearest_commandable(x, y, robot_xy)
        if snapped is not None:
            # 스냅이 전진을 통째로 삼켰는지 본다.
            #
            # TERRAIN_SNAP_MAX_M(1.5m)이 GOAL_REACHED_DISTANCE_M(0.5m)보다 훨씬 커서,
            # 1.5m 앞 hop이 로봇 발밑으로 옮겨질 수 있다. 그대로 발행하면 즉시 "도착"
            # 처리되고 다시 계획 -> 또 같은 방향 -> 또 되끌림으로 제자리에서 돈다.
            # 실측(2026-08-22): 요청 (4.50,1.50)이 (3.22,1.58)로 1.28m 되끌렸는데
            # 로봇(3.27,1.84)에서 0.26m라 도착 반경 안이었다. 탐색 goal 329건 중 40%가
            # 1m 넘게 되끌리고 있었다.
            #
            # "원래는 갈 거리가 있었는데 스냅 후 없어진" 경우만 거른다 - 원래부터
            # 가까운 목표(마지막 접근 등)는 그대로 통과시켜야 한다.
            if robot_xy is not None:
                before = math.dist(robot_xy, requested)
                after = math.dist(robot_xy, snapped)
                if before > config.GOAL_REACHED_DISTANCE_M >= after:
                    if trace is not None:
                        trace("SNAP_NO_PROGRESS",
                              f"({requested[0]:.2f},{requested[1]:.2f}) -> "
                              f"({snapped[0]:.2f},{snapped[1]:.2f}) "
                              f"robot_dist {before:.2f}m -> {after:.2f}m label={label} "
                              f"- not published")
                    return None
            return snapped, snapped

        # waypointConverter는 통과 후보가 하나도 없으면 목표를 갈아끼우지 않고 우리
        # 좌표를 그대로 쓴다(`if (minInd >= 0)`). 좁아서 아무것도 통과 못 하는 방이
        # 정확히 그 경우라 발행을 막으면 안 된다. 후보가 "있는데 목표 근처에만 없는"
        # 경우에만 막아야 한다 - 그때만 엉뚱한 데로 끌려가기 때문이다.
        if not monitor.has_commandable_points(robot_xy):
            if trace is not None:
                trace("PASSTHRU",
                      f"goal=({requested[0]:.2f},{requested[1]:.2f}) label={label} "
                      f"no candidate anywhere - waypointConverter will use it as-is")
            return requested, None
        # 여기까지 왔다 = "통과 후보는 어딘가 있는데 목표 근처(1.5m)엔 없다".
        # 예전엔 무조건 거부했지만, 그러면 로봇에게 아무 명령도 안 가서 한 발짝도 못
        # 움직인다(실측 2026-08-24: mission3 step 1/2가 이 상태로 4초 만에 포기).
        # 저쪽은 우리 1.5m 규칙을 모르고 후보가 있으면 argmin을 고르므로, 그 점을 우리가
        # 예측해서 **전진이 되는 경우에만** 그 점으로 보낸다. 물체 앞까지는 못 가도
        # 그쪽으로 전진하면 terrain이 자라 다음 tick엔 더 가까운 점이 생긴다.
        predicted = (
            monitor.predict_converter_choice(x, y, robot_xy)
            if config.PREDICT_CONVERTER_FALLBACK_ENABLED else None
        )
        if predicted is not None and robot_xy is not None:
            point, gap_to_goal, candidate_count = predicted
            robot_gap = math.dist(robot_xy, requested)
            travel = math.dist(robot_xy, point)
            gain = robot_gap - gap_to_goal
            if (
                gain >= config.PREDICT_CONVERTER_MIN_GAIN_M
                and travel > config.GOAL_REACHED_DISTANCE_M
            ):
                if trace is not None:
                    trace("PREDICT_FALLBACK",
                          f"goal=({requested[0]:.2f},{requested[1]:.2f}) -> "
                          f"({point[0]:.2f},{point[1]:.2f}) gain={gain:.2f}m "
                          f"travel={travel:.2f}m gap={gap_to_goal:.2f}m "
                          f"of {candidate_count} commandable label={label}")
                return point, point
            if trace is not None:
                trace("PREDICT_REJECT",
                      f"goal=({requested[0]:.2f},{requested[1]:.2f}) -> "
                      f"({point[0]:.2f},{point[1]:.2f}) gain={gain:.2f}m "
                      f"travel={travel:.2f}m - no forward progress label={label}")
        if trace is not None:
            trace("SNAP_FAIL",
                  f"goal=({requested[0]:.2f},{requested[1]:.2f}) label={label} "
                  f"{monitor.last_selection} - not published")
        return None

    def can_publish(self, x: float, y: float) -> bool:
        """부작용 없이 "이 좌표를 지금 발행할 수 있는가"만 본다."""
        return self.resolve(x, y) is not None

    def publish(self, x: float, y: float, theta: float, label: str = "goal") -> Pose2D | None:
        """목표를 발행한다. 발행 **직전에** base autonomy가 그대로 받아줄 지점으로
        옮긴다(Layer 1).

        왜 필요한가: waypointConverter는 우리 좌표를 검사만 하는 게 아니라 자기
        travArea 점으로 갈아끼운다. 실측(2026-08-21, probe 60지점 x 2위치)에서 요청의
        93~97%가 0.3m 넘게, 80%가 1m 넘게 밀렸고 중앙값이 2.1~2.5m였다 - 요청 (-3.00,
        0.00)이 반대 방향 (1.18, -0.32)으로 가는 식이라 사실상 우리 좌표가 무시됐다.
        우리가 먼저 commandable 지점으로 맞춰 보내면 스냅이 일어나지 않는다.

        반환값:
          Pose2D - **실제로 발행된 좌표**. 호출 측은 이것을 current_goal로 저장해야 한다.
                   원본을 저장하면 로봇이 갈 수 없는 좌표를 기준으로 도착 판정을 하게
                   되어 영원히 도착하지 않는다.
          None   - 발행하지 않았다. 이 목표 근처에 base autonomy가 받아줄 지점이 없다.
                   호출 측은 **다른 후보로 넘어가야 한다**.

        None일 때 원본을 대신 발행하면 안 된다(예전엔 그렇게 fail-open 했다). 스냅이
        실패했다는 건 그 좌표 근처에 받아줄 지점이 없다는 뜻이고, 그러면 waypointConverter는
        반드시 로봇 발밑으로 목표를 떨어뜨린다 - 실측(2026-08-21): 요청 (-3.90,-0.90)이
        (-0.34,0.12)로, 로봇(-0.64,0.18)에서 0.31m 지점에 찍혔다. waypointXYRadius(0.3)
        바로 바깥이라 "도착"도 아니고 갈 거리도 없어서 로봇이 그대로 서 있었고, 우리는
        stuck timeout(8~20초)을 통째로 낭비한 뒤에야 다음 후보로 넘어갔다.
        """
        resolved = self.resolve(x, y, label=label, trace_failure=True)
        if resolved is None:
            self.last_snap_distance_m = None
            far = self._far_throw(x, y, label)
            if far is None:
                return None
            effective, snapped = far, None
        else:
            effective, snapped = resolved
        requested = (float(x), float(y))
        message = Pose2D()
        message.x, message.y = effective[0], effective[1]
        message.theta = float(theta)
        self.publisher.publish(message)

        self.last_requested_xy = effective
        self.last_requested_label = label
        # 밀림 측정 창의 기준 시각 (sysnav_node._is_measurable_waypoint).
        self._node._last_goal_publish_time = time.monotonic()
        self.last_snap_distance_m = (
            None if snapped is None else math.dist(requested, effective)
        )
        moved = math.dist(requested, effective)
        trace = getattr(self._node, "_trace_navigation", None)
        if trace is not None and moved >= 0.05:
            trace("SNAP",
                  f"({requested[0]:.2f},{requested[1]:.2f}) -> "
                  f"({effective[0]:.2f},{effective[1]:.2f}) moved={moved:.2f}m label={label}")
        self.publish_requested_marker()
        return message

    def _far_throw(self, x: float, y: float, label: str) -> tuple[float, float] | None:
        """목표 방향으로 adjDisThre(5m)를 넘겨 던진 좌표. 못 던지면 None.

        commandable 지점이 로봇 반경 1.75m 안에만 존재하는 교착을 빠져나가는 유일한
        경로다(config.FAR_THROW_ENABLED 주석의 실측 참고). 5m 밖 목표는 waypointConverter가
        손대지 않고 그대로 넘기고, localPlanner가 자체 회피로 그쪽으로 몰고 간다.

        도착은 못 한다(벽 너머일 수 있다). 그건 의도된 것이다 - 로봇은 갈 수 있는 만큼
        전진하고, 진전이 멈추면 기존 stuck timeout이 회수해서 최신 지도로 다시 고른다.
        """
        if not config.FAR_THROW_ENABLED:
            return None
        robot_xy = self._robot_xy()
        if robot_xy is None:
            return None
        dx, dy = float(x) - robot_xy[0], float(y) - robot_xy[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return None

        # 벽을 코앞에 두고 던지면 localPlanner가 갈 길을 못 찾아 제자리 회전만 한다.
        clear = self._node.coverage_planner.clear_distance_along(
            robot_xy, (dx, dy), config.TERRAIN_ADJ_DIS_M + config.FAR_THROW_MARGIN_M
        )
        if clear < config.FAR_THROW_MIN_CLEAR_M:
            trace = getattr(self._node, "_trace_navigation", None)
            if trace is not None:
                trace("FAR_THROW_SKIP",
                      f"dir=({dx / norm:+.2f},{dy / norm:+.2f}) clear={clear:.2f}m "
                      f"< {config.FAR_THROW_MIN_CLEAR_M:.2f}m label={label}")
            return None

        distance = config.TERRAIN_ADJ_DIS_M + config.FAR_THROW_MARGIN_M
        target = (robot_xy[0] + dx / norm * distance, robot_xy[1] + dy / norm * distance)
        trace = getattr(self._node, "_trace_navigation", None)
        if trace is not None:
            trace("FAR_THROW",
                  f"({x:.2f},{y:.2f}) -> ({target[0]:.2f},{target[1]:.2f}) "
                  f"{distance:.1f}m out, clear={clear:.2f}m label={label}")
        return target

    def _marker(self, marker_id: int, marker_type: int, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = config.OBJECT_MARKER_FRAME_ID
        marker.header.stamp = stamp
        marker.ns = _REQUESTED_NAMESPACE
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def publish_requested_marker(self, actual_xy: tuple[float, float] | None = None) -> None:
        """우리가 요청한 좌표를 청록 구로 그린다. base autonomy가 확정한 좌표(actual_xy)를
        알고 있으면 그 사이를 빨간 선으로 잇고 밀려난 거리를 글자로 띄운다 - "우리 목표가
        실제로 얼마나 옮겨졌나"를 RViz에서 한눈에 보기 위한 것이다."""
        if self.last_requested_xy is None:
            return
        x, y = self.last_requested_xy
        stamp = self._node.get_clock().now().to_msg()
        markers: list[Marker] = []

        sphere = self._marker(0, Marker.SPHERE, stamp)
        sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = x, y, 0.2
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.3
        (sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a) = _REQUESTED_COLOR
        markers.append(sphere)

        text = self._marker(1, Marker.TEXT_VIEW_FACING, stamp)
        text.pose.position.x, text.pose.position.y, text.pose.position.z = x, y, 0.75
        text.scale.z = 0.22
        (text.color.r, text.color.g, text.color.b, text.color.a) = _REQUESTED_COLOR
        text.text = f"requested ({self.last_requested_label})"
        markers.append(text)

        if actual_xy is not None:
            offset = math.hypot(actual_xy[0] - x, actual_xy[1] - y)
            line = self._marker(2, Marker.LINE_LIST, stamp)
            line.scale.x = 0.05
            (line.color.r, line.color.g, line.color.b, line.color.a) = _DISPLACEMENT_COLOR
            for px, py in ((x, y), actual_xy):
                point = Point()
                point.x, point.y, point.z = float(px), float(py), 0.2
                line.points.append(point)
            markers.append(line)

            offset_text = self._marker(3, Marker.TEXT_VIEW_FACING, stamp)
            offset_text.pose.position.x = (x + actual_xy[0]) / 2.0
            offset_text.pose.position.y = (y + actual_xy[1]) / 2.0
            offset_text.pose.position.z = 0.45
            offset_text.scale.z = 0.22
            (offset_text.color.r, offset_text.color.g, offset_text.color.b,
             offset_text.color.a) = _DISPLACEMENT_COLOR
            offset_text.text = f"pushed {offset:.2f}m"
            markers.append(offset_text)

        self.requested_marker_pub.publish(MarkerArray(markers=markers))

    def reset_step_markers(self) -> None:
        """새 mission3 task 시작 시 이전 task의 goal1/2/3 마커를 지운다."""
        self._goal_markers = []
        clear = Marker()
        clear.header.frame_id = config.OBJECT_MARKER_FRAME_ID
        clear.header.stamp = self._node.get_clock().now().to_msg()
        clear.ns = _MISSION3_GOAL_NAMESPACE
        clear.action = Marker.DELETEALL
        self.goal_marker_pub.publish(MarkerArray(markers=[clear]))

    def add_step_goal_marker(
        self,
        step_index: int,
        x: float,
        y: float,
        label: str,
        gate_segment=None,
        gate_crossed: bool = False,
    ) -> None:
        """mission3 step(0-based index)이 실제로 향하는 최종 goal 좌표를 goal{N} 라벨로
        추가한다 - "success는 떴는데 실제로는 물체 앞까지 안 갔다"를 RViz에서 눈으로 바로
        비교하기 위함(로봇 실제 이동 경로 vs 여기 찍힌 goal 위치). 이전 step의 마커는
        지우지 않고 계속 쌓아서 한 task의 goal1/2/3을 한눈에 같이 볼 수 있게 한다.

        gate_segment("take the path between A and B"의 A-B 선분)가 있으면 같이 그린다.
        통과 전은 노랑, 통과 후는 초록이라 제약을 실제로 만족했는지가 한눈에 보인다.
        마커 id는 step당 4칸(sphere/text/gate line/gate endpoints)을 쓴다."""
        stamp = self._node.get_clock().now().to_msg()

        sphere = Marker()
        sphere.header.frame_id = config.OBJECT_MARKER_FRAME_ID
        sphere.header.stamp = stamp
        sphere.ns = _MISSION3_GOAL_NAMESPACE
        sphere.id = step_index * _MARKERS_PER_STEP
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(x)
        sphere.pose.position.y = float(y)
        sphere.pose.position.z = 0.2
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.25
        sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = _MISSION3_GOAL_COLOR

        text = Marker()
        text.header.frame_id = config.OBJECT_MARKER_FRAME_ID
        text.header.stamp = stamp
        text.ns = _MISSION3_GOAL_NAMESPACE
        text.id = step_index * _MARKERS_PER_STEP + 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(x)
        text.pose.position.y = float(y)
        text.pose.position.z = 0.55
        text.pose.orientation.w = 1.0
        text.scale.z = 0.25
        text.color.r, text.color.g, text.color.b, text.color.a = (1.0, 1.0, 1.0, 1.0)
        text.text = label

        markers = [sphere, text]
        if gate_segment is not None:
            markers.extend(self._gate_markers(step_index, gate_segment, gate_crossed, stamp))

        replaced = {marker.id for marker in markers}
        self._goal_markers = [
            marker for marker in self._goal_markers if marker.id not in replaced
        ]
        self._goal_markers.extend(markers)
        self.goal_marker_pub.publish(MarkerArray(markers=list(self._goal_markers)))

    @staticmethod
    def _gate_markers(step_index: int, segment, crossed: bool, stamp) -> list[Marker]:
        """A-B 게이트 선분 + 양 끝점. "이 사이를 지나가야 한다"를 RViz에 그대로 그린다."""
        color = _MISSION3_GATE_CROSSED_COLOR if crossed else _MISSION3_GATE_COLOR
        (ax, ay), (bx, by) = segment

        def endpoint(px: float, py: float) -> Point:
            point = Point()
            point.x, point.y, point.z = float(px), float(py), 0.2
            return point

        line = Marker()
        line.header.frame_id = config.OBJECT_MARKER_FRAME_ID
        line.header.stamp = stamp
        line.ns = _MISSION3_GOAL_NAMESPACE
        line.id = step_index * _MARKERS_PER_STEP + 2
        line.type = Marker.LINE_LIST
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.08
        line.color.r, line.color.g, line.color.b, line.color.a = color
        line.points = [endpoint(ax, ay), endpoint(bx, by)]

        # 선분 끝이 어느 물체였는지 보이도록 양 끝에 작은 구를 찍는다(SPHERE_LIST라
        # 마커 하나로 두 점을 그린다 - id 소비를 늘리지 않으려고).
        ends = Marker()
        ends.header.frame_id = config.OBJECT_MARKER_FRAME_ID
        ends.header.stamp = stamp
        ends.ns = _MISSION3_GOAL_NAMESPACE
        ends.id = step_index * _MARKERS_PER_STEP + 3
        ends.type = Marker.SPHERE_LIST
        ends.action = Marker.ADD
        ends.pose.orientation.w = 1.0
        ends.scale.x = ends.scale.y = ends.scale.z = 0.2
        ends.color.r, ends.color.g, ends.color.b, ends.color.a = color
        ends.points = [endpoint(ax, ay), endpoint(bx, by)]

        return [line, ends]

    @staticmethod
    def object_approach_pose(robot_pose: dict, object_position, standoff: float = config.TARGET_STANDOFF_DISTANCE_M) -> tuple[float, float, float]:
        ox, oy = float(object_position[0]), float(object_position[1])
        rx, ry = float(robot_pose["x"]), float(robot_pose["y"])
        dx, dy = ox - rx, oy - ry
        distance = math.hypot(dx, dy)
        if distance <= 1e-6:
            return rx, ry, float(robot_pose["yaw"])
        usable = min(standoff, max(0.0, distance - 0.15))
        gx, gy = ox - usable * dx / distance, oy - usable * dy / distance
        return gx, gy, math.atan2(oy - gy, ox - gx)
