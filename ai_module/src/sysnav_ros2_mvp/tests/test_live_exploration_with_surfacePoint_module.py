"""Callable module wrapper around the surface-point exploration policy.

tests/test_live_exploration_with_surfacePoint.py is a standalone ROS node: it
owns subscriptions, timers and a shutdown path, which is what you want for an
experiment but not for the real pipeline. This file exposes the same policy as
something the existing sysnav_node can simply *call* during its own control
loop - the "1. explore room -> 2. scene graph -> 3. navigate" flow needs the
exploration phase to be a step, not a separate process fighting for the robot.

Design constraints that follow from that:

  * No ROS imports here. Goals leave through a `publish_goal(x, y, theta)`
    callback and messages through an optional `logger`, so the same object runs
    inside sysnav_node, inside the experiment node, or in an offline test with
    a synthetic room.
  * Non-blocking. Planning costs ~150-200 ms on a room-sized map, well over the
    0.2 s control period, so `update()` hands planning to an optional `submit`
    callable (e.g. `node.worker.submit`) and returns immediately, exactly as
    sysnav_node already does for perception. Without `submit` it plans inline,
    which is fine offline.
  * Idempotent per tick. `update(pose)` may be called every control cycle; it
    publishes a goal only when the goal actually changes.

Typical use inside sysnav_node's control loop:

    if self.explorer is None:
        self.explorer = SurfaceExplorer(
            planner=self.coverage_planner,
            publish_goal=self.goal_publisher.publish,
            submit=self.worker.submit,
            logger=self.get_logger(),
        )
    status = self.explorer.update(pose)
    if status.done:
        # exploration finished - hand over to scene-graph / mission logic
        self.state = "OBSERVE"

Run it standalone (same behaviour as the experiment node, driven through this
API instead) inside the sysnav container:

    python3 tests/test_live_exploration_with_surfacePoint_module.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import time

from sysnav import config
from sysnav.exploration.coverage_planner import CoveragePlanner
from sysnav.exploration.viewpoint_memory import ViewpointMemory


def _load_surface_policy():
    """Import SurfaceCoveragePolicy from the sibling experiment file.

    Loaded by path rather than by name: this file is run both as
    `python3 tests/<file>.py` (sys.path[0] = tests/) and imported as
    `tests.<file>` from the repo root, and a plain import statement would only
    work in one of those.
    """
    module_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_live_exploration_with_surfacePoint.py",
    )
    spec = importlib.util.spec_from_file_location("_surface_policy_impl", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_policy_module = _load_surface_policy()
SurfaceCoveragePolicy = _policy_module.SurfaceCoveragePolicy
surface_point_mask = _policy_module.surface_point_mask


class ExplorationStatus:
    """What the host loop needs to know after one `update()` call.

    Deliberately a plain class rather than a dataclass: this file gets loaded by
    path through importlib in places, and @dataclass resolves its own module out
    of sys.modules, which a path-loaded module is not registered in.
    """

    def __init__(
        self,
        done: bool = False,
        reason: str = "",
        phase: str = "PLANNING",          # PLANNING | FOLLOWING | DONE
        goal: dict | None = None,
        surface_ratio: float = 0.0,
        surface_uncovered: int = 0,
        unknown_side_cells: int = 0,
        floor_ratio: float = 0.0,
        waypoints_published: int = 0,
        goals_skipped: int = 0,
    ) -> None:
        self.done = done
        self.reason = reason
        self.phase = phase
        self.goal = goal
        self.surface_ratio = surface_ratio
        self.surface_uncovered = surface_uncovered
        self.unknown_side_cells = unknown_side_cells
        self.floor_ratio = floor_ratio
        self.waypoints_published = waypoints_published
        self.goals_skipped = goals_skipped

    def summary(self) -> str:
        return (
            f"{self.phase} surface={self.surface_ratio:.0%} "
            f"({self.surface_uncovered} cells left, {self.unknown_side_cells} unknown-side) "
            f"floor={self.floor_ratio:.0%} waypoints={self.waypoints_published} "
            f"skipped={self.goals_skipped}"
        )


class _Goal:
    def __init__(
        self,
        x: float,
        y: float,
        theta: float,
        is_viewpoint: bool = False,
        last_progress_time: float | None = None,
    ) -> None:
        self.x = x
        self.y = y
        self.theta = theta
        self.is_viewpoint = is_viewpoint
        self.best_distance_m: float | None = None
        self.last_progress_time = (
            time.monotonic() if last_progress_time is None else last_progress_time
        )


class SurfaceExplorer:
    """Drive surface-point exploration from someone else's control loop.

    The caller keeps ownership of ROS: it feeds poses in and receives goals
    through `publish_goal`. Mapping stays the caller's job too - whatever
    already calls `CoveragePlanner.update_from_scan()` keeps doing so, and this
    object reads the resulting map.
    """

    def __init__(
        self,
        planner: CoveragePlanner,
        publish_goal,
        submit=None,
        logger=None,
        viewpoint_memory: ViewpointMemory | None = None,
    ) -> None:
        self.planner = planner
        self.publish_goal = publish_goal
        self.submit = submit
        self.logger = logger
        self.viewpoint_memory = viewpoint_memory or ViewpointMemory()
        self.policy = SurfaceCoveragePolicy(planner)

        self.route: list[dict] = []
        self.goal: _Goal | None = None
        self.phase = "PLANNING"
        self.done = False
        self.done_reason = ""
        self._plan_future = None
        self.waypoints_published = 0
        self.goals_skipped_unreachable = 0
        self.goals_in_occupied_cell = 0
        self.started_at = time.monotonic()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def update(self, pose: dict, now: float | None = None) -> ExplorationStatus:
        """Advance exploration by one control tick. Safe to call every cycle."""
        now = time.monotonic() if now is None else now
        if self.done:
            return self._status()

        if self.planner.origin_x is None:
            # No scan has landed yet, so there is no map to reason about.
            return self._status(phase="WAITING_FOR_MAP")

        if self.goal is not None:
            self._advance_goal(pose, now)
            if self.done:
                return self._status()
            if self.goal is not None:
                return self._status()

        if not self.route:
            self._plan(pose)
            if self.done or not self.route:
                return self._status()

        self._publish_next_goal(pose, now)
        return self._status()

    def stats(self) -> dict:
        surface = self.policy.surface_stats()
        floor = self.planner.floor_coverage()
        return {
            "surface": surface,
            "floor": floor,
            "elapsed_sec": time.monotonic() - self.started_at,
            "plan_cycles": self.policy.plan_cycles,
            "waypoints_published": self.waypoints_published,
            "goals_skipped_unreachable": self.goals_skipped_unreachable,
            "goals_in_occupied_cell": self.goals_in_occupied_cell,
            "viewpoints": len(self.viewpoint_memory.snapshot()),
            "done_reason": self.done_reason,
        }

    def report(self) -> str:
        """Multi-line summary for the end of a run."""
        stats = self.stats()
        surface, floor = stats["surface"], stats["floor"]
        cell_area = self.planner.resolution ** 2
        if floor["free_cells"] == 0:
            verdict = "INCONCLUSIVE - nothing was mapped"
        elif surface["uncovered"] == 0:
            verdict = "COMPLETE - every surface point seen from within d_cover"
        elif surface["ratio"] >= 0.90:
            verdict = f"NEARLY COMPLETE - {surface['ratio']:.1%} of surfaces covered"
        else:
            verdict = (
                f"INCOMPLETE - {surface['ratio']:.1%} of surfaces covered, "
                f"{surface['uncovered']} cells never approached"
            )
        return (
            "\n===== surface exploration report =====\n"
            f"verdict              : {verdict}\n"
            f"stop reason          : {stats['done_reason'] or '(still running)'}\n"
            f"elapsed              : {stats['elapsed_sec']:.1f}s\n"
            f"surface coverage     : {surface['ratio']:.1%} "
            f"({surface['covered']}/{surface['total']} cells, "
            f"d_cover={config.FRONTIER_COVERAGE_RADIUS_M:.1f}m)\n"
            f"  still unknown-side : {surface['frontier_like']} cells\n"
            f"floor coverage       : {floor['ratio']:.1%} "
            f"({floor['observed_cells']}/{floor['free_cells']} cells)\n"
            f"mapped free area     : {floor['free_cells'] * cell_area:.1f} m2\n"
            f"planning cycles      : {stats['plan_cycles']}\n"
            f"waypoints published  : {stats['waypoints_published']}\n"
            f"viewpoints reached   : {stats['viewpoints']}\n"
            f"goals skipped (stuck): {stats['goals_skipped_unreachable']}\n"
            f"goals on OCCUPIED    : {stats['goals_in_occupied_cell']}\n"
            "======================================"
        )

    def reset(self) -> None:
        """Start over on the current map (keeps the occupancy grid).

        Clears the observed mask, so every surface counts as uncovered again -
        useful for a deliberate second sweep. The map itself is kept: walls are
        still needed for planning and there is no reason to redraw them.
        """
        self.planner.reset_observed()
        self.viewpoint_memory.clear()
        self.policy.global_horizon.clear()
        self.route.clear()
        self.goal = None
        self.done = False
        self.done_reason = ""
        self.phase = "PLANNING"
        self._plan_future = None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _plan(self, pose: dict) -> None:
        if self.submit is None:
            route = self.policy.plan(pose)
            self._consume_plan(route)
            return
        if self._plan_future is None:
            self._plan_future = self.submit(self.policy.plan, pose)
            self.phase = "PLANNING"
            return
        if not self._plan_future.done():
            return
        route = self._plan_future.result()
        self._plan_future = None
        self._consume_plan(route)

    def _consume_plan(self, route: list[dict]) -> None:
        if route:
            self.route = list(route)
            return
        self.done = True
        self.done_reason = self.policy.last_note
        self.phase = "DONE"
        self._log(f"exploration finished: {self.done_reason}")

    def _publish_next_goal(self, pose: dict, now: float) -> None:
        waypoint = self.route.pop(0)
        self.publish_goal(waypoint["x"], waypoint["y"], waypoint["theta"])
        self.goal = _Goal(
            x=float(waypoint["x"]),
            y=float(waypoint["y"]),
            theta=float(waypoint["theta"]),
            is_viewpoint=bool(waypoint.get("is_viewpoint")),
            last_progress_time=now,
        )
        self.phase = "FOLLOWING"
        self.waypoints_published += 1

        cell = self.planner.world_to_grid(self.goal.x, self.goal.y)
        if cell is not None and self.planner.grid[cell] == config.OCC_OCCUPIED:
            self.goals_in_occupied_cell += 1
            self._log(
                f"goal ({self.goal.x:.2f}, {self.goal.y:.2f}) sits on a cell marked OCCUPIED",
                warning=True,
            )
        distance = math.hypot(self.goal.x - pose["x"], self.goal.y - pose["y"])
        self._log(
            f"-> goal=({self.goal.x:.2f}, {self.goal.y:.2f}) dist={distance:.2f}m "
            f"remaining_in_route={len(self.route)}"
        )

    def _advance_goal(self, pose: dict, now: float) -> None:
        goal = self.goal
        assert goal is not None
        distance = math.hypot(goal.x - pose["x"], goal.y - pose["y"])

        if distance <= config.GOAL_REACHED_DISTANCE_M:
            if goal.is_viewpoint:
                self.viewpoint_memory.add(goal.x, goal.y, goal.theta)
            self.goal = None
            return

        if (
            goal.best_distance_m is None
            or distance <= goal.best_distance_m - config.EXPLORATION_STUCK_PROGRESS_M
        ):
            goal.best_distance_m = distance
            goal.last_progress_time = now
            return

        if now - goal.last_progress_time >= config.EXPLORATION_STUCK_TIMEOUT_SEC:
            self.goals_skipped_unreachable += 1
            self._log(
                f"skip - no progress for {config.EXPLORATION_STUCK_TIMEOUT_SEC:.0f}s toward "
                f"({goal.x:.2f}, {goal.y:.2f}), remaining={distance:.2f}m",
                warning=True,
            )
            # Treat an unreachable goal as visited so the same spot is not
            # picked again on the next planning cycle.
            self.viewpoint_memory.add(goal.x, goal.y, goal.theta)
            self.goal = None

    def _status(self, phase: str | None = None) -> ExplorationStatus:
        surface = self.policy.surface_stats()
        floor = self.planner.floor_coverage()
        return ExplorationStatus(
            done=self.done,
            reason=self.done_reason,
            phase=phase or self.phase,
            goal=None if self.goal is None else {"x": self.goal.x, "y": self.goal.y, "theta": self.goal.theta},
            surface_ratio=surface["ratio"],
            surface_uncovered=surface["uncovered"],
            unknown_side_cells=surface["frontier_like"],
            floor_ratio=floor["ratio"],
            waypoints_published=self.waypoints_published,
            goals_skipped=self.goals_skipped_unreachable,
        )

    def _log(self, message: str, warning: bool = False) -> None:
        if self.logger is None:
            return
        if warning and hasattr(self.logger, "warning"):
            self.logger.warning(message)
        elif hasattr(self.logger, "info"):
            self.logger.info(message)


# ---------------------------------------------------------------------------
# Optional standalone runner - same experiment as
# test_live_exploration_with_surfacePoint.py, but driven through the API above,
# which also serves as a worked example of embedding it.
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import PointCloud2

        from sysnav.navigation.goal_publisher import GoalPublisher
    except ImportError as error:  # pragma: no cover
        raise SystemExit(
            "rclpy is unavailable - run this inside the sysnav container after sourcing "
            f"/home/docker/ai_module/install/setup.bash ({error})"
        )

    from concurrent.futures import ThreadPoolExecutor

    from sysnav.ros_helpers import (
        closest_stamped_item,
        message_stamp_to_sec,
        odometry_to_pose,
        pointcloud2_to_xyz,
    )

    class ExplorationRunner(Node):
        def __init__(self) -> None:
            super().__init__("sysnav_surface_exploration_module")
            self.planner = CoveragePlanner()
            self.worker = ThreadPoolExecutor(max_workers=1)
            self.map_worker = ThreadPoolExecutor(max_workers=1)
            self.map_future = None
            self.explorer = SurfaceExplorer(
                planner=self.planner,
                publish_goal=GoalPublisher(self).publish,
                submit=self.worker.submit,
                logger=self.get_logger(),
            )
            self.latest_pose = None
            self.pose_buffer: list[tuple[float, dict]] = []
            self.pending_scan = None
            self.last_map_update = 0.0
            self.last_progress = 0.0
            self.started_at = time.monotonic()
            self.finished = False

            self.create_subscription(
                Odometry, config.TOPIC_STATE, self._on_state, qos_profile_sensor_data
            )
            self.create_subscription(
                PointCloud2, config.TOPIC_SCAN, self._on_scan, qos_profile_sensor_data
            )
            self.create_timer(config.CONTROL_PERIOD_SEC, self._tick)
            self.get_logger().info(
                "surface exploration (module API) started - make sure the real sysnav node is stopped"
            )

        def _on_state(self, msg) -> None:
            pose = odometry_to_pose(msg)
            self.latest_pose = pose
            self.pose_buffer.append((pose["stamp"], pose))
            if len(self.pose_buffer) > config.POSE_BUFFER_SIZE:
                del self.pose_buffer[: -config.POSE_BUFFER_SIZE]

        def _on_scan(self, msg) -> None:
            stamp = message_stamp_to_sec(msg)
            pose = closest_stamped_item(
                list(self.pose_buffer), stamp, config.SENSOR_SYNC_TOLERANCE_SEC
            )
            if pose is None and self.latest_pose is not None:
                pose = dict(self.latest_pose)
            if pose is not None:
                self.pending_scan = (msg, dict(pose))

        def _tick(self) -> None:
            now = time.monotonic()
            # Mapping stays the host's responsibility, exactly as it is in
            # sysnav_node - SurfaceExplorer only reads the resulting map.
            if (
                self.pending_scan is not None
                and now - self.last_map_update >= config.MAP_UPDATE_INTERVAL_SEC
                and (self.map_future is None or self.map_future.done())
            ):
                scan_msg, scan_pose = self.pending_scan
                self.pending_scan = None
                self.last_map_update = now
                self.map_future = self.map_worker.submit(
                    lambda: self.planner.update_from_scan(pointcloud2_to_xyz(scan_msg), scan_pose)
                )

            if self.finished or self.latest_pose is None:
                if self.latest_pose is None and now - self.started_at > 30.0:
                    self.get_logger().error(
                        f"no {config.TOPIC_STATE} in 30s - is the simulator running?"
                    )
                    self.finished = True
                return

            status = self.explorer.update(dict(self.latest_pose), now)
            if now - self.last_progress >= 5.0:
                self.last_progress = now
                self.get_logger().info(f"[{now - self.started_at:5.1f}s] {status.summary()}")
            if status.done:
                self.finished = True
                self.get_logger().info(self.explorer.report())

    rclpy.init()
    node = ExplorationRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("interrupted" + node.explorer.report())
    finally:
        node.worker.shutdown(wait=False, cancel_futures=True)
        node.map_worker.shutdown(wait=False, cancel_futures=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
