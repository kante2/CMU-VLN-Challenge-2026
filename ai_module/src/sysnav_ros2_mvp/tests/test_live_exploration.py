"""Live exploration-only check against the running simulator.

Step 1 of the intended pipeline (explore room -> scene graph -> goal -> navigate)
in isolation: this node does mapping + frontier exploration and nothing else -
no YOLO/SAM2, no Gemini, no scene graph, no missions. Use it to answer one
question: does the robot actually finish exploring the whole room?

Prerequisites:
  1. the simulator is up (./docker/run_scene.sh <scene>)
  2. the real sysnav node is NOT running - both publish to
     config.TOPIC_WAYPOINT and would fight over the robot

Run inside the sysnav container:
    docker exec -it iros2026_sysnav_module bash -lc \
      "source /home/docker/ai_module/install/setup.bash && \
       python3 /home/docker/ai_module/src/sysnav_ros2_mvp/tests/test_live_exploration.py"

It prints a progress line every few seconds, then a summary when
plan_route() reports no reachable frontier left, and writes a top-down PNG to
config.DEBUG_DIR/live_exploration_result.png.

Exploration is judged complete by the same signal the missions use: plan_route()
returning an empty route. "remaining_surface_points" in the summary is the
honest cross-check - a large number there means exploration stopped while
free/unknown boundary was still left (something blocked it), not because the
room was actually finished.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from sysnav import config
from sysnav.exploration.coverage_planner import CoveragePlanner
from sysnav.exploration.viewpoint_memory import ViewpointMemory
from sysnav.ros_helpers import (
    closest_stamped_item,
    message_stamp_to_sec,
    odometry_to_pose,
    pointcloud2_to_xyz,
)

# Import-time guard so pytest can collect this file on a host without ROS
# (it holds no unittest cases - it is a runnable node, not an offline test).
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2

    from sysnav.navigation.goal_publisher import GoalPublisher

    ROS_AVAILABLE = True
except ImportError:  # pragma: no cover
    ROS_AVAILABLE = False
    Node = object

PROGRESS_INTERVAL_SEC = 5.0
# Give up waiting for sensors rather than hanging silently if the simulator or
# its topics are not actually up.
SENSOR_WAIT_TIMEOUT_SEC = 30.0


class LiveExplorationChecker(Node):
    def __init__(self) -> None:
        super().__init__("sysnav_live_exploration_check")

        self.planner = CoveragePlanner()
        self.viewpoint_memory = ViewpointMemory()
        self.goal_publisher = GoalPublisher(self)

        self.latest_pose: dict | None = None
        self.pose_buffer: list[tuple[float, dict]] = []
        self.pending_scan: tuple[object, dict] | None = None

        self.state = "WAIT_SENSORS"
        self.phase = "FRONTIER"  # FRONTIER -> FLOOR_COVERAGE
        self.round = 1
        self.round_results: list[dict] = []
        self.route: list[dict] = []
        self.current_goal: dict | None = None
        self.goal_best_distance_m: float | None = None
        self.goal_last_progress_time: float | None = None

        # Mapping (~65ms) and planning (~200ms on a room-sized map) both exceed
        # a slice of the 0.2s control period, so they run on workers exactly as
        # sysnav_node.py does. Running them inline stalls the ROS callbacks and
        # the robot ends up chasing stale goals.
        self.map_worker = ThreadPoolExecutor(max_workers=1)
        self.plan_worker = ThreadPoolExecutor(max_workers=1)
        self.map_future: Future | None = None
        self.plan_future: Future | None = None

        self.started_at = time.monotonic()
        self.last_progress_print = 0.0
        self.last_map_update = 0.0
        self.waypoints_published = 0
        self.goals_skipped_unreachable = 0
        self.goals_in_occupied_cell = 0

        self.create_subscription(
            Odometry, config.TOPIC_STATE, self.state_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            PointCloud2, config.TOPIC_SCAN, self.scan_callback, qos_profile_sensor_data
        )
        self.create_timer(config.CONTROL_PERIOD_SEC, self.control_loop)

        self.get_logger().info(
            "Live exploration check started - mapping + frontier exploration only "
            "(no perception, no missions). Make sure the real sysnav node is stopped."
        )

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------

    def state_callback(self, msg: Odometry) -> None:
        pose = odometry_to_pose(msg)
        self.latest_pose = pose
        self.pose_buffer.append((pose["stamp"], pose))
        if len(self.pose_buffer) > config.POSE_BUFFER_SIZE:
            del self.pose_buffer[: -config.POSE_BUFFER_SIZE]

    def scan_callback(self, msg: PointCloud2) -> None:
        stamp = message_stamp_to_sec(msg)
        pose = closest_stamped_item(
            list(self.pose_buffer), stamp, config.SENSOR_SYNC_TOLERANCE_SEC
        )
        if pose is None and self.latest_pose is not None:
            pose = dict(self.latest_pose)
        if pose is not None:
            # Mapping runs in control_loop, not here: update_from_scan() is the
            # expensive call and doing it on the sensor callback would stall
            # incoming scans.
            self.pending_scan = (msg, dict(pose))

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def control_loop(self) -> None:
        now = time.monotonic()

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

        if self.state == "DONE":
            return

        if self.latest_pose is None:
            if now - self.started_at >= SENSOR_WAIT_TIMEOUT_SEC:
                self.get_logger().error(
                    f"No {config.TOPIC_STATE} message in {SENSOR_WAIT_TIMEOUT_SEC:.0f}s - "
                    "is the simulator running?"
                )
                self.finish(reason="no_sensor_data")
            return
        pose = dict(self.latest_pose)

        if self.state == "WAIT_SENSORS":
            # One mapping cycle must land before planning, otherwise the grid is
            # entirely unknown and plan_route() fails for the wrong reason.
            if self.planner.origin_x is not None:
                self.state = "PLAN"
            return

        if now - self.last_progress_print >= PROGRESS_INTERVAL_SEC:
            self.last_progress_print = now
            self.print_progress(pose)

        if self.state == "PLAN":
            if self.plan_future is None:
                self.plan_future = self.plan_worker.submit(self.plan_job, pose)
                return
            if not self.plan_future.done():
                return
            route = self.plan_future.result()
            self.plan_future = None
            if not route:
                self.finish(
                    reason="no_reachable_frontier" if self.phase == "FRONTIER"
                    else "floor_coverage_exhausted"
                )
                return
            self.route = list(route)
            self.publish_next_goal(pose)
            return

        if self.state == "FOLLOW":
            self.follow_goal(pose, now)

    def plan_job(self, pose: dict) -> list[dict]:
        """Frontier first; once that is exhausted, keep going on floor coverage.

        Frontier exhaustion only means the map is drawn - the LiDAR reaches far
        enough to map a room from a couple of poses, while the camera that has
        to recognize objects only saw part of it. The second phase targets
        mapped floor the robot never got close to.
        """
        route = self.planner.plan_route(pose, self.viewpoint_memory, room_segmentation=None)
        if route:
            self.phase = "FRONTIER"
            return route
        if not config.EXPLORATION_FLOOR_COVERAGE_ENABLED:
            return []
        if self.phase != "FLOOR_COVERAGE":
            self.phase = "FLOOR_COVERAGE"
            self.get_logger().info(
                "frontier exhausted -> switching to floor-coverage exploration "
                f"(observed {self.planner.floor_coverage()['ratio']:.1%} of mapped floor, "
                f"target {config.EXPLORATION_FLOOR_COVERAGE_TARGET:.0%}); "
                f"frontier diagnostics: {self.planner.describe_last_plan_failure()}"
            )
        route = self.planner.plan_floor_coverage(pose)
        if route:
            return route
        return self.start_next_round_or_stop(pose)

    def start_next_round_or_stop(self, pose: dict) -> list[dict]:
        """A finished round wipes the observed mask and sweeps the room again.

        A round that ends short of the target still gets a successor: clearing
        the mask makes the floor around the robot count as unobserved again, so
        there is reachable work even when the leftover corner of the previous
        round was unreachable. EXPLORATION_COVERAGE_ROUNDS is the only stop
        condition needed - it already bounds the total, so a failed round cannot
        loop forever.
        """
        coverage = self.planner.floor_coverage()
        self.round_results.append({
            "round": self.round,
            "coverage": coverage["ratio"],
            "reason": self.planner.last_floor_coverage_diagnostics.get("reason", "?"),
        })
        if self.round >= config.EXPLORATION_COVERAGE_ROUNDS:
            return []

        self.round += 1
        self.planner.reset_observed()
        # Visited-viewpoint memory has to go too: plan_route()/is_near_visited
        # would otherwise reject everything already covered in round 1.
        self.viewpoint_memory.clear()
        self.phase = "FRONTIER"
        self.get_logger().info(
            f"round {self.round - 1} ended at {coverage['ratio']:.1%} coverage "
            f"({self.planner.last_floor_coverage_diagnostics.get('reason', '?')}) -> "
            f"starting round {self.round}/{config.EXPLORATION_COVERAGE_ROUNDS} "
            "(observed mask cleared, occupancy map kept)"
        )
        # The occupancy map is already complete, so the frontier phase will find
        # nothing and this falls straight through to a fresh floor-coverage sweep.
        return self.plan_job(pose)

    def publish_next_goal(self, pose: dict) -> None:
        if not self.route:
            self.state = "PLAN"
            self.current_goal = None
            return
        goal = self.route.pop(0)
        self.goal_publisher.publish(goal["x"], goal["y"], goal["theta"])
        self.current_goal = goal
        self.goal_best_distance_m = None
        self.goal_last_progress_time = time.monotonic()
        self.waypoints_published += 1
        self.state = "FOLLOW"

        # A goal on a cell the map itself calls OCCUPIED means something erased
        # or mis-stamped that cell; the robot will drive into it.
        cell = self.planner.world_to_grid(goal["x"], goal["y"])
        if cell is not None and self.planner.grid[cell] == config.OCC_OCCUPIED:
            self.goals_in_occupied_cell += 1
            self.get_logger().warning(
                f"goal ({goal['x']:.2f}, {goal['y']:.2f}) sits on a cell marked OCCUPIED"
            )

        distance = math.hypot(goal["x"] - pose["x"], goal["y"] - pose["y"])
        self.get_logger().info(
            f"-> goal=({goal['x']:.2f}, {goal['y']:.2f}) dist={distance:.2f}m "
            f"is_viewpoint={goal.get('is_viewpoint')} coverage={goal.get('coverage_score', 0)} "
            f"remaining_in_route={len(self.route)}"
        )

    def follow_goal(self, pose: dict, now: float) -> None:
        if self.current_goal is None:
            self.state = "PLAN"
            return
        goal = self.current_goal
        distance = math.hypot(goal["x"] - pose["x"], goal["y"] - pose["y"])

        if distance <= config.GOAL_REACHED_DISTANCE_M:
            if goal.get("is_viewpoint"):
                self.viewpoint_memory.add(
                    goal["x"], goal["y"], goal["theta"], goal.get("coverage_score")
                )
            self.publish_next_goal(pose)
            return

        if (
            self.goal_best_distance_m is None
            or distance <= self.goal_best_distance_m - config.EXPLORATION_STUCK_PROGRESS_M
        ):
            self.goal_best_distance_m = distance
            self.goal_last_progress_time = now
            return

        if now - (self.goal_last_progress_time or now) >= config.EXPLORATION_STUCK_TIMEOUT_SEC:
            self.goals_skipped_unreachable += 1
            self.get_logger().warning(
                f"skip - no progress for {config.EXPLORATION_STUCK_TIMEOUT_SEC:.0f}s toward "
                f"({goal['x']:.2f}, {goal['y']:.2f}), remaining={distance:.2f}m"
            )
            # Treat an unreachable goal as visited so the same spot is not
            # picked again, matching sysnav_node.py's FOLLOW_EXPLORATION branch.
            self.viewpoint_memory.add(
                goal["x"], goal["y"], goal["theta"], goal.get("coverage_score")
            )
            self.publish_next_goal(pose)

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def map_stats(self) -> dict:
        grid = self.planner.snapshot_grid()
        cell_area = self.planner.resolution ** 2
        free = int((grid == config.OCC_FREE).sum())
        occupied = int((grid == config.OCC_OCCUPIED).sum())
        # S(=표면)가 아니라 미탐색 경계 개수를 보고해야 하는 자리다 - 이제
        # surface_point_mask()는 벽 표면까지 포함하므로 0이 되지 않는다.
        surface = int(self.planner.frontier_mask(grid).sum())
        return {
            "free_cells": free,
            "free_area_m2": free * cell_area,
            "occupied_cells": occupied,
            "remaining_surface_points": surface,
        }

    def print_progress(self, pose: dict) -> None:
        stats = self.map_stats()
        coverage = self.planner.floor_coverage()
        elapsed = time.monotonic() - self.started_at
        self.get_logger().info(
            f"[{elapsed:5.1f}s] r{self.round}/{config.EXPLORATION_COVERAGE_ROUNDS} "
            f"{self.phase}/{self.state} "
            f"pose=({pose['x']:.2f}, {pose['y']:.2f}) "
            f"mapped={stats['free_area_m2']:.1f}m2 "
            f"observed={coverage['ratio']:.0%} (unseen {coverage['unobserved_area_m2']:.1f}m2) "
            f"frontier_cells={stats['remaining_surface_points']} "
            f"waypoints={self.waypoints_published} skipped={self.goals_skipped_unreachable}"
        )

    def finish(self, reason: str) -> None:
        self.state = "DONE"
        stats = self.map_stats()
        elapsed = time.monotonic() - self.started_at
        surface = stats["remaining_surface_points"]
        coverage = self.planner.floor_coverage()
        target = config.EXPLORATION_FLOOR_COVERAGE_TARGET
        rounds_done = len(self.round_results)
        round_summary = ", ".join(
            f"r{item['round']}={item['coverage']:.1%}({item.get('reason', '?')})"
            for item in self.round_results
        ) or "-"
        if stats["free_cells"] == 0:
            # Nothing was ever mapped, so "no frontier left" says nothing about
            # the room - reporting completeness here would be a false pass.
            verdict = "INCONCLUSIVE - nothing was mapped (no sensor data reached this node)"
        elif coverage["ratio"] >= target:
            verdict = (
                f"LOOKS COMPLETE - observed {coverage['ratio']:.1%} of mapped floor "
                f"({rounds_done}/{config.EXPLORATION_COVERAGE_ROUNDS} rounds)"
            )
        else:
            # Mapped-but-never-approached floor is the case frontier-only
            # exploration silently declares finished.
            verdict = (
                f"INCOMPLETE - only observed {coverage['ratio']:.1%} of mapped floor "
                f"(target {target:.0%}); {coverage['unobserved_area_m2']:.1f} m2 never seen up close"
            )
        self.get_logger().info(
            "\n===== exploration finished =====\n"
            f"reason               : {reason}\n"
            f"verdict              : {verdict}\n"
            f"last phase           : {self.phase}\n"
            f"rounds completed     : {rounds_done}/{config.EXPLORATION_COVERAGE_ROUNDS} {round_summary}\n"
            f"elapsed              : {elapsed:.1f}s\n"
            f"mapped free area     : {stats['free_area_m2']:.1f} m2 ({stats['free_cells']} cells)\n"
            f"observed floor       : {coverage['ratio']:.1%} "
            f"({coverage['observed_cells']}/{coverage['free_cells']} cells, "
            f"radius {config.EXPLORATION_OBSERVE_RADIUS_M:.1f}m)\n"
            f"never seen up close  : {coverage['unobserved_area_m2']:.1f} m2\n"
            f"occupied cells       : {stats['occupied_cells']}\n"
            f"remaining frontier   : {surface} cells\n"
            f"viewpoints visited   : {len(self.viewpoint_memory.snapshot())}\n"
            f"waypoints published  : {self.waypoints_published}\n"
            f"goals skipped (stuck): {self.goals_skipped_unreachable}\n"
            f"goals on OCCUPIED    : {self.goals_in_occupied_cell}\n"
            f"frontier diagnostics : {self.planner.describe_last_plan_failure()}\n"
            f"floor cov diagnostics: {self.planner.describe_last_floor_coverage()}\n"
            "================================"
        )
        self.save_debug_png()
        self.save_state_npz()

    def save_state_npz(self) -> None:
        """Dump the raw map + observed mask so coverage can be scored against the
        scene's ground-truth traversable area afterwards.

        The percentages above are self-referential - the denominator is "floor
        discovered so far", not the real room. tests/check_gt_coverage.py reads
        this file on the host (config.DEBUG_DIR is bind-mounted) together with
        map/<scene>.zip's traversable_area.ply to produce a true number.
        """
        if self.planner.origin_x is None or self.planner.origin_y is None:
            return
        path = os.path.join(config.DEBUG_DIR, "live_exploration_result.npz")
        try:
            np.savez_compressed(
                path,
                grid=self.planner.snapshot_grid(),
                observed=self.planner.snapshot_observed(),
                origin_x=self.planner.origin_x,
                origin_y=self.planner.origin_y,
                resolution=self.planner.resolution,
                observe_radius_m=config.EXPLORATION_OBSERVE_RADIUS_M,
            )
            self.get_logger().info(
                f"map state written to {path} - score it against ground truth with:\n"
                "  python3 tests/check_gt_coverage.py <scene_name>   (run on the host, "
                "from ai_module/src/sysnav_ros2_mvp)"
            )
        except Exception as error:  # pragma: no cover
            self.get_logger().warning(f"could not write state npz: {error}")

    def save_debug_png(self) -> None:
        try:
            import cv2
        except ImportError:  # pragma: no cover
            return
        grid = self.planner.snapshot_grid()
        observed = self.planner.snapshot_observed()
        img = np.zeros((*grid.shape, 3), dtype=np.uint8)
        img[grid == config.OCC_UNKNOWN] = (90, 90, 90)
        img[grid == config.OCC_FREE] = (255, 255, 255)
        # Mapped floor the robot never got close to - the gap frontier-only
        # exploration leaves behind, drawn so it is visible at a glance.
        img[(grid == config.OCC_FREE) & (~observed)] = (170, 200, 255)
        img[grid == config.OCC_OCCUPIED] = (0, 0, 0)
        for item in self.viewpoint_memory.snapshot():
            cell = self.planner.world_to_grid(item["x"], item["y"])
            if cell is not None:
                cv2.circle(img, (cell[1], cell[0]), 2, (0, 160, 255), -1)
        known = np.argwhere(grid != config.OCC_UNKNOWN)
        if len(known):
            (r0, c0), (r1, c1) = known.min(axis=0), known.max(axis=0)
            pad = 6
            img = img[max(0, r0 - pad):r1 + pad, max(0, c0 - pad):c1 + pad]
        if img.size:
            img = cv2.resize(
                img, (img.shape[1] * 4, img.shape[0] * 4), interpolation=cv2.INTER_NEAREST
            )
            path = os.path.join(config.DEBUG_DIR, "live_exploration_result.png")
            try:
                cv2.imwrite(path, img)
                self.get_logger().info(
                    f"debug map written to {path} "
                    "(white=observed floor, pale blue=mapped but never approached, "
                    "gray=unknown, black=obstacle, orange dots=viewpoints)"
                )
            except Exception as error:  # pragma: no cover
                self.get_logger().warning(f"could not write debug PNG: {error}")


def main() -> None:
    if not ROS_AVAILABLE:
        raise SystemExit(
            "rclpy is unavailable - run this inside the sysnav container, after "
            "sourcing /home/docker/ai_module/install/setup.bash"
        )
    rclpy.init()
    node = LiveExplorationChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("interrupted - reporting what was mapped so far")
        if node.state != "DONE":
            node.finish(reason="interrupted")
    finally:
        node.map_worker.shutdown(wait=False, cancel_futures=True)
        node.plan_worker.shutdown(wait=False, cancel_futures=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
