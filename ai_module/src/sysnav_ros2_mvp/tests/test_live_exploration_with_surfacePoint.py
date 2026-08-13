"""Live exploration driven by the paper's surface-point coverage (SysNav Sec. IV-B-1).

Why this exists next to tests/test_live_exploration.py: that one explores until
the *frontier* is gone, then bolts on a separate floor-coverage phase and repeats
the whole sweep for several rounds. This one follows the paper instead, which
removes the need for both the second phase and the rounds.

The paper defines the surface point set S as

    "the generalized boundary between free and non-free space
     (including occupied and unknown space)"

Our production frontier_extractor._mask() only marks free cells adjacent to
UNKNOWN, i.e. the classic frontier. That single omission is what makes
frontier-only exploration quit early: once the LiDAR has swept the room, no
unknown space touches free space, S goes empty, and exploration declares
victory after a couple of viewpoints - even though the camera that has to
recognize objects has barely seen the place.

With the paper's definition, S never empties out. When unknown space is gone,
S becomes the *wall and furniture surfaces*, so exploration keeps going until
the robot has been within d_cover of every surface. That is both the paper's
intent and what object recognition actually needs: objects sit on and against
surfaces, not in the middle of the floor. Frontier-chasing and
surface-inspection stop being two phases - they are one objective at two
stages of the same map.

Two more things taken from the paper:

  * Uncovered surface Ŝ is remembered across the whole run. Coverage is
    `observed & S` - the planner's observed mask already records which free
    cells were seen from within the observe radius, so no extra bookkeeping is
    needed and progress can never be silently forgotten.
  * A rolling local/global horizon: candidates sampled but not visited are kept
    in a global horizon and TSP-ordered, instead of being thrown away every
    cycle (EXPLORATION_MAX_CANDIDATES_PER_CYCLE=1 in the production planner).
    The sweep becomes systematic, which is what makes one pass enough.

Prerequisites and usage are the same as tests/test_live_exploration.py:

  1. the simulator is up (./docker/run_scene.sh <scene>)
  2. the real sysnav node is NOT running (it publishes to the same topic)

    docker exec -it iros2026_sysnav_module bash -lc \
      "source /home/docker/ai_module/install/setup.bash && \
       python3 /home/docker/ai_module/src/sysnav_ros2_mvp/tests/test_live_exploration_with_surfacePoint.py"

Afterwards, score it against the scene's ground truth on the host:

    python3 tests/check_gt_coverage.py <scene_name>

This file drives CoveragePlanner for mapping and reuses its A*/visibility-graph
primitives, but keeps the surface-coverage policy local - the production
missions keep using plan_route()/plan_floor_coverage() untouched, so the two
strategies can be compared on the same scene.

The policy itself lives in SurfaceCoveragePolicy, deliberately free of any ROS
dependency, so it can be exercised against the synthetic room in
tests/test_room_exploration.py without a simulator.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np

from sysnav import config
from sysnav.exploration import visibility_path
from sysnav.exploration.coverage_planner import CoveragePlanner
from sysnav.exploration.viewpoint_memory import ViewpointMemory
from sysnav.ros_helpers import (
    closest_stamped_item,
    message_stamp_to_sec,
    odometry_to_pose,
    pointcloud2_to_xyz,
)

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
SENSOR_WAIT_TIMEOUT_SEC = 30.0

# How many candidates one planning cycle commits to as a TSP tour. The paper
# selects a set and solves a TSP over it; going one-at-a-time (the production
# planner's EXPLORATION_MAX_CANDIDATES_PER_CYCLE=1) re-decides constantly and
# wastes the sampling work.
MAX_TOUR_CANDIDATES = 4
# Cap on the global horizon so scoring stays cheap on a big map.
GLOBAL_HORIZON_MAX = 400
_NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def surface_point_mask(grid: np.ndarray) -> np.ndarray:
    """The paper's S: free cells bordering non-free space (occupied OR unknown).

    Contrast with frontier_extractor._mask(), which drops the occupied side and
    therefore empties out as soon as the map is complete.
    """
    free = grid == config.OCC_FREE
    non_free = (grid == config.OCC_UNKNOWN) | (grid == config.OCC_OCCUPIED)
    padded = np.pad(non_free, 1, constant_values=False)
    adjacent = np.zeros_like(non_free, dtype=bool)
    for dr, dc in _NEIGHBORS_8:
        adjacent |= padded[1 + dr:1 + dr + grid.shape[0], 1 + dc:1 + dc + grid.shape[1]]
    return free & adjacent


class SurfaceCoveragePolicy:
    """Surface-point coverage planning, with no ROS dependency.

    Kept separate from the node so it can be run against the synthetic room
    offline (tests/test_room_exploration.py) instead of only in the simulator.
    """

    def __init__(self, planner: CoveragePlanner) -> None:
        self.planner = planner
        # Candidates sampled but not yet visited (the paper's global horizon).
        self.global_horizon: list[tuple[int, int]] = []
        self.last_note = "-"
        self.plan_cycles = 0

    def surface_stats(self) -> dict:
        """Coverage is `observed & S`: the observed mask already records which
        free cells were seen from within the observe radius, so the covered
        surface set persists for the whole run without extra bookkeeping."""
        grid = self.planner.snapshot_grid()
        observed = self.planner.snapshot_observed()
        surface = surface_point_mask(grid)
        total = int(surface.sum())
        covered = int((surface & observed).sum())
        # How much of S is still the unknown boundary tells us which stage of
        # the same objective we are in - no phase flag needed.
        unknown_side = int((surface & _adjacent_to(grid, config.OCC_UNKNOWN)).sum())
        return {
            "total": total,
            "covered": covered,
            "uncovered": total - covered,
            "ratio": (covered / total) if total else 0.0,
            "frontier_like": unknown_side,
        }

    def plan(self, pose: dict) -> list[dict]:
        """One planning cycle: score the horizon by uncovered surface gain, take
        a TSP tour over the best few, keep the rest for later."""
        self.plan_cycles += 1
        planner = self.planner
        grid = planner.snapshot_grid()
        observed = planner.snapshot_observed()
        if planner.origin_x is None:
            self.last_note = "origin_not_ready"
            return []
        robot_cell = planner.world_to_grid(pose["x"], pose["y"])
        if robot_cell is None:
            self.last_note = "robot_cell_out_of_map"
            return []

        occupied = (grid == config.OCC_OCCUPIED).astype(np.uint8)
        inflation = max(1, int(round(config.ROBOT_CLEARANCE_M / planner.resolution)))
        inflated = cv2.dilate(
            occupied, np.ones((2 * inflation + 1, 2 * inflation + 1), np.uint8)
        ).astype(bool)
        traversable = (grid == config.OCC_FREE) & (~inflated)
        start = planner._nearest_traversable(traversable, *robot_cell, radius=10)
        if start is None:
            self.last_note = "robot_not_near_traversable"
            return []

        # Visibility for scoring uses a thin wall margin, not the body clearance:
        # surface points sit right against obstacles by definition, so a
        # clearance-inflated check would call almost all of them invisible.
        los_margin = max(1, int(round(config.FRONTIER_LOS_WALL_MARGIN_M / planner.resolution)))
        los_blocking = cv2.dilate(
            occupied, np.ones((2 * los_margin + 1, 2 * los_margin + 1), np.uint8)
        ).astype(bool)

        surface = surface_point_mask(grid)
        uncovered = surface & (~observed)
        targets = np.argwhere(uncovered)
        if len(targets) == 0:
            self.last_note = "all_surface_covered"
            return []

        pool = self.sample_pool(traversable, uncovered, inflation)
        d_cover_cells = config.FRONTIER_COVERAGE_RADIUS_M / planner.resolution
        target_xy = targets.astype(np.float64)
        target_list = [tuple(int(v) for v in cell) for cell in targets]

        scored: list[tuple[int, tuple[int, int]]] = []
        for cell in pool:
            within = np.hypot(
                target_xy[:, 0] - cell[0], target_xy[:, 1] - cell[1]
            ) <= d_cover_cells
            nearby = [target_list[i] for i in np.nonzero(within)[0]]
            if len(nearby) < config.EXPLORATION_MIN_SCORE_DELTA:
                continue
            gain = sum(
                1 for target in nearby
                if planner._line_of_sight(los_blocking, cell, target)
            )
            if gain >= config.EXPLORATION_MIN_SCORE_DELTA:
                scored.append((gain, cell))

        # Candidates whose score fell below delta are covered now; the rest stay
        # in the global horizon so the sampling work is never thrown away.
        self.global_horizon = [cell for _, cell in scored][:GLOBAL_HORIZON_MAX]
        if not scored:
            self.last_note = f"no_candidate_above_delta (uncovered={len(target_list)})"
            return []

        selected = self.select_tour(scored, start, uncovered, los_blocking, d_cover_cells)
        route = self.build_route(selected, traversable, inflated, pose, start)
        if not route:
            # Every candidate in the tour turned out to be unreachable (a pocket
            # behind furniture, or a fragmented traversable graph). Exploration
            # must not end here while surface is still uncovered - fall back to
            # the rest of the scored candidates, nearest first.
            fallback = sorted(
                (cell for _, cell in scored if cell not in selected),
                key=lambda cell: math.hypot(cell[0] - start[0], cell[1] - start[1]),
            )
            for cell in fallback:
                route = self.build_route([cell], traversable, inflated, pose, start)
                if route:
                    self.last_note = f"fallback_candidate (tour of {len(selected)} unreachable)"
                    return route
            self.last_note = f"no_reachable_candidate (uncovered={len(target_list)})"
            return []
        self.last_note = (
            f"tour={len(selected)} best_gain={max(g for g, _ in scored)} "
            f"horizon={len(self.global_horizon)}"
        )
        return route

    def sample_pool(
        self, traversable: np.ndarray, uncovered: np.ndarray, inflation: int
    ) -> list[tuple[int, int]]:
        """Local horizon (fresh samples) + global horizon (kept from before) +
        an anchor beside uncovered surface so a small leftover patch can never
        be missed by random sampling alone."""
        planner = self.planner
        traversable_cells = np.argwhere(traversable)
        pool: list[tuple[int, int]] = []
        if len(traversable_cells):
            sample_n = min(config.EXPLORATION_CANDIDATE_SAMPLES, len(traversable_cells))
            idx = planner._rng.choice(len(traversable_cells), size=sample_n, replace=False)
            pool = [(int(traversable_cells[i][0]), int(traversable_cells[i][1])) for i in idx]

        pool.extend(self.global_horizon)

        uncovered_cells = np.argwhere(uncovered)
        if len(uncovered_cells):
            stride = max(1, len(uncovered_cells) // 25)
            for cell in uncovered_cells[::stride]:
                anchor = planner._nearest_traversable(
                    traversable, int(cell[0]), int(cell[1]), radius=inflation + 3
                )
                if anchor is not None:
                    pool.append(anchor)
        return list(dict.fromkeys(pool))

    def select_tour(
        self,
        scored: list[tuple[int, tuple[int, int]]],
        start: tuple[int, int],
        uncovered: np.ndarray,
        los_blocking: np.ndarray,
        d_cover_cells: float,
    ) -> list[tuple[int, int]]:
        """Greedy set-cover over Ŝ, distance-decayed, then TSP-ordered.

        Each pick removes what it covers from the remaining uncovered set before
        the next pick, so two candidates staring at the same wall are not both
        selected. The distance decay is the same guard plan_route() uses against
        picking opposite ends of the room in turn.
        """
        remaining = uncovered.copy()
        halflife = max(
            1e-6, config.EXPLORATION_DISTANCE_PENALTY_HALFLIFE_M / self.planner.resolution
        )
        selected: list[tuple[int, int]] = []
        available = list(scored)

        for _ in range(MAX_TOUR_CANDIDATES):
            best_cell = None
            best_priority = 0.0
            best_covered: list[tuple[int, int]] = []
            anchor = selected[-1] if selected else start
            for _, cell in available:
                covered = self.visible_cells(cell, remaining, los_blocking, d_cover_cells)
                if len(covered) < config.EXPLORATION_MIN_SCORE_DELTA:
                    continue
                distance = math.hypot(cell[0] - anchor[0], cell[1] - anchor[1])
                priority = len(covered) / (1.0 + distance / halflife)
                if priority > best_priority:
                    best_priority = priority
                    best_cell = cell
                    best_covered = covered
            if best_cell is None:
                break
            selected.append(best_cell)
            available = [item for item in available if item[1] != best_cell]
            for row, col in best_covered:
                remaining[row, col] = False

        if len(selected) > 1:
            selected = self.planner._solve_tsp(start, selected)
        return selected

    def visible_cells(
        self,
        cell: tuple[int, int],
        remaining: np.ndarray,
        los_blocking: np.ndarray,
        d_cover_cells: float,
    ) -> list[tuple[int, int]]:
        radius = int(math.ceil(d_cover_cells))
        r0 = max(0, cell[0] - radius)
        r1 = min(remaining.shape[0], cell[0] + radius + 1)
        c0 = max(0, cell[1] - radius)
        c1 = min(remaining.shape[1], cell[1] + radius + 1)
        window = np.argwhere(remaining[r0:r1, c0:c1])
        visible = []
        for dr, dc in window:
            target = (int(dr) + r0, int(dc) + c0)
            if math.hypot(target[0] - cell[0], target[1] - cell[1]) > d_cover_cells:
                continue
            if self.planner._line_of_sight(los_blocking, cell, target):
                visible.append(target)
        return visible

    def build_route(
        self,
        selected: list[tuple[int, int]],
        traversable: np.ndarray,
        inflated: np.ndarray,
        pose: dict,
        start: tuple[int, int],
    ) -> list[dict]:
        """Concatenate the tour into waypoints, preferring the continuous
        visibility-graph route over grid A* for the same reason plan_route()
        does: grid hops can graze the clearance boundary in tight spaces."""
        planner = self.planner
        polygon = visibility_path.build_traversable_polygon(traversable, planner.grid_to_world)
        route: list[dict] = []
        leg_start = start
        leg_start_world = (float(pose["x"]), float(pose["y"]))
        for cell in selected:
            goal_world = planner.grid_to_world(*cell)
            theta = math.atan2(
                goal_world[1] - leg_start_world[1], goal_world[0] - leg_start_world[0]
            )
            polyline = visibility_path.shortest_path(
                polygon, leg_start_world, goal_world,
                simplify_tolerance=2.0 * planner.resolution,
            )
            if polyline is not None:
                hops = planner._waypoints_from_world_polyline(polyline, theta, 1)
                if hops:
                    route.extend(hops)
                    leg_start, leg_start_world = cell, goal_world
                    continue
            path = planner._astar_path(traversable, leg_start, cell)
            if path is None:
                continue
            route.extend(planner._leg_waypoints(path, inflated, theta, 1))
            leg_start, leg_start_world = cell, goal_world
        return route


class SurfaceCoverageChecker(Node):
    def __init__(self) -> None:
        super().__init__("sysnav_surface_exploration_check")

        self.planner = CoveragePlanner()
        self.policy = SurfaceCoveragePolicy(self.planner)
        self.viewpoint_memory = ViewpointMemory()
        self.goal_publisher = GoalPublisher(self)

        self.latest_pose: dict | None = None
        self.pose_buffer: list[tuple[float, dict]] = []
        self.pending_scan: tuple[object, dict] | None = None

        self.state = "WAIT_SENSORS"
        self.route: list[dict] = []
        self.current_goal: dict | None = None
        self.goal_best_distance_m: float | None = None
        self.goal_last_progress_time: float | None = None

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
            "Surface-point exploration check started - one objective, no phases, no rounds "
            f"(d_cover={config.FRONTIER_COVERAGE_RADIUS_M:.1f}m, "
            f"observe_radius={config.EXPLORATION_OBSERVE_RADIUS_M:.1f}m, "
            f"delta={config.EXPLORATION_MIN_SCORE_DELTA}). "
            "Make sure the real sysnav node is stopped."
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
            if self.planner.origin_x is not None:
                self.state = "PLAN"
            return

        if now - self.last_progress_print >= PROGRESS_INTERVAL_SEC:
            self.last_progress_print = now
            self.print_progress(pose)

        if self.state == "PLAN":
            if self.plan_future is None:
                self.plan_future = self.plan_worker.submit(self.policy.plan, pose)
                return
            if not self.plan_future.done():
                return
            route = self.plan_future.result()
            self.plan_future = None
            if not route:
                self.finish(reason=self.policy.last_note)
                return
            self.route = list(route)
            self.publish_next_goal(pose)
            return

        if self.state == "FOLLOW":
            self.follow_goal(pose, now)

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

        cell = self.planner.world_to_grid(goal["x"], goal["y"])
        if cell is not None and self.planner.grid[cell] == config.OCC_OCCUPIED:
            self.goals_in_occupied_cell += 1
            self.get_logger().warning(
                f"goal ({goal['x']:.2f}, {goal['y']:.2f}) sits on a cell marked OCCUPIED"
            )

        distance = math.hypot(goal["x"] - pose["x"], goal["y"] - pose["y"])
        self.get_logger().info(
            f"-> goal=({goal['x']:.2f}, {goal['y']:.2f}) dist={distance:.2f}m "
            f"is_viewpoint={goal.get('is_viewpoint')} remaining_in_route={len(self.route)}"
        )

    def follow_goal(self, pose: dict, now: float) -> None:
        if self.current_goal is None:
            self.state = "PLAN"
            return
        goal = self.current_goal
        distance = math.hypot(goal["x"] - pose["x"], goal["y"] - pose["y"])

        if distance <= config.GOAL_REACHED_DISTANCE_M:
            if goal.get("is_viewpoint"):
                self.viewpoint_memory.add(goal["x"], goal["y"], goal["theta"])
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
            self.viewpoint_memory.add(goal["x"], goal["y"], goal["theta"])
            self.publish_next_goal(pose)

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def print_progress(self, pose: dict) -> None:
        surface = self.policy.surface_stats()
        floor = self.planner.floor_coverage()
        elapsed = time.monotonic() - self.started_at
        self.get_logger().info(
            f"[{elapsed:5.1f}s] {self.state} pose=({pose['x']:.2f}, {pose['y']:.2f}) "
            f"surface={surface['ratio']:.0%} ({surface['uncovered']} cells left, "
            f"{surface['frontier_like']} still unknown-side) "
            f"floor={floor['ratio']:.0%} mapped={floor['free_cells'] * self.planner.resolution ** 2:.1f}m2 "
            f"waypoints={self.waypoints_published} skipped={self.goals_skipped_unreachable} "
            f"plan={self.policy.last_note}"
        )

    def finish(self, reason: str) -> None:
        self.state = "DONE"
        surface = self.policy.surface_stats()
        floor = self.planner.floor_coverage()
        elapsed = time.monotonic() - self.started_at
        cell_area = self.planner.resolution ** 2

        if floor["free_cells"] == 0:
            verdict = "INCONCLUSIVE - nothing was mapped (no sensor data reached this node)"
        elif surface["uncovered"] == 0:
            verdict = "COMPLETE - every surface point was seen from within d_cover"
        elif surface["ratio"] >= 0.90:
            verdict = f"NEARLY COMPLETE - {surface['ratio']:.1%} of surfaces covered"
        else:
            verdict = (
                f"INCOMPLETE - {surface['ratio']:.1%} of surfaces covered, "
                f"{surface['uncovered']} surface cells never approached"
            )

        self.get_logger().info(
            "\n===== surface-point exploration finished =====\n"
            f"reason               : {reason}\n"
            f"verdict              : {verdict}\n"
            f"elapsed              : {elapsed:.1f}s\n"
            f"surface coverage     : {surface['ratio']:.1%} "
            f"({surface['covered']}/{surface['total']} cells, d_cover="
            f"{config.FRONTIER_COVERAGE_RADIUS_M:.1f}m)\n"
            f"  still unknown-side : {surface['frontier_like']} cells "
            "(>0 means genuinely unexplored space remains)\n"
            f"floor coverage       : {floor['ratio']:.1%} "
            f"({floor['observed_cells']}/{floor['free_cells']} cells) "
            "<- comparable to test_live_exploration.py\n"
            f"mapped free area     : {floor['free_cells'] * cell_area:.1f} m2\n"
            f"planning cycles      : {self.policy.plan_cycles}\n"
            f"waypoints published  : {self.waypoints_published}\n"
            f"viewpoints reached   : {len(self.viewpoint_memory.snapshot())}\n"
            f"goals skipped (stuck): {self.goals_skipped_unreachable}\n"
            f"goals on OCCUPIED    : {self.goals_in_occupied_cell}\n"
            "=============================================="
        )
        self.save_debug_png()
        self.save_state_npz()

    def save_debug_png(self) -> None:
        grid = self.planner.snapshot_grid()
        observed = self.planner.snapshot_observed()
        surface = surface_point_mask(grid)
        img = np.zeros((*grid.shape, 3), dtype=np.uint8)
        img[grid == config.OCC_UNKNOWN] = (90, 90, 90)
        img[grid == config.OCC_FREE] = (245, 245, 245)
        img[(grid == config.OCC_FREE) & (~observed)] = (170, 200, 255)
        img[surface & observed] = (0, 180, 0)        # surface seen from close range
        img[surface & (~observed)] = (0, 0, 230)     # surface still to inspect
        img[grid == config.OCC_OCCUPIED] = (0, 0, 0)
        for item in self.viewpoint_memory.snapshot():
            cell = self.planner.world_to_grid(item["x"], item["y"])
            if cell is not None:
                cv2.circle(img, (cell[1], cell[0]), 2, (0, 200, 255), -1)

        known = np.argwhere(grid != config.OCC_UNKNOWN)
        if len(known):
            (r0, c0), (r1, c1) = known.min(axis=0), known.max(axis=0)
            pad = 6
            img = img[max(0, r0 - pad):r1 + pad, max(0, c0 - pad):c1 + pad]
        if not img.size:
            return
        img = cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
        path = os.path.join(config.DEBUG_DIR, "surface_exploration_result.png")
        try:
            cv2.imwrite(path, img)
            self.get_logger().info(
                f"debug map written to {path} (green=surface covered, red=surface not yet "
                "approached, pale blue=floor not approached, orange=viewpoints)"
            )
        except Exception as error:  # pragma: no cover
            self.get_logger().warning(f"could not write debug PNG: {error}")

    def save_state_npz(self) -> None:
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
                f"map state written to {path} - score against ground truth with:\n"
                "  python3 tests/check_gt_coverage.py <scene_name>   (on the host, from "
                "ai_module/src/sysnav_ros2_mvp)"
            )
        except Exception as error:  # pragma: no cover
            self.get_logger().warning(f"could not write state npz: {error}")


def _adjacent_to(grid: np.ndarray, value: int) -> np.ndarray:
    match = grid == value
    padded = np.pad(match, 1, constant_values=False)
    adjacent = np.zeros_like(match, dtype=bool)
    for dr, dc in _NEIGHBORS_8:
        adjacent |= padded[1 + dr:1 + dr + grid.shape[0], 1 + dc:1 + dc + grid.shape[1]]
    return adjacent


def main() -> None:
    if not ROS_AVAILABLE:
        raise SystemExit(
            "rclpy is unavailable - run this inside the sysnav container, after sourcing "
            "/home/docker/ai_module/install/setup.bash"
        )
    rclpy.init()
    node = SurfaceCoverageChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("interrupted - reporting what was covered so far")
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
