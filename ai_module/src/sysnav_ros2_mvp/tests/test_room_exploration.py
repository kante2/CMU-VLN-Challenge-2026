"""Standalone check for the frontier exploration algorithm alone - no ROS, no
simulator, no perception/mission logic. Drives CoveragePlanner.plan_route()
against a synthetic ground-truth room (two areas joined by a narrow doorway,
plus a tight furniture cluster) and checks two things:

  1. repeated planning cycles converge to covering the whole reachable floor
  2. every waypoint it hands out lies in genuinely free space

Run standalone (also writes a debug PNG):
    cd ai_module/src/sysnav_ros2_mvp
    PYTHONPATH=. python3 tests/test_room_exploration.py

Or as a normal test:
    PYTHONPATH=. python3 -m pytest tests/test_room_exploration.py -v -s

CoveragePlanner samples candidates through an unseeded RNG, so every run
differs. Both the runner and the tests below seed it explicitly - without that
the pass/fail outcome is a coin flip rather than a signal.
"""

from __future__ import annotations

import math
import os
import unittest

import numpy as np

from sysnav import config
from sysnav.exploration.coverage_planner import CoveragePlanner
from sysnav.exploration.viewpoint_memory import ViewpointMemory

# ---------------------------------------------------------------------------
# Synthetic ground-truth room. Two rectangular areas divided by a wall with a
# narrow doorway gap (tests frontier connectivity through a doorway) plus a
# tight furniture cluster in the second area (tests narrow-space routing - the
# class of problem the visibility-graph routing in coverage_planner.py
# targets). All coordinates are meters in the planner's world frame.
# ---------------------------------------------------------------------------

ROOM_BOUNDS = (0.0, 0.0, 8.0, 6.0)  # x_min, y_min, x_max, y_max (free interior)
WALL_THICKNESS = 0.6
DIVIDER_X = 4.0
DOORWAY_Y = (2.5, 3.5)  # gap in the divider wall

FURNITURE = [
    (5.3, 1.3, 5.7, 1.7),
    (6.2, 1.3, 6.6, 1.7),
    (5.3, 2.2, 5.7, 2.6),
    (6.2, 2.2, 6.6, 2.6),
]

START_POSE = {"x": 1.0, "y": 1.0, "yaw": 0.0}


def _occupied_rects() -> list[tuple[float, float, float, float]]:
    x0, y0, x1, y1 = ROOM_BOUNDS
    t = WALL_THICKNESS
    rects = [
        (x0 - t, y0 - t, x1 + t, y0),  # bottom wall
        (x0 - t, y1, x1 + t, y1 + t),  # top wall
        (x0 - t, y0 - t, x0, y1 + t),  # left wall
        (x1, y0 - t, x1 + t, y1 + t),  # right wall
        (DIVIDER_X - t / 2, y0, DIVIDER_X + t / 2, DOORWAY_Y[0]),  # divider, lower part
        (DIVIDER_X - t / 2, DOORWAY_Y[1], DIVIDER_X + t / 2, y1),  # divider, upper part
    ]
    rects.extend(FURNITURE)
    return rects


_OCCUPIED_RECTS = _occupied_rects()


def is_occupied(x: float, y: float) -> bool:
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in _OCCUPIED_RECTS)


def is_true_free(x: float, y: float) -> bool:
    x0, y0, x1, y1 = ROOM_BOUNDS
    return x0 <= x <= x1 and y0 <= y <= y1 and not is_occupied(x, y)


def simulate_scan(pose: dict, n_rays: int = 180, max_range: float = 10.0, step: float = 0.05) -> np.ndarray:
    """Ray-march against the ground truth from `pose`; return hit points as an
    (N, 3) array in the sensor frame. config.T_SENSOR_TO_BASE is the identity,
    so sensor frame == base frame, which is what update_from_scan() expects."""
    yaw = float(pose["yaw"])
    points = []
    for i in range(n_rays):
        angle = yaw + 2.0 * math.pi * i / n_rays
        dx, dy = math.cos(angle), math.sin(angle)
        distance = step
        hit = None
        while distance <= max_range:
            wx = pose["x"] + dx * distance
            wy = pose["y"] + dy * distance
            if is_occupied(wx, wy):
                hit = (wx, wy)
                break
            distance += step
        if hit is None:
            continue
        rel_x, rel_y = hit[0] - pose["x"], hit[1] - pose["y"]
        bx = math.cos(-yaw) * rel_x - math.sin(-yaw) * rel_y
        by = math.sin(-yaw) * rel_x + math.cos(-yaw) * rel_y
        points.append((bx, by, 0.3))  # z within MAP_OBSTACLE_Z_MIN/MAX_M
    return np.asarray(points, dtype=np.float64) if points else np.zeros((0, 3))


def coverage_ratio(planner: CoveragePlanner) -> float:
    """Fraction of truly-free room cells the planner has marked FREE."""
    x0, y0, x1, y1 = ROOM_BOUNDS
    step = planner.resolution
    total = covered = 0
    y = y0 + step / 2
    while y < y1:
        x = x0 + step / 2
        while x < x1:
            if is_true_free(x, y):
                total += 1
                cell = planner.world_to_grid(x, y)
                if cell is not None and planner.grid[cell] == config.OCC_FREE:
                    covered += 1
            x += step
        y += step
    return covered / total if total else 0.0


def run_exploration(max_cycles: int = 150, seed: int = 0) -> dict:
    """Drive plan_route() cycle by cycle until it reports no reachable frontier
    left (or max_cycles). The robot is teleported to each chosen viewpoint:
    this exercises the exploration ALGORITHM's coverage behavior, not physical
    collision avoidance, which needs the real simulator.

    A waypoint landing in ground-truth-occupied space is recorded rather than
    followed - moving there would let scans leak through the wall and corrupt
    the coverage number, hiding the very defect worth reporting.
    """
    planner = CoveragePlanner()
    planner._rng = np.random.default_rng(seed)
    planner.reset(START_POSE)
    viewpoint_memory = ViewpointMemory()

    pose = dict(START_POSE)
    planner.update_from_scan(simulate_scan(pose), pose)

    invalid_waypoints: list[tuple[float, float]] = []
    # Waypoints the planner issued onto a cell it ALREADY had marked occupied.
    # That is a planner defect, unlike a waypoint that only turns out to be
    # occupied later - planning always runs on an incomplete map, and the robot
    # discovering an obstacle after being sent that way is normal.
    waypoints_on_known_obstacles: list[tuple[float, float]] = []
    cycles = 0
    for cycles in range(1, max_cycles + 1):
        route = planner.plan_route(pose, viewpoint_memory, room_segmentation=None)
        if not route:
            break
        # Walk the whole route, exactly as sysnav_node's FOLLOW_EXPLORATION does
        # (publish_next_exploration_goal pops one waypoint at a time). Jumping
        # straight to route[-1] would skip every earlier candidate, leaving the
        # surface they were chosen to cover unobserved - the planner would then
        # keep re-picking them and never converge.
        # Snapshot the map as it was when the whole route was issued. Checking
        # planner.grid while consuming the route would be wrong: scans taken at
        # earlier waypoints reveal obstacles the planner could not have known
        # about, and a later waypoint would be blamed for them.
        grid_at_issue = planner.snapshot_grid()
        for waypoint in route:
            cell = planner.world_to_grid(waypoint["x"], waypoint["y"])
            if cell is not None and grid_at_issue[cell] == config.OCC_OCCUPIED:
                waypoints_on_known_obstacles.append((waypoint["x"], waypoint["y"]))
            if not is_true_free(waypoint["x"], waypoint["y"]):
                invalid_waypoints.append((waypoint["x"], waypoint["y"]))
                continue
            pose = {"x": waypoint["x"], "y": waypoint["y"], "yaw": waypoint["theta"]}
            planner.update_from_scan(simulate_scan(pose), pose)
            if waypoint.get("is_viewpoint"):
                viewpoint_memory.add(
                    waypoint["x"], waypoint["y"], waypoint["theta"],
                    waypoint.get("coverage_score"),
                )
    return {
        "planner": planner,
        "cycles": cycles,
        "coverage": coverage_ratio(planner),
        "invalid_waypoints": invalid_waypoints,
        "waypoints_on_known_obstacles": waypoints_on_known_obstacles,
        "final_pose": pose,
    }


class RoomExplorationTest(unittest.TestCase):
    MAX_CYCLES = 150
    MIN_COVERAGE_RATIO = 0.85
    SEEDS = (0, 1, 2, 3, 4)

    def test_frontier_exploration_covers_whole_room(self):
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                result = run_exploration(self.MAX_CYCLES, seed=seed)
                print(
                    f"\n[seed {seed}] cycles={result['cycles']} "
                    f"coverage={result['coverage']:.1%} "
                    f"invalid_waypoints={len(result['invalid_waypoints'])}"
                )
                self.assertGreaterEqual(
                    result["coverage"], self.MIN_COVERAGE_RATIO,
                    f"seed {seed}: exploration only covered {result['coverage']:.1%} "
                    f"in {result['cycles']} cycles "
                    f"({result['planner'].describe_last_plan_failure()})",
                )

    def test_plan_route_alone_covers_the_room_in_one_pass(self):
        """plan_route()가 surface point 기준으로 바뀐 뒤에는 별도 floor-coverage
        단계나 라운드 반복 없이 한 패스로 방을 다 훑어야 한다.

        관측 반경을 좁혀(1.2m) "LiDAR가 지도는 다 그렸지만 가까이서 본 곳은 일부"인
        상황을 강제로 만든다 - 기본 3m 반경이면 8x6m 합성 방은 너무 쉽게 덮여서
        차이가 드러나지 않는다. 예전 frontier 기준 구현은 이 조건에서 72.5%에서
        멈췄고, 그래서 plan_floor_coverage()라는 두 번째 단계가 필요했다."""
        original_radius = config.EXPLORATION_OBSERVE_RADIUS_M
        config.EXPLORATION_OBSERVE_RADIUS_M = 1.2
        try:
            result = run_exploration(self.MAX_CYCLES, seed=0)
            planner = result["planner"]
            floor = planner.floor_coverage()["ratio"]
            surface = planner.surface_coverage()["ratio"]
            print(
                f"\n[plan_route single pass] {result['cycles']} cycles: "
                f"floor={floor:.1%} surface={surface:.1%} "
                f"(stop: {planner.describe_last_plan_failure()})"
            )
            # 발급 시점에 이미 장애물로 알고 있던 셀에 목표를 찍는 건 플래너 결함이다.
            # 반면 나중에야 장애물로 밝혀지는 waypoint는 불완전 지도로 계획하는 이
            # 시나리오(관측 반경 1.2m로 좁혀 가구에 훨씬 가까이 붙어 다닌다)에서는
            # 정상이므로 개수만 남기고 통과시킨다.
            self.assertEqual(
                result["waypoints_on_known_obstacles"], [],
                "planner issued a goal onto a cell it already knew was occupied",
            )
            if result["invalid_waypoints"]:
                print(
                    f"  (참고: 발급 후에 장애물로 밝혀진 waypoint "
                    f"{len(result['invalid_waypoints'])}개 - 불완전 지도 계획의 정상 결과)"
                )
            self.assertGreaterEqual(
                floor, config.EXPLORATION_FLOOR_COVERAGE_TARGET,
                f"plan_route alone only reached {floor:.1%} floor coverage, below the "
                f"{config.EXPLORATION_FLOOR_COVERAGE_TARGET:.0%} target - the surface-point "
                "objective is supposed to make the separate floor phase unnecessary",
            )
            # plan_floor_coverage()는 이제 보조 수단이다: 이미 다 덮인 상태에서는
            # 할 일이 없다고 답해야 한다(예전엔 여기서 일을 찾아야 정상이었다).
            leftover = planner.plan_floor_coverage(result["final_pose"])
            self.assertEqual(
                leftover, [],
                "plan_route left uncovered floor behind for the fallback phase to pick up",
            )
        finally:
            config.EXPLORATION_OBSERVE_RADIUS_M = original_radius

    def test_surface_point_policy_beats_frontier_only(self):
        """논문(Sec. IV-B-1)의 surface point 정의로 탐사하면 frontier만 좇는 것보다
        한 패스에서 훨씬 많이 훑는지 확인한다.

        frontier(free/unknown 경계)는 지도가 완성되면 사라져서 탐사가 조기 종료되지만,
        논문의 S는 free/non-free 경계(occupied 포함)라 벽·가구 표면으로 남는다 - 그래서
        단계 전환이나 라운드 반복 없이 한 번에 방을 다 훑게 된다."""
        import importlib.util

        module_path = os.path.join(os.path.dirname(__file__), "test_live_exploration_with_surfacePoint.py")
        spec = importlib.util.spec_from_file_location("surface_policy_module", module_path)
        surface_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(surface_module)

        planner = CoveragePlanner()
        planner._rng = np.random.default_rng(0)
        planner.reset(START_POSE)
        policy = surface_module.SurfaceCoveragePolicy(planner)

        pose = dict(START_POSE)
        planner.update_from_scan(simulate_scan(pose), pose)
        invalid_waypoints = 0
        cycles = 0
        for cycles in range(1, self.MAX_CYCLES + 1):
            route = policy.plan(pose)
            if not route:
                break
            for waypoint in route:
                if not is_true_free(waypoint["x"], waypoint["y"]):
                    invalid_waypoints += 1
                    continue
                pose = {"x": waypoint["x"], "y": waypoint["y"], "yaw": waypoint["theta"]}
                planner.update_from_scan(simulate_scan(pose), pose)

        surface = policy.surface_stats()
        floor = planner.floor_coverage()
        print(
            f"\n[surface policy] {cycles} cycles: surface={surface['ratio']:.1%} "
            f"floor={floor['ratio']:.1%} map_vs_truth={coverage_ratio(planner):.1%} "
            f"(stop: {policy.last_note})"
        )
        self.assertEqual(invalid_waypoints, 0, "surface policy commanded a waypoint into an obstacle")
        self.assertGreaterEqual(
            surface["ratio"], 0.90,
            f"surface coverage stalled at {surface['ratio']:.1%}",
        )
        # frontier-only가 같은 방에서 남기는 미관측 바닥(약 77%, 이 파일의
        # floor-coverage 테스트 참고)보다 확실히 높아야 의미가 있다.
        self.assertGreater(
            floor["ratio"], 0.90,
            f"single-pass floor coverage only reached {floor['ratio']:.1%}",
        )

    def test_waypoints_stay_in_free_space(self):
        """A waypoint inside ground-truth-occupied space means the robot is
        being commanded into a wall or a piece of furniture. update_from_scan()
        stamps a ~1m box of OCC_FREE around the robot unconditionally, which
        can erase a thin obstacle it drove past; the erased cell then reads as
        traversable and can be handed out as a goal."""
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                result = run_exploration(self.MAX_CYCLES, seed=seed)
                self.assertEqual(
                    result["invalid_waypoints"], [],
                    f"seed {seed}: {len(result['invalid_waypoints'])} waypoint(s) landed in "
                    f"occupied space, first few: {result['invalid_waypoints'][:5]}",
                )


def _save_debug_png(planner: CoveragePlanner, path: str) -> None:
    import cv2

    grid = planner.grid
    img = np.zeros((*grid.shape, 3), dtype=np.uint8)
    img[grid == config.OCC_UNKNOWN] = (90, 90, 90)
    img[grid == config.OCC_FREE] = (255, 255, 255)
    img[grid == config.OCC_OCCUPIED] = (0, 0, 0)
    x0, y0, x1, y1 = ROOM_BOUNDS
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1]):
        a, b = planner.world_to_grid(ax, ay), planner.world_to_grid(bx, by)
        if a and b:
            cv2.line(img, (a[1], a[0]), (b[1], b[0]), (255, 120, 0), 1)
    # Crop to the mapped region so the room isn't a speck in a 300x300 frame.
    known = np.argwhere(grid != config.OCC_UNKNOWN)
    if len(known):
        r0, c0 = known.min(axis=0)
        r1, c1 = known.max(axis=0)
        pad = 6
        img = img[max(0, r0 - pad):r1 + pad, max(0, c0 - pad):c1 + pad]
    img = cv2.resize(img, (img.shape[1] * 6, img.shape[0] * 6), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, img)


if __name__ == "__main__":
    import sys

    ok = True
    for seed in RoomExplorationTest.SEEDS:
        result = run_exploration(RoomExplorationTest.MAX_CYCLES, seed=seed)
        invalid = result["invalid_waypoints"]
        print(
            f"seed={seed}  cycles={result['cycles']}  coverage={result['coverage']:.1%}  "
            f"invalid_waypoints={len(invalid)}"
            + (f"  first={invalid[0]}" if invalid else "")
        )
        if result["coverage"] < RoomExplorationTest.MIN_COVERAGE_RATIO or invalid:
            ok = False
        if seed == RoomExplorationTest.SEEDS[0]:
            out_path = "/tmp/room_exploration_result.png"
            _save_debug_png(result["planner"], out_path)
            print(f"  debug image: {out_path} (gray=unknown, white=free, black=occupied, blue=true room)")
    sys.exit(0 if ok else 1)
