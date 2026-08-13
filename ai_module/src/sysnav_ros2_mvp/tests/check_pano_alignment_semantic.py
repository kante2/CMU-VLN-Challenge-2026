"""Score camera-LiDAR alignment objectively against the simulator's semantic image.

Overlays can only be judged by eye, and "the points look about right" is not a
measurement. The simulator publishes /camera/semantic_image: the same 1920x640
panorama with one flat colour per surface category. That turns alignment into a
number.

The idea: a LiDAR return whose height equals the floor is, by construction, a
floor hit. Project it and read the semantic colour under it. If the projection
is correct, floor hits land on floor-coloured pixels; if the vertical FOV (or a
pitch offset) is wrong, they slide onto the wall and the agreement collapses.
Wall hits at camera height give the same test in the other direction.

`purity` below is the fraction of a group's points that land on that group's own
most common colour, so it needs no knowledge of which colour means what - the
correct geometry is simply the one that keeps each group on a single surface.

Usage, with the simulator running:

    docker exec -it iros2026_sysnav_module bash -lc \
      "source /home/docker/ai_module/install/setup.bash && \
       python3 /home/docker/ai_module/src/sysnav_ros2_mvp/tests/check_pano_alignment_semantic.py"

Reports purity per candidate vertical FOV, and per pitch offset at the best FOV,
so both can be read off instead of guessed. Also writes a diagnostic image with
floor hits drawn over the semantic panorama.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from sysnav import config
from sysnav.ros_helpers import closest_stamped_item, image_msg_to_rgb, message_stamp_to_sec, pointcloud2_to_xyz

V_FOV_CANDIDATES = [float(v) for v in os.getenv("PANO_V_FOV_LIST", "90,120,150,180,200").split(",")]
PITCH_OFFSETS_DEG = [float(v) for v in os.getenv("PANO_PITCH_LIST", "-6,-3,0,3,6").split(",")]
# Sensor height above the floor in this stack (map->sensor z).
SENSOR_HEIGHT_M = 0.75
# A return is treated as a floor hit when its height sits within this band of
# the floor plane; tight enough to exclude furniture tops, loose enough to
# survive scan noise.
FLOOR_BAND_M = 0.06
# Wall hits are taken near sensor height, where a wall is the only thing a
# horizontal ray can strike at a distance.
WALL_BAND_M = 0.06
WALL_MIN_RANGE_M = 2.0
# Semantic colours are flat per category, but JPEG-free raw images still carry a
# little dither at edges; quantise so near-identical colours count as one.
COLOR_QUANTIZE = 8


def project(points_sensor: np.ndarray, width: int, height: int, v_fov_deg: float,
            pitch_offset_deg: float = 0.0) -> dict:
    """PanoramaLidarGrounder._project with v_fov and pitch exposed."""
    matrix = np.asarray(config.T_LIDAR_TO_CAMERA, dtype=np.float64)
    homogeneous = np.column_stack([points_sensor.astype(np.float64), np.ones(len(points_sensor))])
    points_camera = (homogeneous @ matrix.T)[:, :3]
    x, y, z = points_camera[:, 0], points_camera[:, 1], points_camera[:, 2]
    horizontal = np.hypot(x, y)
    ranges = np.linalg.norm(points_camera, axis=1)
    valid = (
        np.isfinite(points_camera).all(axis=1)
        & (ranges >= config.GROUNDING_MIN_RANGE_M)
        & (ranges <= config.GROUNDING_MAX_RANGE_M)
        & (horizontal > 1e-6)
    )
    indices = np.flatnonzero(valid)
    if not len(indices):
        return {"indices": indices, "u": np.empty(0, int), "v": np.empty(0, int)}
    x, y, z, horizontal = x[valid], y[valid], z[valid], horizontal[valid]
    yaw = np.arctan2(x, y) + math.radians(config.PANORAMA_YAW_OFFSET_DEG)
    down = np.arctan2(z, horizontal) + math.radians(pitch_offset_deg)
    yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
    u = np.floor(((yaw / (2.0 * math.pi)) + 0.5) * width).astype(np.int32) % width
    v = np.floor(((down / math.radians(v_fov_deg)) + 0.5) * height).astype(np.int32)
    inside = (v >= 0) & (v < height)
    return {"indices": indices[inside], "u": u[inside], "v": v[inside]}


def _modal_color(region: np.ndarray) -> tuple[int, int, int]:
    flat = region.reshape(-1, 3) // COLOR_QUANTIZE
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    best = colors[int(np.argmax(counts))]
    return tuple(int(c) * COLOR_QUANTIZE for c in best)


def project_pinhole_segments(points_sensor: np.ndarray, width: int, height: int,
                             v_fov_deg: float, segments: int = 4) -> dict:
    """Alternative model: `segments` rectilinear cameras stitched side by side.

    A stitched multi-camera panorama is not necessarily equirectangular. If each
    camera is a normal pinhole, the mapping inside its slice is tan-based, so an
    equirectangular formula drifts more and more toward the slice edges - a
    periodic error with the slice width as its period. Testing this model
    against the equirectangular one tells the two apart.
    """
    matrix = np.asarray(config.T_LIDAR_TO_CAMERA, dtype=np.float64)
    homogeneous = np.column_stack([points_sensor.astype(np.float64), np.ones(len(points_sensor))])
    points_camera = (homogeneous @ matrix.T)[:, :3]
    x, y, z = points_camera[:, 0], points_camera[:, 1], points_camera[:, 2]
    horizontal = np.hypot(x, y)
    ranges = np.linalg.norm(points_camera, axis=1)
    valid = (
        np.isfinite(points_camera).all(axis=1)
        & (ranges >= config.GROUNDING_MIN_RANGE_M)
        & (ranges <= config.GROUNDING_MAX_RANGE_M)
        & (horizontal > 1e-6)
    )
    indices = np.flatnonzero(valid)
    if not len(indices):
        return {"indices": indices, "u": np.empty(0, int), "v": np.empty(0, int)}

    x, y, z, horizontal = x[valid], y[valid], z[valid], horizontal[valid]
    yaw = np.arctan2(x, y)
    yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
    down = np.arctan2(z, horizontal)

    segment_angle = 2.0 * math.pi / segments
    segment_width = width / segments
    # Slice index, and the yaw measured from that slice's optical axis.
    slot = np.floor((yaw + math.pi) / segment_angle).astype(np.int64) % segments
    center = -math.pi + (slot + 0.5) * segment_angle
    local_yaw = yaw - center

    half_h = math.tan(segment_angle / 2.0)
    half_v = math.tan(math.radians(v_fov_deg) / 2.0)
    u = (slot + 0.5) * segment_width + (segment_width / 2.0) * (np.tan(local_yaw) / half_h)
    # Pinhole vertical mapping needs the distance along the optical axis, not the
    # full horizontal distance, or the corners come out wrong.
    axis_distance = horizontal * np.cos(local_yaw)
    v = (height / 2.0) + (height / 2.0) * (z / np.maximum(axis_distance, 1e-6)) / half_v

    u = np.floor(u).astype(np.int32) % width
    v = np.floor(v).astype(np.int32)
    inside = (v >= 0) & (v < height) & np.isfinite(v)
    return {"indices": indices[inside], "u": u[inside], "v": v[inside]}


def floor_edge_rows(semantic: np.ndarray, floor_color: np.ndarray, step: int = 8) -> dict[int, int]:
    """Row where the floor region ends (going upward) for each sampled column.

    Scanning up from the bottom and stopping at the first non-floor pixel finds
    the visual floor/wall junction - or the base of whatever furniture stands
    there, which is the same thing for this test: it is where the floor stops
    being visible in that direction.
    """
    height, width = semantic.shape[:2]
    quantized = semantic // COLOR_QUANTIZE
    target = floor_color // COLOR_QUANTIZE
    is_floor = np.all(quantized == target, axis=2)
    edges: dict[int, int] = {}
    for u in range(0, width, step):
        column = is_floor[:, u]
        if not column[height - 1]:
            continue  # bottom pixel is not floor (robot body, furniture) - skip
        row = height - 1
        while row >= 0 and column[row]:
            row -= 1
        if row < 0:
            continue
        edges[u] = row + 1  # topmost floor row in this column
    return edges


def lidar_floor_edge(points_sensor: np.ndarray, floor_hits: np.ndarray, width: int,
                     step: int = 8) -> dict[int, np.ndarray]:
    """Farthest floor return per azimuth column - the LiDAR's view of the same
    junction, expressed as the 3D point so it can be projected at any v_fov."""
    points = points_sensor[floor_hits]
    if not len(points):
        return {}
    # Azimuth in the same convention the projection uses (camera x right, y forward).
    yaw = np.arctan2(-points[:, 1], points[:, 0])
    column = np.floor(((yaw / (2.0 * math.pi)) + 0.5) * width).astype(np.int64) % width
    bucket = (column // step) * step
    horizontal = np.hypot(points[:, 0], points[:, 1])
    farthest: dict[int, np.ndarray] = {}
    for key in np.unique(bucket):
        selection = bucket == key
        index = int(np.argmax(horizontal[selection]))
        farthest[int(key)] = points[selection][index]
    return farthest


def purity(semantic: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[float, tuple[int, int, int], int]:
    """Fraction of sampled pixels sharing the group's most common colour."""
    if not len(u):
        return 0.0, (0, 0, 0), 0
    sampled = semantic[v, u] // COLOR_QUANTIZE
    colors, counts = np.unique(sampled, axis=0, return_counts=True)
    best = int(np.argmax(counts))
    modal = tuple(int(c) * COLOR_QUANTIZE for c in colors[best])
    return float(counts[best] / len(u)), modal, len(u)


def main() -> None:
    try:
        import cv2
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, PointCloud2
    except ImportError as error:  # pragma: no cover
        raise SystemExit(f"run this inside the sysnav container ({error})")

    class SemanticAlignmentCheck(Node):
        def __init__(self) -> None:
            super().__init__("sysnav_semantic_alignment_check")
            self.semantic_msg = None
            self.scan_buffer: list[tuple[float, object]] = []
            self.started = time.monotonic()
            self.reported = False
            self.create_subscription(Image, "/camera/semantic_image", self._on_semantic, qos_profile_sensor_data)
            self.create_subscription(PointCloud2, config.TOPIC_SCAN, self._on_scan, qos_profile_sensor_data)
            self.create_timer(0.3, lambda: self._tick(cv2))
            self.get_logger().info("semantic alignment check started (needs /camera/semantic_image)")

        def _on_semantic(self, msg) -> None:
            self.semantic_msg = msg

        def _on_scan(self, msg) -> None:
            self.scan_buffer.append((message_stamp_to_sec(msg), msg))
            if len(self.scan_buffer) > config.SCAN_BUFFER_SIZE:
                del self.scan_buffer[: -config.SCAN_BUFFER_SIZE]

        def _tick(self, cv2) -> None:
            if self.reported:
                raise SystemExit(0)
            if self.semantic_msg is None or not self.scan_buffer:
                if time.monotonic() - self.started > 30.0:
                    self.get_logger().error(
                        "no semantic image / scan in 30s - is the simulator running?"
                    )
                    raise SystemExit(1)
                return
            if time.monotonic() - self.started < 2.0:
                return
            self.reported = True
            self._report(cv2)

        def _report(self, cv2) -> None:
            semantic_msg = self.semantic_msg
            semantic_stamp = message_stamp_to_sec(semantic_msg)
            # The semantic panorama arrives at ~1 Hz, so pair with a wider window
            # than SENSOR_SYNC_TOLERANCE_SEC and report the gap that was used.
            scan_msg = closest_stamped_item(list(self.scan_buffer), semantic_stamp, 1.0)
            if scan_msg is None:
                self.get_logger().warning("no scan within 1s of the semantic image - retrying")
                self.reported = False
                return
            dt_ms = (message_stamp_to_sec(scan_msg) - semantic_stamp) * 1000.0

            semantic = image_msg_to_rgb(semantic_msg)
            height, width = semantic.shape[:2]
            points = pointcloud2_to_xyz(scan_msg)

            # Group returns by what they must have hit, from geometry alone.
            z = points[:, 2]
            horizontal = np.hypot(points[:, 0], points[:, 1])
            ranges = np.linalg.norm(points, axis=1)
            floor_hits = np.abs(z + SENSOR_HEIGHT_M) <= FLOOR_BAND_M
            wall_hits = (np.abs(z) <= WALL_BAND_M) & (ranges >= WALL_MIN_RANGE_M)

            self.get_logger().info(
                f"\nsemantic {width}x{height}, scan {len(points)} pts, semantic-scan dt={dt_ms:+.0f}ms\n"
                f"floor-height returns : {int(floor_hits.sum())} "
                f"(|z + {SENSOR_HEIGHT_M:.2f}| <= {FLOOR_BAND_M:.2f} m)\n"
                f"wall-height returns  : {int(wall_hits.sum())} "
                f"(|z| <= {WALL_BAND_M:.2f} m, range >= {WALL_MIN_RANGE_M:.1f} m)\n"
                "purity = fraction landing on that group's own most common semantic colour "
                "(1.00 = every point on one surface)"
            )

            if floor_hits.sum() < 20 or wall_hits.sum() < 20:
                self.get_logger().warning(
                    "too few floor/wall returns to judge - move the robot into open space and retry"
                )

            # ---------------------------------------------------------------
            # Primary, unbiased metric: floor/wall junction row per azimuth.
            #
            # The purity sweep below is biased - shrinking v_fov pushes every
            # point away from the horizon and therefore deeper into the large
            # floor region, so purity rises no matter what. This one compares
            # two independent measurements of the SAME edge in the SAME column,
            # so the panorama's distortion is exactly the unknown being fitted
            # and a wrong v_fov shows up as a real error.
            # ---------------------------------------------------------------
            floor_color = np.asarray(
                _modal_color(semantic[int(height * 0.85):, :]), dtype=np.int32
            )
            edges_semantic = floor_edge_rows(semantic, floor_color)
            edges_lidar = lidar_floor_edge(points, floor_hits, width)
            shared = sorted(set(edges_semantic) & set(edges_lidar))
            self.get_logger().info(
                f"\nfloor colour (bottom rows): RGB{tuple(int(c) for c in floor_color)}\n"
                f"columns with both a semantic floor edge and a LiDAR floor return: {len(shared)}"
            )
            if len(shared) >= 20:
                self.get_logger().info(
                    "--- junction-row error per vertical FOV (lower is better) ---"
                )
                edge_best = None
                for v_fov in V_FOV_CANDIDATES:
                    stack = np.array([edges_lidar[u] for u in shared])
                    projected = project(stack, width, height, v_fov)
                    if len(projected["v"]) != len(shared):
                        # A point can fall outside the image at small v_fov;
                        # compare only what survived, and say how many did.
                        pass
                    predicted = projected["v"]
                    observed = np.array([edges_semantic[u] for u in shared])[projected["indices"]]
                    error = predicted.astype(np.int64) - observed.astype(np.int64)
                    median_abs = float(np.median(np.abs(error)))
                    self.get_logger().info(
                        f"  v_fov={v_fov:6.1f}deg  median|err|={median_abs:6.1f}px  "
                        f"median err={float(np.median(error)):+7.1f}px  n={len(error)}"
                    )
                    if edge_best is None or median_abs < edge_best[0]:
                        edge_best = (median_abs, v_fov)
                self.get_logger().info(
                    f"  -> best fit v_fov={edge_best[1]:.0f}deg (median|err|={edge_best[0]:.1f}px). "
                    "A positive median error means projected points land BELOW the real junction."
                )

                # Same test under the 4-camera pinhole model.
                self.get_logger().info(
                    "--- same test, 4 stitched pinhole cameras instead of equirectangular ---"
                )
                stack = np.array([edges_lidar[u] for u in shared])
                observed_all = np.array([edges_semantic[u] for u in shared])
                pin_best = None
                for v_fov in V_FOV_CANDIDATES:
                    projected = project_pinhole_segments(stack, width, height, v_fov)
                    if not len(projected["indices"]):
                        continue
                    error = projected["v"].astype(np.int64) - observed_all[projected["indices"]]
                    median_abs = float(np.median(np.abs(error)))
                    self.get_logger().info(
                        f"  v_fov={v_fov:6.1f}deg  median|err|={median_abs:6.1f}px  n={len(error)}"
                    )
                    if pin_best is None or median_abs < pin_best[0]:
                        pin_best = (median_abs, v_fov)
                if pin_best is not None:
                    verdict = (
                        "equirectangular" if edge_best[0] <= pin_best[0] else "4x pinhole"
                    )
                    self.get_logger().info(
                        f"  -> best pinhole fit v_fov={pin_best[1]:.0f}deg "
                        f"(median|err|={pin_best[0]:.1f}px)\n"
                        f"  MODEL VERDICT: {verdict} fits better "
                        f"(equirect {edge_best[0]:.1f}px vs pinhole {pin_best[0]:.1f}px)"
                    )

                # Periodic error is the giveaway for a stitched panorama treated
                # as equirectangular: report error by position within a slice.
                projected = project(stack, width, height, edge_best[1])
                error = projected["v"].astype(np.int64) - observed_all[projected["indices"]]
                columns = np.array(shared)[projected["indices"]]
                slice_width = width // 4
                phase = columns % slice_width
                self.get_logger().info(
                    "--- error vs position inside a 480px slice (equirect, best v_fov) ---"
                )
                for low in range(0, slice_width, slice_width // 4):
                    high = low + slice_width // 4
                    selection = (phase >= low) & (phase < high)
                    if selection.sum():
                        self.get_logger().info(
                            f"  slice offset {low:3d}-{high:3d}px: median err "
                            f"{float(np.median(error[selection])):+6.1f}px (n={int(selection.sum())})"
                        )
                self.get_logger().info(
                    "  A U-shape (small error mid-slice, large at both edges) means the panorama "
                    "is stitched pinhole segments, not equirectangular."
                )
            else:
                self.get_logger().warning(
                    "not enough shared columns for the junction test - the robot may be too "
                    "close to furniture; move it into open space and retry"
                )

            best = None
            self.get_logger().info(
                "--- purity sweep (BIASED toward small v_fov - see comment above) ---"
            )
            for v_fov in V_FOV_CANDIDATES:
                floor = project(points[floor_hits], width, height, v_fov)
                wall = project(points[wall_hits], width, height, v_fov)
                floor_purity, floor_color, floor_n = purity(semantic, floor["u"], floor["v"])
                wall_purity, wall_color, wall_n = purity(semantic, wall["u"], wall["v"])
                combined = (floor_purity * floor_n + wall_purity * wall_n) / max(1, floor_n + wall_n)
                distinct = "same-surface!" if floor_color == wall_color else ""
                self.get_logger().info(
                    f"  v_fov={v_fov:6.1f}deg  floor purity={floor_purity:.2f} (n={floor_n}, "
                    f"RGB{floor_color})  wall purity={wall_purity:.2f} (n={wall_n}, RGB{wall_color})  "
                    f"combined={combined:.3f} {distinct}"
                )
                if best is None or combined > best[0]:
                    best = (combined, v_fov)

            best_v_fov = best[1]
            self.get_logger().info(f"--- pitch offset sweep at v_fov={best_v_fov:.0f}deg ---")
            best_pitch = None
            for pitch in PITCH_OFFSETS_DEG:
                floor = project(points[floor_hits], width, height, best_v_fov, pitch)
                wall = project(points[wall_hits], width, height, best_v_fov, pitch)
                floor_purity, _, floor_n = purity(semantic, floor["u"], floor["v"])
                wall_purity, _, wall_n = purity(semantic, wall["u"], wall["v"])
                combined = (floor_purity * floor_n + wall_purity * wall_n) / max(1, floor_n + wall_n)
                self.get_logger().info(
                    f"  pitch={pitch:+5.1f}deg  floor purity={floor_purity:.2f}  "
                    f"wall purity={wall_purity:.2f}  combined={combined:.3f}"
                )
                if best_pitch is None or combined > best_pitch[0]:
                    best_pitch = (combined, pitch)

            self.get_logger().info(
                f"\nbest: v_fov={best_v_fov:.0f}deg, pitch_offset={best_pitch[1]:+.1f}deg "
                f"(combined purity {best_pitch[0]:.3f})\n"
                f"production uses PANORAMA_V_FOV_DEG={config.PANORAMA_V_FOV_DEG:.0f}deg and "
                f"PANORAMA_PITCH_OFFSET_DEG={config.PANORAMA_PITCH_OFFSET_DEG:.1f}deg"
            )

            # Diagnostic image: floor hits over the semantic panorama at the
            # production setting. Correct alignment keeps every marker inside
            # the floor region.
            overlay = cv2.cvtColor(semantic, cv2.COLOR_RGB2BGR)
            floor = project(points[floor_hits], width, height, config.PANORAMA_V_FOV_DEG)
            for u, v in zip(floor["u"], floor["v"]):
                cv2.circle(overlay, (int(u), int(v)), 2, (255, 255, 255), -1)
            wall = project(points[wall_hits], width, height, config.PANORAMA_V_FOV_DEG)
            for u, v in zip(wall["u"], wall["v"]):
                cv2.circle(overlay, (int(u), int(v)), 2, (0, 0, 255), -1)
            cv2.putText(overlay, f"white = floor-height returns, red = wall-height returns (v_fov={config.PANORAMA_V_FOV_DEG:.0f})",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(overlay, f"white = floor-height returns, red = wall-height returns (v_fov={config.PANORAMA_V_FOV_DEG:.0f})",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            path = os.path.join(config.DEBUG_DIR, "semantic_alignment.jpg")
            cv2.imwrite(path, overlay)
            self.get_logger().info(f"diagnostic image -> {path}")

    rclpy.init()
    node = SemanticAlignmentCheck()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
