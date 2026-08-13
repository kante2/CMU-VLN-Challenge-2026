"""Save LiDAR-on-panorama overlays at several vertical FOVs to settle which one is right.

Symptom this exists for: projected LiDAR only covers a band across the middle of
the panorama, while the real scan clearly returns floor points that ought to show
up lower in the image.

Both readings are self-consistent, which is why eyeballing one overlay cannot
decide it. With the scan measured in this sim (elevation -32deg .. +57deg):

    v_fov=180  ->  points span v 121..447, and the empty area below v~447
                   corresponds to floor closer than ~1.2 m, which the LiDAR
                   genuinely cannot reach. Nothing is wrong.
    v_fov=120  ->  the same -32deg lands at v~493, so the lower band should be
                   populated, and the current projection is squeezing everything
                   toward the middle.

The discriminator is the image itself: a floor point at horizontal distance d
sits 0.75 m below the camera, so only the correct v_fov puts projected points
exactly on the wall/floor junction and along the tile lines. This tool renders
the same scan at each candidate v_fov so the correct one can be read off.

Usage, with the simulator running:

    docker exec -it iros2026_sysnav_module bash -lc \
      "source /home/docker/ai_module/install/setup.bash && \
       python3 /home/docker/ai_module/src/sysnav_ros2_mvp/tests/check_pano_projection_vfov.py"

Writes config.DEBUG_DIR/pano_vfov_<value>.jpg per candidate, plus
pano_vfov_compare.jpg stacking them, and prints the geometry each implies.

How to read the output:
  * Correct v_fov: points land ON the wall/floor junction line and follow the
    floor tile pattern outward; the wall band tracks the wall/ceiling edge.
  * Too large (points squeezed toward the horizon): the lower rows stay empty
    even though the scan has downward returns.
  * Too small (points spread past the real geometry): floor points fall below
    the actual floor line, or wall points climb over the ceiling edge.

The projection here is intentionally a local copy with a v_fov parameter -
production _project() now reads config.PANORAMA_V_FOV_DEG (measured 120deg); this
local copy keeps the parameter free so the value can be re-verified in other
scenes and robot positions.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from sysnav import config
from sysnav.ros_helpers import (
    closest_stamped_item,
    image_msg_to_rgb,
    message_stamp_to_sec,
    pointcloud2_to_xyz,
)

# 180 is what production assumes today; 120 is what the 3:1 aspect ratio would
# imply for a uniform equirectangular image.
V_FOV_CANDIDATES = [float(v) for v in os.getenv("PANO_V_FOV_LIST", "90,120,150,180").split(",")]
CAMERA_HEIGHT_ABOVE_FLOOR_M = 0.75  # sensor height in this stack (map->sensor z)


def project(points_sensor: np.ndarray, width: int, height: int, v_fov_deg: float) -> dict:
    """Same math as PanoramaLidarGrounder._project, with v_fov exposed.

    Camera frame after T_LIDAR_TO_CAMERA: x right, y forward, z down.
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
        return {"indices": indices, "u": np.empty(0, int), "v": np.empty(0, int),
                "ranges": np.empty(0), "elevation_deg": np.empty(0)}

    x, y, z, horizontal, ranges = x[valid], y[valid], z[valid], horizontal[valid], ranges[valid]
    yaw = np.arctan2(x, y) + math.radians(config.PANORAMA_YAW_OFFSET_DEG)
    down = np.arctan2(z, horizontal) + math.radians(config.PANORAMA_PITCH_OFFSET_DEG)
    yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi

    u = np.floor(((yaw / (2.0 * math.pi)) + 0.5) * width).astype(np.int32) % width
    # The only difference from production: pi becomes the chosen vertical FOV.
    v = np.floor(((down / math.radians(v_fov_deg)) + 0.5) * height).astype(np.int32)
    inside = (v >= 0) & (v < height)
    indices, u, v, ranges, down = indices[inside], u[inside], v[inside], ranges[inside], down[inside]

    pixel_id = v.astype(np.int64) * width + u.astype(np.int64)
    order = np.argsort(ranges)
    _, first = np.unique(pixel_id[order], return_index=True)
    keep = order[first]
    return {
        "indices": indices[keep],
        "u": u[keep],
        "v": v[keep],
        "ranges": ranges[keep],
        "elevation_deg": -np.degrees(down[keep]),  # positive = upward
    }


def floor_row(v_fov_deg: float, height: int, distance_m: float) -> int:
    """Image row where a floor point `distance_m` away should land."""
    down = math.atan2(CAMERA_HEIGHT_ABOVE_FLOOR_M, distance_m)
    return int(round(((down / math.radians(v_fov_deg)) + 0.5) * height))


def bottom_row_distance(v_fov_deg: float) -> float:
    """Floor distance the very bottom row corresponds to, per this v_fov."""
    down = math.radians(v_fov_deg) / 2.0
    if down >= math.pi / 2 - 1e-6:
        return 0.0
    return CAMERA_HEIGHT_ABOVE_FLOOR_M / math.tan(down)


def main() -> None:
    try:
        import cv2
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, PointCloud2
    except ImportError as error:  # pragma: no cover
        raise SystemExit(f"run this inside the sysnav container ({error})")

    class VfovSweep(Node):
        def __init__(self) -> None:
            super().__init__("sysnav_pano_vfov_sweep")
            self.latest_image = None
            self.scan_buffer: list[tuple[float, object]] = []
            self.started = time.monotonic()
            self.saved = False
            self.create_subscription(Image, config.TOPIC_IMAGE, self._on_image, qos_profile_sensor_data)
            self.create_subscription(PointCloud2, config.TOPIC_SCAN, self._on_scan, qos_profile_sensor_data)
            self.create_timer(0.2, lambda: self._tick(cv2))
            self.get_logger().info(
                f"vertical-FOV sweep over {V_FOV_CANDIDATES} -> {config.DEBUG_DIR}"
            )

        def _on_image(self, msg) -> None:
            self.latest_image = msg

        def _on_scan(self, msg) -> None:
            self.scan_buffer.append((message_stamp_to_sec(msg), msg))
            if len(self.scan_buffer) > config.SCAN_BUFFER_SIZE:
                del self.scan_buffer[: -config.SCAN_BUFFER_SIZE]

        def _tick(self, cv2) -> None:
            if self.saved:
                raise SystemExit(0)
            if self.latest_image is None or not self.scan_buffer:
                if time.monotonic() - self.started > 30.0:
                    self.get_logger().error("no image/scan in 30s - is the simulator running?")
                    raise SystemExit(1)
                return
            # Let a couple of frames arrive so image and scan are both fresh.
            if time.monotonic() - self.started < 2.0:
                return
            self.saved = True
            self._render(cv2)

        def _render(self, cv2) -> None:
            image_msg = self.latest_image
            image_stamp = message_stamp_to_sec(image_msg)
            scan_msg = closest_stamped_item(
                list(self.scan_buffer), image_stamp, config.SENSOR_SYNC_TOLERANCE_SEC
            )
            if scan_msg is None:
                self.get_logger().warning("no scan close enough to the image - retrying")
                self.saved = False
                return

            image_rgb = image_msg_to_rgb(image_msg)
            height, width = image_rgb.shape[:2]
            points = pointcloud2_to_xyz(scan_msg)
            base = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            horiz = np.hypot(points[:, 0], points[:, 1])
            rng = np.linalg.norm(points, axis=1)
            ok = (rng > config.GROUNDING_MIN_RANGE_M) & (horiz > 1e-6)
            elevation = np.degrees(np.arctan2(points[ok, 2], horiz[ok]))
            self.get_logger().info(
                f"\nimage {width}x{height}, scan {len(points)} pts\n"
                f"scan elevation      : {elevation.min():+.1f} .. {elevation.max():+.1f} deg "
                f"(median {np.median(elevation):+.1f})\n"
                f"camera height       : {CAMERA_HEIGHT_ABOVE_FLOOR_M:.2f} m above floor"
            )

            panels = []
            for v_fov in V_FOV_CANDIDATES:
                overlay = base.copy()
                result = project(points, width, height, v_fov)
                # Colour by elevation: warm = looking up, cool = looking down.
                if len(result["indices"]):
                    elev = result["elevation_deg"]
                    normalized = np.clip((elev + 90.0) / 180.0, 0.0, 1.0)
                    colors = cv2.applyColorMap(
                        (normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
                    ).reshape(-1, 3)
                    for u, v, color in zip(result["u"], result["v"], colors):
                        cv2.circle(overlay, (int(u), int(v)), 1, tuple(int(c) for c in color), -1)

                # Reference rows: where floor at 1/2/4 m must appear if this
                # v_fov is correct. Points should sit on these, and the real
                # wall/floor junction in the image should agree.
                cv2.line(overlay, (0, height // 2), (width, height // 2), (0, 255, 255), 1)
                for distance, color in ((1.0, (0, 0, 255)), (2.0, (0, 165, 255)), (4.0, (0, 255, 0))):
                    row = floor_row(v_fov, height, distance)
                    if 0 <= row < height:
                        cv2.line(overlay, (0, row), (width, row), color, 1)
                        cv2.putText(overlay, f"floor {distance:.0f}m", (8, max(12, row - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                v_min = int(result["v"].min()) if len(result["v"]) else -1
                v_max = int(result["v"].max()) if len(result["v"]) else -1
                label = (
                    f"v_fov={v_fov:.0f}deg  points v {v_min}..{v_max} of {height}  "
                    f"bottom row = floor at {bottom_row_distance(v_fov):.2f}m  "
                    f"(yellow=horizon)"
                )
                cv2.putText(overlay, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(overlay, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

                path = os.path.join(config.DEBUG_DIR, f"pano_vfov_{int(v_fov)}.jpg")
                cv2.imwrite(path, overlay)
                panels.append(overlay)
                self.get_logger().info(
                    f"  v_fov={v_fov:5.0f}deg -> v {v_min}..{v_max}, "
                    f"floor@1m row {floor_row(v_fov, height, 1.0)}, "
                    f"floor@2m row {floor_row(v_fov, height, 2.0)}, "
                    f"bottom row = floor at {bottom_row_distance(v_fov):.2f}m   {path}"
                )

            if panels:
                compare = np.vstack(panels)
                # Keep the stacked image manageable for viewing.
                scale = min(1.0, 1600.0 / compare.shape[1])
                if scale < 1.0:
                    compare = cv2.resize(compare, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                path = os.path.join(config.DEBUG_DIR, "pano_vfov_compare.jpg")
                cv2.imwrite(path, compare)
                self.get_logger().info(
                    f"stacked comparison -> {path}\n"
                    "Pick the panel where points sit on the wall/floor junction and follow the "
                    "tile lines; that is the true vertical FOV."
                )

    rclpy.init()
    node = VfovSweep()
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
