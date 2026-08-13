"""Overlay LiDAR points on the panorama image to inspect camera-LiDAR alignment.

The 3D position of every detected object comes from LiDAR points that fall
inside a SAM2 mask, so a projection that is off by even a few degrees puts
objects in the wrong place - and nothing downstream can detect that. This tool
makes the alignment visible: it grabs a time-synced image + scan pair the same
way perception does, projects the scan through the *production* code path
(PanoramaLidarGrounder._project), and writes an overlay.

What to look for in the overlay:

  * points must trace the wall/floor and wall/ceiling boundaries. A band that
    sits consistently above or below those lines is a vertical (pitch) error -
    correct with config.PANORAMA_PITCH_OFFSET_DEG.
  * points must sit on the objects they belong to. A constant left/right shift
    is a yaw error - correct with config.PANORAMA_YAW_OFFSET_DEG.
  * a shift that only appears while the robot is turning is not calibration but
    synchronization: the image and the scan were captured at different times.
    Compare the stamp delta printed for each capture.

Usage, with the simulator running (the real sysnav node may stay up - this only
subscribes):

    docker exec -it iros2026_sysnav_module bash -lc \
      "source /home/docker/ai_module/install/setup.bash && \
       python3 /home/docker/ai_module/src/sysnav_ros2_mvp/tests/check_camera_lidar_alignment.py"

Writes config.DEBUG_DIR/alignment_overlay_<n>.jpg (default 3 captures, a few
seconds apart) plus a printed geometry report.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from sysnav import config
from sysnav.perception.lidar_grounding import PanoramaLidarGrounder
from sysnav.ros_helpers import (
    closest_stamped_item,
    image_msg_to_rgb,
    message_stamp_to_sec,
    odometry_to_pose,
    pointcloud2_to_xyz,
)

CAPTURES = int(os.getenv("ALIGNMENT_CAPTURES", "3"))
CAPTURE_INTERVAL_SEC = 3.0
WAIT_TIMEOUT_SEC = 30.0


def describe_geometry(width: int, height: int) -> str:
    """Report the panorama geometry the projection implicitly assumes.

    _project() maps yaw over 2*pi across the width and the down-angle over pi
    across the height, i.e. it hardcodes a 360 x 180 degree panorama with the
    robot's forward direction at the horizontal centre. There is no config knob
    for the vertical FOV here (tmah_vlm has PANO_V_FOV_DEG for exactly this),
    so if the camera ever changes, this is the assumption that silently breaks.
    """
    uniform_v_fov = 360.0 * height / width if width else float("nan")
    return (
        f"image                     : {width} x {height} (aspect {width / height:.2f})\n"
        f"projection uses           : h_fov=360deg (yaw/2pi * width), "
        f"v_fov={config.PANORAMA_V_FOV_DEG:.0f}deg, forward at u={width / 2:.0f}\n"
        f"uniform-equirect v_fov    : {uniform_v_fov:.0f}deg "
        "(what the aspect ratio implies - equal px/deg on both axes)\n"
        f"  -> these agree, which is the expected calibration. The old hardcoded 180deg\n"
        f"     put projected points a median of 26px too high; measured with\n"
        f"     tests/check_pano_alignment_semantic.py against the simulator's semantic\n"
        f"     panorama (see the config.PANORAMA_V_FOV_DEG comment).\n"
        f"yaw offset                : {config.PANORAMA_YAW_OFFSET_DEG:.1f}deg\n"
        f"pitch offset              : {config.PANORAMA_PITCH_OFFSET_DEG:.1f}deg\n"
        f"lidar->camera translation : {config.T_LIDAR_TO_CAMERA[0][3]:.2f}, "
        f"{config.T_LIDAR_TO_CAMERA[1][3]:.2f}, {config.T_LIDAR_TO_CAMERA[2][3]:.2f} "
        "(camera frame: x right, y forward, z down)"
    )


def main() -> None:
    try:
        import cv2
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Image, PointCloud2
    except ImportError as error:  # pragma: no cover
        raise SystemExit(f"run this inside the sysnav container ({error})")

    class AlignmentChecker(Node):
        def __init__(self) -> None:
            super().__init__("sysnav_alignment_check")
            self.grounder = PanoramaLidarGrounder()
            self.latest_image = None
            self.scan_buffer: list[tuple[float, object]] = []
            self.pose_buffer: list[tuple[float, dict]] = []
            self.captures = 0
            self.last_capture = 0.0
            self.started = time.monotonic()
            self.reported_geometry = False

            self.create_subscription(Image, config.TOPIC_IMAGE, self._on_image, qos_profile_sensor_data)
            self.create_subscription(PointCloud2, config.TOPIC_SCAN, self._on_scan, qos_profile_sensor_data)
            self.create_subscription(Odometry, config.TOPIC_STATE, self._on_state, qos_profile_sensor_data)
            self.create_timer(0.2, self._tick)
            self.get_logger().info(
                f"alignment check started - will save {CAPTURES} overlay(s) to {config.DEBUG_DIR}"
            )

        def _on_image(self, msg) -> None:
            self.latest_image = msg

        def _on_scan(self, msg) -> None:
            self.scan_buffer.append((message_stamp_to_sec(msg), msg))
            if len(self.scan_buffer) > config.SCAN_BUFFER_SIZE:
                del self.scan_buffer[: -config.SCAN_BUFFER_SIZE]

        def _on_state(self, msg) -> None:
            pose = odometry_to_pose(msg)
            self.pose_buffer.append((pose["stamp"], pose))
            if len(self.pose_buffer) > config.POSE_BUFFER_SIZE:
                del self.pose_buffer[: -config.POSE_BUFFER_SIZE]

        def _tick(self) -> None:
            now = time.monotonic()
            if self.captures >= CAPTURES:
                raise SystemExit(0)
            if self.latest_image is None or not self.scan_buffer:
                if now - self.started > WAIT_TIMEOUT_SEC:
                    self.get_logger().error(
                        f"no {config.TOPIC_IMAGE} / {config.TOPIC_SCAN} data in "
                        f"{WAIT_TIMEOUT_SEC:.0f}s - is the simulator running?"
                    )
                    raise SystemExit(1)
                return
            if now - self.last_capture < CAPTURE_INTERVAL_SEC:
                return
            self.last_capture = now
            self._capture(cv2)

        def _capture(self, cv2) -> None:
            # Same pairing rule the perception pipeline uses: the image stamp is
            # the reference, the scan and pose are matched to it.
            image_msg = self.latest_image
            image_stamp = message_stamp_to_sec(image_msg)
            scan_msg = closest_stamped_item(
                list(self.scan_buffer), image_stamp, config.SENSOR_SYNC_TOLERANCE_SEC
            )
            if scan_msg is None:
                self.get_logger().warning(
                    "no scan within SENSOR_SYNC_TOLERANCE_SEC of the image - skipping"
                )
                return
            pose = closest_stamped_item(
                list(self.pose_buffer), image_stamp, config.SENSOR_SYNC_TOLERANCE_SEC
            )
            scan_stamp = message_stamp_to_sec(scan_msg)

            image_rgb = image_msg_to_rgb(image_msg)
            height, width = image_rgb.shape[:2]
            if not self.reported_geometry:
                self.reported_geometry = True
                self.get_logger().info("\n" + describe_geometry(width, height))

            points_sensor = pointcloud2_to_xyz(scan_msg)
            projection = self.grounder._project(points_sensor, (height, width))
            kept = projection["indices"]
            overlay = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if len(kept):
                points = points_sensor[kept]
                ranges = np.linalg.norm(points, axis=1)
                # Colour by range so the geometry is readable: near = red,
                # far = blue. A correct projection paints walls at a consistent
                # distance and follows their edges.
                normalized = np.clip(
                    (ranges - config.GROUNDING_MIN_RANGE_M)
                    / max(1e-6, config.GROUNDING_MAX_RANGE_M - config.GROUNDING_MIN_RANGE_M),
                    0.0, 1.0,
                )
                colors = (cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                          .reshape(-1, 3))
                for (u, v, color) in zip(projection["u"], projection["v"], colors):
                    cv2.circle(overlay, (int(u), int(v)), 1, tuple(int(c) for c in color), -1)

            # Horizon line: where the projection puts zero elevation. Points on
            # a flat wall at camera height should straddle it.
            cv2.line(overlay, (0, height // 2), (width, height // 2), (0, 255, 255), 1)
            cv2.line(overlay, (width // 2, 0), (width // 2, height), (0, 255, 255), 1)
            stamp_delta_ms = (scan_stamp - image_stamp) * 1000.0
            yaw_text = "?" if pose is None else f"{math.degrees(pose['yaw']):.0f}deg"
            label = (
                f"projected {len(kept)}/{len(points_sensor)} pts  "
                f"image-scan dt={stamp_delta_ms:+.0f}ms  yaw={yaw_text}  "
                f"(yellow: horizon v={height // 2}, forward u={width // 2})"
            )
            cv2.putText(overlay, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(overlay, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            self.captures += 1
            path = os.path.join(config.DEBUG_DIR, f"alignment_overlay_{self.captures}.jpg")
            cv2.imwrite(path, overlay)
            self.get_logger().info(
                f"[{self.captures}/{CAPTURES}] {path}  "
                f"projected={len(kept)}/{len(points_sensor)}  dt={stamp_delta_ms:+.0f}ms  yaw={yaw_text}"
            )

    rclpy.init()
    node = AlignmentChecker()
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
