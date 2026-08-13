"""Score 3D object grounding against the scene's ground-truth object list.

Fixing the projection (config.PANORAMA_V_FOV_DEG) moved the wall/floor junction
error from 26px to 7px, but that is a projection metric. What actually matters is
whether objects land in the right place in the map, and this measures exactly
that: run the real perception path once on a live frame, then compare each
grounded position against `<scene>/object_list.txt` from the scene zip - the
simulator's own object inventory (id, centre xyz, size, yaw, label).

Because the only thing that changes between runs is the projection constant, the
same command run twice gives a direct before/after:

    # inside the sysnav container
    PANORAMA_V_FOV_DEG=180 python3 tests/check_object_grounding.py loft
    PANORAMA_V_FOV_DEG=120 python3 tests/check_object_grounding.py loft

Prerequisites: the simulator is running, the real sysnav node is NOT (it would
compete for the GPU), and map/<scene>.zip is reachable. The scene zip lives on
the host, so pass --object-list if it is not mounted in this container; the
default path also accepts the copy under /home/docker/maps.

Detection prompts come from the ground-truth labels themselves, so a miss is a
real miss rather than a vocabulary mismatch. Matching is nearest-GT-of-the-same-
label within --max-match-distance; unmatched detections are reported separately
rather than silently dropped, since a projection error shows up as exactly that.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import zipfile

import numpy as np

from sysnav import config
from sysnav.ros_helpers import (
    closest_stamped_item,
    image_msg_to_rgb,
    message_stamp_to_sec,
    odometry_to_pose,
    pointcloud2_to_xyz,
)

DEFAULT_MAP_DIRS = ["/home/docker/maps", os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "map")]
# Labels worth scoring: furniture-scale things YOLO-World can actually see. Tiny
# fixtures (light switch, focus light) and structural surfaces (wall, floor) are
# excluded - they are either invisible at this resolution or not objects.
SKIP_LABELS = {"wall", "floor", "ceiling", "light switch", "focus light", "door frame"}
MAX_PROMPTS = 12


def load_ground_truth(object_list_path: str | None, scene: str) -> list[dict]:
    """Parse object_list.txt (from the scene zip or a loose file)."""
    text = None
    if object_list_path and os.path.exists(object_list_path):
        if object_list_path.endswith(".zip"):
            with zipfile.ZipFile(object_list_path) as archive:
                names = [n for n in archive.namelist() if n.endswith("object_list.txt")]
                if not names:
                    raise SystemExit(f"{object_list_path} has no object_list.txt")
                text = archive.read(names[0]).decode("utf-8", errors="replace")
        else:
            text = open(object_list_path, encoding="utf-8", errors="replace").read()
    else:
        for directory in DEFAULT_MAP_DIRS:
            candidate = os.path.join(directory, f"{scene}.zip")
            if os.path.exists(candidate):
                with zipfile.ZipFile(candidate) as archive:
                    names = [n for n in archive.namelist() if n.endswith("object_list.txt")]
                    if names:
                        text = archive.read(names[0]).decode("utf-8", errors="replace")
                break
    if text is None:
        raise SystemExit(
            f"could not find object_list.txt for scene '{scene}'. Looked for "
            f"{[os.path.join(d, scene + '.zip') for d in DEFAULT_MAP_DIRS]}; "
            "pass --object-list explicitly."
        )

    objects = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # id x y z length width height yaw "label"
        label_start = line.find('"')
        if label_start == -1:
            continue
        label = line[label_start:].strip().strip('"').lower()
        numbers = line[:label_start].split()
        if len(numbers) < 8:
            continue
        try:
            values = [float(v) for v in numbers[:8]]
        except ValueError:
            continue
        objects.append({
            "object_id": int(values[0]),
            "position": np.array(values[1:4], dtype=np.float64),
            "size": np.array(values[4:7], dtype=np.float64),
            "yaw": values[7],
            "label": label,
        })
    return objects


def choose_prompts(ground_truth: list[dict]) -> list[str]:
    counts: dict[str, int] = {}
    for item in ground_truth:
        if item["label"] in SKIP_LABELS:
            continue
        counts[item["label"]] = counts.get(item["label"], 0) + 1
    ordered = sorted(counts.items(), key=lambda pair: -pair[1])
    return [label for label, _ in ordered[:MAX_PROMPTS]]


def score(detections: list[dict], ground_truth: list[dict], max_distance: float) -> dict:
    """Match each detection to the nearest same-label GT object."""
    used: set[int] = set()
    matched: list[dict] = []
    unmatched: list[dict] = []
    for detection in detections:
        candidates = [
            item for item in ground_truth
            if item["label"] == detection["category"] and item["object_id"] not in used
        ]
        if not candidates:
            unmatched.append(detection)
            continue
        position = np.asarray(detection["position"], dtype=np.float64)
        distances = [float(np.linalg.norm(position - item["position"])) for item in candidates]
        best = int(np.argmin(distances))
        if distances[best] > max_distance:
            unmatched.append(detection)
            continue
        used.add(candidates[best]["object_id"])
        horizontal = float(np.linalg.norm(position[:2] - candidates[best]["position"][:2]))
        matched.append({
            "detection": detection,
            "gt": candidates[best],
            "distance": distances[best],
            "horizontal": horizontal,
            "vertical": float(position[2] - candidates[best]["position"][2]),
        })
    return {"matched": matched, "unmatched": unmatched}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene", help="scene name, e.g. loft")
    parser.add_argument("--object-list", default=None, help="path to object_list.txt or the scene zip")
    parser.add_argument("--max-match-distance", type=float, default=2.0,
                        help="a detection farther than this from any same-label GT counts as unmatched (m)")
    parser.add_argument("--frames", type=int, default=1, help="how many snapshots to accumulate")
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.object_list, args.scene)
    prompts = choose_prompts(ground_truth)
    print(f"scene                : {args.scene}")
    print(f"ground-truth objects : {len(ground_truth)} ({len(prompts)} labels prompted)")
    print(f"prompts              : {prompts}")
    print(f"PANORAMA_V_FOV_DEG   : {config.PANORAMA_V_FOV_DEG}  <- the variable under test")
    print()

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Image, PointCloud2

        from sysnav.perception.detector import YoloWorldDetector
        from sysnav.perception.lidar_grounding import PanoramaLidarGrounder
        from sysnav.perception.segmenter import Sam2Segmenter
    except ImportError as error:  # pragma: no cover
        raise SystemExit(f"run this inside the sysnav container ({error})")

    class GroundingCheck(Node):
        def __init__(self) -> None:
            super().__init__("sysnav_grounding_check")
            self.detector = YoloWorldDetector()
            self.segmenter = Sam2Segmenter()
            self.grounder = PanoramaLidarGrounder()
            self.latest_image = None
            self.scan_buffer: list[tuple[float, object]] = []
            self.pose_buffer: list[tuple[float, dict]] = []
            self.detections: list[dict] = []
            self.frames_done = 0
            self.started = time.monotonic()
            self.create_subscription(Image, config.TOPIC_IMAGE, self._on_image, qos_profile_sensor_data)
            self.create_subscription(PointCloud2, config.TOPIC_SCAN, self._on_scan, qos_profile_sensor_data)
            self.create_subscription(Odometry, config.TOPIC_STATE, self._on_state, qos_profile_sensor_data)
            self.create_timer(0.5, self._tick)
            self.get_logger().info("loading models and waiting for a synced frame ...")

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
            if self.frames_done >= args.frames:
                raise SystemExit(0)
            if self.latest_image is None or not self.scan_buffer or not self.pose_buffer:
                if time.monotonic() - self.started > 40.0:
                    self.get_logger().error("no sensor data - is the simulator running?")
                    raise SystemExit(1)
                return
            image_msg = self.latest_image
            image_stamp = message_stamp_to_sec(image_msg)
            scan_msg = closest_stamped_item(list(self.scan_buffer), image_stamp,
                                            config.SENSOR_SYNC_TOLERANCE_SEC)
            pose = closest_stamped_item(list(self.pose_buffer), image_stamp,
                                        config.SENSOR_SYNC_TOLERANCE_SEC)
            if scan_msg is None or pose is None:
                return

            image_rgb = image_msg_to_rgb(image_msg)
            detections = self.detector.detect(image_rgb, prompts)
            segmented = self.segmenter.segment(image_rgb, detections)
            grounded = self.grounder.ground(image_rgb, pointcloud2_to_xyz(scan_msg), segmented, dict(pose))
            self.frames_done += 1
            self.get_logger().info(
                f"frame {self.frames_done}/{args.frames}: "
                f"{len(detections)} detections -> {len(segmented)} masks -> {len(grounded)} grounded"
            )
            self.detections.extend(grounded)

    rclpy.init()
    node = GroundingCheck()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        detections = list(node.detections)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if not detections:
        print("no grounded detections - cannot score")
        sys.exit(1)

    result = score(detections, ground_truth, args.max_match_distance)
    matched, unmatched = result["matched"], result["unmatched"]
    print()
    print(f"grounded detections  : {len(detections)}")
    print(f"matched to GT        : {len(matched)}   unmatched: {len(unmatched)}")
    if matched:
        total = np.array([m["distance"] for m in matched])
        horizontal = np.array([m["horizontal"] for m in matched])
        vertical = np.array([m["vertical"] for m in matched])
        print()
        print(f"position error (3D)  : median {np.median(total):.2f} m   mean {total.mean():.2f} m   "
              f"max {total.max():.2f} m")
        print(f"  horizontal (xy)    : median {np.median(horizontal):.2f} m")
        print(f"  vertical (z)       : median {np.median(vertical):+.2f} m   "
              f"mean {vertical.mean():+.2f} m  <- a systematic sign here means the projection "
              "is still tilted")
        within = [float((total <= t).mean()) for t in (0.3, 0.5, 1.0)]
        print(f"  within 0.3/0.5/1.0m: {within[0]:.0%} / {within[1]:.0%} / {within[2]:.0%}")
        print()
        print("per-detection (worst first):")
        for m in sorted(matched, key=lambda item: -item["distance"])[:12]:
            det, gt = m["detection"], m["gt"]
            print(f"  {det['category']:<14} conf={det.get('confidence', 0):.2f} "
                  f"pts={det.get('num_points', 0):<4} "
                  f"got=({det['position'][0]:6.2f},{det['position'][1]:6.2f},{det['position'][2]:6.2f}) "
                  f"gt=({gt['position'][0]:6.2f},{gt['position'][1]:6.2f},{gt['position'][2]:6.2f}) "
                  f"err={m['distance']:.2f}m (dz={m['vertical']:+.2f})")
    if unmatched:
        print()
        print(f"unmatched detections (no same-label GT within {args.max_match_distance:.1f} m):")
        for det in unmatched[:8]:
            print(f"  {det['category']:<14} at ({det['position'][0]:6.2f},"
                  f"{det['position'][1]:6.2f},{det['position'][2]:6.2f})")


if __name__ == "__main__":
    main()
