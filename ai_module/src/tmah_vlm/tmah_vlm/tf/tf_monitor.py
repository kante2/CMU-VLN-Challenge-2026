#!/usr/bin/env python3

import sys
import argparse
from collections import defaultdict

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.utilities import remove_ros_args

from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException


def clean_frame(frame: str) -> str:
    return frame.strip().lstrip("/")


class TFMonitor(Node):
    def __init__(self, args):
        super().__init__("tmah_tf_monitor")

        self.fixed_frame = clean_frame(args.fixed_frame)
        self.period = args.period
        self.once = args.once

        self.dynamic_tf = {}
        self.static_tf = {}

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        self.create_subscription(TFMessage, "/tf", self.tf_callback, 100)
        self.create_subscription(TFMessage, "/tf_static", self.tf_static_callback, 100)

        self.timer = self.create_timer(self.period, self.print_report)

        self.get_logger().info("TF monitor started.")
        self.get_logger().info(f"Fixed frame: {self.fixed_frame}")
        self.get_logger().info("Listening: /tf, /tf_static")

    def tf_callback(self, msg: TFMessage):
        for t in msg.transforms:
            parent = clean_frame(t.header.frame_id)
            child = clean_frame(t.child_frame_id)
            if parent and child:
                self.dynamic_tf[child] = t

    def tf_static_callback(self, msg: TFMessage):
        for t in msg.transforms:
            parent = clean_frame(t.header.frame_id)
            child = clean_frame(t.child_frame_id)
            if parent and child:
                self.static_tf[child] = t

    def get_all_transforms(self):
        merged = {}
        merged.update(self.static_tf)
        merged.update(self.dynamic_tf)
        return merged

    def get_all_frames(self):
        frames = set()
        for t in self.get_all_transforms().values():
            frames.add(clean_frame(t.header.frame_id))
            frames.add(clean_frame(t.child_frame_id))
        return sorted(frames)

    def print_report(self):
        transforms = self.get_all_transforms()
        frames = self.get_all_frames()

        if not transforms:
            self.get_logger().warn("아직 /tf 또는 /tf_static transform을 받지 못했습니다.")
            return

        tree = defaultdict(list)
        for child, t in transforms.items():
            parent = clean_frame(t.header.frame_id)
            source = "static" if child in self.static_tf else "dynamic"
            tree[parent].append((child, source))

        lines = []
        lines.append("")
        lines.append("========== TMAH TF REPORT ==========")
        lines.append(f"Dynamic TF count : {len(self.dynamic_tf)}")
        lines.append(f"Static TF count  : {len(self.static_tf)}")
        lines.append(f"Frame count      : {len(frames)}")
        lines.append("")
        lines.append("[Frames]")
        lines.append(", ".join(frames))
        lines.append("")
        lines.append("[TF Tree]")

        for parent in sorted(tree.keys()):
            lines.append(f"{parent}")
            for child, source in sorted(tree[parent]):
                lines.append(f"  └── {child} ({source})")

        lines.append("")
        lines.append(f"[Transform check: {self.fixed_frame} -> candidate frames]")

        candidate_frames = [
            "odom",
            "base_link",
            "base_footprint",
            "base",
            "body",
            "vehicle",
            "camera",
            "camera_link",
            "camera_frame",
            "lidar",
            "lidar_link",
            "velodyne",
            "sensor",
            "sensor_link",
        ]

        checked = False

        for frame in candidate_frames:
            frame = clean_frame(frame)

            if frame not in frames:
                continue

            if frame == self.fixed_frame:
                continue

            checked = True

            try:
                tf = self.buffer.lookup_transform(
                    self.fixed_frame,
                    frame,
                    Time()
                )

                tr = tf.transform.translation
                rot = tf.transform.rotation

                lines.append(
                    f"OK   {self.fixed_frame} -> {frame} | "
                    f"xyz=({tr.x:.3f}, {tr.y:.3f}, {tr.z:.3f}), "
                    f"quat=({rot.x:.3f}, {rot.y:.3f}, {rot.z:.3f}, {rot.w:.3f})"
                )

            except TransformException as e:
                lines.append(f"MISS {self.fixed_frame} -> {frame} | {str(e)}")

        if not checked:
            lines.append("candidate frame을 찾지 못했습니다. 위 [Frames]에서 실제 frame 이름을 확인하세요.")

        lines.append("====================================")

        self.get_logger().info("\n".join(lines))

        if self.once:
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fixed-frame",
        default="map",
        help="기준 frame. 예: map, odom, world"
    )

    parser.add_argument(
        "--period",
        type=float,
        default=3.0,
        help="출력 주기 sec"
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="한 번만 출력하고 종료"
    )

    non_ros_args = remove_ros_args(sys.argv)[1:]
    args, _ = parser.parse_known_args(non_ros_args)

    rclpy.init(args=sys.argv)
    node = TFMonitor(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
