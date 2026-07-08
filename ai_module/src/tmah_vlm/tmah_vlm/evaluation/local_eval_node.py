#!/usr/bin/env python3
"""
Local evaluation harness for the CMU VLN challenge interface.

The official challenge_evaluation_node is not public. This node mirrors the
observable part of that interface:
  - publishes one challenge question at 1Hz on /challenge_question
  - listens for the expected response topic based on question type
  - records first-response latency and response payload
  - optionally writes a JSON report

It does not compute the hidden official score. Object-reference overlap and
instruction-following trajectory scoring require ground truth and evaluator
logic that are not released.
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import rclpy
from geometry_msgs.msg import Pose2D
from rclpy.node import Node
from std_msgs.msg import Int32, String
from visualization_msgs.msg import Marker


NUMERICAL_PREFIXES = ("how many", "count")
OBJECT_PREFIXES = ("find", "locate", "show me", "where is", "point to",
                   "identify")
INSTRUCTION_PREFIXES = ("go", "first", "take", "avoid", "pass", "move")


def classify_question(question: str) -> str:
    """Return numerical, object_reference, or instruction_following."""
    q = question.strip().lower()
    if q.startswith(NUMERICAL_PREFIXES):
        return "numerical"
    if q.startswith(OBJECT_PREFIXES):
        return "object_reference"
    if q.startswith(INSTRUCTION_PREFIXES):
        return "instruction_following"

    # Training examples include object references like:
    # "The blue chair that is closest to the cup of coffee."
    object_cues = (
        " closest ", " farthest ", " furthest ", " nearest ", " between ",
        " above ", " below ", " on ", " under ", " near "
    )
    if any(cue in f" {q} " for cue in object_cues):
        return "object_reference"
    return "instruction_following"


def marker_to_dict(msg: Marker) -> Dict[str, Any]:
    return {
        "frame_id": msg.header.frame_id,
        "ns": msg.ns,
        "id": msg.id,
        "type": msg.type,
        "position": {
            "x": msg.pose.position.x,
            "y": msg.pose.position.y,
            "z": msg.pose.position.z,
        },
        "orientation": {
            "x": msg.pose.orientation.x,
            "y": msg.pose.orientation.y,
            "z": msg.pose.orientation.z,
            "w": msg.pose.orientation.w,
        },
        "scale": {
            "x": msg.scale.x,
            "y": msg.scale.y,
            "z": msg.scale.z,
        },
    }


def waypoint_to_dict(msg: Pose2D) -> Dict[str, float]:
    return {"x": msg.x, "y": msg.y, "theta": msg.theta}


class LocalEvalNode(Node):
    def __init__(self, question: str, timeout_s: float, output: str):
        super().__init__("tmah_local_eval_node")
        self.question = question
        self.question_type = classify_question(question)
        self.timeout_s = timeout_s
        self.output = output
        self.start_time = time.monotonic()
        self.first_response_time: Optional[float] = None
        self.done = False
        self.report_written = False
        self.report_failed = False
        self.result: Dict[str, Any] = {
            "question": question,
            "question_type": self.question_type,
            "timeout_s": timeout_s,
            "official_score": None,
            "note": (
                "Official scoring is not reproduced here. This local harness "
                "only validates topics, message types, payloads, and timing."
            ),
            "responses": [],
        }

        self.question_pub = self.create_publisher(
            String, "/challenge_question", 5)
        self.create_subscription(
            Int32, "/numerical_response", self.numerical_cb, 5)
        self.create_subscription(
            Marker, "/selected_object_marker", self.marker_cb, 5)
        self.create_subscription(
            Pose2D, "/way_point_with_heading", self.waypoint_cb, 5)
        self.create_timer(1.0, self.publish_question)
        self.create_timer(0.2, self.check_timeout)

        self.get_logger().info(
            f"Local eval started: type={self.question_type}, "
            f"timeout={timeout_s:.1f}s")

    def publish_question(self):
        msg = String()
        msg.data = self.question
        self.question_pub.publish(msg)

    def _elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def _record(self, kind: str, payload: Dict[str, Any]):
        elapsed = self._elapsed()
        if self.first_response_time is None:
            self.first_response_time = elapsed
            self.result["first_response_latency_s"] = elapsed
        self.result["responses"].append({
            "kind": kind,
            "elapsed_s": elapsed,
            "payload": payload,
        })
        self.get_logger().info(
            f"response[{kind}] at {elapsed:.2f}s: {payload}")

    def numerical_cb(self, msg: Int32):
        if self.question_type != "numerical":
            return
        self._record("numerical", {"data": int(msg.data)})

    def marker_cb(self, msg: Marker):
        if self.question_type != "object_reference":
            return
        self._record("object_marker", marker_to_dict(msg))

    def waypoint_cb(self, msg: Pose2D):
        if self.question_type != "instruction_following":
            return
        payload = waypoint_to_dict(msg)
        if not all(math.isfinite(v) for v in payload.values()):
            payload["warning"] = "non-finite waypoint value"
        self._record("waypoint", payload)

    def check_timeout(self):
        if self._elapsed() < self.timeout_s:
            return
        self.result["total_elapsed_s"] = self._elapsed()
        self.result["timed_out"] = self.first_response_time is None
        self.write_report()
        self.done = True

    def write_report(self):
        if self.report_written or self.report_failed or not self.output:
            return
        path = Path(self.output).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self.result, indent=2, sort_keys=True),
                encoding="utf-8")
        except OSError as exc:
            self.result["report_error"] = str(exc)
            self.report_failed = True
            self.get_logger().error(
                f"failed to write local eval report '{path}': {exc}")
            return
        self.report_written = True
        self.get_logger().info(f"wrote local eval report: {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        required=True,
        help="Question to publish on /challenge_question.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Seconds to wait before writing the report and exiting.")
    parser.add_argument(
        "--output",
        default="/home/docker/ai_module/debug/local_eval_report.json",
        help="JSON report path. Use an empty string to disable writing.")
    return parser.parse_known_args()[0]


def main(args=None):
    cli_args = parse_args()
    rclpy.init(args=args)
    node = LocalEvalNode(
        question=cli_args.question,
        timeout_s=cli_args.timeout,
        output=cli_args.output,
    )
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        node.result["total_elapsed_s"] = node._elapsed()
        node.result["interrupted"] = True
        node.write_report()
    finally:
        if node.done and not node.report_written and not node.report_failed:
            node.write_report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
