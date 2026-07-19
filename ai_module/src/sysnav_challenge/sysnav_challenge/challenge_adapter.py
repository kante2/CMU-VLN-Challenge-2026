"""Bridge CMU VLN Challenge questions and answers to the SysNav ROS graph.

Routing:
  * numerical questions -> /sysnav_challenge/legacy_question (TMAH solver)
  * object reference    -> /keyboard_input (SysNav search), then convert the
                           confirmed SysNav object to /selected_object_marker
  * instruction         -> /keyboard_input (SysNav/TARE navigation)
"""

from __future__ import annotations

import threading

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker

from tare_planner.msg import ObjectNodeList, TargetObject


class ChallengeAdapter(Node):
    def __init__(self) -> None:
        super().__init__("sysnav_challenge_adapter")

        self.declare_parameter("question_topic", "/challenge_question") # topic 구독
        self.declare_parameter("sysnav_instruction_topic", "/keyboard_input")
        self.declare_parameter(
            "legacy_question_topic", "/sysnav_challenge/legacy_question"
        )

        question_topic = self.get_parameter("question_topic").value
        instruction_topic = self.get_parameter("sysnav_instruction_topic").value
        legacy_topic = self.get_parameter("legacy_question_topic").value

        self._lock = threading.Lock()
        self._objects = {}
        self._active_question = ""
        self._active_kind = ""
        self._last_question = ""

        self._instruction_pub = self.create_publisher(String, instruction_topic, 10)
        self._legacy_pub = self.create_publisher(String, legacy_topic, 10)
        self._marker_pub = self.create_publisher(
            Marker, "/selected_object_marker", 10
        )

        self.create_subscription(String, question_topic, self._question_callback, 10)
        self.create_subscription(
            ObjectNodeList, "/object_nodes_list", self._objects_callback, 20
        )
        self.create_subscription(
            TargetObject, "/target_object_answer", self._target_callback, 10
        )

        self.get_logger().info(
            f"SysNav challenge adapter ready: {question_topic} -> "
            f"{instruction_topic} / {legacy_topic}"
        )

    @staticmethod
    def _classify(question: str) -> str:
        first = question.strip().lower().split(maxsplit=1)[0] if question.strip() else ""
        if first in {"how", "count"}:
            return "numerical"
        if first == "find":
            return "object_reference"
        return "instruction"

    # question -> callback -> kind -> pub
    def _question_callback(self, msg: String) -> None:
        question = msg.data.strip()
        if not question or question == self._last_question:
            return

        self._last_question = question
        kind = self._classify(question)
        with self._lock:
            self._active_question = question
            self._active_kind = kind

        outgoing = String()
        outgoing.data = question
        if kind == "numerical":
            self._legacy_pub.publish(outgoing)
            self.get_logger().info(f"Numerical -> legacy TMAH: {question}")
            return

        self._instruction_pub.publish(outgoing)
        self.get_logger().info(f"{kind} -> SysNav: {question}")

    def _objects_callback(self, msg: ObjectNodeList) -> None:
        with self._lock:
            for obj in msg.nodes:
                for object_id in obj.object_id:
                    if obj.status:
                        self._objects[int(object_id)] = obj
                    else:
                        self._objects.pop(int(object_id), None)

    def _target_callback(self, msg: TargetObject) -> None:
        if not msg.is_target:
            return
        with self._lock:
            if self._active_kind != "object_reference":
                return
            obj = self._objects.get(int(msg.object_id))
        if obj is None:
            self.get_logger().warn(
                f"Target object {msg.object_id} confirmed before its map node arrived"
            )
            return
        self._publish_selected_marker(obj, msg.object_label)

    def _publish_selected_marker(self, obj, label: str) -> None:
        corners = list(obj.bbox3d)
        if not corners:
            self.get_logger().warn("Confirmed target has no 3D bounding box")
            return

        xs = [p.x for p in corners]
        ys = [p.y for p in corners]
        zs = [p.z for p in corners]

        marker = Marker()
        marker.header.frame_id = obj.header.frame_id or "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "sysnav_selected_object"
        marker.id = int(obj.object_id[0]) if obj.object_id else 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = (min(xs) + max(xs)) / 2.0
        marker.pose.position.y = (min(ys) + max(ys)) / 2.0
        marker.pose.position.z = (min(zs) + max(zs)) / 2.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = max(0.01, max(xs) - min(xs))
        marker.scale.y = max(0.01, max(ys) - min(ys))
        marker.scale.z = max(0.01, max(zs) - min(zs))
        marker.color.r = 0.1
        marker.color.g = 1.0
        marker.color.b = 0.1
        marker.color.a = 0.7
        marker.text = label or obj.label
        self._marker_pub.publish(marker)
        self.get_logger().info(
            f"Published selected object marker: id={marker.id}, label={marker.text}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ChallengeAdapter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
