#!/usr/bin/env python3
"""VLN Orchestrator node — replaces dummy_vlm.

Implements the challenge I/O contract (see ai_module/src/dummy_vlm/src/dummyVLM.cpp):

  Subscribes:
    /challenge_question   std_msgs/String     - the question to answer (1 Hz)
    /state_estimation     nav_msgs/Odometry   - vehicle pose

  Publishes:
    /numerical_response       std_msgs/Int32                  (Numerical /1)
    /selected_object_marker   visualization_msgs/Marker CUBE  (Object Ref /2)
    /way_point_with_heading   geometry_msgs/Pose2D            (Instruction /6)

The node classifies each incoming question with question_router.classify() and
dispatches to one of three handlers. Handlers run the perception/reasoning
pipeline (semantic mapping + VLM, adapted from SysNav) and call back into the
node's publish_* helpers. Per challenge rules the system is relaunched per
question, so there is no cross-question state to manage here.

NOTE: handler internals are scaffolded (safe fallback answers + TODO hooks).
Perception wiring lands once the SysNav semantic_mapping / vlm_node modules are
vendored into ai_module on the Jazzy box.
"""
from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from std_msgs.msg import String, Int32, Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2

from vln_orchestrator.question_router import QType, classify
from vln_orchestrator.handlers.numerical import NumericalHandler
from vln_orchestrator.handlers.object_reference import ObjectReferenceHandler
from vln_orchestrator.handlers.instruction_following import InstructionFollowingHandler
from vln_orchestrator.exploration.explorer import ExplorationController


class VLNOrchestrator(Node):
    def __init__(self) -> None:
        super().__init__("vln_orchestrator")

        # --- subscriptions ---
        self.create_subscription(String, "/challenge_question", self._on_question, 5)
        self.create_subscription(Odometry, "/state_estimation", self._on_pose, 5)

        # --- publishers (exact topics/types from dummyVLM.cpp) ---
        self._numerical_pub = self.create_publisher(Int32, "/numerical_response", 5)
        self._marker_pub = self.create_publisher(Marker, "/selected_object_marker", 5)
        self._waypoint_pub = self.create_publisher(Pose2D, "/way_point_with_heading", 5)

        # Bridge the challenge question into SysNav's exploration coordinator
        # (vlm_node reads the instruction on /keyboard_input, decomposes it, and
        # tells TARE the target so exploration is target-driven).
        self._keyboard_pub = self.create_publisher(String, "/keyboard_input", 5)

        # --- vehicle state ---
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0

        # --- waypoint streaming (instruction-following path + explore goals) ---
        self.REACH_DIST = float(self.declare_parameter("waypoint_reach_dist", 1.0).value)
        self._wp_queue: list[tuple[float, float]] = []

        # --- perception (semantic map), wired only if SysNav is built ---
        self.semantic_map = None
        self._wire_perception()

        # --- handlers ---
        self._handlers = {
            QType.NUMERICAL: NumericalHandler(self),
            QType.OBJECT_REFERENCE: ObjectReferenceHandler(self),
            QType.INSTRUCTION_FOLLOWING: InstructionFollowingHandler(self),
        }

        # --- exploration: drive the scene to build the map BEFORE answering ---
        # 10-min/question budget; leave margin to answer + earn the time bonus.
        self._answer_deadline_s = float(self.declare_parameter("answer_deadline_s", 540.0).value)
        self._explore_budget_s = float(self.declare_parameter("explore_budget_s", 420.0).value)
        self._convergence_timeout_s = float(self.declare_parameter("convergence_timeout_s", 20.0).value)
        self._min_explore_s = float(self.declare_parameter("min_explore_s", 15.0).value)
        # abandon a frontier we can't reach within this long, mark it blocked, re-route
        self._goal_timeout_s = float(self.declare_parameter("goal_timeout_s", 12.0).value)
        # external_exploration=True: something else drives the robot (e.g. TARE in
        # the system). We then only wait for the map to build and answer; we do NOT
        # publish exploration waypoints (avoids fighting TARE's /way_point).
        self._external_exploration = bool(self.declare_parameter("external_exploration", False).value)
        # In external mode TARE's exploration_finish (std_msgs/Bool) is the
        # AUTHORITATIVE "done" signal. Map-stability alone is unreliable here: a
        # stalled or detection-starved map looks "stable" and triggered a wrong
        # early answer. So we wait for this signal; map-stability is only a long
        # backstop in case the signal never arrives (then we still answer in time).
        self._exploration_finish_topic = str(
            self.declare_parameter("exploration_finish_topic", "exploration_finish").value)
        self._stability_backstop_s = float(
            self.declare_parameter("stability_backstop_s", 180.0).value)
        self._exploration_finished = False
        self._terrain_subsample = int(self.declare_parameter("terrain_subsample", 5).value)
        self._explorer = ExplorationController(
            frontier_clearance=float(self.declare_parameter("frontier_clearance", 2.0).value),
            max_step=float(self.declare_parameter("explore_step", 4.0).value),
        )
        self.create_subscription(PointCloud2, "/terrain_map", self._on_terrain, 5)
        self.create_subscription(
            Bool, self._exploration_finish_topic, self._on_exploration_finish, 5)
        self._phase = "idle"          # idle -> explore -> done
        self._question = None
        self._qtype = None
        self._explore_start = 0.0
        self._goal_time = 0.0         # when the current explore goal was issued
        self._keyboard_sent_until = 0.0  # re-publish instruction to vlm_node until
        self._last_obj_count = 0
        self._last_growth_s = 0.0
        self.create_timer(
            float(self.declare_parameter("explore_tick_s", 1.0).value),
            self._explore_tick,
        )

        # one_shot=True (eval-faithful): handle one question per launch; the eval
        # relaunches per question and re-publishes it at 1 Hz. one_shot:=false for
        # dev to handle each NEW question in one session.
        self.one_shot = bool(self.declare_parameter("one_shot", True).value)
        self._answered = False
        self._last_question = None  # dedupe 1 Hz repeats of the same question
        self.get_logger().info(
            f"VLN Orchestrator ready (one_shot={self.one_shot}); "
            "awaiting /challenge_question..."
        )

    # ------------------------------------------------------------------ #
    # Perception wiring
    # ------------------------------------------------------------------ #
    def _wire_perception(self) -> None:
        """Subscribe to SysNav's semantic map IFF its message type is available.

        The ObjectNodeList message ships with SysNav's tare_planner build (on the
        GPU box). When it isn't present (e.g. the current minimal container), we
        skip the subscription and the handlers run their fallbacks — so the node
        works in both worlds without code changes.
        """
        try:
            from tare_planner.msg import ObjectNodeList
        except Exception as e:
            self.get_logger().warn(
                f"perception: tare_planner/ObjectNodeList unavailable ({e}); "
                "running in fallback mode (no semantic map)."
            )
            return
        from vln_orchestrator.perception.semantic_map_adapter import SemanticMap
        self.semantic_map = SemanticMap()
        self.create_subscription(
            ObjectNodeList, "/object_nodes_list", self._on_semantic_map, 50
        )
        self.get_logger().info("perception: subscribed to /object_nodes_list.")

    def _on_semantic_map(self, msg) -> None:
        if self.semantic_map is not None:
            self.semantic_map.update_from_msg(msg)

    def _on_exploration_finish(self, msg: Bool) -> None:
        """TARE signals the scene is fully explored. Authoritative convergence in
        external mode (latches once true)."""
        if msg.data and not self._exploration_finished:
            self._exploration_finished = True
            self.get_logger().info("exploration_finish received from TARE.")

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def _on_pose(self, msg: Odometry) -> None:
        self.vehicle_x = msg.pose.pose.position.x
        self.vehicle_y = msg.pose.pose.position.y
        self._advance_waypoints()

    def _on_question(self, msg: String) -> None:
        question = msg.data.strip()
        if not question:
            return
        if question == self._last_question:   # ignore 1 Hz repeats of the same Q
            return
        if self.one_shot and self._answered:   # eval-faithful: one Q per launch
            return
        self._last_question = question
        self._answered = True            # accept this question; ignore repeats
        self._question = question
        self._qtype = classify(question)
        now = self._now_s()
        self._explore_start = now
        self._last_growth_s = now
        self._phase = "explore"          # drive the scene; answer in _explore_tick
        # feed the instruction to the exploration coordinator (vlm_node) so TARE
        # explores toward the question's target. Republished by _explore_tick for
        # a few seconds in case vlm_node subscribed late.
        self._keyboard_pub.publish(String(data=question))
        # brief re-publish window in case vlm_node subscribed late; kept short
        # because vlm_node re-decomposes (a VLM call) on every repeat.
        self._keyboard_sent_until = now + 2.0
        self.get_logger().info(
            f"Question [{self._qtype.value}]: {question} -> exploring"
        )

    # ------------------------------------------------------------------ #
    # Exploration phase machine
    # ------------------------------------------------------------------ #
    def _on_terrain(self, msg: PointCloud2) -> None:
        """Latest traversable area -> explorer frontiers (subsampled for speed)."""
        try:
            from sensor_msgs_py import point_cloud2
        except Exception:
            return
        pts = []
        for i, p in enumerate(point_cloud2.read_points(
                msg, field_names=("x", "y"), skip_nans=True)):
            if i % self._terrain_subsample == 0:
                pts.append((float(p[0]), float(p[1])))
        self._explorer.set_terrain(pts)

    def _explore_tick(self) -> None:
        if self._phase != "explore":
            return
        now = self._now_s()
        # re-publish the instruction to the coordinator for a few seconds (in case
        # vlm_node subscribed after the first send)
        if self._question is not None and now < self._keyboard_sent_until:
            self._keyboard_pub.publish(String(data=self._question))
        self._explorer.mark_visited(self.vehicle_x, self.vehicle_y)

        # track semantic-map growth (for the convergence criterion)
        if self.semantic_map is not None:
            n = len(self.semantic_map)
            if n > self._last_obj_count:
                self._last_obj_count = n
                self._last_growth_s = now

        elapsed = now - self._explore_start
        if elapsed >= self._answer_deadline_s:
            self._do_answer("deadline", elapsed); return
        if elapsed >= self._explore_budget_s:
            self._do_answer("budget", elapsed); return
        stable = (now - self._last_growth_s) >= self._convergence_timeout_s
        # external explorer (TARE) drives. Its exploration_finish signal is the
        # authoritative "done" marker; map-stability alone is NOT (a stalled or
        # detection-starved map looks "stable" -> answered a partial scene early).
        # Use stability only as a long backstop so we still answer within budget if
        # the signal never arrives. When WE drive, require coverage as before.
        if self._external_exploration:
            backstop = stable and elapsed >= self._stability_backstop_s
            if self._exploration_finished:
                reason = "tare_finished"
            elif backstop:
                reason = "stability_backstop"
            else:
                reason = ""
            converged = bool(reason)
        else:
            converged = stable and self._explorer.is_covered(self.vehicle_x, self.vehicle_y)
            reason = "converged"
        if converged and elapsed >= self._min_explore_s:
            self._do_answer(reason, elapsed); return

        # external explorer (e.g. TARE) drives the robot; we just wait + answer
        if self._external_exploration:
            return

        # if we've sat on the current goal too long, it's unreachable -> mark it
        # blocked and drop it so we re-route to a different frontier
        if self._wp_queue and (now - self._goal_time) >= self._goal_timeout_s:
            gx, gy = self._wp_queue[0]
            self._explorer.mark_blocked(gx, gy)
            self.get_logger().info(f"frontier ({gx:.1f},{gy:.1f}) unreached; re-routing")
            self._wp_queue = []

        # when no active goal, stream the next frontier (reuses the arrival-advance
        # machinery in _on_pose)
        if not self._wp_queue:
            goal = self._explorer.next_goal(self.vehicle_x, self.vehicle_y)
            if goal is not None:
                self.stream_waypoints([goal])
                self._goal_time = now

    def _do_answer(self, reason: str, elapsed: float) -> None:
        self._phase = "done"
        self._wp_queue = []   # stop exploration driving before the handler acts
        self.get_logger().info(
            f"exploration {reason} after {elapsed:.0f}s "
            f"({self._last_obj_count} objects mapped); answering"
        )
        try:
            self._handlers[self._qtype].handle(self._question)
        except Exception as e:  # never crash; always emit something
            self.get_logger().error(f"handler {self._qtype.value} failed: {e}")
            self._handlers[self._qtype].fallback(self._question)

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    # ------------------------------------------------------------------ #
    # Publish helpers (used by handlers)
    # ------------------------------------------------------------------ #
    def publish_numerical(self, value: int) -> None:
        self._numerical_pub.publish(Int32(data=int(value)))
        self.get_logger().info(f"-> /numerical_response {int(value)}")

    def publish_object_marker(self, bbox: dict, label: str, obj_id: int = 0) -> None:
        """bbox: {cx,cy,cz,l,w,h,heading} in the map frame (see dummyVLM CUBE)."""
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.now_msg()
        m.ns = label
        m.id = int(obj_id)
        m.action = Marker.ADD
        m.type = Marker.CUBE
        m.pose.position.x = float(bbox["cx"])
        m.pose.position.y = float(bbox["cy"])
        m.pose.position.z = float(bbox["cz"])
        heading = float(bbox.get("heading", 0.0))
        m.pose.orientation.z = math.sin(heading / 2.0)
        m.pose.orientation.w = math.cos(heading / 2.0)
        m.scale.x = float(bbox["l"])
        m.scale.y = float(bbox["w"])
        m.scale.z = float(bbox["h"])
        m.color.a = 0.5
        m.color.b = 1.0
        self._marker_pub.publish(m)
        self.get_logger().info(
            f"-> /selected_object_marker '{label}' @ "
            f"({bbox['cx']:.2f},{bbox['cy']:.2f},{bbox['cz']:.2f})"
        )

    def publish_waypoint(self, x: float, y: float, heading: float = 0.0) -> None:
        self._waypoint_pub.publish(Pose2D(x=float(x), y=float(y), theta=float(heading)))

    def stream_waypoints(self, points: list) -> None:
        """Queue an ordered list of (x, y) waypoints and publish the first; the
        rest are advanced in _on_pose as the vehicle reaches each (cf. the
        sequential pubPathWaypoints behaviour of dummyVLM)."""
        self._wp_queue = [(float(x), float(y)) for x, y in points]
        if self._wp_queue:
            x, y = self._wp_queue[0]
            self.publish_waypoint(x, y, self._heading_towards(x, y))
            self.get_logger().info(f"streaming {len(self._wp_queue)} waypoint(s)")

    def _advance_waypoints(self) -> None:
        if not self._wp_queue:
            return
        tx, ty = self._wp_queue[0]
        if math.hypot(tx - self.vehicle_x, ty - self.vehicle_y) <= self.REACH_DIST:
            self._wp_queue.pop(0)
            if self._wp_queue:
                nx, ny = self._wp_queue[0]
                self.publish_waypoint(nx, ny, self._heading_towards(nx, ny))
            else:
                self.get_logger().info("waypoint sequence complete.")

    def _heading_towards(self, x: float, y: float) -> float:
        return math.atan2(y - self.vehicle_y, x - self.vehicle_x)

    def now_msg(self):
        return self.get_clock().now().to_msg()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VLNOrchestrator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as e:
        # On SIGINT, rclpy's executor can raise mid-spin while taking a message
        # (a known shutdown race -> RuntimeError "Unable to convert call
        # argument"). Per-question handler errors are already caught in
        # _on_question, so spin-level exceptions are shutdown/transport races:
        # log and exit cleanly rather than dying with a traceback (exit 1).
        node.get_logger().info(f"shutting down ({type(e).__name__})")
    finally:
        node.destroy_node()
        # The signal handler may have already shut the context down; guard so a
        # second shutdown() doesn't raise. Exit clean (0).
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
