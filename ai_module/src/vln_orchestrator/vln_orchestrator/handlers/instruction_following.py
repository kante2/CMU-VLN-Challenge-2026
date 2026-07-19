#!/usr/bin/env python3
"""Instruction-following handler (/6) — drive an ordered, constraint-respecting path.

Wired so far:
  1. Parse the command into ORDERED sub-goals + via/avoid constraints
     (reasoning.instruction_parser — pure logic, validated on all 30 training cmds).

Wired so far (perception-gated):
  2. Localise each ordered landmark in the semantic map (SemanticMap.locate).
  3. Stream the resulting (x, y) sequence to /way_point_with_heading via the
     node's arrival-advancing streamer (reach ~1.0 m; cf. pubPathWaypoints).

Still pending (needs SysNav route_planner on the GPU box):
  - obstacle-aware path planning between landmarks and honouring avoid/via
    regions; we currently send straight-line landmark waypoints in order.

Scored 0-6 with partial credit; penalties for wrong order / missed / forbidden.
"""
from __future__ import annotations

from vln_orchestrator.handlers.base import BaseHandler
from vln_orchestrator.reasoning.instruction_parser import GoalKind, parse_instruction
from vln_orchestrator.reasoning.route import build_route, locate_region


class InstructionFollowingHandler(BaseHandler):
    def handle(self, question: str) -> None:
        parsed = parse_instruction(question)
        order = " -> ".join(
            f"{g.kind.value}:{g.landmark}" for g in parsed.subgoals
        )
        self.log.info(f"parsed instruction: {order}")
        if parsed.avoid_regions:
            self.log.info(
                "avoid: " + "; ".join(g.landmark for g in parsed.avoid_regions)
            )

        sm = getattr(self.node, "semantic_map", None)
        if sm is not None and len(sm):
            waypoints: list[tuple[float, float]] = []
            for g in parsed.ordered_waypoints:          # GOTO/VIA/STOP, in order
                # VIA points are regions ("between the two columns") -> centroid;
                # GOTO/STOP are single objects -> resolve.
                if g.kind == GoalKind.VIA:
                    pt = locate_region(sm, g.landmark)
                else:
                    inst = sm.locate(g.landmark)
                    pt = (inst.bbox["cx"], inst.bbox["cy"]) if inst else None
                if pt is not None:
                    waypoints.append(pt)
                    self.log.info(f"  located {g.kind.value}:{g.landmark!r} -> "
                                  f"({pt[0]:.2f},{pt[1]:.2f})")
                else:
                    self.log.warn(f"  could not locate {g.landmark!r}; skipping")
            if waypoints:
                # bend the ordered path around AVOID regions (base autonomy handles
                # fine obstacle avoidance between the waypoints we emit).
                avoid_centers = []
                for g in parsed.avoid_regions:
                    c = locate_region(sm, g.landmark)
                    if c is not None:
                        avoid_centers.append(c)
                route = build_route(waypoints, avoid_centers)
                if len(route) > len(waypoints):
                    self.log.info(f"  detoured around {len(avoid_centers)} avoid region(s)")
                self.node.stream_waypoints(route)
                return
            self.log.warn("InstructionFollowingHandler: no landmarks located; fallback.")
        else:
            self.log.warn("InstructionFollowingHandler: no semantic map; fallback.")
        self.fallback(question)

    def fallback(self, question: str) -> None:
        # Hold position (publish current pose) so the topic is exercised and the
        # robot does not wander. Real pipeline emits the planned ordered sequence.
        self.node.publish_waypoint(self.node.vehicle_x, self.node.vehicle_y, 0.0)
