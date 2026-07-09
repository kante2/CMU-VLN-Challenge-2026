#!/usr/bin/env python3
"""
Lightweight HOV-SG style scene graph for tmah_vlm.

This package keeps the graph code separate from the live ROS pipeline. The
current implementation records object-level observations from GroundingDINO +
LiDAR grounding, then stores them under a simple hierarchy:

    building -> floor -> room -> object

Floor and room segmentation can be replaced later with stronger HOV-SG style
logic without changing the object-reference handler.
"""

from tmah_vlm.graph.scene_graph import SceneGraph

__all__ = ["SceneGraph"]
