#!/usr/bin/env python3
"""
Lightweight SORT3D-style object-centric reasoning for tmah_vlm.

This package intentionally stays separate from the live ROS handlers. It adapts
SORT3D's useful pieces to the challenge stack without pulling in the full LLM,
captioning, or semantic-mapping runtime.
"""

from tmah_vlm.sort3d.objects import Sort3DObject
from tmah_vlm.sort3d.pipeline import Sort3DLite

__all__ = ["Sort3DObject", "Sort3DLite"]
