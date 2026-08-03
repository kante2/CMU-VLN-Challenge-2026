#!/usr/bin/env python3
"""
Runtime adapter between object_reference.py and the scene graph.

Keeping this as a tiny adapter means the live perception pipeline only needs to
call record_object_observation(...). Graph construction policy remains isolated
inside tmah_vlm/graph.
"""

import os

from tmah_vlm import config
from tmah_vlm.graph.nodes import ObjectObservation
from tmah_vlm.graph.render_image import render_scene_graph
from tmah_vlm.graph.scene_graph import SceneGraph
from tmah_vlm.graph.visualizer import publish_scene_graph_markers


def stamp_to_sec(stamp):
    if stamp is None:
        return 0.0
    return float(getattr(stamp, "sec", 0.0)) + float(getattr(stamp, "nanosec", 0.0)) * 1e-9


def get_scene_graph(node):
    if not hasattr(node, "scene_graph") or node.scene_graph is None:
        node.scene_graph = SceneGraph(
            merge_distance_m=float(getattr(config, "SCENE_GRAPH_MERGE_DISTANCE_M", 0.75))
        )
    return node.scene_graph


def get_vlm_captioner(node):
    if not getattr(config, "ENABLE_VLM_CAPTIONER", False):
        return None
    if hasattr(node, "vlm_captioner_failed") and node.vlm_captioner_failed:
        return None
    if hasattr(node, "vlm_captioner") and node.vlm_captioner is not None:
        return node.vlm_captioner

    try:
        from tmah_vlm.sort3d.caption.vlm_captioner import FlorenceCaptioner
        node.vlm_captioner = FlorenceCaptioner(
            model_id=getattr(config, "CAPTION_MODEL_ID", "microsoft/Florence-2-base"),
            device=getattr(config, "CAPTION_DEVICE", "cpu"),
            max_new_tokens=getattr(config, "CAPTION_MAX_NEW_TOKENS", 64),
        )
        node.get_logger().info(
            f"[Caption] VLM captioner loaded: {getattr(config, 'CAPTION_MODEL_ID', '')} "
            f"on {getattr(config, 'CAPTION_DEVICE', 'cpu')}"
        )
        return node.vlm_captioner
    except Exception as error:
        node.vlm_captioner_failed = True
        node.get_logger().warn(f"[Caption] VLM captioner unavailable, use rule captions: {error}")
        return None


def make_observation(question, detection, result, image_stamp=None):
    point = tuple(result["point"])
    bbox_center = tuple(result.get("bbox_center") or point)
    bbox_size = tuple(
        result.get("bbox_size") or
        (float(getattr(config, "BBOX3D_DEFAULT_SIZE_M", 0.4)),) * 3
    )

    return ObjectObservation(
        label=str(getattr(detection, "label", result.get("target_name", "object"))),
        score=float(getattr(detection, "score", 0.0)),
        question=str(question),
        box_2d=tuple(float(v) for v in getattr(detection, "box", (0.0, 0.0, 0.0, 0.0))),
        point=point,
        bbox_center=bbox_center,
        bbox_size=bbox_size,
        method=str(result.get("method", "")),
        matched_points=int(result.get("n_matched", 0)),
        stamp_sec=stamp_to_sec(image_stamp),
    )


def maybe_update_vlm_caption(node, object_node, detection, image):
    if image is None:
        return
    if object_node.caption_source == "vlm":
        return

    captioner = get_vlm_captioner(node)
    if captioner is None:
        return

    try:
        caption = captioner.caption_crop(
            image,
            getattr(detection, "box", (0.0, 0.0, image.width, image.height)),
            margin_px=getattr(config, "CAPTION_CROP_MARGIN_PX", 16),
        )
        if caption:
            object_node.caption = caption
            object_node.caption_source = "vlm"
            node.get_logger().info(
                f"[Caption] object={object_node.object_id}, caption={caption}"
            )
    except Exception as error:
        node.get_logger().warn(f"[Caption] VLM caption failed: {error}")


def record_object_observation(node, question, detection, result, image_stamp=None, image=None):
    """
    Add one grounded object to the online HOV-SG style graph and save snapshots.

    Returns the updated ObjectNode, or None if graph recording fails. Failures
    are logged but never allowed to break navigation.
    """
    log = node.get_logger()

    try:
        graph = get_scene_graph(node)
        observation = make_observation(question, detection, result, image_stamp)
        object_node = graph.add_observation(observation)
        maybe_update_vlm_caption(node, object_node, detection, image)

        latest_path = os.path.join(config.DEBUG_DIR, "scene_graph_latest.json")
        graph.save_json(latest_path)
        image_path = os.path.join(config.DEBUG_DIR, "scene_graph_latest.jpg")
        render_scene_graph(latest_path, image_path)

        layout_dir = os.path.join(config.DEBUG_DIR, "scene_graph")
        graph.save_hovsg_layout(layout_dir)
        publish_scene_graph_markers(node)

        log.info(
            f"[SceneGraph] object={object_node.object_id}, "
            f"name={object_node.name}, observations={len(object_node.observations)}, "
            f"saved={latest_path}, image={image_path}"
        )
        return object_node
    except Exception as error:
        log.warn(f"[SceneGraph] update failed: {error}")
        return None
