# tmah_vlm Graph

This folder is a lightweight, online adaptation of the HOV-SG idea.

HOV-SG builds an offline hierarchical scene graph from posed RGB-D sequences:

```text
RGB-D + pose sequence
  -> SAM masks
  -> CLIP/OpenCLIP features
  -> 3D point cloud/object fusion
  -> building -> floor -> room -> object graph
```

For the challenge node, we already have a live perception pipeline:

```text
GroundingDINO detection
  -> selected 2D box
  -> LiDAR-backed 3D target/bbox
  -> marker + waypoint
```

So this module records successful live detections into the same hierarchy:

```text
building
  -> floor_0
    -> room_0_0
      -> object nodes
```

Current behavior:

- `runtime.py` is the adapter used by `handlers/object_reference.py`.
- `scene_graph.py` owns object insertion, nearby-object merging, query, and JSON export.
- `nodes.py` defines compact floor/room/object/observation metadata.
- `edges.py` computes object-object spatial relation edges (`near`, `left_of`,
  `right_of`, `in_front_of`, `behind`, `above`, `below`).
- `visualizer.py` publishes all graph objects as RViz `MarkerArray`.
- `scene_graph_latest.json` is saved under `config.DEBUG_DIR`.
- A HOV-SG-like folder layout is also saved under `config.DEBUG_DIR/scene_graph/graph`.
- RViz topic: `/scene_graph_markers`.

Intentional simplification:

- No SAM/OpenCLIP dependency is required in the live ROS loop.
- Floor and room segmentation are placeholders for now.
- Object query uses lexical matching now; CLIP embeddings can replace that scorer later.
