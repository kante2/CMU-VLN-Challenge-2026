# SORT3D-Lite For TMAH VLM

This folder is a local, lightweight adaptation of the SORT3D idea. It does not
run the full SORT3D ROS system or any external LLM. Instead it provides the data
shape and deterministic tools we need inside `tmah_vlm`.

Pipeline:

1. Load object instances from the online `node.scene_graph`.
   `object_list.txt` loading exists only for local debugging with known maps.
2. Attach cheap rule-based captions such as `a small paper cup on or above the table`.
3. Filter objects relevant to an instruction.
4. Run spatial tools: `find_near`, `find_between`, `find_above`, `find_below`,
   `find_left`, `find_right`, `closest_to`, `furthest_from`.
5. Convert the selected object to a navigation waypoint with `go_near` or
   `go_between`.

Example:

```python
from tmah_vlm.sort3d import Sort3DLite

sort3d = Sort3DLite.from_object_list(
    "/home/docker/ai_module/Navigation-Physical-Experiment/src/base_autonomy/"
    "vehicle_simulator/mesh/unity/object_list.txt"
)
selection = sort3d.select_target("Find the potted plant on the file cabinet.")
waypoint = sort3d.action_for_selection(selection, {"x": 0.0, "y": 0.0})
```

The live handler now uses this module only from the online scene graph:

- relation-heavy `find ...` queries try graph reasoning first
- if GroundingDINO returns no detections, graph reasoning is used as a fallback
- no ground-truth `object_list.txt` is required for hidden evaluation
