# SysNav integration for CMU VLN Challenge 2026

This package keeps the challenge simulator/base autonomy unchanged and runs
SysNav inside the AI module. It connects the official challenge interface to
SysNav as follows:

| Challenge interface | SysNav integration |
|---|---|
| `/challenge_question` | adapter input |
| numerical question | existing TMAH numerical solver |
| object-reference question (`Find ...`) | SysNav search and semantic map |
| instruction-following question | SysNav VLM + TARE navigation |
| `/selected_object_marker` | generated from confirmed SysNav `ObjectNode` bbox |
| `/way_point` | TARE output consumed directly by base autonomy |

## Important limitations

SysNav's released VLM uses Gemini or DashScope through an online API. The
challenge test environment may not allow outbound network access. For an actual
offline submission, replace `vlm_node` with a local model before evaluation.

SysNav also expects two TensorRT detector engines which are not stored in Git.
Run the model preparation once on the target GPU before launching.

## One-time container preparation

The current `tmah_module` mounts the complete `ai_module/src` directory, so the
cloned `SysNav` and this package are visible at runtime. Install SysNav's system
and Python dependencies in a dedicated image (recommended), then run:

```bash
docker exec -it iros2026_tmah_module bash
bash /home/docker/ai_module/src/sysnav_challenge/scripts/build_in_container.sh
exit
docker exec -u root -it iros2026_tmah_module \
  bash /home/docker/ai_module/src/sysnav_challenge/scripts/prepare_models_in_container.sh
```

For a short integration test, create a Gemini API key in Google AI Studio and
set it before launch. Gemini is the default provider:

```bash
export GEMINI_API_KEY="..."
export VLM_PROVIDER=gemini  # optional; gemini is already the default
export GEMINI_MODEL=gemini-3.5-flash
export GEMINI_MODEL_LITE=gemini-3.5-flash
```

## Run

Start the official challenge simulator in the system container first. Then:

```bash
docker exec -it iros2026_tmah_module bash
source /opt/ros/jazzy/setup.bash
source /home/docker/ai_module/install/setup.bash
source /home/docker/ai_module/install_sysnav/setup.bash
ros2 launch sysnav_challenge sysnav_challenge.launch.py
```

For the challenge's single-room environment and semantic-map visualization:

```bash
ros2 launch sysnav_challenge sysnav_challenge.launch.py \
  single_room:=true use_rviz:=true use_legacy_solvers:=false
```

RViz opens in the `map` frame with `/terrain_map`, semantic object points
(`/obj_points`), 3D boxes (`/obj_boxes`), labels (`/obj_labels`), the single
room boundary, and the exploration path enabled. In single-room mode TARE marks
exploration complete instead of asking the VLM to select a nonexistent next
room.

For a safe node-only smoke test that does not send motion waypoints, add
`auto_start:=false`. The launch uses the challenge-specific object vocabulary
in `config/challenge_objects.yaml`; rerun model preparation after editing it.

Send a challenge question normally:

```bash
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Find teal pillow on the sofa farthest from the window'}"
```

Do not run `test_lidar_mapping.py`, `test_lidar_product_mapping.py`, or the old
`tmah_vlm.launch` at the same time. They can publish competing navigation goals.

Useful diagnostics:

```bash
ros2 topic hz /registered_scan
ros2 topic hz /terrain_map
ros2 topic hz /state_estimation
ros2 topic echo /way_point
ros2 topic echo /object_nodes_list
ros2 topic echo /selected_object_marker
```
