# Submission Notes — AI Module (`sysnav`)

This submission replaces the provided `dummy_vlm` with our module, **`sysnav`**.
Everything we changed lives under `ai_module/`. The `docker/` folder at the repository
root is byte-identical to the upstream challenge repository.

- Docker Hub image: https://hub.docker.com/r/kante2/cmu-vln-2026-sysnav
- Dependencies: `ai_module/requirements.txt`
- Source: `ai_module/src/sysnav_ros2_mvp/`

## Build

Exactly the standard procedure from `docker/README.md` — no changes needed:

```bash
xhost +
cd docker
docker compose -f compose_gpu.yml up --build -d
```

**An Nvidia GPU is required** — please use `compose_gpu.yml`, not `compose.yml`.
Our detection and segmentation models run on CUDA (`YOLO_DEVICE=0`, `SAM2_DEVICE=cuda`);
the CPU-only compose file reserves no GPU and the module will fail to start its
perception pipeline. `compose_gpu.yml` already passes the GPU to the `ai_module`
service, so no change to it is needed.

`ai_module/docker/Dockerfile` pulls our published image (which already contains
PyTorch, SAM2, Ultralytics, CLIP and all model weights), copies this repository's
`sysnav` sources over it and rebuilds that one ROS package. The build therefore takes
a few minutes rather than an hour, and the code in this repository always wins over
whatever is baked into the image.

## Run

**Terminal 1 — base autonomy system** (unchanged from `docker/README.md`):

```bash
docker exec -it iros2026_system bash
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
```

**Terminal 2 — our AI module.** Replace `<KEY>` with the Gemini API key we sent
separately (see *API key* below):

```bash
docker exec -e GEMINI_API_KEY=<KEY> -it iros2026_ai_module /home/docker/run_sysnav.sh
```

`/home/docker/run_sysnav.sh` sources ROS 2 and our workspace, then runs
`ros2 launch sysnav sysnav.launch.py`. It is a one-liner because `docker exec` does not
go through the image `ENTRYPOINT` and a non-interactive shell does not read `~/.bashrc`.

The node then waits for `/challenge_question` and answers on the standard topics.

## API key

`sysnav` calls the Gemini API at runtime (allowed per challenge FAQ #3). We are **not**
shipping the key in this public repository or in the public image; it is provided
separately with the submission.

**The key must be present in the environment before the node process starts** — the
`-e GEMINI_API_KEY=<KEY>` in the command above. Injecting it after the node has started
has no effect, because the clients read the variable once during construction.

Without the key the node still runs, explores and publishes waypoints, but every
LLM-backed step fails, so please make sure it is set.

## Topics

Subscribed (all from the base autonomy system, per *System Outputs*):

| Topic | Type |
|-|-|
| `/challenge_question` | `std_msgs/String` |
| `/state_estimation` | `nav_msgs/Odometry` |
| `/camera/image` | `sensor_msgs/Image` |
| `/sensor_scan` | `sensor_msgs/PointCloud2` |
| `/terrain_map` | `sensor_msgs/PointCloud2` |

Published (per *System Inputs*):

| Topic | Type | Used for |
|-|-|-|
| `/way_point_with_heading` | `geometry_msgs/Pose2D` | navigation, all question types |
| `/selected_object_marker` | `visualization_msgs/Marker` | object-reference questions |
| `/numerical_response` | `std_msgs/Int32` | numerical questions |

## Note on DDS

The image pins `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`, matching the base image's own
default (`~/.bashrc` in `zhangjicmu/ubuntu24_ros` exports the same). We pin it as an
`ENV` because `~/.bashrc` is only read by interactive shells, so a module launched via
`bash -c` would silently fall back to Fast DDS.

That matters here: both containers run with `network_mode: host` but private IPC
namespaces, so their `/dev/shm` are separate. Fast DDS then selects shared-memory
transport between them, discovery succeeds but no data is ever delivered — topics show
up in `ros2 topic list` while the module receives zero messages and simply waits.

If the system container is started in a way that leaves it on Fast DDS, please launch
our module with `-e RMW_IMPLEMENTATION=rmw_fastrtps_cpp` so both sides match. To check
which one the system container is using:

```bash
docker exec iros2026_system bash -c 'ls /dev/shm | grep -c fastrtps'   # 0 => CycloneDDS
```
