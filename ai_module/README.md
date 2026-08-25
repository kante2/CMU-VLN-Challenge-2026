# SysNav Run Guide (Docker Hub Submission Image)

CMU VLN Challenge 2026 — run the setup in three terminals: **A** (simulator),
**B** (sysnav node), **C** (queries).

This guide covers running the submitted image straight from Docker Hub. Every command
below was executed end-to-end on the submission image before publishing this document.

> **An Nvidia GPU is required.** Use `compose_gpu.yml`, not `compose.yml` — detection and
> segmentation run on CUDA (`YOLO_DEVICE=0`, `SAM2_DEVICE=cuda`).

---

## 1. Docker Hub submission image

- Repository: https://hub.docker.com/r/kante2/cmu-vln-2026-sysnav
- **Image: `kante2/cmu-vln-2026-sysnav:submission-v3`**

Everything needed at runtime is baked into the image — **no downloads happen during
evaluation**:

| Path (under `/home/docker/ai_module/`) | Contents |
|-|-|
| `weights/yolov8x-worldv2.pt` | YOLO-World detector (140 MB) |
| `weights/yolo12s.pt` | YOLO12 detector (19 MB) |
| `weights/sam2.1_hiera_tiny.pt` | SAM2 segmenter (149 MB) |
| `weights/clip/ViT-B-32.pt` | CLIP text encoder for open-vocabulary prompts (338 MB) |
| `install/sysnav` | colcon build output |

`USER=docker`, `WORKDIR=/home/docker/ai_module`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`.

You do not need to pull manually — step 2 pulls the image as part of the build — but you
can pre-fetch it:

```bash
docker pull kante2/cmu-vln-2026-sysnav:submission-v3
```

---

## 2. Build and start the containers

The standard procedure from `docker/README.md`, unchanged:

```bash
cd ~/CMU-VLN-Challenge-2026/docker
xhost +
docker compose -f compose_gpu.yml up --build -d
```

This starts `iros2026_system` (simulator + autonomy stack) and `iros2026_ai_module`
(our module). `ai_module/docker/Dockerfile` pulls the image above, copies this
repository's `sysnav` sources over it and rebuilds that one ROS package, so the build
takes a few minutes and **the code in this repository always wins over the image**.

If the containers already exist and are stopped:

```bash
docker start iros2026_system iros2026_ai_module
```

---

## 3. Terminal A — simulator

```bash
docker exec -it iros2026_system bash
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
```

---

## 4. Terminal B — sysnav node

Replace `<KEY>` with the Gemini API key provided separately with this submission:

```bash
docker exec -e GEMINI_API_KEY=<KEY> -it iros2026_ai_module /home/docker/run_sysnav.sh
```

`run_sysnav.sh` sources ROS 2 and our workspace, then runs
`ros2 launch sysnav sysnav.launch.py`. It exists because `docker exec` does not go
through the image `ENTRYPOINT`, and a non-interactive shell never reads `~/.bashrc`.

**The key must be in the environment before the node starts** — hence `-e`. The Gemini
clients read the variable once during construction, so injecting it afterwards has no
effect. Without it the node still explores and publishes waypoints, but every
LLM-backed step fails.

No `colcon build` is needed here; the build already happened in step 2.

On startup you should see:

```
[sysnav_node]: SysNav frontier-coverage planner started
```

---

## 5. Terminal C — queries

```bash
docker exec -it iros2026_ai_module bash
source /opt/ros/jazzy/setup.bash
ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: 'Find the toilet'}"
```

The mission type is inferred from the sentence (`sysnav/task/mission_classifier.py`).
Terminal B should then show the Gemini parse, the Ultralytics banner and the
perception pipeline:

```
[sysnav_llm_query_parser]: LLM parse: target=toilet, attributes=[], relation_chain=[]
[sysnav_node]: 📩 NEW QUESTION [object_reference] - Task #1: "Find the toilet"
Ultralytics 8.4.118 🚀 Python-3.12.3 torch-2.13.0+cu130 CUDA:0
[sysnav_perception]: [Perception] YOLO-World detected: toilet=0.39[yolo_world]
[sysnav_perception]: [Perception] SAM2 segmented 1/1 detections
[sysnav_perception]: [Perception] LiDAR-grounded to 3D: [('toilet', (1.11, 2.48, 0.48), 'precise', 234)]
[sysnav_node]: ➡️ DEPARTING - exploration goal=(0.47, 3.25, 2.40)
```

---

## Notes

- **Debug output.** The node writes visualisations to `/home/docker/ai_module/debug`
  inside the container and creates the directory itself. Nothing needs to be mounted;
  the files are optional diagnostics.
- **DDS.** The image pins `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`, matching the base
  image's own default. Both containers run with `network_mode: host` but private IPC
  namespaces, so their `/dev/shm` are separate; under Fast DDS the two sides discover
  each other but no data is ever delivered — topics appear in `ros2 topic list` while
  the module receives zero messages. If the system container is started in a way that
  leaves it on Fast DDS, launch our module with `-e RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
  so both sides match. To check which one it is using:

  ```bash
  docker exec iros2026_system bash -c 'ls /dev/shm | grep -c fastrtps'   # 0 => CycloneDDS
  ```
