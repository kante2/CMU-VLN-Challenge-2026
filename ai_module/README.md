# SysNav Run Guide

CMU VLN Challenge 2026 — Run the setup in three terminal windows: Terminal A (simulator), Terminal B (sysnav node), and Terminal C (queries).
This document only covers the procedure for using the submitted image directly from Docker Hub without additional build steps.

---

## 0. Docker Hub Submission Image

- **Docker Hub**: https://hub.docker.com/r/kante2/cmu-vln-2026-sysnav/tags
- **Image**: `kante2/cmu-vln-2026-sysnav:submission-v2` (`latest` contains the same content)
- The image already includes the **model weights and build artifacts**, so no extra downloads or builds are needed at runtime.
  - `yolov8x-worldv2.pt` (YOLO-World), `sam2.1_hiera_tiny.pt` (SAM2)
  - `/home/docker/ai_module/install/sysnav` (colcon build completed)
  - `USER=docker`, `WORKDIR=/home/docker/ai_module`

```bash
cd ~/CMU-VLN-Challenge-2026
docker pull kante2/cmu-vln-2026-sysnav:submission-v2
```

---

## 1. Prepare the `.env` File — **API key is provided separately**

`sysnav` uses the Google Gemini API for query parsing and candidate selection. Since the key is not committed to the repository
(`.gitignore` includes `ai_module/.env`), the **API key should be sent through the designated channel** by the organizers.
After that, create the `.env` file with the script below and fill in the provided key.

```bash
cd ~/CMU-VLN-Challenge-2026
./ai_module/docker/create_env.sh <API_KEY>      # Creates ai_module/.env with 
```

---

## 2. Start the Containers

Run this from the repository root.

```bash
cd ~/CMU-VLN-Challenge-2026/docker
xhost +
docker compose -f compose_gpu.yml up --build -d system sysnav_module
```

기존 컨테이너가 이미 있으면 (Exited 상태 등):s
```bash
docker start iros2026_system iros2026_sysnav_module
```

---

## 3. Terminal A — Simulator
Use the official method provided by the organizers if available.

This is the method we use.

```bash
cd ~/CMU-VLN-Challenge-2026
docker exec -it iros2026_system bash
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
```

---

## 4. Terminal B — sysnav Node (Submission Image)

```bash
cd ~/CMU-VLN-Challenge-2026
# mkdir -p ~/CMU-VLN-Challenge-2026/ai_module/debug
# sudo chmod -R 777 ~/CMU-VLN-Challenge-2026/ai_module/debug
docker exec -it iros2026_sysnav_module bash
source /opt/ros/jazzy/setup.bash
source /home/docker/ai_module/install/setup.bash
ros2 launch sysnav sysnav.launch.py
```
- If the container does not exist, it will be created automatically, then `ros2 launch sysnav sysnav.launch.py` will run.
- Since the image already contains the build output, **no colcon rebuild is required**.

On successful startup, the log should show: `[sysnav_node]: SysNav single-room MVP started`
Once a query is received and recognition starts, you should see the ultralytics (YOLO-World) banner (🚀) and model loading logs.

---

## 5. Terminal C — Queries

```bash
cd ~/CMU-VLN-Challenge-2026
docker exec -it iros2026_sysnav_module bash
source /opt/ros/jazzy/setup.bash
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Find the toilet'}"
```

Mission type is automatically inferred from the sentence (`ai_module/src/sysnav_ros2_mvp/sysnav/task/mission_classifier.py`).

---
