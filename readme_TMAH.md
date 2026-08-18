# SysNav Run Guide (Based on the Docker Hub Submission Image)

CMU VLN Challenge 2026 — Run the setup in three terminal windows: Terminal A (simulator), Terminal B (sysnav node), and Terminal C (queries).
This document only covers the procedure for using the submitted image directly from Docker Hub without additional build steps.

---

## 0. Docker Hub Submission Image

- **Docker Hub**: https://hub.docker.com/r/parkjaeil00/cmu-vln-2026-sysnav
- **Image**: `parkjaeil00/cmu-vln-2026-sysnav:submission-v1` (`latest` contains the same content)
- linux/amd64, compressed size 9.1GB / unpacked size 27.4GB
- The image already includes the **model weights and build artifacts**, so no extra downloads or builds are needed at runtime.
  - `yolov8x-worldv2.pt` (YOLO-World), `sam2.1_hiera_tiny.pt` (SAM2)
  - `/home/docker/ai_module/install/sysnav` (colcon build completed)
  - `USER=docker`, `WORKDIR=/home/docker/ai_module`

```bash
cd ~/CMU-VLN-Challenge-2026
docker pull parkjaeil00/cmu-vln-2026-sysnav:submission-v1
```

---

## 1. Prepare the `.env` File — **API key is provided separately**

`sysnav` uses the Google Gemini API for query parsing and candidate selection. Since the key is not committed to the repository
(`.gitignore` includes `ai_module/.env`), the **API key should be sent through the designated channel** by the organizers.
After that, create the `.env` file with the script below and fill in the provided key.

```bash
cd ~/CMU-VLN-Challenge-2026
./docker/create_env.sh <API_KEY>      # Creates ai_module/.env with GEMINI_API_KEY= (empty value)
```

---

## 2. Start the Containers

Run this from the repository root.

```bash
cd ~/CMU-VLN-Challenge-2026
xhost +                              # X11 permission for simulator/RViz GUI
./docker/start_containers.sh         # Starts system + sysnav_submission
```

---

## 3. Terminal A — Simulator
Use the official method provided by the organizers if available.

This is the method we use.

```bash
cd ~/CMU-VLN-Challenge-2026
./docker/A_시뮬레이터.sh                   # Run the current scene as-is
./docker/run_scene.sh hotel_room_1        # Run a different scene (requires map/<scene_name>.zip)
```

Example scene names: `hotel_room_1`, `home_building_1`, `office_2`, `livingroom_1` …
The map zip files should be placed in the repository’s `map/` folder so they are mounted into the container at `/home/docker/maps`.

---

## 4. Terminal B — sysnav Node (Submission Image)

```bash
cd ~/CMU-VLN-Challenge-2026
mkdir -p ~/CMU-VLN-Challenge-2026/ai_module/debug
sudo chmod -R 777 ~/CMU-VLN-Challenge-2026/ai_module/debug
./docker/B_sysnav_실행_제출이미지.sh
```
- If the container does not exist, it will be created automatically, then `ros2 launch sysnav sysnav.launch.py` will run.
- Since the image already contains the build output, **no colcon rebuild is required**.

On successful startup, the log should show: `[sysnav_node]: SysNav single-room MVP started`
Once a query is received and recognition starts, you should see the ultralytics (YOLO-World) banner (🚀) and model loading logs.

---

## 5. Terminal C — Queries

```bash
cd ~/CMU-VLN-Challenge-2026
./docker/C_질의.sh                          # Automatically selects the running sysnav container and connects
./docker/C_질의.sh iros2026_sysnav_submission  # Explicitly specify the container
```

Publish questions from the connected shell (example using `hotel_room_1`):

```bash
# Mission 1 — Numerical → /numerical_response (Int32)
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'How many pillows are on the bed?'}"

# Mission 2 — Object Reference → /selected_object_marker (Marker)
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Find the bedside table farthest from the window.'}"

# Mission 3 — Instruction-Following → /way_point_with_heading (Pose2D sequence)
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Go to the bedside table closest to the window and stop at the chair closest to the TV.'}"
```

Mission type is automatically inferred from the sentence (`sysnav/task/mission_classifier.py`).

---

## 6. Check the Results

```bash
# Response topics (open another terminal for this)
./docker/C_질의.sh
ros2 topic echo /numerical_response
ros2 topic echo /selected_object_marker
ros2 topic echo /way_point_with_heading
```

---

## 7. Full Sequence Summary

```bash
# First time only
docker pull parkjaeil00/cmu-vln-2026-sysnav:submission-v1
./docker/create_env.sh <GEMINI_API_KEY_PROVIDED_BY_ORGANIZERS>

# For each run
xhost +
./docker/start_containers.sh

# Terminal A (optional)
./docker/run_scene.sh hotel_room_1

# Terminal B
./docker/B_sysnav_실행_제출이미지.sh

# Terminal C
./docker/C_질의.sh
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Find the bedside table farthest from the window.'}"

# Status check on host
./docker/ui_checker.sh
```