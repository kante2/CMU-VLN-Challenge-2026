# SysNav ROS2 Single-Room MVP

이 패키지는 다음 파이프라인을 하나의 ROS2 프로세스와 상태 머신으로 연결한 구현 예시다.

```text
/challenge_question
        ↓
query_parser.py
        ↓
YOLOv8x-WorldV2 → SAM2 → LiDAR 3D grounding
        ↓
Object Association / Merge
        ↓
Target 후보 존재?
 ├─ Yes → Gemini 2.5 Flash → object approach waypoint
 └─ No  → online occupancy map → frontier/coverage route
        ↓
/way_point_with_heading
```

## 사용 토픽

| 방향 | 토픽 | 타입 | 용도 |
|---|---|---|---|
| Subscribe | `/challenge_question` | `std_msgs/msg/String` | 질문 수신 |
| Subscribe | `/state_estimation` | `nav_msgs/msg/Odometry` | 로봇 pose |
| Subscribe | `/camera/image` | `sensor_msgs/msg/Image` | 파노라마 RGB |
| Subscribe | `/sensor_scan` | `sensor_msgs/msg/PointCloud2` | 3D LiDAR |
| Publish | `/way_point_with_heading` | `geometry_msgs/msg/Pose2D` | 목표/탐색 waypoint |

## 포함 범위

- 질의에서 target/reference object 추출
- YOLOv8x-WorldV2 open-vocabulary detection
- YOLO bbox를 SAM2 bbox prompt로 사용
- equirectangular panorama에 LiDAR point projection
- SAM2 mask 내부 LiDAR point로 3D object 생성
- category + 3D distance + extent + color histogram 기반 object association
- object ID, category, 3D position, point cloud, representative image, observation count, last-seen 관리
- Gemini 2.5 Flash 후보 선택
- target object standoff waypoint 생성
- LiDAR 기반 고정 크기 occupancy map
- free/unknown frontier, coverage score, 다중 waypoint 순서화
- 이동 중 perception 재실행 및 목표 발견 시 탐색 중단

## 미포함 범위

- Room Segmentation
- Room/Object/Viewpoint graph edge
- Room-query Navigation
- Early-stop Room Navigation
- 정식 SLAM/loop closure
- navigation action feedback

현재 `/state_estimation`을 전역 map pose처럼 사용한다. 실제 로봇에서는 SLAM 또는 simulator가 제공하는 안정적인 전역 pose가 필요하다.

## 설치

ROS2 Jazzy를 source한다.

```bash
source /opt/ros/jazzy/setup.bash
python3 -m pip install -r requirements.txt
```

CUDA 환경에 맞는 PyTorch는 별도로 설치하는 것이 안전하다.

SAM2 공식 저장소를 설치한다.

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

환경변수를 지정한다.

```bash
export YOLO_WORLD_WEIGHTS="yolov8x-worldv2.pt"
export SAM2_CHECKPOINT="/absolute/path/to/sam2.1_hiera_tiny.pt"
export SAM2_MODEL_CFG="configs/sam2.1/sam2.1_hiera_t.yaml"
export GEMINI_API_KEY="YOUR_API_KEY"
export GEMINI_MODEL="gemini-2.5-flash"
```

## Calibration

`sysnav/config.py`에서 아래 값을 실제 센서에 맞게 수정해야 한다.

```python
T_LIDAR_TO_CAMERA
T_SENSOR_TO_BASE
PANORAMA_YAW_OFFSET_DEG
PANORAMA_PITCH_OFFSET_DEG
```

기본 가정은 다음과 같다.

```text
LiDAR: x forward, y left, z up
Panorama camera: x right, y forward, z down
```

기본 extrinsic은 예시값이므로 그대로 사용하면 안 될 수 있다.

## 빌드

```bash
cd ~/ros2_ws/src
unzip sysnav_ros2_mvp.zip
cd ~/ros2_ws
colcon build --packages-select sysnav --symlink-install
source install/setup.bash
```

## 실행

```bash
ros2 run sysnav sysnav
```

또는

```bash
ros2 launch sysnav sysnav.launch.py
```

질문 발행 예시:

```bash
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Find the white chair'}"
```

## 상태 흐름

```text
IDLE
  ↓ 질문
OBSERVE
  ├─ 후보 있음 → SELECT_TARGET → NAVIGATE_TARGET → SUCCESS
  └─ 후보 없음 → PLAN_EXPLORATION → FOLLOW_EXPLORATION
                                      ↓ 새 관측
                                    OBSERVE
```

## 주의사항

1. 센서 callback은 데이터만 저장하고 YOLO/SAM2/Gemini는 worker thread에서 실행한다.
2. 이미지와 가장 가까운 timestamp의 PointCloud2와 odometry를 사용한다.
3. `/way_point_with_heading` 도착 여부는 별도 feedback이 없어 odometry 거리로 판정한다.
4. 탐색 occupancy map은 MVP용 ray-tracing map이다. 정식 SLAM map으로 교체할 수 있다.
5. Gemini 호출 실패 시 confidence/거리 기반 fallback을 사용한다.
6. RTX 8GB에서는 SAM2 tiny/small부터 테스트하는 것이 좋다.
