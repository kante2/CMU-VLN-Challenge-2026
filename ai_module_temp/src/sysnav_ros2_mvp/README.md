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
 ├─ Yes → Gemini Flash → object approach waypoint
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
- Gemini Flash 후보 선택
- target object standoff waypoint 생성
- LiDAR 기반 고정 크기 occupancy map
- free/unknown frontier, coverage score, 다중 waypoint 순서화
- 이동 중 perception 재실행 및 목표 발견 시 탐색 중단
- Single Room용 `Room_0` Scene Graph 생성
- LiDAR novel voxel coverage가 임계값을 넘을 때만 대표 Viewpoint Node와 RGB 이미지 저장
- `Viewpoint -> Room (lies_in)` edge
- `Object -> Room (lies_in)` edge
- `Viewpoint -> Object (observes)` edge
- 질의에 spatial constraint가 있을 때만 `Object -> Object` edge 생성
- Scene Graph 변경 시 debug 폴더의 JSON/DOT/PNG 자동 갱신

## 미포함 범위

- Multi-Room Segmentation 및 Room 경계 자동 생성
- Room category 자동 분류
- Cross-Room Navigation
- Room-query Navigation
- Early-stop Room Navigation
- 정식 SLAM/loop closure
- navigation action feedback

현재 `/state_estimation`을 전역 map pose처럼 사용한다. 실제 로봇에서는 SLAM 또는 simulator가 제공하는 안정적인 전역 pose가 필요하다.


## Scene Graph 구현

상세한 Viewpoint coverage 구현과 튜닝 항목은 `VIEWPOINT_IMPLEMENTATION.md`에 정리되어 있다.


새로 추가된 핵심 파일은 다음과 같다.

```text
sysnav/scene_graph/
├── scene_graph_manager.py
├── scene_graph_visualizer.py
└── viewpoint_coverage.py

sysnav/reasoning/
└── spatial_relation_reasoner.py
```

현재 Single Room이므로 `Room_0` 하나를 생성한다. Viewpoint는 매 perception 프레임마다
추가하지 않는다. 현재 360° LiDAR 관측을 map-frame voxel 집합 `C_t`로 만들고, 기존
Viewpoint들의 coverage 합집합 `C_prev`와 비교한다.

```text
C_prev = 기존 Viewpoint coverage의 합집합
C_novel = C_t - C_prev

|C_novel| > omega
    ├─ True  → 현재 pose를 새 Viewpoint로 추가하고 panorama 저장
    └─ False → Viewpoint를 추가하지 않음
```

각 Viewpoint는 논문의 `A(v_i^v) = {p_i, C_i, I_i}`에 대응하도록 pose, coverage voxel
region, panorama image path를 저장한다. `ObjectMemory.update()`가 반환한 실제
`object_id`를 이용해 대표 Viewpoint와 Object의 visibility edge를 연결한다.

```text
Viewpoint_i ── lies_in ──> Room_0
Object_j    ── lies_in ──> Room_0
Viewpoint_i ── observes ──> Object_j
Object_a    ── on/near/... ──> Object_b
```

Object-Object edge는 모든 객체 쌍에 대해 미리 만들지 않는다. 질문에 `on`, `near`,
`left of`, `in front of`, `between` 같은 관계가 있을 때만 생성한다. 현재 프레임만 보는
것이 아니라 Scene Graph에서 목표와 기준 객체를 모두 `observes`하는 과거 Viewpoint를
검색하고, 저장된 panorama를 다시 불러 관계를 검증한다. Gemini API를 사용할 수 있으면
annotation된 RGB로 판단하고, 호출할 수 없거나 응답이 실패하면 해당 Viewpoint pose와
3D bounding box 기반 기하 판별로 fallback한다.

Scene Graph가 갱신될 때마다 `DEBUG_DIR` 안의 같은 파일을 덮어쓴다.

```text
/home/docker/ai_module/debug/
├── scene_graph_latest.json   # 전체 node/edge 원본
├── scene_graph_latest.dot    # Graphviz 입력
├── scene_graph_latest.png    # 바로 확인 가능한 그래프 그림
└── scene_graph_viewpoints/
    ├── viewpoint_000001.jpg
    └── ...
```

`selection_job()`은 검증된 Object-Object edge가 있으면 해당 edge의 source object들을
우선 후보로 사용하고, 그 후보들 중 최종 객체를 Gemini selector가 선택한다.

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
export GEMINI_MODEL="gemini-3.5-flash"

# Scene Graph debug export와 관계 검증
export SYSNAV_SCENE_GRAPH_EXPORT="1"
export SYSNAV_SCENE_GRAPH_SAVE_IMAGES="1"
export SYSNAV_SCENE_GRAPH_USE_GEMINI="1"

# SysNav Viewpoint coverage: |C_t - C_prev| > omega
export SYSNAV_VIEWPOINT_COVERAGE_DISTANCE_M="4.0"
export SYSNAV_VIEWPOINT_COVERAGE_VOXEL_SIZE_M="0.40"
export SYSNAV_VIEWPOINT_NOVEL_VOXEL_THRESHOLD="120"
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
5. Gemini target selection 실패 시 confidence/거리 기반 fallback을 사용한다.
6. Object-Object relation Gemini 검증 실패 시 3D geometry fallback을 사용한다.
7. `in_front_of`, `left_of` 등의 방향 관계는 관계 증거가 된 Viewpoint의 yaw 기준이다.
8. `VIEWPOINT_NOVEL_VOXEL_THRESHOLD`는 논문의 `omega`에 해당하며 실제 LiDAR 밀도에 맞춰 튜닝해야 한다.
9. coverage는 360° LiDAR ray를 voxelize한 구현이며, 논문이 공개하지 않은 수치 파라미터는 환경변수로 조절한다.
10. RTX 8GB에서는 SAM2 tiny/small부터 테스트하는 것이 좋다.
