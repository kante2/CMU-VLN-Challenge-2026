
# 컨테이너 생성 명령어

cd /home/kante/CMU-VLN-Challenge-2026/docker
xhost +
docker compose -f compose_gpu.yml up --build -d
이 한 줄로 아래 3개 컨테이너가 빌드/생성/실행됩니다:

iros2026_system — 시뮬레이터/autonomy (이미지 pull)
iros2026_ai_module — ../ai_module/docker/Dockerfile 빌드
iros2026_tmah_module — ../ai_module/docker/Dockerfile.tmah 빌드

# 권한 문제
sudo chown -R kante:kante /home/kante/CMU-VLN-Challenge-2026/ai_module/debug

# 시뮬 겹침 초기화
pkill -9 -f autonomy_stack_mecanum_wheel_platform
pkill -9 -f static_transform_publisher
pkill -9 -f joy_node
pkill -9 -f default_server_endpoint

# ------------------------------------------------------------------------------
# 맨 처음에 모든 컨테이너를 실행해야 함
docker start iros2026_system iros2026_ai_module iros2026_tmah_module

#  tmah_vlm src 수정 후엔 반드시:
docker exec iros2026_tmah_module bash -lc "cd /home/docker/ai_module && source /opt/ros/jazzy/setup.bash && colcon build --symlink-install --packages-select tmah_vlm"
# 그리고 노드 재시작 (orphan 프로세스가 GPU 물고 있으니 pid로 확실히 kill)


A — 시뮬레이터/autonomy 실행 (터미널 A)
docker exec -it iros2026_system bash
컨테이너 안에서
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh

# 참고, 로봇이 여러대 겹친걸로 뜨면 이거한 후, 다시 컨테이너 접근
docker restart iros2026_system 

B - new : TMAH 팀이 운영하는 컨테이너 실행 명령어
docker exec -it iros2026_tmah_module bash
ros2 launch tmah_vlm tmah_vlm.launch

C-질문 일회 던지기 용도 (터미널 C, 또 새 창)
docker exec -it iros2026_ai_module bash
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Find teal pillow on the sofa farthest from the window'}"
  이거 쏘면 Unity/RVIZ 화면에서 로봇이 waypoint 따라 움직이고 대상 객체에 박스가 뜬다.

D - 키보드 컨트롤 하는 방법
RVIZ에 뜨고 있음, 


# ==============================================================================
# tmah_vlm 코드 구조 (2026-07 리팩토링 완료)
# ==============================================================================

경로: `ai_module/src/tmah_vlm/tmah_vlm/`

## 한 줄 요약
질문(문장) 하나가 들어오면 → 저장 → 타이머가 꺼내서 → 첫 단어로 분기 → 유형별
solver가 "검출 → 선택 → 3D 위치 → 발행"까지 처리한다.

## 질문이 처리되는 흐름 (여기부터 읽으면 됨)

```
/challenge_question (문장 수신)
        │
        ▼
① common/callback.py  question_callback()     ← 문장을 node.pending_question에 "저장만"
        │                                         (무거운 VLM 추론 절대 안 함)
        ▼
② main_node.py  main_control_loop()  (config.MAIN_LOOP_PERIOD_SEC 타이머)
        │        - pending 질문 꺼내고 준비됐는지 확인 (question_process/dispatch.py)
        ▼
③ question_process/dispatch.py  dispatch_question()   ← 문장 첫 단어로 분기하는 핵심 지점
        │
        ├─ "find ..."               → t3_object_reference_solver  object_reference_process()
        ├─ "how many/count ..."     → t2_numerical_solver         numerical_process()
        └─ 그 외                     → t1_instruction_solver       instruction_process()
```

- **문장이 들어오는 곳**: `common/callback.py` 의 `question_callback` (저장만)
- **문장이 갈라지는 곳**: `question_process/dispatch.py` 의 `dispatch_question` (첫 단어 분기 — 개선 시 여기)
- **문장별 실제 로직**: 세 solver의 `*_process`

## 폴더 트리 (기능 단위로 묶음)

```
tmah_vlm/
├── main_node.py     진입점. 조립 + main_control_loop + main() 만 담당
├── config.py        토픽명/프레임명/임계값/상수 전부 모음. 환경 바뀌면 여기부터 확인
├── nav_publish.py   nav/challenge로 나가는 발행 전부 (waypoint / marker / count)
│
├── common/          센서 무관 노드 뼈대·수명주기 (도메인 로직 아님)
│   ├── initialize.py  __init__이 부르는 initialize_* (상태/모델/구독/발행/타이머) + 모델 백그라운드 로딩
│   ├── callback.py    question_callback · pose_callback — "최신값 저장"만
│   └── helpers.py     solver 공용 상태조회 (robot pose, heartbeat, stamp_to_sec)
│
├── question_process/  문장 → 미션 선택 + 구조체 생성
│   ├── dispatch.py     dispatch_question(분기) + peek/ready/print_waiting(처리 준비 판단)
│   ├── query_parser.py 질문 문장 → GroundingDINO 검출어 추출
│   └── context.py      solver별 작업변수 ctx 생성 make_*_context() (SimpleNamespace 구조체)
│
├── t1_instruction_solver/      그 외 질문 (instruction_process, 아직 stub: 앞 1m 직진)
├── t2_numerical_solver/        "how many/count" (numerical_process, 현재 시야 개수 세기)
├── t3_object_reference_solver/ "find" (object_reference_process)
│
├── sensor_process/  카메라 → 2Dbox → segmentation → LiDAR 좌표변환 → ray → 3D 위치
│   ├── sensor_process.py  ★ 흐름파일 — 공용 센서 스텝을 파이프라인 순서로 모음
│   │                        (grab_camera_image → detect_candidate_boxes →
│   │                         load_scan_points_in_map → segment_selected_object →
│   │                         estimate_target_3d_pose). t2/t3가 여기서 가져다 씀
│   ├── detector.py / segmenter.py / selector.py   GroundingDINO / SAM / Qwen
│   ├── image_utils.py     ROS Image → PIL + grab_camera_image
│   ├── coordinate_transform.py  TF (map↔sensor↔camera)
│   ├── projector.py       2D↔3D 재투영, segmentation에 맞는 LiDAR ray → 3D 위치/크기
│   ├── scan_transform.py  image stamp에 동기화된 scan → map frame 점군
│   ├── bbox_estimator.py / bbox_wireframe.py   3D 박스 크기 / 모서리 좌표
│   ├── visualize.py       디버그 오버레이/이미지/텍스트 저장
│   └── callback.py        image_callback · scan_callback — "최신값 저장"만
│
└── reasoning/       누적 관측 기반 공간/관계 추론 (한 부모로 묶음)
    ├── spatial/       공간관계: relations(결정론적 관계함수), candidate_filter, relation_parser
    ├── graph/         누적 scene graph 구축/렌더/시각화 (관계 질문의 근거)
    └── sort3d/        SORT3D-lite 추론 (scene graph 기반 관계 질문 fallback)
```

## solver 코드 스타일 규칙 (새 기능 추가 시 이 패턴 유지)

각 solver의 `*_process` 함수는 **조건문 + 함수 호출만** 나열한다. 함수 이름을 위에서
아래로 읽으면 그게 곧 파이프라인 순서다. 예 (`t3_object_reference.py`):

```python
def object_reference_process(node, question):
    ctx = make_object_ref_context(question)            # 작업변수 구조체(ctx) 1개 생성
    if not sensors_and_models_ready(node): return
    grab_camera_image(node, ctx)          # ctx.image, ctx.image_stamp 채움
    extract_target_object(ctx)            # ctx.detect_prompt 채움
    detect_candidate_boxes(node, ctx)     # ctx.detections 채움
    ...
    publish_object_result(node, ctx)
```

핵심 원칙:
1. **context 구조체(`question_process/context.py`)** 에 그 질문 처리에 필요한 작업변수를 다 모아둔다.
   인스턴스는 질문마다 process 진입부에서 새로 만든다(질문 간 값 안 섞이게).
2. 각 스텝 함수는 `ctx`를 받아 **자기 필드 하나만 채우고**(return 대신 인자 업데이트),
   다음 함수가 그 필드를 읽어 이어서 쓴다. `x = get_something(node)` 처럼 함수 안에서
   값을 뽑아 쓰지 않는다 — 중간값도 ctx 필드로 올려 전용 스텝으로 분리한다.
3. 진입점 이름은 `<파일이름>_process(node, ...)` 형태. 함수 위에는 한 줄 기능 주석.
4. 센서 공용 스텝(카메라→box→scan→seg→3D)은 solver마다 재정의하지 말고
   `sensor_process/sensor_process.py`에서 import해서 쓴다. 발행은 `nav_publish.py`로 모은다.

## 3차 리팩토링 요약 (2026-07-14, 구조 변경·동작 그대로)

- **top-level 폴더 10 → 7.** 없어진 폴더: `context/`, `geometry/`, `perception/`.
- `perception/{camera,lidar}` + `geometry/` + `common/scan_transform` → **`sensor_process/`** 로 통합.
  카메라→3D 파이프라인의 공용 스텝을 흐름파일 `sensor_process/sensor_process.py`에 모아
  t2/t3가 중복 정의 없이 import (solver 파일이 얇아짐).
- `context/context` → `question_process/context`, `context/helpers` + `main_node.dispatch_question`
  → **`question_process/dispatch.py`**, `perception/camera/query_parser` → `question_process/query_parser`.
- 세 solver에 흩어져 있던 marker/waypoint/count 발행 → **`nav_publish.py`** 한 파일로 통일
  (`t3_object_reference_solver/publish.py` 삭제).
- `spatial/` · `graph/` · `sort3d/` → **`reasoning/`** 한 부모 아래로 이동 (내부 구조는 그대로).
- marker 색/선굵기, t1 전진거리, 타이머 주기 등 흩어진 상수 → **`config.py`** 로 흡수.
- 모든 이동은 `git mv`(히스토리 유지), import 90곳 + `setup.py` 엔트리포인트 갱신,
  `py_compile` + import 교차검증 스크립트로 확인.

주의: 구조가 크게 바뀌었으니 컨테이너에서 클린 재빌드
(`rm -rf build/tmah_vlm install/tmah_vlm && colcon build --symlink-install --packages-select tmah_vlm`,
권한 에러 나면 CLAUDE.md의 07-13 항목 참고) 후 `ros2 launch tmah_vlm tmah_vlm.launch` 로 확인할 것.

