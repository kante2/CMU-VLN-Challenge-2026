
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
① node/callbacks.py  question_callback()      ← 문장을 node.pending_question에 "저장만"
        │                                         (무거운 VLM 추론 절대 안 함)
        ▼
② main_node.py  main_control_loop()  (0.2초 타이머)
        │        - pending 질문 꺼내고 준비됐는지 확인
        ▼
③ main_node.py  dispatch_question()           ← 문장 첫 단어로 분기하는 핵심 지점
        │
        ├─ "find ..."               → t3_object_reference_solver  object_reference_process()
        ├─ "how many/count ..."     → t2_numerical_solver         numerical_process()
        └─ 그 외                     → t1_instruction_solver       instruction_process()
```

- **문장이 들어오는 곳**: `node/callbacks.py` 의 `question_callback` (저장만)
- **문장이 갈라지는 곳**: `main_node.py` 의 `dispatch_question` (첫 단어 하드코딩 분기 — 개선 시 여기)
- **문장별 실제 로직**: 세 solver의 `*_process`

## 폴더 트리 (도메인별로 묶음)

```
tmah_vlm/
├── main_node.py     진입점. 조립 + main_control_loop + dispatch_question 만 담당
├── config.py        토픽명/프레임명/임계값 전부 모음. 환경 바뀌면 여기부터 확인
│
├── node/            노드 뼈대·수명주기 (도메인 로직 아님)
│   ├── setup.py       __init__이 부르는 initialize_* (상태/모델/구독/발행/타이머) + 모델 백그라운드 로딩
│   ├── callbacks.py   센서/질문 콜백 — 전부 "최신값 저장"만
│   ├── helpers.py     solver 공용 상태조회 (pending 확인, robot pose, image-scan 동기화, heartbeat)
│   └── context.py     solver별 작업변수 ctx 생성 함수 make_*_context() (SimpleNamespace 구조체 반환)
│
├── t1_instruction_solver/      그 외 질문 (instruction_process, 아직 stub: 앞 1m 직진)
├── t2_numerical_solver/        "how many/count" (numerical_process, 현재 시야 개수 세기)
├── t3_object_reference_solver/ "find" (object_reference_process) + publish.py (marker/waypoint 발행)
│
├── perception/     2D 인식: image_utils(ROS→PIL), query_parser(검출어 추출),
│                   detector(GroundingDINO), visualize(디버그 저장), segmenter(SAM), selector(Qwen 시각선택)
├── geometry/       3D 기하: coordinate_transform(TF), projector(2D→3D 재투영),
│                   bbox_estimator(3D박스 크기), bbox_wireframe(모서리 좌표)
├── spatial/        공간관계: relations(결정론적 관계함수), candidate_filter(관계로 후보 좁히기), relation_parser
├── graph/          누적 scene graph 구축/렌더/시각화 (관계 질문의 근거)
└── sort3d/         SORT3D-lite 추론 (scene graph 기반 관계 질문 fallback)
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
1. **context 구조체(`node/context.py`)** 에 그 질문 처리에 필요한 작업변수를 다 모아둔다.
   인스턴스는 질문마다 process 진입부에서 새로 만든다(질문 간 값 안 섞이게).
2. 각 스텝 함수는 `ctx`를 받아 **자기 필드 하나만 채우고**(return 대신 인자 업데이트),
   다음 함수가 그 필드를 읽어 이어서 쓴다. `x = get_something(node)` 처럼 함수 안에서
   값을 뽑아 쓰지 않는다 — 중간값도 ctx 필드로 올려 전용 스텝으로 분리한다.
3. 진입점 이름은 `<파일이름>_process(node, ...)` 형태. 함수 위에는 한 줄 기능 주석.

## 이번 리팩토링 요약 (구조 변경, 동작은 그대로)

- `vlm_node.py` → `main_node.py` 로 rename (setup.py 엔트리포인트도 갱신)
- `handlers/{object_reference,numerical,instruction}.py` → `t1/t2/t3_*_solver/` 3개 폴더로 분리
- 각 solver를 **context 구조체 + 출력-인자 스타일**로 재작성, 함수명 직관화 + 주석 추가
- **파일 하나짜리 폴더 9개**(initialize/callback/helper/context/segmentation/reasoning/
  tf/grounding/bbox3d/spatial_reasoning/object_filter)를 **4개 도메인**(node/perception/
  geometry/spatial)으로 통합 → top-level 폴더 15개 → 9개
- `graph`·`sort3d` 는 파일 많은 진짜 서브시스템이라 그대로 유지

주의: 구조가 크게 바뀌었으니 컨테이너에서 `colcon build --symlink-install` 한 번 재실행 후
`ros2 launch tmah_vlm tmah_vlm.launch` 로 확인할 것.

