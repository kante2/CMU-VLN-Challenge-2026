# CMU-VLN-Challenge-2026

<!-- ============================================================================
     이 파일 위쪽(이 섹션)은 "현재 구조/파이프라인"의 단일 진실이다.
     아래 날짜별 섹션들은 그 시점의 디버깅 기록이라 옛 경로(perception/·geometry/·
     node/·context/ 등)를 그대로 쓰고 있다 — 경로가 헷갈리면 항상 이 섹션을 먼저 믿을 것.
     ============================================================================ -->

## 현재 코드 구조 & 파이프라인 (최신 = 2026-07-14 3차 리팩터링, 여기부터 읽기)

경로: `ai_module/src/tmah_vlm/tmah_vlm/`. 더 자세한 트리는 저장소 루트 `readme_kante.md`.

**설계 원칙(전 파일 공통)**: 클래스 메서드 대신 **`node`를 첫 인자로 받는 자유 함수**.
solver 진입점은 `<파일이름>_process(node, question)`. 각 process는 **조건문 + 함수 호출만**
나열하고(함수 이름을 위→아래로 읽으면 그게 파이프라인 순서), 각 스텝 함수는 질문마다 새로
만든 **ctx(SimpleNamespace 구조체)** 를 받아 **자기 필드 하나만 채운다**(return 대신 인자
업데이트) → 다음 함수가 그 필드를 읽어 이어 씀. 상수는 전부 `config.py`.

### 질문 하나가 처리되는 전체 흐름

```
/challenge_question (문장)
   │
   │ ① common/callback.py  question_callback()      → node.pending_question 에 "저장만"
   │    (모든 센서 콜백도 최신값 저장만 — 무거운 VLM 추론은 콜백에서 절대 안 함)
   ▼
   ② main_node.py  main_control_loop()  (config.MAIN_LOOP_PERIOD_SEC 타이머)
   │    peek_pending_question / ready_to_process  (question_process/dispatch.py)
   ▼
   ③ question_process/dispatch.py  dispatch_question()   ← 문장 첫 단어로 미션 분기(핵심 지점)
   │    ├─ "find ..."            → t3_object_reference_solver.object_reference_process()
   │    ├─ "how many/count ..."  → t2_numerical_solver.numerical_process()
   │    └─ 그 외                  → t1_instruction_solver.instruction_process()
   ▼
   ④ solver process()  = ctx 생성 → 센서 스텝 → 추론 스텝 → 발행     (아래 t3 예시)
```

**t3(find)의 파이프라인** (t2/t1은 이 부분집합):
```
make_object_ref_context(q)                     # question_process/context.py : 작업변수 ctx 생성
─ 센서 (sensor_process/sensor_process.py 공용 스텝) ─
grab_camera_image        → ctx.image, image_stamp        # 카메라 최신 프레임 스냅샷
extract_target_object    → ctx.detect_prompt             # question_process/query_parser.py
detect_candidate_boxes   → ctx.detections                # GroundingDINO 2D 박스 (sensor_process/detector.py)
load_scan_points_in_map  → ctx.scan_points_map           # image에 동기화된 LiDAR → map frame (scan_transform)
─ 추론/선택 (reasoning/ + solver 로직) ─
narrow_candidates_by_relation → ctx.candidate_indices    # reasoning/spatial : 공간관계로 후보 좁힘
pick_final_candidate     → ctx.selected                  # Qwen 시각 선택 (sensor_process/selector.py)
─ 센서 3D (sensor_process) ─
segment_selected_object  → ctx.segmentation_mask         # SAM 마스크 (sensor_process/segmenter.py)
estimate_target_3d_pose  → ctx.result                    # seg에 맞는 LiDAR ray → 3D 위치/크기 (projector.py)
compute_approach_waypoint→ ctx.waypoint
─ 발행 ─
publish_object_result(node, ctx)                         # nav_publish.py : marker(CUBE+wireframe) + waypoint
record_observation_in_graph                              # reasoning/graph : scene graph에 누적(다음 관계질문용)
```
※ 공간관계 질문은 검출 전에 누적 scene graph(reasoning/sort3d)로 먼저 풀어보고, 되면 조기 종료.

### 폴더 = 기능 (top-level 7개)

| 폴더/파일 | 역할 |
|---|---|
| `main_node.py` | 진입점. 조립 + main_control_loop + main() 만 |
| `config.py` | 토픽·프레임·임계값·상수 전부. 환경 바뀌면 여기부터 |
| `nav_publish.py` | nav/challenge 발행 전부 (waypoint / marker / count) |
| `common/` | 센서 무관 노드 뼈대: initialize, callback(question·pose), helpers(get_robot_pose/heartbeat) |
| `question_process/` | 문장→미션: dispatch(분기+준비판단), query_parser(질문→검출어), context(ctx 생성) |
| `sensor_process/` | 카메라→2Dbox→seg→LiDAR변환→ray→3D. **sensor_process.py = 공용 센서 스텝 흐름파일** + detector/segmenter/selector/image_utils/coordinate_transform/projector/scan_transform/bbox_*/visualize/callback |
| `reasoning/` | 누적 관측 기반 추론: spatial(관계 파싱·필터), graph(scene graph), sort3d(SORT3D-lite fallback) |
| `t1/t2/t3_*_solver/` | 질문 유형별 process (find / count / 그 외) |

**규칙 요약**: 센서 공용 스텝은 solver마다 재정의하지 말고 `sensor_process/sensor_process.py`에서
import. 발행은 전부 `nav_publish.py`. 새 상수는 `config.py`. 새 기능은 "폴더 하나 = 기능 하나"
+ `node` 자유함수 + ctx(출력-인자) 패턴 유지.

> 구조가 바뀌면 컨테이너에서 클린 재빌드 후 `ros2 launch tmah_vlm tmah_vlm.launch` 로 확인
> (재빌드/권한 절차는 아래 "SAM 세그멘테이션" 섹션 07-13 항목, 최신 이동 내역은 "3차 리팩터링" 섹션).

---

## tmah_module 디버그 이미지 저장 안 되던 문제 (2026-07-08~09, 해결됨)

증상: `ai_module/debug`에 결과 이미지가 안 쌓임. 원인이 3개 겹쳐 있었음.

### 1. `/sensor_scan` 타입 불일치 (진짜 버그)
`test_pano_lidar_overlay.py`가 `/sensor_scan`을 `LaserScan`으로 구독했는데 실제 퍼블리셔는
`PointCloud2`를 쏨. ROS2는 타입 다르면 에러 없이 그냥 매칭을 안 시켜서 `scan_callback`이
아예 안 불렸음 (`/camera/image`는 타입이 맞아서 정상 동작 → "토픽은 오는데 저장만 안 됨"처럼 보임).
→ `PointCloud2` + `sensor_msgs_py.point_cloud2.read_points()`로 수정 (`projector.py`의
`pointcloud_to_xyz()`와 동일 방식).

### 2. 호스트에서 소스 고쳐도 컨테이너에 반영 안 됨
`compose_gpu.yml`의 `tmah_module`은 `debug`만 바인드 마운트, `src`는 `Dockerfile.tmah`가
빌드 시점에 `COPY`로 이미지에 박아넣는 구조라 실행 중 컨테이너엔 반영 안 됐음.
→ `src`도 바인드 마운트 추가:
```yaml
volumes:
  - ../ai_module/debug:/home/docker/ai_module/debug
  - ../ai_module/src:/home/docker/ai_module/src
```
`colcon build --symlink-install`이라 이제 `python3 xxx.py` 직접 실행이든
`ros2 launch tmah_vlm tmah_vlm.launch`든 재빌드 없이 즉시 반영됨.
(`Dockerfile.tmah` 자체를 바꿀 때만 재빌드 필요.)

### 3. `ai_module/debug` 권한 — 호스트(uid 1000)·컨테이너(uid 1001) 양쪽 다 필요
바인드 마운트 디렉터리는 **양쪽 uid 모두에게 ACL로 rwx를 줘야** 함. 하나만 걸면 반대쪽이 막힘.
```bash
sudo chown -R kante:kante ai_module/debug   # 기존 1001 소유 파일 정리(1회성)

setfacl -R    -m u:kante:rwx ai_module/debug   # 호스트(kante, uid 1000)
setfacl -R -d -m u:kante:rwx ai_module/debug

setfacl -R    -m u:1001:rwx  ai_module/debug   # 컨테이너(docker, uid 1001)
setfacl -R -d -m u:1001:rwx  ai_module/debug
```
`-d`(default ACL)라 하위에 새로 생기는 파일/폴더에도 자동 상속됨.

**재발 시**: 같은 패턴(host bind mount + 다른 uid로 도는 컨테이너)의 새 마운트 경로가 생기면
위 setfacl 4줄을 그대로 그 경로에 적용.

컨테이너 유저를 호스트 uid(1000)로 맞추는 방식(`user: "1000:1000"`)은 베이스 이미지의
`/home/docker`가 750/`docker:docker` 소유라 `$HOME`, `~/.ros` 접근이 깨져서 안 씀.

## 3D 위치 추정 정확도 개선 (2026-07-09)

**v_fov / TF 실측 보정**
- `config.PANO_V_FOV_DEG`: 120 → 180. LiDAR point를 카메라 이미지에 직접 투영해서
  벽/천장 경계선과 비교한 결과 180이 실측과 맞음 (120은 어긋남).
- `config.STATIC_TF_FALLBACKS`의 `sensor→camera` z: 0.85 → 0.1. 컨테이너에서
  `ros2 run tf2_ros tf2_echo sensor camera`로 라이브 TF 실측해서 확인함 (0.85는 틀린 값이었음,
  회전 쿼터니언은 원래 맞았음). fallback이라 평소엔 안 쓰이지만 live TF 준비 안 된 순간엔 이 값이 씀.

**회전 중 TF 어긋나는 버그** (`geometry/coordinate_transform.py`, 구 `tf/`)
- 원인: `get_matrix()`가 항상 `Time()`(=최신)으로 TF 조회. GroundingDINO+Qwen 추론에
  수백ms~수초 걸리는데, 그 사이 로봇이 회전하면 "센서 캡처 시각"이 아니라 "추론 끝난 시각"의
  TF를 써서 투영이 밀림 (정지 상태에선 안 드러남).
- 수정: `get_matrix`/`transform_point`/`transform_points`/`transform_direction`/
  `get_frame_origin`에 `stamp` 파라미터 추가. 스캔 관련은 `scan_msg.header.stamp`, camera ray
  관련은 `image_stamp` 사용. 캡처 시각 TF가 없으면 최신 TF → static fallback 순으로 degrade.
  `t3_object_reference_solver/t3_object_reference.py`의 `grab_camera_image()`(구
  `prepare_image()`)가 이미지와 stamp를 같은 스냅샷에서 같이 꺼내는 것도 이 때문
  (콜백이 계속 최신값으로 덮어쓰므로 따로 읽으면 어긋남).

**좌표 재구성 방식 개선** (`geometry/projector.py`, 구 `grounding/`)
- 기존: "box 중심 ray * median depth"로 좌표 재구성 (물체가 정확히 ray 위에 있다고 가정).
- 변경: `weighted_centroid_target()`이 선택된 depth cluster의 **실제 3D point들의 weighted
  centroid**를 사용 (가중치는 depth bin 선택에 쓰던 `center_ray_weights` 재사용).

## 3D bounding box 크기 추정 — `geometry/bbox_*` (2026-07-09 신규, 07-10 `geometry/`로 이동)

RViz marker가 원래 크기를 몰라서 고정 0.4m 큐브였음. 이제:
- `geometry/bbox_estimator.py`(구 `bbox3d/estimator.py`): 선택된 cluster point들의 5~95
  percentile 범위로 robust 크기 추정 (min/max 그대로 쓰면 이상치 point 하나에 박스가 확 커짐).
  `config.BBOX3D_*`로 튜닝 (`MIN_POINTS`, `PERCENTILE`, `MIN_SIZE_M`/`MAX_SIZE_M`,
  추정 실패 시 `DEFAULT_SIZE_M`).
- `geometry/bbox_wireframe.py`(구 `bbox3d/wireframe.py`): center+size → 12개 모서리
  `LINE_LIST` 좌표(24점) 생성.

## RViz marker 토픽 — `/selected_object_marker`는 절대 `MarkerArray`로 바꾸지 말 것

`dummy_vlm`(`ai_module/src/dummy_vlm/src/dummyVLM.cpp`)과 챌린지 자체
`visualizationTools` 노드(`iros2026_system` 컨테이너 안)가 이미 이 토픽을
**`Marker`(단수) 타입**으로 구독 중인 고정 규격이다. 한 번 `MarkerArray`로 바꿨다가
두 구독자 다 조용히 연결이 끊기는 문제를 겪었음 (타입이 다르면 에러 없이 그냥 안 붙음 —
LaserScan/PointCloud2 미스매치 때와 같은 패턴). `ros2 topic info <topic> -v`로 구독자
타입을 먼저 확인하는 습관을 들일 것.

현재 구조: `/selected_object_marker`는 `Marker`(CUBE, 초록) 그대로 유지, wireframe은
별도 토픽 `/selected_object_marker_wireframe`(`Marker`, `LINE_LIST`)로 분리해서 발행.
발행 코드는 `t3_object_reference_solver/publish.py`의 `publish_object_marker()`
(구 `handlers/object_reference.py`의 `publish_marker()`).

## 코드 구조 리팩터링 (2026-07-09~10)

`ai_module/src/tmah_vlm/tmah_vlm/` 전체 구조. 클래스 메서드 대신 **`node`를 인자로 받는
자유 함수** 패턴 (C++ 스타일 — Process 함수가 이름 있는 함수들을 flat하게 순서대로 호출).
전체 개요는 저장소 루트 `readme_kante.md`의 "tmah_vlm 코드 구조" 섹션 참고.

**진입점 / 질문 흐름**
- `main_node.py`(구 `vlm_node.py`) — 조립 + `main_control_loop` + `dispatch_question`만.
  질문은 `node/callbacks.py`의 `question_callback`이 저장만 → 0.2초 타이머
  `main_control_loop`이 꺼내 → `dispatch_question`이 첫 단어로 분기(find / how many·count / 그 외).
  (07-13에 `node/` 폴더 자체가 해체됐음 — 아래 "2차 코드 구조 리팩터링" 섹션 참고.
  `question_callback`은 이제 `common/callback.py`, `main_timer` 등록은 `main_node.py`의
  `main()` 함수로 옮김.)

**질문 유형별 solver (구 `handlers/`)** — 폴더 하나 = 유형 하나
- `t1_instruction_solver/`(instruction_process, stub) / `t2_numerical_solver/`
  (numerical_process) / `t3_object_reference_solver/`(object_reference_process + `publish.py`)
- 각 `*_process`는 **조건문 + 함수 호출만** 나열 (함수 이름 = 파이프라인 순서).
- **context 구조체 + 출력-인자 스타일**: `context/context.py`(구 `node/context.py`)의 `make_*_context()` 팩토리 함수가
  작업변수를 다 담은 ctx(SimpleNamespace 구조체 — 클래스 아님)를 질문마다 새로 만들고, 각 스텝
  함수는 ctx를 받아 **자기 필드 하나만 채운다**(return 대신 인자
  업데이트) → 다음 함수가 그 필드를 읽어 씀. `x = get_something(node)`처럼 함수 안에서 값을
  뽑아 쓰지 않고, 중간값(예: robot_pose)도 ctx 필드로 올려 전용 스텝으로 분리한다.

**노드 뼈대 → `node/` (07-10에 4개 1-파일 폴더를 통합, 07-13에 다시 해체됨 — 아래 2차 리팩터링 섹션 참고)**
- ~~`node/setup.py`~~ → `common/initialize.py`
- ~~`node/callbacks.py`~~ → `common/callback.py` + `perception/camera/callback.py` + `perception/lidar/callback.py`
- ~~`node/helpers.py`~~ → `common/helpers.py` + `context/helpers.py`
- ~~`node/context.py`~~ → `context/context.py`

**도메인 폴더 (07-10: 1-파일 폴더 9개를 4개로 통합, top-level 15개 → 9개. 07-13에 perception/geometry 다시 일부 변경됨)**
- `perception/` = 구 perception + segmentation(segmenter) + reasoning(selector) — 2D 인식
  (07-13에 `perception/camera/`로 이동, `perception/lidar/`도 신설 — 아래 참고)
- `geometry/` = 구 tf + grounding + bbox3d — 3D 기하 (`coordinate_transform`, `projector`;
  `bbox_estimator`/`bbox_wireframe`는 07-13에 `perception/lidar/`로 이동)
- `spatial/` = 구 spatial_reasoning + object_filter — 공간관계 파싱·필터
- `graph/`, `sort3d/` = 파일 많은 진짜 서브시스템이라 그대로 유지

**새 기능 추가 규칙**: 폴더 하나 = 도메인 하나, 함수는 `node`를 명시적으로 받고, solver
진입점은 `<파일이름>_process(node, question)` + ctx(출력-인자) 패턴 유지.

죽은 코드 삭제(07-09): `grounding/panorama.py`, `grounding/projector_backup_0708_2014.py`.

주의: 구조가 크게 바뀌었으니 컨테이너에서 `colcon build --symlink-install` 한 번 재실행 후
`ros2 launch tmah_vlm tmah_vlm.launch`로 확인할 것.

## SAM 세그멘테이션 핑크가 안 나오던 문제 (2026-07-10, 해결됨)

증상: 오버레이 디버그 이미지(`proj_*.jpg`)에 세그멘테이션 실루엣(마젠타/핑크)이 안 뜨고,
3D method가 `segmentation_mask_*`가 아니라 `bbox_ray_bundle_*`(폴백)로 떨어짐. heartbeat엔
`segmenter=ok`라 정상처럼 보임. 원인이 2개 겹쳐 있었고 **둘 다 별개**로 잡아야 했음.

### 1. GPU Out of Memory (진짜 원인)
7.52GB GPU에 GroundingDINO + Qwen2.5-VL이 이미 ~6.2GB 상주. SAM 로드는 되지만
**추론(`segment()`) 시점에 768MB를 못 잡아 매 호출 CUDA OOM** → 예외 → 마스크 None →
핑크 없음 + ray 폴백. `segment_selected_object`가 예외를 조용히 삼켜 로그로만 티가 났음
(진단용으로 성공/미로딩/예외+traceback을 매 쿼리 찍도록 로그 보강해둠).
→ **SAM만 CPU로** 돌려 해결: `config.SEGMENTATION_DEVICE = "cpu"` 추가 →
`common/initialize.py`(구 `node/setup.py`)의 `SAMSegmenter(device=config.SEGMENTATION_DEVICE)`로 전달.
세그멘테이션은 쿼리당 1회라 CPU로 ~4.5초는 감수 가능. GPU 여유 생기면 `"cuda"`로 되돌리면 됨.
확인법: 로드 시 `SAM segmenter loaded (device=cpu)`, 쿼리 시 `[ObjectRef] segmentation mask ok: ...px`,
method가 `segmentation_mask_centroid_mode`, 오버레이 범례에 `magenta=segmentation mask`.

### 2. `install/`이 stale 복사본 — 오늘 수정이 하나도 안 돌고 있었음
`--symlink-install`인데도 `install/.../site-packages/`에 예전 plain-build 때의 `tmah_vlm/`
**복사 디렉터리**(구조: `handlers/`, `vlm_node.py` = 리팩터 전)가 남아 egg-link(→build)를
가리고 있었음. 그래서 노드가 하루 종일 Jul-9 옛 코드를 돌렸고 오늘 수정이 전혀 반영 안 됨.
→ **클린 재빌드**로 해결: `rm -rf build/tmah_vlm install/tmah_vlm && colcon build --symlink-install --packages-select tmah_vlm`

**중요(CLAUDE.md 기존 설명 정정)**: 이 환경의 `--symlink-install`은 python 모듈을 **복사본으로**
깐다(`build/tmah_vlm/tmah_vlm/*.py`가 symlink 아님, `readlink`로 확인). 즉 "재빌드 없이 즉시 반영"은
**안 맞음** — src 수정 후엔 매번 `colcon build --symlink-install --packages-select tmah_vlm` 필요.

**추가(2026-07-13)**: 위 클린 재빌드(`rm -rf build/tmah_vlm install/tmah_vlm && colcon build ...`)가
`PermissionError: ... hook/ament_prefix_path.ps1`류 에러로 실패할 수 있음. 원인: `build/`, `install/`
(bind mount 아닌 컨테이너 내부 파일)이 이전에 `root`로 빌드된 적이 있어 `root` 소유로 남아있는데,
`docker exec`는 기본으로 `docker`(uid 1001) 유저로 들어가서 덮어쓸 권한이 없음.
→ 재빌드 전에 root로 한 번 지우고 소유권 정리:
```bash
docker exec -u root iros2026_tmah_module bash -lc \
  "rm -rf /home/docker/ai_module/build/tmah_vlm /home/docker/ai_module/install/tmah_vlm && \
   chown -R docker:docker /home/docker/ai_module/build /home/docker/ai_module/install"
```
그다음 평소처럼 (uid 1001로) `colcon build --symlink-install --packages-select tmah_vlm` 실행.

### 3. orphan 프로세스가 GPU를 물고 안 죽음
`pkill`이나 창 닫기로 노드를 죽이면 launch가 띄운 자식 python 프로세스가 init으로 reparent돼
살아남아 GPU(~6GB)를 계속 점유 → 다음 실행이 무조건 OOM("Process NNN has 6.19 GiB in use").
→ 재시작 전 반드시 확인·정리:
```bash
docker exec iros2026_tmah_module bash -lc "nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
# 남은 pid 있으면 명시적으로: kill -9 <pid> (pkill로 안 죽는 경우 있음)
```

## 2차 코드 구조 리팩터링 — `node/` 해체, `perception/`을 camera·lidar로 분리 (2026-07-13)

목적: `node/` 안에 성격이 다른 코드(초기화, 콜백, 상태조회, ctx생성)가 한 폴더에 섞여있었고,
`helpers`라는 이름도 무슨 함수가 들어있는지 헷갈렸음. 센서 종류(camera/lidar)별로 나눠서
어떤 코드가 어느 센서를 다루는지 파일 위치만 보고 알 수 있게 재구조화.

**새 구조**
- `common/`(구 `node/`) — 특정 센서에 안 속하는 노드 뼈대·수명주기.
  - `initialize.py`(구 `node/setup.py`) — 초기화·모델 백그라운드 로딩·구독/발행 등록.
  - `callback.py`(구 `node/callbacks.py`의 일부) — `question_callback`, `pose_callback`
    (센서별 콜백은 아래 `perception/*/callback.py`로 이동).
  - `helpers.py`(구 `node/helpers.py`의 일부) — **"get 계열"** 공용 상태 조회:
    `get_robot_pose`, `get_synced_scan_for_latest_image`, `get_scan_points_in_map`,
    `heartbeat`, `stamp_to_sec`.
- `context/`(신규) — 질문 처리 흐름 관련.
  - `context.py`(구 `node/context.py`, 내용 그대로) — solver별 ctx 생성 함수 `make_*_context()`.
  - `helpers.py`(구 `node/helpers.py`의 나머지) — **"질문 텍스트/처리 준비 판단"**:
    `peek_pending_question`, `ready_to_process`, `print_waiting_reason`.
- `perception/camera/`(구 `perception/`의 기존 6개 파일 그대로 이동) — `detector.py`,
  `image_utils.py`, `query_parser.py`, `segmenter.py`, `selector.py`, `visualize.py` +
  `callback.py`(구 `node/callbacks.py`의 `image_callback`).
- `perception/lidar/`(신규, 구 `geometry/bbox_*.py` 이동) — `bbox_estimator.py`,
  `bbox_wireframe.py` + `callback.py`(구 `node/callbacks.py`의 `scan_callback`).
- `geometry/`에는 `coordinate_transform.py`, `projector.py`만 남음 — camera/lidar ray를
  같이 다루는 공통 기하코드라 perception 밑으로 안 내리고 그대로 유지하기로 결정.

**의사결정 메모**
- 처음엔 "센서에 안 속하는 공용 인프라" 폴더 이름을 `global/`로 하려 했으나, **`global`은
  파이썬 예약어라 `from tmah_vlm.global.callback import ...` 같은 dotted import가 SyntaxError
  가 남**. `common/`으로 변경.
- `main_control_loop`용 `create_timer(0.2, ...)` 등록 위치를 `TmahVLM.__init__`에서
  `main_node.py`의 `main()` 함수로 옮김 (`node = TmahVLM(); node.create_timer(0.2,
  node.main_control_loop)`). `health_timer`/`scene_graph_marker_timer`는 여전히
  `common/initialize.py`의 `initialize_timers()` 안에 있음 — `main_control_loop`만 예외.

**적용 방법**: `git mv`로 히스토리 유지하며 이동, 전체 import 22곳을 새 경로로 수정,
`py_compile` + 자체 스크립트로 모든 `tmah_vlm.*` import 경로/심볼 존재 여부 교차검증 완료.

주의: 구조가 또 한 번 크게 바뀌었으니 컨테이너에서 클린 재빌드
(`rm -rf build/tmah_vlm install/tmah_vlm && colcon build --symlink-install --packages-select tmah_vlm`,
권한 에러 나면 위 "SAM 세그멘테이션" 섹션의 07-13 추가 항목 참고) 후
`ros2 launch tmah_vlm tmah_vlm.launch`로 확인할 것. 저장소 루트 `readme_kante.md`의
"tmah_vlm 코드 구조" 섹션도 아직 이 변경 전 내용이라 별도로 갱신 필요.

## 3차 코드 구조 리팩터링 — 기능 단위 재편, top-level 10 → 7 (2026-07-14)

목적: t1~t3 solver 외 top-level 폴더(`common`/`context`/`geometry`/`perception`/`spatial`/
`graph`/`sort3d`)가 너무 많아 코드 흐름이 안 보였음. **동작 변경 없이(pure move/rename)**
기능 단위로 다시 묶어 질문 하나가 처리되는 흐름을 파일 위치만으로 따라갈 수 있게 함.
(위 07-13 "2차 리팩터링" 및 그 이전 섹션의 경로 설명은 이 변경으로 낡음 — 아래가 최신.)

**새 구조 (top-level)**
- `common/` — 센서 무관 노드 뼈대: `initialize`, `callback`(question/pose), `helpers`
  (get_robot_pose/heartbeat/stamp_to_sec).
- `question_process/`(신규) — 문장 → 미션 선택. `dispatch.py`(구 `main_node.dispatch_question`
  + 구 `context/helpers`의 peek/ready/print_waiting), `query_parser.py`(구
  `perception/camera/query_parser`), `context.py`(구 `context/context`, make_*_context).
- `sensor_process/`(신규) — 카메라→2Dbox→seg→LiDAR변환→ray→3D 위치. 구
  `perception/camera/*`(query_parser 제외) + `perception/lidar/bbox_*` + `geometry/*`
  + `common/scan_transform` + 두 콜백을 합친 `callback.py`를 전부 이 아래로 이동.
  **흐름파일 `sensor_process/sensor_process.py`** 신설: t2/t3에 흩어져 중복되던 공용 센서 스텝
  (grab_camera_image / detect_candidate_boxes / load_scan_points_in_map /
  segment_selected_object / estimate_target_3d_pose)을 파이프라인 순서로 모아 두 solver가
  import해서 씀(solver 파일이 얇아짐).
- `nav_publish.py`(신규, top-level 단일 파일) — 세 solver에 흩어졌던 waypoint/marker/count
  발행을 통일. `t3_object_reference_solver/publish.py`는 삭제. (scene graph 마커 발행은
  추론 디버그 시각화라 `reasoning/graph/visualizer.py`에 그대로 둠.)
- `reasoning/`(신규 부모) — `spatial/`·`graph/`·`sort3d/`를 이 아래로 이동(내부 구조 그대로).
  `reasoning/sort3d/reasoning/`(sort3d 내부 하위폴더)는 경로로 구분되니 그대로 둠.
- 없어진 폴더: `context/`, `geometry/`, `perception/`.

**부수 변경**
- 흩어진 상수 → `config.py` 흡수: `MARKER_CUBE_RGBA`/`MARKER_WIREFRAME_RGBA`/
  `MARKER_WIREFRAME_LINE_WIDTH_M`, `INSTRUCTION_FORWARD_DISTANCE_M`,
  `MAIN_LOOP_PERIOD_SEC`/`HEALTH_TIMER_PERIOD_SEC`/`SCENE_GRAPH_MARKER_PERIOD_SEC`.
- `setup.py` 엔트리포인트: `scene_graph_json_markers`가 `tmah_vlm.reasoning.graph.json_marker_node:main`
  로 갱신(`tmah_vlm = ...main_node:main`은 그대로). launch는 console-script 이름만 참조라 수정 없음.
- `main_node.py` 상단 docstring, 저장소 루트 `readme_kante.md` 구조 섹션 갱신 완료.

**검증(완료)**: 모든 이동 `git mv`(히스토리 유지). import 90곳 sed 일괄 갱신 후
`py_compile` + AST 교차검증 스크립트(모듈/심볼 존재)로 확인. 컨테이너에서 클린 재빌드
(`colcon build --symlink-install --packages-select tmah_vlm`) 성공, install source 후
main_node/json_marker_node 포함 핵심 15개 모듈 `importlib.import_module` 전부 OK.
남은 것: `ros2 launch`로 실 센서 런타임(find/how many 각 1건) 최종 확인.

## 알 수 없는 파일 변경 — 미해결

이번 세션 중 내가 직접 실행하지 않은 변경이 감지됐으나 원인 파악도, 정리도 안 된 상태:
- `ai_module/src/tmah_vlm/tmah_vlm/readme_occlusion_fix.md`, `readme_pipeline_structure.md`가
  삭제됨 (git엔 커밋된 상태로 남아있음, `git checkout -- <path>`로 복구 가능)
- 저장소 루트에 빈 파일 `use`가 새로 생김

다른 터미널에서 동시 작업 중 발생한 것으로 추정. 다음 세션에서 사용자에게 의도한 변경인지
확인하고 정리할 것 (건드리지 말고 먼저 물어볼 것).
