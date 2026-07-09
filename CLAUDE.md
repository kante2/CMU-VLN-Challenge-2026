# CMU-VLN-Challenge-2026

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

**회전 중 TF 어긋나는 버그** (`tf/coordinate_transform.py`)
- 원인: `get_matrix()`가 항상 `Time()`(=최신)으로 TF 조회. GroundingDINO+Qwen 추론에
  수백ms~수초 걸리는데, 그 사이 로봇이 회전하면 "센서 캡처 시각"이 아니라 "추론 끝난 시각"의
  TF를 써서 투영이 밀림 (정지 상태에선 안 드러남).
- 수정: `get_matrix`/`transform_point`/`transform_points`/`transform_direction`/
  `get_frame_origin`에 `stamp` 파라미터 추가. 스캔 관련은 `scan_msg.header.stamp`, camera ray
  관련은 `image_stamp` 사용. 캡처 시각 TF가 없으면 최신 TF → static fallback 순으로 degrade.
  `handlers/object_reference.py`의 `prepare_image()`가 이미지와 stamp를 같은 스냅샷에서
  같이 꺼내는 것도 이 때문(콜백이 계속 최신값으로 덮어쓰므로 따로 읽으면 어긋남).

**좌표 재구성 방식 개선** (`grounding/projector.py`)
- 기존: "box 중심 ray * median depth"로 좌표 재구성 (물체가 정확히 ray 위에 있다고 가정).
- 변경: `weighted_centroid_target()`이 선택된 depth cluster의 **실제 3D point들의 weighted
  centroid**를 사용 (가중치는 depth bin 선택에 쓰던 `center_ray_weights` 재사용).

## 3D bounding box 크기 추정 — `bbox3d/` (2026-07-09 신규)

RViz marker가 원래 크기를 몰라서 고정 0.4m 큐브였음. 이제:
- `bbox3d/estimator.py`: 선택된 cluster point들의 5~95 percentile 범위로 robust 크기 추정
  (min/max 그대로 쓰면 이상치 point 하나에 박스가 확 커짐). `config.BBOX3D_*`로 튜닝
  (`MIN_POINTS`, `PERCENTILE`, `MIN_SIZE_M`/`MAX_SIZE_M`, 추정 실패 시 `DEFAULT_SIZE_M`).
- `bbox3d/wireframe.py`: center+size → 12개 모서리 `LINE_LIST` 좌표(24점) 생성.

## RViz marker 토픽 — `/selected_object_marker`는 절대 `MarkerArray`로 바꾸지 말 것

`dummy_vlm`(`ai_module/src/dummy_vlm/src/dummyVLM.cpp`)과 챌린지 자체
`visualizationTools` 노드(`iros2026_system` 컨테이너 안)가 이미 이 토픽을
**`Marker`(단수) 타입**으로 구독 중인 고정 규격이다. 한 번 `MarkerArray`로 바꿨다가
두 구독자 다 조용히 연결이 끊기는 문제를 겪었음 (타입이 다르면 에러 없이 그냥 안 붙음 —
LaserScan/PointCloud2 미스매치 때와 같은 패턴). `ros2 topic info <topic> -v`로 구독자
타입을 먼저 확인하는 습관을 들일 것.

현재 구조: `/selected_object_marker`는 `Marker`(CUBE, 초록) 그대로 유지, wireframe은
별도 토픽 `/selected_object_marker_wireframe`(`Marker`, `LINE_LIST`)로 분리해서 발행.

## 코드 구조 리팩터링 (2026-07-09)

폴더를 기능별로 분리하고, 클래스 메서드 대신 **`node`를 인자로 받는 자유 함수** 패턴으로
통일함 (다른 프로젝트의 C++ 스타일 — Process 함수가 이름 있는 함수들을 flat하게 순서대로
호출하는 패턴 — 을 참고).

- `initialize/setup.py` — `TmahVLM.__init__()`이 부르는 `initialize_state/modules/
  subscribers/publishers/timers` + `load_models`
- `callback/sensor_callbacks.py` — 센서/질문 콜백 (최신값 저장만)
- `helper/node_helpers.py` — `main_control_loop`와 `handlers/*.py`가 공용으로 쓰는
  상태 조회 함수 (`get_robot_pose`, `get_synced_scan_for_latest_image`, `heartbeat` 등)
- `handlers/*.py` — 진입점 함수명 통일: `handle` → `object_reference_process` /
  `numerical_process` / `instruction_process` (파일 이름 = 함수 이름 접두어)
- `vlm_node.py`는 이제 이 함수들을 부르는 "조립"만 함 (453줄 → 171줄)
- 죽은 코드 삭제: `grounding/panorama.py`, `grounding/projector_backup_0708_2014.py`
  (아무 데서도 안 쓰던 파일)

새 기능 추가할 때는 이 패턴 유지: 폴더 하나 = 기능 하나, 함수는 `node`를 명시적으로 받고,
진입점은 `<파일이름>_process(node, ...)` 형태로.

## 알 수 없는 파일 변경 — 미해결

이번 세션 중 내가 직접 실행하지 않은 변경이 감지됐으나 원인 파악도, 정리도 안 된 상태:
- `ai_module/src/tmah_vlm/tmah_vlm/readme_occlusion_fix.md`, `readme_pipeline_structure.md`가
  삭제됨 (git엔 커밋된 상태로 남아있음, `git checkout -- <path>`로 복구 가능)
- 저장소 루트에 빈 파일 `use`가 새로 생김

다른 터미널에서 동시 작업 중 발생한 것으로 추정. 다음 세션에서 사용자에게 의도한 변경인지
확인하고 정리할 것 (건드리지 말고 먼저 물어볼 것).
