# SysNav 실행 가이드 (Docker Hub 제출 이미지 기준)

CMU VLN Challenge 2026 — 터미널 A(시뮬레이터) / B(sysnav 노드) / C(질의) 3창 구성으로 실행합니다.
이 문서는 **Docker Hub에 올린 제출 이미지를 그대로 받아서 실행**하는 절차만 정리한 것입니다.
(로컬 소스를 고쳐가며 개발하는 절차는 `readme_kante2.md` 참고.)

---

## 0. 제출 이미지

- **Docker Hub**: https://hub.docker.com/r/parkjaeil00/cmu-vln-2026-sysnav
- **이미지**: `parkjaeil00/cmu-vln-2026-sysnav:submission-v1` (`latest`도 동일 내용)
- linux/amd64, 압축 9.1GB / 전개 27.4GB
- 이미지 안에 **모델 가중치와 빌드 결과가 모두 포함**되어 있어 실행 시 추가 다운로드·빌드가 없습니다.
  - `yolov8x-worldv2.pt` (YOLO-World), `sam2.1_hiera_tiny.pt` (SAM2)
  - `/home/docker/ai_module/install/sysnav` (colcon build 완료 상태)
  - `USER=docker`, `WORKDIR=/home/docker/ai_module`

```bash
docker pull parkjaeil00/cmu-vln-2026-sysnav:submission-v1
```

시뮬레이터 이미지는 챌린지 공식 이미지를 그대로 씁니다:
`zhangjicmu/ubuntu24_ros:cmu_vla_challenge_simulation`

---

## 1. `.env` 준비 — **API 키는 별도 전달**

`sysnav`는 질의 파싱/후보 선택에 Google Gemini API를 사용합니다. 키는 저장소에 커밋하지 않으므로
(`.gitignore`에 `ai_module/.env` 등록됨) **API 키는 운영진 측에 별도 채널로 전달**하며,
아래 스크립트로 `.env`를 만든 뒤 전달받은 키를 채워 넣으면 됩니다.

```bash
./docker/create_env.sh                # GEMINI_API_KEY= (빈칸) 상태로 ai_module/.env 생성
./docker/create_env.sh <API_KEY>      # 전달받은 키를 바로 채워서 생성
./docker/create_env.sh --force        # 이미 있는 .env 덮어쓰기 (기본은 덮어쓰지 않고 종료)
```

생성되는 `ai_module/.env` (mode 600):

```
GEMINI_API_KEY=          # ← 별도 전달받은 키를 여기에
GEMINI_MODEL=gemini-3.5-flash
YOLO_WORLD_WEIGHTS=/home/docker/ai_module/weights/yolov8x-worldv2.pt
SAM2_CHECKPOINT=/home/docker/ai_module/weights/sam2.1_hiera_tiny.pt
SAM2_MODEL_CFG=configs/sam2.1/sam2.1_hiera_t.yaml
```

템플릿 원본은 저장소에 커밋된 `ai_module/.env.example`입니다 (스크립트 없이 복사해서 써도 됩니다).

> **주의 1** — `.env`는 docker compose가 컨테이너를 **생성하는 시점에만** 읽습니다.
> 컨테이너를 만든 뒤 키를 바꿨다면 재생성해야 반영됩니다:
> `docker compose -f docker/compose_gpu.yml up -d --force-recreate sysnav_submission`
>
> **주의 2** — 키가 비어 있어도 노드는 뜨고 동작은 합니다. 다만 LLM 질의 파싱/후보 선택이
> confidence·거리 기반 fallback으로 떨어져 성능이 낮아집니다 (에러는 나지 않음).

---

## 2. 컨테이너 기동

```bash
xhost +                              # 시뮬레이터/RViz GUI용 X 권한
./docker/start_containers.sh         # system + sysnav_submission 기동
```

`start_containers.sh` 동작: 이미 실행 중이면 건너뛰고, 정지 상태면 `docker start`,
컨테이너가 없으면 `docker compose -f docker/compose_gpu.yml up -d <서비스>`로 생성합니다.

| 인자 | 기동 대상 |
| --- | --- |
| (없음) 또는 `submission` | `iros2026_system` + `iros2026_sysnav_submission` (제출 이미지, 기본) |
| `dev` | `iros2026_system` + `iros2026_sysnav_module` (로컬 빌드 개발용) |
| `both` | 셋 다 |

compose 서비스 `sysnav_submission`은 제출 이미지 전용입니다 — **로컬 빌드를 하지 않고**,
호스트 소스를 바인드 마운트하지 **않아서** 이미지 안의 제출본 코드가 그대로 실행됩니다.
(`ai_module/debug` 폴더와 `.env`만 컨테이너로 들어갑니다.)

---

## 3. 터미널 A — 시뮬레이터

```bash
./docker/A_시뮬레이터.sh                   # 현재 씬 그대로 실행
./docker/run_scene.sh hotel_room_1        # 씬을 바꿔서 실행 (map/<씬이름>.zip 필요)
```

씬 이름 예: `hotel_room_1`, `home_building_1`, `office_2`, `livingroom_1` …
맵 zip은 저장소의 `map/` 폴더에 두면 컨테이너 안 `/home/docker/maps`로 마운트됩니다.

---

## 4. 터미널 B — sysnav 노드 (제출 이미지)

```bash
./docker/B_sysnav_실행_제출이미지.sh
```
- 컨테이너가 없으면 자동 생성한 뒤 `ros2 launch sysnav sysnav.launch.py` 실행
- 이미지 안에 빌드가 끝나 있으므로 **colcon 재빌드 없음**
- 참고: 로컬 빌드 개발용은 `./docker/B_sysnav_실행.sh` (컨테이너 안에서 colcon build 후 launch)

정상 기동 시 로그: `[sysnav_node]: SysNav single-room MVP started`
질의를 받아 인식이 시작되면 ultralytics(YOLO-World) 배너(🚀)와 모델 로딩 로그가 출력됩니다.

---

## 5. 터미널 C — 질의

```bash
./docker/C_질의.sh                          # 실행 중인 sysnav 컨테이너 자동 선택 후 접속
./docker/C_질의.sh iros2026_sysnav_submission  # 컨테이너 직접 지정
```

접속한 셸에서 질문을 발행합니다 (아래는 `hotel_room_1` 기준 예시):

```bash
# Mission 1 — Numerical → /numerical_response (Int32)
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'How many pillows are on the bed?'}"

# Mission 2 — Object Reference → /selected_object_marker (Marker)
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Find the bedside table farthest from the window.'}"

# Mission 3 — Instruction-Following → /way_point_with_heading (Pose2D 시퀀스)
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Go to the bedside table closest to the window and stop at the chair closest to the TV.'}"
```

미션 타입은 문장으로 자동 분류됩니다 (`sysnav/task/mission_classifier.py`).
씬별 실제 채점 문장 전체는 `questions/questions.json` 및 `readme_kante2.md` 8번 섹션에 있습니다.

---

## 6. 결과 확인

```bash
# 응답 토픽 (창을 하나 더 띄워서)
./docker/C_질의.sh
ros2 topic echo /numerical_response
ros2 topic echo /selected_object_marker
ros2 topic echo /way_point_with_heading
```

- **미션 상태 대시보드** (호스트 브라우저): `./docker/ui_checker.sh`
  또는 `xdg-open ai_module/debug/mission_status_latest.html` — 1초마다 자동 새로고침되며
  현재 미션/상태/경과시간/후보 수/선택된 물체를 한 화면에서 보여줍니다.
- **디버그 이미지**: `ai_module/debug/sysnav_detect_*.jpg` (bbox + segmentation + 3D 좌표 오버레이).
  끄려면 `.env`에 `SYSNAV_SAVE_DEBUG_IMAGES=0`.
- **RViz**: 로봇이 `/way_point_with_heading`을 따라 이동하는지 확인.

---

## 7. 트러블슈팅

### 7-1. 토픽 목록은 보이는데 데이터가 안 옴 (가장 흔함)

증상: sysnav 컨테이너에서 `ros2 topic list`에는 `/camera/image`, `/state_estimation`이 보이는데
`ros2 topic hz`는 계속 `does not appear to be published yet`. sysnav 노드가 `OBSERVE`에서 멈추고
GPU 사용량이 0이며 ultralytics 배너(🚀)도 안 뜸.

원인: 시뮬레이터 컨테이너(`iros2026_system`)가 **IPC 네임스페이스를 호스트와 공유하지 않는 상태**로
생성됨. 두 컨테이너 모두 `network_mode: host`라 Fast DDS가 같은 머신으로 판단해 공유메모리(/dev/shm)
전송을 고르는데, IPC가 격리돼 있으면 **discovery(UDP)는 되고 샘플은 전달되지 않습니다.**

확인:
```bash
docker inspect iros2026_system --format 'IpcMode={{.HostConfig.IpcMode}}'   # host 여야 정상
```

해결 — compose에는 `ipc: host`가 이미 선언돼 있으므로 컨테이너를 **재생성**하면 됩니다
(설정은 생성 시점에만 적용되어 `docker restart`로는 안 바뀝니다):
```bash
xhost +
docker compose -f docker/compose_gpu.yml up -d --force-recreate system
./docker/A_시뮬레이터.sh        # 시뮬레이터 재기동
```

### 7-2. GPU를 물고 있는 orphan 프로세스

노드를 강제 종료하면 launch가 띄운 자식 프로세스가 살아남아 GPU를 계속 점유하고,
다음 실행이 CUDA OOM으로 실패할 수 있습니다.
```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
kill -9 <pid>
```

### 7-3. 로봇이 여러 대 겹쳐 보임 / 시뮬 중복 실행

```bash
pkill -9 -f autonomy_stack_mecanum_wheel_platform
pkill -9 -f static_transform_publisher
pkill -9 -f joy_node
pkill -9 -f default_server_endpoint
# 또는
docker restart iros2026_system
```

### 7-4. 이미지 안 소스가 제출본과 같은지 확인

```bash
docker exec iros2026_sysnav_submission bash -lc \
  "cd /home/docker/ai_module/src/sysnav_ros2_mvp && find . -name '*.py' | LC_ALL=C sort | xargs md5sum | md5sum"
(cd ai_module/src/sysnav_ros2_mvp && find . -name '*.py' | LC_ALL=C sort | xargs md5sum | md5sum)
```
(`submission-v1` 이미지와 커밋 `daaba3d`의 소스 76개 `.py`가 일치함을 확인했습니다.)

---

## 8. 전체 순서 요약

```bash
# 최초 1회
docker pull parkjaeil00/cmu-vln-2026-sysnav:submission-v1
./docker/create_env.sh <전달받은_GEMINI_API_KEY>

# 매 실행
xhost +
./docker/start_containers.sh

# 터미널 A
./docker/run_scene.sh hotel_room_1

# 터미널 B
./docker/B_sysnav_실행_제출이미지.sh

# 터미널 C
./docker/C_질의.sh
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Find the bedside table farthest from the window.'}"

# 상태 확인 (호스트)
./docker/ui_checker.sh
```
