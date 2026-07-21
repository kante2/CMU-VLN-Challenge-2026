
# sysnav_ros2_mvp 실행 방법 (2026-07-21)

경로: `ai_module/src/sysnav_ros2_mvp/`
tmah_vlm과 별개 의존성 스택(YOLO-World + SAM2 + Gemini)이라 독립 컨테이너 `sysnav_module`로 분리됨.
자세한 파이프라인/토픽/상태 흐름은 `ai_module/src/sysnav_ros2_mvp/README.md` 참고.

## 0. 사전 준비 — `.env` 채우기 (완료됨, 07-21)

`ai_module/.env`에 Gemini API 키 채워넣음 (`.gitignore`에 이미 등록되어 있어 커밋 안 됨):
```
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
YOLO_WORLD_WEIGHTS=/home/docker/ai_module/weights/yolov8x-worldv2.pt
SAM2_CHECKPOINT=/home/docker/ai_module/weights/sam2.1_hiera_tiny.pt
SAM2_MODEL_CFG=configs/sam2.1/sam2.1_hiera_t.yaml
```
`sysnav_module`은 compose에서 이 파일을 `env_file`로 읽음. 키가 비어있으면 Gemini 후보 선택 단계가
confidence/거리 기반 fallback으로만 동작함 (에러는 안 남).

## 1. 컨테이너 빌드 & 기동

```bash
cd /home/kante/CMU-VLN-Challenge-2026/docker
xhost +
docker compose -f compose_gpu.yml up --build -d system sysnav_module
```
`system`(시뮬레이터+오토노미)과 `sysnav_module`만 띄우면 됨. `ai_module`/`tmah_module`은 무관한
별도 스택이라 안 띄워도 됨.

기존 컨테이너가 이미 있으면 (Exited 상태 등):
```bash
docker start iros2026_system iros2026_sysnav_module
```

**주의**: `.env`는 컨테이너 **생성 시점**에만 읽힘. 이미 생성된 `sysnav_module` 컨테이너가 있는
상태에서 `.env`를 또 고치면 `docker start`로는 반영 안 됨 — 재생성 필요:
```bash
docker compose -f compose_gpu.yml up -d --force-recreate sysnav_module
```


## ----------------------명령어 요약--------------------
터미널 A — 시뮬레이터 (이미 켜져있다면 생략)


docker exec -it iros2026_system bash
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
터미널 B — sysnav 실행 (컨테이너 재시작됐으니 새로 exec)


docker exec -it iros2026_sysnav_module bash
source /opt/ros/jazzy/setup.bash
source /home/docker/ai_module/install/setup.bash
ros2 launch sysnav sysnav.launch.py
터미널 C — 질의


docker exec -it iros2026_sysnav_module bash
source /opt/ros/jazzy/setup.bash
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Find the white chair'}"

## --------------------------------------------------

## 2. A — 시뮬레이터/autonomy 실행 (터미널 A)

```bash
docker exec -it iros2026_system bash
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
```
로봇이 여러 대 겹친 걸로 뜨면:
```bash
docker restart iros2026_system
```
그리고 다시 컨테이너 접근해서 재실행.

## 3. B — sysnav 노드 실행 (터미널 B)

`sysnav_module`은 `src/sysnav_ros2_mvp`를 바인드 마운트함. 이미지 빌드 시점 소스로 이미
`colcon build`가 한 번 끝난 상태지만, **호스트에서 소스를 수정했다면 마운트가 그 위를 덮어써서
컨테이너 안 `install/`은 옛 빌드 그대로**임 (tmah_module 때와 같은 패턴). 소스 수정 후엔 항상
재빌드:
```bash
docker exec -it iros2026_sysnav_module bash
source /opt/ros/jazzy/setup.bash
cd /home/docker/ai_module
colcon build --symlink-install --packages-select sysnav
source install/setup.bash
ros2 launch sysnav sysnav.launch.py
```
(또는 `ros2 run sysnav sysnav`)

## 4. C — 질문 던지기 (터미널 C, 또 새 창)

```bash
docker exec -it iros2026_sysnav_module bash
source /opt/ros/jazzy/setup.bash
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Find the white chair'}"
```
`network_mode: host`라 어느 컨테이너에서 쏴도 상관없음 (system, sysnav_module 다 같은 ROS2 네트워크).

## 5. 확인

- RViz에서 로봇이 `/way_point_with_heading`을 따라 이동하는지
- `ros2 topic echo /way_point_with_heading`
- 상태 흐름: `IDLE → OBSERVE → (후보 있음) SELECT_TARGET → NAVIGATE_TARGET → SUCCESS`
  또는 `(후보 없음) PLAN_EXPLORATION → FOLLOW_EXPLORATION → (새 관측 시) OBSERVE`로 순환

## 6. 디버그 이미지 — `ai_module/debug`에 detection 결과 저장 (2026-07-21 추가)

`sysnav/perception/perception_pipeline.py`의 `process()`가 매 perception job마다 (bbox +
segmentation mask + 3D position 텍스트) 오버레이 이미지를 `ai_module/debug/sysnav_detect_*.jpg`로
저장함 (`sysnav/perception/debug_visualize.py`의 `save_debug_image()`).
- 끄고 싶으면 `.env`에 `SYSNAV_SAVE_DEBUG_IMAGES=0` 추가.
- `compose_gpu.yml`의 `sysnav_module`에 `../ai_module/debug:/home/docker/ai_module/debug` 마운트
  추가함 (기존엔 `src`만 마운트되어 있었음). **컨테이너를 새로 만들어야 반영됨**
  (`docker compose -f compose_gpu.yml up -d --force-recreate sysnav_module`).
- 컨테이너 uid(1001)가 호스트(kante, uid 1000) 소유 폴더에 쓸 수 있어야 해서 ACL 추가함
  (tmah_module 때와 같은 패턴, 아래 명령 1회 실행 완료됨):
  ```bash
  setfacl -R    -m u:1001:rwx  ai_module/debug
  setfacl -R -d -m u:1001:rwx  ai_module/debug
  ```
## 주의사항

1. **`sysnav/config.py`의 `T_LIDAR_TO_CAMERA`, `T_SENSOR_TO_BASE`,
   `PANORAMA_YAW_OFFSET_DEG`/`PANORAMA_PITCH_OFFSET_DEG`가 예시값** — 실측 TF 확인 없이 그대로
   쓰면 3D grounding이 어긋날 수 있음. tmah_vlm 쪽에서 v_fov/TF 실측 보정한 이력 있음
   (CLAUDE.md "3D 위치 추정 정확도 개선" 섹션 참고) — 같은 방식으로 확인 필요.
2. RTX 8GB 환경이라 SAM2 tiny 체크포인트(`sam2.1_hiera_tiny.pt`)로 세팅되어 있음
   (Dockerfile.sysnav 빌드 시점에 미리 다운로드됨, 재검증 완료).
3. 재시작 전 orphan 프로세스가 GPU 물고 있는지 확인하는 습관 (tmah_vlm 쪽에서 겪었던 문제,
   `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`).
4. GPU가 통째로 죽는 경우(`nvidia-smi`가 "Unknown Error"로 실패, 컨테이너 문제 아님) 발생한 적
   있음 — 이땐 host 재부팅으로 해결됨. 컨테이너/이미지 재빌드로는 안 고쳐짐.
