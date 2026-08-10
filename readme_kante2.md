
## 빠른 실행 — sh 스크립트 3개 (2026-08-10 추가)

컨테이너가 이미 떠있는 상태(아래 1번 "컨테이너 빌드 & 기동" 완료 후)라면, 터미널 3개를
열고 각각 아래 스크립트만 실행하면 됨 (전부 `docker exec -it`로 컨테이너 안까지 들어가서
소싱까지 자동으로 함).

- **터미널 A** — 시뮬레이터: `./docker/A_시뮬레이터.sh`
  (씬을 바꿔서 켜고 싶으면 대신 `./docker/run_scene.sh <씬이름>`, 예: `./docker/run_scene.sh hotel_room_1`)
- **터미널 B** — sysnav 노드 실행: `./docker/B_sysnav_실행.sh`
  (컨테이너 안에서 `source /opt/ros/jazzy/setup.bash && source /home/docker/ai_module/install/setup.bash && ros2 launch sysnav sysnav.launch.py`를 그대로 실행)
- **터미널 C** — 질의: `./docker/C_질의.sh`
  (컨테이너 접속 + ROS2 소싱까지만 자동으로 해주고 셸을 넘겨줌 — `ros2 topic pub` 명령은 직접 입력.
  예: `ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: 'Find the bowl near the trash can.'}"`)
- **대시보드 확인** (호스트에서, 컨테이너 밖): `./docker/ui_checker.sh`

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

## start CONTAINER----------------------------------------------------
docker start iros2026_system iros2026_sysnav_module


## -----------------------------------------------------------------
# 시뮬 겹침 초기화
pkill -9 -f autonomy_stack_mecanum_wheel_platform
pkill -9 -f static_transform_publisher
pkill -9 -f joy_node
pkill -9 -f default_server_endpoint

## ----------------------코드 수정후 재빌드----------------------------
docker exec -it iros2026_sysnav_module bash
source /opt/ros/jazzy/setup.bash
cd /home/docker/ai_module
colcon build --symlink-install --packages-select sysnav
source install/setup.bash
ros2 launch sysnav sysnav.launch.py


## ----------------------명령어 요약--------------------
터미널 A — 시뮬레이터 (이미 켜져있다면 생략)

docker exec -it iros2026_system bash
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh

or
./docker/run_scene.sh <씬이름>
# 예: ./docker/run_scene.sh office_2
# 예: ./docker/run_scene.sh home_building_1


터미널 B — sysnav 실행 (컨테이너 재시작됐으니 새로 exec)


docker exec -it iros2026_sysnav_module bash
source /opt/ros/jazzy/setup.bash
source /home/docker/ai_module/install/setup.bash
ros2 launch sysnav sysnav.launch.py

터미널 C — 질의


docker exec -it iros2026_sysnav_module bash
source /opt/ros/jazzy/setup.bash
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Find the toilet'}"

ros2 topic pub --once /challenge_question std_msgs/msg/String \
"{data: 'Find the bowl closest to the knife rack near the trash can.'}"

ros2 topic pub --once /challenge_question std_msgs/msg/String \
"{data: 'Find the bowl near the trash can.'}"

ros2 topic pub --once /challenge_question std_msgs/msg/String \
"{data: 'First, go near the bedside table closest to the bench, then take the path between the TV and the bed to the picture closest to the TV.'}"



## --------------------------------------------------
## A - 시뮬레이션 킬때, 방을 변경하고 싶으면 다음 sh을 실행 
./docker/run_scene.sh home_building_1

./docker/run_scene.sh hotel_room_1

<위 실행 이전에 , 세팅 방법>
맵 zip 파일을 map/ 폴더에 넣기


/home/kante/CMU-VLN-Challenge-2026/map/<씬이름>.zip
예: map/office_2.zip

컨테이너 실행/최신화 (한 번만, 이미 떠있으면 생략)


docker compose -f docker/compose_gpu.yml up -d system   # GPU 없으면 compose.yml
map/ 폴더가 컨테이너 안 /home/docker/maps로 마운트되게 compose 파일에 이미 설정해놨어서, 이 명령 한 번이면 zip이 컨테이너에서 바로 보임.

실행

./docker/run_scene.sh <씬이름>
예: ./docker/run_scene.sh office_2


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
  "{data: 'Find the toilet'}"
```
`network_mode: host`라 어느 컨테이너에서 쏴도 상관없음 (system, sysnav_module 다 같은 ROS2 네트워크).

## 4-1. 미션별 질의 테스트 (2026-08-07 추가, mission dispatch 리팩터 이후)

문장 첫 부분으로 미션 타입(Numerical/Object Reference/Instruction-Following)을 자동
분류함 (`sysnav/task/mission_classifier.py`). 미션마다 나가는 토픽이 다름 - 셋 다
확인하려면 아래 세 문장을 각각 던져볼 것.

**Mission 1 - Numerical** → `/numerical_response` (Int32)
```bash
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'How many red pillows are on the sofa?'}"
```

**Mission 2 - Object Reference** → `/selected_object_marker` (Marker, 단수)
```bash
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Find the chair closest to the table.'}"
```

**Mission 3 - Instruction-Following** → `/way_point_with_heading` (Pose2D 시퀀스)
```bash
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Go to the chair near the window, then take the path near the table, and stop at the sofa.'}"
```
물체 이름은 지금 띄운 씬에 맞게 바꿀 것. `questions/questions.json`에 씬별 실제 채점
문장이 있으니 그걸 그대로 써도 됨 (씬 이름으로 검색).

## 5. 확인

- RViz에서 로봇이 `/way_point_with_heading`을 따라 이동하는지
- 미션별로 나가는 토픽 확인:
  - Numerical: `ros2 topic echo /numerical_response`
  - Object Reference: `ros2 topic echo /selected_object_marker`
  - Instruction-Following: `ros2 topic echo /way_point_with_heading` (여러 번 순서대로 나옴)
- 상태 흐름: `IDLE → OBSERVE → (후보 있음) SELECT_TARGET → NAVIGATE_TARGET → SUCCESS`
  또는 `(후보 없음) PLAN_EXPLORATION → FOLLOW_EXPLORATION → (새 관측 시) OBSERVE`로 순환
  (이건 Object Reference 기준. Numerical은 후보를 찾아도 안 멈추고 탐색이 완전히
  끝날 때까지 `PLAN_EXPLORATION`/`FOLLOW_EXPLORATION`을 반복하다가
  `MISSION1_FINALIZE_COUNT → SUCCESS`로 감. Instruction-Following은
  `MISSION3_SELECT_STEP ↔ MISSION3_NAVIGATE_STEP`을 목적지 개수만큼 반복하다가
  `SUCCESS`로 감. 자세한 건 `MISSION_1/2/3_*_CLAUDE.txt` 참고)
- 아래 7번 대시보드로 지금 어느 미션/상태인지 실시간으로 보는 게 제일 편함

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
## 7. 미션 상태 대시보드 (HTML, 2026-08-07 추가)

`sysnav/mission_dashboard.py`가 매 control_loop 사이클마다(1초 스로틀링)
`ai_module/debug/mission_status_latest.html`을 통째로 다시 써서 덮어씀 - 위 6번
디버그 이미지랑 같은 폴더라 마운트/ACL 추가 설정 필요 없음.

호스트에서 브라우저로 열면 됨 (터미널에 경로만 치면 bash가 실행하려다 Permission
denied 남 - 반드시 브라우저나 xdg-open으로 열 것):
```bash
xdg-open /home/kante/CMU-VLN-Challenge-2026/ai_module/debug/mission_status_latest.html
```
또는 브라우저 주소창에 `file:///home/kante/CMU-VLN-Challenge-2026/ai_module/debug/mission_status_latest.html`.

1초마다 자동 새로고침(`<meta refresh>`)됨. 지금 미션 타입/상태(색깔 배지)/10분 타임
리밋 진행바/미션별 상세(타겟, 후보 수, 선택된 물체, 현재 step)/마지막으로 발행한
응답값을 한 화면에서 볼 수 있음.

## 8. 씬별 실제 테스트 문장 (`questions/questions.json` 전체, 2026-08-07 추가)

채점 데이터셋에 있는 실제 문장 그대로. `1-Numerical`/`2-ObjRef`/`3-Instr` 아무거나
위 4-1번 `ros2 topic pub` 템플릿의 `data:` 값에 그대로 넣어서 쓰면 됨.

### arabic_room
- [1-Numerical] How many sofas are below a window?
- [2-ObjRef] Find the pillow closest to the book on the stool.
- [2-ObjRef] Find the wall lamp that is between a door frame and a window.
- [3-Instr] Go near the stool under the picture and stop at the small table farthest from the columns.
- [3-Instr] First, go to the potted plant furthest from the hookah, then take the path between the two columns, and stop at the tray on the table.

### chinese_room
- [1-Numerical] Count the number of chairs with pillows on them.
- [2-ObjRef] Find the bowl on the table closest to the folding screen.
- [2-ObjRef] Find the pillow on the chair that is closest to the TV.
- [3-Instr] Go near the potted plant on the table and stop at the painting near the TV.
- [3-Instr] First, go near the tea table with the elephant figurine on it, then stop at the table with the horse figurine on it, avoiding the path between the chair and the folding screen.

### home_building_1
- [1-Numerical] How many pillows are on the sofa under the pictures?
- [2-ObjRef] Find the clock on the TV cabinet.
- [2-ObjRef] Find the bowl closest to the knife rack near the trash can.
- [3-Instr] Go to the coffee table with the kettle on it and stop at the dining table near the big picture.
- [3-Instr] First, go to the nightstand with a clock on it, then take the path between the dining table and the picture, and stop at the trash can closest to the refridgerator.

### home_building_2
- [1-Numerical] How many red pillows are on the sofa?
- [2-ObjRef] Find the lamp on the nightstand that has the photo on it.
- [2-ObjRef] Find the speaker on the TV cabinet closest to the potted plant on the TV cabinet.
- [3-Instr] Go near the magazine on the ottoman, then go to the potted plant on the dressing table.
- [3-Instr] Take the path between the sofa and the coffee table and go to the kettle on the dining table, then go to the potted plant between the curtain and the TV.

### hotel_room_1
- [1-Numerical] How many pillows are on the bed?
- [2-ObjRef] Find the bedside table farthest from the window.
- [2-ObjRef] Find the picture above the suitcase furthest from the floor.
- [3-Instr] Go to the bedside table closest to the window and stop at the chair closest to the TV.
- [3-Instr] First, go near the bedside table closest to the bench, then take the path between the TV and the bed to the picture closest to the TV.

### hotel_room_2
- [1-Numerical] How many pictures are above the bed?
- [2-ObjRef] Find the flowers near the window.
- [2-ObjRef] Find the picture closest to the bench.
- [3-Instr] Go between the bench and the bed and stop at the lamp closest to the fireplace.
- [3-Instr] First, go to the picture closest to the door, then take the path between the TV cabinet and the bed, and stop by the curtain closest to the TV.

### japanese_room
- [1-Numerical] How many calligraphy paintings are above the display ledge?
- [2-ObjRef] The lantern between the vase and the stone decoration that is closest to the vase.
- [2-ObjRef] The red pillow closest to the sushi.
- [3-Instr] Go near the small table with a vase on it and then to the flowers near the jar.
- [3-Instr] Go to the lantern closest to the fan decoration, then take the path near the wardrobe doors to the flowers on the display ledge.

### livingroom_1
- [1-Numerical] How many chairs are near the table with a vase on it?
- [2-ObjRef] Find the vase on the cabinet below the picture.
- [2-ObjRef] Find the pillow on the sofa that is closest to the windows.
- [3-Instr] Go to the potted plant closest to the pyramid candle holder and stop at the vase between the TV and the door.
- [3-Instr] First, go near the lamp closest to the black chair, then take the path between the sofa and the round tables, and stop at the cabinet with a picture above it.

### livingroom_2
- [1-Numerical] How many cups are on the coffee table?
- [2-ObjRef] Find the stool closest to the shelf near the TV cabinet.
- [2-ObjRef] Find the pillow on the sofa that is closest to the lamp.
- [3-Instr] Go to the microwave on the kitchen counter and then go to the crystal ball decoration on the shelf near the TV.
- [3-Instr] First, go to the chair near the window, then stop at the soccer ball near the couch, avoiding the path between the TV and the tea table.

### livingroom_3
- [1-Numerical] How many photos are on the TV cabinet?
- [2-ObjRef] Find the potted plant near the books on the cabinet.
- [2-ObjRef] Find the vase between the cabinet and the stool.
- [3-Instr] Take the path near the TV and go to the pillow farthest from the lamp.
- [3-Instr] First, go near the stool, then take the path near the cabinet, and stop at the bowl on the table.

### livingroom_4
- [1-Numerical] How many pillows are on a sofa?
- [2-ObjRef] Find the picture closest to a window.
- [2-ObjRef] Find the fossil decoration closest to the phone.
- [3-Instr] Go near the chair closest to the bookcase and stop at the table with the flowers on it.
- [3-Instr] First, go near the fireplace, then go to the window closest to the bookcase, and stop at the chair farthest from the mirror.

### loft
- [1-Numerical] How many black pillows are on the sofa?
- [2-ObjRef] The blue chair that is closest to the cup of coffee.
- [2-ObjRef] Find the potted plant between a vase and the cabinet with a TV on it.
- [3-Instr] Go to the cup near the TV remote and avoid the path near the cabinet.
- [3-Instr] Go near the fireplace, pass by the stairs, then stop at the sphere decoration on the cabinet.

### office_1
- [1-Numerical] How many computer monitors are on the table closest to the map wall decal?
- [2-ObjRef] Find the potted plant on the file cabinet.
- [2-ObjRef] Find the paper cup on the table closest to the projector screen.
- [3-Instr] Go to the potted plant furthest from the projector screen then stop at the water cooler near the window.
- [3-Instr] First, go near the potted plant on the shelf, then take the path between the two tables, and stop at the bench closest to the map wall decal.

### office_2
- [1-Numerical] How many potted plants are on a table?
- [2-ObjRef] Find the computer monitor closest to the cabinet with a phone on it.
- [2-ObjRef] Find the box on the cabinet that is closest to the whiteboard.
- [3-Instr] Go near the potted plant on the cabinet and stop at the window closest to the clock.
- [3-Instr] First, go to the trash can near the cabinet, then go to the folder on the cabinet closest to the whiteboard, and finally, to the door near the exit sign.

### studio
- [1-Numerical] How many framed records are above the couch?
- [2-ObjRef] Find the vase closest to the guitar.
- [2-ObjRef] Find the beer bottle furthest from the couch.
- [3-Instr] Go to the vases on the cabinet below the TV and stop at the guitar near the couch.
- [3-Instr] First, go to the vase closest to the easel, then, take the path between the couch and the table and stop at the window closest to the couch.

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
