
# 컨테이너 생성 명령어

cd /home/kante/CMU-VLN-Challenge-2026/docker
xhost +
docker compose -f compose_gpu.yml up --build -d
이 한 줄로 아래 3개 컨테이너가 빌드/생성/실행됩니다:

iros2026_system — 시뮬레이터/autonomy (이미지 pull)
iros2026_ai_module — ../ai_module/docker/Dockerfile 빌드
iros2026_tmah_module — ../ai_module/docker/Dockerfile.tmah 빌드



A — 시뮬레이터/autonomy 실행 (터미널 A)
docker exec -it iros2026_system bash
컨테이너 안에서
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh

# 참고, 로봇이 여러대 겹친걸로 뜨면 이거한 후, 다시 컨테이너 접근
docker restart iros2026_system 

B — dummy VLM 실행 (터미널 B, 새 창)
docker exec -it iros2026_ai_module bash
컨테이너 안에서
ros2 launch dummy_vlm dummy_vlm.launch


# 참고 
더미는 커닝을 하는 구조이다.
readObjectListFile() → data/object_list.txt에서 정답 객체 하나(좌표·크기·라벨)를 미리 읽어둠
readWaypointFile() → data/waypoints.ply에서 정답 경로를 미리 읽어둠
즉 센서를 안 본다. 정답이 파일에 이미 박혀있고 그걸 재생하는 구조.

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

