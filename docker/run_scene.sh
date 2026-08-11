#!/bin/bash
# 씬(맵)을 골라서 iros2026_system 컨테이너의 시뮬레이션을 실행.
#
# 사용법:
#   ./docker/run_scene.sh <scene_name> [launch_script]
#
#   scene_name    : map/<scene_name>.zip 의 이름 (예: home_building_1, hotel_room_1)
#   launch_script : autonomy_stack_mecanum_wheel_platform/ 안의 실행 스크립트.
#                   기본값 system_simulation.sh
#                   (system_simulation_with_exploration_planner.sh 등도 가능)
#
# 동작:
#   1. map/<scene_name>.zip 을 컨테이너 안 mesh/scenes/<scene_name>/ 에 처음 한 번만 압축 해제 (캐시)
#   2. mesh/unity 심볼릭 링크를 mesh/scenes/<scene_name> 로 전환
#   3. 이전에 떠있던 시뮬레이션 프로세스 정리
#   4. launch_script 실행
#
# map/ 폴더는 docker-compose.yml / docker-compose_gpu.yml 에서
# /home/docker/maps 로 읽기전용 마운트되어 있어야 함.

set -euo pipefail

SCENE="${1:?사용법: $0 <scene_name> [launch_script]}"
LAUNCH_SCRIPT="${2:-system_simulation.sh}"
CONTAINER=iros2026_system

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_DIR="$SCRIPT_DIR/../map"
MAP_ZIP="$MAP_DIR/${SCENE}.zip"

if [ ! -f "$MAP_ZIP" ]; then
  echo "map/${SCENE}.zip 이 없음. 사용 가능한 씬:" >&2
  ls "$MAP_DIR" 2>/dev/null | sed -n 's/\.zip$//p' >&2
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "${CONTAINER} 컨테이너가 안 떠있음. 먼저 'docker compose -f docker/compose_gpu.yml up -d' (GPU 없으면 compose.yml) 실행할 것" >&2
  exit 1
fi

echo "[run_scene] 기존 시뮬레이션 프로세스 정리 중"
# 주의: pkill -f를 "docker exec ... bash -lc '...pkill -f 패턴...'" 형태로 감싸면
# 그 bash -lc 프로세스 자신의 cmdline에도 패턴 문자열이 그대로 들어있어서 pkill이 자기 부모(=이 명령
# 전체)를 잡아 죽여버려 행(hang)이 남. docker exec가 pkill을 바로 실행하게 해서 (중간 bash -lc 없이)
# 패턴 문자열이 다른 프로세스의 cmdline에 실리지 않게 해야 함.
docker exec "$CONTAINER" pkill -f "ros2 launch vehicle_simulator" 2>/dev/null || true
docker exec "$CONTAINER" pkill -f "Model.x86_64" 2>/dev/null || true
docker exec "$CONTAINER" pkill -f "rviz2" 2>/dev/null || true
sleep 1

echo "[run_scene] 씬 준비 중: ${SCENE}"
docker exec "$CONTAINER" bash -lc "
  set -e
  LOCAL_PLANNER_SRC=/home/docker/autonomy_stack_mecanum_wheel_platform/src/base_autonomy/local_planner/src/localPlanner.cpp
  # localPlanner의 narrow-passage corridor 반경은 ROS parameter가 아니라 C++
  # 상수다. 새 컨테이너에서도 동일한 튜닝이 재현되도록 최초 한 번만 바꾸고 해당
  # package만 다시 빌드한다.
  if grep -q 'float searchRadius = 0.45;' \"\$LOCAL_PLANNER_SRC\"; then
    sed -i 's/float searchRadius = 0.45;/float searchRadius = 0.32;/' \"\$LOCAL_PLANNER_SRC\"
    source /opt/ros/jazzy/setup.bash
    cd /home/docker/autonomy_stack_mecanum_wheel_platform
    colcon build --packages-select local_planner
  fi
  # compose가 제공한 tuned launch overlay는 local_planner 빌드가 끝난 뒤 설치
  # 공간에 복사해야 빌드가 upstream 기본값(0.5m footprint 등)으로 되돌리지 않는다.
  if [ -f /tmp/sysnav_local_planner.launch ]; then
    cp /tmp/sysnav_local_planner.launch \
      /home/docker/autonomy_stack_mecanum_wheel_platform/install/local_planner/share/local_planner/launch/local_planner.launch
  fi

  MESH_DIR=/home/docker/autonomy_stack_mecanum_wheel_platform/src/base_autonomy/vehicle_simulator/mesh
  SCENES_DIR=\$MESH_DIR/scenes
  mkdir -p \"\$SCENES_DIR\"

  # 컨테이너 이미지에 원래 박혀있던 씬(첫 실행시엔 mesh/unity가 실디렉토리)은
  # 한 번만 scenes/_original 로 보존하고 그 자리를 심볼릭 링크로 바꿈
  if [ -e \"\$MESH_DIR/unity\" ] && [ ! -L \"\$MESH_DIR/unity\" ]; then
    mv \"\$MESH_DIR/unity\" \"\$SCENES_DIR/_original\"
  fi

  if [ ! -d \"\$SCENES_DIR/${SCENE}\" ]; then
    echo '  - 처음 쓰는 씬이라 압축 해제 중 (다음부턴 캐시돼서 바로 전환됨)'
    TMP=\$(mktemp -d)
    unzip -q /home/docker/maps/${SCENE}.zip -d \"\$TMP\"
    mv \"\$TMP/${SCENE}\" \"\$SCENES_DIR/${SCENE}\"
    rm -rf \"\$TMP\"
  fi

  ln -sfn \"scenes/${SCENE}\" \"\$MESH_DIR/unity\"
  echo \"  - mesh/unity -> scenes/${SCENE}\"
"

echo "[run_scene] ${LAUNCH_SCRIPT} 실행 (씬: ${SCENE})"
docker exec -it "$CONTAINER" bash -lc "/home/docker/autonomy_stack_mecanum_wheel_platform/${LAUNCH_SCRIPT}"
