#!/bin/bash
# 터미널 B (제출 이미지 버전) — Docker Hub 이미지 parkjaeil00/cmu-vln-2026-sysnav 로 sysnav 실행
# 이미지 안에 이미 colcon build 되어 있고 src 마운트도 없으므로 재빌드 없이 바로 launch.
# 사용법: ./B_sysnav_실행_제출이미지.sh   (컨테이너가 없으면 자동 생성)

set -e
cd "$(dirname "$0")"

if [ -z "$(docker ps -q -f name=^/iros2026_sysnav_submission$)" ]; then
  docker compose -f compose_gpu.yml up -d sysnav_submission
fi

docker exec -it iros2026_sysnav_submission bash -c \
  "source /opt/ros/jazzy/setup.bash && source /home/docker/ai_module/install/setup.bash && ros2 launch sysnav sysnav.launch.py; exec bash"
