#!/bin/bash
# 터미널 C — 질의 (컨테이너 접속 + ROS2 소싱까지만 자동, 퍼블리시 명령은 직접 입력)
# 사용법: ./C_질의.sh              (실행 중인 sysnav 컨테이너 자동 선택)
#         ./C_질의.sh <컨테이너명>  (직접 지정)
# 접속 후 예: ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: 'Find the bowl near the trash can.'}"

set -e

TARGET="$1"
if [ -z "$TARGET" ]; then
  for c in iros2026_sysnav_submission iros2026_sysnav_module; do
    if [ -n "$(docker ps -q -f name=^/${c}$)" ]; then TARGET="$c"; break; fi
  done
fi
if [ -z "$TARGET" ]; then
  echo "실행 중인 sysnav 컨테이너가 없음. 먼저 터미널 B 스크립트를 실행할 것." >&2
  exit 1
fi
echo "[C] 접속 대상: $TARGET"

docker exec -it "$TARGET" bash -c "source /opt/ros/jazzy/setup.bash && exec bash"
