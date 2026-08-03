#!/bin/bash
# 터미널 C — 질의
# 사용법: ./C_질의.sh "Find the white chair"   (인자 생략 시 기본 질문 사용)

set -e

QUESTION="${1:-Find the white chair}"

docker exec -it iros2026_sysnav_module bash -c "source /opt/ros/jazzy/setup.bash && ros2 topic pub --once /challenge_question std_msgs/msg/String \"{data: '${QUESTION}'}\""
