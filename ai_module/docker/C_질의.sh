#!/bin/bash
# 터미널 C — 질의 (컨테이너 접속 + ROS2 소싱까지만 자동, 퍼블리시 명령은 직접 입력)
# 사용법: ./C_질의.sh
# 접속 후 예: ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: 'Find the bowl near the trash can.'}"

set -e

docker exec -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -it iros2026_sysnav_module bash -c "source /opt/ros/jazzy/setup.bash && exec bash"
