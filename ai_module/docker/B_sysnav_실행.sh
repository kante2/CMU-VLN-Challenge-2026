#!/bin/bash
# 터미널 B — sysnav 노드 실행 (재빌드 후 launch)
# 사용법: ./B_sysnav_실행.sh

set -e

docker exec -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -it iros2026_sysnav_module bash -c "source /opt/ros/jazzy/setup.bash && cd /home/docker/ai_module && colcon build --symlink-install --packages-select sysnav && source install/setup.bash && ros2 launch sysnav sysnav.launch.py; exec bash"
