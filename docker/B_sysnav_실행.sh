#!/bin/bash
# 터미널 B — sysnav 노드 실행
# 호스트에서 src 수정했다면 재빌드 필요:
#   docker exec -it iros2026_sysnav_module bash
#   source /opt/ros/jazzy/setup.bash
#   cd /home/docker/ai_module && colcon build --symlink-install --packages-select sysnav

set -e

docker exec -it iros2026_sysnav_module bash -c "source /opt/ros/jazzy/setup.bash && source /home/docker/ai_module/install/setup.bash && ros2 launch sysnav sysnav.launch.py"
