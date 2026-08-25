#!/bin/bash
# 터미널 A — 시뮬레이터/autonomy 실행
# 로봇이 여러 대 겹친 걸로 뜨면: docker restart iros2026_system 후 재실행

set -e

docker exec -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -it iros2026_system bash -c "/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh"
