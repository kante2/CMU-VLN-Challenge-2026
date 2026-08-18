#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEBUG_DIR="${REPO_ROOT}/ai_module/debug"

sudo chmod -R a+rwx "$DEBUG_DIR" 2>/dev/null || true

if [ -z "$(docker ps -q -f name=^/iros2026_sysnav_submission$)" ]; then
  docker compose -f compose_gpu.yml up -d sysnav_submission
fi

docker exec -u root -it iros2026_sysnav_submission bash -lc \
  "chmod -R a+rwx /home/docker/ai_module/debug && source /opt/ros/jazzy/setup.bash && source /home/docker/ai_module/install/setup.bash && ros2 launch sysnav sysnav.launch.py; exec bash"