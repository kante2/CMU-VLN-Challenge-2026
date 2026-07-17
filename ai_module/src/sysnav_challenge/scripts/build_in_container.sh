#!/usr/bin/env bash
set -eo pipefail

cd /home/docker/ai_module
source src/sysnav_challenge/scripts/apply_sysnav_patch.sh
source /opt/ros/jazzy/setup.bash
set -u

colcon --log-base log_sysnav build --symlink-install \
  --build-base build_sysnav \
  --install-base install_sysnav \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  --base-paths src/SysNav/src src/sysnav_challenge \
  --packages-select tare_planner semantic_mapping vlm_node sysnav_challenge

echo "Build complete. Source install/setup.bash, then install_sysnav/setup.bash."
