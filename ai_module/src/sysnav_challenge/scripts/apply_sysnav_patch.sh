#!/usr/bin/env bash
set -eo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
challenge_dir="$(cd "$script_dir/.." && pwd)"
sysnav_dir="$(cd "$challenge_dir/../SysNav" && pwd)"
patch_file="$challenge_dir/patches/sysnav_challenge.patch"

if grep -q 'kSingleRoomMode' \
    "$sysnav_dir/src/exploration_planner/tare_planner/include/sensor_coverage_planner/sensor_coverage_planner_ground.h" \
  && grep -q 'gemini-3.5-flash' \
    "$sysnav_dir/src/vlm_node/vlm_node/constants.py"; then
  echo "SysNav challenge patch is already applied."
elif git -C "$sysnav_dir" apply --check "$patch_file"; then
  git -C "$sysnav_dir" apply "$patch_file"
  echo "Applied SysNav challenge patch."
else
  echo "SysNav patch does not apply cleanly; check the submodule revision." >&2
  exit 1
fi
