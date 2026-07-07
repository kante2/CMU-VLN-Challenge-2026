#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FIXED_FRAME="${1:-${FIXED_FRAME:-map}}"

echo "[INFO] fixed frame: ${FIXED_FRAME}"
echo "[INFO] script dir  : ${SCRIPT_DIR}"

if [ -n "${ROS_DISTRO}" ] && [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
elif [ -f "/opt/ros/humble/setup.bash" ]; then
    source "/opt/ros/humble/setup.bash"
elif [ -f "/opt/ros/foxy/setup.bash" ]; then
    source "/opt/ros/foxy/setup.bash"
else
    echo "[WARN] ROS setup.bash를 찾지 못했습니다."
fi

if [ -f "${HOME}/ai_module/install/setup.bash" ]; then
    source "${HOME}/ai_module/install/setup.bash"
fi

if [ -z "${DISPLAY}" ]; then
    echo "[WARN] DISPLAY가 비어 있습니다."
    echo "[WARN] RViz가 안 뜨면 호스트 터미널에서 먼저 xhost + 를 실행하세요."
fi

echo "[INFO] Checking topic types..."
ros2 topic info /tf || true
ros2 topic info /tf_static || true
ros2 topic info /sensor_scan || true
ros2 topic info /registered_scan || true
ros2 topic info /overall_map || true
ros2 topic info /path || true

TMP_RVIZ="/tmp/tmah_tf_scan_${FIXED_FRAME}.rviz"

sed "s/Fixed Frame: map/Fixed Frame: ${FIXED_FRAME}/g" \
    "${SCRIPT_DIR}/tmah_tf_scan.rviz" > "${TMP_RVIZ}"

echo "[INFO] Starting TF monitor..."
python3 "${SCRIPT_DIR}/tf_monitor.py" --fixed-frame "${FIXED_FRAME}" --period 3.0 &
TF_MONITOR_PID=$!

cleanup() {
    echo "[INFO] Stopping TF monitor..."
    kill "${TF_MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[INFO] Starting RViz..."
rviz2 -d "${TMP_RVIZ}"
