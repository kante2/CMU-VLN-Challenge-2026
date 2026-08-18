#!/bin/bash
# 컨테이너 기동 스크립트 (docker start / 없으면 compose로 생성)
#
#   ./docker/start_containers.sh                 # system + 제출이미지 sysnav (기본)
#   ./docker/start_containers.sh submission      # 위와 동일 (명시)
#   ./docker/start_containers.sh dev             # system + 로컬빌드 sysnav_module
#   ./docker/start_containers.sh both            # system + 둘 다
#
# 기동 후: 터미널 A ./docker/A_시뮬레이터.sh
#          터미널 B ./docker/B_sysnav_실행_제출이미지.sh  (dev면 ./docker/B_sysnav_실행.sh)
#          터미널 C ./docker/C_질의.sh

set -euo pipefail
cd "$(dirname "$0")"

COMPOSE="compose_gpu.yml"
MODE="${1:-submission}"

case "$MODE" in
  submission) SERVICES=(system sysnav_submission) ;;
  dev)        SERVICES=(system sysnav_module) ;;
  both)       SERVICES=(system sysnav_module sysnav_submission) ;;
  *) echo "사용법: $0 [submission|dev|both]" >&2; exit 1 ;;
esac

declare -A CNAME=(
  [system]=iros2026_system
  [sysnav_module]=iros2026_sysnav_module
  [sysnav_submission]=iros2026_sysnav_submission
)

# GUI(RViz/시뮬레이터)용 X 접근 허용 — 이미 허용돼 있으면 무해
command -v xhost >/dev/null 2>&1 && xhost +local:docker >/dev/null 2>&1 || true

for svc in "${SERVICES[@]}"; do
  c="${CNAME[$svc]}"
  if [ -n "$(docker ps -q -f "name=^/${c}$")" ]; then
    echo "[start] $c : 이미 실행 중 — 건너뜀"
  elif [ -n "$(docker ps -aq -f "name=^/${c}$")" ]; then
    echo "[start] $c : 정지 상태 — docker start"
    docker start "$c" >/dev/null
  else
    echo "[start] $c : 컨테이너 없음 — compose로 생성 ($svc)"
    docker compose -f "$COMPOSE" up -d "$svc"
  fi
done

echo
docker ps --filter "name=iros2026_" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
