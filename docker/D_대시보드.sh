#!/bin/bash
# 터미널 D — 미션 상태 대시보드 (호스트에서 브라우저로 열기, 컨테이너 접속 아님)
# 사용법: ./D_대시보드.sh

set -e

DASHBOARD_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/ai_module/debug/mission_status_latest.html"

if [ ! -f "$DASHBOARD_PATH" ]; then
    echo "아직 파일이 없습니다: $DASHBOARD_PATH"
    echo "sysnav 노드가 최소 한 번 control_loop을 돌아야 생성됩니다 (터미널 B 먼저 실행)."
    exit 1
fi

xdg-open "$DASHBOARD_PATH"
