#!/bin/bash
# 터미널 D — 미션 상태 대시보드 (호스트에서 브라우저로 열기, 컨테이너 접속 아님)
# 사용법: ./ui_checker.sh
#
# 대시보드에 있는 것:
#   - 현재 상태/미션/경과시간, 로봇 위치와 목표
#   - base autonomy가 우리 목표를 얼마나 옮겼는지(Waypoint pushed)
#   - "지금 하는 일" — 진행 중인 LLM 질의/백그라운드 작업과 경과 시간
#   - "활동 로그" — 상태 전이·작업·LLM 질의·주행 판단이 한 타임라인에 최신순으로
#
# 페이지는 <meta refresh>로 1초마다 스스로 새로고침한다(서버 불필요).

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DASHBOARD_PATH="$REPO_ROOT/ai_module/debug/mission_status_latest.html"
WAIT_SEC="${WAIT_SEC:-60}"

if [ ! -f "$DASHBOARD_PATH" ]; then
    echo "대시보드가 아직 없습니다: $DASHBOARD_PATH"
    echo "sysnav 노드가 control_loop을 한 번 돌면 생성됩니다 (터미널 B 먼저 실행)."
    echo "최대 ${WAIT_SEC}초 기다립니다... (Ctrl-C로 중단)"
    for _ in $(seq "$WAIT_SEC"); do
        [ -f "$DASHBOARD_PATH" ] && break
        sleep 1
    done
fi

if [ ! -f "$DASHBOARD_PATH" ]; then
    echo "여전히 파일이 없습니다. sysnav 노드가 실행 중인지 확인하세요."
    echo "  docker exec iros2026_sysnav_module bash -lc 'ros2 node list'"
    exit 1
fi

echo "여는 중: $DASHBOARD_PATH"
xdg-open "$DASHBOARD_PATH" >/dev/null 2>&1 &
