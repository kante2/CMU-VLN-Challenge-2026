"""디버깅용 미션 상태 대시보드.

room_segmentation_latest.png/scene_graph_latest.png와 같은 패턴 - 매 사이클
현재 상태 전체를 새로 그려서 ai_module/debug/mission_status_latest.html을
통째로 덮어쓴다. 서버 없이 그냥 브라우저(또는 VSCode Simple Browser)로 그
파일을 열어두면 <meta refresh>가 자동으로 새로고침한다.

sysnav_node.py가 snapshot dict를 만들어서 넘기고, 이 파일은 순수하게
dict -> HTML 문자열 변환만 한다 (다른 *_visualizer.py들과 동일한 책임 분리).
"""

from __future__ import annotations

import html
import os
import time
from pathlib import Path

from sysnav import config
from sysnav.activity_log import activity

_STATE_COLORS = {
    "SUCCESS": "#15803d",
    "FAILED": "#b91c1c",
    "IDLE": "#6b7280",
}
_DEFAULT_STATE_COLOR = "#2563eb"

_MISSION_LABELS = {
    "numerical": "Mission 1 - Numerical",
    "object_reference": "Mission 2 - Object Reference",
    "instruction_following": "Mission 3 - Instruction-Following",
}


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


def _state_badge(state: str) -> str:
    color = _STATE_COLORS.get(state, _DEFAULT_STATE_COLOR)
    return (
        f'<span style="display:inline-block;padding:6px 16px;border-radius:14px;'
        f'font-weight:700;font-size:15px;color:white;background:{color};">{_esc(state)}</span>'
    )


def _row(label: str, value_html: str) -> str:
    return (
        f'<tr><td style="padding:6px 14px;color:#6b7280;white-space:nowrap;'
        f'font-size:13px;">{_esc(label)}</td>'
        f'<td style="padding:6px 14px;font-family:ui-monospace,monospace;font-size:13px;">'
        f"{value_html}</td></tr>"
    )


def _time_bar(elapsed_sec: float | None, limit_sec: float) -> str:
    if elapsed_sec is None:
        return "<span style=\"color:#9ca3af;\">-</span>"
    ratio = max(0.0, min(1.0, elapsed_sec / limit_sec))
    color = "#15803d" if ratio < 0.6 else ("#b45309" if ratio < 0.9 else "#b91c1c")
    minutes, seconds = divmod(int(elapsed_sec), 60)
    limit_minutes = int(limit_sec // 60)
    return (
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<div style="width:160px;height:10px;background:#e5e7eb;border-radius:5px;overflow:hidden;">'
        f'<div style="width:{ratio * 100:.1f}%;height:100%;background:{color};"></div></div>'
        f"<span>{minutes}:{seconds:02d} / {limit_minutes}:00</span></div>"
    )


def _describe_step(step: dict) -> str:
    if step.get("resolve") == "category":
        parsed = step["parsed"]
        desc = parsed.get("target", "?")
        attrs = parsed.get("attributes") or []
        if attrs:
            desc = f"{', '.join(attrs)} {desc}"
        relation = parsed.get("relation")
        refs = parsed.get("reference_objects") or []
        if relation and refs:
            desc += f" [{relation} {', '.join(refs)}]"
    else:
        refs = ", ".join(ref.get("target", "?") for ref in step.get("point_refs", []))
        desc = f"{step.get('point_mode')}({refs})"
    kind = "stop" if step.get("is_stop") else "waypoint"
    return f"[{kind}] {desc}"


def _describe_forbidden(forbidden: dict) -> str:
    refs = ", ".join(ref.get("target", "?") for ref in forbidden.get("point_refs", []))
    return f"{forbidden.get('point_mode')}({refs})"


def _plan_list_html(steps: list[dict], current_index: int) -> str:
    if not steps:
        return "<span style=\"color:#9ca3af;\">(no steps)</span>"
    items = []
    for index, step in enumerate(steps):
        if index < current_index:
            marker, color = "✓", "#15803d"  # done
        elif index == current_index:
            marker, color = "▶", "#2563eb"  # in progress
        else:
            marker, color = "○", "#9ca3af"  # pending
        items.append(
            f'<li style="color:{color};margin-bottom:2px;">{marker} '
            f'{_esc(_describe_step(step))}</li>'
        )
    return f'<ol style="margin:2px 0 0 -18px;padding:0;list-style:none;">{"".join(items)}</ol>'


def _mission_detail_rows(snapshot: dict) -> str:
    mission_type = snapshot.get("mission_type")
    task = snapshot.get("task") or {}
    rows = []

    if mission_type == "instruction_following":
        steps = task.get("steps") or []
        idx = snapshot.get("mission3_step_index", 0)
        rows.append(_row("Parsed via", task.get("parser", "rules")))
        rows.append(_row("Progress", f"{min(idx, len(steps))} / {len(steps)} steps done"))
        rows.append(_row("Plan", _plan_list_html(steps, idx)))
        forbidden = task.get("global_forbidden") or []
        if forbidden:
            forbidden_desc = ", ".join(_describe_forbidden(item) for item in forbidden)
            active = snapshot.get("mission3_forbidden_active")
            status = "active (routing around it)" if active else "not yet resolved"
            # 제약이 끝내 미적용이면 눈에 띄게 표시한다 - SUCCESS 배지만 보고
            # "다 됐다"고 오해하면 채점에서 감점되는 걸 놓친다.
            style = ' style="color:#b91c1c;font-weight:700"' if not active else ""
            rows.append(_row("Forbidden constraint",
                             f"<span{style}>{_esc(forbidden_desc)} - {status}</span>"))
    elif mission_type == "numerical":
        rows.append(_row("Target category", task.get("target", "?")))
        rows.append(_row("Attributes", ", ".join(task.get("attributes") or []) or "-"))
        candidate_count = snapshot.get("candidate_count")
        rows.append(_row("Candidates observed so far", "-" if candidate_count is None else str(candidate_count)))
    elif mission_type == "object_reference":
        rows.append(_row("Target category", task.get("target", "?")))
        rows.append(_row("Attributes", ", ".join(task.get("attributes") or []) or "-"))
        goal = snapshot.get("current_goal")
        if goal and goal.get("object_id") is not None:
            rows.append(_row("Selected object_id", str(goal["object_id"])))

    return "\n".join(rows)


_CATEGORY_STYLE = {
    "state": ("#2563eb", "상태"),
    "job":   ("#7c3aed", "작업"),
    "llm":   ("#c2410c", "LLM"),
    "nav":   ("#0f766e", "주행"),
    "warn":  ("#b91c1c", "경고"),
}


def _now_panel() -> str:
    """"지금 무엇을 하고 있는가" - 진행 중인 장기 작업(LLM 질의 등)을 크게 보여준다.
    아무것도 없으면 주행/대기 중이라는 뜻이다."""
    items = activity.inflight()
    if not items:
        return (
            '<div style="padding:10px 14px;border-radius:8px;background:#f3f4f6;'
            'color:#6b7280;font-size:13px;">진행 중인 백그라운드 작업 없음 '
            '(주행 또는 센서 대기 중)</div>'
        )
    blocks = []
    for item in items:
        color, tag = _CATEGORY_STYLE.get(item["category"], ("#374151", item["category"]))
        blocks.append(
            f'<div style="padding:10px 14px;border-radius:8px;margin-bottom:6px;'
            f'background:{color}12;border-left:4px solid {color};">'
            f'<span style="color:{color};font-weight:700;font-size:12px;">{_esc(tag)}</span>'
            f'<span style="margin-left:10px;font-size:14px;font-weight:600;">{_esc(item["label"])}</span>'
            f'<span style="float:right;color:#6b7280;font-family:ui-monospace,monospace;'
            f'font-size:13px;">{item["elapsed_sec"]:.1f}초 경과</span></div>'
        )
    return "".join(blocks)


def _activity_panel(limit: int = 40) -> str:
    events = activity.recent(limit)
    if not events:
        return '<div style="color:#9ca3af;font-size:13px;">아직 기록된 활동이 없습니다.</div>'
    rows = []
    for event in events:
        color, tag = _CATEGORY_STYLE.get(event["category"], ("#374151", event["category"]))
        clock = time.strftime("%H:%M:%S", time.localtime(event["time"]))
        detail = (
            f'<span style="color:#9ca3af;margin-left:8px;">{_esc(event["detail"])}</span>'
            if event.get("detail") else ""
        )
        rows.append(
            f'<tr><td style="padding:3px 8px;color:#9ca3af;font-family:ui-monospace,monospace;'
            f'font-size:12px;white-space:nowrap;">{clock}</td>'
            f'<td style="padding:3px 8px;white-space:nowrap;"><span style="color:{color};'
            f'font-weight:700;font-size:11px;">{_esc(tag)}</span></td>'
            f'<td style="padding:3px 8px;font-size:13px;">{_esc(event["message"])}{detail}</td></tr>'
        )
    return (
        '<div style="max-height:420px;overflow-y:auto;border:1px solid #f3f4f6;'
        'border-radius:8px;">'
        f'<table style="border-collapse:collapse;width:100%;">{"".join(rows)}</table></div>'
    )



def _stat(label: str, value: str, hint: str = "") -> str:
    hint_html = (
        f'<div style="color:#9ca3af;font-size:11px;margin-top:2px;">{_esc(hint)}</div>'
        if hint else ""
    )
    return (
        '<div style="flex:1 1 150px;min-width:150px;padding:10px 12px;background:#f9fafb;'
        'border-radius:8px;">'
        f'<div style="color:#6b7280;font-size:11px;">{_esc(label)}</div>'
        f'<div style="font-family:ui-monospace,monospace;font-size:16px;font-weight:600;'
        f'margin-top:2px;">{_esc(value)}</div>{hint_html}</div>'
    )


# 대시보드와 같은 폴더(config.DEBUG_DIR)에 노드가 매 사이클 덮어쓰는 지도 그림들.
# 상대 경로로 걸면 file:// 로 열어도 그대로 보인다(서버 불필요).
_MAP_IMAGES = [
    ("exploration_debug_latest.png", "탐색 지도",
     "회색=탐사 완료, 검정=벽/미탐사, 노랑=frontier, 파랑=로봇"),
    ("room_segmentation_latest.png", "방 분할", "색=방 라벨, 회색=미분류 영역"),
    ("scene_graph_latest.png", "Scene Graph", "room / viewpoint / object 노드"),
]


def _map_images_panel() -> str:
    """지도 그림을 그대로 띄운다.

    <meta refresh>로 페이지가 1초마다 새로 로드되지만, 파일 이름이 그대로라 브라우저가
    이미지를 캐시해서 옛 그림이 계속 보인다. 그래서 파일 mtime을 쿼리스트링에 붙여
    내용이 바뀔 때만 새로 받게 한다."""
    debug_dir = Path(config.DEBUG_DIR)
    cards = []
    for filename, title, caption in _MAP_IMAGES:
        path = debug_dir / filename
        if not path.exists():
            cards.append(
                f'<div style="flex:1 1 300px;min-width:280px;">'
                f'<div style="font-size:12px;font-weight:600;margin-bottom:4px;">{_esc(title)}</div>'
                f'<div style="padding:24px;background:#f9fafb;border-radius:8px;color:#9ca3af;'
                f'font-size:12px;text-align:center;">아직 생성되지 않음</div></div>'
            )
            continue
        try:
            stamp = int(path.stat().st_mtime)
        except OSError:
            stamp = 0
        cards.append(
            f'<div style="flex:1 1 300px;min-width:280px;">'
            f'<div style="font-size:12px;font-weight:600;margin-bottom:4px;">{_esc(title)}'
            f'<span style="color:#9ca3af;font-weight:400;margin-left:6px;">'
            f'{time.strftime("%H:%M:%S", time.localtime(stamp))}</span></div>'
            f'<img src="{_esc(filename)}?t={stamp}" style="width:100%;border-radius:8px;'
            f'background:#111;display:block;">'
            f'<div style="color:#9ca3af;font-size:11px;margin-top:4px;">{_esc(caption)}</div></div>'
        )
    return '<div style="display:flex;flex-wrap:wrap;gap:14px;">' + "".join(cards) + "</div>"



def _map_panel(snapshot: dict) -> str:
    """지금까지 만든 지도 현황. 우리 occupancy grid는 전역 누적이고 /terrain_map은
    base autonomy가 만드는 로컬 롤링 윈도우라, 둘을 나란히 보여줘야 "우리는 안다고
    보는데 base autonomy는 모르는 구역"이 드러난다."""
    stats = snapshot.get("map_stats") or {}
    graph = snapshot.get("graph_counts") or {}
    room = snapshot.get("room_summary") or {}
    if not stats:
        return '<div style="color:#9ca3af;font-size:13px;">아직 지도가 없습니다.</div>'

    frontier = stats.get("frontier_cells", 0)
    frontier_hint = "남은 탐색 경계" if frontier else "탐색할 경계 없음"
    tiles = [
        _stat("탐사 면적", f'{stats.get("mapped_area_m2", 0):.1f} m²',
              f'free {stats.get("free_area_m2", 0):.1f} m² / '
              f'{stats.get("mapped_cells", 0)}셀 @ {stats.get("resolution_m", 0):.2f}m'),
        _stat("frontier", f'{frontier} 셀', frontier_hint),
        _stat("방 / 문", f'{room.get("rooms", 0)} / {room.get("doorways", 0)}',
              f'현재 방 R{room["active_room_id"]}' if room.get("active_room_id") else "현재 방 미분류"),
        _stat("viewpoint", str(graph.get("viewpoints", 0)),
              f'방문 기록 {snapshot.get("viewpoint_memory_count", 0)}'),
        _stat("물체", str(graph.get("objects", 0)),
              f'메모리 {snapshot.get("object_memory_count", 0)} / '
              f'관계 edge {graph.get("object_object_edges", 0)}'),
        _stat("terrain_map", _esc(snapshot.get("terrain_summary") or "-"),
              "base autonomy 제공(로컬)"),
    ]
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:8px;">' + "".join(tiles) + "</div>"
        '<div style="color:#9ca3af;font-size:11px;margin-top:8px;">'
        'occupancy grid는 전역 누적(60m 범위), terrain_map은 로봇 주변 롤링 윈도우입니다. '
        '지도는 디스크에 저장되지 않아 노드를 재시작하면 처음부터 다시 만듭니다.</div>'
    )



def export_mission_dashboard(snapshot: dict) -> str | None:
    """snapshot: sysnav_node._update_mission_dashboard()가 채운 dict. 저장된
    경로(실패/비활성화면 None)."""
    if not config.SAVE_DEBUG_IMAGES:
        return None

    state = snapshot.get("state", "IDLE")
    task = snapshot.get("task")
    mission_type = snapshot.get("mission_type")
    mission_label = _MISSION_LABELS.get(mission_type, "-")

    rows = [
        _row("Task #", snapshot.get("task_id", "-")),
        _row("Question", _esc((task or {}).get("raw", "-"))),
        _row("Mission", mission_label),
        _row("Elapsed / limit", _time_bar(snapshot.get("elapsed_sec"), config.MISSION_TIME_LIMIT_SEC)),
    ]
    if task is not None:
        rows.append(
            '<tr><td colspan="2" style="padding:14px 14px 4px;color:#9ca3af;'
            'font-size:11px;font-weight:700;letter-spacing:0.05em;'
            'text-transform:uppercase;">Mission detail</td></tr>'
        )
        rows.append(_mission_detail_rows(snapshot))
    rows.append(_row("Last published response", _esc(snapshot.get("last_response_summary") or "-")))
    pose = snapshot.get("pose")
    if pose:
        rows.append(_row("Robot pose", f"({pose['x']:.2f}, {pose['y']:.2f}, yaw={pose.get('yaw', 0):.2f})"))
    goal = snapshot.get("current_goal")
    if goal:
        rows.append(_row("Current goal", f"({goal.get('x', 0):.2f}, {goal.get('y', 0):.2f}) type={goal.get('type', '-')}"))
    # base autonomy(waypointConverter)가 우리 좌표를 obstacleDisThre(0.75m) 조건에 맞는
    # 지점으로 갈아끼운 정도. 값이 크면 "우리가 찍은 곳으로는 애초에 갈 수 없다"는 뜻이라,
    # planner를 의심할지 좌표 실행 가능성을 의심할지 여기서 바로 갈린다.
    requested = snapshot.get("requested_waypoint_xy")
    actual = snapshot.get("actual_waypoint_xy")
    displacement = snapshot.get("waypoint_displacement_m")
    snap = snapshot.get("waypoint_snap_m")
    if snap is not None:
        rows.append(_row("Waypoint snapped before publish (Layer 1)", f"{snap:.2f} m"))
    if requested and actual and displacement is not None:
        warn = " style=\"color:#c0392b;font-weight:bold\"" if displacement >= config.WAYPOINT_DISPLACEMENT_WARN_M else ""
        rows.append(_row(
            "Waypoint pushed by base autonomy",
            f"<span{warn}>{displacement:.2f} m</span>"
            f" &nbsp; ours=({requested[0]:.2f}, {requested[1]:.2f})"
            f" &rarr; actual=({actual[0]:.2f}, {actual[1]:.2f})",
        ))
    target_xy = snapshot.get("target_goal_xy")
    if target_xy:
        distance = snapshot.get("target_distance_m")
        distance_text = "-" if distance is None else f"{distance:.2f}m"
        rows.append(_row(
            "Target progress",
            f"dest=({target_xy[0]:.2f}, {target_xy[1]:.2f}), dist={distance_text}, "
            f"hops_left={snapshot.get('target_hops_remaining', 0)}, "
            f"replans={snapshot.get('target_replans', 0)}",
        ))

    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{config.MISSION_DASHBOARD_REFRESH_SEC:.0f}">
<title>SysNav Mission Status</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Arial, sans-serif; background: #f9fafb; margin: 0; padding: 24px; color: #111827; }}
  .card {{ background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 20px 24px; max-width: 720px; margin: 0 auto 16px; }}
  h1 {{ font-size: 18px; margin: 0 0 14px; display:flex; align-items:center; gap:12px; }}
  h2 {{ font-size: 14px; margin: 0 0 10px; color:#374151; }}
  .card {{ max-width: 1100px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  tr:not(:last-child) td {{ border-bottom: 1px solid #f3f4f6; }}
  .footer {{ text-align:center; color:#9ca3af; font-size:12px; margin-top: 8px; }}
</style>
</head>
<body>
  <div class="card">
    <h1>SysNav Mission Status {_state_badge(state)}</h1>
    <table>
      {"".join(rows)}
    </table>
  </div>
  <div class="card">
    <h2>지도 현황</h2>
    {_map_panel(snapshot)}
    <div style="margin-top:14px;">{_map_images_panel()}</div>
  </div>
  <div class="card">
    <h2>지금 하는 일</h2>
    {_now_panel()}
  </div>
  <div class="card">
    <h2>활동 로그 <span style="font-weight:400;color:#9ca3af;font-size:12px;">(최신순)</span></h2>
    {_activity_panel()}
  </div>
  <div class="footer">auto-refreshes every {config.MISSION_DASHBOARD_REFRESH_SEC:.0f}s - open this file directly in a browser</div>
</body>
</html>
"""

    out_path = Path(config.DEBUG_DIR) / "mission_status_latest.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(".tmp.html")
    temp_path.write_text(doc, encoding="utf-8")
    os.replace(temp_path, out_path)
    os.chmod(out_path, 0o644)
    return str(out_path)
