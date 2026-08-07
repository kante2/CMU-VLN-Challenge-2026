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
from pathlib import Path

from sysnav import config

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


def _mission_detail_rows(snapshot: dict) -> str:
    mission_type = snapshot.get("mission_type")
    task = snapshot.get("task") or {}
    rows = []

    if mission_type == "instruction_following":
        steps = task.get("steps") or []
        idx = snapshot.get("mission3_step_index", 0)
        rows.append(_row("Step", f"{min(idx + 1, len(steps))} / {len(steps)}"))
        if idx < len(steps):
            step = steps[idx]
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
                desc = f"{step.get('point_mode')}({', '.join(r.get('target', '?') for r in step.get('point_refs', []))})"
            kind = "stop" if step.get("is_stop") else "waypoint"
            rows.append(_row("Current step", f"[{kind}] {_esc(desc)}"))
        rows.append(_row("Forbidden region active", "yes" if snapshot.get("mission3_forbidden_active") else "no"))
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
