"""Extract text logs and selfdrive timeline spans."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LogEntry:
  mono_time: float
  level: int
  source: str
  message: str
  origin: str


@dataclass
class TimelineEntry:
  start_time: float
  end_time: float
  kind: str


def _append_timeline(timeline: list[TimelineEntry], mono_time: float, kind: str) -> None:
  if timeline and timeline[-1].kind == kind:
    timeline[-1].end_time = max(timeline[-1].end_time, mono_time)
    return
  timeline.append(TimelineEntry(start_time=mono_time, end_time=mono_time, kind=kind))


def _alert_kind(status: str, enabled: bool) -> str:
  if not enabled:
    return "disengaged"
  if status == "userPrompt":
    return "alert_info"
  if status == "warning":
    return "alert_warning"
  if status in {"critical", "emergency"}:
    return "alert_critical"
  return "engaged"


def extract_logs_and_timeline(events) -> tuple[list[LogEntry], list[TimelineEntry]]:
  logs: list[LogEntry] = []
  timeline: list[TimelineEntry] = []
  last_alert_key = ""

  for event in events:
    try:
      which = event.which()
    except Exception:
      continue
    tm = float(event.logMonoTime) / 1e9

    if which == "selfdriveState":
      try:
        sd = event.selfdriveState
        _append_timeline(timeline, tm, _alert_kind(str(sd.alertStatus), bool(sd.enabled)))
      except Exception:
        pass

    if which == "logMessage":
      try:
        msg = event.logMessage
        logs.append(LogEntry(tm, int(msg.level), str(msg.source), str(msg.message), "log"))
      except Exception:
        pass
    elif which == "errorLogMessage":
      try:
        msg = event.errorLogMessage
        logs.append(LogEntry(tm, 40, str(msg.source), str(msg.message), "log"))
      except Exception:
        pass
    elif which == "operatingSystemLog":
      try:
        msg = event.operatingSystemLog
        logs.append(LogEntry(tm, int(msg.level), str(msg.source), str(msg.message), "os"))
      except Exception:
        pass
    elif which == "selfdriveState":
      try:
        sd = event.selfdriveState
        if sd.alertText1:
          key = f"{sd.alertText1}|{sd.alertText2}"
          if key != last_alert_key:
            last_alert_key = key
            text = str(sd.alertText1)
            if sd.alertText2:
              text += f" — {sd.alertText2}"
            logs.append(LogEntry(tm, 30, "selfdriveState", text, "alert"))
      except Exception:
        pass

  logs.sort(key=lambda e: e.mono_time)
  timeline.sort(key=lambda e: e.start_time)
  return logs, timeline