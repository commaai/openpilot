#!/usr/bin/env python3
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from openpilot.tools.lib.logreader import LogReader
from openpilot.selfdrive.test.process_replay.migration import migrate_all


class TimelineType(Enum):
  NONE = auto()
  ENGAGED = auto()
  ALERT_INFO = auto()
  ALERT_WARNING = auto()
  ALERT_CRITICAL = auto()
  USER_BOOKMARK = auto()


class FindFlag(Enum):
  NEXT_ENGAGEMENT = auto()
  NEXT_DISENGAGEMENT = auto()
  NEXT_USER_BOOKMARK = auto()
  NEXT_INFO = auto()
  NEXT_WARNING = auto()
  NEXT_CRITICAL = auto()


@dataclass
class TimelineEntry:
  start_time: float
  end_time: float
  type: TimelineType
  text1: str = ""
  text2: str = ""


class Timeline:
  def __init__(self):
    self._entries: list[TimelineEntry] = []
    self._lock = threading.Lock()
    self._thread: Optional[threading.Thread] = None
    self._should_exit = threading.Event()

  def __del__(self):
    self._should_exit.set()
    if self._thread is not None and self._thread.is_alive():
      self._thread.join()

  def initialize(self, route, route_start_ts: int, local_cache: bool,
                 callback: Callable[[LogReader], None]) -> None:
    self._thread = threading.Thread(
      target=self._build_timeline,
      args=(route, route_start_ts, local_cache, callback),
      daemon=True
    )
    self._thread.start()

  def find(self, cur_ts: float, flag: FindFlag) -> Optional[float]:
    with self._lock:
      entries = list(self._entries)

    for entry in entries:
      if entry.type == TimelineType.ENGAGED:
        if flag == FindFlag.NEXT_ENGAGEMENT and entry.start_time > cur_ts:
          return entry.start_time
        elif flag == FindFlag.NEXT_DISENGAGEMENT and entry.end_time > cur_ts:
          return entry.end_time
      elif entry.start_time > cur_ts:
        if (flag == FindFlag.NEXT_USER_BOOKMARK and entry.type == TimelineType.USER_BOOKMARK) or \
           (flag == FindFlag.NEXT_INFO and entry.type == TimelineType.ALERT_INFO) or \
           (flag == FindFlag.NEXT_WARNING and entry.type == TimelineType.ALERT_WARNING) or \
           (flag == FindFlag.NEXT_CRITICAL and entry.type == TimelineType.ALERT_CRITICAL):
          return entry.start_time
    return None

  def find_alert_at_time(self, target_time: float) -> Optional[TimelineEntry]:
    with self._lock:
      entries = list(self._entries)

    for entry in entries:
      if entry.start_time > target_time:
        break
      if entry.end_time >= target_time and entry.type in (
          TimelineType.ALERT_INFO, TimelineType.ALERT_WARNING, TimelineType.ALERT_CRITICAL):
        return entry
    return None

  def get_entries(self) -> list[TimelineEntry]:
    with self._lock:
      return list(self._entries)

  def _build_timeline(self, route, route_start_ts: int, local_cache: bool,
                      callback: Callable[[LogReader], None]) -> None:
    current_engaged_idx: Optional[int] = None
    current_alert_idx: Optional[int] = None
    staging_entries: list[TimelineEntry] = []

    for segment in route.segments:
      if self._should_exit.is_set():
        break

      qlog_path = segment.qlog_path
      if qlog_path is None:
        continue

      try:
        lr = LogReader(qlog_path)
      except Exception:
        continue

      for msg in migrate_all(lr):
        if self._should_exit.is_set():
          break

        seconds = (msg.logMonoTime - route_start_ts) / 1e9

        if msg.which() == 'selfdriveState':
          ss = msg.selfdriveState
          current_engaged_idx = self._update_engagement_status(
            ss.enabled, current_engaged_idx, seconds, staging_entries)
          current_alert_idx = self._update_alert_status(
            ss.alertSize, ss.alertStatus, ss.alertText1, ss.alertText2,
            current_alert_idx, seconds, staging_entries)
        elif msg.which() == 'userBookmark':
          staging_entries.append(TimelineEntry(
            start_time=seconds,
            end_time=seconds,
            type=TimelineType.USER_BOOKMARK
          ))

      # Sort and update the timeline entries after each segment
      sorted_entries = sorted(staging_entries, key=lambda e: e.start_time)
      with self._lock:
        self._entries = sorted_entries

      callback(lr)

  def _update_engagement_status(self, enabled: bool, idx: Optional[int], seconds: float,
                                  entries: list[TimelineEntry]) -> Optional[int]:
    if idx is not None:
      entries[idx].end_time = seconds

    if enabled:
      if idx is None:
        idx = len(entries)
        entries.append(TimelineEntry(
          start_time=seconds,
          end_time=seconds,
          type=TimelineType.ENGAGED
        ))
    else:
      idx = None
    return idx

  def _update_alert_status(self, alert_size, alert_status, text1: str, text2: str,
                            idx: Optional[int], seconds: float,
                            entries: list[TimelineEntry]) -> Optional[int]:
    # Map alertStatus enum to TimelineType
    status_map = {
      'normal': TimelineType.ALERT_INFO,
      'userPrompt': TimelineType.ALERT_WARNING,
      'critical': TimelineType.ALERT_CRITICAL,
    }

    entry = entries[idx] if idx is not None else None
    if entry is not None:
      entry.end_time = seconds

    # Check if there's an active alert (alertSize != NONE means alertSize > 0)
    # alertSize is an enum: none=0, small=1, mid=2, full=3
    if str(alert_size) != 'none':
      status_str = str(alert_status)
      alert_type = status_map.get(status_str, TimelineType.ALERT_INFO)

      if entry is None or entry.type != alert_type or entry.text1 != text1 or entry.text2 != text2:
        idx = len(entries)
        entries.append(TimelineEntry(
          start_time=seconds,
          end_time=seconds,
          type=alert_type,
          text1=text1,
          text2=text2
        ))
    else:
      idx = None
    return idx
