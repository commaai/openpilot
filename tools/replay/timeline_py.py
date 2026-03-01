"""Pure-Python timeline builder for replay.

Reads qlogs via tools/lib and tracks engagement windows and alert events,
producing the same (start_time, end_time, type_int) tuples that the C++
Timeline used to provide.
"""
import threading
from enum import IntEnum

from cereal import log as capnp_log
from openpilot.tools.lib.logreader import LogReader


class TimelineType(IntEnum):
  Non = 0  # "None" is a Python keyword
  Engaged = 1
  AlertInfo = 2
  AlertWarning = 3
  AlertCritical = 4
  UserBookmark = 5


class FindFlag(IntEnum):
  nextEngagement = 0
  nextDisEngagement = 1
  nextUserBookmark = 2
  nextInfo = 3
  nextWarning = 4
  nextCritical = 5


_ALERT_TYPE_MAP = {
  capnp_log.SelfdriveState.AlertStatus.normal: TimelineType.AlertInfo,
  capnp_log.SelfdriveState.AlertStatus.userPrompt: TimelineType.AlertWarning,
  capnp_log.SelfdriveState.AlertStatus.critical: TimelineType.AlertCritical,
}


class Timeline:
  """Builds an engagement/alert timeline from qlogs in a background thread."""

  def __init__(self):
    self._entries = []  # list of [start, end, type, text1, text2]
    self._lock = threading.Lock()
    self._thread = None
    self._stop = threading.Event()

  def build_async(self, qlog_paths, route_start_ts):
    """Start background timeline build.

    Args:
      qlog_paths: dict mapping segment_num -> qlog path/url (None for missing).
      route_start_ts: route start timestamp in nanoseconds.
    """
    self._thread = threading.Thread(
      target=self._build, args=(qlog_paths, route_start_ts), daemon=True
    )
    self._thread.start()

  def stop(self):
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=5)

  def get_entries(self):
    """Return current timeline as list of (start_time, end_time, type_int)."""
    with self._lock:
      return [(e[0], e[1], int(e[2])) for e in self._entries]

  def find(self, cur_ts, flag):
    """Find next timeline event matching flag after cur_ts. Returns seconds or None."""
    with self._lock:
      entries = list(self._entries)

    for e in entries:
      start, end, etype = e[0], e[1], e[2]
      if etype == TimelineType.Engaged:
        if flag == FindFlag.nextEngagement and start > cur_ts:
          return start
        elif flag == FindFlag.nextDisEngagement and end > cur_ts:
          return end
      elif start > cur_ts:
        if flag == FindFlag.nextUserBookmark and etype == TimelineType.UserBookmark:
          return start
        elif flag == FindFlag.nextInfo and etype == TimelineType.AlertInfo:
          return start
        elif flag == FindFlag.nextWarning and etype == TimelineType.AlertWarning:
          return start
        elif flag == FindFlag.nextCritical and etype == TimelineType.AlertCritical:
          return start
    return None

  def _build(self, qlog_paths, route_start_ts):
    staging = []
    current_engaged_idx = None
    current_alert_idx = None

    for seg_num in sorted(qlog_paths.keys()):
      if self._stop.is_set():
        break

      path = qlog_paths[seg_num]
      if not path:
        continue

      try:
        lr = LogReader(path)
      except Exception:
        continue

      for msg in lr:
        if self._stop.is_set():
          break

        if msg.which() == "selfdriveState":
          seconds = (msg.logMonoTime - route_start_ts) / 1e9
          cs = msg.selfdriveState

          # Update engagement
          if current_engaged_idx is not None:
            staging[current_engaged_idx][1] = seconds
          if cs.enabled:
            if current_engaged_idx is None:
              current_engaged_idx = len(staging)
              staging.append([seconds, seconds, TimelineType.Engaged, "", ""])
          else:
            current_engaged_idx = None

          # Update alerts
          entry = staging[current_alert_idx] if current_alert_idx is not None else None
          if entry is not None:
            entry[1] = seconds

          alert_size = str(cs.alertSize)
          if alert_size != "none":
            atype = _ALERT_TYPE_MAP.get(cs.alertStatus, TimelineType.AlertInfo)
            text1 = cs.alertText1
            text2 = cs.alertText2
            if entry is None or entry[2] != atype or entry[3] != text1 or entry[4] != text2:
              current_alert_idx = len(staging)
              staging.append([seconds, seconds, atype, text1, text2])
          else:
            current_alert_idx = None

        elif msg.which() == "userBookmark":
          seconds = (msg.logMonoTime - route_start_ts) / 1e9
          staging.append([seconds, seconds, TimelineType.UserBookmark, "", ""])

      # Sort and publish after each segment
      sorted_entries = sorted(staging, key=lambda e: e[0])
      with self._lock:
        self._entries = sorted_entries
