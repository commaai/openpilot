"""ReplayStream - loads CAN data from openpilot routes."""

import bisect
import time
from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal as QtSignal, Qt

from openpilot.tools.cabana.pycabana.dbc.dbc import MessageId, CanData
from openpilot.tools.cabana.pycabana.streams.abstract import AbstractStream


@dataclass
class CanEvent:
  """Single CAN event with timestamp."""
  ts: float  # seconds since start
  msg_id: MessageId
  dat: bytes


class LogLoaderThread(QThread):
  """Background thread that loads log data and processes it."""

  dataReady = QtSignal(object, object, object)  # last_msgs, msg_ids, events
  progress = QtSignal(int, int)
  finished = QtSignal(str, float)  # fingerprint, duration

  BATCH_SIZE = 10000
  YIELD_INTERVAL = 1000  # Yield GIL every N messages

  def __init__(self, route: str, parent=None):
    super().__init__(parent)
    self.route = route
    self._stop_requested = False

  def run(self):
    fingerprint = ""
    duration = 0.0
    try:
      from openpilot.tools.lib.logreader import LogReader
      lr = LogReader(self.route)
      time.sleep(0.001)  # Yield GIL after LogReader init
      total_msgs = 0
      can_msgs = 0
      last_msgs: dict[MessageId, CanData] = {}
      all_events: list[CanEvent] = []
      start_ts = 0
      last_emit_count = 0
      last_yield = 0

      for msg in lr:
        if self._stop_requested:
          break

        total_msgs += 1
        which = msg.which()

        if which == 'carParams' and not fingerprint:
          fingerprint = msg.carParams.carFingerprint

        if which == 'can':
          for c in msg.can:
            msg_id = MessageId(c.src, c.address)
            mono_time = msg.logMonoTime

            if start_ts == 0:
              start_ts = mono_time

            ts = (mono_time - start_ts) / 1e9
            duration = max(duration, ts)

            # Store event for seeking
            all_events.append(CanEvent(ts=ts, msg_id=msg_id, dat=bytes(c.dat)))

            if msg_id not in last_msgs:
              last_msgs[msg_id] = CanData()

            last_msgs[msg_id].count += 1
            last_msgs[msg_id].dat = bytes(c.dat)
            last_msgs[msg_id].ts = ts
            can_msgs += 1

            if can_msgs - last_emit_count >= self.BATCH_SIZE:
              snapshot = {k: CanData(v.ts, v.count, v.freq, v.dat) for k, v in last_msgs.items()}
              self.dataReady.emit(snapshot, set(last_msgs.keys()), None)
              self.progress.emit(can_msgs, total_msgs)
              last_emit_count = can_msgs
              time.sleep(0.01)  # Give main thread time to process

            # Periodically yield GIL to allow main thread to process events
            if can_msgs - last_yield >= self.YIELD_INTERVAL:
              time.sleep(0)
              last_yield = can_msgs

      snapshot = {k: CanData(v.ts, v.count, v.freq, v.dat) for k, v in last_msgs.items()}
      self.dataReady.emit(snapshot, set(last_msgs.keys()), all_events)
      self.progress.emit(can_msgs, total_msgs)
    except Exception as e:
      import traceback

      print(f"Error loading route: {e}")
      traceback.print_exc()
    finally:
      self.finished.emit(fingerprint, duration)

  def stop(self):
    self._stop_requested = True


class ReplayStream(AbstractStream):
  """Stream that replays CAN data from an openpilot route."""

  loadProgress = QtSignal(int, int)
  loadFinished = QtSignal()
  seeked = QtSignal(float)  # Emitted when seek completes (time in seconds)

  def __init__(self, parent=None):
    super().__init__(parent)
    self._route: str = ""
    self._fingerprint: str = ""
    self._loader_thread: LogLoaderThread | None = None
    self._loading: bool = False
    self._duration: float = 0.0
    self._current_time: float = 0.0
    self._all_events: list[CanEvent] = []
    self._event_timestamps: list[float] = []  # For binary search

  def __del__(self):
    try:
      self.stop()
    except RuntimeError:
      pass

  def loadRoute(self, route: str) -> bool:
    if self._loading:
      return False

    self._route = route
    self._loading = True
    self._duration = 0.0
    self._current_time = 0.0
    self._all_events = []
    self._event_timestamps = []

    self.events.clear()
    self.last_msgs.clear()
    self._msg_ids.clear()
    self.start_ts = 0

    self._loader_thread = LogLoaderThread(route, self)
    self._loader_thread.dataReady.connect(self._onDataReady, Qt.ConnectionType.QueuedConnection)
    self._loader_thread.progress.connect(self._onProgress, Qt.ConnectionType.QueuedConnection)
    self._loader_thread.finished.connect(self._onLoadFinished, Qt.ConnectionType.QueuedConnection)
    self._loader_thread.start()

    return True

  def _onDataReady(self, snapshot: dict[MessageId, CanData], msg_ids: set[MessageId], events: list[CanEvent] | None):
    self.last_msgs = snapshot
    new_ids = msg_ids - self._msg_ids
    self._msg_ids = msg_ids

    # Store events when loading completes (final emit has all events)
    if events is not None:
      self._all_events = events
      self._event_timestamps = [e.ts for e in events]

    self.emitMsgsReceived(has_new=bool(new_ids))

  def _onProgress(self, can_msgs: int, total_msgs: int):
    self.loadProgress.emit(can_msgs, total_msgs)

  def _onLoadFinished(self, fingerprint: str, duration: float):
    self._loading = False
    self._fingerprint = fingerprint
    self._duration = duration
    self._current_time = duration  # Start at end (showing all data)
    self.loadFinished.emit()

  def seekTo(self, time_sec: float) -> None:
    """Seek to a specific time, updating last_msgs to state at that time."""
    if not self._all_events:
      return

    time_sec = max(0, min(time_sec, self._duration))
    self._current_time = time_sec

    # Find all events up to this time
    idx = bisect.bisect_right(self._event_timestamps, time_sec)

    # Rebuild last_msgs from events up to this point
    new_last_msgs: dict[MessageId, CanData] = {}
    counts: dict[MessageId, int] = {}

    for i in range(idx):
      event = self._all_events[i]
      if event.msg_id not in new_last_msgs:
        new_last_msgs[event.msg_id] = CanData()
        counts[event.msg_id] = 0

      counts[event.msg_id] += 1
      new_last_msgs[event.msg_id].count = counts[event.msg_id]
      new_last_msgs[event.msg_id].dat = event.dat
      new_last_msgs[event.msg_id].ts = event.ts

    self.last_msgs = new_last_msgs
    self._msg_ids = set(new_last_msgs.keys())
    self.emitMsgsReceived(has_new=False)
    self.seeked.emit(time_sec)

  def stop(self):
    if self._loader_thread is not None:
      self._loader_thread.stop()
      if self._loader_thread.isRunning():
        if not self._loader_thread.wait(2000):
          # Force terminate if thread doesn't stop gracefully
          self._loader_thread.terminate()
          self._loader_thread.wait(1000)
      self._loader_thread = None

  @property
  def routeName(self) -> str:
    return self._route

  @property
  def carFingerprint(self) -> str:
    return self._fingerprint

  @property
  def isLoading(self) -> bool:
    return self._loading

  @property
  def duration(self) -> float:
    return self._duration

  @property
  def currentTime(self) -> float:
    return self._current_time
