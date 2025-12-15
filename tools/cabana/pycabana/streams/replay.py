"""ReplayStream - loads CAN data from openpilot routes."""

from PySide6.QtCore import QThread, Signal as QtSignal

from openpilot.tools.cabana.pycabana.dbc.dbc import CanEvent
from openpilot.tools.cabana.pycabana.streams.abstract import AbstractStream


class LogLoaderThread(QThread):
  """Background thread that loads log data."""

  progress = QtSignal(int, int)  # (current, total)
  finished = QtSignal()
  eventLoaded = QtSignal(object)  # CanEvent

  def __init__(self, route: str, parent=None):
    super().__init__(parent)
    self.route = route
    self._stop_requested = False

  def run(self):
    try:
      from openpilot.tools.lib.logreader import LogReader

      print(f"[LogLoaderThread] Starting to load route: {self.route}")
      lr = LogReader(self.route)
      total_msgs = 0
      can_msgs = 0

      for msg in lr:
        if self._stop_requested:
          break

        total_msgs += 1
        if msg.which() == 'can':
          for c in msg.can:
            event = CanEvent(
              src=c.src,
              address=c.address,
              mono_time=msg.logMonoTime,
              dat=bytes(c.dat),
            )
            self.eventLoaded.emit(event)
            can_msgs += 1

        # Emit progress every 1000 messages
        if total_msgs % 1000 == 0:
          self.progress.emit(can_msgs, total_msgs)
          print(f"[LogLoaderThread] Progress: {can_msgs} CAN msgs, {total_msgs} total msgs")

      self.progress.emit(can_msgs, total_msgs)
      print(f"[LogLoaderThread] Finished: {can_msgs} CAN msgs, {total_msgs} total msgs")
    except Exception as e:
      import traceback

      print(f"[LogLoaderThread] Error loading route: {e}")
      traceback.print_exc()
    finally:
      self.finished.emit()

  def stop(self):
    self._stop_requested = True


class ReplayStream(AbstractStream):
  """Stream that replays CAN data from an openpilot route."""

  # Additional signals for replay-specific events
  loadProgress = QtSignal(int, int)  # (can_msgs, total_msgs)
  loadFinished = QtSignal()

  def __init__(self, parent=None):
    super().__init__(parent)
    self._route: str = ""
    self._fingerprint: str = ""
    self._loader_thread: LogLoaderThread | None = None
    self._loading = False

  def loadRoute(self, route: str) -> bool:
    """Start loading a route. Returns True if loading started."""
    if self._loading:
      return False

    self._route = route
    self._loading = True

    # Clear existing data
    self.events.clear()
    self.last_msgs.clear()
    self._msg_ids.clear()
    self.start_ts = 0

    # Start loader thread
    self._loader_thread = LogLoaderThread(route, self)
    self._loader_thread.eventLoaded.connect(self._onEventLoaded)
    self._loader_thread.progress.connect(self._onProgress)
    self._loader_thread.finished.connect(self._onLoadFinished)
    self._loader_thread.start()

    return True

  def _onEventLoaded(self, event: CanEvent):
    """Handle an event loaded from the log."""
    self.updateEvent(event)

  def _onProgress(self, can_msgs: int, total_msgs: int):
    """Handle progress update."""
    self.loadProgress.emit(can_msgs, total_msgs)
    # Emit msgsReceived so UI can update
    self.emitMsgsReceived(has_new=True)

  def _onLoadFinished(self):
    """Handle load completion."""
    self._loading = False
    self.loadFinished.emit()
    # Final update
    self.emitMsgsReceived(has_new=False)

    # Try to extract fingerprint from carParams
    self._extractFingerprint()

  def _extractFingerprint(self):
    """Try to extract car fingerprint from loaded data."""
    try:
      from openpilot.tools.lib.logreader import LogReader

      lr = LogReader(self._route)
      for msg in lr:
        if msg.which() == 'carParams':
          self._fingerprint = msg.carParams.carFingerprint
          break
    except Exception:
      pass  # Fingerprint extraction is optional

  def stop(self):
    """Stop loading."""
    if self._loader_thread and self._loader_thread.isRunning():
      self._loader_thread.stop()
      self._loader_thread.wait()

  @property
  def routeName(self) -> str:
    return self._route

  @property
  def carFingerprint(self) -> str:
    return self._fingerprint

  @property
  def isLoading(self) -> bool:
    return self._loading
