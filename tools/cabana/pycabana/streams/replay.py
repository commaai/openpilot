"""ReplayStream - loads CAN data from openpilot routes."""

from PySide6.QtCore import QThread, Signal as QtSignal, Qt

from openpilot.tools.cabana.pycabana.dbc.dbc import MessageId, CanData
from openpilot.tools.cabana.pycabana.streams.abstract import AbstractStream


class LogLoaderThread(QThread):
  """Background thread that loads log data and processes it."""

  dataReady = QtSignal(object, object)
  progress = QtSignal(int, int)
  finished = QtSignal()

  BATCH_SIZE = 10000

  def __init__(self, route: str, parent=None):
    super().__init__(parent)
    self.route = route
    self._stop_requested = False

  def run(self):
    try:
      from openpilot.tools.lib.logreader import LogReader

      lr = LogReader(self.route)
      total_msgs = 0
      can_msgs = 0
      last_msgs: dict[MessageId, CanData] = {}
      start_ts = 0
      last_emit_count = 0

      for msg in lr:
        if self._stop_requested:
          break

        total_msgs += 1
        if msg.which() == 'can':
          for c in msg.can:
            msg_id = MessageId(c.src, c.address)
            mono_time = msg.logMonoTime

            if start_ts == 0:
              start_ts = mono_time

            if msg_id not in last_msgs:
              last_msgs[msg_id] = CanData()

            last_msgs[msg_id].count += 1
            last_msgs[msg_id].dat = bytes(c.dat)
            last_msgs[msg_id].ts = (mono_time - start_ts) / 1e9
            can_msgs += 1

            if can_msgs - last_emit_count >= self.BATCH_SIZE:
              snapshot = {k: CanData(v.ts, v.count, v.freq, v.dat) for k, v in last_msgs.items()}
              self.dataReady.emit(snapshot, set(last_msgs.keys()))
              self.progress.emit(can_msgs, total_msgs)
              last_emit_count = can_msgs

      snapshot = {k: CanData(v.ts, v.count, v.freq, v.dat) for k, v in last_msgs.items()}
      self.dataReady.emit(snapshot, set(last_msgs.keys()))
      self.progress.emit(can_msgs, total_msgs)
    except Exception as e:
      import traceback

      print(f"Error loading route: {e}")
      traceback.print_exc()
    finally:
      self.finished.emit()

  def stop(self):
    self._stop_requested = True


class ReplayStream(AbstractStream):
  """Stream that replays CAN data from an openpilot route."""

  loadProgress = QtSignal(int, int)
  loadFinished = QtSignal()

  def __init__(self, parent=None):
    super().__init__(parent)
    self._route: str = ""
    self._fingerprint: str = ""
    self._loader_thread: LogLoaderThread | None = None
    self._loading = False

  def __del__(self):
    self.stop()

  def loadRoute(self, route: str) -> bool:
    if self._loading:
      return False

    self._route = route
    self._loading = True

    self.events.clear()
    self.last_msgs.clear()
    self._msg_ids.clear()
    self.start_ts = 0

    self._loader_thread = LogLoaderThread(route, self)
    self._loader_thread.dataReady.connect(self._onDataReady, Qt.QueuedConnection)
    self._loader_thread.progress.connect(self._onProgress, Qt.QueuedConnection)
    self._loader_thread.finished.connect(self._onLoadFinished, Qt.QueuedConnection)
    self._loader_thread.start()

    return True

  def _onDataReady(self, snapshot: dict[MessageId, CanData], msg_ids: set[MessageId]):
    self.last_msgs = snapshot
    new_ids = msg_ids - self._msg_ids
    self._msg_ids = msg_ids
    self.emitMsgsReceived(has_new=bool(new_ids))

  def _onProgress(self, can_msgs: int, total_msgs: int):
    self.loadProgress.emit(can_msgs, total_msgs)

  def _onLoadFinished(self):
    self._loading = False
    self.loadFinished.emit()
    self._extractFingerprint()

  def _extractFingerprint(self):
    try:
      from openpilot.tools.lib.logreader import LogReader

      lr = LogReader(self._route)
      for msg in lr:
        if msg.which() == 'carParams':
          self._fingerprint = msg.carParams.carFingerprint
          break
    except Exception:
      pass

  def stop(self):
    if self._loader_thread is not None:
      self._loader_thread.stop()
      if self._loader_thread.isRunning():
        self._loader_thread.wait(5000)

  @property
  def routeName(self) -> str:
    return self._route

  @property
  def carFingerprint(self) -> str:
    return self._fingerprint

  @property
  def isLoading(self) -> bool:
    return self._loading
