import threading
import time

from openpilot.common.params import Params


class ParameterUpdater:
  def __init__(self):
    self.params = Params()
    self.mutex = threading.Lock()
    self.is_metric = False
    self.experimental_mode = False

    self._update()  # Initial update
    self.stop_event = threading.Event()
    self.update_thread = threading.Thread(target=self._update_periodically)
    self.update_thread.start()

  def stop(self):
    self.stop_event.set()
    self.update_thread.join()

  def _update(self):
    is_metric = self.params.get_bool("IsMetric")
    experimental_mode = self.params.get_bool("ExperimentalMode")
    with self.mutex:
      self.is_metric = is_metric
      self.experimental_mode = experimental_mode

  def _update_periodically(self):
    while not self.stop_event.is_set():
      self._update()
      time.sleep(0.1)
