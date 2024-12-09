import threading
import time

from openpilot.common.params import Params


class ParameterUpdater:
  def __init__(self, params_to_update: list[str], update_interval: float = 0.1):
    self.params = Params()
    self.params_to_update = params_to_update
    self.param_values: dict[str, str | None] = {}
    self.update_interval = update_interval
    self.mutex = threading.Lock()
    self.stop_event = threading.Event()
    self.update_thread: threading.Thread | None = None

    # Initial update
    self._update()

  def get(self, param: str) -> str | None:
    with self.mutex:
      return self.param_values[param]

  def get_bool(self, param: str) -> bool:
    return self.get(param) == b'1'

  def get_int(self, param: str, def_val: int = 0) -> int:
    value = self.get(param)
    try:
      return int(value) if value is not None else def_val
    except (ValueError, TypeError):
      return def_val

  def start(self) -> None:
    if self.update_thread is None:
      self.stop_event.clear()
      self.update_thread = threading.Thread(target=self._update_periodically, daemon=True)
      self.update_thread.start()

  def stop(self) -> None:
    if self.update_thread:
      self.stop_event.set()
      self.update_thread.join()
      self.update_thread = None

  def _update(self) -> None:
    new_values = {param: self.params.get(param) for param in self.params_to_update}
    with self.mutex:
      self.param_values = new_values

  def _update_periodically(self) -> None:
    while not self.stop_event.is_set():
      self._update()
      time.sleep(self.update_interval)
