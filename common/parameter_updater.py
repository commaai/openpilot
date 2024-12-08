import threading
import time

from openpilot.common.params import Params


class ParameterUpdater:
  def __init__(self, params_to_update: dict[str, str], update_interval: float = 0.1):
    """
    params_to_update: A dictionary where keys are parameter names, and values are their types ('bool' or 'str').
    Example: {"IsMetric": "bool", "LongitudinalPersonality": "str"}
    """
    self.params = Params()
    self.params_to_update = params_to_update
    self.param_values = {param: None for param in params_to_update}
    self.update_interval = update_interval

    self._update()  # Initial update

    self.mutex = threading.Lock()
    self.stop_event = threading.Event()
    self.update_thread: threading.Thread | None = None

  def start(self):
    if self.update_thread is None:
      self.update_thread = threading.Thread(target=self._update_periodically, daemon=True)
      self.update_thread.start()

  def stop(self):
    if self.update_thread:
      self.stop_event.set()
      self.update_thread.join()

  def get_param_value(self, param: str):
    with self.mutex:
      return self.param_values.get(param)

  def _update(self):
    new_values: dict[str, bool | str | None] = {}
    for param, param_type in self.params_to_update.items():
      if param_type == "bool":
        new_values[param] = self.params.get_bool(param)
      elif param_type == "str":
        new_values[param] = self.params.get(param)
      else:
        raise ValueError(f"Unsupported type {param_type} for parameter {param}")

    with self.mutex:
      self.param_values = new_values

  def _update_periodically(self):
    while not self.stop_event.is_set():
      self._update()
      time.sleep(self.update_interval)
