import threading
import time
from typing import Dict, Union
from openpilot.common.params import Params


class ParameterUpdater:
  def __init__(self, params_to_update: Dict[str, str]):
    """
    params_to_update: A dictionary where keys are parameter names, and values are their types ('bool' or 'str').
    Example: {"IsMetric": "bool", "LongitudinalPersonality": "str"}
    """
    self.params = Params()
    self.params_to_update = params_to_update
    self.param_values: Dict[str, Union[bool, str]] = {param: None for param in params_to_update}

    self._update()  # Initial update

    self.mutex = threading.Lock()
    self.stop_event = threading.Event()
    self.update_thread = None

  def start(self) -> None:
    if self.update_thread is None or not self.update_thread.is_alive():
      self.update_thread = threading.Thread(target=self._update_periodically, daemon=True)
      self.update_thread.start()

  def stop(self) -> None:
    if self.update_thread and self.update_thread.is_alive():
      self.stop_event.set()
      self.update_thread.join()

  def get_param_value(self, param: str) -> Union[bool, str, None]:
    with self.mutex:
      return self.param_values.get(param)

  def _update(self) -> None:
    new_values: Dict[str, Union[bool, str]] = {}
    for param, param_type in self.params_to_update.items():
      if param_type == "bool":
        new_values[param] = self.params.get_bool(param)
      elif param_type == "str":
        new_values[param] = self.params.get(param)
      else:
        raise ValueError(f"Unsupported type {param_type} for parameter {param}")

    with self.mutex:
      self.param_values = new_values

  def _update_periodically(self) -> None:
    while not self.stop_event.is_set():
      self._update()
      time.sleep(0.1)
