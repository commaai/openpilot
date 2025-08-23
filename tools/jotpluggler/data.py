import threading
import numpy as np
from collections.abc import Callable
from openpilot.common.swaglog import cloudlog
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.log_time_series import msgs_to_time_series


class DataManager:
  def __init__(self):
    self.time_series_data = {}
    self.loading = False
    self.route_start_time_mono = 0.0
    self.duration = 0.0
    self._callbacks: list[Callable[[dict], None]] = []

  def add_callback(self, callback: Callable[[dict], None]):
    self._callbacks.append(callback)

  def remove_callback(self, callback: Callable[[dict], None]):
    if callback in self._callbacks:
      self._callbacks.remove(callback)

  def _notify_callbacks(self, data: dict):
    for callback in self._callbacks:
      try:
        callback(data)
      except Exception as e:
        cloudlog.exception(f"Error in data callback: {e}")

  def get_current_value(self, path: str, time_s: float, last_index: int | None = None):
    try:
      abs_time_s = self.route_start_time_mono + time_s
      msg_type, field_path = path.split('/', 1)
      ts_data = self.time_series_data[msg_type]
      t, v = ts_data['t'], ts_data[field_path]

      if len(t) == 0:
        return None, None

      if last_index is None:  # jump
        idx = np.searchsorted(t, abs_time_s, side='right') - 1
      else:  # continuous playback
        idx = last_index
        while idx < len(t) - 1 and t[idx + 1] < abs_time_s:
          idx += 1

      idx = max(0, idx)
      return v[idx], idx

    except (KeyError, IndexError):
      return None, None

  def get_all_paths(self) -> list[str]:
    all_paths = []
    for msg_type, data in self.time_series_data.items():
      for key in data.keys():
        if key != 't':
          all_paths.append(f"{msg_type}/{key}")
    return all_paths

  def is_path_plottable(self, path: str) -> bool:
    try:
      msg_type, field_path = path.split('/', 1)
      value_array = self.time_series_data.get(msg_type, {}).get(field_path)
      if value_array is not None:
        return np.issubdtype(value_array.dtype, np.number) or np.issubdtype(value_array.dtype, np.bool_)
    except (ValueError, KeyError):
      pass
    return False

  def get_time_series(self, path: str):
    try:
      msg_type, field_path = path.split('/', 1)
      ts_data = self.time_series_data[msg_type]
      time_array = ts_data['t']
      values = ts_data[field_path]

      if len(time_array) == 0:
        return None

      rel_time = time_array - self.route_start_time_mono
      return rel_time, values

    except (KeyError, ValueError):
      return None

  def load_route(self, route_name: str):
    if self.loading:
      return

    self.loading = True
    threading.Thread(target=self._load_route_background, args=(route_name,), daemon=True).start()

  def _load_route_background(self, route_name: str):
    try:
      lr = LogReader(route_name)
      raw_data = msgs_to_time_series(lr)
      processed_data = self._expand_list_fields(raw_data)

      min_time = float('inf')
      max_time = float('-inf')
      for data in processed_data.values():
        if len(data['t']) > 0:
          min_time = min(min_time, data['t'][0])
          max_time = max(max_time, data['t'][-1])

      self.time_series_data = processed_data
      self.route_start_time_mono = min_time if min_time != float('inf') else 0.0
      self.duration = max_time - min_time if max_time != float('-inf') else 0.0

      self._notify_callbacks({'time_series_data': processed_data, 'route_start_time_mono': self.route_start_time_mono, 'duration': self.duration})

    except Exception as e:
      cloudlog.exception(f"Error loading route {route_name}: {e}")
    finally:
      self.loading = False

  def _expand_list_fields(self, time_series_data):
    expanded_data = {}
    for msg_type, data in time_series_data.items():
      expanded_data[msg_type] = {}
      for field, values in data.items():
        if field == 't':
          expanded_data[msg_type]['t'] = values
          continue

        if values.dtype == object:  # ragged array
          lens = np.fromiter((len(v) for v in values), dtype=int, count=len(values))
          max_len = lens.max() if lens.size else 0
          if max_len > 0:
            arr = np.full((len(values), max_len), None, dtype=object)
            for i, v in enumerate(values):
              arr[i, : lens[i]] = v
            for i in range(max_len):
              sub_arr = arr[:, i]
              expanded_data[msg_type][f"{field}/{i}"] = sub_arr
        elif values.ndim > 1:  # regular multidimensional array
          for i in range(values.shape[1]):
            col_data = values[:, i]
            expanded_data[msg_type][f"{field}/{i}"] = col_data
        else:
          expanded_data[msg_type][field] = values
    return expanded_data
