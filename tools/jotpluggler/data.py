import threading
import numpy as np
from abc import ABC, abstractmethod
from typing import Any
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.log_time_series import msgs_to_time_series


# TODO: support cereal/ZMQ streaming
class DataSource(ABC):
  @abstractmethod
  def load_data(self) -> dict[str, Any]:
    pass

  @abstractmethod
  def get_duration(self) -> float:
    pass


class LogReaderSource(DataSource):
  def __init__(self, route_name: str):
    self.route_name = route_name
    self._duration = 0.0
    self._start_time_mono = 0.0

  def load_data(self) -> dict[str, Any]:
    lr = LogReader(self.route_name)
    raw_time_series = msgs_to_time_series(lr)
    processed_data = self._expand_list_fields(raw_time_series)

    # Calculate timing information
    times = [data['t'] for data in processed_data.values() if 't' in data and len(data['t']) > 0]
    if times:
      all_times = np.concatenate(times)
      self._start_time_mono = all_times.min()
      self._duration = all_times.max() - self._start_time_mono

    return {'time_series_data': processed_data, 'route_start_time_mono': self._start_time_mono, 'duration': self._duration}

  def get_duration(self) -> float:
    return self._duration

  # TODO: lists are expanded, but lists of structs are not
  def _expand_list_fields(self, time_series_data):
    expanded_data = {}
    for msg_type, data in time_series_data.items():
      expanded_data[msg_type] = {}
      for field, values in data.items():
        if field == 't':
          expanded_data[msg_type]['t'] = values
          continue

        if isinstance(values, np.ndarray) and values.dtype == object:  # ragged array
          lens = np.fromiter((len(v) for v in values), dtype=int, count=len(values))
          max_len = lens.max() if lens.size else 0
          if max_len > 0:
            arr = np.full((len(values), max_len), None, dtype=object)
            for i, v in enumerate(values):
              arr[i, : lens[i]] = v
            for i in range(max_len):
              sub_arr = arr[:, i]
              expanded_data[msg_type][f"{field}/{i}"] = sub_arr
        elif isinstance(values, np.ndarray) and values.ndim > 1:  # regular array
          for i in range(values.shape[1]):
            col_data = values[:, i]
            expanded_data[msg_type][f"{field}/{i}"] = col_data
        else:
          expanded_data[msg_type][field] = values
    return expanded_data


class DataLoadedEvent:
  def __init__(self, data: dict[str, Any]):
    self.data = data


class Observer(ABC):
  @abstractmethod
  def on_data_loaded(self, event: DataLoadedEvent):
    pass


class DataManager:
  def __init__(self):
    self.time_series_data = {}
    self.loading = False
    self.route_start_time_mono = 0.0
    self.duration = 100.0
    self._observers: list[Observer] = []

  def add_observer(self, observer: Observer):
    self._observers.append(observer)

  def remove_observer(self, observer: Observer):
    if observer in self._observers:
      self._observers.remove(observer)

  def _notify_observers(self, event: DataLoadedEvent):
    for observer in self._observers:
      observer.on_data_loaded(event)

  def get_current_value_for_path(self, path: str, time_s: float, last_index: int | None = None):
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
      if value_array is not None:  # only numbers and bools are plottable
        return np.issubdtype(value_array.dtype, np.number) or np.issubdtype(value_array.dtype, np.bool_)
    except (ValueError, KeyError):
      pass
    return False

  def get_time_series_data(self, path: str) -> tuple | None:
    try:
      msg_type, field_path = path.split('/', 1)
      ts_data = self.time_series_data[msg_type]
      time_array = ts_data['t']
      plot_values = ts_data[field_path]

      if len(time_array) == 0:
        return None

      rel_time_array = time_array - self.route_start_time_mono
      return rel_time_array, plot_values

    except (KeyError, ValueError):
      return None

  def load_route(self, route_name: str):
    if self.loading:
      return

    self.loading = True
    data_source = LogReaderSource(route_name)
    threading.Thread(target=self._load_in_background, args=(data_source,), daemon=True).start()

  def _load_in_background(self, data_source: DataSource):
    try:
      data = data_source.load_data()
      self.time_series_data = data['time_series_data']
      self.route_start_time_mono = data['route_start_time_mono']
      self.duration = data['duration']

      self._notify_observers(DataLoadedEvent(data))

    except Exception as e:
      print(f"Error loading route: {e}")
    finally:
      self.loading = False
