import uuid
import threading
import numpy as np
from collections import deque
import dearpygui.dearpygui as dpg
from abc import ABC, abstractmethod


class ViewPanel(ABC):
  """Abstract base class for all view panels that can be displayed in a plot container"""

  def __init__(self, panel_id: str = None):
    self.panel_id = panel_id or str(uuid.uuid4())
    self.title = "Untitled Panel"

  @abstractmethod
  def clear(self):
    pass

  @abstractmethod
  def create_ui(self, parent_tag: str):
    pass

  @abstractmethod
  def destroy_ui(self):
    pass

  @abstractmethod
  def get_panel_type(self) -> str:
    pass

  @abstractmethod
  def update(self):
    pass


class TimeSeriesPanel(ViewPanel):
  def __init__(self, data_manager, playback_manager, worker_manager, panel_id: str | None = None):
    super().__init__(panel_id)
    self.data_manager = data_manager
    self.playback_manager = playback_manager
    self.worker_manager = worker_manager
    self.title = "Time Series Plot"
    self.plot_tag = f"plot_{self.panel_id}"
    self.x_axis_tag = f"{self.plot_tag}_x_axis"
    self.y_axis_tag = f"{self.plot_tag}_y_axis"
    self.timeline_indicator_tag = f"{self.plot_tag}_timeline"
    self._ui_created = False
    self._series_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    self._last_plot_duration = 0
    self._update_lock = threading.RLock()
    self._results_deque: deque[tuple[str, list, list]] = deque()
    self._new_data = False

  def create_ui(self, parent_tag: str):
    self.data_manager.add_observer(self.on_data_loaded)
    with dpg.plot(height=-1, width=-1, tag=self.plot_tag, parent=parent_tag, drop_callback=self._on_series_drop, payload_type="TIMESERIES_PAYLOAD"):
      dpg.add_plot_legend()
      dpg.add_plot_axis(dpg.mvXAxis, no_label=True, tag=self.x_axis_tag)
      dpg.add_plot_axis(dpg.mvYAxis, no_label=True, tag=self.y_axis_tag)
      timeline_series_tag = dpg.add_inf_line_series(x=[0], label="Timeline", parent=self.y_axis_tag, tag=self.timeline_indicator_tag)
      dpg.bind_item_theme(timeline_series_tag, "global_timeline_theme")

    for series_path in list(self._series_data.keys()):
      self.add_series(series_path)
    self._ui_created = True

  def update(self):
    with self._update_lock:
      if not self._ui_created:
        return

      if self._new_data:  # handle new data in main thread
        self._new_data = False
        for series_path in list(self._series_data.keys()):
          self.add_series(series_path, update=True)

      while self._results_deque:  # handle downsampled results in main thread
        results = self._results_deque.popleft()
        for series_path, downsampled_time, downsampled_values in results:
          series_tag = f"series_{self.panel_id}_{series_path}"
          if dpg.does_item_exist(series_tag):
            dpg.set_value(series_tag, (downsampled_time, downsampled_values.astype(float)))

      # update timeline
      current_time_s = self.playback_manager.current_time_s
      dpg.set_value(self.timeline_indicator_tag, [[current_time_s], [0]])

      # update timeseries legend label
      for series_path, (time_array, value_array) in self._series_data.items():
        position = np.searchsorted(time_array, current_time_s, side='right') - 1
        if position >= 0 and (current_time_s - time_array[position]) <= 1.0:
          value = value_array[position]
          formatted_value = f"{value:.5f}" if np.issubdtype(type(value), np.floating) else str(value)
          series_tag = f"series_{self.panel_id}_{series_path}"
          if dpg.does_item_exist(series_tag):
            dpg.configure_item(series_tag, label=f"{series_path}: {formatted_value}")

      # downsample if plot zoom changed significantly
      plot_duration = dpg.get_axis_limits(self.x_axis_tag)[1] - dpg.get_axis_limits(self.x_axis_tag)[0]
      if plot_duration > self._last_plot_duration * 2 or plot_duration < self._last_plot_duration * 0.5:
        self._downsample_all_series(plot_duration)

  def _downsample_all_series(self, plot_duration):
    plot_width = dpg.get_item_rect_size(self.plot_tag)[0]
    if plot_width <= 0 or plot_duration <= 0:
      return

    self._last_plot_duration = plot_duration
    target_points_per_second = plot_width / plot_duration
    work_items = []
    for series_path, (time_array, value_array) in self._series_data.items():
      if len(time_array) == 0:
        continue
      series_duration = time_array[-1] - time_array[0] if len(time_array) > 1 else 1
      points_per_second = len(time_array) / series_duration
      if points_per_second > target_points_per_second * 2:
        target_points = max(int(target_points_per_second * series_duration), plot_width)
        work_items.append((series_path, time_array, value_array, target_points))
      elif dpg.does_item_exist(f"series_{self.panel_id}_{series_path}"):
        dpg.set_value(f"series_{self.panel_id}_{series_path}", (time_array, value_array.astype(float)))

    if work_items:
      self.worker_manager.submit_task(
        TimeSeriesPanel._downsample_worker, work_items, callback=lambda results: self._results_deque.append(results), task_id=f"downsample_{self.panel_id}"
      )

  def add_series(self, series_path: str, update: bool = False):
    with self._update_lock:
      if update or series_path not in self._series_data:
        self._series_data[series_path] = self.data_manager.get_timeseries(series_path)

      time_array, value_array = self._series_data[series_path]
      series_tag = f"series_{self.panel_id}_{series_path}"
      if dpg.does_item_exist(series_tag):
        dpg.set_value(series_tag, (time_array, value_array.astype(float)))
      else:
        line_series_tag = dpg.add_line_series(x=time_array, y=value_array.astype(float), label=series_path, parent=self.y_axis_tag, tag=series_tag)
        dpg.bind_item_theme(line_series_tag, "global_line_theme")
        dpg.fit_axis_data(self.x_axis_tag)
        dpg.fit_axis_data(self.y_axis_tag)
      plot_duration = dpg.get_axis_limits(self.x_axis_tag)[1] - dpg.get_axis_limits(self.x_axis_tag)[0]
      self._downsample_all_series(plot_duration)

  def destroy_ui(self):
    with self._update_lock:
      self.data_manager.remove_observer(self.on_data_loaded)
      if dpg.does_item_exist(self.plot_tag):
        dpg.delete_item(self.plot_tag)
      self._ui_created = False

  def get_panel_type(self) -> str:
    return "timeseries"

  def clear(self):
    with self._update_lock:
      for series_path in list(self._series_data.keys()):
        self.remove_series(series_path)

  def remove_series(self, series_path: str):
    with self._update_lock:
      if series_path in self._series_data:
        if dpg.does_item_exist(f"series_{self.panel_id}_{series_path}"):
          dpg.delete_item(f"series_{self.panel_id}_{series_path}")
        del self._series_data[series_path]

  def on_data_loaded(self, data: dict):
    self._new_data = True

  def _on_series_drop(self, sender, app_data, user_data):
    self.add_series(app_data)

  @staticmethod
  def _downsample_worker(series_path, time_array, value_array, target_points):
    if len(time_array) <= target_points:
      return series_path, time_array, value_array

    step = len(time_array) / target_points
    indices = []

    for i in range(target_points):
      start_idx = int(i * step)
      end_idx = int((i + 1) * step)
      if start_idx == end_idx:
        indices.append(start_idx)
      else:
        bucket_values = value_array[start_idx:end_idx]
        min_idx = start_idx + np.argmin(bucket_values)
        max_idx = start_idx + np.argmax(bucket_values)
        if min_idx != max_idx:
          indices.extend([min(min_idx, max_idx), max(min_idx, max_idx)])
        else:
          indices.append(min_idx)
    indices = sorted(set(indices))
    return series_path, time_array[indices], value_array[indices]
