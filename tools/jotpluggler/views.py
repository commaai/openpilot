import os
import re
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
    self._series_data: dict[str, tuple[list, list]] = {}
    self._last_plot_duration = 0
    self._update_lock = threading.RLock()
    self.results_deque: deque[tuple[str, list, list]] = deque()
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

      while self.results_deque:  # handle downsampled results in main thread
        results = self.results_deque.popleft()
        for series_path, downsampled_time, downsampled_values in results:
          series_tag = f"series_{self.panel_id}_{series_path}"
          if dpg.does_item_exist(series_tag):
            dpg.set_value(series_tag, [downsampled_time, downsampled_values])

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
        dpg.set_value(f"series_{self.panel_id}_{series_path}", [time_array, value_array])

    if work_items:
      self.worker_manager.submit_task(
        TimeSeriesPanel._downsample_worker, work_items, callback=lambda results: self.results_deque.append(results), task_id=f"downsample_{self.panel_id}"
      )

  def add_series(self, series_path: str, update: bool = False):
    with self._update_lock:
      if update or series_path not in self._series_data:
        self._series_data[series_path] = self.data_manager.get_timeseries(series_path)

      time_array, value_array = self._series_data[series_path]
      series_tag = f"series_{self.panel_id}_{series_path}"
      if dpg.does_item_exist(series_tag):
        dpg.set_value(series_tag, [time_array, value_array])
      else:
        line_series_tag = dpg.add_line_series(x=time_array, y=value_array, label=series_path, parent=self.y_axis_tag, tag=series_tag)
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


class DataTreeNode:
  def __init__(self, name: str, full_path: str = "", parent=None):
    self.name = name
    self.full_path = full_path
    self.parent = parent
    self.children: dict[str, DataTreeNode] = {}
    self.is_leaf = False
    self.child_count = 0
    self.is_plottable_cached: bool | None = None
    self.ui_created = False
    self.children_ui_created = False
    self.ui_tag: str | None = None


class DataTreeView:
  MAX_NODES_PER_FRAME = 50

  def __init__(self, data_manager, playback_manager):
    self.data_manager = data_manager
    self.playback_manager = playback_manager
    self.current_search = ""
    self.data_tree = DataTreeNode(name="root")
    self.build_queue: deque[tuple[DataTreeNode, str | None, str | int]] = deque()
    self._all_paths_cache: set[str] = set()
    self._item_handlers: set[str] = set()
    self.avg_char_width = None
    self.data_manager.add_observer(self._on_data_loaded)
    self.queued_search = None
    self.new_data = False
    self._ui_lock = threading.RLock()

  def create_ui(self, parent_tag: str):
    with dpg.child_window(parent=parent_tag, border=False, width=-1, height=-1):
      dpg.add_text("Available Data")
      dpg.add_separator()
      dpg.add_input_text(tag="search_input", width=-1, hint="Search fields...", callback=self.search_data)
      dpg.add_separator()
      with dpg.group(tag="data_tree_container", track_offset=True):
        pass

  def _on_data_loaded(self, data: dict):
    if data.get('segment_added'):
      self.new_data = True

  def _populate_tree(self):
    self._clear_ui()
    self.data_tree = self._add_paths_to_tree(self._all_paths_cache, incremental=False)
    if self.data_tree:
      self._request_children_build(self.data_tree)

  def _add_paths_to_tree(self, paths, incremental=False):
    search_term = self.current_search.strip().lower()
    filtered_paths = [path for path in paths if self._should_show_path(path, search_term)]
    target_tree = self.data_tree if incremental else DataTreeNode(name="root")

    if not filtered_paths:
      return target_tree

    parent_nodes_to_recheck = set()
    for path in sorted(filtered_paths):
      parts = path.split('/')
      current_node = target_tree
      current_path_prefix = ""

      for i, part in enumerate(parts):
        current_path_prefix = f"{current_path_prefix}/{part}" if current_path_prefix else part
        if i < len(parts) - 1:
          parent_nodes_to_recheck.add(current_node)  # for incremental changes from new data
        if part not in current_node.children:
          current_node.children[part] = DataTreeNode(name=part, full_path=current_path_prefix, parent=current_node)
        current_node = current_node.children[part]

      if not current_node.is_leaf:
        current_node.is_leaf = True

    self._calculate_child_counts(target_tree)
    if incremental:
      for p_node in parent_nodes_to_recheck:
        p_node.children_ui_created = False
        self._request_children_build(p_node)
    return target_tree

  def update_frame(self, font):
    with self._ui_lock:
      if self.avg_char_width is None and dpg.is_dearpygui_running():
        self.avg_char_width = self.calculate_avg_char_width(font)

      if self.new_data:
        current_paths = set(self.data_manager.get_all_paths())
        new_paths = current_paths - self._all_paths_cache
        if new_paths:
          all_paths_empty = not self._all_paths_cache
          self._all_paths_cache = current_paths
          if all_paths_empty:
            self._populate_tree()
          else:
            self._add_paths_to_tree(new_paths, incremental=True)
        self.new_data = False
        return

      if self.queued_search is not None:
        self.current_search = self.queued_search
        self._all_paths_cache = set(self.data_manager.get_all_paths())
        self._populate_tree()
        self.queued_search = None
        return

      nodes_processed = 0
      while self.build_queue and nodes_processed < self.MAX_NODES_PER_FRAME:
        child_node, parent_tag, before_tag = self.build_queue.popleft()
        if not child_node.ui_created:
          if child_node.is_leaf:
            self._create_leaf_ui(child_node, parent_tag, before_tag)
          else:
            self._create_tree_node_ui(child_node, parent_tag, before_tag)
        nodes_processed += 1

  def search_data(self):
    self.queued_search = dpg.get_value("search_input")

  def _clear_ui(self):
    for handler_tag in self._item_handlers:
      dpg.configure_item(handler_tag, show=False)
    dpg.set_frame_callback(dpg.get_frame_count() + 1, callback=self._delete_handlers, user_data=list(self._item_handlers))
    self._item_handlers.clear()

    if dpg.does_item_exist("data_tree_container"):
      dpg.delete_item("data_tree_container", children_only=True)

    self.build_queue.clear()

  def _delete_handlers(self, sender, app_data, user_data):
    for handler in user_data:
      dpg.delete_item(handler)

  def _calculate_child_counts(self, node: DataTreeNode):
    if node.is_leaf:
      node.child_count = 0
    else:
      node.child_count = len(node.children)
      for child in node.children.values():
        self._calculate_child_counts(child)

  def _create_tree_node_ui(self, node: DataTreeNode, parent_tag: str, before: str | int):
    tag = f"tree_{node.full_path}"
    node.ui_tag = tag
    label = f"{node.name} ({node.child_count} fields)"
    search_term = self.current_search.strip().lower()
    should_open = bool(search_term) and len(search_term) > 1 and any(search_term in path for path in self._get_descendant_paths(node))
    if should_open and node.parent and node.parent.child_count > 100 and node.child_count > 2: # don't fully autoexpand large lists (only affects procLog rn)
      label += " (+)"
      should_open = False

    with dpg.tree_node(label=label, parent=parent_tag, tag=tag, default_open=should_open, open_on_arrow=True, open_on_double_click=True, before=before):
      with dpg.item_handler_registry() as handler_tag:
        dpg.add_item_toggled_open_handler(callback=lambda s, a, u: self._request_children_build(node, handler_tag))
        dpg.add_item_visible_handler(callback=lambda s, a, u: self._request_children_build(node, handler_tag))
      dpg.bind_item_handler_registry(tag, handler_tag)
      self._item_handlers.add(handler_tag)

    node.ui_created = True

  def _create_leaf_ui(self, node: DataTreeNode, parent_tag: str, before: str | int):
    half_split_size = dpg.get_item_rect_size("sidebar_window")[0] // 2

    with dpg.group(parent=parent_tag, horizontal=True, xoffset=half_split_size, tag=f"group_{node.full_path}", before=before) as draggable_group:
      dpg.add_text(node.name)
      dpg.add_text("N/A", tag=f"value_{node.full_path}")
      if node.is_plottable_cached is None:
        node.is_plottable_cached = self.data_manager.is_plottable(node.full_path)
      if node.is_plottable_cached:
        with dpg.drag_payload(parent=draggable_group, drag_data=node.full_path, payload_type="TIMESERIES_PAYLOAD"):
          dpg.add_text(f"Plot: {node.full_path}")

    with dpg.item_handler_registry() as handler_tag:
      dpg.add_item_visible_handler(callback=self._on_item_visible, user_data=node.full_path)
    dpg.bind_item_handler_registry(draggable_group, handler_tag)
    self._item_handlers.add(handler_tag)

    node.ui_created = True
    node.ui_tag = f"value_{node.full_path}"

  def _on_item_visible(self, sender, app_data, user_data):
    with self._ui_lock:
      path = user_data
      group_tag = f"group_{path}"
      value_tag = f"value_{path}"

      if not self.avg_char_width or not dpg.does_item_exist(group_tag) or not dpg.does_item_exist(value_tag):
        return

      value_column_width = dpg.get_item_rect_size("sidebar_window")[0] // 2
      dpg.configure_item(group_tag, xoffset=value_column_width)

      value = self.data_manager.get_value_at(path, self.playback_manager.current_time_s)
      if value is not None:
        formatted_value = self.format_and_truncate(value, value_column_width, self.avg_char_width)
        dpg.set_value(value_tag, formatted_value)
      else:
        dpg.set_value(value_tag, "N/A")

  def _request_children_build(self, node: DataTreeNode, handler_tag=None):
    with self._ui_lock:
      if not node.children_ui_created and (node.name == "root" or (node.ui_tag is not None and dpg.get_value(node.ui_tag))):  # check root or node expanded
        parent_tag = "data_tree_container" if node.name == "root" else node.ui_tag
        sorted_children = sorted(node.children.values(), key=self._natural_sort_key)

        for i, child_node in enumerate(sorted_children):
          if not child_node.ui_created:
            before_tag: int | str = 0
            for j in range(i + 1, len(sorted_children)):  # when incrementally building get "before_tag" for correct ordering
              next_child = sorted_children[j]
              if next_child.ui_created:
                candidate_tag = f"group_{next_child.full_path}" if next_child.is_leaf else f"tree_{next_child.full_path}"
                if dpg.does_item_exist(candidate_tag):
                  before_tag = candidate_tag
                  break
            self.build_queue.append((child_node, parent_tag, before_tag))
        node.children_ui_created = True

  def _should_show_path(self, path: str, search_term: str) -> bool:
    if 'DEPRECATED' in path and not os.environ.get('SHOW_DEPRECATED'):
      return False
    return not search_term or search_term in path.lower()

  def _natural_sort_key(self, node: DataTreeNode):
    node_type_key = node.is_leaf
    parts = [int(p) if p.isdigit() else p.lower() for p in re.split(r'(\d+)', node.name) if p]
    return (node_type_key, parts)

  def _get_descendant_paths(self, node: DataTreeNode):
    for child_name, child_node in node.children.items():
      child_name_lower = child_name.lower()
      if child_node.is_leaf:
        yield child_name_lower
      else:
        for path in self._get_descendant_paths(child_node):
          yield f"{child_name_lower}/{path}"

  @staticmethod
  def calculate_avg_char_width(font):
    sample_text = "abcdefghijklmnopqrstuvwxyz0123456789"
    if size := dpg.get_text_size(sample_text, font=font):
      return size[0] / len(sample_text)
    return None

  @staticmethod
  def format_and_truncate(value, available_width: float, avg_char_width: float) -> str:
    s = f"{value:.5f}" if np.issubdtype(type(value), np.floating) else str(value)
    max_chars = int(available_width / avg_char_width) - 3
    if len(s) > max_chars:
      return s[: max(0, max_chars)] + "..."
    return s
