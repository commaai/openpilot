import os
import re
import uuid
import threading
import numpy as np
from collections import deque
import dearpygui.dearpygui as dpg
from abc import ABC, abstractmethod
from openpilot.tools.jotpluggler.data import DataManager


class ViewPanel(ABC):
  """Abstract base class for all view panels that can be displayed in a plot container"""

  def __init__(self, panel_id: str = None):
    self.panel_id = panel_id or str(uuid.uuid4())
    self.title = "Untitled Panel"

  @abstractmethod
  def create_ui(self, parent_tag: str):
    pass

  @abstractmethod
  def destroy_ui(self):
    pass

  @abstractmethod
  def get_panel_type(self) -> str:
    pass


class TimeSeriesPanel(ViewPanel):
  def __init__(self, data_manager: DataManager, playback_manager, panel_id: str | None = None):
    super().__init__(panel_id)
    self.data_manager = data_manager
    self.playback_manager = playback_manager
    self.title = "Time Series Plot"
    self.plot_tag: str | None = None
    self.x_axis_tag: str | None = None
    self.y_axis_tag: str | None = None
    self.timeline_indicator_tag: str | None = None
    self._ui_created = False
    self._series_data: dict[str, tuple] = {}
    self.data_manager.add_observer(self.on_data_loaded)

  def create_ui(self, parent_tag: str):
    self.plot_tag = f"plot_{self.panel_id}"
    self.x_axis_tag = f"{self.plot_tag}_x_axis"
    self.y_axis_tag = f"{self.plot_tag}_y_axis"
    self.timeline_indicator_tag = f"{self.plot_tag}_timeline"

    with dpg.plot(height=-1, width=-1, tag=self.plot_tag, parent=parent_tag, drop_callback=self._on_series_drop, payload_type="TIMESERIES_PAYLOAD"):
      dpg.add_plot_legend()
      dpg.add_plot_axis(dpg.mvXAxis, no_label=True, tag=self.x_axis_tag)
      dpg.add_plot_axis(dpg.mvYAxis, no_label=True, tag=self.y_axis_tag)
      timeline_series_tag = dpg.add_inf_line_series(x=[0], label="Timeline", parent=self.y_axis_tag, tag=self.timeline_indicator_tag)
      dpg.bind_item_theme(timeline_series_tag, "global_timeline_theme")

    for series_path in list(self._series_data.keys()):
      self.add_series(series_path)

    self._ui_created = True

  def update_timeline_indicator(self, current_time_s: float):
    if not self._ui_created or not dpg.does_item_exist(self.timeline_indicator_tag):
      return

    dpg.set_value(self.timeline_indicator_tag, [[current_time_s], [0]])

    for series_path, (rel_time_array, value_array) in self._series_data.items():
      position = np.searchsorted(rel_time_array, current_time_s, side='right') - 1
      value = None

      if position >= 0 and (current_time_s - rel_time_array[position]) <= 1.0:
        value = value_array[position]

      if value is not None:
        if np.issubdtype(type(value), np.floating):
          formatted_value = f"{value:.5f}"
        else:
          formatted_value = str(value)

        series_tag = f"series_{self.panel_id}_{series_path}"
        legend_label = f"{series_path}: {formatted_value}"

        if dpg.does_item_exist(series_tag):
          dpg.configure_item(series_tag, label=legend_label)

  def add_series(self, series_path: str, update: bool = False) -> bool:
    if update or series_path not in self._series_data:
      self._series_data[series_path] = self.data_manager.get_timeseries(series_path)

    rel_time_array, value_array = self._series_data[series_path]
    series_tag = f"series_{self.panel_id}_{series_path}"

    if dpg.does_item_exist(series_tag):
      dpg.set_value(series_tag, [rel_time_array, value_array])
    else:
      line_series_tag = dpg.add_line_series(x=rel_time_array, y=value_array, label=series_path, parent=self.y_axis_tag, tag=series_tag)
      dpg.bind_item_theme(line_series_tag, "global_line_theme")
      dpg.fit_axis_data(self.x_axis_tag)
      dpg.fit_axis_data(self.y_axis_tag)
    return True

  def destroy_ui(self):
    if self.plot_tag and dpg.does_item_exist(self.plot_tag):
      dpg.delete_item(self.plot_tag)
    self._ui_created = False

  def get_panel_type(self) -> str:
    return "timeseries"

  def clear_all_series(self):
    for series_path in list(self._series_data.keys()):
      self.remove_series(series_path)

  def remove_series(self, series_path: str):
    if series_path in self._series_data:
      series_tag = f"series_{self.panel_id}_{series_path}"
      if dpg.does_item_exist(series_tag):
        dpg.delete_item(series_tag)
      del self._series_data[series_path]

  def on_data_loaded(self, data: dict):
    for series_path in list(self._series_data.keys()):
      self.add_series(series_path, update=True)

  def _on_series_drop(self, sender, app_data, user_data):
    self.add_series(app_data)


class DataTreeNode:
  def __init__(self, name: str, full_path: str = ""):
    self.name = name
    self.full_path = full_path
    self.children: dict[str, DataTreeNode] = {}
    self.is_leaf = False
    self.child_count = 0
    self.is_plottable_cached: bool | None = None
    self.ui_created = False
    self.ui_tag: str | None = None


class DataTreeView:
  MAX_ITEMS_PER_FRAME = 50

  def __init__(self, data_manager: DataManager, ui_lock: threading.Lock):
    self.data_manager = data_manager
    self.ui_lock = ui_lock
    self.current_search = ""
    self.data_tree = DataTreeNode(name="root")
    self.ui_render_queue: deque[tuple[DataTreeNode, str, str, bool]] = deque() # (node, parent_tag, search_term, is_leaf)
    self.visible_expanded_nodes: set[str] = set()
    self.created_leaf_paths: set[str] = set()
    self._all_paths_cache: list[str] = []
    self._previous_paths_set: set[str] = set()
    self.data_manager.add_observer(self._on_data_loaded)

  def _on_data_loaded(self, data: dict):
    with self.ui_lock:
      if data.get('segment_added'):
        current_paths = set(self.data_manager.get_all_paths())
        new_paths = current_paths - self._previous_paths_set
        if new_paths:
          self._all_paths_cache = list(current_paths)
          if not self._previous_paths_set:
            self._populate_tree()
          else:
            self._add_paths_to_tree(new_paths, incremental=True)
          self._previous_paths_set = current_paths.copy()

  def _populate_tree(self):
    self._clear_ui()
    search_term = self.current_search.strip().lower()
    self.data_tree = self._add_paths_to_tree(self._all_paths_cache, incremental=False)
    for child in sorted(self.data_tree.children.values(), key=self._natural_sort_key):
      self.ui_render_queue.append((child, "data_tree_container", search_term, child.is_leaf))


  def _add_paths_to_tree(self, paths, incremental=False):
    search_term = self.current_search.strip().lower()
    filtered_paths = [path for path in paths if self._should_show_path(path, search_term)]
    target_tree = self.data_tree if incremental else DataTreeNode(name="root")

    if not filtered_paths:
      return target_tree

    nodes_to_update = set() if incremental else None

    for path in sorted(filtered_paths):
      parts = path.split('/')
      current_node = target_tree
      current_path_prefix = ""

      for i, part in enumerate(parts):
        current_path_prefix = f"{current_path_prefix}/{part}" if current_path_prefix else part

        if part not in current_node.children:
          current_node.children[part] = DataTreeNode(name=part, full_path=current_path_prefix)
          if incremental:
            nodes_to_update.add(current_node)

        current_node = current_node.children[part]
        if incremental and i < len(parts) - 1:
          nodes_to_update.add(current_node)

      if not current_node.is_leaf:
        current_node.is_leaf = True
        if incremental:
          nodes_to_update.add(current_node)

    self._calculate_child_counts(target_tree)
    if incremental:
      self._queue_new_ui_items(filtered_paths, search_term)
    return target_tree

  def _queue_new_ui_items(self, new_paths, search_term):
    for path in new_paths:
      parts = path.split('/')
      parent_path = '/'.join(parts[:-1]) if len(parts) > 1 else ""
      if parent_path == "" or parent_path in self.visible_expanded_nodes:
        parent_tag = "data_tree_container" if parent_path == "" else f"tree_{parent_path}"
        if dpg.does_item_exist(parent_tag):
          node = self.data_tree
          for part in parts:
            node = node.children[part]
          self.ui_render_queue.append((node, parent_tag, search_term, True))

  def update_frame(self):
    items_processed = 0
    while self.ui_render_queue and items_processed < self.MAX_ITEMS_PER_FRAME: # process up to MAX_ITEMS_PER_FRAME to maintain performance
      node, parent_tag, search_term, is_leaf = self.ui_render_queue.popleft()
      if is_leaf:
        self._create_leaf_ui(node, parent_tag)
      else:
        self._create_node_ui(node, parent_tag, search_term)
      items_processed += 1

  def search_data(self, search_term: str):
    self.current_search = search_term
    self._all_paths_cache = self.data_manager.get_all_paths()
    self._previous_paths_set = set(self._all_paths_cache)  # Reset tracking after search
    self._populate_tree()

  def _clear_ui(self):
    dpg.delete_item("data_tree_container", children_only=True)
    self.ui_render_queue.clear()
    self.visible_expanded_nodes.clear()
    self.created_leaf_paths.clear()

  def _calculate_child_counts(self, node: DataTreeNode):
    if node.is_leaf:
      node.child_count = 0
    else:
      node.child_count = len(node.children)
      for child in node.children.values():
        self._calculate_child_counts(child)

  def _create_node_ui(self, node: DataTreeNode, parent_tag: str, search_term: str):
    if node.is_leaf:
      self._create_leaf_ui(node, parent_tag)
    else:
      self._create_tree_node_ui(node, parent_tag, search_term)

  def _create_tree_node_ui(self, node: DataTreeNode, parent_tag: str, search_term: str):
    if not dpg.does_item_exist(parent_tag):
      return
    node_tag = f"tree_{node.full_path}"
    node.ui_tag = node_tag

    label = f"{node.name} ({node.child_count} fields)"
    should_open = (bool(search_term) and len(search_term) > 1 and any(search_term in path for path in self._get_descendant_paths(node)))

    with dpg.tree_node(label=label, parent=parent_tag, tag=node_tag, default_open=should_open, open_on_arrow=True, open_on_double_click=True) as tree_node:
      with dpg.item_handler_registry() as handler:
        dpg.add_item_toggled_open_handler(callback=lambda s, d, u: self._on_node_expanded(node, search_term))
      dpg.bind_item_handler_registry(tree_node, handler)

    node.ui_created = True

    if should_open:
      self.visible_expanded_nodes.add(node.full_path)
      self._queue_children(node, node_tag, search_term)

  def _create_leaf_ui(self, node: DataTreeNode, parent_tag: str):
    if not dpg.does_item_exist(parent_tag):
      return
    half_split_size = dpg.get_item_rect_size("data_pool_window")[0] // 2
    with dpg.group(parent=parent_tag, horizontal=True, xoffset=half_split_size, tag=f"group_{node.full_path}") as draggable_group:
      dpg.add_text(node.name)
      dpg.add_text("N/A", tag=f"value_{node.full_path}")

      if node.is_plottable_cached is None:
        node.is_plottable_cached = self.data_manager.is_plottable(node.full_path)

      if node.is_plottable_cached:
        with dpg.drag_payload(parent=draggable_group, drag_data=node.full_path, payload_type="TIMESERIES_PAYLOAD"):
          dpg.add_text(f"Plot: {node.full_path}")

    node.ui_created = True
    node.ui_tag = f"value_{node.full_path}"
    self.created_leaf_paths.add(node.full_path)

  def _queue_children(self, node: DataTreeNode, parent_tag: str, search_term: str):
    for child in sorted(node.children.values(), key=self._natural_sort_key):
      self.ui_render_queue.append((child, parent_tag, search_term, child.is_leaf))

  def _on_node_expanded(self, node: DataTreeNode, search_term: str):
    node_tag = f"tree_{node.full_path}"
    if not dpg.does_item_exist(node_tag):
      return

    is_expanded = dpg.get_value(node_tag)

    if is_expanded:
      if node.full_path not in self.visible_expanded_nodes:
        self.visible_expanded_nodes.add(node.full_path)
        self._queue_children(node, node_tag, search_term)
    else:
      self.visible_expanded_nodes.discard(node.full_path)
      self._remove_children_from_queue(node.full_path)

  def _remove_children_from_queue(self, collapsed_node_path: str):
    new_queue: deque[tuple] = deque()
    for node, parent_tag, search_term, is_leaf in self.ui_render_queue:
      # Keep items that are not children of the collapsed node
      if not node.full_path.startswith(collapsed_node_path + "/"):
        new_queue.append((node, parent_tag, search_term, is_leaf))
    self.ui_render_queue = new_queue

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
