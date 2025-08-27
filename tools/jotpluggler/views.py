import os
import re
import uuid
import threading
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

  @abstractmethod
  def preserve_data(self):
    pass


class TimeSeriesPanel(ViewPanel):
  def __init__(self, data_manager: DataManager, playback_manager, panel_id: str | None = None):
    super().__init__(panel_id)
    self.data_manager = data_manager
    self.playback_manager = playback_manager
    self.title = "Time Series Plot"
    self.plotted_series: set[str] = set()
    self.plot_tag: str | None = None
    self.x_axis_tag: str | None = None
    self.y_axis_tag: str | None = None
    self.timeline_indicator_tag: str | None = None
    self._ui_created = False
    self._preserved_series_data: list[tuple[str, tuple]] = []  # TODO: the way we do this right now doesn't make much sense
    self._series_legend_tags: dict[str, str] = {}  # Maps series_path to legend tag
    self.data_manager.add_observer(self.on_data_loaded)

  def preserve_data(self):
    self._preserved_series_data = []
    if self.plotted_series and self._ui_created:
      for series_path in self.plotted_series:
        time_value_data = self.data_manager.get_timeseries(series_path)
        if time_value_data:
          self._preserved_series_data.append((series_path, time_value_data))

  def create_ui(self, parent_tag: str):
    self.plot_tag = f"plot_{self.panel_id}"
    self.x_axis_tag = f"{self.plot_tag}_x_axis"
    self.y_axis_tag = f"{self.plot_tag}_y_axis"
    self.timeline_indicator_tag = f"{self.plot_tag}_timeline"

    with dpg.plot(height=-1, width=-1, tag=self.plot_tag, parent=parent_tag, drop_callback=self._on_series_drop, payload_type="TIMESERIES_PAYLOAD"):
      dpg.add_plot_legend()
      dpg.add_plot_axis(dpg.mvXAxis, label="", tag=self.x_axis_tag)
      dpg.add_plot_axis(dpg.mvYAxis, label="", tag=self.y_axis_tag)

      timeline_series_tag = dpg.add_inf_line_series(x=[0], label="Timeline", parent=self.y_axis_tag, tag=self.timeline_indicator_tag)
      dpg.bind_item_theme(timeline_series_tag, "global_timeline_theme")

    # Restore series from preserved data
    if self._preserved_series_data:
      self.plotted_series.clear()
      for series_path, (rel_time_array, value_array) in self._preserved_series_data:
        self._add_series_with_data(series_path, rel_time_array, value_array)
      self._preserved_series_data = []

    self._ui_created = True

  def update_timeline_indicator(self, current_time_s: float):
    if not self._ui_created or not dpg.does_item_exist(self.timeline_indicator_tag):
      return

    dpg.set_value(self.timeline_indicator_tag, [[current_time_s], [0]])  # vertical line position

    if self.plotted_series:  # update legend labels with current values
      for series_path in self.plotted_series:
        value = self.data_manager.get_value_at(series_path, current_time_s)

        if value is not None:
          if isinstance(value, (int, float)):
            if isinstance(value, float):
              formatted_value = f"{value:.4f}" if abs(value) < 1000 else f"{value:.3e}"
            else:
              formatted_value = str(value)
          else:
            formatted_value = str(value)

          series_tag = f"series_{self.panel_id}_{series_path.replace('/', '_')}"
          legend_label = f"{series_path}: {formatted_value}"

          if dpg.does_item_exist(series_tag):
            dpg.configure_item(series_tag, label=legend_label)

  def _add_series_with_data(self, series_path: str, rel_time_array, value_array) -> bool:
    if series_path in self.plotted_series:
      return False

    series_tag = f"series_{self.panel_id}_{series_path.replace('/', '_')}"
    line_series_tag = dpg.add_line_series(x=rel_time_array.tolist(), y=value_array.tolist(), label=series_path, parent=self.y_axis_tag, tag=series_tag)

    dpg.bind_item_theme(line_series_tag, "global_line_theme")

    self.plotted_series.add(series_path)
    dpg.fit_axis_data(self.x_axis_tag)
    dpg.fit_axis_data(self.y_axis_tag)
    return True

  def destroy_ui(self):
    if self.plot_tag and dpg.does_item_exist(self.plot_tag):
      dpg.delete_item(self.plot_tag)

    self._series_legend_tags.clear()
    self._ui_created = False

  def get_panel_type(self) -> str:
    return "timeseries"

  def add_series(self, series_path: str) -> bool:
    if series_path in self.plotted_series:
      return False

    time_value_data = self.data_manager.get_timeseries(series_path)
    if time_value_data is None:
      return False

    rel_time_array, value_array = time_value_data
    return self._add_series_with_data(series_path, rel_time_array, value_array)

  def clear_all_series(self):
    for series_path in self.plotted_series.copy():
      self.remove_series(series_path)

  def remove_series(self, series_path: str):
    if series_path in self.plotted_series:
      series_tag = f"series_{self.panel_id}_{series_path.replace('/', '_')}"
      if dpg.does_item_exist(series_tag):
        dpg.delete_item(series_tag)
      self.plotted_series.remove(series_path)
      if series_path in self._series_legend_tags:
        del self._series_legend_tags[series_path]

  def on_data_loaded(self, data: dict):
    for series_path in self.plotted_series.copy():
      self._update_series_data(series_path)

  def _update_series_data(self, series_path: str) -> bool:
    time_value_data = self.data_manager.get_timeseries(series_path)
    if time_value_data is None:
      return False

    rel_time_array, value_array = time_value_data
    series_tag = f"series_{self.panel_id}_{series_path.replace('/', '_')}"

    if dpg.does_item_exist(series_tag):
      dpg.set_value(series_tag, [rel_time_array.tolist(), value_array.tolist()])
      dpg.fit_axis_data(self.x_axis_tag)
      dpg.fit_axis_data(self.y_axis_tag)
      return True
    else:
      self.plotted_series.discard(series_path)
      return False

  def _on_series_drop(self, sender, app_data, user_data):
    series_path = app_data
    self.add_series(series_path)


class DataTreeNode:
  def __init__(self, name: str, full_path: str = ""):
    self.name = name
    self.full_path = full_path
    self.children: dict[str, DataTreeNode] = {}
    self.is_leaf = False

class DataTreeView:
  def __init__(self, data_manager: DataManager, ui_lock: threading.Lock):
    self.data_manager = data_manager
    self.ui_lock = ui_lock
    self.current_search = ""
    self.data_tree = DataTreeNode(name="root")
    self.active_leaf_nodes: list[DataTreeNode] = []
    self.data_manager.add_observer(self.on_data_loaded)

  def on_data_loaded(self, data: dict):
    with self.ui_lock:
      self.populate_data_tree()

  def populate_data_tree(self):
    if not dpg.does_item_exist("data_tree_container"):
      return

    dpg.delete_item("data_tree_container", children_only=True)
    search_term = self.current_search.strip().lower()

    self.data_tree = DataTreeNode(name="root")
    all_paths = self.data_manager.get_all_paths()

    for path in sorted(all_paths):
      if not self._should_display_path(path, search_term):
        continue

      parts = path.split('/')
      current_node = self.data_tree
      current_path_prefix = ""

      for part in parts:
        current_path_prefix = f"{current_path_prefix}/{part}" if current_path_prefix else part
        if part not in current_node.children:
          current_node.children[part] = DataTreeNode(name=part, full_path=current_path_prefix)
        current_node = current_node.children[part]

      current_node.is_leaf = True

    self._create_ui_from_tree_recursive(self.data_tree, "data_tree_container", search_term)
    self.update_active_nodes_list()

  def _should_display_path(self, path: str, search_term: str) -> bool:
    if 'DEPRECATED' in path and not os.environ.get('SHOW_DEPRECATED'):
      return False
    return not search_term or search_term in path.lower()

  def _natural_sort_key(self, node: DataTreeNode):
    node_type_key = node.is_leaf
    parts = [int(p) if p.isdigit() else p.lower() for p in re.split(r'(\d+)', node.name) if p]
    return (node_type_key, parts)

  def _create_ui_from_tree_recursive(self, node: DataTreeNode, parent_tag: str, search_term: str):
    sorted_children = sorted(node.children.values(), key=self._natural_sort_key)

    for child in sorted_children:
      if child.is_leaf:
        is_plottable = self.data_manager.is_plottable(child.full_path)

        # Create draggable item
        with dpg.group(parent=parent_tag) as draggable_group:
          with dpg.table(header_row=False, borders_innerV=False, borders_outerH=False, borders_outerV=False, policy=dpg.mvTable_SizingStretchProp):
            dpg.add_table_column(init_width_or_weight=0.5)
            dpg.add_table_column(init_width_or_weight=0.5)
            with dpg.table_row():
              dpg.add_text(child.name)
              dpg.add_text("N/A", tag=f"value_{child.full_path}")

        # Add drag payload if plottable
        if is_plottable:
          with dpg.drag_payload(parent=draggable_group, drag_data=child.full_path, payload_type="TIMESERIES_PAYLOAD"):
            dpg.add_text(f"Plot: {child.full_path}")

      else:
        node_tag = f"tree_{child.full_path}"
        label = child.name

        should_open = bool(search_term) and len(search_term) > 1 and any(search_term in path for path in self._get_all_descendant_paths(child))

        with dpg.tree_node(label=label, parent=parent_tag, tag=node_tag, default_open=should_open):
          dpg.bind_item_handler_registry(node_tag, "tree_node_handler")
          self._create_ui_from_tree_recursive(child, node_tag, search_term)

  def _get_all_descendant_paths(self, node: DataTreeNode):
    for child_name, child_node in node.children.items():
      child_name_lower = child_name.lower()
      if child_node.is_leaf:
        yield child_name_lower
      else:
        for path in self._get_all_descendant_paths(child_node):
          yield f"{child_name_lower}/{path}"

  def search_data(self, search_term: str):
    self.current_search = search_term
    self.populate_data_tree()

  def update_active_nodes_list(self, sender=None, app_data=None, user_data=None):
    self.active_leaf_nodes = self.get_active_leaf_nodes(self.data_tree)

  def get_active_leaf_nodes(self, node: DataTreeNode):
    active_leaves = []
    for child in node.children.values():
      if child.is_leaf:
        active_leaves.append(child)
      else:
        node_tag = f"tree_{child.full_path}"
        if dpg.does_item_exist(node_tag) and dpg.get_value(node_tag):
          active_leaves.extend(self.get_active_leaf_nodes(child))
    return active_leaves
