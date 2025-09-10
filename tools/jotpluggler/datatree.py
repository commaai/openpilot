import os
import re
import threading
import numpy as np
import dearpygui.dearpygui as dpg


class DataTreeNode:
  def __init__(self, name: str, full_path: str = "", parent=None):
    self.name = name
    self.full_path = full_path
    self.parent = parent
    self.children: dict[str, DataTreeNode] = {}
    self.active_children: dict[str, DataTreeNode] = {}
    self.is_leaf = False
    self.is_plottable: bool | None = None
    self.ui_queued = False
    self.ui_created = False
    self.children_ui_created = False
    self.ui_tag: str | None = None


class DataTree:
  MAX_NODES_PER_FRAME = 50

  def __init__(self, data_manager, playback_manager):
    self.data_manager = data_manager
    self.playback_manager = playback_manager
    self.current_search = ""
    self.data_tree = DataTreeNode(name="root")
    self._build_queue: dict[str, tuple[DataTreeNode, str | None, str | int]] = {}
    self._current_filtered_paths: set[str] = set()
    self._item_handlers: dict[str, str] = {}  # ui_tag -> handler_tag
    self._avg_char_width = None
    self._queued_search = None
    self._new_data = False
    self._ui_lock = threading.RLock()
    self.data_manager.add_observer(self._on_data_loaded)

  def create_ui(self, parent_tag: str):
    with dpg.child_window(parent=parent_tag, border=False, width=-1, height=-1):
      dpg.add_text("Available Data")
      dpg.add_separator()
      dpg.add_input_text(tag="search_input", width=-1, hint="Search fields...", callback=self.search_data)
      dpg.add_separator()
      with dpg.group(tag="data_tree_container"):
        pass

  def _on_data_loaded(self, data: dict):
    with self._ui_lock:
      if data.get('segment_added'):
        self._new_data = True
      elif data.get('reset'):
        self._current_filtered_paths = set()
        self._new_data = True

  def update_frame(self, font):
    with self._ui_lock:
      if self._avg_char_width is None and dpg.is_dearpygui_running():
        self._avg_char_width = self.calculate_avg_char_width(font)

      if self._new_data:
        self._process_path_change()
        self._new_data = False
        return

      if self._queued_search is not None:
        self.current_search = self._queued_search
        self._process_path_change()
        self._queued_search = None
        return

      nodes_processed = 0
      while self._build_queue and nodes_processed < self.MAX_NODES_PER_FRAME:
        first_key = next(iter(self._build_queue))
        child_node, parent_tag, before_tag = self._build_queue.pop(first_key)
        if not child_node.ui_created:
          if child_node.is_leaf:
            self._create_leaf_ui(child_node, parent_tag, before_tag)
          else:
            self._create_tree_node_ui(child_node, parent_tag, before_tag)
          child_node.ui_queued = False
        nodes_processed += 1

  def _process_path_change(self):
    search_term = self.current_search.strip().lower()
    all_paths = set(self.data_manager.get_all_paths())
    new_filtered_paths = {path for path in all_paths if self._should_show_path(path, search_term)}
    paths_to_remove = self._current_filtered_paths - new_filtered_paths
    paths_to_add = new_filtered_paths - self._current_filtered_paths

    if paths_to_remove:
      self._remove_paths_from_tree(paths_to_remove)
    if paths_to_add:
      self._add_paths_to_tree(paths_to_add)

    self._apply_expansion_to_tree(self.data_tree, search_term)
    self._current_filtered_paths = new_filtered_paths

  def _remove_paths_from_tree(self, paths):
    item_handlers_to_delete = []
    for path in paths:
      parts = path.split('/')
      current_node = self.data_tree
      for part in parts:
        current_node = current_node.children[part]

      part_array_index = -1
      while len(current_node.active_children) == 0 and part_array_index >= -len(parts):
        if current_node.full_path in self._build_queue:
          self._build_queue.pop(current_node.full_path)
          current_node.ui_queued = False
        if item_handler_tag := self._item_handlers.get(current_node.ui_tag):
          item_handlers_to_delete.append(item_handler_tag)  # queue item handler for deletion
          del self._item_handlers[current_node.ui_tag]
        dpg.delete_item(current_node.ui_tag)  # delete item
        current_node.ui_created = False
        current_node.ui_tag = None
        current_node.children_ui_created = False
        current_node = current_node.parent
        del current_node.active_children[parts[part_array_index]]
        part_array_index -= 1

    dpg.set_frame_callback(dpg.get_frame_count() + 1, callback=self._delete_handlers, user_data=item_handlers_to_delete)

  def _add_paths_to_tree(self, paths):
    parent_nodes_to_recheck = set()
    for path in sorted(paths):
      parts = path.split('/')
      current_node = self.data_tree
      current_path_prefix = ""

      for i, part in enumerate(parts):
        current_path_prefix = f"{current_path_prefix}/{part}" if current_path_prefix else part
        if i < len(parts) - 1:
          parent_nodes_to_recheck.add(current_node)  # for incremental changes from new data
        if part not in current_node.children:
          current_node.children[part] = DataTreeNode(name=part, full_path=current_path_prefix, parent=current_node)
        current_node.active_children[part] = current_node.children[part]
        current_node = current_node.children[part]

      if not current_node.is_leaf:
        current_node.is_leaf = True

    for p_node in parent_nodes_to_recheck:
      p_node.children_ui_created = False
      self._request_children_build(p_node)

  def _get_node_label_and_expand(self, node: DataTreeNode, search_term: str):
    label = f"{node.name} ({len(node.active_children)} fields)"
    expand = len(search_term) > 0 and any(search_term in path for path in self._get_descendant_paths(node))
    if expand and node.parent and len(node.parent.active_children) > 100 and len(node.active_children) > 2:
      label += " (+)" # symbol for large lists which aren't fully expanded for performance (only affects procLog rn)
      expand = False
    return label, expand

  def _apply_expansion_to_tree(self, node: DataTreeNode, search_term: str):
    if node.ui_created and not node.is_leaf and node.ui_tag and dpg.does_item_exist(node.ui_tag):
      label, expand = self._get_node_label_and_expand(node, search_term)
      dpg.set_item_label(node.ui_tag, label)
      dpg.set_value(node.ui_tag, expand)

    for child in node.active_children.values():
      self._apply_expansion_to_tree(child, search_term)

  def search_data(self):
    self._queued_search = dpg.get_value("search_input")

  def _delete_handlers(self, sender, app_data, user_data):
    for handler in user_data:
      dpg.delete_item(handler)

  def _create_tree_node_ui(self, node: DataTreeNode, parent_tag: str, before: str | int):
    tag = f"tree_{node.full_path}"
    node.ui_tag = tag
    search_term = self.current_search.strip().lower()
    label, expand = self._get_node_label_and_expand(node, search_term)

    with dpg.tree_node(
      label=label, parent=parent_tag, tag=tag, default_open=expand, open_on_arrow=True, open_on_double_click=True, before=before, delay_search=True
    ):
      with dpg.item_handler_registry() as handler_tag:
        dpg.add_item_toggled_open_handler(callback=lambda s, a, u: self._request_children_build(node))
        dpg.add_item_visible_handler(callback=lambda s, a, u: self._request_children_build(node))
      dpg.bind_item_handler_registry(tag, handler_tag)
      self._item_handlers[tag] = handler_tag

    node.ui_created = True

  def _create_leaf_ui(self, node: DataTreeNode, parent_tag: str, before: str | int):
    with dpg.group(parent=parent_tag, tag=f"leaf_{node.full_path}", before=before, delay_search=True) as draggable_group:
      with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp, delay_search=True):
        dpg.add_table_column(init_width_or_weight=0.5)
        dpg.add_table_column(init_width_or_weight=0.5)
        with dpg.table_row():
          dpg.add_text(node.name)
          dpg.add_text("N/A", tag=f"value_{node.full_path}")

    if node.is_plottable is None:
      node.is_plottable = self.data_manager.is_plottable(node.full_path)
    if node.is_plottable:
      with dpg.drag_payload(parent=draggable_group, drag_data=node.full_path, payload_type="TIMESERIES_PAYLOAD"):
        dpg.add_text(f"Plot: {node.full_path}")

    with dpg.item_handler_registry() as handler_tag:
      dpg.add_item_visible_handler(callback=self._on_item_visible, user_data=node.full_path)
    dpg.bind_item_handler_registry(draggable_group, handler_tag)
    self._item_handlers[draggable_group] = handler_tag

    node.ui_created = True
    node.ui_tag = f"leaf_{node.full_path}"

  def _on_item_visible(self, sender, app_data, user_data):
    with self._ui_lock:
      path = user_data
      value_tag = f"value_{path}"
      if not dpg.does_item_exist(value_tag):
        return
      value_column_width = dpg.get_item_rect_size("sidebar_window")[0] // 2
      value = self.data_manager.get_value_at(path, self.playback_manager.current_time_s)
      if value is not None:
        formatted_value = self.format_and_truncate(value, value_column_width, self._avg_char_width)
        dpg.set_value(value_tag, formatted_value)
      else:
        dpg.set_value(value_tag, "N/A")

  def _request_children_build(self, node: DataTreeNode):
    with self._ui_lock:
      if not node.children_ui_created and (node.name == "root" or (node.ui_tag is not None and dpg.get_value(node.ui_tag))):  # check root or node expanded
        parent_tag = "data_tree_container" if node.name == "root" else node.ui_tag
        sorted_children = sorted(node.active_children.values(), key=self._natural_sort_key)
        next_existing: list[int | str] = [0] * len(sorted_children)
        current_before_tag: int | str = 0

        for i in range(len(sorted_children) - 1, -1, -1):  # calculate "before_tag" for correct ordering when incrementally building tree
          child = sorted_children[i]
          next_existing[i] = current_before_tag
          if child.ui_created:
            candidate_tag = f"leaf_{child.full_path}" if child.is_leaf else f"tree_{child.full_path}"
            if dpg.does_item_exist(candidate_tag):
              current_before_tag = candidate_tag

        for i, child_node in enumerate(sorted_children):
          if not child_node.ui_created and child_node.full_path not in self._build_queue:
            before_tag = next_existing[i]
            self._build_queue[child_node.full_path] = (child_node, parent_tag, before_tag)
            child_node.ui_queued = True
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
    for child_name, child_node in node.active_children.items():
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
