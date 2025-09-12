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
    self.filtered_children: dict[str, DataTreeNode] = {}
    self.created_children: dict[str, DataTreeNode] = {}
    self.is_leaf = False
    self.is_plottable: bool | None = None
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
    self._build_queue: dict[str, tuple[DataTreeNode, DataTreeNode, str | int]] = {} # full_path -> (node, parent, before_tag)
    self._current_created_paths: set[str] = set()
    self._current_filtered_paths: set[str] = set()
    self._path_to_node: dict[str, DataTreeNode] = {}  # full_path -> node
    self._expanded_tags: set[str] = set()
    self._item_handlers: dict[str, str] = {}  # ui_tag -> handler_tag
    self._char_width = None
    self._queued_search = None
    self._new_data = False
    self._ui_lock = threading.RLock()
    self._handlers_to_delete = []
    self.data_manager.add_observer(self._on_data_loaded)

  def create_ui(self, parent_tag: str):
    with dpg.child_window(parent=parent_tag, border=False, width=-1, height=-1):
      dpg.add_text("Timeseries List")
      dpg.add_separator()
      dpg.add_input_text(tag="search_input", width=-1, hint="Search fields...", callback=self.search_data)
      dpg.add_separator()
      with dpg.child_window(border=False, width=-1, height=-1):
        with dpg.group(tag="data_tree_container"):
          pass

  def _on_data_loaded(self, data: dict):
    with self._ui_lock:
      if data.get('segment_added') or data.get('reset'):
        self._new_data = True

  def update_frame(self, font):
    if self._handlers_to_delete:  # we need to do everything in main thread, frame callbacks are flaky
      dpg.render_dearpygui_frame()  # wait a frame to ensure queued callbacks are done
      with self._ui_lock:
        for handler in self._handlers_to_delete:
          dpg.delete_item(handler)
        self._handlers_to_delete.clear()

    with self._ui_lock:
      if self._char_width is None:
        if size := dpg.get_text_size(" ", font=font):
          self._char_width = size[0]

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
        child_node, parent, before_tag = self._build_queue.pop(next(iter(self._build_queue)))
        parent_tag = "data_tree_container" if parent.name == "root" else parent.ui_tag
        if not child_node.ui_created:
          if child_node.is_leaf:
            self._create_leaf_ui(child_node, parent_tag, before_tag)
          else:
            self._create_tree_node_ui(child_node, parent_tag, before_tag)
        parent.created_children[child_node.name] = parent.children[child_node.name]
        self._current_created_paths.add(child_node.full_path)
        nodes_processed += 1

  def _process_path_change(self):
    self._build_queue.clear()
    search_term = self.current_search.strip().lower()
    all_paths = set(self.data_manager.get_all_paths())
    new_filtered_leafs = {path for path in all_paths if self._should_show_path(path, search_term)}
    new_filtered_paths = set(new_filtered_leafs)
    for path in new_filtered_leafs:
      parts = path.split('/')
      for i in range(1, len(parts)):
        prefix = '/'.join(parts[:i])
        new_filtered_paths.add(prefix)
    created_paths_to_remove = self._current_created_paths - new_filtered_paths
    filtered_paths_to_remove = self._current_filtered_paths - new_filtered_leafs

    if created_paths_to_remove or filtered_paths_to_remove:
      self._remove_paths_from_tree(created_paths_to_remove, filtered_paths_to_remove)
      self._apply_expansion_to_tree(self.data_tree, search_term)

    paths_to_add = new_filtered_leafs - self._current_created_paths
    if paths_to_add:
      self._add_paths_to_tree(paths_to_add)
      self._apply_expansion_to_tree(self.data_tree, search_term)
    self._current_filtered_paths = new_filtered_paths

  def _remove_paths_from_tree(self, created_paths_to_remove, filtered_paths_to_remove):
    for path in sorted(created_paths_to_remove, reverse=True):
      current_node = self._path_to_node[path]

      if len(current_node.created_children) == 0:
        self._current_created_paths.remove(current_node.full_path)
        if item_handler_tag := self._item_handlers.get(current_node.ui_tag):
          dpg.configure_item(item_handler_tag, show=False)
          self._handlers_to_delete.append(item_handler_tag)
          del self._item_handlers[current_node.ui_tag]
        dpg.delete_item(current_node.ui_tag)
        current_node.ui_created = False
        current_node.ui_tag = None
        current_node.children_ui_created = False
        del current_node.parent.created_children[current_node.name]
        del current_node.parent.filtered_children[current_node.name]

    for path in filtered_paths_to_remove:
      parts = path.split('/')
      current_node = self._path_to_node[path]

      part_array_index = -1
      while len(current_node.filtered_children) == 0 and part_array_index >= -len(parts):
        current_node = current_node.parent
        if parts[part_array_index] in current_node.filtered_children:
          del current_node.filtered_children[parts[part_array_index]]
        part_array_index -= 1

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
          self._path_to_node[current_path_prefix] = current_node.children[part]
        current_node.filtered_children[part] = current_node.children[part]
        current_node = current_node.children[part]

      if not current_node.is_leaf:
        current_node.is_leaf = True

    for p_node in parent_nodes_to_recheck:
      p_node.children_ui_created = False
      self._request_children_build(p_node)

  def _get_node_label_and_expand(self, node: DataTreeNode, search_term: str):
    label = f"{node.name} ({len(node.filtered_children)} fields)"
    expand = len(search_term) > 0 and any(search_term in path for path in self._get_descendant_paths(node))
    if expand and node.parent and len(node.parent.filtered_children) > 100 and len(node.filtered_children) > 2:
      label += " (+)"  # symbol for large lists which aren't fully expanded for performance (only affects procLog rn)
      expand = False
    return label, expand

  def _apply_expansion_to_tree(self, node: DataTreeNode, search_term: str):
    if node.ui_created and not node.is_leaf and node.ui_tag and dpg.does_item_exist(node.ui_tag):
      label, expand = self._get_node_label_and_expand(node, search_term)
      if expand:
        self._expanded_tags.add(node.ui_tag)
        dpg.set_value(node.ui_tag, expand)
      elif node.ui_tag in self._expanded_tags:  # not expanded and was expanded
        self._expanded_tags.remove(node.ui_tag)
        dpg.set_value(node.ui_tag, expand)
        dpg.delete_item(node.ui_tag, children_only=True)  # delete children (not visible since collapsed)
        self._reset_ui_state_recursive(node)
        node.children_ui_created = False
      dpg.set_item_label(node.ui_tag, label)
    for child in node.created_children.values():
      self._apply_expansion_to_tree(child, search_term)

  def _reset_ui_state_recursive(self, node: DataTreeNode):
    for child in node.created_children.values():
      if child.ui_tag is not None:
        if item_handler_tag := self._item_handlers.get(child.ui_tag):
          self._handlers_to_delete.append(item_handler_tag)
          dpg.configure_item(item_handler_tag, show=False)
          del self._item_handlers[child.ui_tag]
        self._reset_ui_state_recursive(child)
        child.ui_created = False
        child.ui_tag = None
        child.children_ui_created = False
        self._current_created_paths.remove(child.full_path)
    node.created_children.clear()

  def search_data(self):
    with self._ui_lock:
      self._queued_search = dpg.get_value("search_input")

  def _create_tree_node_ui(self, node: DataTreeNode, parent_tag: str, before: str | int):
    node.ui_tag = f"tree_{node.full_path}"
    search_term = self.current_search.strip().lower()
    label, expand = self._get_node_label_and_expand(node, search_term)
    if expand:
      self._expanded_tags.add(node.ui_tag)
    elif node.ui_tag in self._expanded_tags:
      self._expanded_tags.remove(node.ui_tag)

    with dpg.tree_node(
      label=label, parent=parent_tag, tag=node.ui_tag, default_open=expand, open_on_arrow=True, open_on_double_click=True, before=before, delay_search=True
    ):
      with dpg.item_handler_registry() as handler_tag:
        dpg.add_item_toggled_open_handler(callback=lambda s, a, u: self._request_children_build(node))
        dpg.add_item_visible_handler(callback=lambda s, a, u: self._request_children_build(node))
      dpg.bind_item_handler_registry(node.ui_tag, handler_tag)
      self._item_handlers[node.ui_tag] = handler_tag
    node.ui_created = True

  def _create_leaf_ui(self, node: DataTreeNode, parent_tag: str, before: str | int):
    node.ui_tag = f"leaf_{node.full_path}"
    with dpg.group(parent=parent_tag, tag=node.ui_tag, before=before, delay_search=True):
      with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp, delay_search=True):
        dpg.add_table_column(init_width_or_weight=0.5)
        dpg.add_table_column(init_width_or_weight=0.5)
        with dpg.table_row():
          dpg.add_text(node.name)
          dpg.add_text("N/A", tag=f"value_{node.full_path}")

    if node.is_plottable is None:
      node.is_plottable = self.data_manager.is_plottable(node.full_path)
    if node.is_plottable:
      with dpg.drag_payload(parent=node.ui_tag, drag_data=node.full_path, payload_type="TIMESERIES_PAYLOAD"):
        dpg.add_text(f"Plot: {node.full_path}")

    with dpg.item_handler_registry() as handler_tag:
      dpg.add_item_visible_handler(callback=self._on_item_visible, user_data=node.full_path)
    dpg.bind_item_handler_registry(node.ui_tag, handler_tag)
    self._item_handlers[node.ui_tag] = handler_tag
    node.ui_created = True

  def _on_item_visible(self, sender, app_data, user_data):
    with self._ui_lock:
      path = user_data
      value_tag = f"value_{path}"
      if not dpg.does_item_exist(value_tag):
        return
      value_column_width = dpg.get_item_rect_size(f"leaf_{path}")[0] // 2
      value = self.data_manager.get_value_at(path, self.playback_manager.current_time_s)
      if value is not None:
        formatted_value = self.format_and_truncate(value, value_column_width, self._char_width)
        dpg.set_value(value_tag, formatted_value)
      else:
        dpg.set_value(value_tag, "N/A")

  def _request_children_build(self, node: DataTreeNode):
    with self._ui_lock:
      if not node.children_ui_created and (node.name == "root" or (node.ui_tag is not None and dpg.get_value(node.ui_tag))):  # check root or node expanded
        sorted_children = sorted(node.filtered_children.values(), key=self._natural_sort_key)
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
          if not child_node.ui_created:
            before_tag = next_existing[i]
            self._build_queue[child_node.full_path] = (child_node, node, before_tag)
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
    for child_name, child_node in node.filtered_children.items():
      child_name_lower = child_name.lower()
      if child_node.is_leaf:
        yield child_name_lower
      else:
        for path in self._get_descendant_paths(child_node):
          yield f"{child_name_lower}/{path}"

  @staticmethod
  def format_and_truncate(value, available_width: float, char_width: float) -> str:
    s = f"{value:.5f}" if np.issubdtype(type(value), np.floating) else str(value)
    max_chars = int(available_width / char_width)
    if len(s) > max_chars:
      return s[: max(0, max_chars - 3)] + "..."
    return s
