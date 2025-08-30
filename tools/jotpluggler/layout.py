import dearpygui.dearpygui as dpg
from openpilot.tools.jotpluggler.data import DataManager
from openpilot.tools.jotpluggler.views import TimeSeriesPanel

GRIP_SIZE = 4
MIN_PANE_SIZE = 60


class PlotLayoutManager:
  def __init__(self, data_manager: DataManager, playback_manager, scale: float = 1.0):
    self.data_manager = data_manager
    self.playback_manager = playback_manager
    self.scale = scale
    self.container_tag = "plot_layout_container"
    self.active_panels: list = []

    initial_panel = TimeSeriesPanel(data_manager, playback_manager)
    self.layout = {"type": "panel", "panel": initial_panel}

  def create_ui(self, parent_tag: str):
    if dpg.does_item_exist(self.container_tag):
      dpg.delete_item(self.container_tag)

    with dpg.child_window(tag=self.container_tag, parent=parent_tag, border=False, width=-1, height=-1, no_scrollbar=True):
      container_width, container_height = dpg.get_item_rect_size(self.container_tag)
      self._create_ui_recursive(self.layout, self.container_tag, [], container_width, container_height)

  def _create_ui_recursive(self, layout: dict, parent_tag: str, path: list[int], width: int, height: int):
    if layout["type"] == "panel":
      self._create_panel_ui(layout, parent_tag, path)
    else:
      self._create_split_ui(layout, parent_tag, path, width, height)

  def _create_panel_ui(self, layout: dict, parent_tag: str, path: list[int]):
    panel_tag = self._path_to_tag(path, "panel")
    panel = layout["panel"]
    self.active_panels.append(panel)

    with dpg.child_window(tag=panel_tag, parent=parent_tag, border=True, width=-1, height=-1, no_scrollbar=True):
      with dpg.group(horizontal=True):
        dpg.add_input_text(default_value=panel.title, width=int(100 * self.scale), callback=lambda s, v: setattr(panel, "title", v))
        dpg.add_combo(items=["Time Series"], default_value="Time Series", width=int(100 * self.scale))
        dpg.add_button(label="Clear", callback=lambda: self.clear_panel(panel), width=int(40 * self.scale))
        dpg.add_button(label="Delete", callback=lambda: self.delete_panel(path), width=int(40 * self.scale))
        dpg.add_button(label="Split H", callback=lambda: self.split_panel(path, 0), width=int(40 * self.scale))
        dpg.add_button(label="Split V", callback=lambda: self.split_panel(path, 1), width=int(40 * self.scale))

      dpg.add_separator()

      content_tag = self._path_to_tag(path, "content")
      with dpg.child_window(tag=content_tag, border=False, height=-1, width=-1, no_scrollbar=True):
        panel.create_ui(content_tag)

  def _create_split_ui(self, layout: dict, parent_tag: str, path: list[int], width: int, height: int):
    split_tag = self._path_to_tag(path, "split")
    orientation = layout["orientation"]
    min_pane_size = int(MIN_PANE_SIZE * self.scale)
    grip_size = int(GRIP_SIZE * self.scale)
    num_grips = len(layout["children"]) - 1

    with dpg.group(tag=split_tag, parent=parent_tag, horizontal=orientation == 0):
      for i, (child_layout, proportion) in enumerate(zip(layout["children"], layout["proportions"], strict=True)):
        child_path = path + [i]
        container_tag = self._path_to_tag(child_path, "container")

        size = [width, height]  # pass through since get_item_rect_size is unavailble until rendered
        fill_size = [-1, -1]  # fill up to the border upon resize
        calculated_size = max(min_pane_size, int((size[orientation] - (num_grips * grip_size)) * proportion))
        size[orientation] = fill_size[orientation] = calculated_size

        with dpg.child_window(tag=container_tag, width=fill_size[0], height=fill_size[1], border=False, no_scrollbar=True):
          self._create_ui_recursive(child_layout, container_tag, child_path, size[0], size[1])

        if i < len(layout["children"]) - 1:  # Add grip between panes (except after the last pane)
          self._create_grip(split_tag, path, i, orientation)

  def clear_panel(self, panel):
    panel.clear()

  def delete_panel(self, panel_path: list[int]):
    if not panel_path:  # Root deletion
      old_panel = self.layout["panel"]
      old_panel.destroy_ui()
      self.active_panels.remove(old_panel)
      new_panel = TimeSeriesPanel(self.data_manager, self.playback_manager)
      self.layout = {"type": "panel", "panel": new_panel}
      self._rebuild_ui_at_path([])
      return

    parent, child_index = self._get_parent_and_index(panel_path)
    layout_to_delete = parent["children"][child_index]
    self._cleanup_ui_recursive(layout_to_delete, panel_path)

    parent["children"].pop(child_index)
    parent["proportions"].pop(child_index)

    if len(parent["children"]) == 1:  # remove parent and collapse
      remaining_child = parent["children"][0]
      if len(panel_path) == 1:  # parent is at root level - promote remaining child to root
        self.layout = remaining_child
        self._rebuild_ui_at_path([])
      else:  # replace parent with remaining child in grandparent
        grandparent_path = panel_path[:-2]
        parent_index = panel_path[-2]
        self._replace_layout_at_path(grandparent_path + [parent_index], remaining_child)
        self._rebuild_ui_at_path(grandparent_path + [parent_index])
    else:  # redistribute proportions
      equal_prop = 1.0 / len(parent["children"])
      parent["proportions"] = [equal_prop] * len(parent["children"])
      self._rebuild_ui_at_path(panel_path[:-1])

  def split_panel(self, panel_path: list[int], orientation: int):
    current_layout = self._get_layout_at_path(panel_path)
    existing_panel = current_layout["panel"]
    new_panel = TimeSeriesPanel(self.data_manager, self.playback_manager)
    parent, child_index = self._get_parent_and_index(panel_path)

    if parent is None:  # Root split
      self.layout = {
        "type": "split",
        "orientation": orientation,
        "children": [{"type": "panel", "panel": existing_panel}, {"type": "panel", "panel": new_panel}],
        "proportions": [0.5, 0.5],
      }
      self._rebuild_ui_at_path([])
    elif parent["type"] == "split" and parent["orientation"] == orientation:  # Same orientation - insert into existing split
      parent["children"].insert(child_index + 1, {"type": "panel", "panel": new_panel})
      parent["proportions"] = [1.0 / len(parent["children"])] * len(parent["children"])
      self._rebuild_ui_at_path(panel_path[:-1])
    else:  # Different orientation - create new split level
      new_split = {"type": "split", "orientation": orientation, "children": [current_layout, {"type": "panel", "panel": new_panel}], "proportions": [0.5, 0.5]}
      self._replace_layout_at_path(panel_path, new_split)
      self._rebuild_ui_at_path(panel_path)

  def _rebuild_ui_at_path(self, path: list[int]):
    layout = self._get_layout_at_path(path)
    if path:
      container_tag = self._path_to_tag(path, "container")
    else:  # Root update
      container_tag = self.container_tag

    self._cleanup_ui_recursive(layout, path)
    dpg.delete_item(container_tag, children_only=True)
    width, height = dpg.get_item_rect_size(container_tag)
    self._create_ui_recursive(layout, container_tag, path, width, height)

  def _cleanup_ui_recursive(self, layout: dict, path: list[int]):
    if layout["type"] == "panel":
      panel = layout["panel"]
      panel.destroy_ui()
      if panel in self.active_panels:
        self.active_panels.remove(panel)
    else:
      # Clean up grip handler registries for splits BEFORE recursing
      for i in range(len(layout["children"]) - 1):
        grip_tag = self._path_to_tag(path, f"grip_{i}")
        handler_tag = f"{grip_tag}_handler"
        if dpg.does_item_exist(handler_tag):
          dpg.delete_item(handler_tag)

      # Recursively cleanup children
      for i, child in enumerate(layout["children"]):
        child_path = path + [i]
        self._cleanup_ui_recursive(child, child_path)

  def update_all_panels(self):
    for panel in self.active_panels:
      panel.update()

  def on_viewport_resize(self):
    self._resize_splits_recursive(self.layout, [])

  def _resize_splits_recursive(self, layout: dict, path: list[int]):
    if layout["type"] == "split":
      split_tag = self._path_to_tag(path, "split")
      if dpg.does_item_exist(split_tag):
        parent_tag = dpg.get_item_parent(split_tag)
        grip_size = int(GRIP_SIZE * self.scale)
        min_pane_size = int(MIN_PANE_SIZE * self.scale)
        num_grips = len(layout["children"]) - 1
        orientation = layout["orientation"]
        available_sizes = dpg.get_item_rect_size(parent_tag)
        size_properties = ("width", "height")

        for i, proportion in enumerate(layout["proportions"]):
          child_path = path + [i]
          container_tag = self._path_to_tag(child_path, "container")
          if dpg.does_item_exist(container_tag):
            new_size = max(min_pane_size, int((available_sizes[orientation] - (num_grips * grip_size)) * proportion))
            dpg.configure_item(container_tag, **{size_properties[orientation]: new_size})
          self._resize_splits_recursive(layout["children"][i], child_path)

  def _get_layout_at_path(self, path: list[int]) -> dict:
    current = self.layout
    for index in path:
      current = current["children"][index]
    return current

  def _get_parent_and_index(self, path: list[int]) -> tuple:
    return (None, -1) if not path else (self._get_layout_at_path(path[:-1]), path[-1])

  def _replace_layout_at_path(self, path: list[int], new_layout: dict):
    if not path:
      self.layout = new_layout
    else:
      parent, index = self._get_parent_and_index(path)
      parent["children"][index] = new_layout

  def _path_to_tag(self, path: list[int], prefix: str = "") -> str:
    path_str = "_".join(map(str, path)) if path else "root"
    return f"{prefix}_{path_str}" if prefix else path_str

  def _create_grip(self, parent_tag: str, path: list[int], grip_index: int, orientation: int):
    grip_tag = self._path_to_tag(path, f"grip_{grip_index}")
    grip_size = int(GRIP_SIZE * self.scale)
    width = grip_size if orientation == 0 else -1
    height = grip_size if orientation == 1 else -1

    with dpg.child_window(tag=grip_tag, parent=parent_tag, width=width, height=height, no_scrollbar=True, border=False):
      button_tag = dpg.add_button(label="", width=-1, height=-1)

    with dpg.item_handler_registry(tag=f"{grip_tag}_handler"):
      user_data = (path, grip_index, orientation)
      dpg.add_item_active_handler(callback=self._on_grip_drag, user_data=user_data)
      dpg.add_item_deactivated_handler(callback=self._on_grip_end, user_data=user_data)
    dpg.bind_item_handler_registry(button_tag, f"{grip_tag}_handler")

  def _on_grip_drag(self, sender, app_data, user_data):
    path, grip_index, orientation = user_data
    layout = self._get_layout_at_path(path)

    if "_drag_data" not in layout:
      layout["_drag_data"] = {"initial_proportions": layout["proportions"][:], "start_mouse": dpg.get_mouse_pos(local=False)[orientation]}
      return

    drag_data = layout["_drag_data"]
    current_coord = dpg.get_mouse_pos(local=False)[orientation]
    delta = current_coord - drag_data["start_mouse"]

    split_tag = self._path_to_tag(path, "split")
    if not dpg.does_item_exist(split_tag):
      return
    total_size = dpg.get_item_rect_size(split_tag)[orientation]
    num_grips = len(layout["children"]) - 1
    usable_size = max(100, total_size - (num_grips * int(GRIP_SIZE * self.scale)))

    delta_prop = delta / usable_size

    left_idx = grip_index
    right_idx = left_idx + 1
    initial = drag_data["initial_proportions"]
    min_prop = int(MIN_PANE_SIZE * self.scale) / usable_size

    new_left = max(min_prop, initial[left_idx] + delta_prop)
    new_right = max(min_prop, initial[right_idx] - delta_prop)

    total_available = initial[left_idx] + initial[right_idx]
    if new_left + new_right > total_available:
      if new_left > new_right:
        new_left = total_available - new_right
      else:
        new_right = total_available - new_left

    layout["proportions"] = initial[:]
    layout["proportions"][left_idx] = new_left
    layout["proportions"][right_idx] = new_right

    self._resize_splits_recursive(layout, path)

  def _on_grip_end(self, sender, app_data, user_data):
    path, _, _ = user_data
    self._get_layout_at_path(path).pop("_drag_data", None)

