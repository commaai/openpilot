import dearpygui.dearpygui as dpg
from openpilot.tools.jotpluggler.data import DataManager
from openpilot.tools.jotpluggler.views import TimeSeriesPanel

GRIP_SIZE = 4
MIN_PANE_SIZE = 60


class PlotLayoutManager:
  def __init__(self, data_manager: DataManager, playback_manager, worker_manager, scale: float = 1.0):
    self.data_manager = data_manager
    self.playback_manager = playback_manager
    self.worker_manager = worker_manager
    self.scale = scale
    self.container_tag = "plot_layout_container"
    self.active_panels: list = []

    self.grip_size = int(GRIP_SIZE * self.scale)
    self.min_pane_size = int(MIN_PANE_SIZE * self.scale)

    initial_panel = TimeSeriesPanel(data_manager, playback_manager, worker_manager)
    self.layout: dict = {"type": "panel", "panel": initial_panel}

  def create_ui(self, parent_tag: str):
    if dpg.does_item_exist(self.container_tag):
      dpg.delete_item(self.container_tag)

    with dpg.child_window(tag=self.container_tag, parent=parent_tag, border=False, width=-1, height=-1, no_scrollbar=True, no_scroll_with_mouse=True):
      container_width, container_height = dpg.get_item_rect_size(self.container_tag)
      self._create_ui_recursive(self.layout, self.container_tag, [], container_width, container_height)

  def _create_ui_recursive(self, layout: dict, parent_tag: str, path: list[int], width: int, height: int):
    if layout["type"] == "panel":
      self._create_panel_ui(layout, parent_tag, path, width, height)
    else:
      self._create_split_ui(layout, parent_tag, path, width, height)

  def _create_panel_ui(self, layout: dict, parent_tag: str, path: list[int], width: int, height:int):
    panel_tag = self._path_to_tag(path, "panel")
    panel = layout["panel"]
    self.active_panels.append(panel)
    text_size = int(13 * self.scale)
    bar_height = (text_size+24) if width < int(279 * self.scale + 80) else (text_size+8) # adjust height to allow for scrollbar

    with dpg.child_window(parent=parent_tag, border=True, width=-1, height=-1, no_scrollbar=True):
      with dpg.group(horizontal=True):
        with dpg.child_window(tag=panel_tag, width=-(text_size + 16), height=bar_height, horizontal_scrollbar=True, no_scroll_with_mouse=True, border=False):
          with dpg.group(horizontal=True):
            dpg.add_input_text(default_value=panel.title, width=int(100 * self.scale), callback=lambda s, v: setattr(panel, "title", v))
            dpg.add_combo(items=["Time Series"], default_value="Time Series", width=int(100 * self.scale))
            dpg.add_button(label="Clear", callback=lambda: self.clear_panel(panel), width=int(40 * self.scale))
            dpg.add_image_button(texture_tag="split_h_texture", callback=lambda: self.split_panel(path, 0), width=text_size, height=text_size)
            dpg.add_image_button(texture_tag="split_v_texture", callback=lambda: self.split_panel(path, 1), width=text_size, height=text_size)
        dpg.add_image_button(texture_tag="x_texture", callback=lambda: self.delete_panel(path), width=text_size, height=text_size)

      dpg.add_separator()

      content_tag = self._path_to_tag(path, "content")
      with dpg.child_window(tag=content_tag, border=False, height=-1, width=-1, no_scrollbar=True):
        panel.create_ui(content_tag)

  def _create_split_ui(self, layout: dict, parent_tag: str, path: list[int], width: int, height: int):
    split_tag = self._path_to_tag(path, "split")
    orientation, _, pane_sizes = self._get_split_geometry(layout, (width, height))

    with dpg.group(tag=split_tag, parent=parent_tag, horizontal=orientation == 0):
      for i, child_layout in enumerate(layout["children"]):
        child_path = path + [i]
        container_tag = self._path_to_tag(child_path, "container")
        pane_width, pane_height = [(pane_sizes[i], -1), (-1, pane_sizes[i])][orientation] # fill 2nd dim up to the border
        with dpg.child_window(tag=container_tag, width=pane_width, height=pane_height, border=False, no_scrollbar=True):
          child_width, child_height = [(pane_sizes[i], height), (width, pane_sizes[i])][orientation]
          self._create_ui_recursive(child_layout, container_tag, child_path, child_width, child_height)
        if i < len(layout["children"]) - 1:
          self._create_grip(split_tag, path, i, orientation)

  def clear_panel(self, panel):
    panel.clear()

  def delete_panel(self, panel_path: list[int]):
    if not panel_path:  # Root deletion
      old_panel = self.layout["panel"]
      old_panel.destroy_ui()
      self.active_panels.remove(old_panel)
      new_panel = TimeSeriesPanel(self.data_manager, self.playback_manager, self.worker_manager)
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
    new_panel = TimeSeriesPanel(self.data_manager, self.playback_manager, self.worker_manager)
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
      for i in range(len(layout["children"]) - 1):
        handler_tag = f"{self._path_to_tag(path, f'grip_{i}')}_handler"
        if dpg.does_item_exist(handler_tag):
          dpg.delete_item(handler_tag)

      for i, child in enumerate(layout["children"]):
        self._cleanup_ui_recursive(child, path + [i])

  def update_all_panels(self):
    for panel in self.active_panels:
      panel.update()

  def on_viewport_resize(self):
    self._resize_splits_recursive(self.layout, [])

  def _resize_splits_recursive(self, layout: dict, path: list[int], width: int | None = None, height: int | None = None):
    if layout["type"] == "split":
      split_tag = self._path_to_tag(path, "split")
      if dpg.does_item_exist(split_tag):
        available_sizes = (width, height) if width and height else dpg.get_item_rect_size(dpg.get_item_parent(split_tag))
        orientation, _, pane_sizes = self._get_split_geometry(layout, available_sizes)
        size_properties = ("width", "height")

        for i, child_layout in enumerate(layout["children"]):
          child_path = path + [i]
          container_tag = self._path_to_tag(child_path, "container")
          if dpg.does_item_exist(container_tag):
            dpg.configure_item(container_tag, **{size_properties[orientation]: pane_sizes[i]})
            child_width, child_height = [(pane_sizes[i], available_sizes[1]), (available_sizes[0], pane_sizes[i])][orientation]
            self._resize_splits_recursive(child_layout, child_path, child_width, child_height)
    else: # leaf node/panel - adjust bar height to allow for scrollbar
      panel_tag = self._path_to_tag(path, "panel")
      if width is not None and width < int(279 * self.scale + 80):  # scaled widths of the elements in top bar + fixed 8 padding on left and right of each item
        dpg.configure_item(panel_tag, height=(int(13*self.scale) + 24))
      else:
        dpg.configure_item(panel_tag, height=(int(13*self.scale) + 8))

  def _get_split_geometry(self, layout: dict, available_size: tuple[int, int]) -> tuple[int, int, list[int]]:
    orientation = layout["orientation"]
    num_grips = len(layout["children"]) - 1
    usable_size = max(self.min_pane_size, available_size[orientation] - (num_grips * (self.grip_size + 8 * (2-orientation)))) # approximate, scaling is weird
    pane_sizes = [max(self.min_pane_size, int(usable_size * prop)) for prop in layout["proportions"]]
    return orientation, usable_size, pane_sizes

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
    width, height = [(self.grip_size, -1), (-1, self.grip_size)][orientation]

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
    split_tag = self._path_to_tag(path, "split")
    if not dpg.does_item_exist(split_tag):
      return

    _, usable_size, _ = self._get_split_geometry(layout, dpg.get_item_rect_size(split_tag))
    current_coord = dpg.get_mouse_pos(local=False)[orientation]
    delta = current_coord - drag_data["start_mouse"]
    delta_prop = delta / usable_size

    left_idx = grip_index
    right_idx = left_idx + 1
    initial = drag_data["initial_proportions"]
    min_prop = self.min_pane_size / usable_size

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
