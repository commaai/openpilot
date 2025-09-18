import dearpygui.dearpygui as dpg
from openpilot.tools.jotpluggler.data import DataManager
from openpilot.tools.jotpluggler.views import TimeSeriesPanel

GRIP_SIZE = 4
MIN_PANE_SIZE = 60

class LayoutManager:
  def __init__(self, data_manager, playback_manager, worker_manager, scale: float = 1.0):
    self.data_manager = data_manager
    self.playback_manager = playback_manager
    self.worker_manager = worker_manager
    self.scale = scale
    self.container_tag = "plot_layout_container"
    self.tab_bar_tag = "tab_bar_container"
    self.tab_content_tag = "tab_content_area"

    self.active_tab = 0
    initial_panel_layout = PanelLayoutManager(data_manager, playback_manager, worker_manager, scale)
    self.tabs: dict = {0: {"name": "Tab 1", "panel_layout": initial_panel_layout}}
    self._next_tab_id = self.active_tab + 1

  def to_dict(self) -> dict:
    return {
      "tabs": {
        str(tab_id): {
          "name": tab_data["name"],
          "panel_layout": tab_data["panel_layout"].to_dict()
        }
        for tab_id, tab_data in self.tabs.items()
      }
    }

  def clear_and_load_from_dict(self, data: dict):
    tab_ids_to_close = list(self.tabs.keys())
    for tab_id in tab_ids_to_close:
      self.close_tab(tab_id, force=True)

    for tab_id_str, tab_data in data["tabs"].items():
      tab_id = int(tab_id_str)
      panel_layout = PanelLayoutManager.load_from_dict(
        tab_data["panel_layout"], self.data_manager, self.playback_manager,
        self.worker_manager, self.scale
      )
      self.tabs[tab_id] = {
        "name": tab_data["name"],
        "panel_layout": panel_layout
      }

    self.active_tab = min(self.tabs.keys()) if self.tabs else 0
    self._next_tab_id = max(self.tabs.keys()) + 1 if self.tabs else 1

  def create_ui(self, parent_tag: str):
    if dpg.does_item_exist(self.container_tag):
      dpg.delete_item(self.container_tag)

    with dpg.child_window(tag=self.container_tag, parent=parent_tag, border=False, width=-1, height=-1, no_scrollbar=True, no_scroll_with_mouse=True):
      self._create_tab_bar()
      self._create_tab_content()
    dpg.bind_item_theme(self.tab_bar_tag, "tab_bar_theme")

  def _create_tab_bar(self):
    text_size = int(13 * self.scale)
    with dpg.child_window(tag=self.tab_bar_tag, parent=self.container_tag, height=(text_size + 8), border=False, horizontal_scrollbar=True):
      with dpg.group(horizontal=True, tag="tab_bar_group"):
        for tab_id, tab_data in self.tabs.items():
          self._create_tab_ui(tab_id, tab_data["name"])
        dpg.add_image_button(texture_tag="plus_texture", callback=self.add_tab, width=text_size, height=text_size, tag="add_tab_button")
    dpg.bind_item_theme("add_tab_button", "inactive_tab_theme")

  def _create_tab_ui(self, tab_id: int, tab_name: str):
    text_size = int(13 * self.scale)
    tab_width = int(140 * self.scale)
    with dpg.child_window(width=tab_width, height=-1, border=False, no_scrollbar=True, tag=f"tab_window_{tab_id}", parent="tab_bar_group"):
      with dpg.group(horizontal=True, tag=f"tab_group_{tab_id}"):
        dpg.add_input_text(
          default_value=tab_name, width=tab_width - text_size - 16, callback=lambda s, v, u: self.rename_tab(u, v), user_data=tab_id, tag=f"tab_input_{tab_id}"
        )
        dpg.add_image_button(
          texture_tag="x_texture", callback=lambda s, a, u: self.close_tab(u), user_data=tab_id, width=text_size, height=text_size, tag=f"tab_close_{tab_id}"
        )
      with dpg.item_handler_registry(tag=f"tab_handler_{tab_id}"):
        dpg.add_item_clicked_handler(callback=lambda s, a, u: self.switch_tab(u), user_data=tab_id)
      dpg.bind_item_handler_registry(f"tab_group_{tab_id}", f"tab_handler_{tab_id}")

    theme_tag = "active_tab_theme" if tab_id == self.active_tab else "inactive_tab_theme"
    dpg.bind_item_theme(f"tab_window_{tab_id}", theme_tag)

  def _create_tab_content(self):
    with dpg.child_window(tag=self.tab_content_tag, parent=self.container_tag, border=False, width=-1, height=-1, no_scrollbar=True, no_scroll_with_mouse=True):
      if self.active_tab in self.tabs:
        active_panel_layout = self.tabs[self.active_tab]["panel_layout"]
        active_panel_layout.create_ui()

  def add_tab(self):
    new_panel_layout = PanelLayoutManager(self.data_manager, self.playback_manager, self.worker_manager, self.scale)
    new_tab = {"name": f"Tab {self._next_tab_id + 1}", "panel_layout": new_panel_layout}
    self.tabs[self._next_tab_id] = new_tab
    self._create_tab_ui(self._next_tab_id, new_tab["name"])
    dpg.move_item("add_tab_button", parent="tab_bar_group")  # move plus button to end
    self.switch_tab(self._next_tab_id)
    self._next_tab_id += 1

  def close_tab(self, tab_id: int, force = False):
    if len(self.tabs) <= 1 and not force:
      return  # don't allow closing the last tab

    tab_to_close = self.tabs[tab_id]
    tab_to_close["panel_layout"].destroy_ui()
    for suffix in ["window", "group", "input", "close", "handler"]:
      tag = f"tab_{suffix}_{tab_id}"
      if dpg.does_item_exist(tag):
        dpg.delete_item(tag)
    del self.tabs[tab_id]

    if self.active_tab == tab_id and self.tabs: # switch to another tab if we closed the active one
      self.active_tab = next(iter(self.tabs.keys()))
      self._switch_tab_content()
      dpg.bind_item_theme(f"tab_window_{self.active_tab}", "active_tab_theme")

  def switch_tab(self, tab_id: int):
    if tab_id == self.active_tab or tab_id not in self.tabs:
      return

    current_panel_layout = self.tabs[self.active_tab]["panel_layout"]
    current_panel_layout.destroy_ui()
    dpg.bind_item_theme(f"tab_window_{self.active_tab}", "inactive_tab_theme")  # deactivate old tab
    self.active_tab = tab_id
    dpg.bind_item_theme(f"tab_window_{tab_id}", "active_tab_theme")  # activate new tab
    self._switch_tab_content()

  def _switch_tab_content(self):
    dpg.delete_item(self.tab_content_tag, children_only=True)
    active_panel_layout = self.tabs[self.active_tab]["panel_layout"]
    active_panel_layout.create_ui()
    active_panel_layout.update_all_panels()

  def rename_tab(self, tab_id: int, new_name: str):
    if tab_id in self.tabs:
      self.tabs[tab_id]["name"] = new_name

  def update_all_panels(self):
    self.tabs[self.active_tab]["panel_layout"].update_all_panels()

  def on_viewport_resize(self):
    self.tabs[self.active_tab]["panel_layout"].on_viewport_resize()

class PanelLayoutManager:
  def __init__(self, data_manager: DataManager, playback_manager, worker_manager, scale: float = 1.0):
    self.data_manager = data_manager
    self.playback_manager = playback_manager
    self.worker_manager = worker_manager
    self.scale = scale
    self.active_panels: list = []
    self.parent_tag = "tab_content_area"
    self._queue_resize = False
    self._created_handler_tags: set[str] = set()

    self.grip_size = int(GRIP_SIZE * self.scale)
    self.min_pane_size = int(MIN_PANE_SIZE * self.scale)

    initial_panel = TimeSeriesPanel(data_manager, playback_manager, worker_manager)
    self.layout: dict = {"type": "panel", "panel": initial_panel}

  def to_dict(self) -> dict:
    return self._layout_to_dict(self.layout)

  def _layout_to_dict(self, layout: dict) -> dict:
    if layout["type"] == "panel":
      return {
        "type": "panel",
        "panel": layout["panel"].to_dict()
      }
    else:  # split
      return {
        "type": "split",
        "orientation": layout["orientation"],
        "proportions": layout["proportions"],
        "children": [self._layout_to_dict(child) for child in layout["children"]]
      }

  @classmethod
  def load_from_dict(cls, data: dict, data_manager, playback_manager, worker_manager, scale: float = 1.0):
    manager = cls(data_manager, playback_manager, worker_manager, scale)
    manager.layout = manager._dict_to_layout(data)
    return manager

  def _dict_to_layout(self, data: dict) -> dict:
    if data["type"] == "panel":
      panel_data = data["panel"]
      if panel_data["type"] == "timeseries":
        panel = TimeSeriesPanel.load_from_dict(
          panel_data, self.data_manager, self.playback_manager, self.worker_manager
        )
        return {"type": "panel", "panel": panel}
      else:
        # Handle future panel types here or make a general mapping
        raise ValueError(f"Unknown panel type: {panel_data['type']}")
    else:  # split
      return {
        "type": "split",
        "orientation": data["orientation"],
        "proportions": data["proportions"],
        "children": [self._dict_to_layout(child) for child in data["children"]]
      }

  def create_ui(self):
    self.active_panels.clear()
    if dpg.does_item_exist(self.parent_tag):
      dpg.delete_item(self.parent_tag, children_only=True)
    self._cleanup_all_handlers()

    container_width, container_height = dpg.get_item_rect_size(self.parent_tag)
    if container_width == 0 and container_height == 0:
      self._queue_resize = True
    self._create_ui_recursive(self.layout, self.parent_tag, [], container_width, container_height)

  def destroy_ui(self):
    self._cleanup_ui_recursive(self.layout, [])
    self._cleanup_all_handlers()
    self.active_panels.clear()

  def _cleanup_all_handlers(self):
    for handler_tag in list(self._created_handler_tags):
      if dpg.does_item_exist(handler_tag):
        dpg.delete_item(handler_tag)
    self._created_handler_tags.clear()

  def _create_ui_recursive(self, layout: dict, parent_tag: str, path: list[int], width: int, height: int):
    if layout["type"] == "panel":
      self._create_panel_ui(layout, parent_tag, path, width, height)
    else:
      self._create_split_ui(layout, parent_tag, path, width, height)

  def _create_panel_ui(self, layout: dict, parent_tag: str, path: list[int], width: int, height: int):
    panel_tag = self._path_to_tag(path, "panel")
    panel = layout["panel"]
    self.active_panels.append(panel)
    text_size = int(13 * self.scale)
    bar_height = (text_size + 24) if width < int(329 * self.scale + 64) else (text_size + 8)  # adjust height to allow for scrollbar

    with dpg.child_window(parent=parent_tag, border=False, width=-1, height=-1, no_scrollbar=True):
      with dpg.group(horizontal=True):
        with dpg.child_window(tag=panel_tag, width=-(text_size + 16), height=bar_height, horizontal_scrollbar=True, no_scroll_with_mouse=True, border=False):
          with dpg.group(horizontal=True):
            # if you change the widths make sure to change the sum of widths (currently 329 * scale)
            dpg.add_input_text(default_value=panel.title, width=int(150 * self.scale), callback=lambda s, v: setattr(panel, "title", v))
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
        pane_width, pane_height = [(pane_sizes[i], -1), (-1, pane_sizes[i])][orientation]  # fill 2nd dim up to the border
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
      container_tag = self.parent_tag

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
        self._created_handler_tags.discard(handler_tag)

      for i, child in enumerate(layout["children"]):
        self._cleanup_ui_recursive(child, path + [i])

  def update_all_panels(self):
    if self._queue_resize:
      if (size := dpg.get_item_rect_size(self.parent_tag)) != [0, 0]:
        self._queue_resize = False
        self._resize_splits_recursive(self.layout, [], *size)
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
    else:  # leaf node/panel - adjust bar height to allow for scrollbar
      panel_tag = self._path_to_tag(path, "panel")
      if width is not None and width < int(329 * self.scale + 64):  # scaled widths of the elements in top bar + fixed 8 padding on left and right of each item
        dpg.configure_item(panel_tag, height=(int(13 * self.scale) + 24))
      else:
        dpg.configure_item(panel_tag, height=(int(13 * self.scale) + 8))

  def _get_split_geometry(self, layout: dict, available_size: tuple[int, int]) -> tuple[int, int, list[int]]:
    orientation = layout["orientation"]
    num_grips = len(layout["children"]) - 1
    usable_size = max(self.min_pane_size, available_size[orientation] - (num_grips * (self.grip_size + 8 * (2 - orientation))))  # approximate, scaling is weird
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
    handler_tag = f"{grip_tag}_handler"
    width, height = [(self.grip_size, -1), (-1, self.grip_size)][orientation]

    with dpg.child_window(tag=grip_tag, parent=parent_tag, width=width, height=height, no_scrollbar=True, border=False):
      button_tag = dpg.add_button(label="", width=-1, height=-1)

    with dpg.item_handler_registry(tag=handler_tag):
      user_data = (path, grip_index, orientation)
      dpg.add_item_active_handler(callback=self._on_grip_drag, user_data=user_data)
      dpg.add_item_deactivated_handler(callback=self._on_grip_end, user_data=user_data)
    dpg.bind_item_handler_registry(button_tag, handler_tag)
    self._created_handler_tags.add(handler_tag)

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
