import dearpygui.dearpygui as dpg
from openpilot.tools.jotpluggler.data import DataManager
from openpilot.tools.jotpluggler.views import TimeSeriesPanel


class PlotLayoutManager:
  def __init__(self, data_manager: DataManager, playback_manager, scale: float = 1.0):
    self.data_manager = data_manager
    self.playback_manager = playback_manager
    self.scale = scale
    self.container_tag = "plot_layout_container"
    self.active_panels = []

    initial_panel = TimeSeriesPanel(data_manager, playback_manager)
    self.active_panels.append(initial_panel)
    self.layout = {"type": "panel", "panel": initial_panel}

  def create_ui(self, parent_tag: str):
    if dpg.does_item_exist(self.container_tag):
      dpg.delete_item(self.container_tag)

    with dpg.child_window(tag=self.container_tag, parent=parent_tag, border=False, width=-1, height=-1, no_scrollbar=True):
      container_width, container_height = dpg.get_item_rect_size(self.container_tag)
      self._create_ui_recursive(self.layout, self.container_tag, [], container_width, container_height)

  def on_viewport_resize(self):
    self._resize_splits_recursive(self.layout, [])

  def split_panel(self, panel_path: list[int], orientation: str):
    current_layout = self._get_layout_at_path(panel_path)
    existing_panel = current_layout["panel"]
    new_panel = TimeSeriesPanel(self.data_manager, self.playback_manager)
    self.active_panels.append(new_panel)

    parent, child_index = self._get_parent_and_index(panel_path)

    if parent is None:  # Root split
      self.layout = {
        "type": "split",
        "orientation": orientation,
        "children": [{"type": "panel", "panel": existing_panel}, {"type": "panel", "panel": new_panel}],
        "proportions": [0.5, 0.5],
      }
      self._rebuild_ui_at_path([])
    elif parent["type"] == "split" and parent["orientation"] == orientation:
      parent["children"].insert(child_index + 1, {"type": "panel", "panel": new_panel})
      parent["proportions"] = [1.0 / len(parent["children"])] * len(parent["children"])
      self._rebuild_ui_at_path(panel_path[:-1])
    else:
      new_split = {"type": "split", "orientation": orientation, "children": [current_layout, {"type": "panel", "panel": new_panel}], "proportions": [0.5, 0.5]}
      self._replace_layout_at_path(panel_path, new_split)
      self._rebuild_ui_at_path(panel_path)

  def delete_panel(self, panel_path: list[int]):
    if not panel_path:  # Root deletion
      old_panel = self.layout["panel"]
      old_panel.destroy_ui()
      self.active_panels.remove(old_panel)
      new_panel = TimeSeriesPanel(self.data_manager, self.playback_manager)
      self.active_panels.append(new_panel)
      self.layout = {"type": "panel", "panel": new_panel}
      self._rebuild_ui_at_path([])
      return

    parent, child_index = self._get_parent_and_index(panel_path)
    layout_to_delete = parent["children"][child_index]
    self._cleanup_ui_recursive(layout_to_delete)

    parent["children"].pop(child_index)
    parent["proportions"].pop(child_index)

    if len(parent["children"]) == 1: # remove parent and collapse
      remaining_child = parent["children"][0]
      if len(panel_path) == 1: # parent is at root level - promote remaining child to root
        self.layout = remaining_child
        self._rebuild_ui_at_path([])
      else: # replace parent with remaining child in grandparent
        grandparent_path = panel_path[:-2]
        parent_index = panel_path[-2]
        self._replace_layout_at_path(grandparent_path + [parent_index], remaining_child)
        self._rebuild_ui_at_path(grandparent_path + [parent_index])
    else: # redistribute proportions
      equal_prop = 1.0 / len(parent["children"])
      parent["proportions"] = [equal_prop] * len(parent["children"])
      self._rebuild_ui_at_path(panel_path[:-1])

  def update_all_panels(self):
    for panel in self.active_panels:
      panel.update()

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

  def _create_ui_recursive(self, layout: dict, parent_tag: str, path: list[int], width: int, height: int):
    if layout["type"] == "panel":
      self._create_panel_ui(layout, parent_tag, path)
    else:
      self._create_split_ui(layout, parent_tag, path, width, height)

  def _create_panel_ui(self, layout: dict, parent_tag: str, path: list[int]):
    panel_tag = self._path_to_tag(path, "panel")

    with dpg.child_window(tag=panel_tag, parent=parent_tag, border=True, width=-1, height=-1, no_scrollbar=True):
      with dpg.group(horizontal=True):
        dpg.add_input_text(default_value=layout["panel"].title, width=int(100 * self.scale), callback=lambda s, v: setattr(layout["panel"], "title", v))
        dpg.add_combo(items=["Time Series"], default_value="Time Series", width=int(100 * self.scale))
        dpg.add_button(label="Clear", callback=lambda: self._clear_panel(layout["panel"]), width=int(50 * self.scale))
        dpg.add_button(label="Delete", callback=lambda: self.delete_panel(path), width=int(50 * self.scale))
        dpg.add_button(label="Split H", callback=lambda: self.split_panel(path, "horizontal"), width=int(50 * self.scale))
        dpg.add_button(label="Split V", callback=lambda: self.split_panel(path, "vertical"), width=int(50 * self.scale))

      dpg.add_separator()

      content_tag = self._path_to_tag(path, "content")
      with dpg.child_window(tag=content_tag, border=False, height=-1, width=-1, no_scrollbar=True):
        layout["panel"].create_ui(content_tag)

  def _create_split_ui(self, layout: dict, parent_tag: str, path: list[int], width: int, height: int):
    split_tag = self._path_to_tag(path, "split")
    is_horizontal = layout["orientation"] == "horizontal"

    with dpg.group(tag=split_tag, parent=parent_tag, horizontal=is_horizontal):
      for i, (child_layout, proportion) in enumerate(zip(layout["children"], layout["proportions"], strict=True)):
        child_path = path + [i]
        container_tag = self._path_to_tag(child_path, "container")

        if is_horizontal:
          child_width = max(100, int(width * proportion))
          with dpg.child_window(tag=container_tag, width=child_width, height=-1, border=False, no_scrollbar=True):
            self._create_ui_recursive(child_layout, container_tag, child_path, child_width, height)
        else:
          child_height = max(100, int(height * proportion))
          with dpg.child_window(tag=container_tag, width=-1, height=child_height, border=False, no_scrollbar=True):
            self._create_ui_recursive(child_layout, container_tag, child_path, width, child_height)

  def _rebuild_ui_at_path(self, path: list[int]):
    layout = self._get_layout_at_path(path)

    if not path:  # Root update
      dpg.delete_item(self.container_tag, children_only=True)
      container_width, container_height = dpg.get_item_rect_size(self.container_tag)
      self._create_ui_recursive(layout, self.container_tag, path, container_width, container_height)
    else:
      container_tag = self._path_to_tag(path, "container")
      if dpg.does_item_exist(container_tag):
        self._cleanup_ui_recursive(layout)
        dpg.delete_item(container_tag, children_only=True)
        width, height = dpg.get_item_rect_size(container_tag)
        self._create_ui_recursive(layout, container_tag, path, width, height)

  def _cleanup_ui_recursive(self, layout: dict):
    if layout["type"] == "panel":
      panel = layout["panel"]
      panel.destroy_ui()
      if panel in self.active_panels:
        self.active_panels.remove(panel)
    else:
      for child in layout["children"]:
        self._cleanup_ui_recursive(child)

  def _resize_splits_recursive(self, layout: dict, path: list[int]):
    if layout["type"] == "split":
      split_tag = self._path_to_tag(path, "split")
      if dpg.does_item_exist(split_tag):
        parent_tag = dpg.get_item_parent(split_tag)
        available_width, available_height = dpg.get_item_rect_size(parent_tag)

        for i, proportion in enumerate(layout["proportions"]):
          child_path = path + [i]
          container_tag = self._path_to_tag(child_path, "container")
          if dpg.does_item_exist(container_tag):
            if layout["orientation"] == "horizontal":
              dpg.configure_item(container_tag, width=max(100, int(available_width * proportion)))
            else:
              dpg.configure_item(container_tag, height=max(100, int(available_height * proportion)))

          self._resize_splits_recursive(layout["children"][i], child_path)

  def _clear_panel(self, panel):
    if hasattr(panel, 'clear_all_series'):
      panel.clear_all_series()
