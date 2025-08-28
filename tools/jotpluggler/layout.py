import uuid
import dearpygui.dearpygui as dpg
from abc import ABC, abstractmethod
from openpilot.tools.jotpluggler.data import DataManager
from openpilot.tools.jotpluggler.views import ViewPanel, TimeSeriesPanel


class LayoutNode(ABC):
  def __init__(self, node_id: str | None = None):
    self.node_id = node_id or str(uuid.uuid4())
    self.tag: str | None = None

  @abstractmethod
  def create_ui(self, parent_tag: str, width: int = -1, height: int = -1):
    pass

  @abstractmethod
  def destroy_ui(self):
    pass


class LeafNode(LayoutNode):
  """Leaf node that contains a single ViewPanel with controls"""

  def __init__(self, panel: ViewPanel, layout_manager=None, scale: float = 1.0, node_id: str = None):
    super().__init__(node_id)
    self.panel = panel
    self.layout_manager = layout_manager
    self.scale = scale

  def create_ui(self, parent_tag: str, width: int = -1, height: int = -1):
    """Create UI container with controls and panel"""
    self.tag = f"leaf_{self.node_id}"

    with dpg.child_window(tag=self.tag, parent=parent_tag, border=True, width=-1, height=-1, no_scrollbar=True):
      # Control bar
      with dpg.group(horizontal=True):
        dpg.add_input_text(tag=f"title_{self.node_id}", default_value=self.panel.title, width=int(100 * self.scale), callback=self._on_title_change)
        dpg.add_combo(
          items=["Time Series"],  # "Camera", "Text Log", "Map View"],
          tag=f"type_{self.node_id}",
          default_value="Time Series",
          width=int(100 * self.scale),
          callback=self._on_type_change,
        )
        dpg.add_button(label="Clear", callback=self._clear, width=int(50 * self.scale))
        dpg.add_button(label="Delete", callback=self._delete, width=int(50 * self.scale))
        dpg.add_button(label="Split H", callback=lambda: self._split("horizontal"), width=int(50 * self.scale))
        dpg.add_button(label="Split V", callback=lambda: self._split("vertical"), width=int(50 * self.scale))

      dpg.add_separator()

      # Panel content area
      panel_area_tag = f"panel_area_{self.node_id}"
      with dpg.child_window(tag=panel_area_tag, border=False, height=-1, width=-1, no_scrollbar=True):
        self.panel.create_ui(panel_area_tag)

  def destroy_ui(self):
    if self.panel:
      self.panel.destroy_ui()
    if self.tag and dpg.does_item_exist(self.tag):
      dpg.delete_item(self.tag)

  def _on_title_change(self, sender, app_data):
    self.panel.title = app_data

  def _on_type_change(self, sender, app_data):
    print(f"Panel type change requested: {app_data}")

  def _split(self, orientation: str):
    if self.layout_manager:
      self.layout_manager.split_node(self, orientation)

  def _clear(self):
    if hasattr(self.panel, 'clear_all_series'):
      self.panel.clear_all_series()

  def _delete(self):
    if self.layout_manager:
      self.layout_manager.delete_node(self)


class SplitterNode(LayoutNode):
  def __init__(self, children: list[LayoutNode], orientation: str = "horizontal", node_id: str | None = None):
    super().__init__(node_id)
    self.children = children if children else []
    self.orientation = orientation
    self.child_proportions = [1.0 / len(self.children) for _ in self.children] if self.children else []
    self.child_container_tags: list[str] = []  # Track container tags for resizing

  def add_child(self, child: LayoutNode, index: int = None):
    if index is None:
      self.children.append(child)
      self.child_proportions.append(0.0)
    else:
      self.children.insert(index, child)
      self.child_proportions.insert(index, 0.0)
    self._redistribute_proportions()

  def remove_child(self, child: LayoutNode):
    if child in self.children:
      index = self.children.index(child)
      self.children.remove(child)
      self.child_proportions.pop(index)
      child.destroy_ui()
      if self.children:
        self._redistribute_proportions()

  def replace_child(self, old_child: LayoutNode, new_child: LayoutNode):
    try:
      index = self.children.index(old_child)
      self.children[index] = new_child
      return index
    except ValueError:
      return None

  def _redistribute_proportions(self):
    if self.children:
      equal_proportion = 1.0 / len(self.children)
      self.child_proportions = [equal_proportion for _ in self.children]

  def resize_children(self):
    if not self.tag or not dpg.does_item_exist(self.tag):
      return

    available_width, available_height = dpg.get_item_rect_size(dpg.get_item_parent(self.tag))

    for i, container_tag in enumerate(self.child_container_tags):
      if not dpg.does_item_exist(container_tag):
        continue

      proportion = self.child_proportions[i] if i < len(self.child_proportions) else (1.0 / len(self.children))

      if self.orientation == "horizontal":
        new_width = max(100, int(available_width * proportion))
        dpg.configure_item(container_tag, width=new_width)
      else:
        new_height = max(100, int(available_height * proportion))
        dpg.configure_item(container_tag, height=new_height)

      child = self.children[i] if i < len(self.children) else None
      if child and isinstance(child, SplitterNode):
        child.resize_children()

  def create_ui(self, parent_tag: str, width: int = -1, height: int = -1):
    self.tag = f"splitter_{self.node_id}"
    self.child_container_tags = []

    if self.orientation == "horizontal":
      with dpg.group(tag=self.tag, parent=parent_tag, horizontal=True):
        for i, child in enumerate(self.children):
          proportion = self.child_proportions[i]
          child_width = max(100, int(width * proportion))
          container_tag = f"child_container_{self.node_id}_{i}"
          self.child_container_tags.append(container_tag)

          with dpg.child_window(tag=container_tag, width=child_width, height=-1, border=False, no_scrollbar=True, resizable_x=False):
            child.create_ui(container_tag, child_width, height)
    else:
      with dpg.group(tag=self.tag, parent=parent_tag):
        for i, child in enumerate(self.children):
          proportion = self.child_proportions[i]
          child_height = max(100, int(height * proportion))
          container_tag = f"child_container_{self.node_id}_{i}"
          self.child_container_tags.append(container_tag)

          with dpg.child_window(tag=container_tag, width=-1, height=child_height, border=False, no_scrollbar=True, resizable_y=False):
            child.create_ui(container_tag, width, child_height)

  def destroy_ui(self):
    for child in self.children:
      if child:
        child.destroy_ui()
    if self.tag and dpg.does_item_exist(self.tag):
      dpg.delete_item(self.tag)
    self.child_container_tags.clear()


class PlotLayoutManager:
  def __init__(self, data_manager: DataManager, playback_manager, scale: float = 1.0):
    self.data_manager = data_manager
    self.playback_manager = playback_manager
    self.scale = scale
    self.container_tag = "plot_layout_container"
    self._initialize_default_layout()

  def _initialize_default_layout(self):
    panel = TimeSeriesPanel(self.data_manager, self.playback_manager)
    self.root_node = LeafNode(panel, layout_manager=self, scale=self.scale)

  def create_ui(self, parent_tag: str):
    if dpg.does_item_exist(self.container_tag):
      dpg.delete_item(self.container_tag)

    with dpg.child_window(tag=self.container_tag, parent=parent_tag, border=False, width=-1, height=-1, no_scrollbar=True):
      container_width, container_height = dpg.get_item_rect_size(self.container_tag)
      self.root_node.create_ui(self.container_tag, container_width, container_height)

  def on_viewport_resize(self):
    if isinstance(self.root_node, SplitterNode):
      self.root_node.resize_children()

  def split_node(self, node: LeafNode, orientation: str):
    # create new panel for the split
    new_panel = TimeSeriesPanel(self.data_manager, self.playback_manager)  # TODO: create same type of panel as the split
    new_leaf = LeafNode(new_panel, layout_manager=self, scale=self.scale)

    parent_node, child_index = self._find_parent_and_index(node)

    if parent_node is None:  # root node - create new splitter as root
      self.root_node = SplitterNode([node, new_leaf], orientation)
      self._update_ui_for_node(self.root_node, self.container_tag)
    elif isinstance(parent_node, SplitterNode) and parent_node.orientation == orientation:  # same orientation - add to existing splitter
      parent_node.add_child(new_leaf, child_index + 1)
      self._update_ui_for_node(parent_node)
    else:  # different orientation - replace node with new splitter
      new_splitter = SplitterNode([node, new_leaf], orientation)
      self._replace_child_in_parent(parent_node, node, new_splitter)

  def delete_node(self, node: LeafNode):  # TODO: actually delete the node, not just the ui for the node
    parent_node, child_index = self._find_parent_and_index(node)

    if parent_node is None:  # root deletion - replace with new default
      node.destroy_ui()
      self._initialize_default_layout()
      self._update_ui_for_node(self.root_node, self.container_tag)
    elif isinstance(parent_node, SplitterNode):
      parent_node.remove_child(node)
      if len(parent_node.children) == 1:  # collapse splitter --> leaf to just leaf
        remaining_child = parent_node.children[0]
        grandparent_node, parent_index = self._find_parent_and_index(parent_node)

        if grandparent_node is None:  # promote remaining child to root
          parent_node.children.remove(remaining_child)
          self.root_node = remaining_child
          parent_node.destroy_ui()
          self._update_ui_for_node(self.root_node, self.container_tag)
        else:  # replace splitter with remaining child in grandparent node
          self._replace_child_in_parent(grandparent_node, parent_node, remaining_child)
      else:  # update splpitter contents
        self._update_ui_for_node(parent_node)

  def _replace_child_in_parent(self, parent_node: SplitterNode, old_child: LayoutNode, new_child: LayoutNode):
    child_index = parent_node.children.index(old_child)
    child_container_tag = f"child_container_{parent_node.node_id}_{child_index}"

    parent_node.replace_child(old_child, new_child)

    # Clean up old child if it's being replaced (not just moved)
    if old_child != new_child:
      old_child.destroy_ui()

    if dpg.does_item_exist(child_container_tag):
      dpg.delete_item(child_container_tag, children_only=True)
      container_width, container_height = dpg.get_item_rect_size(child_container_tag)
      new_child.create_ui(child_container_tag, container_width, container_height)

  def _update_ui_for_node(self, node: LayoutNode, container_tag: str = None):
    if container_tag:  #  update node in a specific container (usually root)
      dpg.delete_item(container_tag, children_only=True)
      container_width, container_height = dpg.get_item_rect_size(container_tag)
      node.create_ui(container_tag, container_width, container_height)
    else:  # update node in its current location (splitter updates)
      if node.tag and dpg.does_item_exist(node.tag):
        parent_container = dpg.get_item_parent(node.tag)
        node.destroy_ui()
        if parent_container and dpg.does_item_exist(parent_container):
          parent_width, parent_height = dpg.get_item_rect_size(parent_container)
          node.create_ui(parent_container, parent_width, parent_height)

  def _find_parent_and_index(self, target_node: LayoutNode):  # TODO: probably can be stored in child
    def search_recursive(node: LayoutNode | None, parent: LayoutNode | None = None, index: int = 0):
      if node == target_node:
        return parent, index
      if isinstance(node, SplitterNode):
        for i, child in enumerate(node.children):
          result = search_recursive(child, node, i)
          if result[0] is not None:
            return result
      return None, None

    return search_recursive(self.root_node)
