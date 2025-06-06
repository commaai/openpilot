import os
import pyray as rl
from dataclasses import dataclass
from collections.abc import Callable
from abc import ABC, abstractmethod
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.lib.button import gui_button
from openpilot.system.ui.lib.toggle import Toggle
from openpilot.system.ui.lib.toggle import WIDTH as TOGGLE_WIDTH, HEIGHT as TOGGLE_HEIGHT


LINE_PADDING = 40
LINE_COLOR = rl.GRAY
ITEM_PADDING = 20
ITEM_SPACING = 80
ITEM_BASE_HEIGHT = 170
ITEM_TEXT_FONT_SIZE = 50
ITEM_TEXT_COLOR = rl.WHITE
ITEM_DESC_TEXT_COLOR = rl.Color(128, 128, 128, 255)
ITEM_DESC_FONT_SIZE = 40
ITEM_DESC_V_OFFSET = 130
RIGHT_ITEM_PADDING = 20
ICON_SIZE = 80
BUTTON_WIDTH = 250
BUTTON_HEIGHT = 100
BUTTON_BORDER_RADIUS = 50
BUTTON_FONT_SIZE = 35
BUTTON_FONT_WEIGHT = FontWeight.MEDIUM


# Abstract base class for right-side items
class RightItem(ABC):
  def __init__(self, width: int = 100):
    self.width = width
    self.enabled = True

  @abstractmethod
  def draw(self, rect: rl.Rectangle) -> bool:
    pass

  @abstractmethod
  def get_width(self) -> int:
    pass


class ToggleRightItem(RightItem):
  def __init__(self, initial_state: bool = False, width: int = TOGGLE_WIDTH):
    super().__init__(width)
    self.toggle = Toggle(initial_state=initial_state)
    self.state = initial_state
    self.enabled = True

  def draw(self, rect: rl.Rectangle) -> bool:
    if self.toggle.render(rl.Rectangle(rect.x, rect.y + (rect.height - TOGGLE_HEIGHT) / 2, self.width, TOGGLE_HEIGHT)):
      self.state = not self.state
      return True
    return False

  def get_width(self) -> int:
    return self.width

  def set_state(self, state: bool):
    self.state = state
    self.toggle.set_state(state)

  def get_state(self) -> bool:
    return self.state

  def set_enabled(self, enabled: bool):
    self.enabled = enabled


class ButtonRightItem(RightItem):
  def __init__(self, text: str, width: int = BUTTON_WIDTH):
    super().__init__(width)
    self.text = text
    self.enabled = True

  def draw(self, rect: rl.Rectangle) -> bool:
    return (
      gui_button(
        rl.Rectangle(rect.x, rect.y + (rect.height - BUTTON_HEIGHT) / 2, BUTTON_WIDTH, BUTTON_HEIGHT),
        self.text,
        border_radius=BUTTON_BORDER_RADIUS,
        font_weight=BUTTON_FONT_WEIGHT,
        font_size=BUTTON_FONT_SIZE,
        is_enabled=self.enabled,
      )
      == 1
    )

  def get_width(self) -> int:
    return self.width

  def set_enabled(self, enabled: bool):
    self.enabled = enabled


class TextRightItem(RightItem):
  def __init__(self, text: str, color: rl.Color = ITEM_TEXT_COLOR, font_size: int = ITEM_TEXT_FONT_SIZE):
    self.text = text
    self.color = color
    self.font_size = font_size

    font = gui_app.font(FontWeight.NORMAL)
    text_width = measure_text_cached(font, text, font_size).x
    super().__init__(int(text_width + 20))

  def draw(self, rect: rl.Rectangle) -> bool:
    font = gui_app.font(FontWeight.NORMAL)
    text_size = measure_text_cached(font, self.text, self.font_size)

    # Center the text in the allocated rectangle
    text_x = rect.x + (rect.width - text_size.x) / 2
    text_y = rect.y + (rect.height - text_size.y) / 2

    rl.draw_text_ex(font, self.text, rl.Vector2(text_x, text_y), self.font_size, 0, self.color)
    return False

  def get_width(self) -> int:
    return self.width

  def set_text(self, text: str):
    self.text = text
    font = gui_app.font(FontWeight.NORMAL)
    text_width = measure_text_cached(font, text, self.font_size).x
    self.width = int(text_width + 20)


@dataclass
class ListItem:
  title: str
  icon: str | None = None
  description: str | None = None
  description_visible: bool = False
  rect: "rl.Rectangle | None" = None
  callback: Callable | None = None
  right_item: RightItem | None = None

  # Cached properties for performance
  _wrapped_description: str | None = None
  _description_height: float = 0

  def get_right_item(self) -> RightItem | None:
    return self.right_item

  def get_item_height(self, font: rl.Font, max_width: int) -> float:
    if self.description_visible and self.description:
      if not self._wrapped_description:
        wrapped_lines = wrap_text(font, self.description, ITEM_DESC_FONT_SIZE, max_width)
        self._wrapped_description = "\n".join(wrapped_lines)
        self._description_height = len(wrapped_lines) * 20 + 10  # Line height + padding
      return ITEM_BASE_HEIGHT + self._description_height - (ITEM_BASE_HEIGHT - ITEM_DESC_V_OFFSET) + ITEM_SPACING
    return ITEM_BASE_HEIGHT

  def get_content_width(self, total_width: int) -> int:
    if self.right_item:
      return total_width - self.right_item.get_width() - RIGHT_ITEM_PADDING
    return total_width

  def get_right_item_rect(self, item_rect: rl.Rectangle) -> rl.Rectangle:
    if not self.right_item:
      return rl.Rectangle(0, 0, 0, 0)

    right_width = self.right_item.get_width()
    right_x = item_rect.x + item_rect.width - right_width
    right_y = item_rect.y
    return rl.Rectangle(right_x, right_y, right_width, ITEM_BASE_HEIGHT)


class ListView:
  def __init__(self, items: list[ListItem]):
    self._items: list[ListItem] = items
    self._last_dim: tuple[float, float] = (0, 0)
    self.scroll_panel = GuiScrollPanel()

    self._font_normal = gui_app.font(FontWeight.NORMAL)

    # Interaction state
    self._hovered_item: int = -1
    self._last_mouse_pos = rl.Vector2(0, 0)

    self._total_height: float = 0
    self._visible_range = (0, 0)

  def invalid_height_cache(self):
    self._last_dim = (0, 0)

  def render(self, rect: rl.Rectangle):
    if self._last_dim != (rect.width, rect.height):
      self._update_item_rects(rect)
      self._last_dim = (rect.width, rect.height)

    # Update layout and handle scrolling
    content_rect = rl.Rectangle(rect.x, rect.y, rect.width, self._total_height)
    scroll_offset = self.scroll_panel.handle_scroll(rect, content_rect)

    # Handle mouse interaction
    if self.scroll_panel.is_click_valid():
      self._handle_mouse_interaction(rect, scroll_offset)

    # Set scissor mode for clipping
    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))

    # Calculate visible range for performance
    self._calculate_visible_range(rect, -scroll_offset.y)

    # Render only visible items
    for i in range(self._visible_range[0], min(self._visible_range[1], len(self._items))):
      item = self._items[i]
      if item.rect:
        adjusted_rect = rl.Rectangle(item.rect.x, item.rect.y + scroll_offset.y, item.rect.width, item.rect.height)
        self._render_item(item, adjusted_rect, i)

        if i != len(self._items) - 1:
          rl.draw_line_ex(
            rl.Vector2(adjusted_rect.x + LINE_PADDING, adjusted_rect.y + adjusted_rect.height - 1),
            rl.Vector2(
              adjusted_rect.x + adjusted_rect.width - LINE_PADDING * 2, adjusted_rect.y + adjusted_rect.height - 1
            ),
            1.0,
            LINE_COLOR,
          )
    rl.end_scissor_mode()

  def _render_item(self, item: ListItem, rect: rl.Rectangle, index: int):
    content_x = rect.x + ITEM_PADDING
    text_x = content_x

    # Calculate available width for main content
    content_width = item.get_content_width(int(rect.width - ITEM_PADDING * 2))

    # Draw icon if present
    if item.icon:
      icon_texture = gui_app.texture(os.path.join("icons", item.icon), ICON_SIZE, ICON_SIZE)
      rl.draw_texture(
        icon_texture, int(content_x), int(rect.y + (ITEM_BASE_HEIGHT - icon_texture.width) // 2), rl.WHITE
      )
      text_x += ICON_SIZE + ITEM_PADDING

    # Draw main text
    text_size = measure_text_cached(self._font_normal, item.title, ITEM_TEXT_FONT_SIZE)
    item_y = rect.y + (ITEM_BASE_HEIGHT - text_size.y) // 2
    rl.draw_text_ex(self._font_normal, item.title, rl.Vector2(text_x, item_y), ITEM_TEXT_FONT_SIZE, 0, ITEM_TEXT_COLOR)

    # Draw description if visible (adjust width for right item)
    if item.description_visible and item._wrapped_description:
      desc_y = rect.y + ITEM_DESC_V_OFFSET
      desc_max_width = int(content_width - (text_x - content_x))

      # Re-wrap description if needed due to right item
      if (item.right_item and item.description) and not item._wrapped_description:
        wrapped_lines = wrap_text(self._font_normal, item.description, ITEM_DESC_FONT_SIZE, desc_max_width)
        item._wrapped_description = "\n".join(wrapped_lines)

      rl.draw_text_ex(
        self._font_normal,
        item._wrapped_description,
        rl.Vector2(text_x, desc_y),
        ITEM_DESC_FONT_SIZE,
        0,
        ITEM_DESC_TEXT_COLOR,
      )

    # Draw right item if present
    if item.right_item:
      right_rect = item.get_right_item_rect(rect)
      # Adjust for scroll offset
      right_rect.y = right_rect.y
      if item.right_item.draw(right_rect):
        # Right item was clicked/activated
        if item.callback:
          item.callback()

  def _update_item_rects(self, container_rect: rl.Rectangle) -> None:
    current_y: float = 0.0
    self._total_height = 0

    for item in self._items:
      content_width = item.get_content_width(int(container_rect.width - ITEM_PADDING * 2))
      item_height = item.get_item_height(self._font_normal, content_width)
      item.rect = rl.Rectangle(container_rect.x, container_rect.y + current_y, container_rect.width, item_height)
      current_y += item_height
      self._total_height += item_height

  def _calculate_visible_range(self, rect: rl.Rectangle, scroll_offset: float):
    if not self._items:
      self._visible_range = (0, 0)
      return

    visible_top = scroll_offset
    visible_bottom = scroll_offset + rect.height

    start_idx = 0
    end_idx = len(self._items)

    # Find first visible item
    for i, item in enumerate(self._items):
      if item.rect and item.rect.y + item.rect.height >= visible_top:
        start_idx = max(0, i - 1)
        break

    # Find last visible item
    for i in range(start_idx, len(self._items)):
      item = self._items[i]
      if item.rect and item.rect.y > visible_bottom:
        end_idx = min(len(self._items), i + 2)
        break

    self._visible_range = (start_idx, end_idx)

  def _handle_mouse_interaction(self, rect: rl.Rectangle, scroll_offset: rl.Vector2):
    mouse_pos = rl.get_mouse_position()

    self._hovered_item = -1
    if not rl.check_collision_point_rec(mouse_pos, rect):
      return

    content_mouse_y = mouse_pos.y - rect.y - scroll_offset.y

    for i, item in enumerate(self._items):
      if item.rect:
        # Check if mouse is within this item's bounds in content space
        if (
          mouse_pos.x >= rect.x
          and mouse_pos.x <= rect.x + rect.width
          and content_mouse_y >= item.rect.y
          and content_mouse_y <= item.rect.y + item.rect.height
        ):
          item_screen_y = item.rect.y + scroll_offset.y
          if item_screen_y < rect.height and item_screen_y + item.rect.height > 0:
            self._hovered_item = i
            break

    # Handle click on main item (not right item)
    if rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT) and self._hovered_item >= 0:
      item = self._items[self._hovered_item]

      # Check if click was on right item area
      if item.right_item and item.rect:
        adjusted_rect = rl.Rectangle(item.rect.x, item.rect.y + scroll_offset.y, item.rect.width, item.rect.height)
        right_rect = item.get_right_item_rect(adjusted_rect)
        if rl.check_collision_point_rec(mouse_pos, right_rect):
          # Click was handled by right item, don't process main item click
          return

      # Toggle description visibility if item has description
      if item.description:
        item.description_visible = not item.description_visible
        # Force layout update when description visibility changes
        self._last_dim = (0, 0)

      # Call item callback
      if item.callback:
        item.callback()


# Factory functions
def simple_item(title: str, callback: Callable | None = None) -> ListItem:
  return ListItem(title=title, callback=callback)


def toggle_item(
  title: str, description: str = None, initial_state: bool = False, callback: Callable | None = None, icon: str = ""
) -> ListItem:
  toggle = ToggleRightItem(initial_state=initial_state)
  return ListItem(title=title, description=description, right_item=toggle, icon=icon, callback=callback)


def button_item(title: str, button_text: str, description: str = None, callback: Callable | None = None) -> ListItem:
  button = ButtonRightItem(text=button_text)
  return ListItem(title=title, description=description, right_item=button, callback=callback)


def text_item(title: str, value: str, description: str = None, callback: Callable | None = None) -> ListItem:
  text_item = TextRightItem(text=value, color=rl.Color(170, 170, 170, 255))
  return ListItem(title=title, description=description, right_item=text_item, callback=callback)
