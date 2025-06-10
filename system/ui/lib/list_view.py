import os
import pyray as rl
from dataclasses import dataclass
from collections.abc import Callable
from abc import ABC
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.application import gui_app, FontWeight, Widget
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.lib.button import gui_button, ButtonStyle
from openpilot.system.ui.lib.toggle import Toggle, WIDTH as TOGGLE_WIDTH, HEIGHT as TOGGLE_HEIGHT

ITEM_BASE_HEIGHT = 170
LINE_PADDING = 40
LINE_COLOR = rl.GRAY
ITEM_PADDING = 20
ITEM_SPACING = 80
ITEM_TEXT_FONT_SIZE = 50
ITEM_TEXT_COLOR = rl.WHITE
ITEM_DESC_TEXT_COLOR = rl.Color(128, 128, 128, 255)
ITEM_DESC_FONT_SIZE = 40
ITEM_DESC_V_OFFSET = 140
RIGHT_ITEM_PADDING = 20
ICON_SIZE = 80
BUTTON_WIDTH = 250
BUTTON_HEIGHT = 100
BUTTON_BORDER_RADIUS = 50
BUTTON_FONT_SIZE = 35
BUTTON_FONT_WEIGHT = FontWeight.MEDIUM

TEXT_PADDING = 20


def _resolve_value(value, default=""):
  if callable(value):
    return value()
  return value if value is not None else default


# Abstract base class for right-side items
class ItemAction(Widget, ABC):
  def __init__(self, width: int = 100, enabled: bool | Callable[[], bool] = True):
    super().__init__()
    self.width = width
    self._enabled_source = enabled

  @property
  def enabled(self):
    return _resolve_value(self._enabled_source, False)

  def get_width(self) -> int:
    return self.width


class ToggleAction(ItemAction):
  def __init__(self, initial_state: bool = False, callback: Callable | None = None, width: int = TOGGLE_WIDTH, enabled: bool | Callable[[], bool] = True):
    super().__init__(width, enabled)
    self.toggle = Toggle(initial_state=initial_state)
    self.state = initial_state
    self._callback: Callable | None = callback

  def _render(self, rect: rl.Rectangle) -> bool:
    self.toggle.set_enabled(self.enabled)
    result = self.toggle.render(rl.Rectangle(rect.x, rect.y + (rect.height - TOGGLE_HEIGHT) / 2, self.width, TOGGLE_HEIGHT))
    if result == 1 and self._callback is not None:
      self._callback(self.toggle.get_state())
      return True
    return False

  def set_state(self, state: bool):
    self.state = state
    self.toggle.set_state(state)

  def get_state(self) -> bool:
    return self.state


class ButtonAction(ItemAction):
  def __init__(self, text: str | Callable[[], str], callback: Callable | None = None, width: int = BUTTON_WIDTH,
               enabled: bool | Callable[[], bool] = True):
    super().__init__(width, enabled)
    self._text_source = text
    self._callback: Callable | None = callback

  @property
  def text(self):
    return _resolve_value(self._text_source, "Error")

  def _render(self, rect: rl.Rectangle) -> bool:
    if gui_button(
      rl.Rectangle(rect.x, rect.y + (rect.height - BUTTON_HEIGHT) / 2, BUTTON_WIDTH, BUTTON_HEIGHT),
      self.text,
      border_radius=BUTTON_BORDER_RADIUS,
      font_weight=BUTTON_FONT_WEIGHT,
      font_size=BUTTON_FONT_SIZE,
      button_style=ButtonStyle.LIST_ACTION,
      is_enabled=self.enabled) == 1:
      if self._callback is not None:
        self._callback()
      return True
    return False


class TextAction(ItemAction):
  def __init__(self, text: str | Callable[[], str], color: rl.Color = ITEM_TEXT_COLOR, enabled: bool | Callable[[], bool] = True):
    self._text_source = text
    self.color = color

    self._font = gui_app.font(FontWeight.NORMAL)
    initial_text = _resolve_value(text, "")
    text_width = measure_text_cached(self._font, initial_text, ITEM_TEXT_FONT_SIZE).x
    super().__init__(int(text_width + TEXT_PADDING), enabled)

  @property
  def text(self):
    return _resolve_value(self._text_source, "Error")

  def _render(self, rect: rl.Rectangle) -> bool:
    current_text = self.text
    text_size = measure_text_cached(self._font, current_text, ITEM_TEXT_FONT_SIZE)

    text_x = rect.x + (rect.width - text_size.x) / 2
    text_y = rect.y + (rect.height - text_size.y) / 2
    rl.draw_text_ex(self._font, current_text, rl.Vector2(text_x, text_y), ITEM_TEXT_FONT_SIZE, 0, self.color)
    return False

  def get_width(self) -> int:
    text_width = measure_text_cached(self._font, self.text, ITEM_TEXT_FONT_SIZE).x
    return int(text_width + TEXT_PADDING)


class DualButtonAction(ItemAction):
  def __init__(self, left_text: str, right_text: str, left_callback: Callable = None,
               right_callback: Callable = None, enabled: bool | Callable[[], bool] = True):
    super().__init__(width=0, enabled=enabled)  # Width 0 means use full width
    self.left_text, self.right_text = left_text, right_text
    self.left_callback, self.right_callback = left_callback, right_callback

  def _render(self, rect: rl.Rectangle) -> bool:
    button_spacing = 30
    button_height = 120
    button_width = (rect.width - button_spacing) / 2
    button_y = rect.y + (rect.height - button_height) / 2

    left_rect = rl.Rectangle(rect.x, button_y, button_width, button_height)
    right_rect = rl.Rectangle(rect.x + button_width + button_spacing, button_y, button_width, button_height)

    left_clicked = gui_button(left_rect, self.left_text, button_style=ButtonStyle.LIST_ACTION) == 1
    right_clicked = gui_button(right_rect, self.right_text, button_style=ButtonStyle.DANGER) == 1

    if left_clicked and self.left_callback:
      self.left_callback()
      return True
    if right_clicked and self.right_callback:
      self.right_callback()
      return True
    return False


@dataclass
class ListItem:
  title: str
  icon: str | None = None
  description: str | Callable[[], str] | None = None
  description_visible: bool = False
  rect: "rl.Rectangle" = rl.Rectangle(0, 0, 0, 0)
  action_item: ItemAction | None = None
  visible: bool | Callable[[], bool] = True

  # Cached properties for performance
  _prev_max_width: int = 0
  _wrapped_description: str | None = None
  _prev_description: str | None = None
  _description_height: float = 0

  @property
  def is_visible(self) -> bool:
    return bool(_resolve_value(self.visible, True))

  def get_description(self):
    return _resolve_value(self.description, None)

  def get_item_height(self, font: rl.Font, max_width: int) -> float:
    if not self.is_visible:
      return 0

    current_description = self.get_description()
    if self.description_visible and current_description:
      if (
        not self._wrapped_description
        or current_description != self._prev_description
        or max_width != self._prev_max_width
      ):
        self._prev_max_width = max_width
        self._prev_description = current_description

        wrapped_lines = wrap_text(font, current_description, ITEM_DESC_FONT_SIZE, max_width)
        self._wrapped_description = "\n".join(wrapped_lines)
        self._description_height = len(wrapped_lines) * ITEM_DESC_FONT_SIZE + 10
      return ITEM_BASE_HEIGHT + self._description_height - (ITEM_BASE_HEIGHT - ITEM_DESC_V_OFFSET) + ITEM_PADDING
    return ITEM_BASE_HEIGHT

  def get_content_width(self, total_width: int) -> int:
    if self.action_item and self.action_item.get_width() > 0:
      return total_width - self.action_item.get_width() - RIGHT_ITEM_PADDING
    return total_width

  def get_right_item_rect(self, item_rect: rl.Rectangle) -> rl.Rectangle:
    if not self.action_item:
      return rl.Rectangle(0, 0, 0, 0)

    right_width = self.action_item.get_width()
    if right_width == 0:  # Full width action (like DualButtonAction)
      return rl.Rectangle(item_rect.x + ITEM_PADDING, item_rect.y,
                          item_rect.width - (ITEM_PADDING * 2), ITEM_BASE_HEIGHT)

    right_x = item_rect.x + item_rect.width - right_width
    right_y = item_rect.y
    return rl.Rectangle(right_x, right_y, right_width, ITEM_BASE_HEIGHT)


class ListView(Widget):
  def __init__(self, items: list[ListItem]):
    super().__init__()
    self._items = items
    self.scroll_panel = GuiScrollPanel()
    self._font = gui_app.font(FontWeight.NORMAL)
    self._hovered_item = -1

  def _render(self, rect: rl.Rectangle):
    total_height = self._update_item_rects(rect)

    # Update layout and handle scrolling
    content_rect = rl.Rectangle(rect.x, rect.y, rect.width, total_height)
    scroll_offset = self.scroll_panel.handle_scroll(rect, content_rect)

    # Handle mouse interaction
    if self.scroll_panel.is_click_valid():
      self._handle_mouse_interaction(rect, scroll_offset)

    # Set scissor mode for clipping
    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))

    for i, item in enumerate(self._items):
      if not item.is_visible:
        continue

      y = int(item.rect.y + scroll_offset.y)
      if y + item.rect.height <= rect.y or y >= rect.y + rect.height:
        continue

      self._render_item(item, y)

      # Draw separator line
      next_visible_item = self._get_next_visible_item(i)
      if next_visible_item is not None:
        line_y = int(y + item.rect.height - 1)
        rl.draw_line(
          int(item.rect.x) + LINE_PADDING,
          line_y,
          int(item.rect.x + item.rect.width) - LINE_PADDING * 2,
          line_y,
          LINE_COLOR,
        )

    rl.end_scissor_mode()

  def _get_next_visible_item(self, current_index: int) -> int | None:
    for i in range(current_index + 1, len(self._items)):
      if self._items[i].is_visible:
        return i
    return None

  def _update_item_rects(self, container_rect: rl.Rectangle) -> float:
    current_y = 0.0
    for item in self._items:
      if not item.is_visible:
        item.rect = rl.Rectangle(container_rect.x, container_rect.y + current_y, container_rect.width, 0)
        continue

      content_width = item.get_content_width(int(container_rect.width - ITEM_PADDING * 2))
      item_height = item.get_item_height(self._font, content_width)
      item.rect = rl.Rectangle(container_rect.x, container_rect.y + current_y, container_rect.width, item_height)
      current_y += item_height
    return current_y  # total height of all items

  def _render_item(self, item: ListItem, y: int):
    content_x = item.rect.x + ITEM_PADDING
    text_x = content_x

    # Only draw title and icon for items that have them
    if item.title:
      # Draw icon if present
      if item.icon:
        icon_texture = gui_app.texture(os.path.join("icons", item.icon), ICON_SIZE, ICON_SIZE)
        rl.draw_texture(icon_texture, int(content_x), int(y + (ITEM_BASE_HEIGHT - icon_texture.width) // 2), rl.WHITE)
        text_x += ICON_SIZE + ITEM_PADDING

      # Draw main text
      text_size = measure_text_cached(self._font, item.title, ITEM_TEXT_FONT_SIZE)
      item_y = y + (ITEM_BASE_HEIGHT - text_size.y) // 2
      rl.draw_text_ex(self._font, item.title, rl.Vector2(text_x, item_y), ITEM_TEXT_FONT_SIZE, 0, ITEM_TEXT_COLOR)

    # Draw description if visible
    current_description = item.get_description()
    if item.description_visible and current_description and item._wrapped_description:
      rl.draw_text_ex(
        self._font,
        item._wrapped_description,
        rl.Vector2(text_x, y + ITEM_DESC_V_OFFSET),
        ITEM_DESC_FONT_SIZE,
        0,
        ITEM_DESC_TEXT_COLOR,
      )

    # Draw right item if present
    if item.action_item:
      right_rect = item.get_right_item_rect(item.rect)
      right_rect.y = y
      item.action_item.render(right_rect)

  def _handle_mouse_interaction(self, rect: rl.Rectangle, scroll_offset: rl.Vector2):
    mouse_pos = rl.get_mouse_position()

    self._hovered_item = -1
    if not rl.check_collision_point_rec(mouse_pos, rect):
      return

    content_mouse_y = mouse_pos.y - rect.y - scroll_offset.y

    for i, item in enumerate(self._items):
      if not item.is_visible:
        continue

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
      if item.action_item and item.rect:
        # Use the same coordinate system as in _render_item
        adjusted_rect = rl.Rectangle(item.rect.x, item.rect.y + scroll_offset.y, item.rect.width, item.rect.height)
        right_rect = item.get_right_item_rect(adjusted_rect)

        if rl.check_collision_point_rec(mouse_pos, right_rect):
          # Click was on right item, don't toggle description
          return

      # Toggle description visibility if item has description
      if item.description:
        item.description_visible = not item.description_visible


# Factory functions
def toggle_item(title: str, description: str | Callable[[], str] | None = None, initial_state: bool = False,
                callback: Callable | None = None, icon: str = "", enabled: bool | Callable[[], bool] = True,
                visible: bool | Callable[[], bool] = True) -> ListItem:
  action = ToggleAction(initial_state=initial_state, callback=callback, enabled=enabled)
  return ListItem(title=title, description=description, action_item=action, icon=icon, visible=visible)


def button_item(title: str, button_text: str | Callable[[], str], description: str | Callable[[], str] | None = None,
                callback: Callable | None = None, enabled: bool | Callable[[], bool] = True,
                visible: bool | Callable[[], bool] = True) -> ListItem:
  action = ButtonAction(text=button_text, callback=callback, enabled=enabled)
  return ListItem(title=title, description=description, action_item=action, visible=visible)


def text_item(title: str, value: str | Callable[[], str], description: str | Callable[[], str] | None = None,
              callback: Callable | None = None, enabled: bool | Callable[[], bool] = True,
              visible: bool | Callable[[], bool] = True) -> ListItem:
  action = TextAction(text=value, color=rl.Color(170, 170, 170, 255), enabled=enabled)
  return ListItem(title=title, description=description, action_item=action, visible=visible)


def dual_button_item(left_text: str, right_text: str, left_callback: Callable = None, right_callback: Callable = None,
                     description: str | Callable[[], str] | None = None, enabled: bool | Callable[[], bool] = True,
                     visible: bool | Callable[[], bool] = True) -> ListItem:
  action = DualButtonAction(left_text, right_text, left_callback, right_callback, enabled)
  return ListItem(title="", description=description, action_item=action, visible=visible)
