import os
import pyray as rl
from collections.abc import Callable
from abc import ABC
from openpilot.system.ui.lib.application import gui_app, FontWeight, MousePos
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import gui_button, ButtonStyle
from openpilot.system.ui.widgets.toggle import Toggle, WIDTH as TOGGLE_WIDTH, HEIGHT as TOGGLE_HEIGHT

ITEM_BASE_WIDTH = 600
ITEM_BASE_HEIGHT = 170
ITEM_PADDING = 20
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
  def __init__(self, width: int = BUTTON_HEIGHT, enabled: bool | Callable[[], bool] = True):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, width, 0))
    self._enabled_source = enabled

  @property
  def enabled(self):
    return _resolve_value(self._enabled_source, False)


class ToggleAction(ItemAction):
  def __init__(self, initial_state: bool = False, width: int = TOGGLE_WIDTH, enabled: bool | Callable[[], bool] = True):
    super().__init__(width, enabled)
    self.toggle = Toggle(initial_state=initial_state)
    self.state = initial_state

  def set_touch_valid_callback(self, touch_callback: Callable[[], bool]) -> None:
    super().set_touch_valid_callback(touch_callback)
    self.toggle.set_touch_valid_callback(touch_callback)

  def _render(self, rect: rl.Rectangle) -> bool:
    self.toggle.set_enabled(self.enabled)
    self.toggle.render(rl.Rectangle(rect.x, rect.y + (rect.height - TOGGLE_HEIGHT) / 2, self._rect.width, TOGGLE_HEIGHT))
    return False

  def set_state(self, state: bool):
    self.state = state
    self.toggle.set_state(state)

  def get_state(self) -> bool:
    return self.state


class ButtonAction(ItemAction):
  def __init__(self, text: str | Callable[[], str], width: int = BUTTON_WIDTH, enabled: bool | Callable[[], bool] = True):
    super().__init__(width, enabled)
    self._text_source = text

  @property
  def text(self):
    return _resolve_value(self._text_source, "Error")

  def _render(self, rect: rl.Rectangle) -> bool:
    return gui_button(
      rl.Rectangle(rect.x, rect.y + (rect.height - BUTTON_HEIGHT) / 2, BUTTON_WIDTH, BUTTON_HEIGHT),
      self.text,
      border_radius=BUTTON_BORDER_RADIUS,
      font_weight=BUTTON_FONT_WEIGHT,
      font_size=BUTTON_FONT_SIZE,
      button_style=ButtonStyle.LIST_ACTION,
      is_enabled=self.enabled,
    ) == 1


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


class MultipleButtonAction(ItemAction):
  def __init__(self, buttons: list[str], button_width: int, selected_index: int = 0, callback: Callable = None):
    super().__init__(width=len(buttons) * (button_width + 20), enabled=True)
    self.buttons = buttons
    self.button_width = button_width
    self.selected_button = selected_index
    self.callback = callback
    self._font = gui_app.font(FontWeight.MEDIUM)

  def _render(self, rect: rl.Rectangle) -> bool:
    spacing = 20
    button_y = rect.y + (rect.height - BUTTON_HEIGHT) / 2
    clicked = -1

    for i, text in enumerate(self.buttons):
      button_x = rect.x + i * (self.button_width + spacing)
      button_rect = rl.Rectangle(button_x, button_y, self.button_width, BUTTON_HEIGHT)

      # Check button state
      mouse_pos = rl.get_mouse_position()
      is_hovered = rl.check_collision_point_rec(mouse_pos, button_rect)
      is_pressed = is_hovered and rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT) and self._is_pressed
      is_selected = i == self.selected_button

      # Button colors
      if is_selected:
        bg_color = rl.Color(51, 171, 76, 255)  # Green
      elif is_pressed:
        bg_color = rl.Color(74, 74, 74, 255)  # Dark gray
      else:
        bg_color = rl.Color(57, 57, 57, 255)  # Gray

      # Draw button
      rl.draw_rectangle_rounded(button_rect, 1.0, 20, bg_color)

      # Draw text
      text_size = measure_text_cached(self._font, text, 40)
      text_x = button_x + (self.button_width - text_size.x) / 2
      text_y = button_y + (BUTTON_HEIGHT - text_size.y) / 2
      rl.draw_text_ex(self._font, text, rl.Vector2(text_x, text_y), 40, 0, rl.Color(228, 228, 228, 255))

      # Handle click
      if is_hovered and rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT) and self._is_pressed:
        clicked = i

    if clicked >= 0:
      self.selected_button = clicked
      if self.callback:
        self.callback(clicked)
      return True
    return False


class ListItem(Widget):
  def __init__(self, title: str = "", icon: str | None = None, description: str | Callable[[], str] | None = None,
               description_visible: bool = False, callback: Callable | None = None,
               action_item: ItemAction | None = None):
    super().__init__()
    self.title = title
    self.icon = icon
    self.description = description
    self.description_visible = description_visible
    self.callback = callback
    self.action_item = action_item

    self.set_rect(rl.Rectangle(0, 0, ITEM_BASE_WIDTH, ITEM_BASE_HEIGHT))
    self._font = gui_app.font(FontWeight.NORMAL)

    # Cached properties for performance
    self._prev_max_width: int = 0
    self._wrapped_description: str | None = None
    self._prev_description: str | None = None
    self._description_height: float = 0

  def set_touch_valid_callback(self, touch_callback: Callable[[], bool]) -> None:
    super().set_touch_valid_callback(touch_callback)
    if self.action_item:
      self.action_item.set_touch_valid_callback(touch_callback)

  def set_parent_rect(self, parent_rect: rl.Rectangle):
    super().set_parent_rect(parent_rect)
    self._rect.width = parent_rect.width

  def _handle_mouse_release(self, mouse_pos: MousePos):
    if not self.is_visible:
      return

    # Check not in action rect
    if self.action_item:
      action_rect = self.get_right_item_rect(self._rect)
      if rl.check_collision_point_rec(mouse_pos, action_rect):
        # Click was on right item, don't toggle description
        return

    if self.description:
      self.description_visible = not self.description_visible
      content_width = self.get_content_width(int(self._rect.width - ITEM_PADDING * 2))
      self._rect.height = self.get_item_height(self._font, content_width)

  def _render(self, _):
    if not self.is_visible:
      return

    # Don't draw items that are not in parent's viewport
    if ((self._rect.y + self.rect.height) <= self._parent_rect.y or
      self._rect.y >= (self._parent_rect.y + self._parent_rect.height)):
      return

    content_x = self._rect.x + ITEM_PADDING
    text_x = content_x

    # Only draw title and icon for items that have them
    if self.title:
      # Draw icon if present
      if self.icon:
        icon_texture = gui_app.texture(os.path.join("icons", self.icon), ICON_SIZE, ICON_SIZE)
        rl.draw_texture(icon_texture, int(content_x), int(self._rect.y + (ITEM_BASE_HEIGHT - icon_texture.width) // 2), rl.WHITE)
        text_x += ICON_SIZE + ITEM_PADDING

      # Draw main text
      text_size = measure_text_cached(self._font, self.title, ITEM_TEXT_FONT_SIZE)
      item_y = self._rect.y + (ITEM_BASE_HEIGHT - text_size.y) // 2
      rl.draw_text_ex(self._font, self.title, rl.Vector2(text_x, item_y), ITEM_TEXT_FONT_SIZE, 0, ITEM_TEXT_COLOR)

    # Draw description if visible
    current_description = self.get_description()
    if self.description_visible and current_description and self._wrapped_description:
      rl.draw_text_ex(
        self._font,
        self._wrapped_description,
        rl.Vector2(text_x, self._rect.y + ITEM_DESC_V_OFFSET),
        ITEM_DESC_FONT_SIZE,
        0,
        ITEM_DESC_TEXT_COLOR,
      )

    # Draw right item if present
    if self.action_item:
      right_rect = self.get_right_item_rect(self._rect)
      right_rect.y = self._rect.y
      if self.action_item.render(right_rect) and self.action_item.enabled:
        # Right item was clicked/activated
        if self.callback:
          self.callback()

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
    if self.action_item and self.action_item.rect.width > 0:
      return total_width - int(self.action_item.rect.width) - RIGHT_ITEM_PADDING
    return total_width

  def get_right_item_rect(self, item_rect: rl.Rectangle) -> rl.Rectangle:
    if not self.action_item:
      return rl.Rectangle(0, 0, 0, 0)

    right_width = self.action_item.rect.width
    if right_width == 0:  # Full width action (like DualButtonAction)
      return rl.Rectangle(item_rect.x + ITEM_PADDING, item_rect.y,
                          item_rect.width - (ITEM_PADDING * 2), ITEM_BASE_HEIGHT)

    right_x = item_rect.x + item_rect.width - right_width
    right_y = item_rect.y
    return rl.Rectangle(right_x, right_y, right_width, ITEM_BASE_HEIGHT)


# Factory functions
def simple_item(title: str, callback: Callable | None = None) -> ListItem:
  return ListItem(title=title, callback=callback)


def toggle_item(title: str, description: str | Callable[[], str] | None = None, initial_state: bool = False,
                callback: Callable | None = None, icon: str = "", enabled: bool | Callable[[], bool] = True) -> ListItem:
  action = ToggleAction(initial_state=initial_state, enabled=enabled)
  return ListItem(title=title, description=description, action_item=action, icon=icon, callback=callback)


def button_item(title: str, button_text: str | Callable[[], str], description: str | Callable[[], str] | None = None,
                callback: Callable | None = None, enabled: bool | Callable[[], bool] = True) -> ListItem:
  action = ButtonAction(text=button_text, enabled=enabled)
  return ListItem(title=title, description=description, action_item=action, callback=callback)


def text_item(title: str, value: str | Callable[[], str], description: str | Callable[[], str] | None = None,
              callback: Callable | None = None, enabled: bool | Callable[[], bool] = True) -> ListItem:
  action = TextAction(text=value, color=rl.Color(170, 170, 170, 255), enabled=enabled)
  return ListItem(title=title, description=description, action_item=action, callback=callback)


def dual_button_item(left_text: str, right_text: str, left_callback: Callable = None, right_callback: Callable = None,
                     description: str | Callable[[], str] | None = None, enabled: bool | Callable[[], bool] = True) -> ListItem:
  action = DualButtonAction(left_text, right_text, left_callback, right_callback, enabled)
  return ListItem(title="", description=description, action_item=action)


def multiple_button_item(title: str, description: str, buttons: list[str], selected_index: int,
                         button_width: int = BUTTON_WIDTH, callback: Callable = None, icon: str = ""):
  action = MultipleButtonAction(buttons, button_width, selected_index, callback=callback)
  return ListItem(title=title, description=description, icon=icon, action_item=action)
