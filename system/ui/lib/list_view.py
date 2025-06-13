import pyray as rl
from collections.abc import Callable
from abc import ABC, abstractmethod
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.lib.button import gui_button, ButtonStyle
from openpilot.system.ui.lib.toggle import Toggle, WIDTH as TOGGLE_WIDTH, HEIGHT as TOGGLE_HEIGHT
from openpilot.system.ui.lib.widget import Widget

LINE_PADDING = 40
ITEM_BASE_HEIGHT = 170
ITEM_PADDING = 20
ITEM_TEXT_FONT_SIZE = 50
ITEM_TEXT_COLOR = rl.WHITE
ITEM_DESC_TEXT_COLOR = rl.Color(128, 128, 128, 255)
ITEM_DESC_FONT_SIZE = 40
ITEM_DESC_V_OFFSET = 140
ICON_SIZE = 80
BUTTON_WIDTH = 250
BUTTON_HEIGHT = 100
BUTTON_FONT_SIZE = 35


# Type Aliases for Clarity
StrSrc = str | Callable[[], str] | None
BoolSrc = bool | Callable[[], bool]


def _get_value(value, default=""):
  if callable(value):
    return value()
  return value if value is not None else default


class ListItem(Widget, ABC):
  def __init__(self, title, description: StrSrc=None, enabled: BoolSrc=True, visible: BoolSrc=True, icon=None):
    super().__init__()
    self.title = title
    self._icon = icon
    self.description = description
    self.show_desc = False

    self._enabled_source = enabled
    self._visible_source = visible
    self._font = gui_app.font(FontWeight.NORMAL)

    # Cached properties for performance
    self._prev_max_width: int = 0
    self._wrapped_description: str | None = None
    self._prev_description: str | None = None
    self._description_height: float = 0

  @property
  def enabled(self):
    return _get_value(self._enabled_source, True)

  @property
  def is_visible(self):
    return _get_value(self._visible_source, True)

  def set_enabled(self, enabled: bool):
    self._enabled_source = enabled

  def get_desc(self):
    return _get_value(self.description, "")

  def set_icon(self, icon: str):
    self._icon = icon

  def set_desc(self, description: StrSrc):
    self.description = description
    current_description = self.get_desc()
    if current_description != self._prev_description:
      self._update_description_cache(self._prev_max_width, current_description)

  def _update_description_cache(self, max_width: int, current_description: str):
    """Update the cached description wrapping"""
    self._prev_max_width = max_width
    self._prev_description = current_description
    content_width = max_width - ITEM_PADDING * 2

    # Account for icon width
    if self._icon:
      content_width -= ICON_SIZE + ITEM_PADDING

    wrapped_lines = wrap_text(self._font, current_description, ITEM_DESC_FONT_SIZE, content_width)
    self._wrapped_description = "\n".join(wrapped_lines)
    self._description_height = len(wrapped_lines) * ITEM_DESC_FONT_SIZE + 10

  def _get_height(self, max_width: int) -> float:
    if not self.is_visible:
      return 0

    if not self.show_desc:
      return ITEM_BASE_HEIGHT

    current_description = self.get_desc()
    if not current_description:
      return ITEM_BASE_HEIGHT

    if current_description != self._prev_description or max_width != self._prev_max_width:
      self._update_description_cache(max_width, current_description)

    return ITEM_BASE_HEIGHT + self._description_height - (ITEM_BASE_HEIGHT - ITEM_DESC_V_OFFSET) + ITEM_PADDING

  def _render(self, rect: rl.Rectangle):
    # Handle click on title/description area for toggling description
    if self.description and rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT):
      mouse_pos = rl.get_mouse_position()

      text_area_width = rect.width - self.get_action_width() - ITEM_PADDING
      text_area = rl.Rectangle(rect.x, rect.y, text_area_width, rect.height)
      if rl.check_collision_point_rec(mouse_pos, text_area):
        self.show_desc = not self.show_desc

    # Render title and description
    x = rect.x + ITEM_PADDING

    # Draw icon if present
    if self._icon:
      icon_texture = gui_app.texture(f"icons/{self._icon}", ICON_SIZE, ICON_SIZE)
      rl.draw_texture(icon_texture, int(x), int(rect.y + (ITEM_BASE_HEIGHT - ICON_SIZE) // 2), rl.WHITE)
      x += ICON_SIZE + ITEM_PADDING

    text_size = measure_text_cached(self._font, self.title, ITEM_TEXT_FONT_SIZE)
    title_y = rect.y + (ITEM_BASE_HEIGHT - text_size.y) // 2
    rl.draw_text_ex(self._font, self.title, (x, title_y), ITEM_TEXT_FONT_SIZE, 0, ITEM_TEXT_COLOR)

    # Draw description if visible
    if self.show_desc and self._wrapped_description:
      rl.draw_text_ex(self._font, self._wrapped_description, (x, rect.y + ITEM_DESC_V_OFFSET),
                      ITEM_DESC_FONT_SIZE, 0, ITEM_DESC_TEXT_COLOR)

    # Render action if needed
    action_width = self.get_action_width()
    action_rect = rl.Rectangle(rect.x + rect.width - action_width, rect.y, action_width, ITEM_BASE_HEIGHT)
    self.render_action(action_rect)

  @abstractmethod
  def get_action_width(self) -> int:
    """Return the width needed for the action part (right side)"""

  @abstractmethod
  def render_action(self, rect: rl.Rectangle):
    """Render the action part"""


class ToggleItem(ListItem):
  def __init__(self, title: str, description: StrSrc = None, initial_state: bool=False, callback=None, active_icon=None, **kwargs):
    super().__init__(title, description, **kwargs)
    self.toggle = Toggle(initial_state=initial_state)
    self.callback = callback
    self._inactive_icon = kwargs.get('icon', None)
    self._active_icon = active_icon
    if self._active_icon and initial_state:
      self.set_icon(self._active_icon)

  def get_action_width(self) -> int:
    return TOGGLE_WIDTH

  def render_action(self, rect: rl.Rectangle):
    self.toggle.set_enabled(self.enabled)
    toggle_rect = rl.Rectangle(rect.x, rect.y + (rect.height - TOGGLE_HEIGHT) // 2,
                               TOGGLE_WIDTH, TOGGLE_HEIGHT)

    if self.toggle.render(toggle_rect):
      if self._active_icon and self._inactive_icon:
        self.set_icon(self._active_icon if self.toggle.get_state() else self._inactive_icon)

      if self.callback:
        self.callback(self)

  def set_state(self, state: bool):
    self.toggle.set_state(state)

  def get_state(self):
    return self.toggle.get_state()


class ButtonItem(ListItem):
  def __init__(self, title: str, button_text, description=None, callback=None, **kwargs):
    super().__init__(title, description, **kwargs)
    self._button_text_src = button_text
    self._callback = callback

  def get_button_text(self):
    return _get_value(self._button_text_src, "Error")

  def get_action_width(self) -> int:
    return BUTTON_WIDTH

  def render_action(self, rect: rl.Rectangle):
    button_rect = rl.Rectangle(rect.x, rect.y + (rect.height - BUTTON_HEIGHT) // 2, BUTTON_WIDTH, BUTTON_HEIGHT)
    if gui_button(button_rect, self.get_button_text(), border_radius=BUTTON_HEIGHT // 2,
                  font_size=BUTTON_FONT_SIZE, button_style=ButtonStyle.LIST_ACTION, is_enabled=self.enabled):
      if self._callback:
        self._callback()


class TextItem(ListItem):
  def __init__(self, title: str, value: str | Callable[[], str], **kwargs):
    super().__init__(title, **kwargs)
    self._value_src = value
    self.color = rl.Color(170, 170, 170, 255)

  def get_value(self):
    return _get_value(self._value_src, "")

  def get_action_width(self) -> int:
    return int(measure_text_cached(self._font, self.get_value(), ITEM_TEXT_FONT_SIZE).x + ITEM_PADDING)

  def render_action(self, rect: rl.Rectangle):
    value = self.get_value()
    text_size = measure_text_cached(self._font, value, ITEM_TEXT_FONT_SIZE)
    x = rect.x + (rect.width - text_size.x) // 2
    y = rect.y + (rect.height - text_size.y) // 2
    rl.draw_text_ex(self._font, value, rl.Vector2(x, y), ITEM_TEXT_FONT_SIZE, 0, self.color)


class DualButtonItem(Widget):
  def __init__(self, left_text: str, right_text: str, left_callback: Callable, right_callback: Callable):
    super().__init__()
    self.left_text = left_text
    self.right_text = right_text
    self.left_callback = left_callback
    self.right_callback = right_callback
    self._button_spacing = 30
    self._button_height = 120

  def _get_height(self, max_width: int) -> float:
    return ITEM_BASE_HEIGHT

  def _render(self, rect: rl.Rectangle):
    button_width = (rect.width - self._button_spacing) / 2
    button_y = rect.y + (rect.height - self._button_height) / 2

    left_rect = rl.Rectangle(rect.x, button_y, button_width, self._button_height)
    right_rect = rl.Rectangle(rect.x + button_width + self._button_spacing, button_y, button_width, self._button_height)

    left_clicked = gui_button(left_rect, self.left_text, button_style=ButtonStyle.LIST_ACTION)
    right_clicked = gui_button(right_rect, self.right_text, button_style=ButtonStyle.DANGER)

    if left_clicked and self.left_callback is not None:
      self.left_callback()
    if right_clicked and self.right_callback is not None:
      self.right_callback()


class MultipleButtonItem(ListItem):
  def __init__(self, title: str, description: str, buttons: list[str], button_width: int, selected_index: int = 0, callback: Callable = None, **kwargs):
    super().__init__(title, description, **kwargs)
    self.buttons = buttons
    self.button_width = button_width
    self.selected_index = selected_index
    self.callback = callback
    self._font = gui_app.font(FontWeight.MEDIUM)
    self._colors = {
      'normal': rl.Color(57, 57, 57, 255),  # Gray
      'hovered': rl.Color(74, 74, 74, 255),  # Dark gray
      'selected': rl.Color(51, 171, 76, 255),  # Green
      'disabled': rl.Color(153, 51, 171, 76),  # #9933Ab4C - Semi-transparent
      'text': rl.Color(228, 228, 228, 255),  # Light gray
      'text_disabled': rl.Color(51, 228, 228, 228),  # #33E4E4E4 - Semi-transparent
    }

  def get_action_width(self) -> int:
    return self.button_width * len(self.buttons) + (len(self.buttons) - 1) * 20

  def render_action(self, rect: rl.Rectangle) -> bool:
    spacing = 20
    button_y = rect.y + (rect.height - 100) / 2
    clicked = -1

    for i, text in enumerate(self.buttons):
      button_x = rect.x + i * (self.button_width + spacing)
      button_rect = rl.Rectangle(button_x, button_y, self.button_width, 100)

      # Check button state
      mouse_pos = rl.get_mouse_position()
      is_hovered = rl.check_collision_point_rec(mouse_pos, button_rect)
      is_pressed = is_hovered and rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT)
      is_selected = i == self.selected_index

      bg_color = (self._colors['disabled'] if not self.enabled and is_selected else
                  self._colors['selected'] if is_selected else
                  self._colors['hovered'] if is_pressed and self.enabled else
                  self._colors['normal'])
      text_color = self._colors['text_disabled'] if not self.enabled else self._colors['text']

      # Draw button
      rl.draw_rectangle_rounded(button_rect, 1.0, 20, bg_color)

      # Draw text
      text_size = measure_text_cached(self._font, text, 40)
      text_x = button_x + (self.button_width - text_size.x) / 2
      text_y = button_y + (100 - text_size.y) / 2
      rl.draw_text_ex(self._font, text, rl.Vector2(text_x, text_y), 40, 0, text_color)

      # Handle click only if enabled
      if self.enabled and is_hovered and rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT):
        clicked = i

    if clicked >= 0:
      self.selected_index = clicked
      if self.callback:
        self.callback(clicked)
      return True
    return False


class ListView(Widget):
  def __init__(self, items: list[ListItem]):
    super().__init__()
    self.items = items
    self.scroll_panel = GuiScrollPanel()

  def _render(self, rect: rl.Rectangle):
    total_height = sum(item._get_height(int(rect.width)) for item in self.items if item.is_visible)

    # Handle scrolling
    content_rect = rl.Rectangle(rect.x, rect.y, rect.width, total_height)
    scroll_offset = self.scroll_panel.handle_scroll(rect, content_rect)

    # Set scissor mode for clipping
    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))

    y = rect.y + scroll_offset.y
    for i, item in enumerate(self.items):
      if not item.is_visible:
        continue

      item_height = item._get_height(int(rect.width))

      # Skip if outside viewport
      if y + item_height < rect.y or y > rect.y + rect.height:
        y += item_height
        continue

      # Render item
      item.render(rl.Rectangle(rect.x, y, rect.width, item_height))

      # Draw separator line
      if i < len(self.items) - 1:
        line_y = int(y + item_height - 1)
        rl.draw_line(int(rect.x + ITEM_PADDING), line_y, int(rect.x + rect.width - ITEM_PADDING), line_y, rl.GRAY)

      y += item_height

    rl.end_scissor_mode()
