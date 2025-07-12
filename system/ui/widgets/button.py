import pyray as rl
from collections.abc import Callable
from enum import IntEnum
from typing import Self
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import MousePos, Widget
from openpilot.system.ui.widgets.text import Label


class ButtonStyle(IntEnum):
  NORMAL = 0  # Most common, neutral buttons
  PRIMARY = 1  # For main actions
  DANGER = 2  # For critical actions, like reboot or delete
  TRANSPARENT = 3  # For buttons with transparent background and border
  ACTION = 4
  LIST_ACTION = 5  # For list items with action buttons


class TextAlignment(IntEnum):
  LEFT = 0
  CENTER = 1
  RIGHT = 2


ICON_PADDING = 15
DEFAULT_BUTTON_FONT_SIZE = 60
BUTTON_DISABLED_TEXT_COLOR = rl.Color(228, 228, 228, 51)
ACTION_BUTTON_FONT_SIZE = 48

BUTTON_TEXT_COLOR = {
  ButtonStyle.NORMAL: rl.Color(228, 228, 228, 255),
  ButtonStyle.PRIMARY: rl.Color(228, 228, 228, 255),
  ButtonStyle.DANGER: rl.Color(228, 228, 228, 255),
  ButtonStyle.TRANSPARENT: rl.BLACK,
  ButtonStyle.ACTION: rl.Color(0, 0, 0, 255),
  ButtonStyle.LIST_ACTION: rl.Color(228, 228, 228, 255),
}

BUTTON_BACKGROUND_COLORS = {
  ButtonStyle.NORMAL: rl.Color(51, 51, 51, 255),
  ButtonStyle.PRIMARY: rl.Color(70, 91, 234, 255),
  ButtonStyle.DANGER: rl.Color(255, 36, 36, 255),
  ButtonStyle.TRANSPARENT: rl.BLACK,
  ButtonStyle.ACTION: rl.Color(189, 189, 189, 255),
  ButtonStyle.LIST_ACTION: rl.Color(57, 57, 57, 255),
}

BUTTON_PRESSED_BACKGROUND_COLORS = {
  ButtonStyle.NORMAL: rl.Color(74, 74, 74, 255),
  ButtonStyle.PRIMARY: rl.Color(48, 73, 244, 255),
  ButtonStyle.DANGER: rl.Color(255, 36, 36, 255),
  ButtonStyle.TRANSPARENT: rl.BLACK,
  ButtonStyle.ACTION: rl.Color(130, 130, 130, 255),
  ButtonStyle.LIST_ACTION: rl.Color(74, 74, 74, 74),
}

_pressed_buttons: set[str] = set()  # Track mouse press state globally


# TODO: This should be a Widget class

def gui_button(
  rect: rl.Rectangle,
  text: str,
  font_size: int = DEFAULT_BUTTON_FONT_SIZE,
  font_weight: FontWeight = FontWeight.MEDIUM,
  button_style: ButtonStyle = ButtonStyle.NORMAL,
  is_enabled: bool = True,
  border_radius: int = 10,  # Corner rounding in pixels
  text_alignment: TextAlignment = TextAlignment.CENTER,
  text_padding: int = 20,  # Padding for left/right alignment
  icon=None,
) -> int:
  button_id = f"{rect.x}_{rect.y}_{rect.width}_{rect.height}"
  result = 0

  if button_style in (ButtonStyle.PRIMARY, ButtonStyle.DANGER) and not is_enabled:
    button_style = ButtonStyle.NORMAL

  if button_style == ButtonStyle.ACTION and font_size == DEFAULT_BUTTON_FONT_SIZE:
    font_size = ACTION_BUTTON_FONT_SIZE

  # Set background color based on button type
  bg_color = BUTTON_BACKGROUND_COLORS[button_style]
  mouse_over = is_enabled and rl.check_collision_point_rec(rl.get_mouse_position(), rect)
  is_pressed = button_id in _pressed_buttons

  if mouse_over:
    if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
      # Only this button enters pressed state
      _pressed_buttons.add(button_id)
      is_pressed = True

    # Use pressed color when mouse is down over this button
    if is_pressed and rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT):
      bg_color = BUTTON_PRESSED_BACKGROUND_COLORS[button_style]

    # Handle button click
    if rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT) and is_pressed:
      result = 1
      _pressed_buttons.remove(button_id)

  # Clean up pressed state if mouse is released anywhere
  if rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT) and button_id in _pressed_buttons:
    _pressed_buttons.remove(button_id)

  # Draw the button with rounded corners
  roundness = border_radius / (min(rect.width, rect.height) / 2)
  if button_style != ButtonStyle.TRANSPARENT:
    rl.draw_rectangle_rounded(rect, roundness, 20, bg_color)
  else:
    rl.draw_rectangle_rounded(rect, roundness, 20, rl.BLACK)
    rl.draw_rectangle_rounded_lines_ex(rect, roundness, 20, 2, rl.WHITE)

  # Handle icon and text positioning
  font = gui_app.font(font_weight)
  text_size = measure_text_cached(font, text, font_size)
  text_pos = rl.Vector2(0, rect.y + (rect.height - text_size.y) // 2)  # Vertical centering

  # Draw icon if provided
  if icon:
    icon_y = rect.y + (rect.height - icon.height) / 2
    if text:
      if text_alignment == TextAlignment.LEFT:
        icon_x = rect.x + text_padding
        text_pos.x = icon_x + icon.width + ICON_PADDING
      elif text_alignment == TextAlignment.CENTER:
        total_width = icon.width + ICON_PADDING + text_size.x
        icon_x = rect.x + (rect.width - total_width) / 2
        text_pos.x = icon_x + icon.width + ICON_PADDING
      else:  # RIGHT
        text_pos.x = rect.x + rect.width - text_size.x - text_padding
        icon_x = text_pos.x - ICON_PADDING - icon.width
    else:
      # Center icon when no text
      icon_x = rect.x + (rect.width - icon.width) / 2

    rl.draw_texture_v(icon, rl.Vector2(icon_x, icon_y), rl.WHITE if is_enabled else rl.Color(255, 255, 255, 100))
  else:
    # No icon, position text normally
    if text_alignment == TextAlignment.LEFT:
      text_pos.x = rect.x + text_padding
    elif text_alignment == TextAlignment.CENTER:
      text_pos.x = rect.x + (rect.width - text_size.x) // 2
    elif text_alignment == TextAlignment.RIGHT:
      text_pos.x = rect.x + rect.width - text_size.x - text_padding

  # Draw the button text if any
  if text:
    color = BUTTON_TEXT_COLOR[button_style] if is_enabled else BUTTON_DISABLED_TEXT_COLOR
    rl.draw_text_ex(font, text, text_pos, font_size, 0, color)

  return result


# TODO: This could extend the Button class once it's added
class SelectionButton(Widget):
  """A button that can be selected, like a radio button.
  If `toggleable` is `True`, the button will be deselected when clicked again.
  If `on_select_callback` is provided, it will be called when the button is clicked (either selected or deselected).
  """

  MARGIN = 100

  def __init__(
    self,
    text: str,
    font_size: int = DEFAULT_BUTTON_FONT_SIZE,
    font_weight: FontWeight = FontWeight.MEDIUM,
    foreground_color: rl.Color = BUTTON_TEXT_COLOR[ButtonStyle.PRIMARY],
    is_selected: bool = False,
    on_select_callback: Callable[[Self], None] | None = None, # Called when the button is clicked (either selected or deselected)
    toggleable: bool = False, # If true, the button can be deselected when clicked again
  ):
    super().__init__()
    self.font_size = font_size
    self.font_weight = font_weight
    self.foreground_color = foreground_color
    self.is_selected = is_selected
    self.label = Label(text, font_size=font_size, font_weight=font_weight, color=self.foreground_color)
    self.check_icon = gui_app.texture("icons/circled_check.png", 100, 100)
    self.on_select_callback = on_select_callback
    self.toggleable = toggleable

  def set_selected(self, is_selected: bool):
    self.is_selected = is_selected

  def _render(self, rect: rl.Rectangle):
    # Render background, based on selected state
    rl.draw_rectangle_rounded(rect, 0.1, 10, BUTTON_BACKGROUND_COLORS[ButtonStyle.PRIMARY] if self.is_selected else rl.Color(79, 79, 79, 255))
    # Render label
    self.label.render(rl.Rectangle(rect.x + self.MARGIN, rect.y, rect.width - self.MARGIN * 2, rect.height))
    # Render checkmark, if selected
    if self.is_selected:
      checkmark_pos = rl.Vector2(rect.x + rect.width - self.MARGIN - self.check_icon.width, rect.y + rect.height / 2 - self.check_icon.height / 2)
      rl.draw_texture_v(self.check_icon, checkmark_pos, self.foreground_color)

  def _handle_mouse_release(self, mouse_pos: MousePos) -> bool:
    if self.toggleable and self.is_selected:
      self.set_selected(False)
    else:
      self.set_selected(True)
    if self.on_select_callback:
      self.on_select_callback(self)
    return True
