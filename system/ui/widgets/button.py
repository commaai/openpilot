from collections.abc import Callable
from enum import IntEnum

import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight, MousePos
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import TextAlignment, Label


class ButtonStyle(IntEnum):
  NORMAL = 0  # Most common, neutral buttons
  PRIMARY = 1  # For main actions
  DANGER = 2  # For critical actions, like reboot or delete
  TRANSPARENT = 3  # For buttons with transparent background and border
  ACTION = 4
  LIST_ACTION = 5  # For list items with action buttons
  NO_EFFECT = 6
  KEYBOARD = 7
  FORGET_WIFI = 8


ICON_PADDING = 15
DEFAULT_BUTTON_FONT_SIZE = 60
BUTTON_DISABLED_TEXT_COLOR = rl.Color(228, 228, 228, 51)
BUTTON_DISABLED_BACKGROUND_COLOR = rl.Color(51, 51, 51, 255)
ACTION_BUTTON_FONT_SIZE = 48

BUTTON_TEXT_COLOR = {
  ButtonStyle.NORMAL: rl.Color(228, 228, 228, 255),
  ButtonStyle.PRIMARY: rl.Color(228, 228, 228, 255),
  ButtonStyle.DANGER: rl.Color(228, 228, 228, 255),
  ButtonStyle.TRANSPARENT: rl.BLACK,
  ButtonStyle.ACTION: rl.Color(0, 0, 0, 255),
  ButtonStyle.LIST_ACTION: rl.Color(228, 228, 228, 255),
  ButtonStyle.NO_EFFECT: rl.Color(228, 228, 228, 255),
  ButtonStyle.KEYBOARD: rl.Color(221, 221, 221, 255),
  ButtonStyle.FORGET_WIFI: rl.Color(51, 51, 51, 255),
}

BUTTON_BACKGROUND_COLORS = {
  ButtonStyle.NORMAL: rl.Color(51, 51, 51, 255),
  ButtonStyle.PRIMARY: rl.Color(70, 91, 234, 255),
  ButtonStyle.DANGER: rl.Color(255, 36, 36, 255),
  ButtonStyle.TRANSPARENT: rl.BLACK,
  ButtonStyle.ACTION: rl.Color(189, 189, 189, 255),
  ButtonStyle.LIST_ACTION: rl.Color(57, 57, 57, 255),
  ButtonStyle.NO_EFFECT: rl.Color(51, 51, 51, 255),
  ButtonStyle.KEYBOARD: rl.Color(68, 68, 68, 255),
  ButtonStyle.FORGET_WIFI: rl.Color(189, 189, 189, 255),
}

BUTTON_PRESSED_BACKGROUND_COLORS = {
  ButtonStyle.NORMAL: rl.Color(74, 74, 74, 255),
  ButtonStyle.PRIMARY: rl.Color(48, 73, 244, 255),
  ButtonStyle.DANGER: rl.Color(255, 36, 36, 255),
  ButtonStyle.TRANSPARENT: rl.BLACK,
  ButtonStyle.ACTION: rl.Color(130, 130, 130, 255),
  ButtonStyle.LIST_ACTION: rl.Color(74, 74, 74, 74),
  ButtonStyle.NO_EFFECT: rl.Color(51, 51, 51, 255),
  ButtonStyle.KEYBOARD: rl.Color(51, 51, 51, 255),
  ButtonStyle.FORGET_WIFI: rl.Color(130, 130, 130, 255),
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


class Button(Widget):
  def __init__(self,
               text: str,
               click_callback: Callable[[], None] = None,
               font_size: int = DEFAULT_BUTTON_FONT_SIZE,
               font_weight: FontWeight = FontWeight.MEDIUM,
               button_style: ButtonStyle = ButtonStyle.NORMAL,
               border_radius: int = 10,
               text_alignment: TextAlignment = TextAlignment.CENTER,
               text_padding: int = 20,
               enabled: bool = True,
               icon = None,
               multi_touch: bool = False,
               ):

    super().__init__()
    self._button_style = button_style
    self._border_radius = border_radius
    self._background_color = BUTTON_BACKGROUND_COLORS[self._button_style]

    self._label = Label(text, font_size, font_weight, text_alignment, text_padding,
                        BUTTON_TEXT_COLOR[self._button_style], icon=icon)

    self._click_callback = click_callback
    self._multi_touch = multi_touch
    self.enabled = enabled

  def set_text(self, text):
    self._label.set_text(text)

  def _handle_mouse_release(self, mouse_pos: MousePos):
    if self._click_callback and self.enabled:
      self._click_callback()

  def _update_state(self):
    if self.enabled:
      self._label.set_text_color(BUTTON_TEXT_COLOR[self._button_style])
      if self.is_pressed:
        self._background_color = BUTTON_PRESSED_BACKGROUND_COLORS[self._button_style]
      else:
        self._background_color = BUTTON_BACKGROUND_COLORS[self._button_style]
    elif self._button_style != ButtonStyle.NO_EFFECT:
      self._background_color = BUTTON_DISABLED_BACKGROUND_COLOR
      self._label.set_text_color(BUTTON_DISABLED_TEXT_COLOR)

  def _render(self, _):
    roundness = self._border_radius / (min(self._rect.width, self._rect.height) / 2)
    rl.draw_rectangle_rounded(self._rect, roundness, 10, self._background_color)
    self._label.render(self._rect)


class ButtonRadio(Button):
  def __init__(self,
               text: str,
               icon,
               click_callback: Callable[[], None] = None,
               font_size: int = DEFAULT_BUTTON_FONT_SIZE,
               text_alignment: TextAlignment = TextAlignment.LEFT,
               border_radius: int = 10,
               text_padding: int = 20,
               ):

    super().__init__(text, click_callback=click_callback, font_size=font_size,
                     border_radius=border_radius, text_padding=text_padding,
                     text_alignment=text_alignment)
    self._text_padding = text_padding
    self._icon = icon
    self.selected = False

  def _handle_mouse_release(self, mouse_pos: MousePos):
    self.selected = not self.selected
    if self._click_callback:
      self._click_callback()

  def _update_state(self):
    if self.selected:
      self._background_color = BUTTON_BACKGROUND_COLORS[ButtonStyle.PRIMARY]
    else:
      self._background_color = BUTTON_BACKGROUND_COLORS[ButtonStyle.NORMAL]

  def _render(self, _):
    roundness = self._border_radius / (min(self._rect.width, self._rect.height) / 2)
    rl.draw_rectangle_rounded(self._rect, roundness, 10, self._background_color)
    self._label.render(self._rect)

    if self._icon and self.selected:
      icon_y = self._rect.y + (self._rect.height - self._icon.height) / 2
      icon_x = self._rect.x + self._rect.width - self._icon.width - self._text_padding - ICON_PADDING
      rl.draw_texture_v(self._icon, rl.Vector2(icon_x, icon_y), rl.WHITE if self.enabled else rl.Color(255, 255, 255, 100))
