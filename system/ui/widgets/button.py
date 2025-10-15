from collections.abc import Callable
from enum import IntEnum

import pyray as rl

from openpilot.system.ui.lib.application import FontWeight, MousePos
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import TextAlignment, Label


class ButtonStyle(IntEnum):
  NORMAL = 0  # Most common, neutral buttons
  PRIMARY = 1  # For main actions
  DANGER = 2  # For critical actions, like reboot or delete
  TRANSPARENT = 3  # For buttons with transparent background and border
  TRANSPARENT_WHITE_TEXT = 9  # For buttons with transparent background and border and white text
  TRANSPARENT_WHITE_BORDER = 10  # For buttons with transparent background and white border and text
  ACTION = 4
  LIST_ACTION = 5  # For list items with action buttons
  NO_EFFECT = 6
  KEYBOARD = 7
  FORGET_WIFI = 8


ICON_PADDING = 15
DEFAULT_BUTTON_FONT_SIZE = 60
ACTION_BUTTON_FONT_SIZE = 48

BUTTON_TEXT_COLOR = {
  ButtonStyle.NORMAL: rl.Color(228, 228, 228, 255),
  ButtonStyle.PRIMARY: rl.Color(228, 228, 228, 255),
  ButtonStyle.DANGER: rl.Color(228, 228, 228, 255),
  ButtonStyle.TRANSPARENT: rl.BLACK,
  ButtonStyle.TRANSPARENT_WHITE_TEXT: rl.WHITE,
  ButtonStyle.TRANSPARENT_WHITE_BORDER: rl.Color(228, 228, 228, 255),
  ButtonStyle.ACTION: rl.BLACK,
  ButtonStyle.LIST_ACTION: rl.Color(228, 228, 228, 255),
  ButtonStyle.NO_EFFECT: rl.Color(228, 228, 228, 255),
  ButtonStyle.KEYBOARD: rl.Color(221, 221, 221, 255),
  ButtonStyle.FORGET_WIFI: rl.Color(51, 51, 51, 255),
}

BUTTON_DISABLED_TEXT_COLORS = {
  ButtonStyle.TRANSPARENT_WHITE_TEXT: rl.WHITE,
}

BUTTON_BACKGROUND_COLORS = {
  ButtonStyle.NORMAL: rl.Color(51, 51, 51, 255),
  ButtonStyle.PRIMARY: rl.Color(70, 91, 234, 255),
  ButtonStyle.DANGER: rl.Color(255, 36, 36, 255),
  ButtonStyle.TRANSPARENT: rl.BLACK,
  ButtonStyle.TRANSPARENT_WHITE_TEXT: rl.BLANK,
  ButtonStyle.TRANSPARENT_WHITE_BORDER: rl.BLACK,
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
  ButtonStyle.TRANSPARENT_WHITE_TEXT: rl.BLANK,
  ButtonStyle.TRANSPARENT_WHITE_BORDER: rl.BLANK,
  ButtonStyle.ACTION: rl.Color(130, 130, 130, 255),
  ButtonStyle.LIST_ACTION: rl.Color(74, 74, 74, 74),
  ButtonStyle.NO_EFFECT: rl.Color(51, 51, 51, 255),
  ButtonStyle.KEYBOARD: rl.Color(51, 51, 51, 255),
  ButtonStyle.FORGET_WIFI: rl.Color(130, 130, 130, 255),
}

BUTTON_DISABLED_BACKGROUND_COLORS = {
  ButtonStyle.TRANSPARENT_WHITE_TEXT: rl.BLANK,
}


class Button(Widget):
  def __init__(self,
               text: str,
               click_callback: Callable[[], None] | None = None,
               font_size: int = DEFAULT_BUTTON_FONT_SIZE,
               font_weight: FontWeight = FontWeight.MEDIUM,
               button_style: ButtonStyle = ButtonStyle.NORMAL,
               border_radius: int = 10,
               text_alignment: TextAlignment = TextAlignment.CENTER,
               text_padding: int = 0,
               icon=None,
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

  def set_text(self, text):
    self._label.set_text(text)

  def set_button_style(self, button_style: ButtonStyle):
    self._button_style = button_style
    self._background_color = BUTTON_BACKGROUND_COLORS[self._button_style]
    self._label.set_text_color(BUTTON_TEXT_COLOR[self._button_style])

  def _update_state(self):
    if self.enabled:
      self._label.set_text_color(BUTTON_TEXT_COLOR[self._button_style])
      if self.is_pressed:
        self._background_color = BUTTON_PRESSED_BACKGROUND_COLORS[self._button_style]
      else:
        self._background_color = BUTTON_BACKGROUND_COLORS[self._button_style]
    elif self._button_style != ButtonStyle.NO_EFFECT:
      self._background_color = BUTTON_DISABLED_BACKGROUND_COLORS.get(self._button_style, rl.Color(51, 51, 51, 255))
      self._label.set_text_color(BUTTON_DISABLED_TEXT_COLORS.get(self._button_style, rl.Color(228, 228, 228, 51)))

  def _render(self, _):
    roundness = self._border_radius / (min(self._rect.width, self._rect.height) / 2)
    if self._button_style == ButtonStyle.TRANSPARENT_WHITE_BORDER:
      rl.draw_rectangle_rounded(self._rect, roundness, 10, rl.BLACK)
      rl.draw_rectangle_rounded_lines_ex(self._rect, roundness, 10, 2, rl.WHITE)
    else:
      rl.draw_rectangle_rounded(self._rect, roundness, 10, self._background_color)
    self._label.render(self._rect)


class ButtonRadio(Button):
  def __init__(self,
               text: str,
               icon,
               click_callback: Callable[[], None] | None = None,
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
    super()._handle_mouse_release(mouse_pos)
    self.selected = not self.selected

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
