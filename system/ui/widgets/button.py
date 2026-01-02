from collections.abc import Callable
from enum import IntEnum

import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight, MousePos
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import Label, UnifiedLabel
from openpilot.common.filter_simple import FirstOrderFilter


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
  ButtonStyle.DANGER: rl.Color(226, 44, 44, 255),
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
               text: str | Callable[[], str],
               click_callback: Callable[[], None] | None = None,
               font_size: int = DEFAULT_BUTTON_FONT_SIZE,
               font_weight: FontWeight = FontWeight.MEDIUM,
               button_style: ButtonStyle = ButtonStyle.NORMAL,
               border_radius: int = 10,
               text_alignment: int = rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
               text_padding: int = 20,
               icon=None,
               elide_right: bool = False,
               multi_touch: bool = False,
               ):

    super().__init__()
    self._button_style = button_style
    self._border_radius = border_radius
    self._background_color = BUTTON_BACKGROUND_COLORS[self._button_style]

    self._label = Label(text, font_size, font_weight, text_alignment, text_padding=text_padding,
                        text_color=BUTTON_TEXT_COLOR[self._button_style], icon=icon, elide_right=elide_right)

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
               text_alignment: int = rl.GuiTextAlignment.TEXT_ALIGN_LEFT,
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


class IconButton(Widget):
  def __init__(self, texture: rl.Texture):
    super().__init__()
    self._texture = texture
    self._opacity_filter = FirstOrderFilter(1.0, 0.1, 1 / gui_app.target_fps)
    self.set_rect(rl.Rectangle(0, 0, self._texture.width, self._texture.height))

  def set_opacity(self, opacity: float, smooth: bool = False):
    if smooth:
      self._opacity_filter.update(opacity)
    else:
      self._opacity_filter.x = opacity

  def _render(self, rect: rl.Rectangle):
    color = rl.Color(180, 180, 180, int(150 * self._opacity_filter.x)) if self.is_pressed else rl.WHITE
    if not self.enabled:
      color = rl.Color(255, 255, 255, int(255 * 0.9 * 0.35 * self._opacity_filter.x))
    draw_x = rect.x + (rect.width - self._texture.width) / 2
    draw_y = rect.y + (rect.height - self._texture.height) / 2
    rl.draw_texture(self._texture, int(draw_x), int(draw_y), color)


class SmallCircleIconButton(Widget):
  def __init__(self, icon_txt: rl.Texture):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, 100, 100))
    self._opacity_filter = FirstOrderFilter(1.0, 0.1, 1 / gui_app.target_fps)
    self._icon_bg_txt = gui_app.texture("icons_mici/setup/small_button.png", 100, 100)
    self._icon_bg_pressed_txt = gui_app.texture("icons_mici/setup/small_button_pressed.png", 100, 100)
    self._icon_bg_disabled_txt = gui_app.texture("icons_mici/setup/small_button_disabled.png", 100, 100)
    self._icon_txt = icon_txt

  def set_opacity(self, opacity: float, smooth: bool = False):
    if smooth:
      self._opacity_filter.update(opacity)
    else:
      self._opacity_filter.x = opacity

  def _render(self, _):
    white = rl.Color(255, 255, 255, int(255 * self._opacity_filter.x))
    if not self.enabled:
      bg_txt = self._icon_bg_disabled_txt
      icon_white = rl.Color(255, 255, 255, int(white.a * 0.35))
    else:
      bg_txt = self._icon_bg_pressed_txt if self.is_pressed else self._icon_bg_txt
      icon_white = white

    rl.draw_texture(bg_txt, int(self.rect.x), int(self.rect.y), white)
    icon_x = self.rect.x + (self.rect.width - self._icon_txt.width) / 2
    icon_y = self.rect.y + (self.rect.height - self._icon_txt.height) / 2
    rl.draw_texture(self._icon_txt, int(icon_x), int(icon_y), icon_white)


class SmallButton(Widget):
  def __init__(self, text: str):
    super().__init__()
    self._opacity_filter = FirstOrderFilter(1.0, 0.1, 1 / gui_app.target_fps)

    self._load_assets()

    self._label = UnifiedLabel(text, 36, font_weight=FontWeight.MEDIUM,
                               text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                               alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
                               alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE)

    self._bg_disabled_txt = None

  def _load_assets(self):
    self.set_rect(rl.Rectangle(0, 0, 194, 100))
    self._bg_txt = gui_app.texture("icons_mici/setup/reset/small_button.png", 194, 100)
    self._bg_pressed_txt = gui_app.texture("icons_mici/setup/reset/small_button_pressed.png", 194, 100)

  def set_text(self, text: str):
    self._label.set_text(text)

  def set_opacity(self, opacity: float, smooth: bool = False):
    if smooth:
      self._opacity_filter.update(opacity)
    else:
      self._opacity_filter.x = opacity

  def _render(self, _):
    if not self.enabled and self._bg_disabled_txt is not None:
      rl.draw_texture(self._bg_disabled_txt, int(self.rect.x), int(self.rect.y), rl.Color(255, 255, 255, int(255 * self._opacity_filter.x)))
    elif self.is_pressed:
      rl.draw_texture(self._bg_pressed_txt, int(self.rect.x), int(self.rect.y), rl.Color(255, 255, 255, int(255 * self._opacity_filter.x)))
    else:
      rl.draw_texture(self._bg_txt, int(self.rect.x), int(self.rect.y), rl.Color(255, 255, 255, int(255 * self._opacity_filter.x)))

    opacity = 0.9 if self.enabled else 0.35
    self._label.set_color(rl.Color(255, 255, 255, int(255 * opacity * self._opacity_filter.x)))
    self._label.render(self._rect)


class SmallRedPillButton(SmallButton):
  def _load_assets(self):
    self.set_rect(rl.Rectangle(0, 0, 194, 100))
    self._bg_txt = gui_app.texture("icons_mici/setup/small_red_pill.png", 194, 100)
    self._bg_pressed_txt = gui_app.texture("icons_mici/setup/small_red_pill_pressed.png", 194, 100)


class SmallerRoundedButton(SmallButton):
  def _load_assets(self):
    self.set_rect(rl.Rectangle(0, 0, 150, 100))
    self._bg_txt = gui_app.texture("icons_mici/setup/smaller_button.png", 150, 100)
    self._bg_disabled_txt = gui_app.texture("icons_mici/setup/smaller_button_disabled.png", 150, 100)
    self._bg_pressed_txt = gui_app.texture("icons_mici/setup/smaller_button_pressed.png", 150, 100)


class WideRoundedButton(SmallButton):
  def _load_assets(self):
    self.set_rect(rl.Rectangle(0, 0, 316, 100))
    self._bg_txt = gui_app.texture("icons_mici/setup/medium_button_bg.png", 316, 100)
    self._bg_pressed_txt = gui_app.texture("icons_mici/setup/medium_button_pressed_bg.png", 316, 100)


class WidishRoundedButton(SmallButton):
  def _load_assets(self):
    self.set_rect(rl.Rectangle(0, 0, 250, 100))
    self._bg_txt = gui_app.texture("icons_mici/setup/widish_button.png", 250, 100)
    self._bg_pressed_txt = gui_app.texture("icons_mici/setup/widish_button_pressed.png", 250, 100)
    self._bg_disabled_txt = gui_app.texture("icons_mici/setup/widish_button_disabled.png", 250, 100)


class FullRoundedButton(SmallButton):
  def _load_assets(self):
    self.set_rect(rl.Rectangle(0, 0, 520, 100))
    self._bg_txt = gui_app.texture("icons_mici/setup/reset/wide_button.png", 520, 100)
    self._bg_pressed_txt = gui_app.texture("icons_mici/setup/reset/wide_button_pressed.png", 520, 100)
