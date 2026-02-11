import pyray as rl
from typing import Union
from enum import Enum
from collections.abc import Callable
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.scroller import DO_ZOOM
from openpilot.system.ui.lib.application import gui_app, FontWeight, MousePos
from openpilot.common.filter_simple import BounceFilter

try:
  from openpilot.common.params import Params
except ImportError:
  Params = None

SCROLLING_SPEED_PX_S = 50
COMPLICATION_SIZE    = 36
LABEL_COLOR          = rl.Color(255, 255, 255, int(255 * 0.9))
LABEL_HORIZONTAL_PADDING = 40
LABEL_VERTICAL_PADDING = 23  # visually matches 30 in figma
COMPLICATION_GREY    = rl.Color(0xAA, 0xAA, 0xAA, 255)
PRESSED_SCALE = 1.15 if DO_ZOOM else 1.07


class ScrollState(Enum):
  PRE_SCROLL = 0
  SCROLLING = 1
  POST_SCROLL = 2


class BigCircleButton(Widget):
  def __init__(self, icon: str, red: bool = False, icon_size: tuple[int, int] = (64, 53), icon_offset: tuple[int, int] = (0, 0)):
    super().__init__()
    self._red = red
    self._icon_offset = icon_offset

    # State
    self.set_rect(rl.Rectangle(0, 0, 180, 180))
    self._press_state_enabled = True
    self._scale_filter = BounceFilter(1.0, 0.1, 1 / gui_app.target_fps)

    # Icons
    self._txt_icon = gui_app.texture(icon, *icon_size)
    self._txt_btn_disabled_bg = gui_app.texture("icons_mici/buttons/button_circle_disabled.png", 180, 180)

    self._txt_btn_bg = gui_app.texture("icons_mici/buttons/button_circle.png", 180, 180)
    self._txt_btn_pressed_bg = gui_app.texture("icons_mici/buttons/button_circle_hover.png", 180, 180)

    self._txt_btn_red_bg = gui_app.texture("icons_mici/buttons/button_circle_red.png", 180, 180)
    self._txt_btn_red_pressed_bg = gui_app.texture("icons_mici/buttons/button_circle_red_hover.png", 180, 180)

  def set_enable_pressed_state(self, pressed: bool):
    self._press_state_enabled = pressed

  def _draw_content(self, btn_y: float):
    # draw icon
    icon_color = rl.WHITE if self.enabled else rl.Color(255, 255, 255, int(255 * 0.35))
    rl.draw_texture_ex(self._txt_icon, (self._rect.x + (self._rect.width - self._txt_icon.width) / 2 + self._icon_offset[0],
                                        btn_y + (self._rect.height - self._txt_icon.height) / 2 + self._icon_offset[1]), 0, 1.0, icon_color)

  def _render(self, _):
    # draw background
    txt_bg = self._txt_btn_bg if not self._red else self._txt_btn_red_bg
    if not self.enabled:
      txt_bg = self._txt_btn_disabled_bg
    elif self.is_pressed and self._press_state_enabled:
      txt_bg = self._txt_btn_pressed_bg if not self._red else self._txt_btn_red_pressed_bg

    scale = self._scale_filter.update(PRESSED_SCALE if self.is_pressed and self._press_state_enabled else 1.0)
    btn_x = self._rect.x + (self._rect.width * (1 - scale)) / 2
    btn_y = self._rect.y + (self._rect.height * (1 - scale)) / 2
    rl.draw_texture_ex(txt_bg, (btn_x, btn_y), 0, scale, rl.WHITE)

    self._draw_content(btn_y)


class BigCircleToggle(BigCircleButton):
  def __init__(self, icon: str, toggle_callback: Callable | None = None, icon_size: tuple[int, int] = (64, 53), icon_offset: tuple[int, int] = (0, 0)):
    super().__init__(icon, False, icon_size=icon_size, icon_offset=icon_offset)
    self._toggle_callback = toggle_callback

    # State
    self._checked = False

    # Icons
    self._txt_toggle_enabled = gui_app.texture("icons_mici/buttons/toggle_dot_enabled.png", 66, 66)
    self._txt_toggle_disabled = gui_app.texture("icons_mici/buttons/toggle_dot_disabled.png", 66, 66)

  def set_checked(self, checked: bool):
    self._checked = checked

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)

    self._checked = not self._checked
    if self._toggle_callback:
      self._toggle_callback(self._checked)

  def _draw_content(self, btn_y: float):
    super()._draw_content(btn_y)

    # draw status icon
    rl.draw_texture_ex(self._txt_toggle_enabled if self._checked else self._txt_toggle_disabled,
                       (self._rect.x + (self._rect.width - self._txt_toggle_enabled.width) / 2, btn_y + 5),
                       0, 1.0, rl.WHITE)


class BigButton(Widget):
  """A lightweight stand-in for the Qt BigButton, drawn & updated each frame."""

  def __init__(self, text: str, value: str = "", icon: Union[str, rl.Texture] = "", icon_size: tuple[int, int] = (64, 64),
               scroll: bool = False):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, 402, 180))
    self.text = text
    self.value = value
    self._icon_size = icon_size
    self._scroll = scroll
    self.set_icon(icon)

    self._scale_filter = BounceFilter(1.0, 0.1, 1 / gui_app.target_fps)

    self._rotate_icon_t: float | None = None

    self._label = UnifiedLabel(text, font_size=self._get_label_font_size(), font_weight=FontWeight.BOLD,
                               text_color=LABEL_COLOR, alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM, scroll=scroll,
                               line_height=0.9)
    self._sub_label = UnifiedLabel(value, font_size=COMPLICATION_SIZE, font_weight=FontWeight.ROMAN,
                                   text_color=COMPLICATION_GREY, alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM)
    self._update_label_layout()

    self._load_images()

  def set_icon(self, icon: Union[str, rl.Texture]):
    self._txt_icon = gui_app.texture(icon, *self._icon_size) if isinstance(icon, str) and len(icon) else icon

  def set_rotate_icon(self, rotate: bool):
    if rotate and self._rotate_icon_t is not None:
      return
    self._rotate_icon_t = rl.get_time() if rotate else None

  def _load_images(self):
    self._txt_default_bg = gui_app.texture("icons_mici/buttons/button_rectangle.png", 402, 180)
    self._txt_pressed_bg = gui_app.texture("icons_mici/buttons/button_rectangle_pressed.png", 402, 180)
    self._txt_disabled_bg = gui_app.texture("icons_mici/buttons/button_rectangle_disabled.png", 402, 180)
    self._txt_hover_bg = gui_app.texture("icons_mici/buttons/button_rectangle_hover.png", 402, 180)

  def _width_hint(self) -> int:
    # Single line if scrolling, so hide behind icon if exists
    icon_size = self._icon_size[0] if self._txt_icon and self._scroll and self.value else 0
    return int(self._rect.width - LABEL_HORIZONTAL_PADDING * 2 - icon_size)

  def _get_label_font_size(self):
    if len(self.text) <= 18:
      return 48
    else:
      return 42

  def _update_label_layout(self):
    self._label.set_font_size(self._get_label_font_size())
    if self.value:
      self._label.set_alignment_vertical(rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP)
    else:
      self._label.set_alignment_vertical(rl.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM)

  def set_text(self, text: str):
    self.text = text
    self._label.set_text(text)
    self._update_label_layout()

  def set_value(self, value: str):
    self.value = value
    self._sub_label.set_text(value)
    self._update_label_layout()

  def get_value(self) -> str:
    return self.value

  def get_text(self):
    return self.text

  def _draw_content(self, btn_y: float):
    # LABEL ------------------------------------------------------------------
    label_x = self._rect.x + LABEL_HORIZONTAL_PADDING

    label_color = LABEL_COLOR if self.enabled else rl.Color(255, 255, 255, int(255 * 0.35))
    self._label.set_color(label_color)
    label_rect = rl.Rectangle(label_x, btn_y + LABEL_VERTICAL_PADDING, self._width_hint(),
                              self._rect.height - LABEL_VERTICAL_PADDING * 2)
    self._label.render(label_rect)

    if self.value:
      label_y = btn_y + self._rect.height - LABEL_VERTICAL_PADDING
      sub_label_height = self._sub_label.get_content_height(self._width_hint())
      sub_label_rect = rl.Rectangle(label_x, label_y - sub_label_height, self._width_hint(), sub_label_height)
      self._sub_label.render(sub_label_rect)

    # ICON -------------------------------------------------------------------
    if self._txt_icon:
      rotation = 0
      if self._rotate_icon_t is not None:
        rotation = (rl.get_time() - self._rotate_icon_t) * 180

      # draw top right with 30px padding
      x = self._rect.x + self._rect.width - 30 - self._txt_icon.width / 2
      y = btn_y + 30 + self._txt_icon.height / 2
      source_rec = rl.Rectangle(0, 0, self._txt_icon.width, self._txt_icon.height)
      dest_rec = rl.Rectangle(x, y, self._txt_icon.width, self._txt_icon.height)
      origin = rl.Vector2(self._txt_icon.width / 2, self._txt_icon.height / 2)
      rl.draw_texture_pro(self._txt_icon, source_rec, dest_rec, origin, rotation, rl.WHITE)

  def _render(self, _):
    # draw _txt_default_bg
    txt_bg = self._txt_default_bg
    if not self.enabled:
      txt_bg = self._txt_disabled_bg
    elif self.is_pressed:
      txt_bg = self._txt_hover_bg

    scale = self._scale_filter.update(PRESSED_SCALE if self.is_pressed else 1.0)
    btn_x = self._rect.x + (self._rect.width * (1 - scale)) / 2
    btn_y = self._rect.y + (self._rect.height * (1 - scale)) / 2
    rl.draw_texture_ex(txt_bg, (btn_x, btn_y), 0, scale, rl.WHITE)

    self._draw_content(btn_y)


class BigToggle(BigButton):
  def __init__(self, text: str, value: str = "", initial_state: bool = False, toggle_callback: Callable | None = None):
    super().__init__(text, value, "")
    self._checked = initial_state
    self._toggle_callback = toggle_callback

  def _load_images(self):
    super()._load_images()
    self._txt_enabled_toggle = gui_app.texture("icons_mici/buttons/toggle_pill_enabled.png", 84, 66)
    self._txt_disabled_toggle = gui_app.texture("icons_mici/buttons/toggle_pill_disabled.png", 84, 66)

  def set_checked(self, checked: bool):
    self._checked = checked

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)
    self._checked = not self._checked
    if self._toggle_callback:
      self._toggle_callback(self._checked)

  def _draw_pill(self, x: float, y: float, checked: bool):
    # draw toggle icon top right
    if checked:
      rl.draw_texture_ex(self._txt_enabled_toggle, (x, y), 0, 1.0, rl.WHITE)
    else:
      rl.draw_texture_ex(self._txt_disabled_toggle, (x, y), 0, 1.0, rl.WHITE)

  def _draw_content(self, btn_y: float):
    super()._draw_content(btn_y)

    x = self._rect.x + self._rect.width - self._txt_enabled_toggle.width
    y = btn_y
    self._draw_pill(x, y, self._checked)


class BigMultiToggle(BigToggle):
  def __init__(self, text: str, options: list[str], toggle_callback: Callable | None = None,
               select_callback: Callable | None = None):
    super().__init__(text, "", toggle_callback=toggle_callback)
    assert len(options) > 0
    self._options = options
    self._select_callback = select_callback

    self.set_value(self._options[0])

  def _width_hint(self) -> int:
    return int(self._rect.width - LABEL_HORIZONTAL_PADDING * 2 - self._txt_enabled_toggle.width)

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)
    cur_idx = self._options.index(self.value)
    new_idx = (cur_idx + 1) % len(self._options)
    self.set_value(self._options[new_idx])
    if self._select_callback:
      self._select_callback(self.value)

  def _draw_content(self, btn_y: float):
    # don't draw pill from BigToggle
    BigButton._draw_content(self, btn_y)

    checked_idx = self._options.index(self.value)

    x = self._rect.x + self._rect.width - self._txt_enabled_toggle.width
    y = btn_y

    for i in range(len(self._options)):
      self._draw_pill(x, y, checked_idx == i)
      y += 35


class BigMultiParamToggle(BigMultiToggle):
  def __init__(self, text: str, param: str, options: list[str], toggle_callback: Callable | None = None,
               select_callback: Callable | None = None):
    super().__init__(text, options, toggle_callback, select_callback)
    self._param = param

    self._params = Params()
    self._load_value()

  def _load_value(self):
    self.set_value(self._options[self._params.get(self._param) or 0])

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)
    new_idx = self._options.index(self.value)
    self._params.put_nonblocking(self._param, new_idx)


class BigParamControl(BigToggle):
  def __init__(self, text: str, param: str, toggle_callback: Callable | None = None):
    super().__init__(text, "", toggle_callback=toggle_callback)
    self.param = param
    self.params = Params()
    self.set_checked(self.params.get_bool(self.param, False))

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)
    self.params.put_bool(self.param, self._checked)

  def refresh(self):
    self.set_checked(self.params.get_bool(self.param, False))


# TODO: param control base class
class BigCircleParamControl(BigCircleToggle):
  def __init__(self, icon: str, param: str, toggle_callback: Callable | None = None, icon_size: tuple[int, int] = (64, 53),
               icon_offset: tuple[int, int] = (0, 0)):
    super().__init__(icon, toggle_callback, icon_size=icon_size, icon_offset=icon_offset)
    self._param = param
    self.params = Params()
    self.set_checked(self.params.get_bool(self._param, False))

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)
    self.params.put_bool(self._param, self._checked)

  def refresh(self):
    self.set_checked(self.params.get_bool(self._param, False))
