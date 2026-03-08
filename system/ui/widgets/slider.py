import abc
from collections.abc import Callable

import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.common.filter_simple import FirstOrderFilter, BounceFilter


class SliderButton(Widget):
  PRESSED_SCALE = 1.07

  def __init__(self, bg_txt: rl.Texture, bg_pressed_txt: rl.Texture,
               icon_txt: rl.Texture, on_release: Callable[[], None] | None = None):
    super().__init__()
    self._bg_txt = bg_txt
    self._bg_pressed_txt = bg_pressed_txt
    self._icon_txt = icon_txt
    self._on_release = on_release

    self._scale_filter = BounceFilter(1.0, 0.1, 1 / gui_app.target_fps)
    self._press_time: float | None = None

    self.is_dragging = False
    self._start_x = 0.0
    self.scroll_x = 0.0

    self._confirmed = False
    self._opacity = 1.0

  @property
  def button_width(self):
    return self._bg_txt.width

  def set_confirmed(self, confirmed: bool):
    self._confirmed = confirmed

  def set_opacity(self, opacity: float):
    self._opacity = opacity

  def reset(self):
    self.is_dragging = False
    self._press_time = None
    self.scroll_x = 0.0
    self._confirmed = False

  def _handle_mouse_press(self, mouse_pos):
    self._start_x = mouse_pos.x
    self.is_dragging = True
    self._press_time = rl.get_time()

  def _handle_mouse_event(self, mouse_event):
    if mouse_event.left_released and self.is_dragging:
      self.is_dragging = False
      if self._on_release:
        self._on_release()
    if self.is_dragging:
      self.scroll_x = mouse_event.pos.x - self._start_x

  def _render(self, rect):
    white = rl.Color(255, 255, 255, int(255 * self._opacity))

    btn_x = rect.x
    btn_y = rect.y + (rect.height - self._bg_txt.height) / 2

    pressed = self.is_dragging or self._confirmed or \
              (self._press_time is not None and rl.get_time() - self._press_time < 0.075)
    bg_txt = self._bg_pressed_txt if pressed else self._bg_txt
    scale = self._scale_filter.update(self.PRESSED_SCALE if pressed else 1.0)
    scaled_btn_x = btn_x + (self._bg_txt.width * (1 - scale)) / 2
    scaled_btn_y = btn_y + (self._bg_txt.height * (1 - scale)) / 2
    rl.draw_texture_ex(bg_txt, rl.Vector2(scaled_btn_x, scaled_btn_y), 0.0, scale, white)

    icon_x = btn_x + (self._bg_txt.width - self._icon_txt.width) / 2
    icon_y = scaled_btn_y + (self._bg_txt.height - self._icon_txt.height) / 2
    rl.draw_texture_ex(self._icon_txt, rl.Vector2(icon_x, icon_y), 0.0, 1.0, white)


class BigSliderButton(SliderButton):
  def __init__(self, icon: rl.Texture, red: bool = False, **kwargs):
    suffix = "_red" if red else ""
    super().__init__(
      gui_app.texture(f"icons_mici/buttons/button_circle{suffix}.png", 180, 180),
      gui_app.texture(f"icons_mici/buttons/button_circle{suffix}_pressed.png", 180, 180),
      icon,
      **kwargs,
    )


class SliderBase(Widget, abc.ABC):
  HORIZONTAL_PADDING = 8
  CONFIRM_DELAY = 0.2

  _bg_txt: rl.Texture

  def __init__(self, title: str, confirm_callback: Callable | None = None):
    super().__init__()
    self._confirm_callback = confirm_callback

    self._load_assets()

    self._drag_threshold = -self._rect.width // 2

    # State
    self._opacity_filter = FirstOrderFilter(1.0, 0.1, 1 / gui_app.target_fps)
    self._confirmed_time = 0.0
    self._confirm_callback_called = False  # we keep dialog open by default, only call once
    self._scroll_x_circle_filter = FirstOrderFilter(0, 0.05, 1 / gui_app.target_fps)

    self._circle_button = self._child(self._create_circle_button(self._on_circle_release))

    self._label = UnifiedLabel(title, font_size=36, font_weight=FontWeight.SEMI_BOLD, text_color=rl.Color(255, 255, 255, int(255 * 0.65)),
                               alignment=rl.GuiTextAlignment.TEXT_ALIGN_RIGHT,
                               alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE, line_height=0.9)

  @abc.abstractmethod
  def _load_assets(self):
    ...

  @abc.abstractmethod
  def _create_circle_button(self, on_release: Callable) -> SliderButton:
    ...

  @property
  def confirmed(self) -> bool:
    return self._confirmed_time > 0.0

  def reset(self):
    self._circle_button.reset()
    self._confirmed_time = 0.0
    self._confirm_callback_called = False

  def set_opacity(self, opacity: float, smooth: bool = False):
    if smooth:
      self._opacity_filter.update(opacity)
    else:
      self._opacity_filter.x = opacity

  @property
  def slider_percentage(self):
    activated_pos = -self._bg_txt.width + self._circle_button.button_width
    return min(max(-self._scroll_x_circle_filter.x / abs(activated_pos), 0.0), 1.0)

  def _on_confirm(self):
    if self._confirm_callback:
      self._confirm_callback()

  def _on_circle_release(self):
    if self._scroll_x_circle_filter.x < self._drag_threshold:
      self._confirmed_time = rl.get_time()

  def _update_state(self):
    super()._update_state()
    # TODO: this math can probably be cleaned up to remove duplicate stuff
    activated_pos = int(-self._bg_txt.width + self._circle_button.button_width)
    self._circle_button.scroll_x = max(min(self._circle_button.scroll_x, 0), activated_pos)

    if self.confirmed:
      # swiped left to confirm
      self._scroll_x_circle_filter.update(activated_pos)

      # activate once animation completes, small threshold for small floats
      if self._scroll_x_circle_filter.x < (activated_pos + 1):
        if not self._confirm_callback_called and (rl.get_time() - self._confirmed_time) >= self.CONFIRM_DELAY:
          self._confirm_callback_called = True
          self._on_confirm()

    elif not self._circle_button.is_dragging:
      # reset back to right
      self._scroll_x_circle_filter.update(0)
    else:
      # not activated yet, keep movement 1:1
      self._scroll_x_circle_filter.x = self._circle_button.scroll_x

    self._circle_button.set_confirmed(self.confirmed)
    self._circle_button.set_opacity(self._opacity_filter.x)

  def _layout(self):
    bg_txt_x = self._rect.x + (self._rect.width - self._bg_txt.width) / 2
    btn_x = bg_txt_x + self._bg_txt.width - self._circle_button.button_width + self._scroll_x_circle_filter.x
    self._circle_button.set_rect(rl.Rectangle(
      btn_x,
      self._rect.y,
      self._circle_button.button_width,
      self._rect.height,
    ))

  def _render(self, _):
    # TODO: iOS text shimmering animation

    white = rl.Color(255, 255, 255, int(255 * self._opacity_filter.x))

    bg_txt_x = self._rect.x + (self._rect.width - self._bg_txt.width) / 2
    bg_txt_y = self._rect.y + (self._rect.height - self._bg_txt.height) / 2
    rl.draw_texture_ex(self._bg_txt, rl.Vector2(bg_txt_x, bg_txt_y), 0.0, 1.0, white)

    if not self.confirmed:
      self._label.set_text_color(rl.Color(255, 255, 255, int(255 * 0.65 * (1.0 - self.slider_percentage) * self._opacity_filter.x)))
      label_rect = rl.Rectangle(
        self._rect.x + 20,
        self._rect.y,
        self._rect.width - self._circle_button.button_width - 20 * 2.5,
        self._rect.height,
      )
      self._label.render(label_rect)

    self._circle_button.render()


class BigSlider(SliderBase):
  def __init__(self, title: str, icon: rl.Texture, confirm_callback: Callable | None = None, red: bool = False):
    self._icon = icon
    self._red = red
    super().__init__(title, confirm_callback=confirm_callback)
    self._label = UnifiedLabel(title, font_size=48, font_weight=FontWeight.DISPLAY, text_color=rl.Color(255, 255, 255, int(255 * 0.65)),
                               alignment=rl.GuiTextAlignment.TEXT_ALIGN_RIGHT, alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE,
                               line_height=0.875)

  def _load_assets(self):
    self.set_rect(rl.Rectangle(0, 0, 520 + self.HORIZONTAL_PADDING * 2, 180))
    self._bg_txt = gui_app.texture("icons_mici/buttons/slider_bg.png", 520, 180)

  def _create_circle_button(self, on_release):
    return BigSliderButton(icon=self._icon, red=self._red, on_release=on_release)
