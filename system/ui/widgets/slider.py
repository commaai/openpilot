from collections.abc import Callable

import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.common.filter_simple import FirstOrderFilter


class SmallSlider(Widget):
  HORIZONTAL_PADDING = 8
  CONFIRM_DELAY = 0.2

  def __init__(self, title: str, confirm_callback: Callable | None = None):
    # TODO: unify this with BigConfirmationDialogV2
    super().__init__()
    self._confirm_callback = confirm_callback

    self._font = gui_app.font(FontWeight.DISPLAY)

    self._load_assets()

    self._drag_threshold = -self._rect.width // 2

    # State
    self._opacity_filter = FirstOrderFilter(1.0, 0.1, 1 / gui_app.target_fps)
    self._confirmed_time = 0.0
    self._confirm_callback_called = False  # we keep dialog open by default, only call once
    self._start_x_circle = 0.0
    self._scroll_x_circle = 0.0
    self._scroll_x_circle_filter = FirstOrderFilter(0, 0.05, 1 / gui_app.target_fps)

    self._is_dragging_circle = False

    self._label = UnifiedLabel(title, font_size=36, font_weight=FontWeight.MEDIUM, text_color=rl.Color(255, 255, 255, int(255 * 0.65)),
                               alignment=rl.GuiTextAlignment.TEXT_ALIGN_RIGHT,
                               alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE, line_height=0.9)

  def _load_assets(self):
    self.set_rect(rl.Rectangle(0, 0, 316 + self.HORIZONTAL_PADDING * 2, 100))

    self._bg_txt = gui_app.texture("icons_mici/setup/small_slider/slider_bg.png", 316, 100)
    self._circle_bg_txt = gui_app.texture("icons_mici/setup/small_slider/slider_red_circle.png", 100, 100)
    self._circle_arrow_txt = gui_app.texture("icons_mici/setup/small_slider/slider_arrow.png", 37, 32)

  @property
  def confirmed(self) -> bool:
    return self._confirmed_time > 0.0

  def reset(self):
    # reset all slider state
    self._is_dragging_circle = False
    self._confirmed_time = 0.0
    self._confirm_callback_called = False

  def set_opacity(self, opacity: float, smooth: bool = False):
    if smooth:
      self._opacity_filter.update(opacity)
    else:
      self._opacity_filter.x = opacity

  @property
  def slider_percentage(self):
    activated_pos = -self._bg_txt.width + self._circle_bg_txt.width
    return min(max(-self._scroll_x_circle_filter.x / abs(activated_pos), 0.0), 1.0)

  def _on_confirm(self):
    if self._confirm_callback:
      self._confirm_callback()

  def _handle_mouse_event(self, mouse_event):
    super()._handle_mouse_event(mouse_event)

    if mouse_event.left_pressed:
      # touch rect goes to the padding
      circle_button_rect = rl.Rectangle(
        self._rect.x + (self._rect.width - self._circle_bg_txt.width) + self._scroll_x_circle_filter.x - self.HORIZONTAL_PADDING * 2,
        self._rect.y,
        self._circle_bg_txt.width + self.HORIZONTAL_PADDING * 2,
        self._rect.height,
      )
      if rl.check_collision_point_rec(mouse_event.pos, circle_button_rect):
        self._start_x_circle = mouse_event.pos.x
        self._is_dragging_circle = True

    elif mouse_event.left_released:
      # swiped to left
      if self._scroll_x_circle_filter.x < self._drag_threshold:
        self._confirmed_time = rl.get_time()

      self._is_dragging_circle = False

    if self._is_dragging_circle:
      self._scroll_x_circle = mouse_event.pos.x - self._start_x_circle

  def _update_state(self):
    super()._update_state()
    # TODO: this math can probably be cleaned up to remove duplicate stuff
    activated_pos = int(-self._bg_txt.width + self._circle_bg_txt.width)
    self._scroll_x_circle = max(min(self._scroll_x_circle, 0), activated_pos)

    if self._confirmed_time > 0:
      # swiped left to confirm
      self._scroll_x_circle_filter.update(activated_pos)

      # activate once animation completes, small threshold for small floats
      if self._scroll_x_circle_filter.x < (activated_pos + 1):
        if not self._confirm_callback_called and (rl.get_time() - self._confirmed_time) >= self.CONFIRM_DELAY:
          self._on_confirm()
          self._confirm_callback_called = True

    elif not self._is_dragging_circle:
      # reset back to right
      self._scroll_x_circle_filter.update(0)
    else:
      # not activated yet, keep movement 1:1
      self._scroll_x_circle_filter.x = self._scroll_x_circle

  def _render(self, _):
    # TODO: iOS text shimmering animation

    white = rl.Color(255, 255, 255, int(255 * self._opacity_filter.x))

    bg_txt_x = self._rect.x + (self._rect.width - self._bg_txt.width) / 2
    bg_txt_y = self._rect.y + (self._rect.height - self._bg_txt.height) / 2
    rl.draw_texture_ex(self._bg_txt, rl.Vector2(bg_txt_x, bg_txt_y), 0.0, 1.0, white)

    btn_x = bg_txt_x + self._bg_txt.width - self._circle_bg_txt.width + self._scroll_x_circle_filter.x
    btn_y = self._rect.y + (self._rect.height - self._circle_bg_txt.height) / 2

    if self._confirmed_time == 0.0 or self._scroll_x_circle > 0:
      self._label.set_text_color(rl.Color(255, 255, 255, int(255 * 0.65 * (1.0 - self.slider_percentage) * self._opacity_filter.x)))
      label_rect = rl.Rectangle(
        self._rect.x + 20,
        self._rect.y,
        self._rect.width - self._circle_bg_txt.width - 20 * 2.5,
        self._rect.height,
      )
      self._label.render(label_rect)

    # circle and arrow
    rl.draw_texture_ex(self._circle_bg_txt, rl.Vector2(btn_x, btn_y), 0.0, 1.0, white)

    arrow_x = btn_x + (self._circle_bg_txt.width - self._circle_arrow_txt.width) / 2
    arrow_y = btn_y + (self._circle_bg_txt.height - self._circle_arrow_txt.height) / 2
    rl.draw_texture_ex(self._circle_arrow_txt, rl.Vector2(arrow_x, arrow_y), 0.0, 1.0, white)


class LargerSlider(SmallSlider):
  def __init__(self, title: str, confirm_callback: Callable | None = None, green: bool = True):
    self._green = green
    super().__init__(title, confirm_callback=confirm_callback)

  def _load_assets(self):
    self.set_rect(rl.Rectangle(0, 0, 520 + self.HORIZONTAL_PADDING * 2, 115))

    self._bg_txt = gui_app.texture("icons_mici/setup/small_slider/slider_bg_larger.png", 520, 115)
    circle_fn = "slider_green_rounded_rectangle" if self._green else "slider_black_rounded_rectangle"
    self._circle_bg_txt = gui_app.texture(f"icons_mici/setup/small_slider/{circle_fn}.png", 180, 115)
    self._circle_arrow_txt = gui_app.texture("icons_mici/setup/small_slider/slider_arrow.png", 64, 55)


class BigSlider(SmallSlider):
  def __init__(self, title: str, icon: rl.Texture, confirm_callback: Callable | None = None):
    self._icon = icon
    super().__init__(title, confirm_callback=confirm_callback)
    self._label = UnifiedLabel(title, font_size=48, font_weight=FontWeight.DISPLAY, text_color=rl.Color(255, 255, 255, int(255 * 0.65)),
                               alignment=rl.GuiTextAlignment.TEXT_ALIGN_RIGHT, alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE,
                               line_height=0.875)

  def _load_assets(self):
    self.set_rect(rl.Rectangle(0, 0, 520 + self.HORIZONTAL_PADDING * 2, 180))

    self._bg_txt = gui_app.texture("icons_mici/buttons/slider_bg.png", 520, 180)
    self._circle_bg_txt = gui_app.texture("icons_mici/buttons/button_circle.png", 180, 180)
    self._circle_arrow_txt = self._icon


class RedBigSlider(BigSlider):
  def _load_assets(self):
    self.set_rect(rl.Rectangle(0, 0, 520 + self.HORIZONTAL_PADDING * 2, 180))

    self._bg_txt = gui_app.texture("icons_mici/buttons/slider_bg.png", 520, 180)
    self._circle_bg_txt = gui_app.texture("icons_mici/buttons/button_circle_red.png", 180, 180)
    self._circle_arrow_txt = self._icon
