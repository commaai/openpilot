import abc
import math
from collections.abc import Callable

import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.common.filter_simple import FirstOrderFilter, BounceFilter

# Shimmer parameters (matching the shader version)
SHIMMER_BAND_WIDTH = 0.3       # shimmer width as fraction of text width
SHIMMER_BLUR_RADIUS = 0.12     # gaussian blur as fraction of text width
SHIMMER_CYCLE_PERIOD = 2.5     # seconds per full shimmer cycle
SHIMMER_SWEEP_FRACTION = 0.9   # fraction of cycle spent sweeping (rest is pause)
SHIMMER_LOW_OPACITY = 0.65     # text opacity at rest, shimmer brings to 1.0


class SliderBase(Widget, abc.ABC):
  HORIZONTAL_PADDING = 8
  CONFIRM_DELAY = 0.2
  PRESSED_SCALE = 1.07

  _bg_txt: rl.Texture
  _circle_bg_txt: rl.Texture
  _circle_bg_pressed_txt: rl.Texture
  _circle_arrow_txt: rl.Texture

  def __init__(self, title: str, confirm_callback: Callable | None = None, shimmer_offset: float = 0.0):
    super().__init__()
    self._confirm_callback = confirm_callback

    self._load_assets()

    self._drag_threshold = -self._rect.width // 2

    # State
    self._opacity_filter = FirstOrderFilter(1.0, 0.1, 1 / gui_app.target_fps)
    self._confirmed_time = 0.0
    self._confirm_callback_called = False  # we keep dialog open by default, only call once
    self._start_x_circle = 0.0
    self._scroll_x_circle = 0.0
    self._scroll_x_circle_filter = FirstOrderFilter(0, 0.05, 1 / gui_app.target_fps)
    self._circle_scale_filter = BounceFilter(1.0, 0.1, 1 / gui_app.target_fps)
    self._circle_press_time: float | None = None

    self._is_dragging_circle = False

    self._label = UnifiedLabel(title, font_size=36, font_weight=FontWeight.SEMI_BOLD, text_color=rl.WHITE,
                               alignment=rl.GuiTextAlignment.TEXT_ALIGN_RIGHT,
                               alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE, line_height=0.9)

    # Shimmer state
    self._shimmer_offset = shimmer_offset
    self._shimmer_start_time = 0.0

  @abc.abstractmethod
  def _load_assets(self):
    ...

  @property
  def confirmed(self) -> bool:
    return self._confirmed_time > 0.0

  def show_event(self):
    super().show_event()
    self.reset()

  def reset(self, reset_shimmer: bool = True):
    # reset all slider state
    self._is_dragging_circle = False
    self._circle_press_time = None
    self._confirmed_time = 0.0
    self._confirm_callback_called = False
    if reset_shimmer:
      self._shimmer_start_time = rl.get_time() + self._shimmer_offset

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
        self._circle_press_time = rl.get_time()

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

    if self.confirmed:
      # swiped left to confirm
      self._scroll_x_circle_filter.update(activated_pos)

      # activate once animation completes, small threshold for small floats
      if self._scroll_x_circle_filter.x < (activated_pos + 1):
        if not self._confirm_callback_called and (rl.get_time() - self._confirmed_time) >= self.CONFIRM_DELAY:
          self._confirm_callback_called = True
          self._on_confirm()

    elif not self._is_dragging_circle:
      # reset back to right
      self._scroll_x_circle_filter.update(0)
    else:
      # not activated yet, keep movement 1:1
      self._scroll_x_circle_filter.x = self._scroll_x_circle

  def _compute_shimmer_alpha(self, char_x: float, text_left: float, text_width: float) -> float:
    """Compute per-character shimmer alpha based on x position."""
    if text_width <= 0:
      return SHIMMER_LOW_OPACITY

    elapsed = rl.get_time() - self._shimmer_start_time
    sigma = text_width * SHIMMER_BLUR_RADIUS

    # Smooth sweep progress with pause at end
    t_raw = (elapsed % SHIMMER_CYCLE_PERIOD) / SHIMMER_CYCLE_PERIOD
    # smoothstep for sweep fraction
    t_clamped = max(0.0, min(t_raw / SHIMMER_SWEEP_FRACTION, 1.0))
    t = t_clamped * t_clamped * (3.0 - 2.0 * t_clamped)

    # Sweep from right to left
    margin = text_width * SHIMMER_BAND_WIDTH
    text_right = text_left + text_width
    center = text_right + margin - t * (text_width + 2.0 * margin)

    d = char_x - center
    shimmer = math.exp(-0.5 * d * d / (sigma * sigma)) if sigma > 0 else 0.0

    return SHIMMER_LOW_OPACITY + (1.0 - SHIMMER_LOW_OPACITY) * shimmer

  def _render_shimmer_text(self, label_rect: rl.Rectangle, base_alpha: float):
    """Render text with per-character shimmer effect."""
    self._label._update_text_cache(int(label_rect.width))

    if not self._label._cached_wrapped_lines:
      return

    font = self._label._font
    font_size = self._label._font_size
    spacing = self._label._spacing_pixels

    # Compute total visible height for vertical centering
    sizes = self._label._cached_line_sizes
    total_height = 0.0
    for idx, size in enumerate(sizes):
      total_height += size.y if idx == 0 else size.y * self._label._line_height

    # Vertical alignment (middle)
    start_y = label_rect.y + (label_rect.height - total_height) / 2

    # Use widest line for shimmer range so sweep is even across all lines (right-aligned)
    text_width = self._label.text_width
    text_right = label_rect.x + label_rect.width - self._label._text_padding
    text_left = text_right - text_width
    current_y = start_y

    for idx, (line, size, emojis) in enumerate(zip(
        self._label._cached_wrapped_lines, sizes, self._label._cached_line_emojis, strict=True)):
      # Right-aligned
      line_x = text_right - size.x

      # Draw character by character
      cursor_x = line_x
      for ch in line:
        char_width = measure_text_cached(font, ch, font_size, spacing).x
        char_center_x = cursor_x + char_width / 2.0
        shimmer_a = self._compute_shimmer_alpha(char_center_x, text_left, text_width)
        alpha = int(255 * shimmer_a * base_alpha)
        color = rl.Color(255, 255, 255, alpha)
        rl.draw_text_ex(font, ch, rl.Vector2(cursor_x, current_y), font_size, 0, color)
        cursor_x += char_width + spacing

      if idx < len(sizes) - 1:
        current_y += size.y * self._label._line_height

  def _render(self, _):
    white = rl.Color(255, 255, 255, int(255 * self._opacity_filter.x))

    bg_txt_x = self._rect.x + (self._rect.width - self._bg_txt.width) / 2
    bg_txt_y = self._rect.y + (self._rect.height - self._bg_txt.height) / 2
    rl.draw_texture_ex(self._bg_txt, rl.Vector2(bg_txt_x, bg_txt_y), 0.0, 1.0, white)

    btn_x = bg_txt_x + self._bg_txt.width - self._circle_bg_txt.width + self._scroll_x_circle_filter.x
    btn_y = self._rect.y + (self._rect.height - self._circle_bg_txt.height) / 2

    label_alpha = (1.0 - self.slider_percentage) * self._opacity_filter.x
    if label_alpha > 0:
      label_rect = rl.Rectangle(
        self._rect.x + 20,
        self._rect.y,
        self._rect.width - self._circle_bg_txt.width - 20 * 2.5,
        self._rect.height,
      )
      self._render_shimmer_text(label_rect, label_alpha)

    # circle and arrow with grow animation
    circle_pressed = self._is_dragging_circle or self.confirmed or (self._circle_press_time is not None and rl.get_time() - self._circle_press_time < 0.075)
    circle_bg_txt = self._circle_bg_pressed_txt if circle_pressed else self._circle_bg_txt
    scale = self._circle_scale_filter.update(self.PRESSED_SCALE if circle_pressed else 1.0)
    scaled_btn_x = btn_x + (self._circle_bg_txt.width * (1 - scale)) / 2
    scaled_btn_y = btn_y + (self._circle_bg_txt.height * (1 - scale)) / 2
    rl.draw_texture_ex(circle_bg_txt, rl.Vector2(scaled_btn_x, scaled_btn_y), 0.0, scale, white)

    arrow_x = btn_x + (self._circle_bg_txt.width - self._circle_arrow_txt.width) / 2
    arrow_y = scaled_btn_y + (self._circle_bg_txt.height - self._circle_arrow_txt.height) / 2
    rl.draw_texture_ex(self._circle_arrow_txt, rl.Vector2(arrow_x, arrow_y), 0.0, 1.0, white)


class LargerSlider(SliderBase):
  def __init__(self, title: str, confirm_callback: Callable | None = None, green: bool = True, shimmer_offset: float = 0.0):
    self._green = green
    super().__init__(title, confirm_callback=confirm_callback, shimmer_offset=shimmer_offset)

  def _load_assets(self):
    self.set_rect(rl.Rectangle(0, 0, 520 + self.HORIZONTAL_PADDING * 2, 115))

    self._bg_txt = gui_app.texture("icons_mici/setup/small_slider/slider_bg_larger.png", 520, 115)
    circle_fn = "slider_green_rounded_rectangle" if self._green else "slider_black_rounded_rectangle"
    self._circle_bg_txt = gui_app.texture(f"icons_mici/setup/small_slider/{circle_fn}.png", 180, 115)
    self._circle_bg_pressed_txt = gui_app.texture(f"icons_mici/setup/small_slider/{circle_fn}_pressed.png", 180, 115)
    self._circle_arrow_txt = gui_app.texture("icons_mici/setup/small_slider/slider_arrow.png", 64, 55)


class BigSlider(SliderBase):
  def __init__(self, title: str, icon: rl.Texture, confirm_callback: Callable | None = None):
    self._icon = icon
    super().__init__(title, confirm_callback=confirm_callback)
    self._label = UnifiedLabel(title, font_size=48, font_weight=FontWeight.DISPLAY, text_color=rl.WHITE,
                               alignment=rl.GuiTextAlignment.TEXT_ALIGN_RIGHT, alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE,
                               line_height=0.875)

  def _load_assets(self):
    self.set_rect(rl.Rectangle(0, 0, 520 + self.HORIZONTAL_PADDING * 2, 180))

    self._bg_txt = gui_app.texture("icons_mici/buttons/slider_bg.png", 520, 180)
    self._circle_bg_txt = gui_app.texture("icons_mici/buttons/button_circle.png", 180, 180)
    self._circle_bg_pressed_txt = gui_app.texture("icons_mici/buttons/button_circle_pressed.png", 180, 180)
    self._circle_arrow_txt = self._icon


class RedBigSlider(BigSlider):
  def _load_assets(self):
    self.set_rect(rl.Rectangle(0, 0, 520 + self.HORIZONTAL_PADDING * 2, 180))

    self._bg_txt = gui_app.texture("icons_mici/buttons/slider_bg.png", 520, 180)
    self._circle_bg_txt = gui_app.texture("icons_mici/buttons/button_circle_red.png", 180, 180)
    self._circle_bg_pressed_txt = gui_app.texture("icons_mici/buttons/button_circle_red_pressed.png", 180, 180)
    self._circle_arrow_txt = self._icon
