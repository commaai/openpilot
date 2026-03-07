import abc
from collections.abc import Callable

import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight, GL_VERSION
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.common.filter_simple import FirstOrderFilter

SHIMMER_VERTEX_SHADER = GL_VERSION + """
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec4 vertexColor;
uniform mat4 mvp;
out vec2 fragTexCoord;
out vec4 fragColor;
void main() {
  fragTexCoord = vertexTexCoord;
  fragColor = vertexColor;
  gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""

SHIMMER_FRAGMENT_SHADER = GL_VERSION + """
in vec2 fragTexCoord;
in vec4 fragColor;
uniform sampler2D texture0;
uniform vec4 colDiffuse;
uniform float time;
uniform vec2 shimmerRange;
out vec4 finalColor;

void main() {
  vec4 texelColor = texture(texture0, fragTexCoord);
  finalColor = texelColor * colDiffuse * fragColor;

  // --- tweakable parameters ---
  const float BAND_WIDTH = 0.3;          // shimmer width as fraction of text width
  const float BLUR_RADIUS = 0.12;        // gaussian blur as fraction of text width (smaller = sharper)
  const float CYCLE_PERIOD = 2.5;        // seconds per full shimmer cycle
  const float SWEEP_FRACTION = 0.9;      // fraction of cycle spent sweeping (rest is a pause)
  const float LOWERED_OPACITY = 0.65;    // text opacity at rest, shimmer brings it to 1.0

  float range = shimmerRange.y - shimmerRange.x;
  float sigma = range * BLUR_RADIUS;

  float t = smoothstep(0.0, SWEEP_FRACTION, mod(time, CYCLE_PERIOD) / CYCLE_PERIOD);
  // Start one margin past the right edge, end one margin past the left edge
  float margin = range * BAND_WIDTH;
  float center = shimmerRange.y + margin - t * (range + 2.0 * margin);

  float d = gl_FragCoord.x - center;
  float shimmer = exp(-0.5 * d * d / (sigma * sigma));

  finalColor.a *= mix(LOWERED_OPACITY, 1.0, shimmer);
}
"""


class SliderBase(Widget, abc.ABC):
  HORIZONTAL_PADDING = 8
  CONFIRM_DELAY = 0.2

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

    self._is_dragging_circle = False

    self._label = UnifiedLabel(title, font_size=36, font_weight=FontWeight.SEMI_BOLD, text_color=rl.WHITE,
                               alignment=rl.GuiTextAlignment.TEXT_ALIGN_RIGHT,
                               alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE, line_height=0.9)

    # Shimmer shader (lazy init)
    self._shimmer_shader = None
    self._shimmer_time_loc = -1
    self._shimmer_range_loc = -1
    self._shimmer_time_ptr = rl.ffi.new("float[]", [0.0])
    self._shimmer_range_ptr = rl.ffi.new("float[]", [0.0, 0.0])
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

  def hide_event(self):
    super().hide_event()
    if self._shimmer_shader is not None:
      rl.unload_shader(self._shimmer_shader)
      self._shimmer_shader = None

  def reset(self, reset_shimmer: bool = True):
    # reset all slider state
    self._is_dragging_circle = False
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

  def _render_shimmer_label(self, label_rect: rl.Rectangle):
    # Shimmer shader for label
    if self._shimmer_shader is None:
      self._shimmer_shader = rl.load_shader_from_memory(SHIMMER_VERTEX_SHADER, SHIMMER_FRAGMENT_SHADER)
      self._shimmer_time_loc = rl.get_shader_location(self._shimmer_shader, "time")
      self._shimmer_range_loc = rl.get_shader_location(self._shimmer_shader, "shimmerRange")

    self._shimmer_time_ptr[0] = rl.get_time() - self._shimmer_start_time
    # use actual text width (right-aligned) instead of full rect
    text_right = label_rect.x + label_rect.width
    self._shimmer_range_ptr[0] = text_right - self._label.text_width
    self._shimmer_range_ptr[1] = text_right
    rl.set_shader_value(self._shimmer_shader, self._shimmer_time_loc, self._shimmer_time_ptr, rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
    rl.set_shader_value(self._shimmer_shader, self._shimmer_range_loc, self._shimmer_range_ptr, rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2)

    rl.begin_shader_mode(self._shimmer_shader)
    self._label.render(label_rect)
    rl.end_shader_mode()

  def _render(self, _):
    white = rl.Color(255, 255, 255, int(255 * self._opacity_filter.x))

    bg_txt_x = self._rect.x + (self._rect.width - self._bg_txt.width) / 2
    bg_txt_y = self._rect.y + (self._rect.height - self._bg_txt.height) / 2
    rl.draw_texture_ex(self._bg_txt, rl.Vector2(bg_txt_x, bg_txt_y), 0.0, 1.0, white)

    btn_x = bg_txt_x + self._bg_txt.width - self._circle_bg_txt.width + self._scroll_x_circle_filter.x
    btn_y = self._rect.y + (self._rect.height - self._circle_bg_txt.height) / 2

    label_alpha = int(255 * (1.0 - self.slider_percentage) * self._opacity_filter.x)
    if label_alpha > 0:
      self._label.set_text_color(rl.Color(255, 255, 255, label_alpha))
      label_rect = rl.Rectangle(
        self._rect.x + 20,
        self._rect.y,
        self._rect.width - self._circle_bg_txt.width - 20 * 2.5,
        self._rect.height,
      )
      self._render_shimmer_label(label_rect)

    # circle and arrow
    circle_bg_txt = self._circle_bg_pressed_txt if self._is_dragging_circle or self.confirmed else self._circle_bg_txt
    rl.draw_texture_ex(circle_bg_txt, rl.Vector2(btn_x, btn_y), 0.0, 1.0, white)

    arrow_x = btn_x + (self._circle_bg_txt.width - self._circle_arrow_txt.width) / 2
    arrow_y = btn_y + (self._circle_bg_txt.height - self._circle_arrow_txt.height) / 2
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
