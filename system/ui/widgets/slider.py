from collections.abc import Callable
import platform

import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.common.filter_simple import FirstOrderFilter

VERSION = """
#version 300 es
precision mediump float;
"""
if platform.system() == "Darwin":
  VERSION = """
    #version 330 core
  """

VERTEX_SHADER = VERSION + """
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
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

SHIMMER_FRAGMENT_SHADER = VERSION + """
in vec2 fragTexCoord;
in vec4 fragColor;
uniform sampler2D texture0;
uniform float time;
uniform float shimmerWidth;
uniform float shimmerSpeed;
out vec4 finalColor;

void main() {
  vec4 texColor = texture(texture0, fragTexCoord);

  float xPos = fragTexCoord.x;

  float shimmerPos = mod(time * shimmerSpeed, 1.0 + shimmerWidth);

  float distFromShimmer = abs(xPos - shimmerPos);

  float mask = 1.0 - smoothstep(0.0, shimmerWidth, distFromShimmer);

  vec3 shimmerColor = vec3(1.0, 1.0, 1.0);
  vec3 finalRGB = mix(texColor.rgb, shimmerColor, mask * 0.9);

  finalColor = vec4(finalRGB, texColor.a) * fragColor;
}
"""


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
    self._opacity = 1.0
    self._confirmed_time = 0.0
    self._confirm_callback_called = False  # we keep dialog open by default, only call once
    self._start_x_circle = 0.0
    self._scroll_x_circle = 0.0
    self._scroll_x_circle_filter = FirstOrderFilter(0, 0.05, 1 / gui_app.target_fps)

    self._is_dragging_circle = False

    self._label = UnifiedLabel(title, font_size=36, font_weight=FontWeight.MEDIUM, text_color=rl.Color(255, 255, 255, int(255 * 0.65)),
                               alignment=rl.GuiTextAlignment.TEXT_ALIGN_RIGHT,
                               alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE, line_height=0.9)

    self.shader = rl.load_shader_from_memory(VERTEX_SHADER, SHIMMER_FRAGMENT_SHADER)
    self._time_loc = rl.get_shader_location(self.shader, "time")
    self._shimmer_width_loc = rl.get_shader_location(self.shader, "shimmerWidth")
    self._shimmer_speed_loc = rl.get_shader_location(self.shader, "shimmerSpeed")
    self._mvp_loc = rl.get_shader_location(self.shader, "mvp")

    proj = rl.matrix_ortho(0, gui_app.width, gui_app.height, 0, -1, 1)
    rl.set_shader_value_matrix(self.shader, self._mvp_loc, proj)

    shimmer_width_val = rl.ffi.new("float[]", [0.3])  # Width of shimmer zone (0.0 to 1.0)
    shimmer_speed_val = rl.ffi.new("float[]", [0.65])  # Speed of animation (30% faster than 0.5)
    rl.set_shader_value(self.shader, self._shimmer_width_loc, shimmer_width_val, rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
    rl.set_shader_value(self.shader, self._shimmer_speed_loc, shimmer_speed_val, rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    self._text_render_texture: rl.RenderTexture | None = None
    self._text_render_texture_width = 0
    self._text_render_texture_height = 0
    self._last_text_color: rl.Color | None = None

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
    self._last_text_color = None  # Force texture re-render

  def set_opacity(self, opacity: float):
    self._opacity = opacity

  def _ensure_render_texture(self, width: int, height: int):
    if (self._text_render_texture is None or
        self._text_render_texture_width != width or
        self._text_render_texture_height != height):
      if self._text_render_texture is not None:
        rl.unload_render_texture(self._text_render_texture)

      self._text_render_texture = rl.load_render_texture(width, height)
      rl.set_texture_filter(self._text_render_texture.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
      self._text_render_texture_width = width
      self._text_render_texture_height = height

  def _render_text_to_texture(self, label_rect: rl.Rectangle, text_color: rl.Color):
    # Only re-render if color changed
    if (self._last_text_color is not None and
        self._last_text_color.r == text_color.r and
        self._last_text_color.g == text_color.g and
        self._last_text_color.b == text_color.b and
        self._last_text_color.a == text_color.a):
      return

    self._last_text_color = text_color
    width = int(label_rect.width)
    height = int(label_rect.height)
    self._ensure_render_texture(width, height)

    # Render to texture
    rl.begin_texture_mode(self._text_render_texture)
    rl.clear_background(rl.Color(0, 0, 0, 0))  # Transparent background
    self._label.render(rl.Rectangle(0, 0, width, height))
    rl.end_texture_mode()

  def close(self):
    if self._text_render_texture is not None:
      rl.unload_render_texture(self._text_render_texture)
      self._text_render_texture = None

    if self.shader and self.shader.id:
      rl.unload_shader(self.shader)
      self.shader = None

  def __del__(self):
    self.close()

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
    white = rl.Color(255, 255, 255, int(255 * self._opacity))

    bg_txt_x = self._rect.x + (self._rect.width - self._bg_txt.width) / 2
    bg_txt_y = self._rect.y + (self._rect.height - self._bg_txt.height) / 2
    rl.draw_texture_ex(self._bg_txt, rl.Vector2(bg_txt_x, bg_txt_y), 0.0, 1.0, white)

    btn_x = bg_txt_x + self._bg_txt.width - self._circle_bg_txt.width + self._scroll_x_circle_filter.x
    btn_y = self._rect.y + (self._rect.height - self._circle_bg_txt.height) / 2

    if not self._confirmed_time == 0 or self._scroll_x_circle > 0:
      text_color = rl.Color(255, 255, 255, int(255 * 0.65 * (1.0 - self.slider_percentage) * self._opacity))
      self._label.set_text_color(text_color)
      label_rect = rl.Rectangle(
        self._rect.x + 20,
        self._rect.y,
        self._rect.width - self._circle_bg_txt.width - 20 * 3,
        self._rect.height,
      )

      self._render_text_to_texture(label_rect, text_color)

      time_val = rl.ffi.new("float[]", [rl.get_time()])
      rl.set_shader_value(self.shader, self._time_loc, time_val, rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

      rl.begin_shader_mode(self.shader)
      src_rect = rl.Rectangle(0, 0, float(self._text_render_texture_width), -float(self._text_render_texture_height))
      rl.draw_texture_pro(self._text_render_texture.texture, src_rect, label_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)
      rl.end_shader_mode()

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
