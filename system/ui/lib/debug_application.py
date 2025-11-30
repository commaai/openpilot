import cProfile
import io
import os
import pstats
import sys
import time
import pyray as rl
from typing import NamedTuple
from collections import deque
from contextlib import contextmanager
from openpilot.system.hardware import PC
from openpilot.system.ui.lib.application import GuiApplication, MouseState
from openpilot.system.ui.lib.application import MOUSE_THREAD_RATE, GL_VERSION, _DEFAULT_FPS

TOUCH_HISTORY_TIMEOUT = 3.0  # Seconds before touch points fade out

SCALE = float(os.getenv("SCALE", "1.0"))
GRID_SIZE = int(os.getenv("GRID", "0"))
PROFILE_RENDER = int(os.getenv("PROFILE_RENDER", "0"))
PROFILE_STATS = int(os.getenv("PROFILE_STATS", "100"))  # Number of functions to show in profile output
SHOW_FPS = os.getenv("SHOW_FPS") == "1"
SHOW_TOUCHES = os.getenv("SHOW_TOUCHES") == "1"


BURN_IN_MODE = "BURN_IN" in os.environ
BURN_IN_VERTEX_SHADER = (
  GL_VERSION
  + """
in vec3 vertexPosition;
in vec2 vertexTexCoord;
uniform mat4 mvp;
out vec2 fragTexCoord;
void main() {
  fragTexCoord = vertexTexCoord;
  gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""
)
BURN_IN_FRAGMENT_SHADER = (
  GL_VERSION
  + """
in vec2 fragTexCoord;
uniform sampler2D texture0;
out vec4 fragColor;
void main() {
  vec4 sampled = texture(texture0, fragTexCoord);
  float intensity = sampled.b;
  // Map blue intensity to green -> yellow -> red to highlight burn-in risk.
  vec3 start = vec3(0.0, 1.0, 0.0);
  vec3 middle = vec3(1.0, 1.0, 0.0);
  vec3 end = vec3(1.0, 0.0, 0.0);
  vec3 gradient = mix(start, middle, clamp(intensity * 2.0, 0.0, 1.0));
  gradient = mix(gradient, end, clamp((intensity - 0.5) * 2.0, 0.0, 1.0));
  fragColor = vec4(gradient, sampled.a);
}
"""
)


class MousePosWithTime(NamedTuple):
  x: float
  y: float
  t: float


class DebugGuiApplication(GuiApplication):
  def __init__(self, width: int, height: int):
    super().__init__(width, height)

    if PC and os.getenv("SCALE") is None:
      self._scale = self._calculate_auto_scale()
    else:
      self._scale = SCALE

    self._scaled_width = int(self._width * self._scale)
    self._scaled_height = int(self._height * self._scale)

    self._burn_in_shader: rl.Shader | None = None

    # reinitialize mouse with the correct scale for debug UI
    self._mouse = MouseState(self._scale)
    self._render_texture: rl.RenderTexture | None = None

    # debug-specific settings from envs
    self._mouse_history: deque[MousePosWithTime] = deque(maxlen=MOUSE_THREAD_RATE)
    self._show_touches = SHOW_TOUCHES
    self._show_fps = SHOW_FPS
    self._grid_size = GRID_SIZE
    self._profile_render_frames = PROFILE_RENDER
    self._render_profiler: cProfile.Profile | None = None
    self._render_profile_start_time: float | None = None

  def init_window(self, title: str, fps: int = _DEFAULT_FPS):
    with self._startup_profile_context():
      super().init_window(title, fps)

      if BURN_IN_MODE and self._burn_in_shader is None:
        self._burn_in_shader = rl.load_shader_from_memory(BURN_IN_VERTEX_SHADER, BURN_IN_FRAGMENT_SHADER)

      needs_render_texture = self._scale != 1.0 or BURN_IN_MODE
      if needs_render_texture:
        self._render_texture = rl.load_render_texture(self._width, self._height)
        rl.set_texture_filter(self._render_texture.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)

    if self._profile_render_frames > 0:
      self._render_profiler = cProfile.Profile()
      self._render_profile_start_time = time.monotonic()
      self._render_profiler.enable()

  def close(self):
    if self._render_texture is not None:
      rl.unload_render_texture(self._render_texture)
      self._render_texture = None

    if self._burn_in_shader:
      rl.unload_shader(self._burn_in_shader)
      self._burn_in_shader = None

    super().close()

  def set_show_touches(self, show: bool):
    self._show_touches = show

  def set_show_fps(self, show: bool):
    self._show_fps = show

  def _begin_frame(self):
    if self._render_texture:
      rl.begin_texture_mode(self._render_texture)
    else:
      rl.begin_drawing()

    rl.clear_background(rl.BLACK)

  def _end_frame(self):
    if self._render_texture:
      rl.end_texture_mode()
      rl.begin_drawing()
      rl.clear_background(rl.BLACK)
      src_rect = rl.Rectangle(0, 0, float(self._width), -float(self._height))
      dst_rect = rl.Rectangle(0, 0, float(self._scaled_width), float(self._scaled_height))
      texture = self._render_texture.texture
      if texture:
        if BURN_IN_MODE and self._burn_in_shader:
          rl.begin_shader_mode(self._burn_in_shader)
          rl.draw_texture_pro(texture, src_rect, dst_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)
          rl.end_shader_mode()
        else:
          rl.draw_texture_pro(texture, src_rect, dst_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)

    if self._show_fps:
      rl.draw_fps(10, 10)

    if self._show_touches:
      self._draw_touch_points()

    if self._grid_size > 0:
      self._draw_grid()

    rl.end_drawing()

    if self._profile_render_frames > 0 and self._frame >= self._profile_render_frames:
      self._output_render_profile()

  def _draw_touch_points(self):
    current_time = time.monotonic()

    for mouse_event in self._mouse_events:
      if mouse_event.left_pressed:
        self._mouse_history.clear()
      self._mouse_history.append(
        MousePosWithTime(mouse_event.pos.x * self._scale, mouse_event.pos.y * self._scale, current_time)
      )

    # Remove old touch points that exceed the timeout
    while self._mouse_history and (current_time - self._mouse_history[0].t) > TOUCH_HISTORY_TIMEOUT:
      self._mouse_history.popleft()

    if self._mouse_history:
      mouse_pos = self._mouse_history[-1]
      rl.draw_circle(int(mouse_pos.x), int(mouse_pos.y), 15, rl.RED)
      for idx, mouse_pos in enumerate(self._mouse_history):
        perc = idx / len(self._mouse_history)
        color = rl.Color(min(int(255 * (1.5 - perc)), 255), int(min(255 * (perc + 0.5), 255)), 50, 255)
        rl.draw_circle(int(mouse_pos.x), int(mouse_pos.y), 5, color)

  def _draw_grid(self):
    grid_color = rl.Color(60, 60, 60, 255)
    # Draw vertical lines
    x = 0
    while x <= self._scaled_width:
      rl.draw_line(x, 0, x, self._scaled_height, grid_color)
      x += self._grid_size
    # Draw horizontal lines
    y = 0
    while y <= self._scaled_height:
      rl.draw_line(0, y, self._scaled_width, y, grid_color)
      y += self._grid_size

  def _output_render_profile(self):
    self._render_profiler.disable()
    elapsed_ms = (time.monotonic() - self._render_profile_start_time) * 1e3
    avg_frame_time = elapsed_ms / self._frame if self._frame > 0 else 0

    stats_stream = io.StringIO()
    pstats.Stats(self._render_profiler, stream=stats_stream).sort_stats("cumtime").print_stats(PROFILE_STATS)
    print("\n=== Render loop profile ===")
    print(stats_stream.getvalue().rstrip())

    green = "\033[92m"
    reset = "\033[0m"
    print(f"\n{green}Rendered {self._frame} frames in {elapsed_ms:.1f} ms{reset}")
    print(f"{green}Average frame time: {avg_frame_time:.2f} ms ({1000 / avg_frame_time:.1f} FPS){reset}")
    sys.exit(0)

  @contextmanager
  def _startup_profile_context(self):
    if "PROFILE_STARTUP" not in os.environ:
      yield
      return

    profiler = cProfile.Profile()
    start_time = time.monotonic()
    profiler.enable()

    # do the init
    yield

    profiler.disable()
    elapsed_ms = (time.monotonic() - start_time) * 1e3

    stats_stream = io.StringIO()
    pstats.Stats(profiler, stream=stats_stream).sort_stats("cumtime").print_stats(25)
    print("\n=== Startup profile ===")
    print(stats_stream.getvalue().rstrip())

    green = "\033[92m"
    reset = "\033[0m"
    print(f"{green}UI window ready in {elapsed_ms:.1f} ms{reset}")
    sys.exit(0)

  def _calculate_auto_scale(self) -> float:
    # Create temporary window to query monitor info
    rl.init_window(1, 1, "")
    w, h = rl.get_monitor_width(0), rl.get_monitor_height(0)
    rl.close_window()

    if w == 0 or h == 0 or (w >= self._width and h >= self._height):
      return 1.0

    # Apply 0.95 factor for window decorations/taskbar margin
    return max(0.3, min(w / self._width, h / self._height) * 0.95)
