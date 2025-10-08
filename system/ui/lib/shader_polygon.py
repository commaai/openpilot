import platform
import pyray as rl
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, cast
from openpilot.system.ui.lib.application import gui_app

DEBUG = False

MAX_GRADIENT_COLORS = 15  # includes stops as well


@dataclass
class GradientState:
  start: tuple[float, float]
  end: tuple[float, float]
  colors: list[rl.Color]
  stops: list[float]

  def __post_init__(self):
    if len(self.colors) > MAX_GRADIENT_COLORS:
      self.colors = self.colors[:MAX_GRADIENT_COLORS]
      print(f"Warning: GradientState colors truncated to {MAX_GRADIENT_COLORS} entries")

    if len(self.stops) > MAX_GRADIENT_COLORS:
      self.stops = self.stops[:MAX_GRADIENT_COLORS]
      print(f"Warning: GradientState stops truncated to {MAX_GRADIENT_COLORS} entries")

    if not len(self.stops):
      color_count = min(len(self.colors), MAX_GRADIENT_COLORS)
      self.stops = [i / max(1, color_count - 1) for i in range(color_count)]


VERSION = """
#version 300 es
precision highp float;
"""
if platform.system() == "Darwin":
  VERSION = """
    #version 330 core
  """

FRAGMENT_SHADER = VERSION + """
in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec4 fillColor;

// Gradient line defined in *screen pixels*
uniform int useGradient;
uniform vec2 gradientStart;  // e.g. vec2(0, 0)
uniform vec2 gradientEnd;    // e.g. vec2(0, screenHeight)
uniform vec4 gradientColors[15];
uniform float gradientStops[15];
uniform int gradientColorCount;

vec4 getGradientColor(vec2 p) {
  // Compute t from screen-space position
  vec2 d = gradientStart - gradientEnd;
  float len2 = max(dot(d, d), 1e-6);
  float t = clamp(dot(p - gradientEnd, d) / len2, 0.0, 1.0);
  // Clamp to range
  float t0 = gradientStops[0];
  float tn = gradientStops[gradientColorCount-1];
  if (t <= t0) return gradientColors[0];
  if (t >= tn) return gradientColors[gradientColorCount-1];

  for (int i = 0; i < gradientColorCount - 1; i++) {
    float a = gradientStops[i];
    float b = gradientStops[i+1];
    if (t >= a && t <= b) {
      float k = (t - a) / max(b - a, 1e-6);
      return mix(gradientColors[i], gradientColors[i+1], k);
    }
  }
  return gradientColors[gradientColorCount-1];
}

void main() {
  vec2 p = vec2(gl_FragCoord.x, gl_FragCoord.y);
  vec4 color = useGradient == 1 ? getGradientColor(p) : fillColor;

  // TODO: does this do anything?
  // fragTexCoord.y = 0 at inner edge, 1 at outer feather ring (~1 px)
  float alpha = smoothstep(1.0, 0.0, fragTexCoord.y);
  color.a *= alpha;

  finalColor = color;
}
"""

# Default vertex shader
VERTEX_SHADER = VERSION + """
in vec3 vertexPosition;
in vec2 vertexTexCoord;
out vec2 fragTexCoord;
uniform mat4 mvp;

void main() {
  fragTexCoord = vertexTexCoord;
  gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""

UNIFORM_INT = rl.ShaderUniformDataType.SHADER_UNIFORM_INT
UNIFORM_FLOAT = rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT
UNIFORM_VEC2 = rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2
UNIFORM_VEC4 = rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4


class ShaderState:
  _instance: Any = None

  @classmethod
  def get_instance(cls):
    if cls._instance is None:
      cls._instance = cls()
    return cls._instance

  def __init__(self):
    if ShaderState._instance is not None:
      raise Exception("This class is a singleton. Use get_instance() instead.")

    self.initialized = False
    self.shader = None
    self.locations = {
      'mvp': None,
      'fillColor': None,
      'useGradient': None,
      'gradientStart': None,
      'gradientEnd': None,
      'gradientColors': None,
      'gradientStops': None,
      'gradientColorCount': None,
    }

    # Pre-allocated FFI objects
    self.fill_color_ptr = rl.ffi.new("float[]", [0.0, 0.0, 0.0, 0.0])
    self.use_gradient_ptr = rl.ffi.new("int[]", [0])
    self.color_count_ptr = rl.ffi.new("int[]", [0])
    self.gradient_colors_ptr = rl.ffi.new("float[]", MAX_GRADIENT_COLORS * 4)
    self.gradient_stops_ptr = rl.ffi.new("float[]", MAX_GRADIENT_COLORS)

  def initialize(self):
    if self.initialized:
      return

    self.shader = rl.load_shader_from_memory(VERTEX_SHADER, FRAGMENT_SHADER)

    # Cache all uniform locations
    for uniform in self.locations.keys():
      self.locations[uniform] = rl.get_shader_location(self.shader, uniform)

    # Orthographic MVP (origin top-left)
    # TODO: why did this change from the original? any differences?
    proj = rl.matrix_ortho(0, gui_app.width, gui_app.height, 0, -1, 1)
    rl.set_shader_value_matrix(self.shader, self.locations['mvp'], proj)

    self.initialized = True

  def cleanup(self):
    if not self.initialized:
      return
    if self.shader:
      rl.unload_shader(self.shader)
      self.shader = None
    self.initialized = False


def _configure_shader_color(state: ShaderState, color: Optional[rl.Color],  # noqa: UP045
                            gradient: GradientState | None, origin_rect: rl.Rectangle):
  assert (color is not None) != (gradient is not None), "Either color or gradient must be provided"

  use_gradient = 1 if (gradient is not None and len(gradient.colors) >= 1) else 0
  state.use_gradient_ptr[0] = use_gradient
  rl.set_shader_value(state.shader, state.locations['useGradient'], state.use_gradient_ptr, UNIFORM_INT)

  if use_gradient:
    gradient = cast(GradientState, gradient)
    state.color_count_ptr[0] = len(gradient.colors)
    for i in range(len(gradient.colors)):
      c = gradient.colors[i]
      base = i * 4
      state.gradient_colors_ptr[base:base + 4] = [c.r / 255.0, c.g / 255.0, c.b / 255.0, c.a / 255.0]
    rl.set_shader_value_v(state.shader, state.locations['gradientColors'], state.gradient_colors_ptr, UNIFORM_VEC4, len(gradient.colors))

    for i in range(len(gradient.stops)):
      s = float(gradient.stops[i])
      state.gradient_stops_ptr[i] = 0.0 if s < 0.0 else 1.0 if s > 1.0 else s
    rl.set_shader_value_v(state.shader, state.locations['gradientStops'], state.gradient_stops_ptr, UNIFORM_FLOAT, len(gradient.stops))
    rl.set_shader_value(state.shader, state.locations['gradientColorCount'], state.color_count_ptr, UNIFORM_INT)

    # Map normalized start/end to screen pixels
    start_vec = rl.Vector2(origin_rect.x + gradient.start[0] * origin_rect.width, origin_rect.y + gradient.start[1] * origin_rect.height)
    end_vec = rl.Vector2(origin_rect.x + gradient.end[0] * origin_rect.width, origin_rect.y + gradient.end[1] * origin_rect.height)
    rl.set_shader_value(state.shader, state.locations['gradientStart'], start_vec, UNIFORM_VEC2)
    rl.set_shader_value(state.shader, state.locations['gradientEnd'], end_vec, UNIFORM_VEC2)
  else:
    color = color or rl.WHITE
    state.fill_color_ptr[0:4] = [color.r / 255.0, color.g / 255.0, color.b / 255.0, color.a / 255.0]
    rl.set_shader_value(state.shader, state.locations['fillColor'], state.fill_color_ptr, UNIFORM_VEC4)


def triangulate(pts: np.ndarray) -> list[tuple[float, float]]:
  # TODO: consider deduping close screenspace points
  # interleave points to produce a triangle strip
  assert len(pts) % 2 == 0, "Interleaving expects even number of points"

  tri_strip = []
  for i in range(len(pts) // 2):
    tri_strip.append(pts[i])
    tri_strip.append(pts[-i - 1])

  return cast(list, np.array(tri_strip).tolist())


def draw_polygon(origin_rect: rl.Rectangle, points: np.ndarray,
                 color: Optional[rl.Color] = None, gradient: GradientState | None = None):  # noqa: UP045
  """
  Draw a simple filled polygon by triangulating to indexed triangles with earcut
  and rendering them under a lightweight shader. Supports solid color or
  two-stop linear gradient in screen space.
  """
  if points is None or len(points) < 3:
    return

  # Ensure (N,2) float32 contiguous array
  pts = np.ascontiguousarray(points, dtype=np.float32)
  if pts.ndim != 2 or pts.shape[1] != 2:
    return

  # Initialize shader on-demand (after window/context is ready)
  state = ShaderState.get_instance()
  state.initialize()

  _configure_shader_color(state, color, gradient, origin_rect)

  tri_strip = triangulate(pts)

  # Use custom shader for gradient, color here doesn't matter
  rl.begin_shader_mode(state.shader)
  rl.draw_triangle_strip(tri_strip, len(tri_strip), rl.WHITE)
  rl.end_shader_mode()

  if DEBUG:
    for i in range(len(pts)):
      rl.draw_circle_lines(int(pts[i, 0]), int(pts[i, 1]), 3, rl.RED)

    # draw each triangle, need to handle deduped tri_strip
    i = 0
    while i + 2 < len(tri_strip):
      color1 = rl.BLUE if (i) % 3 == 0 else rl.RED if (i) % 3 == 1 else rl.GREEN
      color2 = rl.BLUE if (i) % 3 == 1 else rl.RED if (i) % 3 == 2 else rl.GREEN
      color3 = rl.BLUE if (i) % 3 == 2 else rl.RED if (i) % 3 == 0 else rl.GREEN
      a = rl.Vector2(tri_strip[i][0], tri_strip[i][1])
      b = rl.Vector2(tri_strip[i + 1][0], tri_strip[i + 1][1])
      c = rl.Vector2(tri_strip[i + 2][0], tri_strip[i + 2][1])
      rl.draw_line_ex(a, b, 1, color1)
      rl.draw_line_ex(b, c, 1, color2)
      rl.draw_line_ex(c, a, 1, color3)
      i += 1


def cleanup_shader_resources():
  state = ShaderState.get_instance()
  state.cleanup()
