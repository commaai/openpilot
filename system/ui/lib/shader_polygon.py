import platform
import pyray as rl
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, cast
from openpilot.system.ui.lib.application import gui_app

MAX_GRADIENT_COLORS = 15  # includes stops as well


@dataclass
class Gradient:
  start: tuple[float, float]
  end: tuple[float, float]
  colors: list[rl.Color]
  stops: list[float]

  def __post_init__(self):
    if len(self.colors) > MAX_GRADIENT_COLORS:
      self.colors = self.colors[:MAX_GRADIENT_COLORS]
      print(f"Warning: Gradient colors truncated to {MAX_GRADIENT_COLORS} entries")

    if len(self.stops) > MAX_GRADIENT_COLORS:
      self.stops = self.stops[:MAX_GRADIENT_COLORS]
      print(f"Warning: Gradient stops truncated to {MAX_GRADIENT_COLORS} entries")

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
  vec4 color = useGradient == 1 ? getGradientColor(gl_FragCoord.xy) : fillColor;

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
      'fillColor': None,
      'useGradient': None,
      'gradientStart': None,
      'gradientEnd': None,
      'gradientColors': None,
      'gradientStops': None,
      'gradientColorCount': None,
      'mvp': None,
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
                            gradient: Gradient | None, origin_rect: rl.Rectangle):
  assert (color is not None) != (gradient is not None), "Either color or gradient must be provided"

  use_gradient = 1 if (gradient is not None and len(gradient.colors) >= 1) else 0
  state.use_gradient_ptr[0] = use_gradient
  rl.set_shader_value(state.shader, state.locations['useGradient'], state.use_gradient_ptr, UNIFORM_INT)

  if use_gradient:
    gradient = cast(Gradient, gradient)
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
  """Only supports simple polygons with two chains (ribbon)."""

  # TODO: consider deduping close screenspace points
  # interleave points to produce a triangle strip
  assert len(pts) % 2 == 0, "Interleaving expects even number of points"

  tri_strip = []
  for i in range(len(pts) // 2):
    tri_strip.append(pts[i])
    tri_strip.append(pts[-i - 1])

  return cast(list, np.array(tri_strip).tolist())


def draw_polygon(origin_rect: rl.Rectangle, points: np.ndarray,
                 color: Optional[rl.Color] = None, gradient: Gradient | None = None):  # noqa: UP045

  """
  Draw a ribbon polygon (two chains) with a triangle strip and gradient.
  - Input must be [L0..Lk-1, Rk-1..R0], even count, no crossings/holes.
  """
  if len(points) < 3:
    return

  # Initialize shader on-demand
  state = ShaderState.get_instance()
  state.initialize()

  # Ensure (N,2) float32 contiguous array
  pts = np.ascontiguousarray(points, dtype=np.float32)
  assert pts.ndim == 2 and pts.shape[1] == 2, "points must be (N,2)"

  # Configure gradient shader
  _configure_shader_color(state, color, gradient, origin_rect)

  # Triangulate via interleaving
  tri_strip = triangulate(pts)

  # Draw strip, color here doesn't matter
  rl.begin_shader_mode(state.shader)
  rl.draw_triangle_strip(tri_strip, len(tri_strip), rl.WHITE)
  rl.end_shader_mode()


def cleanup_shader_resources():
  state = ShaderState.get_instance()
  state.cleanup()
