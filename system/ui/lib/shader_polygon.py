import platform
import pyray as rl
import numpy as np
from typing import Any
from openpilot.system.ui.lib.application import gui_app

DEBUG = False

# TODO: this match?
MAX_GRADIENT_COLORS = 15

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

// Two-color fallback gradient (top->bottom or arbitrary line)
uniform vec4 uColorTop;
uniform vec4 uColorBottom;

// Gradient line defined in *screen pixels*
uniform vec2 uGradStart;   // e.g. vec2(0, 0)
uniform vec2 uGradEnd;     // e.g. vec2(0, screenHeight)

// TODO: remove since unused
uniform int  uUseFeather;  // 0 or 1

// Arbitrary-stop gradient support
uniform vec4 uGradColors[15];
uniform float uGradStops[15];
uniform int uGradCount;

vec4 getGradientColor(float t) {
  if (uGradCount <= 0) return mix(uColorTop, uColorBottom, t);
  if (uGradCount == 1) return uGradColors[0];

  // Clamp to range
  float t0 = uGradStops[0];
  float tn = uGradStops[uGradCount-1];
  if (t <= t0) return uGradColors[0];
  if (t >= tn) return uGradColors[uGradCount-1];

  for (int i = 0; i < uGradCount - 1; i++) {
    float a = uGradStops[i];
    float b = uGradStops[i+1];
    if (t >= a && t <= b) {
      float k = (t - a) / max(b - a, 1e-6);
      return mix(uGradColors[i], uGradColors[i+1], k);
    }
  }
  return uGradColors[uGradCount-1];
}

void main() {
  // Compute t from screen-space position
  vec2 p = vec2(gl_FragCoord.x, gl_FragCoord.y);
  vec2 d = uGradEnd - uGradStart;
  float len2 = max(dot(d, d), 1e-6);
  float t = clamp(dot(p - uGradStart, d) / len2, 0.0, 1.0);

  // TODO: fix the flip
  vec4 col = getGradientColor(1.0f - t);

  if (uUseFeather == 1) {
    // TODO: needs more aliasing?
    // fragTexCoord.y = 0 at inner edge, 1 at outer feather ring (~1 px)
    float alpha = smoothstep(1.0, 0.0, fragTexCoord.y);
    col.a *= alpha;
  }

  finalColor = col;
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
      'uColorTop': None,
      'uColorBottom': None,
      'uGradStart': None,
      'uGradEnd': None,
      'uUseFeather': None,
      'uGradColors': None,
      'uGradStops': None,
      'uGradCount': None,
    }
    self._last_w = 0
    self._last_h = 0
    self._feather_ptr = rl.ffi.new("int[]", [0])
    self._grad_count_ptr = rl.ffi.new("int[]", [0])
    self._grad_colors_ptr = rl.ffi.new("float[]", MAX_GRADIENT_COLORS * 4)  # TODO: wtf is this
    self._grad_stops_ptr = rl.ffi.new("float[]", MAX_GRADIENT_COLORS)

  def initialize(self):
    if self.initialized:
      # Update MVP if size changed
      if self._last_w != gui_app.width or self._last_h != gui_app.height:
        proj = rl.matrix_ortho(0, gui_app.width, gui_app.height, 0, -1, 1)
        rl.set_shader_value_matrix(self.shader, self.locations['mvp'], proj)
        self._last_w, self._last_h = gui_app.width, gui_app.height
      return

    # Safe to call only after window/context exists
    self.shader = rl.load_shader_from_memory(VERTEX_SHADER, FRAGMENT_SHADER)

    # Cache locations
    self.locations['mvp'] = rl.get_shader_location(self.shader, "mvp")
    self.locations['uColorTop'] = rl.get_shader_location(self.shader, "uColorTop")
    self.locations['uColorBottom'] = rl.get_shader_location(self.shader, "uColorBottom")
    self.locations['uGradStart'] = rl.get_shader_location(self.shader, "uGradStart")
    self.locations['uGradEnd'] = rl.get_shader_location(self.shader, "uGradEnd")
    self.locations['uUseFeather'] = rl.get_shader_location(self.shader, "uUseFeather")
    self.locations['uGradColors'] = rl.get_shader_location(self.shader, "uGradColors")
    self.locations['uGradStops'] = rl.get_shader_location(self.shader, "uGradStops")
    self.locations['uGradCount'] = rl.get_shader_location(self.shader, "uGradCount")

    # Orthographic MVP (origin top-left)
    proj = rl.matrix_ortho(0, gui_app.width, gui_app.height, 0, -1, 1)
    rl.set_shader_value_matrix(self.shader, self.locations['mvp'], proj)
    self._last_w, self._last_h = gui_app.width, gui_app.height

    # Reasonable defaults
    rl.set_shader_value(self.shader, self.locations['uColorTop'], rl.Vector4(1, 1, 1, 1), UNIFORM_VEC4)
    rl.set_shader_value(self.shader, self.locations['uColorBottom'], rl.Vector4(0, 0, 0, 1), UNIFORM_VEC4)
    rl.set_shader_value(self.shader, self.locations['uGradStart'], rl.Vector2(0, 0), UNIFORM_VEC2)
    rl.set_shader_value(self.shader, self.locations['uGradEnd'], rl.Vector2(0, gui_app.height), UNIFORM_VEC2)
    rl.set_shader_value(self.shader, self.locations['uUseFeather'], self._feather_ptr, UNIFORM_INT)
    rl.set_shader_value(self.shader, self.locations['uGradCount'], self._grad_count_ptr, UNIFORM_INT)

    self.initialized = True

  def cleanup(self):
    if not self.initialized:
      return
    if self.shader:
      rl.unload_shader(self.shader)
      self.shader = None
    self.initialized = False


def _configure_shader_color(state, color, gradient, clipped_rect, original_rect):
  use_gradient = 1 if gradient else 0
  state.use_gradient_ptr[0] = use_gradient
  rl.set_shader_value(state.shader, state.locations['useGradient'], state.use_gradient_ptr, UNIFORM_INT)

  if use_gradient:
    start = np.array(gradient['start']) * np.array([original_rect.width, original_rect.height]) + np.array([original_rect.x, original_rect.y])
    end = np.array(gradient['end']) * np.array([original_rect.width, original_rect.height]) + np.array([original_rect.x, original_rect.y])
    start = start - np.array([clipped_rect.x, clipped_rect.y])
    end = end - np.array([clipped_rect.x, clipped_rect.y])
    state.gradient_start_ptr[0:2] = start.astype(np.float32)
    state.gradient_end_ptr[0:2] = end.astype(np.float32)
    rl.set_shader_value(state.shader, state.locations['gradientStart'], state.gradient_start_ptr, UNIFORM_VEC2)
    rl.set_shader_value(state.shader, state.locations['gradientEnd'], state.gradient_end_ptr, UNIFORM_VEC2)

    colors = gradient['colors']
    color_count = min(len(colors), MAX_GRADIENT_COLORS)
    state.color_count_ptr[0] = color_count
    for i, c in enumerate(colors[:color_count]):
      base_idx = i * 4
      state.gradient_colors_ptr[base_idx:base_idx + 4] = [c.r / 255.0, c.g / 255.0, c.b / 255.0, c.a / 255.0]
    rl.set_shader_value_v(state.shader, state.locations['gradientColors'], state.gradient_colors_ptr, UNIFORM_VEC4, color_count)

    stops = gradient.get('stops', [i / max(1, color_count - 1) for i in range(color_count)])
    stops = np.clip(stops[:color_count], 0.0, 1.0)
    state.gradient_stops_ptr[0:color_count] = stops
    rl.set_shader_value_v(state.shader, state.locations['gradientStops'], state.gradient_stops_ptr, UNIFORM_FLOAT, color_count)
    rl.set_shader_value(state.shader, state.locations['gradientColorCount'], state.color_count_ptr, UNIFORM_INT)
  else:
    color = color or rl.WHITE
    state.fill_color_ptr[0:4] = [color.r / 255.0, color.g / 255.0, color.b / 255.0, color.a / 255.0]
    rl.set_shader_value(state.shader, state.locations['fillColor'], state.fill_color_ptr, UNIFORM_VEC4)


# TODO: remove all this extra dedup junk unless necessary
def triangulate(pts: np.ndarray, min_pair_px: float = 0.5, dedup_eps: float = 0.25) -> list[tuple[float, float]]:
  """
  Build an interleaved triangle strip from a ribbon polygon laid out as
  [L0..Lk-1, Rk-1..R0]. Returns [L0, R0, L1, R1, ...].
  Skips near-zero width pairs and removes adjacent duplicates to avoid
  degenerate triangles.
  """
  # TODO: check this never happens
  # TODO: surely we can simplify this. why are we converting to floats?
  n = len(pts)
  if n < 4 or (n % 2) != 0:
    return []

  k = n // 2
  left = pts[:k]
  right_rev = pts[k:][::-1]

  interleaved: list[tuple[float, float]] = []
  min_pair_px2 = min_pair_px * min_pair_px
  for i in range(min(len(left), len(right_rev))):
    lx, ly = float(left[i, 0]), float(left[i, 1])
    rx, ry = float(right_rev[i, 0]), float(right_rev[i, 1])
    dx = lx - rx
    dy = ly - ry
    # TODO: ?
    if dx * dx + dy * dy < min_pair_px2:
      continue
    interleaved.append((lx, ly))
    interleaved.append((rx, ry))

  # ??
  # Deduplicate adjacent vertices (screen-space)
  if len(interleaved) >= 2:
    deduped: list[tuple[float, float]] = [interleaved[0]]
    lastx, lasty = interleaved[0]
    thr2 = dedup_eps * dedup_eps
    for vx, vy in interleaved[1:]:
      dx = vx - lastx
      dy = vy - lasty
      if dx * dx + dy * dy >= thr2:
        deduped.append((vx, vy))
        lastx, lasty = vx, vy
    interleaved = deduped

  # TODO: no! check this never happens and remove?
  # Ensure even count for pairs (optional: drop last if odd)
  if len(interleaved) % 2 == 1:
    interleaved = interleaved[:-1]

  return interleaved


def draw_polygon(origin_rect: rl.Rectangle, points: np.ndarray, color=None, gradient=None):
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

  # Configure uniforms (arbitrary gradient stops if provided)
  if gradient and 'colors' in gradient and len(gradient['colors']) >= 1:
    cols = gradient['colors']
    stops = gradient.get('stops', [i / max(1, len(cols) - 1) for i in range(len(cols))])
    count = min(len(cols), MAX_GRADIENT_COLORS)
    state._grad_count_ptr[0] = count
    for i in range(count):
      c = cols[i]
      base = i * 4
      state._grad_colors_ptr[base:base + 4] = [c.r / 255.0, c.g / 255.0, c.b / 255.0, c.a / 255.0]
    rl.set_shader_value_v(state.shader, state.locations['uGradColors'], state._grad_colors_ptr, UNIFORM_VEC4, count)

    stops = np.clip(np.asarray(stops, dtype=np.float32)[:count], 0.0, 1.0)
    state._grad_stops_ptr[0:count] = stops
    rl.set_shader_value_v(state.shader, state.locations['uGradStops'], state._grad_stops_ptr, UNIFORM_FLOAT, count)
    rl.set_shader_value(state.shader, state.locations['uGradCount'], state._grad_count_ptr, UNIFORM_INT)

    # Gradient line is provided normalized to rect; convert to screen pixels
    start = np.array(gradient.get('start', (0.0, 1.0)), dtype=np.float32)
    end = np.array(gradient.get('end', (0.0, 0.0)), dtype=np.float32)
    start_px = start * np.array([origin_rect.width, origin_rect.height], dtype=np.float32) + np.array([origin_rect.x, origin_rect.y], dtype=np.float32)
    end_px = end * np.array([origin_rect.width, origin_rect.height], dtype=np.float32) + np.array([origin_rect.x, origin_rect.y], dtype=np.float32)
    rl.set_shader_value(state.shader, state.locations['uGradStart'], rl.Vector2(float(start_px[0]), float(start_px[1])), UNIFORM_VEC2)
    rl.set_shader_value(state.shader, state.locations['uGradEnd'], rl.Vector2(float(end_px[0]), float(end_px[1])), UNIFORM_VEC2)
  else:
    # Solid color
    c = color or rl.WHITE
    vec = rl.Vector4(c.r / 255.0, c.g / 255.0, c.b / 255.0, c.a / 255.0)
    rl.set_shader_value(state.shader, state.locations['uColorTop'], vec, UNIFORM_VEC4)
    rl.set_shader_value(state.shader, state.locations['uColorBottom'], vec, UNIFORM_VEC4)
    state._grad_count_ptr[0] = 0
    rl.set_shader_value(state.shader, state.locations['uGradCount'], state._grad_count_ptr, UNIFORM_INT)

  rl.set_shader_value(state.shader, state.locations['uUseFeather'], state._feather_ptr, UNIFORM_INT)

  tri_strip = triangulate(pts)
  # TODO: check this
  if len(tri_strip) < 4:
    return

  # Use custom shader (for gradient) if configured above; pass WHITE so shader drives color
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
