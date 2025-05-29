import pyray as rl
import numpy as np
from typing import Any

MAX_GRADIENT_COLORS = 15

FRAGMENT_SHADER = """
#version 300 es
precision mediump float;

in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec2 points[100];
uniform int pointCount;
uniform vec4 fillColor;
uniform vec2 resolution;

uniform bool useGradient;
uniform vec2 gradientStart;
uniform vec2 gradientEnd;
uniform vec4 gradientColors[15];
uniform float gradientStops[15];
uniform int gradientColorCount;
uniform vec2 visibleGradientRange;

vec4 getGradientColor(vec2 pos) {
  vec2 gradientDir = gradientEnd - gradientStart;
  float gradientLength = length(gradientDir);
  if (gradientLength < 0.001) return gradientColors[0];

  vec2 normalizedDir = gradientDir / gradientLength;
  vec2 pointVec = pos - gradientStart;
  float projection = dot(pointVec, normalizedDir);

  float t = projection / gradientLength;

  // Gradient clipping: remap t to visible range
  float visibleStart = visibleGradientRange.x;
  float visibleEnd = visibleGradientRange.y;
  float visibleRange = visibleEnd - visibleStart;

  // Remap t to visible range
  if (visibleRange > 0.001) {
    t = visibleStart + t * visibleRange;
  }

  t = clamp(t, 0.0, 1.0);
  for (int i = 0; i < gradientColorCount - 1; i++) {
    if (t >= gradientStops[i] && t <= gradientStops[i+1]) {
      float segmentT = (t - gradientStops[i]) / (gradientStops[i+1] - gradientStops[i]);
      return mix(gradientColors[i], gradientColors[i+1], segmentT);
    }
  }

  return gradientColors[gradientColorCount-1];
}

bool isPointInsidePolygon(vec2 p) {
  if (pointCount < 3) return false;

  int crossings = 0;
  for (int i = 0, j = pointCount - 1; i < pointCount; j = i++) {
    vec2 pi = points[i];
    vec2 pj = points[j];

    // Skip degenerate edges
    if (distance(pi, pj) < 0.001) continue;

    // Ray-casting
    if (((pi.y > p.y) != (pj.y > p.y)) &&
        (p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y + 0.001) + pi.x)) {
      crossings++;
    }
  }
  return (crossings & 1) == 1;
}

float distanceToEdge(vec2 p) {
  float minDist = 1000.0;

  for (int i = 0, j = pointCount - 1; i < pointCount; j = i++) {
    vec2 edge0 = points[j];
    vec2 edge1 = points[i];

    if (distance(edge0, edge1) < 0.0001) continue;

    vec2 v1 = p - edge0;
    vec2 v2 = edge1 - edge0;
    float l2 = dot(v2, v2);

    if (l2 < 0.0001) {
      float dist = length(v1);
      minDist = min(minDist, dist);
      continue;
    }

    float t = clamp(dot(v1, v2) / l2, 0.0, 1.0);
    vec2 projection = edge0 + t * v2;
    float dist = length(p - projection);
    minDist = min(minDist, dist);
  }

  return minDist;
}

float signedDistanceToPolygon(vec2 p) {
  float dist = distanceToEdge(p);
  bool inside = isPointInsidePolygon(p);
  return inside ? dist : -dist;
}

void main() {
  vec2 pixel = fragTexCoord * resolution;

  float signedDist = signedDistanceToPolygon(pixel);

  vec2 pixelGrad = vec2(dFdx(pixel.x), dFdy(pixel.y));
  float pixelSize = length(pixelGrad);
  float aaWidth = max(0.5, pixelSize * 0.5); // Sharper anti-aliasing

  float alpha = smoothstep(-aaWidth, aaWidth, signedDist);
  if (alpha > 0.0) {
    vec4 color = useGradient ? getGradientColor(fragTexCoord) : fillColor;
    finalColor = vec4(color.rgb, color.a * alpha);
  } else {
    finalColor = vec4(0.0);
  }
}
"""

# Default vertex shader
VERTEX_SHADER = """
#version 300 es
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
    self.white_texture = None

    # Shader uniform locations
    self.locations = {
      'pointCount': None,
      'fillColor': None,
      'resolution': None,
      'points': None,
      'useGradient': None,
      'gradientStart': None,
      'gradientEnd': None,
      'gradientColors': None,
      'gradientStops': None,
      'gradientColorCount': None,
      'mvp': None,
      'visibleGradientRange': None,
    }

    # Pre-allocated FFI objects
    self.point_count_ptr = rl.ffi.new("int[]", [0])
    self.resolution_ptr = rl.ffi.new("float[]", [0.0, 0.0])
    self.fill_color_ptr = rl.ffi.new("float[]", [0.0, 0.0, 0.0, 0.0])
    self.use_gradient_ptr = rl.ffi.new("int[]", [0])
    self.gradient_start_ptr = rl.ffi.new("float[]", [0.0, 0.0])
    self.gradient_end_ptr = rl.ffi.new("float[]", [0.0, 0.0])
    self.color_count_ptr = rl.ffi.new("int[]", [0])
    self.visible_gradient_range_ptr = rl.ffi.new("float[]", [0.0, 0.0])
    self.gradient_colors_ptr = rl.ffi.new("float[]", MAX_GRADIENT_COLORS * 4)
    self.gradient_stops_ptr = rl.ffi.new("float[]", MAX_GRADIENT_COLORS)

  def initialize(self):
    if self.initialized:
      return

    self.shader = rl.load_shader_from_memory(VERTEX_SHADER, FRAGMENT_SHADER)

    # Create and cache white texture
    white_img = rl.gen_image_color(2, 2, rl.WHITE)
    self.white_texture = rl.load_texture_from_image(white_img)
    rl.set_texture_filter(self.white_texture, rl.TEXTURE_FILTER_BILINEAR)
    rl.unload_image(white_img)

    # Cache all uniform locations
    for uniform in self.locations.keys():
      self.locations[uniform] = rl.get_shader_location(self.shader, uniform)

    # Setup default MVP matrix
    mvp_ptr = rl.ffi.new("float[16]", [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    rl.set_shader_value_matrix(self.shader, self.locations['mvp'], rl.Matrix(*mvp_ptr))

    self.initialized = True

  def cleanup(self):
    if not self.initialized:
      return

    if self.white_texture:
      rl.unload_texture(self.white_texture)
      self.white_texture = None

    if self.shader:
      rl.unload_shader(self.shader)
      self.shader = None

    self.initialized = False


def _configure_shader_color(state, color, gradient, rect, min_xy, max_xy):
  """Configure shader uniforms for solid color or gradient rendering"""
  use_gradient = 1 if gradient else 0
  state.use_gradient_ptr[0] = use_gradient
  rl.set_shader_value(state.shader, state.locations['useGradient'], state.use_gradient_ptr, UNIFORM_INT)

  if use_gradient:
    # Set gradient start/end
    state.gradient_start_ptr[0:2] = gradient['start']
    state.gradient_end_ptr[0:2] = gradient['end']
    rl.set_shader_value(state.shader, state.locations['gradientStart'], state.gradient_start_ptr, UNIFORM_VEC2)
    rl.set_shader_value(state.shader, state.locations['gradientEnd'], state.gradient_end_ptr, UNIFORM_VEC2)

    # Calculate visible gradient range
    width = max_xy[0] - min_xy[0]
    height = max_xy[1] - min_xy[1]

    gradient_dir = (gradient['end'][0] - gradient['start'][0], gradient['end'][1] - gradient['start'][1])
    is_vertical = abs(gradient_dir[1]) > abs(gradient_dir[0])

    visible_start = 0.0
    visible_end = 1.0

    if is_vertical and height > 0:
      visible_start = (rect.y - min_xy[1]) / height
      visible_end = visible_start + rect.height / height
    elif width > 0:
      visible_start = (rect.x - min_xy[0]) / width
      visible_end = visible_start + rect.width / width

    # Clamp visible range
    visible_start = max(0.0, min(1.0, visible_start))
    visible_end = max(0.0, min(1.0, visible_end))

    state.visible_gradient_range_ptr[0:2] = [visible_start, visible_end]
    rl.set_shader_value(state.shader, state.locations['visibleGradientRange'], state.visible_gradient_range_ptr, UNIFORM_VEC2)

    # Set gradient colors
    colors = gradient['colors']
    color_count = min(len(colors), MAX_GRADIENT_COLORS)
    for i, c in enumerate(colors[:color_count]):
      base_idx = i * 4
      state.gradient_colors_ptr[base_idx:base_idx+4] = [c.r / 255.0, c.g / 255.0, c.b / 255.0, c.a / 255.0]
    rl.set_shader_value_v(state.shader, state.locations['gradientColors'], state.gradient_colors_ptr, UNIFORM_VEC4, color_count)

    # Set gradient stops
    stops = gradient.get('stops', [i / (color_count - 1) for i in range(color_count)])
    state.gradient_stops_ptr[0:color_count] = stops[:color_count]
    rl.set_shader_value_v(state.shader, state.locations['gradientStops'], state.gradient_stops_ptr, UNIFORM_FLOAT, color_count)

    # Set color count
    state.color_count_ptr[0] = color_count
    rl.set_shader_value(state.shader, state.locations['gradientColorCount'], state.color_count_ptr, UNIFORM_INT)
  else:
    color = color or rl.WHITE  # Default to white if no color provided
    state.fill_color_ptr[0:4] = [color.r / 255.0, color.g / 255.0, color.b / 255.0, color.a / 255.0]
    rl.set_shader_value(state.shader, state.locations['fillColor'], state.fill_color_ptr, UNIFORM_VEC4)


def draw_polygon(rect: rl.Rectangle, points: np.ndarray, color=None, gradient=None):
  """
  Draw a complex polygon using shader-based even-odd fill rule

  Args:
      rect: Rectangle defining the drawing area
      points: numpy array of (x,y) points defining the polygon
      color: Solid fill color (rl.Color)
      gradient: Dict with gradient parameters:
          {
              'start': (x1, y1),    # Start point (normalized 0-1)
              'end': (x2, y2),      # End point (normalized 0-1)
              'colors': [rl.Color], # List of colors at stops
              'stops': [float]      # List of positions (0-1)
          }
  """
  if len(points) < 3:
    return

  state = ShaderState.get_instance()
  if not state.initialized:
    state.initialize()

  # Find bounding box
  min_xy = np.min(points, axis=0)
  max_xy = np.max(points, axis=0)

  # Clip coordinates to rectangle
  clip_x = max(rect.x, min_xy[0])
  clip_y = max(rect.y, min_xy[1])
  clip_right = min(rect.x + rect.width, max_xy[0])
  clip_bottom = min(rect.y + rect.height, max_xy[1])

  # Check if polygon is completely off-screen
  if clip_x >= clip_right or clip_y >= clip_bottom:
    return

  clipped_width = clip_right - clip_x
  clipped_height = clip_bottom - clip_y

  clip_rect = rl.Rectangle(clip_x, clip_y, clipped_width, clipped_height)

  # Transform points relative to the CLIPPED area
  transformed_points = points - np.array([clip_x, clip_y])

  # Set shader values
  state.point_count_ptr[0] = len(transformed_points)
  rl.set_shader_value(state.shader, state.locations['pointCount'], state.point_count_ptr, UNIFORM_INT)

  state.resolution_ptr[0:2] = [clipped_width, clipped_height]
  rl.set_shader_value(state.shader, state.locations['resolution'], state.resolution_ptr, UNIFORM_VEC2)

  flat_points = np.ascontiguousarray(transformed_points.flatten().astype(np.float32))
  points_ptr = rl.ffi.cast("float *", flat_points.ctypes.data)
  rl.set_shader_value_v(state.shader, state.locations['points'], points_ptr, UNIFORM_VEC2, len(transformed_points))

  _configure_shader_color(state, color, gradient, clip_rect, min_xy, max_xy)

  # Render
  rl.begin_shader_mode(state.shader)
  rl.draw_texture_pro(
    state.white_texture,
    rl.Rectangle(0, 0, 2, 2),
    clip_rect,
    rl.Vector2(0, 0),
    0.0,
    rl.WHITE,
  )
  rl.end_shader_mode()


def cleanup_shader_resources():
  state = ShaderState.get_instance()
  state.cleanup()
