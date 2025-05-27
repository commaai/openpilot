import pyray as rl
import numpy as np
from typing import Any


FRAGMENT_SHADER = """
#version 300 es
precision mediump float;

in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec2 points[100];
uniform int pointCount;
uniform vec4 fillColor;
uniform vec2 resolution;

uniform int useGradient;
uniform vec2 gradientStart;
uniform vec2 gradientEnd;
uniform vec4 gradientColors[8];
uniform float gradientStops[8];
uniform int gradientColorCount;

vec4 getGradientColor(vec2 pos) {
    vec2 gradientDir = gradientEnd - gradientStart;
    float gradientLength = length(gradientDir);

    if (gradientLength < 0.001) return gradientColors[0];

    float t = clamp(dot(pos - gradientStart, gradientDir) / (gradientLength * gradientLength), 0.0, 1.0);

    // Binary search for better performance with many stops
    int left = 0;
    int right = gradientColorCount - 1;

    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (t <= gradientStops[mid]) {
            right = mid;
        } else {
            left = mid;
        }
    }

    if (left == right) return gradientColors[left];

    float segmentT = (t - gradientStops[left]) / (gradientStops[right] - gradientStops[left]);
    return mix(gradientColors[left], gradientColors[right], segmentT);
}

float distanceToEdge(vec2 p) {
  float minDist = 1000.0;

  for (int i = 0, j = pointCount - 1; i < pointCount; j = i++) {
    vec2 edge0 = points[j];
    vec2 edge1 = points[i];

    if (edge0 == edge1) continue;

    vec2 v1 = p - edge0;
    vec2 v2 = edge1 - edge0;
    float l2 = dot(v2, v2);

    if (l2 < 0.0001) {
      float dist = length(v1);
      minDist = min(minDist, dist);
      continue;
    }

    float t = max(0.0, min(1.0, dot(v1, v2) / l2));

    vec2 projection = edge0 + t * v2;

    float dist = length(p - projection);
    minDist = min(minDist, dist);
  }

  return minDist;
}

bool isPointInsidePolygon(vec2 p) {
  if (pointCount < 3) return false;

  if (pointCount == 3) {
    vec2 v0 = points[0];
    vec2 v1 = points[1];
    vec2 v2 = points[2];

    float d = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);
    if (abs(d) < 0.0001) return false;

    float a = ((v1.y - v2.y) * (p.x - v2.x) + (v2.x - v1.x) * (p.y - v2.y)) / d;
    float b = ((v2.y - v0.y) * (p.x - v2.x) + (v0.x - v2.x) * (p.y - v2.y)) / d;
    float c = 1.0 - a - b;

    return (a >= 0.0 && b >= 0.0 && c >= 0.0);
  }

  bool inside = false;
  for (int i = 0, j = pointCount - 1; i < pointCount; j = i++) {
    if (points[i] == points[j]) continue;

    float dy = points[j].y - points[i].y;
    if (abs(dy) < 0.0001) continue;

    if (((points[i].y > p.y) != (points[j].y > p.y))) {
      float x_intersect = points[i].x + (points[j].x - points[i].x) * (p.y - points[i].y) / dy;
      if (p.x < x_intersect) {
        inside = !inside;
      }
    }
  }
  return inside;
}

void main() {
  vec2 pixel = fragTexCoord * resolution;

  bool inside = isPointInsidePolygon(pixel);

  float dist = distanceToEdge(pixel);

  float aaWidth = 1.0;

  float alpha = inside ?
      min(1.0, dist / aaWidth) :
      max(0.0, 1.0 - dist / aaWidth);

  if (alpha > 0.0) {
    vec4 color;
    if (useGradient) {
      color = getGradientColor(fragTexCoord);
    } else {
      color = fillColor;
    }

    finalColor = vec4(color.rgb, color.a * alpha);
  } else {
    finalColor = vec4(0.0, 0.0, 0.0, 0.0);
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


class ShaderState:
  _instance: Any  = None

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
    }

  def initialize(self):
    if self.initialized:
      return

    vertex_shader = rl.load_shader_from_memory(VERTEX_SHADER, FRAGMENT_SHADER)
    self.shader = vertex_shader

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


def draw_polygon(points: np.ndarray, color=None, gradient=None):
  """
  Draw a complex polygon using shader-based even-odd fill rule

  Args:
      points: List of (x,y) points defining the polygon
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

  # Get shader state singleton
  state = ShaderState.get_instance()

  # Initialize shader if not already done
  if not state.initialized:
    state.initialize()

  # Find bounding box
  min_xy = np.min(points, axis=0)
  min_x, min_y = min_xy
  max_x, max_y = np.max(points, axis=0)

  width = max(1, max_x - min_x)
  height = max(1, max_y - min_y)

  # Transform points to shader space
  transformed_points = points - min_xy

  # Set basic shader uniforms using cached locations
  point_count_ptr = rl.ffi.new("int[]", [len(transformed_points)])
  rl.set_shader_value(state.shader, state.locations['pointCount'], point_count_ptr, rl.ShaderUniformDataType.SHADER_UNIFORM_INT)

  resolution_ptr = rl.ffi.new("float[]", [width, height])
  rl.set_shader_value(state.shader, state.locations['resolution'], resolution_ptr, rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2)

  # Set points
  flat_points = np.ascontiguousarray(transformed_points.flatten().astype(np.float32))
  points_ptr = rl.ffi.cast("float *", flat_points.ctypes.data)
  rl.set_shader_value_v(
    state.shader, state.locations['points'], points_ptr, rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2, len(transformed_points)
  )

  # Set gradient or solid color based on what was provided
  if gradient:
    # Enable gradient
    use_gradient_ptr = rl.ffi.new("int[]", [1])
    rl.set_shader_value(state.shader, state.locations['useGradient'], use_gradient_ptr, rl.ShaderUniformDataType.SHADER_UNIFORM_INT)

    # Set gradient start/end
    start_ptr = rl.ffi.new("float[]", [gradient['start'][0], gradient['start'][1]])
    end_ptr = rl.ffi.new("float[]", [gradient['end'][0], gradient['end'][1]])
    rl.set_shader_value(state.shader, state.locations['gradientStart'], start_ptr, rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2)
    rl.set_shader_value(state.shader, state.locations['gradientEnd'], end_ptr, rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2)

    # Set gradient colors
    colors = gradient['colors']
    color_count = min(len(colors), 8)  # Max 8 colors
    colors_ptr = rl.ffi.new("float[]", color_count * 4)
    for i, c in enumerate(colors[:color_count]):
      colors_ptr[i * 4] = c.r / 255.0
      colors_ptr[i * 4 + 1] = c.g / 255.0
      colors_ptr[i * 4 + 2] = c.b / 255.0
      colors_ptr[i * 4 + 3] = c.a / 255.0
    rl.set_shader_value_v(
      state.shader, state.locations['gradientColors'], colors_ptr, rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4, color_count
    )

    # Set gradient stops
    stops = gradient.get('stops', [i / (color_count - 1) for i in range(color_count)])
    stops_ptr = rl.ffi.new("float[]", color_count)
    for i, s in enumerate(stops[:color_count]):
      stops_ptr[i] = s
    rl.set_shader_value_v(
      state.shader, state.locations['gradientStops'], stops_ptr, rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT, color_count
    )

    # Set color count
    color_count_ptr = rl.ffi.new("int[]", [color_count])
    rl.set_shader_value(state.shader, state.locations['gradientColorCount'], color_count_ptr, rl.ShaderUniformDataType.SHADER_UNIFORM_INT)
  else:
    # Disable gradient
    use_gradient_ptr = rl.ffi.new("int[]", [0])
    rl.set_shader_value(state.shader, state.locations['useGradient'], use_gradient_ptr, rl.ShaderUniformDataType.SHADER_UNIFORM_INT)

    # Set solid color
    if color is None:
      color = rl.WHITE
    fill_color_ptr = rl.ffi.new("float[]", [color.r / 255.0, color.g / 255.0, color.b / 255.0, color.a / 255.0])
    rl.set_shader_value(state.shader, state.locations['fillColor'], fill_color_ptr, rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4)

  # Draw with shader
  rl.begin_shader_mode(state.shader)
  rl.draw_texture_pro(
    state.white_texture,
    rl.Rectangle(0, 0, 2, 2),
    rl.Rectangle(int(min_x), int(min_y), int(width), int(height)),
    rl.Vector2(0, 0),
    0.0,
    rl.WHITE,
  )
  rl.end_shader_mode()


def cleanup_shader_resources():
  state = ShaderState.get_instance()
  state.cleanup()
