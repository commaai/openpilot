import platform
import pyray as rl
from typing import Any
from openpilot.system.ui.lib.application import gui_app

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
uniform float sliderPercentage;
uniform float opacity;
out vec4 finalColor;

void main() {
  vec4 texColor = texture(texture0, fragTexCoord);
  float xPos = fragTexCoord.x;
  float shimmerPos = mod(-time * shimmerSpeed, 1.0 + shimmerWidth);
  float distFromShimmer = abs(xPos - shimmerPos);
  float mask = 1.0 - smoothstep(0.0, shimmerWidth, distFromShimmer);
  vec3 shimmerColor = vec3(1.0, 1.0, 1.0);
  vec3 finalRGB = mix(texColor.rgb, shimmerColor, mask);
  float alphaFade = (1.0 - sliderPercentage) * opacity;
  float finalAlpha = texColor.a * alphaFade;
  finalColor = vec4(finalRGB, finalAlpha) * fragColor;
}
"""

UNIFORM_FLOAT = rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT


class ShimmerShader:
  _instance: Any = None

  @classmethod
  def get_instance(cls):
    if cls._instance is None:
      cls._instance = cls()
      cls._instance.initialize()
    return cls._instance

  def __init__(self):
    if ShimmerShader._instance is not None:
      raise Exception("This class is a singleton. Use get_instance() instead.")

    self.initialized = False
    self.shader = None

    self.locations = {
      'time': None,
      'shimmerWidth': None,
      'shimmerSpeed': None,
      'sliderPercentage': None,
      'opacity': None,
      'mvp': None,
    }

    self.time_ptr = rl.ffi.new("float[]", [0.0])
    self.shimmer_width_ptr = rl.ffi.new("float[]", [0.15])
    self.shimmer_speed_ptr = rl.ffi.new("float[]", [0.6])
    self.slider_percentage_ptr = rl.ffi.new("float[]", [0.0])
    self.opacity_ptr = rl.ffi.new("float[]", [1.0])

  def initialize(self):
    if self.initialized:
      return

    self.shader = rl.load_shader_from_memory(VERTEX_SHADER, SHIMMER_FRAGMENT_SHADER)

    for uniform in self.locations.keys():
      self.locations[uniform] = rl.get_shader_location(self.shader, uniform)

    proj = rl.matrix_ortho(0, gui_app.width, gui_app.height, 0, -1, 1)
    rl.set_shader_value_matrix(self.shader, self.locations['mvp'], proj)
    rl.set_shader_value(self.shader, self.locations['shimmerWidth'], self.shimmer_width_ptr, UNIFORM_FLOAT)
    rl.set_shader_value(self.shader, self.locations['shimmerSpeed'], self.shimmer_speed_ptr, UNIFORM_FLOAT)

    self.initialized = True

  def cleanup(self):
    if not self.initialized:
      return
    if self.shader:
      rl.unload_shader(self.shader)
      self.shader = None

    self.initialized = False

  def set_uniforms(self, time: float, slider_percentage: float, opacity: float):
    if not self.initialized:
      self.initialize()

    self.time_ptr[0] = time
    self.slider_percentage_ptr[0] = slider_percentage
    self.opacity_ptr[0] = opacity

    rl.set_shader_value(self.shader, self.locations['time'], self.time_ptr, UNIFORM_FLOAT)
    rl.set_shader_value(self.shader, self.locations['sliderPercentage'], self.slider_percentage_ptr, UNIFORM_FLOAT)
    rl.set_shader_value(self.shader, self.locations['opacity'], self.opacity_ptr, UNIFORM_FLOAT)


def cleanup_shimmer_shader_resources():
  state = ShimmerShader.get_instance()
  state.cleanup()

