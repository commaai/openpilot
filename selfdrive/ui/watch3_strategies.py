#!/usr/bin/env python3
import platform
import pyray as rl

from msgq.visionipc import VisionStreamType, VisionIpcClient
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.hardware import TICI
from openpilot.selfdrive.ui.onroad.cameraview import CameraView, VERTEX_SHADER, VERSION
from openpilot.system.ui.widgets.label import gui_label


# Different shader strategies to test
SHADERS = {}

# Strategy 1: Just brightness multiplier (simple)
if TICI:
  SHADERS["1: Brightness only"] = """
    #version 300 es
    #extension GL_OES_EGL_image_external_essl3 : enable
    precision mediump float;
    in vec2 fragTexCoord;
    uniform samplerExternalOES texture0;
    out vec4 fragColor;
    uniform float brightness;
    void main() {
      vec4 color = texture(texture0, fragTexCoord);
      color.rgb = pow(color.rgb, vec3(1.0/1.28));
      color.rgb *= brightness;
      fragColor = vec4(color.rgb, color.a);
    }
    """
else:
  SHADERS["1: Brightness only"] = VERSION + """
    in vec2 fragTexCoord;
    uniform sampler2D texture0;
    uniform sampler2D texture1;
    out vec4 fragColor;
    uniform float brightness;
    void main() {
      float y = texture(texture0, fragTexCoord).r;
      vec2 uv = texture(texture1, fragTexCoord).ra - 0.5;
      vec3 rgb = vec3(y + 1.402*uv.y, y - 0.344*uv.x - 0.714*uv.y, y + 1.772*uv.x);
      rgb *= brightness;
      fragColor = vec4(rgb, 1.0);
    }
    """

# Strategy 2: Contrast boost only (no shadow lift)
if TICI:
  SHADERS["2: Contrast only"] = """
    #version 300 es
    #extension GL_OES_EGL_image_external_essl3 : enable
    precision mediump float;
    in vec2 fragTexCoord;
    uniform samplerExternalOES texture0;
    out vec4 fragColor;
    uniform float brightness;
    void main() {
      vec4 color = texture(texture0, fragTexCoord);
      color.rgb = pow(color.rgb, vec3(1.0/1.28));
      if (brightness > 1.0) {
        color.rgb = clamp((color.rgb - 0.5) * brightness + 0.5, 0.0, 1.0);
      } else {
        color.rgb *= brightness;
      }
      fragColor = vec4(color.rgb, color.a);
    }
    """
else:
  SHADERS["2: Contrast only"] = VERSION + """
    in vec2 fragTexCoord;
    uniform sampler2D texture0;
    uniform sampler2D texture1;
    out vec4 fragColor;
    uniform float brightness;
    void main() {
      float y = texture(texture0, fragTexCoord).r;
      vec2 uv = texture(texture1, fragTexCoord).ra - 0.5;
      vec3 rgb = vec3(y + 1.402*uv.y, y - 0.344*uv.x - 0.714*uv.y, y + 1.772*uv.x);
      if (brightness > 1.0) {
        rgb = clamp((rgb - 0.5) * brightness + 0.5, 0.0, 1.0);
      } else {
        rgb *= brightness;
      }
      fragColor = vec4(rgb, 1.0);
    }
    """

# Strategy 3: Gamma adjustment
if TICI:
  SHADERS["3: Gamma adjust"] = """
    #version 300 es
    #extension GL_OES_EGL_image_external_essl3 : enable
    precision mediump float;
    in vec2 fragTexCoord;
    uniform samplerExternalOES texture0;
    out vec4 fragColor;
    uniform float brightness;
    void main() {
      vec4 color = texture(texture0, fragTexCoord);
      color.rgb = pow(color.rgb, vec3(1.0/1.28));
      if (brightness > 1.0) {
        // Gamma curve: lower gamma = brighter
        float gamma = 1.0 / brightness;
        color.rgb = pow(color.rgb, vec3(gamma));
      } else {
        color.rgb *= brightness;
      }
      fragColor = vec4(color.rgb, color.a);
    }
    """
else:
  SHADERS["3: Gamma adjust"] = VERSION + """
    in vec2 fragTexCoord;
    uniform sampler2D texture0;
    uniform sampler2D texture1;
    out vec4 fragColor;
    uniform float brightness;
    void main() {
      float y = texture(texture0, fragTexCoord).r;
      vec2 uv = texture(texture1, fragTexCoord).ra - 0.5;
      vec3 rgb = vec3(y + 1.402*uv.y, y - 0.344*uv.x - 0.714*uv.y, y + 1.772*uv.x);
      if (brightness > 1.0) {
        float gamma = 1.0 / brightness;
        rgb = pow(rgb, vec3(gamma));
      } else {
        rgb *= brightness;
      }
      fragColor = vec4(rgb, 1.0);
    }
    """

# Strategy 4: Shadow lift only
if TICI:
  SHADERS["4: Shadow lift"] = """
    #version 300 es
    #extension GL_OES_EGL_image_external_essl3 : enable
    precision mediump float;
    in vec2 fragTexCoord;
    uniform samplerExternalOES texture0;
    out vec4 fragColor;
    uniform float brightness;
    void main() {
      vec4 color = texture(texture0, fragTexCoord);
      color.rgb = pow(color.rgb, vec3(1.0/1.28));
      if (brightness > 1.0) {
        // Just lift shadows
        color.rgb = color.rgb + (brightness - 1.0) * 0.2;
      } else {
        color.rgb *= brightness;
      }
      fragColor = vec4(color.rgb, color.a);
    }
    """
else:
  SHADERS["4: Shadow lift"] = VERSION + """
    in vec2 fragTexCoord;
    uniform sampler2D texture0;
    uniform sampler2D texture1;
    out vec4 fragColor;
    uniform float brightness;
    void main() {
      float y = texture(texture0, fragTexCoord).r;
      vec2 uv = texture(texture1, fragTexCoord).ra - 0.5;
      vec3 rgb = vec3(y + 1.402*uv.y, y - 0.344*uv.x - 0.714*uv.y, y + 1.772*uv.x);
      if (brightness > 1.0) {
        rgb = rgb + (brightness - 1.0) * 0.2;
      } else {
        rgb *= brightness;
      }
      fragColor = vec4(rgb, 1.0);
    }
    """

# Strategy 5: Contrast + Gamma
if TICI:
  SHADERS["5: Contrast+Gamma"] = """
    #version 300 es
    #extension GL_OES_EGL_image_external_essl3 : enable
    precision mediump float;
    in vec2 fragTexCoord;
    uniform samplerExternalOES texture0;
    out vec4 fragColor;
    uniform float brightness;
    void main() {
      vec4 color = texture(texture0, fragTexCoord);
      color.rgb = pow(color.rgb, vec3(1.0/1.28));
      if (brightness > 1.0) {
        color.rgb = clamp((color.rgb - 0.5) * brightness + 0.5, 0.0, 1.0);
        color.rgb = pow(color.rgb, vec3(0.9));
      } else {
        color.rgb *= brightness;
      }
      fragColor = vec4(color.rgb, color.a);
    }
    """
else:
  SHADERS["5: Contrast+Gamma"] = VERSION + """
    in vec2 fragTexCoord;
    uniform sampler2D texture0;
    uniform sampler2D texture1;
    out vec4 fragColor;
    uniform float brightness;
    void main() {
      float y = texture(texture0, fragTexCoord).r;
      vec2 uv = texture(texture1, fragTexCoord).ra - 0.5;
      vec3 rgb = vec3(y + 1.402*uv.y, y - 0.344*uv.x - 0.714*uv.y, y + 1.772*uv.x);
      if (brightness > 1.0) {
        rgb = clamp((rgb - 0.5) * brightness + 0.5, 0.0, 1.0);
        rgb = pow(rgb, vec3(0.9));
      } else {
        rgb *= brightness;
      }
      fragColor = vec4(rgb, 1.0);
    }
    """

# Strategy 6: Shadow lift + Gamma
if TICI:
  SHADERS["6: Shadow+Gamma"] = """
    #version 300 es
    #extension GL_OES_EGL_image_external_essl3 : enable
    precision mediump float;
    in vec2 fragTexCoord;
    uniform samplerExternalOES texture0;
    out vec4 fragColor;
    uniform float brightness;
    void main() {
      vec4 color = texture(texture0, fragTexCoord);
      color.rgb = pow(color.rgb, vec3(1.0/1.28));
      if (brightness > 1.0) {
        color.rgb = color.rgb + (brightness - 1.0) * 0.15;
        color.rgb = pow(color.rgb, vec3(0.85));
      } else {
        color.rgb *= brightness;
      }
      fragColor = vec4(color.rgb, color.a);
    }
    """
else:
  SHADERS["6: Shadow+Gamma"] = VERSION + """
    in vec2 fragTexCoord;
    uniform sampler2D texture0;
    uniform sampler2D texture1;
    out vec4 fragColor;
    uniform float brightness;
    void main() {
      float y = texture(texture0, fragTexCoord).r;
      vec2 uv = texture(texture1, fragTexCoord).ra - 0.5;
      vec3 rgb = vec3(y + 1.402*uv.y, y - 0.344*uv.x - 0.714*uv.y, y + 1.772*uv.x);
      if (brightness > 1.0) {
        rgb = rgb + (brightness - 1.0) * 0.15;
        rgb = pow(rgb, vec3(0.85));
      } else {
        rgb *= brightness;
      }
      fragColor = vec4(rgb, 1.0);
    }
    """


class StrategyTestCameraView(CameraView):
  """Camera view with custom shader for testing different strategies"""
  def __init__(self, name: str, stream_type: VisionStreamType, shader_code: str, label: str):
    from openpilot.system.ui.widgets import Widget
    # Initialize Widget base class first
    Widget.__init__(self)

    # Don't call CameraView.__init__() - we need to override shader creation
    self._name = name
    self._stream_type = stream_type
    self.label = label
    self.available_streams = []
    self._target_client = None
    self._target_stream_type = None
    self._switching = False
    self._texture_needs_update = True
    self.last_connection_attempt = 0.0

    # Create custom shader
    self.shader = rl.load_shader_from_memory(VERTEX_SHADER, shader_code)
    self._texture1_loc = rl.get_shader_location(self.shader, "texture1") if not TICI else -1
    self._brightness_loc = rl.get_shader_location(self.shader, "brightness")
    self._brightness_val = rl.ffi.new("float[1]", [1.5])  # Use 1.5 for all tests

    self.frame = None
    self.texture_y = None
    self.texture_uv = None
    self.egl_images = {}
    self.egl_texture = None
    self._placeholder_color = None

    self.client = VisionIpcClient(name, stream_type, conflate=True)

    if TICI:
      from openpilot.system.ui.lib.egl import init_egl
      if not init_egl():
        raise RuntimeError("Failed to initialize EGL")
      temp_image = rl.gen_image_color(1, 1, rl.BLACK)
      self.egl_texture = rl.load_texture_from_image(temp_image)
      rl.unload_image(temp_image)


if __name__ == "__main__":
  gui_app.init_window("Driver Camera Strategy Test")

  # Create camera views with different strategies
  cameras = []
  for label, shader_code in SHADERS.items():
    cam = StrategyTestCameraView("camerad", VisionStreamType.VISION_STREAM_DRIVER, shader_code, label)
    cameras.append(cam)

  # Layout: 3x2 grid (6 strategies)
  cols = 3
  rows = 2
  cell_width = gui_app.width // cols
  cell_height = gui_app.height // rows

  for _ in gui_app.render():
    for idx, cam in enumerate(cameras):
      row = idx // cols
      col = idx % cols
      x = col * cell_width
      y = row * cell_height

      rect = rl.Rectangle(x, y, cell_width, cell_height)
      cam.render(rect)

      # Draw label at top of each cell
      label_rect = rl.Rectangle(x + 5, y + 5, cell_width - 10, 50)
      rl.draw_rectangle_rec(label_rect, rl.Color(0, 0, 0, 200))
      gui_label(label_rect, cam.label, font_size=28, color=rl.WHITE,
                alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
                alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE)
