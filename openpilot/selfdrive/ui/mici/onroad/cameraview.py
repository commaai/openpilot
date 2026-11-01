import pyray as rl

from openpilot.cereal.visionipc import VisionStreamType
from openpilot.common.hardware import COMMA_HARDWARE
from openpilot.selfdrive.ui.onroad.cameraview import CameraView as BaseCameraView, VERSION
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus

# Choose fragment shader based on platform capabilities
if COMMA_HARDWARE:
  FRAME_FRAGMENT_SHADER = """
    #version 300 es
    #extension GL_OES_EGL_image_external_essl3 : enable
    precision mediump float;
    in vec2 fragTexCoord;
    uniform samplerExternalOES texture0;
    out vec4 fragColor;
    uniform int engaged;
    uniform int enhance_driver;

    void main() {
      vec4 color = texture(texture0, fragTexCoord);
      if (engaged == 1) {
        float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));  // Luma
        color.rgb = mix(vec3(gray), color.rgb, 0.2);  // 20% saturation
        color.rgb = clamp((color.rgb - 0.5) * 1.2 + 0.5, 0.0, 1.0);  // +20% contrast
        color.rgb = pow(color.rgb, vec3(1.0/1.28));
        fragColor = vec4(color.rgb, color.a);
      } else {
        color.rgb *= 0.85;  // 85% opacity
      }
      if (enhance_driver == 1) {
        float brightness = 1.1;
        color.rgb = color.rgb + 0.15;
        color.rgb = clamp((color.rgb - 0.5) * (brightness * 0.8) + 0.5, 0.0, 1.0);
        color.rgb = color.rgb * color.rgb * (3.0 - 2.0 * color.rgb);
        color.rgb = pow(color.rgb, vec3(0.8));
      }
      fragColor = vec4(color.rgb, color.a);
    }
    """
else:
  FRAME_FRAGMENT_SHADER = VERSION + """
    in vec2 fragTexCoord;
    uniform sampler2D texture0;
    uniform sampler2D texture1;
    out vec4 fragColor;
    uniform int engaged;
    uniform int enhance_driver;

    void main() {
      float y = texture(texture0, fragTexCoord).r;
      vec2 uv = texture(texture1, fragTexCoord).ra - 0.5;
      vec3 rgb = vec3(y + 1.402*uv.y, y - 0.344*uv.x - 0.714*uv.y, y + 1.772*uv.x);
      if (engaged == 1) {
        float gray = dot(rgb, vec3(0.299, 0.587, 0.114));
        rgb = mix(vec3(gray), rgb, 0.2);  // 20% saturation
        rgb = clamp((rgb - 0.5) * 1.2 + 0.5, 0.0, 1.0);  // +20% contrast
      } else {
        rgb *= 0.85;  // 85% opacity
      }
      // TODO: the images out of camerad need some more correction and
      // the ui should apply a gamma curve for the device display
      if (enhance_driver == 1) {
        float brightness = 1.1;
        rgb = rgb + 0.15;
        rgb = clamp((rgb - 0.5) * (brightness * 0.8) + 0.5, 0.0, 1.0);
        rgb = rgb * rgb * (3.0 - 2.0 * rgb);
        rgb = pow(rgb, vec3(0.8));
      }
      fragColor = vec4(rgb, 1.0);
    }
    """


class CameraView(BaseCameraView):
  FRAGMENT_SHADER = FRAME_FRAGMENT_SHADER

  def __init__(self, name: str, stream_type: VisionStreamType):
    super().__init__(name, stream_type)
    self._engaged_loc = rl.get_shader_location(self.shader, "engaged")
    self._engaged_val = rl.ffi.new("int[1]", [1])
    self._enhance_driver_loc = rl.get_shader_location(self.shader, "enhance_driver")
    self._enhance_driver_val = rl.ffi.new("int[1]", [1 if stream_type == VisionStreamType.VISION_STREAM_CABIN else 0])

  def _set_shader_uniforms(self) -> None:
    self._engaged_val[0] = 1 if ui_state.status != UIStatus.DISENGAGED else 0
    rl.set_shader_value(self.shader, self._engaged_loc, self._engaged_val, rl.ShaderUniformDataType.SHADER_UNIFORM_INT)
    rl.set_shader_value(self.shader, self._enhance_driver_loc, self._enhance_driver_val, rl.ShaderUniformDataType.SHADER_UNIFORM_INT)
