import pyray as rl

from msgq.visionipc import VisionStreamType
from openpilot.system.hardware import TICI
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.mici.onroad.cameraview import BaseCameraView, VERSION


# Choose fragment shader based on platform capabilities
if TICI:
  FRAME_FRAGMENT_SHADER = """
    #version 300 es
    #extension GL_OES_EGL_image_external_essl3 : enable
    precision mediump float;
    in vec2 fragTexCoord;
    uniform samplerExternalOES texture0;
    out vec4 fragColor;
    void main() {
      vec4 color = texture(texture0, fragTexCoord);
      fragColor = vec4(pow(color.rgb, vec3(1.0/1.28)), color.a);
    }
    """
else:
  FRAME_FRAGMENT_SHADER = VERSION + """
    in vec2 fragTexCoord;
    uniform sampler2D texture0;
    uniform sampler2D texture1;
    out vec4 fragColor;
    void main() {
      float y = texture(texture0, fragTexCoord).r;
      vec2 uv = texture(texture1, fragTexCoord).ra - 0.5;
      fragColor = vec4(y + 1.402*uv.y, y - 0.344*uv.x - 0.714*uv.y, y + 1.772*uv.x, 1.0);
    }
    """


class CameraView(BaseCameraView):
  def __init__(self, name: str, stream_type: VisionStreamType):
    super().__init__(name, stream_type, FRAME_FRAGMENT_SHADER)


if __name__ == "__main__":
  gui_app.init_window("camera view")
  road = CameraView("camerad", VisionStreamType.VISION_STREAM_ROAD)
  for _ in gui_app.render():
    road.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
