import pyray as rl
from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.system.ui.lib.application import gui_app

FRAME_FRAGMENT_SHADER = """
#version 330 core
in vec2 fragTexCoord; uniform sampler2D texture0, texture1; out vec4 fragColor;
void main() {
  float y = texture(texture0, fragTexCoord).r;
  vec2 uv = texture(texture1, fragTexCoord).ra - 0.5;
  fragColor = vec4(y + 1.402*uv.y, y - 0.344*uv.x - 0.714*uv.y, y + 1.772*uv.x, 1.0);
}"""


class CameraView:
  def __init__(self, name: str, stream_type: VisionStreamType):
    self.client = VisionIpcClient(name, stream_type, False)
    self.shader = rl.load_shader_from_memory(rl.ffi.NULL, FRAME_FRAGMENT_SHADER)
    self.texture_y: rl.Texture | None = None
    self.texture_uv: rl.Texture | None = None
    self.frame = None

  def close(self):
    self._clear_textures()
    if self.shader and self.shader.id:
      rl.unload_shader(self.shader)

  def render(self, rect: rl.Rectangle):
    if not self._ensure_connection():
      return

    buffer = self.client.recv(timeout_ms=0)
    self.frame = buffer if buffer else self.frame
    if not self.frame or not self.texture_y or not self.texture_uv:
      return

    y_data = self.frame.data[: self.frame.uv_offset]
    uv_data = self.frame.data[self.frame.uv_offset :]

    rl.update_texture(self.texture_y, rl.ffi.cast("void *", y_data.ctypes.data))
    rl.update_texture(self.texture_uv, rl.ffi.cast("void *", uv_data.ctypes.data))

    # Calculate scaling to maintain aspect ratio
    scale = min(rect.width / self.frame.width, rect.height / self.frame.height)
    x_offset = rect.x + (rect.width - (self.frame.width * scale)) / 2
    y_offset = rect.y + (rect.height - (self.frame.height * scale)) / 2
    src_rect = rl.Rectangle(0, 0, float(self.frame.width), float(self.frame.height))
    dst_rect = rl.Rectangle(x_offset, y_offset, self.frame.width * scale, self.frame.height * scale)

    rl.begin_shader_mode(self.shader)
    rl.set_shader_value_texture(self.shader, rl.get_shader_location(self.shader, "texture1"), self.texture_uv)
    rl.draw_texture_pro(self.texture_y, src_rect, dst_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)
    rl.end_shader_mode()

  def _ensure_connection(self) -> bool:
    if not self.client.is_connected():
      self.frame = None
      if not self.client.connect(False) or not self.client.num_buffers:
        return False

      self._clear_textures()
      self.texture_y = rl.load_texture_from_image(rl.Image(None, int(self.client.stride),
        int(self.client.height), 1, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE))
      self.texture_uv = rl.load_texture_from_image(rl.Image(None, int(self.client.stride // 2),
        int(self.client.height // 2), 1, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA))
    return True

  def _clear_textures(self):
    if self.texture_y and self.texture_y.id:
      rl.unload_texture(self.texture_y)
    if self.texture_uv and self.texture_uv.id:
      rl.unload_texture(self.texture_uv)

if __name__ == "__main__":
  gui_app.init_window("watch3")
  road_camera_view = CameraView("camerad", VisionStreamType.VISION_STREAM_ROAD)
  driver_camera_view = CameraView("camerad", VisionStreamType.VISION_STREAM_DRIVER)
  wide_road_camera_view = CameraView("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD)
  try:
    for _ in gui_app.render():
      road_camera_view.render(rl.Rectangle(gui_app.width // 4, 0, gui_app.width // 2, gui_app.height // 2))
      driver_camera_view.render(rl.Rectangle(0, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2))
      wide_road_camera_view.render(rl.Rectangle(gui_app.width // 2, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2))
  finally:
    road_camera_view.close()
    driver_camera_view.close()
    wide_road_camera_view.close()
