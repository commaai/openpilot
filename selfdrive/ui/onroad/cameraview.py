import platform
import numpy as np
import pyray as rl

from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import TICI
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.egl import init_egl, create_egl_image, destroy_egl_image, bind_egl_image_to_texture, EGLImage
from openpilot.system.ui.widgets import Widget

CONNECTION_RETRY_INTERVAL = 0.2  # seconds between connection attempts

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


class CameraView(Widget):
  def __init__(self, name: str, stream_type: VisionStreamType):
    super().__init__()
    self._name = name
    # Primary stream
    self.client = VisionIpcClient(name, stream_type, conflate=True)
    self._stream_type = stream_type
    self.available_streams: list[VisionStreamType] = []

    # Target stream for switching
    self._target_client: VisionIpcClient | None = None
    self._target_stream_type: VisionStreamType | None = None
    self._switching: bool = False

    self._texture_needs_update = True
    self.last_connection_attempt: float = 0.0
    self.shader = rl.load_shader_from_memory(VERTEX_SHADER, FRAME_FRAGMENT_SHADER)
    self._texture1_loc: int = rl.get_shader_location(self.shader, "texture1") if not TICI else -1

    self.frame: VisionBuf | None = None
    self.texture_y: rl.Texture | None = None
    self.texture_uv: rl.Texture | None = None

    # EGL resources
    self.egl_images: dict[int, EGLImage] = {}
    self.egl_texture: rl.Texture | None = None

    self._placeholder_color: rl.Color | None = None

    # Initialize EGL for zero-copy rendering on TICI
    if TICI:
      if not init_egl():
        raise RuntimeError("Failed to initialize EGL")

      # Create a 1x1 pixel placeholder texture for EGL image binding
      temp_image = rl.gen_image_color(1, 1, rl.BLACK)
      self.egl_texture = rl.load_texture_from_image(temp_image)
      rl.unload_image(temp_image)

  def _set_placeholder_color(self, color: rl.Color):
    """Set a placeholder color to be drawn when no frame is available."""
    self._placeholder_color = color

  def switch_stream(self, stream_type: VisionStreamType) -> None:
    if self._stream_type == stream_type:
      return

    if self._switching and self._target_stream_type == stream_type:
      return

    cloudlog.debug(f'Preparing switch from {self._stream_type} to {stream_type}')

    if self._target_client:
      del self._target_client

    self._target_stream_type = stream_type
    self._target_client = VisionIpcClient(self._name, stream_type, conflate=True)
    self._switching = True

  @property
  def stream_type(self) -> VisionStreamType:
    return self._stream_type

  def close(self) -> None:
    self._clear_textures()

    # Clean up EGL texture
    if TICI and self.egl_texture:
      rl.unload_texture(self.egl_texture)
      self.egl_texture = None

    # Clean up shader
    if self.shader and self.shader.id:
      rl.unload_shader(self.shader)

    self.client = None

  def _calc_frame_matrix(self, rect: rl.Rectangle) -> np.ndarray:
    if not self.frame:
      return np.eye(3)

    # Calculate aspect ratios
    widget_aspect_ratio = rect.width / rect.height
    frame_aspect_ratio = self.frame.width / self.frame.height

    # Calculate scaling factors to maintain aspect ratio
    zx = min(frame_aspect_ratio / widget_aspect_ratio, 1.0)
    zy = min(widget_aspect_ratio / frame_aspect_ratio, 1.0)

    return np.array([
      [zx, 0.0, 0.0],
      [0.0, zy, 0.0],
      [0.0, 0.0, 1.0]
    ])

  def _render(self, rect: rl.Rectangle):
    if self._switching:
      self._handle_switch()

    if not self._ensure_connection():
      self._draw_placeholder(rect)
      return

    # Try to get a new buffer without blocking
    buffer = self.client.recv(timeout_ms=0)
    if buffer:
      self._texture_needs_update = True
      self.frame = buffer

    if not self.frame:
      self._draw_placeholder(rect)
      return

    transform = self._calc_frame_matrix(rect)
    src_rect = rl.Rectangle(0, 0, float(self.frame.width), float(self.frame.height))

    # Calculate scale
    scale_x = rect.width * transform[0, 0]  # zx
    scale_y = rect.height * transform[1, 1]  # zy

    # Calculate base position (centered)
    x_offset = rect.x + (rect.width - scale_x) / 2
    y_offset = rect.y + (rect.height - scale_y) / 2

    x_offset += transform[0, 2] * rect.width / 2
    y_offset += transform[1, 2] * rect.height / 2

    dst_rect = rl.Rectangle(x_offset, y_offset, scale_x, scale_y)

    # Render with appropriate method
    if TICI:
      self._render_egl(src_rect, dst_rect)
    else:
      self._render_textures(src_rect, dst_rect)

  def _draw_placeholder(self, rect: rl.Rectangle):
    if self._placeholder_color:
      rl.draw_rectangle_rec(rect, self._placeholder_color)

  def _render_egl(self, src_rect: rl.Rectangle, dst_rect: rl.Rectangle) -> None:
    """Render using EGL for direct buffer access"""
    if self.frame is None or self.egl_texture is None:
      return

    idx = self.frame.idx
    egl_image = self.egl_images.get(idx)

    # Create EGL image if needed
    if egl_image is None:
      egl_image = create_egl_image(self.frame.width, self.frame.height, self.frame.stride, self.frame.fd, self.frame.uv_offset)
      if egl_image:
        self.egl_images[idx] = egl_image
      else:
        return

    # Update texture dimensions to match current frame
    self.egl_texture.width = self.frame.width
    self.egl_texture.height = self.frame.height

    # Bind the EGL image to our texture
    bind_egl_image_to_texture(self.egl_texture.id, egl_image)

    # Render with shader
    rl.begin_shader_mode(self.shader)
    rl.draw_texture_pro(self.egl_texture, src_rect, dst_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)
    rl.end_shader_mode()

  def _render_textures(self, src_rect: rl.Rectangle, dst_rect: rl.Rectangle) -> None:
    """Render using texture copies"""
    if not self.texture_y or not self.texture_uv or self.frame is None:
      return

    # Update textures with new frame data
    if self._texture_needs_update:
      y_data = self.frame.data[: self.frame.uv_offset]
      uv_data = self.frame.data[self.frame.uv_offset:]

      rl.update_texture(self.texture_y, rl.ffi.cast("void *", y_data.ctypes.data))
      rl.update_texture(self.texture_uv, rl.ffi.cast("void *", uv_data.ctypes.data))
      self._texture_needs_update = False

    # Render with shader
    rl.begin_shader_mode(self.shader)
    rl.set_shader_value_texture(self.shader, self._texture1_loc, self.texture_uv)
    rl.draw_texture_pro(self.texture_y, src_rect, dst_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)
    rl.end_shader_mode()

  def _ensure_connection(self) -> bool:
    if not self.client.is_connected():
      self.frame = None
      self.available_streams.clear()

      # Throttle connection attempts
      current_time = rl.get_time()
      if current_time - self.last_connection_attempt < CONNECTION_RETRY_INTERVAL:
        return False
      self.last_connection_attempt = current_time

      if not self.client.connect(False) or not self.client.num_buffers:
        return False

      cloudlog.debug(f"Connected to {self._name} stream: {self._stream_type}, buffers: {self.client.num_buffers}")
      self._initialize_textures()
      self.available_streams = self.client.available_streams(self._name, block=False)

    return True

  def _handle_switch(self) -> None:
    """Check if target stream is ready and switch immediately."""
    if not self._target_client or not self._switching:
      return

    # Try to connect target if needed
    if not self._target_client.is_connected():
      if not self._target_client.connect(False) or not self._target_client.num_buffers:
        return

      cloudlog.debug(f"Target stream connected: {self._target_stream_type}")

    # Check if target has frames ready
    target_frame = self._target_client.recv(timeout_ms=0)
    if target_frame:
      self.frame = target_frame  # Update current frame to target frame
      self._complete_switch()

  def _complete_switch(self) -> None:
    """Instantly switch to target stream."""
    cloudlog.debug(f"Switching to {self._target_stream_type}")
    # Clean up current resources
    if self.client:
      del self.client

    # Switch to target
    self.client = self._target_client
    self._stream_type = self._target_stream_type
    self._texture_needs_update = True

    # Reset state
    self._target_client = None
    self._target_stream_type = None
    self._switching = False

    # Initialize textures for new stream
    self._initialize_textures()

  def _initialize_textures(self):
      self._clear_textures()
      if not TICI:
        self.texture_y = rl.load_texture_from_image(rl.Image(None, int(self.client.stride),
          int(self.client.height), 1, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE))
        self.texture_uv = rl.load_texture_from_image(rl.Image(None, int(self.client.stride // 2),
          int(self.client.height // 2), 1, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA))

  def _clear_textures(self):
    if self.texture_y and self.texture_y.id:
      rl.unload_texture(self.texture_y)
      self.texture_y = None

    if self.texture_uv and self.texture_uv.id:
      rl.unload_texture(self.texture_uv)
      self.texture_uv = None

    # Clean up EGL resources
    if TICI:
      for data in self.egl_images.values():
        destroy_egl_image(data)
      self.egl_images = {}


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
