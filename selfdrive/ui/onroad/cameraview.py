import time
import platform
import threading
import numpy as np
import pyray as rl

from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import TICI
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.egl import init_egl, create_egl_image, destroy_egl_image, bind_egl_image_to_texture, EGLImage
from openpilot.system.ui.widgets import Widget
from openpilot.selfdrive.ui.ui_state import ui_state

# CONNECTION_RETRY_INTERVAL = 0.2  # seconds between connection attempts
FRAME_BUFFER_SIZE = 5

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
    self.client: None | VisionIpcClient = None
    self._stream_type = stream_type
    self._requested_stream_type = stream_type
    self.available_streams: list[VisionStreamType] = []

    self._texture_needs_update = True
    self.last_connection_attempt: float = 0.0
    self.shader = rl.load_shader_from_memory(VERTEX_SHADER, FRAME_FRAGMENT_SHADER)
    self._texture1_loc: int = rl.get_shader_location(self.shader, "texture1") if not TICI else -1

    self.frame: VisionBuf | None = None
    self._frames: list[tuple[int, VisionBuf]] = []
    self.texture_y: rl.Texture | None = None
    self.texture_uv: rl.Texture | None = None

    # EGL resources
    self.egl_images: dict[int, EGLImage] = {}
    self.egl_texture: rl.Texture | None = None

    self._placeholder_color: rl.Color | None = None

    # VIPC thread
    self._vipc_thread_running = True
    self._vipc_thread = None
    self._is_vipc_thread_connected = threading.Event()  # current connection state
    self._vipc_thread_connected_event = threading.Event()  # rising edge of connection
    self._vipc_thread_lock = threading.Lock()

    # Initialize EGL for zero-copy rendering on TICI
    if TICI:
      if not init_egl():
        raise RuntimeError("Failed to initialize EGL")

      # Create a 1x1 pixel placeholder texture for EGL image binding
      temp_image = rl.gen_image_color(1, 1, rl.BLACK)
      self.egl_texture = rl.load_texture_from_image(temp_image)
      rl.unload_image(temp_image)

    ui_state.add_offroad_transition_callback(self._offroad_transition)

  def show_event(self):
    if self._vipc_thread is None:
      self._vipc_thread = threading.Thread(target=self._vipc_thread_loop, daemon=True)
      self._vipc_thread.start()

  def _vipc_thread_loop(self):
    """
    void CameraWidget::vipcThread() {
      VisionStreamType cur_stream = requested_stream_type;
      std::unique_ptr<VisionIpcClient> vipc_client;
      VisionIpcBufExtra meta_main = {0};

      while (!QThread::currentThread()->isInterruptionRequested()) {
        if (!vipc_client || cur_stream != requested_stream_type) {
          clearFrames();
          qDebug().nospace() << "connecting to stream " << requested_stream_type << ", was connected to " << cur_stream;
          cur_stream = requested_stream_type;
          vipc_client.reset(new VisionIpcClient(stream_name, cur_stream, false));
        }
        active_stream_type = cur_stream;

        if (!vipc_client->connected) {
          clearFrames();
          auto streams = VisionIpcClient::getAvailableStreams(stream_name, false);
          if (streams.empty()) {
            QThread::msleep(100);
            continue;
          }
          emit vipcAvailableStreamsUpdated(streams);

          if (!vipc_client->connect(false)) {
            QThread::msleep(100);
            continue;
          }
          emit vipcThreadConnected(vipc_client.get());
        }

        if (VisionBuf *buf = vipc_client->recv(&meta_main, 1000)) {
          {
            std::lock_guard lk(frame_lock);
            frames.push_back(std::make_pair(meta_main.frame_id, buf));
            while (frames.size() > FRAME_BUFFER_SIZE) {
              frames.pop_front();
            }
          }
          emit vipcThreadFrameReceived();
        } else {
          if (!isVisible()) {
            vipc_client->connected = false;
          }
        }
      }
    }
    """

    # cur_stream = self._stream_type
    # vipc_client: VisionIpcClient | None = None
    # meta_main = None

    while self._vipc_thread_running:
      if self.client is None or self._stream_type != self._requested_stream_type:
        # self.frame = None
        cloudlog.debug(f"Connecting to stream {self._requested_stream_type}, was connected to {self._stream_type}")
        self._stream_type = self._requested_stream_type
        with self._vipc_thread_lock:
          print('CREATING NEW CLIENT')
          self.client = VisionIpcClient(self._name, self._stream_type, conflate=True)

      if not self.client.is_connected():
        with self._vipc_thread_lock:
          # self.frame = None
          streams = self.client.available_streams(self._name, block=False)
        if not streams:
          time.sleep(0.1)
          continue

        # TODO: don't need to do any blocking since user only reads, right?
        # TODO: or threading.Event with argument passed to slot?
        self.available_streams = streams

        # VisionIpcClient::connect is not thread safe, guard with lock
        with self._vipc_thread_lock:
          if not self.client.connect(False):
            time.sleep(0.1)
            continue

        # TODO: this in main thread!
        # self._emit_vipc_connected()
        print('CONNECTED TO NEW VIPC!')
        self._is_vipc_thread_connected.set()  # to draw placeholder
        self._vipc_thread_connected_event.set()  # to set up textures

      if self.client.is_connected():
        time.sleep(0.1)

  def _offroad_transition(self):
    # Reconnect if not first time going onroad
    if ui_state.is_onroad() and self.frame is not None:
      # Prevent old frames from showing when going onroad. Qt has a separate thread
      # which drains the VisionIpcClient SubSocket for us. Re-connecting is not enough
      # and only clears internal buffers, not the message queue.
      self.frame = None
      with self._vipc_thread_lock:
        if self.client:
          del self.client
        self.client = VisionIpcClient(self._name, self._stream_type, conflate=True)

  def _set_placeholder_color(self, color: rl.Color):
    """Set a placeholder color to be drawn when no frame is available."""
    self._placeholder_color = color

  def switch_stream(self, stream_type: VisionStreamType) -> None:
    self._requested_stream_type = stream_type

  @property
  def stream_type(self) -> VisionStreamType:
    return self._stream_type

  def close(self) -> None:
    # TODO: decide which order
    self._vipc_thread_running = False
    if self._vipc_thread.is_alive():
      self._vipc_thread.join()

    self._clear_textures()

    # Clean up EGL texture
    if TICI and self.egl_texture:
      rl.unload_texture(self.egl_texture)
      self.egl_texture = None

    # Clean up shader
    if self.shader and self.shader.id:
      rl.unload_shader(self.shader)

    self.client = None

  def __del__(self):
    self.close()

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
    # if self._vipc_thread_connected_event.is_set():
    #   print('   INITIALIZING TEXTURES!!!!')
    #   self._initialize_textures()
    #   self._vipc_thread_connected_event.clear()
    #   # return

    # if not self._ensure_connection():
    if not self._is_vipc_thread_connected.is_set():
      print('DRAWING PLACEHOLDER!!')
      self._draw_placeholder(rect)
      return

    # Try to get a new buffer without blocking
    with self._vipc_thread_lock:
      buffer = self.client.recv(timeout_ms=0)

    if self._vipc_thread_connected_event.is_set() and (buffer or self.frame is None):
      print('   INITIALIZING TEXTURES!!!!')
      self._initialize_textures()
      self._vipc_thread_connected_event.clear()

    if buffer:
      self._texture_needs_update = True
      self.frame = buffer

    if not self.frame:
      self._draw_placeholder(rect)
      print('DRAWING PLACEHOLDER NO FRAME!!')
      return

    transform = self._calc_frame_matrix(rect)
    src_rect = rl.Rectangle(0, 0, float(self.frame.width), float(self.frame.height))
    # Flip driver camera horizontally
    if self._stream_type == VisionStreamType.VISION_STREAM_DRIVER:
      src_rect.width = -src_rect.width

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
      print('NOT RENDERING TEXTURES')
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

  def _initialize_textures(self):
    self._clear_textures()
    print('stride', self.client.stride, 'height', self.client.height)
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
  gui_app.init_window("camera view")
  road = CameraView("camerad", VisionStreamType.VISION_STREAM_ROAD)
  for _ in gui_app.render():
    road.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
