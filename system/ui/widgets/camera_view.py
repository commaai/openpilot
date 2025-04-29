import threading
import time
import pyray as rl
import numpy as np

from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.system.ui.lib.application import gui_app


def yuv_to_rgb(y, u, v):
  height, width = y.shape

  ul = np.repeat(np.repeat(u, 2, axis=1), 2, axis=0)
  vl = np.repeat(np.repeat(v, 2, axis=1), 2, axis=0)

  ul = ul[:height, :width]
  vl = vl[:height, :width]

  yuv = np.dstack((y, ul, vl)).astype(np.float32)

  yuv[:, :, 1:] -= 128.0

  m = np.array([
    [1.00000, 1.00000, 1.00000],
    [0.00000, -0.39465, 2.03211],
    [1.13983, -0.58060, 0.00000],
  ], dtype=np.float32)

  rgb = np.dot(yuv, m)
  np.clip(rgb, 0, 255, out=rgb)
  return rgb.astype(np.uint8)


def extract_image(buf):
  height, width, stride = buf.height, buf.width, buf.stride
  uv_offset = buf.uv_offset
  data = buf.data

  y_plane = np.frombuffer(data, dtype=np.uint8, count=uv_offset)
  if stride == width:
    y = y_plane.reshape((height, width))
  else:
    y = np.lib.stride_tricks.as_strided(y_plane, shape=(height, width), strides=(stride, 1))

  uv_height = height // 2
  uv_width = width // 2

  uv_plane = np.frombuffer(data, dtype=np.uint8, offset=uv_offset)

  u = np.lib.stride_tricks.as_strided(uv_plane, shape=(uv_height, uv_width), strides=(stride, 2))
  v = np.lib.stride_tricks.as_strided(uv_plane[1:], shape=(uv_height, uv_width), strides=(stride, 2))

  return yuv_to_rgb(y, u, v)


class RaylibCameraView:
  def __init__(self, stream_name: str, stream_type: VisionStreamType, width: int | None = None, height: int | None = None):
    self.stream_name = stream_name
    self.stream_type = stream_type
    self.render_width = width if width is not None else gui_app.width
    self.render_height = height if height is not None else gui_app.height

    # Frame acquisition state
    self.vipc_client: VisionIpcClient | None = None
    self.vipc_thread: threading.Thread | None = None
    self.running = False
    self.latest_frame_lock = threading.Lock()
    self.latest_vision_buf: VisionBuf | None = None
    self.frame_updated = False
    self.connect_retries = 0

    # Rendering state
    self.rgb_texture: rl.Texture | None = None
    self.current_width: int = 0
    self.current_height: int = 0

    self.start_vipc_thread()

  def start_vipc_thread(self):
    if self.vipc_thread is not None and self.vipc_thread.is_alive():
      self.close()

    self.running = True
    self.vipc_thread = threading.Thread(target=self._vipc_thread_run, name=f"vipc_{self.stream_type}")
    self.vipc_thread.daemon = True
    self.vipc_thread.start()

  def _vipc_thread_run(self):
    print(f"Connecting to {self.stream_name} - {self.stream_type}...")
    while self.running:
      try:
        temp_vipc_client = VisionIpcClient(self.stream_name, self.stream_type, True)

        if not temp_vipc_client.connect(False):
          if self.connect_retries % 100 == 0:
            print(f"[{self.stream_type}] VIPC connect failed, retrying...")
          self.connect_retries += 1
          del temp_vipc_client
          time.sleep(0.1)
          continue

        attempts = 0
        while temp_vipc_client.buffer_len is None and attempts < 20 and self.running:
          time.sleep(0.05)
          attempts += 1

        if temp_vipc_client.buffer_len is None or temp_vipc_client.buffer_len <= 0:
          print(f"[{self.stream_type}] VIPC connected, but buffer length is invalid: {temp_vipc_client.buffer_len}. Retrying connection.")
          del temp_vipc_client
          time.sleep(0.5)
          continue

        self.vipc_client = temp_vipc_client
        self.connect_retries = 0
        print(f"[{self.stream_type}] VIPC connected with valid buffers (len: {self.vipc_client.buffer_len})")

        while self.running and self.vipc_client.is_connected():
          buf = self.vipc_client.recv()
          if buf is None:
            if not self.vipc_client.is_connected():
              print(f"[{self.stream_type}] VIPC receive loop detected disconnection")
              break
            time.sleep(0.01)
            continue

          with self.latest_frame_lock:
            self.latest_vision_buf = buf
            self.frame_updated = True

        if self.vipc_client is not None:
          print(f"[{self.stream_type}] Exiting receive loop (running={self.running}, connected={self.vipc_client.is_connected()})")
          self.vipc_client = None

      except Exception as e:
        print(f"Exception in VIPC thread for {self.stream_type}: {e}")
        time.sleep(1)

      if not self.running:
        break

    print(f"Exiting VIPC thread for {self.stream_type}")
    self.vipc_client = None

  def _ensure_texture(self, width: int, height: int):
    if width <= 0 or height <= 0:
      return

    if self.rgb_texture is None or self.rgb_texture.width != width or self.rgb_texture.height != height:
      if self.rgb_texture is not None:
        rl.unload_texture(self.rgb_texture)
      img_rgb = rl.gen_image_color(width, height, rl.BLANK)
      rl.image_format(img_rgb, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8)
      self.rgb_texture = rl.load_texture_from_image(img_rgb)
      rl.unload_image(img_rgb)
      rl.set_texture_filter(self.rgb_texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
      print(f"[{self.stream_type}] Created RGB texture {width}x{height}")

    self.current_width = width
    self.current_height = height

  def update_frame(self):
    buf = None
    with self.latest_frame_lock:
      if self.frame_updated:
        buf = self.latest_vision_buf
        self.frame_updated = False

    if buf is None:
      return False

    rgb_image_np = extract_image(buf)

    if rgb_image_np is None:
      return False

    height, width = rgb_image_np.shape[:2]
    self._ensure_texture(width, height)

    if self.rgb_texture is None:
      print(f"[{self.stream_type}] RGB Texture not ready for update.")
      return False

    try:
      data_ptr = rl.ffi.cast('void *', rgb_image_np.ctypes.data)
      rl.update_texture(self.rgb_texture, data_ptr)
    except Exception as e:
      print(f"[{self.stream_type}] Error updating texture: {e}")
      return False

    return True

  def render(self, x: int, y: int, w: int, h: int):
    frame_updated = self.update_frame()

    if self.rgb_texture is None:
      rl.draw_rectangle(x, y, w, h, rl.DARKGRAY)
      rl.draw_text("Connecting...", x + 20, y + 20, 40, rl.WHITE)
      return

    source_rect = rl.Rectangle(0, 0, float(self.rgb_texture.width), float(self.rgb_texture.height))
    dest_rect = rl.Rectangle(float(x), float(y), float(w), float(h))
    origin = rl.Vector2(0, 0)

    rl.draw_texture_pro(self.rgb_texture, source_rect, dest_rect, origin, 0.0, rl.WHITE)

  def close(self):
    self.running = False
    if self.vipc_thread is not None and self.vipc_thread.is_alive():
      self.vipc_thread.join(timeout=1.0)
      if self.vipc_thread.is_alive():
        print(f"Warning: VIPC thread for {self.stream_type} did not exit cleanly.")
    self.vipc_thread = None


if __name__ == "__main__":
  gui_app.init_window("watch3")
  road_cam_view = RaylibCameraView("camerad", VisionStreamType.VISION_STREAM_ROAD)
  wide_road_cam_view = RaylibCameraView("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD)
  driver_cam_view = RaylibCameraView("camerad", VisionStreamType.VISION_STREAM_DRIVER)

  for _ in gui_app.render():
    road_cam_view.render(gui_app.width // 3, 0, gui_app.width // 2, gui_app.height // 2)
    wide_road_cam_view.render(0, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2)
    driver_cam_view.render(gui_app.width // 2, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2)

  road_cam_view.close()
  driver_cam_view.close()
