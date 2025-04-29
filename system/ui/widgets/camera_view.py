import threading
import time
import pyray as rl

from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.system.camerad.snapshot.snapshot import extract_image
from openpilot.system.ui.lib.application import gui_app


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
    self.rgb_texture_rect: rl.Rectangle | None = None

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
      self.rgb_texture_rect = rl.Rectangle(0, 0, width, height)
      rl.unload_image(img_rgb)
      rl.set_texture_filter(self.rgb_texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
      print(f"[{self.stream_type}] Created RGB texture {width}x{height}")

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

  def render(self, dest: rl.Rectangle):
    self.update_frame()

    if self.rgb_texture is None or self.rgb_texture_rect is None:
      rl.draw_text_ex(gui_app.font(), "Connecting...", rl.Vector2(dest.x + dest.width / 2 - 100, dest.y + dest.height / 2), 40, 0, rl.WHITE)
      return

    rl.draw_texture_pro(self.rgb_texture, self.rgb_texture_rect, dest, rl.vector2_zero(), 0.0, rl.WHITE)

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

  road_cam_rect = rl.Rectangle(gui_app.width // 4, 0, gui_app.width // 2, gui_app.height // 2)
  wide_road_cam_rect = rl.Rectangle(0, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2)
  driver_cam_rect = rl.Rectangle(gui_app.width // 2, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2)

  for _ in gui_app.render():
    road_cam_view.render(road_cam_rect)
    wide_road_cam_view.render(wide_road_cam_rect)
    driver_cam_view.render(driver_cam_rect)

  road_cam_view.close()
  wide_road_cam_view.close()
  driver_cam_view.close()
