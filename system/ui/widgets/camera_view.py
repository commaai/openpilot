import threading
import time
import pyray as rl

from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.system.camerad.snapshot.snapshot import extract_image
from openpilot.system.ui.lib.application import gui_app


class CameraView:
  def __init__(self, stream_name: str, stream_type: VisionStreamType):
    self.stream_name = stream_name
    self.stream_type = stream_type

    # Frame buffer and synchronization
    self._lock = threading.Lock()
    self._buf: VisionBuf | None = None
    self._updated = False
    self._retries = 0

    # Texture state
    self._texture: rl.Texture | None = None
    self._texture_rect: rl.Rectangle | None = None

    # IPC client and threading
    self.running = True
    self.vipc_client: VisionIpcClient | None = None
    self.vipc_thread = threading.Thread(target=self._thread_loop, name=f"vipc_{self.stream_type}", daemon=True)
    self.vipc_thread.start()

  def _thread_loop(self):
    print(f"Connecting to {self.stream_name} [{self.stream_type}]...")
    while self.running:
      try:
        client = VisionIpcClient(self.stream_name, self.stream_type, True)
        if not client.connect(False):
          if self._retries % 100 == 0:
            print(f"[{self.stream_type}] Connection failed, retrying...")
          self._retries += 1
          time.sleep(0.1)
          continue

        # wait for valid buffers
        for _ in range(20):
          if client.buffer_len:
            break
          time.sleep(0.05)

        if not client.buffer_len:
          print(f"[{self.stream_type}] Invalid buffer length: {client.buffer_len}")
          time.sleep(0.5)
          continue

        self.vipc_client = client
        self._retries = 0
        print(f"[{self.stream_type}] Connected with {client.buffer_len} buffers")

        # receive loop
        while self.running and client.is_connected():
          buf = client.recv()
          if buf is None:
            if not client.is_connected():
              print(f"[{self.stream_type}] Disconnected")
              break
            time.sleep(0.01)
            continue

          with self._lock:
            self._buf = buf
            self._updated = True

        print(f"[{self.stream_type}] Exiting receive loop")
        self.vipc_client = None

      except Exception as e:
        print(f"[{self.stream_type}] Error: {e}")
        time.sleep(1)

    print(f"VIPC thread for {self.stream_type} stopped")
    self.vipc_client = None

  def _ensure_texture(self, w: int, h: int):
    if w <= 0 or h <= 0:
      return
    if not self._texture or self._texture.width != w or self._texture.height != h:
      if self._texture:
        rl.unload_texture(self._texture)
      img = rl.gen_image_color(w, h, rl.BLANK)
      rl.image_format(img, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8)
      self._texture = rl.load_texture_from_image(img)
      self._texture_rect = rl.Rectangle(0, 0, w, h)
      rl.unload_image(img)
      rl.set_texture_filter(self._texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
      print(f"[{self.stream_type}] Texture {w}x{h} created")

  def update_frame(self) -> bool:
    with self._lock:
      if not self._updated:
        return False
      buf = self._buf
      self._updated = False

    if buf is None:
      return False

    img_np = extract_image(buf)
    if img_np is None:
      return False

    h, w = img_np.shape[:2]
    self._ensure_texture(w, h)
    if not self._texture:
      print(f"[{self.stream_type}] Texture not ready")
      return False

    data = rl.ffi.cast('void *', img_np.ctypes.data)
    rl.update_texture(self._texture, data)
    return True

  def render(self, dest: rl.Rectangle):
    self.update_frame()
    if not self._texture or not self._texture_rect:
      pos = rl.Vector2(dest.x + dest.width / 2 - 100, dest.y + dest.height / 2)
      rl.draw_text_ex(gui_app.font(), "Connecting...", pos, 40, 0, rl.WHITE)
      return
    rl.draw_texture_pro(self._texture, self._texture_rect, dest, rl.Vector2(0, 0), 0.0, rl.WHITE)

  def close(self):
    self.running = False
    if self.vipc_thread:
      self.vipc_thread.join(timeout=1.0)
      if self.vipc_thread.is_alive():
        print(f"[{self.stream_type}] Thread did not exit cleanly")
    self.vipc_thread = None


if __name__ == "__main__":
  gui_app.init_window("watch3")

  views = [
    (VisionStreamType.VISION_STREAM_ROAD, (gui_app.width // 4, 0, gui_app.width // 2, gui_app.height // 2)),
    (VisionStreamType.VISION_STREAM_WIDE_ROAD, (0, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2)),
    (VisionStreamType.VISION_STREAM_DRIVER, (gui_app.width // 2, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2)),
  ]

  camera_views = [CameraView("camerad", t) for t, _ in views]
  rects = [rl.Rectangle(*r) for _, r in views]

  for _ in gui_app.render():
    for view, rect in zip(camera_views, rects, strict=True):
      view.render(rect)

  for view in camera_views:
    view.close()
