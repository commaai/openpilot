import numpy as np
import raylib_nb as rl

from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcClient, VisionBuf
from openpilot.common.swaglog import cloudlog
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget
from openpilot.selfdrive.ui.ui_state import ui_state

CONNECTION_RETRY_INTERVAL = 0.2  # seconds between connection attempts


def color_curve(curve) -> np.ndarray:
  """a per channel curve as rl.cpu.nv12's lut, for channel values -512 to 767: they aren't clamped before the curve"""
  values = (np.arange(1280) - 512) / 255
  with np.errstate(invalid="ignore"):
    out = curve(values)
  out = np.clip(np.nan_to_num(out, nan=0.0), 0, 1)
  return np.round(out * 255).astype(np.uint8)


# a gamma for the device's display
def _display_gamma(c):
  return c ** (1 / 1.28)


FRAME_COLOR = (color_curve(_display_gamma), 1.0)


class CameraView(Widget):
  def __init__(self, name: str, stream_type: VisionStreamType):
    super().__init__()
    self._name = name
    # Reuse subscribers: destroying a VisionIpcClient doesn't reclaim its msgq reader slot.
    self._clients = {stream: VisionIpcClient(name, stream, conflate=True) for stream in VisionStreamType}
    self.client = self._clients[stream_type]
    self._stream_type: VisionStreamType = stream_type
    self.available_streams: set[VisionStreamType] = set()

    # Target stream for switching
    self._target_client: VisionIpcClient | None = None
    self._target_stream_type: VisionStreamType | None = None
    self._switching: bool = False

    self._new_frame = True
    self.last_connection_attempt: float = 0.0

    self.frame: VisionBuf | None = None
    self._placeholder_color: rl.Color | None = None

    ui_state.add_offroad_transition_callback(self._offroad_transition)

  def _offroad_transition(self):
    # Discard queued frames from the previous drive without recreating subscribers.
    if ui_state.is_onroad() and self.frame is not None:
      self.frame = None
      for client in self._clients.values():
        client.recv(timeout_ms=0)
      self._target_client = None
      self._target_stream_type = None
      self._switching = False

  def _set_placeholder_color(self, color: rl.Color):
    """Set a placeholder color to be drawn when no frame is available."""
    self._placeholder_color = color

  def switch_stream(self, stream_type: VisionStreamType) -> None:
    if self._stream_type == stream_type:
      return

    if self._switching and self._target_stream_type == stream_type:
      return

    cloudlog.debug(f'Preparing switch from {self._stream_type} to {stream_type}')

    self._target_stream_type = stream_type
    self._target_client = self._clients[stream_type]
    self._switching = True

  @property
  def stream_type(self) -> VisionStreamType:
    return self._stream_type

  def close(self) -> None:
    self.frame = None
    self.available_streams.clear()
    self.client = None
    self._target_client = None
    self._clients.clear()

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
    if self._switching:
      self._handle_switch()

    if not self._ensure_connection():
      self._draw_placeholder(rect)
      return

    # Try to get a new buffer without blocking
    buffer = self.client.recv(timeout_ms=0)
    if buffer:
      self._new_frame = True
      self.frame = buffer
    elif not self.client.is_connected():
      # ensure we clear the displayed frame when the connection is lost
      self.frame = None

    if not self.frame:
      self._draw_placeholder(rect)
      return

    transform = self._calc_frame_matrix(rect)
    src_rect = rl.Rectangle(0, 0, float(self.frame.width), float(self.frame.height))
    # Flip cabin camera horizontally
    if self._stream_type == VisionStreamType.VISION_STREAM_CABIN:
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

    curve, saturation = self._frame_color()
    rl.cpu.target_raylib()
    rl.cpu.nv12(self.frame.data, self.frame.uv_offset, self.frame.stride, self.frame.width, self.frame.height, dst_rect,
                src_rect.width < 0, curve, round(saturation * 256), self._new_frame)
    self._new_frame = False

  def _draw_placeholder(self, rect: rl.Rectangle):
    if self._placeholder_color:
      rl.draw_rectangle_rec(rect, self._placeholder_color)

  def _frame_color(self) -> tuple[np.ndarray, float]:
    """the frame's per channel curve and saturation"""
    return FRAME_COLOR

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
      self.available_streams = {VisionStreamType(s) for s in self.client.available_streams(self._name, block=False)}

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
    # Switch to target
    assert self._target_client is not None and self._target_stream_type is not None
    self.client = self._target_client
    self._stream_type = self._target_stream_type
    self._new_frame = True

    # Reset state
    self._target_client = None
    self._target_stream_type = None
    self._switching = False


if __name__ == "__main__":
  gui_app.init_window("camera view")
  road = CameraView("camerad", VisionStreamType.VISION_STREAM_NARROW_ROAD)
  for _ in gui_app.render():
    road.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
