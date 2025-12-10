import threading
from dataclasses import dataclass
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.common.swaglog import cloudlog


CONNECTION_RETRY_INTERVAL = 0.2  # seconds between connection attempts


@dataclass
class Frame:
  frame: VisionBuf
  _client: VisionIpcClient  # reference to keep buffers valid


class VisionIpcThread:
  def __init__(self, name: str, stream_type: VisionStreamType):
    self._name = name
    self._stream_type = stream_type
    self._switch_type: VisionStreamType | None = None
    self._client: VisionIpcClient | None = None
    self._frame: Frame | None = None
    self._available_streams: list[VisionStreamType] = []
    self._stop_event = threading.Event()
    self._just_connected = True
    self.lock = threading.Lock()
    self._thread: threading.Thread | None = None
    self.connected: bool = False

  def __del__(self):
    self.stop()

  def get_frame(self) -> Frame | None:
    return self._frame.frame if self._frame else None

  @property
  def available_streams(self) -> list[VisionStreamType]:
    with self.lock:
      return self._available_streams.copy()

  @property
  def stream_type(self) -> VisionStreamType:
    with self.lock:
      return self._stream_type

  def just_connected(self) -> bool:
    if self._just_connected:
      self._just_connected = False
      return True
    return False

  def start(self):
    self._stop_event.clear()
    if self._thread is None or not self._thread.is_alive():
      self._thread = threading.Thread(target=self._thread_func, daemon=True)
      self._thread.start()

  def stop(self):
    self._stop_event.set()
    if self._thread:
      self._thread.join()
      self._thread = None

  def switch_stream(self, stream_type: VisionStreamType) -> None:
    with self.lock:
      if self._stream_type != stream_type:
        self._switch_type = stream_type

  def _thread_func(self):
    client = VisionIpcClient(self._name, self._stream_type, conflate=True)
    while not self._stop_event.is_set():
      with self.lock:
        if self._switch_type is not None:
          cloudlog.debug(f'Switching from {self._stream_type} to {self._switch_type}')
          self._stream_type = self._switch_type
          client = VisionIpcClient(self._name, self._stream_type, conflate=True)
          self._just_connected = True
          self._switch_type = None

      if not self._ensure_connection(client):
        self.connected = False
        self._stop_event.wait(CONNECTION_RETRY_INTERVAL)
        continue

      self.connected = True
      if buffer := client.recv(timeout_ms=20):
        with self.lock:
          self._frame = Frame(buffer, client)

  def _ensure_connection(self, client: VisionIpcClient) -> bool:
    if client.is_connected():
      return True

    # Check if we need to clear the frame before reconnecting
    with self.lock:
      if self._frame and client is self._frame._client:
        self._frame = None

    if not client.connect(False) or not client.num_buffers:
      return False

    cloudlog.debug(f"Connected to {self._name} stream: {self._stream_type}, buffers: {client.num_buffers}")

    with self.lock:
      self._just_connected = True
      self._available_streams = client.available_streams(self._name, block=False)

    return True
