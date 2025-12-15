"""CameraView - displays video frames from openpilot routes."""

from PySide6.QtCore import Qt, QThread, Signal as QtSignal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QVBoxLayout, QLabel, QFrame


class FrameLoaderThread(QThread):
  """Background thread for loading video frames."""

  frameReady = QtSignal(int, object)  # frame_idx, numpy array
  loadComplete = QtSignal(int)  # total frames

  def __init__(self, route: str, camera: str = "fcamera", parent=None):
    super().__init__(parent)
    self.route = route
    self.camera = camera
    self._frame_reader = None
    self._stop_requested = False
    self._fps = 20.0  # Default openpilot camera FPS

  def run(self):
    try:
      from openpilot.tools.lib.framereader import FrameReader
      from openpilot.tools.lib.route import Route

      route = Route(self.route)

      # Get camera paths based on camera type
      if self.camera == "fcamera":
        paths = route.camera_paths()
      elif self.camera == "ecamera":
        paths = route.ecamera_paths()
      elif self.camera == "dcamera":
        paths = route.dcamera_paths()
      else:
        paths = route.camera_paths()

      if not paths:
        return

      # For now, just load the first segment
      self._frame_reader = FrameReader(paths[0])
      total_frames = self._frame_reader.frame_count

      self.loadComplete.emit(total_frames)

    except Exception as e:
      print(f"Error initializing FrameReader: {e}")

  def getFrame(self, frame_idx: int):
    """Request a frame to be loaded."""
    if self._frame_reader is None:
      return

    try:
      frame = self._frame_reader.get(frame_idx, pix_fmt="rgb24")
      if frame is not None:
        self.frameReady.emit(frame_idx, frame[0])
    except Exception as e:
      print(f"Error getting frame {frame_idx}: {e}")

  @property
  def fps(self) -> float:
    return self._fps

  def stop(self):
    self._stop_requested = True


class CameraView(QFrame):
  """Widget for displaying video frames from openpilot routes."""

  def __init__(self, parent=None):
    super().__init__(parent)
    self._route: str = ""
    self._loader_thread: FrameLoaderThread | None = None
    self._total_frames: int = 0
    self._current_frame: int = -1
    self._fps: float = 20.0
    self._duration: float = 0.0

    self._setup_ui()

  def _setup_ui(self):
    self.setFrameShape(QFrame.Shape.StyledPanel)

    layout = QVBoxLayout(self)
    layout.setContentsMargins(0, 0, 0, 0)

    # Frame display label
    self.frame_label = QLabel()
    self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    self.frame_label.setMinimumSize(320, 240)
    self.frame_label.setStyleSheet("background-color: #1a1a1a;")
    self.frame_label.setText("No video loaded")
    layout.addWidget(self.frame_label)

  def loadRoute(self, route: str, camera: str = "fcamera"):
    """Load video from a route."""
    self._route = route

    if self._loader_thread is not None:
      self._loader_thread.stop()
      self._loader_thread.wait()

    self._loader_thread = FrameLoaderThread(route, camera, self)
    self._loader_thread.frameReady.connect(self._on_frame_ready)
    self._loader_thread.loadComplete.connect(self._on_load_complete)
    self._loader_thread.start()

  def _on_load_complete(self, total_frames: int):
    self._total_frames = total_frames
    if self._loader_thread:
      self._fps = self._loader_thread.fps
    self._duration = total_frames / self._fps if self._fps > 0 else 0.0

    # Load first frame
    if total_frames > 0:
      self.seekToFrame(0)

  def _on_frame_ready(self, frame_idx: int, frame_data):
    """Handle frame data received from loader thread."""
    if frame_data is None:
      return

    self._current_frame = frame_idx

    # Convert numpy array to QImage
    try:
      import numpy as np
      if isinstance(frame_data, np.ndarray):
        height, width = frame_data.shape[:2]
        bytes_per_line = 3 * width
        qimg = QImage(frame_data.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale to fit label while keeping aspect ratio
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
          self.frame_label.size(),
          Qt.AspectRatioMode.KeepAspectRatio,
          Qt.TransformationMode.SmoothTransformation
        )
        self.frame_label.setPixmap(scaled)
    except Exception as e:
      print(f"Error displaying frame: {e}")

  def seekToTime(self, time_sec: float):
    """Seek to a specific time in seconds."""
    if self._total_frames == 0:
      return

    frame_idx = int(time_sec * self._fps)
    frame_idx = max(0, min(frame_idx, self._total_frames - 1))
    self.seekToFrame(frame_idx)

  def seekToFrame(self, frame_idx: int):
    """Seek to a specific frame index."""
    if self._loader_thread is None:
      return

    if frame_idx == self._current_frame:
      return

    self._loader_thread.getFrame(frame_idx)

  def stop(self):
    """Stop the frame loader."""
    if self._loader_thread is not None:
      self._loader_thread.stop()
      self._loader_thread.wait()

  @property
  def duration(self) -> float:
    return self._duration

  @property
  def fps(self) -> float:
    return self._fps
