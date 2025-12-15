"""VideoWidget - timeline slider, playback controls, and camera view."""

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
  QVBoxLayout,
  QHBoxLayout,
  QSlider,
  QLabel,
  QPushButton,
  QStyle,
  QFrame,
  QComboBox,
)

from openpilot.tools.cabana.pycabana.widgets.camera import CameraView


class TimelineSlider(QSlider):
  """Timeline slider with millisecond precision."""

  def __init__(self, parent=None):
    super().__init__(Qt.Orientation.Horizontal, parent)
    self._factor = 1000.0  # Store time in milliseconds
    self.setRange(0, 0)

  def currentSecond(self) -> float:
    return self.value() / self._factor

  def setCurrentSecond(self, sec: float):
    self.setValue(int(sec * self._factor))

  def setTimeRange(self, min_sec: float, max_sec: float):
    self.setRange(int(min_sec * self._factor), int(max_sec * self._factor))


class VideoWidget(QFrame):
  """Widget with camera view, timeline slider and playback controls."""

  seeked = Signal(float)  # Emitted when user seeks to a time (seconds)

  def __init__(self, parent=None):
    super().__init__(parent)
    self._duration = 0.0
    self._current_time = 0.0
    self._playing = False
    self._playback_speed = 1.0
    self._route: str = ""

    self._setup_ui()
    self._connect_signals()

    # Playback timer
    self._playback_timer = QTimer(self)
    self._playback_timer.setInterval(33)  # ~30fps
    self._playback_timer.timeout.connect(self._on_playback_tick)

  def _setup_ui(self):
    self.setFrameShape(QFrame.Shape.StyledPanel)
    layout = QVBoxLayout(self)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(8)

    # Camera selector and view
    camera_header = QHBoxLayout()
    camera_header.addWidget(QLabel("Camera:"))
    self.camera_combo = QComboBox()
    self.camera_combo.addItems(["Road Camera", "Wide Camera", "Driver Camera"])
    self.camera_combo.setCurrentIndex(0)
    camera_header.addWidget(self.camera_combo)
    camera_header.addStretch()
    layout.addLayout(camera_header)

    # Camera view
    self.camera_view = CameraView()
    self.camera_view.setMinimumHeight(200)
    layout.addWidget(self.camera_view, 1)

    # Time display
    time_layout = QHBoxLayout()
    self.time_label = QLabel("0:00.0 / 0:00.0")
    self.time_label.setStyleSheet("font-family: monospace; font-size: 12px;")
    time_layout.addWidget(self.time_label)
    time_layout.addStretch()

    self.speed_label = QLabel("1.0x")
    self.speed_label.setStyleSheet("color: gray;")
    time_layout.addWidget(self.speed_label)
    layout.addLayout(time_layout)

    # Timeline slider
    self.slider = TimelineSlider()
    self.slider.setMinimumHeight(20)
    layout.addWidget(self.slider)

    # Playback controls
    controls_layout = QHBoxLayout()
    controls_layout.setSpacing(4)

    self.play_btn = QPushButton()
    self.play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
    self.play_btn.setFixedSize(32, 32)
    controls_layout.addWidget(self.play_btn)

    self.stop_btn = QPushButton()
    self.stop_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
    self.stop_btn.setFixedSize(32, 32)
    controls_layout.addWidget(self.stop_btn)

    controls_layout.addSpacing(16)

    # Speed controls
    self.slower_btn = QPushButton("-")
    self.slower_btn.setFixedSize(24, 24)
    self.slower_btn.setToolTip("Slower")
    controls_layout.addWidget(self.slower_btn)

    self.faster_btn = QPushButton("+")
    self.faster_btn.setFixedSize(24, 24)
    self.faster_btn.setToolTip("Faster")
    controls_layout.addWidget(self.faster_btn)

    controls_layout.addStretch()
    layout.addLayout(controls_layout)

  def _connect_signals(self):
    self.slider.sliderMoved.connect(self._on_slider_moved)
    self.slider.sliderPressed.connect(self._on_slider_pressed)
    self.slider.sliderReleased.connect(self._on_slider_released)
    self.play_btn.clicked.connect(self._toggle_playback)
    self.stop_btn.clicked.connect(self._stop_playback)
    self.slower_btn.clicked.connect(self._decrease_speed)
    self.faster_btn.clicked.connect(self._increase_speed)
    self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)

  def loadRoute(self, route: str):
    """Load video from a route."""
    self._route = route
    camera = self._get_camera_name()
    self.camera_view.loadRoute(route, camera)

  def _get_camera_name(self) -> str:
    """Get the camera name from combo box selection."""
    idx = self.camera_combo.currentIndex()
    return ["fcamera", "ecamera", "dcamera"][idx]

  def _on_camera_changed(self, index: int):
    """Handle camera selection change."""
    if self._route:
      camera = self._get_camera_name()
      self.camera_view.loadRoute(self._route, camera)

  def setDuration(self, duration: float):
    """Set the total duration in seconds."""
    self._duration = duration
    self.slider.setTimeRange(0, duration)
    self._update_time_display()

  def setCurrentTime(self, time: float):
    """Set the current playback time."""
    self._current_time = time
    if not self.slider.isSliderDown():
      self.slider.setCurrentSecond(time)
    self._update_time_display()
    # Update camera frame
    self.camera_view.seekToTime(time)

  def _update_time_display(self):
    current = self._format_time(self._current_time)
    total = self._format_time(self._duration)
    self.time_label.setText(f"{current} / {total}")

  def _format_time(self, seconds: float) -> str:
    """Format seconds as M:SS.s"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:04.1f}"

  def _on_slider_moved(self, value):
    time = value / self.slider._factor
    self._current_time = time
    self._update_time_display()

  def _on_slider_pressed(self):
    self._was_playing = self._playing
    if self._playing:
      self._playback_timer.stop()

  def _on_slider_released(self):
    time = self.slider.currentSecond()
    self._current_time = time
    self.camera_view.seekToTime(time)
    self.seeked.emit(time)
    if self._was_playing:
      self._playback_timer.start()

  def _toggle_playback(self):
    if self._playing:
      self._pause()
    else:
      self._play()

  def _play(self):
    self._playing = True
    self.play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
    self._playback_timer.start()

  def _pause(self):
    self._playing = False
    self.play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
    self._playback_timer.stop()

  def _stop_playback(self):
    self._pause()
    self._current_time = 0
    self.slider.setCurrentSecond(0)
    self._update_time_display()
    self.camera_view.seekToTime(0)
    self.seeked.emit(0)

  def _on_playback_tick(self):
    if self._current_time >= self._duration:
      self._pause()
      return

    # Advance time based on playback speed
    dt = self._playback_timer.interval() / 1000.0 * self._playback_speed
    self._current_time = min(self._current_time + dt, self._duration)
    self.slider.setCurrentSecond(self._current_time)
    self._update_time_display()
    self.camera_view.seekToTime(self._current_time)
    self.seeked.emit(self._current_time)

  def _decrease_speed(self):
    speeds = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    idx = 0
    for i, s in enumerate(speeds):
      if s >= self._playback_speed:
        idx = max(0, i - 1)
        break
    self._playback_speed = speeds[idx]
    self.speed_label.setText(f"{self._playback_speed}x")

  def _increase_speed(self):
    speeds = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    idx = len(speeds) - 1
    for i, s in enumerate(speeds):
      if s > self._playback_speed:
        idx = i
        break
    self._playback_speed = speeds[idx]
    self.speed_label.setText(f"{self._playback_speed}x")

  @property
  def isPlaying(self) -> bool:
    return self._playing
