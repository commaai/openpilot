from typing import Optional
from enum import IntEnum
from pathlib import Path

from PySide6.QtCore import QObject, Signal, QSettings, QByteArray, QDir, QStandardPaths
from PySide6.QtWidgets import (
  QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
  QGroupBox, QSpinBox, QComboBox, QLineEdit, QPushButton,
  QDialogButtonBox, QFileDialog
)

# Theme constants
LIGHT_THEME = 1
DARK_THEME = 2

# Cache limits
MIN_CACHE_MINUTES = 30
MAX_CACHE_MINUTES = 120


class DragDirection(IntEnum):
  MsbFirst = 0
  LsbFirst = 1
  AlwaysLE = 2
  AlwaysBE = 3


class Settings(QObject):
  changed = Signal()

  _instance: Optional['Settings'] = None

  def __new__(cls):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance

  def __init__(self):
    if hasattr(self, '_initialized'):
      return
    super().__init__()
    self._initialized = True

    # General settings
    self.absolute_time: bool = False
    self.fps: int = 10
    self.max_cached_minutes: int = 30
    self.chart_height: int = 200
    self.chart_column_count: int = 1
    self.chart_range: int = 3 * 60  # 3 minutes
    self.chart_series_type: int = 0
    self.theme: int = 0
    self.sparkline_range: int = 15  # 15 seconds
    self.multiple_lines_hex: bool = False
    self.log_livestream: bool = True
    self.suppress_defined_signals: bool = False
    self.log_path: str = str(Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.HomeLocation)) / "cabana_live_stream")
    self.last_dir: str = QDir.homePath()
    self.last_route_dir: str = QDir.homePath()
    self.geometry: QByteArray = QByteArray()
    self.video_splitter_state: QByteArray = QByteArray()
    self.window_state: QByteArray = QByteArray()
    self.recent_files: list[str] = []
    self.message_header_state: QByteArray = QByteArray()
    self.drag_direction: DragDirection = DragDirection.MsbFirst

    # Session data
    self.recent_dbc_file: str = ""
    self.active_msg_id: str = ""
    self.selected_msg_ids: list[str] = []
    self.active_charts: list[str] = []

    self._load()

  def _load(self):
    """Load settings from QSettings"""
    s = QSettings("cabana", "cabana")

    self.absolute_time = s.value("absolute_time", self.absolute_time, type=bool)
    self.fps = s.value("fps", self.fps, type=int)
    self.max_cached_minutes = s.value("max_cached_minutes", self.max_cached_minutes, type=int)
    self.chart_height = s.value("chart_height", self.chart_height, type=int)
    self.chart_range = s.value("chart_range", self.chart_range, type=int)
    self.chart_column_count = s.value("chart_column_count", self.chart_column_count, type=int)
    self.last_dir = s.value("last_dir", self.last_dir, type=str)
    self.last_route_dir = s.value("last_route_dir", self.last_route_dir, type=str)
    self.window_state = s.value("window_state", self.window_state, type=QByteArray)
    self.geometry = s.value("geometry", self.geometry, type=QByteArray)
    self.video_splitter_state = s.value("video_splitter_state", self.video_splitter_state, type=QByteArray)
    self.recent_files = s.value("recent_files", self.recent_files, type=list)
    self.message_header_state = s.value("message_header_state", self.message_header_state, type=QByteArray)
    self.chart_series_type = s.value("chart_series_type", self.chart_series_type, type=int)
    self.theme = s.value("theme", self.theme, type=int)
    self.sparkline_range = s.value("sparkline_range", self.sparkline_range, type=int)
    self.multiple_lines_hex = s.value("multiple_lines_hex", self.multiple_lines_hex, type=bool)
    self.log_livestream = s.value("log_livestream", self.log_livestream, type=bool)
    self.log_path = s.value("log_path", self.log_path, type=str)
    self.drag_direction = DragDirection(s.value("drag_direction", int(self.drag_direction), type=int))
    self.suppress_defined_signals = s.value("suppress_defined_signals", self.suppress_defined_signals, type=bool)
    self.recent_dbc_file = s.value("recent_dbc_file", self.recent_dbc_file, type=str)
    self.active_msg_id = s.value("active_msg_id", self.active_msg_id, type=str)
    self.selected_msg_ids = s.value("selected_msg_ids", self.selected_msg_ids, type=list)
    self.active_charts = s.value("active_charts", self.active_charts, type=list)

  def save(self):
    """Save settings to QSettings"""
    s = QSettings("cabana", "cabana")

    s.setValue("absolute_time", self.absolute_time)
    s.setValue("fps", self.fps)
    s.setValue("max_cached_minutes", self.max_cached_minutes)
    s.setValue("chart_height", self.chart_height)
    s.setValue("chart_range", self.chart_range)
    s.setValue("chart_column_count", self.chart_column_count)
    s.setValue("last_dir", self.last_dir)
    s.setValue("last_route_dir", self.last_route_dir)
    s.setValue("window_state", self.window_state)
    s.setValue("geometry", self.geometry)
    s.setValue("video_splitter_state", self.video_splitter_state)
    s.setValue("recent_files", self.recent_files)
    s.setValue("message_header_state", self.message_header_state)
    s.setValue("chart_series_type", self.chart_series_type)
    s.setValue("theme", self.theme)
    s.setValue("sparkline_range", self.sparkline_range)
    s.setValue("multiple_lines_hex", self.multiple_lines_hex)
    s.setValue("log_livestream", self.log_livestream)
    s.setValue("log_path", self.log_path)
    s.setValue("drag_direction", int(self.drag_direction))
    s.setValue("suppress_defined_signals", self.suppress_defined_signals)
    s.setValue("recent_dbc_file", self.recent_dbc_file)
    s.setValue("active_msg_id", self.active_msg_id)
    s.setValue("selected_msg_ids", self.selected_msg_ids)
    s.setValue("active_charts", self.active_charts)

    s.sync()

  def __del__(self):
    """Save settings on destruction"""
    self.save()


class SettingsDlg(QDialog):
  def __init__(self, parent: Optional[QWidget] = None):
    super().__init__(parent)
    self.setWindowTitle(self.tr("Settings"))

    main_layout = QVBoxLayout(self)

    # General settings group
    groupbox = QGroupBox("General")
    form_layout = QFormLayout(groupbox)

    self.theme = QComboBox(self)
    self.theme.setToolTip(self.tr("You may need to restart cabana after changes theme"))
    self.theme.addItems([self.tr("Automatic"), self.tr("Light"), self.tr("Dark")])
    self.theme.setCurrentIndex(settings.theme)
    form_layout.addRow(self.tr("Color Theme"), self.theme)

    self.fps = QSpinBox(self)
    self.fps.setRange(10, 100)
    self.fps.setSingleStep(10)
    self.fps.setValue(settings.fps)
    form_layout.addRow("FPS", self.fps)

    self.cached_minutes = QSpinBox(self)
    self.cached_minutes.setRange(MIN_CACHE_MINUTES, MAX_CACHE_MINUTES)
    self.cached_minutes.setSingleStep(1)
    self.cached_minutes.setValue(settings.max_cached_minutes)
    form_layout.addRow(self.tr("Max Cached Minutes"), self.cached_minutes)

    main_layout.addWidget(groupbox)

    # New Signal Settings group
    groupbox = QGroupBox("New Signal Settings")
    form_layout = QFormLayout(groupbox)

    self.drag_direction = QComboBox(self)
    self.drag_direction.addItems([
      self.tr("MSB First"),
      self.tr("LSB First"),
      self.tr("Always Little Endian"),
      self.tr("Always Big Endian")
    ])
    self.drag_direction.setCurrentIndex(int(settings.drag_direction))
    form_layout.addRow(self.tr("Drag Direction"), self.drag_direction)

    main_layout.addWidget(groupbox)

    # Chart settings group
    groupbox = QGroupBox("Chart")
    form_layout = QFormLayout(groupbox)

    self.chart_height = QSpinBox(self)
    self.chart_height.setRange(100, 500)
    self.chart_height.setSingleStep(10)
    self.chart_height.setValue(settings.chart_height)
    form_layout.addRow(self.tr("Chart Height"), self.chart_height)

    main_layout.addWidget(groupbox)

    # Live stream logging group
    self.log_livestream = QGroupBox(self.tr("Enable live stream logging"), self)
    self.log_livestream.setCheckable(True)
    self.log_livestream.setChecked(settings.log_livestream)
    path_layout = QHBoxLayout(self.log_livestream)

    self.log_path = QLineEdit(settings.log_path, self)
    self.log_path.setReadOnly(True)
    path_layout.addWidget(self.log_path)

    browse_btn = QPushButton(self.tr("B&rowse..."))
    browse_btn.clicked.connect(self._browse_log_path)
    path_layout.addWidget(browse_btn)

    main_layout.addWidget(self.log_livestream)

    # Dialog buttons
    button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    button_box.accepted.connect(self._save)
    button_box.rejected.connect(self.reject)
    main_layout.addWidget(button_box)

    self.setFixedSize(400, self.sizeHint().height())

  def _browse_log_path(self):
    """Open file dialog to select log path"""
    fn = QFileDialog.getExistingDirectory(
      self,
      self.tr("Log File Location"),
      QStandardPaths.writableLocation(QStandardPaths.StandardLocation.HomeLocation),
      QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks
    )
    if fn:
      self.log_path.setText(fn)

  def _save(self):
    """Save settings and close dialog"""
    old_theme = settings.theme
    settings.theme = self.theme.currentIndex()

    if old_theme != settings.theme:
      # Set theme before emit changed
      # Note: In Python, we would need to import and call the theme utility here
      # For now, we'll just emit the signal
      pass

    settings.fps = self.fps.value()
    settings.max_cached_minutes = self.cached_minutes.value()
    settings.chart_height = self.chart_height.value()
    settings.log_livestream = self.log_livestream.isChecked()
    settings.log_path = self.log_path.text()
    settings.drag_direction = DragDirection(self.drag_direction.currentIndex())

    settings.save()
    settings.changed.emit()
    self.accept()


# Global singleton instance
settings = Settings()
