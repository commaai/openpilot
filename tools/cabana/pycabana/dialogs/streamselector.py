"""StreamSelector - dialog for selecting and opening CAN data streams."""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
  QDialog,
  QWidget,
  QVBoxLayout,
  QHBoxLayout,
  QGridLayout,
  QFormLayout,
  QTabWidget,
  QLineEdit,
  QLabel,
  QPushButton,
  QDialogButtonBox,
  QFileDialog,
  QFrame,
  QComboBox,
  QCheckBox,
  QRadioButton,
  QButtonGroup,
)
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtCore import QRegularExpression

from openpilot.tools.cabana.pycabana.streams.abstract import AbstractStream
from openpilot.tools.cabana.pycabana.streams.replay import ReplayStream
from openpilot.tools.cabana.pycabana.settings import Settings


class AbstractOpenStreamWidget(QWidget):
  """Base class for stream source widgets.

  Subclasses must implement open() to return a configured stream.
  """

  def __init__(self, parent: Optional[QWidget] = None):
    super().__init__(parent)

  def open(self) -> Optional[AbstractStream]:
    """Open and return a configured stream, or None on failure."""
    raise NotImplementedError("Subclasses must implement open()")


class OpenReplayWidget(AbstractOpenStreamWidget):
  """Widget for opening replay streams from routes."""

  def __init__(self, parent: Optional[QWidget] = None):
    super().__init__(parent)

    settings = Settings()

    layout = QGridLayout(self)

    # Route input
    layout.addWidget(QLabel("Route"), 0, 0)
    self.route_edit = QLineEdit(self)
    self.route_edit.setPlaceholderText("Enter route name or browse for local/remote route")
    layout.addWidget(self.route_edit, 0, 1)

    # Remote route button
    browse_remote_btn = QPushButton("Remote route...", self)
    layout.addWidget(browse_remote_btn, 0, 2)
    browse_remote_btn.setEnabled(False)  # Not implemented yet

    # Local route button
    browse_local_btn = QPushButton("Local route...", self)
    layout.addWidget(browse_local_btn, 0, 3)

    # Camera options
    camera_layout = QHBoxLayout()
    self.road_camera = QCheckBox("Road camera", self)
    self.driver_camera = QCheckBox("Driver camera", self)
    self.wide_camera = QCheckBox("Wide road camera", self)

    self.road_camera.setChecked(True)

    camera_layout.addWidget(self.road_camera)
    camera_layout.addWidget(self.driver_camera)
    camera_layout.addWidget(self.wide_camera)
    camera_layout.addStretch(1)
    layout.addLayout(camera_layout, 1, 1)

    self.setMinimumWidth(550)

    # Connect signals
    browse_local_btn.clicked.connect(self._browse_local_route)
    browse_remote_btn.clicked.connect(self._browse_remote_route)

  def _browse_local_route(self) -> None:
    """Browse for a local route directory."""
    settings = Settings()
    directory = QFileDialog.getExistingDirectory(
      self,
      "Open Local Route",
      settings.last_route_dir
    )
    if directory:
      self.route_edit.setText(directory)
      settings.last_route_dir = directory
      settings.save()

  def _browse_remote_route(self) -> None:
    """Browse for a remote route (not implemented yet)."""
    # TODO: Implement remote route browser dialog
    pass

  def open(self) -> Optional[AbstractStream]:
    """Open a replay stream from the specified route."""
    route = self.route_edit.text().strip()
    if not route:
      return None

    stream = ReplayStream()
    if stream.loadRoute(route):
      return stream

    return None


class OpenPandaWidget(AbstractOpenStreamWidget):
  """Widget for opening live streams from Panda devices."""

  def __init__(self, parent: Optional[QWidget] = None):
    super().__init__(parent)

    form_layout = QFormLayout(self)

    # Serial selection
    serial_layout = QHBoxLayout()
    self.serial_edit = QComboBox(self)
    serial_layout.addWidget(self.serial_edit)

    refresh_btn = QPushButton("Refresh", self)
    refresh_btn.setSizePolicy(refresh_btn.sizePolicy().horizontalPolicy(), refresh_btn.sizePolicy().verticalPolicy())
    serial_layout.addWidget(refresh_btn)

    form_layout.addRow("Serial", serial_layout)

    # Note: Panda streaming not fully implemented in Python yet
    info_label = QLabel("Panda live streaming is not yet fully implemented in pycabana.")
    info_label.setWordWrap(True)
    form_layout.addRow(info_label)

    refresh_btn.clicked.connect(self._refresh_serials)
    self.serial_edit.currentTextChanged.connect(self._on_serial_changed)

    self._refresh_serials()

  def _refresh_serials(self) -> None:
    """Refresh the list of available Panda devices."""
    self.serial_edit.clear()
    # TODO: Implement actual Panda device discovery
    self.serial_edit.addItem("(No Panda devices found)")

  def _on_serial_changed(self, serial: str) -> None:
    """Handle serial selection change."""
    pass

  def open(self) -> Optional[AbstractStream]:
    """Open a Panda stream (not yet implemented)."""
    # TODO: Implement PandaStream
    return None


class OpenSocketCanWidget(AbstractOpenStreamWidget):
  """Widget for opening SocketCAN streams."""

  def __init__(self, parent: Optional[QWidget] = None):
    super().__init__(parent)

    main_layout = QVBoxLayout(self)
    main_layout.addStretch(1)

    form_layout = QFormLayout()

    # Device selection
    device_layout = QHBoxLayout()
    self.device_edit = QComboBox(self)
    self.device_edit.setFixedWidth(300)
    device_layout.addWidget(self.device_edit)

    refresh_btn = QPushButton("Refresh", self)
    refresh_btn.setFixedWidth(100)
    device_layout.addWidget(refresh_btn)

    form_layout.addRow("Device", device_layout)
    main_layout.addLayout(form_layout)

    main_layout.addStretch(1)

    # Note: SocketCAN not fully implemented in Python yet
    info_label = QLabel("SocketCAN streaming is not yet fully implemented in pycabana.")
    info_label.setWordWrap(True)
    form_layout.addRow(info_label)

    refresh_btn.clicked.connect(self._refresh_devices)
    self.device_edit.currentTextChanged.connect(self._on_device_changed)

    self._refresh_devices()

  def _refresh_devices(self) -> None:
    """Refresh the list of available SocketCAN devices."""
    self.device_edit.clear()
    # TODO: Implement actual SocketCAN device discovery
    self.device_edit.addItem("(No SocketCAN devices found)")

  def _on_device_changed(self, device: str) -> None:
    """Handle device selection change."""
    pass

  def open(self) -> Optional[AbstractStream]:
    """Open a SocketCAN stream (not yet implemented)."""
    # TODO: Implement SocketCanStream
    return None


class OpenDeviceWidget(AbstractOpenStreamWidget):
  """Widget for opening live streams from openpilot devices."""

  def __init__(self, parent: Optional[QWidget] = None):
    super().__init__(parent)

    form_layout = QFormLayout(self)

    # Connection type selection
    self.msgq_radio = QRadioButton("MSGQ", self)
    self.zmq_radio = QRadioButton("ZMQ", self)

    # IP address input for ZMQ
    self.ip_address = QLineEdit(self)
    self.ip_address.setPlaceholderText("Enter device IP Address")

    # IP address validation
    ip_range = "(?:[0-1]?[0-9]?[0-9]|2[0-4][0-9]|25[0-5])"
    pattern = f"^{ip_range}\\.{ip_range}\\.{ip_range}\\.{ip_range}$"
    regex = QRegularExpression(pattern)
    validator = QRegularExpressionValidator(regex, self)
    self.ip_address.setValidator(validator)

    # Button group for radio buttons
    self.connection_group = QButtonGroup(self)
    self.connection_group.addButton(self.msgq_radio, 0)
    self.connection_group.addButton(self.zmq_radio, 1)

    form_layout.addRow(self.msgq_radio)
    form_layout.addRow(self.zmq_radio, self.ip_address)

    # Note: Device streaming not fully implemented in Python yet
    info_label = QLabel("Device live streaming is not yet fully implemented in pycabana.")
    info_label.setWordWrap(True)
    form_layout.addRow(info_label)

    # Default to ZMQ
    self.zmq_radio.setChecked(True)

    # Connect signals
    self.connection_group.buttonToggled.connect(self._on_connection_toggled)

  def _on_connection_toggled(self, button, checked: bool) -> None:
    """Handle connection type toggle."""
    self.ip_address.setEnabled(button == self.zmq_radio and checked)

  def open(self) -> Optional[AbstractStream]:
    """Open a device stream (not yet implemented)."""
    # TODO: Implement DeviceStream
    return None


class StreamSelector(QDialog):
  """Dialog for selecting and opening CAN data streams.

  Provides tabs for different stream sources:
  - Replay: Load data from openpilot routes
  - Panda: Connect to Panda hardware
  - SocketCAN: Connect to SocketCAN devices
  - Device: Connect to live openpilot devices
  """

  def __init__(self, parent: Optional[QWidget] = None):
    super().__init__(parent)

    self.stream_: Optional[AbstractStream] = None
    self.settings = Settings()

    self.setWindowTitle("Open stream")

    layout = QVBoxLayout(self)

    # Tab widget for different stream sources
    self.tab = QTabWidget(self)
    layout.addWidget(self.tab)

    # DBC file selection
    dbc_layout = QHBoxLayout()

    self.dbc_file = QLineEdit(self)
    self.dbc_file.setReadOnly(True)
    self.dbc_file.setPlaceholderText("Choose a dbc file to open")

    file_btn = QPushButton("Browse...", self)

    dbc_layout.addWidget(QLabel("dbc File"))
    dbc_layout.addWidget(self.dbc_file)
    dbc_layout.addWidget(file_btn)

    layout.addLayout(dbc_layout)

    # Separator line
    line = QFrame(self)
    line.setFrameStyle(QFrame.Shape.HLine | QFrame.Shadow.Sunken)
    layout.addWidget(line)

    # Dialog buttons
    self.btn_box = QDialogButtonBox(
      QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel,
      self
    )
    layout.addWidget(self.btn_box)

    # Add stream source widgets
    self._add_stream_widget(OpenReplayWidget(), "&Replay")
    self._add_stream_widget(OpenPandaWidget(), "&Panda")

    # Only add SocketCAN if available (platform dependent)
    if self._socketcan_available():
      self._add_stream_widget(OpenSocketCanWidget(), "&SocketCAN")

    self._add_stream_widget(OpenDeviceWidget(), "&Device")

    # Connect signals
    self.btn_box.rejected.connect(self.reject)
    self.btn_box.accepted.connect(self._on_open_clicked)
    file_btn.clicked.connect(self._browse_dbc_file)

  def _add_stream_widget(self, widget: AbstractOpenStreamWidget, title: str) -> None:
    """Add a stream source widget as a tab."""
    self.tab.addTab(widget, title)

  def _socketcan_available(self) -> bool:
    """Check if SocketCAN is available on this platform."""
    # SocketCAN is typically only available on Linux
    import platform
    return platform.system() == "Linux"

  def _browse_dbc_file(self) -> None:
    """Browse for a DBC file."""
    filename = QFileDialog.getOpenFileName(
      self,
      "Open File",
      self.settings.last_dir,
      "DBC (*.dbc)"
    )[0]

    if filename:
      self.dbc_file.setText(filename)
      from pathlib import Path
      self.settings.last_dir = str(Path(filename).parent)
      self.settings.save()

  def _on_open_clicked(self) -> None:
    """Handle the Open button click."""
    self.setEnabled(False)

    # Get the current widget and try to open its stream
    current_widget = self.tab.currentWidget()
    if isinstance(current_widget, AbstractOpenStreamWidget):
      self.stream_ = current_widget.open()
      if self.stream_:
        self.accept()
        return

    self.setEnabled(True)

  def dbcFile(self) -> str:
    """Get the selected DBC file path."""
    return self.dbc_file.text()

  def stream(self) -> Optional[AbstractStream]:
    """Get the opened stream."""
    return self.stream_
