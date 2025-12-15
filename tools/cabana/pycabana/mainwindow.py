"""MainWindow - main application window for pycabana."""

from PySide2.QtCore import Qt
from PySide2.QtWidgets import (
  QMainWindow,
  QDockWidget,
  QLabel,
  QStatusBar,
  QProgressBar,
  QWidget,
  QVBoxLayout,
)

from openpilot.tools.cabana.pycabana.dbc.dbc import MessageId
from openpilot.tools.cabana.pycabana.streams.abstract import AbstractStream
from openpilot.tools.cabana.pycabana.streams.replay import ReplayStream
from openpilot.tools.cabana.pycabana.widgets.messages import MessagesWidget


class MainWindow(QMainWindow):
  """Main application window."""

  def __init__(self, stream: AbstractStream, parent=None):
    super().__init__(parent)
    self.stream = stream

    self.setWindowTitle("pycabana")
    self.resize(1200, 800)

    self._setup_ui()
    self._connect_signals()

  def _setup_ui(self):
    # Central widget - placeholder for now
    central = QWidget()
    layout = QVBoxLayout(central)
    layout.setContentsMargins(20, 20, 20, 20)

    self.welcome_label = QLabel("Select a message from the list")
    self.welcome_label.setAlignment(Qt.AlignCenter)
    self.welcome_label.setStyleSheet("color: gray; font-size: 14px;")
    layout.addWidget(self.welcome_label)

    self.setCentralWidget(central)

    # Messages dock (left)
    self.messages_widget = MessagesWidget(self.stream)
    self.messages_dock = QDockWidget("Messages", self)
    self.messages_dock.setWidget(self.messages_widget)
    self.messages_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
    self.addDockWidget(Qt.LeftDockWidgetArea, self.messages_dock)

    # Status bar
    self.status_bar = QStatusBar()
    self.setStatusBar(self.status_bar)

    self.status_label = QLabel("Ready")
    self.status_bar.addWidget(self.status_label, 1)

    self.progress_bar = QProgressBar()
    self.progress_bar.setFixedWidth(200)
    self.progress_bar.setVisible(False)
    self.status_bar.addPermanentWidget(self.progress_bar)

    self.msg_count_label = QLabel("0 messages")
    self.status_bar.addPermanentWidget(self.msg_count_label)

  def _connect_signals(self):
    # Message selection
    self.messages_widget.msgSelectionChanged.connect(self._on_msg_selected)

    # Stream signals
    self.stream.msgsReceived.connect(self._on_msgs_received)

    # Replay-specific signals
    if isinstance(self.stream, ReplayStream):
      self.stream.loadProgress.connect(self._on_load_progress)
      self.stream.loadFinished.connect(self._on_load_finished)

  def _on_msg_selected(self, msg_id: MessageId | None):
    """Handle message selection."""
    if msg_id:
      can_data = self.stream.last_msgs.get(msg_id)
      if can_data:
        self.welcome_label.setText(f"Selected: {msg_id}\nCount: {can_data.count}\nFreq: {can_data.freq:.1f} Hz\nData: {can_data.dat.hex(' ').upper()}")
      else:
        self.welcome_label.setText(f"Selected: {msg_id}")
    else:
      self.welcome_label.setText("Select a message from the list")

  def _on_msgs_received(self, msg_ids: set[MessageId], has_new: bool):
    """Handle stream message updates."""
    total_msgs = len(self.stream.last_msgs)
    total_events = sum(d.count for d in self.stream.last_msgs.values())
    self.msg_count_label.setText(f"{total_msgs} messages, {total_events} events")

  def _on_load_progress(self, can_msgs: int, total_msgs: int):
    """Handle loading progress."""
    self.progress_bar.setVisible(True)
    self.progress_bar.setMaximum(0)  # Indeterminate
    self.status_label.setText(f"Loading... {can_msgs} CAN messages")

  def _on_load_finished(self):
    """Handle loading completion."""
    self.progress_bar.setVisible(False)
    total_msgs = len(self.stream.last_msgs)
    total_events = sum(d.count for d in self.stream.last_msgs.values())
    self.status_label.setText(f"Loaded {total_events} events from {total_msgs} messages")

    # Show route info if available
    if isinstance(self.stream, ReplayStream):
      route = self.stream.routeName
      fingerprint = self.stream.carFingerprint
      if route:
        title = f"pycabana - {route}"
        if fingerprint:
          title += f" ({fingerprint})"
        self.setWindowTitle(title)
