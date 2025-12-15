"""MainWindow - main application window for pycabana."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
  QMainWindow,
  QDockWidget,
  QLabel,
  QStatusBar,
  QProgressBar,
)

from openpilot.tools.cabana.pycabana.dbc.dbc import MessageId
from openpilot.tools.cabana.pycabana.dbc.dbcmanager import dbc_manager
from openpilot.tools.cabana.pycabana.streams.abstract import AbstractStream
from openpilot.tools.cabana.pycabana.streams.replay import ReplayStream
from openpilot.tools.cabana.pycabana.widgets.messages import MessagesWidget
from openpilot.tools.cabana.pycabana.widgets.signal import SignalView


class MainWindow(QMainWindow):
  """Main application window."""

  def __init__(self, stream: AbstractStream, dbc_name: str = "", parent=None):
    super().__init__(parent)
    self.stream = stream
    self._dbc_name = dbc_name
    self._selected_msg_id: MessageId | None = None

    self.setWindowTitle("pycabana")
    self.resize(1200, 800)

    self._setup_ui()
    self._connect_signals()

    # Load DBC if provided
    if dbc_name:
      dbc_manager().load(dbc_name)

  def _setup_ui(self):
    # Central widget - signal view
    self.signal_view = SignalView()
    self.setCentralWidget(self.signal_view)

    # Messages dock (left)
    self.messages_widget = MessagesWidget(self.stream)
    self.messages_dock = QDockWidget("Messages", self)
    self.messages_dock.setWidget(self.messages_widget)
    self.messages_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
    self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.messages_dock)

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
    self._selected_msg_id = msg_id
    can_data = self.stream.last_msgs.get(msg_id) if msg_id else None
    self.signal_view.setMessage(msg_id, can_data)

  def _on_msgs_received(self, msg_ids: set[MessageId], has_new: bool):
    """Handle stream message updates."""
    total_msgs = len(self.stream.last_msgs)
    total_events = sum(d.count for d in self.stream.last_msgs.values())
    self.msg_count_label.setText(f"{total_msgs} messages, {total_events} events")

    # Update signal view if showing a message
    if self._selected_msg_id:
      can_data = self.stream.last_msgs.get(self._selected_msg_id)
      self.signal_view.updateValues(can_data)

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

      # Auto-load DBC from fingerprint if no DBC was specified
      if fingerprint and not self._dbc_name:
        self._try_load_dbc_for_fingerprint(fingerprint)

  def _try_load_dbc_for_fingerprint(self, fingerprint: str):
    """Try to load a DBC file based on car fingerprint."""
    # Common DBC name patterns based on fingerprint
    # Fingerprints are typically like "TOYOTA RAV4 2019" -> "toyota_rav4_2019"
    dbc_name = fingerprint.lower().replace(" ", "_") + "_pt_generated"
    if dbc_manager().load(dbc_name):
      self.status_label.setText(f"Loaded DBC: {dbc_name}")
      # Refresh the messages table to show names
      self.messages_widget.model.layoutChanged.emit()
