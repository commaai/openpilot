"""MainWindow - main application window for pycabana."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
  QMainWindow,
  QDockWidget,
  QLabel,
  QStatusBar,
  QProgressBar,
  QSplitter,
  QWidget,
  QVBoxLayout,
  QFileDialog,
)

from openpilot.tools.cabana.pycabana.dbc.dbc import MessageId
from openpilot.tools.cabana.pycabana.dbc.dbcmanager import dbc_manager
from openpilot.tools.cabana.pycabana.dialogs.streamselector import StreamSelector
from openpilot.tools.cabana.pycabana.settings import Settings, SettingsDlg
from openpilot.tools.cabana.pycabana.streams.abstract import AbstractStream
from openpilot.tools.cabana.pycabana.streams.replay import ReplayStream
from openpilot.tools.cabana.pycabana.widgets.charts import ChartsWidget
from openpilot.tools.cabana.pycabana.widgets.detail import DetailWidget
from openpilot.tools.cabana.pycabana.widgets.messages import MessagesWidget
from openpilot.tools.cabana.pycabana.widgets.video import VideoWidget


class MainWindow(QMainWindow):
  """Main application window."""

  def __init__(self, stream: AbstractStream, dbc_name: str = "", parent=None):
    super().__init__(parent)
    self.stream = stream
    self._dbc_name = dbc_name
    self._selected_msg_id: MessageId | None = None

    self.setWindowTitle("pycabana")
    self.resize(1400, 900)

    self._setup_ui()
    self._connect_signals()

    # Load DBC if provided
    if dbc_name:
      dbc_manager().load(dbc_name)

  def _setup_ui(self):
    # ===== Central Widget: DetailWidget =====
    self.detail_widget = DetailWidget(self.stream, self)
    self.setCentralWidget(self.detail_widget)

    # ===== Left Dock: Messages =====
    self.messages_widget = MessagesWidget(self.stream)
    self.messages_dock = QDockWidget(self.tr("MESSAGES"), self)
    self.messages_dock.setObjectName("MessagesPanel")
    self.messages_dock.setWidget(self.messages_widget)
    self.messages_dock.setAllowedAreas(
      Qt.DockWidgetArea.LeftDockWidgetArea |
      Qt.DockWidgetArea.RightDockWidgetArea |
      Qt.DockWidgetArea.TopDockWidgetArea |
      Qt.DockWidgetArea.BottomDockWidgetArea
    )
    self.messages_dock.setFeatures(
      QDockWidget.DockWidgetFeature.DockWidgetMovable |
      QDockWidget.DockWidgetFeature.DockWidgetFloatable |
      QDockWidget.DockWidgetFeature.DockWidgetClosable
    )
    self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.messages_dock)

    # ===== Right Dock: Video + Charts in vertical splitter =====
    self.video_dock = QDockWidget("", self)
    self.video_dock.setObjectName("VideoPanel")
    self.video_dock.setAllowedAreas(
      Qt.DockWidgetArea.LeftDockWidgetArea |
      Qt.DockWidgetArea.RightDockWidgetArea
    )
    self.video_dock.setFeatures(
      QDockWidget.DockWidgetFeature.DockWidgetMovable |
      QDockWidget.DockWidgetFeature.DockWidgetFloatable |
      QDockWidget.DockWidgetFeature.DockWidgetClosable
    )

    # Create splitter for video and charts
    self.video_splitter = QSplitter(Qt.Orientation.Vertical)

    # Video widget (timeline controls)
    self.video_widget = VideoWidget()
    self.video_splitter.addWidget(self.video_widget)

    # Charts container
    charts_container = QWidget()
    charts_layout = QVBoxLayout(charts_container)
    charts_layout.setContentsMargins(0, 0, 0, 0)
    self.charts_widget = ChartsWidget()
    charts_layout.addWidget(self.charts_widget)

    self.video_splitter.addWidget(charts_container)
    self.video_splitter.setStretchFactor(0, 0)  # Video - don't stretch
    self.video_splitter.setStretchFactor(1, 1)  # Charts - stretch
    self.video_splitter.setSizes([150, 400])

    self.video_dock.setWidget(self.video_splitter)
    self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.video_dock)

    # Set dock sizes
    self.resizeDocks([self.messages_dock], [350], Qt.Orientation.Horizontal)
    self.resizeDocks([self.video_dock], [450], Qt.Orientation.Horizontal)

    # ===== Status Bar =====
    self.status_bar = QStatusBar()
    self.setStatusBar(self.status_bar)

    self.help_label = QLabel(self.tr("For Help, Press F1"))
    self.status_bar.addWidget(self.help_label)

    self.status_label = QLabel("")
    self.status_bar.addWidget(self.status_label, 1)

    self.progress_bar = QProgressBar()
    self.progress_bar.setFixedSize(300, 16)
    self.progress_bar.setVisible(False)
    self.status_bar.addPermanentWidget(self.progress_bar)

    self.msg_count_label = QLabel("0 messages")
    self.status_bar.addPermanentWidget(self.msg_count_label)

    # Create menu bar
    self._create_menu()

  def _connect_signals(self):
    # Message selection from messages list
    self.messages_widget.msgSelectionChanged.connect(self._on_msg_selected)

    # Stream signals
    self.stream.msgsReceived.connect(self._on_msgs_received)

    # Video widget seeking
    self.video_widget.seeked.connect(self._on_video_seeked)

    # Replay-specific signals
    if isinstance(self.stream, ReplayStream):
      self.stream.loadProgress.connect(self._on_load_progress)
      self.stream.loadFinished.connect(self._on_load_finished)

    # DBC manager signals
    dbc_manager().dbcLoaded.connect(self._on_dbc_loaded)

  def _on_msg_selected(self, msg_id: MessageId | None):
    """Handle message selection from messages list."""
    self._selected_msg_id = msg_id
    if msg_id is not None:
      self.detail_widget.setMessage(msg_id)

  def _on_msgs_received(self, msg_ids: set[MessageId], has_new: bool):
    """Handle stream message updates."""
    total_msgs = len(self.stream.last_msgs)
    total_events = sum(d.count for d in self.stream.last_msgs.values())
    self.msg_count_label.setText(f"{total_msgs} msgs | {total_events:,} events")

    # Update detail widget
    if self._selected_msg_id:
      self.detail_widget.updateState()

  def _on_load_progress(self, can_msgs: int, total_msgs: int):
    """Handle loading progress."""
    self.progress_bar.setVisible(True)
    self.progress_bar.setMaximum(0)  # Indeterminate
    self.status_label.setText(f"Loading... {can_msgs:,} CAN messages")

  def _on_video_seeked(self, time_sec: float):
    """Handle video widget seeking."""
    if isinstance(self.stream, ReplayStream):
      self.stream.seekTo(time_sec)
    self.charts_widget.setCurrentTime(time_sec)

  def _on_load_finished(self):
    """Handle loading completion."""
    self.progress_bar.setVisible(False)
    total_msgs = len(self.stream.last_msgs)
    total_events = sum(d.count for d in self.stream.last_msgs.values())
    self.status_label.setText(f"Loaded {total_events:,} events from {total_msgs} messages")

    if isinstance(self.stream, ReplayStream):
      route = self.stream.routeName
      fingerprint = self.stream.carFingerprint

      # Set video widget duration
      self.video_widget.setDuration(self.stream.duration)
      self.video_widget.setCurrentTime(0)

      # Pass events to charts widget
      self.charts_widget.setEvents(self.stream._all_events)

      if route:
        title = f"pycabana - {route}"
        if fingerprint:
          title += f" ({fingerprint})"
        self.setWindowTitle(title)

      # Auto-load DBC from fingerprint
      if fingerprint and not self._dbc_name:
        self._try_load_dbc_for_fingerprint(fingerprint)

  def _on_dbc_loaded(self):
    """Handle DBC file loaded."""
    self.messages_widget.model.layoutChanged.emit()
    if self._selected_msg_id:
      self.detail_widget.refresh()

  def _try_load_dbc_for_fingerprint(self, fingerprint: str):
    """Try to load a DBC file based on car fingerprint."""
    dbc_name = fingerprint.lower().replace(" ", "_") + "_pt_generated"
    if dbc_manager().load(dbc_name):
      self.status_label.setText(f"Loaded DBC: {dbc_name}")

  def _create_menu(self):
    """Create the application menu bar."""
    menubar = self.menuBar()

    # ===== File Menu =====
    file_menu = menubar.addMenu(self.tr("&File"))

    open_stream_action = QAction(self.tr("Open Stream..."), self)
    open_stream_action.triggered.connect(self._open_stream)
    file_menu.addAction(open_stream_action)

    close_stream_action = QAction(self.tr("Close Stream"), self)
    close_stream_action.setEnabled(False)
    file_menu.addAction(close_stream_action)

    export_csv_action = QAction(self.tr("Export to CSV..."), self)
    export_csv_action.setEnabled(False)
    file_menu.addAction(export_csv_action)

    file_menu.addSeparator()

    new_dbc_action = QAction(self.tr("New DBC File"), self)
    new_dbc_action.setShortcut(QKeySequence.StandardKey.New)
    new_dbc_action.triggered.connect(self._new_dbc_file)
    file_menu.addAction(new_dbc_action)

    open_dbc_action = QAction(self.tr("Open DBC File..."), self)
    open_dbc_action.setShortcut(QKeySequence.StandardKey.Open)
    open_dbc_action.triggered.connect(self._open_dbc_file)
    file_menu.addAction(open_dbc_action)

    file_menu.addSeparator()

    save_dbc_action = QAction(self.tr("Save DBC..."), self)
    save_dbc_action.setShortcut(QKeySequence.StandardKey.Save)
    save_dbc_action.triggered.connect(self._save_dbc_file)
    file_menu.addAction(save_dbc_action)

    save_dbc_as_action = QAction(self.tr("Save DBC As..."), self)
    save_dbc_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
    save_dbc_as_action.triggered.connect(self._save_dbc_as)
    file_menu.addAction(save_dbc_as_action)

    file_menu.addSeparator()

    settings_action = QAction(self.tr("Settings..."), self)
    settings_action.setShortcut(QKeySequence.StandardKey.Preferences)
    settings_action.triggered.connect(self._open_settings)
    file_menu.addAction(settings_action)

    file_menu.addSeparator()

    exit_action = QAction(self.tr("E&xit"), self)
    exit_action.setShortcut(QKeySequence.StandardKey.Quit)
    exit_action.triggered.connect(self.close)
    file_menu.addAction(exit_action)

    # ===== Edit Menu =====
    edit_menu = menubar.addMenu(self.tr("&Edit"))

    undo_action = QAction(self.tr("&Undo"), self)
    undo_action.setShortcut(QKeySequence.StandardKey.Undo)
    undo_action.triggered.connect(self._undo)
    edit_menu.addAction(undo_action)

    redo_action = QAction(self.tr("&Redo"), self)
    redo_action.setShortcut(QKeySequence.StandardKey.Redo)
    redo_action.triggered.connect(self._redo)
    edit_menu.addAction(redo_action)

    # ===== View Menu =====
    view_menu = menubar.addMenu(self.tr("&View"))

    fullscreen_action = QAction(self.tr("Full Screen"), self)
    fullscreen_action.setShortcut(QKeySequence.StandardKey.FullScreen)
    fullscreen_action.triggered.connect(self._toggle_fullscreen)
    view_menu.addAction(fullscreen_action)

    view_menu.addSeparator()

    view_menu.addAction(self.messages_dock.toggleViewAction())
    view_menu.addAction(self.video_dock.toggleViewAction())

    view_menu.addSeparator()

    reset_layout_action = QAction(self.tr("Reset Window Layout"), self)
    reset_layout_action.triggered.connect(self._reset_layout)
    view_menu.addAction(reset_layout_action)

    # ===== Tools Menu =====
    tools_menu = menubar.addMenu(self.tr("&Tools"))

    find_similar_action = QAction(self.tr("Find Similar Bits"), self)
    find_similar_action.triggered.connect(self._find_similar_bits)
    tools_menu.addAction(find_similar_action)

    find_signal_action = QAction(self.tr("Find Signal"), self)
    find_signal_action.triggered.connect(self._find_signal)
    tools_menu.addAction(find_signal_action)

    # ===== Help Menu =====
    help_menu = menubar.addMenu(self.tr("&Help"))

    help_action = QAction(self.tr("Online Help"), self)
    help_action.setShortcut(QKeySequence.StandardKey.HelpContents)
    help_menu.addAction(help_action)

    help_menu.addSeparator()

    about_action = QAction(self.tr("&About"), self)
    about_action.triggered.connect(self._show_about)
    help_menu.addAction(about_action)

  def _open_stream(self):
    """Open the stream selector dialog."""
    dlg = StreamSelector(self)
    if dlg.exec():
      self.status_label.setText("Stream selector opened")

  def _new_dbc_file(self):
    """Create a new empty DBC file."""
    dbc_manager().clear()
    self.status_label.setText("New DBC file created")
    self._on_dbc_loaded()

  def _open_dbc_file(self):
    """Open a DBC file."""
    settings = Settings()
    filename, _ = QFileDialog.getOpenFileName(
      self,
      self.tr("Open DBC File"),
      settings.last_dir,
      self.tr("DBC Files (*.dbc);;All Files (*)")
    )
    if filename:
      from pathlib import Path
      settings.last_dir = str(Path(filename).parent)
      settings.save()

      if dbc_manager().load(filename):
        self.status_label.setText(f"Loaded DBC: {filename}")
      else:
        self.status_label.setText(f"Failed to load DBC: {filename}")

  def _save_dbc_file(self):
    """Save the current DBC file."""
    # TODO: Track current filename and save to it
    self._save_dbc_as()

  def _save_dbc_as(self):
    """Save the current DBC to a new file."""
    settings = Settings()
    filename, _ = QFileDialog.getSaveFileName(
      self,
      self.tr("Save DBC File"),
      settings.last_dir,
      self.tr("DBC Files (*.dbc)")
    )
    if filename:
      from pathlib import Path
      settings.last_dir = str(Path(filename).parent)
      settings.save()

      if dbc_manager().save(filename):
        self.status_label.setText(f"Saved DBC: {filename}")
      else:
        self.status_label.setText(f"Failed to save DBC: {filename}")

  def _open_settings(self):
    """Open the settings dialog."""
    dlg = SettingsDlg(self)
    dlg.exec()

  def _undo(self):
    """Undo the last action."""
    from openpilot.tools.cabana.pycabana.commands import undo_stack
    undo_stack().undo()

  def _redo(self):
    """Redo the last undone action."""
    from openpilot.tools.cabana.pycabana.commands import undo_stack
    undo_stack().redo()

  def _toggle_fullscreen(self):
    """Toggle fullscreen mode."""
    if self.isFullScreen():
      self.showNormal()
    else:
      self.showFullScreen()

  def _reset_layout(self):
    """Reset window layout to default."""
    # Restore docks to default positions
    self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.messages_dock)
    self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.video_dock)
    self.messages_dock.show()
    self.video_dock.show()
    self.resizeDocks([self.messages_dock], [350], Qt.Orientation.Horizontal)
    self.resizeDocks([self.video_dock], [450], Qt.Orientation.Horizontal)

  def _find_similar_bits(self):
    """Find similar bits tool."""
    from PySide6.QtWidgets import QMessageBox
    QMessageBox.information(self, "Find Similar Bits", "Not yet implemented")

  def _find_signal(self):
    """Find signal tool."""
    from PySide6.QtWidgets import QMessageBox
    QMessageBox.information(self, "Find Signal", "Not yet implemented")

  def _show_about(self):
    """Show the about dialog."""
    from PySide6.QtWidgets import QMessageBox
    QMessageBox.about(
      self,
      self.tr("About pycabana"),
      self.tr(
        "pycabana - PySide6 CAN Bus Analyzer\n\n"
        + "A Python port of cabana for analyzing CAN bus data\n"
        + "from openpilot routes.\n\n"
        + "comma.ai"
      )
    )
