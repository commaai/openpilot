"""DetailWidget - displays detailed information about a selected CAN message."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
  QWidget,
  QVBoxLayout,
  QHBoxLayout,
  QLabel,
  QTabWidget,
  QSplitter,
  QToolBar,
  QRadioButton,
  QTabBar,
  QMenu,
  QSizePolicy,
  QFrame,
)
from PySide6.QtGui import QPixmap, QPainter, QIcon, QAction

from openpilot.tools.cabana.pycabana.dbc.dbc import MessageId
from openpilot.tools.cabana.pycabana.dbc.dbcmanager import dbc_manager
from openpilot.tools.cabana.pycabana.widgets.binary import BinaryView
from openpilot.tools.cabana.pycabana.widgets.signal import SignalView
from openpilot.tools.cabana.pycabana.streams.abstract import AbstractStream


class ElidedLabel(QLabel):
  """Label that elides text with ellipsis if too long."""

  def __init__(self, text: str = "", parent: QWidget | None = None):
    super().__init__(text, parent)
    self.setTextFormat(Qt.TextFormat.PlainText)

  def paintEvent(self, event):
    painter = QPainter(self)
    metrics = painter.fontMetrics()
    elided = metrics.elidedText(self.text(), Qt.TextElideMode.ElideRight, self.width())
    painter.drawText(self.rect(), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided)


class TabBar(QTabBar):
  """Custom tab bar that can be hidden when empty."""

  def __init__(self, parent: QWidget | None = None):
    super().__init__(parent)
    self._auto_hide = False
    self.setTabsClosable(True)

  def setAutoHide(self, enable: bool):
    """Enable auto-hide when no tabs."""
    self._auto_hide = enable
    self._updateVisibility()

  def tabInserted(self, index: int):
    super().tabInserted(index)
    self._updateVisibility()

  def tabRemoved(self, index: int):
    super().tabRemoved(index)
    self._updateVisibility()

  def _updateVisibility(self):
    if self._auto_hide:
      self.setVisible(self.count() > 0)


class HistoryLog(QWidget):
  """Placeholder for history log widget."""

  def __init__(self, parent: QWidget | None = None):
    super().__init__(parent)
    layout = QVBoxLayout(self)
    self._msg_id: MessageId | None = None
    self._stream: AbstractStream | None = None

    label = QLabel("History Log - Not yet implemented")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet("color: gray; font-style: italic; font-size: 14px;")
    layout.addWidget(label)

  def setMessage(self, msg_id: MessageId):
    """Set the message to display history for."""
    self._msg_id = msg_id

  def setStream(self, stream: AbstractStream):
    """Set the stream to read events from."""
    self._stream = stream

  def updateState(self):
    """Update the history log state."""
    pass


class DetailWidget(QWidget):
  """Main detail widget showing message information across multiple tabs."""

  def __init__(self, stream: AbstractStream, parent: QWidget | None = None):
    super().__init__(parent)
    self._stream = stream
    self._msg_id: MessageId | None = None

    self._setup_ui()
    self._connect_signals()

  def _setup_ui(self):
    """Set up the user interface."""
    main_layout = QVBoxLayout(self)
    main_layout.setContentsMargins(0, 0, 0, 0)
    main_layout.setSpacing(0)

    # Tab bar for multiple messages
    self._tabbar = TabBar(self)
    self._tabbar.setUsesScrollButtons(True)
    self._tabbar.setAutoHide(True)
    self._tabbar.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    main_layout.addWidget(self._tabbar)

    # Toolbar
    self._create_toolbar()
    main_layout.addWidget(self._toolbar)

    # Warning widget
    self._warning_widget = QWidget(self)
    warning_layout = QHBoxLayout(self._warning_widget)
    warning_layout.setContentsMargins(8, 4, 8, 4)

    self._warning_icon = QLabel(self._warning_widget)
    self._warning_label = QLabel(self._warning_widget)
    self._warning_label.setWordWrap(True)

    warning_layout.addWidget(self._warning_icon, 0, Qt.AlignmentFlag.AlignTop)
    warning_layout.addWidget(self._warning_label, 1)

    self._warning_widget.hide()
    self._warning_widget.setStyleSheet("background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 4px;")
    main_layout.addWidget(self._warning_widget)

    # Main content splitter
    splitter = QSplitter(Qt.Orientation.Vertical, self)

    self._binary_view = BinaryView(self)
    self._signal_view = SignalView(self)

    splitter.addWidget(self._binary_view)
    splitter.addWidget(self._signal_view)

    self._binary_view.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    self._signal_view.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)

    # Tab widget for Msg/Logs
    self._tab_widget = QTabWidget(self)
    self._tab_widget.setStyleSheet("QTabWidget::pane {border: none; margin-bottom: -2px;}")
    self._tab_widget.setTabPosition(QTabWidget.TabPosition.South)

    self._tab_widget.addTab(splitter, "&Msg")

    self._history_log = HistoryLog(self)
    self._history_log.setStream(self._stream)
    self._tab_widget.addTab(self._history_log, "&Logs")

    main_layout.addWidget(self._tab_widget)

  def _create_toolbar(self):
    """Create the toolbar with message name and controls."""
    self._toolbar = QToolBar(self)
    icon_size = self.style().pixelMetric(self.style().PixelMetric.PM_SmallIconSize)
    self._toolbar.setIconSize(self._toolbar.iconSize())

    # Message name label
    self._name_label = ElidedLabel("", self)
    self._name_label.setStyleSheet("QLabel {font-weight: bold; padding: 4px;}")
    self._toolbar.addWidget(self._name_label)

    # Spacer
    spacer = QWidget()
    spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    self._toolbar.addWidget(spacer)

    # Heatmap controls
    self._toolbar.addWidget(QLabel("Heatmap:", self))
    self._heatmap_live = QRadioButton("Live", self)
    self._heatmap_all = QRadioButton("All", self)
    self._heatmap_live.setChecked(True)

    self._toolbar.addWidget(self._heatmap_live)
    self._toolbar.addWidget(self._heatmap_all)

    # Edit and remove buttons
    self._toolbar.addSeparator()

    self._edit_action = QAction("Edit Message", self)
    self._edit_action.triggered.connect(self._edit_msg)
    self._toolbar.addAction(self._edit_action)

    self._remove_action = QAction("Remove Message", self)
    self._remove_action.triggered.connect(self._remove_msg)
    self._remove_action.setEnabled(False)
    self._toolbar.addAction(self._remove_action)

  def _connect_signals(self):
    """Connect internal signals."""
    # Tab bar signals
    self._tabbar.currentChanged.connect(self._on_tab_changed)
    self._tabbar.tabCloseRequested.connect(self._tabbar.removeTab)
    self._tabbar.customContextMenuRequested.connect(self._show_tabbar_context_menu)

    # Tab widget signals
    self._tab_widget.currentChanged.connect(self._on_content_tab_changed)

    # Stream signals
    self._stream.msgsReceived.connect(self._on_msgs_received)

    # DBC manager signals
    dbc_manager().dbcLoaded.connect(self._on_dbc_loaded)

  def _on_tab_changed(self, index: int):
    """Handle tab bar selection change."""
    if index >= 0:
      msg_id = self._tabbar.tabData(index)
      if isinstance(msg_id, MessageId):
        self.setMessage(msg_id)

  def _on_content_tab_changed(self, index: int):
    """Handle content tab change (Msg/Logs)."""
    self._update_state()

  def _on_msgs_received(self, msg_ids: set[MessageId], has_new: bool):
    """Handle new messages from stream."""
    if self._msg_id and self._msg_id in msg_ids:
      self._update_state()

  def _on_dbc_loaded(self, dbc_name: str):
    """Handle DBC file loaded."""
    self.refresh()

  def _show_tabbar_context_menu(self, pos):
    """Show context menu on tab bar."""
    index = self._tabbar.tabAt(pos)
    if index >= 0:
      menu = QMenu(self)
      action = menu.addAction("Close Other Tabs")
      if menu.exec(self._tabbar.mapToGlobal(pos)):
        # Move selected tab to front
        self._tabbar.moveTab(index, 0)
        self._tabbar.setCurrentIndex(0)
        # Remove all other tabs
        while self._tabbar.count() > 1:
          self._tabbar.removeTab(1)

  def _find_or_add_tab(self, msg_id: MessageId) -> int:
    """Find existing tab or add new one for message."""
    # Search for existing tab
    for index in range(self._tabbar.count()):
      tab_msg_id = self._tabbar.tabData(index)
      if isinstance(tab_msg_id, MessageId) and tab_msg_id == msg_id:
        return index

    # Add new tab
    index = self._tabbar.addTab(str(msg_id))
    self._tabbar.setTabData(index, msg_id)
    self._tabbar.setTabToolTip(index, self._get_msg_name(msg_id))
    return index

  def _get_msg_name(self, msg_id: MessageId) -> str:
    """Get message name from DBC or return default."""
    msg = dbc_manager().msg(msg_id)
    if msg:
      return msg.name
    return f"0x{msg_id.address:X}"

  def setMessage(self, msg_id: MessageId):
    """Set the currently displayed message."""
    if self._msg_id == msg_id:
      return

    self._msg_id = msg_id

    # Update tab bar
    self._tabbar.blockSignals(True)
    index = self._find_or_add_tab(msg_id)
    self._tabbar.setCurrentIndex(index)
    self._tabbar.blockSignals(False)

    # Update views
    self.setUpdatesEnabled(False)

    can_data = self._stream.lastMessage(msg_id)
    self._binary_view.setMessage(msg_id, can_data)
    self._signal_view.setMessage(msg_id, can_data)
    self._history_log.setMessage(msg_id)

    self.refresh()
    self.setUpdatesEnabled(True)

  def refresh(self):
    """Refresh the display with current message state."""
    if not self._msg_id:
      return

    warnings = []
    msg = dbc_manager().msg(self._msg_id)

    if msg:
      can_data = self._stream.lastMessage(self._msg_id)
      if not can_data:
        warnings.append("No messages received.")
      elif msg.size != len(can_data.dat):
        warnings.append(f"Message size ({msg.size}) is incorrect.")

      # Display message name
      msg_name = f"{msg.name} (0x{self._msg_id.address:X})"
      self._name_label.setText(msg_name)
      self._name_label.setToolTip(msg_name)
      self._remove_action.setEnabled(True)
    else:
      # No DBC definition
      msg_name = f"0x{self._msg_id.address:X}"
      self._name_label.setText(msg_name)
      self._name_label.setToolTip(msg_name)
      self._remove_action.setEnabled(False)
      warnings.append("No DBC definition for this message.")

    # Show warnings if any
    if warnings:
      self._warning_label.setText("\n".join(warnings))
      self._warning_icon.setText("⚠")
      self._warning_widget.show()
    else:
      self._warning_widget.hide()

  def _update_state(self):
    """Update the current view based on tab selection."""
    if not self._msg_id:
      return

    can_data = self._stream.lastMessage(self._msg_id)

    # Update based on which tab is active
    if self._tab_widget.currentIndex() == 0:
      # Msg tab
      self._binary_view.updateData(can_data)
      self._signal_view.updateValues(can_data)
    else:
      # Logs tab
      self._history_log.updateState()

  def _edit_msg(self):
    """Open edit message dialog."""
    # TODO: Implement edit message dialog
    print(f"Edit message: {self._msg_id}")

  def _remove_msg(self):
    """Remove message from DBC."""
    # TODO: Implement remove message
    print(f"Remove message: {self._msg_id}")

  def serializeMessageIds(self) -> tuple[str, list[str]]:
    """Serialize tab state for saving."""
    msg_ids = []
    for i in range(self._tabbar.count()):
      msg_id = self._tabbar.tabData(i)
      if isinstance(msg_id, MessageId):
        msg_ids.append(str(msg_id))

    active_id = str(self._msg_id) if self._msg_id else ""
    return (active_id, msg_ids)

  def restoreTabs(self, active_msg_id: str, msg_ids: list[str]):
    """Restore tab state from saved data."""
    self._tabbar.blockSignals(True)

    # Add tabs for each message ID
    for msg_id_str in msg_ids:
      try:
        # Parse "source:address" format
        parts = msg_id_str.split(':')
        if len(parts) == 2:
          source = int(parts[0])
          address = int(parts[1], 16)
          msg_id = MessageId(source=source, address=address)

          # Check if message still exists in DBC
          if dbc_manager().msg(msg_id):
            self._find_or_add_tab(msg_id)
      except (ValueError, IndexError):
        continue

    self._tabbar.blockSignals(False)

    # Set active message
    if active_msg_id:
      try:
        parts = active_msg_id.split(':')
        if len(parts) == 2:
          source = int(parts[0])
          address = int(parts[1], 16)
          active_id = MessageId(source=source, address=address)
          if dbc_manager().msg(active_id):
            self.setMessage(active_id)
      except (ValueError, IndexError):
        pass


class CenterWidget(QWidget):
  """Center widget that shows either welcome screen or detail widget."""

  def __init__(self, stream: AbstractStream, parent: QWidget | None = None):
    super().__init__(parent)
    self._stream = stream
    self._detail_widget: DetailWidget | None = None
    self._welcome_widget: QWidget | None = None

    self._layout = QVBoxLayout(self)
    self._layout.setContentsMargins(0, 0, 0, 0)

    self._show_welcome()

  def _show_welcome(self):
    """Show welcome screen."""
    if self._welcome_widget:
      return

    self._welcome_widget = self._create_welcome_widget()
    self._layout.addWidget(self._welcome_widget)

  def _create_welcome_widget(self) -> QWidget:
    """Create the welcome screen widget."""
    widget = QWidget(self)
    layout = QVBoxLayout(widget)
    layout.addStretch()

    # Logo
    logo = QLabel("CABANA")
    logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
    logo.setStyleSheet("font-size: 50px; font-weight: bold; color: #404040;")
    layout.addWidget(logo)

    # Instructions
    instructions = QLabel("<- Select a message to view details")
    instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
    instructions.setStyleSheet("color: gray; font-size: 14px; margin-top: 20px;")
    layout.addWidget(instructions)

    # Shortcuts
    shortcuts_layout = QVBoxLayout()
    shortcuts_layout.setSpacing(8)
    shortcuts_layout.setContentsMargins(0, 20, 0, 0)

    def add_shortcut(title: str, key: str):
      row = QHBoxLayout()
      row.addStretch()

      label = QLabel(title)
      label.setStyleSheet("color: gray;")
      row.addWidget(label)

      key_label = QLabel(key)
      key_label.setStyleSheet(
        "background-color: #e0e0e0; padding: 4px 8px; "
        "border-radius: 4px; color: #404040; margin-left: 8px;"
      )
      row.addWidget(key_label)
      row.addStretch()

      shortcuts_layout.addLayout(row)

    add_shortcut("Pause", "Space")
    add_shortcut("Help", "F1")
    add_shortcut("WhatsThis", "Shift+F1")

    layout.addLayout(shortcuts_layout)
    layout.addStretch()

    widget.setStyleSheet("background-color: #f5f5f5;")
    widget.setAutoFillBackground(True)

    return widget

  def setMessage(self, msg_id: MessageId):
    """Set the message to display."""
    detail = self.ensureDetailWidget()
    detail.setMessage(msg_id)

  def ensureDetailWidget(self) -> DetailWidget:
    """Ensure detail widget exists and return it."""
    if not self._detail_widget:
      if self._welcome_widget:
        self._layout.removeWidget(self._welcome_widget)
        self._welcome_widget.deleteLater()
        self._welcome_widget = None

      self._detail_widget = DetailWidget(self._stream, self)
      self._layout.addWidget(self._detail_widget)

    return self._detail_widget

  def getDetailWidget(self) -> DetailWidget | None:
    """Get the detail widget if it exists."""
    return self._detail_widget

  def clear(self):
    """Clear the detail widget and show welcome screen."""
    if self._detail_widget:
      self._layout.removeWidget(self._detail_widget)
      self._detail_widget.deleteLater()
      self._detail_widget = None

    self._show_welcome()
