"""HistoryLog - displays individual CAN message events over time."""

from collections import deque
from collections.abc import Callable
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QPersistentModelIndex, QSize, QRect
from PySide6.QtGui import QColor, QPainter, QBrush, QPalette
from PySide6.QtWidgets import (
  QWidget,
  QFrame,
  QVBoxLayout,
  QHBoxLayout,
  QTableView,
  QHeaderView,
  QComboBox,
  QLineEdit,
  QPushButton,
  QFileDialog,
  QStyledItemDelegate,
  QStyleOptionViewItem,
)

from openpilot.tools.cabana.pycabana.dbc.dbc import MessageId, decode_signal
from openpilot.tools.cabana.pycabana.dbc.dbcmanager import dbc_manager
from openpilot.tools.cabana.pycabana.streams.abstract import AbstractStream


# Custom roles for byte data display
ColorsRole = Qt.ItemDataRole.UserRole + 1
BytesRole = Qt.ItemDataRole.UserRole + 2


class Message:
  """Represents a single CAN message event with decoded signal values."""

  def __init__(self, mono_time: int, sig_values: list[float], data: bytes, colors: list[QColor] | None = None):
    self.mono_time = mono_time
    self.sig_values = sig_values
    self.data = data
    self.colors = colors if colors else []


class HistoryLogModel(QAbstractTableModel):
  """Model for displaying history of CAN message events."""

  def __init__(self, stream: AbstractStream, parent=None):
    super().__init__(parent)
    self.stream = stream
    self.msg_id: MessageId | None = None
    self.messages: deque[Message] = deque()
    self.sigs: list = []
    self.hex_mode = False
    self.batch_size = 50
    self.filter_sig_idx = -1
    self.filter_value = 0.0
    self.filter_cmp: Callable[[float, float], bool] | None = None

  def setMessage(self, msg_id: MessageId | None):
    """Set the message to display."""
    self.msg_id = msg_id
    self.reset()

  def reset(self):
    """Reset the model and rebuild signal list."""
    self.beginResetModel()
    self.sigs = []
    if self.msg_id:
      msg = dbc_manager().msg(self.msg_id)
      if msg:
        # Convert dict values to list
        self.sigs = list(msg.sigs.values())
    self.messages.clear()
    self.endResetModel()
    self.setFilter(0, "", None)

  def setHexMode(self, hex_mode: bool):
    """Toggle between signal value mode and hex mode."""
    self.hex_mode = hex_mode
    self.reset()

  def isHexMode(self) -> bool:
    """Check if in hex mode."""
    return len(self.sigs) == 0 or self.hex_mode

  def setFilter(self, sig_idx: int, value: str, cmp: Callable[[float, float], bool] | None):
    """Set filter for signal values."""
    self.filter_sig_idx = sig_idx
    try:
      self.filter_value = float(value) if value else 0.0
    except ValueError:
      self.filter_value = 0.0
    self.filter_cmp = cmp if value else None
    self.updateState(clear=True)

  def updateState(self, clear: bool = False):
    """Update the model with new events from the stream."""
    if clear and len(self.messages) > 0:
      self.beginRemoveRows(QModelIndex(), 0, len(self.messages) - 1)
      self.messages.clear()
      self.endRemoveRows()

    if not self.msg_id:
      return

    # Get current time boundary
    last_msg = self.stream.lastMessage(self.msg_id)
    if not last_msg:
      return

    current_time = int(last_msg.ts * 1e9) + self.stream.start_ts + 1
    min_time = self.messages[0].mono_time if self.messages else 0
    self._fetchData(0, current_time, min_time)

  def canFetchMore(self, parent: QModelIndex | QPersistentModelIndex | None = None) -> bool:
    """Check if more data can be fetched."""
    if not self.msg_id or len(self.messages) == 0:
      return False
    events = self.stream.events.get(self.msg_id, [])
    if not events:
      return False
    return self.messages[-1].mono_time > events[0].mono_time

  def fetchMore(self, parent: QModelIndex | QPersistentModelIndex | None = None):
    """Fetch more historical data."""
    if len(self.messages) > 0:
      self._fetchData(len(self.messages), self.messages[-1].mono_time, 0)

  def _fetchData(self, insert_pos: int, from_time: int, min_time: int):
    """Fetch and decode events from the stream."""
    if not self.msg_id:
      return

    events = self.stream.events.get(self.msg_id, [])
    if not events:
      return

    # Find events in time range (reverse chronological order)
    msgs: list[Message] = []
    for event in reversed(events):
      if event.mono_time >= from_time:
        continue
      if event.mono_time <= min_time:
        break

      # Decode signal values
      sig_values = []
      for sig in self.sigs:
        value = decode_signal(sig, event.dat)
        sig_values.append(value)

      # Apply filter
      if self.filter_cmp and len(sig_values) > self.filter_sig_idx:
        if not self.filter_cmp(sig_values[self.filter_sig_idx], self.filter_value):
          continue

      # Create message entry
      msgs.append(Message(event.mono_time, sig_values, event.dat))

      # Limit batch size when loading newest data
      if len(msgs) >= self.batch_size and min_time == 0:
        break

    # Insert new messages
    if msgs:
      self.beginInsertRows(QModelIndex(), insert_pos, insert_pos + len(msgs) - 1)
      for i, msg in enumerate(msgs):
        self.messages.insert(insert_pos + i, msg)
      self.endInsertRows()

  def rowCount(self, parent: QModelIndex | QPersistentModelIndex | None = None) -> int:
    if parent is not None and parent.isValid():
      return 0
    return len(self.messages)

  def columnCount(self, parent: QModelIndex | QPersistentModelIndex | None = None) -> int:
    if parent is not None and parent.isValid():
      return 0
    return 2 if self.isHexMode() else len(self.sigs) + 1

  def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
    if not index.isValid() or index.row() >= len(self.messages):
      return None

    msg = self.messages[index.row()]
    col = index.column()

    if role == Qt.ItemDataRole.DisplayRole:
      if col == 0:
        # Time column
        time_sec = self.stream.toSeconds(msg.mono_time)
        return f"{time_sec:.3f}"
      if not self.isHexMode() and col <= len(self.sigs):
        # Signal value column
        sig = self.sigs[col - 1]
        value = msg.sig_values[col - 1]
        # Format signal value with unit
        if sig.unit:
          return f"{value:.6g} {sig.unit}"
        return f"{value:.6g}"
    elif role == Qt.ItemDataRole.TextAlignmentRole:
      return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    elif self.isHexMode() and col == 1:
      # Hex mode - return data for delegate
      if role == BytesRole:
        return msg.data
      if role == ColorsRole:
        return msg.colors

    return None

  def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
    if orientation == Qt.Orientation.Horizontal:
      if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.ToolTipRole:
        if section == 0:
          return "Time"
        if self.isHexMode():
          return "Data"
        if section <= len(self.sigs):
          sig = self.sigs[section - 1]
          if sig.unit:
            return f"{sig.name} ({sig.unit})"
          return sig.name
      elif role == Qt.ItemDataRole.BackgroundRole and section > 0 and not self.isHexMode():
        # Signal color with alpha for contrast
        if section <= len(self.sigs):
          sig = self.sigs[section - 1]
          color = QColor(sig.color)
          color.setAlpha(128)
          return QBrush(color)
    return None


class HeaderView(QHeaderView):
  """Custom header view with word-wrapped text and custom sizing."""

  def __init__(self, orientation: Qt.Orientation, parent=None):
    super().__init__(orientation, parent)
    self.setDefaultAlignment(Qt.AlignmentFlag.AlignRight | Qt.TextFlag.TextWordWrap)

  def sectionSizeFromContents(self, logicalIndex: int) -> QSize:
    """Calculate section size with word wrapping."""
    time_col_size = QSize(
      self.fontMetrics().horizontalAdvance("000000.000") + 10,
      self.fontMetrics().height() + 6
    )

    if logicalIndex == 0:
      return time_col_size
    else:
      model = self.model()
      if not model:
        return QSize(100, time_col_size.height())

      col_count = model.columnCount()
      default_size = max(100, (self.rect().width() - time_col_size.width()) // max(1, col_count - 1))

      text = str(model.headerData(logicalIndex, self.orientation(), Qt.ItemDataRole.DisplayRole) or "")
      text = text.replace('_', ' ')

      rect = self.fontMetrics().boundingRect(
        QRect(0, 0, default_size, 2000),
        self.defaultAlignment(),
        text
      )
      size = QSize(rect.width() + 10, rect.height() + 6)
      return QSize(max(size.width(), default_size), size.height())

  def paintSection(self, painter: QPainter, rect: QRect, logicalIndex: int):
    """Paint section with custom background color."""
    model = self.model()
    if model:
      bg_role = model.headerData(logicalIndex, Qt.Orientation.Horizontal, Qt.ItemDataRole.BackgroundRole)
      if bg_role:
        painter.fillRect(rect, bg_role)

      text = str(model.headerData(logicalIndex, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole) or "")
      text = text.replace('_', ' ')

      # Use palette color for text
      painter.setPen(self.palette().color(QPalette.ColorRole.Text))
      painter.drawText(rect.adjusted(5, 3, -5, -3), self.defaultAlignment(), text)


class MessageBytesDelegate(QStyledItemDelegate):
  """Delegate for rendering hex bytes with colors."""

  def __init__(self, parent=None):
    super().__init__(parent)
    # Pre-compute font metrics for fixed-width font
    from PySide6.QtGui import QFontDatabase
    self.fixed_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    from PySide6.QtGui import QFontMetricsF
    fm = QFontMetricsF(self.fixed_font)
    self.byte_width = fm.horizontalAdvance("00 ")
    self.byte_height = fm.height()
    self.h_margin = 6
    self.v_margin = 4

  def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex | QPersistentModelIndex):
    """Paint hex bytes with background colors."""
    data = index.data(BytesRole)
    colors = index.data(ColorsRole)

    if not isinstance(data, bytes):
      super().paint(painter, option, index)
      return

    # Fill background
    if option.state & QStyleOptionViewItem.StateFlag.State_Selected:
      painter.fillRect(option.rect, option.palette.highlight())

    painter.setFont(self.fixed_font)

    # Paint each byte with its color
    x = option.rect.x() + self.h_margin
    y = option.rect.y() + self.v_margin + self.byte_height

    for i, byte_val in enumerate(data):
      # Draw background color if available
      if colors and isinstance(colors, list) and i < len(colors):
        color = colors[i]
        if isinstance(color, QColor) and color.isValid():
          bg_rect = QRect(int(x), option.rect.y(), int(self.byte_width), option.rect.height())
          painter.fillRect(bg_rect, color)

      # Draw hex text
      hex_str = f"{byte_val:02X}"
      painter.setPen(option.palette.color(QPalette.ColorRole.Text))
      painter.drawText(int(x), int(y), hex_str)
      x += self.byte_width

  def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex | QPersistentModelIndex) -> QSize:
    """Calculate size hint for hex bytes."""
    data = index.data(BytesRole)
    if isinstance(data, bytes) and len(data) > 0:
      width = int(len(data) * self.byte_width + 2 * self.h_margin)
      height = int(self.byte_height + 2 * self.v_margin)
      return QSize(width, height)
    return QSize(100, int(self.byte_height + 2 * self.v_margin))


class HistoryLogWidget(QFrame):
  """Widget for displaying message history log."""

  def __init__(self, stream: AbstractStream, parent=None):
    super().__init__(parent)
    self.stream = stream
    self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Plain)

    self._setup_ui()
    self._connect_signals()

  def _setup_ui(self):
    """Set up the UI components."""
    main_layout = QVBoxLayout(self)
    main_layout.setContentsMargins(0, 0, 0, 0)
    main_layout.setSpacing(0)

    # Toolbar
    toolbar = QWidget()
    toolbar.setAutoFillBackground(True)
    toolbar_layout = QHBoxLayout(toolbar)

    # Filters widget
    self.filters_widget = QWidget()
    filter_layout = QHBoxLayout(self.filters_widget)
    filter_layout.setContentsMargins(0, 0, 0, 0)

    self.display_type_cb = QComboBox()
    self.display_type_cb.addItems(["Signal", "Hex"])
    self.display_type_cb.setToolTip("Display signal value or raw hex value")
    filter_layout.addWidget(self.display_type_cb)

    self.signals_cb = QComboBox()
    filter_layout.addWidget(self.signals_cb)

    self.comp_box = QComboBox()
    self.comp_box.addItems([">", "=", "!=", "<"])
    filter_layout.addWidget(self.comp_box)

    self.value_edit = QLineEdit()
    self.value_edit.setClearButtonEnabled(True)
    from PySide6.QtGui import QDoubleValidator
    self.value_edit.setValidator(QDoubleValidator())
    filter_layout.addWidget(self.value_edit)

    toolbar_layout.addWidget(self.filters_widget)
    toolbar_layout.addStretch()

    self.export_btn = QPushButton("Export to CSV...")
    self.export_btn.setEnabled(False)
    toolbar_layout.addWidget(self.export_btn, alignment=Qt.AlignmentFlag.AlignRight)

    main_layout.addWidget(toolbar)

    # Separator line
    line = QFrame()
    line.setFrameStyle(QFrame.Shape.HLine | QFrame.Shadow.Sunken)
    main_layout.addWidget(line)

    # Table view
    self.logs = QTableView()
    self.model = HistoryLogModel(self.stream)
    self.logs.setModel(self.model)

    self.delegate = MessageBytesDelegate()
    self.logs.setItemDelegate(self.delegate)

    self.logs.setHorizontalHeader(HeaderView(Qt.Orientation.Horizontal))
    self.logs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    self.logs.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
    self.logs.verticalHeader().setDefaultSectionSize(int(self.delegate.byte_height + 2 * self.delegate.v_margin))
    self.logs.setFrameShape(QFrame.Shape.NoFrame)

    main_layout.addWidget(self.logs)

  def _connect_signals(self):
    """Connect signal handlers."""
    self.display_type_cb.activated.connect(self._on_display_type_changed)
    self.signals_cb.activated.connect(self._on_filter_changed)
    self.comp_box.activated.connect(self._on_filter_changed)
    self.value_edit.textEdited.connect(self._on_filter_changed)
    self.export_btn.clicked.connect(self._on_export_csv)

    # Stream signals
    self.stream.seekedTo.connect(lambda: self.model.reset())

    # DBC signals
    dbc_manager().dbcLoaded.connect(lambda: self.model.reset())

    # Model signals
    self.model.modelReset.connect(self._on_model_reset)
    self.model.rowsInserted.connect(lambda: self.export_btn.setEnabled(True))

  def setMessage(self, msg_id: MessageId | None):
    """Set the message to display."""
    self.model.setMessage(msg_id)

  def updateState(self):
    """Update the model state with latest data."""
    self.model.updateState()

  def showEvent(self, event):
    """Handle show event."""
    super().showEvent(event)
    self.model.updateState(clear=True)

  def _on_display_type_changed(self, index: int):
    """Handle display type change."""
    self.model.setHexMode(index == 1)

  def _on_model_reset(self):
    """Handle model reset."""
    self.signals_cb.clear()
    for sig in self.model.sigs:
      self.signals_cb.addItem(sig.name)
    self.export_btn.setEnabled(False)
    self.value_edit.clear()
    self.comp_box.setCurrentIndex(0)
    self.filters_widget.setVisible(len(self.model.sigs) > 0)

  def _on_filter_changed(self):
    """Handle filter change."""
    text = self.value_edit.text()
    if not text and not self.value_edit.isModified():
      return

    # Map comparison operator
    cmp_funcs = [
      lambda l, r: l > r,
      lambda l, r: l == r,
      lambda l, r: l != r,
      lambda l, r: l < r,
    ]
    cmp = cmp_funcs[self.comp_box.currentIndex()] if self.comp_box.currentIndex() < len(cmp_funcs) else None

    self.model.setFilter(self.signals_cb.currentIndex(), text, cmp)

  def _on_export_csv(self):
    """Export history to CSV file."""
    if not self.model.msg_id:
      return

    msg_name = dbc_manager().msgName(self.model.msg_id)
    route_name = self.stream.routeName if hasattr(self.stream, 'routeName') else "route"
    default_filename = f"{route_name}_{msg_name}.csv"

    filename, _ = QFileDialog.getSaveFileName(
      self,
      f"Export {msg_name} to CSV file",
      default_filename,
      "CSV files (*.csv)"
    )

    if filename:
      self._export_to_csv(filename)

  def _export_to_csv(self, filename: str):
    """Write data to CSV file."""
    try:
      with open(filename, 'w') as f:
        # Write header
        if self.model.isHexMode():
          f.write("Time,Data\n")
        else:
          header = ["Time"] + [sig.name for sig in self.model.sigs]
          f.write(",".join(header) + "\n")

        # Write data rows
        for msg in self.model.messages:
          time_sec = self.stream.toSeconds(msg.mono_time)
          if self.model.isHexMode():
            hex_str = msg.data.hex(' ').upper()
            f.write(f"{time_sec:.3f},{hex_str}\n")
          else:
            values = [f"{time_sec:.3f}"] + [f"{val:.6g}" for val in msg.sig_values]
            f.write(",".join(values) + "\n")

    except Exception as e:
      print(f"Failed to export CSV: {e}")
