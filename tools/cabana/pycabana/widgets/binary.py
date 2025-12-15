"""BinaryView - displays CAN message bytes as bits with signal coloring."""

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QPersistentModelIndex, Signal as QtSignal
from PySide6.QtGui import QColor, QPainter, QFont, QFontDatabase, QMouseEvent
from PySide6.QtWidgets import (
  QWidget,
  QVBoxLayout,
  QTableView,
  QHeaderView,
  QStyledItemDelegate,
  QStyleOptionViewItem,
)

from openpilot.tools.cabana.pycabana.dbc.dbc import MessageId, CanData
from openpilot.tools.cabana.pycabana.dbc.dbcmanager import dbc_manager
from openpilot.tools.cabana.pycabana.settings import Settings, DragDirection

# Signal colors - same palette as C++ cabana
SIGNAL_COLORS = [
  QColor(102, 86, 169),   # purple
  QColor(76, 175, 80),    # green
  QColor(255, 152, 0),    # orange
  QColor(33, 150, 243),   # blue
  QColor(233, 30, 99),    # pink
  QColor(0, 188, 212),    # cyan
  QColor(255, 235, 59),   # yellow
  QColor(121, 85, 72),    # brown
]

CELL_HEIGHT = 36


def flip_bit_pos(pos: int) -> int:
  """Flip bit position for big endian calculation."""
  return (pos // 8) * 8 + 7 - (pos % 8)


def get_bit_pos(row: int, col: int) -> int:
  """Get absolute bit position from row and column."""
  return row * 8 + (7 - col)


class BinaryViewModel(QAbstractTableModel):
  """Model for the binary view - 8 bit columns + 1 hex column per row."""

  COLUMN_COUNT = 9  # 8 bits + 1 hex byte

  def __init__(self, parent=None):
    super().__init__(parent)
    self.msg_id: MessageId | None = None
    self._data: bytes = b''
    self._row_count = 0
    # items[row][col] = (bit_value, signal_indices, is_msb, is_lsb)
    self._items: list[list[dict]] = []

  def setMessage(self, msg_id: MessageId | None, can_data: CanData | None):
    """Set the message to display."""
    self.beginResetModel()
    self.msg_id = msg_id
    self._data = can_data.dat if can_data else b''
    self._rebuild()
    self.endResetModel()

  def updateData(self, can_data: CanData | None):
    """Update data without rebuilding signal mapping."""
    new_data = can_data.dat if can_data else b''
    if new_data == self._data:
      return

    old_row_count = self._row_count
    self._data = new_data
    new_row_count = len(self._data)

    if new_row_count > old_row_count:
      self.beginInsertRows(QModelIndex(), old_row_count, new_row_count - 1)
      self._row_count = new_row_count
      self._extend_items()
      self.endInsertRows()
    elif new_row_count < old_row_count:
      self.beginRemoveRows(QModelIndex(), new_row_count, old_row_count - 1)
      self._row_count = new_row_count
      self._items = self._items[:new_row_count]
      self.endRemoveRows()

    # Emit data changed for all cells
    if self._row_count > 0:
      self.dataChanged.emit(
        self.index(0, 0),
        self.index(self._row_count - 1, self.COLUMN_COUNT - 1)
      )

  def _rebuild(self):
    """Rebuild the signal mapping for each bit."""
    self._row_count = len(self._data) if self._data else 0
    if self.msg_id is not None:
      # Check DBC for message size
      msg = dbc_manager().msg(self.msg_id)
      if msg:
        self._row_count = max(self._row_count, msg.size)

    self._items = []
    for _ in range(self._row_count):
      row = []
      for _ in range(self.COLUMN_COUNT):
        row.append({'sig_indices': [], 'is_msb': False, 'is_lsb': False})
      self._items.append(row)

    # Map signals to bits
    if self.msg_id:
      msg = dbc_manager().msg(self.msg_id)
      if msg:
        for sig_idx, sig in enumerate(msg.sigs.values()):
          for j in range(sig.size):
            # Calculate bit position based on endianness
            if sig.is_little_endian:
              bit_pos = sig.lsb + j
            else:
              # Big endian: start from MSB
              bit_pos = sig.msb - (j % 8) + (j // 8) * 8

            byte_idx = bit_pos // 8
            bit_idx = bit_pos % 8

            if byte_idx < self._row_count and bit_idx < 8:
              item = self._items[byte_idx][bit_idx]
              item['sig_indices'].append(sig_idx)
              if j == 0:
                item['is_lsb' if sig.is_little_endian else 'is_msb'] = True
              if j == sig.size - 1:
                item['is_msb' if sig.is_little_endian else 'is_lsb'] = True

  def _extend_items(self):
    """Extend items list for new rows."""
    while len(self._items) < self._row_count:
      row = []
      for _ in range(self.COLUMN_COUNT):
        row.append({'sig_indices': [], 'is_msb': False, 'is_lsb': False})
      self._items.append(row)

  def rowCount(self, parent=None):
    if parent is None:
      parent = QModelIndex()
    if parent.isValid():
      return 0
    return self._row_count

  def columnCount(self, parent=None):
    if parent is None:
      parent = QModelIndex()
    if parent.isValid():
      return 0
    return self.COLUMN_COUNT

  def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
    if not index.isValid():
      return None

    row, col = index.row(), index.column()
    if row >= self._row_count:
      return None

    if role == Qt.ItemDataRole.DisplayRole:
      if row < len(self._data):
        byte_val = self._data[row]
        if col == 8:  # Hex column
          return f"{byte_val:02X}"
        else:  # Bit column (0-7, MSB first)
          bit_val = (byte_val >> (7 - col)) & 1
          return str(bit_val)
      return ""

    elif role == Qt.ItemDataRole.UserRole:
      # Return item data for delegate
      if row < len(self._items) and col < len(self._items[row]):
        item = self._items[row][col]
        byte_val = self._data[row] if row < len(self._data) else 0
        bit_val = (byte_val >> (7 - col)) & 1 if col < 8 else byte_val
        return {
          'value': bit_val,
          'byte_value': byte_val,
          'sig_indices': item['sig_indices'],
          'is_msb': item['is_msb'],
          'is_lsb': item['is_lsb'],
          'is_hex': col == 8,
          'valid': row < len(self._data),
        }
      return None

    return None

  def headerData(self, section, orientation, role: int = Qt.ItemDataRole.DisplayRole):
    if role == Qt.ItemDataRole.DisplayRole:
      if orientation == Qt.Orientation.Horizontal:
        if section == 8:
          return "Hex"
        return str(7 - section)  # Show bit position (MSB=7 to LSB=0)
      else:
        return str(section)
    return None


class BinaryItemDelegate(QStyledItemDelegate):
  """Custom delegate for painting binary cells with signal colors."""

  def __init__(self, parent=None):
    super().__init__(parent)
    self._hex_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    self._hex_font.setBold(True)
    self._small_font = QFont()
    self._small_font.setPixelSize(8)

  def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
    item = index.data(Qt.ItemDataRole.UserRole)
    if item is None:
      return

    painter.save()

    rect = option.rect
    is_hex = item['is_hex']
    sig_indices = item['sig_indices']
    valid = item['valid']

    # Background color
    if sig_indices:
      # Use first signal's color
      color_idx = sig_indices[0] % len(SIGNAL_COLORS)
      bg_color = QColor(SIGNAL_COLORS[color_idx])
      bg_color.setAlpha(180 if valid else 80)
      painter.fillRect(rect, bg_color)

      # Draw border for signal boundaries
      border_color = SIGNAL_COLORS[color_idx].darker(130)
      painter.setPen(border_color)
      painter.drawRect(rect.adjusted(0, 0, -1, -1))
    elif valid:
      # No signal - light gray
      painter.fillRect(rect, QColor(60, 60, 60, 40))

    # Mark overlapping signals
    if len(sig_indices) > 1:
      painter.fillRect(rect, QColor(100, 100, 100, 100))

    # Invalid data pattern
    if not valid:
      painter.fillRect(rect, QColor(80, 80, 80, 60))

    # Text
    if valid:
      if is_hex:
        painter.setFont(self._hex_font)
      text = index.data(Qt.ItemDataRole.DisplayRole)
      if text:
        painter.setPen(Qt.GlobalColor.white if sig_indices else Qt.GlobalColor.lightGray)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    # MSB/LSB markers
    if item['is_msb'] or item['is_lsb']:
      painter.setFont(self._small_font)
      painter.setPen(Qt.GlobalColor.white)
      marker = "M" if item['is_msb'] else "L"
      painter.drawText(rect.adjusted(2, 2, -2, -2), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom, marker)

    painter.restore()


class BinaryTableView(QTableView):
  """Custom table view with mouse handling for signal creation."""

  signalCreated = QtSignal(int, int, bool)  # start_bit, size, is_little_endian

  def __init__(self, model: BinaryViewModel, parent=None):
    super().__init__(parent)
    self._model = model
    self._anchor_index: QModelIndex | None = None
    self._resize_sig = None  # Signal being resized (not implemented yet)

  def mousePressEvent(self, event: QMouseEvent):
    """Handle mouse press - start selection."""
    if event.button() == Qt.MouseButton.LeftButton:
      index = self.indexAt(event.pos())
      if index.isValid() and index.column() != 8:  # Not hex column
        self._anchor_index = index
        self._resize_sig = None
        # TODO: Check if clicking on signal edge for resize
        event.accept()
        return
    super().mousePressEvent(event)

  def mouseMoveEvent(self, event: QMouseEvent):
    """Handle mouse move - update selection."""
    if self._anchor_index is not None and self._anchor_index.isValid():
      index = self.indexAt(event.pos())
      if index.isValid() and index.column() != 8:
        # Update selection between anchor and current
        self._updateSelection(index)
    super().mouseMoveEvent(event)

  def mouseReleaseEvent(self, event: QMouseEvent):
    """Handle mouse release - create signal from selection."""
    if event.button() == Qt.MouseButton.LeftButton and self._anchor_index is not None:
      release_index = self.indexAt(event.pos())
      if release_index.isValid() and self._anchor_index.isValid():
        selection = self.selectionModel().selectedIndexes()
        if selection:
          # Calculate signal parameters from selection
          start_bit, size, is_le = self._getSelectionParams(release_index)
          if size > 0:
            self.signalCreated.emit(start_bit, size, is_le)
      self.clearSelection()
      self._anchor_index = None
      self._resize_sig = None
    super().mouseReleaseEvent(event)

  def _updateSelection(self, current_index: QModelIndex):
    """Update selection between anchor and current index."""
    if not self._anchor_index or not current_index.isValid():
      return

    # Clear and set new selection
    self.clearSelection()
    selection = self.selectionModel()

    # Get range of cells to select
    start_row = min(self._anchor_index.row(), current_index.row())
    end_row = max(self._anchor_index.row(), current_index.row())
    start_col = min(self._anchor_index.column(), current_index.column())
    end_col = max(self._anchor_index.column(), current_index.column())

    # For simple rectangular selection
    for row in range(start_row, end_row + 1):
      for col in range(start_col, min(end_col + 1, 8)):  # Exclude hex column
        idx = self._model.index(row, col)
        selection.select(idx, selection.SelectionFlag.Select)

  def _getSelectionParams(self, release_index: QModelIndex) -> tuple[int, int, bool]:
    """Calculate start_bit, size, and endianness from selection."""
    settings = Settings()

    # Determine endianness based on drag direction setting
    is_le = True
    if settings.drag_direction == DragDirection.MsbFirst:
      is_le = release_index.row() < self._anchor_index.row() or (
        release_index.row() == self._anchor_index.row() and release_index.column() > self._anchor_index.column()
      )
    elif settings.drag_direction == DragDirection.LsbFirst:
      is_le = not (release_index.row() < self._anchor_index.row() or (
        release_index.row() == self._anchor_index.row() and release_index.column() > self._anchor_index.column()
      ))
    elif settings.drag_direction == DragDirection.AlwaysLE:
      is_le = True
    elif settings.drag_direction == DragDirection.AlwaysBE:
      is_le = False

    # Get bit positions
    anchor_bit = get_bit_pos(self._anchor_index.row(), self._anchor_index.column())
    release_bit = get_bit_pos(release_index.row(), release_index.column())

    if is_le:
      # Little endian: start_bit is the LSB
      start_bit = min(anchor_bit, release_bit)
      size = abs(anchor_bit - release_bit) + 1
    else:
      # Big endian: start_bit is MSB, calculate flipped positions
      anchor_flipped = flip_bit_pos(anchor_bit)
      release_flipped = flip_bit_pos(release_bit)
      start_bit = get_bit_pos(
        min(self._anchor_index.row(), release_index.row()),
        min(self._anchor_index.column(), release_index.column())
      )
      size = abs(anchor_flipped - release_flipped) + 1

    return start_bit, size, is_le


class BinaryView(QWidget):
  """Widget showing CAN message bytes as colored bits."""

  signalCreated = QtSignal(int, int, bool)  # start_bit, size, is_little_endian

  def __init__(self, parent=None):
    super().__init__(parent)
    self._msg_id: MessageId | None = None

    self._setup_ui()
    self._connect_signals()

  def _setup_ui(self):
    layout = QVBoxLayout(self)
    layout.setContentsMargins(0, 0, 0, 0)

    self._model = BinaryViewModel()
    self._delegate = BinaryItemDelegate()

    self._table = BinaryTableView(self._model)
    self._table.setModel(self._model)
    self._table.setItemDelegate(self._delegate)

    # Configure table appearance
    self._table.setShowGrid(False)
    self._table.setAlternatingRowColors(False)
    self._table.setSelectionMode(QTableView.SelectionMode.ContiguousSelection)

    # Horizontal header (bit positions)
    h_header = self._table.horizontalHeader()
    h_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    h_header.setMinimumSectionSize(30)
    h_header.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))

    # Vertical header (byte indices)
    v_header = self._table.verticalHeader()
    v_header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
    v_header.setDefaultSectionSize(CELL_HEIGHT)
    v_header.setMinimumWidth(30)

    self._table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    layout.addWidget(self._table)

  def _connect_signals(self):
    """Connect internal signals."""
    self._table.signalCreated.connect(self._on_signal_created)

  def _on_signal_created(self, start_bit: int, size: int, is_le: bool):
    """Handle signal creation from table view."""
    if self._msg_id is None:
      return

    from opendbc.can.dbc import Signal
    from openpilot.tools.cabana.pycabana.commands import AddSigCommand, UndoStack

    # Create a new signal
    sig = Signal(
      name="",  # Will be assigned by AddSigCommand
      start_bit=start_bit,
      size=size,
      is_signed=False,
      factor=1.0,
      offset=0.0,
      is_little_endian=is_le,
    )

    cmd = AddSigCommand(self._msg_id, sig)
    UndoStack.push(cmd)

    # Re-emit for parent widgets
    self.signalCreated.emit(start_bit, size, is_le)

  def setMessage(self, msg_id: MessageId | None, can_data: CanData | None):
    """Set the message to display."""
    self._msg_id = msg_id
    self._model.setMessage(msg_id, can_data)

  def updateData(self, can_data: CanData | None):
    """Update data values without changing message selection."""
    self._model.updateData(can_data)
