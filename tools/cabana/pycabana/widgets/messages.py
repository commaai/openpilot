"""MessagesWidget - displays CAN messages in a table."""

from PySide2.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Signal
from PySide2.QtWidgets import (
  QWidget,
  QVBoxLayout,
  QHBoxLayout,
  QLineEdit,
  QTableView,
  QHeaderView,
  QAbstractItemView,
  QPushButton,
)

from openpilot.tools.cabana.pycabana.dbc.dbc import MessageId, CanData
from openpilot.tools.cabana.pycabana.streams.abstract import AbstractStream


class MessageListModel(QAbstractTableModel):
  """Model for the messages table."""

  COLUMNS = ['Bus', 'Address', 'Count', 'Freq', 'Data']
  COL_BUS = 0
  COL_ADDRESS = 1
  COL_COUNT = 2
  COL_FREQ = 3
  COL_DATA = 4

  def __init__(self, stream: AbstractStream, parent=None):
    super().__init__(parent)
    self.stream = stream
    self.msg_ids: list[MessageId] = []

  def rowCount(self, parent=QModelIndex()):
    if parent.isValid():
      return 0
    return len(self.msg_ids)

  def columnCount(self, parent=QModelIndex()):
    if parent.isValid():
      return 0
    return len(self.COLUMNS)

  def data(self, index: QModelIndex, role=Qt.DisplayRole):
    if not index.isValid() or index.row() >= len(self.msg_ids):
      return None

    msg_id = self.msg_ids[index.row()]
    can_data = self.stream.last_msgs.get(msg_id)

    if role == Qt.DisplayRole:
      col = index.column()
      if col == self.COL_BUS:
        return str(msg_id.source)
      elif col == self.COL_ADDRESS:
        return f"0x{msg_id.address:X}"
      elif col == self.COL_COUNT:
        return str(can_data.count) if can_data else "0"
      elif col == self.COL_FREQ:
        return f"{can_data.freq:.1f}" if can_data else "0.0"
      elif col == self.COL_DATA:
        if can_data and can_data.dat:
          return can_data.dat.hex(' ').upper()
        return ""

    elif role == Qt.TextAlignmentRole:
      col = index.column()
      if col in (self.COL_COUNT, self.COL_FREQ):
        return Qt.AlignRight | Qt.AlignVCenter
      return Qt.AlignLeft | Qt.AlignVCenter

    elif role == Qt.UserRole:
      # Return MessageId for selection handling
      return msg_id

    return None

  def headerData(self, section, orientation, role=Qt.DisplayRole):
    if orientation == Qt.Horizontal and role == Qt.DisplayRole:
      if 0 <= section < len(self.COLUMNS):
        return self.COLUMNS[section]
    return None

  def updateMessages(self, msg_ids: set[MessageId], has_new: bool):
    """Update the model with new message IDs."""
    if has_new:
      # Add new message IDs
      new_ids = msg_ids - set(self.msg_ids)
      if new_ids:
        start = len(self.msg_ids)
        end = start + len(new_ids) - 1
        self.beginInsertRows(QModelIndex(), start, end)
        self.msg_ids.extend(sorted(new_ids))
        self.endInsertRows()

    # Emit dataChanged for all rows to update counts/freq/data
    if self.msg_ids:
      top_left = self.index(0, self.COL_COUNT)
      bottom_right = self.index(len(self.msg_ids) - 1, self.COL_DATA)
      self.dataChanged.emit(top_left, bottom_right)

  def getMsgId(self, row: int) -> MessageId | None:
    """Get MessageId for a row."""
    if 0 <= row < len(self.msg_ids):
      return self.msg_ids[row]
    return None


class MessageFilterProxyModel(QSortFilterProxyModel):
  """Proxy model for filtering messages by address or name."""

  def __init__(self, parent=None):
    super().__init__(parent)
    self.filter_text = ""

  def setFilterText(self, text: str):
    self.filter_text = text.lower()
    self.invalidateFilter()

  def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
    if not self.filter_text:
      return True

    source_model = self.sourceModel()
    # Check address column
    address_index = source_model.index(source_row, MessageListModel.COL_ADDRESS)
    address = source_model.data(address_index, Qt.DisplayRole)
    if address and self.filter_text in address.lower():
      return True

    return False


class MessagesWidget(QWidget):
  """Widget displaying the list of CAN messages."""

  msgSelectionChanged = Signal(object)  # MessageId or None

  def __init__(self, stream: AbstractStream, parent=None):
    super().__init__(parent)
    self.stream = stream

    self._setup_ui()
    self._connect_signals()

  def _setup_ui(self):
    layout = QVBoxLayout(self)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    # Filter bar
    filter_layout = QHBoxLayout()
    filter_layout.setContentsMargins(4, 4, 4, 0)

    self.filter_input = QLineEdit()
    self.filter_input.setPlaceholderText("Filter by address...")
    self.filter_input.setClearButtonEnabled(True)
    filter_layout.addWidget(self.filter_input)

    self.clear_btn = QPushButton("Clear")
    self.clear_btn.setFixedWidth(60)
    filter_layout.addWidget(self.clear_btn)

    layout.addLayout(filter_layout)

    # Table view
    self.model = MessageListModel(self.stream)
    self.proxy_model = MessageFilterProxyModel()
    self.proxy_model.setSourceModel(self.model)

    self.table_view = QTableView()
    self.table_view.setModel(self.proxy_model)
    self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
    self.table_view.setSelectionMode(QAbstractItemView.SingleSelection)
    self.table_view.setSortingEnabled(True)
    self.table_view.setAlternatingRowColors(True)
    self.table_view.verticalHeader().setVisible(False)

    # Column sizing
    header = self.table_view.horizontalHeader()
    header.setSectionResizeMode(MessageListModel.COL_BUS, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(MessageListModel.COL_ADDRESS, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(MessageListModel.COL_COUNT, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(MessageListModel.COL_FREQ, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(MessageListModel.COL_DATA, QHeaderView.Stretch)

    layout.addWidget(self.table_view)

  def _connect_signals(self):
    # Filter input
    self.filter_input.textChanged.connect(self.proxy_model.setFilterText)
    self.clear_btn.clicked.connect(self.filter_input.clear)

    # Selection
    self.table_view.selectionModel().selectionChanged.connect(self._on_selection_changed)

    # Stream updates
    self.stream.msgsReceived.connect(self._on_msgs_received)

  def _on_selection_changed(self, selected, deselected):
    """Handle selection change in the table."""
    indexes = self.table_view.selectionModel().selectedRows()
    if indexes:
      # Map proxy index to source index
      proxy_index = indexes[0]
      source_index = self.proxy_model.mapToSource(proxy_index)
      msg_id = self.model.getMsgId(source_index.row())
      self.msgSelectionChanged.emit(msg_id)
    else:
      self.msgSelectionChanged.emit(None)

  def _on_msgs_received(self, msg_ids: set[MessageId], has_new: bool):
    """Handle new messages from stream."""
    self.model.updateMessages(msg_ids, has_new)

  def selectMessage(self, msg_id: MessageId):
    """Programmatically select a message."""
    for row, mid in enumerate(self.model.msg_ids):
      if mid == msg_id:
        source_index = self.model.index(row, 0)
        proxy_index = self.proxy_model.mapFromSource(source_index)
        self.table_view.selectRow(proxy_index.row())
        break
