"""SignalView - displays decoded signal values for a CAN message."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
  QWidget,
  QVBoxLayout,
  QLabel,
  QTableWidget,
  QTableWidgetItem,
  QHeaderView,
)

from openpilot.tools.cabana.pycabana.dbc.dbc import MessageId, CanData, decode_signal
from openpilot.tools.cabana.pycabana.dbc.dbcmanager import dbc_manager


class SignalView(QWidget):
  """Widget showing decoded signal values for a message."""

  def __init__(self, parent=None):
    super().__init__(parent)
    self._msg_id: MessageId | None = None

    self._setup_ui()

  def _setup_ui(self):
    layout = QVBoxLayout(self)
    layout.setContentsMargins(8, 8, 8, 8)

    # Header label
    self.header_label = QLabel("No message selected")
    self.header_label.setStyleSheet("font-weight: bold; font-size: 14px;")
    layout.addWidget(self.header_label)

    # Signals table
    self.table = QTableWidget()
    self.table.setColumnCount(2)
    self.table.setHorizontalHeaderLabels(["Signal", "Value"])
    self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
    self.table.verticalHeader().setVisible(False)
    self.table.setAlternatingRowColors(True)
    self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    layout.addWidget(self.table)

  def setMessage(self, msg_id: MessageId | None, can_data: CanData | None):
    """Update to show signals for the given message."""
    self._msg_id = msg_id

    if msg_id is None:
      self.header_label.setText("No message selected")
      self.table.setRowCount(0)
      return

    msg = dbc_manager().msg(msg_id)
    if msg is None:
      self.header_label.setText(f"0x{msg_id.address:X} (no DBC)")
      self.table.setRowCount(0)
      return

    self.header_label.setText(f"{msg.name} (0x{msg_id.address:X})")

    # Populate signals table
    signals = list(msg.sigs.values())
    self.table.setRowCount(len(signals))

    data = can_data.dat if can_data else b''

    for i, sig in enumerate(signals):
      # Signal name
      name_item = QTableWidgetItem(sig.name)
      name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
      self.table.setItem(i, 0, name_item)

      # Signal value
      if data:
        value = decode_signal(sig, data)
        value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
      else:
        value_str = "N/A"
      value_item = QTableWidgetItem(value_str)
      value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
      value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
      self.table.setItem(i, 1, value_item)

  def updateValues(self, can_data: CanData | None):
    """Update signal values without changing the selected message."""
    if self._msg_id is None:
      return

    msg = dbc_manager().msg(self._msg_id)
    if msg is None:
      return

    data = can_data.dat if can_data else b''
    signals = list(msg.sigs.values())

    for i, sig in enumerate(signals):
      if i < self.table.rowCount():
        if data:
          value = decode_signal(sig, data)
          value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
        else:
          value_str = "N/A"
        item = self.table.item(i, 1)
        if item:
          item.setText(value_str)
