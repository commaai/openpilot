"""DBCManager - singleton for managing loaded DBC files."""

from PySide6.QtCore import QObject, Signal as QtSignal

from opendbc.can.dbc import DBC, Msg, Signal

from openpilot.tools.cabana.pycabana.dbc.dbc import MessageId


class DBCManager(QObject):
  """Singleton that manages loaded DBC files."""

  _instance: "DBCManager | None" = None

  # Signals
  dbcLoaded = QtSignal(str)  # dbc name
  signalAdded = QtSignal(MessageId, Signal)  # msg_id, signal
  signalRemoved = QtSignal(Signal)  # signal
  signalUpdated = QtSignal(Signal)  # signal
  msgUpdated = QtSignal(MessageId)  # msg_id
  msgRemoved = QtSignal(MessageId)  # msg_id
  DBCFileChanged = QtSignal()
  maskUpdated = QtSignal()

  def __init__(self, parent=None):
    super().__init__(parent)
    self._dbc: DBC | None = None
    self._name: str = ""

  @classmethod
  def instance(cls) -> "DBCManager":
    if cls._instance is None:
      cls._instance = DBCManager()
    return cls._instance

  def load(self, dbc_name: str) -> bool:
    """Load a DBC file by name."""
    try:
      self._dbc = DBC(dbc_name)
      self._name = dbc_name
      self.dbcLoaded.emit(dbc_name)
      return True
    except Exception as e:
      print(f"Failed to load DBC {dbc_name}: {e}")
      return False

  def save(self, filename: str) -> bool:
    """Save the DBC to a file."""
    if self._dbc is None:
      return False
    try:
      from openpilot.tools.cabana.pycabana.dbc.dbcfile import DBCFile
      dbc_file = DBCFile(self._dbc)
      dbc_string = dbc_file.to_dbc_string()
      with open(filename, 'w') as f:
        f.write(dbc_string)
      self._name = filename
      return True
    except Exception as e:
      print(f"Failed to save DBC to {filename}: {e}")
      return False

  def clear(self) -> None:
    """Clear the current DBC and create an empty one."""
    self._dbc = DBC("")
    self._name = ""
    self.DBCFileChanged.emit()

  @property
  def name(self) -> str:
    return self._name

  @property
  def dbc(self) -> DBC | None:
    return self._dbc

  def msg(self, msg_id: MessageId) -> Msg | None:
    """Get message definition by ID."""
    if self._dbc is None:
      return None
    return self._dbc.msgs.get(msg_id.address)

  def msgName(self, msg_id: MessageId) -> str:
    """Get message name, or empty string if unknown."""
    msg = self.msg(msg_id)
    return msg.name if msg else ""

  def addSignal(self, msg_id: MessageId, sig: Signal) -> None:
    """Add a signal to a message."""
    msg = self.msg(msg_id)
    if msg and self._dbc:
      msg.sigs[sig.name] = sig
      self.signalAdded.emit(msg_id, sig)
      self.maskUpdated.emit()

  def updateSignal(self, msg_id: MessageId, sig_name: str, new_sig: Signal) -> None:
    """Update a signal in a message."""
    msg = self.msg(msg_id)
    if msg and sig_name in msg.sigs:
      # If name changed, remove old and add new
      if sig_name != new_sig.name:
        del msg.sigs[sig_name]
        msg.sigs[new_sig.name] = new_sig
      else:
        msg.sigs[sig_name] = new_sig
      self.signalUpdated.emit(new_sig)
      self.maskUpdated.emit()

  def removeSignal(self, msg_id: MessageId, sig_name: str) -> None:
    """Remove a signal from a message."""
    msg = self.msg(msg_id)
    if msg and sig_name in msg.sigs:
      sig = msg.sigs[sig_name]
      self.signalRemoved.emit(sig)
      del msg.sigs[sig_name]
      self.maskUpdated.emit()

  def updateMsg(self, msg_id: MessageId, name: str, size: int, node: str, comment: str) -> None:
    """Update or create a message."""
    if self._dbc is None:
      return

    # Get existing message or create new one
    msg = self.msg(msg_id)
    if msg:
      # Update existing message
      msg.name = name
      msg.size = size
      if hasattr(msg, 'transmitter'):
        msg.transmitter = node
      if hasattr(msg, 'comment'):
        msg.comment = comment
    else:
      # Create new message
      new_msg = Msg(name=name, address=msg_id.address, size=size, sigs={})
      if hasattr(new_msg, 'transmitter'):
        new_msg.transmitter = node
      if hasattr(new_msg, 'comment'):
        new_msg.comment = comment
      self._dbc.msgs[msg_id.address] = new_msg

    self.msgUpdated.emit(msg_id)

  def removeMsg(self, msg_id: MessageId) -> None:
    """Remove a message from the DBC."""
    if self._dbc and msg_id.address in self._dbc.msgs:
      del self._dbc.msgs[msg_id.address]
      self.msgRemoved.emit(msg_id)
      self.maskUpdated.emit()

  def newMsgName(self, msg_id: MessageId) -> str:
    """Generate a new message name."""
    return f"NEW_MSG_{msg_id.address:X}"

  def newSignalName(self, msg_id: MessageId) -> str:
    """Generate a new signal name for a message."""
    msg = self.msg(msg_id)
    if not msg:
      return "NEW_SIGNAL_1"

    # Find the next available signal name
    i = 1
    while True:
      name = f"NEW_SIGNAL_{i}"
      if name not in msg.sigs:
        return name
      i += 1


def dbc_manager() -> DBCManager:
  """Global accessor for DBCManager singleton."""
  return DBCManager.instance()
