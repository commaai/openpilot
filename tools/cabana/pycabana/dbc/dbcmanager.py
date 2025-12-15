"""DBCManager - singleton for managing loaded DBC files."""

from PySide6.QtCore import QObject, Signal as QtSignal

from opendbc.can.dbc import DBC, Msg

from openpilot.tools.cabana.pycabana.dbc.dbc import MessageId


class DBCManager(QObject):
  """Singleton that manages loaded DBC files."""

  _instance: "DBCManager | None" = None

  dbcLoaded = QtSignal(str)  # dbc name

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


def dbc_manager() -> DBCManager:
  """Global accessor for DBCManager singleton."""
  return DBCManager.instance()
