"""Undo/redo commands for pycabana."""

from typing import Optional

from PySide6.QtCore import QCoreApplication, QObject
from PySide6.QtGui import QUndoCommand, QUndoStack
from PySide6.QtWidgets import QApplication

from opendbc.can.dbc import Signal, Msg

from openpilot.tools.cabana.pycabana.dbc.dbc import MessageId
from openpilot.tools.cabana.pycabana.dbc.dbcmanager import dbc_manager
from openpilot.tools.cabana.pycabana.streams.abstract import AbstractStream


# Signal type constants (matching cabana::Signal::Type)
class SignalType:
  Normal = 0
  Multiplexed = 1
  Multiplexor = 2


def _get_stream() -> Optional[AbstractStream]:
  """Get the global stream instance from the application."""
  app = QApplication.instance()
  if app and hasattr(app, 'stream'):
    return app.stream
  return None


class EditMsgCommand(QUndoCommand):
  """Command to edit or create a message."""

  def __init__(
    self,
    msg_id: MessageId,
    name: str,
    size: int,
    node: str,
    comment: str,
    parent: Optional[QUndoCommand] = None
  ):
    super().__init__(parent)
    self.id = msg_id
    self.new_name = name
    self.new_size = size
    self.new_node = node
    self.new_comment = comment

    self.old_name = ""
    self.old_size = 0
    self.old_node = ""
    self.old_comment = ""

    msg = dbc_manager().msg(msg_id)
    if msg:
      self.old_name = msg.name
      self.old_size = msg.size
      self.old_node = getattr(msg, 'transmitter', '')
      self.old_comment = getattr(msg, 'comment', '')
      self.setText(QObject.tr(f"edit message {name}:{msg_id.address}"))
    else:
      self.setText(QObject.tr(f"new message {name}:{msg_id.address}"))

  def undo(self) -> None:
    if not self.old_name:
      dbc_manager().removeMsg(self.id)
    else:
      dbc_manager().updateMsg(self.id, self.old_name, self.old_size, self.old_node, self.old_comment)

  def redo(self) -> None:
    dbc_manager().updateMsg(self.id, self.new_name, self.new_size, self.new_node, self.new_comment)


class RemoveMsgCommand(QUndoCommand):
  """Command to remove a message."""

  def __init__(self, msg_id: MessageId, parent: Optional[QUndoCommand] = None):
    super().__init__(parent)
    self.id = msg_id
    self.message: Optional[Msg] = None
    self.signals: list[Signal] = []

    msg = dbc_manager().msg(msg_id)
    if msg:
      # Store a copy of the message
      self.message = msg
      # Store copies of all signals
      self.signals = list(msg.sigs.values()) if hasattr(msg, 'sigs') else []
      self.setText(QObject.tr(f"remove message {msg.name}:{msg_id.address}"))

  def undo(self) -> None:
    if self.message:
      dbc_manager().updateMsg(
        self.id,
        self.message.name,
        self.message.size,
        getattr(self.message, 'transmitter', ''),
        getattr(self.message, 'comment', '')
      )
      # Restore all signals
      for sig in self.signals:
        dbc_manager().addSignal(self.id, sig)

  def redo(self) -> None:
    if self.message:
      dbc_manager().removeMsg(self.id)


class AddSigCommand(QUndoCommand):
  """Command to add a signal to a message."""

  def __init__(self, msg_id: MessageId, sig: Signal, parent: Optional[QUndoCommand] = None):
    super().__init__(parent)
    self.id = msg_id
    self.signal = sig
    self.msg_created = False

    msg_name = dbc_manager().msgName(msg_id)
    self.setText(QObject.tr(f"add signal {sig.name} to {msg_name}:{msg_id.address}"))

  def undo(self) -> None:
    dbc_manager().removeSignal(self.id, self.signal.name)
    if self.msg_created:
      dbc_manager().removeMsg(self.id)

  def redo(self) -> None:
    msg = dbc_manager().msg(self.id)
    if not msg:
      # Create a new message if it doesn't exist
      self.msg_created = True
      stream = _get_stream()
      last_msg = stream.lastMessage(self.id) if stream else None
      msg_size = len(last_msg.dat) if last_msg else 8
      dbc_manager().updateMsg(
        self.id,
        dbc_manager().newMsgName(self.id),
        msg_size,
        "",
        ""
      )

    # Update signal name and max value
    self.signal.name = dbc_manager().newSignalName(self.id)
    # Calculate max value: 2^size - 1
    max_val = (2 ** self.signal.size) - 1
    # Create a new signal with updated values
    updated_signal = Signal(
      name=self.signal.name,
      start_bit=self.signal.start_bit,
      msb=self.signal.msb,
      lsb=self.signal.lsb,
      size=self.signal.size,
      is_signed=self.signal.is_signed,
      factor=self.signal.factor,
      offset=self.signal.offset,
      is_little_endian=self.signal.is_little_endian,
      type=self.signal.type
    )
    dbc_manager().addSignal(self.id, updated_signal)


class RemoveSigCommand(QUndoCommand):
  """Command to remove a signal from a message."""

  def __init__(self, msg_id: MessageId, sig: Signal, parent: Optional[QUndoCommand] = None):
    super().__init__(parent)
    self.id = msg_id
    self.sigs: list[Signal] = []

    # Store the signal to be removed
    self.sigs.append(sig)

    # If removing a multiplexor, also store all multiplexed signals
    if sig.type == SignalType.Multiplexor:
      msg = dbc_manager().msg(msg_id)
      if msg and hasattr(msg, 'sigs'):
        for s in msg.sigs.values():
          if s.type == SignalType.Multiplexed:
            self.sigs.append(s)

    msg_name = dbc_manager().msgName(msg_id)
    self.setText(QObject.tr(f"remove signal {sig.name} from {msg_name}:{msg_id.address}"))

  def undo(self) -> None:
    for sig in self.sigs:
      dbc_manager().addSignal(self.id, sig)

  def redo(self) -> None:
    for sig in self.sigs:
      dbc_manager().removeSignal(self.id, sig.name)


class EditSignalCommand(QUndoCommand):
  """Command to edit a signal."""

  def __init__(
    self,
    msg_id: MessageId,
    sig: Signal,
    new_sig: Signal,
    parent: Optional[QUndoCommand] = None
  ):
    super().__init__(parent)
    self.id = msg_id
    # Store pairs of (old_sig, new_sig)
    self.sigs: list[tuple[Signal, Signal]] = []

    self.sigs.append((sig, new_sig))

    # If converting multiplexor to normal, convert all multiplexed signals to normal
    if sig.type == SignalType.Multiplexor and new_sig.type == SignalType.Normal:
      msg = dbc_manager().msg(msg_id)
      if msg and hasattr(msg, 'sigs'):
        for s in msg.sigs.values():
          if s.type == SignalType.Multiplexed:
            # Create a new signal with Normal type
            new_s = Signal(
              name=s.name,
              start_bit=s.start_bit,
              msb=s.msb,
              lsb=s.lsb,
              size=s.size,
              is_signed=s.is_signed,
              factor=s.factor,
              offset=s.offset,
              is_little_endian=s.is_little_endian,
              type=SignalType.Normal
            )
            self.sigs.append((s, new_s))

    msg_name = dbc_manager().msgName(msg_id)
    self.setText(QObject.tr(f"edit signal {sig.name} in {msg_name}:{msg_id.address}"))

  def undo(self) -> None:
    for old_sig, new_sig in self.sigs:
      dbc_manager().updateSignal(self.id, new_sig.name, old_sig)

  def redo(self) -> None:
    for old_sig, new_sig in self.sigs:
      dbc_manager().updateSignal(self.id, old_sig.name, new_sig)


class UndoStack:
  """Singleton undo stack for pycabana."""

  _instance: Optional[QUndoStack] = None

  @classmethod
  def instance(cls) -> QUndoStack:
    """Get the global undo stack instance."""
    if cls._instance is None:
      app = QCoreApplication.instance()
      cls._instance = QUndoStack(app)
    return cls._instance

  @classmethod
  def push(cls, cmd: QUndoCommand) -> None:
    """Push a command onto the undo stack."""
    cls.instance().push(cmd)
