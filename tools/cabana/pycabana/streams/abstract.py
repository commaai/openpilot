"""Abstract base class for CAN data streams."""

from PySide6.QtCore import QObject, Signal

from openpilot.tools.cabana.pycabana.dbc.dbc import MessageId, CanEvent, CanData


class AbstractStream(QObject):
  """Base class for all CAN data streams.

  Subclasses must implement start() and populate events via updateEvent().
  """

  # Emitted when messages are received/updated
  # Args: (msg_ids: set[MessageId], has_new_ids: bool)
  msgsReceived = Signal(set, bool)

  # Emitted when stream seeks to a new time (for replay)
  # Args: (time_seconds: float)
  seekedTo = Signal(float)

  # Emitted when stream starts
  streamStarted = Signal()

  def __init__(self, parent=None):
    super().__init__(parent)
    self.events: dict[MessageId, list[CanEvent]] = {}
    self.last_msgs: dict[MessageId, CanData] = {}
    self.start_ts: int = 0  # first event timestamp (nanoseconds)
    self._msg_ids: set[MessageId] = set()

  def start(self) -> None:
    """Start the stream. Subclasses should override."""
    self.streamStarted.emit()

  def stop(self) -> None:
    """Stop the stream. Subclasses should override."""
    pass

  def lastMessage(self, msg_id: MessageId) -> CanData | None:
    """Get the last processed data for a message."""
    return self.last_msgs.get(msg_id)

  def allEvents(self) -> list[CanEvent]:
    """Get all events across all messages, sorted by time."""
    all_evts = []
    for evts in self.events.values():
      all_evts.extend(evts)
    all_evts.sort(key=lambda e: e.mono_time)
    return all_evts

  def updateEvent(self, event: CanEvent) -> None:
    """Process a single CAN event, update internal state."""
    msg_id = MessageId(event.src, event.address)

    # Track first timestamp
    if self.start_ts == 0:
      self.start_ts = event.mono_time

    # Add to events list
    if msg_id not in self.events:
      self.events[msg_id] = []
    self.events[msg_id].append(event)

    # Update last_msgs
    if msg_id not in self.last_msgs:
      self.last_msgs[msg_id] = CanData()
    self.last_msgs[msg_id].update(event, self.start_ts)

    # Track new message IDs
    self._msg_ids.add(msg_id)

  def emitMsgsReceived(self, has_new: bool = True) -> None:
    """Emit the msgsReceived signal with current message IDs."""
    self.msgsReceived.emit(self._msg_ids.copy(), has_new)

  def toSeconds(self, mono_time: int) -> float:
    """Convert monotonic time to seconds from start."""
    return (mono_time - self.start_ts) / 1e9

  @property
  def routeName(self) -> str:
    """Return route name if applicable. Subclasses can override."""
    return ""

  @property
  def carFingerprint(self) -> str:
    """Return car fingerprint if known. Subclasses can override."""
    return ""
