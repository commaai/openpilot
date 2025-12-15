"""Core data structures for pycabana."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MessageId:
  """Unique identifier for a CAN message (bus + address)."""

  source: int = 0  # bus number
  address: int = 0  # CAN address

  def __hash__(self):
    return hash((self.source, self.address))

  def __str__(self):
    return f"{self.source}:{self.address:X}"

  def __lt__(self, other):
    return (self.source, self.address) < (other.source, other.address)


@dataclass
class CanEvent:
  """A single CAN message event."""

  src: int  # bus number
  address: int  # CAN address
  mono_time: int  # monotonic time in nanoseconds
  dat: bytes  # message data (up to 64 bytes for CAN FD)

  @property
  def size(self) -> int:
    return len(self.dat)


@dataclass
class CanData:
  """Processed message data for display, updated incrementally."""

  ts: float = 0.0  # last timestamp in seconds
  count: int = 0  # total message count
  freq: float = 0.0  # messages per second (rolling average)
  dat: bytes = b''  # last data bytes

  # For frequency calculation
  _freq_ts: float = field(default=0.0, repr=False)
  _freq_count: int = field(default=0, repr=False)

  def update(self, event: CanEvent, start_ts: int) -> None:
    """Update with a new event."""
    self.count += 1
    self.dat = event.dat
    self.ts = (event.mono_time - start_ts) / 1e9  # convert to seconds

    # Update frequency every second
    if self.ts - self._freq_ts >= 1.0:
      if self._freq_ts > 0:
        self.freq = (self.count - self._freq_count) / (self.ts - self._freq_ts)
      self._freq_ts = self.ts
      self._freq_count = self.count
