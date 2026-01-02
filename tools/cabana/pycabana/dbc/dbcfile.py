"""DBC file parser and generator for pycabana."""

import re
from pathlib import Path
from typing import Optional
from enum import Enum

from PySide6.QtCore import QObject

from openpilot.tools.cabana.pycabana.dbc.dbc import MessageId

DEFAULT_NODE_NAME = "XXX"


def double_to_string(value: float) -> str:
  """Convert double to string with maximum precision."""
  # Python's default g format with 15 digits (equivalent to double precision)
  result = f"{value:.15g}"
  return result


def num_decimals(num: float) -> int:
  """Count the number of decimal places in a number."""
  string = str(num)
  dot_pos = string.find('.')
  return 0 if dot_pos == -1 else len(string) - dot_pos - 1


def flip_bit_pos(start_bit: int) -> int:
  """Flip bit position for big-endian signals."""
  return 8 * (start_bit // 8) + 7 - start_bit % 8


def update_msb_lsb(sig: 'Signal') -> None:
  """Update MSB and LSB based on endianness."""
  if sig.is_little_endian:
    sig.lsb = sig.start_bit
    sig.msb = sig.start_bit + sig.size - 1
  else:
    sig.lsb = flip_bit_pos(flip_bit_pos(sig.start_bit) + sig.size - 1)
    sig.msb = sig.start_bit


class SignalType(Enum):
  """Signal type enumeration."""
  Normal = 0
  Multiplexed = 1
  Multiplexor = 2


class Signal:
  """Signal definition in a CAN message."""

  def __init__(self) -> None:
    self.type: SignalType = SignalType.Normal
    self.name: str = ""
    self.start_bit: int = 0
    self.msb: int = 0
    self.lsb: int = 0
    self.size: int = 0
    self.factor: float = 1.0
    self.offset: float = 0.0
    self.is_signed: bool = False
    self.is_little_endian: bool = True
    self.min: float = 0.0
    self.max: float = 0.0
    self.unit: str = ""
    self.comment: str = ""
    self.receiver_name: str = ""
    self.val_desc: list[tuple[float, str]] = []
    self.precision: int = 0
    self.multiplex_value: int = 0
    self.multiplexor: Optional[Signal] = None

  def update(self) -> None:
    """Update signal properties after changes."""
    update_msb_lsb(self)
    if not self.receiver_name:
      self.receiver_name = DEFAULT_NODE_NAME

    # Calculate precision
    self.precision = max(num_decimals(self.factor), num_decimals(self.offset))

  def __eq__(self, other: object) -> bool:
    """Check if two signals are equal."""
    if not isinstance(other, Signal):
      return False
    return (self.name == other.name and self.size == other.size and
            self.start_bit == other.start_bit and
            self.msb == other.msb and self.lsb == other.lsb and
            self.is_signed == other.is_signed and self.is_little_endian == other.is_little_endian and
            self.factor == other.factor and self.offset == other.offset and
            self.min == other.min and self.max == other.max and
            self.comment == other.comment and self.unit == other.unit and
            self.val_desc == other.val_desc and
            self.multiplex_value == other.multiplex_value and self.type == other.type and
            self.receiver_name == other.receiver_name)


class Msg:
  """Message definition in a DBC file."""

  def __init__(self) -> None:
    self.address: int = 0
    self.name: str = ""
    self.size: int = 0
    self.comment: str = ""
    self.transmitter: str = ""
    self.sigs: list[Signal] = []
    self.mask: list[int] = []
    self.multiplexor: Optional[Signal] = None

  def add_signal(self, sig: Signal) -> Signal:
    """Add a signal to the message."""
    self.sigs.append(sig)
    self.update()
    return sig

  def update_signal(self, sig_name: str, new_sig: Signal) -> Optional[Signal]:
    """Update an existing signal."""
    s = self.sig(sig_name)
    if s:
      idx = self.sigs.index(s)
      self.sigs[idx] = new_sig
      self.update()
      return new_sig
    return None

  def remove_signal(self, sig_name: str) -> None:
    """Remove a signal from the message."""
    self.sigs = [s for s in self.sigs if s.name != sig_name]
    self.update()

  def sig(self, sig_name: str) -> Optional[Signal]:
    """Get signal by name."""
    for s in self.sigs:
      if s.name == sig_name:
        return s
    return None

  def index_of(self, sig: Signal) -> int:
    """Get index of signal."""
    try:
      return self.sigs.index(sig)
    except ValueError:
      return -1

  def new_signal_name(self) -> str:
    """Generate a new unique signal name."""
    i = 1
    while True:
      new_name = f"NEW_SIGNAL_{i}"
      if self.sig(new_name) is None:
        return new_name
      i += 1

  def update(self) -> None:
    """Update message properties after changes."""
    if not self.transmitter:
      self.transmitter = DEFAULT_NODE_NAME

    self.mask = [0x00] * self.size
    self.multiplexor = None

    # Sort signals
    def sort_key(s: Signal) -> tuple:
      return (s.type != SignalType.Multiplexor, s.multiplex_value, s.start_bit, s.name)

    self.sigs.sort(key=sort_key)

    # Update each signal
    for sig in self.sigs:
      if sig.type == SignalType.Multiplexor:
        self.multiplexor = sig
      sig.update()

      # Update mask
      i = sig.msb // 8
      bits = sig.size
      while 0 <= i < self.size and bits > 0:
        lsb = sig.lsb if (sig.lsb // 8) == i else i * 8
        msb = sig.msb if (sig.msb // 8) == i else (i + 1) * 8 - 1

        sz = msb - lsb + 1
        shift = lsb - (i * 8)

        self.mask[i] |= ((1 << sz) - 1) << shift

        bits -= sz
        i = i - 1 if sig.is_little_endian else i + 1

    # Set multiplexor references
    for sig in self.sigs:
      sig.multiplexor = self.multiplexor if sig.type == SignalType.Multiplexed else None
      if not sig.multiplexor:
        if sig.type == SignalType.Multiplexed:
          sig.type = SignalType.Normal
        sig.multiplex_value = 0

  def get_signals(self) -> list[Signal]:
    """Get all signals in the message."""
    return self.sigs


class DBCFile(QObject):
  """DBC file parser and generator."""

  def __init__(self, name_or_path: str, content: Optional[str] = None) -> None:
    """
    Initialize DBCFile.

    Args:
      name_or_path: Either a file path (if content is None) or a name (if content is provided)
      content: Optional DBC content as string
    """
    super().__init__()

    self.filename: str = ""
    self.name_: str = ""
    self.header: str = ""
    self.msgs: dict[int, Msg] = {}

    if content is not None:
      # Create from name and content
      self.name_ = name_or_path
      self.filename = ""
      self._parse(content)
    else:
      # Load from file
      path = Path(name_or_path)
      if not path.exists():
        raise RuntimeError("Failed to open file.")
      self.name_ = path.stem
      self.filename = str(path)
      with open(path, encoding='utf-8') as f:
        self._parse(f.read())

  def save(self) -> bool:
    """Save to current filename."""
    assert self.filename, "Filename is empty"
    return self._write_contents(self.filename)

  def save_as(self, new_filename: str) -> bool:
    """Save to a new filename."""
    self.filename = new_filename
    return self.save()

  def _write_contents(self, fn: str) -> bool:
    """Write DBC contents to file."""
    try:
      with open(fn, 'w', encoding='utf-8') as f:
        f.write(self.generate_dbc())
      return True
    except Exception:
      return False

  def update_msg(self, msg_id: MessageId, name: str, size: int, node: str, comment: str) -> None:
    """Update or create a message."""
    if msg_id.address not in self.msgs:
      self.msgs[msg_id.address] = Msg()

    m = self.msgs[msg_id.address]
    m.address = msg_id.address
    m.name = name
    m.size = size
    m.transmitter = DEFAULT_NODE_NAME if not node else node
    m.comment = comment

  def remove_msg(self, msg_id: MessageId) -> None:
    """Remove a message."""
    if msg_id.address in self.msgs:
      del self.msgs[msg_id.address]

  def get_messages(self) -> dict[int, Msg]:
    """Get all messages."""
    return self.msgs

  def msg(self, address_or_name: int | str) -> Optional[Msg]:
    """Get message by address or name."""
    if isinstance(address_or_name, int):
      return self.msgs.get(address_or_name)
    else:
      # Search by name
      for m in self.msgs.values():
        if m.name == address_or_name:
          return m
      return None

  def signal(self, address: int, name: str) -> Optional[Signal]:
    """Get signal by address and name."""
    m = self.msg(address)
    return m.sig(name) if m else None

  def name(self) -> str:
    """Get the name of the DBC file."""
    return "untitled" if not self.name_ else self.name_

  def is_empty(self) -> bool:
    """Check if the DBC file is empty."""
    return len(self.msgs) == 0 and not self.name_

  def _parse(self, content: str) -> None:
    """Parse DBC content."""
    self.msgs.clear()

    lines = content.split('\n')
    line_num = 0
    current_msg: Optional[Msg] = None
    multiplexor_cnt = 0
    seen_first = False

    i = 0
    while i < len(lines):
      line_num += 1
      raw_line = lines[i]
      line = raw_line.strip()

      seen = True
      try:
        if line.startswith("BO_ "):
          multiplexor_cnt = 0
          current_msg = self._parse_bo(line)
        elif line.startswith("SG_ "):
          self._parse_sg(line, current_msg, multiplexor_cnt)
        elif line.startswith("VAL_ "):
          self._parse_val(line)
        elif line.startswith("CM_ BO_"):
          i = self._parse_cm_bo(line, lines, i)
        elif line.startswith("CM_ SG_ "):
          i = self._parse_cm_sg(line, lines, i)
        else:
          seen = False
      except Exception as e:
        raise RuntimeError(f"[{self.filename}:{line_num}]{e}: {line}") from e

      if seen:
        seen_first = True
      elif not seen_first:
        self.header += raw_line + "\n"

      i += 1

    # Update all messages
    for m in self.msgs.values():
      m.update()

  def _parse_bo(self, line: str) -> Msg:
    """Parse BO_ line."""
    bo_pattern = r'^BO_ (?P<address>\w+) (?P<name>\w+) *: (?P<size>\w+) (?P<transmitter>\w+)'
    match = re.match(bo_pattern, line)

    if not match:
      raise RuntimeError("Invalid BO_ line format")

    address = int(match.group('address'))
    if address in self.msgs:
      raise RuntimeError(f"Duplicate message address: {address}")

    msg = Msg()
    msg.address = address
    msg.name = match.group('name')
    msg.size = int(match.group('size'))
    msg.transmitter = match.group('transmitter').strip()

    self.msgs[address] = msg
    return msg

  def _parse_cm_bo(self, line: str, lines: list[str], current_idx: int) -> int:
    """Parse CM_ BO_ (message comment) line."""
    parse_line = line

    # Handle multi-line comments
    if not parse_line.endswith('";'):
      # Find the end of the comment
      for j in range(current_idx + 1, len(lines)):
        parse_line += " " + lines[j].strip()
        if '";' in lines[j]:
          current_idx = j
          break

    msg_comment_pattern = r'^CM_ BO_ *(?P<address>\w+) *"(?P<comment>(?:[^"\\]|\\.)*)"\s*;'
    match = re.match(msg_comment_pattern, parse_line)

    if not match:
      raise RuntimeError("Invalid message comment format")

    address = int(match.group('address'))
    if address in self.msgs:
      comment = match.group('comment').strip().replace('\\"', '"')
      self.msgs[address].comment = comment

    return current_idx

  def _parse_sg(self, line: str, current_msg: Optional[Msg], multiplexor_cnt: int) -> None:
    """Parse SG_ (signal) line."""
    if not current_msg:
      raise RuntimeError("No Message")

    # Try normal signal pattern first
    sg_pattern = r'^SG_ (\w+) *: (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] "(.*)" (.*)'
    match = re.match(sg_pattern, line)
    offset = 0

    # Try multiplexed signal pattern
    if not match:
      sgm_pattern = r'^SG_ (\w+) (\w+) *: (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] "(.*)" (.*)'
      match = re.match(sgm_pattern, line)
      offset = 1

    if not match:
      raise RuntimeError("Invalid SG_ line format")

    name = match.group(1)
    if current_msg.sig(name) is not None:
      raise RuntimeError("Duplicate signal name")

    sig = Signal()

    # Handle multiplexing
    if offset == 1:
      indicator = match.group(2)
      if indicator == "M":
        multiplexor_cnt += 1
        if multiplexor_cnt >= 2:
          raise RuntimeError("Multiple multiplexor")
        sig.type = SignalType.Multiplexor
      else:
        sig.type = SignalType.Multiplexed
        sig.multiplex_value = int(indicator[1:])

    sig.name = name
    sig.start_bit = int(match.group(offset + 2))
    sig.size = int(match.group(offset + 3))
    sig.is_little_endian = int(match.group(offset + 4)) == 1
    sig.is_signed = match.group(offset + 5) == "-"
    sig.factor = float(match.group(offset + 6))
    sig.offset = float(match.group(offset + 7))
    sig.min = float(match.group(8 + offset))
    sig.max = float(match.group(9 + offset))
    sig.unit = match.group(10 + offset)
    sig.receiver_name = match.group(11 + offset).strip()

    current_msg.sigs.append(sig)

  def _parse_cm_sg(self, line: str, lines: list[str], current_idx: int) -> int:
    """Parse CM_ SG_ (signal comment) line."""
    parse_line = line

    # Handle multi-line comments
    if not parse_line.endswith('";'):
      # Find the end of the comment
      for j in range(current_idx + 1, len(lines)):
        parse_line += " " + lines[j].strip()
        if '";' in lines[j]:
          current_idx = j
          break

    sg_comment_pattern = r'^CM_ SG_ *(\w+) *(\w+) *"((?:[^"\\]|\\.)*)"\s*;'
    match = re.match(sg_comment_pattern, parse_line)

    if not match:
      raise RuntimeError("Invalid CM_ SG_ line format")

    address = int(match.group(1))
    sig_name = match.group(2)
    s = self.signal(address, sig_name)

    if s:
      comment = match.group(3).strip().replace('\\"', '"')
      s.comment = comment

    return current_idx

  def _parse_val(self, line: str) -> None:
    """Parse VAL_ (value description) line."""
    val_pattern = r'VAL_ (\w+) (\w+) (\s*[-+]?[0-9]+\s+".+?"[^;]*)'
    match = re.match(val_pattern, line)

    if not match:
      raise RuntimeError("invalid VAL_ line format")

    address = int(match.group(1))
    sig_name = match.group(2)
    s = self.signal(address, sig_name)

    if s:
      desc_list = match.group(3).strip().split('"')
      for i in range(0, len(desc_list), 2):
        val_str = desc_list[i].strip()
        if val_str and (i + 1) < len(desc_list):
          desc = desc_list[i + 1].strip()
          s.val_desc.append((float(val_str), desc))

  def generate_dbc(self) -> str:
    """Generate DBC file content."""
    dbc_string = ""
    comment = ""
    val_desc = ""

    for address, m in self.msgs.items():
      transmitter = DEFAULT_NODE_NAME if not m.transmitter else m.transmitter
      dbc_string += f"BO_ {address} {m.name}: {m.size} {transmitter}\n"

      if m.comment:
        escaped_comment = m.comment.replace('"', '\\"')
        comment += f'CM_ BO_ {address} "{escaped_comment}";\n'

      for sig in m.get_signals():
        multiplexer_indicator = ""
        if sig.type == SignalType.Multiplexor:
          multiplexer_indicator = "M "
        elif sig.type == SignalType.Multiplexed:
          multiplexer_indicator = f"m{sig.multiplex_value} "

        endian_char = '1' if sig.is_little_endian else '0'
        sign_char = '-' if sig.is_signed else '+'
        receiver = DEFAULT_NODE_NAME if not sig.receiver_name else sig.receiver_name

        dbc_string += (
          f" SG_ {sig.name} {multiplexer_indicator}: {sig.start_bit}|{sig.size}@{endian_char}{sign_char} "
          + f"({double_to_string(sig.factor)},{double_to_string(sig.offset)}) "
          + f"[{double_to_string(sig.min)}|{double_to_string(sig.max)}] "
          + f'"{sig.unit}" {receiver}\n'
        )

        if sig.comment:
          escaped_comment = sig.comment.replace('"', '\\"')
          comment += f'CM_ SG_ {address} {sig.name} "{escaped_comment}";\n'

        if sig.val_desc:
          text_parts = []
          for val, desc in sig.val_desc:
            text_parts.append(f'{int(val)} "{desc}"')
          val_desc += f"VAL_ {address} {sig.name} {' '.join(text_parts)};\n"

      dbc_string += "\n"

    return self.header + dbc_string + comment + val_desc
