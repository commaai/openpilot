import re
import os
from dataclasses import dataclass
from collections.abc import Callable

from opendbc import DBC_PATH

# TODO: these should just be passed in along with the DBC file
from opendbc.car.honda.hondacan import honda_checksum
from opendbc.car.toyota.toyotacan import toyota_checksum
from opendbc.car.subaru.subarucan import subaru_checksum
from opendbc.car.chrysler.chryslercan import chrysler_checksum, fca_giorgio_checksum
from opendbc.car.hyundai.hyundaicanfd import hkg_can_fd_checksum
from opendbc.car.volkswagen.mqbcan import volkswagen_mqb_meb_checksum, xor_checksum
from opendbc.car.tesla.teslacan import tesla_checksum
from opendbc.car.body.bodycan import body_checksum
from opendbc.car.psa.psacan import psa_checksum


class SignalType:
  DEFAULT = 0
  COUNTER = 1
  HONDA_CHECKSUM = 2
  TOYOTA_CHECKSUM = 3
  BODY_CHECKSUM = 4
  VOLKSWAGEN_MQB_MEB_CHECKSUM = 5
  XOR_CHECKSUM = 6
  SUBARU_CHECKSUM = 7
  CHRYSLER_CHECKSUM = 8
  HKG_CAN_FD_CHECKSUM = 9
  FCA_GIORGIO_CHECKSUM = 10
  TESLA_CHECKSUM = 11
  PSA_CHECKSUM = 12


@dataclass
class Signal:
  name: str
  start_bit: int
  msb: int
  lsb: int
  size: int
  is_signed: bool
  factor: float
  offset: float
  is_little_endian: bool
  type: int = SignalType.DEFAULT
  calc_checksum: 'Callable[[int, Signal, bytearray], int] | None' = None


@dataclass
class Msg:
  name: str
  address: int
  size: int
  sigs: dict[str, Signal]


@dataclass
class Val:
  name: str
  address: int
  def_val: str
  sigs: dict[str, Signal] | None = None


BO_RE = re.compile(r"^BO_ (\w+) (\w+) *: (\w+) (\w+)")
SG_RE = re.compile(r"^SG_ (\w+) : (\d+)\|(\d+)@(\d)([+-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[[0-9.+\-eE]+\|[0-9.+\-eE]+\] \".*\" .*")
SGM_RE = re.compile(r"^SG_ (\w+) (\w+) *: (\d+)\|(\d+)@(\d)([+-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[[0-9.+\-eE]+\|[0-9.+\-eE]+\] \".*\" .*")
VAL_RE = re.compile(r"^VAL_ (\w+) (\w+) (.*);")
VAL_SPLIT_RE = re.compile(r'["]+')


@dataclass
class DBC:
  name: str
  msgs: dict[int, Msg]
  addr_to_msg: dict[int, Msg]
  name_to_msg: dict[str, Msg]
  vals: list[Val]

  def __init__(self, name: str):
    dbc_path = name
    if not os.path.exists(dbc_path):
      dbc_path = os.path.join(DBC_PATH, name + ".dbc")

    self._parse(dbc_path)

  def _parse(self, path: str):
    self.name = os.path.basename(path).replace(".dbc", "")
    with open(path) as f:
      lines = f.readlines()

    checksum_state = get_checksum_state(self.name)
    be_bits = [j + i * 8 for i in range(64) for j in range(7, -1, -1)]
    self.msgs: dict[int, Msg] = {}
    self.addr_to_msg: dict[int, Msg] = {}
    self.name_to_msg: dict[str, Msg] = {}
    self.vals: list[Val] = []
    address = 0
    signals_temp: dict[int, dict[str, Signal]] = {}
    for line_num, line in enumerate(lines, 1):
      line = line.strip()
      if line.startswith("BO_ "):
        m = BO_RE.match(line)
        if not m:
          continue
        address = int(m.group(1), 0)
        msg_name = m.group(2)
        size = int(m.group(3), 0)
        sigs = {}
        self.msgs[address] = Msg(msg_name, address, size, sigs)
        self.addr_to_msg[address] = self.msgs[address]
        self.name_to_msg[msg_name] = self.msgs[address]
        signals_temp[address] = sigs
      elif line.startswith("SG_ "):
        m = SG_RE.search(line)
        offset = 0
        if not m:
          m = SGM_RE.search(line)
          if not m:
            continue
          offset = 1
        sig_name = m.group(1)
        start_bit = int(m.group(2 + offset))
        size = int(m.group(3 + offset))
        is_little_endian = m.group(4 + offset) == "1"
        is_signed = m.group(5 + offset) == "-"
        factor = float(m.group(6 + offset))
        offset_val = float(m.group(7 + offset))

        if is_little_endian:
          lsb = start_bit
          msb = start_bit + size - 1
        else:
          idx = be_bits.index(start_bit)
          lsb = be_bits[idx + size - 1]
          msb = start_bit

        sig = Signal(sig_name, start_bit, msb, lsb, size, is_signed, factor, offset_val, is_little_endian)
        set_signal_type(sig, checksum_state, self.name, line_num)
        signals_temp[address][sig_name] = sig
      elif line.startswith("VAL_ "):
        m = VAL_RE.search(line)
        if not m:
          continue
        val_addr = int(m.group(1), 0)
        sgname = m.group(2)
        defs = m.group(3)
        words = [w.strip() for w in VAL_SPLIT_RE.split(defs) if w.strip()]
        words = [w.upper().replace(" ", "_") for w in words]
        val_def = " ".join(words).strip()
        self.vals.append(Val(sgname, val_addr, val_def))
    for addr, sigs in signals_temp.items():
      self.msgs[addr].sigs = sigs


# ***** checksum functions *****

def tesla_setup_signal(sig: Signal, dbc_name: str, line_num: int) -> None:
  if sig.name.endswith("Counter"):
    sig.type = SignalType.COUNTER
  elif sig.name.endswith("Checksum"):
    sig.type = SignalType.TESLA_CHECKSUM
    sig.calc_checksum = tesla_checksum


@dataclass
class ChecksumState:
  checksum_size: int
  counter_size: int
  checksum_start_bit: int
  counter_start_bit: int
  little_endian: bool
  checksum_type: int
  calc_checksum: Callable[[int, Signal, bytearray], int] | None
  setup_signal: Callable[[Signal, str, int], None] | None = None


def get_checksum_state(dbc_name: str) -> ChecksumState | None:
  if dbc_name.startswith(("honda_", "acura_")):
    return ChecksumState(4, 2, 3, 5, False, SignalType.HONDA_CHECKSUM, honda_checksum)
  elif dbc_name.startswith(("toyota_", "lexus_")):
    return ChecksumState(8, -1, 7, -1, False, SignalType.TOYOTA_CHECKSUM, toyota_checksum)
  elif dbc_name.startswith("hyundai_canfd_generated"):
    return ChecksumState(16, -1, 0, -1, True, SignalType.HKG_CAN_FD_CHECKSUM, hkg_can_fd_checksum)
  elif dbc_name.startswith(("vw_mqb", "vw_mqbevo", "vw_meb")):
    return ChecksumState(8, 4, 0, 0, True, SignalType.VOLKSWAGEN_MQB_MEB_CHECKSUM, volkswagen_mqb_meb_checksum)
  elif dbc_name.startswith("vw_pq"):
    return ChecksumState(8, 4, 0, -1, True, SignalType.XOR_CHECKSUM, xor_checksum)
  elif dbc_name.startswith("subaru_global_"):
    return ChecksumState(8, -1, 0, -1, True, SignalType.SUBARU_CHECKSUM, subaru_checksum)
  elif dbc_name.startswith("chrysler_"):
    return ChecksumState(8, -1, 7, -1, False, SignalType.CHRYSLER_CHECKSUM, chrysler_checksum)
  elif dbc_name.startswith("fca_giorgio"):
    return ChecksumState(8, -1, 7, -1, False, SignalType.FCA_GIORGIO_CHECKSUM, fca_giorgio_checksum)
  elif dbc_name.startswith("comma_body"):
    return ChecksumState(8, 4, 7, 3, False, SignalType.BODY_CHECKSUM, body_checksum)
  elif dbc_name.startswith("tesla_model3_party"):
    return ChecksumState(8, -1, 0, -1, True, SignalType.TESLA_CHECKSUM, tesla_checksum, tesla_setup_signal)
  elif dbc_name.startswith("psa_"):
    return ChecksumState(4, 4, 7, 3, False, SignalType.PSA_CHECKSUM, psa_checksum)
  return None


def set_signal_type(sig: Signal, chk: ChecksumState | None, dbc_name: str, line_num: int) -> None:
  sig.calc_checksum = None
  if chk:
    if chk.setup_signal:
      chk.setup_signal(sig, dbc_name, line_num)
    if sig.name == "CHECKSUM":
      sig.type = chk.checksum_type
      sig.calc_checksum = chk.calc_checksum
    elif sig.name == "COUNTER":
      sig.type = SignalType.COUNTER
