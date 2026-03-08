#!/usr/bin/env python3
from __future__ import annotations
import enum, collections
from typing import Iterator
from tinygrad.helpers import colored
from extra.assembly.amd.sqtt import PacketType, bits

# ═══════════════════════════════════════════════════════════════════════════════
# STALL REASONS
# ═══════════════════════════════════════════════════════════════════════════════

class StallReason(enum.IntEnum):
  # Based on CUpti_ActivityPCSamplingStallReason
  INVALID = 0
  NONE = 1              # selected, selected_not_issued
  INST_FETCH = 2        # branch_resolving, no_instructions
  EXEC_DEPENDENCY = 3   # short_scoreboard, wait
  MEMORY_DEPENDENCY = 4 # long_scoreboard
  TEXTURE = 5           # tex_throttle
  SYNC = 6              # barrier, membar
  CONSTANT_MEMORY = 7   # imc_miss
  PIPE_BUSY = 8         # mio_throttle, math_pipe_throttle
  MEMORY_THROTTLE = 9   # drain, lg_throttle
  NOT_SELECTED = 10     # not_selected
  OTHER = 11            # misc, dispatch_stall
  SLEEPING = 12         # sleeping

STALL_KEY_MAP_AMPERE: dict[int, StallReason] = {
  1: StallReason.MEMORY_THROTTLE, 15: StallReason.MEMORY_THROTTLE,
  2: StallReason.CONSTANT_MEMORY,
  3: StallReason.SYNC,
  6: StallReason.INST_FETCH, 11: StallReason.INST_FETCH,
  7: StallReason.EXEC_DEPENDENCY, 10: StallReason.EXEC_DEPENDENCY,
  9: StallReason.MEMORY_DEPENDENCY,
  12: StallReason.PIPE_BUSY,
  17: StallReason.OTHER, 20: StallReason.OTHER,
  18: StallReason.NONE,
}

STALL_KEY_MAP_BLACKWELL: dict[int, StallReason] = {
  0x01: StallReason.MEMORY_THROTTLE, 0x0e: StallReason.MEMORY_THROTTLE,
  0x02: StallReason.SYNC,
  0x05: StallReason.INST_FETCH, 0x0a: StallReason.INST_FETCH,
  0x06: StallReason.EXEC_DEPENDENCY, 0x09: StallReason.EXEC_DEPENDENCY,
  0x08: StallReason.MEMORY_DEPENDENCY,
  0x0b: StallReason.PIPE_BUSY, 0x0f: StallReason.PIPE_BUSY,
  0x10: StallReason.OTHER, 0x13: StallReason.OTHER,
  0x11: StallReason.NONE,
}

# Lookup table for extracting sample bytes from 32-byte packet (bytes 0-3, 8-31, skipping header at 4-7)
LOOKUP_28B = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

# ═══════════════════════════════════════════════════════════════════════════════
# PACKET HEADER
# ═══════════════════════════════════════════════════════════════════════════════

class PMAHeader(PacketType):
  num_bytes  = bits[4:0]    # number of sample bytes in this packet
  tpc_id_lo  = bits[15:8]   # TPC identifier low 8 bits
  tpc_id_hi  = bits[27:25]  # TPC identifier high 3 bits
  dropped    = bits[28:28]  # dropped flag (resets byte accumulator)
  @property
  def tpc_id(self) -> int: return self.tpc_id_lo | (self.tpc_id_hi << 8)

# ═══════════════════════════════════════════════════════════════════════════════
# 8-BYTE SAMPLE FORMAT (Ampere/Ada/Hopper)
# ═══════════════════════════════════════════════════════════════════════════════

class PMASampleAmpere8B(PacketType):
  pc_raw     = bits[44:0]   # raw PC value (pc_offset = pc_raw << 4)
  stall_key  = bits[49:45]  # stall reason key
  wave_id    = bits[55:50]  # warp/wave identifier
  active     = bits[62:62]  # 1 if warp was executing, 0 if scheduled but not issued
  @property
  def pc_offset(self) -> int: return self.pc_raw << 4
  @property
  def stall_reason(self) -> StallReason: return STALL_KEY_MAP_AMPERE.get(self.stall_key, StallReason.OTHER)

# ═══════════════════════════════════════════════════════════════════════════════
# 9-BYTE SAMPLE FORMAT (Blackwell+)
# ═══════════════════════════════════════════════════════════════════════════════

class PMASampleBlackwell9B(PacketType):
  stall_key  = bits[5:0]    # stall reason key
  pc_raw     = bits[60:8]   # raw PC value (pc_offset = pc_raw << 4)
  wave_hi    = bits[7:6]    # wave_id high 2 bits
  wave_lo    = bits[71:68]  # wave_id low 4 bits
  active     = bits[67:67]  # 1 if warp was executing, 0 if scheduled but not issued
  @property
  def pc_offset(self) -> int: return self.pc_raw << 4
  @property
  def stall_reason(self) -> StallReason: return STALL_KEY_MAP_BLACKWELL.get(self.stall_key, StallReason.OTHER)
  @property
  def wave_id(self) -> int: return (self.wave_hi << 4) | self.wave_lo

PMASample = PMASampleAmpere8B|PMASampleBlackwell9B

def decode(data: bytes, sm_version: int = 0x800) -> Iterator[tuple[PMASample, int]]:
  use_9byte = sm_version >= 0xa04
  record_size = 9 if use_9byte else 8
  sample_cls = PMASampleBlackwell9B if use_9byte else PMASampleAmpere8B

  tpc_state: dict[int, list[int]] = collections.defaultdict(list)
  for pkt_idx in range(len(data) // 32):
    pkt = data[pkt_idx * 32:(pkt_idx + 1) * 32]
    hdr = PMAHeader.from_raw(int.from_bytes(pkt[4:8], 'little'))

    if hdr.dropped: tpc_state[hdr.tpc_id].clear()

    for i in range(hdr.num_bytes):
      tpc_state[hdr.tpc_id].append(pkt[LOOKUP_28B[i]])

    while len(tpc_state[hdr.tpc_id]) >= record_size:
      yield sample_cls.from_raw(int.from_bytes(bytes(tpc_state[hdr.tpc_id][:record_size]), 'little')), hdr.tpc_id
      del tpc_state[hdr.tpc_id][:record_size]

# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

STALL_COLORS = {
  StallReason.NONE: "green", StallReason.INST_FETCH: "yellow", StallReason.EXEC_DEPENDENCY: "cyan",
  StallReason.MEMORY_DEPENDENCY: "red", StallReason.SYNC: "magenta", StallReason.CONSTANT_MEMORY: "blue",
  StallReason.PIPE_BUSY: "yellow", StallReason.MEMORY_THROTTLE: "RED", StallReason.OTHER: "white",
}

def decode_tpc_id(tpc_id:int) -> tuple[int, int, int]:
  # NOTE: valid only for ops_nv, cuda encoding is different
  return (tpc_id >> 5, (tpc_id >> 1) & 0xf, tpc_id & 1)

def print_samples(samples:list[tuple[PMASample, int]]) -> None:
  if not samples: return
  base_pc = min(s.pc_offset for s, _ in samples)
  for s, tpc_id in samples:
    gpc, tpc, sm = decode_tpc_id(tpc_id)
    stall_str = colored(f"{s.stall_reason.name:17}", STALL_COLORS.get(s.stall_reason, "white"))
    print(f"pc=0x{s.pc_offset - base_pc:06x} {stall_str} ev={s.stall_key:2d} active={s.active} wave={s.wave_id:2d} gpc={gpc} tpc={tpc} sm={sm}")

def print_packets(data:bytes, sm_version:int=0x800) -> None:
  record_size = 9 if sm_version >= 0x890 else 8
  tpc_state: dict[int, list[int]] = collections.defaultdict(list)
  for i in range(len(data) // 32):
    pkt = data[i * 32:(i + 1) * 32]
    hdr = PMAHeader.from_raw(int.from_bytes(pkt[4:8], 'little'))
    if hdr.dropped: tpc_state[hdr.tpc_id].clear()
    for j in range(hdr.num_bytes): tpc_state[hdr.tpc_id].append(pkt[LOOKUP_28B[j]])
    # Show complete records extracted from this packet
    records = []
    while len(tpc_state[hdr.tpc_id]) >= record_size:
      records.append(bytes(tpc_state[hdr.tpc_id][:record_size]).hex())
      del tpc_state[hdr.tpc_id][:record_size]
    leftover = len(tpc_state[hdr.tpc_id])
    print(f"Pkt {i:3d}: tpc={hdr.tpc_id:4d} n={hdr.num_bytes:2d} drop={hdr.dropped} left={leftover} | {' '.join(records)}")

def print_aggregated(samples:list[tuple[PMASample, int]]) -> None:
  if not samples: return
  base_pc = min(s.pc_offset for s, _ in samples)
  counter: collections.Counter[tuple[int, StallReason]] = collections.Counter((s.pc_offset, s.stall_reason) for s, _ in samples)
  print(f"\nAggregated samples (base_pc=0x{base_pc:x}):")
  for (pc, reason), cnt in sorted(counter.items()):
    stall_str = colored(f"{reason.name:17}", STALL_COLORS.get(reason, "white"))
    print(f"  pc=0x{pc - base_pc:06x} {stall_str} samples={cnt:4d}")

if __name__ == "__main__":
  import sys, pickle

  if len(sys.argv) < 2:
    print("Usage: python decode.py <pkl_file> [--raw] [--sm=0xNNN]")
    sys.exit(1)

  with open(sys.argv[1], "rb") as f:
    data = pickle.load(f)

  if isinstance(data, dict):
    sm_version = 0x800  # default to Ampere
    for arg in sys.argv:
      if arg.startswith("--sm="): sm_version = int(arg[5:], 0)
    dumps = [(i, x, sm_version) for i, x in enumerate(data["pma_raw_dumps"])]
  else:
    devs = {e.device: e for e in data if type(e).__name__ == "ProfileDeviceEvent"}
    dumps = []
    for i, e in enumerate(e for e in data if type(e).__name__ == "ProfilePMAEvent"):
      dumps.append((i, e.blob, devs[e.device].props.get('sm_version', 0x800)))

  for dump_idx, raw, sm_ver in dumps:
    print(f"\n{'='*60}\nDump {dump_idx} ({len(raw)} bytes, {len(raw)//32} packets)\n{'='*60}")
    if "--raw" in sys.argv: print_packets(raw, sm_ver)
    else:
      samples = list(decode(raw, sm_ver))
      print(f"\nDecoded {len(samples)} samples:")
      print_samples(samples)
      print_aggregated(samples)
