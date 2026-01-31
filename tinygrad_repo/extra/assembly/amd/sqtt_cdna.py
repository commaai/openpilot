"""SQTT (SQ Thread Trace) packet decoder for CDNA/MI300 GPUs.

CDNA uses a completely different 16-bit header format from RDNA's nibble-based encoding.
"""
from __future__ import annotations
from typing import Iterator
from extra.assembly.amd.dsl import bits
from extra.assembly.amd.sqtt import PacketType

# CDNA pkt_fmt -> size in bytes (extracted from rocprof hash table)
CDNA_PKT_SIZES = {0: 2, 1: 8, 2: 8, 3: 4, 4: 2, 5: 6, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 8, 12: 6, 13: 4, 14: 8, 15: 6}

class CDNA_DELTA(PacketType):
  """pkt_fmt=0: 16-bit timestamp delta packet"""
  encoding = bits[3:0] == 0
  delta = bits[11:4]      # (data >> 4) & 0xff
  unk_0 = bits[12:12]     # (data >> 0xc) & 1
  unk_1 = bits[15:13]     # (data >> 0xd)

class CDNA_TIMESTAMP(PacketType):
  """pkt_fmt=1: 64-bit timestamp packet (case 0x0)"""
  encoding = bits[3:0] == 1
  unk_0 = bits[15:4]
  timestamp = bits[63:16]   # stored as (data_word >> 0x10) in low 46 bits of local_58

class CDNA_PKT_2(PacketType):
  """pkt_fmt=2: 64-bit packet (case 0x4)"""
  encoding = bits[3:0] == 2
  unk_0 = bits[6:5]       # (data >> 5) & 3
  unk_1 = bits[7:7]       # (data >> 7) + 1 & 1
  unk_padding = bits[63:8]

class CDNA_WAVESTART(PacketType):
  """pkt_fmt=3: 32-bit WAVESTART packet (case 0x8)"""
  encoding = bits[3:0] == 3
  unk_0 = bits[5:5]       # (data >> 5) & 1
  unk_1 = bits[9:6]       # (data >> 6) & 0xf
  wave = bits[13:10]      # (data >> 10) & 0xf
  simd = bits[15:14]      # (data >> 0xe) & 3
  cu = bits[17:16]        # (data >> 0x10) & 3
  unk_5 = bits[19:18]     # (data >> 0x12) & 3
  unk_6 = bits[28:22]     # (data >> 0x16) & 0x7f
  unk_padding = bits[31:29]

class CDNA_PKT_4(PacketType):
  """pkt_fmt=4: 16-bit packet (case 0xc, same as 0x8/0x14)"""
  encoding = bits[3:0] == 4
  unk_0 = bits[5:5]       # (data_word >> 5) & 1
  unk_1 = bits[9:6]       # (data_word >> 6) & 0xf
  unk_2 = bits[13:10]     # (data_word >> 10) & 0xf
  unk_3 = bits[15:14]     # (data_word >> 0xe)

class CDNA_PKT_5(PacketType):
  """pkt_fmt=5: 48-bit packet (case 0x10)"""
  encoding = bits[3:0] == 5
  unk_0 = bits[6:5]       # (data >> 5) & 3
  unk_1 = bits[7:7]       # (data >> 7) + 1 & 1
  unk_2 = bits[15:9]      # (data >> 9) & 0x7f
  unk_padding = bits[47:16]

class CDNA_WAVEEND(PacketType):
  """pkt_fmt=6: 16-bit WAVEEND packet (case 0x14, same as 0x8/0xc)"""
  encoding = bits[3:0] == 6
  unk_0 = bits[5:5]       # (data_word >> 5) & 1
  unk_1 = bits[9:6]       # (data_word >> 6) & 0xf
  wave = bits[13:10]      # (data_word >> 10) & 0xf
  simd = bits[15:14]      # (data_word >> 0xe)

class CDNA_EXEC(PacketType):
  """pkt_fmt=10: 16-bit EXEC packet (case 0x24)"""
  encoding = bits[3:0] == 10
  unk_0 = bits[8:5]       # (data_word >> 5) & 0xf
  unk_1 = bits[10:9]      # (data_word >> 9) & 3
  unk_2 = bits[15:11]     # (data_word >> 0xb)

class CDNA_PKT_11(PacketType):
  """pkt_fmt=11: 64-bit packet (case 0x28)"""
  encoding = bits[3:0] == 11
  unk_0 = bits[8:5]       # (data_word >> 5) & 0xf
  unk_1 = bits[10:9]      # (data_word >> 9) & 3
  unk_2 = bits[15:15]     # (data_word >> 0xf) & 1
  unk_padding = bits[63:16]

class CDNA_INST(PacketType):
  """pkt_fmt=13: 32-bit INST packet (case 0x30)"""
  encoding = bits[3:0] == 13
  unk_0 = bits[6:5]       # (data >> 5) & 3
  unk_1 = bits[9:8]       # (data >> 8) & 3
  unk_2 = bits[11:10]     # (data >> 10) & 3
  unk_3 = bits[13:12]     # (data >> 0xc) & 3
  unk_4 = bits[15:14]     # (data >> 0xe) & 3
  unk_5 = bits[19:18]     # (data >> 0x12) & 3
  unk_6 = bits[21:20]     # (data >> 0x14) & 3
  unk_7 = bits[23:22]     # (data >> 0x16) & 3
  unk_8 = bits[25:24]     # (data >> 0x18) & 3
  unk_9 = bits[27:26]     # (data >> 0x1a) & 3
  unk_padding = bits[31:28]

class CDNA_PKT_14(PacketType):
  """pkt_fmt=14: 64-bit packet (case 0x34)"""
  encoding = bits[3:0] == 14
  unk_0 = bits[5:5]       # (data >> 5) & 1
  unk_1 = bits[9:6]       # (data >> 6) & 0xf
  unk_2 = bits[11:10]     # (data >> 10) & 3
  unk_3 = bits[24:12]     # (data >> 0xc) & 0x1fff
  unk_4 = bits[37:25]     # (data >> 0x19) & 0x1fff
  unk_5 = bits[50:38]     # (data >> 0x26) & 0x1fff
  unk_6 = bits[51:51]     # (data >> 0x33) & 1
  unk_padding = bits[63:52]

class CDNA_PKT_15(PacketType):
  """pkt_fmt=15: 48-bit packet (case 0x38, same as 0x10)"""
  encoding = bits[3:0] == 15
  unk_0 = bits[6:5]       # (data >> 5) & 3
  unk_1 = bits[7:7]       # (data >> 7) + 1 & 1
  unk_2 = bits[15:9]      # (data >> 9) & 0x7f
  unk_padding = bits[47:16]

CDNA_PKT_TYPES: dict[int, type[PacketType]] = {
  0: CDNA_DELTA, 1: CDNA_TIMESTAMP, 2: CDNA_PKT_2, 3: CDNA_WAVESTART, 4: CDNA_PKT_4,
  5: CDNA_PKT_5, 6: CDNA_WAVEEND, 10: CDNA_EXEC, 11: CDNA_PKT_11, 13: CDNA_INST, 14: CDNA_PKT_14, 15: CDNA_PKT_15,
}

# Validate CDNA packet definitions
for pkt_fmt, pkt_cls in CDNA_PKT_TYPES.items():
  assert pkt_cls.encoding.default == pkt_fmt, f"{pkt_cls.__name__} encoding {pkt_cls.encoding.default} != pkt_fmt {pkt_fmt}"
  assert CDNA_PKT_SIZES[pkt_fmt] * 2 == pkt_cls._size_nibbles, f"{pkt_cls.__name__} size {pkt_cls._size_nibbles//2} != {CDNA_PKT_SIZES[pkt_fmt]}"

def decode(data: bytes) -> Iterator[PacketType]:
  """Decode CDNA SQTT blob using 16-bit header format."""
  pos, time, ts_offset = 0, 0, None
  while pos + 2 <= len(data):
    header = int.from_bytes(data[pos:pos+2], 'little')
    pkt_fmt = header & 0xf
    pkt_size = CDNA_PKT_SIZES[pkt_fmt]
    if pos + pkt_size > len(data): break

    raw = int.from_bytes(data[pos:pos+pkt_size], 'little')
    # pkt_fmt=0 has delta in bits[11:4], accumulate it
    if pkt_fmt == 0: time += ((raw >> 4) & 0xff) * 4
    # pkt_fmt=1 with unk_0=0 is absolute timestamp - use it to anchor time
    if pkt_fmt == 1 and ((raw >> 4) & 0xfff) == 0:
      abs_ts = raw >> 16
      if ts_offset is None: ts_offset = abs_ts - time  # first timestamp: save offset
      else: time = ((abs_ts - ts_offset) & ~3) - 4     # subsequent: compute time, align to 4, subtract 4
    pkt_cls = CDNA_PKT_TYPES[pkt_fmt]
    yield pkt_cls.from_raw(raw, time)
    pos += pkt_size

if __name__ == "__main__":
  import sys, pickle
  if len(sys.argv) < 2:
    print("Usage: python sqtt_cdna.py <pkl_file>")
    sys.exit(1)
  with open(sys.argv[1], "rb") as f:
    data = pickle.load(f)
  sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
  for i, event in enumerate(sqtt_events):
    print(f"\n=== event {i} ===")
    for pkt in decode(event.blob):
      print(f"{pkt._time:8}: {pkt}")
