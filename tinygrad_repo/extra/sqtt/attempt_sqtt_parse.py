import pickle, sys
from tinygrad.helpers import getenv, Timing, colored
from extra.sqtt.roc import decode, ProfileSQTTEvent

# do these enums match fields in the packets?
#from tinygrad.runtime.support.amd import import_soc
#soc = import_soc([11])
#perf_sel = {getattr(soc, k):k for k in dir(soc) if k.startswith("SQ_PERF_")}

# Instruction packets (one per ISA op)
# NOTE: these are bad guesses and may be wrong! feel free to update if you know better
# some names were taken from SQ_TT_TOKEN_MASK_TOKEN_EXCLUDE_SHIFT

# we see 18 opcodes
# opcodes(18):  1  2  3  4  5  6  8  9  F 10 11 12 14 15 16 17 18 19
# if you exclude everything, you are left with 6
# opcodes( 6): 10 11 14 15 16 17
# sometimes we see a lot of B, but not repeatable

# not seen
# 7 A C

# NOTE: INST runs before EXEC

OPCODE_COLORS = {
  # dispatches are BLACK
  0x1: "BLACK",
  0x18: "BLACK",

  # execs are yellow
  0x2: "yellow",
  0x3: "yellow",
  0x4: "YELLOW",
  0x5: "YELLOW",

  # waves are blue
  0x8: "blue",
  0x9: "blue",
  0x6: "cyan",
  0xb: "cyan",
}

OPCODE_NAMES = {
  # gated by SQ_TT_TOKEN_EXCLUDE_VALUINST_SHIFT (but others must be enabled for it to show)
  0x01: "VALUINST",
  # gated by SQ_TT_TOKEN_EXCLUDE_VMEMEXEC_SHIFT
  0x02: "VMEMEXEC",
  # gated by SQ_TT_TOKEN_EXCLUDE_ALUEXEC_SHIFT
  0x03: "ALUEXEC",
  # gated by SQ_TT_TOKEN_EXCLUDE_IMMEDIATE_SHIFT
  0x04: "IMMEDIATE",
  0x05: "IMMEDIATE_MASK",

  # gated by SQ_TT_TOKEN_EXCLUDE_WAVERDY_SHIFT
  0x06: "WAVERDY",
  # gated by SQ_TT_TOKEN_EXCLUDE_WAVESTARTEND_SHIFT
  0x08: "WAVEEND",
  0x09: "WAVESTART",
  # gated by SQ_TT_TOKEN_EXCLUDE_WAVEALLOC_SHIFT
  0x0B: "WAVEALLOC",  # FFF00

  # gated by NOT SQ_TT_TOKEN_EXCLUDE_PERF_SHIFT
  0x0D: "PERF",
  # gated by SQ_TT_TOKEN_EXCLUDE_EVENT_SHIFT
  0x12: "EVENT",
  0x13: "EVENT_BIG",  # FFFFF800
  # some gated by SQ_TT_TOKEN_EXCLUDE_REG_SHIFT, some always there. something is broken with the timing on this
  0x14: "REG",
  # gated by SQ_TT_TOKEN_EXCLUDE_INST_SHIFT
  0x18: "INST",
  # gated by SQ_TT_TOKEN_EXCLUDE_UTILCTR_SHIFT
  0x19: "UTILCTR",

  # this is the first (8 byte) packet in the bitstream
  0x17: "LAYOUT_HEADER",       # layout/mode/group + selectors A/B (reversed)

  # pure time (no extra bits)
  0x0F: "TS_DELTA_SHORT",
  0x10: "NOP",
  0x11: "TS_WAVE_STATE",     # almost pure time, has a small flag

  # not a good name, but seen and understood mostly
  0x15: "SNAPSHOT",          # small delta + 50-ish bits of snapshot
  0x16: "TS_DELTA_OR_MARK",  # 36-bit long delta or 36-bit marker

  # packets we haven't seen / rarely see 0x0b
  0x07: "TS_DELTA_S8_W3_7",         # shift=8,  width=3  (small delta)
  0x0A: "TS_DELTA_S5_W2_A",         # shift=5,  width=2
  0x0C: "TS_DELTA_S5_W3_B",         # shift=5,  width=3 (different consumer)
}

#  SALU    =  0x0 / s_mov_b32
#  SMEM    =  0x1 / s_load_b*
#  JUMP    =  0x3 / s_cbranch_scc0
#  NEXT    =  0x4 / s_cbranch_execz
#  MESSAGE =  0x9 / s_sendmsg
#  VALU    =  0xb / v_(exp,log)_f32_e32
#  VALU    =  0xd / v_lshlrev_b64
#  VALU    =  0xe / v_mad_u64_u32
#  VMEM    = 0x21 / global_load_b32
#  VMEM    = 0x22 / global_load_b32
#  VMEM    = 0x24 / global_store_b32
#  VMEM    = 0x25 / global_store_b64
#  VMEM    = 0x27 / global_store
#  VMEM    = 0x28 / global_store_b64
#  LDS     = 0x29 / ds_load_b128
#  LDS     = 0x2b / ds_store_b32
#  LDS     = 0x2e / ds_store_b128
#  ????    = 0x5a / hidden global_load  instruction
#  ????    = 0x5b / hidden global_load  instruction
#  ????    = 0x5c / hidden global_store instruction
#  VALU    = 0x73 / v_cmpx_eq_u32_e32 (not normal VALUINST)
OPNAME = {
  0x0: "SALU",
  0x1: "SMEM",
  0x3: "JUMP",
  0x4: "NEXT",
  0x9: "MESSAGE",
  0xb: "VALU",
  0xd: "VALU",
  0xe: "VALU",
  0x21: "VMEM_LOAD",
  0x22: "VMEM_LOAD",
  0x24: "VMEM_STORE",
  0x25: "VMEM_STORE",
  0x26: "VMEM_STORE",
  0x27: "VMEM_STORE",
  0x28: "VMEM_STORE",
  0x29: "LDS_LOAD",
  0x2b: "LDS_STORE",
  0x2e: "LDS_STORE",
  0x50: "__SIMD_LDS_LOAD",
  0x51: "__SIMD_LDS_LOAD",
  0x54: "__SIMD_LDS_STORE",
  0x5a: "__SIMD_VMEM_LOAD",
  0x5b: "__SIMD_VMEM_LOAD",
  0x5c: "__SIMD_VMEM_STORE",
  0x5d: "__SIMD_VMEM_STORE",
  0x5e: "__SIMD_VMEM_STORE",
  0x5f: "__SIMD_VMEM_STORE",
  0x72: "SALU_OR",
  0x73: "VALU_CMPX",
}

ALUSRC = {
  1: "SALU",
  2: "VALU",
  3: "VALU_SALU",
}

MEMSRC = {
  0: "LDS",
  1: "__LDS",
  2: "VMEM",
  3: "__VMEM",
}


# these tables are from rocprof trace decoder
# rocprof_trace_decoder_parse_data-0x11c6a0
# parse_sqtt_180 = b *rocprof_trace_decoder_parse_data-0x11c6a0+0x110040

# ---------- 1. local_138: 256-byte state->opcode table ----------

STATE_TO_OPCODE: bytes = bytes([
  0x10, 0x16, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x17, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x07, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x19, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x00, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x11, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x12, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x15, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x16, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x17, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x07, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x19, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x00, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x11, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x13, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x15, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
])

# opcode mask (the bits used to determine the opcode, worked out by looking at the repeats in STATE_TO_OPCODE)

opcode_mask = {
  0x10: 0b1111,

  0x16: 0b1111111,
  0x17: 0b1111111,
  0x07: 0b1111111,
  0x19: 0b1111111,
  0x11: 0b1111111,
  0x12: 0b11111111,
  0x13: 0b11111111,
  0x15: 0b1111111,

  0x18: 0b111,
  0x1: 0b111,

  0x5: 0b11111,
  0x6: 0b11111,
  0xb: 0b11111,
  0x8: 0b11111,
  0xc: 0b11111,
  0xd: 0b11111,

  0xf: 0b1111,
  0x14: 0b1111,

  0x9: 0b11111,
  0xa: 0b11111,

  0x4: 0b1111,
  0x3: 0b1111,
  0x2: 0b1111,
}

# ---------- 2. DAT_0012e280: nibble budget per opcode&0x1F ----------

NIBBLE_BUDGET = [
  0x08, 0x0C, 0x08, 0x08, 0x0C, 0x18, 0x18, 0x40, 0x14, 0x20, 0x30, 0x14, 0x34, 0x1C, 0x30, 0x08,
  0x04, 0x18, 0x18, 0x20, 0x40, 0x40, 0x30, 0x40, 0x14, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
]

# ---------- 3. delta_map from your hash nodes ----------

# opcode -> (shift, width)
DELTA_MAP_DEFAULT = {
  0x01: (3,  3),   # shift=3,  end=6
  0x02: (4,  2),   # shift=4,  end=6
  0x03: (4,  2),   # shift=4,  end=6
  0x04: (4,  3),   # shift=4,  end=7
  0x05: (5,  3),   # shift=5,  end=8
  0x06: (5,  3),   # shift=5,  end=8
  0x07: (8,  3),   # shift=8,  end=11
  0x08: (5,  3),   # shift=5,  end=8
  0x09: (5,  2),   # shift=5,  end=7
  0x0A: (5,  2),   # shift=5,  end=7
  0x0B: (5,  3),   # shift=5,  end=8
  0x0C: (5,  3),   # shift=5,  end=8
  0x0D: (5,  3),   # shift=5,  end=8
  # NOTE: 0x0e can never be decoded, it's not in the STATE_TO_OPCODE table
  #0x0E: (7,  2),   # shift=7,  end=9
  0x0F: (4,  4),   # shift=4,  end=8
  0x10: (0,  0),   # shift=0,  end=0  (no delta)
  0x11: (7,  9),   # shift=7,  end=16
  0x12: (8,  3),   # shift=8,  end=11
  0x13: (8,  3),   # shift=8,  end=11
  0x14: (4,  3),   # shift=4,  end=7
  0x15: (7,  3),   # shift=7,  end=10
  0x16: (12, 36),  # shift=12, end=48 (36-bit field, matches the 0x16 special-case)
  0x17: (0,  0),   # shift=0,  end=0  (no delta)
  0x18: (4,  3),   # shift=4,  end=7
  0x19: (7,  2),   # shift=7,  end=9
}

# ---------- 4. One-line-per-packet parser ----------

def reg_mask(opcode):
  nb_bits = NIBBLE_BUDGET[opcode & 0x1F]
  shift, width = DELTA_MAP_DEFAULT[opcode]
  delta_mask = ((1 << width) - 1) << shift
  assert delta_mask & opcode_mask[opcode] == 0, "masks shouldn't overlap"
  return ((1 << nb_bits) - 1) & ~(delta_mask | opcode_mask[opcode])

def decode_packet_fields(opcode: int, reg: int) -> str:
  """
  Decode packet payloads conservatively, using:
    - NIBBLE_BUDGET[opcode & 0x1F] to mask reg down to true width.
    - DELTA_MAP_DEFAULT[opcode] to expose the "primary" field (often delta).
    - Per-opcode layouts derived from rocprof's decompiled consumers.
  """
  # --- 0. Restrict to real packet bits not used in delta ---------------------------------
  pkt = reg & reg_mask(opcode)
  fields: list[str] = []

  match opcode:
    case 0x01: # VALUINST
      # 6 bit field
      flag = (pkt >> 6) & 1
      wave = pkt >> 7
      fields.append(f"wave={wave:x}")
      if flag: fields.append("flag")
    case 0x02: # VMEMEXEC
      # 2 bit field (pipe is a guess)
      src = pkt>>6
      fields.append(f"src={src} [{MEMSRC.get(src, '')}]")
    case 0x03: # ALUEXEC
      # 2 bit field
      src = pkt>>6
      fields.append(f"src={src} [{ALUSRC.get(src, '')}]")
    case 0x04: # IMMEDIATE_4
      # 5 bit field (actually 4)
      wave = pkt >> 7
      fields.append(f"wave={wave:x}")
    case 0x05: # IMMEDIATE_5
      # 16 bit field
      # 1 bit per wave
      fields.append(f"mask={pkt>>8:016b}")
    case 0x6:
      # wave ready FFFF00
      # 16 bit field
      # 1 bit per wave
      fields.append(f"mask={pkt>>8:016b}")
    case 0x0d:
      # 20 bit field
      fields.append(f"arg = {pkt>>8:X}")
    case 0x12:
      fields.append(f"event = {pkt>>11:X}")
    case 0x15:
      fields.append(f"snap = {pkt>>10:X}")
    case 0x19:
      # wave end
      fields.append(f"ctr = {pkt>>9:X}")
    case 0xf:
      extracted_delta = (reg >> 4) & 0xF
      fields.append(f"strange_delta=0x{extracted_delta:x}")
    case 0x11:
      # DELTA_MAP_DEFAULT: shift=7, width=9 -> small delta.
      # FF0000 is the mask
      coarse    = pkt >> 16
      fields.append(f"coarse=0x{coarse:02x}")
      # From decomp:
      #  - when layout<3 and coarse&1, it sets a "has interesting wave" flag
      #  - when coarse&8, it marks all live waves as "terminated"
      if coarse & 0x01:
        fields.append("flag_wave_interest=1")
      if coarse & 0x08:
        fields.append("flag_terminate_all=1")
    case 0x8:
      # wave end, this is 20 bits (FFF00)
      flag7   = (pkt >> 8) & 1
      simd    = (pkt >> 9) & 3
      cu      = ((pkt >> 11) & 0x7) | (flag7 << 3)
      wave    = (pkt >> 15) & 0x1f
      fields.append(f"wave={wave:x}")
      fields.append(f"simd={simd}")
      fields.append(f"cu={cu}")
    case 0x9:
      # From case 9 (WAVESTART) in multiple consumers:
      #   flag7  = (w >> 7) & 1        (low bit of uVar41)
      #   cls2   = (w >> 8) & 3        (class / group)
      #   slot4  = (w >> 10) & 0xf     (slot / group index)
      #   idx_lo = (w >> 0xd) & 0x1f   (low index, layout<4 path)
      #   idx_hi = (w >> 0xf) & 0x1f   (high index, layout>=4 path)
      #   id7    = (w >> 0x19) & 0x7f  (7-bit id)
      flag7   = (pkt >> 7) & 1
      simd    = (pkt >> 8) & 3
      cu      = ((pkt >> 10) & 0x7) | (flag7 << 3)
      wave    = (pkt >> 13) & 0x1F
      id7     = (pkt >> 17)
      fields.append(f"wave={wave:x}")
      fields.append(f"simd={simd}")
      fields.append(f"cu={cu}")
      fields.append(f"id7=0x{id7:x}")
    case 0x18:
      # FFF88 is the mask
      # From case 0x18:
      #   low3   = w & 7
      #   grp3   = (w >> 3) or (w >> 4) & 7   (layout-dependent)
      #   flags  = bits 6 (B6) and 7 (B7)
      #   hi8    = (w >> 0xc) & 0xff   (layout 4 path)
      #   hi7    = (w >> 0xd) & 0x7f   (other layouts)
      #   idx5   = (w >> 7) or (w >> 8) & 0x1f, used as wave index
      flag1 = (pkt >> 3) & 1
      flag2 = (pkt >> 7) & 1
      wave = (pkt >> 8) & 0x1F
      op = (pkt >> 13)
      fields.append(f"wave={wave:x}")
      fields.append(f"op=0x{op:02x} [{OPNAME.get(op, '')}]")
      if flag1: fields.append("flag1")
      if flag2: fields.append("flag2")
    case 0x14:
      subop   = (pkt >> 16) & 0xFFFF       # (short)(w >> 0x10)
      val32   = (pkt >> 32) & 0xFFFFFFFF   # (uint)(w >> 0x20)
      slot    = (pkt >> 7) & 0x7           # index in local_168[...] tables
      hi_byte = (pkt >> 8) & 0xFF          # determines config vs marker

      fields.append(f"subop=0x{subop:04x}")
      fields.append(f"slot={slot}")
      fields.append(f"val32=0x{val32:08x}")

      if hi_byte & 0x80:
        # Config flavour: writes config words into per-slot state arrays.
        fields.append("kind=config")
        if subop == 0x000C:
          fields.append("slot=lo")
        elif subop == 0x000D:
          fields.append("slot=hi")
      else:
        # COR marker: subop 0xC342, payload "COR\0" → start of a COR region.
        if subop == 0xC342:
          fields.append("kind=cor_stream")
          if val32 == 0x434F5200:
            fields.append("cor_magic='COR\\0'")
    case 0x16:
      # Bits:
      #   bit8  -> 0x100
      #   bit9  -> 0x200
      #   bits 12..47 -> 36-bit field used as delta or marker
      bit8 = bool(pkt & 0x100)
      bit9 = bool(pkt & 0x200)
      if not bit9:
        mode = "delta"
      elif not bit8:
        mode = "marker"
      else:
        mode = "other"
      # need to use reg here
      val36 = (reg >> 12) & ((1 << 36) - 1)
      fields.append(f"mode={mode}")
      if mode != "delta":
        fields.append(f"val36=0x{val36:x}")
    case 0x17:
      # From decomp (two sites with identical logic):
      #   layout = (w >> 7) & 0x3f
      #   mode   = (w >> 0xd) & 3
      #   group  = (w >> 0xf) & 7
      #   sel_a  = (w >> 0x1c) & 0xf
      #   sel_b  = (w >> 0x21) & 7
      #   flag4  = (w >> 0x3b) & 1  (only meaningful when layout == 4)
      layout = (pkt >> 7)  & 0x3F
      simd   = (pkt >> 13) & 0x3  # you can change this by changing traced simd
      group  = (pkt >> 15) & 0x7
      sel_a  = (pkt >> 0x1C) & 0xF
      sel_b  = (pkt >> 0x21) & 0x7
      flag4  = (pkt >> 0x3B) & 0x1

      fields.append(f"layout={layout}")
      fields.append(f"group={group}")
      fields.append(f"simd={simd}")
      fields.append(f"sel_a={sel_a}")
      fields.append(f"sel_b={sel_b}")
      if layout == 4:
        fields.append(f"layout4_flag={flag4}")
    case _:
      fields.append(f"{pkt:X} & {reg_mask(opcode):X}")
  return ",".join(fields)

FILTER_LEVEL = getenv("FILTER", 1)

DEFAULT_FILTER: tuple[int, ...] = tuple()
# NOP + pure time + "sample"
if FILTER_LEVEL >= 0: DEFAULT_FILTER += (0x10, 0xf, 0x11)
# reg + event + sample + marker
# TODO: events are probably good
if FILTER_LEVEL >= 1: DEFAULT_FILTER += (0x14, 0x12, 0x16)
# instruction runs + valuinst
if FILTER_LEVEL >= 2: DEFAULT_FILTER += (0x01, 0x02, 0x03)
# instructions dispatch (inst, immed)
if FILTER_LEVEL >= 3: DEFAULT_FILTER += (0x4, 0x5, 0x18)
# waves
if FILTER_LEVEL >= 4: DEFAULT_FILTER += (0x6, 0x8, 0x9)

def parse_sqtt_print_packets(data: bytes, filter=DEFAULT_FILTER, verbose=True) -> None:
  """
  Minimal debug: print ONE LINE per decoded token (packet).

  Now prints only the actual nibbles that belong to each packet, instead of
  the full 64-bit shift register.
  """
  n = len(data)
  time = 0
  last_printed_time = 0
  reg = 0          # shift register
  offset = 0       # bit offset, in steps of 4 (one nibble)
  nib_budget = 0x40
  flags = 0
  token_index = 0
  opcodes_seen = set()

  while (offset >> 3) < n:
    # 1) Fill register with nibbles according to nib_budget
    if nib_budget != 0:
      target = offset + 4 + ((nib_budget - 1) & ~3)
      while offset != target and (offset >> 3) < n:
        byte = data[offset >> 3]
        nib = (byte >> (offset & 4)) & 0xF
        reg = ((reg >> 4) | (nib << 60)) & ((1 << 64) - 1)
        offset += 4
      if offset != target: break  # don't parse past the end

    # 2) Decode token from low 8 bits
    opcode = STATE_TO_OPCODE[reg & 0xFF]
    opcodes_seen.add(opcode)

    # 4) Set next nibble budget based on opcode
    nib_budget = NIBBLE_BUDGET[opcode & 0x1F]

    # 5) Get delta
    shift, width = DELTA_MAP_DEFAULT[opcode]
    delta = (reg >> shift) & ((1 << width) - 1)

    # 6) Update time and handle special opcodes 0xF/0x16
    if opcode == 0x16:
      two_bits = (reg >> 8) & 0x3
      if two_bits == 1:
        flags |= 0x01

      # Common 36-bit field at bits [12..47]
      if (reg & 0x200) == 0:
        # delta mode: add 36-bit delta to time
        pass
      elif (reg & 0x100) == 0:
        # marker / other modes: no time advance
        # real marker: bit9=1, bit8=0, non-zero payload
        # "other" 0x16 variants, ignored for timing
        delta = 0
      else:
        raise RuntimeError("unknown 0x16 delta")
    elif opcode == 0x0F:
      # opcode 0x0F has an offset of 4 to the delta
      # update: it's actually computed to be 8 to match WAVESTART
      delta = delta + 8

    # Append extra decoded fields into the note string
    note = decode_packet_fields(opcode, reg)

    # this delta happens before the instruction
    time += delta
    token_index += 1

    if verbose and (filter is None or opcode not in filter):
      print(f"{time:8d} +{time-last_printed_time:8d} : "+colored(f"{OPCODE_NAMES[opcode]:18s} ", OPCODE_COLORS.get(opcode, "white"))+f"{note}")
      last_printed_time = time

  # Optional summary at the end
  print(f"# done: tokens={token_index:_}, final_time={time}, flags=0x{flags:02x}")
  if verbose:
    print(f"opcodes({len(opcodes_seen):2d}):",
          ' '.join([colored(f"{op:2X}", "WHITE" if op in opcodes_seen else "BLACK") for op in sorted(opcode_mask)]))


def parse(fn:str):
  with Timing(f"unpickle {fn}: "): dat = pickle.load(open(fn, "rb"))
  #if getenv("ROCM", 0):
  #  with Timing(f"decode {fn}: "): ctx = decode(dat)
  dat_sqtt = [x for x in dat if isinstance(x, ProfileSQTTEvent)]
  print(f"got {len(dat_sqtt)} SQTT events in {fn}")
  return dat_sqtt

if __name__ == "__main__":
  fn = "extra/sqtt/examples/profile_gemm_run_0.pkl"
  dat_sqtt = parse(sys.argv[1] if len(sys.argv) > 1 else fn)
  for i,dat in enumerate(dat_sqtt):
    with Timing(f"decode pkt {i} with len {len(dat.blob):_}: "):
      parse_sqtt_print_packets(dat.blob, verbose=getenv("V", 1))
