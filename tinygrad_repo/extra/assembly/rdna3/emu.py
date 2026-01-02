# RDNA3 emulator - pure Python implementation for testing
from __future__ import annotations
import ctypes, struct, math
from typing import Callable
from extra.assembly.rdna3.lib import Inst, Inst32, Inst64, RawImm

Program = dict[int, Inst]  # pc (word offset) -> instruction
from extra.assembly.rdna3.autogen import (
  SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, VOP1, VOP2, VOP3, VOP3SD, VOP3P, VOPC, DS, FLAT, VOPD, SrcEnum,
  SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, SMEMOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp, DSOp, FLATOp, GLOBALOp, VOPDOp
)
from extra.assembly.rdna3.alu import (
  f32, i32, f16, i16, sext, vopc, SALU, VALU,
  SOP1_BASE, SOP2_BASE, SOPC_BASE, SOPK_BASE, VOP1_BASE, VOP2_BASE
)

WAVE_SIZE, SGPR_COUNT, VGPR_COUNT = 32, 128, 256
VCC_LO, VCC_HI, NULL, M0, EXEC_LO, EXEC_HI, SCC = SrcEnum.VCC_LO, SrcEnum.VCC_HI, SrcEnum.NULL, SrcEnum.M0, SrcEnum.EXEC_LO, SrcEnum.EXEC_HI, SrcEnum.SCC
# Pre-computed inline constant table for src operands 128-254 (index = src - 128)
_INLINE_CONSTS = [0] * 127
for _i in range(65): _INLINE_CONSTS[_i] = _i  # 128-192 -> 0-64
for _i in range(1, 17): _INLINE_CONSTS[64 + _i] = ((-_i) & 0xffffffff)  # 193-208 -> -1 to -16
for _k, _v in {SrcEnum.POS_HALF: 0x3f000000, SrcEnum.NEG_HALF: 0xbf000000, SrcEnum.POS_ONE: 0x3f800000, SrcEnum.NEG_ONE: 0xbf800000,
               SrcEnum.POS_TWO: 0x40000000, SrcEnum.NEG_TWO: 0xc0000000, SrcEnum.POS_FOUR: 0x40800000, SrcEnum.NEG_FOUR: 0xc0800000,
               SrcEnum.INV_2PI: 0x3e22f983}.items(): _INLINE_CONSTS[_k - 128] = _v

_valid_mem_ranges: list[tuple[int, int]] = []
def set_valid_mem_ranges(ranges: set[tuple[int, int]]) -> None: global _valid_mem_ranges; _valid_mem_ranges = list(ranges)
def _mem_valid(addr: int, size: int) -> bool:
  for s, z in _valid_mem_ranges:
    if s <= addr and addr + size <= s + z: return True
  return not _valid_mem_ranges
def _ctypes_at(addr: int, size: int): return (ctypes.c_uint8 if size == 1 else ctypes.c_uint16 if size == 2 else ctypes.c_uint32).from_address(addr)
def mem_read(addr: int, size: int) -> int: return _ctypes_at(addr, size).value if _mem_valid(addr, size) else 0
def mem_write(addr: int, size: int, val: int) -> None:
  if _mem_valid(addr, size): _ctypes_at(addr, size).value = val

# Memory op tables - (cnt, sz, sign) for loads, (cnt, sz) for stores
def _mem_ops(ops, suffix_map):
  return {getattr(e, f"{p}_{s}"): v for e in ops for s, v in suffix_map.items() for p in [e.__name__.replace("Op", "")]}
_LOAD_MAP = {'LOAD_B32': (1,4,0), 'LOAD_B64': (2,4,0), 'LOAD_B96': (3,4,0), 'LOAD_B128': (4,4,0), 'LOAD_U8': (1,1,0), 'LOAD_I8': (1,1,1), 'LOAD_U16': (1,2,0), 'LOAD_I16': (1,2,1)}
_STORE_MAP = {'STORE_B32': (1,4), 'STORE_B64': (2,4), 'STORE_B96': (3,4), 'STORE_B128': (4,4), 'STORE_B8': (1,1), 'STORE_B16': (1,2)}
FLAT_LOAD = _mem_ops([GLOBALOp, FLATOp], _LOAD_MAP)
FLAT_STORE = _mem_ops([GLOBALOp, FLATOp], _STORE_MAP)
DS_LOAD: dict[int, tuple[int,int,int]] = {DSOp.DS_LOAD_B32: (1,4,0), DSOp.DS_LOAD_B64: (2,4,0), DSOp.DS_LOAD_B128: (4,4,0), DSOp.DS_LOAD_U8: (1,1,0), DSOp.DS_LOAD_I8: (1,1,1), DSOp.DS_LOAD_U16: (1,2,0), DSOp.DS_LOAD_I16: (1,2,1)}
DS_STORE: dict[int, tuple[int,int]] = {DSOp.DS_STORE_B32: (1,4), DSOp.DS_STORE_B64: (2,4), DSOp.DS_STORE_B128: (4,4), DSOp.DS_STORE_B8: (1,1), DSOp.DS_STORE_B16: (1,2)}
FLAT_D16_LO = {getattr(e, f"{e.__name__.replace('Op', '')}_{s}"): v for e in [FLATOp, GLOBALOp] for s, v in [('LOAD_D16_U8', (1, 0)), ('LOAD_D16_I8', (1, 1)), ('LOAD_D16_B16', (2, 0))]}
FLAT_D16_HI = {getattr(e, f"{e.__name__.replace('Op', '')}_{s}"): v for e in [FLATOp, GLOBALOp] for s, v in [('LOAD_D16_HI_U8', (1, 0)), ('LOAD_D16_HI_I8', (1, 1)), ('LOAD_D16_HI_B16', (2, 0))]}
FLAT_D16_STORE = {getattr(e, f"{e.__name__.replace('Op', '')}_{s}"): v for e in [FLATOp, GLOBALOp] for s, v in [('STORE_D16_HI_B8', 1), ('STORE_D16_HI_B16', 2)]}
SMEM_LOAD: dict[int, int] = {SMEMOp.S_LOAD_B32: 1, SMEMOp.S_LOAD_B64: 2, SMEMOp.S_LOAD_B128: 4, SMEMOp.S_LOAD_B256: 8, SMEMOp.S_LOAD_B512: 16}
SOPK_WAIT = {SOPKOp.S_WAITCNT_VSCNT, SOPKOp.S_WAITCNT_VMCNT, SOPKOp.S_WAITCNT_EXPCNT, SOPKOp.S_WAITCNT_LGKMCNT}

class WaveState:
  __slots__ = ('sgpr', 'vgpr', 'scc', 'pc', 'literal', '_pend_sgpr')
  def __init__(self):
    self.sgpr, self.vgpr = [0] * SGPR_COUNT, [[0] * VGPR_COUNT for _ in range(WAVE_SIZE)]
    self.sgpr[EXEC_LO] = 0xffffffff  # wave32: all lanes active
    self.scc = self.pc = self.literal = 0
    self._pend_sgpr = {}

  @property
  def vcc(self) -> int: return self.sgpr[VCC_LO] | (self.sgpr[VCC_HI] << 32)
  @vcc.setter
  def vcc(self, v: int) -> None: self.sgpr[VCC_LO] = v & 0xffffffff; self.sgpr[VCC_HI] = (v >> 32) & 0xffffffff
  @property
  def exec_mask(self) -> int: return self.sgpr[EXEC_LO] | (self.sgpr[EXEC_HI] << 32)
  @exec_mask.setter
  def exec_mask(self, v: int) -> None: self.sgpr[EXEC_LO] = v & 0xffffffff; self.sgpr[EXEC_HI] = (v >> 32) & 0xffffffff

  def rsgpr(self, i: int) -> int:
    if i == NULL: return 0
    if i == SCC: return self.scc
    return self.sgpr[i] if i < SGPR_COUNT else 0
  def wsgpr(self, i: int, v: int) -> None:
    if i < SGPR_COUNT and i != NULL: self.sgpr[i] = v & 0xffffffff
  def rsgpr64(self, i: int) -> int: return self.rsgpr(i) | (self.rsgpr(i+1) << 32)
  def wsgpr64(self, i: int, v: int) -> None: self.wsgpr(i, v & 0xffffffff); self.wsgpr(i+1, (v >> 32) & 0xffffffff)

  def rsrc(self, v: int, lane: int) -> int:
    if v < SGPR_COUNT: return self.sgpr[v]
    if v == SCC: return self.scc
    if v < 255: return _INLINE_CONSTS[v - 128]
    if v == 255: return self.literal
    return self.vgpr[lane][v - 256] if v <= 511 else 0

  def rsrc64(self, v: int, lane: int) -> int:
    return self.rsrc(v, lane) | ((self.rsrc(v+1, lane) if v < VCC_LO or 256 <= v <= 511 else 0) << 32)

  def pend_sgpr_lane(self, reg: int, lane: int, val: int) -> None:
    if reg not in self._pend_sgpr: self._pend_sgpr[reg] = 0
    if val: self._pend_sgpr[reg] |= (1 << lane)

  def commit_pends(self) -> None:
    for reg, val in self._pend_sgpr.items(): self.sgpr[reg] = val
    self._pend_sgpr.clear()

def decode_format(word: int) -> tuple[type[Inst] | None, bool]:
  hi2 = (word >> 30) & 0x3
  if hi2 == 0b11:
    enc = (word >> 26) & 0xf
    if enc == 0b1101: return SMEM, True
    if enc == 0b0101:
      op = (word >> 16) & 0x3ff
      return (VOP3SD, True) if op in (288, 289, 290, 764, 765, 766, 767, 768, 769, 770) else (VOP3, True)
    return {0b0011: (VOP3P, True), 0b0110: (DS, True), 0b0111: (FLAT, True), 0b0010: (VOPD, True)}.get(enc, (None, True))
  if hi2 == 0b10:
    enc = (word >> 23) & 0x7f
    return {0b1111101: (SOP1, False), 0b1111110: (SOPC, False), 0b1111111: (SOPP, False)}.get(enc, (SOPK, False) if ((word >> 28) & 0xf) == 0b1011 else (SOP2, False))
  enc = (word >> 25) & 0x7f
  return (VOPC, False) if enc == 0b0111110 else (VOP1, False) if enc == 0b0111111 else (VOP2, False)

def _unwrap(v) -> int: return v.val if isinstance(v, RawImm) else v.value if hasattr(v, 'value') else v

def decode_program(data: bytes) -> Program:
  result: Program = {}
  i = 0
  while i < len(data):
    word = int.from_bytes(data[i:i+4], 'little')
    inst_class, is_64 = decode_format(word)
    if inst_class is None: i += 4; continue
    base_size = 8 if is_64 else 4
    inst = inst_class.from_bytes(data[i:i+base_size])
    for name, val in inst._values.items(): setattr(inst, name, _unwrap(val))
    has_literal = any(getattr(inst, fld, None) == 255 for fld in ('src0', 'src1', 'src2', 'ssrc0', 'ssrc1', 'srcx0', 'srcy0'))
    if inst_class == VOP2 and inst.op in (44, 45, 55, 56): has_literal = True
    if inst_class == VOPD and (inst.opx in (1, 2) or inst.opy in (1, 2)): has_literal = True
    if inst_class == SOP2 and inst.op in (69, 70): has_literal = True
    if has_literal: inst._literal = int.from_bytes(data[i+base_size:i+base_size+4], 'little')
    inst._words = inst.size() // 4  # cache size for step_wave
    result[i // 4] = inst
    i += inst._words * 4
  return result

# ═══════════════════════════════════════════════════════════════════════════════
# SCALAR EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════
def exec_sop1(st: WaveState, inst: SOP1) -> int:
  s0, op = st.rsrc(inst.ssrc0, 0), inst.op
  # 64-bit and special ops handled inline
  if op == SOP1Op.S_MOV_B64: st.wsgpr64(inst.sdst, st.rsrc64(inst.ssrc0, 0)); return 0
  if op == SOP1Op.S_NOT_B64: r = (~st.rsrc64(inst.ssrc0, 0)) & 0xffffffffffffffff; st.wsgpr64(inst.sdst, r); st.scc = int(r != 0); return 0
  if op == SOP1Op.S_BITSET0_B32: st.wsgpr(inst.sdst, st.rsgpr(inst.sdst) & ~(1 << (s0 & 0x1f))); return 0
  if op == SOP1Op.S_BITSET1_B32: st.wsgpr(inst.sdst, st.rsgpr(inst.sdst) | (1 << (s0 & 0x1f))); return 0
  if op == SOP1Op.S_AND_SAVEEXEC_B32: old = st.exec_mask & 0xffffffff; st.exec_mask = s0 & old; st.scc = int(st.exec_mask != 0); st.wsgpr(inst.sdst, old); return 0
  if op == SOP1Op.S_OR_SAVEEXEC_B32: old = st.exec_mask & 0xffffffff; st.exec_mask = s0 | old; st.scc = int(st.exec_mask != 0); st.wsgpr(inst.sdst, old); return 0
  if op == SOP1Op.S_AND_NOT1_SAVEEXEC_B32: old = st.exec_mask & 0xffffffff; st.exec_mask = s0 & (~old & 0xffffffff); st.scc = int(st.exec_mask != 0); st.wsgpr(inst.sdst, old); return 0
  if op == SOP1Op.S_GETPC_B64: return -3
  if op == SOP1Op.S_SETPC_B64: return -4
  if op == SOP1Op.S_SWAPPC_B64: return -5
  if (fn := SALU.get(SOP1_BASE + op)) is None: raise NotImplementedError(f"SOP1 op {op}")
  r, scc = fn(s0, 0, st.scc); st.wsgpr(inst.sdst, r); st.scc = scc; return 0

_SOP2_64: dict[int, Callable[[int, int], int]] = {SOP2Op.S_AND_B64: lambda a, b: a & b, SOP2Op.S_OR_B64: lambda a, b: a | b, SOP2Op.S_XOR_B64: lambda a, b: a ^ b}
def exec_sop2(st: WaveState, inst: SOP2) -> int:
  s0, s1, op = st.rsrc(inst.ssrc0, 0), st.rsrc(inst.ssrc1, 0), inst.op
  # 64-bit ops handled inline
  if op == SOP2Op.S_LSHL_B64: r = (st.rsrc64(inst.ssrc0, 0) << (s1 & 0x3f)) & 0xffffffffffffffff; st.wsgpr64(inst.sdst, r); st.scc = int(r != 0); return 0
  if op == SOP2Op.S_LSHR_B64: r = st.rsrc64(inst.ssrc0, 0) >> (s1 & 0x3f); st.wsgpr64(inst.sdst, r); st.scc = int(r != 0); return 0
  if op == SOP2Op.S_ASHR_I64: r = sext(st.rsrc64(inst.ssrc0, 0), 64) >> (s1 & 0x3f); st.wsgpr64(inst.sdst, r & 0xffffffffffffffff); st.scc = int(r != 0); return 0
  if (fn := _SOP2_64.get(op)): r = fn(st.rsrc64(inst.ssrc0, 0), st.rsrc64(inst.ssrc1, 0)); st.wsgpr64(inst.sdst, r); st.scc = int(r != 0); return 0
  if op == SOP2Op.S_CSELECT_B64: st.wsgpr64(inst.sdst, st.rsrc64(inst.ssrc0, 0) if st.scc else st.rsrc64(inst.ssrc1, 0)); return 0
  if op == SOP2Op.S_FMAC_F32: st.wsgpr(inst.sdst, i32(f32(st.rsgpr(inst.sdst)) + f32(s0) * f32(s1))); return 0
  if op == SOP2Op.S_FMAAK_F32: st.wsgpr(inst.sdst, i32(f32(s0) * f32(s1) + f32(inst._literal or 0))); return 0
  if op == SOP2Op.S_FMAMK_F32: st.wsgpr(inst.sdst, i32(f32(s0) * f32(inst._literal or 0) + f32(s1))); return 0
  if (fn := SALU.get(SOP2_BASE + op)) is None: raise NotImplementedError(f"SOP2 op {op}")
  r, scc = fn(s0, s1, st.scc); st.wsgpr(inst.sdst, r); st.scc = scc; return 0

def exec_sopc(st: WaveState, inst: SOPC) -> int:
  s0, s1, op = st.rsrc(inst.ssrc0, 0), st.rsrc(inst.ssrc1, 0), inst.op
  if op == SOPCOp.S_CMP_EQ_U64: st.scc = int(st.rsrc64(inst.ssrc0, 0) == st.rsrc64(inst.ssrc1, 0)); return 0
  if op == SOPCOp.S_CMP_LG_U64: st.scc = int(st.rsrc64(inst.ssrc0, 0) != st.rsrc64(inst.ssrc1, 0)); return 0
  if (fn := SALU.get(SOPC_BASE + op)) is None: raise NotImplementedError(f"SOPC op {op}")
  st.scc = fn(s0, s1, st.scc)[1]; return 0

_SOPK_CMP = frozenset((SOPKOp.S_CMPK_EQ_I32, SOPKOp.S_CMPK_LG_I32, SOPKOp.S_CMPK_GT_I32, SOPKOp.S_CMPK_GE_I32,
  SOPKOp.S_CMPK_LT_I32, SOPKOp.S_CMPK_LE_I32, SOPKOp.S_CMPK_EQ_U32, SOPKOp.S_CMPK_LG_U32,
  SOPKOp.S_CMPK_GT_U32, SOPKOp.S_CMPK_GE_U32, SOPKOp.S_CMPK_LT_U32, SOPKOp.S_CMPK_LE_U32))
def exec_sopk(st: WaveState, inst: SOPK) -> int:
  simm, s0, op = inst.simm16, st.rsgpr(inst.sdst), inst.op
  if op in SOPK_WAIT: return 0
  if (fn := SALU.get(SOPK_BASE + op)) is None: raise NotImplementedError(f"SOPK op {op}")
  r, scc = fn(s0, simm, st.scc)
  if op not in _SOPK_CMP: st.wsgpr(inst.sdst, r)
  st.scc = scc; return 0

def exec_sopp(st: WaveState, inst: SOPP) -> int:
  if inst.op == SOPPOp.S_ENDPGM: return -1
  if inst.op == SOPPOp.S_BARRIER: return -2
  if inst.op == SOPPOp.S_BRANCH: return sext(inst.simm16, 16)
  if inst.op == SOPPOp.S_CBRANCH_SCC0: return sext(inst.simm16, 16) if st.scc == 0 else 0
  if inst.op == SOPPOp.S_CBRANCH_SCC1: return sext(inst.simm16, 16) if st.scc == 1 else 0
  # In wave32 mode, only VCC_LO is used for lane masks; VCC_HI is a free SGPR
  if inst.op == SOPPOp.S_CBRANCH_VCCZ: return sext(inst.simm16, 16) if (st.vcc & 0xffffffff) == 0 else 0
  if inst.op == SOPPOp.S_CBRANCH_VCCNZ: return sext(inst.simm16, 16) if (st.vcc & 0xffffffff) != 0 else 0
  if inst.op == SOPPOp.S_CBRANCH_EXECZ: return sext(inst.simm16, 16) if st.exec_mask == 0 else 0
  if inst.op == SOPPOp.S_CBRANCH_EXECNZ: return sext(inst.simm16, 16) if st.exec_mask != 0 else 0
  # Scheduling hints and wait instructions are no-ops in emulation
  if inst.op <= 31: return 0  # S_NOP, S_CLAUSE, S_DELAY_ALU, S_WAITCNT, etc.
  # S_WAKEUP(52), S_SETPRIO(53), S_SENDMSG(54), S_SENDMSGHALT(55), perf counters, S_ICACHE_INV(60) are no-ops
  if inst.op in (52, 53, 54, 55, 56, 57, 60): return 0
  raise NotImplementedError(f"SOPP op {inst.op}")

def exec_smem(st: WaveState, inst: SMEM) -> int:
  addr = st.rsgpr64(inst.sbase * 2) + sext(inst.offset, 21)
  if inst.soffset not in (NULL, 0x7f): addr += st.rsrc(inst.soffset, 0)
  if (cnt := SMEM_LOAD.get(inst.op)) is None: raise NotImplementedError(f"SMEM op {inst.op}")
  for i in range(cnt): st.wsgpr(inst.sdata + i, mem_read((addr + i * 4) & 0xffffffffffffffff, 4))
  return 0

# ═══════════════════════════════════════════════════════════════════════════════
# VECTOR EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════
def f64(hi: int, lo: int) -> float: return struct.unpack('<d', struct.pack('<Q', (hi << 32) | lo))[0]
def i64_parts(f: float) -> tuple[int, int]:
  if math.isnan(f): val = 0x7ff8000000000000
  elif math.isinf(f): val = 0x7ff0000000000000 if f > 0 else 0xfff0000000000000
  else: val = struct.unpack('<Q', struct.pack('<d', f))[0]
  return val & 0xffffffff, (val >> 32) & 0xffffffff

def exec_vop1(st: WaveState, inst: VOP1, lane: int) -> None:
  if inst.op == VOP1Op.V_NOP: return
  V, s0 = st.vgpr[lane], st.rsrc(inst.src0, lane)
  if inst.op == VOP1Op.V_READFIRSTLANE_B32:
    first = (st.exec_mask & -st.exec_mask).bit_length() - 1 if st.exec_mask else 0
    st.wsgpr(inst.vdst, st.rsrc(inst.src0, first) if inst.src0 >= 256 else s0); return
  # F64 ops handled inline
  if inst.op == VOP1Op.V_CVT_F64_F32: V[inst.vdst], V[inst.vdst+1] = i64_parts(float(f32(s0))); return
  if inst.op == VOP1Op.V_CVT_F64_I32: V[inst.vdst], V[inst.vdst+1] = i64_parts(float(sext(s0, 32))); return
  if inst.op == VOP1Op.V_CVT_F64_U32: V[inst.vdst], V[inst.vdst+1] = i64_parts(float(s0)); return
  if inst.op in (VOP1Op.V_CVT_F32_F64, VOP1Op.V_CVT_I32_F64, VOP1Op.V_CVT_U32_F64):
    src = inst.src0 - 256 if inst.src0 >= 256 else inst.src0
    lo, hi = (V[src], V[src+1]) if inst.src0 >= 256 else (st.sgpr[src], st.sgpr[src+1])
    v = f64(hi, lo)
    if inst.op == VOP1Op.V_CVT_F32_F64: V[inst.vdst] = i32(v)
    elif inst.op == VOP1Op.V_CVT_I32_F64: V[inst.vdst] = (max(-0x80000000, min(0x7fffffff, int(v))) & 0xffffffff) if math.isfinite(v) else 0
    else: V[inst.vdst] = max(0, min(0xffffffff, int(v))) if math.isfinite(v) and v == v else 0
    return
  if (fn := VALU.get(VOP1_BASE + inst.op)): V[inst.vdst] = fn(s0, 0, 0); return
  raise NotImplementedError(f"VOP1 op {inst.op}")

def exec_vop2(st: WaveState, inst: VOP2, lane: int) -> None:
  V, s0, s1, op = st.vgpr[lane], st.rsrc(inst.src0, lane), st.vgpr[lane][inst.vsrc1], inst.op
  if op == VOP2Op.V_CNDMASK_B32: V[inst.vdst] = s1 if (st.vcc >> lane) & 1 else s0; return
  if op == VOP2Op.V_FMAC_F32: V[inst.vdst] = i32(f32(s0)*f32(s1)+f32(V[inst.vdst])); return
  if op == VOP2Op.V_FMAMK_F32: V[inst.vdst] = i32(f32(s0)*f32(st.literal)+f32(s1)); return
  if op == VOP2Op.V_FMAAK_F32: V[inst.vdst] = i32(f32(s0)*f32(s1)+f32(st.literal)); return
  if op == VOP2Op.V_FMAC_F16: V[inst.vdst] = (V[inst.vdst] & 0xffff0000) | i16(f16(s0)*f16(s1)+f16(V[inst.vdst])); return
  if op == VOP2Op.V_FMAMK_F16: V[inst.vdst] = (V[inst.vdst] & 0xffff0000) | i16(f16(s0)*f16(st.literal)+f16(s1)); return
  if op == VOP2Op.V_FMAAK_F16: V[inst.vdst] = (V[inst.vdst] & 0xffff0000) | i16(f16(s0)*f16(s1)+f16(st.literal)); return
  if op == VOP2Op.V_PK_FMAC_F16:
    lo = i16(f16(s0 & 0xffff) * f16(s1 & 0xffff) + f16(V[inst.vdst] & 0xffff))
    hi = i16(f16((s0 >> 16) & 0xffff) * f16((s1 >> 16) & 0xffff) + f16((V[inst.vdst] >> 16) & 0xffff))
    V[inst.vdst] = lo | (hi << 16); return
  if op == VOP2Op.V_ADD_CO_CI_U32: r = s0+s1+((st.vcc>>lane)&1); st.pend_sgpr_lane(VCC_LO, lane, r >= 0x100000000); V[inst.vdst] = r & 0xffffffff; return
  if op == VOP2Op.V_SUB_CO_CI_U32: b = (st.vcc>>lane)&1; st.pend_sgpr_lane(VCC_LO, lane, s1+b > s0); V[inst.vdst] = (s0-s1-b) & 0xffffffff; return
  if (fn := VALU.get(VOP2_BASE + op)): V[inst.vdst] = fn(s0, s1, 0); return
  raise NotImplementedError(f"VOP2 op {op}")

def vop3_mod(val: int, neg: int, abs_: int, idx: int) -> int:
  if (abs_ >> idx) & 1: val = i32(abs(f32(val)))
  if (neg >> idx) & 1: val = i32(-f32(val))
  return val

def exec_vop3(st: WaveState, inst: VOP3, lane: int) -> None:
  op, src0, src1, src2, vdst, neg, abs_ = inst.op, inst.src0, inst.src1, inst.src2, inst.vdst, inst.neg, getattr(inst, 'abs', 0)
  V = st.vgpr[lane]
  # VOPC encoded in VOP3 (0-255)
  if 0 <= op <= 255:
    base = op & 0x7f
    # For 64-bit comparisons (I64: 80-87, U64: 88-95), read raw 64-bit values (no float modifiers)
    if 80 <= base <= 95:
      s0_64, s1_64 = st.rsrc64(src0, lane), st.rsrc64(src1, lane)
      result = vopc(op, s0_64 & 0xffffffff, s1_64 & 0xffffffff, (s0_64 >> 32) & 0xffffffff, (s1_64 >> 32) & 0xffffffff)
    else:
      s0, s1 = vop3_mod(st.rsrc(src0, lane), neg, abs_, 0), vop3_mod(st.rsrc(src1, lane), neg, abs_, 1)
      result = vopc(op, s0, s1)
    is_cmpx = op >= 128
    st.pend_sgpr_lane(vdst, lane, result)
    if is_cmpx: st.pend_sgpr_lane(EXEC_LO, lane, result)
    return
  s0, s1, s2 = vop3_mod(st.rsrc(src0, lane), neg, abs_, 0), vop3_mod(st.rsrc(src1, lane), neg, abs_, 1), vop3_mod(st.rsrc(src2, lane), neg, abs_, 2)
  # Special ops
  if op == VOP3Op.V_FMAC_F32: V[vdst] = i32(f32(s0)*f32(s1)+f32(V[vdst])); return
  if op == VOP3Op.V_READLANE_B32: st.wsgpr(vdst, st.vgpr[s1 & 0x1f][src0 - 256] if src0 >= 256 else s0); return
  if op == VOP3Op.V_WRITELANE_B32: st.vgpr[s1 & 0x1f][vdst] = s0; return
  if op == VOP3Op.V_CNDMASK_B32:
    mask = st.rsgpr(src2) if src2 < 256 else st.vcc
    V[vdst] = s1 if (mask >> lane) & 1 else s0; return
  if op in (VOP3Op.V_LSHLREV_B64, VOP3Op.V_LSHRREV_B64, VOP3Op.V_ASHRREV_I64):
    v64 = st.rsrc64(src1, lane)
    r = ((v64 << (s0 & 0x3f)) & 0xffffffffffffffff if op == VOP3Op.V_LSHLREV_B64 else
         v64 >> (s0 & 0x3f) if op == VOP3Op.V_LSHRREV_B64 else sext(v64, 64) >> (s0 & 0x3f))
    V[vdst], V[vdst+1] = r & 0xffffffff, (r >> 32) & 0xffffffff; return
  if op in (VOP3Op.V_ADD_F64, VOP3Op.V_MUL_F64, VOP3Op.V_FMA_F64, VOP3Op.V_MAX_F64, VOP3Op.V_MIN_F64):
    a, b = f64(st.rsrc(src0+1, lane), s0), f64(st.rsrc(src1+1, lane), s1)
    c = f64(st.rsrc(src2+1, lane), s2) if op == VOP3Op.V_FMA_F64 else 0.0
    rf = a + b if op == VOP3Op.V_ADD_F64 else a * b if op == VOP3Op.V_MUL_F64 else a * b + c if op == VOP3Op.V_FMA_F64 else max(a, b) if op == VOP3Op.V_MAX_F64 else min(a, b)
    V[vdst], V[vdst+1] = i64_parts(rf); return
  if (fn := VALU.get(op)): V[vdst] = fn(s0, s1, s2); return
  raise NotImplementedError(f"VOP3 op {op}")

def exec_vopc(st: WaveState, inst: VOPC, lane: int) -> None:
  result, is_cmpx = vopc(inst.op, st.rsrc(inst.src0, lane), st.vgpr[lane][inst.vsrc1]), inst.op >= 128
  st.pend_sgpr_lane(EXEC_LO if is_cmpx else VCC_LO, lane, result)

def exec_vop3sd(st: WaveState, inst: VOP3SD, lane: int) -> None:
  op, src0, src1, src2, vdst, sdst, neg = inst.op, inst.src0, inst.src1, inst.src2, inst.vdst, inst.sdst, inst.neg
  s0, s1, s2 = st.rsrc(src0, lane), st.rsrc(src1, lane), st.rsrc(src2, lane)
  if (neg >> 0) & 1: s0 = i32(-f32(s0))
  if (neg >> 1) & 1: s1 = i32(-f32(s1))
  if (neg >> 2) & 1: s2 = i32(-f32(s2))
  V = st.vgpr[lane]
  if op == VOP3SDOp.V_ADD_CO_U32: r = s0 + s1; V[vdst] = r & 0xffffffff; st.pend_sgpr_lane(sdst, lane, r >= 0x100000000)
  elif op == VOP3SDOp.V_SUB_CO_U32: V[vdst] = (s0 - s1) & 0xffffffff; st.pend_sgpr_lane(sdst, lane, s1 > s0)
  elif op == VOP3SDOp.V_SUBREV_CO_U32: V[vdst] = (s1 - s0) & 0xffffffff; st.pend_sgpr_lane(sdst, lane, s0 > s1)
  elif op == VOP3SDOp.V_ADD_CO_CI_U32:
    cin = (st.rsgpr(src2) >> lane) & 1 if src2 < 256 else (st.vcc >> lane) & 1
    r = s0 + s1 + cin; V[vdst] = r & 0xffffffff; st.pend_sgpr_lane(sdst, lane, r >= 0x100000000)
  elif op == VOP3SDOp.V_SUB_CO_CI_U32:
    cin = (st.rsgpr(src2) >> lane) & 1 if src2 < 256 else (st.vcc >> lane) & 1
    V[vdst] = (s0 - s1 - cin) & 0xffffffff; st.pend_sgpr_lane(sdst, lane, s1 + cin > s0)
  elif op == VOP3SDOp.V_MAD_U64_U32:
    s2_64 = s2 | (st.rsrc(src2+1, lane) << 32); r = s0 * s1 + s2_64
    V[vdst], V[vdst+1] = r & 0xffffffff, (r >> 32) & 0xffffffff
  elif op == VOP3SDOp.V_MAD_I64_I32:
    s2_64 = sext(s2 | (st.rsrc(src2+1, lane) << 32), 64)
    r = (sext(s0, 32) * sext(s1, 32) + s2_64) & 0xffffffffffffffff
    V[vdst], V[vdst+1] = r & 0xffffffff, (r >> 32) & 0xffffffff
  elif op == VOP3SDOp.V_DIV_SCALE_F32: V[vdst] = 0; st.pend_sgpr_lane(sdst, lane, False)
  elif op == VOP3SDOp.V_DIV_SCALE_F64: V[vdst], V[vdst+1] = s0, st.rsrc(src0+1, lane); st.pend_sgpr_lane(VCC_LO, lane, s0 == s2)
  else: raise NotImplementedError(f"VOP3SD op {op}")

def exec_flat(st: WaveState, inst: FLAT, lane: int) -> None:
  op, addr_reg, data_reg, vdst, offset, saddr, V = inst.op, inst.addr, inst.data, inst.vdst, sext(inst.offset, 13), inst.saddr, st.vgpr[lane]
  addr = V[addr_reg] | (V[addr_reg+1] << 32)
  addr = (st.rsgpr64(saddr) + V[addr_reg] + offset) & 0xffffffffffffffff if saddr not in (NULL, 0x7f) else (addr + offset) & 0xffffffffffffffff
  if op in FLAT_LOAD:
    cnt, sz, sign = FLAT_LOAD[op]
    for i in range(cnt): val = mem_read(addr + i * sz, sz); V[vdst + i] = sext(val, sz * 8) & 0xffffffff if sign else val
  elif op in FLAT_STORE:
    cnt, sz = FLAT_STORE[op]
    for i in range(cnt): mem_write(addr + i * sz, sz, V[data_reg + i] & ((1 << (sz * 8)) - 1))
  elif op in FLAT_D16_LO: sz, sign = FLAT_D16_LO[op]; val = mem_read(addr, sz); V[vdst] = (V[vdst] & 0xffff0000) | ((sext(val, sz * 8) & 0xffff) if sign else (val & 0xffff))
  elif op in FLAT_D16_HI: sz, sign = FLAT_D16_HI[op]; val = mem_read(addr, sz); V[vdst] = (V[vdst] & 0x0000ffff) | (((sext(val, sz * 8) & 0xffff) if sign else (val & 0xffff)) << 16)
  elif op in FLAT_D16_STORE: mem_write(addr, FLAT_D16_STORE[op], (V[data_reg] >> 16) & ((1 << (FLAT_D16_STORE[op] * 8)) - 1))
  else: raise NotImplementedError(f"FLAT op {op}")

def exec_ds(st: WaveState, inst: DS, lane: int, lds: bytearray) -> None:
  op, addr, vdst, V = inst.op, (st.vgpr[lane][inst.addr] + inst.offset0) & 0xffff, inst.vdst, st.vgpr[lane]
  if op in DS_LOAD:
    cnt, sz, sign = DS_LOAD[op]
    for i in range(cnt): val = int.from_bytes(lds[addr+i*sz:addr+i*sz+sz], 'little'); V[vdst + i] = sext(val, sz * 8) & 0xffffffff if sign else val
  elif op in DS_STORE:
    cnt, sz = DS_STORE[op]
    for i in range(cnt): lds[addr+i*sz:addr+i*sz+sz] = (V[inst.data0 + i] & ((1 << (sz * 8)) - 1)).to_bytes(sz, 'little')
  else: raise NotImplementedError(f"DS op {op}")

VOPD_OPS: dict[int, Callable[[int, int, int, int, int], int]] = {
  VOPDOp.V_DUAL_MUL_F32: lambda a, b, d, l, lit: i32(f32(a)*f32(b)), VOPDOp.V_DUAL_ADD_F32: lambda a, b, d, l, lit: i32(f32(a)+f32(b)),
  VOPDOp.V_DUAL_SUB_F32: lambda a, b, d, l, lit: i32(f32(a)-f32(b)), VOPDOp.V_DUAL_SUBREV_F32: lambda a, b, d, l, lit: i32(f32(b)-f32(a)),
  VOPDOp.V_DUAL_MAX_F32: lambda a, b, d, l, lit: i32(max(f32(a), f32(b))), VOPDOp.V_DUAL_MIN_F32: lambda a, b, d, l, lit: i32(min(f32(a), f32(b))),
  VOPDOp.V_DUAL_MUL_DX9_ZERO_F32: lambda a, b, d, l, lit: i32(0.0 if f32(a) == 0.0 or f32(b) == 0.0 else f32(a)*f32(b)),
  VOPDOp.V_DUAL_MOV_B32: lambda a, b, d, l, lit: a, VOPDOp.V_DUAL_ADD_NC_U32: lambda a, b, d, l, lit: (a + b) & 0xffffffff,
  VOPDOp.V_DUAL_LSHLREV_B32: lambda a, b, d, l, lit: (b << (a & 0x1f)) & 0xffffffff, VOPDOp.V_DUAL_AND_B32: lambda a, b, d, l, lit: a & b,
  VOPDOp.V_DUAL_FMAC_F32: lambda a, b, d, l, lit: i32(f32(a)*f32(b)+f32(d)), VOPDOp.V_DUAL_FMAAK_F32: lambda a, b, d, l, lit: i32(f32(a)*f32(b)+f32(lit)),
  VOPDOp.V_DUAL_FMAMK_F32: lambda a, b, d, l, lit: i32(f32(a)*f32(lit)+f32(b)), VOPDOp.V_DUAL_CNDMASK_B32: lambda a, b, d, l, lit: b if l else a,
}
def exec_vopd(st: WaveState, inst: VOPD, lane: int) -> None:
  V, vdsty, vcc_lane = st.vgpr[lane], (inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1), (st.vcc >> lane) & 1
  sx0, sx1, sy0, sy1, dstx = st.rsrc(inst.srcx0, lane), V[inst.vsrcx1], st.rsrc(inst.srcy0, lane), V[inst.vsrcy1], inst.vdstx
  if (fn := VOPD_OPS.get(inst.opx)): V[dstx] = fn(sx0, sx1, V[dstx], vcc_lane, st.literal)
  else: raise NotImplementedError(f"VOPD opx {inst.opx}")
  if (fn := VOPD_OPS.get(inst.opy)): V[vdsty] = fn(sy0, sy1, V[vdsty], vcc_lane, st.literal)
  else: raise NotImplementedError(f"VOPD opy {inst.opy}")

def exec_vop3p(st: WaveState, inst: VOP3P, lane: int) -> None:
  op, vdst, V = inst.op, inst.vdst, st.vgpr[lane]
  s0, s1, s2 = st.rsrc(inst.src0, lane), st.rsrc(inst.src1, lane), st.rsrc(inst.src2, lane)
  opsel, opsel_hi = [(inst.opsel >> i) & 1 for i in range(3)], [(inst.opsel_hi >> i) & 1 for i in range(2)] + [inst.opsel_hi2]
  neg, neg_hi = inst.neg, inst.neg_hi
  def get_src(src: int, idx: int, for_mix: bool = False) -> float:
    if for_mix:
      if not opsel_hi[idx]: return abs(f32(src)) if (neg_hi >> idx) & 1 else f32(src)
      return float(f16((src >> 16) & 0xffff) if opsel[idx] else f16(src & 0xffff))
    use_hi = opsel[idx]
    val = ((src >> 16) & 0xffff) if use_hi else (src & 0xffff)
    f = f16(val)
    if use_hi and (neg >> idx) & 1: f = -f
    elif not use_hi and (neg_hi >> idx) & 1: f = -f
    return f
  if op == VOP3POp.V_FMA_MIX_F32: V[vdst] = i32(get_src(s0, 0, True) * get_src(s1, 1, True) + get_src(s2, 2, True))
  elif op == VOP3POp.V_FMA_MIXLO_F16: V[vdst] = (V[vdst] & 0xffff0000) | i16(get_src(s0, 0, True) * get_src(s1, 1, True) + get_src(s2, 2, True))
  elif op == VOP3POp.V_FMA_MIXHI_F16: V[vdst] = (V[vdst] & 0x0000ffff) | (i16(get_src(s0, 0, True) * get_src(s1, 1, True) + get_src(s2, 2, True)) << 16)
  else: raise NotImplementedError(f"VOP3P op {op}")

def exec_wmma_f32_16x16x16_f16(st: WaveState, inst: VOP3P, n_lanes: int) -> None:
  src0_base, src1_base, src2_base = (inst.src0 - 256) if inst.src0 >= 256 else inst.src0, (inst.src1 - 256) if inst.src1 >= 256 else inst.src1, (inst.src2 - 256) if inst.src2 >= 256 else inst.src2
  src0_is_vgpr, src1_is_vgpr, src2_is_vgpr, vdst = inst.src0 >= 256, inst.src1 >= 256, inst.src2 >= 256, inst.vdst
  A, B, C = [[0.0] * 16 for _ in range(16)], [[0.0] * 16 for _ in range(16)], [[0.0] * 16 for _ in range(16)]
  for lane in range(min(n_lanes, 16)):
    V = st.vgpr[lane]
    for reg in range(8):
      val = V[src0_base + reg] if src0_is_vgpr else st.sgpr[src0_base + reg]
      A[lane][reg * 2], A[lane][reg * 2 + 1] = f16(val & 0xffff), f16((val >> 16) & 0xffff)
      val = V[src1_base + reg] if src1_is_vgpr else st.sgpr[src1_base + reg]
      B[reg * 2][lane], B[reg * 2 + 1][lane] = f16(val & 0xffff), f16((val >> 16) & 0xffff)
  for row in range(16):
    for col in range(16):
      idx, lane_idx, reg = row * 16 + col, (row * 16 + col) % 32, (row * 16 + col) // 32
      if lane_idx < n_lanes:
        val = st.vgpr[lane_idx][src2_base + reg] if src2_is_vgpr else st.sgpr[src2_base + reg]
        C[row][col] = f32(val)
  for row in range(16):
    for col in range(16):
      for k in range(16): C[row][col] += A[row][k] * B[k][col]
  for row in range(16):
    for col in range(16):
      idx, lane_idx, reg = row * 16 + col, (row * 16 + col) % 32, (row * 16 + col) // 32
      if lane_idx < n_lanes and (st.exec_mask & (1 << lane_idx)): st.vgpr[lane_idx][vdst + reg] = i32(C[row][col])

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION LOOP
# ═══════════════════════════════════════════════════════════════════════════════
SCALAR: dict[type, Callable[..., int]] = {SOP1: exec_sop1, SOP2: exec_sop2, SOPC: exec_sopc, SOPK: exec_sopk, SOPP: exec_sopp, SMEM: exec_smem}
VECTOR: dict[type, Callable[..., None]] = {VOP1: exec_vop1, VOP2: exec_vop2, VOP3: exec_vop3, VOP3SD: exec_vop3sd, VOPC: exec_vopc, FLAT: exec_flat, DS: exec_ds, VOPD: exec_vopd, VOP3P: exec_vop3p}

_WMMA_OPS = frozenset((VOP3POp.V_WMMA_F32_16X16X16_F16, VOP3POp.V_WMMA_F32_16X16X16_BF16, VOP3POp.V_WMMA_F16_16X16X16_F16,
                       VOP3POp.V_WMMA_BF16_16X16X16_BF16, VOP3POp.V_WMMA_I32_16X16X16_IU8, VOP3POp.V_WMMA_I32_16X16X16_IU4))

def step_wave(program: Program, st: WaveState, lds: bytearray, n_lanes: int) -> int:
  inst = program.get(st.pc)
  if inst is None: return 1
  inst_words, st.literal, inst_type = inst._words, inst._literal or 0, type(inst)
  if (handler := SCALAR.get(inst_type)) is not None:
    delta = handler(st, inst)
    if delta == -1: return -1
    if delta == -2: st.pc += inst_words; return -2
    if delta == -3:  # S_GETPC_B64
      sop1 = inst if isinstance(inst, SOP1) else None
      assert sop1 is not None
      next_pc = (st.pc + inst_words) * 4; st.wsgpr(sop1.sdst, next_pc & 0xffffffff); st.wsgpr(sop1.sdst + 1, (next_pc >> 32) & 0xffffffff); st.pc += inst_words; return 0
    if delta == -4:  # S_SETPC_B64
      sop1 = inst if isinstance(inst, SOP1) else None
      assert sop1 is not None
      st.pc = st.rsrc64(sop1.ssrc0, 0) // 4; return 0
    if delta == -5:  # S_SWAPPC_B64
      sop1 = inst if isinstance(inst, SOP1) else None
      assert sop1 is not None
      next_pc = (st.pc + inst_words) * 4; st.wsgpr(sop1.sdst, next_pc & 0xffffffff); st.wsgpr(sop1.sdst + 1, (next_pc >> 32) & 0xffffffff); st.pc = st.rsrc64(sop1.ssrc0, 0) // 4; return 0
    st.pc += inst_words + delta
  else:
    vec_handler, exec_mask = VECTOR[inst_type], st.exec_mask
    if inst_type is DS:
      for lane in range(n_lanes):
        if exec_mask & (1 << lane): vec_handler(st, inst, lane, lds)
    elif inst_type is VOP3P:
      vop3p = inst if isinstance(inst, VOP3P) else None
      assert vop3p is not None
      if vop3p.op in _WMMA_OPS:
        exec_wmma_f32_16x16x16_f16(st, vop3p, n_lanes)
      else:
        for lane in range(n_lanes):
          if exec_mask & (1 << lane): vec_handler(st, vop3p, lane)
    else:
      for lane in range(n_lanes):
        if exec_mask & (1 << lane): vec_handler(st, inst, lane)
    st.commit_pends(); st.pc += inst_words
  return 0

def exec_wave(program: Program, st: WaveState, lds: bytearray, n_lanes: int, wg_id: tuple[int,int,int]=(0,0,0), local_size: tuple[int,int,int]=(1,1,1), wave_start: int=0) -> int:
  while st.pc in program:
    result = step_wave(program, st, lds, n_lanes)
    if result == -1: return 0
    if result == -2: return -2
  return 0

def exec_workgroup(program: Program, workgroup_id: tuple[int, int, int], local_size: tuple[int, int, int], args_ptr: int, dispatch_dim: int) -> None:
  lx, ly, lz = local_size
  total_threads, lds = lx * ly * lz, bytearray(65536)
  waves: list[tuple[WaveState, int, int]] = []
  for wave_start in range(0, total_threads, WAVE_SIZE):
    n_lanes, st = min(WAVE_SIZE, total_threads - wave_start), WaveState()
    st.exec_mask = (1 << n_lanes) - 1
    st.wsgpr64(0, args_ptr)
    gx, gy, gz = workgroup_id
    if dispatch_dim >= 3: st.sgpr[13], st.sgpr[14], st.sgpr[15] = gx, gy, gz
    elif dispatch_dim == 2: st.sgpr[14], st.sgpr[15] = gx, gy
    else: st.sgpr[15] = gx
    for i in range(n_lanes):
      tid = wave_start + i
      st.vgpr[i][0] = tid if local_size == (lx, 1, 1) else ((tid // (lx * ly)) << 20) | (((tid // lx) % ly) << 10) | (tid % lx)
    waves.append((st, n_lanes, wave_start))
  has_barrier = any(isinstance(inst, SOPP) and inst.op == SOPPOp.S_BARRIER for inst in program.values())
  for _ in range(2 if has_barrier else 1):
    for st, n_lanes, wave_start in waves: exec_wave(program, st, lds, n_lanes, workgroup_id, local_size, wave_start)

def run_asm(lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int) -> int:
  data = (ctypes.c_char * lib_sz).from_address(lib).raw
  program = decode_program(data)
  if not program: return -1
  dispatch_dim = 3 if gz > 1 else (2 if gy > 1 else 1)
  for gidz in range(gz):
    for gidy in range(gy):
      for gidx in range(gx): exec_workgroup(program, (gidx, gidy, gidz), (lx, ly, lz), args_ptr, dispatch_dim)
  return 0
