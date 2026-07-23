# RDNA3 emulator v2 - compiles pcode to UOps executed via tinygrad CPU backend
# Each instruction is compiled to a kernel that operates on buffers:
#   arg=0: sgpr - sgpr[0-127], inline constants[128-255], PC_LO=256, PC_HI=257, SCC=258, SCRATCH_STRIDE=259
#   arg=1: vgpr - vgpr[reg * 32 + lane]
#   arg=2: vmem - base address 0, INDEX offsets directly to host memory
#   arg=3: lds - local data share
#   arg=4: scratch - per-lane scratch memory
from __future__ import annotations
import ctypes, functools, re, platform, subprocess, tempfile
from typing import Callable

# Set/restore DAZ+FTZ (denormals-are-zero + flush-to-zero) to match RDNA3 default float mode
# x86: MXCSR bits DAZ(6)+FTZ(15), ARM64: FPCR bit FZ(24)
# Only applied during emulator execution, restored afterward to avoid breaking hypothesis tests
@functools.cache
def _get_ftz_lib():
  machine = platform.machine()
  if machine in ('x86_64', 'AMD64'):
    src = b'''
unsigned int get_fpcr(void){unsigned int m;__asm__ __volatile__("stmxcsr %0":"=m"(m));return m;}
void set_fpcr(unsigned int m){__asm__ __volatile__("ldmxcsr %0"::"m"(m));}
'''
    ftz_bits = 0x8040  # DAZ (bit 6) + FTZ (bit 15)
  elif machine in ('arm64', 'aarch64'):
    src = b'''
unsigned int get_fpcr(void){unsigned long long v;__asm__ __volatile__("mrs %0,fpcr":"=r"(v));return(unsigned int)v;}
void set_fpcr(unsigned int m){unsigned long long v=m;__asm__ __volatile__("msr fpcr,%0"::"r"(v));}
'''
    ftz_bits = 1 << 24  # FZ (bit 24)
  else: return None, 0
  try:
    with tempfile.NamedTemporaryFile(suffix='.so', delete=False) as f:
      subprocess.check_output(['clang', '-shared', '-O2', '-x', 'c', '-', '-o', f.name], input=src)
      lib = ctypes.CDLL(f.name)
      lib.get_fpcr.restype = ctypes.c_uint32
      lib.set_fpcr.argtypes = [ctypes.c_uint32]
      return lib, ftz_bits
  except Exception: return None, 0

class _MXCSRContext:
  """Context manager to set DAZ+FTZ during emulator execution and restore afterward."""
  __slots__ = ('_saved',)
  def __enter__(self):
    lib, ftz_bits = _get_ftz_lib()
    if lib is None: return self
    self._saved = lib.get_fpcr()
    lib.set_fpcr(self._saved | ftz_bits)
    return self
  def __exit__(self, *args):
    lib, _ = _get_ftz_lib()
    if lib is None or not hasattr(self, '_saved'): return
    lib.set_fpcr(self._saved)

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.device import Buffer, BufferSpec, Device
from tinygrad.runtime.autogen import hsa
from tinygrad.helpers import Context, DEBUG, PROFILE, colored
from tinygrad.engine.realize import get_runtime
from tinygrad.codegen import to_program

from tinygrad.renderer.amd import decode_inst
from tinygrad.runtime.autogen.amd.rdna3.str_pcode import PCODE as PCODE_RDNA3
from tinygrad.runtime.autogen.amd.rdna4.str_pcode import PCODE as PCODE_RDNA4
from tinygrad.runtime.autogen.amd.cdna.str_pcode import PCODE as PCODE_CDNA
from tinygrad.runtime.autogen.amd.rdna3 import ins as ir3
from tinygrad.runtime.autogen.amd.rdna4 import ins as ir4
from tinygrad.runtime.autogen.amd.cdna import ins as irc
from tinygrad.renderer.amd.dsl import VCC_LO, EXEC_LO, SCC, ttmp
from tinygrad.runtime.autogen.amd.common import Fmt, OpType
from test.amd.helpers import decode_dpp16
from test.mockgpu.amd.pcode import parse_block, _FUNCS, _set_bits, _to_bool, _val_to_bits

MASK32 = 0xFFFFFFFF

# ═══════════════════════════════════════════════════════════════════════════════
# SQTT TRACE COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

# Global trace storage: populated by run_asm as raw SQTT blobs, consumed by amdgpu.py
sqtt_traces: list[bytes] = []

# Encoder primitives
from tinygrad.renderer.amd.sqtt import _build_decode_tables, PACKET_TYPES_RDNA3, LAYOUT_HEADER, WAVESTART, WAVEEND, INST, IMMEDIATE, VALUINST, InstOp

_NIB_COUNTS: dict = {cls: nc for _, (cls, nc, *_) in _build_decode_tables(PACKET_TYPES_RDNA3)[0].items()}

def _encode_raw(pkt_cls, **kwargs) -> tuple[int, int]:
  raw = pkt_cls.encoding.default
  for k, v in kwargs.items(): raw = pkt_cls.__dict__[k].set(raw, v)
  return raw, _NIB_COUNTS[pkt_cls]

def _emit_nibbles(nibbles: list[int], pkt_cls, **kwargs):
  raw, nc = _encode_raw(pkt_cls, **kwargs)
  for i in range(nc): nibbles.append((raw >> (i * 4)) & 0xF)

def _nibbles_to_bytes(nibbles: list[int]) -> bytes:
  result = bytearray()
  for i in range(0, len(nibbles), 2): result.append(nibbles[i] | ((nibbles[i + 1] if i + 1 < len(nibbles) else 0) << 4))
  return bytes(result)

def _init_sqtt_encoder():
  """Initialize and return SQTT encoder state. Called once per dispatch with tracing enabled."""
  from tinygrad.runtime.autogen.amd.rdna3.enum import SOPPOp as SOPPOp3
  from tinygrad.runtime.autogen.amd.rdna4.enum import SOPPOp as SOPPOp4
  import re

  _SOPP = (ir3.SOPP, ir4.SOPP, irc.SOPP)
  _SMEM = (ir3.SMEM, ir4.SMEM, irc.SMEM)
  _VALU = (ir3.VOP1, ir3.VOP2, ir3.VOP3, ir3.VOP3P, ir3.VOPC, ir3.VOPD, ir3.VOP3SD, ir3.VOP3_SDST, ir3.VOP1_SDST,
           ir4.VOP1, ir4.VOP2, ir4.VOP3, ir4.VOP3P, ir4.VOPC, ir4.VOPD, ir4.VOP3SD, ir4.VOP3_SDST, ir4.VOP1_SDST,
           irc.VOP1, irc.VOP2, irc.VOP3, irc.VOP3P, irc.VOPC, irc.VOP3SD, irc.VOP3_SDST)
  _DS = (ir3.DS, ir4.DS, irc.DS)
  _GLOBAL = (ir3.GLOBAL, ir4.VGLOBAL, irc.GLOBAL)
  _FLAT = (ir3.FLAT, ir4.VFLAT, irc.FLAT)
  _SCRATCH = (ir3.SCRATCH, ir4.VSCRATCH, irc.SCRATCH)

  # SOPP classification sets
  _SOPP_SKIP = {SOPPOp3.S_ENDPGM.value, SOPPOp3.S_ENDPGM_SAVED.value, SOPPOp3.S_ENDPGM_ORDERED_PS_DONE.value,
                SOPPOp3.S_DELAY_ALU.value}
  _SOPP_IMMEDIATE = {SOPPOp3.S_NOP.value, SOPPOp3.S_CLAUSE.value, SOPPOp3.S_WAITCNT.value, SOPPOp3.S_WAITCNT_DEPCTR.value,
                     SOPPOp3.S_WAIT_IDLE.value, SOPPOp3.S_WAIT_EVENT.value, SOPPOp3.S_SLEEP.value,
                     SOPPOp3.S_SET_INST_PREFETCH_DISTANCE.value}
  for _op in (SOPPOp4.S_WAIT_ALU, SOPPOp4.S_WAIT_LOADCNT, SOPPOp4.S_WAIT_STORECNT, SOPPOp4.S_WAIT_SAMPLECNT,
              SOPPOp4.S_WAIT_BVHCNT, SOPPOp4.S_WAIT_EXPCNT, SOPPOp4.S_WAIT_DSCNT, SOPPOp4.S_WAIT_KMCNT,
              SOPPOp4.S_WAIT_LOADCNT_DSCNT, SOPPOp4.S_WAIT_STORECNT_DSCNT):
    _SOPP_IMMEDIATE.add(_op.value)
  _SOPP_BARRIER = {SOPPOp3.S_BARRIER.value}
  if hasattr(SOPPOp4, 'S_BARRIER_WAIT'): _SOPP_BARRIER.add(SOPPOp4.S_BARRIER_WAIT.value)
  if hasattr(SOPPOp4, 'S_BARRIER_LEAVE'): _SOPP_BARRIER.add(SOPPOp4.S_BARRIER_LEAVE.value)
  _SOPP_BRANCH = {SOPPOp3.S_BRANCH.value, SOPPOp3.S_CBRANCH_SCC0.value, SOPPOp3.S_CBRANCH_SCC1.value,
                  SOPPOp3.S_CBRANCH_VCCZ.value, SOPPOp3.S_CBRANCH_VCCNZ.value,
                  SOPPOp3.S_CBRANCH_EXECZ.value, SOPPOp3.S_CBRANCH_EXECNZ.value}

  # VALU sub-classification patterns
  _VALUT_4_RE = re.compile(r'V_(EXP|LOG|RCP|RSQ|SQRT|SIN|COS|CEIL|FLOOR|TRUNC|RNDNE|FRACT|FREXP)_')
  _VALUB_2_RE = re.compile(r'V_(LSHLREV|LSHRREV|ASHRREV)_(B|I)64')
  _VALUB_4_RE = re.compile(r'V_MAD_(U|I)64')
  _VALUB_16_RE = re.compile(r'V_\w+_F64')

  def _valu_op(op_name: str) -> InstOp|None:
    if 'CMPX' in op_name: return InstOp.VALU1_WR_EXEC
    if _VALUB_2_RE.search(op_name): return InstOp.VALUB_2
    if _VALUB_4_RE.search(op_name): return InstOp.VALUB_4
    if _VALUB_16_RE.search(op_name): return InstOp.VALUB_16
    if _VALUT_4_RE.search(op_name): return InstOp.VALUT_4
    return None

  def _mem_op(t, op_name: str) -> InstOp:
    is_store = "STORE" in op_name
    if issubclass(t, _DS): return InstOp.LDS_WR_2 if is_store else InstOp.LDS_RD
    if issubclass(t, _GLOBAL): return InstOp.SGMEM_WR_2 if is_store else InstOp.SGMEM_RD_1
    if issubclass(t, _FLAT): return InstOp.FLAT_WR_3 if is_store else InstOp.FLAT_RD_2
    if issubclass(t, _SCRATCH): return InstOp.FLAT_WR_3 if is_store else InstOp.FLAT_RD_2
    return InstOp.SALU

  nibbles: list[int] = []
  started: set[int] = set()
  _emit_nibbles(nibbles, LAYOUT_HEADER, layout=3, sel_a=6)

  def emit(wave_id: int, inst, branch_taken: bool|None):
    """Emit an SQTT packet for one executed instruction."""
    w = wave_id & 0x1F
    if wave_id not in started:
      _emit_nibbles(nibbles, WAVESTART, delta=1, simd=0, wgp=0, wave=w, id7=wave_id)
      started.add(wave_id)
    inst_type, inst_op, op_name = type(inst), inst.op.value if hasattr(inst, 'op') else 0, inst.op.name if hasattr(inst, 'op') else ""
    if issubclass(inst_type, _SOPP):
      if inst_op in _SOPP_SKIP: return
      elif inst_op in _SOPP_IMMEDIATE: _emit_nibbles(nibbles, IMMEDIATE, delta=1, wave=w)
      elif inst_op in _SOPP_BARRIER: _emit_nibbles(nibbles, INST, delta=1, wave=w, op=InstOp.BARRIER)
      elif inst_op in _SOPP_BRANCH:
        _emit_nibbles(nibbles, INST, delta=1, wave=w, op=InstOp.JUMP if branch_taken else InstOp.JUMP_NO)
      else: _emit_nibbles(nibbles, INST, delta=1, wave=w, op=InstOp.SALU)
    elif issubclass(inst_type, _VALU):
      op = _valu_op(op_name)
      if op is None: _emit_nibbles(nibbles, VALUINST, delta=1, wave=w)
      else: _emit_nibbles(nibbles, INST, delta=1, wave=w, op=op)
    elif issubclass(inst_type, _SMEM): _emit_nibbles(nibbles, INST, delta=1, wave=w, op=InstOp.SMEM_RD)
    else: _emit_nibbles(nibbles, INST, delta=1, wave=w, op=_mem_op(inst_type, op_name))

  def finish(wave_id: int):
    """Emit WAVEEND for a completed wave."""
    if wave_id in started: _emit_nibbles(nibbles, WAVEEND, delta=1, simd=0, wgp=0, wave=wave_id & 0x1F)

  def finalize() -> bytes:
    """Pad and return the encoded SQTT blob."""
    while len(nibbles) % 2 != 0: nibbles.append(0)
    nibbles.extend([0] * 32)
    while len(nibbles) % 64 != 0: nibbles.append(0)
    return _nibbles_to_bytes(nibbles)

  return emit, finish, finalize

def _c(val, dtype=dtypes.uint32): return UOp.const(dtype, val)

def _u64(lo: UOp, hi: UOp) -> UOp:
  """Combine two 32-bit UOps into a 64-bit UOp."""
  return lo.cast(dtypes.uint64) | (hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))

def _split64(val: UOp) -> tuple[UOp, UOp]:
  """Split a 64-bit value into (lo, hi) 32-bit values."""
  v64 = val.bitcast(dtypes.uint64) if val.dtype == dtypes.float64 else val.cast(dtypes.uint64) if val.dtype != dtypes.uint64 else val
  return v64.cast(dtypes.uint32), (v64 >> UOp.const(dtypes.uint64, 32)).cast(dtypes.uint32)

_SRC_MOD_TYPES = {16: (dtypes.uint16, dtypes.half, 0x7FFF), 32: (dtypes.uint32, dtypes.float32, 0x7FFFFFFF),
                  64: (dtypes.uint64, dtypes.float64, 0x7FFFFFFFFFFFFFFF)}
def _apply_src_mods(val: UOp, mod_bit: int, abs_bits: int, neg_bits: int, bits: int = 32) -> UOp:
  """Apply abs/neg modifiers to source value based on bit width (16, 32, or 64)."""
  if not (abs_bits & (1 << mod_bit)) and not (neg_bits & (1 << mod_bit)): return val
  ut, ft, mask = _SRC_MOD_TYPES[bits]
  fv = val.cast(ut).bitcast(ft) if bits == 16 else val.bitcast(ft) if val.dtype == ut else val
  if abs_bits & (1 << mod_bit): fv = (fv.bitcast(ut) & UOp.const(ut, mask)).bitcast(ft)
  if neg_bits & (1 << mod_bit): fv = fv.neg()
  return fv.bitcast(ut).cast(dtypes.uint32) if bits == 16 else fv.bitcast(ut)

# Map VOPD ops to VOP2 ops for pcode lookup (both RDNA3 and RDNA4)
VOPD_TO_VOP2 = {
  ir3.VOPDOp.V_DUAL_FMAC_F32: ir3.VOP2Op.V_FMAC_F32_E32, ir3.VOPDOp.V_DUAL_MUL_F32: ir3.VOP2Op.V_MUL_F32_E32,
  ir3.VOPDOp.V_DUAL_ADD_F32: ir3.VOP2Op.V_ADD_F32_E32, ir3.VOPDOp.V_DUAL_SUB_F32: ir3.VOP2Op.V_SUB_F32_E32,
  ir3.VOPDOp.V_DUAL_SUBREV_F32: ir3.VOP2Op.V_SUBREV_F32_E32, ir3.VOPDOp.V_DUAL_MAX_F32: ir3.VOP2Op.V_MAX_F32_E32,
  ir3.VOPDOp.V_DUAL_MIN_F32: ir3.VOP2Op.V_MIN_F32_E32, ir3.VOPDOp.V_DUAL_ADD_NC_U32: ir3.VOP2Op.V_ADD_NC_U32_E32,
  ir3.VOPDOp.V_DUAL_LSHLREV_B32: ir3.VOP2Op.V_LSHLREV_B32_E32, ir3.VOPDOp.V_DUAL_AND_B32: ir3.VOP2Op.V_AND_B32_E32,
  ir3.VOPDOp.V_DUAL_MOV_B32: ir3.VOP1Op.V_MOV_B32_E32, ir3.VOPDOp.V_DUAL_CNDMASK_B32: ir3.VOP2Op.V_CNDMASK_B32_E32,
  ir3.VOPDOp.V_DUAL_FMAAK_F32: ir3.VOP2Op.V_FMAAK_F32_E32, ir3.VOPDOp.V_DUAL_FMAMK_F32: ir3.VOP2Op.V_FMAMK_F32_E32,
  ir3.VOPDOp.V_DUAL_DOT2ACC_F32_F16: ir3.VOP2Op.V_DOT2ACC_F32_F16_E32,
  # RDNA4 mappings (same VOP1/VOP2 targets, RDNA4 uses _NUM_ suffix for min/max)
  ir4.VOPDOp.V_DUAL_FMAC_F32: ir3.VOP2Op.V_FMAC_F32_E32, ir4.VOPDOp.V_DUAL_MUL_F32: ir3.VOP2Op.V_MUL_F32_E32,
  ir4.VOPDOp.V_DUAL_ADD_F32: ir3.VOP2Op.V_ADD_F32_E32, ir4.VOPDOp.V_DUAL_SUB_F32: ir3.VOP2Op.V_SUB_F32_E32,
  ir4.VOPDOp.V_DUAL_SUBREV_F32: ir3.VOP2Op.V_SUBREV_F32_E32, ir4.VOPDOp.V_DUAL_MAX_NUM_F32: ir3.VOP2Op.V_MAX_F32_E32,
  ir4.VOPDOp.V_DUAL_MIN_NUM_F32: ir3.VOP2Op.V_MIN_F32_E32, ir4.VOPDOp.V_DUAL_ADD_NC_U32: ir3.VOP2Op.V_ADD_NC_U32_E32,
  ir4.VOPDOp.V_DUAL_LSHLREV_B32: ir3.VOP2Op.V_LSHLREV_B32_E32, ir4.VOPDOp.V_DUAL_AND_B32: ir3.VOP2Op.V_AND_B32_E32,
  ir4.VOPDOp.V_DUAL_MOV_B32: ir3.VOP1Op.V_MOV_B32_E32, ir4.VOPDOp.V_DUAL_CNDMASK_B32: ir3.VOP2Op.V_CNDMASK_B32_E32,
  ir4.VOPDOp.V_DUAL_FMAAK_F32: ir3.VOP2Op.V_FMAAK_F32_E32, ir4.VOPDOp.V_DUAL_FMAMK_F32: ir3.VOP2Op.V_FMAMK_F32_E32,
  ir4.VOPDOp.V_DUAL_DOT2ACC_F32_F16: ir3.VOP2Op.V_DOT2ACC_F32_F16_E32,
}
def _wave_size(arch: str) -> int: return 64 if arch.startswith("cdna") else 32
# Special registers stored after inline constants (256-259)
PC_LO_IDX, PC_HI_IDX, SCRATCH_STRIDE_IDX = 256, 257, 259
# SGPR buffer: 0-127 = SGPRs, 128-255 = inline constants, 256-259 = special registers
SGPR_COUNT = 260
# Sentinel PC value for s_endpgm
ENDPGM_PC = 0xFFFFFFFFFFFFFFFF

def _op_name(inst) -> str:
  if hasattr(inst, 'opx'): return f"{inst.opx.name}_{inst.opy.name}"  # VOPD has opx/opy not op
  return inst.op.name if hasattr(inst.op, 'name') else str(inst.op)

def _to_u32(val: UOp) -> UOp:
  if val.dtype == dtypes.uint32: return val
  if val.dtype.itemsize == 4: return val.bitcast(dtypes.uint32)  # same size: bitcast (float32->uint32)
  return val.cast(dtypes.uint32)  # different size: cast (bool, int16, etc)
def _lane_active(exec_mask: UOp, lane: UOp) -> UOp:
  if exec_mask.dtype == dtypes.uint64: return ((exec_mask >> lane.cast(dtypes.uint64)) & UOp.const(dtypes.uint64, 1)).ne(UOp.const(dtypes.uint64, 0))
  return ((exec_mask >> lane.cast(dtypes.uint32)) & _c(1)).ne(_c(0))
def _hi16(v: UOp) -> UOp: return (v >> _c(16)) & _c(0xFFFF)
def _cond(cond, if_true, if_false):
  """Select between values based on condition (works with UOp or bool)."""
  return cond.where(if_true, if_false) if isinstance(cond, UOp) else if_true if cond else if_false
def _cond_hi16(cond, val: UOp) -> UOp: return _cond(cond, _hi16(val), val)
def _apply_opsel(val: UOp, sel_bit: int, opsel: int) -> UOp: return _hi16(val) if opsel & (1 << sel_bit) else val

def _set_lane_bit(old: UOp, lane: UOp, val: UOp, exec_mask: UOp) -> UOp:
  """Set/clear a single bit in a mask based on lane index, respecting exec mask."""
  if old.dtype in (dtypes.uint64, dtypes.int64):
    dt = dtypes.uint64
    mask = UOp.const(dt, 1) << lane.cast(dt)
    new_bit = _to_u32(val).cast(dt) << lane.cast(dt)
    cleared = old.cast(dt) & (mask ^ UOp.const(dt, 0xFFFFFFFFFFFFFFFF))
    return _lane_active(exec_mask, lane).where(cleared | new_bit, old.cast(dt))
  mask = _c(1) << lane.cast(dtypes.uint32)
  new_bit = _to_u32(val) << lane.cast(dtypes.uint32)
  cleared = old & (mask ^ _c(MASK32))
  return _lane_active(exec_mask, lane).where(cleared | new_bit, old)

def _val_to_u32(val: UOp) -> UOp:
  """Convert any value to uint32 for storage (bitcast floats, cast ints)."""
  if val.dtype == dtypes.uint32: return val
  if val.dtype == dtypes.float32: return val.bitcast(dtypes.uint32)
  if val.dtype == dtypes.half: return val.bitcast(dtypes.uint16).cast(dtypes.uint32)
  if val.dtype in (dtypes.uint16, dtypes.int16): return val.cast(dtypes.uint32)
  return val.cast(dtypes.uint32)

_pcode_fixes = {
  'V_DIV_FMAS_F32': ('D0.f32 = 2.0F ** 32 * fma(S0.f32, S1.f32, S2.f32)',
    'D0.f32 = (exponent(S2.f32) > 127) ? (2.0F ** 64 * fma(S0.f32, S1.f32, S2.f32)) : (2.0F ** -64 * fma(S0.f32, S1.f32, S2.f32))'),
  'V_DIV_FMAS_F64': ('D0.f64 = 2.0 ** 64 * fma(S0.f64, S1.f64, S2.f64)',
    'D0.f64 = (exponent(S2.f64) > 1023) ? (2.0 ** 128 * fma(S0.f64, S1.f64, S2.f64)) : (2.0 ** -128 * fma(S0.f64, S1.f64, S2.f64))'),
  'V_DIV_FIXUP_F32': ('D0.f32 = sign_out ? -abs(S0.f32) : abs(S0.f32)',
    'D0.f32 = isNAN(S0.f32) ? (sign_out ? -INF.f32 : +INF.f32) : (sign_out ? -abs(S0.f32) : abs(S0.f32))'),
  'V_DIV_FIXUP_F64': ('D0.f64 = sign_out ? -abs(S0.f64) : abs(S0.f64)',
    'D0.f64 = isNAN(S0.f64) ? (sign_out ? -INF : +INF) : (sign_out ? -abs(S0.f64) : abs(S0.f64))'),
  'V_TRIG_PREOP_F64': ("result = 64'F((1201'B(2.0 / PI)[1200 : 0] << shift.u32) & 1201'0x1fffffffffffff)", "result = trig_preop_result(shift)"),
}

def _get_pcode_dict(op) -> dict:
  """Return the PCODE dictionary for the given opcode based on its architecture."""
  return PCODE_CDNA if 'cdna' in type(op).__module__ else PCODE_RDNA4 if 'rdna4' in type(op).__module__ else PCODE_RDNA3

# Pcode parser
@functools.cache
def get_pcode(op) -> str:
  op_name = op.name
  pcode_dict = _get_pcode_dict(op)
  if op not in pcode_dict and op_name.endswith('_E64'):
    # VOP3 ops ending in _E64 may share pcode with VOP1 _E32 equivalents
    import importlib
    enum_mod = importlib.import_module(type(op).__module__)
    vop1_cls = getattr(enum_mod, 'VOP1Op', None)
    e32_name = op_name.replace('_E64', '_E32')
    if vop1_cls and hasattr(vop1_cls, e32_name): op = vop1_cls[e32_name]
  pcode = pcode_dict[op]
  fix_name = op_name.replace('_E64', '').replace('_E32', '')
  if fix_name in _pcode_fixes: pcode = pcode.replace(*_pcode_fixes[fix_name])
  if 'V_DIV_SCALE' in op_name:
    dt, exp_lim, ldexp_val = ('f32', '23', '64') if 'F32' in op_name else ('f64', '52', '128')
    for old, new in [(f'S2.{dt} / S1.{dt} == DENORM.{dt}', f'divWouldBeDenorm(S2.{dt}, S1.{dt})'), (f"1.0 / 64'F(S1.{dt}) == DENORM.f64", '0'),
                     (f'1.0 / S1.{dt} == DENORM.{dt}', '0'), (f'S1.{dt} == DENORM.{dt}', f'isDENORM(S1.{dt})'),
                     (f'D0.{dt} = NAN.{dt}', f'VCC = 0x1LL;\nD0.{dt} = NAN.{dt}'),
                     (f'elsif isDENORM(S1.{dt}) then\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})', f'elsif 1 == 0 then\nD0.{dt} = S0.{dt}'),
                      (f'elsif exponent(S2.{dt}) <= {exp_lim} then\n// Numerator is tiny\n'
                       f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})',
                       f'elsif exponent(S2.{dt}) <= {exp_lim} then\nVCC = 0x1LL;\n'
                       f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})'),
                      (f'elsif divWouldBeDenorm(S2.{dt}, S1.{dt}) then\nVCC = 0x1LL;\n'
                       f'if S0.{dt} == S2.{dt} then\n// Only scale the numerator\n'
                       f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nendif',
                       f'elsif divWouldBeDenorm(S2.{dt}, S1.{dt}) then\n'
                       f'VCC = 0x1LL;\nD0.{dt} = S0.{dt}'),
                      (f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nendif\nelsif',
                       f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nelse\n'
                       f'D0.{dt} = S0.{dt}\nendif\nelsif')]:
      pcode = pcode.replace(old, new)
    lines = pcode.rstrip().split('\n')
    for i in range(len(lines) - 1, -1, -1):
      if lines[i].strip() == 'endif':
        lines.insert(i, f'else\nD0.{dt} = S0.{dt}')
        break
    pcode = '\n'.join(lines) + f';\nif isDENORM(S1.{dt}) then\nD0.{dt} = NAN.{dt}\nendif'
    pcode = pcode.replace('VCC = 0x0LL', 'VCC.u64[laneId] = 0').replace('VCC = 0x1LL', 'VCC.u64[laneId] = 1')
  return pcode

def parse_pcode(pcode: str, srcs: dict[str, UOp | int] | None = None) -> tuple[dict, list]:
  env: dict = srcs.copy() if srcs else {}
  assigns: list[tuple[str, UOp]] = []
  raw_lines = [l.strip().rstrip(';') for l in pcode.split('\n') if l.strip() and not l.strip().startswith('//')]
  # TODO: pcode.py should tokenize full pcode string instead of line-by-line, then this hack can be removed
  lines: list[str] = []
  for l in raw_lines:
    if lines and re.search(r'(&&|\|\||[&|+\-*/^])\s*$', lines[-1]): lines[-1] = lines[-1] + ' ' + l
    else: lines.append(l)
  _, final, _ = parse_block(lines, 0, env, assigns=assigns)
  sliced = set(d.split('[')[0] for d, _ in assigns if '[' in d)
  for var, val in final.items():
    if var in ['D0', 'S0', 'SCC', 'VCC', 'EXEC', 'PC', 'RETURN_DATA', 'VDATA'] and isinstance(val, UOp):
      if var in sliced and not any(re.match(rf'{var}\.\w+\s*=', l) for l in lines): continue
      for l in lines:
        if (m := re.match(rf'{var}\.(\w+(?:\[\w+\])?)', l)):
          assigns.append((f'{var}.{m.group(1)}', val))
          break
      else: assigns.append((var, val))
  return env, assigns

def _write_64bit(val: UOp, wfn, reg_or_addr, is_mem: bool, *args) -> list[UOp]:
  """Write a 64-bit value as two 32-bit writes. args passed to wfn after reg/addr and lo/hi value."""
  lo, hi = _split64(val)
  incr = 4 if is_mem else 1  # 4 bytes for memory addresses, 1 for register indices
  return [wfn(reg_or_addr, lo, *args), wfn(reg_or_addr + (UOp.const(reg_or_addr.dtype, incr) if isinstance(reg_or_addr, UOp) else incr), hi, *args)]

def _write_val(bits: int, val: UOp, wfn, reg_or_addr, *args, is_mem: bool = False) -> list[UOp]:
  """Write value, splitting 64-bit if needed. bits=64 for 64-bit writes, otherwise 32-bit."""
  return _write_64bit(val, wfn, reg_or_addr, is_mem, *args) if bits == 64 else [wfn(reg_or_addr, _to_u32(val), *args)]

def _mem_store(mem: UOp, addr: UOp, val: UOp, active: UOp, addr_bits: int = 32, data_bits: int = 32) -> list[UOp]:
  """Conditional memory store with sub-word support. Returns list of store UOps."""
  adt = dtypes.uint64 if addr_bits == 64 else dtypes.uint32
  word_addr = addr >> UOp.const(adt, 2)
  idx = mem.index(word_addr.valid(active))
  if data_bits == 32: return [idx.store(active.where(_to_u32(val), idx))]
  # Sub-word store: read-modify-write with mask
  byte_pos = addr.cast(dtypes.uint32) & _c(3)
  byte_shift = byte_pos * _c(8)
  val_u32, size_mask = val.cast(dtypes.uint32), _c(0xFF if data_bits == 8 else 0xFFFF)
  mask = size_mask << byte_shift
  new_word = (idx & (mask ^ _c(0xFFFFFFFF))) | ((val_u32 & size_mask) << byte_shift)
  if data_bits == 8: return [idx.store(active.where(new_word, idx))]
  # 16-bit cross-word case: byte_pos == 3 means value spans two words
  is_cross = byte_pos.eq(_c(3))
  cross_word0 = (idx & _c(0x00FFFFFF)) | ((val_u32 & _c(0xFF)) << _c(24))
  store0 = idx.store(active.where(is_cross.where(cross_word0, new_word), idx))
  next_idx = mem.index((word_addr + UOp.const(adt, 1)).valid(active & is_cross))
  cross_word1 = (next_idx & _c(0xFFFFFF00)) | ((val_u32 >> _c(8)) & _c(0xFF))
  return [store0, next_idx.store((active & is_cross).where(cross_word1, next_idx))]

def _mem_store_bytes(mem: UOp, addr: UOp, val: UOp, active: UOp, data_bits: int = 32) -> list[UOp]:
  """Store to byte-addressable memory (scratch). addr is byte offset, mem is uint8 buffer."""
  stores = []
  val_u32 = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
  for i in range(data_bits // 8):
    byte_val = (val_u32 >> UOp.const(dtypes.uint32, i * 8)) & UOp.const(dtypes.uint32, 0xFF)
    stores.append(mem.index((addr + UOp.const(dtypes.uint64, i)).valid(active)).store(byte_val.cast(dtypes.uint8)))
  return stores

def _collect_data_slices(assigns: list[tuple[str, UOp]], data_prefix: str, pcode_vars: dict | None = None, op_name: str = "") -> dict[int, UOp]:
  """Collect bit slices from assigns into {dword_idx: value} dict."""
  slices = {}
  for dest, val in assigns:
    if dest.startswith(f'{data_prefix}['):
      if (m := re.match(rf'{data_prefix}\[(\d+)\s*:\s*(\d+)\]', dest)):
        hi_bit, low_bit = int(m.group(1)), int(m.group(2))
        dword_idx = low_bit // 32
        # D16 loads preserve bits - use final value from pcode_vars which has hi bits preserved
        if pcode_vars and 'D16' in op_name and dword_idx == 0 and hi_bit < 32:
          slices[0] = _to_u32(pcode_vars.get(data_prefix, val))
        else: slices[dword_idx] = _to_u32(val)
    elif dest.startswith(data_prefix): slices[0] = _to_u32(val)
  return slices

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION COMPILER - converts decoded instruction to UOp SINK
# ═══════════════════════════════════════════════════════════════════════════════

class _Ctx:
  """Context for instruction compilation - holds buffers and helpers."""
  __slots__ = ('inst_size', 'dyn_fields', '_axis_id', 'wave_size', 'vgpr', 'accvgpr')
  sgpr = UOp.param(0, dtypes.uint32, (SGPR_COUNT,))
  vmem = UOp.param(2, dtypes.uint32, (1 << 46,))
  lds = UOp.param(3, dtypes.uint32, (16384,))
  scratch = UOp.param(4, dtypes.uint8, (1 << 30,))
  # Cache PARAM UOps by wave_size so all _Ctx instances with same wave_size share identical UOp references
  _vgpr_cache: dict[int, UOp] = {}
  _accvgpr_cache: dict[int, UOp] = {}

  def __init__(self, inst_size: int, wave_size: int = 32):
    self.inst_size, self._axis_id, self.wave_size = inst_size, 0, wave_size
    self.dyn_fields: list[tuple[int, int]] = []  # (lo, hi) of fields read dynamically
    if wave_size not in _Ctx._vgpr_cache: _Ctx._vgpr_cache[wave_size] = UOp.param(1, dtypes.uint32, (256 * wave_size,))
    self.vgpr = _Ctx._vgpr_cache[wave_size]
    if wave_size == 64:
      if wave_size not in _Ctx._accvgpr_cache: _Ctx._accvgpr_cache[wave_size] = UOp.param(5, dtypes.uint32, (256 * wave_size,))
      self.accvgpr = _Ctx._accvgpr_cache[wave_size]
    else:
      self.accvgpr = self.vgpr

  def range(self, n: int | None = None) -> UOp:
    """Create a lane range UOp with unique axis ID."""
    if n is None: n = self.wave_size
    self._axis_id += 1
    return UOp.range(n, self._axis_id, AxisType.LOOP, dtype=dtypes.int)

  def unroll_lanes(self, get_lane_bit, exec_mask: UOp, apply_exec: bool = True) -> UOp:
    """Combine lane bits into a mask using RANGE+REDUCE (32-bit for RDNA, 64-bit for CDNA)."""
    lane = self.range()
    if self.wave_size <= 32:
      bit = get_lane_bit(lane).cast(dtypes.uint32) << lane.cast(dtypes.uint32)
      result = bit.reduce(lane, arg=Ops.ADD)
    else:
      bit = get_lane_bit(lane).cast(dtypes.uint64) << lane.cast(dtypes.uint64)
      result = bit.reduce(lane, arg=Ops.ADD)
    return result & exec_mask if apply_exec else result

  def inst_word(self, dword_idx: int) -> UOp:
    """Read instruction dword from vmem at PC + dword_idx*4."""
    pc = self.rpc()
    addr = pc if dword_idx == 0 else pc + UOp.const(dtypes.uint64, dword_idx * 4)
    return self.vmem.index(addr >> UOp.const(dtypes.uint64, 2)).load()

  def inst_field(self, field) -> UOp:
    """Extract field bits from instruction encoding. Tracks field for canonical key computation."""
    lo, hi = field.lo, field.hi
    self.dyn_fields.append((lo, hi))
    dword_idx = lo // 32
    lo_in_dword = lo % 32
    hi_in_dword = hi % 32
    word = self.inst_word(dword_idx)
    if lo // 32 == hi // 32:  # Same dword
      mask = (1 << (hi - lo + 1)) - 1
      shifted = word if lo_in_dword == 0 else word >> UOp.const(dtypes.uint32, lo_in_dword)
      return shifted & UOp.const(dtypes.uint32, mask)
    else:  # Spans two dwords
      lo_bits = 32 - lo_in_dword
      lo_mask = (1 << lo_bits) - 1
      hi_mask = (1 << (hi_in_dword + 1)) - 1
      lo_part = (word >> UOp.const(dtypes.uint32, lo_in_dword)) & UOp.const(dtypes.uint32, lo_mask)
      hi_part = self.inst_word(dword_idx + 1) & UOp.const(dtypes.uint32, hi_mask)
      return lo_part | (hi_part << UOp.const(dtypes.uint32, lo_bits))

  def inst_field_signed(self, field) -> UOp:
    """Extract field and sign-extend based on field width."""
    val = self.inst_field(field)
    width = field.hi - field.lo + 1
    sign_bit = 1 << (width - 1)
    return (val.cast(dtypes.int) ^ _c(sign_bit, dtypes.int)) - _c(sign_bit, dtypes.int)

  def canonical_mask(self, inst_bytes: bytes) -> tuple[int, int, int]:
    """Compute canonical (base, mask, size) for cache lookup.
    base = instruction bits with dynamic fields zeroed
    mask = bitmask with 1s for static bits, 0s for dynamic bits
    size = instruction size in bytes"""
    size = self.inst_size
    base = int.from_bytes(inst_bytes[:size], 'little')
    mask = (1 << (size * 8)) - 1  # all 1s initially
    for lo, hi in self.dyn_fields:
      field_mask = ((1 << (hi - lo + 1)) - 1) << lo
      base &= ~field_mask  # zero dynamic bits in base
      mask &= ~field_mask  # zero dynamic bits in mask
    return base, mask, size

  def rexec(self) -> UOp:
    """Read full EXEC mask (32-bit for RDNA, 64-bit for CDNA)."""
    lo = self.rsgpr_dyn(_c(EXEC_LO.offset))
    if self.wave_size <= 32: return lo
    hi = self.rsgpr_dyn(_c(EXEC_LO.offset + 1))
    return _u64(lo, hi)

  # Dynamic register access (takes UOp index instead of int)
  def rsgpr_dyn(self, reg: UOp, valid: UOp | None = None) -> UOp:
    """Read SGPR with dynamic register index."""
    if valid is not None: return self.sgpr.index(reg.valid(valid)).load()
    return self.sgpr.index(reg).load()

  def wsgpr_dyn(self, reg: UOp, val: UOp) -> UOp:
    """Write SGPR with dynamic register index. On RDNA, index 124 = NULL (writes discarded). On CDNA, index 124 = M0 (read/write)."""
    # RDNA: NULL (124) discards writes. CDNA: M0 (124) is writable.
    valid = None if self.wave_size == 64 else reg.ne(_c(124))
    return self.sgpr.index(reg.valid(valid) if valid is not None else reg).store(val.cast(dtypes.uint32))

  def wmask(self, reg: UOp, val: UOp) -> list[UOp]:
    """Write a lane mask (VCC/EXEC). Splits into lo/hi for wave64."""
    if self.wave_size > 32:
      lo, hi = _split64(val)
      return [self.wsgpr_dyn(reg, lo), self.wsgpr_dyn(reg + _c(1), hi)]
    return [self.wsgpr_dyn(reg, val)]

  def rmask(self, reg: UOp) -> UOp:
    """Read a lane mask (VCC/EXEC). Combines lo/hi for wave64."""
    if self.wave_size > 32: return _u64(self.rsgpr_dyn(reg), self.rsgpr_dyn(reg + _c(1)))
    return self.rsgpr_dyn(reg)

  def rvgpr_dyn(self, reg: UOp, lane: UOp, valid: UOp | None = None) -> UOp:
    """Read VGPR with dynamic register index."""
    idx = reg.cast(dtypes.int) * _c(self.wave_size, dtypes.int) + lane.cast(dtypes.int)
    return self.vgpr.index(idx.valid(valid)).load() if valid is not None else self.vgpr.index(idx).load()

  def wvgpr_dyn(self, reg: UOp, lane: UOp, val: UOp, exec_mask: UOp, after: UOp | None = None) -> UOp:
    """Write VGPR with dynamic register index."""
    buf = self.vgpr.after(after) if after is not None else self.vgpr
    offset = reg.cast(dtypes.int) * _c(self.wave_size, dtypes.int) + lane.cast(dtypes.int)
    return buf.index(offset.valid(_lane_active(exec_mask, lane))).store(val.cast(dtypes.uint32))

  def raccvgpr_dyn(self, reg: UOp, lane: UOp, valid: UOp | None = None) -> UOp:
    """Read ACCVGPR with dynamic register index (CDNA only)."""
    idx = reg.cast(dtypes.int) * _c(self.wave_size, dtypes.int) + lane.cast(dtypes.int)
    return self.accvgpr.index(idx.valid(valid)).load() if valid is not None else self.accvgpr.index(idx).load()

  def waccvgpr_dyn(self, reg: UOp, lane: UOp, val: UOp, exec_mask: UOp, after: UOp | None = None) -> UOp:
    """Write ACCVGPR with dynamic register index (CDNA only)."""
    buf = self.accvgpr.after(after) if after is not None else self.accvgpr
    offset = reg.cast(dtypes.int) * _c(self.wave_size, dtypes.int) + lane.cast(dtypes.int)
    return buf.index(offset.valid(_lane_active(exec_mask, lane))).store(val.cast(dtypes.uint32))

  def rsrc_dyn(self, off: UOp, lane: UOp | None, bits: int = 32, literal: UOp | None = None, is_f64: bool = False, do_cast: bool = True) -> UOp:
    """Read source operand with dynamic offset. Handles SGPR/inline constants (<256), VGPR (>=256).
    If lane is None, only scalar access is supported (off must be < 256).
    is_f64: True for F64 operations where 64-bit literals go in high 32 bits."""
    is_float_const = (off >= _c(240)) & (off <= _c(248))
    is_vgpr = off >= _c(256)
    is_sgpr = is_vgpr.ne(True)
    sgpr_lo = self.rsgpr_dyn(off, is_sgpr)

    if lane is not None:
      vgpr_reg = off - _c(256)
      vgpr_lo = self.rvgpr_dyn(vgpr_reg, lane, is_vgpr)
      vgpr_val = _u64(vgpr_lo, self.rvgpr_dyn(vgpr_reg + _c(1), lane, is_vgpr)) if bits == 64 else vgpr_lo

    if bits == 64:
      sgpr_hi = self.rsgpr_dyn(off + _c(1), is_sgpr)
      sgpr_val = _u64(sgpr_lo, sgpr_hi)
      # Integer inline constants: sign-extend 32-bit value from buffer to 64-bit
      # Float constants: cast F32 to F64
      int_inline = sgpr_lo.cast(dtypes.int32).cast(dtypes.int64)
      float_inline = sgpr_lo.bitcast(dtypes.float32).cast(dtypes.float64)
      # compute inline
      inline = is_float_const.where(float_inline.bitcast(dtypes.uint64), int_inline.bitcast(dtypes.uint64))
      # Literal handling: F64 VOP puts literal in high 32 bits; B64/I64/U64 VOP and SOP zero-extend
      if literal is not None:
        lit_val = literal.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32) if is_f64 else literal.cast(dtypes.uint64)
        inline = off.eq(_c(255)).where(lit_val, inline)
      scalar_val = (off < _c(128)).where(sgpr_val, inline)
    else:
      scalar_val = sgpr_lo
      if literal is not None: scalar_val = off.eq(_c(255)).where(literal, scalar_val)
      if bits == 16 and do_cast:  # Float constants: cast F32 to F16
        scalar_val = is_float_const.where(scalar_val.bitcast(dtypes.float32).cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32), scalar_val)

    return is_vgpr.where(vgpr_val, scalar_val) if lane is not None else scalar_val

  def rpc(self) -> UOp:
    """Read PC as 64-bit byte address."""
    # Index at PC_LO, then cast to uint64 ptr and load
    return _u64(self.rsgpr_dyn(_c(PC_LO_IDX)), self.rsgpr_dyn(_c(PC_HI_IDX)))

  def inc_pc(self) -> list[UOp]:
    """Increment PC by instruction size in bytes. Returns [store]."""
    new_pc = self.rpc() + UOp.const(dtypes.uint64, self.inst_size)
    lo, hi = _split64(new_pc)
    return [self.wsgpr_dyn(_c(PC_LO_IDX), lo), self.wsgpr_dyn(_c(PC_HI_IDX), hi)]

  def scalar_stores(self, assigns: list[tuple[str, UOp]], sdst_reg: UOp, sdst_size: int = 1) -> list[UOp]:
    """Generate stores for scalar assigns with dynamic destination register (D0, SCC, EXEC, VCC)."""
    stores: list[UOp] = []
    for dest, val in assigns:
      if dest.startswith('D0'):
        if sdst_size == 2:
          lo, hi = _split64(val)
          stores.extend([self.wsgpr_dyn(sdst_reg, lo), self.wsgpr_dyn(sdst_reg + _c(1), hi)])
        else: stores.append(self.wsgpr_dyn(sdst_reg, _val_to_u32(val)))
      elif dest.startswith('SCC'): stores.append(self.wsgpr_dyn(_c(SCC.offset), _to_u32(val)))
      elif dest.startswith('EXEC'):
        if self.wave_size > 32 and val.dtype in (dtypes.uint64, dtypes.int64):
          lo, hi = _split64(val)
          stores.extend([self.wsgpr_dyn(_c(EXEC_LO.offset), lo), self.wsgpr_dyn(_c(EXEC_LO.offset + 1), hi)])
        else: stores.append(self.wsgpr_dyn(_c(EXEC_LO.offset), _to_u32(val)))
      elif dest.startswith('VCC'): stores.extend(self.wmask(_c(VCC_LO.offset), val))
    return stores

  def compile_sop_pcode(self, op, srcs: dict[str, UOp | int], sdst_reg: UOp, sdst_size: int) -> UOp:
    """Compile a scalar instruction with dynamic destination register."""
    pcode = get_pcode(op)
    srcs.update({'VCC': self.rmask(_c(VCC_LO.offset)), 'EXEC': self.rexec(), 'SCC': self.rsgpr_dyn(_c(SCC.offset)),
                 '_wave_size': self.wave_size})
    if 'D0' not in srcs: srcs['D0'] = self.rsgpr_dyn(sdst_reg)  # D0 is current dest value for read-modify-write ops
    _, assigns = parse_pcode(pcode, srcs)
    return UOp.sink(*self.scalar_stores(assigns, sdst_reg, sdst_size), *self.inc_pc())

  def compile_lane_pcode(self, op, inst) -> UOp:
    """Compile cross-lane ops (READLANE/WRITELANE/PERMLANE) using pcode parser."""
    pcode = get_pcode(op)
    op_name = op.name if hasattr(op, 'name') else str(op)
    src0_off, vdst_off = self.inst_field(type(inst).src0), self.inst_field(type(inst).vdst)
    src0_reg = (src0_off >= _c(256)).where(src0_off - _c(256), _c(0))  # VGPR index or 0
    src1_off = self.inst_field(type(inst).src1) if hasattr(type(inst), 'src1') else None
    src2_off = self.inst_field(type(inst).src2) if hasattr(type(inst), 'src2') else None
    src1_reg = (src1_off >= _c(256)).where(src1_off - _c(256), src1_off) if src1_off is not None else _c(0)
    src2_reg = (src2_off >= _c(256)).where(src2_off - _c(256), src2_off) if src2_off is not None else _c(0)
    exec_val = self.rexec()
    exec_lo = exec_val.cast(dtypes.uint32) if exec_val.dtype == dtypes.uint64 else exec_val
    srcs = {
      'SRC0': src0_reg, 'VDST': vdst_off, 'EXEC_LO': exec_lo, 'EXEC': exec_val if exec_val.dtype == dtypes.uint64 else exec_val.cast(dtypes.uint64),
      '_vgpr': self.vgpr, '_wave_size': self.wave_size, 'SRC1': src1_reg, 'SRC2': src2_reg,
      'S0': self.rsrc_dyn(src0_off, _c(0, dtypes.int)) if 'WRITELANE' in op_name else src0_reg,
      'S1': self.rsrc_dyn(src1_off, _c(0, dtypes.int)) if src1_off is not None else _c(0),
      'S2': self.rsrc_dyn(src2_off, _c(0, dtypes.int)) if src2_off is not None else _c(0),
    }
    _, assigns = parse_pcode(pcode, srcs)
    stores = []
    for dest, val in assigns:
      if dest.startswith('D0'): stores.append(self.wsgpr_dyn(vdst_off, val.cast(dtypes.uint32)))
      elif dest.startswith('VGPR['): stores.append(self.vgpr.index(val[0]).store(val[1].cast(dtypes.uint32)))
    return UOp.sink(*stores, *self.inc_pc())

  def compile_vop_pcode(self, op, srcs: dict[str, UOp | int], lane: UOp, vdst_reg: UOp, exec_mask: UOp,
                        opsel_dst_hi: bool | UOp = False, sdst_reg: int | None = None, clmp: int = 0,
                        src0_off: UOp | None = None) -> UOp:
    """Compile VOP instruction. Returns sink with stores and inc_pc."""
    pcode = get_pcode(op)
    vcc_reg = sdst_reg if sdst_reg is not None else VCC_LO.offset
    if 'VCC' not in srcs: srcs['VCC'] = self.rmask(_c(vcc_reg))
    srcs.update({'EXEC': exec_mask, 'SCC': self.rsgpr_dyn(_c(SCC.offset)), 'laneId': lane, 'VDST': vdst_reg,
                  'ROUND_MODE': _c(0), 'ROUND_TOWARD_ZERO': _c(0), 'ROUND_NEAREST_EVEN': _c(0), '_vgpr': self.vgpr, '_wave_size': self.wave_size,
                  'MAX_FLOAT_F32': UOp.const(dtypes.float32, 3.4028234663852886e38),
                  # CDNA SDWA byte/word select constants (E32 always uses BYTE0/WORD0 defaults)
                  'SDWA_SRC0_SEL': _c(0), 'BYTE0': _c(0), 'BYTE1': _c(1), 'BYTE2': _c(2), 'BYTE3': _c(3),
                  'WORD0': _c(0), 'WORD1': _c(1)})  # rounding mode and SDWA constants
    _, assigns = parse_pcode(pcode, srcs)

    # For integer ops with clamp, compute overflow using wide arithmetic
    # NOTE: MUL_LO ops don't saturate - they always return the low bits
    int_saturate = None
    if clmp and any(p in op.name for p in ('_NC_U', '_MAD_U', '_NC_I', '_MAD_I')):
      is_signed, is_16bit = '_I' in op.name and '_U' not in op.name, '16' in op.name
      if not (is_16bit and is_signed):  # Skip 16-bit signed ops due to codegen issues
        s0, s1, s2 = srcs.get('S0'), srcs.get('S1'), srcs.get('S2')
        if s0 is not None and s1 is not None:
          narrow_dt = dtypes.uint16 if is_16bit else (dtypes.int32 if is_signed else dtypes.uint32)
          wide_dt = dtypes.int32 if is_16bit else dtypes.int64
          narrow_max, narrow_min = (0xFFFF, 0) if is_16bit else ((0x7FFFFFFF, -0x80000000) if is_signed else (0xFFFFFFFF, 0))
          def to_wide(x): return (x.bitcast(narrow_dt) if x.dtype.itemsize == narrow_dt.itemsize else x.cast(narrow_dt)).cast(wide_dt)
          is_sub, is_mad = 'SUB' in op.name, 'MAD' in op.name
          full = (to_wide(s0) * to_wide(s1) + to_wide(s2)) if is_mad and s2 is not None else \
                 (to_wide(s1) - to_wide(s0)) if is_sub and 'SUBREV' in op.name else \
                 (to_wide(s0) - to_wide(s1)) if is_sub else (to_wide(s0) + to_wide(s1))
          int_saturate = full.clamp(narrow_min, narrow_max).cast(narrow_dt)
    # V_SUB_U32 / V_ADD_U32 with clamp: unsigned saturate (SUB underflow->0, ADD overflow->0xFFFFFFFF)
    if clmp and int_saturate is None and any(p in op.name for p in ('_SUB_U32', '_ADD_U32', '_SUB_U16', '_ADD_U16')):
      s0, s1 = srcs.get('S0'), srcs.get('S1')
      if s0 is not None and s1 is not None:
        assert isinstance(s0, UOp) and isinstance(s1, UOp)
        a, b = (s1.cast(dtypes.uint32), s0.cast(dtypes.uint32)) if 'SUBREV' in op.name else (s0.cast(dtypes.uint32), s1.cast(dtypes.uint32))
        if 'SUB' in op.name:
          int_saturate = (a < b).where(_c(0), a - b)  # underflow -> 0
        else:
          raw_sum = a + b
          int_saturate = (raw_sum < a).where(_c(0xFFFFFFFF), raw_sum)  # overflow -> MAX

    raw_stores: list = []
    vcc_val, exec_val = None, None
    for dest, val in assigns:
      # VGPR bit-slice assignment: VGPR[lane][reg][hi:lo] = (vgpr_idx, rhs_val, hi, lo[, cond]) -> read-modify-write
      if dest.startswith('VGPR[') and re.search(r'\[\d+:\d+\]', dest):
        # VGPR bit-slice: (vgpr_idx, rhs_val, hi_bit, lo_bit) - hi/lo are UOp constants
        hi_bit, lo_bit = int(val[2].arg), int(val[3].arg)
        width = hi_bit - lo_bit + 1
        old = self.vgpr.index(val[0]).load()
        new_val = _set_bits(old, _val_to_bits(val[1]), width, lo_bit).cast(dtypes.uint32)
        active = _lane_active(exec_mask, lane)
        if len(val) > 4: active = active & _to_bool(val[4])
        raw_stores.append(('vgpr_direct', self.vgpr.index(val[0].valid(active)).store(new_val)))
        continue
      if 'D0' in dest and '[laneId]' in dest:
        old_vcc = self.rmask(_c(VCC_LO.offset))
        new_vcc = _set_lane_bit(old_vcc, lane, val, exec_mask)
        raw_stores.extend([('vcc', s) for s in self.wmask(_c(VCC_LO.offset), new_vcc)])
      elif dest.startswith('D0'):
        dest_suffix = re.match(r'D0\.(\w+)', dest)
        if dest_suffix is not None:
          target_dt = {'u16': dtypes.uint16, 'i16': dtypes.int16, 'f16': dtypes.half}.get(dest_suffix.group(1))
          if target_dt is not None and val.dtype != target_dt: val = val.cast(target_dt)
        if (slice_match := re.match(r'D0\[(\d+)\s*:\s*(\d+)\]', dest)):
          d0_hi_bit, d0_lo_bit = int(slice_match.group(1)), int(slice_match.group(2))
          if d0_hi_bit != 31 or d0_lo_bit != 0:
            d0_width, slice_mask = d0_hi_bit - d0_lo_bit + 1, (1 << (d0_hi_bit - d0_lo_bit + 1)) - 1
            val_bits = val.bitcast(dtypes.uint16).cast(dtypes.uint32) if val.dtype == dtypes.half else \
                       val.cast(dtypes.uint32) if val.dtype in (dtypes.uint16, dtypes.int16) else \
                       val.cast(dtypes.uint32) & UOp.const(dtypes.uint32, slice_mask)
            raw_stores.append(('vgpr_slice', (d0_lo_bit, d0_width, val_bits)))
            continue
        # For integer ops with clamp, use pre-computed saturated value; for floats, clamp to [0,1]
        if int_saturate is not None: val = int_saturate
        elif clmp and val.dtype in (dtypes.float32, dtypes.half, dtypes.float64):
          clamped = val.maximum(UOp.const(val.dtype, 0.0)).minimum(UOp.const(val.dtype, 1.0))
          val = _FUNCS['isNAN'](val).where(UOp.const(val.dtype, 0.0), clamped)
        if val.dtype in (dtypes.uint64, dtypes.int64, dtypes.float64):
          lo, hi = _split64(val)
          raw_stores.extend([('vgpr', self.wvgpr_dyn(vdst_reg, lane, lo, exec_mask)),
                             ('vgpr', self.wvgpr_dyn(vdst_reg + _c(1), lane, hi, exec_mask))])
        elif val.dtype in (dtypes.half, dtypes.uint16, dtypes.int16):
          result, old_val = _val_to_u32(val), self.rvgpr_dyn(vdst_reg, lane)
          hi_result = (old_val & UOp.const(dtypes.uint32, 0xFFFF)) | (result << UOp.const(dtypes.uint32, 16))
          # GFX9/CDNA zeroes upper 16 bits on lo-half write; RDNA preserves them
          lo_result = (result & UOp.const(dtypes.uint32, 0xFFFF)) if self.wave_size == 64 else \
                      (old_val & UOp.const(dtypes.uint32, 0xFFFF0000)) | (result & UOp.const(dtypes.uint32, 0xFFFF))
          result = opsel_dst_hi.where(hi_result, lo_result) if isinstance(opsel_dst_hi, UOp) else hi_result if opsel_dst_hi else lo_result
          raw_stores.append(('vgpr', self.wvgpr_dyn(vdst_reg, lane, result, exec_mask)))
        else: raw_stores.append(('vgpr', self.wvgpr_dyn(vdst_reg, lane, _val_to_u32(val), exec_mask)))
      elif dest.startswith('S0') and src0_off is not None:
        # Write back to src0 VGPR (e.g. v_swap_b32). src0_off is raw encoding (256+ = VGPR)
        src0_vgpr = src0_off - _c(256)
        raw_stores.append(('vgpr_s0', self.wvgpr_dyn(src0_vgpr, lane, _val_to_u32(val), exec_mask)))
      elif dest.startswith('VCC'): vcc_val = val
      elif dest.startswith('EXEC'): exec_val = val
      elif dest.startswith('SCC'): raw_stores.append(('scc', self.wsgpr_dyn(_c(SCC.offset), _to_u32(val))))

    lane_stores = [s for t, s in raw_stores if t in ('vgpr', 'vgpr_s0', 'vgpr_direct')]
    stores, scalar_stores = [], [s for t, s in raw_stores if t == 'scc']
    slice_stores = [s for t, s in raw_stores if t == 'vgpr_slice']
    if slice_stores:
      result = self.rvgpr_dyn(vdst_reg, lane)
      for lo_bit, width, val_bits in slice_stores:
        mask = UOp.const(dtypes.uint32, ((1 << width) - 1) << lo_bit)
        result = (result & (mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))) | (val_bits << UOp.const(dtypes.uint32, lo_bit))
      lane_stores.append(self.wvgpr_dyn(vdst_reg, lane, result, exec_mask))
    # VCC/EXEC mask writes must be computed BEFORE VGPR stores to avoid reading modified VGPRs.
    # When vdst overlaps with src operands (e.g. v_add_co_u32 v[0], vcc, s[8], v[0]), the carry
    # computation reads the original source values only if its range loop runs before the VGPR write loop.
    mask_stores: list[UOp] = []
    for mask_val, reg in [(vcc_val, vcc_reg), (exec_val, EXEC_LO.offset)]:
      if mask_val is None: continue
      def get_bit(l, v=mask_val): return (_to_u32(v.substitute({lane: l})) & _c(1)).cast(dtypes.uint32)
      mask_stores.extend(self.wmask(_c(reg), self.unroll_lanes(get_bit, exec_mask, apply_exec=False)))
    stores.extend(mask_stores)
    if lane_stores: stores.append(UOp.sink(*lane_stores).end(lane))
    stores.extend(scalar_stores)
    return UOp.sink(*stores, *self.inc_pc())

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

def _compile_sopp(inst: ir3.SOPP | ir4.SOPP, ctx: _Ctx) -> UOp:
  simm16 = ctx.inst_field_signed(type(inst).simm16).cast(dtypes.int16)
  if inst.op in (ir3.SOPPOp.S_ENDPGM, ir4.SOPPOp.S_ENDPGM, irc.SOPPOp.S_ENDPGM):
    return UOp.sink(ctx.wsgpr_dyn(_c(PC_LO_IDX), UOp.const(dtypes.uint32, 0xFFFFFFFF)),
                          ctx.wsgpr_dyn(_c(PC_HI_IDX), UOp.const(dtypes.uint32, 0xFFFFFFFF)))
  # S_BARRIER: advance PC past the barrier instruction. The execution loop detects barriers before executing and handles synchronization.
  barrier_ops = {ir3.SOPPOp.S_BARRIER, irc.SOPPOp.S_BARRIER}
  if hasattr(ir4.SOPPOp, 'S_BARRIER_WAIT'): barrier_ops.add(ir4.SOPPOp.S_BARRIER_WAIT)
  if inst.op in barrier_ops: return UOp.sink(*ctx.inc_pc())
  # S_NOP and S_WAITCNT are no-ops in emulator (no pipeline/cache to wait on)
  if inst.op in (ir3.SOPPOp.S_NOP, ir4.SOPPOp.S_NOP, irc.SOPPOp.S_NOP, irc.SOPPOp.S_WAITCNT): return UOp.sink(*ctx.inc_pc())
  # NOTE: we ignore SOPPs without PCODE
  if inst.op in _get_pcode_dict(inst.op):
    pcode = get_pcode(inst.op)
    pc_bytes = ctx.rpc()  # PC is already 64-bit byte address
    vcc, exec_val = ctx.rmask(_c(VCC_LO.offset)), ctx.rexec()
    srcs: dict[str, UOp|int] = {'PC': pc_bytes.cast(dtypes.int64), 'SIMM16': simm16, 'SCC': ctx.rsgpr_dyn(_c(SCC.offset)), 'VCC': vcc,
            'VCCZ': vcc.eq(UOp.const(vcc.dtype, 0)).cast(dtypes.uint32),
            'EXECZ': exec_val.eq(UOp.const(exec_val.dtype, 0)).cast(dtypes.uint32)}
    for dest, val in parse_pcode(pcode, srcs)[1]:
      if dest == 'PC' or dest.startswith('PC.'):
        lo, hi = _split64(val.cast(dtypes.uint64))
        return UOp.sink(ctx.wsgpr_dyn(_c(PC_LO_IDX), lo), ctx.wsgpr_dyn(_c(PC_HI_IDX), hi))
  return UOp.sink(*ctx.inc_pc())

def _compile_smem(inst: ir3.SMEM | ir4.SMEM, ctx: _Ctx) -> UOp:
  # Cache invalidation instructions are no-ops in the emulator (we don't model caches)
  if '_INV' in inst.op.name: return UOp.sink(*ctx.inc_pc())
  # Dynamic sbase field (bits 5:0) - SGPR pair, field value * 2 = register offset
  sbase = ctx.inst_field(type(inst).sbase) * _c(2)
  # Dynamic sdata field (bits 12:6) - destination SGPR
  sdata_reg = ctx.inst_field(type(inst).sdata)
  # RDNA4 uses 'ioffset', RDNA3 uses 'offset' - use type(inst) to get correct field
  offset_field = type(inst).ioffset if hasattr(type(inst), 'ioffset') else type(inst).offset  # type: ignore[union-attr]
  offset = ctx.inst_field_signed(offset_field)  # signed immediate
  # Dynamic soffset field - SGPR for additional offset (NULL=124 reads as 0, CDNA soffset_en=0 means no soffset)
  soffset_val = _c(0).cast(dtypes.uint64)
  if not (isinstance(inst, irc.SMEM) and not inst.soffset_en):
    soffset_val = ctx.rsgpr_dyn(ctx.inst_field(type(inst).soffset)).cast(dtypes.uint64)
  addr = _u64(ctx.rsgpr_dyn(sbase), ctx.rsgpr_dyn(sbase + _c(1))) + offset.cast(dtypes.uint64) + soffset_val
  # S_LOAD_(DTYPE) series: B32/DWORD=1, B64/DWORDX2=2, U8=0.25, I8=-0.25, etc.
  op_name = _op_name(inst)
  assert (op_name).startswith('S_LOAD_'), f"unexpected SMEM op: {op_name}"
  part = op_name.rsplit('_', 1)[1]  # B32, DWORD, DWORDX2, U8, I8, etc.
  nval = int(part.removeprefix('DWORD').removeprefix('X') or '1') if 'DWORD' in part else int(part[1:]) / 32 * (-1 if part[0] == 'I' else 1)
  ndwords = max(1, int(abs(nval)))
  dword_base = addr >> UOp.const(dtypes.uint64, 2)
  vals = [ctx.vmem.index(dword_base + UOp.const(dtypes.uint64, i)) for i in range(ndwords)]
  if abs(nval) < 1:
    nbits = int(abs(nval) * 32)
    byte_off = (addr & UOp.const(dtypes.uint64, 3)).cast(dtypes.uint32) * UOp.const(dtypes.uint32, 8)
    extracted = (vals[0] >> byte_off) & UOp.const(dtypes.uint32, (1 << nbits) - 1)
    vals[0] = extracted.cast({8: dtypes.int8, 16: dtypes.int16}[nbits]).cast(dtypes.int32).bitcast(dtypes.uint32) if nval < 0 else extracted
  stores = [ctx.wsgpr_dyn(sdata_reg + _c(i), vals[i]) for i in range(ndwords)]
  return UOp.sink(*stores, *ctx.inc_pc())

def _compile_sop(inst: ir3.SOP1|ir3.SOP2|ir3.SOPC|ir3.SOPK|ir4.SOP1|ir4.SOP2|ir4.SOPC|ir4.SOPK|irc.SOP1|irc.SOP2|irc.SOPC|irc.SOPK, ctx: _Ctx) -> UOp:
  bits = inst.canonical_op_bits
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None  # type: ignore[union-attr]

  if isinstance(inst, (ir3.SOPK, ir4.SOPK, irc.SOPK)):
    sdst_off = ctx.inst_field(type(inst).sdst)
    simm16 = ctx.inst_field(type(inst).simm16)
    # Sign-extend simm16
    simm16_sext = simm16.cast(dtypes.int16).cast(dtypes.int32)
    # RDNA4 pcodes use S0.i16 for the immediate (e.g., S_MULK_I32), RDNA3 uses S0 for the register (e.g., S_CMPK_*)
    # CDNA pcode uses S0 for the immediate in MOVK/MULK/ADDK/CMOVK, but S0 = register for CMPK/SETREG
    op_name = _op_name(inst)
    if isinstance(inst, ir4.SOPK): s0 = simm16
    elif isinstance(inst, irc.SOPK) and 'CMPK' not in op_name and 'SETREG' not in op_name: s0 = simm16_sext
    else: s0 = ctx.rsgpr_dyn(sdst_off)
    srcs: dict[str, UOp|int] = {'S0': s0, 'S1': simm16_sext, 'SIMM16': simm16_sext, 'D0': ctx.rsgpr_dyn(sdst_off)}
    dst_off, dst_size = sdst_off, 1
    # S_GETREG_B32: extract bits from HW register. Handle as special case since HW_REGISTERS is not a normal variable.
    # HW register values are stored at SGPR[SGPR_COUNT-16 + hwRegId] by _init_wave.
    if 'GETREG' in op_name:
      hw_reg_id = simm16.cast(dtypes.uint32) & _c(0x3F)
      offset = (simm16.cast(dtypes.uint32) >> _c(6)) & _c(0x1F)
      size = ((simm16.cast(dtypes.uint32) >> _c(11)) & _c(0x1F)) + _c(1)
      hw_val = ctx.rsgpr_dyn(_c(SGPR_COUNT - 16) + hw_reg_id)
      mask = (_c(1) << size) - _c(1)
      result = (hw_val >> offset) & mask
      return UOp.sink(ctx.wsgpr_dyn(sdst_off, result), *ctx.inc_pc())
  elif isinstance(inst, (ir3.SOP1, ir4.SOP1, irc.SOP1)):
    # S_BARRIER_SIGNAL: no-op in emulator, barrier sync handled by execution loop
    if isinstance(inst, ir4.SOP1) and inst.op in _BARRIER_SOP1_OPS: return UOp.sink(*ctx.inc_pc())
    sdst_off = ctx.inst_field(type(inst).sdst)
    ssrc0_off = ctx.inst_field(type(inst).ssrc0)
    srcs = {'S0': ctx.rsrc_dyn(ssrc0_off, None, bits['s0'], literal)}
    dst_off, dst_size = sdst_off, bits['d'] // 32
  elif isinstance(inst, (ir3.SOP2, ir4.SOP2, irc.SOP2)):
    sdst_off = ctx.inst_field(type(inst).sdst)
    ssrc0_off = ctx.inst_field(type(inst).ssrc0)
    ssrc1_off = ctx.inst_field(type(inst).ssrc1)
    srcs = {'S0': ctx.rsrc_dyn(ssrc0_off, None, bits['s0'], literal),
            'S1': ctx.rsrc_dyn(ssrc1_off, None, bits['s1'], literal)}
    if literal is not None: srcs['SIMM32'] = literal
    dst_off, dst_size = sdst_off, bits['d'] // 32
  elif isinstance(inst, (ir3.SOPC, ir4.SOPC, irc.SOPC)):
    ssrc0_off = ctx.inst_field(type(inst).ssrc0)
    ssrc1_off = ctx.inst_field(type(inst).ssrc1)
    srcs = {'S0': ctx.rsrc_dyn(ssrc0_off, None, bits['s0'], literal),
            'S1': ctx.rsrc_dyn(ssrc1_off, None, bits['s1'], literal)}
    dst_off, dst_size = _c(0), 0  # SOPC writes to SCC, not sdst
  else:
    raise RuntimeError(f"unknown SOP type: {type(inst).__name__}")

  return ctx.compile_sop_pcode(inst.op, srcs, dst_off, dst_size)

def _sdwa_select(val: UOp, sel: UOp, sext: UOp) -> UOp:
  """Apply SDWA byte/word selection and optional sign extension to a 32-bit value."""
  # sel: 0-3=BYTE_0..3, 4=WORD_0, 5=WORD_1, 6=DWORD
  b0 = val & _c(0xFF)
  b1 = (val >> _c(8)) & _c(0xFF)
  b2 = (val >> _c(16)) & _c(0xFF)
  b3 = (val >> _c(24)) & _c(0xFF)
  w0 = val & _c(0xFFFF)
  w1 = (val >> _c(16)) & _c(0xFFFF)
  selected = sel.eq(_c(1)).where(b1, sel.eq(_c(2)).where(b2, sel.eq(_c(3)).where(b3,
    sel.eq(_c(4)).where(w0, sel.eq(_c(5)).where(w1, sel.eq(_c(6)).where(val, b0))))))
  # Sign extend when sext=1
  is_byte = sel < _c(4)
  byte_sext = (selected & _c(0x80)).ne(_c(0)).where(selected | _c(0xFFFFFF00), selected)
  word_sext = (selected & _c(0x8000)).ne(_c(0)).where(selected | _c(0xFFFF0000), selected)
  return sext.ne(_c(0)).where(is_byte.where(byte_sext, word_sext), selected)

def _sdwa_write(old: UOp, val: UOp, dst_sel: UOp, dst_unused: UOp) -> UOp:
  """Apply SDWA destination selection: write selected byte/word, handle unused bits."""
  # dst_unused: 0=PAD(zero), 1=SEXT, 2=PRESERVE
  # dst_sel: 0-3=BYTE, 4=WORD_0, 5=WORD_1, 6=DWORD
  is_byte = dst_sel < _c(4)
  is_word = (dst_sel >= _c(4)) & (dst_sel < _c(6))
  shift = is_byte.where(dst_sel * _c(8), (dst_sel - _c(4)) * _c(16))
  mask = is_byte.where(_c(0xFF), is_word.where(_c(0xFFFF), _c(0xFFFFFFFF)))
  placed = (val & mask) << shift
  preserve_mask = (mask << shift) ^ _c(0xFFFFFFFF)
  preserved = (old & preserve_mask) | placed
  # For PAD and SEXT, unused bits are zero (PAD) or sign-extended (SEXT). For DWORD, just return val.
  return dst_sel.eq(_c(6)).where(val, dst_unused.eq(_c(2)).where(preserved, placed))

def _dpp_quad_sel(quad_lane: UOp, sels: tuple[int, int, int, int]) -> UOp:
  sel = _c(sels[0], dtypes.int)
  for i, src in enumerate(sels[1:], start=1): sel = quad_lane.eq(_c(i, dtypes.int)).where(_c(src, dtypes.int), sel)
  return sel

def _dpp16_ctrl(lane: UOp, dpp: int, row_mask: int, bank_mask: int, wave_size: int) -> tuple[UOp, UOp, UOp]:
  """Return (src_lane, row/bank enabled, in-bounds) for a DPP16 swizzle."""
  lane_i = lane.cast(dtypes.int)
  row_base, lane_in_row = lane_i & _c(~15, dtypes.int), lane_i & _c(15, dtypes.int)
  row = lane_i // _c(16, dtypes.int)
  bank = lane_in_row >> _c(2, dtypes.int)
  enabled = (((_c(row_mask) >> row.cast(dtypes.uint32)) & _c(1)).ne(_c(0)) &
             (((_c(bank_mask) >> bank.cast(dtypes.uint32)) & _c(1)).ne(_c(0))))
  op, arg = decode_dpp16(dpp)
  src_lane, valid = lane_i, UOp.const(dtypes.bool, True)

  if op == 'quad_perm':
    assert isinstance(arg, tuple)
    src_lane = (lane_i & _c(~3, dtypes.int)) + _dpp_quad_sel(lane_i & _c(3, dtypes.int), arg)
  else:
    assert isinstance(arg, int)
    if op == 'row_shl': src_lane, valid = row_base + lane_in_row + _c(arg, dtypes.int), lane_in_row <= _c(15 - arg, dtypes.int)
    elif op == 'row_shr': src_lane, valid = row_base + lane_in_row - _c(arg, dtypes.int), lane_in_row >= _c(arg, dtypes.int)
    elif op == 'row_ror': src_lane = row_base + ((lane_in_row - _c(arg, dtypes.int)) & _c(15, dtypes.int))
    elif op == 'row_mirror': src_lane = row_base + (_c(15, dtypes.int) - lane_in_row)
    elif op == 'row_half_mirror': src_lane = row_base + ((lane_in_row & _c(8, dtypes.int)) | (_c(7, dtypes.int) - (lane_in_row & _c(7, dtypes.int))))
    elif op == 'row_bcast': src_lane = row_base
    elif op == 'wave_shl': src_lane, valid = lane_i + _c(arg, dtypes.int), lane_i < _c(wave_size - arg, dtypes.int)
    elif op == 'wave_rol': src_lane = (lane_i + _c(arg, dtypes.int)) % _c(wave_size, dtypes.int)
    elif op == 'wave_shr': src_lane, valid = lane_i - _c(arg, dtypes.int), lane_i >= _c(arg, dtypes.int)
    elif op == 'wave_ror': src_lane = (lane_i - _c(arg, dtypes.int)) % _c(wave_size, dtypes.int)
    else: raise NotImplementedError(f"DPP16 control {dpp:#x} ({op}:{arg}) not implemented in emulator")
  return src_lane, enabled, valid

def _load_dpp16_src0(ctx: _Ctx, inst, lane: UOp, fallback: UOp) -> UOp:
  """Load a DPP16-swizzled src0 value from vsrc0."""
  src_lane, enabled, valid = _dpp16_ctrl(lane, getattr(inst, 'dpp', 0) or 0, getattr(inst, 'row_mask', 0xf) or 0xf,
                                         getattr(inst, 'bank_mask', 0xf) or 0xf, ctx.wave_size)
  safe_src_lane = (enabled & valid).where(src_lane, _c(0, dtypes.int))
  swizzled = ctx.rvgpr_dyn(ctx.inst_field(type(inst).vsrc0), safe_src_lane)
  invalid = UOp.const(fallback.dtype, 0) if getattr(inst, 'bc', 0) else fallback
  return enabled.where(valid.where(swizzled, invalid), fallback)

def _compile_sdwa(inst: irc.VOP1_SDWA | irc.VOP2_SDWA | irc.VOP2_SDWA_SDST | irc.VOPC_SDWA_SDST, ctx: _Ctx) -> UOp:
  """Compile CDNA SDWA (Sub-Dword Access) VOP1/VOP2/VOPC instructions."""
  is_vopc = isinstance(inst, irc.VOPC_SDWA_SDST)
  exec_mask = ctx.rexec()
  # sd=1 means use sdst register, sd=0 means use VCC (for VOPC_SDWA_SDST and VOP2_SDWA_SDST)
  if isinstance(inst, (irc.VOP2_SDWA_SDST, irc.VOPC_SDWA_SDST)):
    sdst_off = _c(inst.sdst.offset) if getattr(inst, 'sd', False) else _c(VCC_LO.offset)
  else:
    sdst_off = _c(VCC_LO.offset)
  # Read SDWA fields (these are dynamic but shared across lanes)
  src0_sel = ctx.inst_field(type(inst).src0_sel)
  src0_sext = ctx.inst_field(type(inst).src0_sext)
  vsrc0_reg = ctx.inst_field(type(inst).vsrc0)
  pcode = get_pcode(inst.op)
  if isinstance(inst, (irc.VOP2_SDWA, irc.VOP2_SDWA_SDST, irc.VOPC_SDWA_SDST)):
    src1_sel = ctx.inst_field(type(inst).src1_sel)
    src1_sext = ctx.inst_field(type(inst).src1_sext)
    vsrc1_reg = ctx.inst_field(type(inst).vsrc1)

  # For VOPC: use unroll_lanes to build the bitmask from scratch (no read-modify-write on stale data)
  if is_vopc:
    def get_cmp_bit(lane) -> UOp:
      lc = lane.cast(dtypes.int) if isinstance(lane, UOp) else _c(lane, dtypes.int)
      s0_raw = ctx.rsgpr_dyn(vsrc0_reg) if inst.s0 else ctx.rvgpr_dyn(vsrc0_reg, lc)
      s0 = _sdwa_select(s0_raw, src0_sel, src0_sext)
      s1_raw = ctx.rsgpr_dyn(vsrc1_reg) if inst.s1 else ctx.rvgpr_dyn(vsrc1_reg, lc)
      s1 = _sdwa_select(s1_raw, src1_sel, src1_sext)
      srcs = {'S0': s0, 'S1': s1, 'laneId': lc}
      for dest, val in parse_pcode(pcode, srcs)[1]:
        if '[laneId]' in dest and ('D0' in dest or 'EXEC' in dest): return val.cast(dtypes.uint32)
      return _c(0)
    new_result = ctx.unroll_lanes(get_cmp_bit, exec_mask, apply_exec=False) & exec_mask
    stores = ctx.wmask(sdst_off, new_result)
    return UOp.sink(*stores, *ctx.inc_pc())

  # Non-VOPC path: VOP1_SDWA, VOP2_SDWA, VOP2_SDWA_SDST — uses lane loop
  lane = ctx.range()
  vdst_reg = ctx.inst_field(type(inst).vdst)  # type: ignore[union-attr]
  s0_raw = ctx.rsgpr_dyn(vsrc0_reg) if inst.s0 else ctx.rvgpr_dyn(vsrc0_reg, lane)
  s0 = _sdwa_select(s0_raw, src0_sel, src0_sext)
  if isinstance(inst, (irc.VOP2_SDWA, irc.VOP2_SDWA_SDST)):
    s1_raw = ctx.rsgpr_dyn(vsrc1_reg) if inst.s1 else ctx.rvgpr_dyn(vsrc1_reg, lane)
    s1 = _sdwa_select(s1_raw, src1_sel, src1_sext)
    srcs:dict[str, UOp | int] = {'S0': s0, 'S1': s1, 'D0': ctx.rvgpr_dyn(vdst_reg, lane)}
  else:
    srcs = {'S0': s0}
  # dst_sel and dst_unused
  has_dst_sel = hasattr(type(inst), 'dst_sel')
  if has_dst_sel:
    dst_sel = ctx.inst_field(type(inst).dst_sel)  # type: ignore[union-attr]
    dst_unused = ctx.inst_field(type(inst).dst_unused)  # type: ignore[union-attr]
  srcs.update({'VCC': ctx.rmask(_c(VCC_LO.offset)), 'EXEC': exec_mask, 'SCC': ctx.rsgpr_dyn(_c(SCC.offset)),
               'laneId': lane, 'VDST': vdst_reg, 'ROUND_MODE': _c(0), 'ROUND_TOWARD_ZERO': _c(0),
               'ROUND_NEAREST_EVEN': _c(0), '_vgpr': ctx.vgpr, '_wave_size': ctx.wave_size,
               'SDWA_SRC0_SEL': _c(0), 'BYTE0': _c(0), 'BYTE1': _c(1), 'BYTE2': _c(2), 'BYTE3': _c(3),
               'WORD0': _c(0), 'WORD1': _c(1)})
  _, assigns = parse_pcode(pcode, srcs)
  stores = []
  vcc_val = None
  for dest, val in assigns:
    if 'D0' in dest and '[laneId]' in dest:
      vcc_val = val
    elif dest.startswith('D0'):
      result = _val_to_u32(val)
      if has_dst_sel:
        old = ctx.rvgpr_dyn(vdst_reg, lane)
        result = _sdwa_write(old, result, dst_sel, dst_unused)
      stores.append(ctx.wvgpr_dyn(vdst_reg, lane, result, exec_mask))
    elif dest.startswith('VCC'):
      old_vcc = ctx.rmask(_c(VCC_LO.offset))
      stores.extend(ctx.wmask(_c(VCC_LO.offset), _set_lane_bit(old_vcc, lane, val, exec_mask)))
  if vcc_val is not None:
    # Initialize sdst to 0 before lane loop (old value may be unrelated data), then set lane bits in loop
    init_stores = [ctx.wsgpr_dyn(sdst_off, _c(0)), ctx.wsgpr_dyn(sdst_off + _c(1), _c(0))]
    old_sdst = ctx.rmask(sdst_off)
    stores.extend(ctx.wmask(sdst_off, _set_lane_bit(old_sdst, lane, vcc_val, exec_mask)))
    if stores:
      return UOp.sink(*init_stores, UOp.sink(*stores).end(lane), *ctx.inc_pc())
    return UOp.sink(*init_stores, *ctx.inc_pc())
  if stores:
    return UOp.sink(UOp.sink(*stores).end(lane), *ctx.inc_pc())
  return UOp.sink(*ctx.inc_pc())

def _compile_vop12(inst: ir3.VOP1 | ir3.VOP1_SDST | ir3.VOP1_DPP16 | ir3.VOP2 | ir3.VOP2_DPP16 |
                   ir4.VOP1 | ir4.VOP1_SDST | ir4.VOP1_DPP16 | ir4.VOP2 | ir4.VOP2_DPP16 |
                   irc.VOP1 | irc.VOP1_DPP16 | irc.VOP2 | irc.VOP2_DPP16, ctx: _Ctx) -> UOp:
  op_name = _op_name(inst)
  if op_name in ('V_READFIRSTLANE_B32_E32', 'V_PERMLANE64_B32_E32'): return ctx.compile_lane_pcode(inst.op, inst)
  # v_accvgpr_mov_b32: ACCVGPR[vdst] = ACCVGPR[src0] (VOP1 encoding, no pcode)
  if 'ACCVGPR_MOV' in op_name:
    lane, exec_mask = ctx.range(), ctx.rexec()
    vdst_reg = ctx.inst_field(type(inst).vdst)  # VGPRField: raw ACCVGPR index (0-255)
    acc_src0_off = ctx.inst_field(type(inst).src0)  # SrcField: raw 256 + ACCVGPR index
    val = ctx.raccvgpr_dyn(acc_src0_off - _c(256), lane)
    return UOp.sink(ctx.waccvgpr_dyn(vdst_reg, lane, val, exec_mask).end(lane), *ctx.inc_pc())
  lane, exec_mask, bits = ctx.range(), ctx.rexec(), inst.canonical_op_bits
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None  # type: ignore[union-attr]
  is_f64 = 'F64' in op_name and 'B64' not in op_name
  is_float = any(x in op_name for x in ('F16', 'F32', 'F64'))
  is_dpp16 = hasattr(type(inst), 'dpp') and hasattr(type(inst), 'vsrc0')
  vdst_reg = ctx.inst_field(type(inst).vdst)
  write_hi_half = bits['d'] == 16 and (vdst_reg >= _c(128))
  if isinstance(write_hi_half, UOp): vdst_reg = write_hi_half.where(vdst_reg - _c(128), vdst_reg)
  elif write_hi_half: vdst_reg -= 128
  src0_off: UOp | None = None
  if isinstance(inst, (ir3.VOP1, ir4.VOP1, irc.VOP1)):
    # Handle VOP1 hi-half source operand (src0 >= v[128] for 16-bit ops)
    d0 = _cond_hi16(write_hi_half, ctx.rvgpr_dyn(vdst_reg, lane))
    if is_dpp16:
      s0 = _load_dpp16_src0(ctx, inst, lane, d0)
    else:
      src0_off = ctx.inst_field(type(inst).src0)
      s0 = ctx.rsrc_dyn(src0_off, lane, bits['s0'], literal, is_f64)
    if bits['s0'] == 16 and not is_dpp16:
      src0_hi = src0_off >= _c(384)
      # Only compute hi-half when src0_off >= 384, use guarded index to prevent OOB access
      src0_reg = src0_hi.where(src0_off - _c(384), _c(0))
      s0 = src0_hi.where(_hi16(ctx.rvgpr_dyn(src0_reg, lane)), s0)
    if is_dpp16 and is_float:
      s0 = _apply_src_mods(s0, 0, 1 if getattr(inst, 'src0_abs', 0) else 0, 1 if getattr(inst, 'src0_neg', 0) else 0, bits['s0'])
    srcs:dict[str, UOp | int] = {'S0': s0, 'D0': d0}
  else:
    vsrc1_reg = ctx.inst_field(type(inst).vsrc1)
    vsrc1_hi = bits['s0'] == 16 and (vsrc1_reg >= _c(128))
    vsrc1_actual = _cond(vsrc1_hi, vsrc1_reg - _c(128), vsrc1_reg)
    if bits['s1'] == 64:
      s1 = _u64(ctx.rvgpr_dyn(vsrc1_reg, lane), ctx.rvgpr_dyn(vsrc1_reg + _c(1), lane))
      d0 = _u64(ctx.rvgpr_dyn(vdst_reg, lane), ctx.rvgpr_dyn(vdst_reg + _c(1), lane))
    else:
      s1 = _cond_hi16(vsrc1_hi, ctx.rvgpr_dyn(vsrc1_actual, lane))
      d0 = _cond_hi16(write_hi_half, ctx.rvgpr_dyn(vdst_reg, lane))  # FMAC/FMAMK hi-half dest needs hi-half accumulator
    # Handle VOP2 hi-half src0 operand (src0 >= v[128] for 16-bit ops)
    if is_dpp16:
      s0 = _load_dpp16_src0(ctx, inst, lane, d0)
    else:
      src0_off = ctx.inst_field(type(inst).src0)
      s0 = ctx.rsrc_dyn(src0_off, lane, bits['s0'], literal, is_f64)
    if bits['s0'] == 16 and not is_dpp16:
      src0_hi = src0_off >= _c(384)
      # Only compute hi-half when src0_off >= 384, use guarded index to prevent OOB access
      src0_reg = src0_hi.where(src0_off - _c(384), _c(0))
      s0 = src0_hi.where(_hi16(ctx.rvgpr_dyn(src0_reg, lane)), s0)
    if is_dpp16 and is_float:
      s0 = _apply_src_mods(s0, 0, 1 if getattr(inst, 'src0_abs', 0) else 0, 1 if getattr(inst, 'src0_neg', 0) else 0, bits['s0'])
      s1 = _apply_src_mods(s1, 0, 1 if getattr(inst, 'src1_abs', 0) else 0, 1 if getattr(inst, 'src1_neg', 0) else 0, bits['s1'])
    srcs = {'S0': s0, 'S1': s1, 'D0': d0}
    # FMAAK_(DTYPE)_E32 series
    if 'V_FMAA' in _op_name(inst) or 'V_FMAM' in _op_name(inst):
      assert literal is not None
      srcs['SIMM32'] = literal
  return ctx.compile_vop_pcode(inst.op, srcs, lane, vdst_reg, exec_mask, opsel_dst_hi=write_hi_half, src0_off=src0_off)

def _compile_vopc(inst: ir3.VOPC|ir3.VOPC_DPP16|ir3.VOP3|ir4.VOPC|ir4.VOPC_DPP16|ir4.VOP3|irc.VOPC|irc.VOP3, ctx: _Ctx,
                  opsel: int = 0, abs_bits: int = 0, neg_bits: int = 0) -> UOp:
  exec_mask, op_name, bits = ctx.rexec(), _op_name(inst), inst.canonical_op_bits
  is_cmpx, is_vopc = 'CMPX' in op_name, hasattr(inst, 'vsrc1')  # is_vopc: e32 vs e64
  is_dpp16 = hasattr(type(inst), 'dpp') and hasattr(type(inst), 'vsrc0')

  # Handle both VOPC (vsrc1) and VOP3 (src1) instruction formats - read operands dynamically
  if is_vopc:
    src0_off = ctx.inst_field(type(inst).src0)
    vsrc1_off = ctx.inst_field(type(inst).vsrc1)  # type: ignore[union-attr]
    # For 16-bit ops, vsrc1 >= 128 means hi-half of v[vsrc1-128]
    if bits['s0'] == 16:
      vsrc1_hi = vsrc1_off >= _c(128)
      src1_off = _c(256) + vsrc1_hi.where(vsrc1_off - _c(128), vsrc1_off)
    else:
      vsrc1_hi = False
      src1_off = _c(256) + vsrc1_off
  else:
    src0_off = ctx.inst_field(type(inst).src0)
    src1_off = ctx.inst_field(type(inst).src1)  # type: ignore[union-attr]
    dst_off = ctx.inst_field(type(inst).vdst)  # type: ignore[union-attr]
    vsrc1_hi = False
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None  # type: ignore[union-attr]

  is_float, is_f64, pcode = any(x in op_name for x in ('_F32', '_F64', '_F16')), '_F64' in op_name, get_pcode(inst.op)
  def get_cmp_bit(lane) -> UOp:
    lc = lane.cast(dtypes.int) if isinstance(lane, UOp) else _c(lane, dtypes.int)
    s0 = _load_dpp16_src0(ctx, inst, lc, _c(0)) if is_dpp16 else ctx.rsrc_dyn(src0_off, lc, bits['s0'], literal, is_f64)
    s1 = _cond_hi16(vsrc1_hi, ctx.rsrc_dyn(src1_off, lc, bits['s1'], literal, is_f64)) if bits['s0'] == 16 \
      else ctx.rsrc_dyn(src1_off, lc, bits['s1'], literal, is_f64)
    if bits['s0'] == 16 and opsel: s0, s1 = _apply_opsel(s0, 0, opsel), _apply_opsel(s1, 1, opsel)
    if is_float:
      if is_dpp16:
        s0 = _apply_src_mods(s0, 0, 1 if getattr(inst, 'src0_abs', 0) else 0, 1 if getattr(inst, 'src0_neg', 0) else 0, bits['s0'])
        s1 = _apply_src_mods(s1, 0, 1 if getattr(inst, 'src1_abs', 0) else 0, 1 if getattr(inst, 'src1_neg', 0) else 0, bits['s1'])
      s0 = _apply_src_mods(s0, 0, abs_bits, neg_bits, bits['s0'])
      s1 = _apply_src_mods(s1, 1, abs_bits, neg_bits, bits['s1'])
    for dest, val in parse_pcode(pcode, {'S0': s0, 'S1': s1, 'laneId': lc, 'D0': UOp.const(dtypes.uint64, 0)})[1]:
      if '[laneId]' in dest and ('D0' in dest or 'EXEC' in dest): return val.cast(dtypes.uint32)
    return _c(0)

  new_bits = ctx.unroll_lanes(get_cmp_bit, exec_mask, apply_exec=False)
  # Both VOPC and VOP3 clear inactive lane bits (hardware verified)
  new_result = new_bits & exec_mask

  # CMPX e32: writes EXEC only; CMPX e64: writes both EXEC and SDST; non-CMPX: writes dst only
  if is_cmpx:
    stores = ctx.wmask(_c(EXEC_LO.offset), new_result)
    if not is_vopc: stores.extend(ctx.wmask(dst_off, new_result))
  else:
    stores = ctx.wmask(dst_off, new_result) if not is_vopc else ctx.wmask(_c(VCC_LO.offset), new_result)
  return UOp.sink(*stores, *ctx.inc_pc())


def _compile_bitop3(inst, ctx: _Ctx, exec_mask: UOp, bits: dict, op_name: str) -> UOp:
  """BITOP3: 3-input truth table. abs/neg/omod encode the truth table, not source modifiers."""
  lane = ctx.range()
  vdst_reg = ctx.inst_field(type(inst).vdst)
  ops = inst.canonical_operands
  src0 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src0), lane, bits['s0'], None, 's0' in ops and ops['s0'][0] == Fmt.FMT_NUM_F64)
  src1 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src1), lane, bits['s1'], None, 's1' in ops and ops['s1'][0] == Fmt.FMT_NUM_F64)
  src2 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src2), lane, bits['s2'], None, 's2' in ops and ops['s2'][0] == Fmt.FMT_NUM_F64)
  # Truth table: TTBL = { omod[1:0], abs[2:0], neg[2:0] } = 8-bit LUT
  ttbl = ((getattr(inst, 'omod', 0) or 0) << 6) | ((getattr(inst, 'abs', 0) or 0) << 3) | (getattr(inst, 'neg', 0) or 0)
  is_16 = 'B16' in op_name
  dt, mask = (dtypes.uint16, 0xFFFF) if is_16 else (dtypes.uint32, 0xFFFFFFFF)
  s0, s1, s2 = src0.cast(dt), src1.cast(dt), src2.cast(dt)
  def bnot(v): return v ^ UOp.const(dt, mask)
  result = UOp.const(dt, 0)
  for i in range(8):
    if not (ttbl & (1 << i)): continue
    result = result | ((s0 if i & 4 else bnot(s0)) & (s1 if i & 2 else bnot(s1)) & (s2 if i & 1 else bnot(s2)))
  return UOp.sink(ctx.wvgpr_dyn(vdst_reg, lane, result.cast(dtypes.uint32), exec_mask).end(lane), *ctx.inc_pc())

def _compile_vop3(inst: ir3.VOP3 | ir4.VOP3 | irc.VOP3, ctx: _Ctx) -> UOp:
  exec_mask = ctx.rexec()
  bits = inst.canonical_op_bits
  opsel, op_name = getattr(inst, 'opsel', 0) or 0, _op_name(inst)

  # Lane operations
  if op_name in ('V_READLANE_B32', 'V_READFIRSTLANE_B32', 'V_READFIRSTLANE_B32_E64', 'V_WRITELANE_B32'):
    return ctx.compile_lane_pcode(inst.op, inst)

  # V_PERMLANE16_B32 / V_PERMLANEX16_B32: cross-lane swizzle via pcode
  if 'PERMLANE16' in op_name or 'PERMLANEX16' in op_name:
    return ctx.compile_lane_pcode(inst.op, inst)

   # VOP3 VOPC (v_cmp_*_e64) - delegate to unified VOPC handler
  if 'V_CMP' in op_name or 'V_CMPX' in op_name:
    return _compile_vopc(inst, ctx, opsel=opsel, abs_bits=getattr(inst, 'abs', 0) or 0, neg_bits=getattr(inst, 'neg', 0) or 0)

  # BITOP3: abs/neg/omod encode truth table, not source modifiers
  if 'BITOP3' in op_name:
    return _compile_bitop3(inst, ctx, exec_mask, bits, op_name)

  # VOP3 specific fields
  vdst_reg = ctx.inst_field(type(inst).vdst)
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None  # type: ignore[union-attr]
  abs_bits, neg_bits = getattr(inst, 'abs', 0) or 0, getattr(inst, 'neg', 0) or 0

  # VOP3_SDST: v_s_* instructions goes to SGPR
  if 'V_S_' in op_name:
    src0 = _apply_src_mods(ctx.rsrc_dyn(ctx.inst_field(type(inst).src0), _c(0, dtypes.int), bits['s0'], literal), 0, abs_bits, neg_bits, bits['s0'])
    srcs = {'S0': src0, 'EXEC': exec_mask, 'SCC': ctx.rsgpr_dyn(_c(SCC.offset)), 'laneId': _c(0, dtypes.int),
            'ROUND_MODE': _c(0), 'ROUND_TOWARD_ZERO': _c(0)}
    _, assigns = parse_pcode(get_pcode(inst.op), srcs)
    stores = [ctx.wsgpr_dyn(vdst_reg, _val_to_u32(val)) for dest, val in assigns if dest.startswith('D0')]
    return UOp.sink(*stores, *ctx.inc_pc())

  # Regular VOP3 - read operands dynamically
  lane = ctx.range()
  ops = inst.canonical_operands
  src0 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src0), lane, bits['s0'], literal, 's0' in ops and ops['s0'][0] == Fmt.FMT_NUM_F64)
  src1 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src1), lane, bits['s1'], literal, 's1' in ops and ops['s1'][0] == Fmt.FMT_NUM_F64)
  src2 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src2), lane, bits['s2'], literal, 's2' in ops and ops['s2'][0] == Fmt.FMT_NUM_F64)
  if bits['s0'] == 16:
    src0 = _apply_opsel(src0, 0, opsel)
    src1 = _apply_opsel(src1, 1, opsel)
    src2 = _apply_opsel(src2, 2, opsel)
  src0 = _apply_src_mods(src0, 0, abs_bits, neg_bits, bits['s0'])
  src1 = _apply_src_mods(src1, 1, abs_bits, neg_bits, bits['s1'])
  src2 = _apply_src_mods(src2, 2, abs_bits, neg_bits, bits['s2'])
  srcs = {'S0': src0, 'S1': src1, 'S2': src2, 'OPSEL': UOp.const(dtypes.uint32, opsel)}
  if 'CNDMASK' in op_name and src2 is not None: srcs['VCC'] = src2
  # FMAC instructions need D0 (accumulator) from destination register
  if 'FMAC' in op_name: srcs['D0'] = ctx.rvgpr_dyn(vdst_reg, lane)
  opsel_dst_hi = bool(opsel & 0b1000) and bits['d'] == 16
  return ctx.compile_vop_pcode(inst.op, srcs, lane, vdst_reg, exec_mask, opsel_dst_hi=opsel_dst_hi, clmp=getattr(inst, 'clmp', 0))

def _compile_vinterp(inst: ir3.VINTERP | ir4.VINTERP, ctx: _Ctx) -> UOp:
  lane, exec_mask = ctx.range(), ctx.rexec()
  inst_type = type(inst)
  vdst_reg = ctx.inst_field(inst_type.vdst)
  src0_off, src1_off, src2_off = ctx.inst_field(inst_type.src0), ctx.inst_field(inst_type.src1), ctx.inst_field(inst_type.src2)
  src0_reg = (src0_off >= _c(256)).where(src0_off - _c(256), src0_off)
  src2_reg = (src2_off >= _c(256)).where(src2_off - _c(256), src2_off)
  srcs = {
    'SRC0': src0_reg, 'SRC2': src2_reg,
    'S0': ctx.rsrc_dyn(src0_off, lane), 'S1': ctx.rsrc_dyn(src1_off, lane), 'S2': ctx.rsrc_dyn(src2_off, lane),
  }
  return ctx.compile_vop_pcode(inst.op, srcs, lane, vdst_reg, exec_mask)

def _compile_vop3sd(inst: ir3.VOP3SD | ir4.VOP3SD | irc.VOP3SD, ctx: _Ctx) -> UOp:
  exec_mask = ctx.rexec()
  bits, pcode, ops = inst.canonical_op_bits, get_pcode(inst.op), inst.canonical_operands

  # Read operands dynamically from instruction encoding
  vdst_reg, sdst_off = ctx.inst_field(type(inst).vdst), ctx.inst_field(type(inst).sdst)
  src0_off, src1_off, src2_off = ctx.inst_field(type(inst).src0), ctx.inst_field(type(inst).src1), ctx.inst_field(type(inst).src2)
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None  # type: ignore[union-attr]

  has_carry_in = 's2' in ops and ops['s2'][2] == OpType.OPR_SREG
  vcc_in_off = src2_off if has_carry_in else sdst_off

  def load_srcs(lane_uop):
    ret = {'VCC': ctx.rmask(vcc_in_off), 'EXEC': exec_mask, 'SCC': ctx.rsgpr_dyn(_c(SCC.offset)), 'laneId': lane_uop}
    ret['S0'] = ctx.rsrc_dyn(src0_off, lane_uop, bits['s0'], literal, ops['s0'][0] == Fmt.FMT_NUM_F64)
    ret['S1'] = ctx.rsrc_dyn(src1_off, lane_uop, bits['s1'], literal, ops['s1'][0] == Fmt.FMT_NUM_F64)
    if 's2' in ops: ret['S2'] = ctx.rsrc_dyn(src2_off, lane_uop, bits['s2'], literal, ops['s2'][0] == Fmt.FMT_NUM_F64)
    return ret

  lane = ctx.range()
  srcs = load_srcs(lane)
  _, assigns = parse_pcode(pcode, srcs)

  has_per_lane_vcc = any('[laneId]' in dest for dest, _ in assigns if dest.startswith('VCC') or dest.startswith('D0.u64'))
  clmp = getattr(inst, 'clmp', 0)
  if has_per_lane_vcc:
    # VCC computation: RANGE+REDUCE gets axis ID first (lower ID = runs first)
    # This ensures VCC reads source values BEFORE VGPR stores modify them
    def get_vcc_bit(lane_uop) -> UOp:
      vcc_bit = _c(0)
      for dest, val in parse_pcode(pcode, load_srcs(lane_uop))[1]:
        if dest.startswith('VCC') or (dest.startswith('D0.u64') and '[laneId]' in dest): vcc_bit = val.cast(dtypes.uint32)
      return vcc_bit
    final_vcc = ctx.unroll_lanes(get_vcc_bit, exec_mask)
    # VGPR stores: RANGE gets axis ID second (higher ID = runs after VCC loop)
    lane3 = ctx.range()
    d0_val, vcc_per_lane = None, None
    for dest, val in parse_pcode(pcode, load_srcs(lane3))[1]:
      if dest.startswith('D0') and '[laneId]' not in dest: d0_val = val
      if dest.startswith('VCC') or (dest.startswith('D0.u64') and '[laneId]' in dest): vcc_per_lane = val
    vgpr_stores = []
    if d0_val is not None:
      # Apply clamp using carry/borrow bit: ADD overflow->0xFFFFFFFF, SUB underflow->0
      if clmp and vcc_per_lane is not None:
        is_sub = 'SUB' in inst.op.name
        sat_val = _c(0) if is_sub else _c(0xFFFFFFFF)
        d0_val = vcc_per_lane.cast(dtypes.bool).where(sat_val, d0_val.cast(dtypes.uint32))
      if d0_val.dtype in (dtypes.uint64, dtypes.int64, dtypes.float64):
        lo, hi = _split64(d0_val)
        vgpr_stores.extend([ctx.wvgpr_dyn(vdst_reg, lane3, lo, exec_mask), ctx.wvgpr_dyn(vdst_reg + _c(1), lane3, hi, exec_mask)])
      else:
        d0_u32 = d0_val.bitcast(dtypes.uint32) if d0_val.dtype in (dtypes.float32, dtypes.half) else d0_val.cast(dtypes.uint32)
        vgpr_stores.append(ctx.wvgpr_dyn(vdst_reg, lane3, d0_u32, exec_mask))
    # Write carry output (wmask handles lo/hi split for wave64)
    vcc_writes = ctx.wmask(sdst_off, final_vcc)
    return UOp.sink(*vcc_writes, UOp.group(*vgpr_stores).end(lane3), *ctx.inc_pc())
  else:
    return ctx.compile_vop_pcode(inst.op, srcs, lane, vdst_reg, exec_mask, sdst_reg=inst.sdst.offset)

def _compile_mfma(inst: irc.VOP3P, ctx: _Ctx) -> UOp:
  """CDNA MFMA matrix multiply-accumulate emulation.

  Uses local temp arrays to cache inputs, avoiding aliasing issues when vdst overlaps src0/src1.
  Phase 1: Read all input f32 values from VGPRs into temp arrays (range loop over 64 lanes).
  Phase 2: Compute 256 output values using temp arrays and write to VGPRs (range loop over 64 lanes)

  Register layout (wave64):
  - 16x16: 4 groups of 16 lanes. Each lane in group holds k_per_grp elements. 4 output ACCVGPRs per lane.
  - 32x32: 2 groups of 32 lanes. lanes%16 = M/N index within block, lanes//16 selects block. 16 output ACCVGPRs per lane.
  - 4x4: 16 groups of 4 lanes. 4 output ACCVGPRs per lane.
  """
  import re as _re
  op_name = _op_name(inst)
  exec_mask = ctx.rexec()
  vdst_reg = ctx.inst_field(type(inst).vdst)
  src0_off = ctx.inst_field(type(inst).src0)
  src1_off = ctx.inst_field(type(inst).src1)
  src0_r = src0_off - _c(256)  # VGPR-relative index (only valid when src is VGPR)
  src1_r = src1_off - _c(256)
  src2_off = ctx.inst_field(type(inst).src2)
  use_acc = bool(getattr(inst, 'acc_cd', 0))
  # Check if sources are VGPRs (offset >= 256) vs inline constants/SGPRs
  src0_is_vgpr = src0_off >= _c(256)
  src1_is_vgpr = src1_off >= _c(256)

  m = _re.search(r'(\d+)X(\d+)X(\d+)', op_name)
  if m is None: raise ValueError(f"could not parse MFMA dimensions from {op_name}")
  M, N, K = int(m.group(1)), int(m.group(2)), int(m.group(3))

  is_bf16 = 'BF16' in op_name
  is_fp8 = 'FP8' in op_name or 'F8' in op_name
  is_i8 = 'I8' in op_name
  # Source type is the LAST type in the name: V_MFMA_F32_16X16X32_**F16** -> source is F16, not F32
  src_type = op_name.rsplit('_', 1)[-1]  # e.g. "F16", "BF16", "F32", "I8"
  is_f32_src = src_type == 'F32'
  is_int_out = 'I32' in op_name.split('_')[2]  # V_MFMA_I32_...

  # Determine elements per VGPR and conversion function
  if is_i8: vpg = 4
  elif is_f32_src: vpg = 1
  elif is_fp8: vpg = 4
  else: vpg = 2

  # For 16x16: grp_size=16, n_grps=4, out_per_lane=4
  # For 32x32: grp_size=32, n_grps=2, out_per_lane=16
  # For 4x4: grp_size=4, n_grps=16, out_per_lane=4
  if M == 16 and N == 16:
    grp_size, n_grps, out_per_lane = 16, 4, 4
  elif M == 32 and N == 32:
    grp_size, n_grps, out_per_lane = 32, 2, 16
  elif M == 4 and N == 4:
    grp_size, n_grps, out_per_lane = 4, 16, 4
  else:
    raise RuntimeError(f"unsupported MFMA shape {M}x{N}x{K}")

  # For 4x4: each group independently computes a 4x4 block. K is NOT split across groups.
  # For 16x16/32x32: K IS split across groups (each group has K/n_grps elements).
  k_per_grp = K if M == 4 else K // n_grps
  # Temp array size: for 4x4, store all 16 independent blocks; for others, store shared MxK/NxK
  n_a_elems = n_grps * M * K if M == 4 else M * K
  n_b_elems = n_grps * N * K if M == 4 else N * K

  # src2 can be VGPR (>=256) or inline constant/SGPR (<256)
  src2_is_vgpr = src2_off >= _c(256)
  src2_r = src2_off - _c(256)
  if is_int_out:
    acc_scalar = ctx.rsgpr_dyn(src2_off, src2_is_vgpr.ne(True)).cast(dtypes.int32)
  else:
    acc_scalar = ctx.rsgpr_dyn(src2_off, src2_is_vgpr.ne(True)).bitcast(dtypes.float32)

  # Phase 1: Read all A and B values from VGPRs into temp arrays.
  # Layout: tmp[0..n_a_elems-1] = A[m][k], tmp[n_a_elems..n_a_elems+n_b_elems-1] = B[n][k]
  # Within each group of lanes, lane%grp_sub gives M/N index, lane//grp_sub gives sub-block
  grp_sub = min(M, 16)  # lanes within group mapped to M/N dimension
  b_off = UOp.const(dtypes.int, n_a_elems)
  acc_dt = dtypes.int32 if is_int_out else dtypes.float32
  # Use uint32 temp array to prevent optimizer from eliminating f16→f32 bitcast chains.
  # The optimizer folds bitcast(uint32→float32) stores to float32 arrays, losing the conversion.
  tmp = UOp.placeholder((n_a_elems + n_b_elems,), dtypes.uint32, slot=0, addrspace=AddrSpace.LOCAL)

  def cvt_elem(raw: UOp, sub_idx: int) -> UOp:
    if is_i8:
      # Extract i8, sign-extend to i32
      byte_val = (raw >> UOp.const(dtypes.uint32, sub_idx * 8)) & UOp.const(dtypes.uint32, 0xFF)
      return (byte_val.cast(dtypes.int32) ^ UOp.const(dtypes.int32, 0x80)) - UOp.const(dtypes.int32, 0x80)
    elif is_f32_src:
      return raw  # already uint32 (f32 bit pattern)
    elif is_fp8:
      return ((raw >> UOp.const(dtypes.uint32, sub_idx * 8)) & UOp.const(dtypes.uint32, 0xFF)).cast(dtypes.uint32)
    elif is_bf16:
      # bf16→f32 bits: just shift left by 16 (bf16 is upper 16 bits of f32)
      return ((raw >> UOp.const(dtypes.uint32, sub_idx * 16)) & UOp.const(dtypes.uint32, 0xFFFF)) << UOp.const(dtypes.uint32, 16)
    else:
      # f16→f32 conversion using float arithmetic to avoid UOp optimizer eliminating the conversion.
      # The optimizer folds bitcast(uint32→float32) chains, so we compute the float value directly.
      h = (raw >> UOp.const(dtypes.uint32, sub_idx * 16)) & UOp.const(dtypes.uint32, 0xFFFF)
      sign = (h >> UOp.const(dtypes.uint32, 15)) & UOp.const(dtypes.uint32, 1)
      exp = (h >> UOp.const(dtypes.uint32, 10)) & UOp.const(dtypes.uint32, 0x1F)
      mant = h & UOp.const(dtypes.uint32, 0x3FF)
      # Use bf16 path: shift left by 16 to create bf16 bits, then shift mantissa and adjust exponent in float domain
      # bf16 bits = (sign << 15) | (exp_bf16 << 7) | mant_bf16 -- but f16 and bf16 have different formats
      # Instead: construct f32 bits properly, use a local uint32 array to force materialization
      f32_bits = (sign << UOp.const(dtypes.uint32, 31)) | \
                 ((exp + UOp.const(dtypes.uint32, 112)) << UOp.const(dtypes.uint32, 23)) | \
                 (mant << UOp.const(dtypes.uint32, 13))
      is_zero = exp.eq(UOp.const(dtypes.uint32, 0))
      # Return uint32 (f32 bit pattern) — stored directly to uint32 temp array, bitcast to float on read
      return is_zero.where(UOp.const(dtypes.uint32, 0), f32_bits)

  read_lane = ctx.range()
  # For 32x32: lane%16 = M/N index within 16-wide block, lane//16 = which of 4 quarter-waves
  # Groups: lanes 0-31 = group 0, lanes 32-63 = group 1
  # Within group: (lane%32)%16 = M/N[0-15], (lane%32)//16 selects M/N[0-15] or [16-31]
  lane_in_grp = read_lane % UOp.const(dtypes.int, grp_size)
  grp_idx = read_lane // UOp.const(dtypes.int, grp_size)

  if M == 32:
    # 32x32: lane_in_grp%16 = sub-row/col (0-15), lane_in_grp//16 = block (0=rows 0-15, 1=rows 16-31)
    sub_mn = lane_in_grp % UOp.const(dtypes.int, 16)
    block_mn = lane_in_grp // UOp.const(dtypes.int, 16)
    mn_idx = block_mn * UOp.const(dtypes.int, 16) + sub_mn  # actual M/N index (0-31)
  else:
    mn_idx = lane_in_grp  # for 16x16 and 4x4

  read_stores = []
  for kl in range(k_per_grp):
    reg_idx, sub_idx = kl // vpg, kl % vpg
    # Read A/B sources. Use rsrc_dyn for inline constants/SGPRs (src_off < 256), rvgpr_dyn for VGPRs (src_off >= 256).
    a_raw = src0_is_vgpr.where(ctx.rvgpr_dyn(src0_r + _c(reg_idx), read_lane),
                                ctx.rsrc_dyn(src0_off, _c(0, dtypes.int), 32))
    a_val = cvt_elem(a_raw, sub_idx)
    if M == 4:
      a_idx = grp_idx * UOp.const(dtypes.int, M * K) + mn_idx * UOp.const(dtypes.int, K) + UOp.const(dtypes.int, kl)
    else:
      a_idx = mn_idx * UOp.const(dtypes.int, K) + grp_idx * UOp.const(dtypes.int, k_per_grp) + UOp.const(dtypes.int, kl)
    read_stores.append(tmp.index(a_idx).store(a_val))

    b_raw = src1_is_vgpr.where(ctx.rvgpr_dyn(src1_r + _c(reg_idx), read_lane),
                                ctx.rsrc_dyn(src1_off, _c(0, dtypes.int), 32))
    b_val = cvt_elem(b_raw, sub_idx)
    if M == 4:
      b_idx = b_off + grp_idx * UOp.const(dtypes.int, N * K) + mn_idx * UOp.const(dtypes.int, K) + UOp.const(dtypes.int, kl)
    else:
      b_idx = b_off + mn_idx * UOp.const(dtypes.int, K) + grp_idx * UOp.const(dtypes.int, k_per_grp) + UOp.const(dtypes.int, kl)
    read_stores.append(tmp.index(b_idx).store(b_val))

  read_phase = UOp.group(*read_stores).end(read_lane)

  # Phase 2: Compute dot products and write outputs.
  # For 16x16: each lane computes 4 outputs. n_idx = lane%16, grp selects which 4 rows.
  # For 32x32: each lane computes 16 outputs. Layout: lane%16 selects n within block, lane//16 selects column block.
  #   Output mapping: out_reg r at lane l -> D[m][n] where
  #   n = (l%32)%16 + ((l%32)//16)*16, m = (l//32)*4 + r (for r in 0..3), with 4 groups of 4 rows -> 16 outputs total
  #   Actually: 16 ACCVGPRs per lane, organized as 4 groups (l//32 gives half, each half has 2 sub-groups) of 4 rows
  tmp2 = tmp.after(read_phase)

  compute_lane = ctx.range()
  compute_stores = []

  if M == 32 and N == 32:
    # 32x32: each lane has 16 output ACCVGPRs
    # Lane mapping: n = (lane%32)%16 + ((lane%32)//16)*16, gives column 0-31
    # Row groups: 4 groups of 4, covering rows 0-31. Group g (0-3): rows g*4 .. g*4+3
    # group assignment: lane//16 gives quarter (0-3), each quarter maps to 4 rows
    c_lane_in_32 = compute_lane % UOp.const(dtypes.int, 32)
    c_sub = c_lane_in_32 % UOp.const(dtypes.int, 16)
    c_block = c_lane_in_32 // UOp.const(dtypes.int, 16)
    n_idx = c_block * UOp.const(dtypes.int, 16) + c_sub
    c_half = compute_lane // UOp.const(dtypes.int, 32)  # 0 or 1

    for out_reg in range(16):
      # Each half covers 8 rows. out_reg 0-3: rows 0-3 (half0) or 16-19 (half1)
      # out_reg 4-7: rows 4-7 (half0) or 20-23 (half1), etc.
      # Actually: for 32x32, the output layout per lane is:
      # acc[0:3] -> rows 0-3 (half 0) or rows 0-3 (half 1)?
      # Let me use the ISA doc: for 32x32, D has 16 dwords per lane. The mapping is:
      # acc[r] at lane l -> D[m][n] where n = (l%32)%16 + ((l%32)//16)*16
      # m = (l//32)*16 + (r//4)*4 + (r%4)  ... giving rows in blocks of 4
      # So: m_base = half * 16 + (out_reg // 4) * 4 + (out_reg % 4)
      m_base = c_half * UOp.const(dtypes.int, 16) + UOp.const(dtypes.int, (out_reg // 4) * 4 + (out_reg % 4))

      acc_v = (ctx.raccvgpr_dyn if use_acc else ctx.rvgpr_dyn)(src2_r + _c(out_reg), compute_lane, src2_is_vgpr)
      if is_int_out: acc_v = acc_v.cast(dtypes.int32)
      else: acc_v = acc_v.bitcast(dtypes.float32)
      acc = src2_is_vgpr.where(acc_v, acc_scalar)

      for k in range(K):
        a_val = tmp2.index(m_base * UOp.const(dtypes.int, K) + UOp.const(dtypes.int, k)).bitcast(acc_dt)
        b_val = tmp2.index(b_off + n_idx * UOp.const(dtypes.int, K) + UOp.const(dtypes.int, k)).bitcast(acc_dt)
        acc = acc + a_val * b_val

      if is_int_out:
        compute_stores.append((ctx.waccvgpr_dyn if use_acc else ctx.wvgpr_dyn)(
          vdst_reg + _c(out_reg), compute_lane, acc.cast(dtypes.uint32), exec_mask))
      else:
        compute_stores.append((ctx.waccvgpr_dyn if use_acc else ctx.wvgpr_dyn)(
          vdst_reg + _c(out_reg), compute_lane, acc.bitcast(dtypes.uint32), exec_mask))
  else:
    # 16x16 and 4x4: each lane computes out_per_lane outputs
    n_idx = compute_lane % UOp.const(dtypes.int, grp_sub)
    c_grp = compute_lane // UOp.const(dtypes.int, grp_sub)

    for out_reg in range(out_per_lane):
      acc_v = (ctx.raccvgpr_dyn if use_acc else ctx.rvgpr_dyn)(src2_r + _c(out_reg), compute_lane, src2_is_vgpr)
      if is_int_out: acc_v = acc_v.cast(dtypes.int32)
      else: acc_v = acc_v.bitcast(dtypes.float32)
      acc = src2_is_vgpr.where(acc_v, acc_scalar)

      if M == 4:
        # 4x4: each group is independent. A/B indexed per-group.
        m_base = c_grp * UOp.const(dtypes.int, M * K) + UOp.const(dtypes.int, out_reg * K)
        for k in range(K):
          a_val = tmp2.index(m_base + UOp.const(dtypes.int, k)).bitcast(acc_dt)
          b_val = tmp2.index(b_off + c_grp * UOp.const(dtypes.int, N*K) + n_idx * UOp.const(dtypes.int, K)+UOp.const(dtypes.int, k)).bitcast(acc_dt)
          acc = acc + a_val * b_val
      else:
        # 16x16: K is split across groups. Shared MxK/NxK arrays.
        m_base = c_grp * UOp.const(dtypes.int, out_per_lane) + UOp.const(dtypes.int, out_reg)
        for k in range(K):
          a_val = tmp2.index(m_base * UOp.const(dtypes.int, K) + UOp.const(dtypes.int, k)).bitcast(acc_dt)
          b_val = tmp2.index(b_off + n_idx * UOp.const(dtypes.int, K) + UOp.const(dtypes.int, k)).bitcast(acc_dt)
          acc = acc + a_val * b_val

      if is_int_out:
        compute_stores.append((ctx.waccvgpr_dyn if use_acc else ctx.wvgpr_dyn)(
          vdst_reg + _c(out_reg), compute_lane, acc.cast(dtypes.uint32), exec_mask))
      else:
        compute_stores.append((ctx.waccvgpr_dyn if use_acc else ctx.wvgpr_dyn)(
          vdst_reg + _c(out_reg), compute_lane, acc.bitcast(dtypes.uint32), exec_mask))

  compute_phase = UOp.group(*compute_stores).end(compute_lane)
  return UOp.sink(read_phase, compute_phase, *ctx.inc_pc())

def _compile_wmma(inst: ir3.VOP3P | ir4.VOP3P | irc.VOP3P, ctx: _Ctx) -> UOp:
  op_name = _op_name(inst)
  exec_mask = ctx.rexec()
  vdst_reg = ctx.inst_field(type(inst).vdst)
  src0_r = ctx.inst_field(type(inst).src0) - _c(256)
  src1_r = ctx.inst_field(type(inst).src1) - _c(256)
  src2_r = ctx.inst_field(type(inst).src2) - _c(256)
  is_f16_output = 'F16_16X16X16_F16' in op_name or 'BF16_16X16X16_BF16' in op_name  # F16/BF16 output vs F32 output
  is_bf16 = 'BF16' in op_name
  cvt = _FUNCS['bf16_to_f32'] if is_bf16 else _FUNCS['f16_to_f32']
  is_rdna4 = isinstance(inst, ir4.VOP3P)
  # read 16x16 F16/BF16 matrix from VGPRs → flat f32 array[row*16+k]
  def read_f16_val(src, lane, vgpr, half):
    v = ctx.rvgpr_dyn(src + _c(vgpr), UOp.const(dtypes.int, lane))
    return cvt((v >> UOp.const(dtypes.uint32, 16)) if half else (v & UOp.const(dtypes.uint32, 0xFFFF)))

  # RDNA3: 16 lanes × 8 VGPRs × 2 halves, k maps linearly
  # RDNA4: 32 lanes × 4 VGPRs × 2 halves, k bits are scrambled (k[2] goes to lane bit 4)
  def read_f16_mat(src):
  # (row, k) → (lane, vgpr, half)
    def ab_map(i, k):
      elem, lane = ((k & 3) | ((k >> 1) & 4), i + ((k >> 2) & 1) * 16) if is_rdna4 else (k, i)
      return lane, elem // 2, elem % 2
    return [read_f16_val(src, *ab_map(row, k)) for row in range(16) for k in range(16)]
  mat_a, mat_b = read_f16_mat(src0_r), read_f16_mat(src1_r)
  # (row, col) -> (lane, vgpr)
  def d_map(m, n):
    lane_bit, vgpr = (m >> 3, m & 7) if is_rdna4 else (m & 1, m >> 1)
    return n + lane_bit * 16, vgpr
  if is_f16_output:
    # read accumulator C with f16 layout: for RDNA4, pairs of f32 vgprs pack into one f16 vgpr
    # for RDNA3, same layout as f32 but only lo 16 bits used
    mat_c = [read_f16_val(src2_r, *((lane, vgpr // 2, vgpr % 2) if is_rdna4 else (lane, vgpr, 0)))
             for m in range(16) for n in range(16) for lane, vgpr in [d_map(m, n)]]
    mat_d = [sum(mat_a[r*16+k] * mat_b[c*16+k] for k in range(16)) + mat_c[r*16+c] for r in range(16) for c in range(16)]
    def f32_to_f16_bits(v: UOp) -> UOp: return v.cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32)
    def f32_to_bf16_bits(v: UOp) -> UOp: return (v.bitcast(dtypes.uint32) >> UOp.const(dtypes.uint32, 16)) & UOp.const(dtypes.uint32, 0xFFFF)
    out_cvt = f32_to_bf16_bits if is_bf16 else f32_to_f16_bits
    if is_rdna4:  # pack 2 f16 per VGPR: adjacent m values share (lane, vgpr) since vgpr=m&7, half=m&1
      stores = [ctx.wvgpr_dyn(vdst_reg + _c(d_map(m, n)[1] // 2), UOp.const(dtypes.int, d_map(m, n)[0]),
                out_cvt(mat_d[m*16+n]) | (out_cvt(mat_d[(m+1)*16+n]) << UOp.const(dtypes.uint32, 16)), exec_mask)
                for n in range(16) for m in range(0, 16, 2)]
    else:  # (rdna3) 1 f16 per VGPR (lo half only)
      stores = [ctx.wvgpr_dyn(vdst_reg + _c(d_map(m, n)[1]), UOp.const(dtypes.int, d_map(m, n)[0]), out_cvt(mat_d[m*16+n]), exec_mask)
                for m in range(16) for n in range(16)]
  else: # f32
    mat_c = [ctx.rvgpr_dyn(src2_r + _c(d_map(m, n)[1]), UOp.const(dtypes.int, d_map(m, n)[0])).bitcast(dtypes.float32)
             for m in range(16) for n in range(16)]
    mat_d = [sum(mat_a[r*16+k] * mat_b[c*16+k] for k in range(16)) + mat_c[r*16+c] for r in range(16) for c in range(16)]
    stores = [ctx.wvgpr_dyn(vdst_reg + _c(d_map(m, n)[1]), UOp.const(dtypes.int, d_map(m, n)[0]), mat_d[m*16+n].bitcast(dtypes.uint32), exec_mask)
              for m in range(16) for n in range(16)]
  return UOp.sink(*stores, *ctx.inc_pc())

def _compile_vop3p(inst: ir3.VOP3P | ir4.VOP3P | irc.VOP3P, ctx: _Ctx) -> UOp:
  op_name = _op_name(inst)
  if 'WMMA' in op_name and ('16X16X16_F16' in op_name or '16X16X16_BF16' in op_name): return _compile_wmma(inst, ctx)
  if 'MFMA' in op_name and any(f'{s}X{s}X' in op_name for s in ('4', '16', '32')) and isinstance(inst, irc.VOP3P): return _compile_mfma(inst, ctx)

  # ACCVGPR_WRITE/READ/MOV: copies between VGPR and ACCVGPR register files
  # Detect by checking operand types for ACCVGPR involvement
  ops = inst.operands
  src0_is_acc = ops.get('src0', (None, None, None))[2] in (OpType.OPR_SRC_ACCVGPR, OpType.OPR_ACCVGPR)
  vdst_is_acc = ops.get('vdst', (None, None, None))[2] in (OpType.OPR_ACCVGPR,)
  if src0_is_acc or vdst_is_acc:
    lane = ctx.range()
    exec_mask = ctx.rexec()
    vdst_reg = ctx.inst_field(type(inst).vdst)
    src0_off = ctx.inst_field(type(inst).src0)
    if src0_is_acc and not vdst_is_acc:
      # v_accvgpr_read: VGPR[vdst] = ACCVGPR[src0]
      val = ctx.raccvgpr_dyn(src0_off - _c(256), lane)
      return UOp.sink(ctx.wvgpr_dyn(vdst_reg, lane, val, exec_mask).end(lane), *ctx.inc_pc())
    elif vdst_is_acc and not src0_is_acc:
      # v_accvgpr_write: ACCVGPR[vdst] = src0 (src0 can be VGPR or SGPR/const)
      src0 = ctx.rsrc_dyn(src0_off, lane, 32)
      return UOp.sink(ctx.waccvgpr_dyn(vdst_reg, lane, src0, exec_mask).end(lane), *ctx.inc_pc())
    else:
      # v_accvgpr_mov: ACCVGPR[vdst] = ACCVGPR[src0]
      val = ctx.raccvgpr_dyn(src0_off - _c(256), lane)
      return UOp.sink(ctx.waccvgpr_dyn(vdst_reg, lane, val, exec_mask).end(lane), *ctx.inc_pc())

  lane = ctx.range()
  exec_mask = ctx.rexec()
  vdst_reg = ctx.inst_field(type(inst).vdst)
  is_pk_f32 = 'PK' in op_name and 'F32' in op_name and 'MOV' not in op_name  # CDNA packed F32 ops
  is_pk_mov_b32 = 'PK_MOV_B32' in op_name  # CDNA packed MOV needs special handling
  do_cast = any(x in op_name for x in ('F16', 'F32', 'BF16')) and 'IU' not in op_name and not is_pk_f32
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None  # type: ignore[union-attr]
  src0 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src0), lane, 16, literal=literal, do_cast=do_cast)
  src1 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src1), lane, 16, literal=literal, do_cast=do_cast)
  src2 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src2), lane, 16, literal=literal, do_cast=do_cast)
  opsel, opsel_hi = getattr(inst, 'opsel', 0) or 0, getattr(inst, 'opsel_hi', 3) if getattr(inst, 'opsel_hi', 3) is not None else 3
  opsel_hi2 = getattr(inst, 'opsel_hi2', 1) if getattr(inst, 'opsel_hi2', 1) is not None else 1
  neg, neg_hi = getattr(inst, 'neg', 0) or 0, getattr(inst, 'neg_hi', 0) or 0

  if is_pk_mov_b32:
    # v_pk_mov_b32: D[lo] = src0[opsel_bit0 ? hi : lo], D[hi] = src1[opsel_bit1 ? hi : lo]
    src_offs = [ctx.inst_field(type(inst).src0), ctx.inst_field(type(inst).src1)]
    def _pk_mov_sel(src_lo: UOp, src_off: UOp, sel_bit: int) -> UOp:
      is_vgpr = src_off >= _c(256)
      vgpr_lo = ctx.rvgpr_dyn(src_off - _c(256), lane) if lane is not None else _c(0)
      vgpr_hi = ctx.rvgpr_dyn(src_off - _c(256) + _c(1), lane) if lane is not None else _c(0)
      is_sgpr_pair = src_off < _c(128)
      sgpr_hi = ctx.rsgpr_dyn(src_off + _c(1), is_sgpr_pair)
      scalar_sel = is_sgpr_pair.where(sgpr_hi, src_lo) if sel_bit else src_lo
      return is_vgpr.where(vgpr_hi if sel_bit else vgpr_lo, scalar_sel)
    lo_val = _pk_mov_sel(src0, src_offs[0], opsel & 1)
    hi_val = _pk_mov_sel(src1, src_offs[1], opsel & 2)
    result = _u64(lo_val, hi_val)
    lo_out, hi_out = _split64(result)
    stores = [ctx.wvgpr_dyn(vdst_reg, lane, lo_out, exec_mask), ctx.wvgpr_dyn(vdst_reg + _c(1), lane, hi_out, exec_mask)]
    return UOp.sink(UOp.group(*stores).end(lane), *ctx.inc_pc())

  srcs: dict[str, UOp | int] = {}
  if is_pk_f32:
    # CDNA packed F32: read 32-bit sources, build 64-bit packed values using opsel.
    # For VGPRs: opsel selects between v[reg] (0) and v[reg+1] (1) for each half.
    # For SGPR pairs (off < 128): s[N] = lo float32, s[N+1] = hi float32.
    # For inline constants (128 <= off < 256): broadcast same value to both halves.
    src_offs = [ctx.inst_field(type(inst).src0), ctx.inst_field(type(inst).src1), ctx.inst_field(type(inst).src2)]
    def build_pk_f32(src_lo: UOp, src_off: UOp, opsel_lo: int, opsel_hi_bit: int, neg_lo: int, neg_hi_bit: int) -> UOp:
      is_vgpr = src_off >= _c(256)
      vgpr_lo = ctx.rvgpr_dyn(src_off - _c(256), lane) if lane is not None else _c(0)
      vgpr_hi = ctx.rvgpr_dyn(src_off - _c(256) + _c(1), lane) if lane is not None else _c(0)
      # For SGPR pairs, opsel selects between s[N] (0) and s[N+1] (1); inline constants always broadcast.
      is_sgpr_pair = src_off < _c(128)
      sgpr_hi = ctx.rsgpr_dyn(src_off + _c(1), is_sgpr_pair)
      scalar_lo_sel = src_lo if not opsel_lo else is_sgpr_pair.where(sgpr_hi, src_lo)
      scalar_hi_sel = src_lo if not opsel_hi_bit else is_sgpr_pair.where(sgpr_hi, src_lo)
      lo = is_vgpr.where(vgpr_hi if opsel_lo else vgpr_lo, scalar_lo_sel)
      hi = is_vgpr.where(vgpr_hi if opsel_hi_bit else vgpr_lo, scalar_hi_sel)
      if neg_lo: lo = lo ^ UOp.const(dtypes.uint32, 0x80000000)
      if neg_hi_bit: hi = hi ^ UOp.const(dtypes.uint32, 0x80000000)
      return _u64(lo, hi)
    srcs = {'S0': build_pk_f32(src0, src_offs[0], opsel & 1, opsel_hi & 1, neg & 1, neg_hi & 1),
            'S1': build_pk_f32(src1, src_offs[1], opsel & 2, opsel_hi & 2, neg & 2, neg_hi & 2),
            'S2': build_pk_f32(src2, src_offs[2], opsel & 4, 1 if opsel_hi2 else 0, neg & 4, neg_hi & 4)}
  elif 'FMA_MIX' in op_name or 'MAD_MIX' in op_name:
    combined_opsel_hi = (opsel_hi & 0x3) | ((opsel_hi2 & 0x1) << 2)
    # For FMA_MIX: neg_hi is ABS (not neg!), neg is actual negation
    def apply_abs(v, bit, opsel_hi_bit, opsel_bit):
      if not (neg_hi & bit): return v
      # Apply abs based on whether source is f32 or f16
      if not (combined_opsel_hi & opsel_hi_bit): return v & UOp.const(dtypes.uint32, 0x7FFFFFFF)  # f32 abs
      if opsel & opsel_bit: return v & UOp.const(dtypes.uint32, 0x7FFF0000)  # f16 hi abs (preserve lo)
      return v & UOp.const(dtypes.uint32, 0xFFFF7FFF)  # f16 lo abs (preserve hi)
    def apply_neg_mix(v, bit, opsel_hi_bit, opsel_bit):
      if not (neg & bit): return v
      if not (combined_opsel_hi & opsel_hi_bit): return v ^ UOp.const(dtypes.uint32, 0x80000000)  # f32 neg
      if opsel & opsel_bit: return v ^ UOp.const(dtypes.uint32, 0x80000000)  # f16 hi neg
      return v ^ UOp.const(dtypes.uint32, 0x00008000)  # f16 lo neg
    s0_mod = apply_neg_mix(apply_abs(src0, 1, 1, 1), 1, 1, 1)
    s1_mod = apply_neg_mix(apply_abs(src1, 2, 2, 2), 2, 2, 2)
    s2_mod = apply_neg_mix(apply_abs(src2, 4, 4, 4), 4, 4, 4)
    srcs = {'S@0': s0_mod, 'S@1': s1_mod, 'S@2': s2_mod,
            'OPSEL_HI': UOp.const(dtypes.uint32, combined_opsel_hi), 'OPSEL': UOp.const(dtypes.uint32, opsel)}
  else:
    def get_half_bits(val: UOp, use_hi: bool, apply_neg: bool = False) -> UOp:
      bits = ((val >> UOp.const(dtypes.uint32, 16)) if use_hi else val) & UOp.const(dtypes.uint32, 0xFFFF)
      if apply_neg: bits = bits.cast(dtypes.uint16).bitcast(dtypes.half).neg().bitcast(dtypes.uint16).cast(dtypes.uint32)
      return bits
    def build_remapped_src(src: UOp, opsel_lo_bit: int, opsel_hi_bit: int, neg_lo_bit: int, neg_hi_bit: int) -> UOp:
      lo = get_half_bits(src, bool(opsel_lo_bit), bool(neg_lo_bit))
      hi = get_half_bits(src, bool(opsel_hi_bit), bool(neg_hi_bit))
      return lo | (hi << UOp.const(dtypes.uint32, 16))
    # DOT IU instructions use NEG bits for signed/unsigned selection, not fp16 negation
    is_dot_iu = 'DOT' in op_name and 'IU' in op_name
    n0, n1, n2, nh0, nh1, nh2 = (0, 0, 0, 0, 0, 0) if is_dot_iu else (neg & 1, neg & 2, neg & 4, neg_hi & 1, neg_hi & 2, neg_hi & 4)
    srcs = {'S0': build_remapped_src(src0, opsel & 1, opsel_hi & 1, n0, nh0),
            'S1': build_remapped_src(src1, opsel & 2, opsel_hi & 2, n1, nh1),
            'S2': build_remapped_src(src2, opsel & 4, 1 if opsel_hi2 else 0, n2, nh2)}
    if is_dot_iu: srcs['NEG'] = UOp.const(dtypes.uint32, neg)
  return ctx.compile_vop_pcode(inst.op, srcs, lane, vdst_reg, exec_mask)

def _compile_vopd(inst: ir3.VOPD | ir4.VOPD, ctx: _Ctx) -> UOp:
  exec_mask = ctx.rexec()
  # Read operands dynamically - use type(inst) to get correct field descriptors
  inst_type = type(inst)
  vdstx_reg = ctx.inst_field(inst_type.vdstx)
  # vdsty has complex encoding: actual = (raw << 1) | ((vdstx & 1) ^ 1)
  vdsty_raw = ctx.inst_field(inst_type.vdsty)
  vdsty_reg = (vdsty_raw << _c(1)) | ((vdstx_reg & _c(1)) ^ _c(1))
  srcx0_off = ctx.inst_field(inst_type.srcx0)
  srcy0_off = ctx.inst_field(inst_type.srcy0)
  vsrcx1_reg = ctx.inst_field(inst_type.vsrcx1)
  vsrcy1_reg = ctx.inst_field(inst_type.vsrcy1)
  literal = ctx.inst_field(inst_type.literal) if hasattr(inst_type, 'literal') else None

  lane = ctx.range()
  srcy0, srcy1 = ctx.rsrc_dyn(srcy0_off, lane, literal=literal), ctx.rvgpr_dyn(vsrcy1_reg, lane)
  all_stores = []
  srcs:dict[str, UOp | int] = {}
  for op, src0_off, vsrc1_reg, vdst_reg, label in [(inst.opx, srcx0_off, vsrcx1_reg, vdstx_reg, 'X'),
                                                    (inst.opy, srcy0_off, vsrcy1_reg, vdsty_reg, 'Y')]:
    vop = VOPD_TO_VOP2.get(op)
    assert vop is not None, f"no VOP mapping for VOPD {label}: {op}"
    if label == 'Y': srcs = {'S0': srcy0, 'S1': srcy1, 'D0': ctx.rvgpr_dyn(vdst_reg, lane)}
    else: srcs = {'S0': ctx.rsrc_dyn(src0_off, lane, literal=literal), 'S1': ctx.rvgpr_dyn(vsrc1_reg, lane), 'D0': ctx.rvgpr_dyn(vdst_reg, lane)}
    # VOP2_FMAAK/FMAMK_(DTYPE)_E32
    if vop in (ir3.VOP2Op.V_FMAAK_F32_E32, ir3.VOP2Op.V_FMAMK_F32_E32, ir3.VOP2Op.V_FMAAK_F32_E32, ir3.VOP2Op.V_FMAMK_F32_E32):
      assert literal is not None
      srcs['SIMM32'] = literal
    if op in (ir3.VOPDOp.V_DUAL_CNDMASK_B32, ir4.VOPDOp.V_DUAL_CNDMASK_B32): srcs['VCC'] = ctx.rmask(_c(VCC_LO.offset))
    pcode = get_pcode(vop)
    srcs.update({'VCC': ctx.rmask(_c(VCC_LO.offset)), 'EXEC': exec_mask, 'SCC': ctx.rsgpr_dyn(_c(SCC.offset)), 'laneId': lane})
    for dest, val in parse_pcode(pcode, srcs)[1]:
      if dest.startswith('D0'): all_stores.append(ctx.wvgpr_dyn(vdst_reg, lane, _val_to_u32(val), exec_mask, after=srcy1))
  return UOp.sink(UOp.group(*all_stores).end(lane), *ctx.inc_pc())

def _compile_mem_op(inst: ir3.DS|ir3.FLAT|ir3.GLOBAL|ir3.SCRATCH|ir4.DS|ir4.VFLAT|ir4.VGLOBAL|ir4.VSCRATCH
                    |irc.DS|irc.FLAT|irc.GLOBAL|irc.SCRATCH, ctx: _Ctx) -> UOp:
  """Unified memory operation compiler for DS, FLAT, GLOBAL, SCRATCH."""
  exec_mask, op_name = ctx.rexec(), _op_name(inst)
  pcode = get_pcode(inst.op)
  # CDNA pcode uses CalcGlobalAddr/CalcDsAddr to compute address from raw components, but make_addr already handles this.
  # Strip the addr computation line and use pre-computed ADDR directly (rename 'addr' -> 'ADDR' in remaining pcode).
  if isinstance(inst, (irc.GLOBAL, irc.FLAT, irc.SCRATCH, irc.DS, ir4.VSCRATCH)) and 'Calc' in pcode and 'Addr' in pcode:
    pcode = re.sub(r'addr\s*=\s*Calc\w+Addr\([^)]*\)\s*;?\n?', '', pcode).replace('MEM[addr', 'MEM[ADDR')

  is_lds = isinstance(inst, (ir3.DS, ir4.DS, irc.DS))
  is_scratch = isinstance(inst, (ir3.SCRATCH, ir4.VSCRATCH, irc.SCRATCH))
  # CDNA acc bit: when set, VGPR operands (vdst/vdata) target ACCVGPR file instead of VGPR
  use_acc = bool(getattr(inst, 'acc', 0))
  mem = ctx.lds if is_lds else ctx.scratch if is_scratch else ctx.vmem
  addr_shift = UOp.const(dtypes.uint32 if is_lds else dtypes.uint64, 2)

  # Extract register info - all dynamic for deduplication
  if is_lds:
    addr_reg = ctx.inst_field(type(inst).addr)  # type: ignore[union-attr]
    vdata_reg = ctx.inst_field(type(inst).data0)  # type: ignore[union-attr]
    vdst_reg = ctx.inst_field(type(inst).vdst)
    offset0 = ctx.inst_field(type(inst).offset0)  # type: ignore[union-attr]
    offset1 = ctx.inst_field(type(inst).offset1)  # type: ignore[union-attr]
    offset = (offset1 << _c(8)) | offset0  # DS offset is 16-bit: (offset1 << 8) | offset0
    saddr_reg = None
  elif isinstance(inst, (ir4.VGLOBAL, ir4.VSCRATCH, ir4.VFLAT)):  # RDNA4: vaddr, vsrc, ioffset
    addr_reg = ctx.inst_field(type(inst).vaddr)
    vdata_reg = ctx.inst_field(type(inst).vsrc)
    vdst_reg = ctx.inst_field(type(inst).vdst)
    offset = ctx.inst_field_signed(type(inst).ioffset)
    offset0, offset1 = _c(0), _c(0)
    saddr_reg = ctx.inst_field(type(inst).saddr) if hasattr(type(inst), 'saddr') else None
  else:  # RDNA3: addr, data, offset
    addr_reg = ctx.inst_field(type(inst).addr)
    vdata_reg = ctx.inst_field(type(inst).data)  # type: ignore[union-attr]
    vdst_reg = ctx.inst_field(type(inst).vdst)
    offset = ctx.inst_field_signed(type(inst).offset)  # type: ignore[union-attr]
    offset0, offset1 = _c(0), _c(0)
    saddr_reg = ctx.inst_field(type(inst).saddr) if hasattr(type(inst), 'saddr') else None  # type: ignore[union-attr]

  # Data width from canonical_op_bits (32/64/96/128), default to 32 for untyped ops
  data_bits_mem = inst.canonical_op_bits.get('data', 32)
  is_atomic, glc = 'ATOMIC' in op_name, getattr(inst, 'glc', 0)
  has_data1 = is_lds and hasattr(inst, 'data1') and inst.data1 is not None
  data1_reg = ctx.inst_field(type(inst).data1) if is_lds else _c(0)  # type: ignore[union-attr]

  # DS_PERMUTE/DS_BPERMUTE: cross-lane VGPR access via pcode
  if is_lds and 'PERMUTE' in op_name:
    pcode = get_pcode(inst.op)
    srcs = {'ADDR': addr_reg, 'DATA0': vdata_reg, 'VDST': vdst_reg, 'OFFSET': offset,
            'EXEC': exec_mask.cast(dtypes.uint64), '_vgpr': ctx.vgpr, '_wave_size': ctx.wave_size}
    _, assigns = parse_pcode(pcode, srcs)
    stores = [ctx.vgpr.index(val[0]).store(val[1].cast(dtypes.uint32)) for dest, val in assigns if dest.startswith('VGPR[')]
    return UOp.sink(*stores, *ctx.inc_pc())

  def make_addr(lane: UOp) -> UOp:
    if is_lds:
      addr = ctx.rvgpr_dyn(addr_reg, lane)
      # Some DS pcode (e.g. DS_STORE_B16) uses MEM[ADDR] without adding OFFSET explicitly.
      # In those cases, add the instruction offset to ADDR here.
      if 'OFFSET' not in pcode: addr = addr + offset
      return addr
    offset64 = offset.cast(dtypes.uint64)
    # Dynamic saddr check: saddr < 124 means valid SGPR, otherwise use VGPR pair for address
    use_saddr = (saddr_reg < _c(124)) if saddr_reg is not None else UOp.const(dtypes.bool, False)
    if is_scratch:
      scratch_stride = ctx.rsgpr_dyn(_c(SCRATCH_STRIDE_IDX)).cast(dtypes.uint64)
      base = lane.cast(dtypes.uint64) * scratch_stride
      # SVE (Scratch VGPR Enable): when SVE=1, VADDR is used as offset; when SVE=0, VADDR is ignored
      sve = getattr(inst, 'sve', 0)
      vaddr = ctx.rvgpr_dyn(addr_reg, lane).cast(dtypes.uint64)
      addr_offset = vaddr if sve == 1 else UOp.const(dtypes.uint64, 0)
      # Add saddr value only if use_saddr is true (saddr < 124)
      saddr_contrib = use_saddr.where(ctx.rsgpr_dyn(saddr_reg).cast(dtypes.uint64), UOp.const(dtypes.uint64, 0)) \
        if saddr_reg is not None else UOp.const(dtypes.uint64, 0)
      return base + addr_offset + saddr_contrib + offset64
    # FLAT/GLOBAL: choose between SGPR base (saddr) or VGPR pair (addr) based on saddr validity
    saddr_base = _u64(ctx.rsgpr_dyn(saddr_reg), ctx.rsgpr_dyn(saddr_reg + _c(1))) if saddr_reg is not None else UOp.const(dtypes.uint64, 0)
    vaddr_base = _u64(ctx.rvgpr_dyn(addr_reg, lane), ctx.rvgpr_dyn(addr_reg + _c(1), lane))
    # When saddr is valid: base = saddr pair, vaddr is 32-bit offset; otherwise: base = 0, vaddr is 64-bit address
    base_addr = use_saddr.where(saddr_base + ctx.rvgpr_dyn(addr_reg, lane).cast(dtypes.uint64), vaddr_base)
    return base_addr + offset64

  def wmem(addr: UOp, val: UOp, active: UOp, data_bits: int = 32) -> UOp:
    if data_bits < 32:
      # Sub-dword LDS write: read-modify-write within the uint32 slot
      word_addr = addr >> addr_shift
      idx = mem.index(word_addr.valid(active))
      byte_pos = addr.cast(dtypes.uint32) & _c(3)
      byte_shift = byte_pos * _c(8)
      size_mask = _c(0xFF if data_bits == 8 else 0xFFFF)
      mask = size_mask << byte_shift
      new_word = (idx & (mask ^ _c(0xFFFFFFFF))) | ((val.cast(dtypes.uint32) & size_mask) << byte_shift)
      return idx.store(active.where(new_word, idx))
    idx = mem.index(addr >> addr_shift)
    return idx.store(active.where(val, idx.load()))

  def make_srcs(lane: UOp) -> dict:
    addr = make_addr(lane)
    if is_lds:
      if data_bits_mem == 128:
        data = {'DATA': ctx.rvgpr_dyn(vdata_reg, lane), 'DATA1': ctx.rvgpr_dyn(vdata_reg + _c(1), lane),
                'DATA2': ctx.rvgpr_dyn(vdata_reg + _c(2), lane), 'DATA3': ctx.rvgpr_dyn(vdata_reg + _c(3), lane)}
      elif data_bits_mem == 96:
        data = {'DATA': ctx.rvgpr_dyn(vdata_reg, lane), 'DATA1': ctx.rvgpr_dyn(vdata_reg + _c(1), lane),
                'DATA2': ctx.rvgpr_dyn(vdata_reg + _c(2), lane)}
      elif data_bits_mem <= 32:
        data = {'DATA': ctx.rvgpr_dyn(vdata_reg, lane), 'DATA2': ctx.rvgpr_dyn(data1_reg, lane) if has_data1 else UOp.const(dtypes.uint32, 0)}
      else:
        data = {'DATA': _u64(ctx.rvgpr_dyn(vdata_reg, lane), ctx.rvgpr_dyn(vdata_reg + _c(1), lane)),
                'DATA2': _u64(ctx.rvgpr_dyn(data1_reg, lane), ctx.rvgpr_dyn(data1_reg + _c(1), lane)) if has_data1 else UOp.const(dtypes.uint64, 0)}
      # RDNA3 uses ADDR/OFFSET, RDNA4 uses vgpr_a/offset (lowercase) + CalcDsAddr function
      return {'ADDR': addr, 'ADDR_BASE': addr, 'OFFSET': offset, 'OFFSET0': offset0, 'OFFSET1': offset1, '_lds': mem, 'laneId': lane,
              'vgpr_a': ctx.rvgpr_dyn(addr_reg, lane), 'offset': offset, 'offset0': offset0, 'offset1': offset1, **data}
    active = _lane_active(exec_mask, lane)
    # saddr < 124 means valid SGPR pair, otherwise use 0 (NULL means no saddr contribution)
    use_saddr = (saddr_reg < _c(124)) if saddr_reg is not None else UOp.const(dtypes.bool, False)
    saddr_raw = _u64(ctx.rsgpr_dyn(saddr_reg), ctx.rsgpr_dyn(saddr_reg + _c(1))) if saddr_reg is not None else UOp.const(dtypes.uint64, 0)
    saddr_base = use_saddr.where(saddr_raw, UOp.const(dtypes.uint64, 0))
    # Sign-extend offset to 64-bit for the final address calculation
    ioffset64 = offset.cast(dtypes.int64).cast(dtypes.uint64)
    # v_addr for CalcGlobalAddr: when saddr valid, use low 32 bits as offset; otherwise full 64-bit address. Include ioffset.
    vaddr_full = _u64(ctx.rvgpr_dyn(addr_reg, lane), ctx.rvgpr_dyn(addr_reg + _c(1), lane))
    vaddr_lo = ctx.rvgpr_dyn(addr_reg, lane).cast(dtypes.uint64)
    vaddr_base = use_saddr.where(vaddr_lo + ioffset64, vaddr_full + ioffset64)
    if is_atomic:
      atomic_data = _u64(ctx.rvgpr_dyn(vdata_reg, lane), ctx.rvgpr_dyn(vdata_reg + _c(1), lane)) \
        if data_bits_mem == 64 else ctx.rvgpr_dyn(vdata_reg, lane)
      return {'ADDR': addr, 'DATA': atomic_data, '_vmem': mem, '_active': active,
              'laneId': lane, 'v_addr': vaddr_base, 's_saddr': saddr_base}
    # acc bit: read/write ACCVGPR instead of VGPR for data operands
    _rvdata = (lambda r, l, *a: ctx.raccvgpr_dyn(r, l)) if use_acc else ctx.rvgpr_dyn
    vdata = _rvdata(vdata_reg, lane).cast(dtypes.uint64) if 'STORE' in op_name \
      else _rvdata(vdst_reg, lane) if 'D16' in op_name else UOp.const(dtypes.uint32, 0)
    if 'STORE' in op_name and data_bits_mem >= 64:
      vdata = vdata | (_rvdata(vdata_reg + _c(1), lane).cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))
    srcs = {'ADDR': addr, 'VDATA': vdata, '_vmem': mem, '_active': active,
            'laneId': lane, 'v_addr': vaddr_base, 's_saddr': saddr_base, 'SADDR': saddr_base, 'OFFSET': offset}
    for i in range(data_bits_mem // 32):
      srcs[f'VDATA{i}'] = _rvdata(vdata_reg + _c(i), lane) if 'STORE' in op_name else UOp.const(dtypes.uint32, 0)
    return srcs

  def make_stores(dest: str, val: UOp, lane: UOp, active: UOp, writes_return_data: bool) -> list[UOp]:
    # Parse bit width from dest format: MEM[...].b32 or RETURN_DATA[63:32].b64
    parts = dest.rsplit('.', 1)
    data_bits = int(parts[1][1:]) if len(parts) == 2 else 32
    if dest.startswith('MEM['):
      if is_lds or is_atomic:
        if data_bits < 32 and is_lds: return [wmem(val[0], val[1], active, data_bits)]
        return _write_val(data_bits, val[1], wmem, val[0], active, is_mem=True)
      if is_scratch: return _mem_store_bytes(mem, val[0], val[1], active, data_bits)
      return _mem_store(mem, val[0], val[1], active, 64, data_bits)
    if dest.startswith('RETURN_DATA') and writes_return_data:
      _wdata = (lambda r, v, l, e: ctx.waccvgpr_dyn(r, l, v, e)) if use_acc else (lambda r, v, l, e: ctx.wvgpr_dyn(r, l, v, e))
      if (m := re.match(r'RETURN_DATA\[(\d+)\s*:\s*(\d+)\]', dest)):
        bit_width, dword_idx = int(m.group(1)) - int(m.group(2)) + 1, int(m.group(2)) // 32
        return _write_val(bit_width, val, _wdata, vdst_reg + _c(dword_idx), lane, exec_mask)
      return _write_val(data_bits, val, _wdata, vdst_reg, lane, exec_mask)
    return []

  # DS-specific: check for 2ADDR pattern needing separate ranges
  if is_lds:
    dummy_lane = ctx.range()
    _, assigns = parse_pcode(pcode, make_srcs(dummy_lane))
    mem_assigns = [d for d, _ in assigns if d.startswith('MEM[')]
    mem_addrs = set(m.group(1) if (m := re.match(r'MEM\[([^\]]+)\]', d)) else d for d in mem_assigns)
    use_separate_ranges = (len(mem_addrs) > 1 or '2ADDR' in op_name) and 'STOREXCHG' not in op_name
    if use_separate_ranges:
      # Split assigns into MEM writes (stores) and RETURN_DATA writes (loads).
      # Stores to different addresses need separate lane ranges. Loads must share a single lane range so the
      # addr vgpr is read before any vdst write (hardware reads addr once, then writes all results).
      store_assigns = [(i, d) for i, (d, _) in enumerate(assigns) if d.startswith('MEM[')]
      load_assigns = [(i, d) for i, (d, _) in enumerate(assigns) if d.startswith('RETURN_DATA')]
      ended: list[UOp] = []
      for i, dest in store_assigns:
        lane = ctx.range()
        active = _lane_active(exec_mask, lane)
        _, lane_assigns = parse_pcode(pcode, make_srcs(lane))
        ended.extend(s.end(lane) for s in make_stores(dest, lane_assigns[i][1], lane, active, True))
      if load_assigns:
        lane = ctx.range()
        active = _lane_active(exec_mask, lane)
        _, lane_assigns = parse_pcode(pcode, make_srcs(lane))
        load_stores: list[UOp] = []
        for i, dest in load_assigns:
          load_stores.extend(make_stores(dest, lane_assigns[i][1], lane, active, True))
        if load_stores: ended.append(UOp.group(*load_stores).end(lane))
      return UOp.sink(*ended, *ctx.inc_pc())

  # Standard path: single lane range
  writes_return_data = '_RTN' in op_name or (is_lds and (op_name.startswith('DS_LOAD') or op_name.startswith('DS_READ'))) or bool(is_atomic and glc)
  lane = ctx.range()
  active = _lane_active(exec_mask, lane)
  pcode_vars, assigns = parse_pcode(pcode, make_srcs(lane))
  stores = [s for dest, val in assigns for s in make_stores(dest, val, lane, active, writes_return_data)]

  # FLAT/GLOBAL/SCRATCH: collect VDATA slices for loads
  if not is_lds and not is_atomic:
    _wdst = ctx.waccvgpr_dyn if use_acc else ctx.wvgpr_dyn
    for dword_idx, val in sorted(_collect_data_slices(assigns, 'VDATA', pcode_vars, op_name).items()):
      stores.append(_wdst(vdst_reg + _c(dword_idx), lane, val, exec_mask))

  return UOp.sink(UOp.group(*stores).end(lane), *ctx.inc_pc())

def _compile_mubuf(inst: irc.MUBUF, ctx: _Ctx) -> UOp:
  """CDNA MUBUF: linear buffer address = base + soffset + (stride * index) + vgpr_offset + inst_offset"""
  exec_mask, op_name = ctx.rexec(), _op_name(inst)
  use_acc, is_store, is_lds = bool(getattr(inst, 'acc', 0)), 'STORE' in op_name, bool(getattr(inst, 'lds', 0))
  n_dwords = 4 if 'X4' in op_name else 2 if 'X2' in op_name else 1

  # instruction fields
  vdata, vaddr = ctx.inst_field(type(inst).vdata), ctx.inst_field(type(inst).vaddr)
  srsrc, soffset = ctx.inst_field(type(inst).srsrc) * _c(4), ctx.inst_field(type(inst).soffset)
  offset, offen, idxen = ctx.inst_field(type(inst).offset), ctx.inst_field(type(inst).offen), ctx.inst_field(type(inst).idxen)

  # V# descriptor: base[0:1], num_records[2], stride=word3[13:0]
  base = _u64(ctx.rsgpr_dyn(srsrc), ctx.rsgpr_dyn(srsrc + _c(1))) & UOp.const(dtypes.uint64, 0xFFFFFFFFFFFF)
  num_records = ctx.rsgpr_dyn(srsrc + _c(2))
  stride = (ctx.rsgpr_dyn(srsrc + _c(3)) & _c(0x3FFF)).cast(dtypes.uint64)

  lane = ctx.range()
  active = _lane_active(exec_mask, lane)

  # soffset: sgpr if < 128, else inline constant
  soff = (soffset < _c(128)).where(ctx.rsgpr_dyn(soffset), soffset - _c(128)).cast(dtypes.uint64)
  # vaddr: index (if idxen) in vaddr, offset (if offen) in vaddr or vaddr+1
  index = idxen.ne(_c(0)).where(ctx.rvgpr_dyn(vaddr, lane), _c(0)).cast(dtypes.uint64)
  voff = offen.ne(_c(0)).where(ctx.rvgpr_dyn(idxen.ne(_c(0)).where(vaddr + _c(1), vaddr), lane), _c(0)).cast(dtypes.uint64)

  # buffer_offset for bounds check, final address
  buffer_offset = (stride * index + voff + offset.cast(dtypes.uint64)).cast(dtypes.uint32)
  in_bounds = active & buffer_offset.__lt__(num_records)
  addr = base + soff + buffer_offset.cast(dtypes.uint64)
  addr = in_bounds.where(addr, UOp.const(dtypes.uint64, 0))  # safe address when OOB
  mem = ctx.vmem

  stores: list[UOp] = []
  if is_lds and not is_store:
    # LDS load: buffer -> LDS (bypass VGPRs), LDS addr = M0[17:0] + lane * elem_size
    lds_base = ctx.rsgpr_dyn(_c(124)) & _c(0x3FFFF)
    lds_addr = lds_base + lane.cast(dtypes.uint32) * _c(n_dwords * 4)
    for i in range(n_dwords):
      word_addr = (addr + UOp.const(dtypes.uint64, i * 4)) >> UOp.const(dtypes.uint64, 2)
      val = in_bounds.where(mem.index(word_addr.cast(dtypes.int64)).load(), _c(0))
      lds_idx = (lds_addr + _c(i * 4)) >> _c(2)
      lds_slot = ctx.lds.index(lds_idx.valid(active))
      stores.append(lds_slot.store(active.where(val, lds_slot)))
  elif is_store:
    for i in range(n_dwords):
      word_addr = (addr + UOp.const(dtypes.uint64, i * 4)) >> UOp.const(dtypes.uint64, 2)
      idx = mem.index(word_addr.cast(dtypes.int64).valid(in_bounds))
      val = (ctx.raccvgpr_dyn if use_acc else ctx.rvgpr_dyn)(vdata + _c(i), lane)
      stores.append(idx.store(in_bounds.where(_to_u32(val), idx)))
  else:
    for i in range(n_dwords):
      word_addr = (addr + UOp.const(dtypes.uint64, i * 4)) >> UOp.const(dtypes.uint64, 2)
      val = in_bounds.where(mem.index(word_addr.cast(dtypes.int64).valid(in_bounds)).load(), _c(0))
      stores.append((ctx.waccvgpr_dyn if use_acc else ctx.wvgpr_dyn)(vdata + _c(i), lane, val, exec_mask))
  return UOp.sink(UOp.group(*stores).end(lane), *ctx.inc_pc())

# Dispatch table: instruction type -> handler function
_INST_HANDLERS: dict[type, Callable[..., UOp]] = {
  ir3.SOPP: _compile_sopp, ir3.SMEM: _compile_smem, ir3.SOP1: _compile_sop, ir3.SOP2: _compile_sop, ir3.SOPC: _compile_sop, ir3.SOPK: _compile_sop,
  ir3.VOP1: _compile_vop12, ir3.VOP1_SDST: _compile_vop12, ir3.VOP1_DPP16: _compile_vop12, ir3.VOP2: _compile_vop12, ir3.VOP2_DPP16: _compile_vop12,
  ir3.VOPC: _compile_vopc, ir3.VOPC_DPP16: _compile_vopc, ir3.VOP3: _compile_vop3, ir3.VINTERP: _compile_vinterp,
  ir3.VOP3_SDST: _compile_vop3, ir3.VOP3SD: _compile_vop3sd, ir3.VOP3P: _compile_vop3p, ir3.VOPD: _compile_vopd,
  ir3.DS: _compile_mem_op, ir3.FLAT: _compile_mem_op, ir3.GLOBAL: _compile_mem_op, ir3.SCRATCH: _compile_mem_op,
  # RDNA4 instruction classes
  ir4.SOPP: _compile_sopp, ir4.SMEM: _compile_smem, ir4.SOP1: _compile_sop, ir4.SOP2: _compile_sop, ir4.SOPC: _compile_sop, ir4.SOPK: _compile_sop,
  ir4.VOP1: _compile_vop12, ir4.VOP1_SDST: _compile_vop12, ir4.VOP1_DPP16: _compile_vop12, ir4.VOP2: _compile_vop12, ir4.VOP2_DPP16: _compile_vop12,
  ir4.VOPC: _compile_vopc, ir4.VOPC_DPP16: _compile_vopc, ir4.VOP3: _compile_vop3, ir4.VINTERP: _compile_vinterp,
  ir4.VOP3_SDST: _compile_vop3, ir4.VOP3SD: _compile_vop3sd, ir4.VOP3P: _compile_vop3p, ir4.VOPD: _compile_vopd,
  ir4.DS: _compile_mem_op, ir4.VFLAT: _compile_mem_op, ir4.VGLOBAL: _compile_mem_op, ir4.VSCRATCH: _compile_mem_op,
  # CDNA instruction classes
  irc.SOPP: _compile_sopp, irc.SMEM: _compile_smem, irc.SOP1: _compile_sop, irc.SOP2: _compile_sop, irc.SOPC: _compile_sop, irc.SOPK: _compile_sop,
  irc.VOP1: _compile_vop12, irc.VOP1_DPP16: _compile_vop12, irc.VOP2: _compile_vop12, irc.VOP2_DPP16: _compile_vop12,
  irc.VOPC: _compile_vopc, irc.VOP3: _compile_vop3,
  irc.VOP3_SDST: _compile_vop3, irc.VOP3SD: _compile_vop3sd, irc.VOP3P: _compile_vop3p,
  irc.VOP1_SDWA: _compile_sdwa, irc.VOP2_SDWA: _compile_sdwa, irc.VOP2_SDWA_SDST: _compile_sdwa, irc.VOPC_SDWA_SDST: _compile_sdwa,
  irc.DS: _compile_mem_op, irc.FLAT: _compile_mem_op, irc.GLOBAL: _compile_mem_op, irc.SCRATCH: _compile_mem_op,
  irc.MUBUF: _compile_mubuf,
}

# ═══════════════════════════════════════════════════════════════════════════════
# PROGRAM DECODE AND COMPILATION
# ═══════════════════════════════════════════════════════════════════════════════

_canonical_runner_cache: list[tuple[type, int, int, int, tuple[UOp, object]]] = []  # [(inst_type, base, mask, size, (prg, runtime)), ...]

@functools.cache
def _get_runner(inst_bytes: bytes, arch: str = "rdna3"):
  """Build and compile instruction to (prg, runtime). Cached by instruction bytes, with canonical dedup."""
  inst = decode_inst(inst_bytes, arch)
  inst_size = inst.size()
  inst_int = int.from_bytes(inst_bytes[:inst_size], 'little')

  # Check if instruction matches any cached canonical pattern (must also match instruction type to avoid variant conflicts)
  for inst_type, base, mask, size, entry in _canonical_runner_cache:
    if type(inst) is inst_type and inst_size == size and (inst_int & mask) == base: return entry

  # Look up handler by type, falling back to base classes for _LIT variants
  handler = _INST_HANDLERS.get(type(inst))
  if handler is None:
    for cls in type(inst).__mro__:
      if cls in _INST_HANDLERS:
        handler = _INST_HANDLERS[cls]
        break
  if handler is None: raise RuntimeError(f"[emu] unimplemented instruction type: {type(inst).__name__} {_op_name(inst)}")

  ctx = _Ctx(inst_size, _wave_size(arch))
  sink = handler(inst, ctx)
  base, mask, size = ctx.canonical_mask(inst_bytes)
  canonical_name = f"{_op_name(inst).lower()}_{base.to_bytes(size, 'little').hex()}"
  sink = sink.replace(arg=KernelInfo(name=canonical_name)).rtag(1)

  # NOTE: renderer output is not reproducible because of _MXCSRContext. PROFILE=0 prevents emulator instruction runners from polluting profiling.
  with Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", CAPTURE_PROCESS_REPLAY=0, PROFILE=0):
    prg = to_program(sink, Device['CPU'].renderer)
    runtime = get_runtime('CPU', prg)
  _canonical_runner_cache.append((type(inst), base, mask, size, (prg, runtime)))
  return prg, runtime

_BARRIER_OPS = {ir3.SOPPOp.S_BARRIER, irc.SOPPOp.S_BARRIER}
if hasattr(ir4.SOPPOp, 'S_BARRIER_WAIT'): _BARRIER_OPS.add(ir4.SOPPOp.S_BARRIER_WAIT)
_BARRIER_SOP1_OPS: set = set()
if hasattr(ir4.SOP1Op, 'S_BARRIER_SIGNAL'): _BARRIER_SOP1_OPS.add(ir4.SOP1Op.S_BARRIER_SIGNAL)
_BRANCH_OPS: set[int] = {op.value for op in (ir3.SOPPOp.S_BRANCH, ir3.SOPPOp.S_CBRANCH_SCC0, ir3.SOPPOp.S_CBRANCH_SCC1,
  ir3.SOPPOp.S_CBRANCH_VCCZ, ir3.SOPPOp.S_CBRANCH_VCCNZ, ir3.SOPPOp.S_CBRANCH_EXECZ, ir3.SOPPOp.S_CBRANCH_EXECNZ)}

def _decode_at(pc: int, arch: str):
  """Decode and compile instruction at absolute address pc. Returns (runner, decoded_inst)."""
  inst_bytes = bytes((ctypes.c_char * 16).from_address(pc).raw)
  inst = decode_inst(inst_bytes, arch)
  try: return _get_runner(bytes(inst_bytes[:inst.size() + 4]), arch), inst
  except Exception as e:
    try: inst_str = repr(inst)
    except Exception: inst_str = f"<{type(inst).__name__}>"
    raise RuntimeError(f"[emu] Failed to compile {inst_str}: {type(e).__name__}: {e}") from e

# ═══════════════════════════════════════════════════════════════════════════════
# WAVE STATE
# ═══════════════════════════════════════════════════════════════════════════════

# Inline float constants (as bit patterns) for GPU instructions
F32_INLINE = {240: 0x3f000000, 241: 0xbf000000, 242: 0x3f800000, 243: 0xbf800000,  # 0.5, -0.5, 1.0, -1.0
              244: 0x40000000, 245: 0xc0000000, 246: 0x40800000, 247: 0xc0800000, 248: 0x3e22f983}  # 2.0, -2.0, 4.0, -4.0, 1/(2*pi)

class WaveState:
  __slots__ = ('vgpr_buf', 'sgpr_buf', 'accvgpr_buf', '_vgpr_mv', '_sgpr_mv', 'n_lanes', 'wave_size')

  def __init__(self, n_lanes: int, wave_size: int = 32):
    self.n_lanes, self.wave_size = n_lanes, wave_size
    vgpr_size = 256 * wave_size
    self.vgpr_buf = Buffer('CPU', vgpr_size, dtypes.uint32).ensure_allocated()
    self.sgpr_buf = Buffer('CPU', SGPR_COUNT, dtypes.uint32).ensure_allocated()
    # CDNA (wave64) has separate ACCVGPR file; RDNA shares with VGPR
    if wave_size == 64:
      self.accvgpr_buf = Buffer('CPU', vgpr_size, dtypes.uint32).ensure_allocated()
      ctypes.memset(self.accvgpr_buf._buf.va_addr, 0, vgpr_size * 4)
    else:
      self.accvgpr_buf = self.vgpr_buf
    self._vgpr_mv = self.vgpr_buf.as_memoryview(force_zero_copy=True).cast('I')
    self._sgpr_mv = self.sgpr_buf.as_memoryview(force_zero_copy=True).cast('I')
    # Zero memory using ctypes memset (much faster than Python loops)
    ctypes.memset(self.vgpr_buf._buf.va_addr, 0, vgpr_size * 4)
    ctypes.memset(self.sgpr_buf._buf.va_addr, 0, SGPR_COUNT * 4)
    # Pre-populate inline constants at indices 128-255
    for i in range(65): self._write_sgpr(128 + i, i)  # 128-192: integers 0-64
    for i in range(16): self._write_sgpr(193 + i, (-(i + 1)) & MASK32)  # 193-208: -1 to -16
    for off, val in F32_INLINE.items(): self._write_sgpr(off, val)  # 240-248: float constants
    # EXEC mask: for 64-lane waves, set both EXEC_LO and EXEC_HI
    if wave_size == 64:
      self._write_sgpr(EXEC_LO.offset, (1 << min(n_lanes, 32)) - 1)
      self._write_sgpr(EXEC_LO.offset + 1, (1 << max(n_lanes - 32, 0)) - 1 if n_lanes > 32 else 0)
    else:
      self._write_sgpr(EXEC_LO.offset, (1 << n_lanes) - 1)
    self._write_sgpr(PC_LO_IDX, 0)
    self._write_sgpr(PC_HI_IDX, 0)

  def _write_sgpr(self, idx: int, val: int): self._sgpr_mv[idx] = val & MASK32
  def _read_sgpr(self, idx: int) -> int: return self._sgpr_mv[idx]
  def _write_vgpr(self, reg: int, lane: int, val: int): self._vgpr_mv[reg * self.wave_size + lane] = val & MASK32
  def _read_vgpr(self, reg: int, lane: int) -> int: return self._vgpr_mv[reg * self.wave_size + lane]

  @property
  def pc(self) -> int: return self._read_sgpr(PC_LO_IDX) | (self._read_sgpr(PC_HI_IDX) << 32)
  @pc.setter
  def pc(self, val: int):
    self._write_sgpr(PC_LO_IDX, val & MASK32)
    self._write_sgpr(PC_HI_IDX, (val >> 32) & MASK32)

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def _init_wave(lib: int, wave_start: int, total_threads: int, lx: int, ly: int, lz: int, args_ptr: int, rsrc2: int,
               scratch_size: int, arch: str, gidx: int, gidy: int, gidz: int, user_data: list[int]|None,
               wave_size: int = 32) -> WaveState:
  """Initialize a single wavefront and return WaveState."""
  n_lanes = min(wave_size, total_threads - wave_start)
  st = WaveState(n_lanes, wave_size)
  st.pc = lib
  if user_data:
    for i, val in enumerate(user_data): st._write_sgpr(i, val)
  else:
    st._write_sgpr(0, args_ptr & MASK32)
    st._write_sgpr(1, (args_ptr >> 32) & MASK32)
  if arch == "rdna4":
    # workgroup IDs only exist in ttmp registers, not normal SGPRs
    st._write_sgpr(ttmp[7].offset, (gidy & 0xFFFF) | ((gidz & 0xFFFF) << 16))
    st._write_sgpr(ttmp[9].offset, gidx)
  else:
    sgpr_idx = (rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT
    for enabled, gid in [(hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X, gidx),
                         (hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y, gidy),
                         (hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z, gidz)]:
      if rsrc2 & enabled:
        st._write_sgpr(sgpr_idx, gid)
        sgpr_idx += 1
  for lane in range(n_lanes):
    tid = wave_start + lane
    st._write_vgpr(0, lane, ((tid // (lx * ly)) << 20) | (((tid // lx) % ly) << 10) | (tid % lx))
  st._write_sgpr(SCRATCH_STRIDE_IDX, scratch_size)
  # Store HW register values at SGPR[SGPR_COUNT-16 .. SGPR_COUNT-1] for s_getreg_b32 emulation.
  # HW_ID (hwRegId=4): WAVE_ID[3:0], SIMD_ID[5:4], PIPE_ID[7:6], CU_ID[11:8], ...
  wave_idx = wave_start // wave_size  # wave index within this workgroup (0, 1, 2, 3 for 256 threads / 64 wave_size)
  hw_id = (wave_idx & 0xF) | ((wave_idx & 0x3) << 4)  # WAVE_ID = wave_idx, SIMD_ID = wave_idx % 4
  st._write_sgpr(SGPR_COUNT - 16 + 4, hw_id)  # HW_REGISTERS[4] = HW_ID
  return st

def run_asm(lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int, rsrc2: int = 0x19c,
            scratch_size: int = 0, arch: str = "rdna3", user_data: list[int]|None = None) -> int:
  """Execute AMD assembly program. scratch_size is private_segment_fixed_size from kernel descriptor (per-lane)."""
  from tinygrad.renderer.amd.dsl import Inst
  program: dict[int, tuple[Callable, list[int], bool, Inst]] = {}  # pc -> (fxn, globals, is_barrier, inst)
  lds_size = ((rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT) * 512
  total_threads = lx * ly * lz
  wave_size = _wave_size(arch)

  # Use Buffer objects with external_ptr=0 for vmem
  vmem_buf = Buffer('CPU', 1 << 40, dtypes.uint32, options=BufferSpec(external_ptr=0)).ensure_allocated()
  lds_buf = Buffer('CPU', max(lds_size // 4, 1), dtypes.uint32).ensure_allocated()
  scratch_buf = Buffer('CPU', scratch_size * wave_size, dtypes.uint8).ensure_allocated() if scratch_size else None

  # Initialize SQTT encoder — emits packets inline as instructions execute (only when profiling)
  if PROFILE:
    sqtt_emit, sqtt_finish, sqtt_finalize = _init_sqtt_encoder()

  def _ensure_compiled(pc: int) -> tuple[Callable, list[int], bool, Inst]:
    if pc not in program:
      prev_len = len(_canonical_runner_cache)
      (prg, runtime), inst = _decode_at(pc, arch)
      is_barrier = (isinstance(inst, (ir3.SOPP, ir4.SOPP, irc.SOPP)) and inst.op in _BARRIER_OPS) or \
                   (isinstance(inst, (ir4.SOP1,)) and inst.op in _BARRIER_SOP1_OPS)
      program[pc] = (runtime.fxn, prg.arg.globals, is_barrier, inst)
      if DEBUG >= 3:
        msg = f"[emu] PC={pc - lib}: {inst!r}"
        print(colored(msg, 'green') if len(_canonical_runner_cache) > prev_len else msg)
    return program[pc]

  # Set DAZ+FTZ during emulator execution, restore afterward to avoid breaking hypothesis tests
  # Only trace the first workgroup (like real HW traces one CU/SIMD), subsequent workgroups run but don't add to trace
  tracing = bool(PROFILE)

  with _MXCSRContext():
    for gidz in range(gz):
      for gidy in range(gy):
        for gidx in range(gx):
          # Initialize all wavefronts for this workgroup
          waves: list[tuple[WaveState, list]] = []
          for wave_start in range(0, total_threads, wave_size):
            st = _init_wave(lib, wave_start, total_threads, lx, ly, lz, args_ptr, rsrc2, scratch_size, arch, gidx, gidy, gidz, user_data,
                            wave_size)
            c_bufs = [ctypes.c_uint64(st.sgpr_buf._buf.va_addr), ctypes.c_uint64(st.vgpr_buf._buf.va_addr),
                      ctypes.c_uint64(vmem_buf._buf.va_addr), ctypes.c_uint64(lds_buf._buf.va_addr),
                      ctypes.c_uint64(scratch_buf._buf.va_addr if scratch_buf else 0),
                      ctypes.c_uint64(st.accvgpr_buf._buf.va_addr)]
            waves.append((st, c_bufs))

          # Execute wavefronts with barrier synchronization
          # Each wave runs until it hits s_barrier or s_endpgm. When all waves have stopped, release barrier waves.
          done = [False] * len(waves)
          for total_inst in range(10_000_000):
            if all(done): break
            for wi, (st, c_bufs) in enumerate(waves):
              if done[wi]: continue
              # Run this wave until barrier or endpgm
              for _ in range(1_000_000):
                pc = st.pc
                if pc == ENDPGM_PC:
                  done[wi] = True
                  if tracing: sqtt_finish(wi)
                  break
                fxn, globals_list, is_barrier, inst = _ensure_compiled(pc)
                if DEBUG >= 5: print(f"  exec gid=({gidx},{gidy},{gidz}) w={wi} PC={pc - lib}: {inst!r}", flush=True)
                fxn(*[c_bufs[g] for g in globals_list])
                if tracing:
                  inst_op = inst.op.value if hasattr(inst, 'op') else 0
                  sqtt_emit(wi, inst, (st.pc != ENDPGM_PC and st.pc != pc + inst.size()) if inst_op in _BRANCH_OPS else None)
                if is_barrier: break  # s_barrier hit: PC already advanced past it, pause this wave
              else: raise RuntimeError("exceeded 1M instructions in single wave, likely infinite loop")
            # All waves have either hit barrier or endpgm — release barrier waves for next round
          else: raise RuntimeError("exceeded 10M total scheduling rounds")
          tracing = False  # only trace the first workgroup

          # Reset LDS for next workgroup
          if lds_size > 0: ctypes.memset(lds_buf._buf.va_addr, 0, max(lds_size, 4))

  if PROFILE: sqtt_traces.append(sqtt_finalize())
  return 0
