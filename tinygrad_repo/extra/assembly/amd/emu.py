# RDNA3 emulator v2 - compiles pcode to UOps executed via tinygrad CPU backend
# Each instruction is compiled to a kernel that operates on buffers:
#   arg=0: sgpr - sgpr[0-127], inline constants[128-255], PC_LO=256, PC_HI=257, SCC=258, SCRATCH_STRIDE=259
#   arg=1: vgpr - vgpr[reg * 32 + lane]
#   arg=2: vmem - base address 0, INDEX offsets directly to host memory
#   arg=3: lds - local data share
#   arg=4: scratch - per-lane scratch memory
from __future__ import annotations
import ctypes, functools, re, platform, subprocess, tempfile
from typing import Any, Callable

# Set/restore DAZ+FTZ (denormals-are-zero + flush-to-zero) in MXCSR to match RDNA3 default float mode
# Only applied during emulator execution, restored afterward to avoid breaking hypothesis tests
@functools.cache
def _get_mxcsr_lib():
  if platform.machine() not in ('x86_64', 'AMD64'): return None
  try:
    src = b'''
unsigned int get_mxcsr(void){unsigned int m;__asm__ __volatile__("stmxcsr %0":"=m"(m));return m;}
void set_mxcsr(unsigned int m){__asm__ __volatile__("ldmxcsr %0"::"m"(m));}
'''
    with tempfile.NamedTemporaryFile(suffix='.so', delete=False) as f:
      subprocess.check_output(['clang', '-shared', '-O2', '-x', 'c', '-', '-o', f.name], input=src)
      lib = ctypes.CDLL(f.name)
      lib.get_mxcsr.restype = ctypes.c_uint32
      lib.set_mxcsr.argtypes = [ctypes.c_uint32]
      return lib
  except Exception: return None

class _MXCSRContext:
  """Context manager to set DAZ+FTZ during emulator execution and restore afterward."""
  __slots__ = ('_saved',)
  def __enter__(self):
    lib = _get_mxcsr_lib()
    if lib is None: return self
    self._saved = lib.get_mxcsr()
    lib.set_mxcsr(self._saved | 0x8040)  # DAZ (bit 6) + FTZ (bit 15)
    return self
  def __exit__(self, *args):
    lib = _get_mxcsr_lib()
    if lib is None or not hasattr(self, '_saved'): return
    lib.set_mxcsr(self._saved)

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.dtype import dtypes
from tinygrad.device import Buffer, BufferSpec
from tinygrad.runtime.autogen import hsa
from tinygrad.helpers import Context, DEBUG, colored
from tinygrad.engine.realize import get_runner

from extra.assembly.amd import decode_inst
from extra.assembly.amd.autogen.rdna3.str_pcode import PCODE as PCODE_RDNA3
from extra.assembly.amd.autogen.rdna4.str_pcode import PCODE as PCODE_RDNA4
from extra.assembly.amd.autogen.rdna3 import ins as ir3
from extra.assembly.amd.autogen.rdna4 import ins as ir4
from extra.assembly.amd.dsl import VCC_LO, EXEC_LO, SCC, ttmp
from extra.assembly.amd.autogen.common import Fmt, OpType
from extra.assembly.amd.pcode import parse_block, _FUNCS

MASK32 = 0xFFFFFFFF

def _c(val, dtype=dtypes.uint32): return UOp.const(dtype, val)

def _u64(lo: UOp, hi: UOp) -> UOp:
  """Combine two 32-bit UOps into a 64-bit UOp."""
  return lo.cast(dtypes.uint64) | (hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))

def _split64(val: UOp) -> tuple[UOp, UOp]:
  """Split a 64-bit value into (lo, hi) 32-bit values."""
  v64 = val.bitcast(dtypes.uint64) if val.dtype == dtypes.float64 else val.cast(dtypes.uint64) if val.dtype != dtypes.uint64 else val
  return v64.cast(dtypes.uint32), (v64 >> UOp.const(dtypes.uint64, 32)).cast(dtypes.uint32)

_SRC_MOD_TYPES = {16: (dtypes.uint16, dtypes.half, 0x7FFF), 64: (dtypes.uint64, dtypes.float64, 0x7FFFFFFFFFFFFFFF), 32: (dtypes.uint32, dtypes.float32, 0x7FFFFFFF)}
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
  # RDNA4 mappings (same VOP1/VOP2 targets, RDNA4 uses _NUM_ suffix for min/max)
  ir4.VOPDOp.V_DUAL_FMAC_F32: ir3.VOP2Op.V_FMAC_F32_E32, ir4.VOPDOp.V_DUAL_MUL_F32: ir3.VOP2Op.V_MUL_F32_E32,
  ir4.VOPDOp.V_DUAL_ADD_F32: ir3.VOP2Op.V_ADD_F32_E32, ir4.VOPDOp.V_DUAL_SUB_F32: ir3.VOP2Op.V_SUB_F32_E32,
  ir4.VOPDOp.V_DUAL_SUBREV_F32: ir3.VOP2Op.V_SUBREV_F32_E32, ir4.VOPDOp.V_DUAL_MAX_NUM_F32: ir3.VOP2Op.V_MAX_F32_E32,
  ir4.VOPDOp.V_DUAL_MIN_NUM_F32: ir3.VOP2Op.V_MIN_F32_E32, ir4.VOPDOp.V_DUAL_ADD_NC_U32: ir3.VOP2Op.V_ADD_NC_U32_E32,
  ir4.VOPDOp.V_DUAL_LSHLREV_B32: ir3.VOP2Op.V_LSHLREV_B32_E32, ir4.VOPDOp.V_DUAL_AND_B32: ir3.VOP2Op.V_AND_B32_E32,
  ir4.VOPDOp.V_DUAL_MOV_B32: ir3.VOP1Op.V_MOV_B32_E32, ir4.VOPDOp.V_DUAL_CNDMASK_B32: ir3.VOP2Op.V_CNDMASK_B32_E32,
  ir4.VOPDOp.V_DUAL_FMAAK_F32: ir3.VOP2Op.V_FMAAK_F32_E32, ir4.VOPDOp.V_DUAL_FMAMK_F32: ir3.VOP2Op.V_FMAMK_F32_E32,
}
WAVE_SIZE = 32
# Special registers stored after inline constants (256-259)
PC_LO_IDX, PC_HI_IDX, SCRATCH_STRIDE_IDX = 256, 257, 259
# SGPR buffer: 0-127 = SGPRs, 128-255 = inline constants, 256-259 = special registers
SGPR_COUNT, VGPR_SIZE = 260, 256 * 32

def _op_name(inst) -> str:
  if hasattr(inst, 'opx'): return f"{inst.opx.name}_{inst.opy.name}"  # VOPD has opx/opy not op
  return inst.op.name if hasattr(inst.op, 'name') else str(inst.op)

def _to_u32(val: UOp) -> UOp:
  if val.dtype == dtypes.uint32: return val
  if val.dtype.itemsize == 4: return val.bitcast(dtypes.uint32)  # same size: bitcast (float32->uint32)
  return val.cast(dtypes.uint32)  # different size: cast (bool, int16, etc)
def _lane_active(exec_mask: UOp, lane: UOp) -> UOp: return ((exec_mask >> lane.cast(dtypes.uint32)) & _c(1)).ne(_c(0))
def _hi16(v: UOp) -> UOp: return (v >> _c(16)) & _c(0xFFFF)
def _cond(cond, if_true, if_false):
  """Select between values based on condition (works with UOp or bool)."""
  return cond.where(if_true, if_false) if isinstance(cond, UOp) else if_true if cond else if_false
def _cond_hi16(cond, val: UOp) -> UOp: return _cond(cond, _hi16(val), val)
def _apply_opsel(val: UOp, sel_bit: int, opsel: int) -> UOp: return _hi16(val) if opsel & (1 << sel_bit) else val

def _set_lane_bit(old: UOp, lane: UOp, val: UOp, exec_mask: UOp) -> UOp:
  """Set/clear a single bit in a 32-bit mask based on lane index, respecting exec mask."""
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

def _apply_clamp(val: UOp, clmp: int | UOp) -> UOp:
  """Apply VOP3 clamp modifier: clamp float results to [0.0, 1.0] range."""
  if isinstance(clmp, int) and clmp == 0: return val
  if val.dtype not in (dtypes.float32, dtypes.half, dtypes.float64): return val
  zero, one = UOp.const(val.dtype, 0.0), UOp.const(val.dtype, 1.0)
  clamped = val.maximum(zero).minimum(one)
  return clmp.ne(_c(0)).where(clamped, val) if isinstance(clmp, UOp) else clamped

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
  return PCODE_RDNA4 if 'rdna4' in type(op).__module__ else PCODE_RDNA3

# Pcode parser
@functools.cache
def get_pcode(op) -> str:
  op_name = op.name
  pcode = _get_pcode_dict(op)[op]
  if op_name in _pcode_fixes: pcode = pcode.replace(*_pcode_fixes[op_name])
  if 'V_DIV_SCALE' in op_name:
    dt, exp_lim, ldexp_val = ('f32', '23', '64') if 'F32' in op_name else ('f64', '52', '128')
    for old, new in [(f'S2.{dt} / S1.{dt} == DENORM.{dt}', f'divWouldBeDenorm(S2.{dt}, S1.{dt})'), (f"1.0 / 64'F(S1.{dt}) == DENORM.f64", '0'),
                     (f'1.0 / S1.{dt} == DENORM.{dt}', '0'), (f'S1.{dt} == DENORM.{dt}', f'isDENORM(S1.{dt})'),
                     (f'D0.{dt} = NAN.{dt}', f'VCC = 0x1LL;\nD0.{dt} = NAN.{dt}'),
                     (f'elsif isDENORM(S1.{dt}) then\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})', f'elsif 1 == 0 then\nD0.{dt} = S0.{dt}'),
                     (f'elsif exponent(S2.{dt}) <= {exp_lim} then\n// Numerator is tiny\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})',
                      f'elsif exponent(S2.{dt}) <= {exp_lim} then\nVCC = 0x1LL;\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})'),
                     (f'elsif divWouldBeDenorm(S2.{dt}, S1.{dt}) then\nVCC = 0x1LL;\nif S0.{dt} == S2.{dt} then\n// Only scale the numerator\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nendif',
                      f'elsif divWouldBeDenorm(S2.{dt}, S1.{dt}) then\nVCC = 0x1LL;\nD0.{dt} = S0.{dt}'),
                     (f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nendif\nelsif', f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nelse\nD0.{dt} = S0.{dt}\nendif\nelsif')]:
      pcode = pcode.replace(old, new)
    lines = pcode.rstrip().split('\n')
    for i in range(len(lines) - 1, -1, -1):
      if lines[i].strip() == 'endif': lines.insert(i, f'else\nD0.{dt} = S0.{dt}'); break
    pcode = '\n'.join(lines) + f';\nif isDENORM(S1.{dt}) then\nD0.{dt} = NAN.{dt}\nendif'
    pcode = pcode.replace('VCC = 0x0LL', 'VCC.u64[laneId] = 0').replace('VCC = 0x1LL', 'VCC.u64[laneId] = 1')
  return pcode

def parse_pcode(pcode: str, srcs: dict[str, UOp] | None = None) -> tuple[dict, list[tuple[str, UOp]]]:
  vars: dict = srcs.copy() if srcs else {}
  assigns: list[tuple[str, UOp]] = []
  raw_lines = [l.strip().rstrip(';') for l in pcode.split('\n') if l.strip() and not l.strip().startswith('//')]
  # TODO: pcode.py should tokenize full pcode string instead of line-by-line, then this hack can be removed
  lines: list[str] = []
  for l in raw_lines:
    if lines and lines[-1].endswith('&&'): lines[-1] = lines[-1] + ' ' + l
    else: lines.append(l)
  _, final, _ = parse_block(lines, 0, vars, assigns=assigns)
  sliced = set(d.split('[')[0] for d, _ in assigns if '[' in d)
  for var, val in final.items():
    if var in ['D0', 'SCC', 'VCC', 'EXEC', 'PC', 'RETURN_DATA', 'VDATA'] and isinstance(val, UOp):
      if var in sliced and not any(re.match(rf'{var}\.\w+\s*=', l) for l in lines): continue
      for l in lines:
        if (m := re.match(rf'{var}\.(\w+(?:\[\w+\])?)', l)): assigns.append((f'{var}.{m.group(1)}', val)); break
      else: assigns.append((var, val))
  return vars, assigns

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
  idx = mem.index(word_addr.cast(dtypes.int), active)
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
  next_idx = mem.index((word_addr + UOp.const(adt, 1)).cast(dtypes.int), active & is_cross)
  cross_word1 = (next_idx & _c(0xFFFFFF00)) | ((val_u32 >> _c(8)) & _c(0xFF))
  return [store0, next_idx.store((active & is_cross).where(cross_word1, next_idx))]

def _mem_store_bytes(mem: UOp, addr: UOp, val: UOp, active: UOp, data_bits: int = 32) -> list[UOp]:
  """Store to byte-addressable memory (scratch). addr is byte offset, mem is uint8 buffer."""
  stores = []
  val_u32 = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
  for i in range(data_bits // 8):
    byte_val = (val_u32 >> UOp.const(dtypes.uint32, i * 8)) & UOp.const(dtypes.uint32, 0xFF)
    stores.append(mem.index((addr + UOp.const(dtypes.uint64, i)).cast(dtypes.int), active).store(byte_val.cast(dtypes.uint8)))
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
  __slots__ = ('inst_size', 'dyn_fields', '_axis_id')
  sgpr = UOp(Ops.PARAM, dtypes.uint32.ptr(SGPR_COUNT), arg=0)
  vgpr = UOp(Ops.PARAM, dtypes.uint32.ptr(VGPR_SIZE), arg=1)
  vmem = UOp(Ops.PARAM, dtypes.uint32.ptr(1 << 46), arg=2)
  lds = UOp(Ops.PARAM, dtypes.uint32.ptr(16384), arg=3)
  scratch = UOp(Ops.PARAM, dtypes.uint8.ptr(1 << 30), arg=4)

  def __init__(self, inst_size: int):
    self.inst_size, self._axis_id = inst_size, 0
    self.dyn_fields: list[tuple[int, int]] = []  # (lo, hi) of fields read dynamically

  def range(self, n: int = 32) -> UOp:
    """Create a lane range UOp with unique axis ID."""
    self._axis_id += 1
    return UOp.range(n, self._axis_id, AxisType.LOOP, dtype=dtypes.int)

  def unroll_lanes(self, get_lane_bit, exec_mask: UOp, apply_exec: bool = True) -> UOp:
    """Combine 32 lane bits into a 32-bit mask using RANGE+REDUCE."""
    lane = self.range()
    bit = get_lane_bit(lane).cast(dtypes.uint32) << lane.cast(dtypes.uint32)
    result = bit.reduce(lane, arg=Ops.ADD)
    return result & exec_mask if apply_exec else result

  def inst_word(self, dword_idx: int) -> UOp:
    """Read instruction dword from vmem at PC + dword_idx*4."""
    pc = self.rpc()
    addr = pc if dword_idx == 0 else pc + UOp.const(dtypes.uint64, dword_idx * 4)
    return self.vmem.index((addr >> UOp.const(dtypes.uint64, 2)).cast(dtypes.int), ptr=True).load()

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

  # Dynamic register access (takes UOp index instead of int)
  def rsgpr_dyn(self, reg: UOp, valid: UOp | None = None) -> UOp:
    """Read SGPR with dynamic register index."""
    return self.sgpr.index(reg.cast(dtypes.int), valid, ptr=True).load() if valid is not None else self.sgpr.index(reg.cast(dtypes.int), ptr=True).load()

  def wsgpr_dyn(self, reg: UOp, val: UOp) -> UOp:
    """Write SGPR with dynamic register index. Writes to NULL (124) are discarded."""
    return self.sgpr.index(reg.cast(dtypes.int), reg.ne(_c(124))).store(val.cast(dtypes.uint32))

  def rvgpr_dyn(self, reg: UOp, lane: UOp, valid: UOp | None = None) -> UOp:
    """Read VGPR with dynamic register index."""
    idx = reg.cast(dtypes.int) * _c(32, dtypes.int) + lane.cast(dtypes.int)
    return self.vgpr.index(idx, valid, ptr=True).load() if valid is not None else self.vgpr.index(idx, ptr=True).load()

  def wvgpr_dyn(self, reg: UOp, lane: UOp, val: UOp, exec_mask: UOp, after: UOp | None = None) -> UOp:
    """Write VGPR with dynamic register index."""
    buf = self.vgpr.after(after) if after is not None else self.vgpr
    offset = reg.cast(dtypes.int) * _c(32, dtypes.int) + lane.cast(dtypes.int)
    return buf.index(offset, _lane_active(exec_mask, lane)).store(val.cast(dtypes.uint32))

  def rsrc_dyn(self, off: UOp, lane: UOp | None, bits: int = 32, literal: UOp | None = None, is_f64: bool = False) -> UOp:
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
      if bits == 16:  # Float constants: cast F32 to F16
        scalar_val = is_float_const.where(scalar_val.bitcast(dtypes.float32).cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32), scalar_val)

    return is_vgpr.where(vgpr_val, scalar_val) if lane is not None else scalar_val

  def rpc(self) -> UOp:
    """Read PC as 64-bit byte address."""
    # Index at PC_LO, then cast to uint64 ptr and load
    return self.sgpr.index(_c(PC_LO_IDX, dtypes.int), ptr=True).cast(dtypes.uint64.ptr(SGPR_COUNT // 2)).load()

  def inc_pc(self) -> list[UOp]:
    """Increment PC by instruction size in bytes. Returns [store]."""
    new_pc = self.rpc() + UOp.const(dtypes.uint64, self.inst_size)
    return [self.sgpr.index(_c(PC_LO_IDX, dtypes.int), ptr=True).cast(dtypes.uint64.ptr(SGPR_COUNT // 2)).store(new_pc)]

  def scalar_stores(self, assigns: list[tuple[str, UOp]], sdst_reg: UOp, sdst_size: int = 1) -> list[UOp]:
    """Generate stores for scalar assigns with dynamic destination register (D0, SCC, EXEC, VCC)."""
    stores: list[UOp] = []
    for dest, val in assigns:
      if dest.startswith('D0'):
        if sdst_size == 2:
          lo, hi = _split64(val)
          stores.extend([self.wsgpr_dyn(sdst_reg, lo), self.wsgpr_dyn(sdst_reg + _c(1), hi)])
        else: stores.append(self.wsgpr_dyn(sdst_reg, _to_u32(val)))
      elif dest.startswith('SCC'): stores.append(self.wsgpr_dyn(_c(SCC.offset), _to_u32(val)))
      elif dest.startswith('EXEC'): stores.append(self.wsgpr_dyn(_c(EXEC_LO.offset), _to_u32(val)))
      elif dest.startswith('VCC'): stores.append(self.wsgpr_dyn(_c(VCC_LO.offset), _to_u32(val)))
    return stores

  def compile_sop_pcode(self, op, srcs: dict[str, UOp], sdst_reg: UOp, sdst_size: int) -> UOp:
    """Compile a scalar instruction with dynamic destination register."""
    pcode = get_pcode(op)
    srcs.update({'VCC': self.rsgpr_dyn(_c(VCC_LO.offset)), 'EXEC': self.rsgpr_dyn(_c(EXEC_LO.offset)), 'SCC': self.rsgpr_dyn(_c(SCC.offset))})
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
    exec_lo = self.rsgpr_dyn(_c(EXEC_LO.offset))
    srcs = {
      'SRC0': src0_reg, 'VDST': vdst_off, 'EXEC_LO': exec_lo, 'EXEC': exec_lo.cast(dtypes.uint64), '_vgpr': self.vgpr,
      'S0': self.rsrc_dyn(src0_off, _c(0, dtypes.int)) if 'WRITELANE' in op_name else src0_reg,
      'S1': self.rsrc_dyn(src1_off, _c(0, dtypes.int)) if src1_off is not None else _c(0),
      'S2': self.rsrc_dyn(src2_off, _c(0, dtypes.int)) if src2_off is not None else _c(0),
    }
    _, assigns = parse_pcode(pcode, srcs)
    stores = []
    for dest, val in assigns:
      if dest.startswith('D0'): stores.append(self.wsgpr_dyn(vdst_off, val.cast(dtypes.uint32)))
      elif dest.startswith('VGPR['): stores.append(self.vgpr.index(val[0].cast(dtypes.int)).store(val[1].cast(dtypes.uint32)))
    return UOp.sink(*stores, *self.inc_pc())

  def compile_vop_pcode(self, op, srcs: dict[str, UOp], lane: UOp, vdst_reg: UOp, exec_mask: UOp,
                        opsel_dst_hi: bool | UOp = False, sdst_reg: int | None = None, clmp: int | UOp = 0) -> UOp:
    """Compile VOP instruction. Returns sink with stores and inc_pc."""
    pcode = get_pcode(op)
    vcc_reg = sdst_reg if sdst_reg is not None else VCC_LO.offset
    if 'VCC' not in srcs: srcs['VCC'] = self.rsgpr_dyn(_c(vcc_reg))
    srcs.update({'EXEC': exec_mask, 'SCC': self.rsgpr_dyn(_c(SCC.offset)), 'laneId': lane,
                 'ROUND_MODE': _c(0), 'ROUND_TOWARD_ZERO': _c(0)})  # rounding mode: 0=RNE, RTZ constant
    _, assigns = parse_pcode(pcode, srcs)

    raw_stores: list = []
    vcc_val, exec_val = None, None
    for dest, val in assigns:
      if 'D0' in dest and '[laneId]' in dest:
        raw_stores.append(('vcc', self.wsgpr_dyn(_c(VCC_LO.offset), _set_lane_bit(self.rsgpr_dyn(_c(VCC_LO.offset)), lane, val, exec_mask))))
      elif dest.startswith('D0'):
        if (slice_match := re.match(r'D0\[(\d+)\s*:\s*(\d+)\]', dest)):
          hi_bit, lo_bit = int(slice_match.group(1)), int(slice_match.group(2))
          if hi_bit != 31 or lo_bit != 0:
            width, slice_mask = hi_bit - lo_bit + 1, (1 << (hi_bit - lo_bit + 1)) - 1
            val_bits = val.bitcast(dtypes.uint16).cast(dtypes.uint32) if val.dtype == dtypes.half else \
                       val.cast(dtypes.uint32) if val.dtype in (dtypes.uint16, dtypes.int16) else val.cast(dtypes.uint32) & UOp.const(dtypes.uint32, slice_mask)
            raw_stores.append(('vgpr_slice', (lo_bit, width, val_bits)))
            continue
        val = _apply_clamp(val, clmp)
        if val.dtype in (dtypes.uint64, dtypes.int64, dtypes.float64):
          lo, hi = _split64(val)
          raw_stores.extend([('vgpr', self.wvgpr_dyn(vdst_reg, lane, lo, exec_mask)), ('vgpr', self.wvgpr_dyn(vdst_reg + _c(1), lane, hi, exec_mask))])
        elif val.dtype in (dtypes.half, dtypes.uint16, dtypes.int16):
          result, old_val = _val_to_u32(val), self.rvgpr_dyn(vdst_reg, lane)
          hi_result = (old_val & UOp.const(dtypes.uint32, 0xFFFF)) | (result << UOp.const(dtypes.uint32, 16))
          lo_result = (old_val & UOp.const(dtypes.uint32, 0xFFFF0000)) | (result & UOp.const(dtypes.uint32, 0xFFFF))
          result = opsel_dst_hi.where(hi_result, lo_result) if isinstance(opsel_dst_hi, UOp) else hi_result if opsel_dst_hi else lo_result
          raw_stores.append(('vgpr', self.wvgpr_dyn(vdst_reg, lane, result, exec_mask)))
        else: raw_stores.append(('vgpr', self.wvgpr_dyn(vdst_reg, lane, _val_to_u32(val), exec_mask)))
      elif dest.startswith('VCC'): vcc_val = val
      elif dest.startswith('EXEC'): exec_val = val
      elif dest.startswith('SCC'): raw_stores.append(('scc', self.wsgpr_dyn(_c(SCC.offset), _to_u32(val))))

    stores, lane_stores, scalar_stores = [], [s for t, s in raw_stores if t == 'vgpr'], [s for t, s in raw_stores if t == 'scc']
    slice_stores = [s for t, s in raw_stores if t == 'vgpr_slice']
    if slice_stores:
      result = self.rvgpr_dyn(vdst_reg, lane)
      for lo_bit, width, val_bits in slice_stores:
        mask = UOp.const(dtypes.uint32, ((1 << width) - 1) << lo_bit)
        result = (result & (mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))) | (val_bits << UOp.const(dtypes.uint32, lo_bit))
      lane_stores.append(self.wvgpr_dyn(vdst_reg, lane, result, exec_mask))
    if lane_stores: stores.append(UOp.sink(*lane_stores).end(lane))
    for mask_val, reg in [(vcc_val, vcc_reg), (exec_val, EXEC_LO.offset)]:
      if mask_val is None: continue
      get_bit = lambda l, v=mask_val: (_to_u32(v.substitute({lane: l})) & _c(1)).cast(dtypes.uint32)
      stores.append(self.wsgpr_dyn(_c(reg), self.unroll_lanes(get_bit, exec_mask, apply_exec=False)))
    stores.extend(scalar_stores)
    return UOp.sink(*stores, *self.inc_pc())

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

def _compile_sopp(inst: ir3.SOPP | ir4.SOPP, ctx: _Ctx) -> UOp:
  simm16 = ctx.inst_field_signed(type(inst).simm16).cast(dtypes.int16)
  if inst.op in (ir3.SOPPOp.S_ENDPGM, ir4.SOPPOp.S_ENDPGM):
    return UOp.sink(ctx.wsgpr_dyn(_c(PC_LO_IDX), UOp.const(dtypes.uint32, 0xFFFFFFFF)),
                          ctx.wsgpr_dyn(_c(PC_HI_IDX), UOp.const(dtypes.uint32, 0xFFFFFFFF)))
  if inst.op in (ir3.SOPPOp.S_NOP, ir4.SOPPOp.S_NOP): return UOp.sink(*ctx.inc_pc())  # S_NOP is a no-op
  # NOTE: we ignore SOPPs without PCODE
  if inst.op in _get_pcode_dict(inst.op):
    pcode = get_pcode(inst.op)
    pc_bytes = ctx.rpc()  # PC is already 64-bit byte address
    vcc, exec_lo = ctx.rsgpr_dyn(_c(VCC_LO.offset)), ctx.rsgpr_dyn(_c(EXEC_LO.offset))
    srcs = {'PC': pc_bytes.cast(dtypes.int64), 'SIMM16': simm16, 'SCC': ctx.rsgpr_dyn(_c(SCC.offset)), 'VCC': vcc,
            'VCCZ': vcc.eq(UOp.const(dtypes.uint32, 0)).cast(dtypes.uint32), 'EXECZ': exec_lo.eq(UOp.const(dtypes.uint32, 0)).cast(dtypes.uint32)}
    for dest, val in parse_pcode(pcode, srcs)[1]:
      if dest == 'PC' or dest.startswith('PC.'):
        lo, hi = _split64(val.cast(dtypes.uint64))
        return UOp.sink(ctx.wsgpr_dyn(_c(PC_LO_IDX), lo), ctx.wsgpr_dyn(_c(PC_HI_IDX), hi))
  return UOp.sink(*ctx.inc_pc())

def _compile_smem(inst: ir3.SMEM | ir4.SMEM, ctx: _Ctx) -> UOp:
  # Cache invalidation instructions are no-ops in the emulator (we don't model caches)
  cache_inv_ops = [ir3.SMEMOp.S_GL1_INV, ir3.SMEMOp.S_DCACHE_INV, ir4.SMEMOp.S_DCACHE_INV]
  if hasattr(ir4.SMEMOp, 'S_GL1_INV'): cache_inv_ops.append(ir4.SMEMOp.S_GL1_INV)
  if inst.op in cache_inv_ops:
    return UOp.sink(*ctx.inc_pc())
  # Dynamic sbase field (bits 5:0) - SGPR pair, field value * 2 = register offset
  sbase = ctx.inst_field(type(inst).sbase) * _c(2)
  # Dynamic sdata field (bits 12:6) - destination SGPR
  sdata_reg = ctx.inst_field(type(inst).sdata)
  # RDNA4 uses 'ioffset', RDNA3 uses 'offset' - use type(inst) to get correct field
  offset_field = type(inst).ioffset if hasattr(type(inst), 'ioffset') else type(inst).offset
  offset = ctx.inst_field_signed(offset_field)  # signed immediate
  # Dynamic soffset field - SGPR for additional offset (NULL=124 reads as 0)
  soffset = ctx.inst_field(type(inst).soffset)
  addr = _u64(ctx.rsgpr_dyn(sbase), ctx.rsgpr_dyn(sbase + _c(1))) + offset.cast(dtypes.uint64) + ctx.rsgpr_dyn(soffset).cast(dtypes.uint64)
  _SMEM_NDWORDS = {ir3.SMEMOp.S_LOAD_B32: 1, ir3.SMEMOp.S_LOAD_B64: 2, ir3.SMEMOp.S_LOAD_B128: 4,
    ir3.SMEMOp.S_LOAD_B256: 8, ir3.SMEMOp.S_LOAD_B512: 16, ir4.SMEMOp.S_LOAD_B32: 1, ir4.SMEMOp.S_LOAD_B64: 2,
    ir4.SMEMOp.S_LOAD_B96: 3, ir4.SMEMOp.S_LOAD_B128: 4, ir4.SMEMOp.S_LOAD_B256: 8, ir4.SMEMOp.S_LOAD_B512: 16}
  ndwords = _SMEM_NDWORDS[inst.op]
  stores = [ctx.wsgpr_dyn(sdata_reg + _c(i), ctx.vmem.index((addr + UOp.const(dtypes.uint64, i * 4) >> UOp.const(dtypes.uint64, 2)).cast(dtypes.int)))
            for i in range(ndwords)]
  return UOp.sink(*stores, *ctx.inc_pc())

def _compile_sop(inst: ir3.SOP1 | ir3.SOP2 | ir3.SOPC | ir3.SOPK | ir4.SOP1 | ir4.SOP2 | ir4.SOPC | ir4.SOPK, ctx: _Ctx) -> UOp:
  bits = inst.canonical_op_bits
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None

  if isinstance(inst, (ir3.SOPK, ir4.SOPK)):
    sdst_off = ctx.inst_field(type(inst).sdst)
    simm16 = ctx.inst_field(type(inst).simm16)
    # Sign-extend simm16
    simm16_sext = simm16.cast(dtypes.int16).cast(dtypes.int32)
    srcs = {'S0': ctx.rsgpr_dyn(sdst_off), 'SIMM16': simm16_sext, 'D0': ctx.rsgpr_dyn(sdst_off)}
    dst_off, dst_size = sdst_off, 1
  elif isinstance(inst, (ir3.SOP1, ir4.SOP1)):
    sdst_off = ctx.inst_field(type(inst).sdst)
    ssrc0_off = ctx.inst_field(type(inst).ssrc0)
    srcs = {'S0': ctx.rsrc_dyn(ssrc0_off, None, bits['s0'], literal)}
    dst_off, dst_size = sdst_off, bits['d'] // 32
  elif isinstance(inst, (ir3.SOP2, ir4.SOP2)):
    sdst_off = ctx.inst_field(type(inst).sdst)
    ssrc0_off = ctx.inst_field(type(inst).ssrc0)
    ssrc1_off = ctx.inst_field(type(inst).ssrc1)
    srcs = {'S0': ctx.rsrc_dyn(ssrc0_off, None, bits['s0'], literal),
            'S1': ctx.rsrc_dyn(ssrc1_off, None, bits['s1'], literal)}
    if literal is not None: srcs['SIMM32'] = literal
    dst_off, dst_size = sdst_off, bits['d'] // 32
  elif isinstance(inst, (ir3.SOPC, ir4.SOPC)):
    ssrc0_off = ctx.inst_field(type(inst).ssrc0)
    ssrc1_off = ctx.inst_field(type(inst).ssrc1)
    srcs = {'S0': ctx.rsrc_dyn(ssrc0_off, None, bits['s0'], literal),
            'S1': ctx.rsrc_dyn(ssrc1_off, None, bits['s1'], literal)}
    dst_off, dst_size = _c(0), 0  # SOPC writes to SCC, not sdst
  else:
    raise RuntimeError(f"unknown SOP type: {type(inst).__name__}")

  return ctx.compile_sop_pcode(inst.op, srcs, dst_off, dst_size)

def _compile_vop12(inst: ir3.VOP1 | ir3.VOP1_SDST | ir3.VOP2 | ir4.VOP1 | ir4.VOP1_SDST | ir4.VOP2, ctx: _Ctx) -> UOp:
  op_name = _op_name(inst)
  if op_name in ('V_READFIRSTLANE_B32_E32', 'V_PERMLANE64_B32_E32'): return ctx.compile_lane_pcode(inst.op, inst)
  lane, exec_mask, bits = ctx.range(), ctx.rsgpr_dyn(_c(EXEC_LO.offset)), inst.canonical_op_bits
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None
  vdst_reg = ctx.inst_field(type(inst).vdst)
  write_hi_half = bits['d'] == 16 and (vdst_reg >= _c(128))
  if isinstance(write_hi_half, UOp): vdst_reg = write_hi_half.where(vdst_reg - _c(128), vdst_reg)
  elif write_hi_half: vdst_reg -= 128
  if isinstance(inst, (ir3.VOP1, ir4.VOP1)):
    # Handle VOP1 hi-half source operand (src0 >= v[128] for 16-bit ops)
    src0_off = ctx.inst_field(type(inst).src0)
    s0 = ctx.rsrc_dyn(src0_off, lane, bits['s0'], literal)
    if bits['s0'] == 16:
      src0_hi = src0_off >= _c(384)
      # Only compute hi-half when src0_off >= 384, use guarded index to prevent OOB access
      src0_reg = src0_hi.where(src0_off - _c(384), _c(0))
      s0 = src0_hi.where(_hi16(ctx.rvgpr_dyn(src0_reg, lane)), s0)
    srcs = {'S0': s0}
  else:
    vsrc1_reg = ctx.inst_field(type(inst).vsrc1)
    vsrc1_hi = bits['s0'] == 16 and (vsrc1_reg >= _c(128))
    vsrc1_actual = _cond(vsrc1_hi, vsrc1_reg - _c(128), vsrc1_reg)
    s1 = _cond_hi16(vsrc1_hi, ctx.rvgpr_dyn(vsrc1_actual, lane))
    d0 = _cond_hi16(write_hi_half, ctx.rvgpr_dyn(vdst_reg, lane))  # FMAC/FMAMK hi-half dest needs hi-half accumulator
    # Handle VOP2 hi-half src0 operand (src0 >= v[128] for 16-bit ops)
    src0_off = ctx.inst_field(type(inst).src0)
    s0 = ctx.rsrc_dyn(src0_off, lane, bits['s0'], literal)
    if bits['s0'] == 16:
      src0_hi = src0_off >= _c(384)
      # Only compute hi-half when src0_off >= 384, use guarded index to prevent OOB access
      src0_reg = src0_hi.where(src0_off - _c(384), _c(0))
      s0 = src0_hi.where(_hi16(ctx.rvgpr_dyn(src0_reg, lane)), s0)
    srcs = {'S0': s0, 'S1': s1, 'D0': d0}
    if inst.op in (ir3.VOP2Op.V_FMAAK_F32_E32, ir3.VOP2Op.V_FMAMK_F32_E32, ir3.VOP2Op.V_FMAAK_F16_E32,
                   ir3.VOP2Op.V_FMAMK_F16_E32):
      assert literal is not None
      srcs['SIMM32'] = literal
  return ctx.compile_vop_pcode(inst.op, srcs, lane, vdst_reg, exec_mask, opsel_dst_hi=write_hi_half)

def _compile_vopc(inst: ir3.VOPC | ir3.VOP3 | ir4.VOPC | ir4.VOP3, ctx: _Ctx, opsel: int = 0, abs_bits: int = 0, neg_bits: int = 0) -> UOp:
  exec_mask, op_name, bits = ctx.rsgpr_dyn(_c(EXEC_LO.offset)), _op_name(inst), inst.canonical_op_bits
  is_cmpx, is_vopc = 'CMPX' in op_name, hasattr(inst, 'vsrc1')  # is_vopc: e32 vs e64

  # Handle both VOPC (vsrc1) and VOP3 (src1) instruction formats - read operands dynamically
  if is_vopc:
    src0_off = ctx.inst_field(type(inst).src0)
    vsrc1_off = ctx.inst_field(type(inst).vsrc1)
    # For 16-bit ops, vsrc1 >= 128 means hi-half of v[vsrc1-128]
    if bits['s0'] == 16:
      vsrc1_hi = vsrc1_off >= _c(128)
      src1_off = _c(256) + vsrc1_hi.where(vsrc1_off - _c(128), vsrc1_off)
    else:
      vsrc1_hi = False
      src1_off = _c(256) + vsrc1_off
  else:
    src0_off = ctx.inst_field(type(inst).src0)
    src1_off = ctx.inst_field(type(inst).src1)
    dst_off = ctx.inst_field(type(inst).vdst)
    vsrc1_hi = False
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None

  is_float, is_f64, pcode = any(x in op_name for x in ('_F32', '_F64', '_F16')), '_F64' in op_name, get_pcode(inst.op)
  def get_cmp_bit(lane) -> UOp:
    lc = lane.cast(dtypes.int) if isinstance(lane, UOp) else _c(lane, dtypes.int)
    s0 = ctx.rsrc_dyn(src0_off, lc, bits['s0'], literal, is_f64)
    s1 = _cond_hi16(vsrc1_hi, ctx.rsrc_dyn(src1_off, lc, bits['s1'], literal, is_f64)) if bits['s0'] == 16 else ctx.rsrc_dyn(src1_off, lc, bits['s1'], literal, is_f64)
    if bits['s0'] == 16 and opsel: s0, s1 = _apply_opsel(s0, 0, opsel), _apply_opsel(s1, 1, opsel)
    if is_float:
      s0 = _apply_src_mods(s0, 0, abs_bits, neg_bits, bits['s0'])
      s1 = _apply_src_mods(s1, 1, abs_bits, neg_bits, bits['s1'])
    for dest, val in parse_pcode(pcode, {'S0': s0, 'S1': s1, 'laneId': lc})[1]:
      if '[laneId]' in dest and ('D0' in dest or 'EXEC' in dest): return val.cast(dtypes.uint32)
    return _c(0)

  new_bits = ctx.unroll_lanes(get_cmp_bit, exec_mask, apply_exec=False)
  # Both VOPC and VOP3 clear inactive lane bits (hardware verified)
  new_result = new_bits & exec_mask

  # CMPX e32: writes EXEC only; CMPX e64: writes both EXEC and SDST; non-CMPX: writes dst only
  if is_cmpx:
    stores = [ctx.wsgpr_dyn(_c(EXEC_LO.offset), new_result)]
    if not is_vopc: stores.append(ctx.wsgpr_dyn(dst_off, new_result))
  else:
    stores = [ctx.wsgpr_dyn(dst_off, new_result)] if not is_vopc else [ctx.wsgpr_dyn(_c(VCC_LO.offset), new_result)]
  return UOp.sink(*stores, *ctx.inc_pc())

def _compile_vop3(inst: ir3.VOP3 | ir4.VOP3, ctx: _Ctx) -> UOp:
  exec_mask = ctx.rsgpr_dyn(_c(EXEC_LO.offset))
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

  # Regular VOP3 - read operands dynamically
  lane = ctx.range()
  vdst_reg = ctx.inst_field(type(inst).vdst)
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None
  ops = inst.canonical_operands
  src0 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src0), lane, bits['s0'], literal, 's0' in ops and ops['s0'][0] == Fmt.FMT_NUM_F64)
  src1 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src1), lane, bits['s1'], literal, 's1' in ops and ops['s1'][0] == Fmt.FMT_NUM_F64)
  src2 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src2), lane, bits['s2'], literal, 's2' in ops and ops['s2'][0] == Fmt.FMT_NUM_F64)
  if bits['s0'] == 16:
    src0 = _apply_opsel(src0, 0, opsel)
    src1 = _apply_opsel(src1, 1, opsel)
    src2 = _apply_opsel(src2, 2, opsel)
  abs_bits, neg_bits = getattr(inst, 'abs', 0) or 0, getattr(inst, 'neg', 0) or 0
  src0 = _apply_src_mods(src0, 0, abs_bits, neg_bits, bits['s0'])
  src1 = _apply_src_mods(src1, 1, abs_bits, neg_bits, bits['s1'])
  src2 = _apply_src_mods(src2, 2, abs_bits, neg_bits, bits['s2'])
  srcs = {'S0': src0, 'S1': src1, 'S2': src2}
  if inst.op in (ir3.VOP3Op.V_CNDMASK_B32_E64, ir3.VOP3Op.V_CNDMASK_B16) and src2 is not None: srcs['VCC'] = src2
  # FMAC instructions need D0 (accumulator) from destination register
  if 'FMAC' in op_name: srcs['D0'] = ctx.rvgpr_dyn(vdst_reg, lane)
  opsel_dst_hi = bool(opsel & 0b1000) and bits['d'] == 16
  return ctx.compile_vop_pcode(inst.op, srcs, lane, vdst_reg, exec_mask, opsel_dst_hi=opsel_dst_hi, clmp=getattr(inst, 'clmp', 0))

def _compile_vop3sd(inst: ir3.VOP3SD | ir4.VOP3SD, ctx: _Ctx) -> UOp:
  exec_mask = ctx.rsgpr_dyn(_c(EXEC_LO.offset))
  bits, pcode, ops = inst.canonical_op_bits, get_pcode(inst.op), inst.canonical_operands

  # Read operands dynamically from instruction encoding
  vdst_reg, sdst_off = ctx.inst_field(type(inst).vdst), ctx.inst_field(type(inst).sdst)
  src0_off, src1_off, src2_off = ctx.inst_field(type(inst).src0), ctx.inst_field(type(inst).src1), ctx.inst_field(type(inst).src2)
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None

  has_carry_in = 's2' in ops and ops['s2'][2] == OpType.OPR_SREG
  vcc_in_off = src2_off if has_carry_in else sdst_off

  def load_srcs(lane_uop):
    ret = {'VCC': ctx.rsgpr_dyn(vcc_in_off), 'EXEC': exec_mask, 'SCC': ctx.rsgpr_dyn(_c(SCC.offset)), 'laneId': lane_uop}
    ret['S0'] = ctx.rsrc_dyn(src0_off, lane_uop, bits['s0'], literal, ops['s0'][0] == Fmt.FMT_NUM_F64)
    ret['S1'] = ctx.rsrc_dyn(src1_off, lane_uop, bits['s1'], literal, ops['s1'][0] == Fmt.FMT_NUM_F64)
    if 's2' in ops: ret['S2'] = ctx.rsrc_dyn(src2_off, lane_uop, bits['s2'], literal, ops['s2'][0] == Fmt.FMT_NUM_F64)
    return ret

  lane = ctx.range()
  srcs = load_srcs(lane)
  _, assigns = parse_pcode(pcode, srcs)

  has_per_lane_vcc = any('[laneId]' in dest for dest, _ in assigns if dest.startswith('VCC') or dest.startswith('D0.u64'))
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
    d0_val = None
    for dest, val in parse_pcode(pcode, load_srcs(lane3))[1]:
      if dest.startswith('D0') and '[laneId]' not in dest: d0_val = val
    vgpr_stores = []
    if d0_val is not None:
      if d0_val.dtype in (dtypes.uint64, dtypes.int64, dtypes.float64):
        lo, hi = _split64(d0_val)
        vgpr_stores.extend([ctx.wvgpr_dyn(vdst_reg, lane3, lo, exec_mask), ctx.wvgpr_dyn(vdst_reg + _c(1), lane3, hi, exec_mask)])
      else:
        d0_u32 = d0_val.bitcast(dtypes.uint32) if d0_val.dtype in (dtypes.float32, dtypes.half) else d0_val.cast(dtypes.uint32)
        vgpr_stores.append(ctx.wvgpr_dyn(vdst_reg, lane3, d0_u32, exec_mask))
    # Write carry output (wsgpr_dyn handles NULL register 124)
    vcc_write = ctx.wsgpr_dyn(sdst_off, final_vcc)
    return UOp.sink(vcc_write, UOp.group(*vgpr_stores).end(lane3), *ctx.inc_pc())
  else:
    return ctx.compile_vop_pcode(inst.op, srcs, lane, vdst_reg, exec_mask, sdst_reg=inst.sdst.offset)

def _compile_wmma(inst: ir3.VOP3P | ir4.VOP3P, ctx: _Ctx) -> UOp:
  op_name = _op_name(inst)
  exec_mask = ctx.rsgpr_dyn(_c(EXEC_LO.offset))
  vdst_reg = ctx.inst_field(type(inst).vdst)
  src0_r = ctx.inst_field(type(inst).src0) - _c(256)
  src1_r = ctx.inst_field(type(inst).src1) - _c(256)
  src2_r = ctx.inst_field(type(inst).src2) - _c(256)
  is_f16_output = 'F16_16X16X16_F16' in op_name or 'BF16_16X16X16_BF16' in op_name  # F16/BF16 output vs F32 output
  is_bf16 = 'BF16' in op_name
  cvt = _FUNCS['bf16_to_f32'] if is_bf16 else _FUNCS['f16_to_f32']
  def read_f16_mat(src):
    return [f for l in range(16) for r in range(8) for v in [ctx.rvgpr_dyn(src + _c(r), UOp.const(dtypes.int, l))]
            for f in [cvt(v & UOp.const(dtypes.uint32, 0xFFFF)), cvt(v >> UOp.const(dtypes.uint32, 16))]]
  mat_a, mat_b = read_f16_mat(src0_r), read_f16_mat(src1_r)
  if is_f16_output:
    # RDNA3 F16/BF16 output: uses 8 VGPRs (same as F32), f16/bf16 values in lo 16 bits of each VGPR
    # Layout: half16 per lane where even indices (0,2,4,...,14) = lo halves of VGPRs 0-7
    # Read accumulator: 8 regs × 32 lanes, each VGPR's lo 16 bits holds one f16/bf16
    mat_c = [cvt(ctx.rvgpr_dyn(src2_r + _c(i // 32), UOp.const(dtypes.int, i % 32)) & UOp.const(dtypes.uint32, 0xFFFF))
              for i in range(256)]
    mat_d = [sum(mat_a[row*16+k] * mat_b[col*16+k] for k in range(16)) + mat_c[row*16+col] for row in range(16) for col in range(16)]
    # Write f16/bf16 results to lo 16 bits of each VGPR
    def f32_to_f16_bits(v: UOp) -> UOp: return v.cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32)
    def f32_to_bf16_bits(v: UOp) -> UOp: return (v.bitcast(dtypes.uint32) >> UOp.const(dtypes.uint32, 16)) & UOp.const(dtypes.uint32, 0xFFFF)
    out_cvt = f32_to_bf16_bits if is_bf16 else f32_to_f16_bits
    stores = [ctx.wvgpr_dyn(vdst_reg + _c(i // 32), UOp.const(dtypes.int, i % 32), out_cvt(mat_d[i]), exec_mask) for i in range(256)]
  else:
    # F32 output: accumulator and output are f32
    mat_c = [ctx.rvgpr_dyn(src2_r + _c(i // 32), UOp.const(dtypes.int, i % 32)).bitcast(dtypes.float32) for i in range(256)]
    mat_d = [sum(mat_a[row*16+k] * mat_b[col*16+k] for k in range(16)) + mat_c[row*16+col] for row in range(16) for col in range(16)]
    stores = [ctx.wvgpr_dyn(vdst_reg + _c(i // 32), UOp.const(dtypes.int, i % 32), mat_d[i].bitcast(dtypes.uint32), exec_mask) for i in range(256)]
  return UOp.sink(*stores, *ctx.inc_pc())

def _compile_vop3p(inst: ir3.VOP3P | ir4.VOP3P, ctx: _Ctx) -> UOp:
  op_name = _op_name(inst)
  if 'WMMA' in op_name and ('16X16X16_F16' in op_name or '16X16X16_BF16' in op_name): return _compile_wmma(inst, ctx)

  lane = ctx.range()
  exec_mask = ctx.rsgpr_dyn(_c(EXEC_LO.offset))
  vdst_reg = ctx.inst_field(type(inst).vdst)
  src0 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src0), lane, 16)
  src1 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src1), lane, 16)
  src2 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src2), lane, 16)
  opsel, opsel_hi = getattr(inst, 'opsel', 0) or 0, getattr(inst, 'opsel_hi', 3) if getattr(inst, 'opsel_hi', 3) is not None else 3
  opsel_hi2 = getattr(inst, 'opsel_hi2', 1) if getattr(inst, 'opsel_hi2', 1) is not None else 1
  neg, neg_hi = getattr(inst, 'neg', 0) or 0, getattr(inst, 'neg_hi', 0) or 0

  if 'FMA_MIX' in op_name:
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
      return get_half_bits(src, bool(opsel_lo_bit), bool(neg_lo_bit)) | (get_half_bits(src, bool(opsel_hi_bit), bool(neg_hi_bit)) << UOp.const(dtypes.uint32, 16))
    # DOT IU instructions use NEG bits for signed/unsigned selection, not fp16 negation
    is_dot_iu = 'DOT' in op_name and 'IU' in op_name
    n0, n1, n2, nh0, nh1, nh2 = (0, 0, 0, 0, 0, 0) if is_dot_iu else (neg & 1, neg & 2, neg & 4, neg_hi & 1, neg_hi & 2, neg_hi & 4)
    srcs = {'S0': build_remapped_src(src0, opsel & 1, opsel_hi & 1, n0, nh0),
            'S1': build_remapped_src(src1, opsel & 2, opsel_hi & 2, n1, nh1),
            'S2': build_remapped_src(src2, opsel & 4, 1 if opsel_hi2 else 0, n2, nh2)}
    if is_dot_iu: srcs['NEG'] = UOp.const(dtypes.uint32, neg)
  return ctx.compile_vop_pcode(inst.op, srcs, lane, vdst_reg, exec_mask)

def _compile_vopd(inst: ir3.VOPD | ir4.VOPD, ctx: _Ctx) -> UOp:
  exec_mask = ctx.rsgpr_dyn(_c(EXEC_LO.offset))
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
  for op, src0_off, vsrc1_reg, vdst_reg, label in [(inst.opx, srcx0_off, vsrcx1_reg, vdstx_reg, 'X'),
                                                    (inst.opy, srcy0_off, vsrcy1_reg, vdsty_reg, 'Y')]:
    vop = VOPD_TO_VOP2.get(op)
    assert vop is not None, f"no VOP mapping for VOPD {label}: {op}"
    if label == 'Y': srcs = {'S0': srcy0, 'S1': srcy1, 'D0': ctx.rvgpr_dyn(vdst_reg, lane)}
    else: srcs = {'S0': ctx.rsrc_dyn(src0_off, lane, literal=literal), 'S1': ctx.rvgpr_dyn(vsrc1_reg, lane), 'D0': ctx.rvgpr_dyn(vdst_reg, lane)}
    if op in (ir3.VOPDOp.V_DUAL_FMAAK_F32, ir3.VOPDOp.V_DUAL_FMAMK_F32, ir4.VOPDOp.V_DUAL_FMAAK_F32, ir4.VOPDOp.V_DUAL_FMAMK_F32):
      assert literal is not None
      srcs['SIMM32'] = literal
    if op in (ir3.VOPDOp.V_DUAL_CNDMASK_B32, ir4.VOPDOp.V_DUAL_CNDMASK_B32): srcs['VCC'] = ctx.rsgpr_dyn(_c(VCC_LO.offset))
    pcode = get_pcode(vop)
    srcs.update({'VCC': ctx.rsgpr_dyn(_c(VCC_LO.offset)), 'EXEC': exec_mask, 'SCC': ctx.rsgpr_dyn(_c(SCC.offset)), 'laneId': lane})
    for dest, val in parse_pcode(pcode, srcs)[1]:
      if dest.startswith('D0'): all_stores.append(ctx.wvgpr_dyn(vdst_reg, lane, _val_to_u32(val), exec_mask, after=srcy1))
  return UOp.sink(UOp.group(*all_stores).end(lane), *ctx.inc_pc())

def _compile_mem_op(inst: ir3.DS | ir3.FLAT | ir3.GLOBAL | ir3.SCRATCH | ir4.DS | ir4.VFLAT | ir4.VGLOBAL | ir4.VSCRATCH, ctx: _Ctx) -> UOp:
  """Unified memory operation compiler for DS, FLAT, GLOBAL, SCRATCH."""
  exec_mask, op_name = ctx.rsgpr_dyn(_c(EXEC_LO.offset)), _op_name(inst)
  pcode = get_pcode(inst.op)

  is_lds = isinstance(inst, (ir3.DS, ir4.DS))
  is_scratch = isinstance(inst, (ir3.SCRATCH, ir4.VSCRATCH))
  mem = ctx.lds if is_lds else ctx.scratch if is_scratch else ctx.vmem
  addr_shift = UOp.const(dtypes.uint32 if is_lds else dtypes.uint64, 2)

  # Extract register info - all dynamic for deduplication
  if is_lds:
    addr_reg = ctx.inst_field(type(inst).addr)
    vdata_reg = ctx.inst_field(type(inst).data0)
    vdst_reg = ctx.inst_field(type(inst).vdst)
    offset0 = ctx.inst_field(type(inst).offset0)
    offset1 = ctx.inst_field(type(inst).offset1)
    offset = offset0  # DS uses offset0 as primary offset
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
    vdata_reg = ctx.inst_field(type(inst).data)
    vdst_reg = ctx.inst_field(type(inst).vdst)
    offset = ctx.inst_field_signed(type(inst).offset)
    offset0, offset1 = _c(0), _c(0)
    saddr_reg = ctx.inst_field(type(inst).saddr) if hasattr(type(inst), 'saddr') else None

  # Data width from canonical_op_bits (32/64/96/128), default to 32 for untyped ops
  data_bits_mem = inst.canonical_op_bits.get('data', 32)
  is_atomic, glc = 'ATOMIC' in op_name, getattr(inst, 'glc', 0)
  has_data1 = is_lds and hasattr(inst, 'data1') and inst.data1 is not None
  data1_reg = ctx.inst_field(type(inst).data1) if is_lds else _c(0)

  # DS_PERMUTE/DS_BPERMUTE: cross-lane VGPR access via pcode
  if is_lds and 'PERMUTE' in op_name:
    pcode = get_pcode(inst.op)
    srcs = {'ADDR': addr_reg, 'DATA0': vdata_reg, 'VDST': vdst_reg, 'OFFSET': offset,
            'EXEC': exec_mask.cast(dtypes.uint64), '_vgpr': ctx.vgpr}
    _, assigns = parse_pcode(pcode, srcs)
    stores = [ctx.vgpr.index(val[0].cast(dtypes.int)).store(val[1].cast(dtypes.uint32)) for dest, val in assigns if dest.startswith('VGPR[')]
    return UOp.sink(*stores, *ctx.inc_pc())

  def make_addr(lane: UOp) -> UOp:
    if is_lds: return ctx.rvgpr_dyn(addr_reg, lane)
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
      saddr_contrib = use_saddr.where(ctx.rsgpr_dyn(saddr_reg).cast(dtypes.uint64), UOp.const(dtypes.uint64, 0)) if saddr_reg is not None else UOp.const(dtypes.uint64, 0)
      return base + addr_offset + saddr_contrib + offset64
    # FLAT/GLOBAL: choose between SGPR base (saddr) or VGPR pair (addr) based on saddr validity
    saddr_base = _u64(ctx.rsgpr_dyn(saddr_reg), ctx.rsgpr_dyn(saddr_reg + _c(1))) if saddr_reg is not None else UOp.const(dtypes.uint64, 0)
    vaddr_base = _u64(ctx.rvgpr_dyn(addr_reg, lane), ctx.rvgpr_dyn(addr_reg + _c(1), lane))
    # When saddr is valid: base = saddr pair, vaddr is 32-bit offset; otherwise: base = 0, vaddr is 64-bit address
    base_addr = use_saddr.where(saddr_base + ctx.rvgpr_dyn(addr_reg, lane).cast(dtypes.uint64), vaddr_base)
    return base_addr + offset64

  def wmem(addr: UOp, val: UOp, active: UOp) -> UOp:
    idx = mem.index((addr >> addr_shift).cast(dtypes.int))
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
      elif data_bits_mem == 32:
        data = {'DATA': ctx.rvgpr_dyn(vdata_reg, lane), 'DATA2': ctx.rvgpr_dyn(data1_reg, lane) if has_data1 else UOp.const(dtypes.uint32, 0)}
      else:
        data = {'DATA': _u64(ctx.rvgpr_dyn(vdata_reg, lane), ctx.rvgpr_dyn(vdata_reg + _c(1), lane)),
                'DATA2': _u64(ctx.rvgpr_dyn(data1_reg, lane), ctx.rvgpr_dyn(data1_reg + _c(1), lane)) if has_data1 else UOp.const(dtypes.uint64, 0)}
      # RDNA3 uses ADDR/OFFSET, RDNA4 uses vgpr_a/offset (lowercase) + CalcDsAddr function
      return {'ADDR': addr, 'ADDR_BASE': addr, 'OFFSET': offset, 'OFFSET0': offset0, 'OFFSET1': offset1, '_lds': mem, 'laneId': lane,
              'vgpr_a': ctx.rvgpr_dyn(addr_reg, lane), 'offset': offset, **data}
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
      return {'ADDR': addr, 'DATA': _u64(ctx.rvgpr_dyn(vdata_reg, lane), ctx.rvgpr_dyn(vdata_reg + _c(1), lane)) if data_bits_mem == 64 else ctx.rvgpr_dyn(vdata_reg, lane),
              '_vmem': mem, '_active': active, 'laneId': lane, 'v_addr': vaddr_base, 's_saddr': saddr_base}
    vdata = ctx.rvgpr_dyn(vdata_reg, lane).cast(dtypes.uint64) if 'STORE' in op_name else ctx.rvgpr_dyn(vdst_reg, lane) if 'D16' in op_name else UOp.const(dtypes.uint32, 0)
    if 'STORE' in op_name and data_bits_mem >= 64: vdata = vdata | (ctx.rvgpr_dyn(vdata_reg + _c(1), lane).cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))
    srcs = {'ADDR': addr, 'VDATA': vdata, '_vmem': mem, '_active': active, 'laneId': lane, 'v_addr': vaddr_base, 's_saddr': saddr_base}
    for i in range(data_bits_mem // 32): srcs[f'VDATA{i}'] = ctx.rvgpr_dyn(vdata_reg + _c(i), lane) if 'STORE' in op_name else UOp.const(dtypes.uint32, 0)
    return srcs

  def make_stores(dest: str, val: UOp, lane: UOp, active: UOp, writes_return_data: bool) -> list[UOp]:
    # Parse bit width from dest format: MEM[...].b32 or RETURN_DATA[63:32].b64
    parts = dest.rsplit('.', 1)
    data_bits = int(parts[1][1:]) if len(parts) == 2 else 32
    if dest.startswith('MEM['):
      if is_lds or is_atomic: return _write_val(data_bits, val[1], wmem, val[0], active, is_mem=True)
      if is_scratch: return _mem_store_bytes(mem, val[0], val[1], active, data_bits)
      return _mem_store(mem, val[0], val[1], active, 64, data_bits)
    if dest.startswith('RETURN_DATA') and writes_return_data:
      if (m := re.match(r'RETURN_DATA\[(\d+)\s*:\s*(\d+)\]', dest)):
        bit_width, dword_idx = int(m.group(1)) - int(m.group(2)) + 1, int(m.group(2)) // 32
        return _write_val(bit_width, val, lambda r, v, l, e: ctx.wvgpr_dyn(r, l, v, e), vdst_reg + _c(dword_idx), lane, exec_mask)
      return _write_val(data_bits, val, lambda r, v, l, e: ctx.wvgpr_dyn(r, l, v, e), vdst_reg, lane, exec_mask)
    return []

  # DS-specific: check for 2ADDR pattern needing separate ranges
  if is_lds:
    dummy_lane = ctx.range()
    _, assigns = parse_pcode(pcode, make_srcs(dummy_lane))
    mem_assigns = [d for d, _ in assigns if d.startswith('MEM[')]
    mem_addrs = set(m.group(1) if (m := re.match(r'MEM\[([^\]]+)\]', d)) else d for d in mem_assigns)
    use_separate_ranges = (len(mem_addrs) > 1 or '2ADDR' in op_name) and 'STOREXCHG' not in op_name
    if use_separate_ranges:
      ended: list[UOp] = []
      for i, (dest, _) in enumerate(assigns):
        lane = ctx.range()
        active = _lane_active(exec_mask, lane)
        _, lane_assigns = parse_pcode(pcode, make_srcs(lane))
        ended.extend(s.end(lane) for s in make_stores(dest, lane_assigns[i][1], lane, active, True))
      return UOp.sink(*ended, *ctx.inc_pc())

  # Standard path: single lane range
  writes_return_data = '_RTN' in op_name or (is_lds and op_name.startswith('DS_LOAD')) or bool(is_atomic and glc)
  lane = ctx.range()
  active = _lane_active(exec_mask, lane)
  pcode_vars, assigns = parse_pcode(pcode, make_srcs(lane))
  stores = [s for dest, val in assigns for s in make_stores(dest, val, lane, active, writes_return_data)]

  # FLAT/GLOBAL/SCRATCH: collect VDATA slices for loads
  if not is_lds and not is_atomic:
    for dword_idx, val in sorted(_collect_data_slices(assigns, 'VDATA', pcode_vars, op_name).items()):
      stores.append(ctx.wvgpr_dyn(vdst_reg + _c(dword_idx), lane, val, exec_mask))

  return UOp.sink(UOp.group(*stores).end(lane), *ctx.inc_pc())

# Dispatch table: instruction type -> handler function
_INST_HANDLERS: dict[type, Callable[..., UOp]] = {
  ir3.SOPP: _compile_sopp, ir3.SMEM: _compile_smem, ir3.SOP1: _compile_sop, ir3.SOP2: _compile_sop, ir3.SOPC: _compile_sop, ir3.SOPK: _compile_sop,
  ir3.VOP1: _compile_vop12, ir3.VOP1_SDST: _compile_vop12, ir3.VOP2: _compile_vop12, ir3.VOPC: _compile_vopc, ir3.VOP3: _compile_vop3,
  ir3.VOP3_SDST: _compile_vop3, ir3.VOP3SD: _compile_vop3sd, ir3.VOP3P: _compile_vop3p, ir3.VOPD: _compile_vopd,
  ir3.DS: _compile_mem_op, ir3.FLAT: _compile_mem_op, ir3.GLOBAL: _compile_mem_op, ir3.SCRATCH: _compile_mem_op,
  # RDNA4 instruction classes
  ir4.SOPP: _compile_sopp, ir4.SMEM: _compile_smem, ir4.SOP1: _compile_sop, ir4.SOP2: _compile_sop, ir4.SOPC: _compile_sop, ir4.SOPK: _compile_sop,
  ir4.VOP1: _compile_vop12, ir4.VOP1_SDST: _compile_vop12, ir4.VOP2: _compile_vop12, ir4.VOPC: _compile_vopc, ir4.VOP3: _compile_vop3,
  ir4.VOP3_SDST: _compile_vop3, ir4.VOP3SD: _compile_vop3sd, ir4.VOP3P: _compile_vop3p, ir4.VOPD: _compile_vopd,
  ir4.DS: _compile_mem_op, ir4.VFLAT: _compile_mem_op, ir4.VGLOBAL: _compile_mem_op, ir4.VSCRATCH: _compile_mem_op,
}

# ═══════════════════════════════════════════════════════════════════════════════
# PROGRAM DECODE AND COMPILATION
# ═══════════════════════════════════════════════════════════════════════════════

_canonical_runner_cache: list[tuple[int, int, int, object]] = []  # [(base, mask, size, runner), ...]

@functools.cache
def _get_runner(inst_bytes: bytes, arch: str = "rdna3"):
  """Build and compile instruction to CompiledRunner. Cached by instruction bytes, with canonical dedup."""
  inst = decode_inst(inst_bytes, arch)
  inst_size = inst.size()
  inst_int = int.from_bytes(inst_bytes[:inst_size], 'little')

  # Check if instruction matches any cached canonical pattern
  for base, mask, size, runner in _canonical_runner_cache:
    if inst_size == size and (inst_int & mask) == base: return runner, False

  # Look up handler by type, falling back to base classes for _LIT variants
  handler = _INST_HANDLERS.get(type(inst))
  if handler is None:
    for cls in type(inst).__mro__:
      if cls in _INST_HANDLERS:
        handler = _INST_HANDLERS[cls]
        break
  if handler is None: raise RuntimeError(f"[emu] unimplemented instruction type: {type(inst).__name__} {_op_name(inst)}")

  ctx = _Ctx(inst_size)
  sink = handler(inst, ctx)
  base, mask, size = ctx.canonical_mask(inst_bytes)
  canonical_name = f"{_op_name(inst).lower()}_{base.to_bytes(size, 'little').hex()}"
  sink = sink.replace(arg=KernelInfo(name=canonical_name)).rtag(1)

  with Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES=""):
    runner = get_runner('CPU', sink)
  _canonical_runner_cache.append((base, mask, size, runner))
  return runner, True

@functools.cache
def decode_program(data: bytes, arch: str = "rdna3") -> dict[int, tuple[str, Callable, list[int], Any]]:
  """Decode program to {pc: (name, fxn, globals, runner)}."""
  result: dict[int, tuple[str, Callable, list[int], Any]] = {}
  i = 0
  while i < len(data):
    inst = decode_inst(data[i:], arch)
    if hasattr(inst, 'op') and inst.op in (ir3.SOPPOp.S_CODE_END, ir4.SOPPOp.S_CODE_END): break
    try:
      runner, is_new = _get_runner(bytes(data[i:i + inst.size() + 4]), arch)
      if DEBUG >= 3:
        try: inst_str = repr(inst)
        except Exception: inst_str = f"<{type(inst).__name__} at PC={i}>"
        msg = f"[emu] PC={i}: {inst_str}"
        print(colored(msg, 'green') if is_new else msg)
      result[i] = (runner.p.function_name, runner._prg.fxn, runner.p.globals, runner)
    except Exception as e:
      try: inst_str = repr(inst)
      except Exception: inst_str = f"<{type(inst).__name__}>"
      raise RuntimeError(f"[emu] Failed to compile PC={i} {inst_str}: {type(e).__name__}: {e}") from e
    i += inst.size()
  return result

# ═══════════════════════════════════════════════════════════════════════════════
# WAVE STATE
# ═══════════════════════════════════════════════════════════════════════════════

# Inline float constants (as bit patterns) for GPU instructions
F32_INLINE = {240: 0x3f000000, 241: 0xbf000000, 242: 0x3f800000, 243: 0xbf800000,  # 0.5, -0.5, 1.0, -1.0
              244: 0x40000000, 245: 0xc0000000, 246: 0x40800000, 247: 0xc0800000, 248: 0x3e22f983}  # 2.0, -2.0, 4.0, -4.0, 1/(2*pi)

class WaveState:
  __slots__ = ('vgpr_buf', 'sgpr_buf', '_vgpr_mv', '_sgpr_mv', 'n_lanes')

  def __init__(self, n_lanes: int = WAVE_SIZE):
    self.n_lanes = n_lanes
    self.vgpr_buf = Buffer('CPU', VGPR_SIZE, dtypes.uint32).ensure_allocated()
    self.sgpr_buf = Buffer('CPU', SGPR_COUNT, dtypes.uint32).ensure_allocated()
    self._vgpr_mv = self.vgpr_buf.as_buffer(force_zero_copy=True).cast('I')
    self._sgpr_mv = self.sgpr_buf.as_buffer(force_zero_copy=True).cast('I')
    # Zero memory using ctypes memset (much faster than Python loops)
    ctypes.memset(self.vgpr_buf._buf.va_addr, 0, VGPR_SIZE * 4)
    ctypes.memset(self.sgpr_buf._buf.va_addr, 0, SGPR_COUNT * 4)
    # Pre-populate inline constants at indices 128-255
    for i in range(65): self._write_sgpr(128 + i, i)  # 128-192: integers 0-64
    for i in range(16): self._write_sgpr(193 + i, (-(i + 1)) & MASK32)  # 193-208: -1 to -16
    for off, val in F32_INLINE.items(): self._write_sgpr(off, val)  # 240-248: float constants
    self._write_sgpr(EXEC_LO.offset, (1 << n_lanes) - 1)
    self._write_sgpr(PC_LO_IDX, 0)
    self._write_sgpr(PC_HI_IDX, 0)

  def _write_sgpr(self, idx: int, val: int): self._sgpr_mv[idx] = val & MASK32
  def _read_sgpr(self, idx: int) -> int: return self._sgpr_mv[idx]
  def _write_vgpr(self, reg: int, lane: int, val: int): self._vgpr_mv[reg * 32 + lane] = val & MASK32
  def _read_vgpr(self, reg: int, lane: int) -> int: return self._vgpr_mv[reg * 32 + lane]

  @property
  def pc(self) -> int: return self._read_sgpr(PC_LO_IDX) | (self._read_sgpr(PC_HI_IDX) << 32)
  @pc.setter
  def pc(self, val: int):
    self._write_sgpr(PC_LO_IDX, val & MASK32)
    self._write_sgpr(PC_HI_IDX, (val >> 32) & MASK32)

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_asm(lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int, rsrc2: int = 0x19c,
            scratch_size: int = 0, arch: str = "rdna3") -> int:
  """Execute AMD assembly program. scratch_size is private_segment_fixed_size from kernel descriptor (per-lane)."""
  program_raw = decode_program(bytes((ctypes.c_char * lib_sz).from_address(lib).raw), arch)
  program = {lib + offset: val for offset, val in program_raw.items()}  # Remap to actual addresses
  lds_size = ((rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT) * 512
  total_threads = lx * ly * lz

  # Use Buffer objects with external_ptr=0 for vmem
  vmem_buf = Buffer('CPU', 1 << 40, dtypes.uint32, options=BufferSpec(external_ptr=0)).ensure_allocated()
  lds_buf = Buffer('CPU', max(lds_size // 4, 1), dtypes.uint32).ensure_allocated()
  scratch_buf = Buffer('CPU', scratch_size * WAVE_SIZE, dtypes.uint8).ensure_allocated() if scratch_size else None

  # Set DAZ+FTZ during emulator execution, restore afterward to avoid breaking hypothesis tests
  with _MXCSRContext():
    for gidz in range(gz):
      for gidy in range(gy):
        for gidx in range(gx):
          for wave_start in range(0, total_threads, WAVE_SIZE):
            n_lanes, st = min(WAVE_SIZE, total_threads - wave_start), WaveState(min(WAVE_SIZE, total_threads - wave_start))
            st.pc = lib  # Set PC to code base address
            st._write_sgpr(0, args_ptr & MASK32)
            st._write_sgpr(1, (args_ptr >> 32) & MASK32)

            # Workgroup IDs in SGPRs after user SGPRs
            sgpr_idx = (rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT
            for enabled, gid in [(hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X, gidx),
                                 (hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y, gidy),
                                 (hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z, gidz)]:
              if rsrc2 & enabled: st._write_sgpr(sgpr_idx, gid); sgpr_idx += 1

            # RDNA4 uses TTMP registers for workgroup IDs: ttmp[9]=gidx, ttmp[10]=gidy, ttmp[11]=gidz
            if arch == "rdna4":
              st._write_sgpr(ttmp[9].offset, gidx)
              st._write_sgpr(ttmp[10].offset, gidy)
              st._write_sgpr(ttmp[11].offset, gidz)

            # v0 = packed workitem IDs, scratch stride in secret SGPR
            for lane in range(n_lanes):
              tid = wave_start + lane
              st._write_vgpr(0, lane, ((tid // (lx * ly)) << 20) | (((tid // lx) % ly) << 10) | (tid % lx))
            st._write_sgpr(SCRATCH_STRIDE_IDX, scratch_size)

            # Pass buffer addresses via ctypes (pre-create to avoid allocation in loop)
            c_bufs = [ctypes.c_uint64(st.sgpr_buf._buf.va_addr), ctypes.c_uint64(st.vgpr_buf._buf.va_addr),
                      ctypes.c_uint64(vmem_buf._buf.va_addr), ctypes.c_uint64(lds_buf._buf.va_addr),
                      ctypes.c_uint64(scratch_buf._buf.va_addr if scratch_buf else 0)]
            for inst_count in range(1_000_000):
              if (pc := st.pc) == 0xFFFFFFFFFFFFFFFF or pc not in program: break
              name, fxn, globals_list, _ = program[pc]
              assert fxn is not None, f"[emu] No fxn for {name} at PC={pc}"
              assert 4 not in globals_list or scratch_buf, f"SCRATCH instruction {name} but scratch_size=0"
              if DEBUG >= 6:
                inst = decode_inst(bytes((ctypes.c_char * 12).from_address(pc).raw), arch)
                print(f"[emu] exec PC={pc:X}: {inst!r}")
              fxn(*[c_bufs[g] for g in globals_list])
            else: raise RuntimeError("exceeded 1M instructions, likely infinite loop")
  return 0
