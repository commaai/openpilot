# RDNA3 emulator v2 - compiles pcode to UOps executed via tinygrad CPU backend
# Each instruction is compiled to a kernel that operates on buffers:
#   arg=0: sgpr - sgpr[0-127], inline constants[128-255], PC_LO=256, PC_HI=257, SCC=258, SCRATCH_STRIDE=259
#   arg=1: vgpr - vgpr[reg * 32 + lane]
#   arg=2: vmem - base address 0, INDEX offsets directly to host memory
#   arg=3: lds - local data share
#   arg=4: scratch - per-lane scratch memory
from __future__ import annotations
import ctypes, functools, re, platform, subprocess, tempfile

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
from tinygrad.codegen import get_program
from tinygrad.device import Device, Buffer, BufferSpec
from tinygrad.runtime.autogen import hsa
from tinygrad.helpers import Context, DEBUG, colored, TUPLE_ORDER, getenv
from tinygrad.renderer import ProgramSpec

from extra.assembly.amd.decode import decode_inst
from extra.assembly.amd.autogen.rdna3.str_pcode import PCODE
from extra.assembly.amd.autogen.rdna3.ins import (SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, VOP1, VOP1_SDST, VOP2, VOP3, VOP3_SDST, VOP3SD, VOP3P, VOPC,
  DS, FLAT, GLOBAL, SCRATCH, VOPD, SOPPOp, SMEMOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOPDOp)
from extra.assembly.amd.dsl import NULL, VCC_LO, EXEC_LO
from extra.assembly.amd.autogen.common import OpType
from extra.assembly.amd.expr_parser import parse_block

MASK32 = 0xFFFFFFFF

# Common UOp constants (avoid repeated allocation)
def _c(val, dtype=dtypes.uint32): return UOp.const(dtype, val)
U32_0, U32_1, U32_16, U32_MASK = _c(0), _c(1), _c(16), _c(MASK32)
IDX_0 = _c(0, dtypes.index)

# Inline float constants (as bit patterns) for GPU instructions
F32_INLINE = {240: 0x3f000000, 241: 0xbf000000, 242: 0x3f800000, 243: 0xbf800000,  # 0.5, -0.5, 1.0, -1.0
              244: 0x40000000, 245: 0xc0000000, 246: 0x40800000, 247: 0xc0800000, 248: 0x3e22f983}  # 2.0, -2.0, 4.0, -4.0, 1/(2*pi)
F64_INLINE = {240: 0x3fe0000000000000, 241: 0xbfe0000000000000, 242: 0x3ff0000000000000, 243: 0xbff0000000000000,
              244: 0x4000000000000000, 245: 0xc000000000000000, 246: 0x4010000000000000, 247: 0xc010000000000000, 248: 0x3fc45f306dc9c883}
F16_INLINE = {240: 0x3800, 241: 0xb800, 242: 0x3c00, 243: 0xbc00, 244: 0x4000, 245: 0xc000, 246: 0x4400, 247: 0xc400, 248: 0x3118}

def _u64(lo: UOp, hi: UOp) -> UOp:
  """Combine two 32-bit UOps into a 64-bit UOp."""
  return lo.cast(dtypes.uint64) | (hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))

def _split64(val: UOp) -> tuple[UOp, UOp]:
  """Split a 64-bit value into (lo, hi) 32-bit values."""
  v64 = val.bitcast(dtypes.uint64) if val.dtype == dtypes.float64 else val.cast(dtypes.uint64) if val.dtype != dtypes.uint64 else val
  return v64.cast(dtypes.uint32), (v64 >> UOp.const(dtypes.uint64, 32)).cast(dtypes.uint32)

def _apply_src_mods(val: UOp, mod_bit: int, abs_bits: int, neg_bits: int, is_16bit: bool = False, is_64bit: bool = False) -> UOp:
  """Apply abs/neg modifiers to source value based on operation type."""
  if not (abs_bits & (1 << mod_bit)) and not (neg_bits & (1 << mod_bit)): return val
  if is_16bit:
    f16_val = val.cast(dtypes.uint16).bitcast(dtypes.half)
    if abs_bits & (1 << mod_bit): f16_val = (f16_val.bitcast(dtypes.uint16) & UOp.const(dtypes.uint16, 0x7FFF)).bitcast(dtypes.half)
    if neg_bits & (1 << mod_bit): f16_val = f16_val.neg()
    return f16_val.bitcast(dtypes.uint16).cast(dtypes.uint32)
  if is_64bit:
    if val.dtype == dtypes.uint64: val = val.bitcast(dtypes.float64)
    if abs_bits & (1 << mod_bit): val = (val.bitcast(dtypes.uint64) & UOp.const(dtypes.uint64, 0x7FFFFFFFFFFFFFFF)).bitcast(dtypes.float64)
    if neg_bits & (1 << mod_bit): val = val.neg()
    return val.bitcast(dtypes.uint64)
  if val.dtype == dtypes.uint32: val = val.bitcast(dtypes.float32)
  if abs_bits & (1 << mod_bit): val = (val.bitcast(dtypes.uint32) & UOp.const(dtypes.uint32, 0x7FFFFFFF)).bitcast(dtypes.float32)
  if neg_bits & (1 << mod_bit): val = val.neg()
  return val.bitcast(dtypes.uint32)

# Map VOPD ops to VOP2 ops for pcode lookup
VOPD_TO_VOP2 = {
  VOPDOp.V_DUAL_FMAC_F32: VOP2Op.V_FMAC_F32_E32, VOPDOp.V_DUAL_MUL_F32: VOP2Op.V_MUL_F32_E32,
  VOPDOp.V_DUAL_ADD_F32: VOP2Op.V_ADD_F32_E32, VOPDOp.V_DUAL_SUB_F32: VOP2Op.V_SUB_F32_E32,
  VOPDOp.V_DUAL_SUBREV_F32: VOP2Op.V_SUBREV_F32_E32, VOPDOp.V_DUAL_MAX_F32: VOP2Op.V_MAX_F32_E32,
  VOPDOp.V_DUAL_MIN_F32: VOP2Op.V_MIN_F32_E32, VOPDOp.V_DUAL_ADD_NC_U32: VOP2Op.V_ADD_NC_U32_E32,
  VOPDOp.V_DUAL_LSHLREV_B32: VOP2Op.V_LSHLREV_B32_E32, VOPDOp.V_DUAL_AND_B32: VOP2Op.V_AND_B32_E32,
  VOPDOp.V_DUAL_MOV_B32: VOP1Op.V_MOV_B32_E32, VOPDOp.V_DUAL_CNDMASK_B32: VOP2Op.V_CNDMASK_B32_E32,
  VOPDOp.V_DUAL_FMAAK_F32: VOP2Op.V_FMAAK_F32_E32, VOPDOp.V_DUAL_FMAMK_F32: VOP2Op.V_FMAMK_F32_E32,
}
WAVE_SIZE = 32
# Special registers stored after inline constants (256-259)
PC_LO_IDX, PC_HI_IDX, SCC_IDX, SCRATCH_STRIDE_IDX = 256, 257, 253, 259
# SGPR buffer: 0-127 = SGPRs, 128-255 = inline constants, 256-259 = special registers
SGPR_COUNT, VGPR_SIZE = 260, 256 * 32

def _is_16bit_op(op_name: str) -> bool: return any(x in op_name for x in ('B16', 'F16', 'I16', 'U16'))
def _op_name(inst) -> str:
  if hasattr(inst, 'opx'): return f"{inst.opx.name}_{inst.opy.name}"  # VOPD has opx/opy not op
  return inst.op.name if hasattr(inst.op, 'name') else str(inst.op)
def _is_64bit_dest(dest: str) -> bool: return any(dest.endswith(x) for x in ('.b64', '.u64', '.i64', '.f64'))
def _to_u32(val: UOp) -> UOp:
  if val.dtype == dtypes.uint32: return val
  if val.dtype.itemsize == 4: return val.bitcast(dtypes.uint32)  # same size: bitcast (float32->uint32)
  return val.cast(dtypes.uint32)  # different size: cast (bool, int16, etc)
def _lane_active(exec_mask: UOp, lane: UOp) -> UOp: return ((exec_mask >> lane.cast(dtypes.uint32)) & U32_1).ne(U32_0)
def _apply_opsel(val: UOp, sel_bit: int, opsel: int) -> UOp:
  return (val >> U32_16) & _c(0xFFFF) if opsel & (1 << sel_bit) else val

def _unroll_lanes(get_lane_bit, exec_mask: UOp, apply_exec: bool = True) -> UOp:
  """Combine 32 lane bits into a 32-bit mask using RANGE+REDUCE. Optionally apply EXEC mask."""
  lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
  bit = get_lane_bit(lane).cast(dtypes.uint32) << lane.cast(dtypes.uint32)
  result = bit.reduce(lane, arg=Ops.ADD)
  return result & exec_mask if apply_exec else result

def _set_lane_bit(old: UOp, lane: UOp, val: UOp, exec_mask: UOp) -> UOp:
  """Set/clear a single bit in a 32-bit mask based on lane index, respecting exec mask."""
  mask = U32_1 << lane.cast(dtypes.uint32)
  new_bit = _to_u32(val) << lane.cast(dtypes.uint32)
  cleared = old & (mask ^ U32_MASK)
  return _lane_active(exec_mask, lane).where(cleared | new_bit, old)

def _val_to_u32(val: UOp) -> UOp:
  """Convert any value to uint32 for storage (bitcast floats, cast ints)."""
  if val.dtype == dtypes.uint32: return val
  if val.dtype == dtypes.float32: return val.bitcast(dtypes.uint32)
  if val.dtype == dtypes.half: return val.bitcast(dtypes.uint16).cast(dtypes.uint32)
  if val.dtype in (dtypes.uint16, dtypes.int16): return val.cast(dtypes.uint32)
  return val.cast(dtypes.uint32)

# Pcode parser
def _apply_pseudocode_fixes(op_name: str, pcode: str) -> str:
  fixes = {
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
  if op_name in fixes: pcode = pcode.replace(fixes[op_name][0], fixes[op_name][1])
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

def parse_pcode(pcode: str, srcs: dict[str, UOp] | None = None, lane: UOp | None = None, op_name: str | None = None) -> tuple[dict[str, UOp], list[tuple[str, UOp]]]:
  if op_name: pcode = _apply_pseudocode_fixes(op_name, pcode)
  vars: dict[str, UOp] = {n: UOp(Ops.DEFINE_VAR, dtypes.uint32, (), (n, U32_0, U32_MASK)) for n in ['S0', 'S1', 'S2', 'D0', 'VCC', 'EXEC', 'SCC', 'SIMM32']}
  if srcs: vars.update(srcs)
  vars.update({'laneId': lane if lane is not None else U32_0, 'WAVE_MODE': {'IEEE': U32_1}, 'WAVE32': _c(True, dtypes.bool), 'WAVE64': _c(False, dtypes.bool)})
  assigns: list[tuple[str, UOp]] = []
  lines = [l.strip().rstrip(';') for l in pcode.split('\n') if l.strip() and not l.strip().startswith('//')]
  _, final, _ = parse_block(lines, 0, vars, assigns=assigns)
  sliced = set(d.split('[')[0] for d, _ in assigns if '[' in d)
  for var, val in final.items():
    if var in ['D0', 'SCC', 'VCC', 'EXEC', 'PC', 'RETURN_DATA', 'VDATA']:
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

def _write_val(dest: str, val: UOp, wfn, reg_or_addr, *args, is_mem: bool = False) -> list[UOp]:
  """Write value, splitting 64-bit if needed based on dest type suffix."""
  return _write_64bit(val, wfn, reg_or_addr, is_mem, *args) if _is_64bit_dest(dest) else [wfn(reg_or_addr, _to_u32(val), *args)]

def _mem_store(mem: UOp, addr: UOp, val: UOp, active: UOp, addr_bits: int = 32, data_bits: int = 32) -> list[UOp]:
  """Conditional memory store: write val to mem[addr] if active, else keep old value. Handles sub-word stores. Returns list of store UOps."""
  adt = dtypes.uint64 if addr_bits == 64 else dtypes.uint32
  shift = UOp.const(adt, 2)
  word_addr = addr >> shift
  # Use .valid(active) to skip load from garbage address when lane is inactive
  idx = mem.index(word_addr.cast(dtypes.index).valid(active))
  # NOTE: Don't call idx.load() - use idx directly as the value. pm_add_loads will add the load op later.
  # Calling .load() here causes LOAD(LOAD) after pm_add_loads runs.
  val_u32 = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
  if data_bits == 8:
    byte_pos = (addr.cast(dtypes.uint32) & UOp.const(dtypes.uint32, 3))  # 0-3
    byte_shift = byte_pos << UOp.const(dtypes.uint32, 3)  # *8
    mask = UOp.const(dtypes.uint32, 0xFF) << byte_shift
    new_word = (idx & (mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))) | ((val_u32 & UOp.const(dtypes.uint32, 0xFF)) << byte_shift)
    return [idx.store(active.where(new_word, idx))]
  elif data_bits == 16:
    # 16-bit stores. byte_pos (0-3) determines placement within 4-byte word.
    # byte_pos 0,1,2: both bytes fit in current word
    # byte_pos 3: crosses word boundary - low byte to byte 3, high byte to next word's byte 0
    byte_pos = addr.cast(dtypes.uint32) & UOp.const(dtypes.uint32, 3)
    byte_shift = byte_pos << UOp.const(dtypes.uint32, 3)  # *8
    low_byte = val_u32 & UOp.const(dtypes.uint32, 0xFF)
    high_byte = (val_u32 >> UOp.const(dtypes.uint32, 8)) & UOp.const(dtypes.uint32, 0xFF)
    # Same-word value (for byte_pos 0,1,2): write 16 bits at byte_pos
    mask_16 = UOp.const(dtypes.uint32, 0xFFFF) << byte_shift
    same_word = (idx & (mask_16 ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))) | ((val_u32 & UOp.const(dtypes.uint32, 0xFFFF)) << byte_shift)
    # Cross-word value for current word (byte_pos=3): write low byte to byte 3
    cross_word0 = (idx & UOp.const(dtypes.uint32, 0x00FFFFFF)) | (low_byte << UOp.const(dtypes.uint32, 24))
    # Detect cross-word case: byte_pos == 3 <=> (byte_pos & 2) && (byte_pos & 1)
    is_cross = ((byte_pos >> UOp.const(dtypes.uint32, 1)) & byte_pos & UOp.const(dtypes.uint32, 1)).cast(dtypes.bool)
    # Select value for current word
    new_word = is_cross.where(cross_word0, same_word)
    store0 = idx.store(active.where(new_word, idx))
    # Next word store for cross-word case: write high byte to byte 0 of next word
    active_cross = active & is_cross
    # Use .valid(active_cross) to skip load from garbage address when lane is inactive or not cross-word
    next_word_addr = (word_addr + UOp.const(adt, 1)).cast(dtypes.index).valid(active_cross)
    next_idx = mem.index(next_word_addr)
    cross_word1 = (next_idx & UOp.const(dtypes.uint32, 0xFFFFFF00)) | high_byte
    store1 = next_idx.store(active_cross.where(cross_word1, next_idx))
    return [store0, store1]
  else:
    new_word = _to_u32(val)
    return [idx.store(active.where(new_word, idx))]

def _mem_store_bytes(mem: UOp, addr: UOp, val: UOp, active: UOp, data_bits: int = 32) -> list[UOp]:
  """Store to byte-addressable memory (scratch). addr is byte offset, mem is uint8 buffer."""
  stores = []
  val_u32 = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
  for i in range(data_bits // 8):
    byte_val = (val_u32 >> UOp.const(dtypes.uint32, i * 8)) & UOp.const(dtypes.uint32, 0xFF)
    idx = (addr + UOp.const(dtypes.uint64, i)).cast(dtypes.index).valid(active)
    stores.append(mem.index(idx).store(byte_val.cast(dtypes.uint8)))
  return stores

def _collect_data_slices(assigns: list, data_prefix: str, pcode_vars: dict = None, op_name: str = "") -> dict[int, UOp]:
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

def _scalar_stores_dyn(assigns: list, wsgpr_dyn, sdst_reg: UOp, sdst_size: int = 1) -> list[UOp]:
  """Generate stores for scalar assigns with dynamic destination register (D0, SCC, EXEC, VCC)."""
  def w64_dyn(reg: UOp, val):
    if val.dtype in (dtypes.uint64, dtypes.int64):
      lo, hi = _split64(val)
      return [wsgpr_dyn(reg, lo), wsgpr_dyn(reg + U32_1, hi)]
    return [wsgpr_dyn(reg, _to_u32(val))]
  stores = []
  for dest, val in assigns:
    if dest.startswith('D0'): stores.extend(w64_dyn(sdst_reg, val) if sdst_size == 2 else [wsgpr_dyn(sdst_reg, _to_u32(val))])
    elif dest.startswith('SCC'): stores.append(wsgpr_dyn(_c(SCC_IDX), _to_u32(val)))
    elif dest.startswith('EXEC'): stores.extend([wsgpr_dyn(_c(EXEC_LO.offset), _split64(val)[0]), wsgpr_dyn(_c(EXEC_LO.offset + 1), _split64(val)[1])] if val.dtype in (dtypes.uint64, dtypes.int64) else [wsgpr_dyn(_c(EXEC_LO.offset), _to_u32(val))])
    elif dest.startswith('VCC'): stores.extend([wsgpr_dyn(_c(VCC_LO.offset), _split64(val)[0]), wsgpr_dyn(_c(VCC_LO.offset + 1), _split64(val)[1])] if val.dtype in (dtypes.uint64, dtypes.int64) else [wsgpr_dyn(_c(VCC_LO.offset), _to_u32(val))])
  return stores

# Counter for unique axis IDs to avoid UOp caching issues
_axis_id_counter = 0
def _next_axis_id() -> int:
  global _axis_id_counter
  _axis_id_counter += 1
  return _axis_id_counter

def compile_sop_pcode_dyn(op, srcs: dict[str, UOp], wsgpr_dyn_fn, rsgpr_dyn_fn, sdst_reg: UOp, sdst_size: int, inc_pc_fn, name: str):
  """Compile a scalar instruction with dynamic destination register. Returns (name, sink) or None."""
  pcode = PCODE.get(op)
  if pcode is None: return None
  # For D0 read, use dynamic rsgpr; for VCC/EXEC/SCC use static offsets
  srcs.update({'VCC': rsgpr_dyn_fn(_c(VCC_LO.offset)), 'EXEC': rsgpr_dyn_fn(_c(EXEC_LO.offset)), 'SCC': rsgpr_dyn_fn(_c(SCC_IDX))})
  # D0 is the current value of destination register (for read-modify-write ops like S_ADDK)
  if 'D0' not in srcs: srcs['D0'] = rsgpr_dyn_fn(sdst_reg)
  _, assigns = parse_pcode(pcode, srcs, lane=None)
  stores = _scalar_stores_dyn(assigns, wsgpr_dyn_fn, sdst_reg, sdst_size)
  if not stores: return None
  return name, UOp.sink(*stores, *inc_pc_fn(), arg=KernelInfo(name=name))

def compile_lane_pcode(op, inst, ctx: '_Ctx', inc_pc_fn, name: str):
  """Compile READLANE/READFIRSTLANE/WRITELANE using pcode parser."""
  pcode = PCODE.get(op)
  if pcode is None: return None

  op_name = op.name if hasattr(op, 'name') else str(op)
  # Dynamic field reads
  src0_off = ctx.inst_field(type(inst).src0)
  vdst_off = ctx.inst_field(type(inst).vdst)
  # src0_reg = VGPR index (src0 - 256 if VGPR, else 0 for inline/SGPR)
  is_vgpr = src0_off >= _c(256)
  src0_reg = is_vgpr.where(src0_off - _c(256), U32_0)
  # vdst for VOP1 is VGPRField but READFIRSTLANE writes to SGPR with same encoding; VOP3 vdst is direct SGPR offset
  # S0 = scalar value for WRITELANE, register index for others; S1 = lane select for READLANE/WRITELANE
  src1_off = ctx.inst_field(type(inst).src1) if hasattr(type(inst), 'src1') else None
  srcs = {
    'SRC0': src0_reg, 'VDST': vdst_off, 'EXEC_LO': ctx.rsgpr_dyn(_c(EXEC_LO.offset)), '_vgpr': ctx.vgpr,
    'S0': ctx.rsrc_dyn(src0_off, IDX_0) if 'WRITELANE' in op_name else src0_reg,
    'S1': ctx.rsrc_dyn(src1_off, IDX_0) if src1_off is not None else U32_0,
  }
  _, assigns = parse_pcode(pcode, srcs, lane=None)

  stores = []
  for dest, val in assigns:
    if dest.startswith('D0'):
      stores.append(ctx.wsgpr_dyn(vdst_off, val.cast(dtypes.uint32)))
    elif dest.startswith('VGPR['):
      idx, write_val = val
      stores.append(ctx.vgpr.index(idx.cast(dtypes.index)).store(write_val.cast(dtypes.uint32)))

  if not stores: return None
  return name, UOp.sink(*stores, *inc_pc_fn(), arg=KernelInfo(name=name))

def compile_vop_pcode(op, srcs: dict[str, UOp], lane: UOp, wvgpr_fn, wsgpr_fn, rsgpr_fn, vdst_reg: UOp, exec_mask: UOp,
                      inc_pc_fn=None, name: str = None, opsel_dst_hi: bool | UOp = False, rvgpr_fn=None, sdst_reg: int | None = None):
  """Compile a VOP instruction using pcode parser. Returns (name, sink) if inc_pc_fn/name provided, else list of store UOps, or None."""
  pcode = PCODE.get(op)
  if pcode is None: return None
  vcc_reg = sdst_reg if sdst_reg is not None else VCC_LO.offset
  if 'VCC' not in srcs: srcs['VCC'] = rsgpr_fn(_c(vcc_reg))
  srcs['EXEC'], srcs['SCC'] = exec_mask, rsgpr_fn(_c(SCC_IDX))
  _, assigns = parse_pcode(pcode, srcs, lane, op_name=op.name)

  # Always use dynamic functions (vdst_reg is always UOp now)
  wvgpr, rvgpr = wvgpr_fn, rvgpr_fn

  raw_stores, vcc_val, exec_val = [], None, None
  for dest, val in assigns:
    if 'D0' in dest and '[laneId]' in dest:
      raw_stores.append(('vcc', wsgpr_fn(_c(VCC_LO.offset), _set_lane_bit(rsgpr_fn(_c(VCC_LO.offset)), lane, val, exec_mask))))
    elif dest.startswith('D0'):
      if (slice_match := re.match(r'D0\[(\d+)\s*:\s*(\d+)\]', dest)):
        hi_bit, lo_bit = int(slice_match.group(1)), int(slice_match.group(2))
        if hi_bit != 31 or lo_bit != 0:
          width, slice_mask = hi_bit - lo_bit + 1, (1 << (hi_bit - lo_bit + 1)) - 1
          val_bits = val.bitcast(dtypes.uint16).cast(dtypes.uint32) if val.dtype == dtypes.half else \
                     val.cast(dtypes.uint32) if val.dtype in (dtypes.uint16, dtypes.int16) else val.cast(dtypes.uint32) & UOp.const(dtypes.uint32, slice_mask)
          raw_stores.append(('vgpr_slice', (lo_bit, width, val_bits)))
          continue
      if val.dtype in (dtypes.uint64, dtypes.int64, dtypes.float64):
        lo, hi = _split64(val)
        raw_stores.extend([('vgpr', wvgpr(vdst_reg, lane, lo, exec_mask)), ('vgpr', wvgpr(vdst_reg + U32_1, lane, hi, exec_mask))])
      elif val.dtype in (dtypes.half, dtypes.uint16, dtypes.int16) and rvgpr_fn is not None:
        result, old_val = _val_to_u32(val), rvgpr(vdst_reg, lane)
        hi_result = (old_val & UOp.const(dtypes.uint32, 0xFFFF)) | (result << UOp.const(dtypes.uint32, 16))
        lo_result = (old_val & UOp.const(dtypes.uint32, 0xFFFF0000)) | (result & UOp.const(dtypes.uint32, 0xFFFF))
        if isinstance(opsel_dst_hi, UOp):
          result = opsel_dst_hi.where(hi_result, lo_result)
        else:
          result = hi_result if opsel_dst_hi else lo_result
        raw_stores.append(('vgpr', wvgpr(vdst_reg, lane, result, exec_mask)))
      else: raw_stores.append(('vgpr', wvgpr(vdst_reg, lane, _val_to_u32(val), exec_mask)))
    elif dest.startswith('VCC'): vcc_val = val  # Collect VCC value to reduce across lanes
    elif dest.startswith('EXEC'): exec_val = val  # Collect EXEC value to reduce across lanes
    elif dest.startswith('SCC'): raw_stores.append(('scc', wsgpr_fn(_c(SCC_IDX), _to_u32(val))))

  stores, lane_stores, scalar_stores = [], [s for t, s in raw_stores if t == 'vgpr'], [s for t, s in raw_stores if t == 'scc']
  slice_stores = [s for t, s in raw_stores if t == 'vgpr_slice']
  if slice_stores:
    result = rvgpr(vdst_reg, lane) if rvgpr_fn else UOp.const(dtypes.uint32, 0)
    for lo_bit, width, val_bits in slice_stores:
      mask = UOp.const(dtypes.uint32, ((1 << width) - 1) << lo_bit)
      result = (result & (mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))) | (val_bits << UOp.const(dtypes.uint32, lo_bit))
    lane_stores.append(wvgpr(vdst_reg, lane, result, exec_mask))
  if lane_stores: stores.append(UOp.sink(*lane_stores).end(lane))
  # VCC/EXEC writes use reduce to combine all lane bits, then write once (fixes multi-lane carry bug)
  # Must use _unroll_lanes pattern with fresh lambda to avoid graph issues with the main lane range
  # VOP2 carry instructions write ALL 32 VCC bits (hardware verified), not just active lane bits
  if vcc_val is not None:
    def get_vcc_bit(l): return (_to_u32(vcc_val.substitute({lane: l})) & U32_1).cast(dtypes.uint32)
    stores.append(wsgpr_fn(_c(vcc_reg), _unroll_lanes(get_vcc_bit, exec_mask, apply_exec=False)))
  if exec_val is not None:
    def get_exec_bit(l): return (_to_u32(exec_val.substitute({lane: l})) & U32_1).cast(dtypes.uint32)
    stores.append(wsgpr_fn(_c(EXEC_LO.offset), _unroll_lanes(get_exec_bit, exec_mask, apply_exec=False)))
  stores.extend(scalar_stores)
  if not stores: return None
  return (name, UOp.sink(*stores, *inc_pc_fn(), arg=KernelInfo(name=name))) if inc_pc_fn else stores

# Buffers: sgpr=0, vgpr=1, vmem=2, lds=3, scratch=4

def _define_bufs():
  sgpr = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(SGPR_COUNT), arg=0)
  vgpr = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(VGPR_SIZE), arg=1)
  vmem = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(1 << 46), arg=2)
  lds = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(16384), arg=3)
  scratch = UOp(Ops.DEFINE_GLOBAL, dtypes.uint8.ptr(1 << 30), arg=4)
  return sgpr, vgpr, vmem, lds, scratch

def _sext(v, bits): return v - (1 << bits) if v & (1 << (bits - 1)) else v

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION COMPILER - converts decoded instruction to UOp SINK
# ═══════════════════════════════════════════════════════════════════════════════

class _Ctx:
  """Context for instruction compilation - holds buffers and helpers."""
  __slots__ = ('sgpr', 'vgpr', 'vmem', 'lds', 'scratch', 'inst_size', 'dyn_fields')

  def __init__(self, sgpr, vgpr, vmem, lds, scratch, inst_size):
    self.sgpr, self.vgpr, self.vmem, self.lds, self.scratch = sgpr, vgpr, vmem, lds, scratch
    self.inst_size = inst_size
    self.dyn_fields: list[tuple[int, int]] = []  # (lo, hi) of fields read dynamically

  def inst_word(self, dword_idx: int) -> UOp:
    """Read instruction dword from vmem at PC + dword_idx*4."""
    pc = self.rpc()
    addr = (pc + UOp.const(dtypes.uint64, dword_idx * 4)) >> UOp.const(dtypes.uint64, 2)
    return self.vmem.index(addr.cast(dtypes.index), ptr=True).load()

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
      return (word >> UOp.const(dtypes.uint32, lo_in_dword)) & UOp.const(dtypes.uint32, mask)
    else:  # Spans two dwords
      lo_bits = 32 - lo_in_dword
      lo_mask = (1 << lo_bits) - 1
      hi_mask = (1 << (hi_in_dword + 1)) - 1
      lo_part = (word >> UOp.const(dtypes.uint32, lo_in_dword)) & UOp.const(dtypes.uint32, lo_mask)
      hi_part = self.inst_word(dword_idx + 1) & UOp.const(dtypes.uint32, hi_mask)
      return lo_part | (hi_part << UOp.const(dtypes.uint32, lo_bits))

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
  def rsgpr_dyn(self, reg: UOp) -> UOp:
    """Read SGPR with dynamic register index."""
    return self.sgpr.index(reg.cast(dtypes.index), ptr=True).load()

  def wsgpr_dyn(self, reg: UOp, val: UOp) -> UOp:
    """Write SGPR with dynamic register index. Writes to NULL (124) are discarded."""
    return self.sgpr.index(reg.cast(dtypes.index).valid(reg.ne(_c(124)))).store(val.cast(dtypes.uint32))

  def rvgpr_dyn(self, reg: UOp, lane: UOp) -> UOp:
    """Read VGPR with dynamic register index."""
    return self.vgpr.index(reg.cast(dtypes.index) * UOp.const(dtypes.index, 32) + lane.cast(dtypes.index), ptr=True).load()

  def wvgpr_dyn(self, reg: UOp, lane: UOp, val: UOp, exec_mask: UOp, after: UOp | None = None) -> UOp:
    """Write VGPR with dynamic register index."""
    buf = self.vgpr.after(after) if after is not None else self.vgpr
    offset = (reg.cast(dtypes.index) * UOp.const(dtypes.index, 32) + lane.cast(dtypes.index)).valid(_lane_active(exec_mask, lane))
    return buf.index(offset).store(val.cast(dtypes.uint32))

  def rsrc_dyn(self, off: UOp, lane: UOp, bits: int = 32, literal: UOp | None = None) -> UOp:
    """Read source operand with dynamic offset. Handles SGPR/inline constants (<256), VGPR (>=256).
    Inline constants 128-255 are pre-populated in SGPR buffer (integers, negatives, F32 floats).
    If literal is provided, it's used when off==255."""
    is_vgpr = off >= _c(256)
    is_in_sgpr = off < _c(256)  # guard for SGPR buffer access (size 260, but only 0-255 used for src)
    if bits == 64:
      is_sgpr = off < _c(128)
      # Guard SGPR reads with .valid() to prevent out-of-bounds access when off >= 256
      sgpr_idx0 = off.cast(dtypes.index).valid(is_in_sgpr)
      sgpr_idx1 = (off + U32_1).cast(dtypes.index).valid(is_in_sgpr)
      sgpr_val = _u64(self.sgpr.index(sgpr_idx0, ptr=True).load(), self.sgpr.index(sgpr_idx1, ptr=True).load())
      # Use .valid() for VGPR reads to avoid invalid memory access when off < 256
      vgpr_reg = off - _c(256)
      vgpr_idx0 = (vgpr_reg.cast(dtypes.index) * UOp.const(dtypes.index, 32) + lane.cast(dtypes.index)).valid(is_vgpr)
      vgpr_idx1 = ((vgpr_reg + U32_1).cast(dtypes.index) * UOp.const(dtypes.index, 32) + lane.cast(dtypes.index)).valid(is_vgpr)
      vgpr_val = _u64(self.vgpr.index(vgpr_idx0, ptr=True).load(), self.vgpr.index(vgpr_idx1, ptr=True).load())
      # 64-bit inline constants need special handling (different from 32-bit values in SGPR buffer)
      inline_idx = off.cast(dtypes.index).valid(is_in_sgpr)
      inline = _u64(self.sgpr.index(inline_idx, ptr=True).load(), self.sgpr.index(inline_idx, ptr=True).load())  # integers: just extend
      if literal is not None: inline = off.eq(_c(255)).where(literal.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32), inline)
      for off_val, val in F64_INLINE.items(): inline = off.eq(_c(off_val)).where(UOp.const(dtypes.uint64, val), inline)
      return is_vgpr.where(vgpr_val, is_sgpr.where(sgpr_val, inline))
    # Guard SGPR read with .valid() to prevent out-of-bounds access when off >= 256
    sgpr_idx = off.cast(dtypes.index).valid(is_in_sgpr)
    sgpr_val = self.sgpr.index(sgpr_idx, ptr=True).load()
    if literal is not None: sgpr_val = off.eq(_c(255)).where(literal, sgpr_val)
    if bits == 16:  # F16 constants differ from pre-populated F32 constants
      for off_val, val in F16_INLINE.items(): sgpr_val = off.eq(_c(off_val)).where(UOp.const(dtypes.uint32, val), sgpr_val)
    vgpr_idx = (off - _c(256)).cast(dtypes.index) * UOp.const(dtypes.index, 32) + lane.cast(dtypes.index)
    vgpr_val = self.vgpr.index(vgpr_idx.valid(is_vgpr), ptr=True).load()
    return is_vgpr.where(vgpr_val, sgpr_val)

  def rsrc_dyn_sized(self, off: UOp, lane: UOp, sizes: dict, key: str, f16: bool = False, literal: UOp | None = None) -> UOp:
    return self.rsrc_dyn(off, lane, 64, literal) if sizes.get(key, 1) == 2 else self.rsrc_dyn(off, lane, 16 if f16 else 32, literal)

  def rpc(self) -> UOp:
    """Read PC as 64-bit byte address."""
    return _u64(self.rsgpr_dyn(_c(PC_LO_IDX)), self.rsgpr_dyn(_c(PC_HI_IDX)))

  def inc_pc(self) -> list[UOp]:
    """Increment PC by instruction size in bytes. Returns [lo_store, hi_store]."""
    new_pc = self.rpc() + UOp.const(dtypes.uint64, self.inst_size)
    lo, hi = _split64(new_pc)
    return [self.wsgpr_dyn(_c(PC_LO_IDX), lo), self.wsgpr_dyn(_c(PC_HI_IDX), hi)]

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

def _compile_sopp(inst: SOPP, ctx: _Ctx, name: str) -> tuple[str, UOp]:
  # Read simm16 dynamically and sign-extend: (val ^ 0x8000) - 0x8000
  simm16_raw = ctx.inst_field(SOPP.simm16)
  simm16 = ((simm16_raw ^ _c(0x8000)) - _c(0x8000)).cast(dtypes.int16)
  if inst.op == SOPPOp.S_ENDPGM:
    return name, UOp.sink(ctx.wsgpr_dyn(_c(PC_LO_IDX), UOp.const(dtypes.uint32, 0xFFFFFFFF)),
                          ctx.wsgpr_dyn(_c(PC_HI_IDX), UOp.const(dtypes.uint32, 0xFFFFFFFF)), arg=KernelInfo(name=name))
  pcode = PCODE.get(inst.op)
  if pcode is not None:
    pc_bytes = ctx.rpc()  # PC is already 64-bit byte address
    vcc, exec_lo = ctx.rsgpr_dyn(_c(VCC_LO.offset)), ctx.rsgpr_dyn(_c(EXEC_LO.offset))
    srcs = {'PC': pc_bytes.cast(dtypes.int64), 'SIMM16': simm16, 'SCC': ctx.rsgpr_dyn(_c(SCC_IDX)), 'VCC': vcc,
            'VCCZ': vcc.eq(UOp.const(dtypes.uint32, 0)).cast(dtypes.uint32), 'EXECZ': exec_lo.eq(UOp.const(dtypes.uint32, 0)).cast(dtypes.uint32)}
    for dest, val in parse_pcode(pcode, srcs, op_name=inst.op.name)[1]:
      if dest == 'PC' or dest.startswith('PC.'):
        lo, hi = _split64(val.cast(dtypes.uint64))
        return name, UOp.sink(ctx.wsgpr_dyn(_c(PC_LO_IDX), lo), ctx.wsgpr_dyn(_c(PC_HI_IDX), hi), arg=KernelInfo(name=name))
  return name, UOp.sink(*ctx.inc_pc(), arg=KernelInfo(name=name))

def _compile_smem(inst: SMEM, ctx: _Ctx, name: str) -> tuple[str, UOp]:
  # Cache invalidation instructions are no-ops in the emulator (we don't model caches)
  if inst.op in (SMEMOp.S_GL1_INV, SMEMOp.S_DCACHE_INV):
    return name, UOp.sink(*ctx.inc_pc(), arg=KernelInfo(name=name))
  # Dynamic sbase field (bits 5:0) - SGPR pair, field value * 2 = register offset
  sbase = ctx.inst_field(SMEM.sbase) * _c(2)
  # Dynamic sdata field (bits 12:6) - destination SGPR
  sdata_reg = ctx.inst_field(SMEM.sdata)
  # Dynamic offset field (bits 52:32) - 21-bit signed immediate
  offset_raw = ctx.inst_field(SMEM.offset)
  offset = (offset_raw.cast(dtypes.int) ^ _c(0x100000, dtypes.int)) - _c(0x100000, dtypes.int)  # sign-extend 21-bit
  # Dynamic soffset field (bits 63:57) - SGPR for additional offset (NULL=124 reads as 0)
  soffset = ctx.inst_field(SMEM.soffset)
  addr = _u64(ctx.rsgpr_dyn(sbase), ctx.rsgpr_dyn(sbase + _c(1))) + offset.cast(dtypes.uint64) + ctx.rsgpr_dyn(soffset).cast(dtypes.uint64)
  ndwords = {SMEMOp.S_LOAD_B32: 1, SMEMOp.S_LOAD_B64: 2, SMEMOp.S_LOAD_B128: 4, SMEMOp.S_LOAD_B256: 8, SMEMOp.S_LOAD_B512: 16}.get(inst.op, 1)
  stores = [ctx.wsgpr_dyn(sdata_reg + _c(i), ctx.vmem.index((addr + UOp.const(dtypes.uint64, i * 4) >> UOp.const(dtypes.uint64, 2)).cast(dtypes.index)))
            for i in range(ndwords)]
  return name, UOp.sink(*stores, *ctx.inc_pc(), arg=KernelInfo(name=name))

def _compile_sop(inst, ctx: _Ctx, name: str) -> tuple[str, UOp]:
  sizes = getattr(inst, 'op_regs', {})
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None

  # Read source operands dynamically
  def rsrc_dyn_scalar(off: UOp, is_64bit: bool) -> UOp:
    """Read scalar source with dynamic offset (SGPR or inline constant).
    For SOP, off is always 0-255 (SGPR or inline constant, never VGPR).
    SGPR buffer has 260 entries: 0-127=SGPRs, 128-255=inline constants, 256-259=special."""
    is_sgpr = off < _c(128)
    # For 64-bit: read SGPR pair if off < 128, else compute inline constant as 64-bit
    # (can't just read from buffer since buffer has 32-bit values)
    if is_64bit:
      sgpr_val = _u64(ctx.rsgpr_dyn(off), ctx.rsgpr_dyn(off + U32_1))
      # Build inline constant: 128-192 = 0-64, 193-208 = -1 to -16
      inline_val = (off - _c(128)).cast(dtypes.uint64)  # positive inline 0-64
      neg_val = (_c(192) - off).cast(dtypes.int64).cast(dtypes.uint64)  # negative -1 to -16
      lit_val = literal.cast(dtypes.uint64) if literal is not None else UOp.const(dtypes.uint64, 0)
      # Select between sgpr, positive inline, negative inline, or literal
      is_neg_inline = (off >= _c(193)) & (off < _c(209))
      is_literal = off.eq(_c(255)) if literal is not None else UOp.const(dtypes.bool, False)
      val = is_sgpr.where(sgpr_val, is_neg_inline.where(neg_val, is_literal.where(lit_val, inline_val)))
      return val
    # 32-bit: read from SGPR buffer (inline constants 128-255 are pre-populated)
    # off is always 0-255 for SOP, all valid SGPR indices
    sgpr_val = ctx.rsgpr_dyn(off)
    # Handle literal (255) - literal value overrides the pre-populated 0
    if literal is not None:
      sgpr_val = off.eq(_c(255)).where(literal, sgpr_val)
    return sgpr_val

  if isinstance(inst, SOPK):
    sdst_off = ctx.inst_field(SOPK.sdst)
    simm16 = ctx.inst_field(SOPK.simm16)
    # Sign-extend simm16
    simm16_sext = simm16.cast(dtypes.int16).cast(dtypes.int32)
    srcs = {'S0': ctx.rsgpr_dyn(sdst_off), 'SIMM16': simm16_sext, 'D0': ctx.rsgpr_dyn(sdst_off)}
    dst_off, dst_size = sdst_off, 1
  elif isinstance(inst, SOP1):
    sdst_off = ctx.inst_field(SOP1.sdst)
    ssrc0_off = ctx.inst_field(SOP1.ssrc0)
    srcs = {'S0': rsrc_dyn_scalar(ssrc0_off, sizes.get('ssrc0', 1) == 2)}
    dst_off, dst_size = sdst_off, sizes.get('sdst', 1)
  elif isinstance(inst, SOP2):
    sdst_off = ctx.inst_field(SOP2.sdst)
    ssrc0_off = ctx.inst_field(SOP2.ssrc0)
    ssrc1_off = ctx.inst_field(SOP2.ssrc1)
    srcs = {'S0': rsrc_dyn_scalar(ssrc0_off, sizes.get('ssrc0', 1) == 2),
            'S1': rsrc_dyn_scalar(ssrc1_off, sizes.get('ssrc1', 1) == 2)}
    if literal is not None: srcs['SIMM32'] = literal
    dst_off, dst_size = sdst_off, sizes.get('sdst', 1)
  elif isinstance(inst, SOPC):
    ssrc0_off = ctx.inst_field(SOPC.ssrc0)
    ssrc1_off = ctx.inst_field(SOPC.ssrc1)
    srcs = {'S0': rsrc_dyn_scalar(ssrc0_off, sizes.get('ssrc0', 1) == 2),
            'S1': rsrc_dyn_scalar(ssrc1_off, sizes.get('ssrc1', 1) == 2)}
    dst_off, dst_size = _c(0), 0  # SOPC writes to SCC, not sdst
  else:
    raise RuntimeError(f"unknown SOP type: {type(inst).__name__}")

  # Use dynamic pcode compilation with dynamic destination
  pcode_result = compile_sop_pcode_dyn(inst.op, srcs, ctx.wsgpr_dyn, ctx.rsgpr_dyn, dst_off, dst_size, ctx.inc_pc, name)
  assert pcode_result is not None, f"no pcode for {type(inst).__name__}: {inst.op.name}"
  return pcode_result

def _compile_vop12(inst, ctx: _Ctx, name: str) -> tuple[str, UOp]:
  op_name = _op_name(inst)
  if op_name == 'V_READFIRSTLANE_B32_E32':
    pcode_result = compile_lane_pcode(inst.op, inst, ctx, ctx.inc_pc, name)
    assert pcode_result is not None, f"no pcode for VOP1: {op_name}"
    return pcode_result
  lane, exec_mask, sizes = UOp.range(32, _next_axis_id(), AxisType.LOOP), ctx.rsgpr_dyn(_c(EXEC_LO.offset)), getattr(inst, 'op_regs', {})
  is_16bit = _is_16bit_op(op_name)
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None
  vdst_reg = ctx.inst_field(VOP1.vdst)
  write_hi_half = is_16bit and (vdst_reg >= _c(128))
  if isinstance(write_hi_half, UOp): vdst_reg = write_hi_half.where(vdst_reg - _c(128), vdst_reg)
  elif write_hi_half: vdst_reg -= 128
  if isinstance(inst, VOP1):
    # Handle VOP1 hi-half source operand (src0 >= v[128] for 16-bit ops)
    src0_off = ctx.inst_field(VOP1.src0)
    s0 = ctx.rsrc_dyn_sized(src0_off, lane, sizes, 'src0', f16=is_16bit, literal=literal)
    if is_16bit:
      src0_hi = src0_off >= _c(384)
      # Only compute hi-half when src0_off >= 384, use guarded index to prevent OOB access
      src0_reg = src0_hi.where(src0_off - _c(384), U32_0)
      s0_hi = (ctx.rvgpr_dyn(src0_reg, lane) >> U32_16) & _c(0xFFFF)
      if isinstance(src0_hi, UOp): s0 = src0_hi.where(s0_hi, s0)
      elif src0_hi: s0 = s0_hi
    srcs = {'S0': s0}
  else:
    vsrc1_reg = ctx.inst_field(VOP2.vsrc1)
    vsrc1_hi = is_16bit and (vsrc1_reg >= _c(128))
    vsrc1_actual = vsrc1_hi.where(vsrc1_reg - _c(128), vsrc1_reg) if isinstance(vsrc1_hi, UOp) else vsrc1_reg - _c(128) if vsrc1_hi else vsrc1_reg
    s1 = ctx.rvgpr_dyn(vsrc1_actual, lane)
    if isinstance(vsrc1_hi, UOp): s1 = vsrc1_hi.where((s1 >> U32_16) & _c(0xFFFF), s1)
    elif vsrc1_hi: s1 = (s1 >> U32_16) & _c(0xFFFF)
    # For FMAC/FMAMK hi-half dest, D0 must also read from hi-half (accumulator is in same half as dest)
    d0 = ctx.rvgpr_dyn(vdst_reg, lane)
    if isinstance(write_hi_half, UOp): d0 = write_hi_half.where((d0 >> U32_16) & _c(0xFFFF), d0)
    elif write_hi_half: d0 = (d0 >> U32_16) & _c(0xFFFF)  # extract hi 16 bits for accumulator
    # Handle VOP2 hi-half src0 operand (src0 >= v[128] for 16-bit ops)
    src0_off = ctx.inst_field(VOP2.src0)
    s0 = ctx.rsrc_dyn(src0_off, lane, bits=16 if is_16bit else 32, literal=literal)
    if is_16bit:
      src0_hi = src0_off >= _c(384)
      # Only compute hi-half when src0_off >= 384, use guarded index to prevent OOB access
      src0_reg = src0_hi.where(src0_off - _c(384), U32_0)
      s0_hi = (ctx.rvgpr_dyn(src0_reg, lane) >> U32_16) & _c(0xFFFF)
      if isinstance(src0_hi, UOp): s0 = src0_hi.where(s0_hi, s0)
      elif src0_hi: s0 = s0_hi
    srcs = {'S0': s0, 'S1': s1, 'D0': d0}
    if inst.op in (VOP2Op.V_FMAAK_F32_E32, VOP2Op.V_FMAMK_F32_E32, VOP2Op.V_FMAAK_F16_E32, VOP2Op.V_FMAMK_F16_E32):
      srcs['SIMM32'] = literal
  pcode_result = compile_vop_pcode(inst.op, srcs, lane, ctx.wvgpr_dyn, ctx.wsgpr_dyn, ctx.rsgpr_dyn, vdst_reg, exec_mask, ctx.inc_pc, name,
                                   opsel_dst_hi=write_hi_half, rvgpr_fn=ctx.rvgpr_dyn)
  assert pcode_result is not None, f"no pcode for {type(inst).__name__}: {inst.op.name}"
  return pcode_result

def _compile_vopc(inst, ctx: _Ctx, name: str, opsel: int = 0, abs_bits: int = 0, neg_bits: int = 0) -> tuple[str, UOp]:
  exec_mask, op_name = ctx.rsgpr_dyn(_c(EXEC_LO.offset)), _op_name(inst)
  is_cmpx, is_16bit, is_64bit = 'CMPX' in op_name, _is_16bit_op(op_name), 'F64' in op_name
  is_vopc = hasattr(inst, 'vsrc1')  # VOPC (e32) vs VOP3 (e64) format

  # Handle both VOPC (vsrc1) and VOP3 (src1) instruction formats - read operands dynamically
  if is_vopc:
    src0_off = ctx.inst_field(VOPC.src0)
    vsrc1_off = ctx.inst_field(VOPC.vsrc1)
    # For 16-bit ops, vsrc1 >= 128 means hi-half of v[vsrc1-128]
    if is_16bit:
      vsrc1_hi = vsrc1_off >= _c(128)
      src1_off = _c(256) + vsrc1_hi.where(vsrc1_off - _c(128), vsrc1_off)
    else:
      vsrc1_hi = False
      src1_off = _c(256) + vsrc1_off
    src0_bits, src1_bits = (64, 64) if is_64bit else (32, 32)
  else:
    src0_off = ctx.inst_field(VOP3.src0)
    src1_off = ctx.inst_field(VOP3.src1)
    dst_off = ctx.inst_field(VOP3.vdst)
    vsrc1_hi = False
    _, src0_bits, _ = inst.operands.get('src0', (None, 32, None))
    _, src1_bits, _ = inst.operands.get('src1', (None, 32, None))
    is_16bit = src0_bits == 16 or src1_bits == 16
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None

  is_float, pcode = any(x in op_name for x in ('_F32', '_F64', '_F16')), PCODE.get(inst.op)
  def get_cmp_bit(lane) -> UOp:
    lc = lane.cast(dtypes.index) if isinstance(lane, UOp) else _c(lane, dtypes.index)
    s0 = ctx.rsrc_dyn(src0_off, lc, src0_bits, literal)
    s1 = ctx.rsrc_dyn(src1_off, lc, src1_bits, literal)
    if is_16bit:
      if isinstance(vsrc1_hi, UOp): s1 = vsrc1_hi.where((s1 >> U32_16) & _c(0xFFFF), s1)
      elif vsrc1_hi: s1 = (s1 >> U32_16) & _c(0xFFFF)
      if opsel: s0, s1 = _apply_opsel(s0, 0, opsel), _apply_opsel(s1, 1, opsel)
    if is_float and (abs_bits or neg_bits):
      s0 = _apply_src_mods(s0, 0, abs_bits, neg_bits, is_16bit, src0_bits == 64)
      s1 = _apply_src_mods(s1, 1, abs_bits, neg_bits, is_16bit, src1_bits == 64)
    if pcode is None: return U32_0
    for dest, val in parse_pcode(pcode, {'S0': s0, 'S1': s1}, lane=lc)[1]:
      if '[laneId]' in dest and ('D0' in dest or 'EXEC' in dest): return val.cast(dtypes.uint32)
    return U32_0

  new_bits = _unroll_lanes(get_cmp_bit, exec_mask, apply_exec=False)
  # Both VOPC and VOP3 clear inactive lane bits (hardware verified)
  new_result = new_bits & exec_mask

  # CMPX e32: writes EXEC only; CMPX e64: writes both EXEC and SDST; non-CMPX: writes dst only
  if is_cmpx:
    stores = [ctx.wsgpr_dyn(_c(EXEC_LO.offset), new_result)]
    if not is_vopc: stores.append(ctx.wsgpr_dyn(dst_off, new_result))
  else:
    stores = [ctx.wsgpr_dyn(dst_off, new_result)] if not is_vopc else [ctx.wsgpr_dyn(_c(VCC_LO.offset), new_result)]
  return name, UOp.sink(*stores, *ctx.inc_pc(), arg=KernelInfo(name=name))

def _compile_vop3(inst: VOP3, ctx: _Ctx, name: str) -> tuple[str, UOp]:
  exec_mask = ctx.rsgpr_dyn(_c(EXEC_LO.offset))
  sizes = getattr(inst, 'op_regs', {})
  opsel, op_name = getattr(inst, 'opsel', 0) or 0, _op_name(inst)

  # Lane operations
  if op_name in ('V_READLANE_B32', 'V_READFIRSTLANE_B32', 'V_READFIRSTLANE_B32_E64', 'V_WRITELANE_B32'):
    pcode_result = compile_lane_pcode(inst.op, inst, ctx, ctx.inc_pc, name)
    assert pcode_result is not None, f"no pcode for VOP3: {op_name}"
    return pcode_result

  # VOP3 VOPC (v_cmp_*_e64) - delegate to unified VOPC handler
  if 'V_CMP' in op_name or 'V_CMPX' in op_name:
    return _compile_vopc(inst, ctx, name, opsel=opsel, abs_bits=getattr(inst, 'abs', 0) or 0, neg_bits=getattr(inst, 'neg', 0) or 0)

  # Regular VOP3 - read operands dynamically
  lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
  is_f16_op = 'F16' in op_name
  vdst_reg = ctx.inst_field(VOP3.vdst)
  src0_off = ctx.inst_field(VOP3.src0)
  src1_off = ctx.inst_field(VOP3.src1)
  src2_off = ctx.inst_field(VOP3.src2) if inst.src2 is not None else None
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None
  src0 = ctx.rsrc_dyn_sized(src0_off, lane, sizes, 'src0', f16=is_f16_op, literal=literal)
  src1 = ctx.rsrc_dyn_sized(src1_off, lane, sizes, 'src1', f16=is_f16_op, literal=literal)
  src2 = ctx.rsrc_dyn_sized(src2_off, lane, sizes, 'src2', f16=is_f16_op, literal=literal) if src2_off is not None else None
  if _is_16bit_op(op_name):
    src0, src1 = _apply_opsel(src0, 0, opsel), _apply_opsel(src1, 1, opsel)
    if src2 is not None: src2 = _apply_opsel(src2, 2, opsel)
  abs_bits, neg_bits = getattr(inst, 'abs', 0) or 0, getattr(inst, 'neg', 0) or 0
  is_16bit_op = _is_16bit_op(op_name)
  if abs_bits or neg_bits:
    src0 = _apply_src_mods(src0, 0, abs_bits, neg_bits, is_16bit_op, sizes.get('src0', 1) == 2)
    if src1 is not None: src1 = _apply_src_mods(src1, 1, abs_bits, neg_bits, is_16bit_op, sizes.get('src1', 1) == 2)
    if src2 is not None: src2 = _apply_src_mods(src2, 2, abs_bits, neg_bits, is_16bit_op, sizes.get('src2', 1) == 2)
  srcs = {'S0': src0, 'S1': src1}
  if src2 is not None: srcs['S2'] = src2
  if inst.op in (VOP3Op.V_CNDMASK_B32_E64, VOP3Op.V_CNDMASK_B16) and src2 is not None: srcs['VCC'] = src2
  # FMAC instructions need D0 (accumulator) from destination register
  if 'FMAC' in op_name: srcs['D0'] = ctx.rvgpr_dyn(vdst_reg, lane)
  opsel_dst_hi = bool(opsel & 0b1000) and _is_16bit_op(op_name)
  if opsel_dst_hi:
    stores = compile_vop_pcode(inst.op, srcs, lane, ctx.wvgpr_dyn, ctx.wsgpr_dyn, ctx.rsgpr_dyn, vdst_reg, exec_mask, opsel_dst_hi=True,
                               rvgpr_fn=ctx.rvgpr_dyn)
    if stores is not None:
      return name, UOp.sink(*stores, *ctx.inc_pc(), arg=KernelInfo(name=name))
  pcode_result = compile_vop_pcode(inst.op, srcs, lane, ctx.wvgpr_dyn, ctx.wsgpr_dyn, ctx.rsgpr_dyn, vdst_reg, exec_mask, ctx.inc_pc, name,
                                   rvgpr_fn=ctx.rvgpr_dyn)
  assert pcode_result is not None, f"no pcode for VOP3: {inst.op.name}"
  return pcode_result

def _compile_vop3sd(inst: VOP3SD, ctx: _Ctx, name: str) -> tuple[str, UOp]:
  exec_mask = ctx.rsgpr_dyn(_c(EXEC_LO.offset))
  sizes, op_name = getattr(inst, 'op_regs', {}), _op_name(inst)
  pcode = PCODE.get(inst.op)
  assert pcode is not None, f"no pcode for VOP3SD: {op_name}"

  # Read operands dynamically from instruction encoding
  vdst_reg = ctx.inst_field(VOP3SD.vdst)
  sdst_off = ctx.inst_field(VOP3SD.sdst)
  src0_off = ctx.inst_field(VOP3SD.src0)
  src1_off = ctx.inst_field(VOP3SD.src1)
  src2_off = ctx.inst_field(VOP3SD.src2) if inst.src2 is not None else None
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None

  has_carry_in = 'src2' in inst.operands and inst.operands['src2'][2] == OpType.OPR_SREG
  vcc_in_off = src2_off if has_carry_in and src2_off is not None else sdst_off

  lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
  src0, src1 = ctx.rsrc_dyn_sized(src0_off, lane, sizes, 'src0', literal=literal), ctx.rsrc_dyn_sized(src1_off, lane, sizes, 'src1', literal=literal)
  src2 = ctx.rsrc_dyn_sized(src2_off, lane, sizes, 'src2', literal=literal) if src2_off is not None else None
  srcs = {'S0': src0, 'S1': src1, 'VCC': ctx.rsgpr_dyn(vcc_in_off), 'EXEC': exec_mask, 'SCC': ctx.rsgpr_dyn(_c(SCC_IDX))}
  if src2 is not None: srcs['S2'] = src2
  _, assigns = parse_pcode(pcode, srcs, lane, op_name=op_name)

  has_per_lane_vcc = any('[laneId]' in dest for dest, _ in assigns if dest.startswith('VCC') or dest.startswith('D0.u64'))
  if has_per_lane_vcc:
    # VCC computation: RANGE+REDUCE gets axis ID first (lower ID = runs first)
    # This ensures VCC reads source values BEFORE VGPR stores modify them
    def get_vcc_bit(lane_uop) -> UOp:
      s0, s1 = ctx.rsrc_dyn_sized(src0_off, lane_uop, sizes, 'src0', literal=literal), ctx.rsrc_dyn_sized(src1_off, lane_uop, sizes, 'src1', literal=literal)
      s2 = ctx.rsrc_dyn_sized(src2_off, lane_uop, sizes, 'src2', literal=literal) if src2_off is not None else None
      lane_srcs = {'S0': s0, 'S1': s1, 'VCC': ctx.rsgpr_dyn(vcc_in_off), 'EXEC': exec_mask, 'SCC': ctx.rsgpr_dyn(_c(SCC_IDX))}
      if s2 is not None: lane_srcs['S2'] = s2
      vcc_bit = U32_0
      for dest, val in parse_pcode(pcode, lane_srcs, lane_uop, op_name=op_name)[1]:
        if dest.startswith('VCC') or (dest.startswith('D0.u64') and '[laneId]' in dest): vcc_bit = val.cast(dtypes.uint32)
      return vcc_bit
    final_vcc = _unroll_lanes(get_vcc_bit, exec_mask)
    # VGPR stores: RANGE gets axis ID second (higher ID = runs after VCC loop)
    lane3 = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    s0, s1 = ctx.rsrc_dyn_sized(src0_off, lane3, sizes, 'src0', literal=literal), ctx.rsrc_dyn_sized(src1_off, lane3, sizes, 'src1', literal=literal)
    s2 = ctx.rsrc_dyn_sized(src2_off, lane3, sizes, 'src2', literal=literal) if src2_off is not None else None
    lane_srcs = {'S0': s0, 'S1': s1, 'VCC': ctx.rsgpr_dyn(vcc_in_off), 'EXEC': exec_mask, 'SCC': ctx.rsgpr_dyn(_c(SCC_IDX))}
    if s2 is not None: lane_srcs['S2'] = s2
    d0_val = None
    for dest, val in parse_pcode(pcode, lane_srcs, lane3, op_name=op_name)[1]:
      if dest.startswith('D0') and '[laneId]' not in dest: d0_val = val
    vgpr_stores = []
    if d0_val is not None:
      if d0_val.dtype in (dtypes.uint64, dtypes.int64, dtypes.float64):
        lo, hi = _split64(d0_val)
        vgpr_stores.extend([ctx.wvgpr_dyn(vdst_reg, lane3, lo, exec_mask), ctx.wvgpr_dyn(vdst_reg + U32_1, lane3, hi, exec_mask)])
      else:
        d0_u32 = d0_val.bitcast(dtypes.uint32) if d0_val.dtype in (dtypes.float32, dtypes.half) else d0_val.cast(dtypes.uint32)
        vgpr_stores.append(ctx.wvgpr_dyn(vdst_reg, lane3, d0_u32, exec_mask))
    # Write carry output (wsgpr_dyn handles NULL register 124)
    vcc_write = ctx.wsgpr_dyn(sdst_off, final_vcc)
    if vgpr_stores:
      # VCC write must come first in sink to ensure VCC loop runs before VGPR loop
      return name, UOp.sink(vcc_write, UOp.sink(*vgpr_stores).end(lane3), *ctx.inc_pc(), arg=KernelInfo(name=name))
    return name, UOp.sink(vcc_write, *ctx.inc_pc(), arg=KernelInfo(name=name))
  else:
    pcode_result = compile_vop_pcode(inst.op, srcs, lane, ctx.wvgpr_dyn, ctx.wsgpr_dyn, ctx.rsgpr_dyn, vdst_reg, exec_mask, ctx.inc_pc, name,
                                     sdst_reg=inst.sdst.offset, rvgpr_fn=ctx.rvgpr_dyn)
    assert pcode_result is not None, f"no pcode for VOP3SD: {op_name}"
    return pcode_result

def _compile_vop3p(inst: VOP3P, ctx: _Ctx, name: str) -> tuple[str, UOp]:
  lane, exec_mask = UOp.range(32, _next_axis_id(), AxisType.LOOP), ctx.rsgpr_dyn(_c(EXEC_LO.offset))
  # Read register fields dynamically for deduplication
  vdst_reg = ctx.inst_field(VOP3P.vdst)
  src0_off = ctx.inst_field(VOP3P.src0)
  src1_off = ctx.inst_field(VOP3P.src1)
  src2_off = ctx.inst_field(VOP3P.src2) if hasattr(inst, 'src2') and inst.src2 is not None else None
  src0 = ctx.rsrc_dyn(src0_off, lane, 16)
  src1 = ctx.rsrc_dyn(src1_off, lane, 16)
  src2 = ctx.rsrc_dyn(src2_off, lane, 16) if src2_off is not None else None
  opsel, opsel_hi = getattr(inst, 'opsel', 0) or 0, getattr(inst, 'opsel_hi', 3) if getattr(inst, 'opsel_hi', 3) is not None else 3
  opsel_hi2 = getattr(inst, 'opsel_hi2', 1) if getattr(inst, 'opsel_hi2', 1) is not None else 1
  neg, neg_hi = getattr(inst, 'neg', 0) or 0, getattr(inst, 'neg_hi', 0) or 0
  def get_half_bits(val: UOp, use_hi: bool, apply_neg: bool = False) -> UOp:
    bits = ((val >> UOp.const(dtypes.uint32, 16)) if use_hi else val) & UOp.const(dtypes.uint32, 0xFFFF)
    if apply_neg: bits = bits.cast(dtypes.uint16).bitcast(dtypes.half).neg().bitcast(dtypes.uint16).cast(dtypes.uint32)
    return bits
  def build_remapped_src(src: UOp, opsel_lo_bit: int, opsel_hi_bit: int, neg_lo_bit: int, neg_hi_bit: int) -> UOp:
    return get_half_bits(src, bool(opsel_lo_bit), bool(neg_lo_bit)) | (get_half_bits(src, bool(opsel_hi_bit), bool(neg_hi_bit)) << UOp.const(dtypes.uint32, 16))
  s0_new = build_remapped_src(src0, opsel & 1, opsel_hi & 1, neg & 1, neg_hi & 1)
  s1_new = build_remapped_src(src1, opsel & 2, opsel_hi & 2, neg & 2, neg_hi & 2)
  s2_new = build_remapped_src(src2, opsel & 4, 1 if opsel_hi2 else 0, neg & 4, neg_hi & 4) if src2 is not None else None
  op_name = _op_name(inst)

  # WMMA: Wave Matrix Multiply-Accumulate
  if 'WMMA' in op_name and ('16X16X16_F16' in op_name or '16X16X16_BF16' in op_name):
    # Dynamic register fields for deduplication
    src0_r = ctx.inst_field(VOP3P.src0) - _c(256)
    src1_r = ctx.inst_field(VOP3P.src1) - _c(256)
    src2_r = ctx.inst_field(VOP3P.src2) - _c(256)
    is_f16_output = 'F16_16X16X16_F16' in op_name or 'BF16_16X16X16_BF16' in op_name  # F16/BF16 output vs F32 output
    is_bf16 = 'BF16' in op_name
    def f16_to_f32(bits: UOp) -> UOp: return bits.cast(dtypes.uint16).bitcast(dtypes.half).cast(dtypes.float32)
    def bf16_to_f32(bits: UOp) -> UOp: return (bits.cast(dtypes.uint32) << UOp.const(dtypes.uint32, 16)).bitcast(dtypes.float32)
    def read_f16_mat(src):
      cvt = bf16_to_f32 if is_bf16 else f16_to_f32
      return [f for l in range(16) for r in range(8) for v in [ctx.rvgpr_dyn(src + _c(r), UOp.const(dtypes.index, l))]
              for f in [cvt(v & UOp.const(dtypes.uint32, 0xFFFF)), cvt(v >> UOp.const(dtypes.uint32, 16))]]
    mat_a, mat_b = read_f16_mat(src0_r), read_f16_mat(src1_r)
    acc_cvt = bf16_to_f32 if is_bf16 else f16_to_f32
    if is_f16_output:
      # RDNA3 F16/BF16 output: uses 8 VGPRs (same as F32), f16/bf16 values in lo 16 bits of each VGPR
      # Layout: half16 per lane where even indices (0,2,4,...,14) = lo halves of VGPRs 0-7
      # Read accumulator: 8 regs × 32 lanes, each VGPR's lo 16 bits holds one f16/bf16
      mat_c = [acc_cvt(ctx.rvgpr_dyn(src2_r + _c(i // 32), UOp.const(dtypes.index, i % 32)) & UOp.const(dtypes.uint32, 0xFFFF))
               for i in range(256)]
      mat_d = [sum(mat_a[row*16+k] * mat_b[col*16+k] for k in range(16)) + mat_c[row*16+col] for row in range(16) for col in range(16)]
      # Write f16/bf16 results to lo 16 bits of each VGPR
      def f32_to_f16_bits(v: UOp) -> UOp: return v.cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32)
      def f32_to_bf16_bits(v: UOp) -> UOp: return (v.bitcast(dtypes.uint32) >> UOp.const(dtypes.uint32, 16)) & UOp.const(dtypes.uint32, 0xFFFF)
      out_cvt = f32_to_bf16_bits if is_bf16 else f32_to_f16_bits
      stores = [ctx.wvgpr_dyn(vdst_reg + _c(i // 32), UOp.const(dtypes.index, i % 32),
                out_cvt(mat_d[i]), exec_mask) for i in range(256)]
    else:
      # F32 output: accumulator and output are f32
      mat_c = [ctx.rvgpr_dyn(src2_r + _c(i // 32), UOp.const(dtypes.index, i % 32)).bitcast(dtypes.float32) for i in range(256)]
      mat_d = [sum(mat_a[row*16+k] * mat_b[col*16+k] for k in range(16)) + mat_c[row*16+col] for row in range(16) for col in range(16)]
      stores = [ctx.wvgpr_dyn(vdst_reg + _c(i // 32), UOp.const(dtypes.index, i % 32), mat_d[i].bitcast(dtypes.uint32), exec_mask) for i in range(256)]
    return name, UOp.sink(*stores, *ctx.inc_pc(), arg=KernelInfo(name=name))

  pcode = PCODE.get(inst.op)
  if pcode is not None:
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
      s2_mod = apply_neg_mix(apply_abs(src2, 4, 4, 4), 4, 4, 4) if src2 is not None else UOp.const(dtypes.uint32, 0)
      srcs = {'S0': s0_mod, 'S1': s1_mod, 'S2': s2_mod,
              'OPSEL_HI': UOp.const(dtypes.uint32, combined_opsel_hi), 'OPSEL': UOp.const(dtypes.uint32, opsel)}
    else:
      srcs = {'S0': s0_new, 'S1': s1_new}
      if s2_new is not None: srcs['S2'] = s2_new
    stores = compile_vop_pcode(inst.op, srcs, lane, ctx.wvgpr_dyn, ctx.wsgpr_dyn, ctx.rsgpr_dyn, vdst_reg, exec_mask, rvgpr_fn=ctx.rvgpr_dyn)
    if stores is not None:
      return name, UOp.sink(*stores, *ctx.inc_pc(), arg=KernelInfo(name=name))
  return name, UOp.sink(*ctx.inc_pc(), arg=KernelInfo(name=name))

def _compile_vopd(inst: VOPD, ctx: _Ctx, name: str) -> tuple[str, UOp]:
  exec_mask = ctx.rsgpr_dyn(_c(EXEC_LO.offset))
  # Read operands dynamically
  vdstx_reg = ctx.inst_field(VOPD.vdstx)
  # vdsty has complex encoding: actual = (raw << 1) | ((vdstx & 1) ^ 1)
  vdsty_raw = ctx.inst_field(VOPD.vdsty)
  vdsty_reg = (vdsty_raw << U32_1) | ((vdstx_reg & U32_1) ^ U32_1)
  srcx0_off = ctx.inst_field(VOPD.srcx0)
  srcy0_off = ctx.inst_field(VOPD.srcy0)
  vsrcx1_reg = ctx.inst_field(VOPD.vsrcx1)
  vsrcy1_reg = ctx.inst_field(VOPD.vsrcy1)
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None

  lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
  srcy0, srcy1 = ctx.rsrc_dyn(srcy0_off, lane, literal=literal), ctx.rvgpr_dyn(vsrcy1_reg, lane)
  all_stores = []
  for op, src0_off, vsrc1_reg, vdst_reg, label in [(inst.opx, srcx0_off, vsrcx1_reg, vdstx_reg, 'X'),
                                                    (inst.opy, srcy0_off, vsrcy1_reg, vdsty_reg, 'Y')]:
    vop = VOPD_TO_VOP2.get(op)
    assert vop is not None, f"no VOP mapping for VOPD {label}: {op}"
    if label == 'Y': srcs = {'S0': srcy0, 'S1': srcy1, 'D0': ctx.rvgpr_dyn(vdst_reg, lane)}
    else: srcs = {'S0': ctx.rsrc_dyn(src0_off, lane, literal=literal), 'S1': ctx.rvgpr_dyn(vsrc1_reg, lane), 'D0': ctx.rvgpr_dyn(vdst_reg, lane)}
    if op in (VOPDOp.V_DUAL_FMAAK_F32, VOPDOp.V_DUAL_FMAMK_F32): srcs['SIMM32'] = literal
    if op == VOPDOp.V_DUAL_CNDMASK_B32: srcs['VCC'] = ctx.rsgpr_dyn(_c(VCC_LO.offset))
    pcode = PCODE.get(vop)
    assert pcode is not None, f"no pcode for VOPD {label}: {vop}"
    srcs.update({'VCC': ctx.rsgpr_dyn(_c(VCC_LO.offset)), 'EXEC': exec_mask, 'SCC': ctx.rsgpr_dyn(_c(SCC_IDX))})
    for dest, val in parse_pcode(pcode, srcs, lane, op_name=vop.name)[1]:
      if dest.startswith('D0'): all_stores.append(ctx.wvgpr_dyn(vdst_reg, lane, _val_to_u32(val), exec_mask, after=srcy1))
  return name, UOp.sink(UOp.group(*all_stores).end(lane), *ctx.inc_pc(), arg=KernelInfo(name=name))

def _compile_mem_op(inst, ctx: _Ctx, name: str) -> tuple[str, UOp]:
  """Unified memory operation compiler for DS, FLAT, GLOBAL, SCRATCH."""
  exec_mask, op_name = ctx.rsgpr_dyn(_c(EXEC_LO.offset)), _op_name(inst)
  pcode = PCODE.get(inst.op)
  if pcode is None: return name, UOp.sink(*ctx.inc_pc(), arg=KernelInfo(name=name))

  is_lds = isinstance(inst, DS)
  is_scratch = isinstance(inst, SCRATCH)
  mem = ctx.lds if is_lds else ctx.scratch if is_scratch else ctx.vmem
  addr_shift = UOp.const(dtypes.uint32 if is_lds else dtypes.uint64, 2)

  # Extract register info - all dynamic for deduplication
  if is_lds:
    addr_reg = ctx.inst_field(DS.addr)
    vdata_reg = ctx.inst_field(DS.data0)
    vdst_reg = ctx.inst_field(DS.vdst)
    offset0 = ctx.inst_field(DS.offset0)
    offset1 = ctx.inst_field(DS.offset1)
    offset = offset0  # DS uses offset0 as primary offset
    saddr_reg = None
  else:
    addr_reg = ctx.inst_field(type(inst).addr)
    vdata_reg = ctx.inst_field(type(inst).data)
    vdst_reg = ctx.inst_field(type(inst).vdst)
    # Dynamic 13-bit signed offset: cast to int, then (val ^ 0x1000) - 0x1000 for sign extension
    raw_offset = ctx.inst_field(type(inst).offset).cast(dtypes.int)
    offset = (raw_offset ^ _c(0x1000, dtypes.int)) - _c(0x1000, dtypes.int)
    offset0, offset1 = 0, 0
    # Dynamic saddr - read field, NULL (124) or >= 128 means no saddr
    saddr_reg = ctx.inst_field(type(inst).saddr) if hasattr(inst, 'saddr') else None

  # Data width
  ndwords = 4 if '_B128' in op_name or 'B128' in op_name else 3 if '_B96' in op_name or 'B96' in op_name else 2 if '_B64' in op_name or 'B64' in op_name else 1
  is_64bit = ndwords >= 2 or '_U64' in op_name or '_I64' in op_name or '_F64' in op_name
  is_atomic, glc = 'ATOMIC' in op_name, getattr(inst, 'glc', 0)
  has_data1 = is_lds and hasattr(inst, 'data1') and inst.data1 is not None
  data1_reg = ctx.inst_field(DS.data1) if is_lds else _c(0)

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
    saddr_base = _u64(ctx.rsgpr_dyn(saddr_reg), ctx.rsgpr_dyn(saddr_reg + U32_1)) if saddr_reg is not None else UOp.const(dtypes.uint64, 0)
    vaddr_base = _u64(ctx.rvgpr_dyn(addr_reg, lane), ctx.rvgpr_dyn(addr_reg + _c(1), lane))
    # When saddr is valid: base = saddr pair, vaddr is 32-bit offset; otherwise: base = 0, vaddr is 64-bit address
    base_addr = use_saddr.where(saddr_base + ctx.rvgpr_dyn(addr_reg, lane).cast(dtypes.uint64), vaddr_base)
    return base_addr + offset64

  def wmem(addr: UOp, val: UOp, active: UOp) -> UOp:
    idx = mem.index((addr >> addr_shift).cast(dtypes.index))
    return idx.store(active.where(val, idx.load()))

  def make_srcs(lane: UOp) -> dict:
    addr = make_addr(lane)
    if is_lds:
      if 'B128' in op_name or 'B96' in op_name:
        data = {'DATA': ctx.rvgpr_dyn(vdata_reg, lane), 'DATA1': ctx.rvgpr_dyn(vdata_reg + _c(1), lane),
                'DATA2': ctx.rvgpr_dyn(vdata_reg + _c(2), lane), 'DATA3': ctx.rvgpr_dyn(vdata_reg + _c(3), lane)}
      elif 'B32' in op_name:
        data = {'DATA': ctx.rvgpr_dyn(vdata_reg, lane), 'DATA2': ctx.rvgpr_dyn(data1_reg, lane) if has_data1 else UOp.const(dtypes.uint32, 0)}
      else:
        data = {'DATA': _u64(ctx.rvgpr_dyn(vdata_reg, lane), ctx.rvgpr_dyn(vdata_reg + _c(1), lane)),
                'DATA2': _u64(ctx.rvgpr_dyn(data1_reg, lane), ctx.rvgpr_dyn(data1_reg + _c(1), lane)) if has_data1 else UOp.const(dtypes.uint64, 0)}
      return {'ADDR': addr, 'ADDR_BASE': addr, 'OFFSET': offset, 'OFFSET0': offset0, 'OFFSET1': offset1, '_lds': mem, **data}
    active = _lane_active(exec_mask, lane)
    if is_atomic:
      return {'ADDR': addr, 'DATA': _u64(ctx.rvgpr_dyn(vdata_reg, lane), ctx.rvgpr_dyn(vdata_reg + _c(1), lane)) if is_64bit else ctx.rvgpr_dyn(vdata_reg, lane),
              '_vmem': mem, '_active': active}
    vdata = ctx.rvgpr_dyn(vdata_reg, lane).cast(dtypes.uint64) if 'STORE' in op_name else ctx.rvgpr_dyn(vdst_reg, lane) if 'D16' in op_name else UOp.const(dtypes.uint32, 0)
    if 'STORE' in op_name and ndwords >= 2: vdata = vdata | (ctx.rvgpr_dyn(vdata_reg + _c(1), lane).cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))
    srcs = {'ADDR': addr, 'VDATA': vdata, '_vmem': mem, '_active': active}
    for i in range(ndwords): srcs[f'VDATA{i}'] = ctx.rvgpr_dyn(vdata_reg + _c(i), lane) if 'STORE' in op_name else UOp.const(dtypes.uint32, 0)
    return srcs

  def make_stores(dest: str, val: UOp, lane: UOp, active: UOp, writes_return_data: bool, pcode_vars: dict) -> list[UOp]:
    if dest.startswith('MEM['):
      if is_lds or is_atomic: return _write_val(dest, val[1], wmem, val[0], active, is_mem=True)
      data_bits = 8 if '.b8' in dest else 16 if '.b16' in dest else 64 if '.b64' in dest else 32
      if is_scratch: return _mem_store_bytes(mem, val[0], val[1], active, data_bits)
      return _mem_store(mem, val[0], val[1], active, 64, data_bits)
    if dest.startswith('RETURN_DATA') and writes_return_data:
      if (m := re.match(r'RETURN_DATA\[(\d+)\s*:\s*(\d+)\]', dest)):
        bit_width, dword_idx = int(m.group(1)) - int(m.group(2)) + 1, int(m.group(2)) // 32
        is_64 = '.b64' if bit_width == 64 else ''
        return _write_val(is_64, val, lambda r, v, l, e: ctx.wvgpr_dyn(r, l, v, e), vdst_reg + _c(dword_idx), lane, exec_mask)
      return _write_val(dest, val, lambda r, v, l, e: ctx.wvgpr_dyn(r, l, v, e), vdst_reg, lane, exec_mask)
    return []

  # DS-specific: check for 2ADDR pattern needing separate ranges
  if is_lds:
    dummy_lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    _, assigns = parse_pcode(pcode, make_srcs(dummy_lane), lane=dummy_lane, op_name=op_name)
    mem_assigns = [d for d, _ in assigns if d.startswith('MEM[')]
    mem_addrs = set(re.match(r'MEM\[([^\]]+)\]', d).group(1) if re.match(r'MEM\[([^\]]+)\]', d) else d for d in mem_assigns)
    use_separate_ranges = (len(mem_addrs) > 1 or '2ADDR' in op_name) and 'STOREXCHG' not in op_name
    if use_separate_ranges:
      ended = []
      for i, (dest, _) in enumerate(assigns):
        lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
        active = _lane_active(exec_mask, lane)
        _, lane_assigns = parse_pcode(pcode, make_srcs(lane), lane=lane, op_name=op_name)
        ended.extend(s.end(lane) for s in make_stores(dest, lane_assigns[i][1], lane, active, True, {}))
      return (name, UOp.sink(*ended, *ctx.inc_pc(), arg=KernelInfo(name=name))) if ended else (name, UOp.sink(*ctx.inc_pc(), arg=KernelInfo(name=name)))

  # Standard path: single lane range
  writes_return_data = '_RTN' in op_name or (is_lds and op_name.startswith('DS_LOAD')) or (is_atomic and glc)
  lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
  active = _lane_active(exec_mask, lane)
  pcode_vars, assigns = parse_pcode(pcode, make_srcs(lane), lane=lane, op_name=op_name)
  stores = [s for dest, val in assigns for s in make_stores(dest, val, lane, active, writes_return_data, pcode_vars)]

  # FLAT/GLOBAL/SCRATCH: collect VDATA slices for loads
  if not is_lds and not is_atomic:
    for dword_idx, val in sorted(_collect_data_slices(assigns, 'VDATA', pcode_vars, op_name).items()):
      stores.append(ctx.wvgpr_dyn(vdst_reg + _c(dword_idx), lane, val, exec_mask))

  if stores: return name, UOp.sink(UOp.sink(*stores).end(lane), *ctx.inc_pc(), arg=KernelInfo(name=name))
  return name, UOp.sink(*ctx.inc_pc(), arg=KernelInfo(name=name))

# Dispatch table: instruction type -> handler function
_INST_HANDLERS: dict[type, callable] = {
  SOPP: _compile_sopp, SMEM: _compile_smem, SOP1: _compile_sop, SOP2: _compile_sop, SOPC: _compile_sop, SOPK: _compile_sop,
  VOP1: _compile_vop12, VOP1_SDST: _compile_vop12, VOP2: _compile_vop12, VOPC: _compile_vopc, VOP3: _compile_vop3, VOP3_SDST: _compile_vop3,
  VOP3SD: _compile_vop3sd, VOP3P: _compile_vop3p, VOPD: _compile_vopd,
  DS: _compile_mem_op, FLAT: _compile_mem_op, GLOBAL: _compile_mem_op, SCRATCH: _compile_mem_op,
}

# ═══════════════════════════════════════════════════════════════════════════════
# PROGRAM DECODE AND COMPILATION
# ═══════════════════════════════════════════════════════════════════════════════

# Backend selection: EMU2_BACKEND=clang (default) or llvm
EMU2_BACKEND = getenv("EMU2_BACKEND", "clang")

def _get_backend():
  """Get renderer, compiler, and program class based on EMU2_BACKEND."""
  if EMU2_BACKEND == "llvm":
    from tinygrad.renderer.llvmir import CPULLVMRenderer
    from tinygrad.runtime.support.compiler_cpu import CPULLVMCompiler
    from tinygrad.runtime.ops_cpu import CPUProgram
    return CPULLVMRenderer(), CPULLVMCompiler(), CPUProgram
  else:  # clang (default)
    from tinygrad.renderer.cstyle import ClangRenderer
    from tinygrad.runtime.support.compiler_cpu import ClangJITCompiler
    from tinygrad.runtime.ops_cpu import CPUProgram
    return ClangRenderer(), ClangJITCompiler(), CPUProgram

_emu_renderer, _emu_compiler, _ProgramClass = _get_backend()

def _elf_symbol_offsets(obj: bytes) -> dict[str, int]:
  """Parse ELF object file and return {symbol_name: offset} for all defined symbols."""
  from tinygrad.runtime.support.elf import elf_loader, libc
  def _strtab(blob: bytes, idx: int) -> str: return blob[idx:blob.find(b'\x00', idx)].decode('utf-8')
  _, sections, _ = elf_loader(obj)
  symtab_sec = next((s for s in sections if s.header.sh_type == libc.SHT_SYMTAB), None)
  if symtab_sec is None: return {}
  strtab_sec = sections[symtab_sec.header.sh_link] if symtab_sec.header.sh_link < len(sections) else None
  if strtab_sec is None: return {}
  symbols = (libc.Elf64_Sym * (symtab_sec.header.sh_size // symtab_sec.header.sh_entsize)).from_buffer_copy(symtab_sec.content)
  return {name: sections[sym.st_shndx].header.sh_addr + sym.st_value
          for sym in symbols if 0 < sym.st_shndx < len(sections) and (name := _strtab(strtab_sec.content, sym.st_name))}

@functools.cache
def _get_inst_sink(inst_bytes: bytes) -> tuple[UOp, tuple[int, int, int]]:
  """Build UOp sink for instruction bytes. Returns (sink, (base, mask, size)) with canonical name."""
  inst = decode_inst(inst_bytes)
  inst_size = inst.size()  # bytes

  sgpr, vgpr, vmem, lds, scratch = _define_bufs()
  ctx = _Ctx(sgpr, vgpr, vmem, lds, scratch, inst_size)

  # Look up handler by type, falling back to base classes for _LIT variants
  handler = _INST_HANDLERS.get(type(inst))
  if handler is None:
    for base in type(inst).__mro__:
      if base in _INST_HANDLERS:
        handler = _INST_HANDLERS[base]
        break
  if handler is None: raise RuntimeError(f"[emu2] unimplemented instruction type: {type(inst).__name__} {_op_name(inst)}")
  _, sink = handler(inst, ctx, "")  # name replaced below
  # Compute canonical mask and name after handler populates dyn_fields
  base, mask, size = ctx.canonical_mask(inst_bytes)
  canonical_name = f"{_op_name(inst).lower()}_{base.to_bytes(size, 'little').hex()}"
  return sink.replace(arg=KernelInfo(name=canonical_name)).rtag(1), (base, mask, size)

_canonical_prg_cache: list[tuple[int, int, int, ProgramSpec]] = []  # [(base, mask, size, prg), ...]
_last_compiled_new: bool = False  # set by _get_inst_prg when compiling new instruction

def _match_canonical(inst_int: int, inst_size: int) -> ProgramSpec | None:
  """Check if instruction matches any cached (base, mask, size) pattern."""
  for base, mask, size, prg in _canonical_prg_cache:
    if inst_size != size: continue  # must match instruction size exactly
    if (inst_int & mask) == base: return prg
  return None

@functools.cache
def _get_inst_prg(inst_bytes: bytes) -> ProgramSpec:
  """Compile instruction bytes to ProgramSpec. Cached by instruction bytes, with canonical dedup."""
  global _last_compiled_new
  # Decode instruction to get size for canonical matching
  inst = decode_inst(inst_bytes)
  inst_size = inst.size()
  inst_int = int.from_bytes(inst_bytes[:inst_size], 'little')
  # Check canonical cache BEFORE building sink (avoids expensive UOp construction)
  if (prg := _match_canonical(inst_int, inst_size)) is not None:
    _last_compiled_new = False
    return prg
  sink, (base, mask, size) = _get_inst_sink(inst_bytes)
  with Context(NOOPT=1, IGNORE_OOB=1, TUPLE_ORDER=0):
    prg = get_program(sink, _emu_renderer)
  _canonical_prg_cache.append((base, mask, size, prg))
  _last_compiled_new = True
  return prg

@functools.cache
def decode_program(data: bytes) -> dict[int, tuple[str, object, list[int], object]]:
  """Decode program to {pc: (name, program, globals, holder)}."""

  # Collect all instruction programs
  inst_info: list[tuple[int, ProgramSpec]] = []  # (pc_bytes, prg)
  i = 0
  while i < len(data):
    inst = decode_inst(data[i:])
    if isinstance(inst, SOPP) and inst.op == SOPPOp.S_CODE_END: break
    try:
      prg = _get_inst_prg(bytes(data[i:i + inst.size() + 4]))
      inst_info.append((i, prg))  # PC is now byte offset
      if DEBUG >= 3:
        try: inst_str = repr(inst)
        except Exception: inst_str = f"<{type(inst).__name__} at PC={i}>"
        msg = f"[emu2] PC={i}: {inst_str}"
        print(colored(msg, 'green') if _last_compiled_new else msg)
        if DEBUG >= 4: print(f"{colored(prg.src, 'BLACK')}")
    except Exception as e:
      try: inst_str = repr(inst)
      except Exception: inst_str = f"<{type(inst).__name__}>"
      raise RuntimeError(f"[emu2] Failed to compile PC={i} {inst_str}: {type(e).__name__}: {e}") from e
    i += inst.size()

  if not inst_info: return {}

  # Batch compile and create function pointers
  from tinygrad.runtime.support.elf import jit_loader
  seen_funcs: set[str] = set()
  combined_src_parts: list[str] = []
  for pc, prg in inst_info:
    if prg.function_name not in seen_funcs:
      seen_funcs.add(prg.function_name)
      combined_src_parts.append(prg.src)
  obj = _emu_compiler.compile_to_obj("\n".join(combined_src_parts))
  sym_offsets = _elf_symbol_offsets(obj)
  cpu_prg = _ProgramClass(Device['CPU'], "emu2_batch", jit_loader(obj))
  base_addr = ctypes.cast(cpu_prg.fxn, ctypes.c_void_p).value
  return {pc: (prg.function_name, ctypes.CFUNCTYPE(None)(base_addr + sym_offsets.get(prg.function_name, 0)), prg.globals, cpu_prg)
          for pc, prg in inst_info}

# ═══════════════════════════════════════════════════════════════════════════════
# WAVE STATE
# ═══════════════════════════════════════════════════════════════════════════════

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
            scratch_size: int = 0) -> int:
  """Execute AMD assembly program. scratch_size is private_segment_fixed_size from kernel descriptor (per-lane)."""
  program_raw = decode_program(bytes((ctypes.c_char * lib_sz).from_address(lib).raw))
  program = {lib + offset: val for offset, val in program_raw.items()}  # Remap to actual addresses
  lds_size = ((rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT) * 512
  total_threads = lx * ly * lz

  # Use Buffer objects with external_ptr=0 for vmem
  vmem_buf = Buffer('CPU', 1 << 40, dtypes.uint32, options=BufferSpec(external_ptr=0)).ensure_allocated()
  lds_buf = Buffer('CPU', max(lds_size // 4, 1), dtypes.uint32).ensure_allocated()
  scratch_buf = Buffer('CPU', scratch_size * WAVE_SIZE, dtypes.uint8).ensure_allocated() if scratch_size else None

  # Set DAZ+FTZ during emulator execution, restore afterward to avoid breaking hypothesis tests
  with _MXCSRContext():
    for gidx in range(gx):
      for gidy in range(gy):
        for gidz in range(gz):
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
              assert fxn is not None, f"[emu2] No fxn for {name} at PC={pc}"
              assert 4 not in globals_list or scratch_buf, f"SCRATCH instruction {name} but scratch_size=0"
              if DEBUG >= 5:
                inst = decode_inst(bytes((ctypes.c_char * 12).from_address(pc).raw))
                print(f"[emu2] exec PC={pc:X}: {inst!r}")
              fxn(*[c_bufs[g] for g in globals_list])
            else: raise RuntimeError("exceeded 1M instructions, likely infinite loop")
  return 0
