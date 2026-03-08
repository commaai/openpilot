# RDNA3/RDNA4/CDNA disassembler
from __future__ import annotations
import re, struct
from typing import Callable
from extra.assembly.amd.dsl import Inst, Reg

# Special register mappings for disassembly
SPECIAL_GPRS = {106: 'vcc_lo', 107: 'vcc_hi', 124: 'null', 125: 'm0', 126: 'exec_lo', 127: 'exec_hi',
                128: '0', 240: '0.5', 241: '-0.5', 242: '1.0', 243: '-1.0', 244: '2.0', 245: '-2.0', 246: '4.0', 247: '-4.0', 248: '0x3e22f983', 253: 'scc'}
SPECIAL_GPRS_CDNA = {106: 'vcc_lo', 107: 'vcc_hi', 124: 'm0', 126: 'exec_lo', 127: 'exec_hi',
                     128: '0', 240: '0.5', 241: '-0.5', 242: '1.0', 243: '-1.0', 244: '2.0', 245: '-2.0', 246: '4.0', 247: '-4.0', 248: '0x3e22f983', 253: 'scc',
                     102: 'flat_scratch_lo', 103: 'flat_scratch_hi', 104: 'xnack_mask_lo', 105: 'xnack_mask_hi',
                     251: 'src_vccz', 252: 'src_execz'}
SPECIAL_PAIRS = {106: 'vcc', 126: 'exec'}
SPECIAL_PAIRS_CDNA = {106: 'vcc', 126: 'exec', 102: 'flat_scratch', 104: 'xnack_mask'}

def decode_src(v, cdna: bool = False) -> str:
  """Decode a source operand encoding to its string representation."""
  v = _unwrap(v)
  gprs = SPECIAL_GPRS_CDNA if cdna else SPECIAL_GPRS
  if v in gprs: return gprs[v]
  if v < 106: return f's{v}'
  if 108 <= v < 124: return f'ttmp{v - 108}'
  if 129 <= v <= 192: return str(v - 128)  # positive integers 1-64
  if 193 <= v <= 208: return str(-(v - 192))  # negative integers -1 to -16
  if v >= 256: return f'v{v - 256}'
  return f's{v}'

def _unwrap(v) -> int:
  """Unwrap Reg to int offset, or return int as-is."""
  return v.offset if isinstance(v, Reg) else v

def _vi(v) -> int:
  """Get VGPR index from Reg or int (for v[N] fields that encode as 256+N)."""
  off = _unwrap(v)
  return off - 256 if off >= 256 else off

# ═══════════════════════════════════════════════════════════════════════════════
# LITERAL FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

_FLOAT_DEC = {240: 0.5, 241: -0.5, 242: 1.0, 243: -1.0, 244: 2.0, 245: -2.0, 246: 4.0, 247: -4.0}

def _lit(inst, v, neg=0, cdna=None) -> str:
  """Format literal/inline constant value."""
  if cdna is None: cdna = _is_cdna(inst)
  v = _unwrap(v)
  if v == 255:
    lit = inst._literal
    if lit is None: return "0"
    s = f"0x{lit:x}"
  elif v in _FLOAT_DEC: s = str(_FLOAT_DEC[v])
  elif 128 <= v <= 192: s = str(v - 128)
  elif 193 <= v <= 208: s = str(-(v - 192))
  elif v < 128: s = decode_src(v, cdna)
  elif v >= 256: s = f"v{v - 256}"
  else: s = decode_src(v, cdna)
  return f"-{s}" if neg else s

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION METADATA - fallback functions when inst.num_srcs()/inst.operands unavailable
# ═══════════════════════════════════════════════════════════════════════════════

def _num_srcs(inst) -> int:
  """Fallback: get number of source operands from instruction name."""
  name = getattr(inst, 'op_name', '') or ''
  n = name.upper()
  # FMAC/MAC ops are 2-source (dst is implicit accumulator), but FMA/MAD ops are 3-source
  if 'FMAC' in n or 'V_MAC_' in n: return 2
  if any(x in n for x in ('FMA', 'MAD', 'CNDMASK', 'BFE', 'BFI', 'LERP', 'MED3', 'SAD', 'DIV_FMAS', 'DIV_FIXUP', 'DIV_SCALE', 'CUBE')): return 3
  # PERMLANE_VAR ops are 2-source, but PERMLANE (non-VAR) are 3-source
  if 'PERMLANE' in n and '_VAR' not in n: return 3
  if any(x in n for x in ('_ADD3', '_LSHL_ADD', '_ADD_LSHL', '_LSHL_OR', '_AND_OR', 'OR3_B32', 'AND_OR_B32', 'ALIGNBIT', 'ALIGNBYTE', 'V_PERM_', 'XOR3', 'XAD', 'MULLIT', 'MINMAX', 'MAXMIN', 'MINIMUMMAXIMUM', 'MAXIMUMMINIMUM', 'MINIMUM3', 'MAXIMUM3', 'MIN3', 'MAX3', 'DOT2', 'CVT_PK_U8_F32', 'DOT4', 'DOT8', 'WMMA', 'SWMMAC')): return 3
  return 2

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

from extra.assembly.amd.autogen.rdna3.ins import (VOP1, VOP1_SDST, VOP1_SDST_LIT, VOP1_LIT, VOP2, VOP2_LIT, VOP3, VOP3_SDST, VOP3_SDST_LIT,
  VOP3_LIT, VOP3SD, VOP3SD_LIT, VOP3P, VOP3P_LIT, VOPC, VOPC_LIT, VOPD, VOPD_LIT, VINTERP, SOP1, SOP1_LIT, SOP2, SOP2_LIT, SOPC, SOPC_LIT,
  SOPK, SOPK_LIT, SOPP, SMEM, DS, FLAT, GLOBAL, SCRATCH, VOP2Op, VOPDOp, SOPPOp, HWREG, MSG)
from extra.assembly.amd.autogen.rdna4.ins import (VOP1 as R4_VOP1, VOP1_SDST as R4_VOP1_SDST, VOP1_SDST_LIT as R4_VOP1_SDST_LIT, VOP1_LIT as R4_VOP1_LIT,
  VOP2 as R4_VOP2, VOP2_LIT as R4_VOP2_LIT, VOP3 as R4_VOP3, VOP3_SDST as R4_VOP3_SDST, VOP3_SDST_LIT as R4_VOP3_SDST_LIT, VOP3_LIT as R4_VOP3_LIT,
  VOP3SD as R4_VOP3SD, VOP3SD_LIT as R4_VOP3SD_LIT, VOP3P as R4_VOP3P, VOP3P_LIT as R4_VOP3P_LIT, VOPC as R4_VOPC, VOPC_LIT as R4_VOPC_LIT,
  VOPD as R4_VOPD, VOPD_LIT as R4_VOPD_LIT, VINTERP as R4_VINTERP, SOP1 as R4_SOP1, SOP1_LIT as R4_SOP1_LIT, SOP2 as R4_SOP2, SOP2_LIT as R4_SOP2_LIT,
  SOPC as R4_SOPC, SOPC_LIT as R4_SOPC_LIT, SOPK as R4_SOPK, SOPK_LIT as R4_SOPK_LIT, SOPP as R4_SOPP, SMEM as R4_SMEM, DS as R4_DS,
  VOPDOp as R4_VOPDOp, HWREG as HWREG_RDNA4, VFLAT as R4_FLAT, VGLOBAL as R4_GLOBAL, VSCRATCH as R4_SCRATCH)
from extra.assembly.amd.autogen.cdna.ins import FLAT as C_FLAT, HWREG as HWREG_CDNA

def _is_cdna(inst: Inst) -> bool: return 'cdna' in inst.__class__.__module__
def _is_r4(inst: Inst) -> bool: return 'rdna4' in inst.__class__.__module__

# CDNA opcode name aliases for disasm (new name -> old name expected by tests)
_CDNA_DISASM_ALIASES = {'v_fmac_f64': 'v_mul_legacy_f32', 'v_dot2c_f32_bf16': 'v_mac_f32', 'v_fmamk_f32': 'v_madmk_f32', 'v_fmaak_f32': 'v_madak_f32'}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _reg(p: str, b: int, n: int = 1) -> str: return f"{p}{_unwrap(b)}" if n == 1 else f"{p}[{_unwrap(b)}:{_unwrap(b)+n-1}]"
def _sreg(b: int, n: int = 1) -> str: return _reg("s", _unwrap(b), n)
def _vreg(b: int, n: int = 1) -> str: b = _unwrap(b); return _reg("v", b - 256 if b >= 256 else b, n)
def _areg(b: int, n: int = 1) -> str: b = _unwrap(b); return _reg("a", b - 256 if b >= 256 else b, n)  # accumulator registers for GFX90a
def _ttmp(b, n: int = 1) -> str | None: b = _unwrap(b); return _reg("ttmp", b - 108, n) if 108 <= b <= 123 else None

def _fmt_sdst(v, n: int = 1, cdna: bool = False) -> str:
  v = _unwrap(v)
  if t := _ttmp(v, n): return t
  pairs = SPECIAL_PAIRS_CDNA if cdna else SPECIAL_PAIRS
  gprs = SPECIAL_GPRS_CDNA if cdna else SPECIAL_GPRS
  if n > 1: return pairs.get(v) or gprs.get(v) or _sreg(v, n)  # also check gprs for null/m0
  return gprs.get(v, f"s{v}")

def _fmt_src(v, n: int = 1, cdna: bool = False) -> str:
  v = _unwrap(v)
  if v == 253: return "src_scc"  # SCC as source operand
  if n == 1: return decode_src(v, cdna)
  if v >= 256: return _vreg(v, n)
  if v <= 101: return _sreg(v, n)  # s0-s101 can be pairs, but 102+ are special on CDNA
  pairs = SPECIAL_PAIRS_CDNA if cdna else SPECIAL_PAIRS
  if n == 2 and v in pairs: return pairs[v]
  if v <= 105: return _sreg(v, n)  # s102-s105 regular pairs for RDNA
  if t := _ttmp(v, n): return t
  return decode_src(v, cdna)

def _fmt_v16(v, base: int = 256, hi_thresh: int = 384) -> str:
  v = _unwrap(v)
  return f"v{(v - base) & 0x7f}.{'h' if v >= hi_thresh else 'l'}"

def _has(op: str, *subs) -> bool: return any(s in op for s in subs)
def _omod(v: int) -> str: return {1: " mul:2", 2: " mul:4", 3: " div:2"}.get(v, "")
def _src16(inst, v: int) -> str: v = _unwrap(v); return _fmt_v16(v) if v >= 256 else _lit(inst, v)  # format 16-bit src: vgpr.h/l or literal
def _mods(*pairs) -> str: return " ".join(m for c, m in pairs if c)
def _fmt_bits(label: str, val: int, count: int) -> str: return f"{label}:[{','.join(str((val >> i) & 1) for i in range(count))}]"

def _vop3_src(inst, v: int, neg: int, abs_: int, hi: int, n: int, f16: bool) -> str:
  """Format VOP3 source operand with modifiers."""
  v = _unwrap(v)
  if v == 255: s = _lit(inst, v)  # literal constant takes priority
  elif n > 1: s = _fmt_src(v, n)
  elif f16 and v >= 256: s = f"v{v - 256}.h" if hi else f"v{v - 256}.l"
  elif v == 253: s = "src_scc"  # VOP3 sources use src_scc not scc
  else: s = _lit(inst, v)
  if abs_: s = f"|{s}|"
  return f"-{s}" if neg else s

def _opsel_str(opsel: int, n: int, need: bool, is16_d: bool) -> str:
  """Format op_sel modifier string."""
  if not need: return ""
  dst_hi = (opsel >> 3) & 1
  if n == 1: return f" op_sel:[{opsel & 1},{dst_hi}]"
  # Use 4-element format if bit 2 is set (src2 selection used) or if 3+ sources
  if n == 2 and not ((opsel >> 2) & 1): return f" op_sel:[{opsel & 1},{(opsel >> 1) & 1},{dst_hi}]"
  return f" op_sel:[{opsel & 1},{(opsel >> 1) & 1},{(opsel >> 2) & 1},{dst_hi}]"

# ═══════════════════════════════════════════════════════════════════════════════
# DISASSEMBLER
# ═══════════════════════════════════════════════════════════════════════════════

def _disasm_vop1(inst: VOP1) -> str:
  name, cdna = inst.op_name.lower() or f'vop1_op_{inst.op}', _is_cdna(inst)
  name = name.replace('_e32', '')  # Strip _e32 suffix
  if any(x in name for x in ('v_nop', 'v_pipeflush', 'v_clrexcp')): return name  # no operands
  if 'readfirstlane' in name:
    src = inst.src0.fmt() if inst.src0.offset >= 256 else decode_src(inst.src0.offset, cdna)
    vdst_off = inst.vdst.offset - 256 if inst.vdst.offset >= 256 else inst.vdst.offset
    return f"{name} {_fmt_sdst(vdst_off, 1, cdna)}, {src}"
  bits = inst.canonical_op_bits
  is16_dst, is16_src = not cdna and bits['d'] == 16, not cdna and bits['s0'] == 16
  # Format dst
  if is16_dst: dst = _fmt_v16(inst.vdst)
  else: dst = inst.vdst.fmt()
  # Format src
  if inst.src0.offset == 255: src = _lit(inst, inst.src0)
  elif is16_src and inst.src0.offset >= 256: src = _fmt_v16(inst.src0)
  elif inst.src0.sz > 1: src = _fmt_src(inst.src0, inst.src0.sz, cdna)
  else: src = _lit(inst, inst.src0)
  return f"{name} {dst}, {src}"

_VOP2_CARRY_OUT = {'v_add_co_u32', 'v_sub_co_u32', 'v_subrev_co_u32'}  # carry out only
_VOP2_CARRY_INOUT = {'v_addc_co_u32', 'v_subb_co_u32', 'v_subbrev_co_u32'}  # carry in and out (CDNA)
_VOP2_CARRY_INOUT_RDNA = {'v_add_co_ci_u32', 'v_sub_co_ci_u32', 'v_subrev_co_ci_u32'}  # carry in and out (RDNA)
def _disasm_vop2(inst: VOP2) -> str:
  name, cdna = inst.op_name.lower(), _is_cdna(inst)
  if cdna: name = _CDNA_DISASM_ALIASES.get(name, name)  # apply CDNA aliases
  suf = "" if cdna or name.endswith('_e32') or (not cdna and inst.op == VOP2Op.V_DOT2ACC_F32_F16_E32) else "_e32"
  lit = inst._literal
  is16 = not cdna and inst.canonical_op_bits['d'] == 16
  # fmaak/madak: dst = src0 * vsrc1 + K, fmamk/madmk: dst = src0 * K + vsrc1
  if 'fmaak' in name or 'madak' in name or (not cdna and inst.op in (VOP2Op.V_FMAAK_F32_E32, VOP2Op.V_FMAAK_F16_E32)):
    if lit is None: return f"op_{inst.op.value if hasattr(inst.op, 'value') else inst.op}"
    if is16: return f"{name}{suf} {_fmt_v16(inst.vdst)}, {_src16(inst, inst.src0)}, {_fmt_v16(inst.vsrc1)}, 0x{lit:x}"
    return f"{name}{suf} {inst.vdst.fmt()}, {_lit(inst, inst.src0)}, {inst.vsrc1.fmt()}, 0x{lit:x}"
  if 'fmamk' in name or 'madmk' in name or (not cdna and inst.op in (VOP2Op.V_FMAMK_F32_E32, VOP2Op.V_FMAMK_F16_E32)):
    if lit is None: return f"op_{inst.op.value if hasattr(inst.op, 'value') else inst.op}"
    if is16: return f"{name}{suf} {_fmt_v16(inst.vdst)}, {_src16(inst, inst.src0)}, 0x{lit:x}, {_fmt_v16(inst.vsrc1)}"
    return f"{name}{suf} {inst.vdst.fmt()}, {_lit(inst, inst.src0)}, 0x{lit:x}, {inst.vsrc1.fmt()}"
  if is16: return f"{name}{suf} {_fmt_v16(inst.vdst)}, {_src16(inst, inst.src0)}, {_fmt_v16(inst.vsrc1)}"
  vcc = "vcc" if cdna else "vcc_lo"
  basename = name.replace('_e32', '')
  if cdna and basename in _VOP2_CARRY_OUT: return f"{name}{suf} {inst.vdst.fmt()}, {vcc}, {_lit(inst, inst.src0)}, {inst.vsrc1.fmt()}"
  if cdna and basename in _VOP2_CARRY_INOUT: return f"{name}{suf} {inst.vdst.fmt()}, {vcc}, {_lit(inst, inst.src0)}, {inst.vsrc1.fmt()}, {vcc}"
  if not cdna and basename in _VOP2_CARRY_INOUT_RDNA: return f"{name}{suf} {inst.vdst.fmt()}, {vcc}, {_lit(inst, inst.src0)}, {inst.vsrc1.fmt()}, {vcc}"
  sn0 = inst.canonical_op_regs.get('s0', 1)
  if inst.vdst.sz > 1 or sn0 > 1 or inst.vsrc1.sz > 1:
    src0 = _lit(inst, inst.src0) if inst.src0.offset == 255 else _fmt_src(inst.src0, sn0, cdna)
    return f"{name.replace('_e32', '')} {inst.vdst.fmt()}, {src0}, {inst.vsrc1.fmt()}"
  return f"{name}{suf} {inst.vdst.fmt()}, {_lit(inst, inst.src0)}, {inst.vsrc1.fmt()}" + (f", {vcc}" if name == 'v_cndmask_b32' else "")

def _disasm_vopc(inst: VOPC) -> str:
  name, cdna = inst.op_name.lower(), _is_cdna(inst)
  bits = inst.canonical_op_bits
  is16 = bits['s0'] == 16
  if cdna:
    s0 = _lit(inst, inst.src0) if inst.src0.offset == 255 else _fmt_src(inst.src0, inst.src0.sz, cdna)
    return f"{name} vcc, {s0}, {inst.vsrc1.fmt()}"  # CDNA VOPC always outputs vcc
  # RDNA: v_cmpx_* writes to exec (no vcc), v_cmp_* writes to vcc_lo
  has_vcc = 'cmpx' not in name
  s0 = _lit(inst, inst.src0) if inst.src0.offset == 255 else inst.src0.fmt() if inst.src0.sz > 1 else _src16(inst, inst.src0.offset) if is16 else _lit(inst, inst.src0)
  s1 = inst.vsrc1.fmt() if inst.vsrc1.sz > 1 else _fmt_v16(inst.vsrc1) if is16 else inst.vsrc1.fmt()
  suf = "" if name.endswith('_e32') else "_e32"
  return f"{name}{suf} vcc_lo, {s0}, {s1}" if has_vcc else f"{name}{suf} {s0}, {s1}"

NO_ARG_SOPP = {SOPPOp.S_BARRIER, SOPPOp.S_WAKEUP, SOPPOp.S_ICACHE_INV,
               SOPPOp.S_WAIT_IDLE, SOPPOp.S_ENDPGM_SAVED, SOPPOp.S_CODE_END, SOPPOp.S_ENDPGM_ORDERED_PS_DONE, SOPPOp.S_TTRACEDATA}

def _disasm_sopp(inst: SOPP) -> str:
  name, cdna = inst.op_name.lower(), _is_cdna(inst)
  is_rdna4 = _is_r4(inst)
  # Ops that have no argument when simm16 == 0
  no_arg_zero = {'s_barrier', 's_wakeup', 's_icache_inv', 's_ttracedata', 's_wait_idle', 's_endpgm_saved',
                 's_endpgm_ordered_ps_done', 's_code_end'}
  if name in no_arg_zero: return name if inst.simm16 == 0 else f"{name} {inst.simm16}"
  if name == 's_endpgm': return name if inst.simm16 == 0 else f"{name} {inst.simm16}"
  if cdna:
    if name == 's_waitcnt':
      # GFX9 format: vmcnt[3:0]=bits[3:0], vmcnt[5:4]=bits[15:14], expcnt=bits[6:4], lgkmcnt=bits[11:8] (4 bits, max 15)
      vm_lo, exp, lgkm, vm_hi = inst.simm16 & 0xf, (inst.simm16 >> 4) & 0x7, (inst.simm16 >> 8) & 0xf, (inst.simm16 >> 14) & 0x3
      vm = vm_lo | (vm_hi << 4)
      p = [f"vmcnt({vm})" if vm != 0x3f else "", f"expcnt({exp})" if exp != 7 else "", f"lgkmcnt({lgkm})" if lgkm != 0xf else ""]
      return f"s_waitcnt {' '.join(x for x in p if x) or '0'}"
    if name.startswith(('s_cbranch', 's_branch')): return f"{name} {inst.simm16}"
    if name == 's_set_gpr_idx_mode':
      flags = [n for i, n in enumerate(['SRC0', 'SRC1', 'SRC2', 'DST']) if inst.simm16 & (1 << i)]
      return f"{name} gpr_idx({','.join(flags)})"
    return f"{name} 0x{inst.simm16:x}" if inst.simm16 else name
  # RDNA (use name-based checks instead of enum-based for cross-arch compatibility)
  if name == 's_waitcnt':
    if is_rdna4:
      return f"{name} {inst.simm16}" if inst.simm16 else f"{name} 0"
    vm, exp, lgkm = (inst.simm16 >> 10) & 0x3f, inst.simm16 & 0xf, (inst.simm16 >> 4) & 0x3f
    p = [f"vmcnt({vm})" if vm != 0x3f else "", f"expcnt({exp})" if exp != 7 else "", f"lgkmcnt({lgkm})" if lgkm != 0x3f else ""]
    return f"s_waitcnt {' '.join(x for x in p if x) or '0'}"
  if name == 's_delay_alu':
    deps = ['VALU_DEP_1','VALU_DEP_2','VALU_DEP_3','VALU_DEP_4','TRANS32_DEP_1','TRANS32_DEP_2','TRANS32_DEP_3','FMA_ACCUM_CYCLE_1','SALU_CYCLE_1','SALU_CYCLE_2','SALU_CYCLE_3']
    skips = ['SAME','NEXT','SKIP_1','SKIP_2','SKIP_3','SKIP_4']
    id0, skip, id1 = inst.simm16 & 0xf, (inst.simm16 >> 4) & 0x7, (inst.simm16 >> 7) & 0xf
    dep = lambda v: deps[v-1] if 0 < v <= len(deps) else str(v)
    p = [f"instid0({dep(id0)})" if id0 else "", f"instskip({skips[skip]})" if skip else "", f"instid1({dep(id1)})" if id1 else ""]
    return f"s_delay_alu {' | '.join(x for x in p if x) or '0'}"
  if name.startswith(('s_cbranch', 's_branch')): return f"{name} {inst.simm16}"
  return f"{name} 0x{inst.simm16:x}"

def _disasm_smem(inst: SMEM) -> str:
  name, cdna = inst.op_name.lower(), _is_cdna(inst)
  if name in ('s_gl1_inv', 's_dcache_inv', 's_dcache_inv_vol', 's_dcache_wb', 's_dcache_wb_vol', 's_icache_inv'): return name
  soe, imm = getattr(inst, 'soe', 0) or getattr(inst, 'soffset_en', 0), getattr(inst, 'imm', 1)
  is_rdna4 = _is_r4(inst)
  offset = inst.ioffset if is_rdna4 else getattr(inst, 'offset', 0)
  if cdna:
    if soe and imm: off_s = f"{decode_src(inst.soffset, cdna)} offset:0x{offset:x}"
    elif imm: off_s = f"0x{offset:x}"
    elif offset < 256: off_s = decode_src(offset, cdna)
    else: off_s = decode_src(inst.soffset, cdna)
  elif offset and inst.soffset != 124: off_s = f"{decode_src(inst.soffset, cdna)} offset:0x{offset:x}"
  elif offset: off_s = f"0x{offset:x}"
  else: off_s = decode_src(inst.soffset, cdna)
  is_buffer = 'buffer' in name or 's_atc_probe_buffer' == name
  sbase_idx, sbase_count = _unwrap(inst.sbase), 4 if is_buffer else 2
  sbase_str = _fmt_src(sbase_idx, sbase_count, cdna) if sbase_count == 2 else _sreg(sbase_idx, sbase_count) if sbase_idx <= 105 else _reg("ttmp", sbase_idx - 108, sbase_count)
  if name in ('s_atc_probe', 's_atc_probe_buffer'): return f"{name} {_unwrap(inst.sdata)}, {sbase_str}, {off_s}"
  if 'prefetch' in name:
    off = getattr(inst, 'ioffset', getattr(inst, 'offset', 0))
    if off >= 0x800000: off = off - 0x1000000
    off_s = f"0x{off:x}" if off > 255 else str(off)
    soff_s = decode_src(inst.soffset, cdna) if inst.soffset != 124 else ("m0" if cdna else "null")
    if 'pc_rel' in name: return f"{name} {off_s}, {soff_s}, {_unwrap(inst.sdata)}"
    return f"{name} {sbase_str}, {off_s}, {soff_s}, {_unwrap(inst.sdata)}"
  # Use get_field_bits for register count
  dst_n = inst.canonical_op_regs.get('d', 1)
  th, scope = getattr(inst, 'th', 0), getattr(inst, 'scope', 0)
  if is_rdna4:  # RDNA4 uses th/scope instead of glc/dlc
    th_names = ['TH_LOAD_RT', 'TH_LOAD_NT', 'TH_LOAD_HT', 'TH_LOAD_LU']
    scope_names = ['SCOPE_CU', 'SCOPE_SE', 'SCOPE_DEV', 'SCOPE_SYS']
    mods = (f" th:{th_names[th]}" if th else "") + (f" scope:{scope_names[scope]}" if scope else "")
    return f"{name} {_fmt_sdst(inst.sdata, dst_n, cdna)}, {sbase_str}, {off_s}{mods}"
  if th or scope:
    th_names = ['TH_LOAD_RT', 'TH_LOAD_NT', 'TH_LOAD_HT', 'TH_LOAD_LU']
    scope_names = ['SCOPE_CU', 'SCOPE_SE', 'SCOPE_DEV', 'SCOPE_SYS']
    mods = (f" th:{th_names[th]}" if th else "") + (f" scope:{scope_names[scope]}" if scope else "")
    return f"{name} {_fmt_sdst(inst.sdata, dst_n, cdna)}, {sbase_str}, {off_s}{mods}"
  if 'discard' in name: return f"{name} {sbase_str}, {off_s}" + _mods((inst.glc, " glc"), (getattr(inst, 'dlc', 0), " dlc"))
  if name in ('s_memrealtime', 's_memtime'): return f"{name} {_fmt_sdst(inst.sdata, dst_n, cdna)}"
  return f"{name} {_fmt_sdst(inst.sdata, dst_n, cdna)}, {sbase_str}, {off_s}" + _mods((inst.glc, " glc"), (getattr(inst, 'dlc', 0), " dlc"))

def _disasm_flat(inst: FLAT) -> str:
  name, cdna, r4 = inst.op_name.lower(), _is_cdna(inst), _is_r4(inst)
  acc = getattr(inst, 'acc', 0)
  reg_fn = _areg if acc else _vreg
  if r4: seg = 'flat' if (cls_name:=inst.__class__.__name__) == 'VFLAT' else ('global' if cls_name == 'VGLOBAL' else 'scratch')
  else: seg = ['flat', 'scratch', 'global'][inst.seg] if inst.seg < 3 else 'flat'
  instr = f"{seg}_{name.split('_', 1)[1] if '_' in name else name}"
  # Global/scratch uses 13-bit signed offset
  offset = inst.ioffset if r4 else inst.offset
  if seg != 'flat':
    if cdna:
      # CDNA: bit 12 is sign bit but not in offset field
      raw = int.from_bytes(inst.to_bytes(), 'little')
      off_val = offset | ((raw >> 12) & 1) << 12  # get bit 12
    else:
      off_val = offset
    off_val = off_val if off_val < 4096 else off_val - 8192  # sign extend 13-bit
  else:
    off_val = offset
  # Use get_field_bits: data for stores/atomics, d for loads
  regs = inst.canonical_op_regs
  w = regs.get('data', regs.get('d', 1)) if 'store' in name or 'atomic' in name else regs.get('d', 1)
  off_s = f" offset:{off_val}" if off_val else ""
  if cdna: mods = f"{off_s}{' sc0' if inst.sc0 else ''}{' nt' if inst.nt else ''}{' sc1' if getattr(inst, 'sc1', 0) else ''}"
  elif r4: mods = f"{off_s}{' scope' if inst.scope else ''}{' th' if inst.th else ''}"
  else: mods = f"{off_s}{' glc' if inst.glc else ''}{' slc' if inst.slc else ''}{' dlc' if inst.dlc else ''}"
  if seg == 'flat': saddr_s = ""
  elif _unwrap(inst.saddr) in (0x7F, 124): saddr_s = ", off"
  elif seg == 'scratch': saddr_s = f", {decode_src(inst.saddr, cdna)}"
  elif _unwrap(inst.saddr) in (SPECIAL_PAIRS_CDNA if cdna else SPECIAL_PAIRS): saddr_s = f", {(SPECIAL_PAIRS_CDNA if cdna else SPECIAL_PAIRS)[_unwrap(inst.saddr)]}"
  elif t := _ttmp(inst.saddr, 2): saddr_s = f", {t}"
  else: saddr_s = f", {_sreg(inst.saddr, 2) if _unwrap(inst.saddr) < 106 else decode_src(_unwrap(inst.saddr), cdna)}"
  if 'addtid' in name: return f"{instr} {reg_fn(inst.data if 'store' in name else inst.vdst)}{saddr_s}{mods}"
  # RDNA4: vaddr instead of addr, vsrc instead of data
  addr = inst.vaddr if r4 else inst.addr
  data = inst.vsrc if r4 else inst.data
  # load_lds_* instructions: vaddr, saddr (no vdst, data goes to LDS)
  if 'load_lds' in name:
    addr_w = 1 if seg == 'scratch' or (_unwrap(inst.saddr) not in (0x7F, 124)) else 2
    addr_s = "off" if not inst.sve and seg == 'scratch' else _vreg(addr, addr_w)
    return f"{instr} {addr_s}{saddr_s}{mods}"
  if seg == 'flat': addr_w = 2  # flat always uses 64-bit vaddr
  elif cdna: addr_w = 1 if seg == 'scratch' or (_unwrap(inst.saddr) not in (0x7F, 124)) else 2
  else: addr_w = 1 if seg == 'scratch' or (_unwrap(inst.saddr) not in (0x7F, 124)) else 2
  addr_s = "off" if not inst.sve and seg == 'scratch' else _vreg(addr, addr_w)
  data_s, vdst_s = reg_fn(data, w), reg_fn(inst.vdst, w // 2 if 'cmpswap' in name else w)
  if 'atomic' in name:
    glc_or_sc0 = inst.sc0 if cdna else inst.glc
    return f"{instr} {vdst_s}, {addr_s}, {data_s}{saddr_s if seg != 'flat' else ''}{mods}" if glc_or_sc0 else f"{instr} {addr_s}, {data_s}{saddr_s if seg != 'flat' else ''}{mods}"
  if 'store' in name: return f"{instr} {addr_s}, {data_s}{saddr_s}{mods}"
  return f"{instr} {reg_fn(inst.vdst, w)}, {addr_s}{saddr_s}{mods}"

def _disasm_ds(inst: DS) -> str:
  op, name = inst.op, inst.op_name.lower()
  acc = getattr(inst, 'acc', 0)
  reg_fn = _areg if acc else _vreg
  gds = " gds" if getattr(inst, 'gds', 0) else ""
  off = f" offset:{inst.offset0 | (inst.offset1 << 8)}" if inst.offset0 or inst.offset1 else ""
  off2 = (" offset0:" + str(inst.offset0) if inst.offset0 else "") + (" offset1:" + str(inst.offset1) if inst.offset1 else "")
  # Use get_field_bits: data for stores/writes/atomics, d for loads
  regs = inst.canonical_op_regs
  w = regs.get('data', regs.get('d', 1)) if 'store' in name or 'write' in name or ('load' not in name and 'read' not in name) else regs.get('d', 1)
  d0, d1, dst, addr = reg_fn(inst.data0, w), reg_fn(inst.data1, w), reg_fn(inst.vdst, w), _vreg(inst.addr)

  if name == 'ds_nop': return name
  if name == 'ds_bvh_stack_rtn_b32': return f"{name} {_vreg(inst.vdst)}, {addr}, {_vreg(inst.data0)}, {_vreg(inst.data1, 4)}{off}{gds}"
  if 'bvh_stack_push' in name:
    d1_regs = 8 if 'push8' in name else 4
    vdst_regs = 2 if 'pop2' in name else 1
    vdst_s = _vreg(inst.vdst, vdst_regs) if vdst_regs > 1 else _vreg(inst.vdst)
    return f"{name} {vdst_s}, {addr}, {_vreg(inst.data0)}, {_vreg(inst.data1, d1_regs)}{off}{gds}"
  if 'gws_sema' in name and 'sema_br' not in name: return f"{name}{off}{gds}"
  if 'gws_' in name: return f"{name} {addr}{off}{gds}"
  if name in ('ds_consume', 'ds_append'): return f"{name} {reg_fn(inst.vdst)}{off}{gds}"
  if 'gs_reg' in name: return f"{name} {reg_fn(inst.vdst, 2)}, {reg_fn(inst.data0)}{off}{gds}"
  if '2addr' in name:
    if 'load' in name: return f"{name} {reg_fn(inst.vdst, regs.get('d', 1))}, {addr}{off2}{gds}"
    if 'store' in name and 'xchg' not in name: return f"{name} {addr}, {d0}, {d1}{off2}{gds}"
    return f"{name} {reg_fn(inst.vdst, regs.get('d', 1))}, {addr}, {d0}, {d1}{off2}{gds}"
  if 'write2' in name: return f"{name} {addr}, {d0}, {d1}{off2}{gds}"
  if 'read2' in name: return f"{name} {reg_fn(inst.vdst, regs.get('d', 1))}, {addr}{off2}{gds}"
  if 'xchg2' in name: return f"{name} {reg_fn(inst.vdst, regs.get('d', 1))}, {addr}, {d0}, {d1}{off2}{gds}"
  if 'load' in name or ('read' in name and 'read2' not in name): return f"{name} {reg_fn(inst.vdst)}{off}{gds}" if 'addtid' in name else f"{name} {dst}, {addr}{off}{gds}"
  if ('store' in name or 'write' in name) and not _has(name, 'cmp', 'xchg', 'write2'):
    return f"{name} {reg_fn(inst.data0)}{off}{gds}" if 'addtid' in name else f"{name} {addr}, {d0}{off}{gds}"
  if 'swizzle' in name or name == 'ds_ordered_count': return f"{name} {reg_fn(inst.vdst)}, {addr}{off}{gds}"
  if 'permute' in name: return f"{name} {reg_fn(inst.vdst)}, {addr}, {reg_fn(inst.data0)}{off}{gds}"
  if 'condxchg' in name: return f"{name} {reg_fn(inst.vdst, 2)}, {addr}, {reg_fn(inst.data0, 2)}{off}{gds}"
  if _has(name, 'cmpst', 'mskor', 'wrap'):
    return f"{name} {dst}, {addr}, {d0}, {d1}{off}{gds}" if '_rtn' in name else f"{name} {addr}, {d0}, {d1}{off}{gds}"
  return f"{name} {dst}, {addr}, {d0}{off}{gds}" if '_rtn' in name else f"{name} {addr}, {d0}{off}{gds}"

def _disasm_vop3(inst: VOP3) -> str:
  op, name = inst.op, inst.op_name.lower()
  n_up = name.upper()
  bits = inst.canonical_op_bits

  # RDNA4 v_s_* scalar VOP3 instructions - vdst is SGPR (VGPRField adds 256)
  if name.startswith('v_s_'):
    src = _lit(inst, inst.src0) if _unwrap(inst.src0) == 255 else ("src_scc" if _unwrap(inst.src0) == 253 else _fmt_src(inst.src0, max(1, bits['s0'] // 32)))
    if inst.neg & 1: src = f"-{src}"
    if inst.abs & 1: src = f"|{src}|"
    clamp = getattr(inst, 'cm', None) or getattr(inst, 'clmp', 0)
    vdst_raw = _unwrap(inst.vdst)
    return f"{name} s{vdst_raw - 256 if vdst_raw >= 256 else vdst_raw}, {src}" + (" clamp" if clamp else "") + _omod(inst.omod)

  # Use get_field_bits for register sizes and 16-bit detection
  r0, r1, r2 = max(1, bits['s0'] // 32), max(1, bits['s1'] // 32), max(1, bits['s2'] // 32)
  dn = max(1, bits['d'] // 32)
  is16_d, is16_s, is16_s2 = bits['d'] == 16, bits['s0'] == 16, bits['s2'] == 16

  s0 = _vop3_src(inst, inst.src0, inst.neg&1, inst.abs&1, inst.opsel&1, r0, is16_s)
  s1 = _vop3_src(inst, inst.src1, inst.neg&2, inst.abs&2, inst.opsel&2, r1, is16_s)
  s2 = _vop3_src(inst, inst.src2, inst.neg&4, inst.abs&4, inst.opsel&4, r2, is16_s2)

  # Format destination
  if 'readlane' in name:
    vdst_off = inst.vdst.offset - 256 if inst.vdst.offset >= 256 else inst.vdst.offset
    dst = _fmt_sdst(vdst_off, 1)
  elif is16_d: dst = f"{inst.vdst.fmt()}.h" if (inst.opsel & 8) else f"{inst.vdst.fmt()}.l"
  else: dst = inst.vdst.fmt()

  clamp = getattr(inst, 'cm', None) or getattr(inst, 'clmp', 0)
  cl, om = " clamp" if clamp else "", _omod(inst.omod)
  nonvgpr_opsel = (inst.src0.offset < 256 and (inst.opsel & 1)) or (inst.src1.offset < 256 and (inst.opsel & 2)) or (inst.src2.offset < 256 and (inst.opsel & 4))
  need_opsel = nonvgpr_opsel or (inst.opsel and not is16_s)

  op_val = inst.op.value if hasattr(inst.op, 'value') else inst.op
  e64 = "" if name.endswith('_e64') else "_e64"
  if op_val < 256:  # VOPC
    vdst_off = inst.vdst.offset - 256 if inst.vdst.offset >= 256 else inst.vdst.offset
    return f"{name}{e64} {s0}, {s1}{cl}" if name.startswith('v_cmpx') else f"{name}{e64} {_fmt_sdst(vdst_off, 1)}, {s0}, {s1}{cl}"
  if op_val < 384:  # VOP2
    n = inst.num_srcs() or 2
    os = _opsel_str(inst.opsel, n, need_opsel, is16_d)
    return f"{name}{e64} {dst}, {s0}, {s1}, {s2}{os}{cl}{om}" if n == 3 else f"{name}{e64} {dst}, {s0}, {s1}{os}{cl}{om}"
  if op_val < 512:  # VOP1
    if re.match(r'v_cvt_f32_(bf|fp)8', name) and inst.opsel:
      os = f" byte_sel:{((inst.opsel & 1) << 1) | ((inst.opsel >> 1) & 1)}"
    else:
      os = _opsel_str(inst.opsel, 1, need_opsel, is16_d)
    if 'v_nop' in name or 'v_pipeflush' in name: return f"{name}{e64}"
    return f"{name}{e64} {dst}, {s0}{os}{cl}{om}"
  # Native VOP3
  n = inst.num_srcs() or 2
  os = f" byte_sel:{inst.opsel >> 2}" if 'cvt_sr' in name and inst.opsel else _opsel_str(inst.opsel, n, need_opsel, is16_d)
  return f"{name} {dst}, {s0}, {s1}, {s2}{os}{cl}{om}" if n == 3 else f"{name} {dst}, {s0}, {s1}{os}{cl}{om}"

def _disasm_vop3sd(inst: VOP3SD) -> str:
  name = inst.op_name.lower()
  def src(reg, neg):
    s = _lit(inst, reg.offset) if reg.offset == 255 else ("src_scc" if reg.offset == 253 else (reg.fmt() if reg.sz > 1 else _lit(inst, reg.offset)))
    return f"neg({s})" if neg and reg.offset == 255 else (f"-{s}" if neg else s)
  s0, s1, s2 = src(inst.src0, inst.neg & 1), src(inst.src1, inst.neg & 2), src(inst.src2, inst.neg & 4)
  # VOP3SD: _co_ ops (add/sub) without _ci_ have only 2 sources, all others (mad, div_scale, _co_ci_) have 3 sources
  has_only_two_srcs = '_co_' in name and '_ci_' not in name and 'mad' not in name
  srcs = f"{s0}, {s1}" if has_only_two_srcs else f"{s0}, {s1}, {s2}"
  clamp = getattr(inst, 'cm', None) or getattr(inst, 'clmp', 0)
  return f"{name} {inst.vdst.fmt()}, {_fmt_sdst(inst.sdst, 1)}, {srcs}{' clamp' if clamp else ''}{_omod(inst.omod)}"

def _disasm_vopd(inst: VOPD) -> str:
  lit = inst._literal
  op_enum = R4_VOPDOp if _is_r4(inst) else VOPDOp
  nx, ny = op_enum(inst.opx).name.lower(), op_enum(inst.opy).name.lower()
  def half(n, vd, s0, vs1):
    vd, vs1 = _vi(vd), _vi(vs1)
    if 'mov' in n: return f"{n} v{vd}, {_lit(inst, s0)}"
    if 'fmamk' in n and lit: return f"{n} v{vd}, {_lit(inst, s0)}, 0x{lit:x}, v{vs1}"
    if 'fmaak' in n and lit: return f"{n} v{vd}, {_lit(inst, s0)}, v{vs1}, 0x{lit:x}"
    return f"{n} v{vd}, {_lit(inst, s0)}, v{vs1}"
  return f"{half(nx, inst.vdstx, inst.srcx0, inst.vsrcx1)} :: {half(ny, inst.vdsty, inst.srcy0, inst.vsrcy1)}"

def _disasm_vop3p(inst: VOP3P) -> str:
  name = inst.op_name.lower()
  is_wmma, is_swmmac, n, is_fma_mix = 'wmma' in name, 'swmmac' in name, inst.num_srcs() or 2, 'fma_mix' in name
  def get_src(reg):
    return _lit(inst, reg.offset) if reg.offset == 255 else reg.fmt()
  src0, src1, src2, dst = get_src(inst.src0), get_src(inst.src1), get_src(inst.src2), inst.vdst.fmt()
  opsel_hi = inst.opsel_hi | (inst.opsel_hi2 << 2)
  clamp = getattr(inst, 'cm', None) or getattr(inst, 'clmp', 0)
  if is_fma_mix:
    def m(s, neg, abs_): return f"-{f'|{s}|' if abs_ else s}" if neg else (f"|{s}|" if abs_ else s)
    src0, src1, src2 = m(src0, inst.neg & 1, inst.neg_hi & 1), m(src1, inst.neg & 2, inst.neg_hi & 2), m(src2, inst.neg & 4, inst.neg_hi & 4)
    mods = ([_fmt_bits("op_sel", inst.opsel, n)] if inst.opsel else []) + ([_fmt_bits("op_sel_hi", opsel_hi, n)] if opsel_hi else []) + (["clamp"] if clamp else [])
  elif is_swmmac:
    mods = ([f"index_key:{inst.opsel}"] if inst.opsel else []) + ([_fmt_bits("neg_lo", inst.neg, n)] if inst.neg else []) + \
           ([_fmt_bits("neg_hi", inst.neg_hi, n)] if inst.neg_hi else []) + (["clamp"] if clamp else [])
  else:
    opsel_hi_default = 7 if n == 3 else 3
    mods = ([_fmt_bits("op_sel", inst.opsel, n)] if inst.opsel else []) + ([_fmt_bits("op_sel_hi", opsel_hi, n)] if opsel_hi != opsel_hi_default else []) + \
           ([_fmt_bits("neg_lo", inst.neg, n)] if inst.neg else []) + ([_fmt_bits("neg_hi", inst.neg_hi, n)] if inst.neg_hi else []) + (["clamp"] if clamp else [])
  return f"{name} {dst}, {src0}, {src1}, {src2}{' ' + ' '.join(mods) if mods else ''}" if n == 3 else f"{name} {dst}, {src0}, {src1}{' ' + ' '.join(mods) if mods else ''}"

def _disasm_sop1(inst: SOP1) -> str:
  op, name, cdna = inst.op, inst.op_name.lower(), _is_cdna(inst)
  # Use get_field_bits for register sizes
  regs = inst.canonical_op_regs
  dst_regs, src_regs = regs.get('d', 1), regs.get('s0', 1)
  src = _lit(inst, inst.ssrc0) if _unwrap(inst.ssrc0) == 255 else _fmt_src(inst.ssrc0, src_regs, cdna)
  if not cdna:
    if 'getpc_b64' in name: return f"{name} {_fmt_sdst(inst.sdst, 2)}"
    if 'setpc_b64' in name or 'rfe_b64' in name: return f"{name} {src}"
    if 'swappc_b64' in name: return f"{name} {_fmt_sdst(inst.sdst, 2)}, {src}"
    if 'sendmsg_rtn' in name:
      v = _unwrap(inst.ssrc0)
      try: msg_str = MSG(v).name if v != 255 else None  # MSG_RTN_ILLEGAL_MSG (255) not supported by LLVM
      except ValueError: msg_str = None
      return f"{name} {_fmt_sdst(inst.sdst, dst_regs)}, sendmsg({msg_str})" if msg_str else f"{name} {_fmt_sdst(inst.sdst, dst_regs)}, 0x{v:x}"
  sop1_src_only = ('S_ALLOC_VGPR', 'S_SLEEP_VAR', 'S_BARRIER_SIGNAL', 'S_BARRIER_SIGNAL_ISFIRST', 'S_BARRIER_INIT', 'S_BARRIER_JOIN', 'S_SET_GPR_IDX_IDX',
                   'S_CBRANCH_JOIN')
  if inst.op_name in sop1_src_only: return f"{name} {src}"
  if cdna:
    if 'getpc_b64' in name: return f"{name} {_fmt_sdst(inst.sdst, 2, cdna)}"
    if 'setpc_b64' in name or 'rfe_b64' in name: return f"{name} {src}"
    if 'swappc_b64' in name: return f"{name} {_fmt_sdst(inst.sdst, 2, cdna)}, {src}"
  return f"{name} {_fmt_sdst(inst.sdst, dst_regs, cdna)}, {src}"

def _disasm_sop2(inst: SOP2) -> str:
  cdna, name = _is_cdna(inst), inst.op_name.lower()
  lit = inst._literal
  # Use get_field_bits for register sizes
  regs = inst.canonical_op_regs
  dn, s0n, s1n = regs['d'], regs['s0'], regs['s1']
  s0 = _lit(inst, inst.ssrc0) if _unwrap(inst.ssrc0) == 255 else _fmt_src(inst.ssrc0, s0n, cdna)
  s1 = _lit(inst, inst.ssrc1) if _unwrap(inst.ssrc1) == 255 else _fmt_src(inst.ssrc1, s1n, cdna)
  dst = _fmt_sdst(inst.sdst, dn, cdna)
  if 'fmamk' in name and lit is not None: return f"{name} {dst}, {s0}, 0x{lit:x}, {s1}"
  if 'fmaak' in name and lit is not None: return f"{name} {dst}, {s0}, {s1}, 0x{lit:x}"
  if name in ('s_cbranch_g_fork', 's_rfe_restore_b64'): return f"{name} {s0}, {s1}"  # no destination
  return f"{name} {dst}, {s0}, {s1}"

def _disasm_sopc(inst: SOPC) -> str:
  cdna, regs, name = _is_cdna(inst), inst.canonical_op_regs, inst.op_name.lower()
  s0 = _lit(inst, inst.ssrc0) if _unwrap(inst.ssrc0) == 255 else _fmt_src(inst.ssrc0, regs['s0'], cdna)
  if name == 's_set_gpr_idx_on':
    imm = _unwrap(inst.ssrc1) & 0xf
    flags = [n for i, n in enumerate(['SRC0', 'SRC1', 'SRC2', 'DST']) if imm & (1 << i)]
    return f"{name} {s0}, gpr_idx({','.join(flags)})"
  s1 = _lit(inst, inst.ssrc1) if _unwrap(inst.ssrc1) == 255 else _fmt_src(inst.ssrc1, regs['s1'], cdna)
  return f"{name} {s0}, {s1}"

_HWREG_BLACKLIST = {'HW_REG_PC_LO', 'HW_REG_PC_HI', 'HW_REG_IB_DBG1', 'HW_REG_FLUSH_IB', 'HW_REG_SHADER_TBA_LO', 'HW_REG_SHADER_TBA_HI',
                    'HW_REG_SHADER_FLAT_SCRATCH_LO', 'HW_REG_SHADER_FLAT_SCRATCH_HI', 'HW_REG_SHADER_CYCLES'}
_HWREG_BLACKLIST_CDNA = {'HW_REG_PC_LO', 'HW_REG_PC_HI', 'HW_REG_IB_DBG1', 'HW_REG_FLUSH_IB', 'HW_REG_SQ_SHADER_TBA_LO', 'HW_REG_SQ_SHADER_TBA_HI',
                         'HW_REG_SQ_SHADER_TMA_LO', 'HW_REG_SQ_SHADER_TMA_HI', 'HW_REG_SQ_PERF_SNAPSHOT_DATA', 'HW_REG_SQ_PERF_SNAPSHOT_DATA1',
                         'HW_REG_SQ_PERF_SNAPSHOT_PC_LO', 'HW_REG_SQ_PERF_SNAPSHOT_PC_HI', 'HW_REG_XCC_ID'}
def _disasm_sopk(inst: SOPK) -> str:
  op, name, cdna = inst.op, inst.op_name.lower(), _is_cdna(inst)
  is_rdna4 = _is_r4(inst)
  hw = HWREG_CDNA if cdna else (HWREG_RDNA4 if is_rdna4 else HWREG)
  blacklist = _HWREG_BLACKLIST_CDNA if cdna else _HWREG_BLACKLIST
  def fmt_hwreg(hid, hoff, hsz):
    try: hr_name = hw(hid).name.replace("HW_REG_WAVE_", "HW_REG_")
    except ValueError: return f"0x{inst.simm16:x}"
    if hr_name in blacklist: return f"0x{inst.simm16:x}"
    return f"hwreg({hr_name})" if hoff == 0 and hsz == 32 else f"hwreg({hr_name}, {hoff}, {hsz})"
  if name == 's_setreg_imm32_b32':
    hid, hoff, hsz = inst.simm16 & 0x3f, (inst.simm16 >> 6) & 0x1f, ((inst.simm16 >> 11) & 0x1f) + 1
    return f"{name} {fmt_hwreg(hid, hoff, hsz)}, 0x{inst._literal:x}"
  if name == 's_version': return f"{name} 0x{inst.simm16:x}"
  if name in ('s_setreg_b32', 's_getreg_b32'):
    hid, hoff, hsz = inst.simm16 & 0x3f, (inst.simm16 >> 6) & 0x1f, ((inst.simm16 >> 11) & 0x1f) + 1
    hs = fmt_hwreg(hid, hoff, hsz)
    return f"{name} {hs}, {_fmt_sdst(inst.sdst, 1, cdna)}" if 'setreg' in name else f"{name} {_fmt_sdst(inst.sdst, 1, cdna)}, {hs}"
  if name in ('s_subvector_loop_begin', 's_subvector_loop_end'):
    return f"{name} {_fmt_sdst(inst.sdst, 1)}, 0x{inst.simm16:x}"
  return f"{name} {_fmt_sdst(inst.sdst, inst.canonical_op_regs['d'], cdna)}, 0x{inst.simm16:x}"

def _disasm_vinterp(inst: VINTERP) -> str:
  mods = _mods((inst.waitexp, f"wait_exp:{inst.waitexp}"), (inst.clmp, "clamp"))
  return f"{inst.op_name.lower()} {inst.vdst.fmt()}, {_lit(inst, inst.src0, inst.neg & 1)}, {_lit(inst, inst.src1, inst.neg & 2)}, {_lit(inst, inst.src2, inst.neg & 4)}" + (" " + mods if mods else "")

DISASM_HANDLERS: dict[type, Callable[..., str]] = {
  VOP1: _disasm_vop1, VOP1_SDST: _disasm_vop1, VOP1_SDST_LIT: _disasm_vop1, VOP1_LIT: _disasm_vop1,
  VOP2: _disasm_vop2, VOP2_LIT: _disasm_vop2, VOPC: _disasm_vopc, VOPC_LIT: _disasm_vopc,
  VOP3: _disasm_vop3, VOP3_SDST: _disasm_vop3, VOP3_SDST_LIT: _disasm_vop3, VOP3_LIT: _disasm_vop3, VOP3SD: _disasm_vop3sd, VOP3SD_LIT: _disasm_vop3sd,
  VOPD: _disasm_vopd, VOPD_LIT: _disasm_vopd, VOP3P: _disasm_vop3p, VOP3P_LIT: _disasm_vop3p,
  VINTERP: _disasm_vinterp, SOPP: _disasm_sopp, SMEM: _disasm_smem, DS: _disasm_ds, FLAT: _disasm_flat, GLOBAL: _disasm_flat, SCRATCH: _disasm_flat,
  SOP1: _disasm_sop1, SOP1_LIT: _disasm_sop1, SOP2: _disasm_sop2, SOP2_LIT: _disasm_sop2,
  SOPC: _disasm_sopc, SOPC_LIT: _disasm_sopc, SOPK: _disasm_sopk, SOPK_LIT: _disasm_sopk,
  # RDNA4
  R4_VOP1: _disasm_vop1, R4_VOP1_SDST: _disasm_vop1, R4_VOP1_SDST_LIT: _disasm_vop1, R4_VOP1_LIT: _disasm_vop1,
  R4_VOP2: _disasm_vop2, R4_VOP2_LIT: _disasm_vop2, R4_VOPC: _disasm_vopc, R4_VOPC_LIT: _disasm_vopc,
  R4_VOP3: _disasm_vop3, R4_VOP3_SDST: _disasm_vop3, R4_VOP3_SDST_LIT: _disasm_vop3, R4_VOP3_LIT: _disasm_vop3,
  R4_VOP3SD: _disasm_vop3sd, R4_VOP3SD_LIT: _disasm_vop3sd, R4_VOP3P: _disasm_vop3p, R4_VOP3P_LIT: _disasm_vop3p,
  R4_FLAT: _disasm_flat, R4_GLOBAL: _disasm_flat, R4_SCRATCH: _disasm_flat,
  R4_VOPD: _disasm_vopd, R4_VOPD_LIT: _disasm_vopd, R4_VINTERP: _disasm_vinterp, R4_SOPP: _disasm_sopp, R4_SMEM: _disasm_smem, R4_DS: _disasm_ds,
  R4_SOP1: _disasm_sop1, R4_SOP1_LIT: _disasm_sop1, R4_SOP2: _disasm_sop2, R4_SOP2_LIT: _disasm_sop2,
  R4_SOPC: _disasm_sopc, R4_SOPC_LIT: _disasm_sopc, R4_SOPK: _disasm_sopk, R4_SOPK_LIT: _disasm_sopk}

def disasm(inst: Inst) -> str: return DISASM_HANDLERS[type(inst)](inst)

# ═══════════════════════════════════════════════════════════════════════════════
# CDNA DISASSEMBLER SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════

from extra.assembly.amd.autogen.cdna.ins import (VOP1 as CDNA_VOP1, VOP1_LIT as CDNA_VOP1_LIT,
  VOP1_SDWA as CDNA_VOP1_SDWA, VOP1_DPP16 as CDNA_VOP1_DPP16,
  VOP2 as CDNA_VOP2, VOP2_LIT as CDNA_VOP2_LIT, VOP2_SDWA as CDNA_VOP2_SDWA, VOP2_DPP16 as CDNA_VOP2_DPP16,
  VOPC as CDNA_VOPC, VOPC_LIT as CDNA_VOPC_LIT, VOPC_SDWA_SDST as CDNA_VOPC_SDWA_SDST,
  VOP3 as CDNA_VOP3, VOP3_SDST as CDNA_VOP3_SDST, VOP3SD as CDNA_VOP3SD, VOP3P as CDNA_VOP3P, VOP3P_MFMA as CDNA_VOP3P_MFMA, VOP3PX2 as CDNA_VOP3PX2,
  SOP1 as CDNA_SOP1, SOP1_LIT as CDNA_SOP1_LIT, SOP2 as CDNA_SOP2, SOP2_LIT as CDNA_SOP2_LIT,
  SOPC as CDNA_SOPC, SOPC_LIT as CDNA_SOPC_LIT, SOPK as CDNA_SOPK, SOPK_LIT as CDNA_SOPK_LIT,
  SOPP as CDNA_SOPP, SMEM as CDNA_SMEM, DS as CDNA_DS,
  FLAT as CDNA_FLAT, GLOBAL as CDNA_GLOBAL, SCRATCH as CDNA_SCRATCH, MUBUF as CDNA_MUBUF)

def _cdna_src(inst, v, neg, abs_=0, n=1):
  s = _lit(inst, v) if v == 255 else _fmt_src(v, n, cdna=True)
  if abs_: s = f"|{s}|"
  return f"neg({s})" if neg and v == 255 else (f"-{s}" if neg else s)

_CDNA_VOP3_ALIASES = {'v_fmac_f64': 'v_mul_legacy_f32', 'v_dot2c_f32_bf16': 'v_mac_f32'}

def _disasm_vop3a(inst) -> str:
  op_val = inst.op.value if hasattr(inst.op, 'value') else inst.op
  name = inst.op_name.lower() or f'vop3a_op_{op_val}'
  n = inst.num_srcs() or _num_srcs(inst)
  cl, om = " clamp" if inst.clmp else "", _omod(inst.omod)
  # _sr_ instructions use 4-element op_sel (src2 for byte selection)
  opsel_n = 3 if '_sr_' in name and n == 2 else n
  opsel = _opsel_str(inst.opsel, opsel_n, inst.opsel != 0, False)
  orig_name = name
  name = _CDNA_VOP3_ALIASES.get(name, name)
  if name != orig_name:
    s0, s1 = _cdna_src(inst, inst.src0, inst.neg&1, inst.abs&1, 1), _cdna_src(inst, inst.src1, inst.neg&2, inst.abs&2, 1)
    s2 = ""
    dst = _vreg(inst.vdst)
  else:
    regs = inst.canonical_op_regs
    dregs, r0, r1, r2 = regs['d'], regs['s0'], regs['s1'], regs['s2']
    s0, s1, s2 = _cdna_src(inst, inst.src0, inst.neg&1, inst.abs&1, r0), _cdna_src(inst, inst.src1, inst.neg&2, inst.abs&2, r1), _cdna_src(inst, inst.src2, inst.neg&4, inst.abs&4, r2)
    dst = _vreg(inst.vdst, dregs) if dregs > 1 else _vreg(inst.vdst)
  if op_val >= 512:
    return f"{name} {dst}, {s0}, {s1}, {s2}{opsel}{cl}{om}" if n == 3 else f"{name} {dst}, {s0}, {s1}{opsel}{cl}{om}"
  if op_val < 256:
    # VOPC: vdst is actually sdst (SGPR pair), but VGPRField adds 256 to the offset
    sdst_val = _unwrap(inst.vdst)
    if sdst_val >= 256: sdst_val -= 256
    sdst = _fmt_sdst(sdst_val, 2, cdna=True)
    return f"{name} {sdst}, {s0}, {s1}{cl}"
  if 320 <= op_val < 512:
    if name in ('v_nop', 'v_clrexcp', 'v_nop_e64', 'v_clrexcp_e64'): return name.replace('_e64', '')
    return f"{name} {dst}, {s0}{cl}{om}"
  if name == 'v_cndmask_b32':
    s2 = _fmt_src(inst.src2, 2, cdna=True)
    return f"{name} {dst}, {s0}, {s1}, {s2}{cl}{om}"
  return f"{name} {dst}, {s0}, {s1}, {s2}{opsel}{cl}{om}" if n == 3 else f"{name} {dst}, {s0}, {s1}{opsel}{cl}{om}"

def _disasm_vop3b(inst) -> str:
  op_val = inst.op.value if hasattr(inst.op, 'value') else inst.op
  name, cdna = inst.op_name.lower() or f'vop3b_op_{op_val}', _is_cdna(inst)
  n = inst.num_srcs() or _num_srcs(inst)
  regs = inst.canonical_op_regs
  dregs, r0, r1, r2 = regs['d'], regs['s0'], regs['s1'], regs['s2']
  s0, s1, s2 = _cdna_src(inst, inst.src0, inst.neg&1, n=r0), _cdna_src(inst, inst.src1, inst.neg&2, n=r1), _cdna_src(inst, inst.src2, inst.neg&4, n=r2)
  # CDNA VOP3_SDST uses vdst field for sdst (but vdst adds 256), RDNA uses separate sdst field
  sdst_val = getattr(inst, 'sdst', None)
  if sdst_val is None and hasattr(inst, 'vdst'):
    sdst_val = _unwrap(inst.vdst)
    if sdst_val >= 256: sdst_val -= 256  # VGPRField adds 256, remove it for SGPR
  # For CDNA VOP3_SDST (VOPC->VOP3), vdst is the scalar dest (sdst), there's no vdst output
  if cdna and 'v_cmp' in name:
    sdst = _fmt_sdst(sdst_val, 2, cdna=True)
    cl, om = " clamp" if inst.clmp else "", _omod(inst.omod)
    return f"{name} {sdst}, {s0}, {s1}{cl}{om}"
  dst = _vreg(inst.vdst, dregs) if dregs > 1 else _vreg(inst.vdst)
  sdst = _fmt_sdst(sdst_val, 2, cdna=cdna)
  cl, om = " clamp" if inst.clmp else "", _omod(inst.omod)
  if name in ('v_addc_co_u32', 'v_subb_co_u32', 'v_subbrev_co_u32'):
    s2 = _fmt_src(inst.src2, 2, cdna=cdna)
  return f"{name} {dst}, {sdst}, {s0}, {s1}, {s2}{cl}{om}" if n == 3 else f"{name} {dst}, {sdst}, {s0}, {s1}{cl}{om}"

def _disasm_cdna_vop3p(inst) -> str:
  name, n = inst.op_name.lower(), inst.num_srcs() or 2
  is_mfma = 'mfma' in name or 'smfmac' in name
  is_accvgpr = 'accvgpr' in name
  get_src = lambda v, sc: _lit(inst, v) if v == 255 else _fmt_src(v, sc, cdna=True)

  # Handle accvgpr read/write (accumulator register operations)
  if is_accvgpr:
    src0_off = _unwrap(inst.src0)
    vdst_off = _vi(inst.vdst)
    if 'read' in name:
      # v_accvgpr_read_b32 vN, aM - reads from accumulator to VGPR
      return f"{name}_b32 v{vdst_off}, a{src0_off - 256 if src0_off >= 256 else src0_off}"
    if 'write' in name:
      # v_accvgpr_write_b32 aM, src - writes to accumulator from source
      src = _lit(inst, inst.src0) if src0_off == 255 else (f"v{src0_off - 256}" if src0_off >= 256 else decode_src(src0_off, cdna=True))
      return f"{name}_b32 a{vdst_off}, {src}"

  # Handle v_mfma_ld_scale_b32 - special 2-operand format: v_mfma_ld_scale_b32 src0, src1
  if 'ld_scale' in name:
    src0, src1 = get_src(inst.src0, 1), get_src(inst.src1, 1)
    mods = ([_fmt_bits("op_sel", inst.opsel, 2)] if inst.opsel else []) + \
           ([_fmt_bits("op_sel_hi", inst.opsel_hi, 2)] if inst.opsel_hi != 3 else [])
    return f"{name} {src0}, {src1}{' ' + ' '.join(mods) if mods else ''}"

  # Handle MFMA instructions with accumulator destinations
  if is_mfma:
    regs = inst.canonical_op_regs
    dregs, r0, r1, r2 = regs['d'], regs['s0'], regs['s1'], regs['s2']
    # Infer register counts from instruction name if not in operands table (e.g., v_mfma_f32_32x32x4_xf32)
    if dregs == 1:
      if '32x32' in name: dregs, r0, r1, r2 = 16, 2, 2, 16
      elif '16x16' in name: dregs, r0, r1, r2 = 4, 2, 2, 4
    # MFMA reuses VOP3P fields differently: clmp -> acc_cd (dest is acc), opsel_hi -> acc (src1/src2 are acc)
    # acc field (bits 60-59): bit 0 = src2 is acc (always for MFMA), bit 1 = src1 is acc
    acc = inst.opsel_hi  # opsel_hi field maps to acc for MFMA
    acc_cd = inst.clmp   # clmp field maps to acc_cd for MFMA (dest is accumulator)
    is_smfmac = 'smfmac' in name  # SMFMAC has different operand semantics
    # Format sources: src0 is always VGPR, src1/src2 depend on acc bits
    def mfma_src(v, sc, is_acc):
      v = _unwrap(v)
      if v == 255: return _lit(inst, v)
      if 128 <= v <= 208 or 240 <= v <= 248: return _lit(inst, v)
      base = v - 256 if v >= 256 else v
      if is_acc: return _areg(base, sc)
      return _vreg(base, sc)
    src0 = get_src(inst.src0, r0)  # src0 is always VGPR
    src1 = mfma_src(inst.src1, r1, acc & 2)  # bit 1 = src1 is acc
    # For SMFMAC, src2 is always a VGPR index (1 register), not accumulator
    src2 = _vreg(inst.src2) if is_smfmac else mfma_src(inst.src2, r2, acc_cd)
    dst = _areg(inst.vdst, dregs) if acc_cd else _vreg(inst.vdst, dregs)
    # MFMA uses neg:[...] not neg_lo:[...], and doesn't support op_sel_hi or clamp
    # Only f64 MFMA instructions support neg modifier
    # f8f6f4 MFMA instructions support cbsz/blgp modifiers
    mods = []
    if 'f8f6f4' in name:
      if inst.neg_hi: mods.append(f"cbsz:{inst.neg_hi}")
      if inst.neg: mods.append(f"blgp:{inst.neg}")
    elif inst.neg and 'f64' in name:
      mods.append(_fmt_bits("neg", inst.neg, n))
    return f"{name} {dst}, {src0}, {src1}, {src2}{' ' + ' '.join(mods) if mods else ''}"

  # Standard VOP3P instructions
  src0, src1, src2, dst = get_src(inst.src0, 1), get_src(inst.src1, 1), get_src(inst.src2, 1), _vreg(inst.vdst)
  opsel_hi = inst.opsel_hi  # CDNA VOP3P only has 2 bits for opsel_hi (no opsel_hi2)
  opsel_hi_default = 3  # CDNA default is 0b11 (2 bits), not 0b111 like RDNA
  mods = ([_fmt_bits("op_sel", inst.opsel, n)] if inst.opsel else []) + ([_fmt_bits("op_sel_hi", opsel_hi, n)] if opsel_hi != opsel_hi_default else []) + \
          ([_fmt_bits("neg_lo", inst.neg, n)] if inst.neg else []) + ([_fmt_bits("neg_hi", inst.neg_hi, n)] if inst.neg_hi else []) + (["clamp"] if inst.clmp else [])
  return f"{name} {dst}, {src0}, {src1}, {src2}{' ' + ' '.join(mods) if mods else ''}" if n == 3 else f"{name} {dst}, {src0}, {src1}{' ' + ' '.join(mods) if mods else ''}"

def _disasm_mubuf(inst) -> str:
  name = inst.op_name.lower()
  # Determine vdata register count from instruction name
  nregs = 4 if 'xyzw' in name else 3 if 'xyz' in name else 2 if 'xy' in name or 'x2' in name or 'f64' in name or 'dwordx2' in name else 1
  vdata = _vreg(inst.vdata, nregs)
  vaddr = _vreg(inst.vaddr) if inst.offen or inst.idxen else None
  srsrc = str(inst.srsrc)
  soffset_val = _unwrap(inst.soffset)
  soffset = f"s{soffset_val}" if soffset_val < 128 else "off"
  offset = f" offset:{inst.offset}" if inst.offset else ""
  offen = " offen" if inst.offen else ""
  idxen = " idxen" if inst.idxen else ""
  lds = " lds" if inst.lds else ""
  sc0 = " sc0" if inst.sc0 else ""
  sc1 = " sc1" if inst.sc1 else ""
  nt = " nt" if inst.nt else ""
  # Handle special cases
  if name in ('buffer_wbl2', 'buffer_inv'):
    return f"{name}{sc0}{sc1}"
  if vaddr:
    return f"{name} {vdata}, {vaddr}, {srsrc}, {soffset}{offen}{idxen}{offset}{sc0}{nt}{sc1}{lds}"
  return f"{name} {vdata}, off, {srsrc}, {soffset}{offset}{sc0}{nt}{sc1}{lds}"

_SDWA_SEL = {0: 'BYTE_0', 1: 'BYTE_1', 2: 'BYTE_2', 3: 'BYTE_3', 4: 'WORD_0', 5: 'WORD_1', 6: 'DWORD'}

def _disasm_vop1_sdwa(inst) -> str:
  name = inst.op_name.lower().replace('_e32', '')
  regs = inst.canonical_op_regs
  dst = _vreg(inst.vdst, regs['d'])
  # When s0=1, vsrc0 is SGPR/constant (VGPRField adds 256, so subtract it back)
  if inst.s0 == 0: src0 = _vreg(inst.vsrc0, regs['s0'])
  else:
    raw = _unwrap(inst.vsrc0) - 256  # VGPRField adds 256
    src0 = decode_src(raw, cdna=True)  # handles SGPRs, constants, specials
  src0_sel = _SDWA_SEL.get(inst.src0_sel, f'SEL{inst.src0_sel}')
  mods = []
  if inst.clmp: mods.append("clamp")
  if inst.omod == 1: mods.append("mul:2")
  elif inst.omod == 2: mods.append("mul:4")
  elif inst.omod == 3: mods.append("div:2")
  mods.append(f"src0_sel:{src0_sel}")
  return f"{name}_sdwa {dst}, {src0} {' '.join(mods)}"

def _decode_dpp(dpp: int) -> str:
  """Decode DPP control value to string."""
  if dpp < 0x100: return f"quad_perm:[{dpp&3},{(dpp>>2)&3},{(dpp>>4)&3},{(dpp>>6)&3}]"
  if 0x100 <= dpp <= 0x10f: return f"row_shl:{dpp & 0xf}"
  if 0x110 <= dpp <= 0x11f: return f"row_shr:{dpp & 0xf}"
  if 0x120 <= dpp <= 0x12f: return f"row_ror:{dpp & 0xf}"
  if dpp == 0x130: return "wave_shl:1"
  if dpp == 0x134: return "wave_rol:1"
  if dpp == 0x138: return "wave_shr:1"
  if dpp == 0x13c: return "wave_ror:1"
  if dpp == 0x140: return "row_mirror"
  if dpp == 0x141: return "row_half_mirror"
  if dpp == 0x142: return "row_bcast:15"
  if dpp == 0x143: return "row_bcast:31"
  if 0x150 <= dpp <= 0x15f: return f"row_newbcast:{dpp & 0xf}"
  if 0x160 <= dpp <= 0x16f: return f"row_share:{dpp & 0xf}"
  if 0x170 <= dpp <= 0x17f: return f"row_xmask:{dpp & 0xf}"
  return f"dpp:{dpp:#x}"

def _disasm_vop1_dpp(inst) -> str:
  name = inst.op_name.lower().replace('_e32', '')
  regs = inst.canonical_op_regs
  dst, src0 = _vreg(inst.vdst, regs['d']), _vreg(inst.vsrc0, regs['s0'])
  dpp_str = _decode_dpp(inst.dpp)
  mods = [dpp_str]
  if inst.row_mask != 0xf: mods.append(f"row_mask:{inst.row_mask:#x}")
  if inst.bank_mask != 0xf: mods.append(f"bank_mask:{inst.bank_mask:#x}")
  if inst.bc: mods.append("bound_ctrl:1")
  return f"{name}_dpp {dst}, {src0} {' '.join(mods)}"

def _disasm_vop2_sdwa(inst) -> str:
  name, cdna = inst.op_name.lower().replace('_e32', ''), _is_cdna(inst)
  regs = inst.canonical_op_regs
  dst = _vreg(inst.vdst, regs['d'])
  # When s0/s1=1, vsrc is SGPR/constant (VGPRField adds 256, so subtract it back)
  src0 = _vreg(inst.vsrc0, regs['s0']) if inst.s0 == 0 else decode_src(_unwrap(inst.vsrc0) - 256, cdna)
  src1 = _vreg(inst.vsrc1, regs['s1']) if inst.s1 == 0 else decode_src(_unwrap(inst.vsrc1) - 256, cdna)
  src0_sel = _SDWA_SEL.get(inst.src0_sel, f'SEL{inst.src0_sel}')
  src1_sel = _SDWA_SEL.get(inst.src1_sel, f'SEL{inst.src1_sel}')
  mods = []
  if inst.clmp: mods.append("clamp")
  if inst.omod == 1: mods.append("mul:2")
  elif inst.omod == 2: mods.append("mul:4")
  elif inst.omod == 3: mods.append("div:2")
  if inst.src0_sel != 6: mods.append(f"src0_sel:{src0_sel}")
  if inst.src1_sel != 6: mods.append(f"src1_sel:{src1_sel}")
  mods_str = ' '.join(mods) if mods else ""
  # CDNA carry instructions and cndmask need vcc operands
  if cdna and name in _VOP2_CARRY_OUT: return f"{name}_sdwa {dst}, vcc, {src0}, {src1} {mods_str}".strip()
  if cdna and name in _VOP2_CARRY_INOUT: return f"{name}_sdwa {dst}, vcc, {src0}, {src1}, vcc {mods_str}".strip()
  if cdna and name == 'v_cndmask_b32': return f"{name}_sdwa {dst}, {src0}, {src1}, vcc {mods_str}".strip()
  return f"{name}_sdwa {dst}, {src0}, {src1} {mods_str}".strip()

def _disasm_vop2_dpp(inst) -> str:
  name, cdna = inst.op_name.lower().replace('_e32', ''), _is_cdna(inst)
  regs = inst.canonical_op_regs
  dst, src0, src1 = _vreg(inst.vdst, regs['d']), _vreg(inst.vsrc0, regs['s0']), _vreg(inst.vsrc1, regs['s1'])
  dpp_str = _decode_dpp(inst.dpp)
  mods = [dpp_str]
  if inst.row_mask != 0xf: mods.append(f"row_mask:{inst.row_mask:#x}")
  if inst.bank_mask != 0xf: mods.append(f"bank_mask:{inst.bank_mask:#x}")
  if inst.bc: mods.append("bound_ctrl:1")
  # CDNA carry instructions and cndmask need vcc operands
  if cdna and name in _VOP2_CARRY_OUT: return f"{name}_dpp {dst}, vcc, {src0}, {src1} {' '.join(mods)}"
  if cdna and name in _VOP2_CARRY_INOUT: return f"{name}_dpp {dst}, vcc, {src0}, {src1}, vcc {' '.join(mods)}"
  if cdna and name == 'v_cndmask_b32': return f"{name}_dpp {dst}, {src0}, {src1}, vcc {' '.join(mods)}"
  return f"{name}_dpp {dst}, {src0}, {src1} {' '.join(mods)}"

def _disasm_vopc_sdwa(inst) -> str:
  name = inst.op_name.lower().replace('_e32', '')
  regs = inst.canonical_op_regs
  sdst = _fmt_sdst(inst.sdst, 2, cdna=True)
  src0 = _vreg(inst.vsrc0, regs['s0']) if getattr(inst, 's0', 0) == 0 else decode_src(_unwrap(inst.vsrc0) - 256, cdna=True)
  src1 = _vreg(inst.vsrc1, regs['s1']) if getattr(inst, 's1', 0) == 0 else decode_src(_unwrap(inst.vsrc1) - 256, cdna=True)
  src0_sel = _SDWA_SEL.get(inst.src0_sel, f'SEL{inst.src0_sel}')
  src1_sel = _SDWA_SEL.get(inst.src1_sel, f'SEL{inst.src1_sel}')
  mods = []
  if inst.src0_sel != 6: mods.append(f"src0_sel:{src0_sel}")
  if inst.src1_sel != 6: mods.append(f"src1_sel:{src1_sel}")
  return f"{name}_sdwa {sdst}, {src0}, {src1} {' '.join(mods)}".strip()

def _disasm_vop3px2(inst) -> str:
  """VOP3PX2 disassembler for scaled MFMA instructions."""
  name = inst.op_name.lower()
  regs = inst.canonical_op_regs
  dregs, r2 = regs['d'], regs['s2']
  # F8F6F4 MFMA: CBSZ selects matrix A format, BLGP selects matrix B format
  # VGPRs: FP8/BF8(0,1)=8, FP6/BF6(2,3)=6, FP4(4)=4
  vgprs = {0: 8, 1: 8, 2: 6, 3: 6, 4: 4}
  r0, r1 = vgprs.get(inst.cbsz, 8), vgprs.get(inst.blgp, 8)
  def mfma_src(v, sc, is_acc):
    v = _unwrap(v)
    if v == 255: return _lit(inst, v)
    base = v - 256 if v >= 256 else v
    return _areg(base, sc) if is_acc else _vreg(base, sc)
  src0, src1, src2 = mfma_src(inst.src0, r0, False), mfma_src(inst.src1, r1, inst.acc & 2), mfma_src(inst.src2, r2, inst.acc_cd)
  dst = _areg(inst.vdst, dregs) if inst.acc_cd else _vreg(inst.vdst, dregs)
  scale_src0, scale_src1 = _vreg(inst.scale_src0), _vreg(inst.scale_src1)
  mods = []
  if inst.opsel: mods.append(_fmt_bits("op_sel", inst.opsel, 3))
  if inst.opsel_hi != 0: mods.append(_fmt_bits("op_sel_hi", inst.opsel_hi, 3))
  if inst.neg: mods.append(_fmt_bits("neg", inst.neg, 3))
  if inst.cbsz: mods.append(f"cbsz:{inst.cbsz}")
  if inst.blgp: mods.append(f"blgp:{inst.blgp}")
  return f"{name} {dst}, {src0}, {src1}, {src2}, {scale_src0}, {scale_src1}{' ' + ' '.join(mods) if mods else ''}"

DISASM_HANDLERS.update({CDNA_VOP1: _disasm_vop1, CDNA_VOP1_LIT: _disasm_vop1,
  CDNA_VOP1_SDWA: _disasm_vop1_sdwa, CDNA_VOP1_DPP16: _disasm_vop1_dpp,
  CDNA_VOP2: _disasm_vop2, CDNA_VOP2_LIT: _disasm_vop2,
  CDNA_VOP2_SDWA: _disasm_vop2_sdwa, CDNA_VOP2_DPP16: _disasm_vop2_dpp,
  CDNA_VOPC: _disasm_vopc, CDNA_VOPC_LIT: _disasm_vopc, CDNA_VOPC_SDWA_SDST: _disasm_vopc_sdwa,
  CDNA_SOP1: _disasm_sop1, CDNA_SOP1_LIT: _disasm_sop1, CDNA_SOP2: _disasm_sop2, CDNA_SOP2_LIT: _disasm_sop2,
  CDNA_SOPC: _disasm_sopc, CDNA_SOPC_LIT: _disasm_sopc, CDNA_SOPK: _disasm_sopk, CDNA_SOPK_LIT: _disasm_sopk, CDNA_SOPP: _disasm_sopp,
  CDNA_SMEM: _disasm_smem, CDNA_DS: _disasm_ds, CDNA_FLAT: _disasm_flat, CDNA_GLOBAL: _disasm_flat, CDNA_SCRATCH: _disasm_flat,
  CDNA_VOP3: _disasm_vop3a, CDNA_VOP3_SDST: _disasm_vop3b, CDNA_VOP3SD: _disasm_vop3b, CDNA_VOP3P: _disasm_cdna_vop3p, CDNA_VOP3P_MFMA: _disasm_cdna_vop3p,
  CDNA_MUBUF: _disasm_mubuf, CDNA_VOP3PX2: _disasm_vop3px2})
