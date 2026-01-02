# RDNA3 assembler and disassembler
from __future__ import annotations
import re
from extra.assembly.rdna3.lib import Inst, RawImm, Reg, SGPR, VGPR, TTMP, s, v, ttmp, _RegFactory, FLOAT_ENC, SRC_FIELDS, unwrap

# Decoding helpers
SPECIAL_GPRS = {106: "vcc_lo", 107: "vcc_hi", 124: "null", 125: "m0", 126: "exec_lo", 127: "exec_hi", 253: "scc"}
SPECIAL_DEC = {**SPECIAL_GPRS, **{v: str(k) for k, v in FLOAT_ENC.items()}}
SPECIAL_PAIRS = {106: "vcc", 126: "exec"}  # Special register pairs (for 64-bit ops)
# GFX11 hwreg names (IDs 16-17 are TBA - not supported, IDs 18-19 are PERF_SNAPSHOT)
HWREG_NAMES = {1: 'HW_REG_MODE', 2: 'HW_REG_STATUS', 3: 'HW_REG_TRAPSTS', 4: 'HW_REG_HW_ID', 5: 'HW_REG_GPR_ALLOC',
               6: 'HW_REG_LDS_ALLOC', 7: 'HW_REG_IB_STS', 15: 'HW_REG_SH_MEM_BASES', 18: 'HW_REG_PERF_SNAPSHOT_PC_LO',
               19: 'HW_REG_PERF_SNAPSHOT_PC_HI', 20: 'HW_REG_FLAT_SCR_LO', 21: 'HW_REG_FLAT_SCR_HI',
               22: 'HW_REG_XNACK_MASK', 23: 'HW_REG_HW_ID1', 24: 'HW_REG_HW_ID2', 25: 'HW_REG_POPS_PACKER', 28: 'HW_REG_IB_STS2'}
HWREG_IDS = {v.lower(): k for k, v in HWREG_NAMES.items()}  # Reverse map for assembler
MSG_NAMES = {128: 'MSG_RTN_GET_DOORBELL', 129: 'MSG_RTN_GET_DDID', 130: 'MSG_RTN_GET_TMA',
             131: 'MSG_RTN_GET_REALTIME', 132: 'MSG_RTN_SAVE_WAVE', 133: 'MSG_RTN_GET_TBA'}
_16BIT_TYPES = ('f16', 'i16', 'u16', 'b16')
def _is_16bit(s: str) -> bool: return any(s.endswith(x) for x in _16BIT_TYPES)

def decode_src(val: int) -> str:
  if val <= 105: return f"s{val}"
  if val in SPECIAL_DEC: return SPECIAL_DEC[val]
  if 108 <= val <= 123: return f"ttmp{val - 108}"
  if 128 <= val <= 192: return str(val - 128)
  if 193 <= val <= 208: return str(-(val - 192))
  if 256 <= val <= 511: return f"v{val - 256}"
  return "lit" if val == 255 else f"?{val}"

def _reg(prefix: str, base: int, cnt: int = 1) -> str: return f"{prefix}{base}" if cnt == 1 else f"{prefix}[{base}:{base+cnt-1}]"
def _sreg(base: int, cnt: int = 1) -> str: return _reg("s", base, cnt)
def _vreg(base: int, cnt: int = 1) -> str: return _reg("v", base, cnt)

def _fmt_sdst(v: int, cnt: int = 1) -> str:
  """Format SGPR destination with special register names."""
  if v == 124: return "null"
  if 108 <= v <= 123: return _reg("ttmp", v - 108, cnt)
  if cnt > 1 and v in SPECIAL_PAIRS: return SPECIAL_PAIRS[v]
  if cnt > 1: return _sreg(v, cnt)
  return {126: "exec_lo", 127: "exec_hi", 106: "vcc_lo", 107: "vcc_hi", 125: "m0"}.get(v, f"s{v}")

def _fmt_ssrc(v: int, cnt: int = 1) -> str:
  """Format SGPR source with special register names and pairs."""
  if cnt == 2:
    if v in SPECIAL_PAIRS: return SPECIAL_PAIRS[v]
    if v <= 105: return _sreg(v, 2)
    if 108 <= v <= 123: return _reg("ttmp", v - 108, 2)
  return decode_src(v)

def _fmt_src_n(v: int, cnt: int) -> str:
  """Format source with given register count (1, 2, or 4)."""
  if cnt == 1: return decode_src(v)
  if v >= 256: return _vreg(v - 256, cnt)
  if v <= 105: return _sreg(v, cnt)
  if cnt == 2 and v in SPECIAL_PAIRS: return SPECIAL_PAIRS[v]
  if 108 <= v <= 123: return _reg("ttmp", v - 108, cnt)
  return decode_src(v)

def _fmt_src64(v: int) -> str:
  """Format 64-bit source (VGPR pair, SGPR pair, or special pair)."""
  return _fmt_src_n(v, 2)

def _parse_sop_sizes(op_name: str) -> tuple[int, ...]:
  """Parse dst and src sizes from SOP instruction name. Returns (dst_cnt, src0_cnt) or (dst_cnt, src0_cnt, src1_cnt)."""
  if op_name in ('s_bitset0_b64', 's_bitset1_b64'): return (2, 1)
  if op_name in ('s_lshl_b64', 's_lshr_b64', 's_ashr_i64', 's_bfe_u64', 's_bfe_i64'): return (2, 2, 1)
  if op_name in ('s_bfm_b64',): return (2, 1, 1)
  # SOPC: s_bitcmp0_b64, s_bitcmp1_b64 - 64-bit src0, 32-bit src1 (bit index)
  if op_name in ('s_bitcmp0_b64', 's_bitcmp1_b64'): return (1, 2, 1)
  if m := re.search(r'_(b|i|u)(32|64)_(b|i|u)(32|64)$', op_name):
    return (2 if m.group(2) == '64' else 1, 2 if m.group(4) == '64' else 1)
  if m := re.search(r'_(b|i|u)(32|64)$', op_name):
    sz = 2 if m.group(2) == '64' else 1
    return (sz, sz)
  return (1, 1)

# Waitcnt helpers (RDNA3 format: bits 15:10=vmcnt, bits 9:4=lgkmcnt, bits 3:0=expcnt)
def waitcnt(vmcnt: int = 0x3f, expcnt: int = 0x7, lgkmcnt: int = 0x3f) -> int:
  return (expcnt & 0x7) | ((lgkmcnt & 0x3f) << 4) | ((vmcnt & 0x3f) << 10)
def decode_waitcnt(val: int) -> tuple[int, int, int]:
  return (val >> 10) & 0x3f, val & 0xf, (val >> 4) & 0x3f  # vmcnt, expcnt, lgkmcnt

# VOP3SD opcodes (shared encoding with VOP3 but different field layout)
# Note: opcodes 0-255 are VOPC promoted to VOP3 - never treat as VOP3SD
VOP3SD_OPCODES = {288, 289, 290, 764, 765, 766, 767, 768, 769, 770}

# Disassembler
def disasm(inst: Inst) -> str:
  op_val = unwrap(inst._values.get('op', 0))
  cls_name = inst.__class__.__name__
  # VOP3 and VOP3SD share encoding - check opcode to determine which
  is_vop3sd = cls_name == 'VOP3' and op_val in VOP3SD_OPCODES
  try:
    from extra.assembly.rdna3 import autogen
    if is_vop3sd:
      op_name = autogen.VOP3SDOp(op_val).name.lower()
    else:
      op_name = getattr(autogen, f"{cls_name}Op")(op_val).name.lower() if hasattr(autogen, f"{cls_name}Op") else f"op_{op_val}"
  except (ValueError, KeyError): op_name = f"op_{op_val}"
  def fmt_src(v): return f"0x{inst._literal:x}" if v == 255 and getattr(inst, '_literal', None) else decode_src(v)

  # VOP1
  if cls_name == 'VOP1':
    vdst, src0 = unwrap(inst._values['vdst']), unwrap(inst._values['src0'])
    if op_name == 'v_nop': return 'v_nop'
    if op_name == 'v_pipeflush': return 'v_pipeflush'
    parts = op_name.split('_')
    is_16bit_dst = any(p in _16BIT_TYPES for p in parts[-2:-1]) or (len(parts) >= 2 and parts[-1] in _16BIT_TYPES and 'cvt' not in op_name)
    is_16bit_src = parts[-1] in _16BIT_TYPES and 'sat_pk' not in op_name
    _F64_OPS = ('v_ceil_f64', 'v_floor_f64', 'v_fract_f64', 'v_frexp_mant_f64', 'v_rcp_f64', 'v_rndne_f64', 'v_rsq_f64', 'v_sqrt_f64', 'v_trunc_f64')
    is_f64_dst = op_name in _F64_OPS or op_name in ('v_cvt_f64_f32', 'v_cvt_f64_i32', 'v_cvt_f64_u32')
    is_f64_src = op_name in _F64_OPS or op_name in ('v_cvt_f32_f64', 'v_cvt_i32_f64', 'v_cvt_u32_f64', 'v_frexp_exp_i32_f64')
    if op_name == 'v_readfirstlane_b32':
      return f"v_readfirstlane_b32 {decode_src(vdst)}, v{src0 - 256 if src0 >= 256 else src0}"
    dst_str = _vreg(vdst, 2) if is_f64_dst else f"v{vdst & 0x7f}.{'h' if vdst >= 128 else 'l'}" if is_16bit_dst else f"v{vdst}"
    src_str = _fmt_src64(src0) if is_f64_src else f"v{(src0 - 256) & 0x7f}.{'h' if src0 >= 384 else 'l'}" if is_16bit_src and src0 >= 256 else fmt_src(src0)
    return f"{op_name}_e32 {dst_str}, {src_str}"

  # VOP2
  if cls_name == 'VOP2':
    vdst, src0_raw, vsrc1 = unwrap(inst._values['vdst']), unwrap(inst._values['src0']), unwrap(inst._values['vsrc1'])
    suffix = "" if op_name == "v_dot2acc_f32_f16" else "_e32"
    is_16bit_op = ('_f16' in op_name or '_i16' in op_name or '_u16' in op_name) and '_f32' not in op_name and '_i32' not in op_name and 'pk_' not in op_name
    if is_16bit_op:
      dst_str = f"v{vdst & 0x7f}.{'h' if vdst >= 128 else 'l'}"
      src0_str = f"v{(src0_raw - 256) & 0x7f}.{'h' if src0_raw >= 384 else 'l'}" if src0_raw >= 256 else fmt_src(src0_raw)
      vsrc1_str = f"v{vsrc1 & 0x7f}.{'h' if vsrc1 >= 128 else 'l'}"
    else:
      dst_str, src0_str, vsrc1_str = f"v{vdst}", fmt_src(src0_raw), f"v{vsrc1}"
    return f"{op_name}{suffix} {dst_str}, {src0_str}, {vsrc1_str}" + (", vcc_lo" if op_name == "v_cndmask_b32" else "")

  # VOPC
  if cls_name == 'VOPC':
    src0, vsrc1 = unwrap(inst._values['src0']), unwrap(inst._values['vsrc1'])
    is_64bit = any(x in op_name for x in ('f64', 'i64', 'u64'))
    is_64bit_vsrc1 = is_64bit and 'class' not in op_name
    is_16bit = any(x in op_name for x in ('_f16', '_i16', '_u16')) and 'f32' not in op_name
    is_cmpx = op_name.startswith('v_cmpx')  # VOPCX writes to exec, no vcc destination
    src0_str = _fmt_src64(src0) if is_64bit else f"v{(src0 - 256) & 0x7f}.{'h' if src0 >= 384 else 'l'}" if is_16bit and src0 >= 256 else fmt_src(src0)
    vsrc1_str = _vreg(vsrc1, 2) if is_64bit_vsrc1 else f"v{vsrc1 & 0x7f}.{'h' if vsrc1 >= 128 else 'l'}" if is_16bit else f"v{vsrc1}"
    return f"{op_name}_e32 {src0_str}, {vsrc1_str}" if is_cmpx else f"{op_name}_e32 vcc_lo, {src0_str}, {vsrc1_str}"

  # SOPP
  if cls_name == 'SOPP':
    simm16 = unwrap(inst._values.get('simm16', 0))
    # No-operand instructions (simm16 is ignored)
    no_imm_ops = ('s_endpgm', 's_barrier', 's_wakeup', 's_icache_inv', 's_ttracedata', 's_ttracedata_imm',
                  's_wait_idle', 's_endpgm_saved', 's_code_end', 's_endpgm_ordered_ps_done')
    if op_name in no_imm_ops: return op_name
    if op_name == 's_waitcnt':
      vmcnt, expcnt, lgkmcnt = decode_waitcnt(simm16)
      parts = []
      if vmcnt != 0x3f: parts.append(f"vmcnt({vmcnt})")
      if expcnt != 0x7: parts.append(f"expcnt({expcnt})")
      if lgkmcnt != 0x3f: parts.append(f"lgkmcnt({lgkmcnt})")
      return f"s_waitcnt {' '.join(parts)}" if parts else "s_waitcnt 0"
    if op_name == 's_delay_alu':
      dep_names = ['VALU_DEP_1','VALU_DEP_2','VALU_DEP_3','VALU_DEP_4','TRANS32_DEP_1','TRANS32_DEP_2','TRANS32_DEP_3','FMA_ACCUM_CYCLE_1','SALU_CYCLE_1','SALU_CYCLE_2','SALU_CYCLE_3']
      skip_names = ['SAME','NEXT','SKIP_1','SKIP_2','SKIP_3','SKIP_4']
      id0, skip, id1 = simm16 & 0xf, (simm16 >> 4) & 0x7, (simm16 >> 7) & 0xf
      def dep_name(v): return dep_names[v-1] if 0 < v <= len(dep_names) else str(v)
      parts = [f"instid0({dep_name(id0)})"] if id0 else []
      if skip: parts.append(f"instskip({skip_names[skip]})")
      if id1: parts.append(f"instid1({dep_name(id1)})")
      return f"s_delay_alu {' | '.join(p for p in parts if p)}" if parts else "s_delay_alu 0"
    if op_name.startswith('s_cbranch') or op_name.startswith('s_branch'):
      return f"{op_name} {simm16}"
    # Most SOPP ops require immediate (s_nop, s_setkill, s_sethalt, s_sleep, s_setprio, s_sendmsg*, etc.)
    return f"{op_name} 0x{simm16:x}"

  # SMEM
  if cls_name == 'SMEM':
    if op_name in ('s_gl1_inv', 's_dcache_inv'): return op_name
    sdata, sbase, soffset, offset = unwrap(inst._values['sdata']), unwrap(inst._values['sbase']), unwrap(inst._values['soffset']), unwrap(inst._values.get('offset', 0))
    glc, dlc = unwrap(inst._values.get('glc', 0)), unwrap(inst._values.get('dlc', 0))
    # Format offset: "soffset offset:X" if both, "0x{offset:x}" if only imm, or decode_src(soffset)
    off_str = f"{decode_src(soffset)} offset:0x{offset:x}" if offset and soffset != 124 else f"0x{offset:x}" if offset else decode_src(soffset)
    sbase_idx, sbase_cnt = sbase * 2, 4 if (8 <= op_val <= 12 or op_name == 's_atc_probe_buffer') else 2
    sbase_str = _fmt_ssrc(sbase_idx, sbase_cnt) if sbase_cnt == 2 else _sreg(sbase_idx, sbase_cnt) if sbase_idx <= 105 else _reg("ttmp", sbase_idx - 108, sbase_cnt)
    if op_name in ('s_atc_probe', 's_atc_probe_buffer'): return f"{op_name} {sdata}, {sbase_str}, {off_str}"
    width = {0:1, 1:2, 2:4, 3:8, 4:16, 8:1, 9:2, 10:4, 11:8, 12:16}.get(op_val, 1)
    mods = [m for m in ["glc" if glc else "", "dlc" if dlc else ""] if m]
    return f"{op_name} {_fmt_sdst(sdata, width)}, {sbase_str}, {off_str}" + (" " + " ".join(mods) if mods else "")

  # FLAT
  if cls_name == 'FLAT':
    vdst, addr, data, saddr, offset, seg = [unwrap(inst._values.get(f, 0)) for f in ['vdst', 'addr', 'data', 'saddr', 'offset', 'seg']]
    instr = f"{['flat', 'scratch', 'global'][seg] if seg < 3 else 'flat'}_{op_name.split('_', 1)[1] if '_' in op_name else op_name}"
    width = {'b32':1, 'b64':2, 'b96':3, 'b128':4, 'u8':1, 'i8':1, 'u16':1, 'i16':1}.get(op_name.split('_')[-1], 1)
    addr_str = _vreg(addr, 2) if saddr == 0x7F else _vreg(addr)
    saddr_str = "" if saddr == 0x7F else f", {_sreg(saddr, 2)}" if saddr < 106 else ", off" if saddr == 124 else f", {decode_src(saddr)}"
    off_str = f" offset:{offset}" if offset else ""
    vdata_str = _vreg(data if 'store' in op_name else vdst, width)
    return f"{instr} {addr_str}, {vdata_str}{saddr_str}{off_str}" if 'store' in op_name else f"{instr} {vdata_str}, {addr_str}{saddr_str}{off_str}"

  # VOP3: vector ops with modifiers (can be 1, 2, or 3 sources depending on opcode range)
  if cls_name == 'VOP3':
    # Handle VOP3SD opcodes (same encoding, different field layout)
    if is_vop3sd:
      vdst = unwrap(inst._values.get('vdst', 0))
      # VOP3SD: sdst is at bits [14:8], but VOP3 decodes opsel at [14:11], abs at [10:8], clmp at [15]
      # We need to reconstruct sdst from these fields
      opsel_raw = unwrap(inst._values.get('opsel', 0))
      abs_raw = unwrap(inst._values.get('abs', 0))
      clmp_raw = unwrap(inst._values.get('clmp', 0))
      sdst = (clmp_raw << 7) | (opsel_raw << 3) | abs_raw
      src0, src1, src2 = [unwrap(inst._values.get(f, 0)) for f in ('src0', 'src1', 'src2')]
      neg = unwrap(inst._values.get('neg', 0))
      omod = unwrap(inst._values.get('omod', 0))
      omod_str = {1: " mul:2", 2: " mul:4", 3: " div:2"}.get(omod, "")
      is_f64 = 'f64' in op_name
      # v_mad_i64_i32/v_mad_u64_u32: 64-bit dst and src2, 32-bit src0/src1
      is_mad64 = 'mad_i64_i32' in op_name or 'mad_u64_u32' in op_name
      def fmt_sd_src(v, neg_bit, is_64bit=False):
        s = _fmt_src64(v) if (is_64bit or is_f64) else fmt_src(v)
        return f"-{s}" if neg_bit else s
      src0_str, src1_str = fmt_sd_src(src0, neg & 1), fmt_sd_src(src1, neg & 2)
      src2_str = fmt_sd_src(src2, neg & 4, is_mad64)
      dst_str = _vreg(vdst, 2) if (is_f64 or is_mad64) else f"v{vdst}"
      sdst_str = _fmt_sdst(sdst, 1)
      # v_add_co_u32, v_sub_co_u32, v_subrev_co_u32, v_add_co_ci_u32, etc. only use 2 sources
      if op_name in ('v_add_co_u32', 'v_sub_co_u32', 'v_subrev_co_u32', 'v_add_co_ci_u32', 'v_sub_co_ci_u32', 'v_subrev_co_ci_u32'):
        return f"{op_name} {dst_str}, {sdst_str}, {src0_str}, {src1_str}"
      # v_div_scale uses 3 sources
      return f"{op_name} {dst_str}, {sdst_str}, {src0_str}, {src1_str}, {src2_str}" + omod_str

    vdst = unwrap(inst._values.get('vdst', 0))
    src0, src1, src2 = [unwrap(inst._values.get(f, 0)) for f in ('src0', 'src1', 'src2')]
    neg, abs_, clmp = unwrap(inst._values.get('neg', 0)), unwrap(inst._values.get('abs', 0)), unwrap(inst._values.get('clmp', 0))
    opsel = unwrap(inst._values.get('opsel', 0))
    # Check if 64-bit op (needs register pairs)
    is_f64 = 'f64' in op_name or 'i64' in op_name or 'u64' in op_name or 'b64' in op_name
    # v_cmp_class_* has 64-bit src0 but 32-bit src1 (class mask)
    is_class = 'class' in op_name
    # Shift ops: v_*rev_*64 have 32-bit shift amount (src0), 64-bit value (src1)
    is_shift64 = 'rev' in op_name and '64' in op_name and op_name.startswith('v_')
    # v_ldexp_f64: 64-bit src0 (mantissa), 32-bit src1 (exponent)
    is_ldexp64 = op_name == 'v_ldexp_f64'
    # v_trig_preop_f64: 64-bit dst/src0, 32-bit src1 (exponent/scale)
    is_trig_preop = op_name == 'v_trig_preop_f64'
    # v_readlane_b32: destination is SGPR (despite vdst field)
    is_readlane = op_name == 'v_readlane_b32'
    # SAD/QSAD/MQSAD instructions have mixed sizes
    # v_qsad_pk_u16_u8, v_mqsad_pk_u16_u8: 64-bit dst/src0/src2, 32-bit src1
    # v_mqsad_u32_u8: 128-bit (4 reg) dst/src2, 64-bit src0, 32-bit src1
    is_sad64 = any(x in op_name for x in ('qsad_pk', 'mqsad_pk'))
    is_mqsad_u32 = 'mqsad_u32' in op_name
    # Detect 16-bit and 64-bit operand sizes for various instruction patterns
    if 'cvt_pk' in op_name:
      is_f16_dst, is_f16_src, is_f16_src2 = False, op_name.endswith('16'), False
    elif m := re.match(r'v_(?:cvt|frexp_exp)_([a-z0-9_]+)_([a-z0-9]+)', op_name):
      dst_type, src_type = m.group(1), m.group(2)
      is_f16_dst, is_f16_src, is_f16_src2 = _is_16bit(dst_type), _is_16bit(src_type), _is_16bit(src_type)
      is_f64_dst, is_f64_src, is_f64 = '64' in dst_type, '64' in src_type, False
    elif re.match(r'v_mad_[iu]32_[iu]16', op_name):
      is_f16_dst, is_f16_src, is_f16_src2 = False, True, False  # 32-bit dst, 16-bit src0/src1, 32-bit src2
    elif 'pack_b32' in op_name:
      is_f16_dst, is_f16_src, is_f16_src2 = False, True, True  # 32-bit dst, 16-bit sources
    else:
      is_16bit_op = any(x in op_name for x in _16BIT_TYPES) and not any(x in op_name for x in ('dot2', 'pk_', 'sad', 'msad', 'qsad', 'mqsad'))
      is_f16_dst = is_f16_src = is_f16_src2 = is_16bit_op
    def fmt_vop3_src(v, neg_bit, abs_bit, hi_bit=False, reg_cnt=1, is_16=False):
      s = _fmt_src_n(v, reg_cnt) if reg_cnt > 1 else f"v{v - 256}.h" if is_16 and v >= 256 and hi_bit else f"v{v - 256}.l" if is_16 and v >= 256 else fmt_src(v)
      if abs_bit: s = f"|{s}|"
      return f"-{s}" if neg_bit else s
    # Determine register count for each source (check for cvt-specific 64-bit flags first)
    is_src0_64 = locals().get('is_f64_src', is_f64 and not is_shift64) or is_sad64 or is_mqsad_u32
    is_src1_64 = is_f64 and not is_class and not is_ldexp64 and not is_trig_preop
    src0_cnt = 2 if is_src0_64 else 1
    src1_cnt = 2 if is_src1_64 else 1
    src2_cnt = 4 if is_mqsad_u32 else 2 if (is_f64 or is_sad64) else 1
    src0_str = fmt_vop3_src(src0, neg & 1, abs_ & 1, opsel & 1, src0_cnt, is_f16_src)
    src1_str = fmt_vop3_src(src1, neg & 2, abs_ & 2, opsel & 2, src1_cnt, is_f16_src)
    src2_str = fmt_vop3_src(src2, neg & 4, abs_ & 4, opsel & 4, src2_cnt, is_f16_src2)
    # Format destination - for 16-bit ops, use .h/.l suffix; readlane uses SGPR dest
    is_dst_64 = locals().get('is_f64_dst', is_f64) or is_sad64
    dst_cnt = 4 if is_mqsad_u32 else 2 if is_dst_64 else 1
    if is_readlane:
      dst_str = _fmt_sdst(vdst, 1)
    elif dst_cnt > 1:
      dst_str = _vreg(vdst, dst_cnt)
    elif is_f16_dst:
      dst_str = f"v{vdst}.h" if (opsel & 8) else f"v{vdst}.l"
    else:
      dst_str = f"v{vdst}"
    clamp_str = " clamp" if clmp else ""
    omod = unwrap(inst._values.get('omod', 0))
    omod_str = {1: " mul:2", 2: " mul:4", 3: " div:2"}.get(omod, "")
    # op_sel for non-VGPR sources (when opsel bits are set but source is not a VGPR)
    # For 16-bit ops with VGPR sources, opsel is encoded in .h/.l suffix
    # For non-VGPR sources or non-16-bit ops, we need explicit op_sel
    has_nonvgpr_opsel = (src0 < 256 and (opsel & 1)) or (src1 < 256 and (opsel & 2)) or (src2 < 256 and (opsel & 4))
    need_opsel = has_nonvgpr_opsel or (opsel and not is_f16_src)
    # Helper to format opsel string based on source count
    def fmt_opsel(num_src):
      if not need_opsel: return ""
      # When dst is .h (for 16-bit ops) and non-VGPR sources have opsel, use all 1s
      if is_f16_dst and (opsel & 8):  # dst is .h
        return f" op_sel:[1,1,1{',1' if num_src == 3 else ''}]"
      # Otherwise output actual opsel values
      if num_src == 3:
        return f" op_sel:[{opsel & 1},{(opsel >> 1) & 1},{(opsel >> 2) & 1},{(opsel >> 3) & 1}]"
      return f" op_sel:[{opsel & 1},{(opsel >> 1) & 1},{(opsel >> 2) & 1}]"
    # Determine number of sources based on opcode range:
    # 0-255: VOPC promoted (comparison, 2 src, sdst)
    # 256-383: VOP2 promoted (2 src)
    # 384-511: VOP1 promoted (1 src)
    # 512+: Native VOP3 (2 or 3 src depending on instruction)
    if op_val < 256:  # VOPC promoted
      # VOPCX (v_cmpx_*) writes to exec, no explicit destination
      if op_name.startswith('v_cmpx'):
        return f"{op_name}_e64 {src0_str}, {src1_str}"
      return f"{op_name}_e64 {_fmt_sdst(vdst, 1)}, {src0_str}, {src1_str}"
    elif op_val < 384:  # VOP2 promoted
      # v_cndmask_b32 in VOP3 format has 3 sources (src2 is mask selector)
      if 'cndmask' in op_name:
        return f"{op_name}_e64 {dst_str}, {src0_str}, {src1_str}, {src2_str}" + fmt_opsel(3) + clamp_str + omod_str
      return f"{op_name}_e64 {dst_str}, {src0_str}, {src1_str}" + fmt_opsel(2) + clamp_str + omod_str
    elif op_val < 512:  # VOP1 promoted
      if op_name in ('v_nop', 'v_pipeflush'): return f"{op_name}_e64"
      return f"{op_name}_e64 {dst_str}, {src0_str}" + fmt_opsel(1) + clamp_str + omod_str
    else:  # Native VOP3 - determine 2 vs 3 sources based on instruction name
      # 3-source ops: fma, mad, min3, max3, med3, div_fixup, div_fmas, sad, msad, qsad, mqsad, lerp, alignbit/byte, cubeid/sc/tc/ma, bfe, bfi, perm_b32, permlane, cndmask
      # Note: v_writelane_b32 is 2-src (src0, src1 with vdst as 3rd operand - read-modify-write)
      is_3src = any(x in op_name for x in ('fma', 'mad', 'min3', 'max3', 'med3', 'div_fix', 'div_fmas', 'sad', 'lerp', 'align', 'cube',
                                            'bfe', 'bfi', 'perm_b32', 'permlane', 'cndmask', 'xor3', 'or3', 'add3', 'lshl_or', 'and_or', 'lshl_add',
                                            'add_lshl', 'xad', 'maxmin', 'minmax', 'dot2', 'cvt_pk_u8', 'mullit'))
      if is_3src:
        return f"{op_name} {dst_str}, {src0_str}, {src1_str}, {src2_str}" + fmt_opsel(3) + clamp_str + omod_str
      return f"{op_name} {dst_str}, {src0_str}, {src1_str}" + fmt_opsel(2) + clamp_str + omod_str

  # VOP3SD: 3-source with scalar destination (v_div_scale_*, v_add_co_u32, v_mad_*64_*32, etc.)
  if cls_name == 'VOP3SD':
    vdst, sdst = unwrap(inst._values.get('vdst', 0)), unwrap(inst._values.get('sdst', 0))
    src0, src1, src2 = [unwrap(inst._values.get(f, 0)) for f in ('src0', 'src1', 'src2')]
    neg, omod, clmp = unwrap(inst._values.get('neg', 0)), unwrap(inst._values.get('omod', 0)), unwrap(inst._values.get('clmp', 0))
    is_f64, is_mad64 = 'f64' in op_name, 'mad_i64_i32' in op_name or 'mad_u64_u32' in op_name
    def fmt_neg(v, neg_bit, is_64=False): return f"-{_fmt_src64(v) if (is_64 or is_f64) else fmt_src(v)}" if neg_bit else _fmt_src64(v) if (is_64 or is_f64) else fmt_src(v)
    srcs = [fmt_neg(src0, neg & 1), fmt_neg(src1, neg & 2), fmt_neg(src2, neg & 4, is_mad64)]
    dst_str, sdst_str = _vreg(vdst, 2) if (is_f64 or is_mad64) else f"v{vdst}", _fmt_sdst(sdst, 1)
    clamp_str, omod_str = " clamp" if clmp else "", {1: " mul:2", 2: " mul:4", 3: " div:2"}.get(omod, "")
    is_2src = op_name in ('v_add_co_u32', 'v_sub_co_u32', 'v_subrev_co_u32')
    suffix = "_e64" if op_name.startswith('v_') and 'co_' in op_name else ""
    return f"{op_name}{suffix} {dst_str}, {sdst_str}, {', '.join(srcs[:2] if is_2src else srcs)}" + clamp_str + omod_str

  # VOPD: dual-issue instructions
  if cls_name == 'VOPD':
    from extra.assembly.rdna3 import autogen
    opx, opy, vdstx, vdsty_enc = [unwrap(inst._values.get(f, 0)) for f in ('opx', 'opy', 'vdstx', 'vdsty')]
    srcx0, vsrcx1, srcy0, vsrcy1 = [unwrap(inst._values.get(f, 0)) for f in ('srcx0', 'vsrcx1', 'srcy0', 'vsrcy1')]
    vdsty = (vdsty_enc << 1) | ((vdstx & 1) ^ 1)  # Decode vdsty
    def fmt_vopd(op, vdst, src0, vsrc1):
      try: name = autogen.VOPDOp(op).name.lower()
      except (ValueError, KeyError): name = f"op_{op}"
      return f"{name} v{vdst}, {fmt_src(src0)}" if 'mov' in name else f"{name} v{vdst}, {fmt_src(src0)}, v{vsrc1}"
    return f"{fmt_vopd(opx, vdstx, srcx0, vsrcx1)} :: {fmt_vopd(opy, vdsty, srcy0, vsrcy1)}"

  # VOP3P: packed vector ops
  if cls_name == 'VOP3P':
    vdst, clmp = unwrap(inst._values.get('vdst', 0)), unwrap(inst._values.get('clmp', 0))
    src0, src1, src2 = [unwrap(inst._values.get(f, 0)) for f in ('src0', 'src1', 'src2')]
    neg, neg_hi = unwrap(inst._values.get('neg', 0)), unwrap(inst._values.get('neg_hi', 0))
    opsel, opsel_hi, opsel_hi2 = unwrap(inst._values.get('opsel', 0)), unwrap(inst._values.get('opsel_hi', 0)), unwrap(inst._values.get('opsel_hi2', 0))
    is_wmma, is_3src = 'wmma' in op_name, any(x in op_name for x in ('fma', 'mad', 'dot', 'wmma'))
    def fmt_bits(name, val, n): return f"{name}:[{','.join(str((val >> i) & 1) for i in range(n))}]"
    # WMMA: f16/bf16 use 8-reg sources, iu8 uses 4-reg, iu4 uses 2-reg; all have 8-reg dst
    if is_wmma:
      src_cnt = 2 if 'iu4' in op_name else 4 if 'iu8' in op_name else 8
      src0_str, src1_str, src2_str = _fmt_src_n(src0, src_cnt), _fmt_src_n(src1, src_cnt), _fmt_src_n(src2, 8)
      dst_str = _vreg(vdst, 8)
    else:
      src0_str, src1_str, src2_str = _fmt_src_n(src0, 1), _fmt_src_n(src1, 1), _fmt_src_n(src2, 1)
      dst_str = f"v{vdst}"
    n = 3 if is_3src else 2
    full_opsel_hi = opsel_hi | (opsel_hi2 << 2)
    mods = [fmt_bits("op_sel", opsel, n)] if opsel else []
    if full_opsel_hi != (0b111 if is_3src else 0b11): mods.append(fmt_bits("op_sel_hi", full_opsel_hi, n))
    if neg: mods.append(fmt_bits("neg_lo", neg, n))
    if neg_hi: mods.append(fmt_bits("neg_hi", neg_hi, n))
    if clmp: mods.append("clamp")
    mod_str = " " + " ".join(mods) if mods else ""
    return f"{op_name} {dst_str}, {src0_str}, {src1_str}, {src2_str}{mod_str}" if is_3src else f"{op_name} {dst_str}, {src0_str}, {src1_str}{mod_str}"

  # VINTERP: interpolation instructions
  if cls_name == 'VINTERP':
    vdst = unwrap(inst._values.get('vdst', 0))
    src0, src1, src2 = [unwrap(inst._values.get(f, 0)) for f in ('src0', 'src1', 'src2')]
    neg, waitexp, clmp = unwrap(inst._values.get('neg', 0)), unwrap(inst._values.get('waitexp', 0)), unwrap(inst._values.get('clmp', 0))
    def fmt_neg_vi(v, neg_bit): return f"-{v}" if neg_bit else v
    srcs = [fmt_neg_vi(f"v{s - 256}" if s >= 256 else fmt_src(s), neg & (1 << i)) for i, s in enumerate([src0, src1, src2])]
    mods = [m for m in [f"wait_exp:{waitexp}" if waitexp else "", "clamp" if clmp else ""] if m]
    return f"{op_name} v{vdst}, {', '.join(srcs)}" + (" " + " ".join(mods) if mods else "")

  # MUBUF/MTBUF helpers
  def _buf_vaddr(vaddr, offen, idxen): return _vreg(vaddr, 2) if offen and idxen else f"v{vaddr}" if offen or idxen else "off"
  def _buf_srsrc(srsrc): srsrc_base = srsrc * 4; return _reg("ttmp", srsrc_base - 108, 4) if 108 <= srsrc_base <= 123 else _sreg(srsrc_base, 4)

  # MUBUF: buffer load/store
  if cls_name == 'MUBUF':
    vdata, vaddr, srsrc, soffset = [unwrap(inst._values.get(f, 0)) for f in ('vdata', 'vaddr', 'srsrc', 'soffset')]
    offset, offen, idxen = unwrap(inst._values.get('offset', 0)), unwrap(inst._values.get('offen', 0)), unwrap(inst._values.get('idxen', 0))
    glc, dlc, slc, tfe = [unwrap(inst._values.get(f, 0)) for f in ('glc', 'dlc', 'slc', 'tfe')]
    if op_name in ('buffer_gl0_inv', 'buffer_gl1_inv'): return op_name
    # Determine data width from op name
    if 'd16' in op_name: width = 2 if any(x in op_name for x in ('xyz', 'xyzw')) else 1
    elif 'atomic' in op_name:
      base_width = 2 if any(x in op_name for x in ('b64', 'u64', 'i64')) else 1
      width = base_width * 2 if 'cmpswap' in op_name else base_width
    else: width = {'b32':1, 'b64':2, 'b96':3, 'b128':4, 'b16':1, 'x':1, 'xy':2, 'xyz':3, 'xyzw':4}.get(op_name.split('_')[-1], 1)
    if tfe: width += 1
    mods = [m for m in ["offen" if offen else "", "idxen" if idxen else "", f"offset:{offset}" if offset else "",
                        "glc" if glc else "", "dlc" if dlc else "", "slc" if slc else "", "tfe" if tfe else ""] if m]
    return f"{op_name} {_vreg(vdata, width)}, {_buf_vaddr(vaddr, offen, idxen)}, {_buf_srsrc(srsrc)}, {decode_src(soffset)}" + (" " + " ".join(mods) if mods else "")

  # MTBUF: typed buffer load/store
  if cls_name == 'MTBUF':
    vdata, vaddr, srsrc, soffset = [unwrap(inst._values.get(f, 0)) for f in ('vdata', 'vaddr', 'srsrc', 'soffset')]
    offset, tbuf_fmt, offen, idxen = [unwrap(inst._values.get(f, 0)) for f in ('offset', 'format', 'offen', 'idxen')]
    glc, dlc, slc = [unwrap(inst._values.get(f, 0)) for f in ('glc', 'dlc', 'slc')]
    mods = [f"format:{tbuf_fmt}"] + [m for m in ["idxen" if idxen else "", "offen" if offen else "", f"offset:{offset}" if offset else "",
                                                  "glc" if glc else "", "dlc" if dlc else "", "slc" if slc else ""] if m]
    width = 2 if 'd16' in op_name and any(x in op_name for x in ('xyz', 'xyzw')) else 1 if 'd16' in op_name else {'x':1, 'xy':2, 'xyz':3, 'xyzw':4}.get(op_name.split('_')[-1], 1)
    return f"{op_name} {_vreg(vdata, width)}, {_buf_vaddr(vaddr, offen, idxen)}, {_buf_srsrc(srsrc)}, {decode_src(soffset)} {' '.join(mods)}"

  # SOP1/SOP2/SOPC/SOPK
  if cls_name in ('SOP1', 'SOP2', 'SOPC', 'SOPK'):
    sizes = _parse_sop_sizes(op_name)
    dst_cnt, src0_cnt = sizes[0], sizes[1]
    src1_cnt = sizes[2] if len(sizes) > 2 else src0_cnt
    if cls_name == 'SOP1':
      sdst, ssrc0 = unwrap(inst._values.get('sdst', 0)), unwrap(inst._values.get('ssrc0', 0))
      if op_name == 's_getpc_b64': return f"{op_name} {_fmt_sdst(sdst, 2)}"
      if op_name in ('s_setpc_b64', 's_rfe_b64'): return f"{op_name} {_fmt_ssrc(ssrc0, 2)}"
      if op_name == 's_swappc_b64': return f"{op_name} {_fmt_sdst(sdst, 2)}, {_fmt_ssrc(ssrc0, 2)}"
      if op_name in ('s_sendmsg_rtn_b32', 's_sendmsg_rtn_b64'):
        return f"{op_name} {_fmt_sdst(sdst, 2 if 'b64' in op_name else 1)}, sendmsg({MSG_NAMES.get(ssrc0, str(ssrc0))})"
      return f"{op_name} {_fmt_sdst(sdst, dst_cnt)}, {_fmt_ssrc(ssrc0, src0_cnt)}"
    if cls_name == 'SOP2':
      sdst, ssrc0, ssrc1 = [unwrap(inst._values.get(f, 0)) for f in ('sdst', 'ssrc0', 'ssrc1')]
      return f"{op_name} {_fmt_sdst(sdst, dst_cnt)}, {_fmt_ssrc(ssrc0, src0_cnt)}, {_fmt_ssrc(ssrc1, src1_cnt)}"
    if cls_name == 'SOPC':
      return f"{op_name} {_fmt_ssrc(unwrap(inst._values.get('ssrc0', 0)), src0_cnt)}, {_fmt_ssrc(unwrap(inst._values.get('ssrc1', 0)), src1_cnt)}"
    if cls_name == 'SOPK':
      sdst, simm16 = unwrap(inst._values.get('sdst', 0)), unwrap(inst._values.get('simm16', 0))
      if op_name == 's_version': return f"{op_name} 0x{simm16:x}"
      if op_name in ('s_setreg_b32', 's_getreg_b32'):
        hwreg_id, hwreg_offset, hwreg_size = simm16 & 0x3f, (simm16 >> 6) & 0x1f, ((simm16 >> 11) & 0x1f) + 1
        hwreg_str = f"0x{simm16:x}" if hwreg_id in (16, 17) else f"hwreg({HWREG_NAMES.get(hwreg_id, str(hwreg_id))}, {hwreg_offset}, {hwreg_size})"
        return f"{op_name} {hwreg_str}, {_fmt_sdst(sdst, 1)}" if op_name == 's_setreg_b32' else f"{op_name} {_fmt_sdst(sdst, 1)}, {hwreg_str}"
      return f"{op_name} {_fmt_sdst(sdst, dst_cnt)}, 0x{simm16:x}"

  # Generic fallback
  def fmt_field(n, v):
    v = unwrap(v)
    if n in SRC_FIELDS: return fmt_src(v) if v != 255 else "0xff"
    if n in ('sdst', 'vdst'): return f"{'s' if n == 'sdst' else 'v'}{v}"
    return f"v{v}" if n == 'vsrc1' else f"0x{v:x}" if n == 'simm16' else str(v)
  ops = [fmt_field(n, inst._values.get(n, 0)) for n in inst._fields if n not in ('encoding', 'op')]
  return f"{op_name} {', '.join(ops)}" if ops else op_name

# Assembler
SPECIAL_REGS = {'vcc_lo': RawImm(106), 'vcc_hi': RawImm(107), 'null': RawImm(124), 'off': RawImm(124), 'm0': RawImm(125), 'exec_lo': RawImm(126), 'exec_hi': RawImm(127), 'scc': RawImm(253)}
FLOAT_CONSTS = {'0.5': 0.5, '-0.5': -0.5, '1.0': 1.0, '-1.0': -1.0, '2.0': 2.0, '-2.0': -2.0, '4.0': 4.0, '-4.0': -4.0}
REG_MAP: dict[str, _RegFactory] = {'s': s, 'v': v, 't': ttmp, 'ttmp': ttmp}

def parse_operand(op: str) -> tuple:
  op = op.strip().lower()
  neg = op.startswith('-') and not op[1:2].isdigit(); op = op[1:] if neg else op
  abs_ = op.startswith('|') and op.endswith('|') or op.startswith('abs(') and op.endswith(')')
  op = op[1:-1] if op.startswith('|') else op[4:-1] if op.startswith('abs(') else op
  hi_half = op.endswith('.h')
  op = re.sub(r'\.[lh]$', '', op)
  if op in FLOAT_CONSTS: return (FLOAT_CONSTS[op], neg, abs_, hi_half)
  if re.match(r'^-?\d+$', op): return (int(op), neg, abs_, hi_half)
  if m := re.match(r'^-?0x([0-9a-f]+)$', op):
    v = -int(m.group(1), 16) if op.startswith('-') else int(m.group(1), 16)
    return (v, neg, abs_, hi_half)
  if op in SPECIAL_REGS: return (SPECIAL_REGS[op], neg, abs_, hi_half)
  if m := re.match(r'^([svt](?:tmp)?)\[(\d+):(\d+)\]$', op): return (REG_MAP[m.group(1)][int(m.group(2)):int(m.group(3))+1], neg, abs_, hi_half)
  if m := re.match(r'^([svt](?:tmp)?)(\d+)$', op):
    reg = REG_MAP[m.group(1)][int(m.group(2))]
    reg.hi = hi_half
    return (reg, neg, abs_, hi_half)
  # hwreg(name, offset, size) or hwreg(name) -> simm16 encoding
  if m := re.match(r'^hwreg\((\w+)(?:,\s*(\d+),\s*(\d+))?\)$', op):
    name_str = m.group(1).lower()
    hwreg_id = HWREG_IDS.get(name_str, int(name_str) if name_str.isdigit() else None)
    if hwreg_id is None: raise ValueError(f"unknown hwreg name: {name_str}")
    offset, size = int(m.group(2)) if m.group(2) else 0, int(m.group(3)) if m.group(3) else 32
    return (((size - 1) << 11) | (offset << 6) | hwreg_id, neg, abs_, hi_half)
  raise ValueError(f"cannot parse operand: {op}")

SMEM_OPS = {'s_load_b32', 's_load_b64', 's_load_b128', 's_load_b256', 's_load_b512',
            's_buffer_load_b32', 's_buffer_load_b64', 's_buffer_load_b128', 's_buffer_load_b256', 's_buffer_load_b512'}
SOP1_SRC_ONLY = {'s_setpc_b64', 's_rfe_b64'}
SOP1_MSG_IMM = {'s_sendmsg_rtn_b32', 's_sendmsg_rtn_b64'}
SOPK_IMM_ONLY = {'s_version'}
SOPK_IMM_FIRST = {'s_setreg_b32'}
SOPK_UNSUPPORTED = {'s_setreg_imm32_b32'}

def asm(text: str) -> Inst:
  from extra.assembly.rdna3 import autogen
  text = text.strip()
  clamp = 'clamp' in text.lower()
  if clamp: text = re.sub(r'\s+clamp\s*$', '', text, flags=re.I)
  modifiers = {}
  if m := re.search(r'\s+wait_exp:(\d+)', text, re.I): modifiers['waitexp'] = int(m.group(1)); text = text[:m.start()] + text[m.end():]
  parts = text.replace(',', ' ').split()
  if not parts: raise ValueError("empty instruction")
  mnemonic, op_str = parts[0].lower(), text[len(parts[0]):].strip()
  # Handle s_waitcnt specially before operand parsing
  if mnemonic == 's_waitcnt':
    vmcnt, expcnt, lgkmcnt = 0x3f, 0x7, 0x3f
    for part in op_str.replace(',', ' ').split():
      if m := re.match(r'vmcnt\((\d+)\)', part): vmcnt = int(m.group(1))
      elif m := re.match(r'expcnt\((\d+)\)', part): expcnt = int(m.group(1))
      elif m := re.match(r'lgkmcnt\((\d+)\)', part): lgkmcnt = int(m.group(1))
      elif re.match(r'^0x[0-9a-f]+$|^\d+$', part): return autogen.s_waitcnt(simm16=int(part, 0))
    return autogen.s_waitcnt(simm16=waitcnt(vmcnt, expcnt, lgkmcnt))
  # Handle VOPD dual-issue instructions: opx dst, src :: opy dst, src
  if '::' in text:
    x_part, y_part = text.split('::')
    x_parts, y_parts = x_part.strip().replace(',', ' ').split(), y_part.strip().replace(',', ' ').split()
    opx_name, opy_name = x_parts[0].upper(), y_parts[0].upper()
    opx, opy = autogen.VOPDOp[opx_name], autogen.VOPDOp[opy_name]
    x_ops, y_ops = [parse_operand(p)[0] for p in x_parts[1:]], [parse_operand(p)[0] for p in y_parts[1:]]
    vdstx, srcx0 = x_ops[0], x_ops[1] if len(x_ops) > 1 else 0
    vsrcx1 = x_ops[2] if len(x_ops) > 2 else VGPR(0)
    vdsty, srcy0 = y_ops[0], y_ops[1] if len(y_ops) > 1 else 0
    vsrcy1 = y_ops[2] if len(y_ops) > 2 else VGPR(0)
    # Handle fmaak/fmamk literals (4th operand on x or y side)
    lit = None
    if 'fmaak' in opx_name.lower() and len(x_ops) > 3: lit = unwrap(x_ops[3])
    elif 'fmamk' in opx_name.lower() and len(x_ops) > 3: lit, vsrcx1 = unwrap(x_ops[2]), x_ops[3]
    elif 'fmaak' in opy_name.lower() and len(y_ops) > 3: lit = unwrap(y_ops[3])
    elif 'fmamk' in opy_name.lower() and len(y_ops) > 3: lit, vsrcy1 = unwrap(y_ops[2]), y_ops[3]
    return autogen.VOPD(opx, opy, vdstx=vdstx, vdsty=vdsty, srcx0=srcx0, vsrcx1=vsrcx1, srcy0=srcy0, vsrcy1=vsrcy1, literal=lit)
  operands, current, depth, in_pipe = [], "", 0, False
  for ch in op_str:
    if ch in '[(': depth += 1
    elif ch in '])': depth -= 1
    elif ch == '|': in_pipe = not in_pipe
    if ch == ',' and depth == 0 and not in_pipe: operands.append(current.strip()); current = ""
    else: current += ch
  if current.strip(): operands.append(current.strip())
  parsed = [parse_operand(op) for op in operands]
  values = [p[0] for p in parsed]
  neg_bits = sum((1 << (i-1)) for i, p in enumerate(parsed) if i > 0 and p[1])
  abs_bits = sum((1 << (i-1)) for i, p in enumerate(parsed) if i > 0 and p[2])
  opsel_bits = (8 if len(parsed) > 0 and parsed[0][3] else 0) | sum((1 << i) for i, p in enumerate(parsed[1:4]) if p[3])
  lit = None
  if mnemonic in ('v_fmaak_f32', 'v_fmaak_f16') and len(values) == 4: lit, values = unwrap(values[3]), values[:3]
  elif mnemonic in ('v_fmamk_f32', 'v_fmamk_f16') and len(values) == 4: lit, values = unwrap(values[2]), [values[0], values[1], values[3]]
  vcc_ops = {'v_add_co_ci_u32', 'v_sub_co_ci_u32', 'v_subrev_co_ci_u32', 'v_add_co_u32', 'v_sub_co_u32', 'v_subrev_co_u32'}
  if mnemonic.replace('_e32', '') in vcc_ops and len(values) >= 5: values = [values[0], values[2], values[3]]
  if mnemonic.startswith('v_cmp') and len(values) >= 3 and operands[0].strip().lower() in ('vcc_lo', 'vcc_hi', 'vcc'):
    values = values[1:]
  # CMPX instructions with _e64 suffix: prepend implicit EXEC_LO destination (vdst=126)
  if 'cmpx' in mnemonic and mnemonic.endswith('_e64') and len(values) == 2:
    values = [VGPR(126, 1)] + values
    # Recalculate modifiers: parsed[0]=src0, parsed[1]=src1 (no vdst in user input)
    neg_bits = sum((1 << i) for i, p in enumerate(parsed[:3]) if p[1])
    abs_bits = sum((1 << i) for i, p in enumerate(parsed[:3]) if p[2])
    opsel_bits = sum((1 << i) for i, p in enumerate(parsed[:2]) if p[3])
  vop3sd_ops = {'v_div_scale_f32', 'v_div_scale_f64'}
  if mnemonic in vop3sd_ops and len(parsed) >= 5:
    neg_bits = sum((1 << i) for i, p in enumerate(parsed[2:5]) if p[1])
    abs_bits = sum((1 << i) for i, p in enumerate(parsed[2:5]) if p[2])
  if mnemonic in SOPK_UNSUPPORTED: raise ValueError(f"unsupported instruction: {mnemonic}")
  elif mnemonic in SOP1_SRC_ONLY:
    return getattr(autogen, mnemonic)(ssrc0=values[0])
  elif mnemonic in SOP1_MSG_IMM:
    return getattr(autogen, mnemonic)(sdst=values[0], ssrc0=RawImm(unwrap(values[1])))
  elif mnemonic in SOPK_IMM_ONLY:
    return getattr(autogen, mnemonic)(simm16=values[0])
  elif mnemonic in SOPK_IMM_FIRST:
    return getattr(autogen, mnemonic)(simm16=values[0], sdst=values[1])
  elif mnemonic in SMEM_OPS and len(operands) >= 3 and re.match(r'^-?[0-9]|^-?0x', operands[2].strip().lower()):
    return getattr(autogen, mnemonic)(sdata=values[0], sbase=values[1], offset=values[2], soffset=RawImm(124))
  elif mnemonic.startswith('buffer_') and len(operands) >= 2 and operands[1].strip().lower() == 'off':
    return getattr(autogen, mnemonic)(vdata=values[0], vaddr=0, srsrc=values[2], soffset=RawImm(unwrap(values[3])) if len(values) > 3 else RawImm(0))
  elif (mnemonic.startswith('flat_load') or mnemonic.startswith('global_load') or mnemonic.startswith('scratch_load')) and len(values) >= 3:
    offset = int(m.group(1)) if (m := re.search(r'offset:(-?\d+)', op_str)) else 0
    return getattr(autogen, mnemonic)(vdst=values[0], addr=values[1], saddr=values[2], offset=offset)
  elif (mnemonic.startswith('flat_store') or mnemonic.startswith('global_store') or mnemonic.startswith('scratch_store')) and len(values) >= 3:
    offset = int(m.group(1)) if (m := re.search(r'offset:(-?\d+)', op_str)) else 0
    return getattr(autogen, mnemonic)(addr=values[0], data=values[1], saddr=values[2], offset=offset)
  for suffix in (['_e32', ''] if not (neg_bits or abs_bits or clamp) else ['', '_e32']):
    if hasattr(autogen, name := mnemonic.replace('.', '_') + suffix):
      use_opsel = 'opsel' in getattr(autogen, name).func._fields
      vals = [type(v)(v.idx, v.count, False) if isinstance(v, Reg) and v.hi and use_opsel else v for v in values]
      inst = getattr(autogen, name)(*vals, literal=lit, **modifiers)
      if neg_bits and 'neg' in inst._fields: inst._values['neg'] = neg_bits
      if opsel_bits and use_opsel: inst._values['opsel'] = opsel_bits
      if abs_bits and 'abs' in inst._fields: inst._values['abs'] = abs_bits
      if clamp and 'clmp' in inst._fields: inst._values['clmp'] = 1
      return inst
  raise ValueError(f"unknown instruction: {mnemonic}")
