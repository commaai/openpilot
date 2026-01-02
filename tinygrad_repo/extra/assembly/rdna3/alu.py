# Pure combinational ALU functions for RDNA3 emulation
from __future__ import annotations
import struct, math
from typing import Callable
from extra.assembly.rdna3.autogen import SOP1Op, SOP2Op, SOPCOp, SOPKOp, VOP1Op, VOP2Op, VOP3Op

# Format base offsets for unified opcode space
SOP2_BASE, SOP1_BASE, SOPC_BASE, SOPK_BASE = 0x000, 0x100, 0x200, 0x300
VOP2_BASE, VOP1_BASE = 0x100, 0x180

# Float conversion helpers
_I, _f, _H, _e = struct.Struct('<I'), struct.Struct('<f'), struct.Struct('<H'), struct.Struct('<e')
def f32(i: int) -> float: return _f.unpack(_I.pack(i & 0xffffffff))[0]
def i32(f: float) -> int:
  if math.isinf(f): return 0x7f800000 if f > 0 else 0xff800000
  try: return _I.unpack(_f.pack(f))[0]
  except (OverflowError, struct.error): return 0x7f800000 if f > 0 else 0xff800000
def f16(i: int) -> float: return _e.unpack(_H.pack(i & 0xffff))[0]
def i16(f: float) -> int:
  if math.isinf(f): return 0x7c00 if f > 0 else 0xfc00
  try: return _H.unpack(_e.pack(f))[0]
  except (OverflowError, struct.error): return 0x7c00 if f > 0 else 0xfc00
def sext(v: int, b: int) -> int: return v - (1 << b) if v & (1 << (b-1)) else v
def clz(x: int) -> int: return 32 - x.bit_length() if x else 32
def cls(x: int) -> int: x &= 0xffffffff; return 31 if x in (0, 0xffffffff) else clz(~x & 0xffffffff if x >> 31 else x) - 1
def _cvt_i32_f32(v): return (0x7fffffff if v > 0 else 0x80000000) if math.isinf(v) else (0 if math.isnan(v) else max(-0x80000000, min(0x7fffffff, int(v))) & 0xffffffff)
def _cvt_u32_f32(v): return (0xffffffff if v > 0 else 0) if math.isinf(v) else (0 if math.isnan(v) or v < 0 else min(0xffffffff, int(v)))

# SALU: op -> fn(s0, s1, scc_in) -> (result, scc_out)
SALU: dict[int, Callable] = {
  # SOP2
  SOP2_BASE + SOP2Op.S_ADD_U32: lambda a, b, scc: ((a + b) & 0xffffffff, int((a + b) >= 0x100000000)),
  SOP2_BASE + SOP2Op.S_SUB_U32: lambda a, b, scc: ((a - b) & 0xffffffff, int(b > a)),
  SOP2_BASE + SOP2Op.S_ADDC_U32: lambda a, b, scc: ((r := a + b + scc) & 0xffffffff, int(r >= 0x100000000)),
  SOP2_BASE + SOP2Op.S_SUBB_U32: lambda a, b, scc: ((a - b - scc) & 0xffffffff, int((b + scc) > a)),
  SOP2_BASE + SOP2Op.S_ADD_I32: lambda a, b, scc: ((r := sext(a, 32) + sext(b, 32)) & 0xffffffff, int(((a >> 31) == (b >> 31)) and ((a >> 31) != ((r >> 31) & 1)))),
  SOP2_BASE + SOP2Op.S_SUB_I32: lambda a, b, scc: ((r := sext(a, 32) - sext(b, 32)) & 0xffffffff, int(((a >> 31) != (b >> 31)) and ((a >> 31) != ((r >> 31) & 1)))),
  SOP2_BASE + SOP2Op.S_AND_B32: lambda a, b, scc: ((r := a & b), int(r != 0)),
  SOP2_BASE + SOP2Op.S_OR_B32: lambda a, b, scc: ((r := a | b), int(r != 0)),
  SOP2_BASE + SOP2Op.S_XOR_B32: lambda a, b, scc: ((r := a ^ b), int(r != 0)),
  SOP2_BASE + SOP2Op.S_AND_NOT1_B32: lambda a, b, scc: ((r := a & (~b & 0xffffffff)), int(r != 0)),
  SOP2_BASE + SOP2Op.S_OR_NOT1_B32: lambda a, b, scc: ((r := a | (~b & 0xffffffff)), int(r != 0)),
  SOP2_BASE + SOP2Op.S_LSHL_B32: lambda a, b, scc: ((r := (a << (b & 0x1f)) & 0xffffffff), int(r != 0)),
  SOP2_BASE + SOP2Op.S_LSHR_B32: lambda a, b, scc: ((r := a >> (b & 0x1f)), int(r != 0)),
  SOP2_BASE + SOP2Op.S_ASHR_I32: lambda a, b, scc: ((r := sext(a, 32) >> (b & 0x1f)) & 0xffffffff, int(r != 0)),
  SOP2_BASE + SOP2Op.S_MUL_I32: lambda a, b, scc: ((sext(a, 32) * sext(b, 32)) & 0xffffffff, scc),
  SOP2_BASE + SOP2Op.S_MUL_HI_U32: lambda a, b, scc: (((a * b) >> 32) & 0xffffffff, scc),
  SOP2_BASE + SOP2Op.S_MUL_HI_I32: lambda a, b, scc: (((sext(a, 32) * sext(b, 32)) >> 32) & 0xffffffff, scc),
  SOP2_BASE + SOP2Op.S_MIN_I32: lambda a, b, scc: (a, 1) if sext(a, 32) < sext(b, 32) else (b, 0),
  SOP2_BASE + SOP2Op.S_MIN_U32: lambda a, b, scc: (a, 1) if a < b else (b, 0),
  SOP2_BASE + SOP2Op.S_MAX_I32: lambda a, b, scc: (a, 1) if sext(a, 32) > sext(b, 32) else (b, 0),
  SOP2_BASE + SOP2Op.S_MAX_U32: lambda a, b, scc: (a, 1) if a > b else (b, 0),
  SOP2_BASE + SOP2Op.S_CSELECT_B32: lambda a, b, scc: (a if scc else b, scc),
  SOP2_BASE + SOP2Op.S_BFE_U32: lambda a, b, scc: ((r := ((a >> (b & 0x1f)) & ((1 << ((b >> 16) & 0x7f)) - 1)) if (b >> 16) & 0x7f else 0), int(r != 0)),
  SOP2_BASE + SOP2Op.S_BFE_I32: lambda a, b, scc: ((r := sext((a >> (b & 0x1f)) & ((1 << w) - 1), w) & 0xffffffff if (w := (b >> 16) & 0x7f) else 0), int(r != 0)),
  SOP2_BASE + SOP2Op.S_PACK_LL_B32_B16: lambda a, b, scc: ((a & 0xffff) | ((b & 0xffff) << 16), scc),
  SOP2_BASE + SOP2Op.S_PACK_LH_B32_B16: lambda a, b, scc: ((a & 0xffff) | (b & 0xffff0000), scc),
  SOP2_BASE + SOP2Op.S_PACK_HH_B32_B16: lambda a, b, scc: (((a >> 16) & 0xffff) | (b & 0xffff0000), scc),
  SOP2_BASE + SOP2Op.S_PACK_HL_B32_B16: lambda a, b, scc: (((a >> 16) & 0xffff) | ((b & 0xffff) << 16), scc),
  SOP2_BASE + SOP2Op.S_ADD_F32: lambda a, b, scc: (i32(f32(a) + f32(b)), scc),
  SOP2_BASE + SOP2Op.S_SUB_F32: lambda a, b, scc: (i32(f32(a) - f32(b)), scc),
  SOP2_BASE + SOP2Op.S_MUL_F32: lambda a, b, scc: (i32(f32(a) * f32(b)), scc),
  # SOP1
  SOP1_BASE + SOP1Op.S_MOV_B32: lambda a, b, scc: (a, scc),
  SOP1_BASE + SOP1Op.S_NOT_B32: lambda a, b, scc: ((r := (~a) & 0xffffffff), int(r != 0)),
  SOP1_BASE + SOP1Op.S_BREV_B32: lambda a, b, scc: (int(f'{a & 0xffffffff:032b}'[::-1], 2), scc),
  SOP1_BASE + SOP1Op.S_CLZ_I32_U32: lambda a, b, scc: (clz(a), scc),
  SOP1_BASE + SOP1Op.S_CLS_I32: lambda a, b, scc: (cls(a), scc),
  SOP1_BASE + SOP1Op.S_SEXT_I32_I8: lambda a, b, scc: (sext(a & 0xff, 8) & 0xffffffff, scc),
  SOP1_BASE + SOP1Op.S_SEXT_I32_I16: lambda a, b, scc: (sext(a & 0xffff, 16) & 0xffffffff, scc),
  SOP1_BASE + SOP1Op.S_ABS_I32: lambda a, b, scc: ((r := abs(sext(a, 32)) & 0xffffffff), int(r != 0)),
  SOP1_BASE + SOP1Op.S_CVT_F32_I32: lambda a, b, scc: (i32(float(sext(a, 32))), scc),
  SOP1_BASE + SOP1Op.S_CVT_F32_U32: lambda a, b, scc: (i32(float(a)), scc),
  SOP1_BASE + SOP1Op.S_CVT_I32_F32: lambda a, b, scc: (_cvt_i32_f32(f32(a)), scc),
  SOP1_BASE + SOP1Op.S_CVT_U32_F32: lambda a, b, scc: (_cvt_u32_f32(f32(a)), scc),
  SOP1_BASE + SOP1Op.S_CEIL_F32: lambda a, b, scc: (i32(math.ceil(f32(a))), scc),
  SOP1_BASE + SOP1Op.S_FLOOR_F32: lambda a, b, scc: (i32(math.floor(f32(a))), scc),
  SOP1_BASE + SOP1Op.S_TRUNC_F32: lambda a, b, scc: (i32(math.trunc(f32(a))), scc),
  SOP1_BASE + SOP1Op.S_RNDNE_F32: lambda a, b, scc: (i32(round(f32(a))), scc),
  SOP1_BASE + SOP1Op.S_CVT_F16_F32: lambda a, b, scc: (i16(f32(a)), scc),
  SOP1_BASE + SOP1Op.S_CVT_F32_F16: lambda a, b, scc: (i32(f16(a)), scc),
  # SOPC
  SOPC_BASE + SOPCOp.S_CMP_EQ_I32: lambda a, b, scc: (0, int(sext(a, 32) == sext(b, 32))),
  SOPC_BASE + SOPCOp.S_CMP_LG_I32: lambda a, b, scc: (0, int(sext(a, 32) != sext(b, 32))),
  SOPC_BASE + SOPCOp.S_CMP_GT_I32: lambda a, b, scc: (0, int(sext(a, 32) > sext(b, 32))),
  SOPC_BASE + SOPCOp.S_CMP_GE_I32: lambda a, b, scc: (0, int(sext(a, 32) >= sext(b, 32))),
  SOPC_BASE + SOPCOp.S_CMP_LT_I32: lambda a, b, scc: (0, int(sext(a, 32) < sext(b, 32))),
  SOPC_BASE + SOPCOp.S_CMP_LE_I32: lambda a, b, scc: (0, int(sext(a, 32) <= sext(b, 32))),
  SOPC_BASE + SOPCOp.S_CMP_EQ_U32: lambda a, b, scc: (0, int(a == b)),
  SOPC_BASE + SOPCOp.S_CMP_LG_U32: lambda a, b, scc: (0, int(a != b)),
  SOPC_BASE + SOPCOp.S_CMP_GT_U32: lambda a, b, scc: (0, int(a > b)),
  SOPC_BASE + SOPCOp.S_CMP_GE_U32: lambda a, b, scc: (0, int(a >= b)),
  SOPC_BASE + SOPCOp.S_CMP_LT_U32: lambda a, b, scc: (0, int(a < b)),
  SOPC_BASE + SOPCOp.S_CMP_LE_U32: lambda a, b, scc: (0, int(a <= b)),
  SOPC_BASE + SOPCOp.S_BITCMP0_B32: lambda a, b, scc: (0, int((a & (1 << (b & 0x1f))) == 0)),
  SOPC_BASE + SOPCOp.S_BITCMP1_B32: lambda a, b, scc: (0, int((a & (1 << (b & 0x1f))) != 0)),
  # SOPK
  SOPK_BASE + SOPKOp.S_MOVK_I32: lambda a, b, scc: (sext(b, 16) & 0xffffffff, scc),
  SOPK_BASE + SOPKOp.S_CMOVK_I32: lambda a, b, scc: ((sext(b, 16) & 0xffffffff) if scc else a, scc),
  SOPK_BASE + SOPKOp.S_ADDK_I32: lambda a, b, scc: ((r := sext(a, 32) + sext(b, 16)) & 0xffffffff, int(((a >> 31) == ((b >> 15) & 1)) and ((a >> 31) != ((r >> 31) & 1)))),
  SOPK_BASE + SOPKOp.S_MULK_I32: lambda a, b, scc: ((sext(a, 32) * sext(b, 16)) & 0xffffffff, scc),
  SOPK_BASE + SOPKOp.S_CMPK_EQ_I32: lambda a, b, scc: (0, int(sext(a, 32) == sext(b, 16))),
  SOPK_BASE + SOPKOp.S_CMPK_LG_I32: lambda a, b, scc: (0, int(sext(a, 32) != sext(b, 16))),
  SOPK_BASE + SOPKOp.S_CMPK_GT_I32: lambda a, b, scc: (0, int(sext(a, 32) > sext(b, 16))),
  SOPK_BASE + SOPKOp.S_CMPK_GE_I32: lambda a, b, scc: (0, int(sext(a, 32) >= sext(b, 16))),
  SOPK_BASE + SOPKOp.S_CMPK_LT_I32: lambda a, b, scc: (0, int(sext(a, 32) < sext(b, 16))),
  SOPK_BASE + SOPKOp.S_CMPK_LE_I32: lambda a, b, scc: (0, int(sext(a, 32) <= sext(b, 16))),
  SOPK_BASE + SOPKOp.S_CMPK_EQ_U32: lambda a, b, scc: (0, int(a == (b & 0xffff))),
  SOPK_BASE + SOPKOp.S_CMPK_LG_U32: lambda a, b, scc: (0, int(a != (b & 0xffff))),
  SOPK_BASE + SOPKOp.S_CMPK_GT_U32: lambda a, b, scc: (0, int(a > (b & 0xffff))),
  SOPK_BASE + SOPKOp.S_CMPK_GE_U32: lambda a, b, scc: (0, int(a >= (b & 0xffff))),
  SOPK_BASE + SOPKOp.S_CMPK_LT_U32: lambda a, b, scc: (0, int(a < (b & 0xffff))),
  SOPK_BASE + SOPKOp.S_CMPK_LE_U32: lambda a, b, scc: (0, int(a <= (b & 0xffff))),
}

# VALU: op -> fn(s0, s1, s2) -> result
VALU: dict[int, Callable] = {
  # VOP2
  VOP2_BASE + VOP2Op.V_ADD_F32: lambda a, b, c: i32(f32(a) + f32(b)),
  VOP2_BASE + VOP2Op.V_SUB_F32: lambda a, b, c: i32(f32(a) - f32(b)),
  VOP2_BASE + VOP2Op.V_SUBREV_F32: lambda a, b, c: i32(f32(b) - f32(a)),
  VOP2_BASE + VOP2Op.V_MUL_F32: lambda a, b, c: i32(f32(a) * f32(b)),
  VOP2_BASE + VOP2Op.V_MIN_F32: lambda a, b, c: i32(min(f32(a), f32(b))),
  VOP2_BASE + VOP2Op.V_MAX_F32: lambda a, b, c: i32(max(f32(a), f32(b))),
  VOP2_BASE + VOP2Op.V_ADD_NC_U32: lambda a, b, c: (a + b) & 0xffffffff,
  VOP2_BASE + VOP2Op.V_SUB_NC_U32: lambda a, b, c: (a - b) & 0xffffffff,
  VOP2_BASE + VOP2Op.V_SUBREV_NC_U32: lambda a, b, c: (b - a) & 0xffffffff,
  VOP2_BASE + VOP2Op.V_AND_B32: lambda a, b, c: a & b,
  VOP2_BASE + VOP2Op.V_OR_B32: lambda a, b, c: a | b,
  VOP2_BASE + VOP2Op.V_XOR_B32: lambda a, b, c: a ^ b,
  VOP2_BASE + VOP2Op.V_XNOR_B32: lambda a, b, c: (~(a ^ b)) & 0xffffffff,
  VOP2_BASE + VOP2Op.V_LSHLREV_B32: lambda a, b, c: (b << (a & 0x1f)) & 0xffffffff,
  VOP2_BASE + VOP2Op.V_LSHRREV_B32: lambda a, b, c: b >> (a & 0x1f),
  VOP2_BASE + VOP2Op.V_ASHRREV_I32: lambda a, b, c: (sext(b, 32) >> (a & 0x1f)) & 0xffffffff,
  VOP2_BASE + VOP2Op.V_MIN_I32: lambda a, b, c: a if sext(a, 32) < sext(b, 32) else b,
  VOP2_BASE + VOP2Op.V_MAX_I32: lambda a, b, c: a if sext(a, 32) > sext(b, 32) else b,
  VOP2_BASE + VOP2Op.V_MIN_U32: lambda a, b, c: min(a, b),
  VOP2_BASE + VOP2Op.V_MAX_U32: lambda a, b, c: max(a, b),
  VOP2_BASE + VOP2Op.V_MUL_I32_I24: lambda a, b, c: (sext(a & 0xffffff, 24) * sext(b & 0xffffff, 24)) & 0xffffffff,
  VOP2_BASE + VOP2Op.V_MUL_HI_I32_I24: lambda a, b, c: ((sext(a & 0xffffff, 24) * sext(b & 0xffffff, 24)) >> 32) & 0xffffffff,
  VOP2_BASE + VOP2Op.V_MUL_U32_U24: lambda a, b, c: ((a & 0xffffff) * (b & 0xffffff)) & 0xffffffff,
  VOP2_BASE + VOP2Op.V_MUL_HI_U32_U24: lambda a, b, c: (((a & 0xffffff) * (b & 0xffffff)) >> 32) & 0xffffffff,
  VOP2_BASE + VOP2Op.V_CVT_PK_RTZ_F16_F32: lambda a, b, c: i16(f32(a)) | (i16(f32(b)) << 16),
  VOP2_BASE + VOP2Op.V_LDEXP_F16: lambda a, b, c: i16(math.ldexp(f16(a), sext(b, 32))),
  VOP2_BASE + VOP2Op.V_ADD_F16: lambda a, b, c: i16(f16(a) + f16(b)),
  VOP2_BASE + VOP2Op.V_SUB_F16: lambda a, b, c: i16(f16(a) - f16(b)),
  VOP2_BASE + VOP2Op.V_MUL_F16: lambda a, b, c: i16(f16(a) * f16(b)),
  VOP2_BASE + VOP2Op.V_MIN_F16: lambda a, b, c: i16(min(f16(a), f16(b))),
  VOP2_BASE + VOP2Op.V_MAX_F16: lambda a, b, c: i16(max(f16(a), f16(b))),
  # VOP1
  VOP1_BASE + VOP1Op.V_MOV_B32: lambda a, b, c: a,
  VOP1_BASE + VOP1Op.V_NOT_B32: lambda a, b, c: (~a) & 0xffffffff,
  VOP1_BASE + VOP1Op.V_BFREV_B32: lambda a, b, c: int(f'{a & 0xffffffff:032b}'[::-1], 2),
  VOP1_BASE + VOP1Op.V_CLZ_I32_U32: lambda a, b, c: clz(a),
  VOP1_BASE + VOP1Op.V_CLS_I32: lambda a, b, c: cls(a),
  VOP1_BASE + VOP1Op.V_CVT_F32_I32: lambda a, b, c: i32(float(sext(a, 32))),
  VOP1_BASE + VOP1Op.V_CVT_F32_U32: lambda a, b, c: i32(float(a)),
  VOP1_BASE + VOP1Op.V_CVT_I32_F32: lambda a, b, c: _cvt_i32_f32(f32(a)),
  VOP1_BASE + VOP1Op.V_CVT_U32_F32: lambda a, b, c: _cvt_u32_f32(f32(a)),
  VOP1_BASE + VOP1Op.V_CVT_F16_F32: lambda a, b, c: i16(f32(a)),
  VOP1_BASE + VOP1Op.V_CVT_F32_F16: lambda a, b, c: i32(f16(a)),
  VOP1_BASE + VOP1Op.V_RCP_F32: lambda a, b, c: i32(1.0 / f32(a) if f32(a) != 0 else math.copysign(float('inf'), f32(a))),
  VOP1_BASE + VOP1Op.V_RCP_IFLAG_F32: lambda a, b, c: i32(1.0 / f32(a) if f32(a) != 0 else math.copysign(float('inf'), f32(a))),
  VOP1_BASE + VOP1Op.V_RSQ_F32: lambda a, b, c: i32(1.0 / math.sqrt(f32(a)) if f32(a) > 0 else (float('nan') if f32(a) < 0 else float('inf'))),
  VOP1_BASE + VOP1Op.V_SQRT_F32: lambda a, b, c: i32(math.sqrt(f32(a)) if f32(a) >= 0 else float('nan')),
  VOP1_BASE + VOP1Op.V_LOG_F32: lambda a, b, c: i32(math.log2(f32(a)) if f32(a) > 0 else (float('-inf') if f32(a) == 0 else float('nan'))),
  VOP1_BASE + VOP1Op.V_EXP_F32: lambda a, b, c: i32(float('inf') if f32(a) > 128 else (0.0 if f32(a) < -150 else math.pow(2.0, f32(a)))),
  VOP1_BASE + VOP1Op.V_SIN_F32: lambda a, b, c: i32(math.sin(f32(a) * 2 * math.pi)),
  VOP1_BASE + VOP1Op.V_COS_F32: lambda a, b, c: i32(math.cos(f32(a) * 2 * math.pi)),
  VOP1_BASE + VOP1Op.V_FLOOR_F32: lambda a, b, c: i32(math.floor(f32(a))),
  VOP1_BASE + VOP1Op.V_CEIL_F32: lambda a, b, c: i32(math.ceil(f32(a))),
  VOP1_BASE + VOP1Op.V_TRUNC_F32: lambda a, b, c: i32(math.trunc(f32(a))),
  VOP1_BASE + VOP1Op.V_RNDNE_F32: lambda a, b, c: i32(round(f32(a))),
  VOP1_BASE + VOP1Op.V_FRACT_F32: lambda a, b, c: i32((v := f32(a)) - math.floor(v)),
  VOP1_BASE + VOP1Op.V_CVT_F32_UBYTE0: lambda a, b, c: i32(float(a & 0xff)),
  VOP1_BASE + VOP1Op.V_CVT_F32_UBYTE1: lambda a, b, c: i32(float((a >> 8) & 0xff)),
  VOP1_BASE + VOP1Op.V_CVT_F32_UBYTE2: lambda a, b, c: i32(float((a >> 16) & 0xff)),
  VOP1_BASE + VOP1Op.V_CVT_F32_UBYTE3: lambda a, b, c: i32(float((a >> 24) & 0xff)),
  VOP1_BASE + VOP1Op.V_FREXP_MANT_F32: lambda a, b, c: i32(math.frexp(v)[0] if (v := f32(a)) != 0 else 0.0),
  VOP1_BASE + VOP1Op.V_FREXP_EXP_I32_F32: lambda a, b, c: (math.frexp(v)[1] if (v := f32(a)) != 0 else 0) & 0xffffffff,
  # VOP3
  VOP3Op.V_FMA_F32: lambda a, b, c: i32(f32(a) * f32(b) + f32(c)),
  VOP3Op.V_DIV_FMAS_F32: lambda a, b, c: i32(f32(a) * f32(b) + f32(c)),
  VOP3Op.V_ADD3_U32: lambda a, b, c: (a + b + c) & 0xffffffff,
  VOP3Op.V_LSHL_ADD_U32: lambda a, b, c: ((a << (b & 0x1f)) + c) & 0xffffffff,
  VOP3Op.V_ADD_LSHL_U32: lambda a, b, c: ((a + b) << (c & 0x1f)) & 0xffffffff,
  VOP3Op.V_XOR3_B32: lambda a, b, c: a ^ b ^ c,
  VOP3Op.V_OR3_B32: lambda a, b, c: a | b | c,
  VOP3Op.V_AND_OR_B32: lambda a, b, c: (a & b) | c,
  VOP3Op.V_LSHL_OR_B32: lambda a, b, c: ((a << (b & 0x1f)) | c) & 0xffffffff,
  VOP3Op.V_XAD_U32: lambda a, b, c: ((a ^ b) + c) & 0xffffffff,
  VOP3Op.V_MAD_U32_U24: lambda a, b, c: ((a & 0xffffff) * (b & 0xffffff) + c) & 0xffffffff,
  VOP3Op.V_MAD_I32_I24: lambda a, b, c: (sext(a & 0xffffff, 24) * sext(b & 0xffffff, 24) + sext(c, 32)) & 0xffffffff,
  VOP3Op.V_BFE_U32: lambda a, b, c: (a >> (b & 0x1f)) & ((1 << (c & 0x1f)) - 1) if c & 0x1f else 0,
  VOP3Op.V_BFE_I32: lambda a, b, c: sext((a >> (b & 0x1f)) & ((1 << w) - 1), w) & 0xffffffff if (w := c & 0x1f) else 0,
  VOP3Op.V_ALIGNBIT_B32: lambda a, b, c: (((a << 32) | b) >> (c & 0x1f)) & 0xffffffff,
  VOP3Op.V_MUL_LO_U32: lambda a, b, c: (a * b) & 0xffffffff,
  VOP3Op.V_MUL_HI_U32: lambda a, b, c: ((a * b) >> 32) & 0xffffffff,
  VOP3Op.V_MUL_HI_I32: lambda a, b, c: ((sext(a, 32) * sext(b, 32)) >> 32) & 0xffffffff,
  VOP3Op.V_LDEXP_F32: lambda a, b, c: i32(math.ldexp(f32(a), sext(b, 32))),
  VOP3Op.V_DIV_FIXUP_F32: lambda a, b, c: i32(math.copysign(float('inf'), f32(c)) if f32(b) == 0.0 else f32(c) / f32(b)),
  VOP3Op.V_PACK_B32_F16: lambda a, b, c: (a & 0xffff) | ((b & 0xffff) << 16),
  VOP3Op.V_CVT_PK_RTZ_F16_F32: lambda a, b, c: i16(f32(a)) | (i16(f32(b)) << 16),
  VOP3Op.V_LSHLREV_B16: lambda a, b, c: ((b & 0xffff) << (a & 0xf)) & 0xffff,
  VOP3Op.V_LSHRREV_B16: lambda a, b, c: (b & 0xffff) >> (a & 0xf),
  VOP3Op.V_ASHRREV_I16: lambda a, b, c: (sext(b & 0xffff, 16) >> (a & 0xf)) & 0xffff,
  VOP3Op.V_ADD_NC_U16: lambda a, b, c: ((a & 0xffff) + (b & 0xffff)) & 0xffff,
  VOP3Op.V_SUB_NC_U16: lambda a, b, c: ((a & 0xffff) - (b & 0xffff)) & 0xffff,
  VOP3Op.V_MUL_LO_U16: lambda a, b, c: ((a & 0xffff) * (b & 0xffff)) & 0xffff,
  VOP3Op.V_MIN_U16: lambda a, b, c: min(a & 0xffff, b & 0xffff),
  VOP3Op.V_MAX_U16: lambda a, b, c: max(a & 0xffff, b & 0xffff),
  VOP3Op.V_MIN_I16: lambda a, b, c: (a & 0xffff) if sext(a & 0xffff, 16) < sext(b & 0xffff, 16) else (b & 0xffff),
  VOP3Op.V_MAX_I16: lambda a, b, c: (a & 0xffff) if sext(a & 0xffff, 16) > sext(b & 0xffff, 16) else (b & 0xffff),
  VOP3Op.V_MAD_U16: lambda a, b, c: ((a & 0xffff) * (b & 0xffff) + (c & 0xffff)) & 0xffff,
  VOP3Op.V_MAD_I16: lambda a, b, c: (sext(a & 0xffff, 16) * sext(b & 0xffff, 16) + sext(c & 0xffff, 16)) & 0xffff,
  VOP3Op.V_FMA_F16: lambda a, b, c: i16(f16(a) * f16(b) + f16(c)),
  VOP3Op.V_MIN3_I32: lambda a, b, c: sorted([sext(a, 32), sext(b, 32), sext(c, 32)])[0] & 0xffffffff,
  VOP3Op.V_MAX3_I32: lambda a, b, c: sorted([sext(a, 32), sext(b, 32), sext(c, 32)])[2] & 0xffffffff,
  VOP3Op.V_MED3_I32: lambda a, b, c: sorted([sext(a, 32), sext(b, 32), sext(c, 32)])[1] & 0xffffffff,
  VOP3Op.V_MIN3_F16: lambda a, b, c: i16(min(f16(a), f16(b), f16(c))),
  VOP3Op.V_MAX3_F16: lambda a, b, c: i16(max(f16(a), f16(b), f16(c))),
  VOP3Op.V_MED3_F16: lambda a, b, c: i16(sorted([f16(a), f16(b), f16(c)])[1]),
  VOP3Op.V_MIN3_U16: lambda a, b, c: min(a & 0xffff, b & 0xffff, c & 0xffff),
  VOP3Op.V_MAX3_U16: lambda a, b, c: max(a & 0xffff, b & 0xffff, c & 0xffff),
  VOP3Op.V_MED3_U16: lambda a, b, c: sorted([a & 0xffff, b & 0xffff, c & 0xffff])[1],
  VOP3Op.V_MIN3_I16: lambda a, b, c: sorted([sext(a & 0xffff, 16), sext(b & 0xffff, 16), sext(c & 0xffff, 16)])[0] & 0xffff,
  VOP3Op.V_MAX3_I16: lambda a, b, c: sorted([sext(a & 0xffff, 16), sext(b & 0xffff, 16), sext(c & 0xffff, 16)])[2] & 0xffff,
  VOP3Op.V_MED3_I16: lambda a, b, c: sorted([sext(a & 0xffff, 16), sext(b & 0xffff, 16), sext(c & 0xffff, 16)])[1] & 0xffff,
}

def _cmp8(a, b): return [False, a < b, a == b, a <= b, a > b, a != b, a >= b, True]
def _cmp6(a, b): return [a < b, a == b, a <= b, a > b, a != b, a >= b]

def vopc(op: int, s0: int, s1: int, s0_hi: int = 0, s1_hi: int = 0) -> int:
  base = op & 0x7f
  if 16 <= base <= 31:  # F32
    f0, f1, cmp, nan = f32(s0), f32(s1), base - 16, math.isnan(f32(s0)) or math.isnan(f32(s1))
    return int([False, f0<f1, f0==f1, f0<=f1, f0>f1, f0!=f1, f0>=f1, not nan, nan, f0<f1 or nan, f0==f1 or nan, f0<=f1 or nan, f0>f1 or nan, f0!=f1 or nan, f0>=f1 or nan, True][cmp])
  if 49 <= base <= 54: return int(_cmp6(sext(s0 & 0xffff, 16), sext(s1 & 0xffff, 16))[base - 49])  # I16
  if 57 <= base <= 62: return int(_cmp6(s0 & 0xffff, s1 & 0xffff)[base - 57])  # U16
  if 64 <= base <= 79:  # I32/U32
    cmp = (base - 64) % 8
    return int(_cmp8(sext(s0, 32), sext(s1, 32))[cmp] if base < 72 else _cmp8(s0, s1)[cmp])
  if 80 <= base <= 95:  # I64/U64
    s0_64, s1_64 = s0 | (s0_hi << 32), s1 | (s1_hi << 32)
    return int(_cmp8(sext(s0_64, 64), sext(s1_64, 64))[(base - 80) % 8] if base < 88 else _cmp8(s0_64, s1_64)[(base - 80) % 8])
  if base == 126:  # CLASS_F32
    f, mask = f32(s0), s1
    if math.isnan(f): return int(bool(mask & 0x3))
    if math.isinf(f): return int(bool(mask & (0x4 if f < 0 else 0x200)))
    if f == 0.0: return int(bool(mask & (0x20 if (s0 >> 31) & 1 else 0x40)))
    exp, sign = (s0 >> 23) & 0xff, (s0 >> 31) & 1
    return int(bool(mask & ((0x10 if sign else 0x80) if exp == 0 else (0x8 if sign else 0x100))))
  raise NotImplementedError(f"VOPC op {op} (base {base})")
