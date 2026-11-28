import functools
from collections.abc import Callable

from tinygrad import UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

from slopkernels.src.amd import (WMMA_ARG, _amd_byte_perm, _amd_fdot2, _amd_load, _cvt_pk_f16, _ds_bpermute,
                                 _fast_fmax, _lane_xor, _nat, warp_reduce)

GGML_BLOCK_SIZE, Q8_GROUP_SIZE = 256, 32
Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K = 8, 10, 11, 12, 13, 14
IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, MXFP4, NVFP4 = 16, 17, 18, 19, 20, 21, 22, 23, 39, 40
QUANT_SIZES = {Q8_0: 272, Q2_K: 84, Q3_K: 110, Q4_K: 144, Q5_K: 176, Q6_K: 210, IQ2_XXS: 66, IQ2_XS: 74,
               IQ3_XXS: 98, IQ1_S: 50, IQ4_NL: 144, IQ3_S: 110, IQ2_S: 82, IQ4_XS: 136, MXFP4: 136, NVFP4: 144}
QUANT_NAMES = {Q8_0: "q8_0", Q2_K: "q2_k", Q3_K: "q3_k", Q4_K: "q4_k", Q5_K: "q5_k", Q6_K: "q6",
               IQ2_XXS: "iq2_xxs", IQ2_XS: "iq2_xs", IQ3_XXS: "iq3_xxs", IQ1_S: "iq1_s", IQ4_NL: "iq4_nl",
               IQ3_S: "iq3_s", IQ2_S: "iq2_s", IQ4_XS: "iq4_xs", MXFP4: "mxfp4", NVFP4: "nvfp4"}
SMALL_BLOCK_QUANTS = {Q8_0: (1, 17), IQ4_NL: (1, 9), MXFP4: (1, 17), NVFP4: (2, 9)}
HALFWORD_QUANTS = {Q8_0, Q2_K, Q3_K, Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S}
OTF_FP8_QUANTS = {Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K, IQ4_XS, IQ3_S, IQ3_XXS, IQ2_S, MXFP4}
_RPW2_QUANTS = {IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_XXS, IQ3_S}
_LDS_GRID_QUANTS = {IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS}
_WMMA_TBL_QUANTS = {IQ2_XXS, IQ2_XS, IQ1_S, IQ2_S, IQ3_XXS, IQ3_S}
_R3_FAST_B_QUANTS = {Q8_0, Q4_K, IQ4_XS, MXFP4, IQ3_XXS, IQ2_S}
IQ4NL_LUT = (0xbfad9881, 0xf6eaddcf, 0x26190d01, 0x71594535)
IQ4NL_U8_LUT = tuple(x ^ 0x80808080 for x in IQ4NL_LUT)
_IQ4NL_VALS = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
MXFP4_LUT = (0x03020100, 0x0c080604, 0xfdfeff00, 0xf4f8fafc)
MXFP4_U8_LUT = tuple(x ^ 0x80808080 for x in MXFP4_LUT)

def _qswitch(qtype:int|UOp, cases:list[tuple[int, UOp]]) -> UOp:
  if isinstance(qtype, int): return dict(cases)[qtype]
  res = cases[-1][1]
  for qt, val in reversed(cases[:-1]): res = UOp(Ops.WHERE, src=(qtype.eq(qt), val, res))
  return res
def _qsize(qtype:int|UOp) -> int|UOp:
  return QUANT_SIZES[qtype] if isinstance(qtype, int) else _qswitch(qtype, [(q, UOp.const(s, dtypes.weakint)) for q, s in QUANT_SIZES.items()])
def _qraw(raw:UOp, qt:int) -> UOp:
  if raw.dtype != dtypes.uint8 or qt == MXFP4: return raw
  return raw.bitcast(dtypes.uint16) if qt in HALFWORD_QUANTS else raw.bitcast(dtypes.uint32)
def _qraw_u32(raw:UOp) -> UOp: return raw if raw.dtype == dtypes.uint32 else raw.bitcast(dtypes.uint32)

def _sh(x:int|UOp) -> int|UOp: return x.cast(dtypes.uint32) if isinstance(x, UOp) else x
def _v4(v:UOp) -> list[UOp]: return [v[0], v[1], v[2], v[3]]
def _half(v:UOp) -> UOp: return v.cast(dtypes.uint16).bitcast(dtypes.float16).float()
def _fast_rcp(v:UOp) -> UOp: return _nat("__builtin_amdgcn_rcpf({})", dtypes.float32, v)
def _lane_bpermute(src_lane:UOp, v:UOp) -> UOp: return _ds_bpermute(src_lane.cast(dtypes.int32) << 2, v, inline=False)
def _pin_f32(v:UOp, *deps:UOp) -> UOp:
  res = _nat('({{ float _v = ({0}); __asm__ volatile("" : "+v"(_v) :: "memory"); _v; }})', dtypes.float32, v, inline=False)
  return res.after(*deps) if deps else res
def _perm_alignbyte(hi:UOp, lo:UOp, sh:int|UOp) -> UOp:
  return _nat("__builtin_amdgcn_alignbyte({}, {}, {})", dtypes.uint32, hi.cast(dtypes.uint32), lo.cast(dtypes.uint32),
              sh.cast(dtypes.uint32) if isinstance(sh, UOp) else UOp.const(sh, dtypes.uint32))
def _splat_h2(u:UOp) -> UOp: return _amd_byte_perm(u, u, UOp.const(0x01000100, dtypes.uint32)).reshape((1,)).bitcast(dtypes.half)
def _f32_to_h2(v:UOp) -> UOp: return _nat("__builtin_bit_cast(unsigned int, __builtin_amdgcn_cvt_pkrtz({0}, {0}))", dtypes.uint32, v)
def _splat_f_h2(f:UOp) -> UOp: return _splat_h2(f.cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32))
def _load_byte(raw:UOp, base:UOp, off:int|UOp) -> UOp:
  w = raw.dtype.itemsize
  return raw[base + off // w].cast(dtypes.uint32) >> _sh((off % w) * 8) & 255
def _load_u32(raw:UOp, base:UOp, off:int|UOp, stream:bool=False) -> UOp:
  if raw.dtype == dtypes.uint32: return _amd_load(raw[base + off // 4]) if stream else raw[base + off // 4]
  if raw.dtype == dtypes.uint16:
    h_off = off // 2 if isinstance(off, int) and off % 2 == 0 else (off.divides(2) if isinstance(off, UOp) else None)
    if h_off is not None: return raw[base + h_off].cast(dtypes.uint32) | (raw[base + h_off + 1].cast(dtypes.uint32) << 16)
  return (_load_byte(raw, base, off) | (_load_byte(raw, base, off + 1) << 8) |
          (_load_byte(raw, base, off + 2) << 16) | (_load_byte(raw, base, off + 3) << 24))
def _nibble_lut(packed:UOp, sh:int|UOp, lut:tuple[int|UOp, ...]) -> UOp:
  nib = (packed >> _sh(sh)) & 0x0f0f0f0f
  lo = _amd_byte_perm(UOp.const(lut[1], dtypes.uint32) if isinstance(lut[1], int) else lut[1],
                      UOp.const(lut[0], dtypes.uint32) if isinstance(lut[0], int) else lut[0], nib & 0x07070707)
  hi = _amd_byte_perm(UOp.const(lut[3], dtypes.uint32) if isinstance(lut[3], int) else lut[3],
                      UOp.const(lut[2], dtypes.uint32) if isinstance(lut[2], int) else lut[2], nib & 0x07070707)
  return _amd_byte_perm(hi, lo, UOp.const(0x03020100, dtypes.uint32) | ((nib & 0x08080808) >> 1))
def _lut16_i8(nib:UOp, lut:tuple[int, ...]) -> UOp:
  lo, hi = UOp.const(lut[0] | (lut[1] << 32), dtypes.uint64), UOp.const(lut[2] | (lut[3] << 32), dtypes.uint64)
  return ((nib < 8).where(lo, hi) >> ((nib & 7).cast(dtypes.uint64) * 8)).cast(dtypes.uint8).bitcast(dtypes.int8).float()
def _e8m0_halved(e:UOp) -> UOp:
  e = e.cast(dtypes.uint32)
  return (e < 2).where((UOp.const(0x00200000, dtypes.uint32) << e).bitcast(dtypes.float), ((e - 1) << 23).bitcast(dtypes.float))
def _ue4m3_halved(e:UOp) -> UOp:
  e = e.cast(dtypes.uint32)
  exponent, mantissa = (e >> 3) & 15, e & 7
  normal = ((exponent + 119) << 23) | (mantissa << 20)
  return e.eq(0x7f).where(0.0, exponent.eq(0).where(mantissa.float() * 2**-10, normal.bitcast(dtypes.float32)))
def _iq_even_signs(s:UOp) -> UOp:
  p = s ^ (s >> 4)
  p = p ^ (p >> 2)
  p = p ^ (p >> 1)
  return s | ((p & 1) << 7)
def _sign_h2(s2:UOp) -> UOp: return ((s2 & 1) << 15) | ((s2 & 2) << 30)
def _iq_signed_word(w:UOp, s:UOp) -> UOp:
  ones = ((s.cast(dtypes.uint32) & 15) * 0x00204081) & 0x01010101
  return (w ^ (ones * 255)) + ones
def _q5_scales(raw:UOp, base:UOp, sub:UOp, stream:bool=True) -> tuple[UOp, UOp, UOp, UOp]:
  if stream: w0, w1, w2, w3 = _v4(_amd_load(raw[base], 4, stream=True))
  else: w0, w1, w2, w3 = raw[base], raw[base + 1], raw[base + 2], raw[base + 3]
  sc_w = (sub < 4).where((w1 >> _sh(sub * 8)) & 63, (((w3 >> _sh((sub - 4) * 8)) & 15) | (((w1 >> _sh((sub - 4) * 8 + 6)) & 3) << 4)))
  mn_w = (sub < 4).where((w2 >> _sh(sub * 8)) & 63, (((w3 >> _sh((sub - 4) * 8 + 4)) & 15) | (((w2 >> _sh((sub - 4) * 8 + 6)) & 3) << 4)))
  return _half(w0 & 0xffff), _half(w0 >> 16), sc_w.float(), mn_w.float()
def _q4k_pair_scales(hdr:list[UOp], sp:UOp) -> tuple[UOp, UOp, UOp, UOp]:
  w0, w1, w2, w3 = hdr
  d, dmin, hi = _half(w0 & 0xffff), _half(w0 >> 16), sp >= 2
  s_lo, m_lo = w1 >> _sh(sp * 16), w2 >> _sh(sp * 16)
  w3s, w1h, w2h = w3 >> _sh((sp - 2) * 16), w1 >> _sh((sp - 2) * 16), w2 >> _sh((sp - 2) * 16)
  sc0 = hi.where((w3s & 15) | (((w1h >> 6) & 3) << 4), s_lo & 63)
  mn0 = hi.where(((w3s >> 4) & 15) | (((w2h >> 6) & 3) << 4), m_lo & 63)
  sc1 = hi.where(((w3s >> 8) & 15) | (((w1h >> 14) & 3) << 4), (s_lo >> 8) & 63)
  mn1 = hi.where(((w3s >> 12) & 15) | (((w2h >> 14) & 3) << 4), (m_lo >> 8) & 63)
  return d * sc0.float(), dmin * mn0.float(), d * sc1.float(), dmin * mn1.float()
def _iq4_scale(raw:UOp, base:UOp, sub:UOp, stream:bool=False) -> UOp:
  h = _amd_load(raw[base], 2, stream=False) if stream else None
  h0, h1 = (h[0], h[1]) if h is not None else (raw[base], raw[base + 1])
  return _half(h0) * ((((h1 >> _sh(4 * sub)) & 15) | (((h0 >> _sh(16 + 2 * sub)) & 3) << 4)).cast(dtypes.int32) - 32).float()
def _iq1_scale(raw:UOp, base:UOp, sub:UOp) -> tuple[UOp, UOp]:
  qh = raw[base + 17 + sub].cast(dtypes.uint32)
  return (((qh >> 12) & 7) * 2 + 1).float() * _half(raw[base]), (qh >> 15).eq(0).where(-0.125, 0.125)
def _load_grid_lds(grid:UOp, qt:int|UOp, tid:UOp, threads:int|UOp, expand_iq3x:bool=True) -> UOp:
  g_len = grid.shape[0]
  if not expand_iq3x:
    g_lds, g_i = UOp.alloc((g_len,), dtypes.uint32, addrspace=AddrSpace.LOCAL), UOp.range((g_len + threads - 1) // threads, 20)
    g_off = tid + g_i * threads
    return g_lds.after(g_lds[g_off.valid(g_off < g_len)].store(grid[g_off.valid(g_off < g_len)].load()).end(g_i).barrier())
  is_iq3x = UOp.const(qt == IQ3_XXS, dtypes.bool) if isinstance(qt, int) else qt.eq(IQ3_XXS)
  g_lds_len = is_iq3x.where(512, g_len)
  g_lds, g_i = UOp.alloc((g_lds_len,), dtypes.uint32, addrspace=AddrSpace.LOCAL), UOp.range((g_len + threads - 1) // threads, 20)
  g_off = tid + g_i * threads
  gv = grid[g_off.valid(g_off < g_len)].load()
  bf = [((gv >> (8 * i)) & 255).float() * UOp.const(1.0 / 64.0, dtypes.float32) for i in range(4)]
  pk0 = _nat("__builtin_bit_cast(unsigned int, __builtin_amdgcn_cvt_pkrtz({0}, {1}))", dtypes.uint32, bf[0], bf[1])
  pk1 = _nat("__builtin_bit_cast(unsigned int, __builtin_amdgcn_cvt_pkrtz({0}, {1}))", dtypes.uint32, bf[2], bf[3])
  st_g = UOp.group(g_lds[is_iq3x.where(g_off * 2, g_off).valid(g_off < g_len)].store(is_iq3x.where(pk0, gv)),
                   g_lds[(g_off * 2 + 1).valid(is_iq3x & (g_off < g_len))].store(pk1))
  return g_lds.after(st_g.end(g_i).barrier())
def _gemv_base(q:int, isz:int, row_i:UOp, grp:UOp, out_features:UOp, in_features:UOp, wg_g:UOp, group_rows:UOp|None) -> UOp:
  g_blk = (group_rows > 0).where(wg_g * (out_features * (in_features // 256)), 0) if group_rows is not None else 0
  if q in SMALL_BLOCK_QUANTS:
    g, bw = SMALL_BLOCK_QUANTS[q]
    g_sblk = (group_rows > 0).where(wg_g * (out_features * (in_features // (Q8_GROUP_SIZE * g))), 0) if group_rows is not None else 0
    return (row_i * (in_features // (Q8_GROUP_SIZE * g)) + grp // g + g_sblk) * bw
  return (row_i * (in_features // 256) + grp // 8 + g_blk) * (QUANT_SIZES[q] // isz)
def _small_block(sb_base:UOp, sub:UOp, qt:int) -> UOp:
  g, bw = SMALL_BLOCK_QUANTS[qt]
  return sb_base + sub // g * bw

# ******** unified single-kernel direct fdot2 GEMV & MoE (RDNA3 / RDNA4 / CDNA) ********

_DOT_U8_QUANTS = {Q8_0, Q4_K, Q5_K, Q6_K, Q2_K, IQ4_XS, IQ4_NL, IQ3_S}

def _direct_group_dot_fn(raw:UOp, base:UOp, sub:UOp, qt:int, grid:UOp|None, wr:UOp, sc_r:UOp,
                         stream_w:bool=True, lane:UOp|None=None, deps:tuple[UOp, ...]=()) -> UOp:
  c8080 = UOp.const(0x80808080, dtypes.uint32)
  sel01, sel23 = UOp.const(0x05010400, dtypes.uint32), UOp.const(0x07030602, dtypes.uint32)
  sel01z, sel23z = UOp.const(0x05000400, dtypes.uint32), UOp.const(0x07000600, dtypes.uint32)
  z, z32 = UOp.const(0.0, dtypes.float32), UOp.const(0, dtypes.uint32)
  def ld(p:UOp, lanes:int|None=None) -> UOp:
    return _amd_load(p, lanes, stream=stream_w) if lanes is not None else p.load(arg="nontemporal" if stream_w else None)
  def put_sc(w_st:UOp, a0:UOp, a1:UOp, b0:UOp, b1:UOp) -> UOp: return UOp.group(w_st, sc_r.after(*deps).store(UOp.stack(a0, a1, b0, b1)))
  def fill8(fn_w:Callable[[UOp], UOp]) -> UOp:
    j = UOp.range(8, 33)
    return wr.after(*deps)[j].store(fn_w(j)).end(j)
  def fill16_pairs(fn_p:Callable[[UOp], tuple[UOp, UOp]]) -> UOp:
    j = UOp.range(8, 33)
    h0, h1 = fn_p(j)
    return UOp.group(wr.after(*deps)[j * 2].store(h0), wr.after(*deps)[j * 2 + 1].store(h1)).end(j)
  def st_v(v:UOp) -> UOp:
    arr = UOp.alloc(v.shape, dtypes.uint32, addrspace=AddrSpace.REG)
    return arr.after(arr.after(*deps).store(v))
  def iq2_wh2(gw:UOp, s4:UOp) -> tuple[UOp, UOp]:
    sel = (gw >> 4) & 0x03030303
    hi_b = _amd_byte_perm(z32, UOp.const(0x00514e48, dtypes.uint32), sel)
    lo_b = _amd_byte_perm(z32, UOp.const(0x00604000, dtypes.uint32), sel)
    return _amd_byte_perm(hi_b, lo_b, sel01) ^ _sign_h2(s4), _amd_byte_perm(hi_b, lo_b, sel23) ^ _sign_h2(s4 >> 2)
  def mxfp4_wh2(pw:UOp) -> tuple[UOp, UOp]:
    hi_b = (_amd_byte_perm(UOp.const(0x4a484644, dtypes.uint32), UOp.const(0x42403c00, dtypes.uint32), pw & 0x07070707)
            | ((pw & 0x08080808) << 4))
    return _amd_byte_perm(hi_b, z32, sel01z), _amd_byte_perm(hi_b, z32, sel23z)
  if qt == IQ3_XXS:
    assert grid is not None
    sc_w = ld(raw[base + 33 + sub * 2], 2).bitcast(dtypes.uint32)[0]
    qv = ld(raw[base + 1 + sub * 4], 4).bitcast(dtypes.uint32)
    sc = _half(ld(raw[base])) * ((sc_w >> 28).float() + 0.5) * 32.0
    def iq3x_p(j:UOp) -> tuple[UOp, UOp]:
      p, k2 = j // 2, j % 2
      g_i = (((p < 2).where(qv[0], qv[1]) >> _sh((p & 1) * 16 + k2 * 8)) & 255) * 2
      s4 = _iq_even_signs((sc_w >> _sh(7 * p)) & 127) >> _sh(k2 * 4)
      return grid[g_i].load() ^ _sign_h2(s4), grid[g_i + 1].load() ^ _sign_h2(s4 >> 2)
    return put_sc(fill16_pairs(iq3x_p), sc, sc, z, z)
  if qt == Q8_0:
    ws = ld(raw[base + 1], 8).bitcast(dtypes.uint32).cat(ld(raw[base + 9], 8).bitcast(dtypes.uint32))
    sc = _half(ld(raw[base]))
    b = UOp.const(-1152.0, dtypes.float32) * sc
    return put_sc(wr.after(*deps).reshape(2, 8)[0].store(ws ^ c8080.reshape((1,)).expand((8,))), sc, sc, b, b)
  if qt in (Q4_K, Q5_K):
    qr = st_v(ld(raw[base + (4 if qt == Q4_K else 12) + (sub // 2) * 8], 8))
    hr = st_v(ld(raw[base + 4], 8)) if qt == Q5_K else None
    sh, (d, dmin, sc, mn) = ((sub & 1) * 4).cast(dtypes.uint32), _q5_scales(raw, base, sub, stream=stream_w)
    ds, dm = d * sc, dmin * mn
    w_st = fill8(lambda j: ((qr[j].load() >> sh) & 0x0f0f0f0f) |
                 ((((hr[j].load() >> sub.cast(dtypes.uint32)) & 0x01010101) << 4) if hr is not None else 0))
    b = -(dm + UOp.const(1024.0, dtypes.float32) * ds)
    return put_sc(w_st, ds, ds, b, b)
  if qt == Q6_K:
    ql_b, qh_b = base + (sub // 4) * 32 + (sub % 2) * 16, base + 64 + (sub // 4) * 16
    ql_sh, qh_sh = ((sub % 4 // 2) * 4).cast(dtypes.uint32), ((sub % 4) * 2).cast(dtypes.uint32)
    qlr = st_v(ld(raw[ql_b], 8).bitcast(dtypes.uint32).cat(ld(raw[ql_b + 8], 8).bitcast(dtypes.uint32)))
    qhr = st_v(ld(raw[qh_b], 8).bitcast(dtypes.uint32).cat(ld(raw[qh_b + 8], 8).bitcast(dtypes.uint32)))
    sc_w, d_sc = ld(raw[base + 96 + sub]), _half(ld(raw[base + 104]))
    s0 = (sc_w & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float() * d_sc
    s1 = ((sc_w >> 8) & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float() * d_sc
    w_st = fill8(lambda j: ((qlr[j].load() >> ql_sh) & 0x0f0f0f0f) | (((qhr[j].load() >> qh_sh) & 0x03030303) << 4))
    return put_sc(w_st, s0, s1, UOp.const(-1056.0, dtypes.float32) * s0, UOp.const(-1056.0, dtypes.float32) * s1)
  if qt == Q3_K:
    assert lane is not None
    b0, b4 = (lane // 8) * 8, (lane // 4) * 4
    hm_m = ld(raw[base + sub * 2], 2).bitcast(dtypes.uint32)[0]
    qs_p = ld(raw[base + 16 + (sub // 4) * 16 + (sub % 4) * 4], 4).bitcast(dtypes.uint32)
    sl16, sh16 = ld(raw[base + 48 + (sub % 4)]).cast(dtypes.uint32), ld(raw[base + 52 + (sub % 2)]).cast(dtypes.uint32)
    qs_sh, hm_sh, d_sc = ((sub % 4) * 2).cast(dtypes.uint32), sub.cast(dtypes.uint32), _half(ld(raw[base + 54]))
    s = [((((sl16 >> (h * 8 + (sub // 4) * 4).cast(dtypes.uint32)) & 15) |
           (((sh16 >> (h * 8 + (sub // 2) * 2).cast(dtypes.uint32)) & 3) << 4)).cast(dtypes.int32) - 32).float() * d_sc for h in (0, 1)]
    c_lo, c_hi = UOp.const(0xbcc0c2c4, dtypes.uint32), UOp.const(0x42403c00, dtypes.uint32)
    def q3_p(j:UOp) -> tuple[UOp, UOp]:
      hm_j = _lane_bpermute(b0 + j, hm_m)
      qs_j = _lane_bpermute(b4 + j // 2, (j % 2).eq(0).where(qs_p[0], qs_p[1]))
      hi_b = _amd_byte_perm(c_hi, c_lo, ((qs_j >> qs_sh) & 0x03030303) | (((hm_j >> hm_sh) & 0x01010101) << 2))
      return _amd_byte_perm(hi_b, z32, sel01z), _amd_byte_perm(hi_b, z32, sel23z)
    return put_sc(fill16_pairs(q3_p), s[0], s[1], z, z)
  if qt == Q2_K:
    qsr = st_v(ld(raw[base + 8 + (sub // 4) * 16], 8).bitcast(dtypes.uint32).cat(ld(raw[base + 16 + (sub // 4) * 16], 8).bitcast(dtypes.uint32)))
    sh, dm, sc_w = ((sub % 4) * 2).cast(dtypes.uint32), ld(raw[base + 40], 2), ld(raw[base + sub]).cast(dtypes.uint32)
    dh, dmh, sc0, sc1 = _half(dm[0]), _half(dm[1]), sc_w & 255, (sc_w >> 8) & 255
    ds0, dm0, ds1, dm1 = (sc0 & 15).float() * dh, (sc0 >> 4).float() * dmh, (sc1 & 15).float() * dh, (sc1 >> 4).float() * dmh
    return put_sc(fill8(lambda j: (qsr[j].load() >> sh) & 0x03030303),
                  ds0, ds1, -(dm0 + UOp.const(1024.0, dtypes.float32) * ds0), -(dm1 + UOp.const(1024.0, dtypes.float32) * ds1))
  if qt in (IQ4_XS, IQ4_NL):
    pv = ld(raw[base + 2 + sub * 4], 4) if qt == IQ4_XS else ld(raw[base + 1], 8).bitcast(dtypes.uint32)
    sc = _iq4_scale(raw, base, sub, stream=stream_w) if qt == IQ4_XS else _half(ld(raw[base]))
    pr, b = st_v(pv), UOp.const(-1152.0, dtypes.float32) * sc
    return put_sc(fill8(lambda j: _nibble_lut(pr[j % 4].load(), _sh((j // 4) * 4), IQ4NL_U8_LUT)), sc, sc, b, b)
  if qt == MXFP4:
    pv = ld(raw[base + 1], 16).bitcast(dtypes.uint32)
    pr, sc = st_v(pv), _e8m0_halved(ld(raw[base]))
    return put_sc(fill16_pairs(lambda j: mxfp4_wh2(pr[j % 4].load() >> _sh((j // 4) * 4))), sc, sc, z, z)
  if qt == NVFP4:
    h64 = sub % 2
    sc_w, pr = ld(raw[base]), st_v(ld(raw[base + 1 + h64 * 4], 4))
    s0, s1 = _ue4m3_halved((sc_w >> _sh(h64 * 16)) & 255), _ue4m3_halved((sc_w >> _sh(h64 * 16 + 8)) & 255)
    return put_sc(fill16_pairs(lambda j: mxfp4_wh2(pr[(j // 4) * 2 + (j % 2)].load() >> _sh(((j % 4) // 2) * 4))), s0, s1, z, z)
  assert grid is not None
  if qt == IQ3_S:
    qh_u = (ld(raw[base + 33 + sub // 2]).cast(dtypes.uint32) >> _sh((sub & 1) * 8)) & 255
    sc = _half(ld(raw[base])) * (1 + 2 * ((ld(raw[base + 53 + sub // 4]).cast(dtypes.uint32) >> _sh((sub & 3) * 4)) & 15)).float()
    qs_v, sg_w = ld(raw[base + 1 + sub * 4], 4).bitcast(dtypes.uint32), ld(raw[base + 37 + sub * 2], 2).bitcast(dtypes.uint32)[0]
    w_st = fill8(lambda j: _iq_signed_word(grid[(((j < 4).where(qs_v[0], qs_v[1]) >> _sh((j % 4) * 8)) & 255) |
                 (((qh_u >> _sh(j)) & 1) << 8)].load(), (sg_w >> _sh(j * 4)) & 15) ^ c8080)
    b = UOp.const(-1152.0, dtypes.float32) * sc
    return put_sc(w_st, sc, sc, b, b)
  if qt == IQ1_S:
    qh_u, qs_w = ld(raw[base + 17 + sub]).cast(dtypes.uint32), ld(raw[base + 1 + sub * 2], 2).bitcast(dtypes.uint32)[0]
    sc, is_p = (((qh_u >> 12) & 7) * 2 + 1).float() * _half(ld(raw[base])), (qh_u >> 15).eq(0)
    hi_t = is_p.where(UOp.const(0xbb003c30, dtypes.uint32), UOp.const(0xbc003bb0, dtypes.uint32))
    lo_t = is_p.where(UOp.const(0x00008000, dtypes.uint32), UOp.const(0x80000000, dtypes.uint32))
    def iq1_p(j:UOp) -> tuple[UOp, UOp]:
      p, k = j // 2, j % 2
      sel = grid[(((qs_w >> _sh(p * 8)) & 255) | (((qh_u >> _sh(3 * p)) & 7) << 8)) * 2 + k].load() & 0x03030303
      hi_b, lo_b = _amd_byte_perm(z32, hi_t, sel), _amd_byte_perm(z32, lo_t, sel)
      return _amd_byte_perm(hi_b, lo_b, sel01), _amd_byte_perm(hi_b, lo_b, sel23)
    return put_sc(fill16_pairs(iq1_p), sc, sc, z, z)
  if qt == IQ2_S:
    qh_u = (ld(raw[base + 33 + sub // 2]).cast(dtypes.uint32) >> _sh((sub & 1) * 8)) & 255
    sc_b = (ld(raw[base + 37 + sub // 2]).cast(dtypes.uint32) >> _sh((sub & 1) * 8)) & 255
    q_w, s_w = ld(raw[base + 1 + sub * 2], 2).bitcast(dtypes.uint32)[0], ld(raw[base + 17 + sub * 2], 2).bitcast(dtypes.uint32)[0]
    d_b = _half(ld(raw[base])) * 0.25
    sf0, sf1 = ((sc_b & 15).float() + 0.5) * d_b, (((sc_b >> 4) & 15).float() + 0.5) * d_b
    def iq2s_p(j:UOp) -> tuple[UOp, UOp]:
      p, k = j // 2, j % 2
      return iq2_wh2(grid[(((q_w >> _sh(p * 8)) & 255) | (((qh_u >> _sh(2 * p)) & 3) << 8)) * 2 + k].load(),
                     ((s_w >> _sh(p * 8)) & 255) >> _sh(k * 4))
    return put_sc(fill16_pairs(iq2s_p), sf0, sf1, z, z)
  if qt == IQ2_XS:
    sc_b, d_b = (ld(raw[base + 33 + sub // 2]).cast(dtypes.uint32) >> _sh((sub & 1) * 8)) & 255, _half(ld(raw[base])) * 0.25
    sf0, sf1 = ((sc_b & 15).float() + 0.5) * d_b, (((sc_b >> 4) & 15).float() + 0.5) * d_b
    def iq2xs_p(j:UOp) -> tuple[UOp, UOp]:
      p, k = j // 2, j % 2
      pk = ld(raw[base + 1 + sub * 4 + p]).cast(dtypes.uint32)
      return iq2_wh2(grid[(pk & 511) * 2 + k].load(), _iq_even_signs(pk >> 9) >> _sh(k * 4))
    return put_sc(fill16_pairs(iq2xs_p), sf0, sf1, z, z)
  qs_w, sg_w = ld(raw[base + 1 + sub * 4], 2).bitcast(dtypes.uint32)[0], ld(raw[base + 3 + sub * 4], 2).bitcast(dtypes.uint32)[0]
  sc = _half(ld(raw[base])) * ((sg_w >> 28).float() + 0.5) * 0.25
  def iq2xxs_p(j:UOp) -> tuple[UOp, UOp]:
    p, k = j // 2, j % 2
    return iq2_wh2(grid[((qs_w >> _sh(p * 8)) & 255) * 2 + k].load(), _iq_even_signs((sg_w >> _sh(7 * p)) & 127) >> _sh(k * 4))
  return put_sc(fill16_pairs(iq2xxs_p), sc, sc, z, z)

def _load_x_32(x:UOp, xt:UOp, grp:UOp, ones_h2:UOp, z:UOp, need_xs:bool|UOp, deps:tuple[UOp, ...]=(), rid:int=23) -> tuple[UOp, UOp, UOp]:
  xh, jx = UOp.alloc((16,), dtypes.uint32, addrspace=AddrSpace.REG), UOp.range(8, rid)
  xv = _amd_load(x[xt, grp * 32 + jx * 4], 4)
  def _pk(a:UOp, b:UOp) -> UOp:
    return a.float().cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32) | (
      b.float().cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32) << 16)
  h0, h1 = _pk(xv[0], xv[1]), _pk(xv[2], xv[3])
  xh = xh.after(UOp.group(xh.after(*deps, jx)[jx * 2].store(h0), xh.after(*deps, jx)[jx * 2 + 1].store(h1)).end(jx))
  if isinstance(need_xs, bool) and not need_xs: return xh, z, z
  x0r, x1r = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG), UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
  x0r, x1r = x0r.after(x0r.after(*deps)[0].store(z)), x1r.after(x1r.after(*deps)[0].store(z))
  r_xs = UOp.range(need_xs.cast(dtypes.weakint), rid + 2) if isinstance(need_xs, UOp) else None
  x0d, x1d = (x0r.after(r_xs), x1r.after(r_xs)) if r_xs is not None else (x0r, x1r)
  js = UOp.range(8, rid + 1)
  upd = UOp.group(x0d[0].store(_amd_fdot2(ones_h2, xh[js].load(), x0d.after(js)[0].load())),
                  x1d[0].store(_amd_fdot2(ones_h2, xh[js + 8].load(), x1d.after(js)[0].load()))).end(js)
  if r_xs is not None: upd = upd.end(r_xs)
  return xh, x0r.after(upd)[0].load(), x1r.after(upd)[0].load()

def _direct_gemv_body(out:UOp, raw:UOp, x:UOp, grids:tuple[UOp, ...], out_features:UOp, in_features:UOp, qt:int|UOp,
                      rpw:int|UOp=1, wk_cap:int|UOp=16, w_rows:int|UOp=1, w1_cond:UOp|None=None,
                      stream_w:bool=True, lds_grid:bool=False, group_rows:UOp|None=None, gate:Callable|None=None,
                      xtoken:Callable|None=None, name:str="linear_", row_mask:Callable|None=None) -> UOp:
  tokens, group_count = out.shape[0], in_features // 32
  assert isinstance(tokens, UOp)
  qtypes = (qt,) if isinstance(qt, int) else tuple(QUANT_NAMES)
  wg_groups, t_per_wg = ((group_rows > 0).where(tokens // group_rows.maximum(1), 1),
                         (group_rows > 0).where(group_rows, tokens)) if group_rows is not None else (tokens, UOp.const(1, dtypes.weakint))
  k_waves = (group_count + 31) // 32
  k_steps = (k_waves + wk_cap - 1) // wk_cap
  waves_k_m = (k_waves + k_steps - 1) // k_steps
  waves_k = w1_cond.where(1, waves_k_m) if w1_cond is not None else waves_k_m
  lanes_k = (waves_k.eq(1) & (group_count <= 8) & (out_features % (4 * rpw * w_rows)).eq(0)).where(
    8, (waves_k.eq(1) & (group_count <= 16) & (out_features % (2 * rpw * w_rows)).eq(0)).where(16, 32))
  opw, t_outer = 32 // lanes_k, t_per_wg
  wg_rows, threads = w_rows * opw * rpw, waves_k * w_rows * 32
  row_blks, steps = out_features // wg_rows, (group_count + waves_k * lanes_k - 1) // (waves_k * lanes_k)
  rb = UOp.range(wg_groups * t_outer * row_blks, 0, AxisType.GLOBAL)
  lane, wave_k, w_row = UOp.range(32, -1, AxisType.WARP), UOp.range(waves_k, 1, AxisType.LOCAL), UOp.range(w_rows, 2, AxisType.LOCAL)
  tid, wg_g, rem = (w_row * waves_k + wave_k) * 32 + lane, rb // (t_outer * row_blks), rb % (t_outer * row_blks)
  t_o, rblk, sub_l, which = rem // row_blks, rem % row_blks, lane % lanes_k, lane // lanes_k
  row_base = ((rblk * w_rows + w_row) * opw + which) * rpw
  is_u8:bool|UOp = (qt in _DOT_U8_QUANTS) if isinstance(qt, int) else functools.reduce(lambda a, b: a | b, (qt.eq(q) for q in _DOT_U8_QUANTS))
  grid = grids[0] if grids else None
  grid_lds = _load_grid_lds(grid, qt, tid, threads) if (lds_grid and grid is not None) else grid
  ones_h2, z = UOp.const(0x3c003c00, dtypes.uint32), UOp.const(0.0, dtypes.float32)
  c6400, sel_lo, sel_hi = UOp.const(0x64006400, dtypes.uint32), UOp.const(0x05010500, dtypes.uint32), UOp.const(0x05030502, dtypes.uint32)
  acc = UOp.alloc((rpw,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0.0)))

  s = UOp.range(steps, 5)
  grp_raw = sub_l + (wave_k + s * waves_k) * lanes_k
  valid_g, grp = grp_raw < group_count, grp_raw.minimum(group_count - 1)
  tok = wg_g * t_per_wg + t_o
  xh_v, xs0_v, xs1_v = _load_x_32(x, xtoken(tok) if xtoken is not None else tok, grp, ones_h2, z, is_u8, deps=(s,))
  r_i = UOp.range(rpw, 26)
  wr, sc_r = UOp.alloc((16,), dtypes.uint32, addrspace=AddrSpace.REG), UOp.alloc((4,), dtypes.float32, addrspace=AddrSpace.REG)
  def _dot_for(q:int) -> UOp:
    rq = _qraw(raw, q)
    base = _gemv_base(q, rq.dtype.itemsize, row_base + r_i, grp, out_features, in_features, wg_g, group_rows)
    if gate is not None: base = gate(tok, base, q, rq.dtype.itemsize)
    return _direct_group_dot_fn(rq, base, grp % 8, q, grid_lds if (lds_grid and q == IQ3_XXS) else grid, wr, sc_r,
                                stream_w=stream_w and q not in (IQ4_XS, IQ3_XXS), lane=lane, deps=(s, r_i))
  fill_done = _qswitch(qt, [(q, _dot_for(q)) for q in qtypes])
  wr_r, sc_v = wr.after(fill_done), sc_r.after(fill_done)
  d0r, d1r = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG), UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
  d0r, d1r = d0r.after(d0r.after(s, r_i, fill_done)[0].store(z)), d1r.after(d1r.after(s, r_i, fill_done)[0].store(z))
  dot_parts: list[UOp] = []
  if not (isinstance(is_u8, bool) and not is_u8):
    r_u8 = UOp.range(is_u8.cast(dtypes.weakint), 34) if isinstance(is_u8, UOp) else None
    d0u, d1u = (d0r.after(r_u8), d1r.after(r_u8)) if r_u8 is not None else (d0r, d1r)
    jd = UOp.range(4, 31)
    w0, w1 = wr_r[jd].load(), wr_r[jd + 4].load()
    d0 = _amd_fdot2(_amd_byte_perm(c6400, w0, sel_hi), xh_v[jd * 2 + 1].load(),
                    _amd_fdot2(_amd_byte_perm(c6400, w0, sel_lo), xh_v[jd * 2].load(), d0u.after(jd)[0].load()))
    d1 = _amd_fdot2(_amd_byte_perm(c6400, w1, sel_hi), xh_v[jd * 2 + 9].load(),
                    _amd_fdot2(_amd_byte_perm(c6400, w1, sel_lo), xh_v[jd * 2 + 8].load(), d1u.after(jd)[0].load()))
    u8_u = UOp.group(d0u[0].store(d0), d1u[0].store(d1)).end(jd)
    dot_parts.append(u8_u.end(r_u8) if r_u8 is not None else u8_u)
  if not (isinstance(is_u8, bool) and is_u8):
    r_wh = UOp.range(is_u8.logical_not().cast(dtypes.weakint), 35) if isinstance(is_u8, UOp) else None
    d0w, d1w = (d0r.after(r_wh), d1r.after(r_wh)) if r_wh is not None else (d0r, d1r)
    jd = UOp.range(8, 36)
    wh_u = UOp.group(d0w[0].store(_amd_fdot2(wr_r[jd].load(), xh_v[jd].load(), d0w.after(jd)[0].load())),
                     d1w[0].store(_amd_fdot2(wr_r[jd + 8].load(), xh_v[jd + 8].load(), d1w.after(jd)[0].load()))).end(jd)
    dot_parts.append(wh_u.end(r_wh) if r_wh is not None else wh_u)
  dot_done = UOp.group(*dot_parts) if len(dot_parts) > 1 else dot_parts[0]
  dot_val = (d0r.after(dot_done)[0].load() * sc_v[0].load() + d1r.after(dot_done)[0].load() * sc_v[1].load() +
             sc_v[2].load() * xs0_v + sc_v[3].load() * xs1_v)
  upd = acc[r_i].store(acc.after(s, r_i)[r_i].load() + valid_g.where(dot_val, z)).end(r_i, s)

  acc_done = acc.after(upd)
  def _red_sub(v:UOp) -> UOp:
    v = warp_reduce(v, masks=(4, 2, 1))
    v = (lanes_k > 8).where(v + _lane_xor(v, 8), v)
    return (lanes_k > 16).where(v + _lane_xor(v, 16), v)
  tok_o, has_wk = wg_g * t_per_wg + t_o, waves_k > 1
  def _mask_out(v:UOp) -> UOp: return (row_mask(tok_o).where(v, z) if row_mask is not None else v).cast(out.dtype)
  r_w1, r_o1 = UOp.range(has_wk.logical_not().cast(dtypes.weakint), 29), UOp.range(rpw, 27)
  st_w1 = out[tok_o, (row_base + r_o1).valid(has_wk.logical_not() & sub_l.eq(0))].store(_mask_out(_red_sub(acc_done[r_o1].load()))).end(r_o1, r_w1)
  r_wk, r_s, r_o2 = UOp.range(has_wk.cast(dtypes.weakint), 30), UOp.range(rpw, 28), UOp.range(rpw, 32)
  w_lds = UOp.alloc((w_rows, rpw, waves_k), dtypes.float32, addrspace=AddrSpace.LOCAL)
  w_lds_r = w_lds.after(w_lds.after(r_wk)[w_row, r_s, wave_k.valid(has_wk & sub_l.eq(0))].store(_red_sub(acc_done[r_s].load())).end(r_s).barrier())
  wk_r = UOp.range(waves_k, 12)
  st_wk = out[tok_o, (row_base + r_o2).valid(has_wk & wave_k.eq(0) & sub_l.eq(0))].store(
    _mask_out(w_lds_r[w_row, r_o2, wk_r].load().reduce(wk_r, arg=(Ops.ADD, 0)))).end(r_o2, r_wk)
  kname = name + (QUANT_NAMES[qt] if isinstance(qt, int) else "quant")
  return UOp.group(st_w1, st_wk).end(lane, wave_k, w_row, rb).sink(arg=KernelInfo(name=kname, opts_to_apply=()))

def _iq4_lds_gemv_kernel(out:UOp, raw:UOp, x:UOp, out_features:UOp, in_features:UOp, arch:UOp) -> UOp:
  # Single-token IQ4_XS decode kernel using a 1KB in-LDS f16x2 byte LUT (zero global LUT load, zero x-sum bias).
  gc = in_features // 32
  k_waves = (gc + 31) // 32
  k_steps = (k_waves + 15) // 16
  wk_full = (k_waves + k_steps - 1) // k_steps
  r3_mid = arch.eq(11) & (in_features < 8192)
  waves_k = r3_mid.where((in_features >= 6144).where(4, 2).minimum(k_waves), wk_full)
  w_rows = r3_mid.where((in_features >= 6144).where((out_features % 2).eq(0).where(2, 1), (out_features % 4).eq(0).where(4, 1)), 1)
  steps = (gc + waves_k * 32 - 1) // (waves_k * 32)
  rb = UOp.range(out_features // w_rows, 0, AxisType.GLOBAL)
  lane, wave_k, w_row = UOp.range(32, -1, AxisType.WARP), UOp.range(waves_k, 1, AxisType.LOCAL), UOp.range(w_rows, 2, AxisType.LOCAL)
  row, tid, thr = rb * w_rows + w_row, (w_row * waves_k + wave_k) * 32 + lane, w_rows * waves_k * 32
  g_lds, gi = UOp.alloc((256,), dtypes.uint32, addrspace=AddrSpace.LOCAL), UOp.range((256 + thr - 1) // thr, 20)
  b = tid + gi * thr
  bm = b.minimum(255).cast(dtypes.uint32)
  u8w = _nibble_lut((bm & 15) | (((bm >> 4) & 15) << 8), 0, IQ4NL_U8_LUT)
  lut_r = g_lds.after(g_lds[b.valid(b < 256)].store(
    _cvt_pk_f16(((u8w & 255).cast(dtypes.int32) - 128).float(), (((u8w >> 8) & 255).cast(dtypes.int32) - 128).float())).end(gi).barrier())
  raw_u32, z = raw.bitcast(dtypes.uint32), UOp.const(0.0, dtypes.float32)
  acc = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0.0)))
  s = UOp.range(steps, 5)
  grp_raw = lane + (wave_k + s * waves_k) * 32
  ok, grp = grp_raw < gc, grp_raw.minimum(gc - 1)
  x0 = [_amd_load(x[0, grp * 32 + j * 4], 4) for j in range(4)]
  x1 = [_amd_load(x[0, grp * 32 + 16 + j * 4], 4) for j in range(4)]
  blk_base, sub = (row * (in_features // 256) + grp // 8) * 34, grp % 8
  pv, sc = _amd_load(raw_u32[blk_base + 2 + sub * 4], 4, stream=False), _iq4_scale(raw_u32, blk_base, sub, stream=False)
  d = z
  for j in range(4):
    w_u = pv[j]
    for b_i in range(4):
      d = _amd_fdot2(lut_r[(w_u >> (b_i * 8)) & 255 if b_i else (w_u & 255)].load(), _cvt_pk_f16(x0[j][b_i], x1[j][b_i]), d)
  upd = acc[0].store(acc.after(s)[0].load() + ok.where(d * sc, z)).end(s)
  wsum = warp_reduce(acc.after(upd)[0].load(), full_wave=True, dpp=True)
  w_lds = UOp.alloc((w_rows, waves_k), dtypes.float32, addrspace=AddrSpace.LOCAL)
  w_r, kr = w_lds.after(w_lds[w_row, wave_k.valid(lane.eq(0))].store(wsum).barrier()), UOp.range(waves_k, 12)
  return out[0, row.valid(wave_k.eq(0) & lane.eq(0))].store(w_r[w_row, kr].load().reduce(kr, arg=(Ops.ADD, 0)).cast(out.dtype)).end(
    lane, wave_k, w_row, rb).sink(arg=KernelInfo(name="linear_quant", opts_to_apply=()))

# ******** 8-lane split q8_1 quantizer + zero-LDS RDNA3/4 sudot4 multi-token GEMV ********

def _sudot4(a:UOp, b:UOp, c:UOp) -> UOp:
  return _nat("__builtin_amdgcn_sudot4(true, (int)({0}), true, (int)({1}), {2}, false)", dtypes.int32, a, b, c)

def _pack4_s8(f0:UOp, f1:UOp, f2:UOp, f3:UOp, inv:UOp) -> UOp:
  return _nat(
    "({{ float _s = ({4}); "
    "unsigned int _b0 = (unsigned int)(int)__builtin_rintf(({0}) * _s) & 255u; "
    "unsigned int _b1 = (unsigned int)(int)__builtin_rintf(({1}) * _s) & 255u; "
    "unsigned int _b2 = (unsigned int)(int)__builtin_rintf(({2}) * _s) & 255u; "
    "unsigned int _b3 = (unsigned int)(int)__builtin_rintf(({3}) * _s) & 255u; "
    "_b0 | (_b1 << 8) | (_b2 << 16) | (_b3 << 24); }})",
    dtypes.uint32, f0, f1, f2, f3, inv)

def _quant_q8_1_kernel(xq:UOp, xds:UOp, x:UOp, tokens:UOp, in_features:UOp) -> UOp:
  G = in_features // 32
  tot_g, gpw = tokens * G, 8
  blk = UOp.range((tot_g + gpw - 1) // gpw, 0, AxisType.GLOBAL)
  lane, wave = UOp.range(32, -1, AxisType.WARP), UOp.range(2, 1, AxisType.LOCAL)
  g_id, sub = blk * gpw + wave * 4 + lane // 8, lane % 8
  ok, g_s = g_id < tot_g, g_id.minimum(tot_g - 1)
  t_i, g_i, z = g_s // G, g_s % G, UOp.const(0.0, dtypes.float32)
  v = _amd_load(x[t_i, g_i * 32 + sub * 4], 4)
  f0, f1, f2, f3 = v[0].float(), v[1].float(), v[2].float(), v[3].float()
  am = _fast_fmax(_fast_fmax(f0.abs(), f1.abs()), _fast_fmax(f2.abs(), f3.abs()))
  sm = (f0 + f1) + (f2 + f3)
  for msk in (2, 1):
    am = _pin_f32(_fast_fmax(am, _lane_xor(am, msk, inline=False)))
    sm = _pin_f32(sm + _lane_xor(sm, msk, inline=False))
  sm_other = _pin_f32(_lane_xor(sm, 4, inline=False))
  am = _pin_f32(_fast_fmax(am, _lane_xor(am, 4, inline=False)))
  d = am * UOp.const(1.0 / 127.0, dtypes.float32)
  inv = (am > UOp.const(1e-30, dtypes.float32)).where(UOp.const(127.0, dtypes.float32) * _fast_rcp(am), z)
  return UOp.group(
    xq[t_i.valid(ok), g_i, sub].store(_pack4_s8(f0, f1, f2, f3, inv)),
    xds[t_i.valid(ok & sub.eq(0)), g_i, 0].store(d),
    xds[t_i.valid(ok & sub.eq(0)), g_i, 1].store(sm),
    xds[t_i.valid(ok & sub.eq(0)), g_i, 2].store(sm_other),
  ).end(lane, wave, blk).sink(arg=KernelInfo(name="amd_quant_q8_1", opts_to_apply=()))

def _sdot4_group_dot(raw:UOp, base:UOp, sub:UOp, qt:int, grid:UOp|None, wr:UOp, sc_r:UOp,
                     stream_w:bool=True, lane:UOp|None=None, deps:tuple[UOp, ...]=()) -> UOp:
  z = UOp.const(0.0, dtypes.float32)
  def ld(p:UOp, lanes:int|None=None, st:bool|None=None) -> UOp:
    s_ = stream_w if st is None else st
    return _amd_load(p, lanes, stream=s_) if lanes is not None else p.load(arg="nontemporal" if s_ else None)
  def st_v(v:UOp) -> UOp:
    arr = UOp.alloc(v.shape, dtypes.uint32, addrspace=AddrSpace.REG)
    return arr.after(arr.after(*deps).store(v))
  def fill8(fn_w:Callable[[UOp], UOp]) -> UOp:
    j = UOp.range(8, 22)
    return wr.after(*deps)[j].store(fn_w(j)).end(j)
  def put_sc(w_st:UOp, a0:UOp, a1:UOp, b0:UOp, b1:UOp) -> UOp: return UOp.group(w_st, sc_r.after(*deps).store(UOp.stack(a0, a1, b0, b1)))
  if qt == Q8_0:
    ws = ld(raw[base + 1], 8).bitcast(dtypes.uint32).cat(ld(raw[base + 9], 8).bitcast(dtypes.uint32))
    sc = _half(ld(raw[base], st=False))
    return put_sc(wr.after(*deps).store(ws), sc, sc, z, z)
  if qt in (IQ4_XS, IQ4_NL):
    pv = ld(raw[base + 2 + sub * 4], 4) if qt == IQ4_XS else ld(raw[base + 1], 8).bitcast(dtypes.uint32)
    sc = _iq4_scale(raw, base, sub, stream=True) if qt == IQ4_XS else _half(ld(raw[base], st=False))
    pr = st_v(pv)
    return put_sc(fill8(lambda j: _nibble_lut(pr[j % 4].load(), _sh((j // 4) * 4), IQ4NL_LUT)), sc, sc, z, z)
  if qt in (Q4_K, Q5_K):
    qr = st_v(ld(raw[base + (4 if qt == Q4_K else 12) + (sub // 2) * 8], 8))
    hr = st_v(ld(raw[base + 4], 8)) if qt == Q5_K else None
    sh, (d, dmin, sc, mn) = ((sub & 1) * 4).cast(dtypes.uint32), _q5_scales(raw, base, sub, stream=True)
    ds, dm = d * sc, dmin * mn
    w_st = fill8(lambda j: ((qr[j].load() >> sh) & 0x0f0f0f0f) |
                 ((((hr[j].load() >> sub.cast(dtypes.uint32)) & 0x01010101) << 4) if hr is not None else 0))
    return put_sc(w_st, ds, ds, -dm, -dm)
  if qt == Q6_K:
    ql_b, qh_b = base + (sub // 4) * 32 + (sub % 2) * 16, base + 64 + (sub // 4) * 16
    ql_sh, qh_sh = ((sub % 4 // 2) * 4).cast(dtypes.uint32), ((sub % 4) * 2).cast(dtypes.uint32)
    qlr = st_v(ld(raw[ql_b], 8).bitcast(dtypes.uint32).cat(ld(raw[ql_b + 8], 8).bitcast(dtypes.uint32)))
    qhr = st_v(ld(raw[qh_b], 8).bitcast(dtypes.uint32).cat(ld(raw[qh_b + 8], 8).bitcast(dtypes.uint32)))
    sc_w, d_sc = ld(raw[base + 96 + sub], st=False), _half(ld(raw[base + 104], st=False))
    sc0 = (sc_w & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float() * d_sc
    sc1 = ((sc_w >> 8) & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float() * d_sc
    c60, c80 = UOp.const(0x60606060, dtypes.uint32), UOp.const(0x80808080, dtypes.uint32)
    w_st = fill8(lambda j: ((((qlr[j].load() >> ql_sh) & 0x0f0f0f0f) | (((qhr[j].load() >> qh_sh) & 0x03030303) << 4)) + c60) ^ c80)
    return put_sc(w_st, sc0, sc1, z, z)
  if qt == Q3_K:
    assert lane is not None
    b0, b4 = (lane // 8) * 8, (lane // 4) * 4
    hm_m = ld(raw[base + sub * 2], 2).bitcast(dtypes.uint32)[0]
    qs_p = ld(raw[base + 16 + (sub // 4) * 16 + (sub % 4) * 4], 4).bitcast(dtypes.uint32)
    sl16, sh16 = ld(raw[base + 48 + (sub % 4)], st=False).cast(dtypes.uint32), ld(raw[base + 52 + (sub % 2)], st=False).cast(dtypes.uint32)
    qs_sh, hm_sh, d_sc = ((sub % 4) * 2).cast(dtypes.uint32), sub.cast(dtypes.uint32), _half(ld(raw[base + 54], st=False))
    s = [((((sl16 >> (h * 8 + (sub // 4) * 4).cast(dtypes.uint32)) & 15) |
           (((sh16 >> (h * 8 + (sub // 2) * 2).cast(dtypes.uint32)) & 3) << 4)).cast(dtypes.int32) - 32).float() * d_sc for h in (0, 1)]
    c7c, c80 = UOp.const(0x7c7c7c7c, dtypes.uint32), UOp.const(0x80808080, dtypes.uint32)
    w_st = fill8(lambda j: ((((_lane_bpermute(b4 + j // 2, (j % 2).eq(0).where(qs_p[0], qs_p[1])) >> qs_sh) & 0x03030303) |
                             (((_lane_bpermute(b0 + j, hm_m) >> hm_sh) & 0x01010101) << 2)) + c7c) ^ c80)
    return put_sc(w_st, s[0], s[1], z, z)
  if qt == Q2_K:
    qsr = st_v(ld(raw[base + 8 + (sub // 4) * 16], 8).bitcast(dtypes.uint32).cat(ld(raw[base + 16 + (sub // 4) * 16], 8).bitcast(dtypes.uint32)))
    sh, dm, sc_w = ((sub % 4) * 2).cast(dtypes.uint32), ld(raw[base + 40], 2, st=False), ld(raw[base + sub], st=False).cast(dtypes.uint32)
    dh, dmh, sc0, sc1 = _half(dm[0]), _half(dm[1]), sc_w & 255, (sc_w >> 8) & 255
    ds0, dm0, ds1, dm1 = (sc0 & 15).float() * dh, (sc0 >> 4).float() * dmh, (sc1 & 15).float() * dh, (sc1 >> 4).float() * dmh
    return put_sc(fill8(lambda j: (qsr[j].load() >> sh) & 0x03030303), ds0, ds1, -dm0, -dm1)
  if qt == MXFP4:
    pr, sc = st_v(ld(raw[base + 1], 16).bitcast(dtypes.uint32)), _e8m0_halved(ld(raw[base], st=False))
    return put_sc(fill8(lambda j: _nibble_lut(pr[j % 4].load(), _sh((j // 4) * 4), MXFP4_LUT)), sc, sc, z, z)
  if qt == NVFP4:
    h64 = sub % 2
    sc_w, pr = ld(raw[base], st=False), st_v(ld(raw[base + 1 + h64 * 4], 4))
    sc0, sc1 = _ue4m3_halved((sc_w >> _sh(h64 * 16)) & 255), _ue4m3_halved((sc_w >> _sh(h64 * 16 + 8)) & 255)
    return put_sc(fill8(lambda j: _nibble_lut(pr[(j // 4) * 2 + (j % 2)].load(), _sh(((j % 4) // 2) * 4), MXFP4_LUT)), sc0, sc1, z, z)
  assert grid is not None
  if qt == IQ3_XXS:
    sc_w = ld(raw[base + 33 + sub * 2], 2).bitcast(dtypes.uint32)[0]
    qv = ld(raw[base + 1 + sub * 4], 4).bitcast(dtypes.uint32)
    sc = _half(ld(raw[base], st=False)) * ((sc_w >> 28).float() + 0.5) * 0.5
    w_st = fill8(lambda j: _iq_signed_word(grid[(((j < 4).where(qv[0], qv[1]) >> _sh(((j // 2) & 1) * 16 + (j % 2) * 8)) & 255)].load(),
                 _iq_even_signs((sc_w >> _sh(7 * (j // 2))) & 127) >> _sh((j % 2) * 4)))
    return put_sc(w_st, sc, sc, z, z)
  if qt == IQ3_S:
    qh_u = (ld(raw[base + 33 + sub // 2], st=False).cast(dtypes.uint32) >> _sh((sub & 1) * 8)) & 255
    sc = _half(ld(raw[base], st=False)) * (1 + 2 * ((ld(raw[base + 53 + sub // 4], st=False).cast(dtypes.uint32) >> _sh((sub & 3) * 4)) & 15)).float()
    qs_v, sg_w = ld(raw[base + 1 + sub * 4], 4).bitcast(dtypes.uint32), ld(raw[base + 37 + sub * 2], 2).bitcast(dtypes.uint32)[0]
    w_st = fill8(lambda j: _iq_signed_word(grid[(((j < 4).where(qs_v[0], qs_v[1]) >> _sh((j % 4) * 8)) & 255) |
                 (((qh_u >> _sh(j)) & 1) << 8)].load(), (sg_w >> _sh(j * 4)) & 15))
    return put_sc(w_st, sc, sc, z, z)
  if qt == IQ1_S:
    qh_u, qs_w = ld(raw[base + 17 + sub], st=False).cast(dtypes.uint32), ld(raw[base + 1 + sub * 2], 2).bitcast(dtypes.uint32)[0]
    sc, delta = (((qh_u >> 12) & 7) * 2 + 1).float() * _half(ld(raw[base], st=False)), (qh_u >> 15).eq(0).where(-0.125, 0.125)
    w_st = fill8(lambda j: grid[(((qs_w >> _sh((j // 2) * 8)) & 255) | (((qh_u >> _sh(3 * (j // 2))) & 7) << 8)) * 2 + (j % 2)].load())
    b = -delta * sc
    return put_sc(w_st, sc, sc, b, b)
  if qt == IQ2_S:
    qh_u = (ld(raw[base + 33 + sub // 2], st=False).cast(dtypes.uint32) >> _sh((sub & 1) * 8)) & 255
    sc_b = (ld(raw[base + 37 + sub // 2], st=False).cast(dtypes.uint32) >> _sh((sub & 1) * 8)) & 255
    q_w, s_w = ld(raw[base + 1 + sub * 2], 2).bitcast(dtypes.uint32)[0], ld(raw[base + 17 + sub * 2], 2).bitcast(dtypes.uint32)[0]
    d_b = _half(ld(raw[base], st=False)) * 0.25
    sf0, sf1 = ((sc_b & 15).float() + 0.5) * d_b, (((sc_b >> 4) & 15).float() + 0.5) * d_b
    w_st = fill8(lambda j: _iq_signed_word(grid[(((q_w >> _sh((j // 2) * 8)) & 255) | (((qh_u >> _sh(2 * (j // 2))) & 3) << 8)) * 2 + (j % 2)].load(),
                 ((s_w >> _sh((j // 2) * 8)) & 255) >> _sh((j % 2) * 4)))
    return put_sc(w_st, sf0, sf1, z, z)
  if qt == IQ2_XS:
    sc_b, d_b = (ld(raw[base + 33 + sub // 2], st=False).cast(dtypes.uint32) >> _sh((sub & 1) * 8)) & 255, _half(ld(raw[base], st=False)) * 0.25
    sf0, sf1 = ((sc_b & 15).float() + 0.5) * d_b, (((sc_b >> 4) & 15).float() + 0.5) * d_b
    w_st = fill8(lambda j: (pk := ld(raw[base + 1 + sub * 4 + j // 2]).cast(dtypes.uint32),
                 _iq_signed_word(grid[(pk & 511) * 2 + (j % 2)].load(), _iq_even_signs(pk >> 9) >> _sh((j % 2) * 4)))[1])
    return put_sc(w_st, sf0, sf1, z, z)
  qs_w, sg_w = ld(raw[base + 1 + sub * 4], 2).bitcast(dtypes.uint32)[0], ld(raw[base + 3 + sub * 4], 2).bitcast(dtypes.uint32)[0]
  sc = _half(ld(raw[base], st=False)) * ((sg_w >> 28).float() + 0.5) * 0.25
  w_st = fill8(lambda j: _iq_signed_word(grid[((qs_w >> _sh((j // 2) * 8)) & 255) * 2 + (j % 2)].load(),
               _iq_even_signs((sg_w >> _sh(7 * (j // 2))) & 127) >> _sh((j % 2) * 4)))
  return put_sc(w_st, sc, sc, z, z)

def _sdot4_gemv_kernel(out:UOp, raw:UOp, xq:UOp, xds:UOp, grids:tuple[UOp, ...], out_features:UOp, in_features:UOp, qt:int|UOp,
                       w_rows:UOp, waves_k:UOp, group_rows:UOp|None=None) -> UOp:
  tokens, G = out.shape[0], in_features // 32
  assert isinstance(tokens, UOp)
  qtypes = (qt,) if isinstance(qt, int) else tuple(QUANT_NAMES)
  wg_groups, t_per_wg = ((group_rows > 0).where(tokens // group_rows.maximum(1), 1),
                         (group_rows > 0).where(group_rows, tokens)) if group_rows is not None else (tokens, UOp.const(1, dtypes.weakint))
  threads, row_blks = waves_k * w_rows * 32, out_features // w_rows
  steps = (G + waves_k * 32 - 1) // (waves_k * 32)
  rb = UOp.range(wg_groups * row_blks, 0, AxisType.GLOBAL)
  lane, wave_k, w_row = UOp.range(32, -1, AxisType.WARP), UOp.range(waves_k, 1, AxisType.LOCAL), UOp.range(w_rows, 2, AxisType.LOCAL)
  tid, wg_g, rblk = (w_row * waves_k + wave_k) * 32 + lane, rb // row_blks, rb % row_blks
  row_i = rblk * w_rows + w_row
  need_xs:bool|UOp = (qt in (Q4_K, Q5_K, Q2_K, IQ1_S)) if isinstance(qt, int) else (qt.eq(Q4_K) | qt.eq(Q5_K) | qt.eq(Q2_K) | qt.eq(IQ1_S))
  grid = grids[0] if grids else None
  grid_lds = _load_grid_lds(grid, qt, tid, threads, expand_iq3x=False) if grid is not None else None
  z, z32i = UOp.const(0.0, dtypes.float32), UOp.const(0, dtypes.int32)
  acc = UOp.alloc((t_per_wg,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0.0)))
  s = UOp.range(steps, 5)
  grp_raw = lane + (wave_k + s * waves_k) * 32
  valid_g, grp = grp_raw < G, grp_raw.minimum(G - 1)
  wr, sc_r = UOp.alloc((8,), dtypes.uint32, addrspace=AddrSpace.REG), UOp.alloc((4,), dtypes.float32, addrspace=AddrSpace.REG)
  def _fill_sdot(q:int) -> UOp:
    rq = _qraw(raw, q)
    base = _gemv_base(q, rq.dtype.itemsize, row_i, grp, out_features, in_features, wg_g, group_rows)
    return _sdot4_group_dot(rq, base, grp % 8, q, grid_lds if q in _LDS_GRID_QUANTS else grid, wr, sc_r, stream_w=True, lane=lane, deps=(s,))
  w_done = _qswitch(qt, [(q, _fill_sdot(q)) for q in qtypes])
  wr_r, sc_v = wr.after(w_done), sc_r.after(w_done)
  t_i = UOp.range(t_per_wg, 6)
  tok_i = wg_g * t_per_wg + t_i
  v0, v1 = _amd_load(xq[tok_i, grp, 0], 4), _amd_load(xq[tok_i, grp, 4], 4)
  xq_buf = UOp.alloc((8,), dtypes.uint32, addrspace=AddrSpace.REG)
  xq_r = xq_buf.after(xq_buf.after(s, t_i).store(v0.cat(v1)))
  xd_v = xds[tok_i, grp, 0].load()
  xs0_v = xds[tok_i, grp, 1].load() if need_xs is True else (z if need_xs is False else need_xs.where(xds[tok_i, grp, 1].load(), z))
  xs1_v = xds[tok_i, grp, 2].load() if need_xs is True else (z if need_xs is False else need_xs.where(xds[tok_i, grp, 2].load(), z))
  s0r, s1r = UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG), UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG)
  s0r, s1r = s0r.after(s0r.after(s, t_i, w_done)[0].store(z32i)), s1r.after(s1r.after(s, t_i, w_done)[0].store(z32i))
  jd = UOp.range(4, 23)
  d_upd = UOp.group(s0r[0].store(_sudot4(wr_r[jd].load(), xq_r[jd].load(), s0r.after(jd)[0].load())),
                    s1r[0].store(_sudot4(wr_r[jd + 4].load(), xq_r[jd + 4].load(), s1r.after(jd)[0].load()))).end(jd)
  delta = ((s0r.after(d_upd)[0].load().float() * sc_v[0].load() + s1r.after(d_upd)[0].load().float() * sc_v[1].load()) * xd_v +
           sc_v[2].load() * xs0_v + sc_v[3].load() * xs1_v)
  upd = acc[t_i].store(acc.after(s)[t_i].load() + valid_g.where(delta, z)).end(t_i, s)
  acc_done, has_wk = acc.after(upd), (waves_k > 1) if isinstance(waves_k, UOp) else UOp.const(waves_k > 1, dtypes.bool)
  r_w1, t_o1 = UOp.range(has_wk.logical_not().cast(dtypes.weakint), 29), UOp.range(t_per_wg, 8)
  st_w1 = out[wg_g * t_per_wg + t_o1, row_i.valid(has_wk.logical_not() & lane.eq(0))].store(
    warp_reduce(acc_done[t_o1].load(), full_wave=True).cast(out.dtype)).end(t_o1, r_w1)
  r_wk, t_s, t_o2 = UOp.range(has_wk.cast(dtypes.weakint), 30), UOp.range(t_per_wg, 9), UOp.range(t_per_wg, 10)
  w_lds = UOp.alloc((w_rows, t_per_wg, waves_k), dtypes.float32, addrspace=AddrSpace.LOCAL)
  w_lds_r = w_lds.after(w_lds.after(r_wk)[w_row, t_s, wave_k.valid(has_wk & lane.eq(0))].store(
    warp_reduce(acc_done[t_s].load(), full_wave=True)).end(t_s).barrier())
  wk_r = UOp.range(waves_k, 12)
  st_wk = out[wg_g * t_per_wg + t_o2, row_i.valid(has_wk & wave_k.eq(0) & lane.eq(0))].store(
    w_lds_r[w_row, t_o2, wk_r].load().reduce(wk_r, arg=(Ops.ADD, 0)).cast(out.dtype)).end(t_o2, r_wk)
  kname = f"sdot4_{QUANT_NAMES[qt]}" if isinstance(qt, int) else "sdot4_quant"
  return UOp.group(st_w1, st_wk).end(lane, wave_k, w_row, rb).sink(arg=KernelInfo(name=kname, opts_to_apply=()))

# ******** shared quant word & exact FP16 WMMA dequant ********

def _quant_word(raw:UOp, base:UOp, sub:UOp, i:int|UOp, qt:int, grid:UOp|None) -> UOp:
  if qt == Q2_K: return (_load_u32(raw, base, 16 + (sub // 4) * 32 + (i // 4) * 16 + (i % 4) * 4) >> _sh((sub % 4) * 2)) & 0x03030303
  if qt == Q3_K:
    qs = (_load_u32(raw, base, 32 + (sub // 4) * 32 + (i // 4) * 16 + (i % 4) * 4) >> _sh((sub % 4) * 2)) & 0x03030303
    return qs | (((_load_u32(raw, base, (i // 4) * 16 + (i % 4) * 4) >> _sh(sub)) & 0x01010101) << 2)
  if qt == Q6_K:
    ql = (_load_u32(raw, base, (sub // 4) * 64 + (sub % 2) * 32 + (i // 4) * 16 + (i % 4) * 4) >> _sh((sub % 4 // 2) * 4)) & 0x0f0f0f0f
    return ql | (((_load_u32(raw, base, 128 + (sub // 4) * 32 + (i // 4) * 16 + (i % 4) * 4) >> _sh((sub % 4) * 2)) & 0x03030303) << 4)
  if qt == IQ4_NL: return _nibble_lut(_load_u32(raw, base, 2 + (i % 4) * 4), (i // 4) * 4, IQ4NL_LUT)
  if qt == NVFP4:
    return _nibble_lut(_load_u32(raw, base, 4 + (sub % 2) * 16 + (i // 4) * 8 + (i % 2) * 4), ((i // 2) % 2) * 4, MXFP4_LUT)
  assert grid is not None
  if qt == IQ3_S:
    return _iq_signed_word(grid[_load_byte(raw, base, 2 + sub * 8 + i) | (((_load_byte(raw, base, 66 + sub) >> _sh(i)) & 1) << 8)].load(),
                           _load_byte(raw, base, 74 + sub * 4 + i // 2) >> _sh((i % 2) * 4))
  if qt == IQ3_XXS:
    return _iq_signed_word(grid[_load_byte(raw, base, 2 + sub * 8 + i)].load(),
                           _iq_even_signs((_load_u32(raw, base, 66 + sub * 4) >> _sh((i // 2) * 7)) & 127) >> _sh((i % 2) * 4))
  if qt == IQ2_XXS:
    return _iq_signed_word(grid[_load_byte(raw, base, 2 + sub * 8 + i // 2) * 2 + i % 2].load(),
                           _iq_even_signs((_load_u32(raw, base, 6 + sub * 8) >> _sh((i // 2) * 7)) & 127) >> _sh((i % 2) * 4))
  if qt == IQ2_XS:
    p = raw[base + 1 + sub * 4 + i // 2].cast(dtypes.uint32)
    return _iq_signed_word(grid[(p & 511) * 2 + i % 2].load(), _iq_even_signs(p >> 9) >> _sh((i % 2) * 4))
  if qt == IQ2_S:
    gi = (_load_byte(raw, base, 2 + sub * 4 + i // 2) | (((_load_byte(raw, base, 66 + sub) >> _sh((i // 2) * 2)) & 3) << 8)) * 2 + i % 2
    return _iq_signed_word(grid[gi].load(), _load_byte(raw, base, 34 + sub * 4 + i // 2) >> _sh((i % 2) * 4))
  gi = (_load_byte(raw, base, 2 + sub * 4 + i // 2) | (((raw[base + 17 + sub].cast(dtypes.uint32) >> _sh((i // 2) * 3)) & 7) << 8)) * 2 + i % 2
  return grid[gi].load()

def _make_wmma_dequant(raw:UOp, qt:int, lane_group:UOp, kgroups:int, k_u:UOp, table:UOp|None) -> Callable[[UOp, UOp, UOp], UOp|tuple[UOp, ...]]:
  w_u, b_u = k_u // 4, k_u % 4
  w_idx = w_u if kgroups == 1 else (lane_group + kgroups * w_u)
  word_indices: tuple[int|UOp, ...] = tuple(range(4)) if kgroups == 1 else tuple(lane_group + kgroups * j for j in range(4 // kgroups))
  def dequant(base:UOp, sub:UOp, half:UOp) -> UOp|tuple[UOp, ...]:
    if qt in (Q4_K, Q5_K):
      d, dmin, sc, mn = _q5_scales(raw, base, sub, stream=False)
      w = (raw[base + (4 if qt == Q4_K else 12) + (sub // 2) * 8 + half * 4 + w_idx].load() >> _sh((sub & 1) * 4)) & 0x0f0f0f0f
      if qt == Q5_K: w |= ((raw[base + 4 + half * 4 + w_idx].load() >> _sh(sub)) & 0x01010101) << 4
      return (((w >> _sh(b_u * 8)) & 255).float() * (d * sc) - dmin * mn).cast(dtypes.float16)
    if qt == IQ4_XS:
      nib = (raw[base + 2 + sub * 4 + w_idx].load() >> _sh(b_u * 8 + half * 4)) & 15
      return (_lut16_i8(nib, IQ4NL_LUT) * _iq4_scale(raw, base, sub)).cast(dtypes.float16)
    if qt in SMALL_BLOCK_QUANTS: base = _small_block(base, sub, qt)
    if qt == IQ4_NL:
      nib = (_load_byte(raw, base, 2 + w_idx * 4 + b_u) >> _sh(half * 4)) & 15
      return (_lut16_i8(nib, IQ4NL_LUT) * _half(raw[base])).cast(dtypes.half)
    if qt == NVFP4:
      nib = (_load_byte(raw, base, 4 + (sub % 2) * 16 + half * 8 + (w_idx % 2) * 4 + b_u) >> _sh((w_idx // 2) * 4)) & 15
      return (_lut16_i8(nib, MXFP4_LUT) * _ue4m3_halved(_load_byte(raw, base, (sub % 2) * 2 + half))).cast(dtypes.half)
    if qt == MXFP4:
      if kgroups != 4:
        b_m = (raw[base + 1 + w_idx * 4 + b_u].cast(dtypes.uint32) >> _sh(half * 4)) & 15
        return (_lut16_i8(b_m, MXFP4_LUT) * _e8m0_halved(raw[base].cast(dtypes.uint32))).cast(dtypes.half)
      e = raw[base].cast(dtypes.uint32)
      small = e < 116
      c_lo = small.where(UOp.const(0x26242000, dtypes.uint32), UOp.const(0x36343000, dtypes.uint32))
      c_hi = small.where(UOp.const(0x2e2c2a28, dtypes.uint32), UOp.const(0x3e3c3a38, dtypes.uint32))
      sc2 = _e8m0_halved(e + small.where(UOp.const(7, dtypes.uint32), UOp.const(3, dtypes.uint32))).cast(dtypes.half).reshape((1,)).expand((2,))
      z_u, s01, s23, out_v = UOp.const(0, dtypes.uint32), UOp.const(0x05000400, dtypes.uint32), UOp.const(0x07000600, dtypes.uint32), list[UOp]()
      for wi in word_indices:
        pw = _load_u32(raw, base, 1 + wi * 4) >> _sh(half * 4)
        fpb = _amd_byte_perm(c_hi, c_lo, pw & 0x07070707) | ((pw & 0x08080808) << 4)
        v01 = _amd_byte_perm(fpb, z_u, s01).reshape((1,)).bitcast(dtypes.half) * sc2
        v23 = _amd_byte_perm(fpb, z_u, s23).reshape((1,)).bitcast(dtypes.half) * sc2
        out_v.extend((v01[0], v01[1], v23[0], v23[1]))
      return tuple(out_v)
    if qt == Q8_0:
      if kgroups == 2:
        return (_load_byte(raw, base, 2 + half * 16 + w_idx * 4 + b_u).cast(dtypes.uint8).bitcast(dtypes.int8).float() *
                _half(raw[base])).cast(dtypes.half)
      sc2 = raw[base].load().bitcast(dtypes.half).reshape((1,)).expand((2,))
      neg, c64 = UOp.const(-1152.0, dtypes.half).reshape((1,)).expand((2,)), UOp.const(0x64006400, dtypes.uint32)
      return tuple(v[k] for w in word_indices for p in range(2)
                   for u in (raw[base + 1 + half * 8 + w * 2 + p].load().cast(dtypes.uint32) ^ UOp.const(0x8080, dtypes.uint32),)
                   for v in ((_amd_byte_perm(c64, u, UOp.const(0x05010500, dtypes.uint32)).reshape((1,)).bitcast(dtypes.half) + neg) * sc2,)
                   for k in (0, 1))
    def byte(o:int|UOp) -> UOp: return _load_byte(raw, base, o)
    zero: int|UOp = 0
    minimum = UOp.const(0, dtypes.float)
    if qt == Q2_K:
      sc = byte(sub * 2 + half)
      scale, minimum = _half(raw[base + 40]) * (sc & 15).float(), _half(raw[base + 41]) * (sc >> 4).float()
    elif qt == Q3_K:
      j = sub * 2 + half
      sc_q3 = (((byte(96 + j % 8) >> _sh((j // 8) * 4)) & 15) | (((byte(104 + j % 4) >> _sh((j // 4) * 2)) & 3) << 4)).cast(dtypes.int32) - 32
      scale, zero = _half(raw[base + 54]) * sc_q3.float(), 4
    elif qt == Q6_K: scale, zero = _half(raw[base + 104]) * byte(192 + sub * 2 + half).cast(dtypes.uint8).bitcast(dtypes.int8).float(), 32
    elif qt == IQ1_S: scale, zero = _iq1_scale(raw, base, sub)
    else:
      scale = _half(raw[base])
      if qt in (IQ2_XS, IQ2_S): scale *= (((byte((66 if qt == IQ2_XS else 74) + sub) >> _sh(half * 4)) & 15).float() + 0.5) * 0.25
      elif qt == IQ3_S: scale *= (1 + 2 * ((byte(106 + sub // 2) >> _sh((sub % 2) * 4)) & 15)).float()
      elif qt == IQ3_XXS: scale *= ((_load_u32(raw, base, 66 + sub * 4) >> 28).float() + 0.5) * 0.5
      elif qt == IQ2_XXS: scale *= ((_load_u32(raw, base, 6 + sub * 8) >> 28).float() + 0.5) * 0.25
    w = _quant_word(raw, base, sub, half * 4 + w_idx, qt, table)
    return (((w >> _sh(b_u * 8)).cast(dtypes.uint8).bitcast(dtypes.int8).float() - zero) * scale - minimum).cast(dtypes.half)
  return dequant

@functools.cache
def _quant_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, *grids:UOp, m_blks:UOp, m_tiles:UOp, out_features:UOp, in_features:UOp,
                                  ggml_type:int|UOp, kgroups:int, lanes:int, qtypes:tuple[int, ...]|None=None) -> UOp:
  ob, tb = UOp.range(out_features // 16, 0, AxisType.GLOBAL), UOp.range(m_blks, 1, AxisType.GLOBAL)
  lane = UOp.range(lanes, -1, axis_type=AxisType.WARP)
  col, lg, rows, frag_k = lane % 16, lane // 16, 256 // lanes, 16 // kgroups
  output, grid = ob * 16 + col, None
  qts = qtypes if qtypes is not None else ((ggml_type,) if isinstance(ggml_type, int) else tuple(QUANT_NAMES))
  if grids:
    g_n = grids[0].shape[0]
    grid = UOp.alloc((g_n,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    g_i, g4 = UOp.range((g_n + lanes * 4 - 1) // (lanes * 4), 20), UOp.range(4, 21, AxisType.UPCAST)
    g_off = (lane + g_i * lanes) * 4 + g4
    grid = grid.after(grid[g_off.valid(g_off < g_n)].store(grids[0][g_off.valid(g_off < g_n)].load()).end(g4, g_i))
  grp, half = UOp.range(in_features // Q8_GROUP_SIZE, 4), UOp.range(2, 5)
  k_u, m_u, mt_u = UOp.range(frag_k, 10, AxisType.UPCAST), UOp.range(rows, 11, AxisType.UPCAST), UOp.range(m_tiles, 12, AxisType.UPCAST)
  w_idx = (k_u // 4) if kgroups == 1 else (lg + kgroups * (k_u // 4))
  a_el = x[(tb * m_tiles + mt_u) * 16 + col, (grp * 2 + half) * 16 + w_idx * 4 + (k_u % 4)].load()
  sb_idx = output * (in_features // GGML_BLOCK_SIZE) + grp // 8
  def _b_for(q:int) -> UOp:
    rq = _qraw(raw, q)
    if q in SMALL_BLOCK_QUANTS:
      g, bw = SMALL_BLOCK_QUANTS[q]
      base, sub = (output * (in_features // (Q8_GROUP_SIZE * g)) + grp // g) * bw, grp % g if g > 1 else UOp.const(0, dtypes.weakint)
    else: base, sub = sb_idx * (QUANT_SIZES[q] // rq.dtype.itemsize), grp % 8
    b_raw = _make_wmma_dequant(rq, q, lg, kgroups, k_u, grid if q in _WMMA_TBL_QUANTS else None)(base, sub, half)
    return ((br := UOp.alloc((frag_k,), dtypes.float16, addrspace=AddrSpace.REG)).after(
      br.after(grp, half).store(UOp.stack(*b_raw)))[k_u].load() if isinstance(b_raw, tuple) else b_raw)
  b_el = _qswitch(ggml_type, [(q, _b_for(q)) for q in qts])
  tc_axes = (((k_u.arg, frag_k), (m_u.arg, 1)), ((k_u.arg, frag_k), (m_u.arg, 1)), ((k_u.arg, 1), (m_u.arg, rows)))
  res = UOp.wmma(a_el, b_el, UOp.const((0.0,) * rows, dtypes.float32), WMMA_ARG[0], lanes, tc_upcast_axes=tc_axes).reduce(grp, half, arg=(Ops.ADD, 0))
  kname = f"linear_{QUANT_NAMES[ggml_type]}_f16_wmma" if isinstance(ggml_type, int) else "linear_quant_f16_wmma"
  return UOp(Ops.STORE, src=(out[(tb * m_tiles + mt_u) * 16 + (lg * rows + m_u if kgroups > 1 else m_u * 2 + lg), output], res.cast(out.dtype))).end(
    k_u, m_u, mt_u, tb, ob, lane).sink(arg=KernelInfo(name=kname, optimized=True))

# ******** RDNA3 Direct-A + Double-Buffered-B 2D FP16 WMMA (with in-WG K-split) ********

@functools.cache
def _rdna3_gemm_2d_kernel(out:UOp, raw:UOp, x:UOp, *grids:UOp, m_blks:UOp, n_blks:UOp, wm:UOp, wn:UOp, tm:UOp, tn:UOp,
                          bk:UOp, out_features:UOp, in_features:UOp, ggml_type:int|UOp, wk:int|UOp=1) -> UOp:
  qts = (ggml_type,) if isinstance(ggml_type, int) else tuple(QUANT_NAMES)
  bm, bn, waves = wm * tm * 16, wn * tn * 16, wm * wn * wk
  threads, k_groups, k8_cnt, k16_cnt = waves * 32, bk // 32, bk // 8, bk // 16
  k16_pw = k16_cnt // wk
  b_total, total_kt = bn * k_groups, (in_features // bk).maximum(1)
  b_iters, swz_m = (b_total + threads - 1) // threads, (k8_cnt - 1).minimum(7)
  x_u32 = x.flatten().bitcast(dtypes.uint32)
  gid = UOp.range(m_blks * n_blks, 0, AxisType.GLOBAL)
  swz = (m_blks % 4).eq(0).where(4, (m_blks % 2).eq(0).where(2, 1))
  grp, in_grp = gid // (swz * n_blks), gid % (swz * n_blks)
  block_m, block_n = grp * swz + in_grp % swz, in_grp // swz
  lane, wave = UOp.range(32, -1, AxisType.WARP), UOp.range(waves, 2, AxisType.LOCAL)
  tid, col, lg, wave_k, wave_mn = wave * 32 + lane, lane % 16, lane // 16, wave % wk, wave // wk
  wave_m, wave_n = wave_mn // wn, wave_mn % wn
  grid = _load_grid_lds(grids[0], ggml_type, tid, threads) if grids else None
  b_elems = bn * k8_cnt * 8
  has_wk = wk > 1 if isinstance(wk, UOp) else UOp.const(wk > 1, dtypes.bool)
  red_f32_elems = has_wk.where(wm * wn * wk * tm * tn * 256, 0)
  b_all_halves = (2 * b_elems).maximum(2 * red_f32_elems)
  B_lds_raw = UOp.alloc((b_all_halves,), dtypes.half, addrspace=AddrSpace.LOCAL)
  B_lds = B_lds_raw[:2 * b_elems].reshape(2, bn, k8_cnt, 8)
  B_h2, acc = B_lds.reshape(2, bn, k8_cnt, 4, 2), UOp.alloc((tm, tn, 8), dtypes.float32, addrspace=AddrSpace.REG)
  acc, exact_q = acc.after(acc.store(acc.const_like(0.0))), out_features < 32

  def load_b_reg_q(q:int, kt:UOp, dep:tuple[UOp, ...], rid:int) -> tuple[UOp, ...]:
    if q not in _R3_FAST_B_QUANTS: return ()
    rq = _qraw(raw, q)
    tw = QUANT_SIZES[q] // rq.dtype.itemsize
    ow = in_features * tw // GGML_BLOCK_SIZE
    stride = {Q4_K: 6, IQ4_XS: 6, Q8_0: 9, MXFP4: 5, IQ3_XXS: 4, IQ2_S: 5}[q]
    buf, bi = UOp.alloc((b_iters, stride), dtypes.uint32, addrspace=AddrSpace.REG), UOp.range(b_iters, rid)
    bp = (tid + bi * threads).minimum(b_total - 1)
    gn = (block_n * bn + bp // k_groups).minimum(out_features - 1)
    gbg = (kt * k_groups + bp % k_groups).minimum(in_features // 32 - 1)
    if q == Q4_K:
      base, sub = (gn * (in_features // 256) + gbg // 8) * 36, gbg % 8
      ds0, dm0, ds1, dm1 = _q4k_pair_scales(_v4(_amd_load(rq[base], 4)), sub // 2)
      vs = _v4(_amd_load(rq[base + 4 + (sub // 2) * 8 + (sub & 1) * 4], 4)) + [
        UOp.stack(ds0.cast(dtypes.half), (-dm0).cast(dtypes.half)).bitcast(dtypes.uint32),
        UOp.stack(ds1.cast(dtypes.half), (-dm1).cast(dtypes.half)).bitcast(dtypes.uint32)]
    elif q == IQ4_XS:
      sub, base = gbg % 8, (gn * (in_features // 256) + gbg // 8) * 34
      hdr = _amd_load(rq[base], 2)
      vs = _v4(_amd_load(rq[base + 2 + sub * 4], 4)) + [hdr[0], hdr[1]]
    elif q == Q8_0:
      wb = (gn * (in_features // 32) + gbg) * 17
      vs = (_v4(_amd_load(rq[wb + 1], 8).bitcast(dtypes.uint32)) +
            _v4(_amd_load(rq[wb + 9], 8).bitcast(dtypes.uint32)) + [rq[wb].load().cast(dtypes.uint32)])
    elif q == MXFP4:
      wb = (gn * (in_features // 32) + gbg) * 17
      vs = _v4(_amd_load(rq[wb + 1], 16, stream=True).bitcast(dtypes.uint32)) + [rq[wb].load().cast(dtypes.uint32)]
    elif q == IQ3_XXS:
      sub, base = gbg % 8, gn * ow + (gbg // 8) * tw
      qs = _amd_load(rq[base + 1 + sub * 4], 4).bitcast(dtypes.uint32)
      vs = [qs[0], qs[1], _amd_load(rq[base + 33 + sub * 2], 2).bitcast(dtypes.uint32)[0], rq[base].load().cast(dtypes.uint32)]
    else:  # IQ2_S
      sub, base = gbg % 8, gn * ow + (gbg // 8) * tw
      vs = [_amd_load(rq[base + 1 + sub * 2], 2).bitcast(dtypes.uint32)[0],
            _amd_load(rq[base + 17 + sub * 2], 2).bitcast(dtypes.uint32)[0],
            rq[base + 33 + sub // 2].load().cast(dtypes.uint32),
            rq[base + 37 + sub // 2].load().cast(dtypes.uint32), rq[base].load().cast(dtypes.uint32)]
    return (buf.after(UOp.group(*(buf.after(*dep)[bi, j].store(vs[j]) for j in range(stride))).end(bi)),)

  def store_b_lds(buf_i:UOp, br_map:dict[int, tuple[UOp, ...]], kt:UOp, dep:tuple[UOp, ...], rid:int) -> UOp:
    fast_qts, slow_qts = [q for q in qts if q in _R3_FAST_B_QUANTS], [q for q in qts if q not in _R3_FAST_B_QUANTS]
    def run_fast() -> UOp:
      bi = UOp.range(b_iters, rid)
      bp_raw = tid + bi * threads
      ok, bp = bp_raw < b_total, bp_raw.minimum(b_total - 1)
      bn_i, bg = bp // k_groups, bp % k_groups
      bn_v, gn, gbg = bn_i.valid(ok), (block_n * bn + bn_i).minimum(out_features - 1), (kt * k_groups + bg).minimum(in_features // 32 - 1)
      dst = B_h2.after(*dep)[buf_i]
      c64, c80, slo, shi = (UOp.const(v, dtypes.uint32) for v in (0x64006400, 0x80808080, 0x05010500, 0x05030502))
      n1152 = UOp.const(0xe480e480, dtypes.uint32).reshape((1,)).bitcast(dtypes.half)
      n1024 = UOp.const(0xe400e400, dtypes.uint32).reshape((1,)).bitcast(dtypes.half)
      tmp_h2 = UOp.alloc((2, 4, 2, 2), dtypes.half, addrspace=AddrSpace.REG)
      td = tmp_h2.after(*dep, bi)
      def fill_f(q:int) -> UOp:
        rq, reg = _qraw(raw, q), br_map[q][0]
        if q == Q4_K:
          sub_q = gbg % 8
          ds0_e, dm0_e, ds1_e, dm1_e = _q4k_pair_scales(_v4(_amd_load(rq[(gn * (in_features // 256) + gbg // 8) * 36], 4)), sub_q // 2)
          m_r, j_r = UOp.range(2, rid + 1), UOp.range(4, rid + 2)
          sp = reg[bi, 4 + m_r].load()
          ds_e, ndm_e = m_r.eq(0).where(ds0_e, ds1_e), m_r.eq(0).where(-dm0_e, -dm1_e)
          ds2, dm2 = _splat_h2(sp), _amd_byte_perm(sp, sp, UOp.const(0x03020302, dtypes.uint32)).reshape((1,)).bitcast(dtypes.half)
          u = reg[bi, j_r].load() >> _sh(m_r * 4)
          u4 = u & 0x0f0f0f0f
          f0, f1 = ((_amd_byte_perm(c64, u4, sl).reshape((1,)).bitcast(dtypes.half) + n1024) * ds2 + dm2 for sl in (slo, shi))
          he = [(((u >> (b * 8)) & 15).float() * ds_e + ndm_e).cast(dtypes.half) for b in range(4)]
          return UOp.group(td[m_r, j_r, 0].store(exact_q.where(UOp.stack(he[0], he[1]), f0)),
                           td[m_r, j_r, 1].store(exact_q.where(UOp.stack(he[2], he[3]), f1))).end(j_r, m_r)
        if q == IQ4_XS:
          sub, h0, h1 = gbg % 8, reg[bi, 4].load(), reg[bi, 5].load()
          sc_f = _half(h0) * ((((h1 >> _sh(4 * sub)) & 15) | (((h0 >> _sh(16 + 2 * sub)) & 3) << 4)).cast(dtypes.int32) - 32).float()
          c4b, zf = UOp.const(0x4b000000, dtypes.uint32), UOp.const(-8388736.0, dtypes.float32)
          k_r, j_r = UOp.range(2, rid + 1), UOp.range(4, rid + 2)
          ux_u = _nibble_lut(reg[bi, j_r].load(), _sh(k_r * 4), IQ4NL_U8_LUT)
          hs = [((_amd_byte_perm(c4b, ux_u, UOp.const(0x07060500 + b, dtypes.uint32)).bitcast(dtypes.float32) + zf) * sc_f).cast(dtypes.half)
                for b in range(4)]
          return UOp.group(td[k_r, j_r, 0].store(UOp.stack(hs[0], hs[1])), td[k_r, j_r, 1].store(UOp.stack(hs[2], hs[3]))).end(j_r, k_r)
        if q == Q8_0:
          sc2, k_r, j_r = _splat_h2(reg[bi, 8].load()), UOp.range(2, rid + 1), UOp.range(4, rid + 2)
          w = reg[bi, k_r * 4 + j_r].load() ^ c80
          h0, h1 = ((_amd_byte_perm(c64, w, sl).reshape((1,)).bitcast(dtypes.half) + n1152) * sc2 for sl in (slo, shi))
          return UOp.group(td[k_r, j_r, 0].store(h0), td[k_r, j_r, 1].store(h1)).end(j_r, k_r)
        if q == MXFP4:
          e = reg[bi, 4].load()
          small = e < 116
          c_lo = small.where(UOp.const(0x26242000, dtypes.uint32), UOp.const(0x36343000, dtypes.uint32))
          c_hi = small.where(UOp.const(0x2e2c2a28, dtypes.uint32), UOp.const(0x3e3c3a38, dtypes.uint32))
          sc2, z_u = _splat_f_h2(_e8m0_halved(e + small.where(UOp.const(7, dtypes.uint32), UOp.const(3, dtypes.uint32)))), UOp.const(0, dtypes.uint32)
          k_r, j_r = UOp.range(2, rid + 1), UOp.range(4, rid + 2)
          pw = reg[bi, j_r].load() >> _sh(k_r * 4)
          fpb = _amd_byte_perm(c_hi, c_lo, pw & 0x07070707) | ((pw & 0x08080808) << 4)
          p0, p1 = UOp.const(0x05000400, dtypes.uint32), UOp.const(0x07000600, dtypes.uint32)
          return UOp.group(td[k_r, j_r, 0].store(_amd_byte_perm(fpb, z_u, p0).reshape((1,)).bitcast(dtypes.half) * sc2),
                           td[k_r, j_r, 1].store(_amd_byte_perm(fpb, z_u, p1).reshape((1,)).bitcast(dtypes.half) * sc2)).end(j_r, k_r)
        if q == IQ3_XXS:
          assert grid is not None
          sc_w, d_u = reg[bi, 2].load(), reg[bi, 3].load()
          sc_f = _half(d_u) * ((sc_w >> 28).float() + 0.5) * 32.0
          sc2, k_r, p_r, m_r = _splat_f_h2(sc_f), UOp.range(2, rid + 1), UOp.range(2, rid + 2), UOp.range(2, rid + 3)
          qw = reg[bi, k_r].load()
          sgn = _iq_even_signs((sc_w >> _sh((k_r * 2 + p_r) * 7)) & 127)
          g_idx, s4 = (qw >> _sh((p_r * 2 + m_r) * 8)) & 255, sgn >> _sh(m_r * 4)
          g0, g1 = grid[g_idx * 2].load() ^ _sign_h2(s4), grid[g_idx * 2 + 1].load() ^ _sign_h2(s4 >> 2)
          h0, h1 = g0.reshape((1,)).bitcast(dtypes.half), g1.reshape((1,)).bitcast(dtypes.half)
          e0 = UOp.stack((h0[0].float() * sc_f).cast(dtypes.half), (h0[1].float() * sc_f).cast(dtypes.half))
          e1 = UOp.stack((h1[0].float() * sc_f).cast(dtypes.half), (h1[1].float() * sc_f).cast(dtypes.half))
          return UOp.group(td[k_r, p_r * 2 + m_r, 0].store(exact_q.where(e0, h0 * sc2)),
                           td[k_r, p_r * 2 + m_r, 1].store(exact_q.where(e1, h1 * sc2))).end(m_r, p_r, k_r)
        # IQ2_S (raw 2048-u32 grid!)
        assert grid is not None
        sub, (q_w, s_w, qh_u, sc_u, d_u) = gbg % 8, (reg[bi, j].load() for j in range(5))
        qh_b, sc_b, d_f = (qh_u >> _sh((sub & 1) * 8)) & 255, (sc_u >> _sh((sub & 1) * 8)) & 255, _half(d_u)
        sf0, sf1 = d_f * ((sc_b & 15).float() + 0.5) * 0.25, d_f * (((sc_b >> 4) & 15).float() + 0.5) * 0.25
        sc2_0, sc2_1 = _splat_f_h2(sf0), _splat_f_h2(sf1)
        k_r, p_r, m_r = UOp.range(2, rid + 1), UOp.range(2, rid + 2), UOp.range(2, rid + 3)
        p_i = k_r * 2 + p_r
        idx_g = ((q_w >> _sh(p_i * 8)) & 255) | (((qh_b >> _sh(2 * p_i)) & 3) << 8)
        sf, sc2 = k_r.eq(0).where(sf0, sf1), k_r.eq(0).where(sc2_0, sc2_1)
        sg = _iq_signed_word(grid[idx_g * 2 + m_r].load(), (s_w >> _sh(p_i * 8 + m_r * 4)) & 15)
        f0, f1 = ((_amd_byte_perm(c64, sg ^ c80, sl).reshape((1,)).bitcast(dtypes.half) + n1152) * sc2 for sl in (slo, shi))
        he = [(((sg >> (b * 8)).cast(dtypes.uint8).bitcast(dtypes.int8).float()) * sf).cast(dtypes.half) for b in range(4)]
        return UOp.group(td[k_r, p_r * 2 + m_r, 0].store(exact_q.where(UOp.stack(he[0], he[1]), f0)),
                         td[k_r, p_r * 2 + m_r, 1].store(exact_q.where(UOp.stack(he[2], he[3]), f1))).end(m_r, p_r, k_r)
      t_r = tmp_h2.after(_qswitch(ggml_type, [(q, fill_f(q)) for q in fast_qts]))
      ki_r, ji_r = UOp.range(2, rid + 4), UOp.range(4, rid + 5)
      is_q4 = (ggml_type == Q4_K) if isinstance(ggml_type, int) else ggml_type.eq(Q4_K)
      bk_v = ((bg // 2) * 4 + ki_r * 2 + (bg % 2)) if is_q4 is True else ((bg * 2 + ki_r) if is_q4 is False else
              is_q4.where((bg // 2) * 4 + ki_r * 2 + (bg % 2), bg * 2 + ki_r))
      return UOp.group(*(dst[bn_v, (bk_v * 2 + (ji_r // 2)) ^ (bn_i & swz_m), (ji_r % 2) * 2 + p].store(t_r[ki_r, ji_r, p])
                         for p in (0, 1))).end(ji_r, ki_r, bi)
    def run_slow_all() -> UOp:
      bi = UOp.range(b_iters, rid)
      bp_raw = tid + bi * threads
      ok, bp = bp_raw < b_total, bp_raw.minimum(b_total - 1)
      bn_i, bg = bp // k_groups, bp % k_groups
      bn_v, gn, gbg = bn_i.valid(ok), (block_n * bn + bn_i).minimum(out_features - 1), (kt * k_groups + bg).minimum(in_features // 32 - 1)
      sb_idx = gn * (in_features // GGML_BLOCK_SIZE) + gbg // 8
      B_dst, k16, k_u = B_lds.after(*dep)[buf_i], UOp.range(2, rid + 1), UOp.range(16, rid + 2)
      def slow_val(q:int) -> UOp:
        rq = _qraw(raw, q)
        if q in SMALL_BLOCK_QUANTS:
          g, bw = SMALL_BLOCK_QUANTS[q]
          base, sub = (gn * (in_features // (32 * g)) + gbg // g) * bw, gbg % g if g > 1 else UOp.const(0, dtypes.weakint)
        else: base, sub = sb_idx * (QUANT_SIZES[q] // rq.dtype.itemsize), gbg % 8
        v = _make_wmma_dequant(rq, q, UOp.const(0, dtypes.weakint), 1, k_u, grid if q in _WMMA_TBL_QUANTS else None)(base, sub, k16)
        assert isinstance(v, UOp)
        return v
      val = _qswitch(ggml_type, [(q, slow_val(q)) for q in slow_qts])
      return B_dst[bn_v, (bg * 4 + k16 * 2 + k_u // 8) ^ (bn_i & swz_m), k_u % 8].store(val).end(k_u, k16, bi)
    if not slow_qts: return run_fast()
    if not fast_qts: return run_slow_all()
    assert isinstance(ggml_type, UOp)
    is_fast = functools.reduce(lambda a, b: a | b, (ggml_type.eq(q) for q in fast_qts))
    return UOp(Ops.WHERE, src=(is_fast, run_fast(), run_slow_all()))

  def compute_slice(kt_cur:UOp, B_r:UOp, acc_in:UOp, kk_start:int|UOp, kk_cnt:UOp, rid:int) -> UOp:
    n_sl = (kk_cnt // 2).maximum(1)
    sl_r, kk_r, tm_r, tn_r = UOp.range(n_sl, rid + 3), UOp.range(kk_cnt // n_sl, rid), UOp.range(tm, rid + 1), UOp.range(tn, rid + 2)
    kk_loc = (kk_start + sl_r * (kk_cnt // n_sl) + kk_r) if not (isinstance(kk_start, int) and kk_start == 0) else (sl_r * (kk_cnt // n_sl) + kk_r)
    kk = has_wk.where(wave_k * k16_pw + kk_loc, kk_loc)
    mr, nr = (block_m * bm + (wave_m * tm + tm_r) * 16 + col).minimum(x.shape[0] - 1), (wave_n * tn + tn_r) * 16 + col
    a_off = (mr * (in_features // 2) + kt_cur * (bk // 2) + kk * 8).minimum(x.shape[0] * (in_features // 2) - 8)
    a_f = _amd_load(x_u32[a_off], 4).cat(_amd_load(x_u32[a_off + 4], 4)).bitcast(dtypes.half)
    b_f = B_r[nr, (kk * 2) ^ (nr & swz_m)].cat(B_r[nr, (kk * 2 + 1) ^ (nr & swz_m)])
    w = acc[tm_r, tn_r].store(UOp.wmma(a_f, b_f, acc_in.after(sl_r, kk_r)[tm_r, tn_r], *WMMA_ARG)).end(tn_r, tm_r, kk_r)
    ti, tj = UOp.range(tm, rid + 4), UOp.range(tn, rid + 5)
    return acc[ti, tj, 7].store(_pin_f32(acc.after(w)[ti, tj, 7].load(), w)).end(tj, ti, sl_r)

  kt = UOp.range(total_kt + 1, 10)
  do_ld, do_cmp = kt < total_kt, kt > 0
  kt_ld, kt_c = kt.minimum(total_kt - 1), (kt - 1).maximum(0)
  bc, bn_i = kt_c % 2, kt % 2
  r_ld, r_cmp = UOp.range(do_ld.cast(dtypes.weakint), 11), UOp.range(do_cmp.cast(dtypes.weakint), 12)
  br_map = {q: load_b_reg_q(q, kt_ld, (kt, r_ld), 30) for q in qts}
  pb = compute_slice(kt_c, B_lds.after(kt, r_cmp)[bc], acc.after(kt, r_cmp), 0, k16_pw, 34).end(r_cmp)
  st_nxt = store_b_lds(bn_i, br_map, kt_ld, (kt, r_ld), 44).end(r_ld)
  done = UOp.group(pb, st_nxt).barrier().end(kt)
  r_w1 = UOp.range(has_wk.logical_not().cast(dtypes.weakint), 193)
  tm_o1, tn_o1, e_o1 = UOp.range(tm, 190), UOp.range(tn, 191), UOp.range(8, 192, AxisType.UPCAST)
  om1 = (block_m * bm + (wave_m * tm + tm_o1) * 16 + e_o1 * 2 + lg).minimum(out.shape[0] - 1)
  on1 = (block_n * bn + (wave_n * tn + tn_o1) * 16 + col).minimum(out_features - 1)
  st_w1 = out[om1.valid(has_wk.logical_not()), on1].store(
    acc.after(done, r_w1)[tm_o1, tn_o1, e_o1].load().cast(out.dtype)).end(e_o1, tn_o1, tm_o1, r_w1)
  r_wk = UOp.range(has_wk.cast(dtypes.weakint), 194)
  red_lds = B_lds_raw.after(done, r_wk).bitcast(dtypes.float32)
  def red_idx(wk_i, ti, tj, e): return (((((wave_mn * wk + wk_i) * tm + ti) * tn + tj) * 32 + lane) * 8 + e)
  tm_s, tn_s, e_s = UOp.range(tm, 180), UOp.range(tn, 181), UOp.range(8, 182, AxisType.UPCAST)
  red_st = red_lds[red_idx(wave_k, tm_s, tn_s, e_s).valid(has_wk)].store(acc.after(done, r_wk)[tm_s, tn_s, e_s].load()).end(e_s, tn_s, tm_s).barrier()
  red_r = red_lds.after(red_st)
  tm_o, tn_o, e_o, k_w = UOp.range(tm, 195), UOp.range(tn, 196), UOp.range(8, 197, AxisType.UPCAST), UOp.range(wk, 185)
  om = (block_m * bm + (wave_m * tm + tm_o) * 16 + e_o * 2 + lg).minimum(out.shape[0] - 1)
  on = (block_n * bn + (wave_n * tn + tn_o) * 16 + col).minimum(out_features - 1)
  tot = red_r[red_idx(k_w, tm_o, tn_o, e_o)].load().reduce(k_w, arg=(Ops.ADD, 0))
  st_wk = out[om.valid(has_wk & wave_k.eq(0)), on].store(tot.cast(out.dtype)).end(e_o, tn_o, tm_o, r_wk)
  kname = f"linear_{QUANT_NAMES[ggml_type]}_f16_wmma" if isinstance(ggml_type, int) else "linear_quant_f16_wmma"
  return UOp.group(st_w1, st_wk).end(wave, lane, gid).sink(arg=KernelInfo(name=kname, opts_to_apply=()))

# ******** RDNA4 Direct-FP16 Zero-LDS Split-K (M<64 & N<2048) + OTF FP8 2D WMMA ********

def _cvt_pk_fp8(a:UOp, b:UOp, old:UOp, hi:bool) -> UOp:
  return _nat(f"__builtin_amdgcn_cvt_pk_fp8_f32({{0}}, {{1}}, {{2}}, {'true' if hi else 'false'})", dtypes.uint32, a, b, old)
def _pack4_fp8(*fs:UOp) -> UOp: return _cvt_pk_fp8(fs[2], fs[3], _cvt_pk_fp8(fs[0], fs[1], UOp.const(0, dtypes.uint32), False), True)
def _b4_to_fp8(u:UOp, sc:UOp, zero_off:float, add:UOp|float=0.0) -> UOp:
  c4b, zf = UOp.const(0x4b000000, dtypes.uint32), UOp.const(zero_off, dtypes.float32)
  return _pack4_fp8(*((_amd_byte_perm(c4b, u, UOp.const(0x07060500 + b, dtypes.uint32)).bitcast(dtypes.float32) + zf) * sc + add for b in range(4)))
def _pk_max_u16(a:UOp, b:UOp) -> UOp:
  return UOp.native(
    "({{ unsigned int _x = ({0}), _y = ({1}); __builtin_bit_cast(unsigned int, __builtin_elementwise_max("
    "__builtin_bit_cast(unsigned short __attribute__((ext_vector_type(2))), _x), "
    "__builtin_bit_cast(unsigned short __attribute__((ext_vector_type(2))), _y))); }})",
    shape=(), dtype=dtypes.uint32, fmt=True).call(a, b, inline=False)
def _h2x2_to_fp8_pk(u0:UOp, u1:UOp, inv_h2:UOp) -> UOp:
  h2t = "decltype(__builtin_amdgcn_cvt_pkrtz(0.0f, 0.0f))"
  return UOp.native(
    f"({{{{ unsigned int _u0 = ({{0}}), _u1 = ({{1}}), _s = ({{2}}); "
    f"{h2t} _a = __builtin_bit_cast({h2t}, _u0) * __builtin_bit_cast({h2t}, _s); "
    f"{h2t} _b = __builtin_bit_cast({h2t}, _u1) * __builtin_bit_cast({h2t}, _s); "
    f"unsigned int _r = __builtin_amdgcn_cvt_pk_fp8_f32((float)_a[0], (float)_a[1], 0u, false); "
    f"__builtin_amdgcn_cvt_pk_fp8_f32((float)_b[0], (float)_b[1], _r, true); }}}})",
    shape=(), dtype=dtypes.uint32, fmt=True).call(u0, u1, inv_h2, inline=True)

@functools.cache
def _rdna4_zlds_f16_kernel(out:UOp, raw:UOp, x:UOp, *grids:UOp, tok:UOp, out_features:UOp, in_features:UOp,
                           ggml_type:int|UOp, waves_k:int|UOp=4, tm:int=2, tn:int|UOp=2) -> UOp:
  qts = (ggml_type,) if isinstance(ggml_type, int) else tuple(QUANT_NAMES)
  bm, bn = tm * 16, tn * 16
  m_blks, n_blks, k_sb = (tok + bm - 1) // bm, (out_features + bn - 1) // bn, in_features // 256
  raw_u32 = _qraw_u32(raw)
  x_u32, grid = x.flatten().bitcast(dtypes.uint32), grids[0] if grids else None
  gid, lane, wk = UOp.range(m_blks * n_blks, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP), UOp.range(waves_k, 2, AxisType.LOCAL)
  block_m, block_n, col, lg = gid // n_blks, gid % n_blks, lane % 16, lane // 16
  acc = UOp.alloc((tm, tn, 8), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0.0)))
  c4b, c80 = UOp.const(0x4b000000, dtypes.uint32), UOp.const(0x80808080, dtypes.uint32)
  zf_0 = UOp.const(-8388608.0, dtypes.float32)
  zf_128 = UOp.const(-8388736.0, dtypes.float32)
  z_f = UOp.const(0.0, dtypes.float32)
  sb_i = UOp.range(k_sb // waves_k, 10)
  sb = wk + sb_i * waves_k
  sp_r = UOp.range(4, 11)
  k64 = sb * 4 + sp_r
  a_reg = UOp.alloc((2, tm, 2, 8), dtypes.half, addrspace=AddrSpace.REG)
  ma_r, tia_r, k2a_r = UOp.range(2, 12), UOp.range(tm, 13), UOp.range(2, 14)
  a_val = _amd_load(x_u32[(block_m * bm + tia_r * 16 + col).minimum(tok - 1) * (in_features // 2) +
                          k64 * 32 + (ma_r * 2 + k2a_r) * 8 + lg * 4], 4).bitcast(dtypes.half)
  a_r = a_reg.after(a_reg.after(sb_i, sp_r)[ma_r, tia_r, k2a_r].store(a_val).end(k2a_r, tia_r, ma_r))
  tj_r = UOp.range(tn, 15)
  gn = (block_n * bn + tj_r * 16 + col).minimum(out_features - 1)
  sb_blk = gn * k_sb + sb
  hdrs: dict[int, tuple[UOp, ...]] = {}
  for q in qts:
    rq = _qraw(raw, q)
    base = sb_blk * (QUANT_SIZES[q] // rq.dtype.itemsize)
    if q in (Q4_K, Q5_K): hdrs[q] = (base, *_v4(_amd_load(raw_u32[base], 4)))
    elif q == IQ4_XS:
      h01 = _amd_load(raw_u32[base], 2)
      hdrs[q] = (base, h01[0], h01[1])
    elif q == Q2_K:
      dm_w = raw_u32[sb_blk * 21 + 20].load()
      hdrs[q] = (sb_blk * 21, _half(dm_w & 0xffff), _half(dm_w >> 16))
    elif q in (Q6_K, Q3_K, IQ3_XXS, IQ3_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ1_S):
      hdrs[q] = (base, _half(rq[base + (104 if q == Q6_K else 54 if q == Q3_K else 0)].load().cast(dtypes.uint32)))
    else: hdrs[q] = (base,)
  m_r = UOp.range(2, 16)
  sub = sp_r * 2 + m_r
  b_buf = UOp.alloc((2, 8), dtypes.half, addrspace=AddrSpace.REG)
  h_r = UOp.range(2, 58)
  u_reg = UOp.alloc((2,), dtypes.uint32, addrspace=AddrSpace.REG)
  f_reg = UOp.alloc((3,), dtypes.float32, addrspace=AddrSpace.REG)
  def _put(u0:UOp, u1:UOp, zf:UOp, sc:UOp, add:UOp=z_f) -> UOp:
    ud, fd = u_reg.after(sb_i, sp_r, tj_r, m_r, h_r), f_reg.after(sb_i, sp_r, tj_r, m_r, h_r)
    return UOp.group(ud.store(UOp.stack(u0, u1)), fd.store(UOp.stack(zf, sc, add)))
  def _deq_b(q:int) -> UOp:
    rq, hdr = _qraw(raw, q), hdrs[q]
    if q in (Q4_K, Q5_K):
      b, h0_k, h1_k, h2_k, h3_k = hdr
      ds0, dm0, ds1, dm1 = _q4k_pair_scales([h0_k, h1_k, h2_k, h3_k], sp_r)
      ds_f, dm_f, sh, off = m_r.eq(0).where(ds0, ds1), m_r.eq(0).where(-dm0, -dm1), _sh(m_r * 4), 4 if q == Q4_K else 12
      p = _amd_load(raw_u32[b + off + sp_r * 8 + h_r * 4 + lg * 2], 2)
      u0, u1 = (p[0] >> sh) & 0x0f0f0f0f, (p[1] >> sh) & 0x0f0f0f0f
      if q == Q5_K:
        hv = _amd_load(raw_u32[b + 4 + h_r * 4 + lg * 2], 2)
        u0, u1 = u0 | (((hv[0] >> _sh(sub)) & 0x01010101) << 4), u1 | (((hv[1] >> _sh(sub)) & 0x01010101) << 4)
      return _put(u0, u1, zf_0, ds_f, dm_f)
    if q == Q6_K:
      b, d_f = hdr
      ql_b, qh_b = b + (sub // 4) * 32 + (sub % 2) * 16, b + 64 + (sub // 4) * 16
      ql_sh, qh_sh = ((sub % 4 // 2) * 4).cast(dtypes.uint32), ((sub % 4) * 2).cast(dtypes.uint32)
      ql = _amd_load(rq[ql_b + h_r * 8 + lg * 4], 4).bitcast(dtypes.uint32)
      qh = _amd_load(rq[qh_b + h_r * 8 + lg * 4], 4).bitcast(dtypes.uint32)
      sc_w = rq[b + 96 + sub].load().cast(dtypes.uint32)
      s_h = ((sc_w >> _sh(h_r * 8)) & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float() * d_f
      return _put(((ql[0] >> ql_sh) & 0x0f0f0f0f) | (((qh[0] >> qh_sh) & 0x03030303) << 4),
                  ((ql[1] >> ql_sh) & 0x0f0f0f0f) | (((qh[1] >> qh_sh) & 0x03030303) << 4), UOp.const(-8388640.0, dtypes.float32), s_h)
    if q == IQ4_XS:
      b, h0, h1 = hdr
      sc_f = _half(h0) * ((((h1 >> _sh(4 * sub)) & 15) | (((h0 >> _sh(16 + 2 * sub)) & 3) << 4)).cast(dtypes.int32) - 32).float()
      p, sh = _amd_load(raw_u32[b + 2 + sub * 4 + lg * 2], 2), _sh(h_r * 4)
      return _put(_nibble_lut(p[0], sh, IQ4NL_U8_LUT), _nibble_lut(p[1], sh, IQ4NL_U8_LUT), zf_128, sc_f)
    if q == Q8_0:
      wb = hdr[0] + sub * 17
      sc_f = _half(rq[wb].load().cast(dtypes.uint32))
      qv = _amd_load(rq[wb + 1 + h_r * 8 + lg * 4], 4).bitcast(dtypes.uint32)
      return _put(qv[0] ^ c80, qv[1] ^ c80, zf_128, sc_f)
    if q == MXFP4:
      boff = hdr[0] + sub * 17
      sc_f = _e8m0_halved(rq[boff].load().cast(dtypes.uint32))
      p, sh = _amd_load(rq[boff + 1 + lg * 8], 8).bitcast(dtypes.uint32), _sh(h_r * 4)
      return _put(_nibble_lut(p[0], sh, MXFP4_U8_LUT), _nibble_lut(p[1], sh, MXFP4_U8_LUT), zf_128, sc_f)
    if q == Q2_K:
      b32, dr, dmr = hdr
      sc_b = ((raw_u32[b32 + sp_r].load() >> _sh(m_r * 16 + h_r * 8)) & 255)
      sh_q2 = _sh((sub % 4) * 2)
      p = _amd_load(raw_u32[b32 + 4 + (sp_r // 2) * 8 + h_r * 4 + lg * 2], 2)
      return _put((p[0] >> sh_q2) & 0x03030303, (p[1] >> sh_q2) & 0x03030303, zf_0, dr * (sc_b & 15).float(), -dmr * (sc_b >> 4).float())
    b, d_f = hdr if len(hdr) == 2 else (hdr[0], z_f)
    wb = _small_block(b, sub, q) if q in SMALL_BLOCK_QUANTS else b
    w_i, ud, fd = UOp.range(2, 59), u_reg.after(sb_i, sp_r, tj_r, m_r, h_r), f_reg.after(sb_i, sp_r, tj_r, m_r, h_r)
    w_v = _quant_word(rq, wb, sub, h_r * 4 + lg * 2 + w_i, q, grid)
    u_st = ud[w_i].store(w_v if q == Q3_K else (w_v ^ c80)).end(w_i)
    if q == Q3_K:
      j = sub * 2 + h_r
      sc_q3 = (((_load_byte(rq, b, 96 + j % 8) >> _sh((j // 8) * 4)) & 15) |
               (((_load_byte(rq, b, 104 + j % 4) >> _sh((j // 4) * 2)) & 3) << 4)).cast(dtypes.int32) - 32
      return UOp.group(u_st, fd.store(UOp.stack(UOp.const(-8388612.0, dtypes.float32), sc_q3.float() * d_f, z_f)))
    if q == IQ1_S:
      sc, zero = _iq1_scale(rq, b, sub)
      return UOp.group(u_st, fd.store(UOp.stack(zf_128, sc, -sc * zero)))
    if q == IQ3_XXS: sc = d_f * ((_load_u32(rq, b, 66 + sub * 4) >> 28).float() + 0.5) * 0.5
    elif q == IQ3_S: sc = d_f * (1 + 2 * ((_load_byte(rq, b, 106 + sub // 2) >> _sh((sub % 2) * 4)) & 15)).float()
    elif q in (IQ2_XS, IQ2_S):
      sc = d_f * (((_load_byte(rq, b, (66 if q == IQ2_XS else 74) + sub) >> _sh(h_r * 4)) & 15).float() + 0.5) * 0.25
    elif q == IQ2_XXS: sc = d_f * ((_load_u32(rq, b, 6 + sub * 8) >> 28).float() + 0.5) * 0.25
    elif q == NVFP4: sc = _ue4m3_halved(_load_byte(rq, wb, (sub % 2) * 2 + h_r))
    else: sc = _half(rq[wb].load().cast(dtypes.uint32))
    return UOp.group(u_st, fd.store(UOp.stack(zf_128, sc, z_f)))
  q_done = _qswitch(ggml_type, [(q, _deq_b(q)) for q in qts])
  u_r, f_r = u_reg.after(q_done), f_reg.after(q_done)
  u0, u1, zf_v, sc_v, add_v = u_r[0].load(), u_r[1].load(), f_r[0].load(), f_r[1].load(), f_r[2].load()
  bi = UOp.range(4, 60)
  sel_b, bd = UOp.const(0x07060500, dtypes.uint32) + bi.cast(dtypes.uint32), b_buf.after(sb_i, sp_r, tj_r, m_r, h_r, q_done)
  b_st = UOp.group(
    bd[h_r, bi].store(((_amd_byte_perm(c4b, u0, sel_b).bitcast(dtypes.float32) + zf_v) * sc_v + add_v).cast(dtypes.half)),
    bd[h_r, bi + 4].store(((_amd_byte_perm(c4b, u1, sel_b).bitcast(dtypes.float32) + zf_v) * sc_v + add_v).cast(dtypes.half))).end(bi, h_r)
  b_r = b_buf.after(b_st)
  ti_0, ti_1 = UOp.range(tm, 17), UOp.range(tm, 18)
  w0_m = acc[ti_0, tj_r].store(UOp.wmma(a_r[m_r, ti_0, 0], b_r[0], acc.after(sb_i, sp_r, tj_r, m_r)[ti_0, tj_r], *WMMA_ARG)).end(ti_0)
  w1_m = acc[ti_1, tj_r].store(UOp.wmma(a_r[m_r, ti_1, 1], b_r[1], acc.after(w0_m)[ti_1, tj_r], *WMMA_ARG)).end(ti_1)
  acc_f = acc.after(w1_m.end(m_r, tj_r, sp_r, sb_i))
  red_lds = UOp.alloc((waves_k, tm, tn, 32, 8), dtypes.float32, addrspace=AddrSpace.LOCAL)
  ti_r, tj_o, e_r = UOp.range(tm, 180), UOp.range(tn, 181), UOp.range(8, 182, AxisType.UPCAST)
  r_lds = red_lds.after(red_lds[wk, ti_r, tj_o, lane, e_r].store(acc_f[ti_r, tj_o, e_r].load()).end(e_r, tj_o, ti_r).barrier())
  tm_o, tn_o, e_o, wk_r = UOp.range(tm, 190), UOp.range(tn, 191), UOp.range(8, 192, AxisType.UPCAST), UOp.range(waves_k, 193)
  val = r_lds[wk_r, tm_o, tn_o, lane, e_o].load().reduce(wk_r, arg=(Ops.ADD, 0))
  om, on = block_m * bm + tm_o * 16 + lg * 8 + e_o, block_n * bn + tn_o * 16 + col
  kname = f"linear_{QUANT_NAMES[ggml_type]}_zlds_f16" if isinstance(ggml_type, int) else "linear_quant_zlds_f16"
  return out[om.valid((om < tok) & wk.eq(0)), on.valid(on < out_features)].store(val.cast(out.dtype)).end(
    e_o, tn_o, tm_o, wk, lane, gid).sink(arg=KernelInfo(name=kname, opts_to_apply=()))

@functools.cache
def _amd_moe_route_kernel(tok_idx:UOp, tile_exp:UOp, tile_off:UOp, tile_cnt:UOp, total_tiles:UOp, sel:UOp, *,
                          num_experts:UOp, ndev:UOp) -> UOp:
  m_tot, max_tiles, bm = sel.shape[0], tile_exp.shape[0], 16
  assert isinstance(m_tot, UOp) and isinstance(max_tiles, UOp)
  blk, tid = UOp.range(1, 0, AxisType.GLOBAL), UOp.range(1024, 1, AxisType.LOCAL)
  dev_off = UOp.range(ndev, 10, AxisType.DEVICE) * num_experts
  cnt_lds, cur_lds = UOp.alloc((512,), dtypes.int32, addrspace=AddrSpace.LOCAL), UOp.alloc((512,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  off_lds, t_off_lds = UOp.alloc((513,), dtypes.int32, addrspace=AddrSpace.LOCAL), UOp.alloc((513,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  z, z_idx = UOp.const(0, dtypes.int32), UOp.const(0, dtypes.weakint)
  c0 = cnt_lds[tid.valid(tid < num_experts)].store(z).barrier()
  ci = UOp.range((m_tot + 1023) // 1024, 11)
  idx = tid + ci * 1024
  ok = idx < m_tot
  e_i = ok.where(sel[idx.valid(ok)].load() - dev_off.cast(dtypes.int32), UOp.const(-1, dtypes.int32))
  e_ok = ok & (e_i >= 0) & (e_i < num_experts.cast(dtypes.int32))
  sink_r = UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG)
  c1 = sink_r[0].store(_nat("(({1}) ? __atomic_fetch_add(({0}), 1, __ATOMIC_RELAXED) : 0)", dtypes.int32,
                            cnt_lds.after(c0, ci)[e_i.cast(dtypes.weakint).maximum(0).minimum(num_experts - 1)],
                            e_ok, inline=False)).end(ci).barrier()
  cnt_r, t0_en = cnt_lds.after(c1), tid.eq(0)
  ps_done = sink_r.after(c1)[0].store(_nat(
    "({{ if ({4}) {{ int _s = 0, _t = 0; for (int _e = 0; _e < (int)({5}); ++_e) {{ int _c = ({0})[_e]; "
    "({1})[_e] = _s; ({2})[_e] = _s; ({3})[_e] = _t; _s += _c; _t += (_c + 15) >> 4; }} "
    "({3})[(int)({5})] = _t; }} 0; }})", dtypes.int32,
    cnt_r[z_idx], off_lds.after(c1)[z_idx], cur_lds.after(c1)[z_idx], t_off_lds.after(c1)[z_idx], t0_en, num_experts, inline=False)).barrier()
  cnt_f, off_r, cur_r, toff_r = cnt_lds.after(ps_done), off_lds.after(ps_done), cur_lds.after(ps_done), t_off_lds.after(ps_done)
  si = UOp.range((m_tot + 1023) // 1024, 14)
  idx2 = tid + si * 1024
  ok2 = idx2 < m_tot
  e2 = ok2.where(sel[idx2.valid(ok2)].load() - dev_off.cast(dtypes.int32), UOp.const(-1, dtypes.int32))
  e2_ok = ok2 & (e2 >= 0) & (e2 < num_experts.cast(dtypes.int32))
  sc_done = sink_r.after(ps_done, si)[0].store(_nat(
    "({{ if ({3}) {{ int _p = __atomic_fetch_add(({0}) + (int)({2}), 1, __ATOMIC_RELAXED); ({1})[_p] = (int)({4}); }} 0; }})", dtypes.int32,
    cur_r.after(si)[z_idx], tok_idx[z_idx], e2, e2_ok, idx2, inline=False)).end(si)
  total_nt, ti_m = toff_r[num_experts].load(), UOp.range((max_tiles + 1023) // 1024, 16)
  t_id = tid + ti_m * 1024
  t_ok, t_act = t_id < max_tiles, t_id.cast(dtypes.int32) < total_nt
  ex_f = _nat("({{ int _lo = 0, _hi = (int)({1}); for (int _k = 0; _k < 9; ++_k) {{ int _m = (_lo + _hi) >> 1; "
              "if (({0})[_m] <= (int)({2})) _lo = _m; else _hi = _m; }} _lo; }})", dtypes.int32,
              toff_r.after(ti_m)[z_idx], num_experts, t_id, inline=False).maximum(0).minimum(num_experts.cast(dtypes.int32) - 1)
  ex_w = ex_f.cast(dtypes.weakint)
  j_t = t_id.cast(dtypes.int32) - toff_r[ex_w].load()
  td_done = sink_r.after(ps_done, ti_m)[0].store(_nat(
    "({{ if ({6}) {{ ({0})[(int)({7})] = (int)({3}); ({1})[(int)({7})] = (int)({4}); ({2})[(int)({7})] = (int)({5}); }} 0; }})", dtypes.int32,
    tile_exp[z_idx], tile_off[z_idx], tile_cnt[z_idx], t_act.where(ex_f, z), t_act.where(off_r[ex_w].load() + j_t * bm, z),
    t_act.where((cnt_f[ex_w].load() - j_t * bm).minimum(bm).maximum(0), z), t_ok, t_id, inline=False)).end(ti_m)
  st_tot = total_tiles[z_idx].store(total_nt)
  return UOp.group(sc_done, td_done, st_tot).end(tid, blk).sink(arg=KernelInfo(name="amd_moe_route", opts_to_apply=()))

def _mx_fp8_luts(e:UOp, ref_e:UOp) -> tuple[UOp, UOp]:
  de, c11 = ref_e - e, UOp.const(11, dtypes.uint32)
  ok, d4 = de <= c11, ((c11 - de.minimum(c11)) << 3) * UOp.const(0x01010101, dtypes.uint32)
  return (ok.where(UOp.const(0x0c080000, dtypes.uint32) + (d4 & UOp.const(0xffffff00, dtypes.uint32)), 0),
          ok.where(UOp.const(0x1c181410, dtypes.uint32) + d4, 0))
def _iq_sgn(s4:UOp) -> UOp: return (((s4 & 15) * 0x00204081) & 0x01010101) << 7

def _fp8_load_b128(q:int, ru:UOp, rd:UOp, gn:UOp, bg:UOp, kt_v:UOp, k_sb:UOp, in_features:UOp,
                   e_u32:int|UOp=0, e_rd:int|UOp=0) -> list[UOp]:
  gbg = kt_v * 4 + bg
  sub, sb = gbg % 8, gn * k_sb + gbg // 8
  if q == MXFP4: return _v4(_amd_load(ru[(b32 := e_u32 + ((gn * (in_features // 32) + kt_v * 4) * 17) // 4) + bg * 4], 4)) + [ru[b32 + 16].load()]
  if q == Q8_0:
    b32, off = e_u32 + (gn * (in_features // 64) + gbg // 2) * 17, (bg & 1) * 9
    return _v4(_amd_load(ru[b32 + off], 4)) + _v4(_amd_load(ru[b32 + off + 4], 4)) + [ru[b32 + 8].load()]
  if q == Q4_K: return _v4(_amd_load(ru[(b := e_u32 + sb * 36) + 4 + ((gbg // 2) % 4) * 8 + (gbg & 1) * 4], 4)) + [ru[b + bg].load()]
  if q == Q5_K:
    b = e_u32 + sb * 44
    qh2 = _amd_load(ru[b + 4 + bg * 2], 2)
    return _v4(_amd_load(ru[b + 12 + ((gbg // 2) % 4) * 8 + (gbg & 1) * 4], 4)) + [qh2[0], qh2[1], ru[b + bg].load()]
  if q == Q6_K:
    b, s4 = e_rd + sb * 105, sub // 4
    qh = _amd_load(rd[b + 64 + s4 * 16 + bg * 4], 4).bitcast(dtypes.uint32)
    ql = _v4(_amd_load(rd[b + s4 * 32 + (bg & 1) * 16 + (bg // 2) * 8], 8).bitcast(dtypes.uint32))
    return ql + [qh[0], qh[1], rd[b + 96 + sub].load().cast(dtypes.uint32), rd[b + 104].load().cast(dtypes.uint32)]
  if q == Q3_K:
    b, s4 = e_rd + sb * 55, sub // 4
    hm, qs = _amd_load(rd[b + bg * 4], 4).bitcast(dtypes.uint32), _amd_load(rd[b + 16 + s4 * 16 + bg * 4], 4).bitcast(dtypes.uint32)
    return [hm[0], hm[1], qs[0], qs[1], rd[b + 48 + (sub % 4)].load().cast(dtypes.uint32),
            rd[b + 52 + (sub & 1)].load().cast(dtypes.uint32), rd[b + 54].load().cast(dtypes.uint32)]
  if q == IQ4_XS: return _v4(_amd_load(ru[(b := e_u32 + sb * 34) + 2 + sub * 4], 4)) + [ru[b + (bg & 1)].load()]
  if q == Q2_K:
    b32 = e_u32 + sb * 21
    return (_v4(_amd_load(ru[b32 + 4 + (sub // 4) * 8], 4)) + _v4(_amd_load(ru[b32 + 8 + (sub // 4) * 8], 4)) +
            [ru[b32 + sub // 2].load(), ru[b32 + 20].load()])
  if q == IQ3_S:
    b = e_rd + sb * 55
    q01 = _amd_load(rd[b + 1 + sub * 4], 4).bitcast(dtypes.uint32)
    return [q01[0], q01[1], _amd_load(rd[b + 37 + sub * 2], 2).bitcast(dtypes.uint32)[0],
            rd[b + 33 + sub // 2].load().cast(dtypes.uint32), rd[b + 53 + sub // 4].load().cast(dtypes.uint32), rd[b].load().cast(dtypes.uint32)]
  if q == IQ3_XXS:
    b = e_rd + sb * 49
    q01 = _amd_load(rd[b + 1 + sub * 4], 4).bitcast(dtypes.uint32)
    return [q01[0], q01[1], _amd_load(rd[b + 33 + sub * 2], 2).bitcast(dtypes.uint32)[0], rd[b].load().cast(dtypes.uint32)]
  b = e_rd + sb * 41
  return [_amd_load(rd[b + 1 + sub * 2], 2).bitcast(dtypes.uint32)[0], _amd_load(rd[b + 17 + sub * 2], 2).bitcast(dtypes.uint32)[0],
          rd[b + 33 + sub // 2].load().cast(dtypes.uint32), rd[b + 37 + sub // 2].load().cast(dtypes.uint32), rd[b].load().cast(dtypes.uint32)]

def _fp8_dequant_b128(q:int, get_u:Callable[[int|UOp], UOp], grid:UOp, bg:UOp, gbg:UOp, sub:UOp, lane:UOp, base_l:UOp,
                      put_r:Callable, put_r4:Callable, reg_u32:Callable, scale_fn:Callable, mx_fn:Callable, rid:int) -> UOp:
  z_u, z_f = UOp.const(0, dtypes.uint32), UOp.const(0.0, dtypes.float32)
  if q == MXFP4:
    rem = bg.cast(dtypes.uint32)
    w0, w4 = get_u(0), bg.eq(3).where(get_u(4), _lane_bpermute(lane + 1, get_u(0)))
    e = (w0 >> (rem * 8)) & 255
    em = e.maximum(_lane_xor(e, 1, dpp=True))
    em = em.maximum(_lane_xor(em, 2, dpp=True))
    ref_e, meta = mx_fn(em)
    l0, l1 = _mx_fp8_luts(e, ref_e)
    def mx_pair(j:UOp) -> tuple[UOp, UOp]:
      lo, hi = get_u(j), j.eq(3).where(w4, get_u(j + 1))
      pw = rem.eq(3).where(hi, _perm_alignbyte(hi, lo, (rem + 1) & 3))
      return (_amd_byte_perm(l1, l0, pw & 0x07070707) | ((pw & 0x08080808) << 4),
              _amd_byte_perm(l1, l0, (pw >> 4) & 0x07070707) | (pw & 0x80808080))
    return put_r4(mx_pair, meta)
  if q == Q8_0:
    is_e = (bg & 1).eq(0)
    sc = _half(is_e.where(get_u(0) & 0xffff, get_u(8) >> 16))
    sm = _fast_fmax(sc.abs(), _lane_xor(sc.abs(), 1, dpp=True))
    sm = _fast_fmax(sm, _lane_xor(sm, 2, dpp=True))
    inv_r, meta = scale_fn(sm)
    sn, c80 = sc * inv_r, UOp.const(0x80808080, dtypes.uint32)
    sel_a = is_e.where(UOp.const(0x05040302, dtypes.uint32), UOp.const(0x03020100, dtypes.uint32))
    return put_r(lambda j: _b4_to_fp8(_amd_byte_perm(get_u(j + 1), get_u(j), sel_a) ^ c80, sn, -8388736.0), meta)
  if q in (Q4_K, Q5_K):
    is_lo = (gbg & 1).eq(0)
    hdr = [_dpp_quad(get_u(4 if q == Q4_K else 6), k) for k in range(4)]
    tsc_need = _fast_fmax(_half(hdr[0] & 0xffff).abs() * UOp.const(4.0 if q == Q4_K else 8.0, dtypes.float32),
                          _half(hdr[0] >> 16).abs() * UOp.const(0.25, dtypes.float32))
    inv_r, meta = scale_fn(tsc_need)
    ds0, dm0, ds1, dm1 = _q4k_pair_scales(hdr, (gbg // 2) % 4)
    ds_m, dm_m = is_lo.where(ds0, ds1) * inv_r, is_lo.where(-dm0, -dm1) * inv_r
    sh = is_lo.where(UOp.const(0, dtypes.uint32), UOp.const(4, dtypes.uint32))
    if q == Q4_K:
      def q4_pair(j:UOp) -> tuple[UOp, UOp]:
        q_m = get_u(j)
        q_o = _lane_xor(q_m, 1, dpp=True)
        return (_b4_to_fp8((is_lo.where(q_m, q_o) >> sh) & 0x0f0f0f0f, ds_m, -8388608.0, dm_m),
                _b4_to_fp8((is_lo.where(q_o, q_m) >> sh) & 0x0f0f0f0f, ds_m, -8388608.0, dm_m))
      return put_r4(q4_pair, meta)
    pa_r = reg_u32(8, lambda j: (q_m := get_u(j % 4), q_o := _lane_xor(q_m, 1, dpp=True),
                   (j < 4).where(is_lo.where(q_m, q_o), is_lo.where(q_o, q_m)))[2], rid + 2)
    qh_r = reg_u32(8, lambda j: _lane_bpermute(base_l + j // 2, get_u(4 + (j % 2))), rid + 3)
    return put_r(lambda j: _b4_to_fp8(((pa_r[j].load() >> sh) & 0x0f0f0f0f) | (((qh_r[j].load() >> _sh(sub)) & 0x01010101) << 4),
                 ds_m, -8388608.0, dm_m), meta)
  if q == Q6_K:
    sc_w, d_f, is_l2 = get_u(6), _half(get_u(7)), bg < 2
    ql_r = reg_u32(8, lambda j: (q_m := get_u(j % 4), q_o := _lane_xor(q_m, 2, dpp=True),
                   (j < 4).where(is_l2.where(q_m, q_o), is_l2.where(q_o, q_m)))[2], rid + 2)
    qh_r = reg_u32(8, lambda j: _lane_bpermute(base_l + j // 2, get_u(4 + (j % 2))), rid + 3)
    inv_r, meta = scale_fn(d_f.abs() * UOp.const(16.0, dtypes.float32))
    d_n = d_f * inv_r
    s0 = (sc_w & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float() * d_n
    s1 = ((sc_w >> 8) & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float() * d_n
    ql_sh, qh_sh = _sh((sub % 4 // 2) * 4), _sh((sub % 4) * 2)
    return put_r(lambda j: _b4_to_fp8(((ql_r[j].load() >> ql_sh) & 0x0f0f0f0f) | (((qh_r[j].load() >> qh_sh) & 0x03030303) << 4),
                 (j < 4).where(s0, s1), -8388640.0), meta)
  if q == Q3_K:
    hm_r = reg_u32(8, lambda j: _lane_bpermute(base_l + j // 2, get_u(j % 2)), rid + 2)
    qs_r = reg_u32(8, lambda j: _lane_bpermute(base_l + j // 2, get_u(2 + (j % 2))), rid + 3)
    sl16, sh16, d_f = get_u(4), get_u(5), _half(get_u(6))
    inv_r, meta = scale_fn(d_f.abs() * UOp.const(0.5, dtypes.float32))
    d_n = d_f * inv_r
    q3_scs = [((((sl16 >> _sh(h * 8 + (sub // 4) * 4)) & 15) |
                (((sh16 >> _sh(h * 8 + (sub // 2) * 2)) & 3) << 4)).cast(dtypes.int32) - 32).float() * d_n for h in (0, 1)]
    luts = [(_pack4_fp8(*(float(v) * sc for v in (-4, -3, -2, -1))), _pack4_fp8(*(float(v) * sc for v in (0, 1, 2, 3)))) for sc in q3_scs]
    qs_sh, hm_sh = _sh((sub % 4) * 2), _sh(sub)
    return put_r(lambda j: _amd_byte_perm((j < 4).where(luts[0][1], luts[1][1]), (j < 4).where(luts[0][0], luts[1][0]),
                 ((qs_r[j].load() >> qs_sh) & 0x03030303) | (((hm_r[j].load() >> hm_sh) & 0x01010101) << 2)), meta)
  if q == IQ4_XS:
    hm, ho = get_u(4), _lane_xor(get_u(4), 1, dpp=True)
    h0, h1 = (bg & 1).eq(0).where(hm, ho), (bg & 1).eq(0).where(ho, hm)
    d_f = _half(h0)
    inv_r, meta = scale_fn(d_f.abs() * UOp.const(32.0 / 3.0, dtypes.float32))
    sc_i = (((h1 >> _sh(4 * sub)) & 15) | (((h0 >> _sh(16 + 2 * sub)) & 3) << 4)).cast(dtypes.int32) - 32
    sc = sc_i.float() * (d_f * inv_r)
    lut = tuple(_pack4_fp8(*(UOp.const(float(v), dtypes.float32) * sc for v in _IQ4NL_VALS[k:k + 4])) for k in (0, 4, 8, 12))
    return put_r(lambda j: _nibble_lut(get_u(j % 4), _sh((j // 4) * 4), lut), meta)
  if q == Q2_K:
    sc16, dm_w = (get_u(8) >> _sh((sub & 1) * 16)) & 0xffff, get_u(9)
    dr, dmr = _half(dm_w & 0xffff), _half(dm_w >> 16)
    inv_r, meta = scale_fn(_fast_fmax(dr.abs(), dmr.abs()) * UOp.const(0.25, dtypes.float32))
    sh_q2 = _sh((sub % 4) * 2)
    luts1 = [_pack4_fp8(*(float(i) * dr * inv_r * (sc & 15).float() - dmr * inv_r * (sc >> 4).float() for i in range(4)))
             for sc in (sc16 & 255, (sc16 >> 8) & 255)]
    return put_r(lambda j: _amd_byte_perm(z_u, (j < 4).where(luts1[0], luts1[1]), (get_u(j) >> sh_q2) & 0x03030303), meta)
  if q == IQ3_S:
    q0, q1, s0, d_f = get_u(0), get_u(1), get_u(2), _half(get_u(5))
    qh_s, sc_n = (get_u(3) >> _sh((sub & 1) * 8)) & 255, (get_u(4) >> _sh((sub & 3) * 4)) & 15
    sc_m = sc_n.maximum(_lane_xor(sc_n, 1, dpp=True))
    sc_m = sc_m.maximum(_lane_xor(sc_m, 2, dpp=True))
    inv_r, meta = scale_fn(d_f.abs() * (1.0 + 2.0 * sc_m.float()) * UOp.const(15.0 / 448.0, dtypes.float32))
    sn = (1.0 + 2.0 * sc_n.float()) * (d_f * inv_r)
    l0, l1 = _pack4_fp8(*(float(v) * sn for v in (1, 3, 5, 7))), _pack4_fp8(*(float(v) * sn for v in (9, 11, 13, 15)))
    return put_r(lambda i: _amd_byte_perm(l1, l0, (grid[(((i < 4).where(q0, q1) >> _sh((i & 3) * 8)) & 255) |
                 (((qh_s >> _sh(i)) & 1) << 8)].load() >> 1) & 0x07070707) ^ _iq_sgn(s0 >> _sh(i * 4)), meta)
  if q == IQ3_XXS:
    q0, q1, sc_w, d_f = get_u(0), get_u(1), get_u(2), _half(get_u(3))
    inv_r, meta = scale_fn(d_f.abs() * UOp.const(1.5, dtypes.float32))
    sn = ((sc_w >> 28).float() + 0.5) * UOp.const(0.5, dtypes.float32) * (d_f * inv_r)
    l0, l1 = _pack4_fp8(*(float(v) * sn for v in (4, 12, 20, 28))), _pack4_fp8(*(float(v) * sn for v in (36, 44, 52, 62)))
    def iq3x_pair(p:UOp) -> tuple[UOp, UOp]:
      qw, sgn = (p < 2).where(q0, q1) >> _sh((p & 1) * 16), _iq_even_signs((sc_w >> _sh(7 * p)) & 127)
      return (_amd_byte_perm(l1, l0, (grid[qw & 255].load() >> 3) & 0x07070707) ^ _iq_sgn(sgn),
              _amd_byte_perm(l1, l0, (grid[(qw >> 8) & 255].load() >> 3) & 0x07070707) ^ _iq_sgn(sgn >> 4))
    return put_r4(iq3x_pair, meta, stride2=True)
  q0, s0, bsh, d_f = get_u(0), get_u(1), _sh((sub & 1) * 8), _half(get_u(4))
  qh_2, sc_b = (get_u(2) >> bsh) & 255, (get_u(3) >> bsh) & 255
  inv_r, meta = scale_fn(d_f.abs() * UOp.const(0.5, dtypes.float32))
  luts2 = [_pack4_fp8(8.0 * sf, 25.0 * sf, 43.0 * sf, z_f)
           for k in (0, 1) for sf in ((((sc_b >> (k * 4)) & 15).float() + 0.5) * UOp.const(0.25, dtypes.float32) * (d_f * inv_r),)]
  return put_r(lambda j: (p := j // 2, k := j % 2, _amd_byte_perm(z_u, (p < 2).where(luts2[0], luts2[1]),
               (grid[(((q0 >> _sh(8 * p)) & 255) | (((qh_2 >> _sh(2 * p)) & 3) << 8)) * 2 + k].load() >> 4) & 0x03030303) ^
               _iq_sgn(s0 >> _sh(8 * p + 4 * k)))[2], meta)

@functools.cache
def _rdna4_quant_x_frag_kernel(xt:UOp, xsc:UOp, x:UOp, tok_idx:UOp, tile_off:UOp, tile_cnt:UOp, total_tiles:UOp, *, x_share:int|UOp=1) -> UOp:
  m_tiles, k_tiles, x_rows, in_features = xt.shape[0], xsc.shape[1], x.shape[0], x.shape[1]
  assert isinstance(m_tiles, UOp) and isinstance(k_tiles, UOp) and isinstance(x_rows, UOp) and isinstance(in_features, UOp)
  gid = UOp.range(m_tiles * k_tiles, 0, AxisType.GLOBAL)
  wave, lane = UOp.range(2, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  tid, m_tile, kt = wave * 32 + lane, gid // k_tiles, gid % k_tiles
  row16, kg, z32 = tid // 4, tid % 4, UOp.const(0, dtypes.uint32)
  guard = _nat("({{ if ((int)({0}) >= (int)({1})) return; 0; }})", dtypes.int32, m_tile, total_tiles[0].load(), inline=False)
  cnt_m, off_m = tile_cnt.after(guard)[m_tile].load().cast(dtypes.weakint), tile_off.after(guard)[m_tile].load().cast(dtypes.weakint)
  r_ok = row16 < cnt_m
  src_row = (r_ok.where(tok_idx.after(guard)[(off_m + row16).valid(r_ok)].load(), UOp.const(0, dtypes.int32)).cast(dtypes.weakint) // x_share
             ).maximum(0).minimum(x_rows - 1)
  x_g, xt_g, xsc_g = x.after(guard), xt.after(guard), xsc.after(guard)
  u_buf, xf, c_i = UOp.alloc((16,), dtypes.uint32, addrspace=AddrSpace.REG), x_g.flatten(), UOp.range(8, 20)
  fv = _amd_load(xf[src_row * in_features + kt * 128 + kg * 32 + c_i * 4], 4)
  u_r = u_buf.after(UOp.group(
    u_buf[2 * c_i].store(_nat("__builtin_bit_cast(unsigned int, __builtin_amdgcn_cvt_pkrtz({0}, {1}))", dtypes.uint32, fv[0], fv[1])),
    u_buf[2 * c_i + 1].store(_nat("__builtin_bit_cast(unsigned int, __builtin_amdgcn_cvt_pkrtz({0}, {1}))", dtypes.uint32, fv[2], fv[3]))).end(c_i))
  rm_buf, m_i = UOp.alloc((1,), dtypes.uint32, addrspace=AddrSpace.REG), UOp.range(16, 21)
  rm_0 = rm_buf.after(rm_buf.after(u_r)[0].store(z32))
  rm = rm_buf.after(rm_buf[0].store(_pk_max_u16(rm_0.after(m_i)[0].load(), u_r[m_i].load() & 0x7fff7fff)).end(m_i))[0].load()
  rm = _pk_max_u16(rm, _lane_xor(rm, 1, dpp=True))
  rm = _pk_max_u16(rm, _lane_xor(rm, 2, dpp=True))
  asc = _fast_fmax(_fast_fmax(_half(rm & 0xffff), _half(rm >> 16)) * UOp.const(1.0 / 448.0, dtypes.float32),
                   UOp.const(1.0 / 32768.0, dtypes.float32))
  inv_h2, ks_i, lg_i = _f32_to_h2(_fast_rcp(asc)), UOp.range(2, 22), UOp.range(2, 23)
  b_off = ks_i * 8 + lg_i * 4
  p0 = r_ok.where(_h2x2_to_fp8_pk(u_r[b_off].load(), u_r[b_off + 1].load(), inv_h2), z32)
  p1 = r_ok.where(_h2x2_to_fp8_pk(u_r[b_off + 2].load(), u_r[b_off + 3].load(), inv_h2), z32)
  st_xt = xt_g[m_tile, kt * 8 + kg * 2 + ks_i, lg_i * 16 + row16].store(UOp.stack(p0, p1)).end(lg_i, ks_i)
  st_sc = xsc_g[m_tile, kt, row16.valid(kg.eq(0))].store(r_ok.where(asc, UOp.const(0.0, dtypes.float32)))
  return UOp.group(st_xt, st_sc).end(lane, wave, gid).sink(arg=KernelInfo(name="rdna4_quant_x_frag", opts_to_apply=()))

@functools.cache
def _rdna4_gemm_fp8_kernel(out:UOp, raw:UOp, xt:UOp, xsc:UOp, grid:UOp, tok_idx:UOp, tile_exp:UOp,
                           tile_off:UOp, tile_cnt:UOp, total_tiles:UOp, *, out_features:UOp,
                           in_features:UOp, ggml_type:int|UOp, qtypes:tuple[int, ...]|None=None) -> UOp:
  # RDNA4 MoE FP8 WMMA kernel (4 waves, BM=16, BN=64, BK=128) sharing _fp8_load_b128/_fp8_dequant_b128 with dense GEMM.
  qts = qtypes if qtypes is not None else ((ggml_type,) if isinstance(ggml_type, int) else tuple(sorted(OTF_FP8_QUANTS)))
  bn, waves, threads, b_iters = 64, 4, 128, 2
  m_tiles_tot = xt.shape[0]
  assert isinstance(m_tiles_tot, UOp)
  n_blks, k_tiles, k_sb = (out_features + bn - 1) // bn, in_features // 128, in_features // 256
  gid = UOp.range(m_tiles_tot * n_blks, 0, AxisType.GLOBAL)
  block_m, block_n = gid // n_blks, gid % n_blks
  lane, wave = UOp.range(32, -1, AxisType.WARP), UOp.range(waves, 2, AxisType.LOCAL)
  tid, col, lg = wave * 32 + lane, lane % 16, lane // 16
  guard = _nat("({{ if ((int)({0}) >= (int)({1})) return; 0; }})", dtypes.int32, block_m, total_tiles[0].load(), inline=False)
  exp_id = tile_exp.after(guard)[block_m].load().cast(dtypes.weakint)
  raw_g, xt_g, xsc_g, out_g, gdep = raw.after(guard), xt.after(guard), xsc.after(guard), out.after(guard), (guard,)
  raw_u32 = _qraw_u32(raw_g)
  B_lds = UOp.alloc((2, bn // 16, 8, 32, 8), dtypes.fp8e4m3, addrspace=AddrSpace.LOCAL)
  B_u32 = B_lds.flatten().bitcast(dtypes.uint32).reshape(2, bn // 16, 8, 32, 2)
  Bsc_lds = UOp.alloc((2, bn), dtypes.float32, addrspace=AddrSpace.LOCAL)
  acc = UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG)
  sub_acc, eps = UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG), UOp.const(1e-30, dtypes.float32)
  acc = acc.after(acc.after(guard).store(acc.const_like(0.0)))

  def load_b_reg_q(q:int, kt:UOp, dep:tuple[UOp, ...], rid:int) -> tuple[UOp, UOp]:
    rq = _qraw(raw_g, q)
    ru, rd = raw_u32.after(*dep), rq.after(*dep)
    e_u32 = exp_id * ((out_features * in_features * QUANT_SIZES[q] // 256) // 4)
    e_rd = exp_id * ((out_features * in_features * QUANT_SIZES[q] // 256) // rq.dtype.itemsize)
    bi = UOp.range(b_iters, rid)
    bp = tid + bi * threads
    uv = _fp8_load_b128(q, ru, rd, (block_n * bn + bp // 4).minimum(out_features - 1), bp % 4, kt, k_sb, in_features, e_u32, e_rd)
    ub = UOp.alloc((b_iters, len(uv)), dtypes.uint32, addrspace=AddrSpace.REG)
    st = UOp.group(*(ub.after(*dep)[bi, j].store(v) for j, v in enumerate(uv))).end(bi)
    return ub.after(st), st

  def load_b_reg(kt:UOp, dep:tuple[UOp, ...], rid:int) -> tuple[dict[int, UOp], UOp]:
    m = {q: load_b_reg_q(q, kt, dep, rid) for q in qts}
    return {q: v[0] for q, v in m.items()}, _qswitch(ggml_type, [(q, v[1]) for q, v in m.items()])

  def store_b_lds(buf_i:int|UOp, bmap:dict[int, UOp], kt:UOp, dep:tuple[UOp, ...], rid:int) -> UOp:
    dst, sdst = B_u32.after(*dep)[buf_i], Bsc_lds.after(*dep)[buf_i]
    bi = UOp.range(b_iters, rid)
    bp = tid + bi * threads
    bn_i, bg, base_l = bp // 4, bp % 4, (lane // 4) * 4
    gbg, sub = kt * 4 + bg, (kt * 4 + bg) % 8
    t8, sc_r, dep_b = UOp.alloc((8,), dtypes.uint32, addrspace=AddrSpace.REG), UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG), (*dep, bi)
    def put_r(fn_w:Callable[[UOp], UOp], sc:UOp) -> UOp:
      jr = UOp.range(8, rid + 1)
      return UOp.group(t8.after(*dep_b)[jr].store(fn_w(jr)).end(jr), sc_r.after(*dep_b)[0].store(sc))
    def put_r4(fn_pair:Callable[[UOp], tuple[UOp, UOp]], sc:UOp, stride2:bool=False) -> UOp:
      jr = UOp.range(4, rid + 1)
      w0, w1 = fn_pair(jr)
      i0, i1 = (2 * jr, 2 * jr + 1) if stride2 else (jr, jr + 4)
      return UOp.group(UOp.group(t8.after(*dep_b)[i0].store(w0), t8.after(*dep_b)[i1].store(w1)).end(jr), sc_r.after(*dep_b)[0].store(sc))
    def reg_u32(n:int, fn:Callable[[UOp], UOp], r_id:int) -> UOp:
      arr, ri = UOp.alloc((n,), dtypes.uint32, addrspace=AddrSpace.REG), UOp.range(n, r_id)
      return arr.after(arr.after(*dep_b)[ri].store(fn(ri)).end(ri))
    def scale_fn(tsc_need:UOp) -> tuple[UOp, UOp]:
      tsc = _fast_fmax(tsc_need, eps)
      return _fast_rcp(tsc), tsc
    def mx_fn(em:UOp) -> tuple[UOp, UOp]: return em, _e8m0_halved(em) * UOp.const(1.0 / 16.0, dtypes.float32)
    def _deq_q(q:int) -> UOp:
      breg = bmap[q]
      return _fp8_dequant_b128(q, lambda j: breg[bi, j].load(), grid, bg, gbg, sub, lane, base_l, put_r, put_r4, reg_u32, scale_fn, mx_fn, rid)
    f_done = _qswitch(ggml_type, [(q, _deq_q(q)) for q in qts])
    t8_r, sc_v, r_blk, r_in = t8.after(f_done), sc_r.after(f_done)[0].load(), bn_i // 16, bn_i % 16
    k_loc, lg_ = UOp.range(2, rid + 4), UOp.range(2, rid + 5)
    t_off = k_loc * 4 + lg_ * 2
    st_b = dst[r_blk, bg * 2 + k_loc, lg_ * 16 + r_in].store(UOp.stack(t8_r[t_off].load(), t8_r[t_off + 1].load())).end(lg_, k_loc)
    return UOp.group(st_b, sdst[bn_i.valid(bg.eq(0))].store(sc_v)).end(bi)

  kt = UOp.range(k_tiles + 1, 10)
  kt_ld, kt_c, do_cmp, do_st = kt.minimum(k_tiles - 1), (kt - 1).maximum(0), kt > 0, kt < k_tiles
  bc, bn_b = kt_c % 2, kt % 2
  b_map, br_v = load_b_reg(kt_ld, gdep + (kt,), 34)
  r_cmp = UOp.range(do_cmp.cast(dtypes.weakint), 12)
  W_r, Wsc_r = B_lds.after(*gdep, kt, r_cmp, br_v)[bc], Bsc_lds.after(*gdep, kt, r_cmp, br_v)[bc]
  sa = sub_acc.after(sub_acc.after(*gdep, kt, r_cmp, br_v).store(sub_acc.const_like(0.0)))
  kg_r = UOp.range(8, 101)
  bv = _amd_load(xt_g[block_m, kt_c * 8 + kg_r, lane, 0], 2).bitcast(dtypes.fp8e4m3)
  w_op = sub_acc.store(UOp.wmma(W_r[wave, kg_r, lane], bv, sa.after(kg_r), *WMMA_ARG)).end(kg_r)
  e_f = UOp.range(8, 121, AxisType.UPCAST)
  fl = acc[e_f].store(acc.after(*gdep, kt, r_cmp)[e_f].load() + sub_acc.after(w_op)[e_f].load() * (
    Wsc_r[wave * 16 + lg * 8 + e_f].load() * xsc_g[block_m, kt_c, col].load())).end(e_f)
  cmp_done = acc[7].store(_pin_f32(acc.after(fl)[7].load(), fl)).end(r_cmp)
  r_st = UOp.range(do_st.cast(dtypes.weakint), 11)
  st_w = store_b_lds(bn_b, b_map, kt_ld, gdep + (kt, r_st, br_v), 48).end(r_st)
  lp = UOp.group(cmp_done, st_w).barrier().end(kt)
  acc_f, e_o = acc.after(lp), UOp.range(8, 192, AxisType.UPCAST)
  on = block_n * bn + wave * 16 + lg * 8 + e_o
  om_ok = col < tile_cnt.after(guard)[block_m].load().cast(dtypes.weakint)
  om = om_ok.where(tok_idx.after(guard)[(tile_off.after(guard)[block_m].load().cast(dtypes.weakint) + col).valid(om_ok)].load(),
                   UOp.const(0, dtypes.int32)).cast(dtypes.weakint)
  kname = f"linear_{QUANT_NAMES[ggml_type]}_fp8_wmma" if isinstance(ggml_type, int) else "linear_quant_fp8_wmma"
  return out_g[om.valid(om_ok), on.valid(on < out_features)].store(acc_f[e_o].load().cast(out.dtype)).end(
    e_o, wave, lane, gid).sink(arg=KernelInfo(name=kname, opts_to_apply=()))

def _lds_barrier(sink_r:UOp, *deps:UOp, bid:int=0) -> UOp:
  v = (sink_r.after(*deps) if deps else sink_r)[0].load()
  b = _nat(f'({{{{ int _x = ({{0}}); __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local"); '
           f'__builtin_amdgcn_s_barrier(); '
           f'__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local"); _x ^ {bid}; }}}})',
           dtypes.int32, v, inline=False)
  return (sink_r.after(*deps) if deps else sink_r)[0].store(b)

def _sched_barrier(sink_r:UOp, *deps:UOp, bid:int=0) -> UOp:
  v = (sink_r.after(*deps) if deps else sink_r)[0].load()
  b = _nat(f"({{{{ int _x = ({{0}}); __builtin_amdgcn_sched_barrier(0); _x ^ {bid}; }}}})", dtypes.int32, v, inline=False)
  return (sink_r.after(*deps) if deps else sink_r)[0].store(b)

def _lds_st1(ptr:UOp, v:UOp, sink_r:UOp, *deps:UOp) -> UOp:
  st = _nat("({{ *({0}) = ({1}); 0; }})", dtypes.int32, ptr, v, inline=False)
  return (sink_r.after(*deps) if deps else sink_r)[0].store(st)

def _lds_st2(ptr:UOp, v0:UOp, v1:UOp, sink_r:UOp, *deps:UOp) -> UOp:
  st = _nat("({{ typedef unsigned int _u2 __attribute__((ext_vector_type(2))); "
            "*((_u2*)(__builtin_assume_aligned(({0}), 8))) = (_u2){{(unsigned int)({1}), (unsigned int)({2})}}; 0; }})",
            dtypes.int32, ptr, v0, v1, inline=False)
  return (sink_r.after(*deps) if deps else sink_r)[0].store(st)

def _dpp_quad(v:UOp, q:int) -> UOp:
  return _nat(f"__builtin_amdgcn_update_dpp(0, (int)({{0}}), {q * 0x55}, 15, 15, true)", dtypes.uint32, v, inline=False)

def _rfl(v:UOp) -> UOp: return _nat("__builtin_amdgcn_readfirstlane((int)({0}))", dtypes.int32, v.cast(dtypes.int32), inline=False)

@functools.cache
def _rdna4_row_amax_kernel(xsc:UOp, x:UOp) -> UOp:
  # 1 wave (32 lanes) per row, 100% coalesced b128 loads (256 f16/iter), ZERO LDS, ZERO barriers!
  m_tiles, x_rows, in_features = xsc.shape[0], x.shape[0], x.shape[1]
  assert isinstance(m_tiles, UOp) and isinstance(x_rows, UOp) and isinstance(in_features, UOp)
  k_chunks = in_features // 256
  gid, lane = UOp.range(m_tiles * 16, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  r_ok, src_row = gid < x_rows, gid.minimum(x_rows - 1)
  x_u32 = x.flatten().bitcast(dtypes.uint32)
  rm_buf = UOp.alloc((1,), dtypes.uint32, addrspace=AddrSpace.REG)
  rm_buf = rm_buf.after(rm_buf[0].store(UOp.const(0, dtypes.uint32)))
  kc = UOp.range(k_chunks, 10)
  v = _amd_load(x_u32[src_row * (k_chunks * 128) + kc * 128 + lane * 4], 4)
  m = _pk_max_u16(_pk_max_u16(v[0] & 0x7fff7fff, v[1] & 0x7fff7fff), _pk_max_u16(v[2] & 0x7fff7fff, v[3] & 0x7fff7fff))
  rm_d = rm_buf[0].store(_pk_max_u16(rm_buf.after(kc)[0].load(), m)).end(kc)
  rm = rm_buf.after(rm_d)[0].load()
  rm = _pk_max_u16(rm, _lane_xor(rm, 1, dpp=True))
  rm = _pk_max_u16(rm, _lane_xor(rm, 2, dpp=True))
  for msk in (4, 8, 16): rm = _pk_max_u16(rm, _lane_bpermute(lane ^ msk, rm))
  asc = _fast_fmax(_fast_fmax(_half(rm & 0xffff), _half(rm >> 16)) * UOp.const(1.0 / 448.0, dtypes.float32),
                   UOp.const(1.0 / 32768.0, dtypes.float32))
  return xsc[gid // 16, (gid % 16).valid(lane.eq(0))].store(r_ok.where(asc, UOp.const(0.0, dtypes.float32))).end(lane, gid).sink(
    arg=KernelInfo(name="rdna4_row_amax", opts_to_apply=()))

@functools.cache
def _rdna4_quant_x_apply_kernel(xt:UOp, xsc:UOp, x:UOp) -> UOp:
  # 1-pass ZERO-LDS ZERO-BARRIER FP8 fragment quantizer using precomputed per-row xsc (reads hot L3 x_f16!).
  m_tiles, k_steps, x_rows = xt.shape[0], xt.shape[1], x.shape[0]
  assert isinstance(m_tiles, UOp) and isinstance(k_steps, UOp) and isinstance(x_rows, UOp)
  k_chunks = k_steps // 16
  gid = UOp.range(m_tiles * k_chunks, 0, AxisType.GLOBAL)
  wave, lane = UOp.range(4, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  tid, mt, kc = wave * 32 + lane, gid // k_chunks, gid % k_chunks
  row16, kg, z32 = tid // 8, tid % 8, UOp.const(0, dtypes.uint32)
  r_ok, src_row = (mt * 16 + row16) < x_rows, (mt * 16 + row16).minimum(x_rows - 1)
  asc = _fast_fmax(xsc[mt, row16].load(), UOp.const(1.0 / 32768.0, dtypes.float32))
  inv_h2 = _f32_to_h2(_fast_rcp(asc).cast(dtypes.half).float())
  x_u32 = x.flatten().bitcast(dtypes.uint32)
  base = src_row * (k_chunks * 128) + kc * 128 + kg * 16
  us = [v for c in range(4) for v in _v4(_amd_load(x_u32[base + c * 4], 4))]
  pks = [r_ok.where(_h2x2_to_fp8_pk(us[2 * i], us[2 * i + 1], inv_h2), z32) for i in range(8)]
  sts = [xt[mt, kc * 16 + kg * 2 + k_sub, lg_i * 16 + row16].store(UOp.stack(pks[k_sub * 4 + lg_i * 2], pks[k_sub * 4 + lg_i * 2 + 1]))
         for k_sub in (0, 1) for lg_i in (0, 1)]
  return UOp.group(*sts).end(lane, wave, gid).sink(arg=KernelInfo(name="rdna4_quant_x_apply", opts_to_apply=()))

@functools.cache
def _rdna4_gemm_fp8_dense_kernel(out:UOp, raw:UOp, xt:UOp, xsc:UOp, grid:UOp, *, out_features:UOp, in_features:UOp,
                                 ggml_type:int|UOp, tm:int|UOp, tn:int|UOp, bk:int, wm:int|UOp=4, qtypes:tuple[int, ...]|None=None) -> UOp:
  # Double-buffered W_lds[2], 1 LDS-only barrier/slab (ZERO s_wait_loadcnt at barriers!), online register wRef, per-row xsc (0 K-loop xsc rescale!).
  # Exact 1:1 thread mapping: threads == bn * 4 (wm == 2 * tn -> 16 wts/thr for bk=64, 32 wts/thr for bk=128; ZERO idle waves!).
  qts = qtypes if qtypes is not None else ((ggml_type,) if isinstance(ggml_type, int) else tuple(sorted(OTF_FP8_QUANTS)))
  wn = 2
  bm, bn, waves = wm * tm * 16, wn * tn * 16, wm * wn
  threads, ns = waves * 32, bk // 16
  assert bk in (64, 128) and (not isinstance(threads, int) or threads == bn * 4)
  m_tiles_tot, tok_out = xt.shape[0], out.shape[0]
  assert isinstance(m_tiles_tot, UOp) and isinstance(tok_out, UOp)
  m_blks, n_blks, k_tiles, k_sb = (tok_out + bm - 1) // bm, (out_features + bn - 1) // bn, in_features // bk, in_features // 256
  gid = UOp.range(m_blks * n_blks, 0, AxisType.GLOBAL)
  swz = (m_blks % 4).eq(0).where(4, (m_blks % 2).eq(0).where(2, 1))
  grp, in_grp = gid // (swz * n_blks), gid % (swz * n_blks)
  block_m, block_n = grp * swz + in_grp % swz, in_grp // swz
  lane, wave = UOp.range(32, -1, AxisType.WARP), UOp.range(waves, 2, AxisType.LOCAL)
  tid, wave_m, wave_n, col, lg = wave * 32 + lane, wave // wn, wave % wn, lane % 16, lane // 16
  raw_u32 = _qraw_u32(raw)
  w_str_u32 = bk // 4 + 2  # +8 byte pad -> conflict-free ds_load_b64
  W_lds_u32 = UOp.alloc((2, bn, w_str_u32), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  sRef = UOp.alloc((bn,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  sRatio = UOp.alloc((2, bn), dtypes.float32, addrspace=AddrSpace.LOCAL)
  sRatioTag = UOp.alloc((2, bn), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  sFlag = UOp.alloc((2, wn), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  acc = UOp.alloc((tm, tn, 8), dtypes.float32, addrspace=AddrSpace.REG)
  ref_r = UOp.alloc((1,), dtypes.uint32, addrspace=AddrSpace.REG)
  sink_r = UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG)
  z_u, z_f, one_f, eps = UOp.const(0, dtypes.uint32), UOp.const(0.0, dtypes.float32), UOp.const(1.0, dtypes.float32), UOp.const(1e-30, dtypes.float32)
  # Init LDS via _nat stores + _lds_barrier (zero Ops.STORE to LOCAL -> zero __syncthreads()!)
  s0_init = sink_r[0].store(UOp.const(0, dtypes.int32))
  r_it1 = UOp.range((tid < 2 * bn).cast(dtypes.weakint), 20)
  st_rt0 = _lds_st1(sRatioTag[tid // bn, tid % bn], z_u, sink_r, s0_init, r_it1).end(r_it1)
  r_it2 = UOp.range((tid < 2 * wn).cast(dtypes.weakint), 21)
  st_fl0 = _lds_st1(sFlag[tid // wn, tid % wn], z_u, sink_r, s0_init, r_it2).end(r_it2)
  init_done = _lds_barrier(sink_r, s0_init, acc.store(acc.const_like(0.0)), ref_r[0].store(z_u), st_rt0, st_fl0, bid=1)
  bn_i, bg = tid // 4, tid % 4
  qg, hg, base_l = bg // 2, bg & 1, (lane // 4) * 4
  gn = (block_n * bn + bn_i).minimum(out_features - 1)
  xt_u32, ksteps_g = xt.flatten(), in_features // 16
  mt_base = block_m * (wm * tm) + wave_m * tm

  wreg_n = {MXFP4: (3, 5), Q8_0: (5, 9), Q4_K: (3, 5), Q5_K: (5, 7), Q6_K: (8, 8), Q3_K: (7, 7),
            IQ4_XS: (3, 5), Q2_K: (6, 10), IQ3_S: (5, 6), IQ3_XXS: (3, 4), IQ2_S: (5, 5)}
  w_reg = {q: UOp.alloc((wreg_n[q][0 if bk == 64 else 1],), dtypes.uint32, addrspace=AddrSpace.REG) for q in qts}

  def load_b_reg_q(q:int, kt_v:UOp, dep:tuple[UOp, ...]) -> UOp:
    rq = _qraw(raw, q)
    ru, rd = raw_u32.after(*dep), rq.after(*dep)
    if bk == 64:  # 16 weights/thread (bg=0..3: qg=0..1, hg=0..1)
      gbg = kt_v * 2 + qg
      sub, sb = gbg % 8, gn * k_sb + gbg // 8
      if q == MXFP4:
        a32 = ((gn * (in_features // 32) + kt_v * 2) * 17) // 4 + bg * 2
        v2 = _amd_load(ru[a32], 2)
        uv = [v2[0], v2[1], ru[a32 + 2].load()]
      elif q == Q8_0:
        b32 = (gn * (in_features // 64) + kt_v) * 17
        off = bg.eq(0).where(0, bg.eq(1).where(4, bg.eq(2).where(9, 13)))
        uv = _v4(_amd_load(ru[b32 + off], 4)) + [ru[b32 + bg.eq(0).where(4, 8)].load()]
      elif q == Q4_K:
        b, pair = sb * 36, kt_v % 4
        v2 = _amd_load(ru[b + 4 + pair * 8 + bg * 2], 2)
        uv = [v2[0], v2[1], ru[b + bg].load()]
      elif q == Q5_K:
        b, pair = sb * 44, kt_v % 4
        qs2, qh2 = _amd_load(ru[b + 12 + pair * 8 + bg * 2], 2), _amd_load(ru[b + 4 + bg * 2], 2)
        uv = [qs2[0], qs2[1], qh2[0], qh2[1], ru[b + bg].load()]
      elif q == Q6_K:
        b, s4 = sb * 105, sub // 4
        ql = _v4(_amd_load(rd[b + s4 * 32 + qg * 16 + hg * 8], 8).bitcast(dtypes.uint32))
        qh = _amd_load(rd[b + 64 + s4 * 16 + (hg * 2 + qg) * 4], 4).bitcast(dtypes.uint32)
        uv = ql + [qh[0], qh[1], rd[b + 96 + sub].load().cast(dtypes.uint32), rd[b + 104].load().cast(dtypes.uint32)]
      elif q == Q3_K:
        b, s4, p_idx = sb * 55, sub // 4, hg * 2 + qg
        hm = _amd_load(rd[b + p_idx * 4], 4).bitcast(dtypes.uint32)
        qs = _amd_load(rd[b + 16 + s4 * 16 + p_idx * 4], 4).bitcast(dtypes.uint32)
        uv = [hm[0], hm[1], qs[0], qs[1], rd[b + 48 + (sub % 4)].load().cast(dtypes.uint32),
              rd[b + 52 + (sub & 1)].load().cast(dtypes.uint32), rd[b + 54].load().cast(dtypes.uint32)]
      elif q == IQ4_XS:
        b = sb * 34
        v2 = _amd_load(ru[b + 2 + sub * 4 + hg * 2], 2)
        uv = [v2[0], v2[1], ru[b + hg].load()]
      elif q == Q2_K:
        b32 = sb * 21
        uv = _v4(_amd_load(ru[b32 + 4 + (sub // 4) * 8 + hg * 4], 4)) + [ru[b32 + sub // 2].load(), ru[b32 + 20].load()]
      elif q == IQ3_S:
        b = sb * 55
        uv = [_amd_load(rd[b + 1 + sub * 4 + hg * 2], 2).bitcast(dtypes.uint32)[0], rd[b + 37 + sub * 2 + hg].load().cast(dtypes.uint32),
              rd[b + 33 + sub // 2].load().cast(dtypes.uint32), rd[b + 53 + sub // 4].load().cast(dtypes.uint32), rd[b].load().cast(dtypes.uint32)]
      elif q == IQ3_XXS:
        b = sb * 49
        uv = [_amd_load(rd[b + 1 + sub * 4 + hg * 2], 2).bitcast(dtypes.uint32)[0],
              _amd_load(rd[b + 33 + sub * 2], 2).bitcast(dtypes.uint32)[0], rd[b].load().cast(dtypes.uint32)]
      else:  # IQ2_S
        b = sb * 41
        uv = [rd[b + 1 + sub * 2 + hg].load().cast(dtypes.uint32), rd[b + 17 + sub * 2 + hg].load().cast(dtypes.uint32),
              rd[b + 33 + sub // 2].load().cast(dtypes.uint32), rd[b + 37 + sub // 2].load().cast(dtypes.uint32), rd[b].load().cast(dtypes.uint32)]
    else: uv = _fp8_load_b128(q, ru, rd, gn, bg, kt_v, k_sb, in_features)
    return UOp.group(*(w_reg[q].after(*dep)[j].store(v) for j, v in enumerate(uv)))

  def load_b_reg(kt_v:UOp, dep:tuple[UOp, ...]) -> UOp:
    return _qswitch(ggml_type, [(q, load_b_reg_q(q, kt_v, dep)) for q in qts])

  def store_b_lds(buf_i:UOp, bmap:dict[int, UOp], kt_v:UOp, dep:tuple[UOp, ...], rid:int, allow_upd:UOp|bool=True) -> UOp:
    dst, rat_d, rtag_d, flg_d = W_lds_u32.after(*dep)[buf_i], sRatio.after(*dep)[buf_i], sRatioTag.after(*dep)[buf_i], sFlag.after(*dep)
    tag = (kt_v + 1).cast(dtypes.uint32)
    t_out = UOp.alloc((4 if bk == 64 else 8,), dtypes.uint32, addrspace=AddrSpace.REG)
    nr_r, rat_r, upd_r = (UOp.alloc((1,), dt, addrspace=AddrSpace.REG) for dt in (dtypes.uint32, dtypes.float32, dtypes.bool))
    def _done(st_t:UOp, new_r_u:UOp, rat_v:UOp, upd:UOp) -> UOp:
      return UOp.group(st_t, nr_r.after(*dep)[0].store(new_r_u), rat_r.after(*dep)[0].store(rat_v), upd_r.after(*dep)[0].store(upd))
    def _upd_ref(tsc_need:UOp) -> tuple[UOp, UOp, UOp, UOp]:
      old_u = _nat("({{ unsigned int _v = ({0}); __builtin_amdgcn_s_setprio(2); _v; }})", dtypes.uint32, ref_r.after(*dep)[0].load(), inline=False)
      old_r = old_u.bitcast(dtypes.float32)
      upd = (tsc_need > old_r) & allow_upd if isinstance(allow_upd, UOp) else (tsc_need > old_r)
      new_r = upd.where(_fast_fmax(tsc_need * UOp.const(8.0, dtypes.float32), eps), old_r)
      inv_r = (new_r > z_f).where(_fast_rcp(new_r), z_f)
      return new_r.bitcast(dtypes.uint32), inv_r, (old_r > z_f).where(old_r * inv_r, z_f), upd
    def _upd_mx_ref(em:UOp) -> tuple[UOp, UOp, UOp]:
      old_r = _nat("({{ unsigned int _v = ({0}); __builtin_amdgcn_s_setprio(2); _v; }})", dtypes.uint32, ref_r.after(*dep)[0].load(), inline=False)
      upd = (em > old_r) & allow_upd if isinstance(allow_upd, UOp) else (em > old_r)
      new_r = upd.where((em + UOp.const(5, dtypes.uint32)).minimum(UOp.const(254, dtypes.uint32)), old_r)
      sh_e = new_r - old_r
      mx_rat = (sh_e >= UOp.const(127, dtypes.uint32)).where(z_f, ((UOp.const(127, dtypes.uint32) - sh_e) << 23).bitcast(dtypes.float32))
      return new_r, mx_rat, upd
    def fill_q(q:int) -> UOp:
      breg = bmap[q]
      def get_u(j:int|UOp) -> UOp: return breg[j].load()
      if bk == 64:  # 16 weights/thread -> 4 fp8 u32s -> 2x ds_store_b64!
        gbg = kt_v * 2 + qg
        sub = gbg % 8
        def put4_r(fn_w:Callable[[UOp], UOp], new_r_u:UOp, rat_v:UOp, upd:UOp) -> UOp:
          jr = UOp.range(4, rid + 1)
          return _done(t_out.after(*dep)[jr].store(fn_w(jr)).end(jr), new_r_u, rat_v, upd)
        def put4_r2(fn_p:Callable[[UOp], tuple[UOp, UOp]], new_r_u:UOp, rat_v:UOp, upd:UOp, stride2:bool=False) -> UOp:
          jr = UOp.range(2, rid + 1)
          w0, w1 = fn_p(jr)
          i0, i1 = (2 * jr, 2 * jr + 1) if stride2 else (jr, jr + 2)
          return _done(UOp.group(t_out.after(*dep)[i0].store(w0), t_out.after(*dep)[i1].store(w1)).end(jr), new_r_u, rat_v, upd)
        if q == MXFP4:
          h2 = (((gn * (in_features // 32) + kt_v * 2) * 17 // 2) & 1).cast(dtypes.uint32)
          rem = h2 * 2 + qg.cast(dtypes.uint32)
          w0, w1, w2 = get_u(0), get_u(1), get_u(2)
          e = (hg.eq(0).where(w0, _lane_xor(w0, 1, dpp=True)) >> (rem * 8)) & 255
          new_r, mx_rat, upd = _upd_mx_ref(e.maximum(_lane_xor(e, 2, dpp=True)))
          l0, l1 = _mx_fp8_luts(e, new_r)
          sh_b = (rem + 1) & 3
          pw0, pw1 = rem.eq(3).where(w1, _perm_alignbyte(w1, w0, sh_b)), rem.eq(3).where(w2, _perm_alignbyte(w2, w1, sh_b))
          ws = (_amd_byte_perm(l1, l0, pw0 & 0x07070707) | ((pw0 & 0x08080808) << 4),
                _amd_byte_perm(l1, l0, pw1 & 0x07070707) | ((pw1 & 0x08080808) << 4),
                _amd_byte_perm(l1, l0, (pw0 >> 4) & 0x07070707) | (pw0 & 0x80808080),
                _amd_byte_perm(l1, l0, (pw1 >> 4) & 0x07070707) | (pw1 & 0x80808080))
          return _done(UOp.group(*(t_out.after(*dep)[i].store(v) for i, v in enumerate(ws))), new_r, mx_rat, upd)
        if q == Q8_0:
          w0, w4 = get_u(0), get_u(4)
          sc = _half(qg.eq(0).where(hg.eq(0).where(w0 & 0xffff, _lane_xor(w0 & 0xffff, 1, dpp=True)), w4 >> 16))
          new_r_u, inv_r, rat_v, upd = _upd_ref(_fast_fmax(sc.abs(), _lane_xor(sc.abs(), 2, dpp=True)))
          sn, c80 = sc * inv_r, UOp.const(0x80808080, dtypes.uint32)
          sel = qg.eq(0).where(UOp.const(0x05040302, dtypes.uint32), UOp.const(0x03020100, dtypes.uint32))
          return put4_r(lambda j: _b4_to_fp8(_amd_byte_perm(get_u(j + 1), get_u(j), sel) ^ c80, sn, -8388736.0), new_r_u, rat_v, upd)
        if q in (Q4_K, Q5_K):
          hdr = [_dpp_quad(get_u(2 if q == Q4_K else 4), k) for k in range(4)]
          tsc_need = _fast_fmax(_half(hdr[0] & 0xffff).abs() * UOp.const(4.0 if q == Q4_K else 8.0, dtypes.float32),
                                _half(hdr[0] >> 16).abs() * UOp.const(0.25, dtypes.float32))
          new_r_u, inv_r, rat_v, upd = _upd_ref(tsc_need)
          ds0, dm0, ds1, dm1 = _q4k_pair_scales(hdr, kt_v % 4)
          is_lo = qg.eq(0)
          ds_m, dm_m = is_lo.where(ds0, ds1) * inv_r, is_lo.where(-dm0, -dm1) * inv_r
          sh = is_lo.where(UOp.const(0, dtypes.uint32), UOp.const(4, dtypes.uint32))
          def q45_w(j:UOp) -> UOp:
            qm = get_u(j % 2)
            qo = _lane_xor(qm, 2, dpp=True)
            qbits = ((j < 2).where(is_lo.where(qm, qo), is_lo.where(qo, qm)) >> sh) & 0x0f0f0f0f
            if q == Q5_K:
              hm = get_u(2 + (j % 2))
              ho = _lane_xor(hm, 2, dpp=True)
              qbits = qbits | ((((j < 2).where(is_lo.where(hm, ho), is_lo.where(ho, hm)) >> _sh(sub)) & 0x01010101) << 4)
            return _b4_to_fp8(qbits, ds_m, -8388608.0, dm_m)
          return put4_r(q45_w, new_r_u, rat_v, upd)
        if q == Q6_K:
          sc_w, d_f = get_u(6), _half(get_u(7))
          new_r_u, inv_r, rat_v, upd = _upd_ref(d_f.abs() * UOp.const(16.0, dtypes.float32))
          s_b = ((sc_w >> _sh(hg * 8)) & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float() * (d_f * inv_r)
          is_lo, ql_sh, qh_sh = qg.eq(0), _sh(((sub % 4) // 2) * 4), _sh((sub % 4) * 2)
          def q6_w(j:UOp) -> UOp:
            qhm = get_u(4 + (j % 2))
            qho = _lane_xor(qhm, 2, dpp=True)
            qhv = (j < 2).where(is_lo.where(qhm, qho), is_lo.where(qho, qhm))
            return _b4_to_fp8(((get_u(j) >> ql_sh) & 0x0f0f0f0f) | (((qhv >> qh_sh) & 0x03030303) << 4), s_b, -8388640.0)
          return put4_r(q6_w, new_r_u, rat_v, upd)
        if q == Q3_K:
          sl16, sh16, d_f = get_u(4), get_u(5), _half(get_u(6))
          new_r_u, inv_r, rat_v, upd = _upd_ref(d_f.abs() * UOp.const(0.5, dtypes.float32))
          q3_sc = ((((sl16 >> _sh(hg * 8 + (sub // 4) * 4)) & 15) |
                    (((sh16 >> _sh(hg * 8 + (sub // 2) * 2)) & 3) << 4)).cast(dtypes.int32) - 32).float() * (d_f * inv_r)
          l0, l1 = _pack4_fp8(*(float(v) * q3_sc for v in (-4, -3, -2, -1))), _pack4_fp8(*(float(v) * q3_sc for v in (0, 1, 2, 3)))
          is_lo, qs_sh, hm_sh = qg.eq(0), _sh((sub % 4) * 2), _sh(sub)
          def q3_w(j:UOp) -> UOp:
            hmm, qsm = get_u(j % 2), get_u(2 + (j % 2))
            hmo, qso = _lane_xor(hmm, 2, dpp=True), _lane_xor(qsm, 2, dpp=True)
            hmv = (j < 2).where(is_lo.where(hmm, hmo), is_lo.where(hmo, hmm))
            qsv = (j < 2).where(is_lo.where(qsm, qso), is_lo.where(qso, qsm))
            return _amd_byte_perm(l1, l0, ((qsv >> qs_sh) & 0x03030303) | (((hmv >> hm_sh) & 0x01010101) << 2))
          return put4_r(q3_w, new_r_u, rat_v, upd)
        if q == IQ4_XS:
          hm, ho = get_u(2), _lane_xor(get_u(2), 1, dpp=True)
          h0, h1 = hg.eq(0).where(hm, ho), hg.eq(0).where(ho, hm)
          d_f = _half(h0)
          new_r_u, inv_r, rat_v, upd = _upd_ref(d_f.abs() * UOp.const(32.0 / 3.0, dtypes.float32))
          sc_i = (((h1 >> _sh(4 * sub)) & 15) | (((h0 >> _sh(16 + 2 * sub)) & 3) << 4)).cast(dtypes.int32) - 32
          sc = sc_i.float() * (d_f * inv_r)
          lut = tuple(_pack4_fp8(*(UOp.const(float(v), dtypes.float32) * sc for v in _IQ4NL_VALS[k:k + 4])) for k in (0, 4, 8, 12))
          return put4_r(lambda j: _nibble_lut(get_u(j % 2), _sh((j // 2) * 4), lut), new_r_u, rat_v, upd)
        if q == Q2_K:
          sc_b, dm_w = (get_u(4) >> _sh((sub & 1) * 16 + hg * 8)) & 255, get_u(5)
          dr, dmr = _half(dm_w & 0xffff), _half(dm_w >> 16)
          new_r_u, inv_r, rat_v, upd = _upd_ref(_fast_fmax(dr.abs(), dmr.abs()) * UOp.const(0.25, dtypes.float32))
          sh = _sh((sub % 4) * 2)
          l0 = _pack4_fp8(*(float(i) * dr * inv_r * (sc_b & 15).float() - dmr * inv_r * (sc_b >> 4).float() for i in range(4)))
          return put4_r(lambda j: _amd_byte_perm(z_u, l0, (get_u(j) >> sh) & 0x03030303), new_r_u, rat_v, upd)
        if q == IQ3_S:
          q_h, s_h, d_f = get_u(0), get_u(1), _half(get_u(4))
          qh_4, sc_n = (get_u(2) >> _sh((sub & 1) * 8 + hg * 4)) & 15, (get_u(3) >> _sh((sub & 3) * 4)) & 15
          sc_m = sc_n.maximum(_lane_xor(sc_n, 2, dpp=True))
          new_r_u, inv_r, rat_v, upd = _upd_ref(d_f.abs() * (1.0 + 2.0 * sc_m.float()) * UOp.const(15.0 / 448.0, dtypes.float32))
          sn = (1.0 + 2.0 * sc_n.float()) * (d_f * inv_r)
          l0, l1 = _pack4_fp8(*(float(v) * sn for v in (1, 3, 5, 7))), _pack4_fp8(*(float(v) * sn for v in (9, 11, 13, 15)))
          return put4_r(lambda i: _amd_byte_perm(
            l1, l0, (grid[((q_h >> _sh(i * 8)) & 255) | (((qh_4 >> _sh(i)) & 1) << 8)].load() >> 1) & 0x07070707) ^
            _iq_sgn(s_h >> _sh(i * 4)), new_r_u, rat_v, upd)
        if q == IQ3_XXS:
          q_h, sc_w, d_f = get_u(0), get_u(1), _half(get_u(2))
          new_r_u, inv_r, rat_v, upd = _upd_ref(d_f.abs() * UOp.const(1.5, dtypes.float32))
          sn = ((sc_w >> 28).float() + 0.5) * UOp.const(0.5, dtypes.float32) * (d_f * inv_r)
          l0, l1 = _pack4_fp8(*(float(v) * sn for v in (4, 12, 20, 28))), _pack4_fp8(*(float(v) * sn for v in (36, 44, 52, 62)))
          def iq3x64_pair(p_i:UOp) -> tuple[UOp, UOp]:
            qw, sgn = q_h >> _sh(p_i * 16), _iq_even_signs((sc_w >> _sh(7 * (hg * 2 + p_i))) & 127)
            return (_amd_byte_perm(l1, l0, (grid[qw & 255].load() >> 3) & 0x07070707) ^ _iq_sgn(sgn),
                    _amd_byte_perm(l1, l0, (grid[(qw >> 8) & 255].load() >> 3) & 0x07070707) ^ _iq_sgn(sgn >> 4))
          return put4_r2(iq3x64_pair, new_r_u, rat_v, upd, stride2=True)
        q_h, s_h, bsh, d_f = get_u(0), get_u(1), _sh((sub & 1) * 8 + hg * 4), _half(get_u(4))
        qh_4, sc_4 = (get_u(2) >> bsh) & 15, (get_u(3) >> bsh) & 15
        new_r_u, inv_r, rat_v, upd = _upd_ref(d_f.abs() * UOp.const(0.5, dtypes.float32))
        sf = (sc_4.float() + 0.5) * UOp.const(0.25, dtypes.float32) * (d_f * inv_r)
        l0 = _pack4_fp8(8.0 * sf, 25.0 * sf, 43.0 * sf, z_f)
        return put4_r(lambda j: (p := j // 2, k := j % 2, _amd_byte_perm(z_u, l0,
                      (grid[(((q_h >> _sh(8 * p)) & 255) | (((qh_4 >> _sh(2 * p)) & 3) << 8)) * 2 + k].load() >> 4) & 0x03030303) ^
                      _iq_sgn(s_h >> _sh(8 * p + 4 * k)))[2], new_r_u, rat_v, upd)

      # bk == 128: 32 weights/thread -> 8 fp8 u32s -> 4x ds_store_b64 (shared with MoE FP8!)
      gbg = kt_v * 4 + bg
      def put_r(fn_w:Callable[[UOp], UOp], meta:tuple[UOp, UOp, UOp]) -> UOp:
        jr = UOp.range(8, rid + 1)
        return _done(t_out.after(*dep)[jr].store(fn_w(jr)).end(jr), *meta)
      def put_r4(fn_pair:Callable[[UOp], tuple[UOp, UOp]], meta:tuple[UOp, UOp, UOp], stride2:bool=False) -> UOp:
        jr = UOp.range(4, rid + 1)
        w0, w1 = fn_pair(jr)
        i0, i1 = (2 * jr, 2 * jr + 1) if stride2 else (jr, jr + 4)
        return _done(UOp.group(t_out.after(*dep)[i0].store(w0), t_out.after(*dep)[i1].store(w1)).end(jr), *meta)
      def _reg_u32(n:int, fn:Callable[[UOp], UOp], r_id:int) -> UOp:
        arr, ri = UOp.alloc((n,), dtypes.uint32, addrspace=AddrSpace.REG), UOp.range(n, r_id)
        return arr.after(arr.after(*dep)[ri].store(fn(ri)).end(ri))
      def scale_fn(tsc_need:UOp) -> tuple[UOp, tuple[UOp, UOp, UOp]]:
        new_r_u, inv_r, rat_v, upd = _upd_ref(tsc_need)
        return inv_r, (new_r_u, rat_v, upd)
      def mx_fn(em:UOp) -> tuple[UOp, tuple[UOp, UOp, UOp]]:
        new_r, mx_rat, upd = _upd_mx_ref(em)
        return new_r, (new_r, mx_rat, upd)
      return _fp8_dequant_b128(q, get_u, grid, bg, gbg, gbg % 8, lane, base_l, put_r, put_r4, _reg_u32, scale_fn, mx_fn, rid)
    f_done = _qswitch(ggml_type, [(q, fill_q(q)) for q in qts])
    tr, new_r_u, rat_v, upd = t_out.after(f_done), nr_r.after(f_done)[0].load(), rat_r.after(f_done)[0].load(), upd_r.after(f_done)[0].load()
    r_u = UOp.range((upd & bg.eq(0)).cast(dtypes.weakint), rid + 4)
    st_u = UOp.group(_lds_st1(rat_d.after(f_done, r_u)[bn_i], rat_v, sink_r, f_done, r_u),
                     _lds_st1(rtag_d.after(f_done, r_u)[bn_i], tag, sink_r, f_done, r_u),
                     _lds_st1(flg_d.after(f_done, r_u)[buf_i, bn_i // (tn * 16)], tag, sink_r, f_done, r_u)).end(r_u)
    ref_st = UOp.group(ref_r.after(f_done)[0].store(new_r_u), st_u)
    if bk == 64:
      is_sp = (ggml_type in (MXFP4, Q4_K, Q5_K, IQ4_XS)) if isinstance(ggml_type, int) else (
        ggml_type.eq(MXFP4) | ggml_type.eq(Q4_K) | ggml_type.eq(Q5_K) | ggml_type.eq(IQ4_XS))
      p0 = is_sp.where(qg * 8 + hg * 2, bg * 4) if isinstance(is_sp, UOp) else (qg * 8 + hg * 2 if is_sp else bg * 4)
      p1 = is_sp.where(qg * 8 + hg * 2 + 4, bg * 4 + 2) if isinstance(is_sp, UOp) else (qg * 8 + hg * 2 + 4 if is_sp else bg * 4 + 2)
      return UOp.group(_lds_st2(dst[bn_i, p0], tr[0].load(), tr[1].load(), sink_r, f_done),
                       _lds_st2(dst[bn_i, p1], tr[2].load(), tr[3].load(), sink_r, f_done), ref_st)
    k_st = UOp.range(4, rid + 5)
    return UOp.group(_lds_st2(dst[bn_i, bg * 8 + k_st * 2], tr[k_st * 2].load(), tr[k_st * 2 + 1].load(), sink_r, f_done, k_st).end(k_st), ref_st)

  def _by_tn(fn:Callable[[int], UOp]) -> UOp:
    return UOp(Ops.WHERE, src=(tn.eq(2), fn(2), fn(1))) if isinstance(tn, UOp) else fn(tn)

  def _rescale_block(kt_c:UOp, cur_b:UOp, dep:tuple[UOp, ...]) -> UOp:
    # Rare (<1%) wRef rescale (per-wave_n sFlag) — ZERO xsc rescale in K-loop!
    tag_c = (kt_c + 1).cast(dtypes.uint32)
    flg_v = _rfl(sFlag.after(*dep)[cur_b, wave_n].load())
    r_res = UOp.range(((kt_c > 0) & flg_v.eq((kt_c + 1).cast(dtypes.int32))).cast(dtypes.weakint), 65)
    rat_r, rtag_r = sRatio.after(*dep, r_res)[cur_b], sRatioTag.after(*dep, r_res)[cur_b]
    ri_r, re_r = UOp.range(tm, 66), UOp.range(8, 67, AxisType.UPCAST)
    def _res_tn(tn_c:int) -> UOp:
      r_c = [rtag_r[(wave_n * tn_c + j) * 16 + col].load().eq(tag_c).where(rat_r[(wave_n * tn_c + j) * 16 + col].load(), one_f)
             for j in range(tn_c)]
      return UOp.group(*[acc[ri_r, j, re_r].store(acc.after(*dep, r_res)[ri_r, j, re_r].load() * r_c[j])
                         for j in range(tn_c)])
    return _by_tn(_res_tn).end(re_r, ri_r, r_res)

  if bk == 64:
    # Large-M (WM=8, TN=4, BK=64): unified single-copy 2-stage pipeline (1x load_b_reg + 1x store_b_lds in UIR!).
    # kt=0: load(0); kt=1: store(0)+load(1)+bar; kt=2..k_tiles+1: rescale(kt-2)+af[0]+store(kt-1)+early load(kt)+64xWMMA+bar!
    assert isinstance(tn, int)
    kt = UOp.range(k_tiles + 2, 10)
    kt_c = (kt - 2).maximum(0)
    cur_b, st_b = kt_c % 2, (kt - 1).maximum(0) % 2
    af_buf = UOp.alloc((2, tm, 2), dtypes.uint32, addrspace=AddrSpace.REG)
    r_pre = UOp.range((kt >= 2).cast(dtypes.weakint), 50)
    res_done = _rescale_block(kt_c, cur_b, (init_done, kt, r_pre))
    ia_0 = UOp.range(tm, 55)
    a0_idx = ((mt_base + ia_0).minimum(m_tiles_tot - 1) * ksteps_g + kt_c * ns) * 64 + lane * 2
    av0 = _amd_load(xt_u32.after(res_done, ia_0)[a0_idx], 2)
    pre_done = UOp.group(af_buf.after(res_done)[0, ia_0, 0].store(av0[0]),
                         af_buf.after(res_done)[0, ia_0, 1].store(av0[1])).end(ia_0, r_pre)
    r_st = UOp.range((kt >= 1).cast(dtypes.weakint), 51)
    st_w = store_b_lds(st_b, {q: w_reg[q].after(init_done, kt, pre_done, r_st) for q in qts},
                       (kt - 1).maximum(0).minimum(k_tiles - 1), (init_done, kt, pre_done, r_st), 48, allow_upd=(kt <= k_tiles)).end(r_st)
    r_wl = UOp.range((kt < k_tiles).cast(dtypes.weakint), 52)
    wl_done = load_b_reg(kt.minimum(k_tiles - 1), (init_done, kt, st_w, r_wl)).end(r_wl)
    r_wm = UOp.range((kt >= 2).cast(dtypes.weakint), 53)
    W_r = W_lds_u32.after(init_done, kt, pre_done, r_wm)[cur_b]
    step_r = UOp.range(ns, 100)
    s_cur, s_nxt = step_r % 2, (step_r + 1) % 2
    r_an = UOp.range((step_r + 1 < ns).cast(dtypes.weakint), 102)
    ia_n = UOp.range(tm, 103)
    an_idx = ((mt_base + ia_n).minimum(m_tiles_tot - 1) * ksteps_g + kt_c * ns + (step_r + 1).minimum(ns - 1)) * 64 + lane * 2
    avn = _amd_load(xt_u32.after(pre_done, st_w, wl_done, r_wm, step_r, r_an, ia_n)[an_idx], 2)
    afn_done = UOp.group(af_buf.after(pre_done, st_w, wl_done, r_wm, step_r, r_an)[s_nxt, ia_n, 0].store(avn[0]),
                         af_buf.after(pre_done, st_w, wl_done, r_wm, step_r, r_an)[s_nxt, ia_n, 1].store(avn[1])).end(ia_n, r_an)
    wf_lo = [W_r[(wave_n * tn + j) * 16 + col, step_r * 4 + lg * 2].load() for j in range(tn)]
    wf_hi = [W_r[(wave_n * tn + j) * 16 + col, step_r * 4 + lg * 2 + 1].load() for j in range(tn)]
    wf_hi[-1] = _nat("({{ unsigned int _v = ({0}); __builtin_amdgcn_sched_barrier(0); if ({1}) __builtin_amdgcn_s_setprio(0); _v; }})",
                     dtypes.uint32, wf_hi[-1], step_r.eq(0), inline=False)
    wm_i = UOp.range(tm, 101)
    af_cur = af_buf.after(pre_done, st_w, wl_done, afn_done, r_wm, step_r)
    a_f8 = UOp.stack(af_cur[s_cur, wm_i, 0].load(), af_cur[s_cur, wm_i, 1].load()).bitcast(dtypes.fp8e4m3)
    w_ops = UOp.group(*[acc[wm_i, j].store(UOp.wmma(a_f8, UOp.stack(wf_lo[j], wf_hi[j]).bitcast(dtypes.fp8e4m3),
                        acc.after(init_done, kt, pre_done, afn_done, r_wm, step_r)[wm_i, j], *WMMA_ARG))
                        for j in range(tn)]).end(wm_i)
    w_ops = acc[0, 0, 7].store(_nat("({{ float _v = ({0}); __builtin_amdgcn_sched_barrier(0); _v; }})",
                                    dtypes.float32, acc.after(w_ops)[0, 0, 7].load(), inline=False))
    cmp_done = w_ops.end(step_r, r_wm)
    b_end = _lds_barrier(sink_r, st_w, wl_done, cmp_done, bid=2).end(kt)
  else:
    # Small/Medium-M (TN=1,2, BK=128): unified single-copy Saber PIPE=1 loop (1x load_b_reg + 1x store_b_lds in UIR!).
    kt = UOp.range(k_tiles + 1, 10)
    kt_c = (kt - 1).maximum(0)
    cur_b = kt_c % 2
    af_all = UOp.alloc((tm, ns, 2), dtypes.uint32, addrspace=AddrSpace.REG)
    r_st = UOp.range((kt > 0).cast(dtypes.weakint), 50)
    st_w = store_b_lds(cur_b, {q: w_reg[q].after(init_done, kt, r_st) for q in qts}, kt_c, (init_done, kt, r_st), 48)
    ia_r, st_r = UOp.range(tm, 55), UOp.range(ns, 56)
    a_idx = ((mt_base + ia_r).minimum(m_tiles_tot - 1) * ksteps_g + kt_c * ns + st_r) * 64 + lane * 2
    av = _amd_load(xt_u32.after(init_done, kt, r_st, ia_r, st_r)[a_idx], 2)
    af_done = UOp.group(af_all.after(init_done, kt, r_st)[ia_r, st_r, 0].store(av[0]),
                        af_all.after(init_done, kt, r_st)[ia_r, st_r, 1].store(av[1])).end(st_r, ia_r)
    st_af_done = UOp.group(st_w, af_done).end(r_st)
    r_wl = UOp.range((kt < k_tiles).cast(dtypes.weakint), 51)
    wl_done = load_b_reg(kt.minimum(k_tiles - 1), (init_done, kt, st_af_done, r_wl)).end(r_wl)
    r_wm = UOp.range((kt > 0).cast(dtypes.weakint), 52)
    b_mid = _lds_barrier(sink_r, st_af_done, wl_done, r_wm, bid=3)
    res_done = _rescale_block(kt_c, cur_b, (init_done, kt, b_mid))
    W_r = W_lds_u32.after(b_mid)[cur_b]
    af_r = af_all.after(b_mid, res_done)
    step_r = UOp.range(ns, 100)
    def _wmma_128(tn_c:int) -> UOp:
      wf_lo = [W_r[(wave_n * tn_c + j) * 16 + col, step_r * 4 + lg * 2].load() for j in range(tn_c)]
      wf_hi = [W_r[(wave_n * tn_c + j) * 16 + col, step_r * 4 + lg * 2 + 1].load() for j in range(tn_c)]
      wf_hi[-1] = _nat("({{ unsigned int _v = ({0}); __builtin_amdgcn_sched_barrier(0); if ({1}) __builtin_amdgcn_s_setprio(0); _v; }})",
                       dtypes.uint32, wf_hi[-1], step_r.eq(0), inline=False)
      wm_i = UOp.range(tm, 101)
      a_f8 = UOp.stack(af_r[wm_i, step_r, 0].load(), af_r[wm_i, step_r, 1].load()).bitcast(dtypes.fp8e4m3)
      return UOp.group(*[acc[wm_i, j].store(UOp.wmma(a_f8, UOp.stack(wf_lo[j], wf_hi[j]).bitcast(dtypes.fp8e4m3),
                         acc.after(init_done, kt, res_done, step_r)[wm_i, j], *WMMA_ARG))
                         for j in range(tn_c)]).end(wm_i)
    b_end = _by_tn(_wmma_128).end(step_r, r_wm, kt)

  acc_f = acc.after(b_end)
  # Publish final register ref_r -> sRef ONCE for epilogue (via _nat store + _lds_barrier)!
  r_sf = UOp.range(bg.eq(0).cast(dtypes.weakint), 185)
  sRef_done = _lds_barrier(sink_r, _lds_st1(sRef.after(b_end, r_sf)[bn_i], ref_r.after(b_end, r_sf)[0].load(), sink_r, b_end, r_sf).end(r_sf), bid=4)
  is_mx = UOp.const(ggml_type == MXFP4, dtypes.bool) if isinstance(ggml_type, int) else ggml_type.eq(MXFP4)
  sRef_u_f = sRef.after(sRef_done)
  ti_o, e_o = UOp.range(tm, 190), UOp.range(8, 192, AxisType.UPCAST)
  mt_o = (mt_base + ti_o).minimum(m_tiles_tot - 1)
  om = block_m * bm + (wave_m * tm + ti_o) * 16 + lg * 8 + e_o
  xs_o = xsc[mt_o, lg * 8 + e_o].load()
  def _ep_tn(tn_c:int) -> UOp:
    w_sc = [is_mx.where(_e8m0_halved(sRef_u_f[(wave_n * tn_c + j) * 16 + col].load()) * UOp.const(1.0 / 16.0, dtypes.float32),
                        sRef_u_f[(wave_n * tn_c + j) * 16 + col].load().bitcast(dtypes.float32)) for j in range(tn_c)]
    return UOp.group(*[out[om.valid(om < tok_out), (block_n * bn + (wave_n * tn_c + j) * 16 + col).valid(
      (block_n * bn + (wave_n * tn_c + j) * 16 + col) < out_features)].store(
      (acc_f[ti_o, j, e_o].load() * w_sc[j] * xs_o).cast(out.dtype)) for j in range(tn_c)])
  ep_group = _by_tn(_ep_tn).end(e_o)
  if bk == 64: ep_group = _sched_barrier(sink_r, ep_group, bid=12)
  kname = f"linear_{QUANT_NAMES[ggml_type]}_fp8_wmma" if isinstance(ggml_type, int) else "linear_quant_fp8_wmma"
  return ep_group.end(ti_o, wave, lane, gid).sink(arg=KernelInfo(name=kname, opts_to_apply=()))
