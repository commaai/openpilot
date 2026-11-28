"""slopkernels: optional AMD RDNA3/RDNA4/CDNA templated UIR kernels (always round-tripped through tinygrad/llm/kernels/)."""
from __future__ import annotations

import functools, os, pathlib, threading
from collections.abc import Callable
from dataclasses import replace

from tinygrad import Context, UOp, dtypes
from tinygrad.dtype import AddrSpace, DType
from tinygrad.uop.ops import AxisType, Ops, ParamArg, PatternMatcher, UPat, graph_rewrite
from tinygrad.uop.symbolic import symbolic
from tinygrad.uop.uir import parse_uir, render_uir

from slopkernels.src import amd as _amd, amd_gguf as _gguf

_UIR_PM = PatternMatcher([
  (UPat(Ops.WHERE, dtype=dtypes.void, src=(UPat.var("c", dtype=dtypes.bool).logical_not(), UPat.var("t"), UPat.var("f"))),
   lambda c, t, f: UOp(Ops.WHERE, src=(c, f, t))),
  (UPat(Ops.WHERE, dtype=dtypes.void, src=(UPat.var("a"), UPat(Ops.WHERE, dtype=dtypes.void, src=(UPat.var("b"), UPat.var("c"), UPat.var("d"))),
                                           UPat.var("d"))), lambda a, b, c, d: UOp(Ops.WHERE, src=(a & b, c, d))),
  (UPat(Ops.WHERE, dtype=dtypes.void, src=(UPat.var("a"), UPat(Ops.WHERE, dtype=dtypes.void, src=(UPat.var("b"), UPat.var("c"), UPat.var("d"))),
                                           UPat.var("c"))), lambda a, b, c, d: UOp(Ops.WHERE, src=(a & b.logical_not(), d, c))),
  (UPat(Ops.WHERE, dtype=dtypes.void, src=(UPat.var("a"), UPat.var("c"),
                                           UPat(Ops.WHERE, dtype=dtypes.void, src=(UPat.var("b"), UPat.var("c"), UPat.var("d"))))),
   lambda a, b, c, d: UOp(Ops.WHERE, src=(a | b, c, d))),
  (UPat(Ops.END, name="e"), lambda e: (
    e.src[0] if not (r := tuple(x for x in e.src[1:] if x.op is Ops.RANGE)) else e.replace(src=(e.src[0], *r)))
   if any(x.op is not Ops.RANGE for x in e.src[1:]) else None),
]) + PatternMatcher([(p, f) for p, f in symbolic.patterns if not (p.op is not None and Ops.AFTER in p.op)])

UIR_DIR = pathlib.Path(__file__).resolve().parent.parent / "tinygrad" / "llm" / "kernels"
_DTYPES: dict[str, DType] = {v.name: v for v in vars(type(dtypes)).values() if isinstance(v, DType)}

def _p(slot:int, shape:tuple[int|UOp, ...], dt:str|DType) -> UOp: return UOp.param(slot, dt if isinstance(dt, DType) else _DTYPES[dt], shape)
def _v(slot:int, name:str) -> UOp:
  return UOp.variable(name, 0, 1 << 20).replace(arg=ParamArg(slot, dtypes.weakint, (0, 1 << 20), name, AddrSpace.ALU, None))
def _t(name:str, lo:int|float=1, hi:int|float=1 << 20, dt:DType=dtypes.weakint) -> UOp: return UOp.template(name, (lo, hi), dtype=dt)

def _b_gemv(wdt:str, xdt:str) -> UOp:
  tok, ib, outf, vc = _t("tokens"), _t("in_blks"), _t("out_features"), _t("val_chunk", 1, 4)
  inf = ib * (32 * vc)
  args = (_p(0, (tok, outf), "f32"), _p(1, (outf, inf), wdt), _p(2, (tok, inf), xdt))
  k_base = _amd._amd_f16_gemv_kernel(*args, in_blks=ib, out_features=outf, tokens=tok, val_chunk=vc)
  if wdt != "f16": return k_base
  k_1t = _amd._amd_f16_gemv_kernel(*args, in_blks=ib, out_features=outf, tokens=tok, val_chunk=vc, fdot2_1t=True)
  k_lds = _amd._amd_f16_gemv_kernel(*args, in_blks=ib, out_features=outf, tokens=tok, val_chunk=vc, lds_mt=True)
  return UOp(Ops.WHERE, src=(tok.eq(1) & (inf >= 512) & (inf % 256).eq(0), k_1t,
             UOp(Ops.WHERE, src=((tok > 1) & (tok <= 8) & (inf % 1024).eq(0) & (outf >= 1024) & (outf % 8).eq(0), k_lds, k_base))))

def _grid_len(qt:UOp) -> UOp:
  tbl = {16: 512, 17: 1024, 18: 256, 19: 4096, 21: 512, 22: 2048}
  return _gguf._qswitch(qt, [(q, UOp.const(tbl.get(q, 1), dtypes.weakint)) for q in _gguf.QUANT_NAMES])

def _b_qgemv() -> UOp:
  qt, tok, inf, outf = _t("qtype", 8, 40), _t("tokens"), _t("in_features", 32), _t("out_features")
  rb, gr = _t("raw_bytes", 1, 1 << 40), _t("group_rows", 0)
  use_sel, xrows, xs, nd = _t("use_sel", 0, 1), _t("x_rows"), _t("x_share", 1, 1024), _t("ndev", 1, 64)
  arch, ne, gl = _t("arch", 0, 12), _t("num_experts", 0, 512), _grid_len(qt)
  p_out, p_raw, p_x = _p(0, (tok, outf), "f32"), _p(1, (rb,), "u8"), _p(2, (xrows, inf), "f32")
  is_moe = use_sel.eq(1)
  p_sel, p_grid = _p(3, (is_moe.where(tok, gl),), "i32"), _p(4, (gl,), "u32")
  expert_bytes = outf * inf * _gguf._qsize(qt) // 256
  local_experts = (rb // expert_bytes).maximum(1)
  def expert(row:UOp) -> UOp: return p_sel[row].cast(dtypes.weakint) - UOp.range(nd, 10, AxisType.DEVICE) * local_experts
  w1 = (inf <= 1024) | (is_moe.logical_not() & tok.eq(1) & (outf <= 1024) & (inf <= 5120)) | (is_moe & (tok >= 16) & (inf <= 2048))
  is_rpw2 = _gguf._qswitch(qt, [(q, UOp.const(q in _gguf._RPW2_QUANTS, dtypes.bool)) for q in _gguf.QUANT_NAMES])
  rpw_dec = is_rpw2.where((inf <= 1024).where(1, (outf % 2).eq(0).where(2, 1)), 1)
  rpw_pre = (outf % 4).eq(0).where(4, (outf % 2).eq(0).where(2, 1))
  rpw = (is_moe & (tok >= 16)).where(rpw_pre, rpw_dec)
  wk_cap = (is_moe & (tok >= 16)).where(4, rpw_dec.eq(2).where(8, 16))
  def _gate(row:UOp, base:UOp, q:int, isz:int) -> UOp:
    return base + is_moe.where(expert(row).maximum(0).minimum(local_experts - 1), 0) * (outf * inf * (_gguf.QUANT_SIZES[q] // isz) // 256)
  k_direct = _gguf._direct_gemv_body(
    p_out, p_raw, p_x, (p_grid,), outf, inf, qt, rpw=rpw, wk_cap=wk_cap, w1_cond=w1,
    lds_grid=True, group_rows=gr, gate=_gate,
    xtoken=lambda row: row // xs, row_mask=lambda row: is_moe.logical_not() | ((expert(row) >= 0) & (expert(row) < local_experts)))
  # 2-kernel SDOT4 pipeline for dense small-batch (4..15 tokens) on RDNA3/4
  xq, xds = UOp.alloc((tok, inf // 32, 8), dtypes.uint32, slot=0), UOp.alloc((tok, inf // 32, 3), dtypes.float32, slot=1)
  k_q8 = _gguf._quant_q8_1_kernel(_p(0, (tok, inf // 32, 8), "u32"), _p(1, (tok, inf // 32, 3), "f32"), _p(2, (tok, inf), "f32"), tok, inf)
  is_iq3x = qt.eq(_gguf.IQ3_XXS)
  wk_s = is_iq3x.where((inf >= 2048).where(2, 1), 1)
  wr_s = is_iq3x.where((outf % 4).eq(0).where(4, 1), (outf % 8).eq(0).where(8, (outf % 4).eq(0).where(4, 1)))
  k_sd = _gguf._sdot4_gemv_kernel(_p(0, (tok, outf), "f32"), _p(1, (rb,), "u8"), _p(2, (tok, inf // 32, 8), "u32"),
                                  _p(3, (tok, inf // 32, 3), "f32"), (_p(4, (gl,), "u32"),), outf, inf, qt,
                                  w_rows=wr_s, waves_k=wk_s, group_rows=gr)
  cq8 = k_q8.call(xq, xds, p_x)
  k_sdot4_pipe = UOp.sink(k_sd.call(p_out, p_raw, xq.after(cq8), xds.after(cq8), p_grid))
  eff_tok = (gr > 0).where(gr, tok)
  cond_sdot4 = (arch >= 11) & is_moe.logical_not() & (inf >= 1024) & (outf % 4).eq(0) & (eff_tok >= 4) & (eff_tok <= 15) & tok.eq(xrows)
  # 3-kernel RDNA4 MoE FP8 WMMA pipeline (route -> quant_x_frag -> gemm_fp8)
  max_tiles, ne_safe = (tok // 16 + ne).maximum(1), ne.maximum(1)
  kt_m = (inf // 128).maximum(1)
  t_idx, t_exp, t_off, t_cnt = (UOp.alloc((n,), dtypes.int32, slot=i + 2) for i, n in enumerate((tok, max_tiles, max_tiles, max_tiles)))
  t_tot = UOp.alloc((1,), dtypes.int32, slot=6)
  xt_m, xsc_m = UOp.alloc((max_tiles, kt_m * 8, 32, 2), dtypes.uint32, slot=7), UOp.alloc((max_tiles, kt_m, 16), dtypes.float32, slot=8)
  k_rt = _gguf._amd_moe_route_kernel(_p(0, (tok,), "i32"), _p(1, (max_tiles,), "i32"), _p(2, (max_tiles,), "i32"),
                                     _p(3, (max_tiles,), "i32"), _p(4, (1,), "i32"), _p(5, (tok,), "i32"), num_experts=ne_safe, ndev=nd)
  k_qxm = _gguf._rdna4_quant_x_frag_kernel(_p(0, (max_tiles, kt_m * 8, 32, 2), "u32"), _p(1, (max_tiles, kt_m, 16), "f32"),
                                           _p(2, (xrows, kt_m * 128), "f32"), _p(3, (tok,), "i32"), _p(4, (max_tiles,), "i32"),
                                           _p(5, (max_tiles,), "i32"), _p(6, (1,), "i32"), x_share=xs)
  k_fp8m = _gguf._rdna4_gemm_fp8_kernel(_p(0, (tok, outf), "f32"), _p(1, (rb,), "u8"), _p(2, (max_tiles, kt_m * 8, 32, 2), "u32"),
                                        _p(3, (max_tiles, kt_m, 16), "f32"), _p(4, (gl,), "u32"), _p(5, (tok,), "i32"),
                                        _p(6, (max_tiles,), "i32"), _p(7, (max_tiles,), "i32"), _p(8, (max_tiles,), "i32"),
                                        _p(9, (1,), "i32"), out_features=outf, in_features=kt_m * 128, ggml_type=qt,
                                        qtypes=tuple(sorted(_gguf.OTF_FP8_QUANTS)))
  cr = k_rt.call(t_idx, t_exp, t_off, t_cnt, t_tot, p_sel)
  t_idx, t_exp, t_off, t_cnt, t_tot = (a.after(cr) for a in (t_idx, t_exp, t_off, t_cnt, t_tot))
  cqm = k_qxm.call(xt_m, xsc_m, p_x, t_idx, t_off, t_cnt, t_tot)
  k_moe_fp8_pipe = UOp.sink(k_fp8m.call(p_out, p_raw, xt_m.after(cqm), xsc_m.after(cqm), p_grid, t_idx, t_exp, t_off, t_cnt, t_tot))
  is_otf = _gguf._qswitch(qt, [(q, UOp.const(q in _gguf.OTF_FP8_QUANTS, dtypes.bool)) for q in _gguf.QUANT_NAMES])
  k_align = (qt.eq(_gguf.Q8_0) | qt.eq(_gguf.MXFP4)).where(128, 256)
  cond_moe_fp8 = (arch.eq(12) & is_moe & nd.eq(1) & ((tok // xs) >= 16) & is_otf & (ne >= 1) & (ne <= 512) &
                  (outf >= 64) & (outf % 64).eq(0) & (inf >= 256) & (inf % k_align).eq(0))
  k_iq4_lds = _gguf._iq4_lds_gemv_kernel(p_out, p_raw, p_x, outf, inf, arch)
  cond_iq4_lds = (arch >= 11) & is_moe.logical_not() & gr.eq(0) & tok.eq(1) & tok.eq(xrows) & qt.eq(_gguf.IQ4_XS) & (inf >= 2048)
  k_sel = UOp(Ops.WHERE, src=(cond_sdot4, k_sdot4_pipe, UOp(Ops.WHERE, src=(cond_iq4_lds, k_iq4_lds, k_direct))))
  return UOp(Ops.WHERE, src=(cond_moe_fp8, k_moe_fp8_pipe, k_sel))

def _b_gemm(kg:int, ln:int) -> UOp:
  qt, mb, mt, inf, outf = _t("qtype", 8, 40), _t("m_blks"), _t("m_tiles", 1, 8), _t("in_features", 32), _t("out_features", 16)
  rb, gl = _t("raw_bytes", 1, 1 << 40), _grid_len(qt)
  tok = mb * (mt * 16)
  p_out, p_raw, p_x, p_grid = _p(0, (tok, outf), "f32"), _p(1, (rb,), "u8"), _p(2, (tok, inf), "f16"), _p(3, (gl,), "u32")
  if kg == 1:  # RDNA3: Direct-A + Double-Buffered-B 2D FP16 WMMA + in-WG K-split (wk=2,4)
    c_lg = (tok % 128).eq(0) & (outf % 64).eq(0) & (inf % 128).eq(0) & (((tok // 128) * (outf // 64)) >= 192)
    c_md = (tok >= 64) & (tok % 64).eq(0) & (outf % 32).eq(0) & (inf % 256).eq(0) & (((tok // 64) * (outf // 32)) >= 96)
    use_w4 = (tok % 32).eq(0) & (inf >= 256) & (inf % 256).eq(0) & ((tok.eq(32)) | ((tok * outf) <= 128 * 1024))
    use_w2 = (tok >= 128) & (tok % 64).eq(0) & (outf % 64).eq(0) & (inf >= 8192) & (inf % 128).eq(0) & (outf > 4096)
    use_wk = use_w4 | use_w2
    wm = use_wk.where(1, c_lg.where(4, c_md.where(2, 1)))
    wn = use_wk.where(use_w4.where(1, 2), 1)
    tm = use_wk.where(use_w4.where(2, 4), c_lg.where(2, c_md.where(2, (tok % 32).eq(0).where(2, 1))))
    tn = use_wk.where(use_w4.where(1, 2), c_lg.where(4, c_md.where(2, 1)))
    bk = use_wk.where(use_w4.where(256, 128), c_lg.where(64, c_md.where(128, (inf % 64).eq(0).where(64, 32))))
    wk = use_wk.where(use_w4.where(4, 2), 1)
    return _gguf._rdna3_gemm_2d_kernel(p_out, p_raw, p_x, p_grid, m_blks=tok // (wm * tm * 16), n_blks=outf // (wn * tn * 16),
                                       wm=wm, wn=wn, tm=tm, tn=tn, bk=bk, out_features=outf, in_features=inf, ggml_type=qt, wk=wk)
  if kg != 2:
    return _gguf._quant_linear_f16_wmma_kernel(p_out, p_raw, p_x, p_grid, m_blks=mb, m_tiles=mt, out_features=outf, in_features=inf,
                                               ggml_type=qt, kgroups=kg, lanes=ln)
  # RDNA4: 3-stage OTF FP8 2D WMMA (M>=64, N>=2048): row_amax -> quant_apply -> dense FP8 WMMA -> ZLDS F16 -> F16 WMMA
  mt_tot, kc_d = mb * mt, (inf // 256).maximum(1)
  xt_d, xsc_d = UOp.alloc((mt_tot, kc_d * 16, 32, 2), dtypes.uint32, slot=0), UOp.alloc((mt_tot, 16), dtypes.float32, slot=1)
  k_ramax = _gguf._rdna4_row_amax_kernel(_p(0, (mt_tot, 16), "f32"), _p(1, (tok, kc_d * 256), "f16"))
  k_qapp = _gguf._rdna4_quant_x_apply_kernel(_p(0, (mt_tot, kc_d * 16, 32, 2), "u32"), _p(1, (mt_tot, 16), "f32"),
                                             _p(2, (tok, kc_d * 256), "f16"))
  fp8_qts = tuple(sorted(_gguf.OTF_FP8_QUANTS))
  p_fp8 = (_p(0, (tok, outf), "f32"), _p(1, (rb,), "u8"), _p(2, (mt_tot, kc_d * 16, 32, 2), "u32"),
           _p(3, (mt_tot, 16), "f32"), _p(4, (gl,), "u32"))
  use_m = tok >= 256
  k_fp8_sm = _gguf._rdna4_gemm_fp8_dense_kernel(*p_fp8, out_features=outf, in_features=kc_d * 256,
                                                ggml_type=qt, wm=use_m.where(4, 2), tm=4, tn=use_m.where(2, 1), bk=128, qtypes=fp8_qts)
  k_fp8_w = _gguf._rdna4_gemm_fp8_dense_kernel(*p_fp8, out_features=outf, in_features=kc_d * 256,
                                               ggml_type=qt, wm=8, tm=4, tn=4, bk=64, qtypes=fp8_qts)
  cra = k_ramax.call(xsc_d, p_x)
  xsc_r = xsc_d.after(cra)
  cqa = k_qapp.call(xt_d, xsc_r, p_x)
  xt_r = xt_d.after(cqa)
  k_fp8_pipe = UOp(Ops.WHERE, src=((tok >= 1024),
                                   UOp.sink(k_fp8_w.call(p_out, p_raw, xt_r, xsc_r, p_grid)),
                                   UOp.sink(k_fp8_sm.call(p_out, p_raw, xt_r, xsc_r, p_grid))))
  is_otf = _gguf._qswitch(qt, [(q, UOp.const(q in _gguf.OTF_FP8_QUANTS, dtypes.bool)) for q in _gguf.QUANT_NAMES])
  cond_fp8 = is_otf & (tok >= 64) & (outf >= 2048) & (outf % 128).eq(0) & (inf >= 512) & (inf % 256).eq(0)
  k_sb = inf // 256
  wk_z, tn_z = ((k_sb >= 4) & (k_sb % 4).eq(0)).where(4, ((k_sb >= 2) & (k_sb % 2).eq(0)).where(2, 1)), (outf >= 32).where(2, 1)
  k_zlds = _gguf._rdna4_zlds_f16_kernel(p_out, p_raw, p_x, p_grid, tok=tok, out_features=outf, in_features=inf,
                                        ggml_type=qt, waves_k=wk_z, tn=tn_z)
  f16_qts = tuple(sorted(_gguf.SMALL_BLOCK_QUANTS))
  k_f16 = _gguf._quant_linear_f16_wmma_kernel(p_out, p_raw, p_x, p_grid, m_blks=mb, m_tiles=mt, out_features=outf, in_features=inf,
                                              ggml_type=qt, kgroups=kg, lanes=ln, qtypes=f16_qts)
  cond_zlds = (inf >= 256) & (inf % 256).eq(0) & (outf >= 16) & (tok >= 16)
  return UOp(Ops.WHERE, src=(cond_fp8, k_fp8_pipe, UOp(Ops.WHERE, src=(cond_zlds, k_zlds, k_f16))))

def _b_flash_decode() -> UOp:
  B, H_KV, G, db, C, nb = _t("B", 1, 64), _t("H_KV", 1, 256), _t("G", 1, 8), _t("d_blks", 1, 8), _t("C", 1, 64), _t("n_blks", 1, 65536)
  voff, H, D, vkl = _t("valid_kv_offset", 0), H_KV * G, db * 64, _v(3, "valid_kv_len")
  b1 = G * ((D + 4) * 2 + 8)
  waves = (16 * b1 <= 65536).where(16, (8 * b1 <= 65536).where(8, (4 * b1 <= 65536).where(4, (2 * b1 <= 65536).where(2, 1))))
  can_gs2 = (G % 2).eq(0) & (waves >= 2)
  gs = (can_gs2 & (db.eq(2) | (db >= 4))).where(2, 1)
  lpt = (db.eq(1) & (waves * G <= 112)).where(8, (db.eq(2) & ((waves // gs) * G <= 112)).where(16, 32))
  k_part = _amd._amd_flash_attention_decode_partial(
    _p(0, (B, H, C, D), "f32"), _p(1, (B, H, C, 2), "f32"), _p(2, (B, H, 1, D), "f16"), _p(3, (2, B, H_KV, nb * 64, D), "f16"),
    valid_kv_len=_v(4, "valid_kv_len") + voff, d_blks=db, group=G, waves=waves, g_split=gs, lanes_per_tok=lpt)
  k_comb = _amd._amd_flash_decode_combine(_p(0, (B, H, 1, D), "f32"), _p(1, (B, H, C, D), "f32"), _p(2, (B, H, C, 2), "f32"),
                                          valid_kv_len=_v(3, "valid_kv_len") + voff)
  partial, stats = UOp.alloc((B, H, C, D), dtypes.float32, slot=0), UOp.alloc((B, H, C, 2), dtypes.float32, slot=1)
  cp = k_part.call(partial, stats, _p(1, (B, H, 1, D), "f16"), _p(2, (2, B, H_KV, nb * 64, D), "f16"), vkl)
  return UOp.sink(k_comb.call(_p(0, (B, H, 1, D), "f32"), partial.after(cp), stats.after(cp), vkl))

def _b_fa(rdna4:bool) -> UOp:
  B, H, H_KV, db, nb, voff = _t("B", 1, 64), _t("H", 1, 256), _t("H_KV", 1, 256), _t("d_blks", 1, 8), _t("n_blks", 1, 65536), _t("valid_kv_offset", 0)
  mb, qfe, vkl, ql, D = _t("m_blks", 1, 4096), _t("q_start_from_end", 0, 1), _v(3, "valid_kv_len"), _v(4, "q_len"), db * 64
  args = (_p(0, (B, H, mb * 32, D), "f32").reshape((B * H, mb * 32, D)),
          _p(1, (B, H, mb * 32, D), "f16").reshape((B * H, mb * 32, D)), _p(2, (2, B, H_KV, nb * 32, D), "f16"))
  vkl_e, qs_e = qfe.eq(1).where(vkl + voff, vkl + voff + ql), qfe.eq(1).where(vkl + voff - ql, vkl + voff)
  k_sm = _amd._amd_flash_attention_smallm(*args, valid_kv_len=vkl_e, q_start=qs_e, d_blks=db, rdna4=rdna4)
  k_gen = _amd._amd_flash_attention(*args, valid_kv_len=vkl_e, q_start=qs_e, d_blks=db, rdna4=rdna4)
  return UOp(Ops.WHERE, src=(mb.eq(1) & (db <= 2) & (nb >= 4), k_sm, k_gen))

def _b_hc() -> UOp:
  tok, feats, iters = _t("tokens"), _t("features", 32), _t("iters", 1, 64)
  eps, neps = _t("eps", 0.0, 1.0, dtypes.float32), _t("norm_eps", 0.0, 1.0, dtypes.float32)
  return _amd._hc_coeffs_kernel(_p(0, (tok, 24), "f32"), _p(1, (tok, 24), "f32"), _p(2, (tok, feats), "f32"),
                                _p(3, (24,), "f32"), _p(4, (3,), "f32"), 4, iters, eps, neps)

def _b_gd() -> UOp:
  tok, ba, hd, vt, kb, ak, arch = (_t("tokens"), _t("batch", 1, 64), _t("heads", 1, 256), _t("v_tiles", 1, 1024),
                                   _t("k_blks", 1, 8), _t("alpha_k", 1, 256), _t("arch", 9, 12))
  K, sp = kb * 32, _v(7, "start_pos")
  args = (_p(0, (ba, hd, tok, vt * 4), "f32"), _p(1, (ba, hd, tok, K), "f32"), _p(2, (ba, hd, tok, K), "f32"),
          _p(3, (ba, hd, tok, vt * 4), "f32"), _p(4, (ba, hd, tok), "f32"), _p(5, (ba, hd, tok, ak), "f32"),
          _p(6, (ba, hd, vt * 4, K), "f32"))
  return _amd._gated_delta_prefill_kernel(*args, k_blks=kb, alpha_k=ak, start_pos=sp, arch=arch)

_BUILDERS: dict[str, Callable[[], UOp]] = {
  **{f"amd_gemv_{w}_f32": functools.partial(_b_gemv, w, "f32") for w in ("f16", "bf16", "f32")},
  "amd_gemv_quant_f32": _b_qgemv,
  **{f"{a}_gemm_f16": functools.partial(_b_gemm, kg, ln) for a, kg, ln in (("rdna4", 2, 32), ("rdna3", 1, 32), ("cdna", 4, 64))},
  "amd_flash_decode": _b_flash_decode,
  "rdna4_flash_attn": functools.partial(_b_fa, True), "rdna3_flash_attn": functools.partial(_b_fa, False),
  "amd_hc_coeffs": _b_hc,
  "amd_gated_delta": _b_gd,
}

def _emit(name:str, fn:Callable[[], UOp]) -> tuple[pathlib.Path, str]:
  UIR_DIR.mkdir(parents=True, exist_ok=True)
  with Context(TRACK_MATCH_STATS=0, SPEC=0):
    u = graph_rewrite(fn(), _UIR_PM, enter_calls=True)
    if allocs := [x for x in u.toposort(enter_calls=True) if x.op is Ops.ALLOC]:
      u = u.substitute({b: b.replace(arg=replace(b.arg, slot=-1 - i)) for i, b in enumerate(allocs)}, enter_calls=True)
      u = u.substitute({b.replace(arg=replace(b.arg, slot=-1 - i)): b.replace(arg=replace(b.arg, slot=i))
                        for i, b in enumerate(allocs)}, enter_calls=True)
    uir = render_uir(u, color=False)
  dst = UIR_DIR / f"{name}.uir"
  tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
  tmp.write_bytes((uir + "\n").encode("utf-8"))
  os.replace(tmp, dst)
  return dst, uir

def generate_all() -> dict[str, str]:
  """Generate all static templated `.uir` files fresh on disk in `tinygrad/llm/kernels/`."""
  return {name: _emit(name, fn)[1] for name, fn in sorted(_BUILDERS.items())}

@functools.cache
def get(name:str) -> UOp|None:
  """Generate `tinygrad/llm/kernels/<name>.uir` fresh on disk and return the parsed templated UOp."""
  if (fn := _BUILDERS.get(name)) is None: return None
  with Context(SPEC=0): return parse_uir(_emit(name, fn)[0].read_text(encoding="utf-8"))

__all__ = ["generate_all", "get"]
