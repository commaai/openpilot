"""End-to-end LLM kernel benchmark & correctness suite for slopkernels on AMD GPUs (e.g. 7900XTX, 9060XT, 9070XT)."""
from __future__ import annotations

import argparse, functools, json, math, pathlib, re, statistics, sys, time
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

if str(root := pathlib.Path(__file__).resolve().parent.parent) not in sys.path: sys.path.insert(0, str(root))

from tinygrad import Device, Tensor, UOp, dtypes
from tinygrad.dtype import DType
from tinygrad.engine.realize import ExecContext, link_linear, pm_exec
from tinygrad.codegen.compile import compile_linear
from tinygrad.helpers import ceildiv, colored, prod, unwrap
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.uop.ops import Ops

import slopkernels

_QUANT_META: dict[str, tuple[int, int, DType, str]] = {
  "Q8_0": (8, 272, dtypes.uint16, "q8_0"), "Q6_K": (14, 210, dtypes.uint16, "q6"), "Q5_K": (13, 176, dtypes.uint32, "q5_k"),
  "Q4_K": (12, 144, dtypes.uint32, "q4_k"), "Q3_K": (11, 110, dtypes.uint16, "q3_k"), "Q2_K": (10, 84, dtypes.uint16, "q2_k"),
  "IQ4_XS": (23, 136, dtypes.uint32, "iq4_xs"), "IQ4_NL": (20, 144, dtypes.uint16, "iq4_nl"), "IQ3_S": (21, 110, dtypes.uint16, "iq3_s"),
  "IQ3_XXS": (18, 98, dtypes.uint16, "iq3_xxs"), "IQ2_S": (22, 82, dtypes.uint16, "iq2_s"), "IQ2_XS": (17, 74, dtypes.uint16, "iq2_xs"),
  "IQ2_XXS": (16, 66, dtypes.uint16, "iq2_xxs"), "IQ1_S": (19, 50, dtypes.uint16, "iq1_s"), "MXFP4": (39, 136, dtypes.uint8, "mxfp4"),
}
_AMD_PEAKS: dict[tuple[tuple[int, ...], int], tuple[str, float, float]] = {
  ((11, 0, 0), 96): ("RX 7900 XTX", 960.0, 122.9), ((11, 0, 0), 84): ("RX 7900 XT", 800.0, 103.0),
  ((11, 0, 0), 80): ("RX 7900 GRE", 576.0, 92.0), ((11, 0, 1), 60): ("RX 7800 XT", 624.0, 74.6),
  ((11, 0, 1), 54): ("RX 7700 XT", 432.0, 70.3), ((11, 0, 2), 32): ("RX 7600", 288.0, 43.5),
  ((12, 0, 1), 64): ("RX 9070 XT", 640.0, 195.0), ((12, 0, 1), 56): ("RX 9070", 640.0, 144.4),
  ((12, 0, 0), 32): ("RX 9060 XT", 320.0, 103.0),
  ((9, 4, 2), 304): ("MI300X", 5300.0, 1307.4), ((9, 5, 0), 256): ("MI350X", 8000.0, 2309.6),
}

@dataclass(frozen=True)
class BenchResult:
  name: str
  median_us: float
  min_us: float
  gb_s: float
  bw_pct: float
  tflops: float
  tflops_pct: float

VerifyFn = Callable[[Tensor], None]
SetupFn = Callable[[], tuple[Callable[[], Tensor], VerifyFn]]

def _arch_info(dev:str) -> tuple[tuple[int, ...], bool]:
  raw = getattr(Device[dev], "target", None)
  target = tuple(int(x) for x in raw) if isinstance(raw, tuple) else (11, 0, 0)
  return target, target[0] == 9

def _peak_specs(dev:str) -> tuple[str, int, float, float]:
  d, target = Device[dev], _arch_info(dev)[0]
  cus = int(getattr(d, "cu_cnt", 0)) * int(getattr(d, "xccs", 1))
  spec = _AMD_PEAKS.get((target, cus)) or next((v for (t, _), v in _AMD_PEAKS.items() if t == target),
                                               (f"gfx{target[0]}{target[1]:x}{target[2]:x}", 0.0, 0.0))
  return spec[0], cus, spec[1], spec[2]

def _packed_words(rows:int, cols:int, qname:str) -> tuple[int, int, DType, str]:
  qtype, blk_bytes, w_dt, kname = _QUANT_META[qname]
  return qtype, (rows * cols * blk_bytes // 256) // w_dt.itemsize, w_dt, kname

def _nonuniform(shape:tuple[int, ...], dtype:DType, dev:str, scale:float=0.5, freq:float=0.017, phase:float=0.3) -> Tensor:
  t = ((Tensor.arange(prod(shape), dtype=dtypes.float32) + Tensor.zeros(1, dtype=dtypes.float32, device=dev)) * freq + phase).sin() * scale
  return t.cast(dtype).reshape(shape).contiguous().realize()

def _make_packed(rows:int, cols:int, qname:str, dev:str, seed:int=7) -> tuple[Tensor, Tensor]:
  qtype, blk_bytes, _, _ = _QUANT_META[qname]
  row_bytes, tile_rows, rng = cols * blk_bytes // 256, min(16, rows), np.random.default_rng(seed + qtype)
  if qtype == 39:
    blocks = rng.integers(0, 256, size=(tile_rows, cols // 32, 17), dtype=np.uint8)
    blocks[:, :, 0] = rng.integers(0x7b, 0x80, size=(tile_rows, cols // 32), dtype=np.uint8)
    tile_np = blocks.reshape(tile_rows, row_bytes)
  else: tile_np = rng.integers(0x20, 0x38, size=(tile_rows, row_bytes), dtype=np.uint8)
  tile_u8 = Tensor(tile_np, device=dev).realize()
  reps = rows // tile_rows
  packed = tile_u8.flatten() if reps == 1 else tile_u8.reshape(1, tile_rows, row_bytes).expand(reps, tile_rows, row_bytes).reshape(-1)
  return packed.contiguous().realize(), tile_u8

def _assert_close(got:np.ndarray, ref:np.ndarray, rtol:float, atol:float) -> None:
  np.testing.assert_allclose(got, ref, rtol=rtol, atol=max(atol, rtol * float(np.max(np.abs(ref))) if ref.size else atol))

def _make_cache_flusher(flush_mb:int) -> Callable[[], None]:
  if flush_mb <= 0: return lambda: None
  src = Tensor.ones(flush_mb * 1024 * 1024 // 4, dtype=dtypes.float32).contiguous().realize()
  sink = src.sum()
  Device[Device.DEFAULT].synchronize()
  linear, externs = sink.linear_with_externs()
  linked = link_linear(compile_linear(linear, beam=0, profile=False, cache=True), list(externs))
  ctx, keep = ExecContext({}, update_stats=False, wait=True, cache=True), (src, sink, linked)
  def _flush() -> None:
    _ = keep
    for raw in linked.src: pm_exec.rewrite(raw.without_after, ctx)
  _flush()
  return _flush

def _time_fn(fn:Callable[[], Tensor], verify:VerifyFn|None, warmup:int, iters:int, flush_fn:Callable[[], None]) -> tuple[float, float]:
  Device[Device.DEFAULT].synchronize()
  out = fn()
  linear, externs = out.linear_with_externs()
  linked = link_linear(compile_linear(linear, beam=0, profile=True, cache=True), list(externs))
  ctx = ExecContext({}, update_stats=False, wait=True, cache=True)
  def _run_once() -> float:
    return sum(sum(map(unwrap, pm_exec.rewrite(c, ctx)[c.arg.hcq is not None:])) for raw in linked.src for c in (raw.without_after,))
  _run_once()
  if verify is not None: verify(out)
  elif not math.isfinite(first := float(out.flatten()[:1].float().item())): raise RuntimeError(f"non-finite benchmark output: {first}")
  for _ in range(max(0, warmup - 1)): _run_once()
  times_us = []
  for _ in range(iters):
    flush_fn()
    times_us.append(_run_once() * 1e6)
  return statistics.median(times_us), min(times_us)

def _krun(name:str, shp:tuple[int, ...], dev:str|tuple[str, ...], *args:Tensor|UOp, axis:int|None=None, **kw:int|float) -> Tensor:
  k, o = unwrap(slopkernels.get(name)).specialize(**kw), UOp.alloc(shp, dtypes.float32, device=dev, axis=axis)
  if k.op is Ops.SINK and k.arg is None:
    k = k.substitute({b: b.replace(src=b.src[:1] + UOp.device_range_src(dev), arg=replace(b.arg, device=dev))
                      for b in k.toposort() if b.op is Ops.ALLOC and b.arg.device is None})
  return Tensor(o.after(k.call(o, *(a.uop if isinstance(a, Tensor) else a for a in args))))

@functools.cache
def _iq_lut(t:int, dev:str|tuple[str, ...]) -> tuple[UOp, int]:
  if t not in (16, 17, 18, 19, 21, 22): return Tensor(np.array([0], dtype=np.uint32), device=dev).realize().uop, 1
  from tinygrad.runtime.autogen import ggml_common as g
  grid, nw = {16:(g.iq2xxs_grid, 2), 17:(g.iq2xs_grid, 2), 19:(g.iq1s_grid, 2), 22:(g.iq2s_grid, 2), 18:(g.iq3xxs_grid, 1), 21:(g.iq3s_grid, 1)}[t]
  words = tuple((v >> (32 * i)) & 0xffffffff for v in grid for i in range(nw))
  return Tensor(np.array(words, dtype=np.uint32), device=dev).realize().uop, len(words)

def build_benchmarks(dev:str, smoke:bool=False, fast:bool=False) -> list[tuple[str, int, int, SetupFn]]:
  target, cdna = _arch_info(dev)
  N, K, FFN = (128, 256, 256) if smoke else (4096, 4096, 14336)
  M_PRE, M_PRE_LG = (16, 32) if smoke else (128, 512)
  E, TOPK, MOE_N, MOE_K = (8, 4, 128, 256) if smoke else (64, 8, 2048, 4096)
  AH, AHKV, AD, AN, AM, CR = *((4, 2, 64, 128, 32) if smoke else (32, 8, 128, 2048, 256)), 4
  arch = "cdna" if cdna else "rdna4" if target[0] == 12 else "rdna3"
  cases: list[tuple[str, int, int, SetupFn]] = []

  def _add_qlinear(qname:str, m:int, n:int, k:int, x_dt:DType) -> None:
    qtype, w_len, w_dt, _ = _packed_words(n, k, qname)
    raw_bytes = w_len * w_dt.itemsize
    def _setup() -> tuple[Callable[[], Tensor], VerifyFn]:
      w_packed, tile_u8 = _make_packed(n, k, qname, dev)
      w_u8 = w_packed if w_packed.dtype == dtypes.uint8 else w_packed.bitcast(dtypes.uint8)
      x = _nonuniform((m, k), x_dt, dev, scale=0.5, freq=0.013)
      def _verify(out:Tensor) -> None:
        w4 = ggml_data_to_tensor(tile_u8[:CR].flatten(), CR * k, qtype).reshape(CR, k)
        _assert_close(out[:, :CR].float().numpy(), (x.float() @ w4.T).numpy(), rtol=4e-2, atol=5e-2)
      if m % 16 == 0 and n % 16 == 0:
        mt = 8 if m % 128 == 0 and m >= 256 else 4 if m % 64 == 0 else 2 if m % 32 == 0 else 1
        lut, _ = _iq_lut(qtype, dev)
        return (lambda: _krun(f"{arch}_gemm_f16", (m, n), dev, w_u8, x.half().contiguous(), lut,
                              qtype=qtype, m_blks=m // (mt * 16), m_tiles=mt, out_features=n, in_features=k, raw_bytes=raw_bytes)), _verify
      lut, _ = _iq_lut(qtype, dev)
      return (lambda: _krun("amd_gemv_quant_f32", (m, n), dev, w_u8, x, lut.bitcast(dtypes.int32), lut, qtype=qtype,
                            tokens=m, x_rows=m, out_features=n, in_features=k, raw_bytes=raw_bytes,
                            group_rows=0, use_sel=0, x_share=1, ndev=1, arch=target[0], num_experts=0)), _verify
    kind = "gemm" if m % 16 == 0 and n % 16 == 0 else "gemv"
    cases.append((f"quant_{kind}_{qname}_{m}x{n}x{k}", w_len * w_dt.itemsize + m * k * x_dt.itemsize + m * n * 4, 2 * m * n * k, _setup))

  gemv_q = ("Q8_0", "Q4_K", "IQ4_XS", "MXFP4") if smoke else (
    ("Q8_0", "Q6_K", "Q4_K", "Q2_K", "IQ4_XS", "IQ3_XXS", "IQ3_S", "IQ1_S", "MXFP4") if fast else
    ("Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K", "IQ4_XS", "IQ4_NL", "IQ3_XXS", "IQ3_S", "IQ2_S", "IQ1_S", "MXFP4"))
  for qn in gemv_q: _add_qlinear(qn, 1, N, K, dtypes.float32)
  if not smoke:
    for qn, m, n, k in ((("Q4_K", 1, FFN, K), ("IQ4_XS", 1, 5120, 17408), ("IQ3_XXS", 1, K, FFN),
                         ("IQ4_XS", 1, 2048, 512), ("Q4_K", 4, N, K)) if fast else
                        (("Q4_K", 1, FFN, K), ("Q8_0", 1, FFN, K), ("MXFP4", 1, FFN, K), ("Q4_K", 1, K, FFN), ("IQ4_XS", 1, 5120, 17408),
                         ("IQ3_XXS", 1, K, FFN), ("IQ4_XS", 1, 1024, K), ("IQ4_XS", 1, 2048, 512),
                         ("Q4_K", 4, N, K), ("IQ4_XS", 4, N, K), ("IQ3_XXS", 4, N, K), ("IQ4_XS", 8, N, K))):
      _add_qlinear(qn, m, n, k, dtypes.float32)

  for qn in (("Q8_0", "Q4_K", "IQ4_XS", "MXFP4") if (smoke or fast) else
             ("Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q2_K", "IQ4_XS", "IQ3_XXS", "IQ3_S", "IQ2_S", "MXFP4")):
    _add_qlinear(qn, M_PRE, N, K, dtypes.float16)
  if not smoke:
    for qn in (("Q4_K", "IQ4_XS", "IQ3_XXS") if fast else ("Q8_0", "Q4_K", "IQ4_XS", "IQ3_XXS", "MXFP4")):
      _add_qlinear(qn, M_PRE_LG, N, K, dtypes.float16)
    for qn, m, n, k in ((("Q4_K", 32, N, K), ("IQ4_XS", 32, N, K), ("IQ4_XS", 128, 5120, 17408)) if fast else
                        (("Q4_K", 32, N, K), ("IQ4_XS", 32, N, K), ("IQ3_XXS", 32, N, K), ("IQ4_XS", 128, 1024, K), ("IQ4_XS", 128, 5120, 17408))):
      _add_qlinear(qn, m, n, k, dtypes.float16)

  moe_specs = [("Q4_K", MOE_N, MOE_K)] if smoke else (
    [("Q8_0", MOE_N, MOE_K), ("Q4_K", MOE_N, MOE_K), ("IQ4_XS", MOE_N, MOE_K), ("IQ3_XXS", MOE_N, MOE_K), ("IQ4_XS", 2048, 512)] if fast else
    [("Q8_0", MOE_N, MOE_K), ("Q4_K", MOE_N, MOE_K), ("IQ4_XS", MOE_N, MOE_K), ("IQ3_XXS", MOE_N, MOE_K), ("IQ1_S", MOE_N, MOE_K),
     ("MXFP4", MOE_N, MOE_K), ("Q4_K", 1024, 2048), ("IQ4_XS", 512, 2048), ("IQ4_XS", 2048, 512)])
  for qname, mn, mk in moe_specs:
    qtype, per_e_words, w_dt, kname = _packed_words(mn, mk, qname)
    def _mk_moe(qname:str=qname, qtype:int=qtype, mn:int=mn, mk:int=mk) -> tuple[Callable[[], Tensor], VerifyFn]:
      w_packed, tile_u8 = _make_packed(E * mn, mk, qname, dev, seed=19)
      w_u8 = w_packed if w_packed.dtype == dtypes.uint8 else w_packed.bitcast(dtypes.uint8)
      sel, x = Tensor(list(range(TOPK)), dtype=dtypes.int32, device=dev).realize(), _nonuniform((1, mk), dtypes.float32, dev, scale=0.5, freq=0.011)
      lut, _ = _iq_lut(qtype, dev)
      def _verify(out:Tensor) -> None:
        w4 = ggml_data_to_tensor(tile_u8[:CR].flatten(), CR * mk, qtype).reshape(CR, mk)
        ref1, got = (x @ w4.T).numpy()[0], out[:, :CR].float().numpy()
        _assert_close(got, np.broadcast_to(ref1, got.shape), rtol=4e-2, atol=5e-2)
      return (lambda: _krun("amd_gemv_quant_f32", (TOPK, mn), dev, w_u8, x, sel, lut, qtype=qtype,
                            tokens=TOPK, x_rows=1, out_features=mn, in_features=mk, raw_bytes=int(w_u8.numel()),
                            group_rows=0, use_sel=1, x_share=TOPK, ndev=1, arch=target[0], num_experts=E)), _verify
    cases.append((f"moe_gemv_{qname}_E{E}k{TOPK}_{mn}x{mk}", TOPK * per_e_words * w_dt.itemsize, 2 * TOPK * mn * mk, _mk_moe))

  if target[0] == 12 and not smoke:
    mt_t, me, mk_t, mn_p, mk_p = 512, 64, 8, 1024, 2048
    qtype, per_e_words, w_dt, _ = _packed_words(mn_p, mk_p, "Q4_K")
    def _mk_moe_pre() -> tuple[Callable[[], Tensor], VerifyFn]:
      w_packed, tile_u8 = _make_packed(me * mn_p, mk_p, "Q4_K", dev, seed=23)
      w_u8 = w_packed if w_packed.dtype == dtypes.uint8 else w_packed.bitcast(dtypes.uint8)
      sel = Tensor(np.random.default_rng(7).integers(0, me, size=(mt_t * mk_t,), dtype=np.int32), device=dev).realize()
      x = _nonuniform((mt_t, mk_p), dtypes.float32, dev, scale=0.5, freq=0.011)
      lut, _ = _iq_lut(qtype, dev)
      m_tot = mt_t * mk_t
      def _verify(out:Tensor) -> None:
        w4 = ggml_data_to_tensor(tile_u8[:CR].flatten(), CR * mk_p, qtype).reshape(CR, mk_p)
        ref = (x[:16].float() @ w4.T).numpy()
        got = out[:16 * mk_t, :CR].float().numpy().reshape(16, mk_t, CR)
        _assert_close(got, np.broadcast_to(ref[:, None, :], got.shape), rtol=4e-2, atol=5e-2)
      return (lambda: _krun("amd_gemv_quant_f32", (m_tot, mn_p), dev, w_u8, x, sel, lut, qtype=qtype,
                            tokens=m_tot, x_rows=mt_t, out_features=mn_p, in_features=mk_p, raw_bytes=int(w_u8.numel()),
                            group_rows=0, use_sel=1, x_share=mk_t, ndev=1, arch=target[0], num_experts=me)), _verify
    cases.append((f"moe_prefill_Q4_K_T{mt_t}_E{me}k{mk_t}_{mn_p}x{mk_p}", me * per_e_words * w_dt.itemsize,
                  2 * mt_t * mk_t * mn_p * mk_p, _mk_moe_pre))

  for m, n, k in (((1, FFN, K),) if fast else ((1, N, K), (1, FFN, K), (4, N, K)) + (() if smoke else ((1, 3072, 8192),))):
    def _mk_dense(m:int=m, n:int=n, k:int=k) -> tuple[Callable[[], Tensor], VerifyFn]:
      w, x = _nonuniform((n, k), dtypes.float16, dev, scale=0.25, freq=0.019), _nonuniform((m, k), dtypes.float32, dev, scale=0.5, freq=0.023)
      return (lambda: _krun("amd_gemv_f16_f32", (m, n), dev, w, x, tokens=m, out_features=n, in_blks=k // 128, val_chunk=4)), \
             (lambda out: _assert_close(out[:, :CR].float().numpy(), (x @ w[:CR].float().T).numpy(), rtol=3e-3, atol=3e-3))
    cases.append((f"dense_gemv_f16_{m}x{n}x{k}", n * k * 2 + m * (k + n) * 4, 2 * m * n * k, _mk_dense))

  if not cdna:
    fa_pre_specs = [(AH, AHKV, AM, AN, AD)] if (smoke or fast) else (
      [(32, 8, 256, 2048, 128), (32, 8, 32, 2048, 128), (24, 4, 256, 2048, 256)] + ([(16, 4, 256, 2048, 512)] if target[0] == 12 else []))
    for fh, fhkv, fm, fn, fd in fa_pre_specs:
      def _mk_fa_pre(fh:int=fh, fhkv:int=fhkv, fm:int=fm, fn:int=fn, fd:int=fd) -> tuple[Callable[[], Tensor], VerifyFn]:
        q = _nonuniform((1, fh, fm, fd), dtypes.float16, dev, scale=0.2, freq=0.029)
        kv = _nonuniform((2, 1, fhkv, fn, fd), dtypes.float16, dev, scale=0.2, freq=0.031, phase=0.7)
        def _verify(out:Tensor) -> None:
          mask = Tensor.full((fm, fn), float("-inf"), dtype=dtypes.float32, device=dev).triu(fn - fm + 1)
          ref = q[:, :1].float().scaled_dot_product_attention(kv[0, :, :1].float(), kv[1, :, :1].float(), attn_mask=mask, enable_gqa=True)
          _assert_close(out[:, :1].float().numpy(), ref.numpy(), rtol=5e-3, atol=5e-3)
        return (lambda: _krun(f"{arch}_flash_attn", (1, fh, fm, fd), dev, q, kv, UOp.const(fn), UOp.const(fm),
                              B=1, H=fh, H_KV=fhkv, m_blks=fm // 32, d_blks=fd // 64, n_blks=fn // 32,
                              valid_kv_offset=0, q_start_from_end=1)), _verify
      io_b = (fh * fm * fd + 2 * fhkv * fn * fd) * 2 + fh * fm * fd * 4
      cases.append((f"flash_attn_prefill_H{fh}_M{fm}_N{fn}_D{fd}", io_b, 4 * fh * fm * fn * fd, _mk_fa_pre))

  for fh, fhkv, ctx, fd in ([(AH, AHKV, AN, AD)] if smoke else [(32, 8, 2048, 128), (32, 8, 8192, 128)] if fast else
                            [(32, 8, 512, 128), (32, 8, 2048, 128), (32, 8, 8192, 128), (24, 4, 2048, 256)]):
    def _mk_fa_dec(fh:int=fh, fhkv:int=fhkv, ctx:int=ctx, fd:int=fd) -> tuple[Callable[[], Tensor], VerifyFn]:
      q = _nonuniform((1, fh, 1, fd), dtypes.float16, dev, scale=0.2, freq=0.037)
      kv = _nonuniform((2, 1, fhkv, ctx, fd), dtypes.float16, dev, scale=0.2, freq=0.041, phase=1.1)
      def _run() -> Tensor:
        return _krun("amd_flash_decode", (1, fh, 1, fd), dev, q, kv, UOp.const(ctx), B=1, H_KV=fhkv, G=fh // fhkv,
                     d_blks=fd // 64, n_blks=ctx // 64, C=min(max(8, 64 // fhkv), 64, ceildiv(ctx, 64)), valid_kv_offset=0)
      def _verify(out:Tensor) -> None:
        ref = q[:, :1].float().scaled_dot_product_attention(kv[0, :, :1].float(), kv[1, :, :1].float(), enable_gqa=True)
        _assert_close(out[:, :1].float().numpy(), ref.numpy(), rtol=5e-3, atol=5e-3)
      return _run, _verify
    cases.append((f"flash_attn_decode_H{fh}_N{ctx}_D{fd}", 2 * fhkv * ctx * fd * 2 + fh * fd * 6, 4 * fh * ctx * fd, _mk_fa_dec))

  gd_specs = [(2, 8, 32, 1)] if smoke else [(16, 64, 128, 1), (48, 1, 128, 1)] if fast else [(16, 64, 128, 1), (48, 1, 128, 1), (16, 64, 128, 128)]
  for gh, gt, gd, ak in gd_specs:
    def _mk_gd(gh:int=gh, gt:int=gt, gd:int=gd, ak:int=ak) -> tuple[Callable[[], Tensor], VerifyFn]:
      q, k, v = (_nonuniform((1, gh, gt, gd), dtypes.float32, dev, scale=sc, freq=fr, phase=ph)
                 for sc, fr, ph in ((0.05, 0.043, 0.3), (0.05, 0.047, 0.5), (0.2, 0.053, 0.9)))
      beta = (_nonuniform((1, gh, gt), dtypes.float32, dev, scale=0.05, freq=0.059) + 0.1).contiguous().realize()
      alpha = (_nonuniform((1, gh, gt, ak), dtypes.float32, dev, scale=0.05, freq=0.061) + 0.9).contiguous().realize()
      state = Tensor.zeros(1, gh, gd, gd, dtype=dtypes.float32, device=dev).contiguous().realize()
      def _verify(out:Tensor) -> None:
        q0, k0, v0, b0, a0 = q[0, 0].numpy(), k[0, 0].numpy(), v[0, 0].numpy(), beta[0, 0].numpy(), alpha[0, 0].numpy()
        st, ref = np.zeros((gd, gd), dtype=np.float32), np.empty((gt, gd), dtype=np.float32)
        for t in range(gt):
          st *= a0[t]
          st += np.outer((v0[t] - st @ k0[t]) * b0[t], k0[t])
          ref[t] = st @ q0[t]
        _assert_close(out[0, 0].numpy(), ref, rtol=2e-3, atol=2e-4)
        _assert_close(state[0, 0].numpy(), st, rtol=2e-3, atol=2e-4)
      return (lambda: _krun("amd_gated_delta", (1, gh, gt, gd), dev, q, k, v, beta, alpha, state, UOp.const(1),
                            batch=1, heads=gh, tokens=gt, v_tiles=gd // 4, k_blks=gd // 32, alpha_k=ak, arch=target[0])), _verify
    tag = "kda" if ak > 1 else ("decode" if gt == 1 else "prefill")
    cases.append((f"gated_delta_{tag}_H{gh}_T{gt}_D{gd}", (4 * gh * gt * gd + 2 * gh * gd * gd) * 4, 6 * gh * gt * gd * gd, _mk_gd))

  def _mk_hc() -> tuple[Callable[[], Tensor], VerifyFn]:
    t_h = 2 if smoke else 32
    mix, flat = _nonuniform((t_h, 24), dtypes.float32, dev, scale=0.2, freq=0.067), _nonuniform((t_h, K), dtypes.float32, dev, scale=0.5, freq=0.071)
    base = _nonuniform((24,), dtypes.float16, dev, scale=0.2, freq=0.073)
    scale = (_nonuniform((3,), dtypes.float16, dev, scale=0.2, freq=0.079) + 1.0).cast(dtypes.float16).contiguous().realize()
    def _verify(out:Tensor) -> None:
      m_n = mix * (flat.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
      pre, post = (m_n[:, :4] * scale[0].float() + base[:4].float()).sigmoid(), (m_n[:, 4:8] * scale[1].float() + base[4:8].float()).sigmoid() * 2.0
      comb = (m_n[:, 8:] * scale[2].float() + base[8:].float()).reshape(t_h, 4, 4).softmax(-1) + 1e-6
      for i in range(3): comb = comb / (comb.sum(-1 if i % 2 else -2, keepdim=True) + 1e-6)
      _assert_close(out.float().numpy(), pre.cat(post, comb.reshape(t_h, 16), dim=-1).numpy(), rtol=2e-3, atol=2e-3)
    return (lambda: _krun("amd_hc_coeffs", (t_h, 24), dev, mix, flat, base.float().contiguous(), scale.float().contiguous(),
                          tokens=t_h, features=K, iters=2, eps=1e-6, norm_eps=1e-6)), _verify
  cases.append((f"hc_coeffs_{2 if smoke else 32}x{K}", (2 if smoke else 32) * K * 4, 0, _mk_hc))
  return cases

def main() -> None:
  ap = argparse.ArgumentParser(description="Benchmark & verify end-to-end LLM slopkernels via UIR on AMD GPUs")
  ap.add_argument("--warmup", type=int, default=2)
  ap.add_argument("--iters", type=int, default=5)
  ap.add_argument("--filter", type=str, default="")
  ap.add_argument("--smoke", action="store_true")
  ap.add_argument("--fast", action="store_true")
  ap.add_argument("--no-verify", action="store_true")
  ap.add_argument("--flush-mb", type=int, default=256)
  ap.add_argument("--json", type=str, default="")
  args = ap.parse_args()

  dev, target = Device.DEFAULT, getattr(Device[Device.DEFAULT], "target", None)
  gpu_name, cus, peak_bw, peak_tf = _peak_specs(dev)
  flush_mb = min(args.flush_mb, 2) if args.smoke else args.flush_mb
  flush_fn, t_start = _make_cache_flusher(flush_mb), time.perf_counter()
  peak_str = f"{gpu_name} ({cus} CU, {peak_bw:.0f} GB/s, {peak_tf:.1f} TF)" if peak_bw > 0 else f"target={target}"
  print(f"slopkernels LLM benchmark | DEV={dev} | {peak_str} | warmup={args.warmup} iters={args.iters} flush={flush_mb}MB "
        f"verify={not args.no_verify} smoke={args.smoke}", flush=True)
  print(f"{'Op':<46} {'Median (us)':>12} {'Min (us)':>10} {'GB/s':>9} {'%BW':>7} {'TFLOPS':>9} {'%TF':>7}", flush=True)
  print("-" * 106, flush=True)
  pat, results = re.compile(args.filter) if args.filter else None, []
  for name, io_bytes, flops, setup_fn in build_benchmarks(dev, smoke=args.smoke, fast=args.fast):
    if pat is not None and not pat.search(name): continue
    step_fn, verify_fn = setup_fn()
    med_us, min_us = _time_fn(step_fn, None if args.no_verify else verify_fn, args.warmup, args.iters, flush_fn)
    gb_s = (io_bytes / 1e9) / (med_us / 1e6) if io_bytes > 0 and med_us > 0 else 0.0
    tflops = (flops / 1e12) / (med_us / 1e6) if flops > 0 and med_us > 0 else 0.0
    bw_pct, tf_pct = (100.0 * gb_s / peak_bw if peak_bw > 0 else 0.0), (100.0 * tflops / peak_tf if peak_tf > 0 else 0.0)
    r = BenchResult(name=name, median_us=med_us, min_us=min_us, gb_s=gb_s, bw_pct=bw_pct, tflops=tflops, tflops_pct=tf_pct)
    results.append(r)
    bw_str = f"{r.gb_s:>9.1f} {bw_pct:>6.1f}%" if peak_bw > 0 else f"{r.gb_s:>9.1f} {'-':>7}"
    tf_str = f"{r.tflops:>9.2f} {tf_pct:>6.1f}%" if peak_tf > 0 and flops > 0 else f"{r.tflops:>9.2f} {'-':>7}"
    if peak_bw > 0 and (peak_tf <= 0 or flops <= 0 or bw_pct >= tf_pct):
      bw_str = colored(bw_str, "green" if bw_pct >= 85 else "yellow" if bw_pct >= 70 else "red")
    elif peak_tf > 0 and flops > 0:
      tf_str = colored(tf_str, "green" if tf_pct >= 70 else "yellow" if tf_pct >= 40 else "red")
    print(f"{r.name:<46} {r.median_us:>12.1f} {r.min_us:>10.1f} {bw_str} {tf_str}", flush=True)
  print("-" * 106, flush=True)
  print(f"Benchmarked & verified {len(results)} LLM ops in {time.perf_counter() - t_start:.1f}s.", flush=True)
  if args.json:
    with open(args.json, "w") as f: json.dump([r.__dict__ for r in results], f, indent=2)

if __name__ == "__main__":
  main()
