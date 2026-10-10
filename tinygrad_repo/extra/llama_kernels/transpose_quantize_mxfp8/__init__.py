from __future__ import annotations
import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.llama_kernels import THREADS_PER_WG, alloc_like, dname_of, compile_hip

TILE_N = THREADS_PER_WG   # 256
BLK = 32

@functools.cache
def _custom_transpose_quantize_mxfp8(q:UOp, e8:UOp, g:UOp, dname:str) -> UOp:
  M, N = g.shape
  num_wg = (M // BLK) * (N // TILE_N)
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(num_wg, "gidx0")
  mem = M * N * 2 + M * N + (M // BLK) * N   # read bf16, write fp8 + e8
  sink = UOp.sink(q.base, e8.base, g.base, threads, workgroups,
                  arg=KernelInfo(f"transpose_quantize_mxfp8_{M}_{N}", estimates=Estimates(ops=M*N, mem=mem)))
  src = (pathlib.Path(__file__).parent/"transpose_quantize_mxfp8.cpp").read_text()
  defines = [f"-DM_DIM={M}", f"-DN_DIM={N}", f"-DTHREADS_PER_WG={THREADS_PER_WG}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

def transpose_quantize_mxfp8(g:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  # fused g.T quantize: returns (q, e8, si) == quantize_mxfp8(g.T) — q (N,M) fp8, e8 (N, M/32), si packed (M/128, N)
  assert g.ndim == 2 and g.dtype == dtypes.bfloat16, f"{g.shape} {g.dtype}"
  from extra.gemm.cdna_asm_gemm import FP8_DTYPE, mx_pack
  M, N = g.shape
  assert M % BLK == 0 and N % TILE_N == 0, f"M={M} must%{BLK}, N={N} must%{TILE_N}"
  device = g.device
  axis = g.uop.axis if isinstance(device, tuple) else None
  out_axis = None if axis is None else (1 if axis == 0 else 0)
  q = alloc_like((N, M), FP8_DTYPE, device, out_axis)
  e8 = alloc_like((N, M // BLK), dtypes.uint8, device, out_axis)
  fxn = functools.partial(_custom_transpose_quantize_mxfp8, dname=dname_of(device))
  q, e8, *_ = Tensor.custom_kernel(q, e8, g, fxn=fxn)
  return q, e8, mx_pack(e8)
