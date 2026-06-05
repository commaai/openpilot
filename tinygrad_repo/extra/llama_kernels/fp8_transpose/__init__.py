from __future__ import annotations
import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.llama_kernels import THREADS_PER_WG, alloc_like, dname_of, compile_hip

TILE = 64

@functools.cache
def _custom_fp8_transpose(out:UOp, inp:UOp, dname:str) -> UOp:
  M, N = inp.shape
  num_wg = (M // TILE) * (N // TILE)
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(num_wg, "gidx0")
  mem = M * N * 2  # one byte read + one byte write per element
  sink = UOp.sink(out.base, inp.base, threads, workgroups,
                  arg=KernelInfo(f"fp8_transpose_{M}_{N}",
                                 estimates=Estimates(ops=M*N, mem=mem)))
  src = (pathlib.Path(__file__).parent/"fp8_transpose.cpp").read_text()
  defines = [f"-DM_DIM={M}", f"-DN_DIM={N}", f"-DTHREADS_PER_WG={THREADS_PER_WG}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

def fast_fp8_transpose(t:Tensor) -> Tensor:
  assert t.ndim == 2, f"fast_fp8_transpose needs 2D input, got shape {t.shape}"
  assert t.dtype in dtypes.fp8s, f"fast_fp8_transpose needs fp8 dtype, got {t.dtype}"
  M, N = t.shape
  assert M % TILE == 0 and N % TILE == 0, f"M={M}, N={N} must be multiples of {TILE}"

  device = t.device
  axis = t.uop.axis if isinstance(device, tuple) else None
  out_axis = None
  if axis == 0: out_axis = 1
  elif axis == 1: out_axis = 0
  elif axis is not None:
    raise ValueError(f"fast_fp8_transpose: unsupported axis {axis}")

  out = alloc_like((N, M), t.dtype, device, out_axis)
  fxn = functools.partial(_custom_fp8_transpose, dname=dname_of(device))
  out, _ = Tensor.custom_kernel(out, t, fxn=fxn)
  return out
