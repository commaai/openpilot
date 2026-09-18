from __future__ import annotations
import functools, math, pathlib

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.llama_kernels import alloc_like, compile_hip, dname_of

THREADS = 256
MAX_PARTIALS = 608

@functools.cache
def _custom_sum_squares(partial:UOp, x:UOp, *, dname:str) -> UOp:
  assert x.dtype == dtypes.bfloat16 and partial.dtype == dtypes.float32
  n, n_partials = x.numel(), partial.numel()
  threads, workgroups = UOp.special(THREADS, "lidx0"), UOp.special(n_partials, "gidx0")
  sink = UOp.sink(partial.base, x.base, threads, workgroups,
                  arg=KernelInfo(f"grad_norm_bf16_partials_{n}_{n_partials}", estimates=Estimates(ops=2*n, mem=2*n+4*n_partials)))
  src = (pathlib.Path(__file__).parent/"sum_squares.cpp").read_text()
  defines = [f"-DN_ELEMS={n}", f"-DN_PARTIALS={n_partials}", f"-DTHREADS={THREADS}"]
  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

def sum_squares_bf16(x:Tensor) -> Tensor:
  assert x.dtype == dtypes.bfloat16
  device = x.device
  axis = x.uop.axis if isinstance(device, tuple) else None
  local_n = math.prod(x.uop.shard_shape if axis is not None else x.shape)
  n_partials = min(MAX_PARTIALS, max(1, (local_n + THREADS * 64 - 1) // (THREADS * 64)))
  partial = alloc_like((n_partials * len(device),), dtypes.float32, device, 0) \
    if isinstance(device, tuple) and axis is not None else alloc_like((n_partials,), dtypes.float32, device)
  partial, *_ = Tensor.custom_kernel(partial, x, fxn=functools.partial(_custom_sum_squares, dname=dname_of(device)))
  return partial.sum()
