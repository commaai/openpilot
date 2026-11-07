import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from extra.llama_kernels import alloc_like, compile_hip

@functools.cache
def _kernel(q:UOp, e8:UOp, si:UOp, src_q:UOp, src_e8:UOp) -> UOp:
  E, M, N = src_q.shape
  assert M % 128 == 0 and N % 128 == 0
  assert src_e8.shape == (E, M, N // 32)
  assert q.shape == (E, N, M) and e8.shape == (E, N, M // 32) and si.shape == (E, M // 128, N)
  sink = UOp.sink(q.base, e8.base, si.base, src_q.base, src_e8.base,
                  UOp.special(128, "lidx0"), UOp.special(E * (M // 32) * (N // 128), "gidx0"),
                  arg=KernelInfo(f"transpose_requantize_mxfp8_{E}_{M}_{N}"))
  src = (pathlib.Path(__file__).parent/"transpose_requantize.cpp").read_text()
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=compile_hip(src, [f"-DE_DIM={E}", f"-DM_DIM={M}", f"-DN_DIM={N}"]))))

def transpose_requantize_mxfp8(q:Tensor, e8:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  E, M, N = q.shape
  assert q.dtype == dtypes.fp8e4m3 and e8.dtype == dtypes.uint8 and e8.shape == (E, M, N // 32)
  axis = q.uop.axis
  out_axis = None if axis is None else (0 if axis == 0 else 3 - axis)
  qt = alloc_like((E, N, M), q.dtype, q.device, out_axis)
  et = alloc_like((E, N, M // 32), dtypes.uint8, q.device, out_axis)
  si = alloc_like((E, M // 128, N), dtypes.uint32, q.device, axis)
  return tuple(Tensor.custom_kernel(qt, et, si, q, e8, fxn=_kernel)[:3])
