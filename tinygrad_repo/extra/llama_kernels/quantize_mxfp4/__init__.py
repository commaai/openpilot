import functools, math, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.llama_kernels import alloc_like, compile_hip

@functools.cache
def _custom_quantize_mxfp4(row_fp4:UOp, row_scale:UOp, col_fp4:UOp, col_scale:UOp, x:UOp, *, shuffle_row:bool, shuffle_col:bool) -> UOp:
  M, N = math.prod(x.shape[:-1]), x.shape[-1]
  assert M % 256 == 0 and N % 256 == 0, f"MXFP4 quantization requires multiples of 256, got {x.shape}"
  name = f"quantize_mxfp4_{int(shuffle_row)}_{int(shuffle_col)}_{M}_{N}"
  mem = M*N*2 + M*N + M*N//16  # read bf16, write row+col fp4 + e8m0
  outputs = (row_fp4, row_scale, col_fp4, col_scale)
  sink = UOp.sink(*(o.base for o in outputs), x.base,
                  *(UOp(Ops.CUSTOM, dtypes.void, (o.base.index(0),), arg="") for o in outputs),
                  UOp.special(256, "lidx0"), UOp.special(M//128, "gidx0"), UOp.special(N//64, "gidx1"),
                  arg=KernelInfo(name, estimates=Estimates(ops=12*M*N, mem=mem)))
  src = (pathlib.Path(__file__).parent/"quantize_mxfp4.cpp").read_text()
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
    UOp(Ops.BINARY, arg=compile_hip(src, [f"-DKERNEL_NAME={name}", f"-DM_DIM={M}", f"-DN_DIM={N}",
                                             f"-DSHUFFLE_ROWWISE_FP4_VALUE={int(shuffle_row)}",
                                             f"-DSHUFFLE_COLWISE_FP4_VALUE={int(shuffle_col)}"]))))

def quantize_mxfp4(x:Tensor, *, shuffle_row:bool=False, shuffle_col:bool=False, flatten_row:bool=False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  assert x.dtype == dtypes.bfloat16 and x.ndim >= 2, f"expected BF16 matrix, got {x.dtype} {x.shape}"
  M, N = math.prod(x.shape[:-1]), x.shape[-1]
  assert M % 256 == 0 and N % 256 == 0, f"MXFP4 quantization requires multiples of 256, got {x.shape}"
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  row_axis = 0 if flatten_row and axis is not None else axis
  col_axis = None if axis is None else (0 if axis == x.ndim-1 else 1)
  outputs = (alloc_like((M, N//2) if flatten_row else (*x.shape[:-1], N//2), dtypes.uint8, x.device, row_axis),
             alloc_like((M, N//32) if flatten_row else (*x.shape[:-1], N//32), dtypes.uint8, x.device, row_axis),
             alloc_like((N, M//2), dtypes.uint8, x.device, col_axis),
             alloc_like((N, M//32), dtypes.uint8, x.device, col_axis))
  fxn = functools.partial(_custom_quantize_mxfp4, shuffle_row=shuffle_row, shuffle_col=shuffle_col)
  return tuple(Tensor.custom_kernel(*outputs, x, fxn=fxn)[:4])
