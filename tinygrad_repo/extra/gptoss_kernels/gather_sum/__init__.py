import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from extra.llama_kernels import alloc_like, compile_hip

@functools.cache
def _gather_sum(out:UOp, table:UOp, indices:UOp) -> UOp:
  groups, tokens, hidden = out.shape
  rows = table.shape[1]
  assert table.shape == (groups, rows, hidden) and indices.shape == (groups, tokens*4)
  assert out.dtype == table.dtype == dtypes.bfloat16 and indices.dtype == dtypes.int32
  assert hidden % 64 == 0
  sink = UOp.sink(out.base, table.base, indices.base, UOp.special(64, "lidx0"), UOp.special(groups*tokens, "gidx0"),
                  arg=KernelInfo(f"gather_sum_{groups}_{tokens}_{rows}_{hidden}"))
  src = (pathlib.Path(__file__).parent/"gather_sum.cpp").read_text()
  lib = compile_hip(src, [f"-DTOKENS={tokens}", f"-DROWS={rows}", f"-DHIDDEN={hidden}", "-fno-fast-math"])
  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def gather_sum(table:Tensor, indices:Tensor) -> Tensor:
  groups, _, hidden = table.shape
  assert indices.shape[0] == groups and indices.shape[1] % 4 == 0
  out = alloc_like((groups, indices.shape[1]//4, hidden), table.dtype, table.device, table.uop.axis)
  return Tensor.custom_kernel(out, table, indices, fxn=_gather_sum)[0]
