import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from extra.llama_kernels import alloc_like, compile_hip

@functools.cache
def _inverse_rows(out:UOp, dest_row:UOp, counts:UOp, off:UOp) -> UOp:
  groups, rows = out.shape
  entries, experts = dest_row.shape[1], counts.shape[1]
  assert rows >= entries and counts.shape == (groups, experts) and off.shape == (groups, experts+1)
  sink = UOp.sink(out.base, dest_row.base, counts.base, off.base, UOp.special(256, "lidx0"),
                  UOp.special(groups*((rows+255)//256), "gidx0"), arg=KernelInfo(f"dispatch_inverse_{groups}_{rows}_{entries}_{experts}"))
  src = (pathlib.Path(__file__).parent/"inverse.cpp").read_text()
  lib = compile_hip(src, [f"-DROWS={rows}", f"-DENTRIES={entries}", f"-DEXPERTS={experts}"])
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def inverse_rows(dest_row:Tensor, counts:Tensor, off:Tensor, rows:int) -> Tensor:
  out = alloc_like((dest_row.shape[0], rows), dtypes.int32, dest_row.device, dest_row.uop.axis)
  return Tensor.custom_kernel(out, dest_row, counts, off, fxn=_inverse_rows)[0]

@functools.cache
def _dispatch_gather(out:UOp, x:UOp, dest_row:UOp, src_row:UOp) -> UOp:
  from extra.gemm.moe_routing import _blk_for
  groups, tokens, hidden = x.shape
  rows = src_row.shape[1]
  assert out.shape == (groups*rows, hidden) and dest_row.shape == (groups, tokens*4)
  assert x.dtype == out.dtype and x.dtype.itemsize == 1
  block = _blk_for(hidden)
  g, row = UOp.range(groups, 0, AxisType.GLOBAL), UOp.range(rows, 1, AxisType.GLOBAL)
  outer, inner = UOp.range(hidden//block, 2), UOp.range(block, 3, AxisType.LOCAL)
  col = outer*block + inner
  source = src_row.index(g, row).load().cast(dtypes.weakint)
  anchor = dest_row.index(g, 0).load() >= 0
  valid = (source >= 0) & anchor
  token = valid.where(source, source.const_like(0)) // 4
  value = valid.where(x.index(g, token, col).load(), UOp.const(0, x.dtype))
  return out.reshape(groups, rows, hidden).index(g, row, col).store(value).end(g, row, outer, inner).sink(
    arg=KernelInfo(f"dispatch_gather_{groups}_{rows}_{tokens}_{hidden}", opts_to_apply=()))

def _dispatch_gather_bwd(gradient:UOp, kernel:UOp) -> tuple:
  from extra.gemm.moe_routing import _gscatter_bwd
  # The first three arguments match scatter's backward; src_row is forward-only.
  x = kernel.src[2]
  return (*_gscatter_bwd(gradient.reshape(x.shape[0], -1, x.shape[-1]), kernel), None)

def dispatch_gather(x:Tensor, dest_row:Tensor, src_row:Tensor) -> Tensor:
  out = alloc_like((x.shape[0]*src_row.shape[1], x.shape[-1]), x.dtype, x.device, x.uop.axis)
  return Tensor.custom_kernel(out, x, dest_row, src_row, fxn=_dispatch_gather, grad_fxn=_dispatch_gather_bwd)[0]
