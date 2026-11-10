from functools import partial

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.helpers import round_up
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp


def pack(x):
  rows, width = x.shape
  return x.pad(((0, round_up(rows, 16)-rows), (0, 0))).reshape(-1, 16, width//16, 4, 4).permute(0, 2, 3, 1, 4).contiguous().reshape(-1, width)


def unpack(x):
  rows, width = x.shape
  return x.reshape(rows//16, width//16, 4, 16, 4).permute(0, 3, 1, 2, 4).reshape(rows, width)


def _gemm(out, x, weight, x_scale, w_scale, *bias, packed_input=True):
  rows, columns = out.shape
  width = weight.shape[1]
  m, n, k = UOp.range(round_up(rows, 64), 0), UOp.range(round_up(columns, 64), 1), UOp.range(width, 2)
  # Each wave reads contiguous FP8 fragments from a 16 by 16 tile.
  def index(row): return (row//16*(width//16)+k//16)*256 + (k%16//4)*64 + (row%16)*4 + k%4
  a = x.flatten().index((index(m) if packed_input else m*width+k).valid(m < rows))
  b = weight.flatten().index(index(n).valid(n < columns))
  value = (a*b).cast(dtypes.float32).reduce(k, arg=Ops.ADD)
  value = value * (x_scale.flatten().index(0) * w_scale.flatten().index(0))
  if bias:
    value = value + bias[0].flatten().index(n.valid(n < columns)).cast(dtypes.float32)
  store = out.flatten().index((m*columns+n).valid((m < rows) & (n < columns))).store(value.cast(out.dtype))
  schedule = Scheduler(store.end(m, n).sink(arg=KernelInfo()), Device[Device.DEFAULT].renderer)
  schedule.convert_loop_to_global()
  ranges = schedule.apply_opt(Opt(OptOps.TC, 0, (-1, 0, 1)))
  for i in (1, 0):
    schedule.apply_opt(Opt(OptOps.SPLIT, schedule.rngs.index(ranges[i]), (4, AxisType.UPCAST)))
  return schedule.get_optimized_ast(name_override='alpamayo_fp8_gemm')


def gemm(x, weight, scales, bias=None, dtype=dtypes.bfloat16, packed_input=True):
  out = Tensor.empty(x.shape[0], weight.shape[0], device=x.device, dtype=dtype)
  return Tensor.custom_kernel(out, pack(x) if packed_input else x, pack(weight), *scales, *(() if bias is None else (bias,)),
                              fxn=partial(_gemm, packed_input=packed_input))[0]
