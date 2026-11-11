from functools import partial, reduce
import math

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import round_up
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp


def _pack(x):
  batch, heads, rows, width = x.shape
  padded_rows, padded_width = round_up(rows, 16), round_up(width, 16)
  x = x.pad(((0, 0), (0, 0), (0, padded_rows-rows), (0, padded_width-width)))
  return x.reshape(batch, heads, padded_rows//16, 16, padded_width//16, 4, 4).permute(0, 1, 2, 4, 5, 3, 6).contiguous()


def _matmul(out, a, b, shape, causal=False, scaled=False):
  batch, heads, kv_heads, rows, columns, width = shape
  padded_rows, padded_columns, padded_width = (round_up(n, 16) for n in (rows, columns, width))
  head = UOp.range(batch*heads, 0)
  m, n, k = UOp.range(round_up(rows, 64), 1), UOp.range(round_up(columns, 64), 2), UOp.range(padded_width, 3)
  def index(h, row, length):
    return ((h*(length//16)+row//16)*(padded_width//16)+k//16)*256 + (k%16//4)*64 + (row%16)*4 + k%4
  aa = a.flatten().index(index(head, m, padded_rows).valid(m < rows))
  kv_head = (head//heads)*kv_heads + (head%heads)//(heads//kv_heads)
  bb = b.flatten().index(index(kv_head, n, padded_columns).valid(n < columns))
  value = (aa*bb).cast(dtypes.float32).reduce(k, arg=Ops.ADD).cast(out.dtype)
  if scaled:
    value = value * UOp.const(width**-0.5, out.dtype)
  if causal:
    value = (n <= m).where(value, UOp.const(-math.inf, out.dtype))
  store = out.flatten().index(((head*rows+m)*columns+n).valid((m < rows) & (n < columns))).store(value)
  schedule = Scheduler(store.end(head, m, n).sink(arg=KernelInfo()), Device[Device.DEFAULT].renderer)
  schedule.convert_loop_to_global()
  ranges = schedule.apply_opt(Opt(OptOps.TC, 0, (-1, 0, 1)))
  for i in (1, 0):
    schedule.apply_opt(Opt(OptOps.SPLIT, schedule.rngs.index(ranges[i]), (4, AxisType.UPCAST)))
  return schedule.get_optimized_ast(name_override='alpamayo_attention_scores' if scaled else 'alpamayo_attention_values')


def _warp_reduce(value, maximum=False):
  shuffle = UOp.custom_function('llvm.amdgcn.ds.swizzle', dtype=dtypes.int32)
  for shift in (16, 8, 4, 2, 1):
    other = shuffle.call(value.bitcast(dtypes.int32), UOp.const(0x1f | (shift << 10), dtypes.int32)).bitcast(dtypes.float32)
    value = value.maximum(other) if maximum else value + other
  return value


def _softmax(out, scores, packed=True):
  batch, heads, rows, columns = scores.shape
  row = UOp.range(batch*heads*rows, 0, AxisType.GLOBAL)
  wave, lane = UOp.range(4, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  cols = [wave*32+lane+i*128 for i in range((columns+127)//128)]
  values = [(c < columns).where(scores.flatten().index((row*columns+c).valid(c < columns)).cast(dtypes.float32),
                               UOp.const(-math.inf, dtypes.float32)) for c in cols]
  maximum = _warp_reduce(reduce(UOp.maximum, values), maximum=True)
  maxima = UOp.alloc((4,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  maxima_store = maxima.index(wave.valid(lane.eq(0))).store(maximum)
  maximum = reduce(UOp.maximum, [maxima.after(maxima_store).index(i) for i in range(4)])
  exps = [((value-maximum)*math.log2(math.e)).exp2() for value in values]
  total = _warp_reduce(sum(exps))
  sums = UOp.alloc((4,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  sums_store = sums.index(wave.valid(lane.eq(0))).store(total)
  inv = sum(sums.after(sums_store).index(i) for i in range(4)).reciprocal()
  # Write the layout consumed by the next WMMA, including zero probabilities for padded keys.
  def index(c):
    if not packed:
      return row*columns+c
    return (((row//rows)*(round_up(rows, 16)//16)+(row%rows)//16)*(round_up(columns, 16)//16)+c//16)*256 + \
           (c%16//4)*64 + (row%rows%16)*4 + c%4
  stores = [out.flatten().index(index(c).valid(c < (round_up(columns, 16) if packed else columns))).store((e*inv).cast(out.dtype))
            for c, e in zip(cols, exps, strict=True)]
  return UOp.group(*stores).end(row, wave, lane).sink(arg=KernelInfo(name='alpamayo_attention_softmax', opts_to_apply=()))


def attention(q, k, v, causal=False):
  batch, heads, queries, width = q.shape
  kv_heads, keys = k.shape[1:3]
  scores = Tensor.empty(batch, heads, queries, keys, device=q.device, dtype=q.dtype)
  scores = Tensor.custom_kernel(scores, _pack(q), _pack(k),
    fxn=partial(_matmul, shape=(batch, heads, kv_heads, queries, keys, width), causal=causal, scaled=True))[0]
  probs = Tensor.empty(batch, heads, round_up(queries, 16), round_up(keys, 16), device=q.device, dtype=q.dtype)
  probs = Tensor.custom_kernel(probs, scores, fxn=_softmax)[0]
  out = Tensor.empty(q.shape, device=q.device, dtype=q.dtype)
  return Tensor.custom_kernel(out, probs, _pack(v.transpose(-2, -1)),
    fxn=partial(_matmul, shape=(batch, heads, kv_heads, queries, width, keys)))[0]
