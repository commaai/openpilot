import functools
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from extra.llama_kernels import alloc_like

ALPHA, LIMIT, LOG2E = 1.702, 7.0, 1.4426950408889634

@functools.cache
def _swiglu_quantize(q:UOp, e8:UOp, h:UOp) -> UOp:
  rows = h.shape[0]
  wg = UOp.range((rows*24 + 255)//256, 0, AxisType.GLOBAL)
  tid = UOp.range(256, 1, AxisType.LOCAL)
  sb = UOp.range(4, 2, AxisType.UPCAST)
  lane = UOp.range(32, 3, AxisType.UPCAST)
  block = (wg*256 + tid)*4 + sb
  row, col = block//96, (block%96)*32 + lane
  valid = (row < rows) & (col < 2880)
  h = h.reshape(rows*5760)
  gate = h.index((row*5760 + col*2).valid(valid)).load().cast(dtypes.float32)
  up = h.index((row*5760 + col*2 + 1).valid(valid)).load().cast(dtypes.float32)
  glu, linear = gate.minimum(LIMIT), up.maximum(-LIMIT).minimum(LIMIT)
  sig = (1.0 + (glu * (ALPHA * -LOG2E)).exp2()).reciprocal()
  act = glu * sig * (linear + 1.0)
  amax = (act < 0.0).where(-act, act).reduce(lane, arg=Ops.MAX)
  exponent = (amax.maximum(1e-38).log2().floor() + 127.0).maximum(0.0).minimum(254.0)
  scaled = (act * (127.0 - exponent).exp2()).maximum(-448.0).minimum(448.0)
  store = q.reshape(rows*3072).index((block*32 + lane).valid(row < rows)).store(scaled.cast(q.dtype)).end(lane)
  return e8.reshape(rows*96).after(store).index(block.valid(row < rows)).store(exponent.cast(dtypes.uint8)).end(sb, tid, wg).sink(
    arg=KernelInfo("gptoss_swiglu_quantize", opts_to_apply=()))

@functools.cache
def _swiglu_backward(out:UOp, h:UOp, gradient:UOp, e8:UOp) -> UOp:
  rows = h.shape[0]
  wg = UOp.range((rows*2880 + 256*8 - 1)//(256*8), 0, AxisType.GLOBAL)
  tid = UOp.range(256, 1, AxisType.LOCAL)
  lane = UOp.range(8, 2, AxisType.UPCAST)
  idx = (wg*256 + tid)*8 + lane
  valid = idx < rows*2880
  row, col = idx//2880, idx%2880
  h, out = h.reshape(rows*5760), out.reshape(rows*5760)
  gate = h.index((idx*2).valid(valid)).load().cast(dtypes.float32)
  up = h.index((idx*2 + 1).valid(valid)).load().cast(dtypes.float32)
  scale = (127.0 - e8.reshape(rows*96).index((row*96 + col//32).valid(valid)).load().cast(dtypes.float32)).exp2()
  dy = gradient.reshape(rows*3072).index((row*3072 + col).valid(valid)).load().cast(dtypes.float32) * scale
  glu, linear = gate.minimum(LIMIT), up.maximum(-LIMIT).minimum(LIMIT)
  sig = (1.0 + (glu * (ALPHA * -LOG2E)).exp2()).reciprocal()
  sprime = sig * (1.0 + ALPHA * glu * (1.0 - sig))
  dgate, dup = dy * sprime * (linear + 1.0), dy * (glu * sig)
  gate_ok = (gate < LIMIT).where(1.0, 0.0)
  up_ok = (up > -LIMIT).where(1.0, 0.0) * (up < LIMIT).where(1.0, 0.0)
  store = out.index((idx*2).valid(valid)).store((dgate * gate_ok).cast(out.dtype))
  return out.after(store).index((idx*2 + 1).valid(valid)).store((dup * up_ok).cast(out.dtype)).end(lane, tid, wg).sink(
    arg=KernelInfo("gptoss_swiglu_backward", opts_to_apply=()))

def _swiglu_quantize_backward(gradient:UOp, kernel:UOp) -> tuple:
  _, e8, h = kernel.src[1:4]
  out = alloc_like(h.shape, h.dtype, h.device, h.axis)
  out = Tensor.custom_kernel(out, Tensor(h), Tensor(gradient).cast(dtypes.bfloat16), Tensor(e8.after(kernel)), fxn=_swiglu_backward)[0]
  return None, None, out.uop

def fused_swiglu_quantize(h:Tensor) -> tuple[Tensor, Tensor]:
  assert h.ndim == 2 and h.shape[1] == 5760 and h.dtype == dtypes.bfloat16
  assert h.uop.shard_shape[0] % 2 == 0
  rows = h.shape[0]
  q = alloc_like((rows, 3072), dtypes.fp8e4m3, h.device, h.uop.axis)
  e8 = alloc_like((rows, 96), dtypes.uint8, h.device, h.uop.axis)
  q, e8, _ = Tensor.custom_kernel(q, e8, h, fxn=_swiglu_quantize, grad_fxn=_swiglu_quantize_backward)
  return q, e8
