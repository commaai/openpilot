import functools
from tinygrad import Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import prod
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from extra.llama_kernels import FP8_MAX, NUM_WG, THREADS_PER_WG, alloc_like

@functools.cache
def _custom_quantize_fp8_with_amax(fp8_out:UOp, amax_out:UOp, x:UOp, amax_state:UOp, device=None) -> UOp:
  VEC = 8
  n_elems = prod(x.shape)
  assert n_elems % (NUM_WG * THREADS_PER_WG * VEC) == 0

  x = x.reshape(n_elems)
  fp8_out = fp8_out.reshape(n_elems)

  wg = UOp.range(NUM_WG, 0, AxisType.GLOBAL)
  tid = UOp.range(THREADS_PER_WG, 1, AxisType.LOCAL)
  it = UOp.range((n_elems // VEC) // (NUM_WG * THREADS_PER_WG), 2, AxisType.LOOP)
  lane = UOp.range(VEC, 3, AxisType.UNROLL)

  idx = (((it * NUM_WG + wg) * THREADS_PER_WG + tid) * VEC) + lane

  scale = FP8_MAX / (amax_state[0].cast(dtypes.float) + 1e-8)
  x_f = x[idx].cast(dtypes.float)
  abs_x = (x_f < 0.0).where(-x_f, x_f)
  scaled = (x_f * scale).maximum(-FP8_MAX).minimum(FP8_MAX)

  fp8_store = fp8_out[idx].store(scaled.cast(fp8_out.dtype)).end(lane)
  lane_max = abs_x.reduce(lane, arg=Ops.MAX)

  lmax = UOp.placeholder((1,), dtypes.float, slot=1, addrspace=AddrSpace.REG)
  lmax_init = lmax.after(wg, tid)[0].store(0.0)
  lmax_prev = lmax.after(lmax_init, it)[0]
  lmax_store = lmax.after(fp8_store)[0].store(lmax_prev.maximum(lane_max))
  lmax_val = lmax.after(lmax_store.end(it))[0]

  lds = UOp.placeholder((THREADS_PER_WG,), dtypes.float, slot=0, addrspace=AddrSpace.LOCAL)
  lds = lds.after(lds[tid].store(lmax_val).barrier())

  step = THREADS_PER_WG // 2
  while step:
    active = tid < step
    other = lds[(tid + step).valid(active)].load()
    lds = lds.after(lds[tid.valid(active)].store(lds[tid].maximum(other)).barrier())
    step //= 2

  device = device[0].split(":")[0] if isinstance(device, tuple) else device.split(":")[0]
  if device in {"AMD", "NULL"}: atomic_arg = "if ({2} > {3}) __hip_atomic_fetch_max((int*){0}, {1}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);"
  else: raise NotImplementedError(f"no atomic max for device {device}")
  amax_idx = amax_out.reshape((1,)).index(UOp.const(dtypes.weakint, 0))
  max_val = lds[0].load()
  atomic = UOp(Ops.CUSTOM, dtypes.void, (amax_idx, max_val.bitcast(dtypes.int32), max_val, amax_idx.load()), arg=atomic_arg)
  return atomic.end(tid, wg).sink(arg=KernelInfo(f"quantize_fp8_with_amax_{n_elems}", opts_to_apply=()))

@functools.cache
def _custom_quantize_fp8_scalar(fp8_out:UOp, x:UOp, amax_state:UOp) -> UOp:
  n_elems = prod(x.shape)
  i = UOp.range(n_elems, 0)

  x_f = x.reshape(n_elems)[i].cast(dtypes.float)
  scale = FP8_MAX / (amax_state[0].cast(dtypes.float) + 1e-8)
  store = fp8_out.reshape(n_elems)[i].store((x_f * scale).cast(fp8_out.dtype))

  return store.end(i).sink(arg=KernelInfo(f"quantize_fp8_scalar_{n_elems}"))

def _quantize_fp8_delayed_bwd(gradient:UOp, kernel:UOp):
  # NOTE: STE-equivalent backward — grad_x = grad_fp8 * scale, scale = FP8_MAX / amax_state.
  # `gradient` is bf16 grad w.r.t. fp8 output (asm_gemm bwd already applied x_scale).
  _, _, x, amax_state = kernel.src[1:]
  device = x.device
  scale = FP8_MAX / (Tensor(amax_state, device=device).float() + 1e-8)
  grad_x = (Tensor(gradient, device=device).float() * scale).cast(dtypes.bfloat16)
  return (None, None, grad_x.uop, None)

def quantize_fp8_delayed(x:Tensor, amax_state:Tensor, amax_out:Tensor, fp8_dtype=dtypes.fp8e4m3) -> tuple[Tensor, Tensor]:
  # NOTE: one-pass bf16 -> fp8 quantize with delayed scaling.
  # Fused kernel reads x once and writes fp8 + scalar amax via global atomic max.
  assert x.dtype == dtypes.bfloat16, f"expected bf16, got {x.dtype}"
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  fp8_out      = alloc_like(x.shape,  fp8_dtype,      x.device, axis)
  n_elems = prod(x.uop.shard_shape)
  assert n_elems % NUM_WG == 0, f"{n_elems=} must divide over {NUM_WG=}"
  fxn = functools.partial(_custom_quantize_fp8_with_amax, device=x.device)
  fp8_out, amax_out, *_ = Tensor.custom_kernel(fp8_out, amax_out, x, amax_state,
                                                fxn=fxn, grad_fxn=_quantize_fp8_delayed_bwd)
  inv_scale = (amax_state.float() + 1e-8) / FP8_MAX
  return fp8_out, inv_scale

def quantize_fp8_scalar(x:Tensor, amax_state:Tensor, fp8_dtype=dtypes.fp8e4m3) -> Tensor:
  # NOTE: pure one-pass bf16 -> fp8 quantize with delayed scalar scale. No amax computation.
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  fp8_out = alloc_like(x.shape, fp8_dtype, x.device, axis)
  fxn = _custom_quantize_fp8_scalar
  fp8_out, *_ = Tensor.custom_kernel(fp8_out, x, amax_state, fxn=fxn)
  return fp8_out
