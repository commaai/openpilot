import functools
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from extra.llama_kernels import FP8_MAX, THREADS_PER_WG, alloc_like

BLK = 32

@functools.cache
def _custom_quantize_mxfp8_qe8(fp8_out:UOp, e8_out:UOp, x:UOp) -> UOp:
  rows, K = x.shape
  n_elems = rows * K
  # One 32-value MX block per thread; packed scale indices are produced separately by mx_pack.
  n_blocks = n_elems // BLK
  assert n_blocks % THREADS_PER_WG == 0, f"{n_blocks=} must divide over {THREADS_PER_WG=}"
  x, fp8_out, e8_out = x.reshape(n_elems), fp8_out.reshape(n_elems), e8_out.reshape(n_blocks)
  wg = UOp.range(n_blocks // THREADS_PER_WG, 0, AxisType.GLOBAL)
  tid = UOp.range(THREADS_PER_WG, 1, AxisType.LOCAL)
  lane = UOp.range(BLK, 3, AxisType.UNROLL)
  block = wg * THREADS_PER_WG + tid
  idx = block * BLK + lane
  x_f = x[idx].cast(dtypes.float)
  abs_x = (x_f < 0.0).where(-x_f, x_f)
  blk_max = abs_x.reduce(lane, arg=Ops.MAX)
  e8f = (blk_max.maximum(1e-38).log2().floor() + 127.0).maximum(0.0).minimum(254.0)
  qscale = (127.0 - e8f).exp2()
  scaled = (x_f * qscale).maximum(-FP8_MAX).minimum(FP8_MAX)
  fp8_store = fp8_out[idx].store(scaled.cast(fp8_out.dtype)).end(lane)
  e8_store = e8_out.after(fp8_store)[block].store(e8f.cast(dtypes.uint8))
  return e8_store.end(tid, wg).sink(arg=KernelInfo(f"quantize_mxfp8_qe8_{n_elems}", opts_to_apply=()))

def _quantize_mxfp8_qe8_bwd(gradient:UOp, kernel:UOp):
  _, e8_out, x = kernel.src[1:]
  rows, K = x.shape
  e8 = Tensor(e8_out.after(kernel), device=x.device).reshape(rows, K // BLK)
  qscale = (127.0 - e8.cast(dtypes.float32)).exp2().reshape(rows, K // BLK, 1).expand(rows, K // BLK, BLK).reshape(rows, K)
  grad_x = (Tensor(gradient, device=x.device).float() * qscale).cast(dtypes.bfloat16)
  return None, None, grad_x.uop

def quantize_mxfp8_fused_qe8(x:Tensor) -> tuple[Tensor, Tensor]:
  assert x.dtype == dtypes.bfloat16 and x.ndim == 2 and x.shape[1] % BLK == 0
  from extra.gemm.cdna_asm_gemm import FP8_DTYPE
  rows, K = x.shape
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  fp8_out = alloc_like((rows, K), FP8_DTYPE, x.device, axis)
  e8_out = alloc_like((rows, K // BLK), dtypes.uint8, x.device, axis)
  fp8_out, e8_out, *_ = Tensor.custom_kernel(fp8_out, e8_out, x, fxn=_custom_quantize_mxfp8_qe8, grad_fxn=_quantize_mxfp8_qe8_bwd)
  return fp8_out, e8_out
