from __future__ import annotations
import functools, math, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.gemm.cdna_asm_gemm import FP8_DTYPE
from extra.llama_kernels import NUM_WG, THREADS_PER_WG, alloc_like, alloc_local, compile_hip, dname_of

COOP_EPILOGUE = getenv("RMSNORM_MX_EP8")

def rmsnorm_mul_fwd(x_in:Tensor, weight:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  x = x_in.float()
  rrms = (x.square().mean(-1, keepdim=True) + eps).rsqrt()
  return ((x * rrms) * weight.float()).cast(x_in.dtype), rrms

@functools.cache
def _rmsnorm_mul_fwd_fxn(x_in_p, w_p, eps, device):
  return rmsnorm_mul_fwd(Tensor(x_in_p, device=device), Tensor(w_p, device=device), eps)

def _rmsnorm_mul_bwd(grad:UOp, call:UOp) -> tuple:
  x_u, weight_u = call.src[1:3]
  x, weight, rrms = Tensor(x_u), Tensor(weight_u), Tensor(call.unbound_outputs[1])
  g = Tensor(grad, device=x_u.device)
  assert x.dtype == weight.dtype == g.dtype == dtypes.bfloat16 and weight.shape == (x.shape[-1],)
  device, axis = x.device, (x.uop.axis if isinstance(x.device, tuple) else None)
  local_rows = math.prod(x.uop.shard_shape[:-1] if axis is not None else x.shape[:-1])
  n_partials = min(NUM_WG, local_rows)
  grad_x = alloc_like(x.shape, dtypes.bfloat16, device, axis)
  partial_shape = (n_partials * len(device), x.shape[-1]) if isinstance(device, tuple) and axis is not None else (n_partials, x.shape[-1])
  grad_weight_partial = alloc_like(partial_shape, dtypes.float32, device, 0 if isinstance(device, tuple) and axis is not None else None)
  grad_x, grad_weight_partial, *_ = Tensor.custom_kernel(grad_x, grad_weight_partial, g.contiguous(), x, weight, rrms,
                                                         fxn=functools.partial(_custom_rmsnorm_mul_bwd, dname=dname_of(device)))
  return grad_x.uop, grad_weight_partial.sum(0).cast(weight_u.dtype).uop

@functools.cache
def _custom_rmsnorm_mul_bwd(grad_x:UOp, grad_weight_partial:UOp, grad:UOp, x:UOp, weight:UOp, rrms:UOp, *, dname:str) -> UOp:
  rows, hidden = math.prod(x.shape[:-1]), x.shape[-1]
  n_partials = grad_weight_partial.shape[0]
  assert rows % (2 * n_partials) == 0, "RMSNorm backward requires complete row pairs per partial"
  assert grad.shape == x.shape == grad_x.shape and grad.dtype == x.dtype == grad_x.dtype == dtypes.bfloat16
  assert weight.shape == (hidden,) and weight.dtype == dtypes.bfloat16 and rrms.shape == (*x.shape[:-1], 1)
  assert rrms.dtype == grad_weight_partial.dtype == dtypes.float32 and grad_weight_partial.shape == (n_partials, hidden)
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(n_partials, "gidx0")
  sink = UOp.sink(grad_x.base, grad_weight_partial.base, grad.base, x.base, weight.base, rrms.base,
                  threads, workgroups,
                  arg=KernelInfo(f"rmsnorm_mul_bwd_{rows}_{hidden}_{n_partials}",
                                 estimates=Estimates(ops=10*rows*hidden, mem=rows*hidden*6+rows*4+n_partials*hidden*4+hidden*2)))
  src = (pathlib.Path(__file__).parent/"rmsnorm_mul_bwd.cpp").read_text()
  defines = [f"-DROWS={rows}", f"-DHIDDEN={hidden}", f"-DNUM_WG={n_partials}", f"-DTHREADS={THREADS_PER_WG}"]
  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

def rmsnorm_mul(x_in:Tensor, weight:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  fxn = _rmsnorm_mul_fwd_fxn(x_in.as_param(0).uop, weight.as_param(1).uop, eps, x_in.device)
  outs = UOp.call_with_outputs((fxn[0].uop, fxn[1].uop), x_in.uop, weight.uop, grad_fxn=_rmsnorm_mul_bwd)
  return Tensor(outs[0]), Tensor(outs[1])

@functools.cache
def _custom_rmsnorm_mul_quantize_mxfp8_fwd(q:UOp, e8:UOp, rrms:UOp, x:UOp, weight:UOp, *, dname:str, eps:float) -> UOp:
  *lead, hidden = x.shape
  rows, padded = math.prod(lead), q.shape[-1]
  num_wg = min(NUM_WG, rows)
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(num_wg, "gidx0")
  sink = UOp.sink(q.base, e8.base, rrms.base, x.base, weight.base, threads, workgroups,
                  arg=KernelInfo(f"rmsnorm_mul_quantize_mxfp8_{rows}_{hidden}_{padded}",
                                 estimates=Estimates(ops=8*rows*hidden, mem=rows*(hidden*2+padded+padded//32+4)+hidden*2)))
  src = (pathlib.Path(__file__).parent/"rmsnorm_mul_quantize_mxfp8.cpp").read_text()
  defines = [f"-DN_ELEMS={rows*hidden}", f"-DHIDDEN={hidden}", f"-DPADDED={padded}",
             f"-DNUM_WG={num_wg}", f"-DTHREADS_PER_WG={THREADS_PER_WG}", f"-DEPS_LITERAL={eps}f", f"-DCOOP_EPILOGUE={COOP_EPILOGUE}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=compile_hip(src, defines))))

@functools.cache
def _custom_rmsnorm_mul_quantize_mxfp8_bwd(grad_x:UOp, grad_weight_partial:UOp, grad_q:UOp, x:UOp, weight:UOp, e8:UOp, rrms:UOp,
                *, dname:str) -> UOp:
  *lead, hidden = x.shape
  rows, padded = math.prod(lead), grad_q.shape[-1]
  num_wg = min(NUM_WG, rows)
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(num_wg, "gidx0")
  sink = UOp.sink(grad_x.base, grad_weight_partial.base, grad_q.base, x.base, weight.base, e8.base, rrms.base,
                  threads, workgroups,
                  arg=KernelInfo(f"rmsnorm_mul_quantize_mxfp8_bwd_{rows}_{hidden}_{padded}",
                                 estimates=Estimates(ops=10*rows*hidden, mem=rows*(hidden*6+padded*2+padded//32+4)+num_wg*hidden*4)))
  src = (pathlib.Path(__file__).parent/"rmsnorm_mul_quantize_mxfp8_bwd.cpp").read_text()
  defines = [f"-DN_ELEMS={rows*hidden}", f"-DHIDDEN={hidden}", f"-DPADDED={padded}", f"-DNUM_WG={num_wg}", f"-DTHREADS_PER_WG={THREADS_PER_WG}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=compile_hip(src, defines))))

def _rmsnorm_mul_quantize_mxfp8_backward(gradient:UOp, kernel:UOp) -> tuple:
  _, e8_u, rrms_u, x_u, weight_u = kernel.src[1:]
  device = x_u.device
  axis = x_u.axis if isinstance(device, tuple) else None
  *lead, hidden = x_u.shape
  num_wg = min(NUM_WG, math.prod(lead))
  grad_x = alloc_like(x_u.shape, x_u.dtype, device, axis)
  grad_weight_partial = alloc_local((num_wg, hidden), dtypes.float32, device, axis)
  grad_q = Tensor(gradient, device=device).cast(dtypes.bfloat16).contiguous()
  grad_x, grad_weight_partial, *_ = Tensor.custom_kernel(
    grad_x, grad_weight_partial, grad_q, Tensor(x_u, device=device), Tensor(weight_u, device=device),
    Tensor(e8_u.after(kernel), device=device), Tensor(rrms_u.after(kernel), device=device),
    fxn=functools.partial(_custom_rmsnorm_mul_quantize_mxfp8_bwd, dname=dname_of(device)))
  grad_weight = grad_weight_partial.sum(0).cast(weight_u.dtype)
  return None, None, None, grad_x.uop, grad_weight.uop

def rmsnorm_mul_quantize_mxfp8(x:Tensor, weight:Tensor, eps:float, padded:int|None=None) -> tuple[Tensor, Tensor, Tensor]:
  """RMSNorm(x)*weight directly to rowwise MXFP8. Returns (q, e8, rrms), without a BF16 normalized round-trip."""
  assert x.dtype == weight.dtype == dtypes.bfloat16 and x.shape[-1] == weight.shape[0], f"{x.shape=} {weight.shape=}"
  hidden = x.shape[-1]
  padded = math.ceil(hidden / 256) * 256 if padded is None else padded
  assert padded >= hidden and padded % 256 == 0 and hidden % 32 == 0
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  q = alloc_like((*x.shape[:-1], padded), FP8_DTYPE, x.device, axis)
  e8 = alloc_like((*x.shape[:-1], padded // 32), dtypes.uint8, x.device, axis)
  rrms = alloc_like((*x.shape[:-1], 1), dtypes.float32, x.device, axis)
  q, e8, rrms, *_ = Tensor.custom_kernel(q, e8, rrms, x, weight,
                                         fxn=functools.partial(_custom_rmsnorm_mul_quantize_mxfp8_fwd, dname=dname_of(x.device), eps=eps),
                                         grad_fxn=_rmsnorm_mul_quantize_mxfp8_backward)
  return q, e8, rrms

@functools.cache
def _custom_fast_final_denom(denom:UOp, rrms:UOp, x:UOp, *, dname:str, eps:float) -> UOp:
  local_x_shape, local_denom_shape, local_rrms_shape = x.shard_shape, denom.shard_shape, rrms.shard_shape
  rows, hidden = math.prod(local_x_shape[:-1]), local_x_shape[-1]
  threads = 64
  sink = UOp.sink(denom.base, rrms.base, x.base,
                  UOp.special(threads, "lidx0"), UOp.special(rows // threads, "gidx0"),
                  arg=KernelInfo(f"fast_final_rmsnorm_denom_rrms_{rows}_{hidden}",
                                 estimates=Estimates(ops=2*rows*hidden, mem=2*rows*hidden+8*rows)))
  src = (pathlib.Path(__file__).parent/"fast_final_denom.cpp").read_text()
  defines = [f"-DROWS={rows}", f"-DHIDDEN={hidden}", f"-DTHREADS={threads}", f"-DEPS_LITERAL={eps}f", "-fno-finite-math-only"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=compile_hip(src, defines))))

def _fast_final_rmsnorm_fwd(x:Tensor, weight:Tensor, eps:float) -> tuple[Tensor, Tensor, Tensor]:
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  local_shape = x.uop.shard_shape if axis is not None else x.shape
  assert axis in (None, 0) and math.prod(local_shape[:-1]) == 16384 and local_shape[-1] == 2880, \
    f"unsupported GPT-OSS final RMS tensor ABI {x.shape}/{x.uop.shard_shape} axis={axis}"
  denom = alloc_like(x.shape[:-1], dtypes.float32, x.device, axis).clone()
  rrms = alloc_like(x.shape[:-1], dtypes.float32, x.device, axis).clone()
  denom, rrms, *_ = Tensor.custom_kernel(denom, rrms, x, fxn=functools.partial(
    _custom_fast_final_denom, dname=dname_of(x.device), eps=eps))
  x_normed = (x.float() / denom.unsqueeze(-1)).cast(x.dtype)
  return x_normed * weight, denom, rrms

@functools.cache
def _fast_final_rmsnorm_fwd_fxn(x_p, weight_p, eps, device):
  return _fast_final_rmsnorm_fwd(Tensor(x_p, device=device), Tensor(weight_p, device=device), eps)

def _fast_final_rmsnorm_bwd(gradient:UOp, call:UOp, *, eps:float) -> tuple:
  x, weight = Tensor(call.src[1]), Tensor(call.src[2])
  denom = Tensor(call.unbound_outputs[1])
  rrms = Tensor(call.unbound_outputs[2])
  grad = Tensor(gradient, device=x.device)
  xf = x.float()
  rrms_ref = (xf.square().mean(-1, keepdim=True) + eps).rsqrt()
  y_ref = (xf * rrms_ref).cast(x.dtype) * weight
  d_x, d_weight = y_ref.gradient(x, weight, gradient=grad)
  denom_saved = denom.unsqueeze(-1)
  rrms_from_denom = 1.0 / denom_saved
  rrms_saved = rrms.unsqueeze(-1)
  replace_dx = {rrms_ref.uop:rrms_from_denom.uop, rrms_ref.uop.src[0]:denom_saved.uop}
  replace_dw = {rrms_ref.uop:rrms_saved.uop}
  return d_x.uop.substitute(replace_dx, walk=True), d_weight.uop.substitute(replace_dw, walk=True)

def fast_final_rmsnorm(x:Tensor, weight:Tensor, eps:float) -> Tensor:
  """GPT-OSS final RMSNorm, with a specialized denominator pass for the training shape."""
  assert x.dtype == weight.dtype == dtypes.bfloat16 and x.shape[-1] == weight.shape[0] == 2880
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  local_shape = x.uop.shard_shape if axis is not None else x.shape
  if axis not in (None, 0) or math.prod(local_shape[:-1]) != 16384:
    xf = x.float()
    return (xf * (xf.square().mean(-1, keepdim=True) + eps).rsqrt()).cast(x.dtype) * weight
  fxn = _fast_final_rmsnorm_fwd_fxn(x.as_param(0).uop, weight.as_param(1).uop, eps, x.device)
  outputs = UOp.call_with_outputs((fxn[0].uop, fxn[1].uop, fxn[2].uop),
    x.uop, weight.uop, grad_fxn=functools.partial(_fast_final_rmsnorm_bwd, eps=eps))
  return Tensor(outputs[0])
