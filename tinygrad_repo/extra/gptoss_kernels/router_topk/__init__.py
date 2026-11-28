import functools, math, pathlib
from tinygrad import Tensor, dtypes, function
from tinygrad.helpers import getenv
from tinygrad.runtime.support.compiler_amd import HIPCompiler, HIPCCCompiler
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from extra.llama_kernels import alloc_like, compile_hip

@functools.cache
def _router_topk_fwd(weights:UOp, indices:UOp, logits:UOp) -> UOp:
  tokens = math.prod(logits.shape[:-1])
  sink = UOp.sink(weights.base, indices.base, logits.base,
                  UOp.special(256, "lidx0"), UOp.special((tokens+255)//256, "gidx0"),
                  arg=KernelInfo(f"moe_router_topk_{tokens}_32_4"))
  src = (pathlib.Path(__file__).parent/"forward.cpp").read_text()
  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                  UOp(Ops.BINARY, arg=compile_hip(src, [f"-DTOKENS={tokens}"]))))

@functools.cache
def _router_topk_bwd_kernel(grad_logits:UOp, bias_partials:UOp, grad_weights:UOp, weights:UOp, indices:UOp) -> UOp:
  tokens = math.prod(grad_logits.shape[:-1])
  sink = UOp.sink(grad_logits.base, bias_partials.base, grad_weights.base, weights.base, indices.base,
                  UOp.special(256, "lidx0"), UOp.special((tokens+255)//256, "gidx0"),
                  arg=KernelInfo(f"moe_router_topk_bwd_{tokens}_32_4"))
  src = (pathlib.Path(__file__).parent/"backward.cpp").read_text()
  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                  UOp(Ops.BINARY, arg=compile_hip(src, [f"-DTOKENS={tokens}"]))))

def router_topk_backward(gradient:Tensor, weights:Tensor, indices:Tensor) -> tuple[Tensor, Tensor]:
  groups, tokens, _ = weights.shape
  grad_logits = alloc_like((groups, tokens, 32), dtypes.float32, weights.device, weights.uop.axis)
  bias_partials = alloc_like((groups, (tokens+255)//256, 32), dtypes.float32, weights.device, weights.uop.axis)
  grad_logits, bias_partials, *_ = Tensor.custom_kernel(grad_logits, bias_partials, gradient.contiguous(), weights, indices,
                                                        fxn=_router_topk_bwd_kernel)
  return grad_logits, bias_partials

def _router_topk_bwd(gradient:UOp, kernel:UOp) -> tuple:
  weights_u, indices_u = kernel.src[1:3]
  grad_logits, _ = router_topk_backward(Tensor(gradient), Tensor(weights_u.after(kernel)), Tensor(indices_u.after(kernel)))
  return None, None, grad_logits.uop

@functools.cache
def _router_weight_bwd_kernel(out:UOp, x:UOp, gradient:UOp) -> UOp:
  rows, hidden = math.prod(x.shape[:-1]), x.shape[-1]
  assert rows % 512 == 0 and hidden % 16 == 0
  sink = UOp.sink(out.base, x.base, gradient.base, UOp.special(1024, "lidx0"), UOp.special(hidden//16, "gidx0"),
                  arg=KernelInfo(f"moe_router_fp32_wgrad_{rows}_{hidden}_32"))
  src = (pathlib.Path(__file__).parent/"weight_backward.cpp").read_text()
  include = pathlib.Path(__file__).parents[2]/"thunder"/"amd"/"include"
  lib = HIPCCCompiler("gfx950", [f"-I{include}", "-std=c++20", "-DKITTENS_CDNA4", "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
                                 f"-DROUTER_M={rows}", f"-DROUTER_K={hidden}"]).compile_cached(src)
  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def router_weight_gradient(x:Tensor, gradient:Tensor) -> Tensor:
  assert x.dtype == dtypes.bfloat16 and gradient.dtype == dtypes.float32
  groups = len(x.device) if isinstance(x.device, tuple) and x.uop.axis is not None else 1
  partial = alloc_like((groups, 32, x.shape[-1]), dtypes.bfloat16, x.device, x.uop.axis)
  partial, *_ = Tensor.custom_kernel(partial, x, gradient, fxn=_router_weight_bwd_kernel)
  return partial.sum(0, dtype=x.dtype)

def _router_bwd(gradient:UOp, call:UOp) -> tuple:
  x, weight, bias = (Tensor(u) for u in call.src[1:4])
  weights, indices = (Tensor(u) for u in call.unbound_outputs)
  grad_logits, bias_partials = router_topk_backward(Tensor(gradient), weights, indices)
  grad_x, = (x.float() @ weight.float().T).gradient(x, gradient=grad_logits.reshape(*x.shape[:-1], 32))
  grad_weight = router_weight_gradient(x, grad_logits)
  return grad_x.uop, grad_weight.uop, bias_partials.sum(1).cast(bias.dtype).sum(0, dtype=bias.dtype).uop

@function(grad_fxn=_router_bwd)
def fused_router(x:Tensor, weight:Tensor, bias:Tensor) -> tuple[Tensor, Tensor]:
  from extra.gemm.moe_routing import router_mfma
  logits = router_mfma(x, weight, bias) if getenv("ROUTER_MFMA", 0) else x.float() @ weight.float().T + bias.float()
  groups = len(x.device) if isinstance(x.device, tuple) else 1
  return fused_router_topk(logits.reshape(groups, -1, 32))

@functools.cache
def _router_input_bwd_kernel(out:UOp, weight:UOp, gradient:UOp, residual:UOp, scales:UOp) -> UOp:
  assert math.prod(out.shape[:-1]) == 16384 and out.shape[-1] == 2880 and weight.shape == (32, 2880)
  assert math.prod(gradient.shape[:-1]) == 16384 and gradient.shape[-1] == 32
  assert residual.shape == (16384, 3072) and scales.shape == (16384, 96)
  sink = UOp.sink(out.base, weight.base, gradient.base, residual.base, scales.base,
                  UOp.special(16, "lidx0"), UOp.special(4, "lidx1"), UOp.special(9, "gidx0"), UOp.special(2048, "gidx1"),
                  arg=KernelInfo("moe_router_dgrad_fused"))
  src = (pathlib.Path(__file__).parent/"input_backward.cpp").read_text()
  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                  UOp(Ops.BINARY, arg=HIPCompiler("gfx950").compile_cached(src))))

def _router_quantize_bwd(gradient:UOp, quantized_gradient:UOp, *, call:UOp) -> tuple:
  x, weight, bias = (Tensor(u) for u in call.src[1:4])
  weights, indices, _, scales = (Tensor(u) for u in call.unbound_outputs)
  grad_logits, bias_partials = router_topk_backward(Tensor(gradient), weights, indices)
  grad_x = alloc_like(x.shape, x.dtype, x.device, x.uop.axis)
  grad_x, *_ = Tensor.custom_kernel(grad_x, weight, grad_logits, Tensor(quantized_gradient), scales, fxn=_router_input_bwd_kernel)
  return grad_x.uop, router_weight_gradient(x, grad_logits).uop, bias_partials.sum(1).cast(bias.dtype).sum(0, dtype=bias.dtype).uop

@function(grad_fxn=_router_quantize_bwd)
def fused_router_quantize(x:Tensor, weight:Tensor, bias:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  from extra.gptoss_kernels.quantize_mxfp8 import quantize_mxfp8_fused_qe8
  weights, indices = fused_router(x, weight, bias)
  quantized, scales = quantize_mxfp8_fused_qe8(x.reshape(-1, 2880).pad(((0, 0), (0, 192))))
  return weights, indices, quantized, scales

def fused_router_topk(logits:Tensor) -> tuple[Tensor, Tensor]:
  assert logits.ndim == 3 and logits.shape[-1] == 32 and logits.dtype == dtypes.float32
  axis = logits.uop.axis
  weights = alloc_like((*logits.shape[:-1], 4), dtypes.float32, logits.device, axis)
  indices = alloc_like(weights.shape, dtypes.int32, logits.device, axis)
  weights, indices, *_ = Tensor.custom_kernel(weights, indices, logits, fxn=_router_topk_fwd, grad_fxn=_router_topk_bwd)
  return weights, indices
