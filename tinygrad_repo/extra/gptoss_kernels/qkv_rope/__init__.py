import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from extra.llama_kernels import alloc_like, compile_hip

@functools.cache
def _forward(q:UOp, k:UOp, v:UOp, xqkv:UOp, freqs:UOp) -> UOp:
  assert xqkv.shape == (2, 8192, 5120) and q.shape == (2, 8192, 64, 64) and k.shape == v.shape == (2, 8192, 8, 64)
  sink = UOp.sink(q.base, k.base, v.base, xqkv.base, freqs.base,
                  UOp.special(256, "lidx0"), UOp.special(2, "gidx0"), UOp.special(8192, "gidx1"),
                  arg=KernelInfo("fused_qkv_rope_forward_gptoss_packed"))
  src = (pathlib.Path(__file__).parent/"forward.cpp").read_text()
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=compile_hip(src, []))))

@functools.cache
def _backward_kernel(out:UOp, dq:UOp, dk:UOp, dv:UOp, freqs:UOp) -> UOp:
  assert out.shape == (2, 8192, 5120) and dq.shape == (2, 64, 8192, 64) and dk.shape == dv.shape == (16, 8192, 8, 64)
  sink = UOp.sink(out.base, dq.base, dk.base, dv.base, freqs.base,
                  UOp.special(256, "lidx0"), UOp.special(2, "gidx0"), UOp.special(128, "gidx1"), UOp.special(80, "gidx2"),
                  arg=KernelInfo("fused_qkv_rope_backward_gptoss"))
  src = (pathlib.Path(__file__).parent/"backward.cpp").read_text()
  include = pathlib.Path(__file__).parents[2]/"thunder"/"amd"/"include"
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=compile_hip(src, [f"-I{include}", "-DKITTENS_CDNA4", "-DHIP_ENABLE_WARP_SYNC_BUILTINS"]))))

def _backward(dq:UOp, dk:UOp, dv:UOp, call:UOp) -> tuple:
  from extra.thunder.amd.fa import _fa_native_grads
  # Consume FA's shuffled dQ and unreduced dK/dV directly, without materializing their logical views.
  native = _fa_native_grads(dq, dk, dv)
  assert native is not None, "GPT-OSS QKV/RoPE backward requires native Flash Attention gradients"
  xqkv, freqs = (Tensor(u) for u in call.src[4:6])
  out = alloc_like(xqkv.shape, xqkv.dtype, xqkv.device, xqkv.uop.axis)
  out, *_ = Tensor.custom_kernel(out, *(Tensor(u) for u in native), freqs, fxn=_backward_kernel)
  return None, None, None, out.uop, None

def fused_qkv_rope(xqkv:Tensor, freqs:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  batch, seq, hidden = xqkv.shape
  assert hidden == 5120 and freqs.shape == (1, seq, 1, 32, 2) and xqkv.dtype == freqs.dtype == dtypes.bfloat16
  q, k, v = (alloc_like((batch, seq, heads, 64), dtypes.bfloat16, xqkv.device, xqkv.uop.axis) for heads in (64, 8, 8))
  q, k, v, *_ = Tensor.custom_kernel(q, k, v, xqkv, freqs, fxn=_forward, grad_fxn=_backward)
  return q, k, v
