import functools, math, pathlib
from tinygrad import Tensor, dtypes, function
from tinygrad.helpers import ALLREDUCE_CAST
from tinygrad.runtime.support.compiler_amd import HIPCompiler
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from extra.llama_kernels import alloc_like, compile_hip

@functools.cache
def _residual_kernel(out:UOp, x:UOp, proj:UOp, bias:UOp, moe:UOp) -> UOp:
  rows = math.prod(x.shape[:-1])
  assert out.shape == x.shape == moe.shape and x.shape[-1] == 2880
  assert proj.shape == (rows, 3072) and bias.shape == (2880,)
  assert all(t.dtype == dtypes.bfloat16 for t in (out, x, proj, bias, moe))
  sink = UOp.sink(out.base, x.base, proj.base, bias.base, moe.base,
                  UOp.special(256, "lidx0"), UOp.special(rows, "gidx0"), arg=KernelInfo("gptoss_residual_join_vec"))
  src = (pathlib.Path(__file__).parent/"residual.cpp").read_text()
  lib = compile_hip(src, [f"-DROWS={rows}", "-DREAL_D=2880", "-DPAD_D=3072", "-fno-fast-math"])
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def _residual_backward(gradient:UOp, call:UOp) -> tuple:
  return gradient, None, None, None, gradient

@function(grad_fxn=_residual_backward)
def residual_join(h:Tensor, x:Tensor, proj:Tensor, bias:Tensor, moe:Tensor) -> Tensor:
  out = alloc_like(x.shape, x.dtype, x.device, x.uop.axis)
  return Tensor.custom_kernel(out, x, proj, bias, moe, fxn=_residual_kernel)[0]

@functools.cache
def _wo_bias_kernel(out:UOp, inp:UOp, *, stage:str) -> UOp:
  assert stage in ("partial", "final")
  partial = stage == "partial"
  assert out.shape == ((16, 2880) if partial else (2880,)) and inp.shape == ((16384, 2880) if partial else (16, 2880))
  assert out.dtype == (dtypes.float32 if partial else dtypes.bfloat16) and inp.dtype == (dtypes.bfloat16 if partial else dtypes.float32)
  sink = UOp.sink(out.base, inp.base, UOp.special(64, "lidx0"), UOp.special(45, "gidx0"),
                  *((UOp.special(16, "gidx1"),) if partial else ()), arg=KernelInfo(f"gptoss_wo_bias_{stage}"))
  src = (pathlib.Path(__file__).parent/f"wo_bias_{stage}.cpp").read_text()
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=HIPCompiler("gfx950").compile_cached(src))))

def wo_bias_gradient(g:Tensor) -> Tensor:
  groups = len(g.device) if isinstance(g.device, tuple) else 1
  assert groups == 1 or (g.uop.axis == 0 and ALLREDUCE_CAST), "WO bias sum requires the baseline BF16 aggregation"
  partial = alloc_like((groups*16, 2880), dtypes.float32, g.device, g.uop.axis)
  local = alloc_like((2880,), dtypes.bfloat16, g.device)
  partial = Tensor.custom_kernel(partial, g.reshape(-1, 2880), fxn=functools.partial(_wo_bias_kernel, stage="partial"))[0]
  local = Tensor.custom_kernel(local, partial, fxn=functools.partial(_wo_bias_kernel, stage="final"))[0]
  return Tensor(local.uop.allreduce(Ops.ADD, g.device)) if groups > 1 else local

def _wo_bias_backward(gradient:UOp, call:UOp) -> tuple:
  return gradient, wo_bias_gradient(Tensor(gradient)).uop

@function(grad_fxn=_wo_bias_backward)
def wo_bias_add(x:Tensor, bias:Tensor) -> Tensor:
  return x + bias
