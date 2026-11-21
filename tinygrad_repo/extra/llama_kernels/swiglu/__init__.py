import functools, math, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.llama_kernels import alloc_like, compile_hip

LOG2E = 1.4426950408889634

@functools.cache
def _custom_swiglu(out:UOp, x_w13:UOp) -> UOp:
  rows, hidden = math.prod(x_w13.shape[:-1]), x_w13.shape[-1]//2
  n_elems = rows * hidden
  out, x_w13 = out.reshape(n_elems), x_w13.reshape(rows, 2*hidden)
  i = UOp.range(n_elems, 0)
  row, col = i // hidden, i % hidden
  act, gate = x_w13[row, col].cast(dtypes.float), x_w13[row, hidden+col].cast(dtypes.float)
  sigmoid = (1.0 + (-LOG2E * act).exp2()).reciprocal()
  store = out[i].store((act * sigmoid * gate).cast(out.dtype))
  return store.end(i).sink(arg=KernelInfo(f"swiglu_fwd_{n_elems}", estimates=Estimates(ops=5*n_elems, mem=6*n_elems)))

@functools.cache
def _custom_swiglu_mxfp4(out:UOp, row_fp4:UOp, row_scale:UOp, col_fp4:UOp, col_scale:UOp, x_w13:UOp) -> UOp:
  M, N = math.prod(out.shape[:-1]), out.shape[-1]
  name = f"swiglu_fwd_mxfp4_{M}_{N}"
  threads, gidx0, gidx1 = UOp.special(256, "lidx0"), UOp.special(M//128, "gidx0"), UOp.special(N//32, "gidx1")
  sink = UOp.sink(out.base, row_fp4.base, row_scale.base, col_fp4.base, col_scale.base, x_w13.base,
                  threads, gidx0, gidx1, arg=KernelInfo(name))
  src = (pathlib.Path(__file__).parent/"swiglu_fwd_mxfp4.cpp").read_text()
  inc = pathlib.Path(__file__).parent.parent/"quantize_mxfp4"
  kittens = pathlib.Path(__file__).parents[2]/"thunder"/"amd"/"include"
  lib = compile_hip(src, [f"-I{inc}", f"-I{kittens}", "-DKITTENS_CDNA4", f"-DKERNEL_NAME={name}", f"-DM_DIM={M}", f"-DN_DIM={N}"])
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

@functools.cache
def _custom_swiglu_bwd(grad_out:UOp, x_w13:UOp, grad_act:UOp) -> UOp:
  rows, hidden = math.prod(x_w13.shape[:-1]), x_w13.shape[-1]//2
  n_elems = rows * hidden
  grad_out, x_w13, grad_act = grad_out.reshape(rows, 2*hidden), x_w13.reshape(rows, 2*hidden), grad_act.reshape(n_elems)
  i = UOp.range(n_elems, 0)
  row, col = i // hidden, i % hidden
  act, gate = x_w13[row, col].cast(dtypes.float), x_w13[row, hidden+col].cast(dtypes.float)
  grad = grad_act[i].cast(dtypes.float)
  sigmoid = (1.0 + (-LOG2E * act).exp2()).reciprocal()
  silu = act * sigmoid
  dact = grad_out[row, col].store((grad * (sigmoid + silu * (1.0 - sigmoid)) * gate).cast(grad_out.dtype))
  dgate = grad_out.after(dact)[row, hidden+col].store((grad * silu).cast(grad_out.dtype))
  return dgate.end(i).sink(arg=KernelInfo(f"swiglu_bwd_{n_elems}", estimates=Estimates(ops=10*n_elems, mem=10*n_elems)))

@functools.cache
def _custom_swiglu_bwd_mxfp4(grad_out:UOp, row_fp4:UOp, row_scale:UOp, col_fp4:UOp, col_scale:UOp,
                              x_w13:UOp, grad_act:UOp) -> UOp:
  M, N = math.prod(x_w13.shape[:-1]), x_w13.shape[-1]
  name = f"swiglu_bwd_mxfp4_{M}_{N}"
  threads, gidx0, gidx1 = UOp.special(512, "lidx0"), UOp.special(M//256, "gidx0"), UOp.special(N//64, "gidx1")
  sink = UOp.sink(grad_out.base, row_fp4.base, row_scale.base, col_fp4.base, col_scale.base,
                  x_w13.base, grad_act.base, threads, gidx0, gidx1, arg=KernelInfo(name))
  src = (pathlib.Path(__file__).parent/"swiglu_bwd_mxfp4.cpp").read_text()
  inc = pathlib.Path(__file__).parent.parent/"quantize_mxfp4"
  kittens = pathlib.Path(__file__).parents[2]/"thunder"/"amd"/"include"
  lib = compile_hip(src, [f"-I{inc}", f"-I{kittens}", "-DKITTENS_CDNA4", f"-DKERNEL_NAME={name}", f"-DM_DIM={M}", f"-DN_DIM={N}"])
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def _swiglu_bwd(gradient:UOp, kernel:UOp, *, prequantize_mxfp4:bool=False):
  x_w13 = kernel.src[-1]
  axis = x_w13.axis if isinstance(x_w13.device, tuple) else None
  grad_out = alloc_like(x_w13.shape, dtypes.bfloat16, x_w13.device, axis)
  M, N = math.prod(x_w13.shape[:-1]), x_w13.shape[-1]
  if prequantize_mxfp4:
    assert M % 256 == 0 and N % 256 == 0, f"MXFP4 SwiGLU gradient requires multiples of 256, got {(M, N)}"
    from extra.llama_kernels.quantize_mxfp4 import alloc_mxfp4_outputs
    quant = alloc_mxfp4_outputs(grad_out, flatten_row=True)
    grad_out, *_ = Tensor.custom_kernel(grad_out, *quant, Tensor(x_w13, device=x_w13.device),
                                        Tensor(gradient, device=x_w13.device), fxn=_custom_swiglu_bwd_mxfp4)
  else:
    grad_out, *_ = Tensor.custom_kernel(grad_out, Tensor(x_w13, device=x_w13.device), Tensor(gradient, device=x_w13.device),
                                        fxn=_custom_swiglu_bwd)
  return (None,)*(len(kernel.src)-2) + (grad_out.uop,)

def swiglu(x_w13:Tensor, *, prequantize_grad_mxfp4:bool=False) -> Tensor:
  assert x_w13.dtype == dtypes.bfloat16 and x_w13.ndim >= 2 and x_w13.shape[-1] % 32 == 0
  *prefix, two_k = x_w13.shape
  axis = x_w13.uop.axis if isinstance(x_w13.device, tuple) else None
  out = alloc_like((*prefix, two_k//2), dtypes.bfloat16, x_w13.device, axis)
  grad_fxn = functools.partial(_swiglu_bwd, prequantize_mxfp4=prequantize_grad_mxfp4)
  return Tensor.custom_kernel(out, x_w13, fxn=_custom_swiglu, grad_fxn=grad_fxn)[0]

def swiglu_mxfp4(x_w13:Tensor) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
  assert x_w13.dtype == dtypes.bfloat16 and x_w13.ndim >= 2 and x_w13.shape[-1] % 512 == 0
  *prefix, two_k = x_w13.shape
  axis = x_w13.uop.axis if isinstance(x_w13.device, tuple) else None
  out = alloc_like((*prefix, two_k//2), dtypes.bfloat16, x_w13.device, axis)
  from extra.llama_kernels.quantize_mxfp4 import alloc_mxfp4_outputs
  quant = alloc_mxfp4_outputs(out)
  grad_fxn = functools.partial(_swiglu_bwd, prequantize_mxfp4=True)
  ret = Tensor.custom_kernel(out, *quant, x_w13, fxn=_custom_swiglu_mxfp4, grad_fxn=grad_fxn)
  return ret[0], tuple(ret[1:5])
