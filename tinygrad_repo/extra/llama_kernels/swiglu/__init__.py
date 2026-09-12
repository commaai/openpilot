import functools, math
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, KernelInfo
from tinygrad.renderer import Estimates
from extra.llama_kernels import alloc_like

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

def _swiglu_bwd(gradient:UOp, kernel:UOp):
  _, x_w13 = kernel.src[1:]
  axis = x_w13.axis if isinstance(x_w13.device, tuple) else None
  grad_out = alloc_like(x_w13.shape, dtypes.bfloat16, x_w13.device, axis)
  grad_out, *_ = Tensor.custom_kernel(grad_out, Tensor(x_w13, device=x_w13.device), Tensor(gradient, device=x_w13.device),
                                      fxn=_custom_swiglu_bwd)
  return (None, grad_out.uop)

def swiglu(x_w13:Tensor) -> Tensor:
  assert x_w13.dtype == dtypes.bfloat16 and x_w13.ndim >= 2 and x_w13.shape[-1] % 32 == 0
  *prefix, two_k = x_w13.shape
  axis = x_w13.uop.axis if isinstance(x_w13.device, tuple) else None
  out = alloc_like((*prefix, two_k//2), dtypes.bfloat16, x_w13.device, axis)
  return Tensor.custom_kernel(out, x_w13, fxn=_custom_swiglu, grad_fxn=_swiglu_bwd)[0]
