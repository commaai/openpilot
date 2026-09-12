from __future__ import annotations
import functools
from tinygrad import Tensor
from tinygrad.uop.ops import UOp

def rmsnorm_mul_fwd(x_in:Tensor, weight:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  x = x_in.float()
  rrms = (x.square().mean(-1, keepdim=True) + eps).rsqrt()
  return ((x * rrms) * weight.float()).cast(x_in.dtype), rrms

@functools.cache
def _rmsnorm_mul_fwd_fxn(x_in_p, w_p, eps, device):
  return rmsnorm_mul_fwd(Tensor(x_in_p, device=device), Tensor(w_p, device=device), eps)

def _rmsnorm_mul_bwd(grad:UOp, call:UOp) -> tuple:
  x = Tensor(call.src[1]).float(); weight = Tensor(call.src[2]).float()
  rrms = Tensor(call.gettuple(1))
  x_normed = x * rrms                                  # recompute unweighted normed (x is call.src[1])
  d_y = Tensor(grad).float()
  dxn = d_y * weight                                   # d/d(x_normed)
  d_x = rrms * (dxn - x_normed * (dxn * x_normed).mean(-1, keepdim=True))
  dw = d_y * x_normed
  d_weight = dw.sum(axis=tuple(range(dw.ndim - 1)))    # reduce batch/seq -> [dim]
  return (d_x.cast(call.src[1].dtype).uop, d_weight.cast(call.src[2].dtype).uop)

def rmsnorm_mul(x_in:Tensor, weight:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  fxn = _rmsnorm_mul_fwd_fxn(x_in.as_param(0).uop, weight.as_param(1).uop, eps, x_in.device)
  call = UOp.maketuple(fxn[0].uop, fxn[1].uop).call(x_in.uop, weight.uop, grad_fxn=_rmsnorm_mul_bwd)
  return Tensor(call.gettuple(0)), Tensor(call.gettuple(1))
