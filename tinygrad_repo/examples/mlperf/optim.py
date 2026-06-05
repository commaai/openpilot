from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn.optim import Optimizer
from tinygrad.helpers import FUSE_OPTIM, getenv
from tinygrad.uop.ops import UOp, Ops

STOCHASTIC_ROUND = getenv("STOCHASTIC_ROUND", 0)
MASTER_WEIGHTS = getenv("MASTER_WEIGHTS", 0)

def stochastic_round_bf16(x:Tensor) -> Tensor:
  bits = x.bitcast(dtypes.uint32)
  if isinstance(x.device, tuple):
    shape = x.uop.shard_shape if x.uop.axis is not None else x.shape
    noise = Tensor(UOp(Ops.MSTACK, dtypes.default_float, tuple(Tensor.rand(*shape, device=d).uop for d in x.device)))
  else:
    noise = x.rand_like()
  noise = (noise * 0xFFFF).cast(dtypes.uint32)
  return ((bits + noise) & 0xFFFF0000).bitcast(dtypes.float32).cast(dtypes.bfloat16)

class GradAccClipAdamW(Optimizer):
  def __init__(self, params:list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, weight_decay=0.0, grad_acc=1, clip_norm=1.0, device=None, fused=FUSE_OPTIM):
    super().__init__(params, lr, device, fused)
    self.b1, self.b2, self.eps, self.wd = b1, b2, eps, weight_decay
    self.b1_t, self.b2_t = (Tensor.ones((1,), dtype=dtypes.float32, device=self.device) for _ in [b1, b2])
    self.m = self._new_optim_param()
    self.v = self._new_optim_param()
    self.grad_acc, self.clip_norm = grad_acc, clip_norm
    self.master_params:list[Tensor]|None = [p.float().contiguous() for p in self.params] if MASTER_WEIGHTS and self.params[0].dtype != dtypes.float32 else None

  def fstep(self, grads:list[Tensor]):
    if self.fused:
      out, extra = self._step([], grads)
      updates = [out[0][self.pos_params[i]:self.pos_params[i+1]].reshape(tt.shape) for i, tt in enumerate(self.params)]
    else:
      updates, extra = self._step([], grads)
    for i, tt in enumerate(self.params): tt.assign(self._apply_update(tt, updates[i], self.master_params[i] if self.master_params else None))
    # collect inv_scale tensors attached to fp8 params (set by _apply_update)
    fp8_inv_scales = [tt._inv_scale for tt in self.params if hasattr(tt, '_inv_scale')]
    to_realize = extra+self.params+self.buffers+(self.master_params or [])+fp8_inv_scales

    Tensor.realize(*to_realize)
    return extra[-1]

  def _step(self, params:list[Tensor], grads:list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    grads = list(grads)

    for i in range(len(grads)):
      if grads[i].device != self.m[i].device: grads[i] = grads[i].to(self.m[i].device)

    if self.fused:
      grads[0].assign(grads[0] / self.grad_acc)
      total_norm = grads[0].float().square().sum().sqrt()
      grads[0].assign((grads[0] * (self.clip_norm / (total_norm + 1e-6)).clamp(max_=1.0)).cast(grads[0].dtype))
    else:
      for i in range(len(grads)):
        grads[i].assign(grads[i] / self.grad_acc)
      total_norm = Tensor.stack(*[g.float().square().sum() for g in grads]).sum().sqrt().contiguous()
      for i in range(len(grads)):
        grads[i].assign((grads[i] * (self.clip_norm / (total_norm + 1e-6)).clamp(max_=1.0)).cast(grads[i].dtype))

    ret = []
    self.b1_t *= self.b1
    self.b2_t *= self.b2
    for i, g in enumerate(grads):
      m_new = self.b1 * self.m[i].float() + (1.0 - self.b1) * g.float()
      v_new = self.b2 * self.v[i].float() + (1.0 - self.b2) * (g.float() * g.float())
      self.m[i].assign(m_new.cast(self.m[i].dtype))
      self.v[i].assign(v_new.cast(self.v[i].dtype))
      m_hat = m_new / (1.0 - self.b1_t)
      v_hat = v_new / (1.0 - self.b2_t)
      up = m_hat / (v_hat.sqrt() + self.eps)
      ret.append(self.lr * up)
    return ret, [self.b1_t, self.b2_t] + self.m + self.v + [total_norm]

  def _apply_update(self, t:Tensor, up:Tensor, master:Tensor|None=None) -> Tensor:
    w = master if master is not None else t
    wd = self.wd if t.ndim >= 3 else 0.0
    up = up.float().shard_like(w) + self.lr.to(w.device) * wd * w.detach()
    new_w = w.detach() - up
    if master is not None: master.assign(new_w)
    if STOCHASTIC_ROUND and t.dtype == dtypes.bfloat16: return stochastic_round_bf16(new_w)
    if t.dtype in dtypes.fp8s:
      from examples.mlperf.models.flat_llama import FP8_MAX
      amax = new_w.float().abs().max(axis=tuple(range(1, new_w.ndim))).detach()  # per-layer amax for (n_layers, out, in)
      scale = FP8_MAX / (amax + 1e-8)
      fp8_w = (new_w * scale.reshape(-1, *([1]*(new_w.ndim-1)))).clamp(-FP8_MAX, FP8_MAX).cast(t.dtype)
      if hasattr(t, '_inv_scale'):
        t._inv_scale.assign(((amax + 1e-8) / FP8_MAX).cast(t._inv_scale.dtype))
      return fp8_w
    return new_w.cast(t.dtype)
