from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn.optim import Optimizer, OptimizerGroup
from tinygrad.helpers import FUSE_OPTIM, getenv
from tinygrad.uop.ops import UOp, Ops, AxisType

STOCHASTIC_ROUND = getenv("STOCHASTIC_ROUND", 0)
MASTER_WEIGHTS = getenv("MASTER_WEIGHTS", 0)
ZERO_OPTIM = getenv("ZERO_OPTIM", 0)
FP8_AMAX_MARGIN = getenv("FP8_AMAX_MARGIN", 1.1)
IMMEDIATE_SCALE = getenv("IMMEDIATE_SCALE", 0)
MXFP8 = getenv("MXFP8", 0)

def stochastic_round_bf16(x:Tensor) -> Tensor:
  bits = x.bitcast(dtypes.uint32)
  if isinstance(x.device, tuple):
    shape = x.uop.shard_shape if x.uop.axis is not None else x.shape
    noise = Tensor(UOp(Ops.MSTACK, dtypes.default_float, tuple(Tensor.rand(*shape, device=d).uop for d in x.device)))
  else:
    noise = x.rand_like()
  noise = (noise * 0xFFFF).cast(dtypes.uint32)
  return ((bits + noise) & 0xFFFF0000).bitcast(dtypes.float32).cast(dtypes.bfloat16)

def clip_grads(grads:list[Tensor], grad_acc, clip_norm) -> Tensor:
  for g in grads: g.assign(g / grad_acc)
  total_norm = Tensor.stack(*[g.float().square().sum() for g in grads]).sum().sqrt().contiguous()
  for g in grads: g.assign((g * (clip_norm / (total_norm + 1e-6)).clamp(max_=1.0)).cast(g.dtype))
  return total_norm

class GradAccClipAdamW(Optimizer):
  def __init__(self, params:list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, weight_decay=0.0, grad_acc=1, clip_norm=1.0, device=None, fused=FUSE_OPTIM):
    super().__init__(params, lr, device, fused)
    self.b1, self.b2, self.eps, self.wd = b1, b2, eps, weight_decay
    self.b1_t, self.b2_t = (Tensor.ones((1,), dtype=dtypes.float32, device=self.device) for _ in [b1, b2])
    self.zero = bool(ZERO_OPTIM) and isinstance(self.device, tuple) and not self.fused
    self.m = [self._zero_shard(x) for x in self._new_optim_param()]
    self.v = [self._zero_shard(x) for x in self._new_optim_param()]
    self.grad_acc, self.clip_norm = grad_acc, clip_norm
    if MASTER_WEIGHTS and self.params[0].dtype != dtypes.float32:
      self.master_params:list[Tensor]|None = [self._zero_shard(p.to(self.device).float().contiguous()) for p in self.params]
    else:
      self.master_params = None

  def _zero_shard(self, t:Tensor) -> Tensor:
    if not self.zero or t.ndim < 2 or (t.shape[0] % len(self.device)) != 0: return t
    return Tensor(t.uop._shard(0, UOp.range(len(self.device), -1, AxisType.DEVICE)).unshard(0)).clone()

  def _zero_gather(self, t:Tensor) -> Tensor:
    if not isinstance(t.device, tuple) or t.uop.axis != 0: return t
    n, sz = len(t.device), t.shape[0] // len(t.device)
    return Tensor.cat(*[t[p*sz:(p+1)*sz] for p in range(n)], dim=0)

  def fschedule_step(self, grads:list[Tensor]) -> list[Tensor]:
    updates, extra = self._step([], grads)
    for i, tt in enumerate(self.params): tt.assign(self._apply_update(tt, updates[i], self.master_params[i] if self.master_params else None))
    fp8_inv_scales = [tt._inv_scale for tt in self.params if hasattr(tt, '_inv_scale')]
    fp8_next_inv_scales = [tt._next_inv_scale for tt in self.params if hasattr(tt, '_next_inv_scale')]
    return extra + self.params + self.buffers + (self.master_params or []) + fp8_inv_scales + fp8_next_inv_scales

  def fstep(self, grads:list[Tensor], grad_norm:Tensor|None=None):
    Tensor.realize(*([grad_norm] if grad_norm is not None else []), *self.fschedule_step(grads))

  def _step(self, params:list[Tensor], grads:list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    grads = list(grads)

    for i in range(len(grads)):
      if grads[i].device != self.m[i].device: grads[i] = grads[i].to(self.m[i].device)
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
    return ret, [self.b1_t, self.b2_t] + self.m + self.v

  def _apply_update(self, t:Tensor, up:Tensor, master:Tensor|None=None) -> Tensor:
    w = master if master is not None else t
    wd = self.wd if t.ndim >= 3 else 0.0
    up = up.float().shard_like(w) + self.lr.to(w.device) * wd * w.detach()
    new_w = w.detach() - up
    if master is not None: master.assign(new_w)
    if self.zero and not (MXFP8 and t.dtype in dtypes.fp8s): new_w = self._zero_gather(new_w)
    # when master is offloaded to a different device than the param, results are resharded back onto the param's (sharded) device
    offloaded = master is not None and master.device != t.device
    if STOCHASTIC_ROUND and t.dtype == dtypes.bfloat16:
      out = stochastic_round_bf16(new_w)
      return out.shard_like(t) if offloaded else out
    if t.dtype in dtypes.fp8s:
      if MXFP8:
        from extra.gemm.cdna_asm_gemm import quantize_mxfp8
        w_q, w_e8, _ = quantize_mxfp8(new_w.reshape(-1, new_w.shape[-1]))
        if self.zero: w_q, w_e8 = self._zero_gather(w_q), self._zero_gather(w_e8)
        new_e8 = w_e8.reshape(t._inv_scale.shape)
        t._inv_scale.assign(new_e8.shard_like(t._inv_scale) if offloaded else new_e8)
        ret = w_q.reshape(t.shape)
        return ret.shard_like(t) if offloaded else ret
      from examples.mlperf.models.flat_llama import FP8_MAX
      if IMMEDIATE_SCALE:
        amax_axis = tuple(range(t._inv_scale.ndim, new_w.ndim))
        new_inv = ((new_w.float().abs().max(axis=amax_axis).detach() + 1e-8) / FP8_MAX).cast(t._inv_scale.dtype)
        t._inv_scale.assign(new_inv.shard_like(t._inv_scale) if offloaded else new_inv)
        scale = new_inv.reciprocal().reshape(*new_inv.shape, *([1]*(new_w.ndim-new_inv.ndim)))
        ret = (new_w * scale).clamp(-FP8_MAX, FP8_MAX).cast(t.dtype)
        return ret.shard_like(t) if offloaded else ret
      # delayed scaling: reuse previous step's inv_scale
      t._inv_scale.assign(t._next_inv_scale)
      inv_scale = t._inv_scale.to(new_w.device) if offloaded else t._inv_scale
      scale = inv_scale.reciprocal().reshape(*inv_scale.shape, *([1]*(new_w.ndim-inv_scale.ndim)))
      scaled = (new_w * scale).clamp(-FP8_MAX, FP8_MAX)
      ret = scaled.cast(t.dtype)
      # update inv_scale for next step from quantized result
      new_amax = (ret.float().abs().max(axis=tuple(range(inv_scale.ndim, ret.ndim))) * inv_scale * FP8_AMAX_MARGIN).detach()
      new_inv = ((new_amax + 1e-8) / FP8_MAX).cast(t._inv_scale.dtype)
      t._next_inv_scale.assign(new_inv.shard_like(t._next_inv_scale) if offloaded else new_inv)
      return ret.shard_like(t) if offloaded else ret
    out = new_w.cast(t.dtype)
    return out.shard_like(t) if offloaded else out

class GradAccClipAdamWGroup(OptimizerGroup):
  def __init__(self, *optimizers:GradAccClipAdamW):
    super().__init__(*optimizers)
    for o in self.optimizers[1:]: o.lr = self.optimizers[0].lr
  def fstep(self, grads:list[Tensor], grad_norm:Tensor|None=None):
    offset = 0
    to_realize = []
    for o in self.optimizers:
      n = len(o.params)
      to_realize += o.fschedule_step(grads[offset:offset+n])
      offset += n
    Tensor.realize(*to_realize, *([grad_norm] if grad_norm is not None else []))
  @property
  def lr(self): return self.optimizers[0].lr
  @property
  def device(self): return self.optimizers[0].device
  @property
  def master_params(self):
    mp = [mp for o in self.optimizers for mp in (o.master_params or [])]
    return mp if mp else None
