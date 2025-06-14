# sorted in order of increasing complexity
import itertools
from tinygrad.helpers import dedup, flatten, getenv, unwrap, FUSE_OPTIM
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes, least_upper_dtype

class Optimizer:
  """
  Base class for all optimizers.
  """
  def __init__(self, params: list[Tensor], lr: float, fused=FUSE_OPTIM):
    # if it's None, but being put into an optimizer, set it to True
    for x in params:
      if x.requires_grad is None: x.requires_grad = True

    self.params: list[Tensor] = dedup([x for x in params if x.requires_grad])
    assert len(self.params) != 0, "optimizer must have at least one param"
    self.device = self.params[0].device
    self.buffers: list[Tensor] = dedup([x for x in params if not x.requires_grad])   # buffers are still realized
    self.fused = fused
    # store lr in at least float32 precision
    self.lr = Tensor(lr if getenv("CONST_LR") else [lr], requires_grad=False, device=self.device,
                     dtype=least_upper_dtype(dtypes.default_float, dtypes.float32))
    if self.fused: self.pos_params = list(itertools.accumulate(self.params, lambda x,y: x+y.flatten().shape[0], initial=0))

  def _new_optim_param(self) -> list[Tensor]:
    if self.fused: return [Tensor.zeros(self.pos_params[-1], dtype=dtypes.float32, device=self.device, requires_grad=False).contiguous()]
    return [Tensor.zeros(*t.shape, dtype=dtypes.float32, device=t.device, requires_grad=False).contiguous() for t in self.params]

  def zero_grad(self):
    """
    Zeroes the gradients of all the parameters.
    """
    for param in self.params: param.grad = None

  def step(self):
    """
    Performs a single optimization step.
    """
    Tensor.realize(*self.schedule_step())

  def schedule_step(self) -> list[Tensor]:
    """
    Returns the tensors that need to be realized to perform a single optimization step.
    """
    if not Tensor.training: raise RuntimeError(
            f"""Tensor.training={Tensor.training}, Tensor.training must be enabled to use the optimizer.
                - help: Consider setting Tensor.training=True before calling Optimizer.step().""")
    if self.fused:
      # optimizer fusion just concatentates all the buffers, runs the _step, then splits them back up
      out, extra = self._step([Tensor.cat(*[t.flatten() for t in self.params], dim=0)],
                              [Tensor.cat(*[unwrap(t.grad).flatten() for t in self.params], dim=0)])
      updated_params = [out[0][self.pos_params[i]:self.pos_params[i+1]].reshape(tt.shape) for i, tt in enumerate(self.params)]
    else:
      updated_params, extra = self._step(self.params, [unwrap(t.grad) for t in self.params])
    for i, tt in enumerate(self.params): tt.assign(updated_params[i])
    return extra+self.params+self.buffers

  def _step(self, params:list[Tensor], grads:list[Tensor]) -> tuple[list[Tensor], list[Tensor]]: raise NotImplementedError

class OptimizerGroup(Optimizer):
  """
  Combines multiple optimizers into one.
  """
  def __init__(self, *optimizers: Optimizer): # pylint: disable=super-init-not-called
    self.optimizers = optimizers
    self.params, self.buffers = flatten([o.params for o in self.optimizers]), flatten([o.buffers for o in self.optimizers])
  def __getitem__(self, i): return self.optimizers[i]
  def zero_grad(self): [o.zero_grad() for o in self.optimizers]
  def schedule_step(self) -> list[Tensor]: return [x for o in self.optimizers for x in o.schedule_step()]

# LARS is essentially just trust ratio to SGD so if we just set the trust coeff 0.0 it's just standard SGD.
def SGD(params: list[Tensor], lr=0.001, momentum=0.0, weight_decay=0.0, nesterov=False, classic=False, fused=FUSE_OPTIM):
  """
  Stochastic Gradient Descent (SGD) optimizer with optional momentum and weight decay.

  `classic` is a boolean flag that determines whether to use the popular momentum update rule or the classic momentum update rule.

  - Described: https://paperswithcode.com/method/sgd
  """
  return LARS(params, lr, momentum, weight_decay, nesterov, classic, tcoef=0.0, fused=fused)

class LARS(Optimizer):
  """
  Layer-wise Adaptive Rate Scaling (LARS) optimizer with optional momentum and weight decay.

  - Described: https://paperswithcode.com/method/lars
  - Paper: https://arxiv.org/abs/1708.03888v3
  """
  def __init__(self, params:list[Tensor], lr=0.001, momentum=0.9, weight_decay=1e-4, nesterov=False, classic=True, tcoef=0.001, fused=FUSE_OPTIM):
    super().__init__(params, lr, fused)
    self.momentum, self.wd, self.nesterov, self.classic, self.tcoef = momentum, weight_decay, nesterov, classic, tcoef
    self.b = self._new_optim_param() if self.momentum else []

  def _step(self, params:list[Tensor], grads:list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    ret = []
    for i, (t, g) in enumerate(zip(params, grads)):
      if self.tcoef != 0:
        r1 = t.detach().square().sum().sqrt()
        r2 = g.square().sum().sqrt()
        r:Tensor|float = (r1 > 0).where((r2 > 0).where(self.tcoef * r1 / (r2 + self.wd * r1), 1.0), 1.0)
      else: r = 1.0
      g = g + self.wd * t.detach()
      # classic momentum does post learning rate update
      if self.classic: g = g * r * self.lr
      if self.momentum:
        # TODO: this contiguous is required for correctness because self.b[i] becomes a non contiguous view
        # the scheduler should detect this and just insert contiguous
        self.b[i].assign(self.momentum * self.b[i].contiguous() + g)  # NOTE: self.b[i] is zero on the first run, no if required
        g = (g + self.momentum * self.b[i]) if self.nesterov else self.b[i]
      # popular momentum does pre learning rate update
      if not self.classic: g = g * r * self.lr
      ret.append((t.detach() - g).cast(t.dtype))
    return ret, self.b

# LAMB is essentially just the trust ratio part of LARS applied to Adam/W so if we just set the trust ratio to 1.0 it's just Adam/W.
def AdamW(params: list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01, fused=FUSE_OPTIM):
  """
  AdamW optimizer with optional weight decay.

  - Described: https://paperswithcode.com/method/adamw
  - Paper: https://arxiv.org/abs/1711.05101v3
  """
  return LAMB(params, lr, b1, b2, eps, weight_decay, adam=True, fused=fused)
def Adam(params: list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, fused=FUSE_OPTIM):
  """
  Adam optimizer.

  - Described: https://paperswithcode.com/method/adam
  - Paper: https://arxiv.org/abs/1412.6980
  """
  return LAMB(params, lr, b1, b2, eps, 0.0, adam=True, fused=fused)

class LAMB(Optimizer):
  """
  LAMB optimizer with optional weight decay.

  - Described: https://paperswithcode.com/method/lamb
  - Paper: https://arxiv.org/abs/1904.00962
  """
  def __init__(self, params: list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, weight_decay=0.0, adam=False, fused=FUSE_OPTIM):
    super().__init__(params, lr, fused)
    self.b1, self.b2, self.eps, self.wd, self.adam = b1, b2, eps, weight_decay, adam
    self.b1_t, self.b2_t = (Tensor.ones((1,), dtype=dtypes.float32, device=self.device, requires_grad=False).contiguous() for _ in [b1, b2])
    self.m = self._new_optim_param()
    self.v = self._new_optim_param()

  def _step(self, params:list[Tensor], grads:list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    ret = []
    self.b1_t *= self.b1
    self.b2_t *= self.b2
    for i, (t, g) in enumerate(zip(params, grads)):
      self.m[i].assign(self.b1 * self.m[i] + (1.0 - self.b1) * g)
      self.v[i].assign(self.b2 * self.v[i] + (1.0 - self.b2) * (g * g))
      m_hat = self.m[i] / (1.0 - self.b1_t)
      v_hat = self.v[i] / (1.0 - self.b2_t)
      up = (m_hat / (v_hat.sqrt() + self.eps)) + self.wd * t.detach()
      if not self.adam:
        r1 = t.detach().square().sum().sqrt()
        r2 = up.square().sum().sqrt()
        r: Tensor|float = Tensor.where(r1 > 0, Tensor.where(r2 > 0, r1 / r2, 1.0), 1.0)
      else:
        r = 1.0
      ret.append((t.detach() - self.lr * r * up).cast(t.dtype))
    return ret, [self.b1_t, self.b2_t] + self.m + self.v
