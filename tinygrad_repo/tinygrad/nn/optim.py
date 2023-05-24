# sorted in order of increasing complexity
from typing import List
from tinygrad.tensor import Tensor

class Optimizer:
  def __init__(self, params : List[Tensor]):
    # if it's None, but being put into an optimizer, set it to True
    for x in params:
      if x.requires_grad is None: x.requires_grad = True

    self.params : List[Tensor] = [x for x in params if x.requires_grad]
    self.buffers : List[Tensor] = [x for x in params if not x.requires_grad]   # buffers are still realized

  # TODO: this probably shouldn't change the gradients, just the ones used by the optimizer
  def clipnorm(self, amount=1):
    for param in self.params:
      assert param.grad is not None
      # clipnorm is the L2 norm, not value: is this right?
      param.grad.assign(param.grad.clip(-(amount**2), (amount**2)))

  def zero_grad(self):
    for param in self.params: param.grad = None

  def realize(self, extra=None):
    # TODO: corealize
    for p in extra + self.params + self.buffers if extra is not None else self.params + self.buffers:
      p.realize()

class SGD(Optimizer):
  def __init__(self, params : List[Tensor], lr=0.001, momentum=0, nesterov=False):
    super().__init__(params)
    self.lr, self.momentum, self.nesterov = lr, momentum, nesterov
    self.b = [Tensor.zeros(*t.shape, device=params[0].device, requires_grad=False) for t in self.params] if self.momentum else []

  # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
  def step(self) -> None:
    for i, t in enumerate(self.params):
      assert t.grad is not None
      g = t.grad
      if self.momentum:
        self.b[i].assign(self.momentum * self.b[i] + g)
        g = (g + self.momentum * self.b[i]) if self.nesterov else self.b[i]
      t.assign(t.detach() - g * self.lr)
    self.realize(self.b)

class RMSprop(Optimizer):
  def __init__(self, params : List[Tensor], lr=0.001, decay=0.9, eps=1e-8):
    super().__init__(params)
    self.lr, self.decay, self.eps = lr, decay, eps

    self.v = [Tensor.zeros(*t.shape, device=params[0].device, requires_grad=False) for t in self.params]

  def step(self) -> None:
    for i, t in enumerate(self.params):
      assert t.grad is not None
      self.v[i].assign(self.decay * self.v[i] + (1.0 - self.decay) * (t.grad * t.grad))
      t.assign(t.detach() - (t.grad * self.lr).div(self.v[i].sqrt() + self.eps))
    self.realize(self.v)

class Adam(Optimizer):
  def __init__(self, params : List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    super().__init__(params)
    # NOTE: self.t is a tensor so Adam can be jitted
    self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, Tensor([0], requires_grad=False).realize()

    self.m = [Tensor.zeros(*t.shape, device=params[0].device, requires_grad=False) for t in self.params]
    self.v = [Tensor.zeros(*t.shape, device=params[0].device, requires_grad=False) for t in self.params]

  def step(self) -> None:
    self.t = self.t + 1
    a = self.lr * ((1.0 - self.b2**self.t)**0.5) / (1.0 - self.b1**self.t)
    for i, t in enumerate(self.params):
      assert t.grad is not None
      self.m[i].assign(self.b1 * self.m[i] + (1.0 - self.b1) * t.grad)
      self.v[i].assign(self.b2 * self.v[i] + (1.0 - self.b2) * (t.grad * t.grad))
      t.assign(t.detach() - a * self.m[i].div(self.v[i].sqrt() + self.eps))
    self.realize([self.t] + self.m + self.v)

def get_parameters(obj) -> List[Tensor]:
  parameters : List[Tensor] = []
  if isinstance(obj, Tensor):
    parameters.append(obj)
  elif isinstance(obj, (list, tuple)):
    for x in obj: parameters.extend(get_parameters(x))
  elif hasattr(obj, '__dict__'):
    for v in obj.__dict__.values(): parameters.extend(get_parameters(v))
  return parameters
