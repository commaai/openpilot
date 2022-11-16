# sorted in order of increasing complexity
from tinygrad.tensor import Tensor

class Optimizer:
  def __init__(self, params):
    # if it's None, but being put into an optimizer, set it to True
    for x in params:
      if x.requires_grad is None:
        x.requires_grad = True

    self.params = [x for x in params if x.requires_grad]

  # TODO: this probably shouldn't change the gradients, just the ones used by the optimizer
  def clipnorm(self, amount=1):
    for param in self.params:
      # clipnorm is the L2 norm, not value: is this right?
      param.grad.assign(param.grad.clip(-(amount**2), (amount**2)))

  def zero_grad(self):
    for param in self.params:
      param.grad = None

  def realize(self, extra=None):
    # TODO: corealize
    for p in self.params + extra if extra is not None else self.params:
      p.realize()

class SGD(Optimizer):
  def __init__(self, params, lr=0.001):
    super().__init__(params)
    self.lr = lr

  def step(self):
    for t in self.params:
      t.assign(t.detach() - t.grad * self.lr)
    self.realize()

class RMSprop(Optimizer):
  def __init__(self, params, lr=0.001, decay=0.9, eps=1e-8):
    super().__init__(params)
    self.lr, self.decay, self.eps = lr, decay, eps

    self.v = [Tensor.zeros(*t.shape, device=params[0].device, requires_grad=False) for t in self.params]

  def step(self):
    for i, t in enumerate(self.params):
      self.v[i] = self.decay * self.v[i] + (1.0 - self.decay) * (t.grad * t.grad)
      t.assign(t.detach() - (t.grad * self.lr).div(self.v[i].sqrt() + self.eps))
    self.realize(self.v)

class Adam(Optimizer):
  def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    super().__init__(params)
    self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0

    self.m = [Tensor.zeros(*t.shape, device=params[0].device, requires_grad=False) for t in self.params]
    self.v = [Tensor.zeros(*t.shape, device=params[0].device, requires_grad=False) for t in self.params]

  def step(self):
    self.t = self.t + 1
    a = self.lr * ((1.0 - self.b2**self.t)**0.5) / (1.0 - self.b1**self.t)
    for i, t in enumerate(self.params):
      self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * t.grad
      self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * (t.grad * t.grad)
      t.assign(t.detach() - a * self.m[i].div(self.v[i].sqrt() + self.eps))
    self.realize(self.m + self.v)

def get_parameters(obj):
  parameters = []
  if isinstance(obj, Tensor):
    parameters.append(obj)
  elif isinstance(obj, list) or isinstance(obj, tuple):
    for x in obj:
      parameters.extend(get_parameters(x))
  elif hasattr(obj, '__dict__'):
    for v in obj.__dict__.values():
      parameters.extend(get_parameters(v))
  return parameters
