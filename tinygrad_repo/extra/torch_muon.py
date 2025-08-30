import torch

#credit to KellerJordan at https://github.com/KellerJordan/Muon/tree/master
#some changes: classic momentum instead of weighting gradient
#added ns_steps, ns_params, nesterov as hyperparams
def zeropower_via_newtonschulz5(G:torch.tensor, steps:int, params:tuple[int, ...]):
  """
  Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
  quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
  of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
  zero even beyond the point where the iteration no longer converges all the way to one everywhere
  on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
  where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
  performance at all relative to UV^T, where USV^T = G is the SVD.
  """
  assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng

  a, b, c = params
  X = G
  if G.size(-2) > G.size(-1):
    X = X.mT

  # Ensure spectral norm is at most 1
  X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
  # Perform the NS iterations
  for _ in range(steps):
    A = X @ X.mT
    B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
    X = a * X + B @ X

  if G.size(-2) > G.size(-1):
    X = X.mT

  return X

def muon_update(grad, momentum, beta=0.95, ns_steps=5, ns_params=(3.4445, -4.7750,  2.0315), nesterov=True):
  if beta:
    momentum.mul_(beta).add_(grad)
    update = grad.add(momentum,alpha=beta) if nesterov else momentum
  else: update = grad
  if update.ndim == 4: # for the case of conv filters
    update = update.view(len(update), -1)
  update = zeropower_via_newtonschulz5(update, steps=ns_steps, params=ns_params)
  return update

class SingleDeviceMuon(torch.optim.Optimizer):
  """
  Muon variant for usage in non-distributed settings.
  """
  def __init__(self, params, lr=0.02, weight_decay=0.0, momentum=0.95, ns_steps=5, ns_params=(3.4445, -4.7750,  2.0315), nesterov=True):
    defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps, ns_params=ns_params, nesterov=nesterov)
    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):

    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group["params"]:
        if p.grad is None:
          p.grad = torch.zeros_like(p)  # Force synchronization
        state = self.state[p]
        if len(state) == 0:
          state["momentum_buffer"] = torch.zeros_like(p)
        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"], ns_steps=group["ns_steps"],
                             ns_params=group["ns_params"], nesterov=group["nesterov"])
        p.mul_(1.0 - group["lr"] * group["weight_decay"])

        p.add_(update.reshape(p.shape), alpha=-group["lr"])

    return loss
