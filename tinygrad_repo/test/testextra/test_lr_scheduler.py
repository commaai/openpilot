import numpy as np
import torch
import unittest
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_parameters
from tinygrad.nn.optim import Adam, SGD
from tinygrad.helpers import DEBUG
from extra.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from extra.training import train, evaluate
from extra.datasets import fetch_mnist

np.random.seed(1337)
Tensor.manual_seed(1337)

X_train, Y_train, X_test, Y_test = fetch_mnist()

class TinyBobNet:
  def __init__(self):
    self.l1 = Tensor.scaled_uniform(784, 128)
    self.l2 = Tensor.scaled_uniform(128, 10)

  def parameters(self):
    return get_parameters(self)

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).log_softmax()

def lr_scheduler_training(sched_fn=None, args=None):
  model = TinyBobNet()
  optim = Adam(model.parameters(), lr=0.01)
  if sched_fn is not None: sched = sched_fn(optim, **args)
  for _ in range(25):
    train(model, X_train, Y_train, optim, 100)
    if sched_fn is not None:
      if isinstance(sched, ReduceLROnPlateau):
        sched.step(evaluate(model, X_test, Y_test))
      else:
        sched.step()
  return evaluate(model, X_test, Y_test)

def current_lr(optim): return optim.param_groups[0]['lr'] if hasattr(optim, 'param_groups') else optim.lr
def get_lrs(optim, sched, epochs, steps=1, accs=None):
  lr = current_lr(optim)
  if not isinstance(lr, float): lr = lr.numpy()[0]
  lrs = [lr]
  for e in range(epochs):
    for _ in range(steps):
      optim.step()
    sched.step() if accs is None else sched.step(accs[e])
    lr = current_lr(optim)
    if not isinstance(lr, float): lr = lr.numpy()[0]
    lrs.append(lr)
  return lrs

class TestLrScheduler(unittest.TestCase):
  def setUp(self):
    self.old_training = Tensor.training
    Tensor.training = True
  def tearDown(self):
    Tensor.training = self.old_training

  def _test_lr_scheduler(self, tinygrad_sched, torch_sched, epochs, opts, atol, rtol, adam=True):
    accs = opts.pop('accs', None)
    test_tensor = Tensor([0.], requires_grad=True)    # NOTE: optimizers are broken on 0-dim tensors because it broadcasts to [lr]
    test_tensor.mean().backward()
    if adam:
      tinygrad_optim, torch_optim = Adam([test_tensor], lr=0.01), torch.optim.Adam([torch.tensor([0.], requires_grad=True)], lr=0.01)
    else:
      tinygrad_optim, torch_optim = SGD([test_tensor], lr=0.01), torch.optim.SGD([torch.tensor([0.], requires_grad=True)], lr=0.01)
    tinygrad_sched, torch_sched = tinygrad_sched(tinygrad_optim, **opts), torch_sched(torch_optim, **opts)

    tinygrad_lrs = get_lrs(tinygrad_optim, tinygrad_sched, epochs, accs=accs)
    torch_lrs = get_lrs(torch_optim, torch_sched, epochs, accs=accs)

    np.testing.assert_allclose(tinygrad_lrs, torch_lrs, atol=atol, rtol=rtol)

  def _test_multisteplr(self, epochs, opts, atol, rtol, adam=True):
    self._test_lr_scheduler(MultiStepLR, torch.optim.lr_scheduler.MultiStepLR, epochs, opts, atol, rtol, adam=adam)
  def _test_reducelronplateau(self, epochs, opts, atol, rtol):
    opts['accs'] = np.random.randn(epochs)
    self._test_lr_scheduler(ReduceLROnPlateau, torch.optim.lr_scheduler.ReduceLROnPlateau, epochs, opts, atol, rtol)
  def _test_cosineannealinglr(self, epochs, opts, atol, rtol):
    opts['T_max'] = epochs
    self._test_lr_scheduler(CosineAnnealingLR, torch.optim.lr_scheduler.CosineAnnealingLR, epochs, opts, atol, rtol)
  def _test_onecyclelr(self, epochs, opts, atol, rtol):
    opts['total_steps'] = epochs
    self._test_lr_scheduler(OneCycleLR, torch.optim.lr_scheduler.OneCycleLR, epochs, opts, atol, rtol)

  def test_multisteplr(self): self._test_multisteplr(10, {'milestones': [1, 2, 7]}, 1e-6, 1e-6)
  def test_multisteplr_gamma(self): self._test_multisteplr(10, {'milestones': [1, 2, 7], 'gamma': 0.1337}, 1e-6, 1e-6)

  def test_reducelronplateau(self): self._test_reducelronplateau(100, {}, 1e-6, 1e-6)
  def test_reducelronplateau_max(self): self._test_reducelronplateau(100, {'mode': 'max'}, 1e-6, 1e-6)
  def test_reducelronplateau_factor(self): self._test_reducelronplateau(100, {'factor': 0.1337}, 1e-6, 1e-6)
  def test_reducelronplateau_patience(self): self._test_reducelronplateau(100, {'patience': 3}, 1e-6, 1e-6)
  def test_reducelronplateau_threshold(self): self._test_reducelronplateau(100, {'threshold': 1e-6}, 1e-6, 1e-6)
  def test_reducelronplateau_threshold_mode(self): self._test_reducelronplateau(100, {'threshold_mode': 'abs'}, 1e-6, 1e-6)

  def test_cosineannealinglr(self): self._test_cosineannealinglr(100, {}, 1e-6, 1e-6)
  def test_cosineannealinglr_eta_min(self): self._test_cosineannealinglr(100, {'eta_min': 0.001}, 1e-6, 1e-6)

  def test_multistep_2step(self):
    # was making this fail with LRU=1, some issue with epoch_counter
    if DEBUG>=2: print("first")
    self._test_multisteplr(1, {'milestones': [1]}, 1e-6, 1e-6, adam=False)
    if DEBUG>=2: print("second")
    self._test_multisteplr(1, {'milestones': [1], 'gamma': 0.133}, 1e-6, 1e-6, adam=False)
    if DEBUG>=2: print("third")

  def test_onecyclelr(self): self._test_onecyclelr(100, {'pct_start': 0.3, 'anneal_strategy': 'linear',
                                                         'cycle_momentum': False, 'div_factor': 25.0,
                                                         'final_div_factor': 10000.0, 'max_lr':1e-5}, 1e-6, 1e-6)
  @unittest.skip("slow")
  def test_training(self):
    without = lr_scheduler_training()
    sched_fns = [MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR]
    argss = [{'milestones': [5, 7, 10, 15], 'gamma': 0.5}, {'factor': 0.5, 'patience': 2}, {'T_max': 25, 'eta_min': 0.001},
             {'pct_start': 0.3, 'anneal_strategy': 'linear', 'cycle_momentum': False, 'div_factor': 25.0, 'final_div_factor': 10000.0,
              'max_lr':1e-5, 'total_steps': 25}]
    for sched_fn, args in zip(sched_fns, argss):
      with_sched = lr_scheduler_training(sched_fn, args)
      assert with_sched > without

if __name__ == '__main__':
  unittest.main()
