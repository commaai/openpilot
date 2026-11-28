import numpy as np
import torch
import unittest
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.optim import Adam, SGD, AdamW, Muon, LAMB
from tinygrad.helpers import Context
from test.helpers import needs_devices

np.random.seed(1337)
x_init = np.random.randn(1,4).astype(np.float32)
W_init = np.random.randn(4,4).astype(np.float32)

def _param(tensor, val):
  return tensor(val, requires_grad=True) if tensor is torch.tensor else tensor(val)

class TeenyNet:
  def __init__(self, tensor):
    self.x = _param(tensor, x_init.copy())
    self.W = _param(tensor, W_init.copy())
  def forward(self):
    return (self.x * self.W).sum()

def step(tensor, optim, steps=1, **kwargs):
  net = TeenyNet(tensor)
  optim = optim([net.x, net.W], **kwargs)
  for _ in range(steps):
    out = net.forward()
    optim.zero_grad()
    out.backward()
    optim.step()
  return net.x.detach().numpy(), net.W.detach().numpy()

class TestOptim(unittest.TestCase):
  def setUp(self): self.enterContext(Context(TRAINING=1))

  def _test_optim(self, tinygrad_optim, torch_optim, steps, opts, atol, rtol):
    for x,y in zip(step(Tensor, tinygrad_optim, steps, **opts),
                   step(torch.tensor, torch_optim, steps, **opts)):
      np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)

  def _test_sgd(self, steps, opts, atol, rtol): self._test_optim(SGD, torch.optim.SGD, steps, opts, atol, rtol)
  def _test_adam(self, steps, opts, atol, rtol): self._test_optim(Adam, torch.optim.Adam, steps, opts, atol, rtol)
  def _test_adamw(self, steps, opts, atol, rtol): self._test_optim(AdamW, torch.optim.AdamW, steps, opts, atol, rtol)
  def _test_muon(self, steps, opts, atol, rtol): self._test_optim(Muon, torch.optim.Muon, steps, opts, atol, rtol)

  def test_multistep_sgd_high_lr_momentum(self): self._test_sgd(2, {'lr': 10, 'momentum': 0.9}, 1e-5, 3e-4)
  def test_multistep_sgd_high_lr_nesterov_momentum_wd(self):
    self._test_sgd(2, {'lr': 9, 'momentum': 0.9, 'nesterov': True, 'weight_decay': 0.1}, 1e-5, 3e-4)

  def test_multistep_muon_high_lr_teeny(self): self._test_muon(2, {'lr': 1.1}, 1e-2, 5e-4)

  def test_multistep_adam_high_lr(self): self._test_adam(2, {'lr': 10}, 2e-3, 5e-4)

  def test_multistep_adamw_high_lr(self): self._test_adamw(2, {'lr': 10}, 5e-4, 2e-3)

  def test_duped_weights(self):
    losses = []
    for i in range(2):
      w = Tensor(x_init.copy())
      opt = SGD([w], lr=0.1) if i == 0 else SGD([w, w], lr=0.1)

      loss = None
      for _ in range(2):
        loss = w.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
      losses.append(loss.numpy())

    np.testing.assert_allclose(losses[0], losses[1], atol=1e-4, rtol=0)

  @unittest.skipUnless(dtypes.half in Device[Device.DEFAULT].renderer.supported_dtypes(), "need half")
  def test_mixed_precision(self):
    self.enterContext(Context(DEFAULT_FLOAT=dtypes.half))
    # weight update would overflow without upcasting
    self._test_sgd(1, {'lr': 1e10}, 1e-6, 3e-4)

  def test_assert_tensor_train(self):
    t = Tensor.ones((1,1))
    optimizer = Adam([t])
    optimizer.zero_grad()
    t.sum().backward()
    with Context(TRAINING=0):
      self.assertRaises(RuntimeError, optimizer.step)
    with Context(TRAINING=1):
      optimizer.step()

  def test_lamb_cpu_offload(self):
    # test that LAMB works when optimizer params (m, v, b1_t, b2_t) are moved to CPU
    t = Tensor(x_init.copy())
    opt = LAMB([t])
    # move optimizer state to CPU
    for p in opt.m + opt.v + [opt.b1_t, opt.b2_t]: p.to_("CPU")
    # run a step
    t.sum().backward()
    opt.step()
    self.assertEqual(t.device, Device.DEFAULT)
    self.assertEqual(opt.m[0].device, "CPU")

  @needs_devices(2)
  def test_lamb_cpu_offload_multi(self):
    ds = tuple(f"{Device.DEFAULT}:{i}" for i in range(2))
    t = Tensor(x_init.copy()).shard(ds, axis=1)
    ds = t.device
    opt = LAMB([t])
    # move optimizer state to CPU
    for p in opt.m + opt.v + [opt.b1_t, opt.b2_t]: p.to_("CPU")
    # run a step
    t.sum().backward()
    opt.step()
    self.assertEqual(t.device, ds)
    self.assertEqual(opt.m[0].device, "CPU")

if __name__ == '__main__':
  unittest.main()
