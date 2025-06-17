import numpy as np
import torch
import unittest
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.optim import Adam, SGD, AdamW
from tinygrad.helpers import CI
from tinygrad.device import is_dtype_supported

np.random.seed(1337)
x_init = np.random.randn(1,4).astype(np.float32)
W_init = np.random.randn(4,4).astype(np.float32)
m_init = np.random.randn(1,4).astype(np.float32)

class TeenyNet:
  def __init__(self, tensor):
    self.x = tensor(x_init.copy(), requires_grad=True)
    self.W = tensor(W_init.copy(), requires_grad=True)
  def forward(self):
    return (self.x * self.W).sum()

class TinyNet:
  def __init__(self, tensor):
    self.x = tensor(x_init.copy(), requires_grad=True)
    self.W = tensor(W_init.copy(), requires_grad=True)
    self.m = tensor(m_init.copy())

  def forward(self):
    out = self.x.matmul(self.W).relu()
    # print(out.detach().numpy())
    out = out.log_softmax(1)
    out = out.mul(self.m).add(self.m).sum()
    return out

def step(tensor, optim, steps=1, teeny=False, **kwargs):
  net = TeenyNet(tensor) if teeny else TinyNet(tensor)
  optim = optim([net.x, net.W], **kwargs)
  for _ in range(steps):
    out = net.forward()
    optim.zero_grad()
    out.backward()
    optim.step()
  return net.x.detach().numpy(), net.W.detach().numpy()

@unittest.skipIf(CI and Device.DEFAULT in {"CUDA", "NV"}, "slow")
class TestOptim(unittest.TestCase):
  def setUp(self):
    self.old_training = Tensor.training
    Tensor.training = True
  def tearDown(self):
    Tensor.training = self.old_training

  def _test_optim(self, tinygrad_optim, torch_optim, steps, opts, atol, rtol):
    for x,y in zip(step(Tensor, tinygrad_optim, steps, **opts),
                   step(torch.tensor, torch_optim, steps, **opts)):
      np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)

  def _test_sgd(self, steps, opts, atol, rtol): self._test_optim(SGD, torch.optim.SGD, steps, opts, atol, rtol)
  def _test_adam(self, steps, opts, atol, rtol): self._test_optim(Adam, torch.optim.Adam, steps, opts, atol, rtol)
  def _test_adamw(self, steps, opts, atol, rtol): self._test_optim(AdamW, torch.optim.AdamW, steps, opts, atol, rtol)

  def test_multistep_sgd_high_lr_teeny(self): self._test_sgd(2, {'lr': 1.1, 'teeny': True}, 1e-6, 1e-5)
  def test_multistep_adam_high_lr_teeny(self): self._test_adam(2, {'lr': 1.1, 'teeny': True}, 2e-4, 5e-4)

  def test_sgd(self): self._test_sgd(1, {'lr': 0.001}, 1e-6, 0)
  def test_sgd_high_lr(self): self._test_sgd(1, {'lr': 10}, 1e-6, 1e-5)
  def test_sgd_wd(self): self._test_sgd(1, {'lr': 0.001, 'weight_decay': 0.1}, 1e-6, 0)
  def test_sgd_high_lr_wd(self): self._test_sgd(1, {'lr': 10, 'weight_decay': 0.1}, 1e-6, 1e-5)

  def test_multistep_sgd(self): self._test_sgd(10, {'lr': 0.001}, 1e-6, 0)
  def test_multistep_sgd_high_lr(self): self._test_sgd(10, {'lr': 10}, 1e-6, 3e-4)
  def test_multistep_sgd_wd(self): self._test_sgd(10, {'lr': 0.001, 'weight_decay': 0.1}, 1e-6, 0)
  def test_multistep_sgd_high_lr_wd(self): self._test_sgd(10, {'lr': 9, 'weight_decay': 0.1}, 1e-6, 3e-4)

  def test_multistep_sgd_momentum(self): self._test_sgd(10, {'lr': 0.001, 'momentum': 0.9}, 1e-6, 0)
  def test_multistep_sgd_high_lr_momentum(self): self._test_sgd(10, {'lr': 10, 'momentum': 0.9}, 1e-5, 3e-4)
  def test_multistep_sgd_momentum_wd(self): self._test_sgd(10, {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.1}, 1e-6, 0)
  def test_multistep_sgd_high_lr_momentum_wd(self): self._test_sgd(10, {'lr': 10, 'momentum': 0.9, 'weight_decay': 0.1}, 1e-5, 3e-4)

  def test_multistep_sgd_nesterov_momentum(self): self._test_sgd(10, {'lr': 0.001, 'momentum': 0.9, 'nesterov': True}, 1e-5, 0)
  def test_multistep_sgd_high_lr_nesterov_momentum(self): self._test_sgd(10, {'lr': 10, 'momentum': 0.9, 'nesterov': True}, 1e-5, 3e-4)
  def test_multistep_sgd_nesterov_momentum_wd(self):
    self._test_sgd(10, {'lr': 0.001, 'momentum': 0.9, 'nesterov': True, 'weight_decay': 0.1}, 1e-5, 0)
  def test_multistep_sgd_high_lr_nesterov_momentum_wd(self):
    self._test_sgd(10, {'lr': 9, 'momentum': 0.9, 'nesterov': True, 'weight_decay': 0.1}, 1e-5, 3e-4)

  def test_adam(self): self._test_adam(1, {'lr': 0.001}, 1e-5, 0)
  def test_adam_high_lr(self): self._test_adam(1, {'lr': 10}, 1e-4, 1e-4)
  def test_adamw(self): self._test_adamw(1, {'lr': 0.001}, 1e-5, 0)
  def test_adamw_high_lr(self): self._test_adamw(1, {'lr': 10}, 1e-4, 1e-4)

  def test_multistep_adam(self): self._test_adam(10, {'lr': 0.001}, 1e-5, 0)
  def test_multistep_adam_high_lr(self): self._test_adam(10, {'lr': 10}, 2e-3, 5e-4)

  def test_multistep_adamw(self): self._test_adamw(10, {'lr': 0.001}, 1e-5, 0)
  def test_multistep_adamw_high_lr(self): self._test_adamw(10, {'lr': 10}, 5e-4, 2e-3)

  def test_duped_weights(self):
    for Opt in [Adam, AdamW, SGD]:
      losses = []
      for i in range(2):
        w = Tensor(x_init.copy())
        opt = Opt([w], lr=0.1) if i == 0 else Opt([w, w], lr=0.1)

        loss = None
        for _ in range(3):
          loss = w.sum()
          opt.zero_grad()
          loss.backward()
          opt.step()
        losses.append(loss.numpy())

      np.testing.assert_allclose(losses[0], losses[1], atol=1e-4, rtol=0)

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_mixed_precision(self):
    old_default_float, dtypes.default_float = dtypes.default_float, dtypes.half
    # weight update would overflow without upcasting
    self._test_sgd(10, {'lr': 1e10}, 1e-6, 3e-4)
    self._test_adam(1, {'lr': 1e10}, 1e-4, 1e-4)
    self._test_adamw(1, {'lr': 1e10}, 1e-4, 1e-4)
    dtypes.default_float = old_default_float

  def test_assert_tensor_train(self):
    t = Tensor.ones((1,1), requires_grad=True)
    optimizer = Adam([t])
    optimizer.zero_grad()
    old_state = Tensor.training
    t.sum().backward()
    Tensor.training = False
    self.assertRaises(RuntimeError, optimizer.step)
    Tensor.training = True
    optimizer.step()
    Tensor.training = old_state

if __name__ == '__main__':
  unittest.main()
