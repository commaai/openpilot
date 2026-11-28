import unittest, math
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import AdamW
from examples.mlperf.lr_schedulers import CosineAnnealingLRWithWarmup, LambdaLR, LambdaLinearScheduler

np.random.seed(1337)
x_init = np.random.randn(1,4).astype(np.float32)
W_init = np.random.randn(4,4).astype(np.float32)
m_init = np.random.randn(1,4).astype(np.float32)

class TinyNet:
  def __init__(self):
    self.x = Tensor(x_init.copy())
    self.W = Tensor(W_init.copy())
    self.m = Tensor(m_init.copy())

  def forward(self):
    out = self.x.matmul(self.W).relu()
    out = out.log_softmax(1)
    out = out.mul(self.m).add(self.m).sum()
    return out

class TestCosineAnnealingLRWithWarmup(unittest.TestCase):
  # only tests the lr
  def _test_lr(self, base_lr, end_lr, warmup_steps, decay_steps):
    net = TinyNet()
    optim = AdamW([net.W], lr=0.0)
    tiny_lr = CosineAnnealingLRWithWarmup(optim, base_lr, end_lr, warmup_steps, decay_steps)
    lr = []
    for _ in range(warmup_steps+decay_steps):
      lr.append(optim.lr.item())
      tiny_lr.step()
    # reimplemented in python
    expected = []
    for i in range(warmup_steps): expected.append((i+1)/warmup_steps*base_lr)
    for i in range(decay_steps): expected.append(end_lr+(base_lr-end_lr)*(1+math.cos((i+1)/decay_steps*math.pi))/2)
    np.testing.assert_allclose(lr, expected, rtol=1e-5)

  def test_lr_0(self): self._test_lr(3e-4, 8e-5, 3, 5)
  def test_lr_1(self): self._test_lr(3e-4, 8e-5, 10, 20)
  def test_lr_llama3(self): self._test_lr(8e-5, 8e-7, 20, 100)

class TestLambdaLRLinearWarmup(unittest.TestCase):
  def test_linear_lr_warmup(self):
    BS, BASE_LR = 304, 2.5e-7
    lr = BS * BASE_LR
    # Use a dummy Tensor parameter for optimizer because the lr_scheduler only needs the optimizer's device and lr, the params aren't touched.
    optimizer = AdamW([Tensor([1.])])
    lambda_lr_callback = LambdaLinearScheduler(1000, 1.0, 1.0, 1e-06, 10000000000000).schedule
    lr_scheduler = LambdaLR(optimizer, Tensor(lr, device=optimizer.device), lambda_lr_callback)
    lrs = {}

    # with above settings, optimizer.lr should warm up to lr over 1000 steps linearly
    for i in range(1200):
      lr_scheduler.step()
      if i in {0, 499, 998, 999, 1000, 1199}:
        lrs[i] = optimizer.lr.item()

    np.testing.assert_allclose(lr, lrs[999], rtol=0, atol=1e-11)
    np.testing.assert_equal(lrs[999], lrs[1000])
    np.testing.assert_equal(lrs[999], lrs[1199])
    np.testing.assert_allclose(lrs[999] / lrs[0], 1000, rtol=0, atol=1)
    np.testing.assert_allclose(lrs[999] / lrs[499], 2, rtol=0, atol=1e-5)

if __name__ == '__main__':
  unittest.main()
