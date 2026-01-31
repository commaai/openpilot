# simple tests
import unittest
import torch
import warnings
from tinygrad.helpers import getenv, GlobalCounters
if getenv("TINY_BACKEND2"):
  import extra.torch_backend.backend2
  device = "cpu"
else:
  import extra.torch_backend.backend
  device = "tiny"


class TestKernelFusionRegression(unittest.TestCase):
  def _realize(self, t): _ = t.detach().cpu().numpy()

  def _check_kernel_count(self, fn, expected_kernels):
    torch.manual_seed(42)
    GlobalCounters.reset()
    fn().detach().cpu().numpy()
    expectation = f"{GlobalCounters.kernel_count} vs {expected_kernels} expected."
    if GlobalCounters.kernel_count < expected_kernels: warnings.warn(f"{expectation} Expectation can be lowered.", UserWarning)
    self.assertLessEqual(GlobalCounters.kernel_count, expected_kernels, f"{expectation}")

  def test_elementwise_fusion(self):
    def fn():
      x = torch.randn(128, 128, device=device)
      return (x + 1.0) * 2.0 - 0.5
    self._check_kernel_count(fn, 6)

  def test_relu_fusion(self):
    def fn():
      x = torch.randn(1, 3, 32, 32, device=device)
      conv = torch.nn.Conv2d(3, 16, 3, padding=1).to(device)
      with torch.no_grad():
        return torch.nn.functional.relu(conv(x))
    self._check_kernel_count(fn, 8)

  def test_batchnorm_fusion(self):
    def fn():
      x = torch.randn(2, 3, 16, 16, device=device)
      conv = torch.nn.Conv2d(3, 8, 3, padding=1).to(device)
      bn = torch.nn.BatchNorm2d(8).to(device)
      bn.eval()
      with torch.no_grad():
        return torch.nn.functional.relu(bn(conv(x)))
    self._check_kernel_count(fn, 16)

  def test_reduce_fusion(self):
    def fn():
      x = torch.randn(64, 64, device=device)
      return (x * 2.0).sum()
    self._check_kernel_count(fn, 7)

  def test_matmul_elementwise_fusion(self):
    def fn():
      x = torch.randn(32, 32, device=device)
      w = torch.randn(32, 32, device=device)
      return torch.nn.functional.relu(x @ w + 1.0)
    self._check_kernel_count(fn, 6)

  def test_pooling_fusion(self):
    def fn():
      x = torch.randn(1, 8, 16, 16, device=device)
      return torch.nn.functional.max_pool2d(x * 2.0, 2)
    self._check_kernel_count(fn, 5)

  def test_residual_add_relu_fusion(self):
    def fn():
      x = torch.randn(1, 8, 16, 16, device=device)
      identity = torch.randn(1, 8, 16, 16, device=device)
      out = x + identity
      return torch.nn.functional.relu(out)
    self._check_kernel_count(fn, 6)

  def test_inplace_add_relu_fusion(self):
    def fn():
      x = torch.randn(1, 16, 32, 32, device=device)
      y = torch.randn(1, 16, 32, 32, device=device)
      x += y
      return torch.nn.functional.relu(x)
    self._check_kernel_count(fn, 6)

  def test_conv_bn_add_relu_fusion(self):
    def fn():
      x = torch.randn(1, 8, 16, 16, device=device)
      identity = torch.randn(1, 8, 16, 16, device=device)
      conv = torch.nn.Conv2d(8, 8, 3, padding=1, bias=False).to(device)
      bn = torch.nn.BatchNorm2d(8).to(device)
      bn.eval()
      with torch.no_grad():
        out = bn(conv(x))
        out += identity
        return torch.nn.functional.relu(out)
    self._check_kernel_count(fn, 16)

  def test_multiple_inplace_ops_fusion(self):
    def fn():
      x = torch.randn(64, 64, device=device)
      x += 1.0
      x *= 2.0
      return torch.nn.functional.relu(x)
    self._check_kernel_count(fn, 4)

  def test_view_inplace_no_fusion_break(self):
    def fn():
      x = torch.randn(4, 64, device=device)
      view = x[1:3]
      view += 1.0
      return x.sum()
    self._check_kernel_count(fn, 8)

  def test_batchnorm_running_stats_update(self):
    def fn():
      x = torch.randn(2, 8, 8, 8, device=device)
      bn = torch.nn.BatchNorm2d(8).to(device)
      bn.train()
      with torch.no_grad():
        return bn(x)
    self._check_kernel_count(fn, 10)

  # this is a minimal extra/other_mnist/beautiful_mnist_torch.py to cover fusion for training with optimizer
  def test_mnist_training_fusion(self):
    def fn():
      model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(8*14*14, 10)
      ).to(device)
      optimizer = torch.optim.Adam(model.parameters(), 1e-3)
      x = torch.randn(32, 1, 28, 28, device=device)
      labels = torch.randint(0, 10, (32,), device=device)
      out = model(x)
      loss = torch.nn.functional.cross_entropy(out, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      return loss
    self._check_kernel_count(fn, 33)

if __name__ == "__main__":
  unittest.main()
