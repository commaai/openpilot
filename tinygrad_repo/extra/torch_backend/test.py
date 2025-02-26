# simple tests
import unittest
import torch
import numpy as np
from tinygrad.helpers import getenv
if getenv("TINY_BACKEND2"):
  import extra.torch_backend.backend2
  device = "cpu"
else:
  import extra.torch_backend.backend
  device = "tiny"

class TestTorchBackend(unittest.TestCase):
  def test_numpy_ones(self):
    a = torch.ones(4, device=device)
    np.testing.assert_equal(a.cpu().numpy(), [1,1,1,1])

  def test_numpy_ones(self):
    a = torch.ones(4, dtype=torch.int32, device=device)
    assert a.dtype == torch.int32
    np.testing.assert_equal(a.cpu().numpy(), [1,1,1,1])

  def test_plus(self):
    a = torch.ones(4, device=device)
    b = torch.ones(4, device=device)
    c = a+b
    np.testing.assert_equal(c.cpu().numpy(), [2,2,2,2])

  def test_exp2(qself):
    a = torch.ones(4, device=device)
    b = a.exp2()
    print(b)

  def test_eq(self):
    a = torch.ones(4, device=device)
    b = torch.ones(4, device=device)
    c = a == b
    print(c.cpu().numpy())

  def test_isfinite(self):
    a = torch.ones(4, device=device)
    np.testing.assert_equal(torch.isfinite(a).cpu().numpy(), [True, True, True, True])

  # TODO: why
  def test_str(self):
    a = torch.ones(4, device=device)
    print(str(a))

if __name__ == "__main__":
  unittest.main()
