import unittest
from tinygrad.helpers import getenv
import torch
import tinygrad.frontend.torch
torch.set_default_device("tiny")
import numpy as np

@unittest.skipIf(getenv("GPUS",1)<=1, "only single GPU")
class TestTorchBackendMultiGPU(unittest.TestCase):
  def test_transfer(self):
    a = torch.Tensor([[1,2],[3,4]]).to("tiny:0")
    b = torch.Tensor([[3,2],[1,0]]).to("tiny:1")
    self.assertNotEqual(a.device, b.device)
    np.testing.assert_array_equal(a.cpu(), a.to("tiny:1").cpu())
    np.testing.assert_array_equal(b.cpu(), b.to("tiny:1").cpu())

  def test_basic_ops(self):
    a = torch.Tensor([[1,2],[3,4]]).to("tiny:0")
    b = torch.Tensor([[3,2],[1,0]]).to("tiny:1")
    c1 = a + b.to("tiny:0")
    c2 = b + a.to("tiny:1")
    np.testing.assert_array_equal(c1.cpu(), torch.full((2,2),4).cpu())
    np.testing.assert_array_equal(c1.cpu(), c2.cpu())

  # TODO: torch.distributed functions

if __name__ == "__main__":
  unittest.main()

