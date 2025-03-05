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

  def test_expand(self):
    a = torch.Tensor([1,2,3,4]).to(device)
    out = a.reshape(4,1).expand(4,4)
    np.testing.assert_equal(out.cpu().numpy(), [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]])

  def test_reshape(self):
    a = torch.Tensor([[1,2],[3,4]]).to(device)
    np.testing.assert_equal(a.reshape(4).cpu().numpy(), [1,2,3,4])
    np.testing.assert_equal(a.reshape(2,1,2).cpu().numpy(), [[[1,2]],[[3,4]]])
    np.testing.assert_equal(a.unsqueeze(1).cpu().numpy(), [[[1,2]],[[3,4]]])
    np.testing.assert_equal(a.unsqueeze(1).unsqueeze(1).cpu().numpy(), [[[[1,2]]],[[[3,4]]]])
    np.testing.assert_equal(a.unsqueeze(1).unsqueeze(1).squeeze().cpu().numpy(), [[1,2],[3,4]])

  def test_permute(self):
    a = torch.Tensor([[1,2],[3,4]]).to(device)
    print(a.stride())
    null = a.permute(0,1)
    perm = a.permute(1,0)
    back = perm.permute(1,0)
    np.testing.assert_equal(a.cpu().numpy(), [[1,2],[3,4]])
    np.testing.assert_equal(null.cpu().numpy(), [[1,2],[3,4]])
    np.testing.assert_equal(perm.cpu().numpy(), [[1,3],[2,4]])
    np.testing.assert_equal(back.cpu().numpy(), [[1,2],[3,4]])

  def test_shrink(self):
    a = torch.Tensor([1,2,3,4]).to(device)
    np.testing.assert_equal(a[:3].cpu().numpy(), [1,2,3])
    np.testing.assert_equal(a[1:].cpu().numpy(), [2,3,4])

  def test_plus_inplace(self):
    a = torch.ones(4, device=device)
    b = torch.ones(4, device=device)
    a += b
    a += b
    np.testing.assert_equal(a.cpu().numpy(), [3,3,3,3])

  def test_exp2(qself):
    a = torch.ones(4, device=device)
    b = a.exp2()
    np.testing.assert_equal(b.cpu().numpy(), [2,2,2,2])

  def test_isfinite(self):
    a = torch.ones(4, device=device)
    np.testing.assert_equal(torch.isfinite(a).cpu().numpy(), [True, True, True, True])

  def test_eq(self):
    a = torch.ones(4, device=device)
    b = torch.ones(4, device=device)
    c = a == b
    print(c.cpu().numpy())

  @unittest.skip("meh")
  def test_str(self):
    a = torch.ones(4, device=device)
    print(str(a))

if __name__ == "__main__":
  unittest.main()
