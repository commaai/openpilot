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

  def test_as_strided(self):
    a = torch.arange(70, device=device).reshape(1,1,10,7)
    a = a.as_strided((1,1,10,5), (0,0,7,1), storage_offset=0)
    a = a.as_strided((1,1,5,5), (50,50,7,1), storage_offset=21)
    np.testing.assert_equal(a.cpu().numpy().sum(-1), [[[115,150,185,220,255]]])

  def test_plus_inplace(self):
    a = torch.ones(4, device=device)
    b = torch.ones(4, device=device)
    a += b
    a += b
    np.testing.assert_equal(a.cpu().numpy(), [3,3,3,3])

  def test_exp2(self):
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
    print(c.cpu())

  def test_maxpool2d_backward(self):
    x = torch.arange(3*3, device=device).reshape(1, 1, 3, 3).requires_grad_(True)
    torch.nn.functional.max_pool2d(x, kernel_size=2, stride=1).sum().backward()
    np.testing.assert_equal(x.grad.squeeze().cpu().numpy(), [[0, 0, 0], [0, 1, 1], [0, 1, 1]])

  def test_copy_cast(self):
    x = torch.zeros(4, device=device, dtype=torch.int64)
    y = torch.ones(4, device=device, dtype=torch.float32).to(dtype=torch.int64)
    res1 = x ^ y # an operation that only works on int types
    print(res1.cpu())
    y = y.cpu().float().to(device=device, dtype=torch.int64)
    res2 = x ^ y
    print(res2.cpu())

  def test_topk(self):
    # test topk return_types
    a = torch.tensor([1, 3, 2, 4], device=device)
    out = torch.topk(a, k=2)
    np.testing.assert_equal(out.values.cpu().numpy(), [4, 3])
    np.testing.assert_equal(out.indices.cpu().numpy(), [3, 1])

  def test_masked_select(self):
    a = torch.tensor([4, 3, 2, 1], device=device)
    mask = torch.tensor([True, False, True, False], device=device)
    out = torch.masked_select(a, mask)
    np.testing.assert_equal(out.cpu().numpy(), [4, 2])
    mask = torch.tensor(True, device=device)
    out = torch.masked_select(a, mask)
    np.testing.assert_equal(out.cpu().numpy(), [4, 3, 2, 1])

  @unittest.skip("meh")
  def test_str(self):
    a = torch.ones(4, device=device)
    print(str(a))

  @unittest.skip("failed")
  def test_floor_div(self):
    a = torch.tensor([10., 7., 5.], device=device)
    b = torch.tensor([3., 2., 2.], device=device)
    result = a // b
    np.testing.assert_equal(result.cpu().numpy(), [3., 3., 2.])

if __name__ == "__main__":
  unittest.main()
