# simple tests
import unittest
import torch
import numpy as np
from tinygrad.helpers import getenv, Context, GlobalCounters
if getenv("TINY_BACKEND2"):
  import extra.torch_backend.backend2
  device = "cpu"
else:
  import extra.torch_backend.backend
  device = "tiny"

class TestTorchBackend(unittest.TestCase):
  def test_randperm_generator_out(self):
    n = 10
    out = torch.empty(n, dtype=torch.long, device=device)
    res = torch.randperm(n, out=out).cpu().numpy()
    np.testing.assert_equal(set(res), set(range(n)))
    np.testing.assert_equal(out.cpu().numpy(), res)

    res2 = torch.randperm(n).cpu().numpy()
    np.testing.assert_equal(set(res2), set(range(n)))

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

  def test_amax(self):
    x = torch.tensor([[[ 1.5,  2.3,  3.1,  4.7],
                       [ 5.2,  6.8,  7.4,  12.9],
                       [ 9.0, 12.3, 11.6, 10.1]],
                      [[13.2, 16.9, 15.5, 14.1],
                       [17.1, 24.9, 19.8, 20.2],
                       [21.0, 22.3, 23.6, 18.4]]], device=device)

    y1 = torch.amax(x)
    expected = np.array([24.9], dtype=np.float32)
    np.testing.assert_equal(y1.cpu().numpy(), expected)

    y2 = torch.amax(x, dim=(1,2))
    expected = np.array([12.9, 24.9], dtype=np.float32)
    np.testing.assert_equal(y2.cpu().numpy(), expected)

    y3 = torch.amax(x, dim=2)
    expected = np.array([[4.7, 12.9, 12.3], [16.9, 24.9, 23.6]], dtype=np.float32)
    np.testing.assert_equal(y3.cpu().numpy(), expected)

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

  def test_isin_tensor_tensor_out(self):
    a = torch.tensor([1, 2, 3], device=device)
    b = torch.tensor([2, 4], device=device)
    expected_base = torch.tensor([False, True, False], device=device)
    for assume_unique in [False, True]:
      for invert, expected in [(False, expected_base), (True, ~expected_base)]:
        out = torch.empty_like(a, dtype=torch.bool)
        res = torch.ops.aten.isin.Tensor_Tensor_out(a, b, invert=invert, assume_unique=assume_unique, out=out)
        np.testing.assert_equal(out.cpu().numpy(), expected.cpu().numpy())

  def test_uniform(self):
    for torch_dtype in [torch.float32, torch.float16]:
      a = torch.rand(10, 10, device=device, dtype=torch_dtype)
      self.assertEqual(a.dtype, torch_dtype)

  def test_normal(self):
    for torch_dtype in [torch.float32, torch.float16]:
      a = torch.randn(10, 10, device=device, dtype=torch_dtype)
      self.assertEqual(a.dtype, torch_dtype)

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

  def test_mnist_index(self):
    with Context(FUSE_ARANGE=1, SPLIT_REDUCEOP=0):
      GlobalCounters.reset()
      from tinygrad.nn.datasets import mnist
      X_train, Y_train, _, _ = mnist()
      X_train = torch.tensor(X_train.float().numpy(), device=device)
      Y_train = torch.tensor(Y_train.cast('int64').numpy(), device=device)
      samples = torch.randint(0, X_train.shape[0], (32,))
      X,Y = X_train[samples], Y_train[samples]
      X.cpu(), Y.cpu()
      self.assertLessEqual(GlobalCounters.global_ops, 10_000_000)

if __name__ == "__main__":
  unittest.main()
