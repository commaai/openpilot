# simple tests
import unittest
import torch
import numpy as np
from tinygrad.helpers import getenv, GlobalCounters
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

  def test_numpy_ones_int32(self):
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


  def test_amin(self):
    x = torch.tensor([[[ 1.5,  2.3,  3.1,  4.7],
                       [ 5.2,  6.8,  7.4,  12.9],
                       [ 9.0, 12.3, 11.6, 10.1]],
                      [[13.2, 16.9, 15.5, 14.1],
                       [17.1, 24.9, 19.8, 20.2],
                       [21.0, 22.3, 23.6, 18.4]]], device=device)

    y1 = torch.amin(x)
    expected = np.array([1.5], dtype=np.float32)
    np.testing.assert_equal(y1.cpu().numpy(), expected)

    y2 = torch.amin(x, dim=(1,2))
    expected = np.array([1.5, 13.2], dtype=np.float32)
    np.testing.assert_equal(y2.cpu().numpy(), expected)

    y3 = torch.amin(x, dim=2)
    expected = np.array([[1.5, 5.2, 9.0], [13.2, 17.1, 18.4]], dtype=np.float32)
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
    x = torch.arange(3*3, dtype=torch.float32, device=device).reshape(1, 1, 3, 3).requires_grad_(True)
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

  def test_equal(self):
    tensor_a = torch.tensor([[1, 2], [3, 4]], device=device)
    tensor_b = torch.tensor([[1, 2], [3, 4]], device=device)
    tensor_c = torch.tensor([[1, 2], [1, 2]], device=device)
    assert torch.equal(tensor_a, tensor_b)
    assert not torch.equal(tensor_a, tensor_c)

  @unittest.skip("# TODO: this test is slow")
  def test_linalg_svd(self):
    A = torch.randn(5, 5, device=device)
    U, S, Vh = torch.linalg.svd(A)
    np.testing.assert_equal(U.shape, (5,5))
    np.testing.assert_equal(Vh.shape, (5,5))
    np.testing.assert_allclose(torch.dist(A, U @ torch.diag(S) @ Vh).cpu().numpy(), 0, atol=1e-5)

    A = torch.randn(5, 3, device=device)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    np.testing.assert_equal(U.shape, (5,3))
    np.testing.assert_equal(Vh.shape, (3,3))
    np.testing.assert_allclose(torch.dist(A, U @ torch.diag(S) @ Vh).cpu().numpy(), 0, atol=1e-5)

  def test_linalg_eigh(self):
    a = torch.tensor([[1, 2], [2, 1]], dtype=torch.float32, device=device)
    w, v = torch.linalg.eigh(a)
    np.testing.assert_equal(w.cpu().numpy(), [-1, 3])
    recon = (v @ torch.diag(w) @ v.T).cpu().numpy()
    np.testing.assert_allclose(recon, a.cpu().numpy(), atol=1e-6)

  def test_linalg_det(self):
    a = torch.diag(torch.tensor([1,2,3,4,5], dtype = torch.float32, device=device))
    b = torch.linalg.det(a)
    np.testing.assert_equal(b.cpu().numpy(), 120.0)

  def test_linalg_cross(self):
    a = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32, device=device)
    b = torch.tensor([[0, 0, 1]], dtype=torch.float32, device=device)
    cross = torch.linalg.cross(a, b)
    np.testing.assert_equal(cross.cpu().numpy(), np.array([[0, -1, 0], [1, 0, 0]], dtype=np.float32))

  def test_scalar_assign(self):
    a = torch.tensor([1, 2, 3], device=device)
    a[1] = 4
    np.testing.assert_equal(a.cpu().numpy(), [1, 4, 3])

  @unittest.skip("meh")
  def test_str(self):
    a = torch.ones(4, device=device)
    print(str(a))

  def test_floor_div(self):
    a = torch.tensor([10., 7., 5.], device=device)
    b = torch.tensor([3., 2., 2.], device=device)
    result = a // b
    np.testing.assert_equal(result.cpu().numpy(), [3., 3., 2.])

  def test_mnist_index(self):
    GlobalCounters.reset()
    from tinygrad.nn.datasets import mnist
    X_train, Y_train, _, _ = mnist()
    X_train = torch.tensor(X_train.float().numpy(), device=device)
    Y_train = torch.tensor(Y_train.cast('int64').numpy(), device=device)
    samples = torch.randint(0, X_train.shape[0], (32,))
    X,Y = X_train[samples], Y_train[samples]
    X.cpu(), Y.cpu()
    self.assertLessEqual(GlobalCounters.global_ops, 10_000_000)

  def _test_diagonal(self, *shape):
    a = torch.randn(*shape, dtype=torch.float32, device=device)
    ref = np.diagonal(a.cpu().numpy(), axis1=-2, axis2=-1)
    diag = torch.linalg.diagonal(a)
    np.testing.assert_equal(diag.cpu().numpy(), ref)
    np.testing.assert_equal(diag[-1].cpu().numpy(), ref[-1])

  def test_diagonal_cube(self): self._test_diagonal(3, 3, 3)
  def test_diagonal_rectangular(self): self._test_diagonal(4, 5, 6)
  def test_diagonal_4d(self): self._test_diagonal(2, 3, 4, 5)

  def test_pad_circular_simple(self):
    a = torch.arange(4, dtype=torch.float32, device=device).reshape(1,1,2,2)
    padded = torch.nn.functional.pad(a, (1,1,1,1), mode="circular")
    expected = np.array([[[[3.,2.,3.,2.], [1.,0.,1.,0.], [3.,2.,3.,2.], [1.,0.,1.,0.]]]], dtype=np.float32)
    np.testing.assert_allclose(padded.cpu().numpy(), expected)

  def test_pad_circular_backward(self):
    a = torch.arange(4, dtype=torch.float32, device=device).reshape(1,1,2,2).requires_grad_(True)
    padded = torch.nn.functional.pad(a, (1,1,1,1), mode="circular")
    loss = padded.sum()
    loss.backward()
    expected_grad = np.array([[[[4., 4.], [4., 4.]]]], dtype=np.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad)


  def test_matmul_backward(self):
    x = torch.randn(3, 4, device=device, dtype=torch.float32, requires_grad=True)
    y = torch.randn(4, 5, device=device, dtype=torch.float32, requires_grad=True)
    z = (x @ y).sum()
    z.backward()
    assert x.grad is not None
    assert y.grad is not None
    assert x.grad.shape == x.shape
    assert y.grad.shape == y.shape

  def test_matmul_broadcast_backward(self):
    x = torch.randn(2, 3, 4, device=device, dtype=torch.float32, requires_grad=True)
    y = torch.randn(4, 5, device=device, dtype=torch.float32, requires_grad=True)
    z = (x @ y).sum()
    z.backward()
    assert x.grad is not None
    assert y.grad is not None
    assert x.grad.shape == x.shape
    assert y.grad.shape == y.shape

  def test_diag_vector_to_matrix(self):
    vec = torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float32, device=device)
    mat = torch.diag(vec)
    expected = np.diag([1., 2., 3., 4., 5.])
    np.testing.assert_allclose(mat.cpu().numpy(), expected, rtol=1e-5)
    assert mat.shape == (5, 5)

  def test_diagonal_matrix_to_vector(self):
    mat = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=torch.float32, device=device)
    vec = torch.linalg.diagonal(mat)
    expected = np.array([1., 5., 9.])
    np.testing.assert_allclose(vec.cpu().numpy(), expected, rtol=1e-5)
    assert vec.shape == (3,)

  def test_permute_2(self):
    a = torch.randn(2, 3, 4, dtype=torch.float32, device=device)
    b = a.permute(2, 0, 1)
    assert b.shape == (4, 2, 3)
    np.testing.assert_equal(b.cpu().numpy(), a.cpu().numpy().transpose(2, 0, 1))

  def test_batchnorm_unsqueeze(self):
    bn = torch.nn.BatchNorm2d(4).to(device)
    x = torch.randn(8, 4, 3, 3, device=device)
    out = bn(x)
    self.assertEqual(out.shape, x.shape)

  def test_slice_inplace_zero(self):
    a = torch.ones((3, 3), device=device)
    b = a[1:, 1:]
    b.zero_()
    expected = np.array([[1., 1., 1.],
                         [1., 0., 0.],
                         [1., 0., 0.]])
    np.testing.assert_equal(a.cpu().numpy(), expected)

  def test_slice_inplace_fill(self):
    a = torch.ones((3, 3), device=device)
    b = a[1:, 1:]
    b.fill_(5.0)
    expected = np.array([[1., 1., 1.],
                         [1., 5., 5.],
                         [1., 5., 5.]])
    np.testing.assert_equal(a.cpu().numpy(), expected)

  def test_fill_tensor_value(self):
    a = torch.zeros((2, 2), dtype=torch.float32, device=device)
    value = torch.tensor(3, dtype=torch.int64, device=device)
    a.fill_(value)
    expected = np.full((2, 2), 3, dtype=np.float32)
    np.testing.assert_equal(a.cpu().numpy(), expected)

  def test_slice_inplace_mul(self):
    a = torch.ones((3, 3), device=device)
    b = a[1:, 1:]
    b *= 2
    expected = np.array([[1., 1., 1.],
                         [1., 2., 2.],
                         [1., 2., 2.]])
    np.testing.assert_equal(a.cpu().numpy(), expected)

  def test_permute_slice_zero(self):
    a = torch.ones((3, 3), device=device)
    b = a[1:, 1:].permute(1, 0)
    b.zero_()
    expected = np.array([[1., 1., 1.],
                         [1., 0., 0.],
                         [1., 0., 0.]])
    np.testing.assert_equal(a.cpu().numpy(), expected)

  def test_permute_slice_mul(self):
    a = torch.ones((3, 3), device=device)
    b = a[1:, 1:].permute(1, 0)
    b *= 2
    expected = np.array([[1., 1., 1.],
                         [1., 2., 2.],
                         [1., 2., 2.]])
    np.testing.assert_equal(a.cpu().numpy(), expected)

  def test_simple_slice_setitem(self):
    a = torch.tensor([10, 20, 30], device=device)
    a[1] = 99
    np.testing.assert_equal(a.cpu().numpy(), [10, 99, 30])

  def test_2d_slice_setitem(self):
    a = torch.zeros((3, 3), device=device)
    a[1, 2] = 99
    self.assertEqual(a[1, 2].item(), 99)
    self.assertEqual(a.sum().item(), 99)

  def test_view_copy(self):
    a = torch.tensor([10, 20, 30], device=device)
    view = a[1]
    view.copy_(torch.tensor(88, device=device))
    np.testing.assert_equal(a.cpu().numpy(), [10, 88, 30])

  def test_diag_2d_input(self):
    a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device)
    d = torch.diag(a)
    np.testing.assert_equal(d.cpu().numpy(), [1, 5, 9])

  def test_diag_1d_input(self):
    a = torch.tensor([1, 2, 3], device=device)
    d = torch.diag(a)
    expected = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    np.testing.assert_equal(d.cpu().numpy(), expected)

  def test_permute_view_tracking(self):
    a = torch.ones((2, 3, 4), device=device)
    b = a.permute(2, 0, 1)
    self.assertEqual(b.shape, (4, 2, 3))

  def test_detach_view_creation(self):
    a = torch.tensor([1.0, 2.0, 3.0], device=device)
    b = a.detach()
    np.testing.assert_equal(b.cpu().numpy(), [1.0, 2.0, 3.0])

  def test_view_zero_inplace(self):
    a = torch.ones((4, 4), device=device)
    view = a[1:3, 1:3]
    view.zero_()
    self.assertEqual(view.sum().item(), 0)

  def test_view_fill_inplace(self):
    a = torch.zeros((4, 4), device=device)
    view = a[1:3, 1:3]
    view.fill_(5)
    self.assertEqual(view.sum().item(), 20)

  def test_permute_contiguous(self):
    a = torch.tensor([[1, 2], [3, 4]], device=device)
    b = a.permute(1, 0)
    c = b.contiguous()
    expected = [[1, 3], [2, 4]]
    np.testing.assert_equal(c.cpu().numpy(), expected)

  def test_diag_2d_extract_diagonal(self):
    a = torch.tensor([[1, 2], [3, 4]], device=device)
    result = torch.diag(a)
    np.testing.assert_equal(result.cpu().numpy(), [1, 4])

  def test_slice_inplace_multiply_offset_preservation(self):
    a = torch.tensor([1, 2, 3], device=device)
    a[1:] *= 2
    np.testing.assert_equal(a.cpu().numpy(), [1, 4, 6])

  def test_slice_inplace_mul_pattern(self):
    a = torch.tensor([1, 2, 3, 4], device=device)
    a[:2] *= 3
    a[2:] *= 2
    np.testing.assert_equal(a.cpu().numpy(), [3, 6, 6, 8])

  def test_chained_slice_column(self):
    a = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    torch_res = a[:, 1:2][:, 0:1].cpu().numpy()
    cpu_res = torch.arange(16, dtype=torch.float32).reshape(4, 4)[:, 1:2][:, 0:1].numpy()
    np.testing.assert_equal(torch_res, cpu_res)

  def test_slice_with_step(self):
    a = torch.arange(20, dtype=torch.float32, device=device)
    torch_res = a[::2][1:4].cpu().numpy()
    cpu_res = torch.arange(20, dtype=torch.float32)[::2][1:4].numpy()
    np.testing.assert_equal(torch_res, cpu_res)

  def test_slice_negative_dim(self):
    a = torch.arange(13, dtype=torch.int32, device=device).repeat(8, 1)
    torch_chunks = a.chunk(3, -1)
    cpu_chunks = torch.arange(13, dtype=torch.int32).repeat(8, 1).chunk(3, -1)
    assert len(torch_chunks) == len(cpu_chunks)
    for i in range(len(torch_chunks)):
      np.testing.assert_equal(torch_chunks[i].cpu().numpy(), cpu_chunks[i].numpy())

  def test_dot_vector_matrix(self):
    a = torch.arange(65, dtype=torch.float32, device=device)
    b = torch.arange(65*45, dtype=torch.float32, device=device).reshape(65, 45)
    torch_res = a.matmul(b).reshape(-1).cpu().numpy()
    cpu_res = torch.arange(65, dtype=torch.float32).matmul(torch.arange(65*45, dtype=torch.float32).reshape(65, 45)).numpy()
    np.testing.assert_equal(torch_res, cpu_res)

  def test_alias_passthrough(self):
    a = torch.randn(3, 3, device=device)
    alias_view = torch.ops.aten.alias(a)
    alias_view += 1
    np.testing.assert_equal(a.cpu().numpy(), alias_view.cpu().numpy())

  def test_split_simple_vector(self):
    a = torch.arange(10, dtype=torch.float32, device=device)
    torch_chunks = a.split([1,4,5])
    cpu_chunks = torch.arange(10, dtype=torch.float32).split([1,4,5])
    for tc, cc in zip(torch_chunks, cpu_chunks):
      np.testing.assert_equal(tc.cpu().numpy(), cc.cpu().numpy())

  def test_split_matches_torch(self):
    a = torch.arange(10, dtype=torch.float32, device=device)
    torch_chunks = a.split([1,4,5])
    tiny_chunks = [chunk.cpu().numpy() for chunk in torch_chunks]
    cpu_chunks = [torch.arange(10, dtype=torch.float32).split([1,4,5])[i].numpy() for i in range(3)]
    for tr, cr in zip(tiny_chunks, cpu_chunks): np.testing.assert_equal(tr, cr)

  def test_sum_matches_torch(self):
    a = torch.arange(6, dtype=torch.float32, device=device).reshape(2,3)
    torch_res = a.sum().cpu().numpy()
    cpu_res = torch.arange(6, dtype=torch.float32).reshape(2,3).sum().numpy()
    np.testing.assert_equal(torch_res, cpu_res)

  def test_view_matches_torch(self):
    a = torch.arange(6, dtype=torch.float32, device=device)
    torch_res = a.view(2, 3).cpu().numpy()
    cpu_res = torch.arange(6, dtype=torch.float32).view(2, 3).numpy()
    np.testing.assert_equal(torch_res, cpu_res)

  def test_view_zero_with_indices(self):
    a = torch.tensor([1, 2, 3, 4], device=device)
    a[1:3].zero_()
    np.testing.assert_equal(a.cpu().numpy(), [1, 0, 0, 4])

  def test_view_fill_with_indices(self):
    a = torch.tensor([1, 2, 3, 4], device=device)
    a[::2].fill_(9)
    np.testing.assert_equal(a.cpu().numpy(), [9, 2, 9, 4])

  def test_nested_slice_inplace_ops(self):
    a = torch.tensor([1, 2, 3, 4, 5, 6], device=device)
    a[:3] += 10
    a[3:] *= 2
    np.testing.assert_equal(a.cpu().numpy(), [11, 12, 13, 8, 10, 12])

  def test_diag_1d(self):
    a = torch.tensor([1, 2, 3], device=device)
    result = torch.diag(a)
    expected = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    np.testing.assert_equal(result.cpu().numpy(), expected)

  def test_diag_backward(self):
    a = torch.randn(5, dtype=torch.float32, device=device, requires_grad=True)
    b = torch.diag(a)
    b.sum().backward()
    assert a.grad is not None

  def test_diagonal(self):
    a = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=torch.float32, device=device, requires_grad=True)
    b = torch.diagonal(a)
    expected = torch.tensor([1., 5., 9.], dtype=torch.float32)
    self.assertEqual(b.shape, (3,))
    np.testing.assert_allclose(b.detach().cpu().numpy(), expected.numpy(), rtol=1e-5)

  def test_diagonal_backward(self):
    a = torch.randn(5, 5, dtype=torch.float32, device=device, requires_grad=True)
    b = torch.diagonal(a)
    b.sum().backward()
    assert a.grad is not None

  def test_expand_backward(self):
    a = torch.randn(4, 3, 1, 6, dtype=torch.float32, device=device, requires_grad=True)
    b = a.expand(4, 3, 2, 6)
    b.sum().backward()
    assert a.grad is not None

  def test_einsum_backward(self):
    a = torch.randn(10, 10, dtype=torch.float32, device=device, requires_grad=True)
    b = torch.einsum('ij->ji', a)
    b.sum().backward()
    assert a.grad is not None

  def test_diag_backward_gradient_values(self):
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device, requires_grad=True)
    b = torch.diag(a)
    loss = b.sum()
    loss.backward()
    expected_grad = torch.ones(3, dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_diag_backward_gradient_values_2d_to_1d(self):
    a = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]], dtype=torch.float32, device=device, requires_grad=True)
    b = torch.diagonal(a)
    loss = b.sum()
    loss.backward()
    expected_grad = torch.tensor([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]], dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_expand_backward_gradient_values(self):
    a = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32, device=device, requires_grad=True)
    b = a.expand(3, 4)
    loss = b.sum()
    loss.backward()
    expected_grad = torch.tensor([[4.0], [4.0], [4.0]], dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_expand_backward_with_leading_dims(self):
    a = torch.tensor([[1.0, 2.0]], dtype=torch.float32, device=device, requires_grad=True)
    b = a.expand(3, 1, 2)
    loss = b.sum()
    loss.backward()
    expected_grad = torch.tensor([[3.0, 3.0]], dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_diag_2d_to_1d_backward(self):
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device, requires_grad=True)
    b = torch.diag(a)
    loss = b.sum()
    loss.backward()
    expected_grad = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_expand_complex_backward(self):
    a = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32, device=device, requires_grad=True)
    b = a.expand(2, 3, 2)
    loss = b.sum()
    loss.backward()
    expected_grad = torch.tensor([[[6.0, 6.0]]], dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_diag_backward_with_scaling(self):
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device, requires_grad=True)
    b = torch.diag(a)
    loss = (b * torch.tensor([[2.0, 0.0, 0.0],
                               [0.0, 3.0, 0.0],
                               [0.0, 0.0, 4.0]], device=device)).sum()
    loss.backward()
    expected_grad = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_repeat_basic(self):
    a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
    b = a.repeat(2, 1)
    expected = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.float32)
    np.testing.assert_equal(b.cpu().numpy(), expected.numpy())

  def test_repeat_multidim(self):
    a = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
    b = a.repeat(2, 3)
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3).repeat(2, 3)
    np.testing.assert_equal(b.cpu().numpy(), expected.numpy())

  def test_repeat_backward(self):
    a = torch.tensor([[1.0, 2.0]], dtype=torch.float32, device=device, requires_grad=True)
    b = a.repeat(3, 2)
    loss = b.sum()
    loss.backward()
    expected_grad = torch.tensor([[6.0, 6.0]], dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_cumsum_1d(self):
    a = torch.tensor([1, 2, 3, 4], dtype=torch.float32, device=device)
    b = torch.cumsum(a, dim=0)
    expected = torch.tensor([1, 3, 6, 10], dtype=torch.float32)
    np.testing.assert_equal(b.cpu().numpy(), expected.numpy())

  def test_cumsum_2d(self):
    a = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
    b = torch.cumsum(a, dim=0)
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4).cumsum(dim=0)
    np.testing.assert_equal(b.cpu().numpy(), expected.numpy())

    c = torch.cumsum(a, dim=1)
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4).cumsum(dim=1)
    np.testing.assert_equal(c.cpu().numpy(), expected.numpy())

  def test_cumsum_backward(self):
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=device, requires_grad=True)
    b = torch.cumsum(a, dim=0)
    loss = b.sum()
    loss.backward()
    expected_grad = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_constant_pad_nd_1d(self):
    a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
    b = torch.nn.functional.pad(a, (1, 2), mode='constant', value=0)
    expected = torch.tensor([0, 1, 2, 3, 0, 0], dtype=torch.float32)
    np.testing.assert_equal(b.cpu().numpy(), expected.numpy())

  def test_constant_pad_nd_2d(self):
    a = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
    b = torch.nn.functional.pad(a, (1, 1, 1, 1), mode='constant', value=0)
    expected = torch.nn.functional.pad(torch.arange(6, dtype=torch.float32).reshape(2, 3), (1, 1, 1, 1), mode='constant', value=0)
    np.testing.assert_equal(b.cpu().numpy(), expected.numpy())

  def test_constant_pad_nd_2d_backward(self):
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device, requires_grad=True)
    b = torch.nn.functional.pad(a, (1, 1, 1, 1), mode='constant', value=0)
    loss = b.sum()
    loss.backward()
    expected_grad = torch.ones((2, 2), dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_negative_strides_cumsum_backward(self):
    a = torch.randn(5, device=device, requires_grad=True)
    b = torch.cumsum(a, dim=0)
    b.sum().backward()
    grad = a.grad.cpu().numpy()
    self.assertEqual(len(grad), 5)

  def test_cumsum_fix_gradient_values(self):
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=device, requires_grad=True)
    b = torch.cumsum(a, dim=0)
    loss = b.sum()
    loss.backward()
    expected = np.array([4.0, 3.0, 2.0, 1.0])
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected, rtol=1e-5)

  def test_cumsum_arange_large(self):
    # Tests cumsum with an unrealized arange input with size > 512 (the split threshold)
    # This exercises the _split_cumalu path which uses a two-stage algorithm
    for size in [513, 1022]:
      a = torch.arange(size, dtype=torch.float32, device=device)
      result = torch.cumsum(a, dim=0)
      expected = torch.arange(size, dtype=torch.float32).cumsum(dim=0)
      np.testing.assert_allclose(result.cpu().numpy(), expected.numpy(), rtol=1e-5)

  def test_diag_1d_to_2d(self):
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device, requires_grad=True)
    b = torch.diag(a)
    expected = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    np.testing.assert_equal(b.detach().cpu().numpy(), expected)

  def test_diag_2d_to_1d(self):
    c = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, device=device)
    d = torch.diag(c)
    np.testing.assert_equal(d.cpu().numpy(), [1, 5, 9])

  def test_biased_conv2d(self):
    # Test case for two sequential conv2d with same weights/bias and ReLU in between, this is as special case from test_ops.py
    torch.manual_seed(0)
    C = 8
    x_cpu = torch.randn(1, C, 5, 5, requires_grad=True)
    w_cpu = torch.randn(C, C, 1, 1, requires_grad=True)
    b_cpu = torch.randn(C, requires_grad=True)
    x_tiny = x_cpu.detach().to(device).requires_grad_(True)
    w_tiny = w_cpu.detach().to(device).requires_grad_(True)
    b_tiny = b_cpu.detach().to(device).requires_grad_(True)
    out_cpu = torch.nn.functional.conv2d(torch.nn.functional.conv2d(x_cpu, w_cpu, b_cpu).relu(), w_cpu, b_cpu)
    out_tiny = torch.nn.functional.conv2d(torch.nn.functional.conv2d(x_tiny, w_tiny, b_tiny).relu(), w_tiny, b_tiny)
    grad_out = torch.randn_like(out_cpu)
    out_cpu.backward(grad_out)
    out_tiny.backward(grad_out.to(device))
    np.testing.assert_allclose(x_tiny.grad.cpu().numpy(), x_cpu.grad.numpy(), atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(w_tiny.grad.cpu().numpy(), w_cpu.grad.numpy(), atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(b_tiny.grad.cpu().numpy(), b_cpu.grad.numpy(), atol=1e-4, rtol=1e-3)


from tinygrad import Tensor
class TestBackendHelpers(unittest.TestCase):

  def test_calculate_storage_offset_no_shrink(self):
    t = Tensor.ones(3, 4)
    assert extra.torch_backend.backend.calculate_storage_offset(t) == 0

  def test_calculate_storage_offset_with_shrink(self):
    t = Tensor.ones(10, 10)[2:5, 3:7]
    # strides for (10, 10) are [10, 1]
    # offset = 2*10 + 3*1 = 23
    assert extra.torch_backend.backend.calculate_storage_offset(t) == 23

  def test_calculate_storage_offset_multiple_shrinks(self):
    t = Tensor.ones(5, 6, 7)[1:3, 2:4, 3:5]
    # strides for (5, 6, 7) are [42, 7, 1]
    # offset = 1*42 + 2*7 + 3*1 = 42 + 14 + 3 = 59
    assert extra.torch_backend.backend.calculate_storage_offset(t) == 59

  def test_calculate_storage_offset_with_reshape(self):
    t = Tensor.ones(10, 10)
    orig_offset = extra.torch_backend.backend.calculate_storage_offset(t)
    assert orig_offset == 0
    t = t.reshape(100)
    assert extra.torch_backend.backend.calculate_storage_offset(t) == orig_offset

  def test_slice_values_match_torch(self):
    torch_cpu = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    torch_tiny = torch_cpu.to(device)
    sliced_cpu = torch_cpu[2:5, 3:7]
    sliced_tiny = torch_tiny[2:5, 3:7]
    np.testing.assert_equal(sliced_tiny.cpu().numpy(), sliced_cpu.numpy())

  def test_slice_values_match_torch_3d(self):
    torch_cpu_3d = torch.arange(210, dtype=torch.float32).reshape(5, 6, 7)
    torch_tiny_3d = torch_cpu_3d.to(device)
    sliced_cpu_3d = torch_cpu_3d[1:3, 2:4, 3:5]
    sliced_tiny_3d = torch_tiny_3d[1:3, 2:4, 3:5]
    np.testing.assert_equal(sliced_tiny_3d.cpu().numpy(), sliced_cpu_3d.numpy())

  def test_topk_out(self):
    a = torch.tensor([1, 3, 2, 4], device=device)
    values = torch.empty(2, device=device)
    indices = torch.empty(2, dtype=torch.int64, device=device)
    ret_values, ret_indices = torch.topk(a, k=2, out=(values, indices))
    np.testing.assert_equal(values.cpu().numpy(), [4, 3])
    np.testing.assert_equal(indices.cpu().numpy(), [3, 1])
    assert ret_values is values
    assert ret_indices is indices

  def test_sort_out(self):
    a = torch.tensor([3, 1, 4, 2], device=device)
    values = torch.empty(4, device=device)
    indices = torch.empty(4, dtype=torch.int64, device=device)
    ret_values, ret_indices = torch.sort(a, out=(values, indices))
    np.testing.assert_equal(values.cpu().numpy(), [1, 2, 3, 4])
    np.testing.assert_equal(indices.cpu().numpy(), [1, 3, 0, 2])
    assert ret_values is values
    assert ret_indices is indices

  def test_cat_out(self):
    a = torch.tensor([1, 2], device=device)
    b = torch.tensor([3, 4], device=device)
    out = torch.empty(4, device=device)
    ret = torch.cat([a, b], out=out)
    np.testing.assert_equal(out.cpu().numpy(), [1, 2, 3, 4])
    assert ret is out

  def test_scatter_add_out(self):
    src = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device, dtype=torch.float32)
    index = torch.tensor([[0, 1, 2], [0, 1, 2]], device=device)
    input = torch.zeros(3, 3, device=device, dtype=torch.float32)
    out = torch.zeros(3, 3, device=device, dtype=torch.float32)
    ret = torch.scatter_add(input, 0, index, src, out=out)
    expected = torch.tensor([[5, 0, 0], [0, 7, 0], [0, 0, 9]], dtype=torch.float32)
    np.testing.assert_allclose(out.cpu().numpy(), expected.cpu().numpy())
    assert ret is out

  def test_floor_divide_inplace_identity(self):
    x = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device=device)
    y = torch.tensor([2, 4, 5, 8], dtype=torch.int32, device=device)
    ret = x.floor_divide_(y)
    assert ret is x
    np.testing.assert_equal(x.cpu().numpy(), [5, 5, 6, 5])

  def test_lshift_inplace_identity(self):
    x = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=device)
    ret = x.__ilshift__(2)
    assert ret is x
    np.testing.assert_equal(x.cpu().numpy(), [4, 8, 12, 16])

  def test_rshift_inplace_identity(self):
    x = torch.tensor([16, 32, 48, 64], dtype=torch.int32, device=device)
    ret = x.__irshift__(2)
    assert ret is x
    np.testing.assert_equal(x.cpu().numpy(), [4, 8, 12, 16])

  def test_relu_inplace_identity(self):
    x = torch.tensor([-1.0, 2.0, -3.0, 4.0], device=device)
    ret = x.relu_()
    assert ret is x
    np.testing.assert_equal(x.cpu().numpy(), [0.0, 2.0, 0.0, 4.0])

  def test_random_inplace_identity(self):
    x = torch.zeros(10, dtype=torch.int32, device=device)
    ret = x.random_()
    assert ret is x
    assert x.shape == (10,)

  def test_random_from_inplace_identity(self):
    x = torch.zeros(10, dtype=torch.int32, device=device)
    ret = x.random_(5, 10)
    assert ret is x
    # values should be in range [5, 10)
    assert torch.all(x >= 5).item() and torch.all(x < 10).item()

  def test_uniform_inplace_identity(self):
    x = torch.zeros(10, device=device)
    ret = x.uniform_(0.0, 1.0)
    assert ret is x
    # values should be in range [0, 1)
    assert torch.all(x >= 0.0).item() and torch.all(x < 1.0).item()

  def test_normal_inplace_identity(self):
    x = torch.zeros(100, device=device)
    ret = x.normal_(0.0, 1.0)
    assert ret is x
    # just check that values changed from zeros
    assert not torch.all(x == 0.0).item()

  def test_logical_or_inplace_identity(self):
    x = torch.tensor([True, False, True, False], device=device)
    y = torch.tensor([False, False, True, True], device=device)
    ret = x.logical_or_(y)
    assert ret is x
    np.testing.assert_equal(x.cpu().numpy(), [True, False, True, True])

  def test_masked_fill_scalar_inplace_identity(self):
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    mask = torch.tensor([True, False, True, False], device=device)
    ret = x.masked_fill_(mask, 0.0)
    assert ret is x
    np.testing.assert_equal(x.cpu().numpy(), [0.0, 2.0, 0.0, 4.0])

  def test_masked_fill_tensor_inplace_identity(self):
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    mask = torch.tensor([True, False, True, False], device=device)
    value = torch.tensor(99.0, device=device)
    ret = x.masked_fill_(mask, value)
    assert ret is x
    np.testing.assert_equal(x.cpu().numpy(), [99.0, 2.0, 99.0, 4.0])

  def test_zero_inplace_identity(self):
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    ret = x.zero_()
    assert ret is x
    np.testing.assert_equal(x.cpu().numpy(), [0.0, 0.0, 0.0, 0.0])

  def test_fill_scalar_inplace_identity(self):
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    ret = x.fill_(5.0)
    assert ret is x
    np.testing.assert_equal(x.cpu().numpy(), [5.0, 5.0, 5.0, 5.0])

  def test_fill_tensor_inplace_identity(self):
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    value = torch.tensor(7.0, device=device)
    ret = x.fill_(value)
    assert ret is x
    np.testing.assert_equal(x.cpu().numpy(), [7.0, 7.0, 7.0, 7.0])

  def test_add_tensor_inplace_identity(self):
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    y = torch.tensor([10.0, 20.0, 30.0, 40.0], device=device)
    ret = x.add_(y)
    assert ret is x
    np.testing.assert_equal(x.cpu().numpy(), [11.0, 22.0, 33.0, 44.0])

  def test_add_scalar_inplace_identity(self):
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    ret = x.add_(10.0)
    assert ret is x
    np.testing.assert_equal(x.cpu().numpy(), [11.0, 12.0, 13.0, 14.0])

  def test_mul_tensor_inplace_identity(self):
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    y = torch.tensor([2.0, 3.0, 4.0, 5.0], device=device)
    ret = x.mul_(y)
    assert ret is x
    np.testing.assert_equal(x.cpu().numpy(), [2.0, 6.0, 12.0, 20.0])

  def test_mul_scalar_inplace_identity(self):
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    ret = x.mul_(2.0)
    assert ret is x
    np.testing.assert_equal(x.cpu().numpy(), [2.0, 4.0, 6.0, 8.0])

if __name__ == "__main__":
  unittest.main()
