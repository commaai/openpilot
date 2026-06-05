import unittest
import torch
import tinygrad.nn.torch
torch.set_default_device("tiny")
import numpy as np

class TestTorchBackendInplace(unittest.TestCase):
  def test_zero(self):
    a = torch.ones(4)
    a.zero_()
    np.testing.assert_equal(a.cpu().numpy(), [0,0,0,0])

  def test_view_zero(self):
    a = torch.ones(4)
    a.view((2, 2)).zero_()
    np.testing.assert_equal(a.cpu().numpy(), [0,0,0,0])

  def test_slice_zero(self):
    a = torch.ones(4)
    a[2:].zero_()
    np.testing.assert_equal(a.cpu().numpy(), [1,1,0,0])

  def test_slice_permute_zero(self):
    a = torch.ones((3,2))
    a.permute(1,0)[1:].zero_()
    np.testing.assert_equal(a.cpu().numpy(), [[1,0],[1,0],[1,0]])

  def test_slice_fill(self):
    a = torch.zeros(4)
    a[2:].fill_(2)
    np.testing.assert_equal(a.cpu().numpy(), [0,0,2,2])

  def test_slice_mul(self):
    a = torch.ones(4)
    a[:2] *= 3
    a[2:] *= 2
    np.testing.assert_equal(a.cpu().numpy(), [3,3,2,2])

  def test_stacked_mul(self):
    a = torch.ones((3,3))
    b = a[1:,1:].permute(1,0)
    c = b[1:,:]
    b *= 2
    c *= 3
    np.testing.assert_equal(a.cpu().numpy(), [[1,1,1],[1,2,6],[1,2,6]])

  def test_flatten_reshape_add(self):
    a = torch.zeros((2,2,12,32))
    b = a.flatten()
    c = b.reshape((48,32))
    a += 1
    b += 1
    c += 1
    np.testing.assert_equal(c.cpu().numpy(), torch.full((48,32),3).cpu().numpy())

  def test_noncontig(self):
    a = torch.empty_strided((4,4),(1,4), dtype=torch.int64)
    # self.assertFalse(a.is_contiguous()) # TODO: we are contiguous when it's not required
    a.zero_()
    b = a.view((4,4))
    b[1:3,:] += 1
    np.testing.assert_equal(a.cpu().numpy(), [[0]*4,[1]*4,[1]*4,[0]*4])

  def test_detach(self):
    a = torch.zeros(4)
    d = a.detach()
    d += torch.arange(4)
    np.testing.assert_array_equal(a.cpu(), torch.arange(4).cpu())

  def test_inplace_view_metadata(self):
    a = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    ret = a.squeeze_(0)
    self.assertIs(ret, a)
    self.assertEqual(a.shape, torch.Size([2, 3]))
    ret = a.unsqueeze_(1)
    self.assertIs(ret, a)
    self.assertEqual(a.shape, torch.Size([2, 1, 3]))
    ret = a.transpose_(0, 2)
    self.assertIs(ret, a)
    self.assertEqual(a.shape, torch.Size([3, 1, 2]))

  def test_t_inplace_metadata(self):
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    ret = a.t_()
    self.assertIs(ret, a)
    self.assertEqual(a.shape, torch.Size([3, 2]))
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    np.testing.assert_array_equal(a.cpu().numpy(), expected.cpu().numpy())

  def test_squeeze_matmul(self):
    # squeeze_ is used internally by PyTorch for vector-matrix matmul (unsqueeze -> mm -> squeeze_)
    a = torch.arange(65, dtype=torch.float32)
    b = torch.arange(65*45, dtype=torch.float32).reshape(65, 45)
    result = a.matmul(b)
    self.assertEqual(result.shape, torch.Size([45]))
    # verify correctness
    a_cpu = torch.arange(65, dtype=torch.float32, device='cpu')
    b_cpu = torch.arange(65*45, dtype=torch.float32, device='cpu').reshape(65, 45)
    expected = a_cpu.matmul(b_cpu)
    np.testing.assert_allclose(result.cpu().numpy(), expected.numpy(), rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
  unittest.main()
