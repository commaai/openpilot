import unittest
import torch
import tinygrad.frontend.torch
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

if __name__ == "__main__":
  unittest.main()
