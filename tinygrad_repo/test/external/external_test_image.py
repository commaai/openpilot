#!/usr/bin/env python
import os
import unittest
import numpy as np
if 'IMAGE' not in os.environ:
  os.environ['IMAGE'] = '2'
os.environ['GPU'] = '1'
os.environ['OPT'] = '2'
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d

class TestImage(unittest.TestCase):
  def test_create_image(self):
    t = Tensor.ones(128, 128, 1)
    t = t.reshape(128, 32, 4) + 3
    t.realize()
    np.testing.assert_array_equal(t.numpy(), np.ones((128,32,4))*4)

  def test_sum_image(self):
    t1 = Tensor.ones(16, 16, 1).reshape(16, 4, 4) + 3
    t1.realize()
    t1 = t1.sum()
    t1.realize()
    assert t1.numpy() == 16*4*4*4, f"got {t1.numpy()}"

  def test_add_image(self):
    t1 = Tensor.ones(16, 16, 1).reshape(16, 4, 4) + 3
    t2 = Tensor.ones(16, 16, 1).reshape(16, 4, 4) + 4
    t1.realize()
    t2.realize()
    t3 = t1 + t2
    t3.realize()
    np.testing.assert_array_equal(t3.numpy(), np.ones((16,4,4))*9)

  def test_padded_conv(self):
    bs, in_chans, out_chans = 1,12,32
    tiny_conv = Conv2d(in_chans, out_chans, 3, bias=None, padding=1)
    tiny_dat = Tensor.ones(bs, 12, 64, 128)
    tiny_conv(tiny_dat).realize()

  def test_op_conv(self):
    bs, in_chans, out_chans = 1,12,32
    tiny_conv = Conv2d(in_chans, out_chans, 3, bias=None, padding=1)
    tiny_dconv = Conv2d(out_chans, out_chans, 1, bias=None, padding=0)
    tiny_dat = Tensor.ones(bs, 12, 64, 128)
    p2 = tiny_conv(tiny_dat).relu()
    p2 = tiny_dconv(p2)
    p2.realize()

if __name__ == '__main__':
  unittest.main()
