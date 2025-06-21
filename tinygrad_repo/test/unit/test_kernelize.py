import unittest
from tinygrad import Tensor
from tinygrad.uop import Ops

class TestKernelize(unittest.TestCase):
  def test_add_reshaped(self):
    a = Tensor.ones(16,16).contiguous()
    b = Tensor.zeros(16,16).contiguous()
    ret = (a+b).sum(axis=1)
    ret_reshaped_1 = ret.reshape(4,4)
    ret_reshaped_2 = ret.reshape(2,8)
    ret.kernelize()
    self.assertIs(ret_reshaped_1.uop.src[0], ret_reshaped_2.uop.src[0])

  def test_two_reduce(self):
    a = Tensor.ones(16,16).contiguous()
    a1 = a.sum(axis=1)
    a0 = a1.sum(axis=0)
    a0.kernelize()
    self.assertIs(a1.uop.base.op, Ops.ASSIGN)

  def test_two_reduce_w_add(self):
    a = Tensor.ones(16,16).contiguous()
    a1 = a.sum(axis=1)
    a0 = (a1+1).sum(axis=0)
    a0.kernelize()
    # NOTE: the +1 is fused with a1, so a1 is not kernelized
    self.assertIs(a1.uop.base.op, Ops.REDUCE_AXIS)
    # the input to the REDUCE_AXIS is an ASSIGN though
    self.assertIs(a1.uop.base.src[0].base.op, Ops.ASSIGN)

if __name__ == '__main__':
  unittest.main()
