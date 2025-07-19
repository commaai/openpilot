import unittest
from tinygrad import Tensor
from tinygrad.uop.ops import Ops

class TestSimpleSchedule(unittest.TestCase):
  def test_reduce_doesnt_split(self):
    a = Tensor.empty(16,16).sum(axis=1)
    a1 = a.reshape(4,4)
    a2 = a.reshape(16,1,1)
    Tensor.kernelize(a1, a2)
    kernels = [x for x in a1.uop.sink(a2.uop).toposort() if x.op is Ops.KERNEL]
    self.assertEqual(len(kernels), 1)

if __name__ == '__main__':
  unittest.main()
