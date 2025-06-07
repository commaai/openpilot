import unittest, time
from tinygrad.uop.ops import UOp
from tinygrad.dtype import dtypes

# it's about 1 ms per 1k UOps on M3
N = 10000

class TestMicrobenchmarks(unittest.TestCase):
  def setUp(self):
    self.st = time.perf_counter()
  def tearDown(self):
    et = (time.perf_counter() - self.st)
    print(f"{self._testMethodName} {et*1e3:.2f} ms")

  def test_uop_instant_creation(self):
    for i in range(N): UOp.const(dtypes.int, 100+i)

  def test_uop_list_creation(self):
    [UOp.const(dtypes.int, 100+i) for i in range(N)]

  def test_uop_add_2n(self):
    a = UOp.const(dtypes.int, 2)
    for _ in range(N): a = a + a

  def test_uop_toposort(self):
    a = UOp.const(dtypes.int, 0)
    for i in range(N): a = a + UOp.const(dtypes.int, 100+i)
    self.setUp()
    self.assertEqual(len(a.toposort()), 2*N+1)

  def test_uop_toposort_2n(self):
    a = UOp.const(dtypes.int, 0)
    for i in range(N): a = a + a
    self.setUp()
    self.assertEqual(len(a.toposort()), N+1)

  def test_uop_simplify(self):
    a = UOp.const(dtypes.int, 2)
    for _ in range(N): (a+a).simplify()

if __name__ == '__main__':
  unittest.main()

