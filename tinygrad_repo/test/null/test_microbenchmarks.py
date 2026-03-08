import unittest, time
from tinygrad import dtypes, Tensor, UOp, getenv
from tinygrad.helpers import Profiling

PYPROFILE = getenv("PYPROFILE")
class TestBench(unittest.TestCase):
  @staticmethod
  def setUpClass():
    # no fixed cost
    Tensor.empty(10,10)
    Tensor.randn(10,10)

  def start_time(self): self.st = time.perf_counter()
  def setUp(self):
    # it's about 1 ms per 1k UOps on M3
    if PYPROFILE:
      self.prof = Profiling()
      self.prof.__enter__()
    else:
      self.prof = None
    self.N = 10000
    self.start_time()

  def tearDown(self):
    et = (time.perf_counter() - self.st)
    if self.prof is not None: self.prof.__exit__()
    print(f"{self._testMethodName:30s} {et*1e6/self.N:.2f} us")

  def test_uop_instant_creation(self):
    for i in range(self.N): UOp.const(dtypes.int, 100+i)

  def test_uop_list_creation(self):
    [UOp.const(dtypes.int, 100+i) for i in range(self.N)]

  def test_uop_add_2n(self):
    a = UOp.const(dtypes.int, 2)
    for _ in range(self.N): a = a + a

  def test_uop_toposort(self):
    a = UOp.const(dtypes.int, 0)
    for i in range(self.N): a = a + UOp.const(dtypes.int, 100+i)
    self.start_time()
    self.assertEqual(len(a.toposort()), 2*self.N+1)

  def test_uop_toposort_2n(self):
    a = UOp.const(dtypes.int, 0)
    for _ in range(self.N): a = a + a
    self.start_time()
    self.assertEqual(len(a.toposort()), self.N+1)

  def test_uop_simplify(self):
    a = UOp.const(dtypes.int, 2)
    for _ in range(self.N): (a+a).simplify()

  def test_uop_simplify_complex(self):
    self.N //= 10 # this test is slow
    x = UOp.variable("x", 0, 10)
    y = UOp.variable("y", 0, 10)
    expr = (x*2)+5+(x*4)+(y*2)+y
    for _ in range(self.N): expr.simplify()

  def test_uop_simplify_div(self):
    self.N //= 10 # this test is slow
    x = UOp.variable("x", 0, 10)
    y = UOp.variable("y", 0, 10)
    z = UOp.variable("z", 0, 10)
    expr = (x*4+y*8)//(z*2)
    for _ in range(self.N): expr.simplify()

  def test_uop_chain_free(self):
    a = UOp.const(dtypes.int, 2)
    for _ in range(self.N): a = a + a
    self.start_time()
    del a

  def test_tensor_zeros(self):
    self.N //= 10 # this test is slow
    for _ in range(self.N): Tensor.zeros(10, 10)

  def test_tensor_add(self):
    self.N //= 10 # this test is slow
    a = Tensor.zeros(10, 10)
    b = Tensor.zeros(10, 10)
    for _ in range(self.N): a+b

  def test_tensor_empty(self):
    self.N //= 10 # this test is slow
    for _ in range(self.N): Tensor.empty(10, 10)

  def test_tensor_rand(self):
    self.N //= 100 # this test is very slow
    for _ in range(self.N): Tensor.rand(10, 10)

  def test_tensor_randn(self):
    self.N //= 100 # this test is very slow
    for _ in range(self.N): Tensor.randn(10, 10)

if __name__ == '__main__':
  unittest.main()

