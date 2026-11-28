import unittest
from tinygrad import Tensor, TinyJit
from tinygrad.engine.jit import JitError

class TestJit(unittest.TestCase):
  def test_jit_duplicate_fail(self):
    # the jit doesn't support duplicate arguments
    @TinyJit
    def add(a, b): return (a+b).realize()
    a = Tensor([1, 2])
    with self.assertRaisesRegex(JitError, "duplicate inputs"):
      add(a, a)

  def test_jit_shape_views_mismatch(self):
    @TinyJit
    def add(a): return (a+1).realize()
    with self.assertRaisesRegex(JitError, "args mismatch"):
      for i in range(1,5):
        # a has an offset that the kernel doesn't know about
        a = Tensor.empty(10, 10).realize()[:, i:i+2]
        add(a)

  def test_jit_shape_views_mismatch_alt(self):
    @TinyJit
    def add(a): return (a+1).realize()
    for _ in range(2):
      add(Tensor.empty(()))
    with self.assertRaisesRegex(JitError, "args mismatch"):
      add(Tensor.empty(2))

  def test_nothing_jitted(self):
    @TinyJit
    def add(a, b): return None
    a, b = Tensor.empty(2), Tensor.empty(2)
    add(a, b)
    with self.assertRaisesRegex(JitError, "didn't JIT anything"):
      add(a, b)

  def test_jit_output_non_tensor_fail(self):
    @TinyJit
    def f(a, b, i): return (a+b).realize(), i
    a, b = Tensor.empty(2), Tensor.empty(2)
    f(a, b, 1)
    with self.assertRaisesRegex(JitError, "non-Tensor value"):
      f(a, b, 1)

if __name__ == '__main__':
  unittest.main()
