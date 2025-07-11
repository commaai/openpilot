import unittest
import numpy as np

from tinygrad.helpers import BEAM, Timing, CI, Context
from tinygrad import Variable, Tensor
from tinygrad.nn import Conv2d

def rand(*shape):
  return Tensor(np.random.rand(*shape).astype(np.float32))

class TestBeamSearch(unittest.TestCase):
  def setUp(self):
    self.old_beam = BEAM.value
    BEAM.value = 2
  def tearDown(self):
    BEAM.value = self.old_beam

  def test_variable_ast_beam(self):
    with Context(IGNORE_OOB=1):
      a = rand(3, 3).reshape((Variable("a", 1, 10).bind(3), 3))
      a = (a+1).realize()

  def test_big_prime_number(self):
    a = rand(367, 367)
    b = rand(367, 367)
    c = (a@b).realize()
    np.testing.assert_allclose(c.numpy(), a.numpy() @ b.numpy(), atol=1e-4, rtol=1e-4)

  def test_big_prime_number_max(self):
    a = -rand(367, 367)
    b = rand(367, 367)
    # if incorrectly padded 0, the max would be 0 instead of a negative number
    c = (a*b).max(1)
    np.testing.assert_allclose(c.numpy(), (a.numpy() * b.numpy()).max(1), atol=1e-4, rtol=1e-4)

  def test_big_prime_number_sum(self):
    a = rand(367, 367)
    b = rand(367, 367)
    # if incorrectly padded 0, the sum would be inf
    c = (a/b).sum(1).realize()
    np.testing.assert_allclose(c.numpy(), (a.numpy() / b.numpy()).sum(1), atol=1e-4, rtol=1e-4)

  def test_variable_big_prime_number(self):
    v = Variable("v", 1, 400).bind(367)
    a = rand(367, 367)
    b = rand(367, 367)
    with Context(IGNORE_OOB=1):
      c = (a.reshape(367, v) @ b.reshape(v, 367)).realize()
      np.testing.assert_allclose(c.numpy(), a.numpy() @ b.numpy(), atol=1e-4, rtol=1e-4)

  def test_variable_shrink_prime_number(self):
    v = Variable("v", 1, 400).bind(367)
    a = rand(400, 367)
    with Context(IGNORE_OOB=1):
      b = (a.shrink(((0,v), None))+1).reshape(367,367).realize()
      np.testing.assert_allclose(b.numpy(), a.numpy()[:367]+1, atol=1e-4, rtol=1e-4)

  def test_no_mutate_rawbuffers(self):
    a = rand(3, 3).realize()
    desired = a.numpy() + 1
    a.assign(a+1)
    actual = a.numpy()
    np.testing.assert_allclose(actual, desired)

  @unittest.skipIf(CI, "flaky. CL_OUT_OF_RESOURCES")
  def test_conv_beam(self):
    c = Conv2d(3, 16, (3,3))
    x = rand(1,3,32,32)
    with Timing():
      c(x).realize()

  @unittest.skip("flaky, Fatal Python error: Floating point exception")
  def test_large_ast(self):
    a = Tensor.rand(3, 3)
    for _ in range(5):
      for _ in range(4):
        a = (a + a) * a
    a.realize()

if __name__ == '__main__':
  unittest.main()
