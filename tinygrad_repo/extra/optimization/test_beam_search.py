import unittest
import numpy as np

from tinygrad.helpers import BEAM, Timing, CI, prod
from tinygrad import Variable, Device, Tensor
from tinygrad.nn import Conv2d
from tinygrad.uop.ops import AxisType
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.codegen.opt.search import get_kernel_actions

def rand(*shape):
  return Tensor(np.random.rand(*shape).astype(np.float32))

class TestBeamSearch(unittest.TestCase):
  def setUp(self):
    self.old_beam = BEAM.value
    BEAM.value = 2
  def tearDown(self):
    BEAM.value = self.old_beam

  def test_variable_ast_beam(self):
    vi = Variable("a", 1, 10).bind(3)
    a = rand(10, 3)[:vi]
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
    a = rand(367, 400)
    b = rand(400, 367)
    c = (a[:, :v] @ b[:v, :]).realize()
    np.testing.assert_allclose(c.numpy(), a[:, :367].numpy() @ b[:367, :].numpy(), atol=1e-4, rtol=1e-4)

  def test_variable_shrink_prime_number(self):
    v = Variable("v", 1, 400).bind(367)
    a = rand(400, 367)
    b = (a.shrink(((0,v), None))+1)[:367,:367].realize()
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

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tc_up(self):
    tc = Device[Device.DEFAULT].renderer.tensor_cores[0]
    size = max(tc.dims[0], tc.dims[1]) * 8
    a, b = Tensor.rand(size, size, dtype=tc.dtype_in), Tensor.rand(size, size, dtype=tc.dtype_in)
    ast = a.matmul(b, dtype=tc.dtype_out).schedule()[-1].ast
    s = Scheduler(ast, Device[Device.DEFAULT].renderer)
    s.apply_opt(Opt(OptOps.TC, 0, (-1, 0, 1)))
    up = prod([x for x, t in zip(s.full_shape, s.axis_types) if t in (AxisType.UPCAST, AxisType.UNROLL)])
    actions = get_kernel_actions(s, include_0=False, max_up=int(up))
    upcasted = [s for s in actions.values() if any(opt.op in (OptOps.UPCAST, OptOps.UNROLL) for opt in s.applied_opts)]
    assert len(upcasted) > 0, f"expected upcast/unroll actions after TC with max_up={up}, but got none"

  def test_max_up(self):
    a = Tensor.rand(16, 16)
    ast = a.schedule()[-1].ast
    s = Scheduler(ast, Device[Device.DEFAULT].renderer)
    for max_up in (2, 4):
      actions = get_kernel_actions(s, include_0=False, max_up=max_up)
      for up_opts in [s.applied_opts for s in actions.values() if any(opt.op in (OptOps.UPCAST, OptOps.UNROLL) for opt in s.applied_opts)]:
        assert len([opt for opt in up_opts if opt.arg > max_up]) == 0 and len([op for op in up_opts if op.arg <= max_up]) > 0

if __name__ == '__main__':
  unittest.main()
