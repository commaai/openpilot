import unittest
import numpy as np
from tinygrad import Tensor, Variable
from tinygrad.helpers import Context

class TestTensorVariable(unittest.TestCase):
  def test_add_tvar(self):
    vv = Variable("a", 0, 10).bind(1)
    ret = (Tensor(vv) + 3).item()
    assert ret == 4

  def test_inner_tvar_node(self):
    vv = Variable("w", 0, 10).bind(2)
    ret = Tensor.from_uop(vv * 4).item()
    assert ret == 8

  def test_inner_tvar_mul(self):
    vv = Variable("w", 0, 10).bind(2)
    assert (Tensor(3) * vv).item() == 6

  def test_inner_tvar_mul_node(self):
    vv = Variable("w", 0, 10).bind(2)
    assert (Tensor(3) * (vv * 4)).item() == 24

  def test_symbolic_mean(self):
    with Context(IGNORE_OOB=1):
      vv = Variable("a", 1, 10).bind(2)
      t = Tensor.ones(2, 2).contiguous().reshape(2, vv)
      ret = t.mean().item()
      assert ret == 1

  def test_symbolic_mean_2d(self):
    with Context(IGNORE_OOB=1):
      vv = Variable("a", 1, 10).bind(2)
      vv2 = Variable("b", 1, 10).bind(2)
      t = Tensor.ones(2, 2).contiguous().reshape(vv2, vv)
      ret = t.mean().item()
      assert ret == 1

  def test_symbolic_mean_2d_axis_1(self):
    with Context(IGNORE_OOB=1):
      vv = Variable("a", 1, 10).bind(2)
      vv2 = Variable("b", 1, 10).bind(2)
      t = Tensor.ones(2, 2).contiguous().reshape(vv2, vv)
      ret = t.mean(axis=1).reshape(2, 1).numpy()
      assert np.all(ret == 1)

  def test_symbolic_mean_2d_add(self):
    with Context(IGNORE_OOB=1):
      add_term = Variable("c", 0, 10).bind(1)
      vv = Variable("a", 1, 10).bind(1)
      vv2 = Variable("b", 1, 10).bind(1)
      t = Tensor.ones(2, 2).contiguous().reshape(vv2+add_term, vv+add_term)
      ret = t.mean().item()
      assert ret == 1

  def test_symbolic_var(self):
    with Context(IGNORE_OOB=1):
      vv = Variable("a", 1, 10).bind(2)
      t = Tensor.ones(2, 2).contiguous().reshape(2, vv)
      ret = t.var().item()
      assert ret == 0

  def test_symbolic_pad(self):
    vv = Variable("a", 1, 10).bind(2)
    t = Tensor.ones(2, 2).contiguous()
    t = t.pad([vv, vv, vv, vv]).mean()
    ones = 4
    zeros = 6+6+4+4+6+6
    self.assertAlmostEqual(t.item(), ones/(ones+zeros))

  def test_symbolic_arange(self):
    vv = Variable("a", 1, 10)
    ret = Tensor.arange(0, vv.bind(4))
    self.assertListEqual(ret.reshape(4).tolist(), [0,1,2,3])

  def test_symbolic_arange_sym_start(self):
    vv = Variable("a", 1, 6)
    ret = Tensor.arange(vv.bind(4), 7)
    self.assertListEqual(ret.reshape(3).tolist(), [4,5,6])

  # TODO: add vmin/vmax pattern for symbolic denominator
  @unittest.expectedFailure
  def test_symbolic_arange_sym_step(self):
    vv = Variable("step", 1, 3)
    ret = Tensor.arange(0, 10, vv.bind(2))
    self.assertListEqual(ret.reshape(5).tolist(), [0,2,4,6,8])

  def test_symbolic_arange_two_vars(self):
    begin = Variable("b", 1, 5)
    end = Variable("e", 6, 10)
    ret = Tensor.arange(begin.bind(4), end.bind(7))
    self.assertListEqual(ret.reshape(3).tolist(), [4,5,6])

if __name__ == '__main__':
  unittest.main()
