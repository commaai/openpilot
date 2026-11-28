import unittest
from tinygrad import Tensor, Variable, dtypes, Device
from tinygrad.helpers import Context
from tinygrad.codegen.compile import estimate_uop, compile_linear
from tinygrad.uop.ops import sym_infer
from tinygrad.renderer.ptx import PTXRenderer

class TestArange(unittest.TestCase):
  def test_cat_complexity(self):
    x = Tensor.arange(2**10) + Tensor.empty((), dtype=dtypes.uint32)
    out = x.cat(x).cat(Tensor.empty(1, dtype=dtypes.uint32))
    linear = compile_linear(out.schedule_linear())
    self.assertLessEqual(estimate_uop(linear.src[-1]).ops, out.numel()*20)

  def test_symbolic_complexity(self):
    out = Tensor.arange(Variable("n", 1, 2**10).bind(2**9)) + Tensor.empty((), dtype=dtypes.int32)
    linear, externs = out.linear_with_externs()
    var_vals = {f"e{i}": x.arg.val for i,x in enumerate(externs) if x.is_bound_var}
    self.assertLessEqual(sym_infer(estimate_uop(compile_linear(linear).src[-1]).ops, var_vals), 2**10*20)

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "PTX indexing is weird")
  def test_tri_complexity(self):
    with Context(NOOPT=1):
      t = Tensor.ones(256, 256).contiguous().realize()
      linear = compile_linear(t.triu().schedule_linear())
      self.assertLessEqual(estimate_uop(linear.src[-1]).ops, 4 * 256 * 256)

if __name__ == '__main__':
  unittest.main()
