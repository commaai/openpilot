import unittest

from tinygrad import Tensor, UOp
from tinygrad.engine.realize import compile_linear
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops


class TestSearch(unittest.TestCase):
  def test_beam_symbolic_kernel(self):
    size = UOp.variable("size", 1, 8)
    out = Tensor.empty(8, 8, device="CPU")[:size.bind(4)] + 1
    linear, _ = out.linear_with_vars()
    with Context(BEAM=1, IGNORE_BEAM_CACHE=1, CACHELEVEL=0): compiled = compile_linear(linear, beam=1)
    program = next(u for u in compiled.toposort(enter_calls=True) if u.op is Ops.PROGRAM)
    self.assertNotEqual(program.src[0].arg.applied_opts, ())


if __name__ == "__main__":
  unittest.main()
