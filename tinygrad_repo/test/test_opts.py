import unittest
from tinygrad import Tensor, Device
from tinygrad.helpers import RANGEIFY, CPU_LLVM
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.engine.realize import get_program

@unittest.skipIf(RANGEIFY>0, "arg is partial contig in rangeify")
class TestOpts(unittest.TestCase):
  def test_opt_upcast(self):
    opts = (Opt(OptOps.UPCAST, 0, 4),)
    a = Tensor.empty(16)
    b = Tensor.empty(16)
    out = (a+b).contiguous(arg=opts)
    s = out.schedule()
    self.assertEqual(s[-1].ast.arg.opts_to_apply, opts)
    if Device.DEFAULT in {"CPU", "CL", "METAL"} and not CPU_LLVM:
      prg = get_program(s[-1].ast)
      self.assertIn('float4', prg.src)

if __name__ == '__main__':
  unittest.main()

