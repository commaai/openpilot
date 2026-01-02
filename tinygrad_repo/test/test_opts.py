import unittest
from tinygrad import Tensor, Device
from tinygrad.helpers import CPU_LLVM, CPU_LVP
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.engine.realize import get_program

class TestOpts(unittest.TestCase):
  def test_opt_upcast(self):
    opts = (Opt(OptOps.UPCAST, 0, 4),)
    a = Tensor.empty(16)
    b = Tensor.empty(16)
    out = (a+b).contiguous(arg=opts)
    s = out.schedule()
    self.assertEqual(s[-1].ast.arg.opts_to_apply, opts)
    if Device.DEFAULT in {"CPU", "CL", "METAL"} and not CPU_LLVM and not CPU_LVP:
      prg = get_program(s[-1].ast, renderer=Device[Device.DEFAULT].renderer)
      self.assertIn('float4', prg.src)

if __name__ == '__main__':
  unittest.main()

