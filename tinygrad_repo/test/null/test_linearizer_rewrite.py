import unittest
from tinygrad import Tensor, Context, Device
from tinygrad.codegen import to_program
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.uop.ops import KernelInfo

class TestLinearizerRewrite(unittest.TestCase):
  def test_reduction(self):
    t = Tensor.ones((64,64), device="NULL").contiguous().realize()
    out = (t*2).sum(axis=1)
    with Context(SPLIT_REDUCEOP=0):
      si = out.schedule_linear().src[-1]
      opts_to_apply = []
      opts_to_apply.append(Opt(OptOps.UPCAST, 0, 4))
      opts_to_apply.append(Opt(OptOps.UNROLL, 0, 4))
      ast = si.src[0].replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply)))
      prg = to_program(ast, Device["CPU"].renderer)
      print(prg.src[2].arg)

  def test_arange(self):
    out = Tensor.arange(32).clone("NULL")
    with Context(SPLIT_REDUCEOP=0):
      si = out.schedule_linear().src[-1]
      opts_to_apply = []
      opts_to_apply.append(Opt(OptOps.UPCAST, 0, 4))
      ast = si.src[0].replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply)))
      prg = to_program(ast, Device["CPU"].renderer)
      print(prg.src[2].arg)

  def test_kernel_info(self):
    out = Tensor.arange(4).clone("NULL")
    si = out.schedule_linear().src[-1]

    ast = si.src[0].replace(arg=KernelInfo(opts_to_apply=()))
    prg = to_program(ast, Device["CPU"].renderer)
    assert prg.src[0].arg.applied_opts == (), f"expected no opts, got {prg}"

    #prg = to_program(ast.replace(arg=KernelInfo()), Device["CPU"].renderer)
    #assert prg.src[0].arg.applied_opts != (), f"expected opts to apply, got {prg.src[0].arg.applied_opts}"

    prg = to_program(ast.replace(arg=KernelInfo(name="custom")), Device["CPU"].renderer)
    self.assertEqual(prg.arg.name, "custom")

if __name__ == '__main__':
  unittest.main()
