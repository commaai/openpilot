import unittest
from tinygrad import Tensor, Context, Device
from tinygrad.engine.realize import get_program
from tinygrad.codegen.opt.kernel import Opt, OptOps
from tinygrad.uop.ops import KernelInfo

class TestLinearizerRewrite(unittest.TestCase):
  def test_reduction(self):
    t = Tensor.ones((64,64), device="NULL").contiguous().realize()
    out = (t*2).sum(axis=1)
    with Context(SPLIT_REDUCEOP=0, DEVECTORIZE=0):
      si = out.schedule()[-1]
      opts_to_apply = []
      opts_to_apply.append(Opt(OptOps.UPCAST, 0, 4))
      opts_to_apply.append(Opt(OptOps.UNROLL, 0, 4))
      ast = si.ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply)))
      prg = get_program(ast, Device["CPU"].renderer)
      print(prg.src)

  def test_arange(self):
    out = Tensor.arange(32, device="NULL")
    with Context(SPLIT_REDUCEOP=0, DEVECTORIZE=0):
      si = out.schedule()[-1]
      opts_to_apply = []
      opts_to_apply.append(Opt(OptOps.UPCAST, 0, 4))
      opts_to_apply.append(Opt(OptOps.UNROLL, 0, 4))
      ast = si.ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply)))
      prg = get_program(ast, Device["CPU"].renderer)
      print(prg.src)

  def test_kernel_info(self):
    out = Tensor.arange(4, device="NULL")
    si = out.schedule()[-1]

    ast = si.ast.replace(arg=KernelInfo(opts_to_apply=()))
    prg = get_program(ast, Device["CPU"].renderer)
    assert prg.applied_opts == (), f"expected no opts, got {prg}"

    prg = get_program(ast.replace(arg=None), Device["CPU"].renderer)
    assert prg.applied_opts != (), f"expected opts to apply, got {prg.applied_opts}"

    prg = get_program(ast.replace(arg=KernelInfo(name="custom")), Device["CPU"].renderer)
    self.assertEqual(prg.name, "custom")

if __name__ == '__main__':
  unittest.main()
