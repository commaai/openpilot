import unittest
from tinygrad import Tensor, Context, Device
from tinygrad.codegen.kernel import Kernel, Opt, OptOps

class TestLinearizerRewrite(unittest.TestCase):
  def test_reduction(self):
    t = Tensor.ones((64,64), device="NULL").contiguous().realize()
    out = (t*2).sum(axis=1)
    with Context(SPLIT_REDUCEOP=0, DEVECTORIZE=0):
      si = out.schedule()[-1]
      k = Kernel(si.ast, Device["CPU"].renderer)
      k.apply_opt(Opt(OptOps.UPCAST, 0, 4))
      k.apply_opt(Opt(OptOps.UNROLL, 0, 4))
      prg = k.to_program()
      print(prg.src)

  def test_arange(self):
    out = Tensor.arange(32, device="NULL")
    with Context(SPLIT_REDUCEOP=0, DEVECTORIZE=0):
      si = out.schedule()[-1]
      k = Kernel(si.ast, Device["CPU"].renderer)
      k.apply_opt(Opt(OptOps.UPCAST, 0, 4))
      k.apply_opt(Opt(OptOps.UNROLL, 0, 4))
      prg = k.to_program()
      print(prg.src)

if __name__ == '__main__':
  unittest.main()
