import unittest
from tinygrad import dtypes, Device, Tensor, Context
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.engine.realize import get_program, ExecItem, CompiledRunner

class TestDefineReg(unittest.TestCase):
  def test_simple(self, at=AxisType.UPCAST):
    N = 16
    bout = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=0).view(ShapeTracker.from_shape((N,N)))
    a = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=1).view(ShapeTracker.from_shape((N,N)))
    a_col = UOp(Ops.DEFINE_REG, dtypes.float.ptr(N, AddrSpace.REG), arg=0).view(ShapeTracker.from_shape((N,N), (0,1)))

    out = a_col.load(a_col.store(a.load()))
    sink = bout.store(out).sink(arg=KernelInfo(name="regcopy", axis_types=(AxisType.LOOP, at)))
    prg = get_program(sink, Device.default.renderer)

    with Context(DEBUG=0):
      a = Tensor.randn(N, N).realize()
      b = Tensor.empty(N, N).realize()
    hrunner = CompiledRunner(prg)
    ExecItem(hrunner, [b.uop.buffer, a.uop.buffer]).run(wait=True)
    with Context(DEBUG=0):
      self.assertEqual((b-a).mean().item(), 0.0)

  @unittest.skipIf(getenv("PTX"), "ptx needs regs to be unrolled")
  def test_simple_loop(self): self.test_simple(AxisType.LOOP)

if __name__ == '__main__':
  unittest.main()
