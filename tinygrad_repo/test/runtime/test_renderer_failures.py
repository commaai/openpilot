import unittest
import numpy as np
from tinygrad.device import Device, Buffer
from tinygrad.dtype import dtypes, ConstType
from tinygrad.engine.realize import run_linear
from tinygrad.helpers import prod
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.runtime.ops_python import PythonRenderer
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.tensor import Tensor

def _test_uop_result(inputs:list[Tensor], sink:UOp, local_size=None):
  for x in inputs: x.realize()
  sz = 1 if local_size is None else prod(local_size)
  outs = [UOp.new_buffer(Device.DEFAULT, sz, u.src[1].dtype) for u in sink.src if u.op is Ops.STORE]
  for u in outs: u.buffer.allocate().copy_from(Buffer("PYTHON", u.buffer.nbytes, initial_value=bytes(u.buffer.nbytes)))
  run_linear(UOp(Ops.LINEAR, src=(sink.call(*outs, *(x.uop.base for x in inputs)),)))
  return [Tensor(u).numpy() for u in outs]

def _setup_and_test_alu(alu_op:Ops, input_val:ConstType, *alu_src_uops:UOp):
  dtype = alu_src_uops[0].dtype
  a = UOp.param(0, dtype, 1)
  b = UOp.param(1, dtype, 1)
  idx = UOp.const(0)
  ld = b.index(idx).load()
  alu = ld.alu(alu_op, *alu_src_uops)
  store = UOp.store(a.index(idx), alu)
  return _test_uop_result([Tensor([input_val])], UOp(Ops.SINK, src=(store,), arg=KernelInfo()))[0]

class TestRendererFailures(unittest.TestCase):
  @unittest.skipIf(not isinstance(Device[Device.DEFAULT].renderer, (PTXRenderer, PythonRenderer)), "test is for ptx or python renderer")
  def test_gated_store_with_alu(self):
    a = UOp.param(0, dtypes.int, 4)
    gate_alu = (lidx0:=UOp.special(4, 'lidx0')).ne(0)
    gated_alu_store = UOp(Ops.STORE, src=(a.index(lidx0.valid(gate_alu)), UOp.const(1).cast(dtypes.int)))
    sink = UOp(Ops.SINK, src=(gated_alu_store,), arg=KernelInfo())
    ret = _test_uop_result([], sink, local_size=[4, 1, 1])[0]
    np.testing.assert_equal(ret, [0, 1, 1, 1])

  @unittest.skipIf(not isinstance(Device[Device.DEFAULT].renderer, (PTXRenderer, PythonRenderer)), "test is for ptx or python renderer")
  def test_gated_store_with_alu_2d(self):
    a = UOp.param(0, dtypes.int, 8)
    gate_alu_0 = (lidx0:=UOp.special(4, 'lidx0')).ne(0)
    gate_alu_1 = (lidx1:=UOp.special(2, 'lidx1')).ne(0)
    gated_alu_store = UOp(Ops.STORE, src=(a.index((lidx0+lidx1*4).valid(gate_alu_0&gate_alu_1)), UOp.const(1).cast(dtypes.int)))
    sink = UOp(Ops.SINK, src=(gated_alu_store,), arg=KernelInfo())
    ret = _test_uop_result([], sink, local_size=[4, 2, 1])[0]
    np.testing.assert_equal(ret, [0, 0, 0, 0, 0, 1, 1, 1])

@unittest.skipIf(not isinstance(Device[Device.DEFAULT].renderer, CStyleLanguage), "uops are for cstyle")
class TestCStyleFailures(unittest.TestCase):
  def test_inline_const_alu(self):
    # CPU doesn't use the max function
    ret = _setup_and_test_alu(Ops.MAX, 1, UOp.const(dtypes.int.min+1).cast(dtypes.int))
    self.assertEqual(ret[0], 1)

if __name__ == '__main__':
  unittest.main()
