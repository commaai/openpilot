import unittest
from typing import List, cast
import numpy as np
from tinygrad.device import Buffer, Device, is_dtype_supported
from tinygrad.dtype import dtypes, ConstType
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import dedup, flatten, prod
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.runtime.ops_python import PythonRenderer
from tinygrad.uop.ops import UOp, Ops, python_alu
from tinygrad.renderer import ProgramSpec
from tinygrad.tensor import Tensor, _to_np_dtype
from tinygrad.codegen import full_rewrite
from tinygrad.engine.realize import lower_schedule_item

def _test_uop_result(inputs:List[Tensor], stores:List[UOp], local_size=None):
  for x in inputs: x.realize()
  # NOTE: we only toposort the stores
  uops: List[UOp] = []
  def _recursive_add(uop:UOp) -> List[UOp]: return flatten([_recursive_add(x) for x in uop.src])+[uop]
  uops = dedup(flatten(_recursive_add(st) for st in stores))
  outbufs = [Buffer(Device.DEFAULT, sz:=(1 if local_size is None else prod(local_size)), (dtype:=u.src[1].dtype), \
      initial_value=np.zeros(sz, dtype=_to_np_dtype(dtype)).data) for u in uops if u.op is Ops.STORE]
  inbufs = [cast(UOp,x.uop).base.buffer for x in inputs]
  src = Device[Device.DEFAULT].renderer.render(uops)
  ei = CompiledRunner(ProgramSpec("test", src, Device.DEFAULT, uops[-1], uops=uops, local_size=local_size))
  ei.exec(outbufs+inbufs)
  return [np.frombuffer(x.as_buffer(), _to_np_dtype(x.dtype)) for x in outbufs]

def _setup_and_test_alu(alu_op:Ops, input_val:ConstType, *alu_src_uops:UOp):
  dtype = alu_src_uops[0].dtype
  a = UOp(Ops.DEFINE_GLOBAL, dtype.ptr(), (), 0)
  b = UOp(Ops.DEFINE_GLOBAL, dtype.ptr(), (), 1)
  idx = UOp.const(dtypes.int, 0)
  ld = UOp(Ops.LOAD, dtype, (b.index(idx),))
  alu = ld.alu(alu_op, *alu_src_uops)
  store = UOp.store(a.index(idx), alu)
  sink = UOp(Ops.SINK, dtypes.void, (store,))
  uops = full_rewrite(sink, Device[Device.DEFAULT].renderer)
  return _test_uop_result([Tensor([input_val])], uops)[0]

class TestRendererFailures(unittest.TestCase):
  @unittest.skipIf(not isinstance(Device[Device.DEFAULT].renderer, (PTXRenderer, PythonRenderer)), "test is for ptx or python renderer")
  def test_gated_store_with_alu(self):
    a = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0)
    gate_alu = (lidx0:=UOp(Ops.SPECIAL, dtypes.int, (), ('lidx0', 4))).ne(0)
    gated_alu_store = UOp(Ops.STORE, dtypes.void, (a.index(lidx0, gate_alu), UOp.const(dtypes.int, 1)))
    sink = UOp(Ops.SINK, dtypes.void, (gated_alu_store,))
    uops = full_rewrite(sink, Device[Device.DEFAULT].renderer)
    ret = _test_uop_result([], uops, local_size=[4, 1, 1])[0]
    np.testing.assert_equal(ret, [0, 1, 1, 1])

  @unittest.skipIf(not isinstance(Device[Device.DEFAULT].renderer, (PTXRenderer, PythonRenderer)), "test is for ptx or python renderer")
  def test_gated_store_with_alu_2d(self):
    a = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0)
    gate_alu_0 = (lidx0:=UOp(Ops.SPECIAL, dtypes.int, (), ('lidx0', 4))).ne(0)
    gate_alu_1 = (lidx1:=UOp(Ops.SPECIAL, dtypes.int, (), ('lidx1', 2))).ne(0)
    gated_alu_store = UOp(Ops.STORE, dtypes.void, (a.index(lidx0+lidx1*4, gate_alu_0&gate_alu_1), UOp.const(dtypes.int, 1)))
    sink = UOp(Ops.SINK, dtypes.void, (gated_alu_store,))
    uops = full_rewrite(sink, Device[Device.DEFAULT].renderer)
    ret = _test_uop_result([], uops, local_size=[4, 2, 1])[0]
    np.testing.assert_equal(ret, [0, 0, 0, 0, 0, 1, 1, 1])

@unittest.skipIf(not isinstance(Device[Device.DEFAULT].renderer, CStyleLanguage), "uops are for cstyle")
class TestCStyleFailures(unittest.TestCase):
  def test_inline_const_alu(self):
    # CPU doesn't use the max function
    ret = _setup_and_test_alu(Ops.MAX, 1, UOp.const(dtypes.int, dtypes.min(dtypes.int)+1))
    self.assertEqual(ret[0], 1)

  def _test_src_strip_paren(self, op: Ops, should_strip_paren:bool=True):
    dtype = "bool" if op in (Ops.OR, Ops.XOR, Ops.AND) else None
    ret = Tensor.empty(1, dtype=dtype)
    for _ in range(5): ret = python_alu[op](ret, Tensor.empty(1, dtype=dtype))
    schedule = ret.schedule()
    assert len(schedule) == 1
    ei = lower_schedule_item(schedule[0])
    src = ei.prg.p.src
    self.assertEqual("("*5 not in src, should_strip_paren)

  def test_repeat_add(self): self._test_src_strip_paren(Ops.ADD)
  def test_repeat_mul(self): self._test_src_strip_paren(Ops.MUL)
  def test_repeat_xor(self): self._test_src_strip_paren(Ops.XOR)
  def test_repeat_or(self): self._test_src_strip_paren(Ops.OR)
  def test_repeat_and(self): self._test_src_strip_paren(Ops.AND)
  def test_repeat_sub(self): self._test_src_strip_paren(Ops.SUB, should_strip_paren=False)

@unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, WGSLRenderer), "tests for wgsl renderer")
class TestWGSLFailures(unittest.TestCase):
  def test_multiply_infinity(self):
    # multiplying a positive constant by infinity should return infinity
    # WGSL pipelines do not handle this reliably, some of which return zero, unless infinity always comes from a read on a dynamic buffer
    ret = _setup_and_test_alu(Ops.MUL, 5.0, UOp.const(dtypes.float32, float("inf")))
    self.assertEqual(ret[0], float("inf"))

@unittest.skipIf(not isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "tests for ptx renderer")
class TestPTXFailures(unittest.TestCase):
  @unittest.skip("INDEX can only have a gate ALU parent, not an IF")
  def test_gated_store_with_if(self):
    a = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0)
    gate_alu = (lidx0:=UOp(Ops.SPECIAL, dtypes.int, (), ('lidx0', 4))).ne(0)
    val = UOp.const(dtypes.int, 1)
    if_uop = UOp(Ops.IF, dtypes.void, (gate_alu,))
    gated_alu_store = UOp(Ops.STORE, dtypes.void, (a.index(lidx0, if_uop), val))
    sink = UOp(Ops.SINK, dtypes.void, (gated_alu_store,))
    uops = full_rewrite(sink, Device[Device.DEFAULT].renderer)
    ret = _test_uop_result([], uops, local_size=[4, 1, 1])[0]
    np.testing.assert_equal(ret, [0, 1, 1, 1])

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_gated_define_acc_with_half_dtype(self):
    a = Tensor.randn(32, 32, dtype=dtypes.half).realize()
    b = Tensor.randn(34, 32, dtype=dtypes.half).realize()
    result = a.pad((1,1)).matmul(b, dtype=dtypes.half).numpy()
    reference = a.pad((1,1)).matmul(b, dtype=dtypes.float).numpy()
    np.testing.assert_allclose(result, reference, atol=1e-2, rtol=1e-2)

if __name__ == '__main__':
  unittest.main()
