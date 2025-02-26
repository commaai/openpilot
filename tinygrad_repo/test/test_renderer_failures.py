import unittest
from typing import List, cast
import numpy as np
from tinygrad.codegen.devectorizer import full_graph_rewrite
from tinygrad.codegen.linearize import linearize_uop
from tinygrad.device import Buffer, Device, is_dtype_supported
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import dedup, flatten, prod
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.ops import UOp, Ops
from tinygrad.renderer import ProgramSpec
from tinygrad.tensor import Tensor, _to_np_dtype

def _test_uop_result(inputs:List[Tensor], stores:List[UOp], local_size=None):
  for x in inputs: x.realize()
  # NOTE: we only toposort the stores
  uops: List[UOp] = []
  def _recursive_add(uop:UOp) -> List[UOp]: return flatten([_recursive_add(x) for x in uop.src])+[uop]
  uops = dedup(flatten(_recursive_add(st) for st in stores))
  outbufs = [Buffer(Device.DEFAULT, sz:=(1 if local_size is None else prod(local_size)), (dtype:=u.src[1].dtype), \
      initial_value=np.zeros(sz, dtype=_to_np_dtype(dtype)).data) for u in uops if u.op is Ops.STORE]
  inbufs = [cast(UOp,x.lazydata).base.buffer for x in inputs]
  src = Device[Device.DEFAULT].renderer.render(uops)
  ei = CompiledRunner(ProgramSpec("test", src, Device.DEFAULT, uops[-1], uops=uops, local_size=local_size))
  ei.exec(outbufs+inbufs)
  return [np.frombuffer(x.as_buffer(), _to_np_dtype(x.dtype)) for x in outbufs]

@unittest.skipIf(not isinstance(Device[Device.DEFAULT].renderer, CStyleLanguage), "uops are for cstyle")
class TestCStyleFailures(unittest.TestCase):
  def test_inline_const_alu(self):
    a = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0)
    b = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 1)
    idx = UOp.const(dtypes.int, 0)
    ld = UOp(Ops.LOAD, dtypes.int, (b.index(idx),))
    alu = ld.alu(Ops.MAX, UOp.const(dtypes.int, dtypes.min(dtypes.int)+1))
    store = UOp.store(a.index(idx), alu)
    sink = UOp(Ops.SINK, dtypes.void, (store,))
    uops = linearize_uop(full_graph_rewrite(sink, Device[Device.DEFAULT].renderer))
    # CPU doesn't use the max function
    ret = _test_uop_result([Tensor([1])], uops)[0]
    self.assertEqual(ret[0], 1)

@unittest.skipIf(not isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "tests for ptx renderer")
class TestPTXFailures(unittest.TestCase):
  def test_gated_store_with_alu(self):
    a = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0)
    gate_alu = (lidx0:=UOp(Ops.SPECIAL, dtypes.int, (), ('lidx0', 4))).ne(0)
    gated_alu_store = UOp(Ops.STORE, dtypes.void, (a.index(lidx0, gate_alu), UOp.const(dtypes.int, 1)))
    sink = UOp(Ops.SINK, dtypes.void, (gated_alu_store,))
    uops = linearize_uop(full_graph_rewrite(sink, Device[Device.DEFAULT].renderer))
    ret = _test_uop_result([], uops, local_size=[4, 1, 1])[0]
    np.testing.assert_equal(ret, [0, 1, 1, 1])

  def test_gated_store_with_if(self):
    a = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0)
    gate_alu = (lidx0:=UOp(Ops.SPECIAL, dtypes.int, (), ('lidx0', 4))).ne(0)
    val = UOp.const(dtypes.int, 1)
    if_uop = UOp(Ops.IF, dtypes.void, (gate_alu,))
    gated_alu_store = UOp(Ops.STORE, dtypes.void, (a.index(lidx0, if_uop), val))
    sink = UOp(Ops.SINK, dtypes.void, (gated_alu_store,))
    uops = linearize_uop(full_graph_rewrite(sink, Device[Device.DEFAULT].renderer))
    ret = _test_uop_result([], uops, local_size=[4, 1, 1])[0]
    np.testing.assert_equal(ret, [0, 1, 1, 1])

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_gated_define_acc_with_half_dtype(self):
    a = Tensor.randn(32, 32, dtype=dtypes.half).realize()
    b = Tensor.randn(34, 32, dtype=dtypes.half).realize()
    result = a.pad((1,1)).matmul(b, acc_dtype=dtypes.half).numpy()
    reference = a.pad((1,1)).matmul(b, acc_dtype=dtypes.float).numpy()
    np.testing.assert_allclose(result, reference)

if __name__ == '__main__':
  unittest.main()
