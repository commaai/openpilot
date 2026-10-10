import unittest
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.renderer.wgsl import WGSLRenderer
from test.helpers import check_schedule
from tinygrad.uop.ops import UOp, Ops, KernelInfo, python_alu
from tinygrad.tensor import Tensor

@unittest.skipIf(not isinstance(Device[Device.DEFAULT].renderer, CStyleLanguage), "uops are for cstyle")
class TestCStyleFailures(unittest.TestCase):
  def _test_src_strip_paren(self, op: Ops, should_strip_paren:bool=True):
    dtype = "bool" if op in (Ops.OR, Ops.XOR, Ops.AND) else None
    ret = Tensor.empty(1, dtype=dtype)
    for _ in range(5): ret = python_alu[op](ret, Tensor.empty(1, dtype=dtype))
    linear, _ = check_schedule(ret, 1)
    src = to_program(linear.src[0].src[0], Device[Device.DEFAULT].renderer).src[2].arg
    self.assertEqual("("*5 not in src, should_strip_paren)

  def test_repeat_add(self): self._test_src_strip_paren(Ops.ADD)

  def test_repeat_mul(self): self._test_src_strip_paren(Ops.MUL)

  def test_repeat_xor(self): self._test_src_strip_paren(Ops.XOR)

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, WGSLRenderer), "wgsl ends up with '(' * 5")
  def test_repeat_or(self): self._test_src_strip_paren(Ops.OR)

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, WGSLRenderer), "wgsl ends up with '(' * 5")
  def test_repeat_and(self): self._test_src_strip_paren(Ops.AND)

  def test_repeat_sub(self): self._test_src_strip_paren(Ops.SUB, should_strip_paren=False)

class TestWGSLFailures(unittest.TestCase):
  def test_folded_packed_store(self):
    b = UOp.param(0, dtypes.char, 4)
    idx = b.index(UOp.const(0).cast(dtypes.int))
    store = UOp.store(idx, idx.cast(dtypes.uint32).load() & UOp.const(0xffffff00).cast(dtypes.uint32))
    src = WGSLRenderer(Target("WEBGPU")).render(UOp.sink(store, arg=KernelInfo()).toposort())
    self.assertIn("atomicAnd(&data0_4[0],4294967040u);", src)
    self.assertNotIn("atomicAdd", src)

if __name__ == '__main__':
  unittest.main()
