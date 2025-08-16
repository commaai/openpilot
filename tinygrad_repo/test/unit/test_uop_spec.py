from __future__ import annotations
import unittest

from tinygrad import Tensor
from tinygrad.helpers import DEBUG
from tinygrad.uop.ops import UOp, Ops, print_uops
from tinygrad.uop.spec import type_verify, ast_spec, tensor_uop_spec
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad import dtypes
from tinygrad.shape.view import View
from tinygrad.engine.realize import get_program
from tinygrad.device import Device

class InvalidASTException(Exception): pass
def helper_test_verify_ast(*stores:UOp):
  sink = UOp(Ops.SINK, dtypes.void, stores)
  if DEBUG >= 3:
    for op in stores: print(op)
  try: type_verify(list(sink.toposort()), ast_spec)
  except RuntimeError as e: raise InvalidASTException(e.args)
  program = get_program(sink, Device[Device.DEFAULT].renderer)

  if DEBUG >= 6: print_uops(program.uops)
  if DEBUG >= 4: print(program.src)

class TestUOpSpec(unittest.TestCase):
  def test_tiny_add(self):
    dtype = dtypes.int
    buf_0 = UOp(Ops.DEFINE_GLOBAL, dtype.ptr(), (), 0)
    buf_1 = UOp(Ops.DEFINE_GLOBAL, dtype.ptr(), (), 1)
    buf_2 = UOp(Ops.DEFINE_GLOBAL, dtype.ptr(), (), 2)
    a = UOp(Ops.LOAD, dtype, (buf_1.view(ShapeTracker.from_shape((32, 1))),))
    b = UOp(Ops.LOAD, dtype, (buf_2.view(ShapeTracker.from_shape((32, 1))),))
    store = UOp(Ops.STORE, dtypes.void, (buf_0.view(ShapeTracker.from_shape((32, 1))), a+b))
    helper_test_verify_ast(store)

  def test_no_implicit_broadcasting(self):
    bufs = [UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), i) for i in range(2)]
    a = UOp(Ops.LOAD, dtypes.float, (bufs[1].view(ShapeTracker.from_shape((4, 32))),))
    b = a + UOp(Ops.REDUCE_AXIS, dtypes.float, (a,), (Ops.MAX, (1,)))
    st = UOp(Ops.STORE, dtypes.void, (bufs[0].view(ShapeTracker.from_shape((4, 32))), b))
    with self.assertRaises(InvalidASTException): helper_test_verify_ast(st)

  def test_shrink_ok(self):
    bufs = [UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), i) for i in range(2)]
    a = UOp(Ops.LOAD, dtypes.float, (bufs[1].view(ShapeTracker((View((32, 32), strides=(32, 1), offset=0, mask=None, contiguous=True),))),))
    b = UOp(Ops.LOAD, dtypes.float, (bufs[1].view(ShapeTracker((View((32, 32), strides=(0, 1), offset=0, mask=None, contiguous=False),))),))
    st = UOp.store(bufs[0].view(ShapeTracker.from_shape((32, 32))), a+b)
    helper_test_verify_ast(st)

  def test_reduce_store(self):
    bufs = [UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), i) for i in range(2)]
    a = UOp(Ops.LOAD, dtypes.float, (bufs[1].view(ShapeTracker.from_shape((32, 1))),))
    r = UOp(Ops.REDUCE_AXIS, dtypes.float, (a,), (Ops.ADD, (0,)))
    st = UOp.store(bufs[0].view(ShapeTracker.from_shape((32, 1))), r)
    with self.assertRaises(InvalidASTException): helper_test_verify_ast(st)

  def test_reduce_add_store(self):
    bufs = [UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), i) for i in range(2)]
    a = UOp(Ops.LOAD, dtypes.float, (bufs[1].view(ShapeTracker.from_shape((32, 1))),))
    r = UOp(Ops.REDUCE_AXIS, dtypes.float, (a,), (Ops.ADD, (0,)))
    st = UOp.store(bufs[0].view(ShapeTracker.from_shape((32, 1))), r+a)
    with self.assertRaises(InvalidASTException): helper_test_verify_ast(st)

  def test_assert_swizzle(self):
    buf = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 0)
    a = UOp(Ops.LOAD, dtypes.float, (buf.view(ShapeTracker.from_shape((32, 1))),))
    r = UOp(Ops.REDUCE_AXIS, dtypes.float, (a,), (Ops.ADD, (0,)))
    st = UOp.store(buf.view(ShapeTracker.from_shape((32, 1))), r.view(r.st.expand((32, 1)))+a)
    with self.assertRaisesRegex(InvalidASTException, "UOp verification failed"): helper_test_verify_ast(st)

  def test_const_view_always_valid(self):
    buf = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 0)
    a = UOp.const(dtypes.int, 0).replace(src=(UOp(Ops.VIEW, dtypes.void, (), ShapeTracker.from_shape(())),))
    st = UOp.store(buf.view(ShapeTracker.from_shape(())), a.cast(dtypes.float))
    helper_test_verify_ast(st)

  def test_assert_masked_view_in_const(self):
    t = Tensor(6).uop
    a = t.replace(src=(t.src[0].replace(arg=t.st.reshape((1,)).pad(((0, 1),))),))
    with self.assertRaisesRegex(RuntimeError, "UOp verification failed"):
      type_verify([a], tensor_uop_spec)

if __name__ == '__main__':
  unittest.main()
