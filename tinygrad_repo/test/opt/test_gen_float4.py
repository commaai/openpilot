import unittest
from tinygrad import Device, Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.engine.realize import get_program
from tinygrad.helpers import AMX

@unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "need backends that support float4")
class TestFloat4(unittest.TestCase):
  @staticmethod
  def count_float4(uops: list[UOp], n=4):
    return (len([uop for uop in uops if uop.op is Ops.LOAD and uop.dtype == dtypes.float.vec(n)]),
            len([uop for uop in uops if uop.op is Ops.STORE and uop.src[1].dtype == dtypes.float.vec(n)]))
  @staticmethod
  def count_half4(uops: list[UOp]):
    return (len([uop for uop in uops if uop.op is Ops.LOAD and uop.dtype == dtypes.half.vec(4)]),
            len([uop for uop in uops if uop.op is Ops.STORE and uop.src[1].dtype == dtypes.half.vec(4)]))

  def test_float4_basic(self):
    a = Tensor.empty(2, 8).realize()
    b = Tensor.empty(2, 8).realize()
    c = a + b

    s = c.schedule()[0]
    realized_ast = s.ast
    opts_to_apply = [Opt(op=OptOps.UPCAST, axis=0, arg=4)]
    program = get_program(realized_ast, renderer=Device[Device.DEFAULT].renderer, opts=opts_to_apply)

    assert TestFloat4.count_float4(program.uops) == (2, 1)

  @unittest.skipIf(Device.DEFAULT in {"CPU"} and AMX, "CPU with AMX upcasts float up to size 16")
  def test_float4_multidim(self):
    a = Tensor.empty(2, 8).realize()
    b = Tensor.empty(2, 8).realize()
    c = a + b

    s = c.schedule()[0]
    uops = get_program(s.ast, renderer=Device[Device.DEFAULT].renderer,
                       opts=[Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=2)]).uops
    assert TestFloat4.count_float4(uops) == (4, 2)

  @unittest.skipUnless(Device.DEFAULT in {"CPU"} and AMX, "Only CPU with AMX upcasts float up to size 16")
  def test_float4_multidim_amx(self):
    def kernel_for_shape(size, shift):
      a = Tensor.empty(2, size).realize()
      b = Tensor.empty(2, size).realize()
      c = a + b

      s = c.schedule()[0]
      return get_program(s.ast, renderer=Device[Device.DEFAULT].renderer,
                         opts=[Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=shift)]).uops

    sizes = [12, 8, 16]
    shifts = [3, 2, 4]
    expected_upcast_size = [4, 8, 16]
    expected_output = [(6,3), (2,1), (2,1)]

    for i in range(len(sizes)):
      assert TestFloat4.count_float4(kernel_for_shape(sizes[i], shifts[i]), expected_upcast_size[i]) == expected_output[i]

  def test_float4_unaligned_load(self):
    a = Tensor.empty(9).realize().shrink(((1, 9),))
    b = Tensor.empty(9).realize().shrink(((1, 9),))
    c = a + b

    s = c.schedule()[0]
    realized_ast = s.ast
    opts_to_apply = [Opt(op=OptOps.UPCAST, axis=0, arg=4)]
    program = get_program(realized_ast, renderer=Device[Device.DEFAULT].renderer, opts=opts_to_apply)

    assert TestFloat4.count_float4(program.uops) == (0, 1)

  @unittest.skipIf(Device.DEFAULT in {"CPU"} and AMX, "CPU with AMX upcasts float up to size 16")
  def test_float4_multidim_unaligned_load(self):
    a = Tensor.empty(2, 9).realize().shrink(((0, 2), (1, 9),))
    b = Tensor.empty(2, 9).realize().shrink(((0, 2), (1, 9),))
    c = a + b

    s = c.schedule()[0]
    uops = get_program(s.ast, renderer=Device[Device.DEFAULT].renderer,
                       opts=[Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=2)]).uops

    assert TestFloat4.count_float4(uops) == (0, 2)

  @unittest.skipUnless(Device.DEFAULT in {"CPU"} and AMX, "Only CPU with AMX upcasts float up to size 16")
  def test_float4_multidim_unaligned_load_amx(self):
    def kernel_for_shape(size, shift):
      a = Tensor.empty(2, size).realize().shrink(((0, 2), (1, size),))
      b = Tensor.empty(2, size).realize().shrink(((0, 2), (1, size),))
      c = a + b

      s = c.schedule()[0]
      return get_program(s.ast, renderer=Device[Device.DEFAULT].renderer,
                         opts=[Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=shift)]).uops

    sizes = [13, 9, 17]
    shifts = [3, 2, 4]
    expected_upcast_size = [4, 8, 16]
    expected_output = [(0,3), (0,1), (0,1)]

    for i in range(len(sizes)):
      assert TestFloat4.count_float4(kernel_for_shape(sizes[i], shifts[i]), expected_upcast_size[i]) == expected_output[i]

  def test_float4_sometimes_unaligned(self):
    a = Tensor.empty(1, 1, 8).realize()
    b = Tensor.empty(1, 1, 5).realize().shrink(((0, 1), (0, 1), (1, 5)))
    c = a.conv2d(b)
    # only the first and last conv dot products are aligned in a, and b is never aligned, so no
    # float4 should be emitted (the reduce axis of size 4 is the float4 axis here)

    s = c.schedule()[0]
    uops = get_program(s.ast, renderer=Device[Device.DEFAULT].renderer, opts=[Opt(op=OptOps.UNROLL, axis=0, arg=4)]).uops

    assert TestFloat4.count_float4(uops) == (0, 0)

  def test_float4_multidim_sometimes_unaligned(self):
    a = Tensor.empty(1, 1, 7).realize()
    b = Tensor.empty(1, 1, 5).realize().shrink(((0, 1), (0, 1), (1, 5)))
    c = a.conv2d(b)
    # the first conv dot product is aligned in a. If we upcast the output and reduce
    # dimension, then we could do float4 for only that one set of loads, but we currently
    # don't.
    # UPDATE: now we do this fusion

    s = c.schedule()[0]
    uops = get_program(s.ast, renderer=Device[Device.DEFAULT].renderer,
                       opts=[Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.UNROLL, axis=0, arg=0)]).uops

    assert TestFloat4.count_float4(uops) in {(0,1), (1,1)}

  def test_float4_expand(self):
    a = Tensor.empty(9).realize().shrink(((1, 9),))
    b = Tensor.empty(2).realize().reshape((2, 1)).expand((2,4)).reshape((8,))
    c = a + b

    # we will upcast the top axis of sz 4. they should not be coalesced into float4,
    # since the top axis is not contiguous.

    s = c.schedule()[0]
    uops = get_program(s.ast, renderer=Device[Device.DEFAULT].renderer, opts=[Opt(op=OptOps.UPCAST, axis=0, arg=4)]).uops

    assert TestFloat4.count_float4(uops) == (0, 1)

  def test_float4_heterogeneous(self):
    a = Tensor.empty(8).realize()
    b = Tensor.empty(9).realize().shrink(((1, 9),))
    c = a + b

    # should float4 b but not a

    s = c.schedule()[0]
    uops = get_program(s.ast, renderer=Device[Device.DEFAULT].renderer, opts=[Opt(op=OptOps.UPCAST, axis=0, arg=4)]).uops

    assert TestFloat4.count_float4(uops) == (1, 1)

if __name__ == '__main__':
  unittest.main()
