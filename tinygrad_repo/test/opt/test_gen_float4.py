import unittest
from tinygrad import Device, Tensor, Variable, dtypes
from tinygrad.uop.ops import UOp, Ops
from tinygrad.codegen import to_program
from tinygrad.codegen.opt import Opt, OptOps

from test.helpers import replace_opts

@unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "need backends that support float4")
class TestFloat4(unittest.TestCase):
  @staticmethod
  def count_float4(uops: list[UOp], n=4):
    return (len([uop for uop in uops if uop.op is Ops.LOAD and uop.dtype.scalar() == dtypes.float and uop.shape == (4,)]),
            len([uop for uop in uops if uop.op is Ops.STORE and uop.src[1].dtype.scalar() == dtypes.float and uop.shape == (4,)]))
  @staticmethod
  def count_half4(uops: list[UOp]):
    return (len([uop for uop in uops if uop.op is Ops.LOAD and uop.dtype.scalar() == dtypes.half and uop.shape == (4,)]),
            len([uop for uop in uops if uop.op is Ops.STORE and uop.src[1].dtype.scalar() == dtypes.half and uop.shape == (4,)]))

  def test_float4_basic(self):
    a = Tensor.empty(2, 8).realize()
    b = Tensor.empty(2, 8).realize()
    c = a + b

    s = c.schedule_linear().src[0]
    realized_ast = s.src[0]
    opts_to_apply = [Opt(op=OptOps.UPCAST, axis=0, arg=4)]
    program = to_program(replace_opts(realized_ast, opts_to_apply), renderer=Device[Device.DEFAULT].renderer)

    assert TestFloat4.count_float4(tuple(program.src[1].src)) == (2, 1)

  def test_float4_multidim(self):
    a = Tensor.empty(2, 8).realize()
    b = Tensor.empty(2, 8).realize()
    c = a + b

    s = c.schedule_linear().src[0]
    uops = tuple(to_program(replace_opts(s.src[0], [Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=2)]),
                       renderer=Device[Device.DEFAULT].renderer).src[1].src)
    assert TestFloat4.count_float4(uops) == (4, 2)

  def test_float4_unaligned_load(self):
    a = Tensor.empty(9).realize().shrink(((1, 9),))
    b = Tensor.empty(9).realize().shrink(((1, 9),))
    c = a + b

    s = c.schedule_linear().src[0]
    realized_ast = s.src[0]
    opts_to_apply = [Opt(op=OptOps.UPCAST, axis=0, arg=4)]
    program = to_program(replace_opts(realized_ast, opts_to_apply), renderer=Device[Device.DEFAULT].renderer)

    assert TestFloat4.count_float4(tuple(program.src[1].src)) == (0, 1)

  def test_float4_multidim_unaligned_load(self):
    a = Tensor.empty(2, 9).realize().shrink(((0, 2), (1, 9),))
    b = Tensor.empty(2, 9).realize().shrink(((0, 2), (1, 9),))
    c = a + b

    s = c.schedule_linear().src[0]
    uops = tuple(to_program(replace_opts(s.src[0], [Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=2)]),
                       renderer=Device[Device.DEFAULT].renderer).src[1].src)

    assert TestFloat4.count_float4(uops) == (0, 2)

  def test_float4_sometimes_unaligned(self):
    a = Tensor.empty(1, 1, 8).realize()
    b = Tensor.empty(1, 1, 5).realize().shrink(((0, 1), (0, 1), (1, 5)))
    c = a.conv2d(b)
    # only the first and last conv dot products are aligned in a, and b is never aligned, so no
    # float4 should be emitted (the reduce axis of size 4 is the float4 axis here)

    s = c.schedule_linear().src[0]
    uops = tuple(to_program(replace_opts(s.src[0], [Opt(op=OptOps.UNROLL, axis=0, arg=4)]), renderer=Device[Device.DEFAULT].renderer).src[1].src)

    assert TestFloat4.count_float4(uops) == (0, 0)

  def test_float4_multidim_sometimes_unaligned(self):
    a = Tensor.empty(1, 1, 7).realize()
    b = Tensor.empty(1, 1, 5).realize().shrink(((0, 1), (0, 1), (1, 5)))
    c = a.conv2d(b)
    # the first conv dot product is aligned in a. If we upcast the output and reduce
    # dimension, then we could do float4 for only that one set of loads, but we currently
    # don't.
    # UPDATE: now we do this fusion

    s = c.schedule_linear().src[0]
    uops = tuple(to_program(replace_opts(s.src[0], [Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.UNROLL, axis=0, arg=0)]),
                       renderer=Device[Device.DEFAULT].renderer).src[1].src)

    assert TestFloat4.count_float4(uops) in {(0,1), (1,1)}

  def test_float4_expand(self):
    a = Tensor.empty(9).realize().shrink(((1, 9),))
    b = Tensor.empty(2).realize().reshape((2, 1)).expand((2,4)).reshape((8,))
    c = a + b

    # we will upcast the top axis of sz 4. they should not be coalesced into float4,
    # since the top axis is not contiguous.

    s = c.schedule_linear().src[0]
    uops = tuple(to_program(replace_opts(s.src[0], [Opt(op=OptOps.UPCAST, axis=0, arg=4)]), renderer=Device[Device.DEFAULT].renderer).src[1].src)

    assert TestFloat4.count_float4(uops) == (0, 1)

  def test_float4_heterogeneous(self):
    a = Tensor.empty(8).realize()
    b = Tensor.empty(9).realize().shrink(((1, 9),))
    c = a + b

    # should float4 b but not a

    s = c.schedule_linear().src[0]
    uops = tuple(to_program(replace_opts(s.src[0], [Opt(op=OptOps.UPCAST, axis=0, arg=4)]), renderer=Device[Device.DEFAULT].renderer).src[1].src)

    assert TestFloat4.count_float4(uops) == (1, 1)

  def test_float4_aligned_variable(self):
    x = Variable('x', 0, 4, multiple_of=4).bind(4)
    a = Tensor.empty(4).realize()
    b = Tensor.empty(12).realize().shrink(((x, x+4),))
    c = a + b

    # should float4 both

    s = c.linear_with_vars()[0].src[0]
    uops = tuple(to_program(replace_opts(s.src[0], [Opt(op=OptOps.UPCAST, axis=0, arg=4)]), renderer=Device[Device.DEFAULT].renderer).src[1].src)

    assert TestFloat4.count_float4(uops) == (2, 1)

  def test_float4_unaligned_variable(self):
    x = Variable('x', 0, 4, multiple_of=2).bind(4)
    a = Tensor.empty(4).realize()
    b = Tensor.empty(12).realize().shrink(((x, x+4),))
    c = a + b

    # should float4 a but not b

    s = c.linear_with_vars()[0].src[0]
    uops = tuple(to_program(replace_opts(s.src[0], [Opt(op=OptOps.UPCAST, axis=0, arg=4)]), renderer=Device[Device.DEFAULT].renderer).src[1].src)

    assert TestFloat4.count_float4(uops) == (1, 1)

if __name__ == '__main__':
  unittest.main()
