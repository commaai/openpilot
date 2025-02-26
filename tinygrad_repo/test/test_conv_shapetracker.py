#!/usr/bin/env python
import unittest
from tinygrad.ops import Ops
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d
from tinygrad.shape.shapetracker import ShapeTracker, View
from tinygrad.helpers import prod
from test.unit.test_shapetracker import shapetracker_getitem

class TestConvShapetracker(unittest.TestCase):
  def test_conv_3x3_one_view(self):
    conv = Conv2d(16, 32, (3, 3))
    # first run to init the weights, they are scheduled.
    conv(Tensor.empty(1, 16, 10, 10)).schedule()
    # run it again to get the kernels
    sched = [si for si in conv(Tensor.empty(1, 16, 10, 10)).schedule() if si.ast.op is Ops.SINK]
    assert len(sched) == 1, f"conv should only have one kernel, getting {len(sched)}"
    for st in [x.st_arg for x in sched[0].ast.toposort if x.op is Ops.LOAD]:
      assert len(st.views) == 1

  def test_conv_2x2_backward_one_view(self):
    X = Tensor.rand(1, 1, 3, 3, requires_grad=True)
    conv = Conv2d(1, 1, (2, 2), bias=False)
    conv(X).mean().backward()
    si = X.grad.schedule()[-1]
    print(si)
    ldb = [x for x in si.ast.toposort if x.op is Ops.LOAD][0]
    st: ShapeTracker = ldb.st_arg.simplify()
    print(si.bufs[1].size)
    self.assertEqual(si.bufs[1].size, st.real_size())
    for v in st.views: print(v)

    # same st
    test_st = ShapeTracker((
      View(shape=(1, 1, 2, 4, 2, 4), strides=(0, 0, 2, 8, 1, 4), offset=0, mask=((0, 1), (0, 1), (0, 2), (0, 2), (0, 2), (0, 2)), contiguous=False),
      View(shape=(1, 1, 1, 1, 3, 3, 3, 3), strides=(0, 0, 0, 0, 24, 8, 3, 1), offset=0,
           mask=((0, 1), (0, 1), (0, 1), (0, 1), (0, 2), (0, 3), (0, 2), (0, 3)), contiguous=False)))
    #test_st = ShapeTracker((
    #  View(shape=(2,4), strides=(1,4), offset=0, mask=None, contiguous=False),
    #)).simplify()
      #View(shape=(1, 1, 2, 4, 2, 4), strides=(0, 0, 2, 8, 1, 4), offset=0, mask=((0, 1), (0, 1), (0, 2), (0, 2), (0, 2), (0, 2)), contiguous=False),
      #View(shape=(1, 1, 1, 1, 3, 3, 3, 3), strides=(0, 0, 0, 0, 24, 8, 3, 1), offset=0,
      #     mask=((0, 1), (0, 1), (0, 1), (0, 1), (0, 2), (0, 3), (0, 2), (0, 3)), contiguous=False))).simplify()
    print("*** new ***")
    for v in test_st.views: print(v)
    for i in range(prod(st.shape)):
      i1, i2 = shapetracker_getitem(st, i), shapetracker_getitem(test_st, i)
      print(i, i1, i2, si.bufs[1].size, i1==i2)
      #self.assertEqual(i1, i2)

    with self.assertRaises(AssertionError):
      assert len(st.views) <= 2

if __name__ == '__main__':
  unittest.main()
