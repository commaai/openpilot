import unittest
from tinygrad import Tensor
from tinygrad.ops import UPat, Ops, UOp

# NOTE: unlike before base for a realized tensor is always a BUFFER
realized_pattern = UPat(Ops.BUFFER)
# after realization, base tensor uops become RESHAPE(BUFFER)
buffer_view_pattern = UPat(Ops.RESHAPE, src=(UPat(Ops.BUFFER),))
const_pattern = UPat(Ops.CONST, src=(UPat(Ops.VIEW, src=(UPat(Ops.DEVICE),),)))
def is_pattern_uop(u:UOp, pat:UPat): assert pat.match(u, {}), f"{u}\nis not\n{pat}"
def is_pattern(ten:Tensor, pat:UPat): is_pattern_uop(ten.lazydata, pat)

class TestTensorMutates(unittest.TestCase):
  def test_mutate_add(self):
    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    ret = a+b
    pa = a.lazydata
    pb = b.lazydata
    pr = ret.lazydata
    ret.schedule()
    self.assertIsNot(pa, a.lazydata)
    self.assertIsNot(pb, b.lazydata)
    self.assertIsNot(pr, ret.lazydata)
    for t in [a,b,ret]: is_pattern_uop(t.lazydata.base, realized_pattern)

  def test_reshape_is_same_parent(self):
    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    c = a+b
    d = (a+b).reshape(3,1)
    d.realize()
    is_pattern_uop(d.lazydata.base, realized_pattern)
    is_pattern_uop(c.lazydata.base, realized_pattern)
    # NOTE: we keep movement ops on top of the buffer view
    is_pattern_uop(c.lazydata, UPat(Ops.BUFFER))
    is_pattern_uop(d.lazydata, UPat(Ops.VIEW, src=(realized_pattern,)))

  def test_reshape_is_same_child(self):
    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    c = a+b
    d = (a+b).reshape(3,1)
    c.realize()
    is_pattern_uop(c.lazydata.base, realized_pattern)
    is_pattern_uop(d.lazydata.base, realized_pattern)

class TestTensorUopRepresentation(unittest.TestCase):
  def test_realized(self):
    a = Tensor([1.,2,3]).realize()
    print(a.lazydata)
    is_pattern_uop(a.lazydata.base, realized_pattern)

  def test_add_realized(self):
    a = Tensor([1.,2,3]).realize()
    b = Tensor([4.,5,6]).realize()
    c = a+b
    print(c.lazydata)
    is_pattern(c, UPat(Ops.ADD, src=(realized_pattern, realized_pattern)))

  def test_const_pattern(self):
    a = Tensor(1)
    print(a.lazydata)
    is_pattern(a, const_pattern) # const in tensor has a DEVICE and VIEW src
    is_pattern(a, UPat.cvar("x")) # even cvar works!

  def test_consts_do_not_realize(self):
    a = Tensor(1)
    print(a.lazydata)
    pre_realize = a.lazydata
    a.realize()
    assert a.lazydata is pre_realize

  def test_viewed_consts_do_not_realize(self):
    a = Tensor.ones(10, 10)
    print(a.lazydata)
    a.realize()
    is_pattern(a, const_pattern)
    self.assertEqual(a.lazydata.shape, (10, 10))

  # currently, CONSTs have a "fake" BUFFER. this should be fixed
  # current:
  # UOp(Ops.EXPAND, dtypes.float, arg=(10, 10), src=(
  #   UOp(Ops.RESHAPE, dtypes.float, arg=(1, 1), src=(
  #     UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
  #       UOp(Ops.BUFFER, dtypes.float, arg=(-1, 'METAL', 1), src=()),
  #       UOp(Ops.CONST, dtypes.float, arg=1.0, src=()),)),)),))
  # expected:
  # UOp(Ops.EXPAND, dtypes.float, arg=(10, 10), src=(
  #   UOp(Ops.RESHAPE, dtypes.float, arg=(1, 1), src=(
  #     UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
  #       UOp(Ops.CONST, dtypes.float, arg=1.0, src=(
  #         UOp(Ops.DEVICE, dtypes.void, arg="METAL", src=()),)),)),))
  def test_consts_dont_have_buffers(self):
    a = Tensor.ones(10, 10)
    print(a.lazydata)
    buffers_in_parents = [x.op for x in a.lazydata.toposort() if x.op is Ops.BUFFER]
    self.assertEqual(len(buffers_in_parents), 0)

  # currently, COPY has an extra BUFFER on the output
  # current:
  # UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(3,), strides=(1,), offset=0, mask=None, contiguous=True),)), src=(
  #   UOp(Ops.BUFFER, dtypes.float, arg=(2, 'TEST', 3), src=()),
  #   UOp(Ops.COPY, dtypes.float, arg=('TEST', False), src=(
  #     UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(3,), strides=(1,), offset=0, mask=None, contiguous=True),)), src=(
  #       UOp(Ops.BUFFER, dtypes.float, arg=(1, 'METAL', 3), src=()),)),)),))
  # expected:
  # UOp(Ops.COPY, dtypes.float, arg=('TEST', False), src=(
  #   UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(3,), strides=(1,), offset=0, mask=None, contiguous=True),)), src=(
  #     UOp(Ops.BUFFER, dtypes.float, arg=(1, 'METAL', 3), src=()),))
  # update: now the arg is just a single bool, the first source is a device.
  def test_copyin(self):
    a = Tensor([1.,2,3]).realize()
    c = a.to("TEST")   # NOTE: this isn't checked
    print(c.lazydata)
    is_pattern(c, UPat(Ops.COPY, src=(realized_pattern, UPat(Ops.DEVICE))))

  def test_empty_buf(self):
    a = Tensor.empty(3, 3)
    is_pattern(a, UPat(Ops.RESHAPE, src=(UPat(Ops.BUFFER),)))
    vi = UOp.variable("i", 1, 3).bind(1)
    a = Tensor.empty(3, vi)
    is_pattern(a, UPat(Ops.RESHAPE, src=(UPat(Ops.BUFFER),)))
    self.assertEqual(a.lazydata.base.buffer.size, 9)

if __name__ == '__main__':
  unittest.main()
