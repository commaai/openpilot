import unittest
from tinygrad import Tensor
from tinygrad.ops import UPat, Ops

realized_pattern = UPat(Ops.VIEW, src=(UPat(Ops.BUFFER),))
const_pattern = UPat(Ops.VIEW, src=(UPat(Ops.BUFFER), UPat(Ops.CONST)))
def is_pattern(ten:Tensor, pat:UPat): assert pat.match(ten.lazydata, {})

class TestTensorUopRepresentation(unittest.TestCase):
  def test_realized(self):
    a = Tensor([1.,2,3]).realize()
    print(a.lazydata)
    is_pattern(a, realized_pattern)

  def test_add_realized(self):
    a = Tensor([1.,2,3]).realize()
    b = Tensor([4.,5,6]).realize()
    c = a+b
    print(c.lazydata)
    is_pattern(c, UPat(Ops.ADD, src=(realized_pattern, realized_pattern)))

  def test_const_pattern(self):
    a = Tensor(1)
    print(a.lazydata)
    is_pattern(a, const_pattern)

  def test_consts_do_not_realize(self):
    a = Tensor(1)
    print(a.lazydata)
    pre_realize = a.lazydata
    a.realize()
    assert a.lazydata is pre_realize

  def test_viewed_consts_do_not_realize(self):
    a = Tensor.ones(10, 10)
    print(a.lazydata)
    pre_realize = a.lazydata
    a.realize()
    assert a.lazydata is pre_realize

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
  @unittest.expectedFailure
  def test_consts_dont_have_buffers(self):
    a = Tensor.ones(10, 10)
    print(a.lazydata)
    buffers_in_parents = [x.op for x in a.lazydata.toposort if x.op is Ops.BUFFER]
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
  @unittest.expectedFailure
  def test_copyin(self):
    a = Tensor([1.,2,3]).realize()
    c = a.to("TEST")   # NOTE: this isn't checked
    print(c.lazydata)
    # NOTE: this is wrong, COPY has an extra buffer for some reason
    is_pattern(c, UPat(Ops.COPY, src=(realized_pattern,)))

if __name__ == '__main__':
  unittest.main()
