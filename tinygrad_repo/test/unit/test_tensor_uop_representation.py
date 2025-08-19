import unittest
from tinygrad import Tensor
from tinygrad.uop.ops import UPat, Ops, UOp

# NOTE: unlike before base for a realized tensor is always a BUFFER
realized_pattern = UPat(Ops.BUFFER)
# after realization, base tensor uops become RESHAPE(BUFFER)
buffer_view_pattern = UPat(Ops.RESHAPE, src=(UPat(Ops.BUFFER),))
const_pattern = UPat(Ops.CONST, src=(UPat(Ops.VIEW, src=(UPat(Ops.DEVICE),),)))
def is_pattern_uop(u:UOp, pat:UPat): assert pat.match(u, {}), f"{u}\nis not\n{pat}"
def is_pattern(ten:Tensor, pat:UPat): is_pattern_uop(ten.uop, pat)

class TestTensorMutates(unittest.TestCase):
  def test_mutate_add(self):
    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    ret = a+b
    pa = a.uop
    pb = b.uop
    pr = ret.uop
    ret.schedule()
    self.assertIsNot(pa, a.uop)
    self.assertIsNot(pb, b.uop)
    self.assertIsNot(pr, ret.uop)
    for t in [a,b,ret]: is_pattern_uop(t.uop.base, realized_pattern)

  def test_reshape_is_same_parent(self):
    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    c = a+b
    d = (a+b).reshape(3,1)
    d.realize()
    is_pattern_uop(d.uop.base, realized_pattern)
    is_pattern_uop(c.uop.base, realized_pattern)
    # NOTE: we keep movement ops on top of the buffer view
    is_pattern_uop(c.uop, UPat(Ops.BUFFER))
    is_pattern_uop(d.uop, UPat(Ops.VIEW, src=(realized_pattern,)))

  def test_reshape_is_same_child(self):
    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    c = a+b
    d = (a+b).reshape(3,1)
    c.realize()
    is_pattern_uop(c.uop.base, realized_pattern)
    is_pattern_uop(d.uop.base, realized_pattern)

class TestTensorUopRepresentation(unittest.TestCase):
  def test_realized(self):
    a = Tensor([1.,2,3]).realize()
    print(a.uop)
    is_pattern_uop(a.uop.base, realized_pattern)

  def test_add_realized(self):
    a = Tensor([1.,2,3]).realize()
    b = Tensor([4.,5,6]).realize()
    c = a+b
    print(c.uop)
    is_pattern(c, UPat(Ops.ADD, src=(realized_pattern, realized_pattern)))

  def test_const_pattern(self):
    a = Tensor(1)
    print(a.uop)
    is_pattern(a, const_pattern) # const in tensor has a DEVICE and VIEW src
    is_pattern(a, UPat.cvar("x")) # even cvar works!

  def test_consts_do_not_realize(self):
    a = Tensor(1)
    print(a.uop)
    pre_realize = a.uop
    a.realize()
    assert a.uop is pre_realize

  def test_viewed_consts_do_not_realize(self):
    a = Tensor.ones(10, 10)
    print(a.uop)
    a.realize()
    is_pattern(a, const_pattern)
    self.assertEqual(a.uop.shape, (10, 10))

  # CONST is EXPAND -> RESHAPE -> CONST -> DEVICE
  def test_consts_dont_have_buffers(self):
    a = Tensor.ones(10, 10)
    buffers_in_parents = [x.op for x in a.uop.toposort() if x.op is Ops.BUFFER]
    self.assertEqual(len(buffers_in_parents), 0)
    is_pattern(a, UPat(Ops.EXPAND, src=(UPat(Ops.RESHAPE, src=(const_pattern,)),)))

  # COPY has a copyin source and a device.
  def test_copyin(self):
    a = Tensor([1.,2,3]).realize()
    c = a.to("TEST")   # NOTE: this isn't checked
    print(c.uop)
    is_pattern(c, UPat(Ops.COPY, src=(realized_pattern, UPat(Ops.DEVICE)), arg=None))

  def test_empty_buf(self):
    a = Tensor.empty(3, 3)
    is_pattern(a, UPat(Ops.RESHAPE, src=(UPat(Ops.BUFFER),)))
    vi = UOp.variable("i", 1, 3).bind(1)
    a = Tensor.empty(3, vi)
    is_pattern(a, UPat(Ops.RESHAPE, src=(UPat(Ops.BUFFER),)))
    self.assertEqual(a.uop.base.buffer.size, 9)

if __name__ == '__main__':
  unittest.main()
