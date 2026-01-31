import unittest
from tinygrad import Tensor
from tinygrad.uop.ops import UPat, Ops, UOp

# NOTE: unlike before base for a realized tensor is always a BUFFER
realized_pattern = UPat(Ops.BUFFER)
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
    is_pattern_uop(c.uop.base, realized_pattern)
    assert d.uop is not d.uop.base

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
    is_pattern(c, UPat(Ops.ADD))
    for s in c.uop.src: is_pattern_uop(s.base, realized_pattern)

if __name__ == '__main__':
  unittest.main()
