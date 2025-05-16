import unittest, random
from tinygrad.dtype import dtypes
from tinygrad.ops import print_uops, UOp, Ops
from tinygrad.codegen.linearize import block_reorder
from tinygrad.renderer.cstyle import OpenCLRenderer

def is_toposorted(lst:list[UOp]):
  seen = set()
  for u in lst:
    if any(p not in seen for p in u.src): return False
    seen.add(u)
  return True

class TestBlockReorder(unittest.TestCase):
  def _test_randomize(self, golden:list[UOp]):
    # test random order is always same
    for _ in range(50):
      # shuffle and form a valid toposort
      lst = golden[:]
      random.shuffle(lst)
      topolst = []
      for u in lst:
        for p in u.toposort():
          if p not in topolst: topolst.append(p)
      assert is_toposorted(topolst)

      for x,y in zip(golden, this_order:=block_reorder(topolst)):
        if x is not y:
          print_uops(golden)
          print_uops(this_order)
        self.assertIs(x, y)

  def _test_render(self, golden:list[UOp]):
    return OpenCLRenderer().render(golden)

  def test_loads(self):
    a = UOp(Ops.DEFINE_GLOBAL, dtype=dtypes.float.ptr(), arg=0)
    b = UOp(Ops.DEFINE_GLOBAL, dtype=dtypes.float.ptr(), arg=1)
    c = UOp(Ops.DEFINE_GLOBAL, dtype=dtypes.float.ptr(), arg=2)
    v1 = UOp(Ops.SPECIAL, dtype=dtypes.int, arg=("gidx0", 4))
    v2 = UOp(Ops.SPECIAL, dtype=dtypes.int, arg=("gidx1", 4))
    v1 = v1*27
    v2 = v2*4
    loads = [
      a.index(v1).load(dtype=dtypes.float),
      a.index(v1+1).load(dtype=dtypes.float),
      a.index(v1+2).load(dtype=dtypes.float),
      a.index(v1+3).load(dtype=dtypes.float),
      b.index(v2).load(dtype=dtypes.float),
      b.index(v2+1).load(dtype=dtypes.float),
      b.index(v2+2).load(dtype=dtypes.float),
      b.index(v2+3).load(dtype=dtypes.float)]
    #random.shuffle(loads)
    sink = c.store(sum(loads)).sink()

    # determine golden order
    golden = block_reorder(sink.toposort())

    # render for test
    print(self._test_render(golden))
    #print_uops(golden)

    # assert the loads are in this order
    self.assertListEqual([g.src[0].src[1].render() for g in golden if g.op is Ops.LOAD],
                         ['(gidx1*4)', '((gidx1*4)+1)', '((gidx1*4)+2)', '((gidx1*4)+3)',
                          '(gidx0*27)', '((gidx0*27)+1)', '((gidx0*27)+2)', '((gidx0*27)+3)'])

    # assert math is after loads
    first_math = [i for i,g in enumerate(golden) if g.op is Ops.ADD and g.dtype == dtypes.float][0]
    assert not any(x.op is Ops.LOAD for x in golden[first_math:])

    # confirm the sort is stable
    self._test_randomize(golden)

if __name__ == '__main__':
  unittest.main()
