import unittest, itertools
from tinygrad.dtype import dtypes
from tinygrad.ops import Ops, UOp, GroupOp # noqa: F401
from tinygrad.ops import PatternMatcher, UPat

class TestPatternMatcher(unittest.TestCase):
  def test_simple_match(self):
    matcher = PatternMatcher([(UPat(Ops.CONST, name="x", dtype=dtypes.float), lambda x: x)])
    c1 = UOp(Ops.CONST, dtypes.float, arg=1.0)
    c2 = UOp(Ops.CONST, dtypes.int, arg=1)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_upat_any(self):
    def test(a, x=None, y=None, z=None):
      #print(x,y,z)
      if y is not None: return a+y
    matcher = PatternMatcher([
      (UPat.var("a")+UPat.any(UPat.var("x"), UPat.var("y"), UPat.var("z")), test),
    ])
    v1 = UOp.variable("a", 0, 10)
    v2 = UOp.variable("b", 0, 10)
    c1 = v1+v2
    self.assertEqual(matcher.rewrite(c1), c1)

  @unittest.skip("closures aren't supported on pattern matchers")
  def test_match_sz_0(self):
    match_cnt = 0
    def fxn(x):
      nonlocal match_cnt
      match_cnt += 1
      assert len(x.src) == 0
      return UOp(Ops.CONST, src=(UOp(Ops.CONST),))
    matcher = PatternMatcher([(UPat(Ops.CONST, src=(), name="x"), fxn)])
    c1 = UOp(Ops.CONST, dtypes.float, arg=1.0)
    # second rewrite shouldn't match anything
    c1 = matcher.rewrite(c1)
    c1 = matcher.rewrite(c1)
    self.assertEqual(match_cnt, 1)

  def test_match_sz_0_ctx(self):
    def fxn(ctx, x):
      ctx.append(True)
      assert len(x.src) == 0
      return UOp(Ops.CONST, src=(UOp(Ops.CONST),))
    matcher = PatternMatcher([(UPat(Ops.CONST, src=(), name="x"), fxn)])
    c1 = UOp(Ops.CONST, dtypes.float, arg=1.0)
    # second rewrite shouldn't match anything
    ctx = []
    c1 = matcher.rewrite(c1, ctx)
    c1 = matcher.rewrite(c1, ctx)
    self.assertEqual(len(ctx), 1)

  def test_uop(self):
    matcher = PatternMatcher([(UPat(Ops.CONST, name="x"), lambda x: x)])
    c1 = UOp(Ops.CONST, dtypes.float, arg=1.0)
    c2 = UOp(Ops.ADD, dtypes.float, (c1, c1))
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_uop_set(self):
    matcher = PatternMatcher([(UPat((Ops.CONST, Ops.CAST), name="x"), lambda x: x)])
    c1 = UOp(Ops.CONST, dtypes.bool, arg=False)
    c2 = UOp(Ops.CAST, dtypes.int, (c1,))
    c3 = UOp(Ops.CONST, dtypes.float, arg=1.0)
    c4 = UOp(Ops.ADD, dtypes.float, (c3, c3))
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), c2)
    self.assertEqual(matcher.rewrite(c4), None)

  def test_arg(self):
    matcher = PatternMatcher([
      (UPat(Ops.CONST, arg=0, name="x"), lambda x: x),
      (UPat(Ops.CONST, arg=False, name="x"), lambda x: x),
      (UPat(Ops.MAX, name="x"), lambda x: x),
    ])
    c1 = UOp(Ops.CONST, dtypes.float, arg=0.0)
    c2 = UOp(Ops.CONST, dtypes.bool, arg=False)
    c3 = UOp(Ops.MAX, dtypes.float, (c1, c1))
    c4 = UOp(Ops.MUL, dtypes.float, (c1, c1))
    c5 = UOp(Ops.CONST, dtypes.int, arg=-1)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), c2)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c4), None)
    self.assertEqual(matcher.rewrite(c5), None)

  def test_filter_arg(self):
    matcher = PatternMatcher([
      (UPat(Ops.MUL, src=[UPat(Ops.CONST, name="c"), UPat(Ops.CONST, arg=2)], name="x"),
       lambda x,c: x if c.arg in {1, -1} else None)
    ])
    y1 = UOp(Ops.CONST, dtypes.int, arg=1)
    y2 = UOp(Ops.CONST, dtypes.int, arg=2)
    y3 = UOp(Ops.CONST, dtypes.int, arg=-1)
    c1 = UOp(Ops.MUL, dtypes.int, (y1, y2))
    c2 = UOp(Ops.MUL, dtypes.int, (y2, y2))
    c3 = UOp(Ops.MUL, dtypes.int, (y3, y2))
    c4 = UOp(Ops.MUL, dtypes.int, (y2, y1))
    c5 = UOp(Ops.MUL, dtypes.int, (y2, y3))
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c4), c4)
    self.assertEqual(matcher.rewrite(c5), c5)

  def test_dup_name(self):
    matcher = PatternMatcher([(UPat(GroupOp.ALU, name="x", src=(UPat(Ops.CONST, name="y"), UPat(Ops.CONST, name="y"))), lambda x, y: x)])
    y1 = UOp(Ops.CONST, dtypes.float, arg=1.0)
    y2 = UOp(Ops.CONST, dtypes.float, arg=1.0)
    c1 = UOp(Ops.ADD, dtypes.float, (y1, y1))
    c2 = UOp(Ops.ADD, dtypes.float, (y1, y2))
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), c1)

  def test_dtype(self):
    matcher = PatternMatcher([(UPat(Ops.CONST, name="x", dtype=dtypes.float32), lambda x: x)])
    c1 = UOp(Ops.CONST, dtypes.float, arg=1.0)
    c2 = UOp(Ops.CONST, dtypes.float64, arg=1.0)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_dtype_set(self):
    matcher = PatternMatcher([(UPat(Ops.CONST, name="x", dtype={dtypes.float32, dtypes.float64}), lambda x: x)])
    c1 = UOp(Ops.CONST, dtypes.float, arg=1.0)
    c2 = UOp(Ops.CONST, dtypes.float64, arg=1.0)
    c3 = UOp(Ops.CONST, dtypes.float16, arg=1.0)
    c4 = UOp(Ops.CONST, dtypes.int, arg=1)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), c2)
    self.assertEqual(matcher.rewrite(c3), None)
    self.assertEqual(matcher.rewrite(c4), None)

  def test_src_one(self):
    matcher = PatternMatcher([(UPat(GroupOp.ALU, name="x", src=(UPat(Ops.CONST), UPat(Ops.CONST))), lambda x: x)])
    c1 = UOp(Ops.CONST, dtypes.float, arg=1.0)
    c2 = UOp(Ops.CONST, dtypes.float, arg=2.0)
    c3 = UOp(Ops.ADD, dtypes.float, (c1,c2))
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c2), None)
    # that CONST/ALU -> ALU/CONST rewrite is now instant
    """
    matcher = PatternMatcher([(UPat(GroupOp.ALU, name="x", src=(UPat(Ops.CONST), UPat(GroupOp.ALU))), lambda x: x)])
    c4 = UOp(Ops.ADD, dtypes.float, (c1,c3))
    c5 = UOp(Ops.ADD, dtypes.float, (c3,c1))
    self.assertEqual(matcher.rewrite(c3), None)
    self.assertEqual(matcher.rewrite(c4), c4)
    self.assertEqual(matcher.rewrite(c5), None)
    """

  def test_src_permutations(self):
    matcher = PatternMatcher([(UPat(GroupOp.ALU, name="x", src=[UPat(Ops.CONST), UPat(GroupOp.ALU)]), lambda x: x)])
    c1 = UOp(Ops.CONST, dtypes.float, arg=1.0)
    c2 = UOp(Ops.CONST, dtypes.float, arg=2.0)
    c3 = UOp(Ops.ADD, dtypes.float, (c1,c2))
    c4 = UOp(Ops.ADD, dtypes.float, (c3,c2))
    c5 = UOp(Ops.ADD, dtypes.float, (c2,c3))
    c6 = UOp(Ops.ADD, dtypes.float, (c3,c4))
    self.assertEqual(matcher.rewrite(c3), None)
    self.assertEqual(matcher.rewrite(c4), c4)
    self.assertEqual(matcher.rewrite(c5), c5)
    self.assertEqual(matcher.rewrite(c6), None)

  def test_src_repeat(self):
    matcher = PatternMatcher([(UPat(GroupOp.ALU, name="x", src=UPat(Ops.CONST)), lambda x: x)])
    c1 = UOp(Ops.CONST, dtypes.float, arg=1.0)
    c2 = UOp(Ops.CONST, dtypes.float, arg=2.0)
    c3 = UOp(Ops.ADD, dtypes.float, (c1,c2))
    c4 = UOp(Ops.ADD, dtypes.float, (c2,c3))
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c4), None)

  def test_allow_len(self):
    matcher = PatternMatcher([(UPat(Ops.MULACC, name="x", src=(UPat(Ops.CONST),), allow_any_len=True), lambda x: x)])
    c1 = UOp(Ops.CONST, dtypes.float, arg=1.0)
    c2 = UOp(Ops.CONST, dtypes.float, arg=2.0)
    c3 = UOp(Ops.CONST, dtypes.float, arg=3.0)
    c4 = UOp(Ops.EXP2, dtypes.float, (c1,))
    c5 = UOp(Ops.ADD, dtypes.float, (c1,c2))
    c6 = UOp(Ops.MULACC, dtypes.float, (c1,c2,c3))
    self.assertEqual(matcher.rewrite(c4), None)
    self.assertEqual(matcher.rewrite(c5), None)
    self.assertEqual(matcher.rewrite(c6), c6)

  def test_deep_src_permutations(self):
    c1 = UOp(Ops.CONST, dtypes.float, arg=1.0)
    c2 = UOp(Ops.CONST, dtypes.float, arg=2.0)
    u1 = (c1 + c2) + c1
    u2 = (c2 + c1) + c1
    matcher = PatternMatcher([
      (UPat(GroupOp.ALU, src=[UPat(GroupOp.ALU, src=[UPat(name='a'), UPat(name='b')]), UPat(name='b')]), lambda a,b: b)
    ])
    self.assertIsNotNone(matcher.rewrite(u1))
    self.assertIsNotNone(matcher.rewrite(u2))

  def _assert_eq_upat(self, a:UPat, b:UPat):
    assert (sorted(map(str,a.op)) if a.op else [] == (sorted(map(str,b.op)) if b.op else []))
    assert (sorted(a.dtype) if a.dtype else [] == (sorted(b.dtype) if b.dtype else []))
    assert (a.name, type(a.src)) == (b.name, type(b.src))
    def simple_src(u:UPat):
      if u.src is None: return []
      if isinstance(u.src, itertools.repeat): return next(u.src[0])
      return u.src[0]
    for a,b in zip(simple_src(a), simple_src(b)): self._assert_eq_upat(a, b)

if __name__ == '__main__':
  unittest.main(verbosity=2)
