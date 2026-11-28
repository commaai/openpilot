import unittest, itertools, dis
from tinygrad.helpers import DEBUG
from tinygrad.uop.upat import _get_code, upat_compile
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, GroupOp # noqa: F401
from tinygrad.uop.ops import PatternMatcher, UPat, rewrite_group

# the rewrite uses a global defined after its PatternMatcher, like codegen's lower_call -> to_program
late_pms = [PatternMatcher([(UPat(Ops.CONST, name="x"), lambda x: late_global(x))], compiled=c) for c in (False, True)]
def late_global(x:UOp) -> UOp: return x.sink()

class TestPatternMatcher(unittest.TestCase):
  def test_simple_match(self):
    matcher = PatternMatcher([(UPat(Ops.CONST, name="x", dtype=dtypes.weakfloat), lambda x: x.sink())])
    c1 = UOp.const(1.0)
    c2 = UOp.const(1)
    self.assertEqual(matcher.rewrite(c1), c1.sink())
    self.assertEqual(matcher.rewrite(c2), None)

  def test_upat_any(self):
    def test(a, x=None, y=None, z=None):
      #print(x,y,z)
      if y is not None: return (a+y).sink()
    matcher = PatternMatcher([
      (UPat.var("a")+UPat.any(UPat.var("x"), UPat.var("y"), UPat.var("z")), test),
    ])
    v1 = UOp.variable("a", 0, 10)
    v2 = UOp.variable("b", 0, 10)
    c1 = v1+v2
    self.assertEqual(matcher.rewrite(c1), c1.sink())

  def test_minimum_len(self):
    matcher = PatternMatcher([
      (UPat(Ops.NOOP, src=(UPat(Ops.NOOP), UPat(Ops.NOOP)), allow_any_len=True), lambda: True),
    ])
    self.assertTrue(matcher.rewrite(UOp(Ops.NOOP, src=(UOp(Ops.NOOP), UOp(Ops.NOOP), UOp(Ops.NOOP),))))
    self.assertTrue(matcher.rewrite(UOp(Ops.NOOP, src=(UOp(Ops.NOOP), UOp(Ops.NOOP)))))
    self.assertIsNone(matcher.rewrite(UOp(Ops.NOOP, src=(UOp(Ops.NOOP),))))

  @unittest.skip("closures aren't supported on pattern matchers")
  def test_match_sz_0(self):
    match_cnt = 0
    def fxn(x):
      nonlocal match_cnt
      match_cnt += 1
      assert len(x.src) == 0
      return UOp(Ops.CONST, src=(UOp(Ops.CONST),))
    matcher = PatternMatcher([(UPat(Ops.CONST, src=(), name="x"), fxn)])
    c1 = UOp(Ops.CONST, arg=1.0)
    # second rewrite shouldn't match anything
    c1 = matcher.rewrite(c1)
    c1 = matcher.rewrite(c1)
    self.assertEqual(match_cnt, 1)

  def test_match_sz_0_ctx(self):
    def fxn(ctx, x):
      ctx.append(True)
      assert len(x.src) == 0
      return x.replace(src=(UOp(Ops.NOOP),))
    matcher = PatternMatcher([(UPat(Ops.NOOP, src=(), name="x"), fxn)])
    c1 = UOp(Ops.NOOP)
    # second rewrite shouldn't match anything
    ctx = []
    c1 = matcher.rewrite(c1, ctx)
    c1 = matcher.rewrite(c1, ctx)
    self.assertEqual(len(ctx), 1)

  def test_uop(self):
    matcher = PatternMatcher([(UPat(Ops.CONST, name="x"), lambda x: x.sink())])
    c1 = UOp.const(1.0)
    c2 = UOp(Ops.ADD, src=(c1, c1))
    self.assertEqual(matcher.rewrite(c1), c1.sink())
    self.assertEqual(matcher.rewrite(c2), None)

  def test_uop_set(self):
    matcher = PatternMatcher([(UPat((Ops.CONST, Ops.CAST), name="x"), lambda x: x.sink())])
    c1 = UOp.const(False)
    c2 = UOp(Ops.CAST, arg=dtypes.int, src=(c1,))
    c3 = UOp.const(1.0)
    c4 = UOp(Ops.ADD, src=(c3, c3))
    self.assertEqual(matcher.rewrite(c1), c1.sink())
    self.assertEqual(matcher.rewrite(c2), c2.sink())
    self.assertEqual(matcher.rewrite(c4), None)

  def test_arg(self):
    matcher = PatternMatcher([
      (UPat(Ops.CONST, arg=0, name="x"), lambda x: x.sink()),
      (UPat(Ops.CONST, arg=False, name="x"), lambda x: x.sink()),
      (UPat(Ops.MAX, name="x"), lambda x: x.sink()),
    ])
    c1 = UOp.const(0.0)
    c2 = UOp.const(False)
    c3 = UOp(Ops.MAX, src=(c1, c1))
    c4 = UOp(Ops.MUL, src=(c1, c1))
    c5 = UOp.const(-1)
    self.assertEqual(matcher.rewrite(c1), c1.sink())
    self.assertEqual(matcher.rewrite(c2), c2.sink())
    self.assertEqual(matcher.rewrite(c3), c3.sink())
    self.assertEqual(matcher.rewrite(c4), None)
    self.assertEqual(matcher.rewrite(c5), None)

  def test_filter_arg(self):
    matcher = PatternMatcher([
      (UPat(Ops.MUL, src=[UPat(Ops.CONST, name="c"), UPat(Ops.CONST, arg=2)], name="x"),
       lambda x,c: x.sink() if c.val in {1, -1} else None)
    ])
    y1 = UOp.const(1)
    y2 = UOp.const(2)
    y3 = UOp.const(-1)
    c1 = UOp(Ops.MUL, src=(y1, y2))
    c2 = UOp(Ops.MUL, src=(y2, y2))
    c3 = UOp(Ops.MUL, src=(y3, y2))
    c4 = UOp(Ops.MUL, src=(y2, y1))
    c5 = UOp(Ops.MUL, src=(y2, y3))
    self.assertEqual(matcher.rewrite(c1), c1.sink())
    self.assertEqual(matcher.rewrite(c2), None)
    self.assertEqual(matcher.rewrite(c3), c3.sink())
    self.assertEqual(matcher.rewrite(c4), c4.sink())
    self.assertEqual(matcher.rewrite(c5), c5.sink())

  def test_dup_name(self):
    matcher = PatternMatcher([(UPat(GroupOp.ALU, name="x", src=(UPat(Ops.CONST, name="y"), UPat(Ops.CONST, name="y"))), lambda x, y: x.sink())])
    y1 = UOp.const(1.0)
    y2 = UOp.const(1.0)
    c1 = UOp(Ops.ADD, src=(y1, y1))
    c2 = UOp(Ops.ADD, src=(y1, y2))
    self.assertEqual(matcher.rewrite(c1), c1.sink())
    self.assertEqual(matcher.rewrite(c2), c1.sink())

  def test_dtype(self):
    # a concrete const dtype lives on the pair's CAST
    matcher = PatternMatcher([(UPat(Ops.CAST, name="x", dtype=dtypes.float32), lambda x: x.sink())])
    c1 = UOp.const(1.0).cast(dtypes.float32)
    c2 = UOp.const(1.0).cast(dtypes.float64)
    self.assertEqual(matcher.rewrite(c1), c1.sink())
    self.assertEqual(matcher.rewrite(c2), None)

  def test_dtype_set(self):
    matcher = PatternMatcher([(UPat(Ops.CAST, name="x", dtype={dtypes.float32, dtypes.float64}), lambda x: x.sink())])
    c1 = UOp.const(1.0).cast(dtypes.float32)
    c2 = UOp.const(1.0).cast(dtypes.float64)
    c3 = UOp.const(1.0).cast(dtypes.float16)
    c4 = UOp.const(1).cast(dtypes.int)
    self.assertEqual(matcher.rewrite(c1), c1.sink())
    self.assertEqual(matcher.rewrite(c2), c2.sink())
    self.assertEqual(matcher.rewrite(c3), None)
    self.assertEqual(matcher.rewrite(c4), None)

  def test_src_one(self):
    matcher = PatternMatcher([(UPat(GroupOp.ALU, name="x", src=(UPat(Ops.CONST), UPat(Ops.CONST))), lambda x: x.sink())])
    c1 = UOp.const(1.0)
    c2 = UOp.const(2.0)
    c3 = UOp(Ops.ADD, src=(c1,c2))
    self.assertEqual(matcher.rewrite(c3), c3.sink())
    self.assertEqual(matcher.rewrite(c2), None)
    # that CONST/ALU -> ALU/CONST rewrite is now instant
    """
    matcher = PatternMatcher([(UPat(GroupOp.ALU, name="x", src=(UPat(Ops.CONST), UPat(GroupOp.ALU))), lambda x: x)])
    c4 = UOp(Ops.ADD, src=(c1,c3))
    c5 = UOp(Ops.ADD, src=(c3,c1))
    self.assertEqual(matcher.rewrite(c3), None)
    self.assertEqual(matcher.rewrite(c4), c4)
    self.assertEqual(matcher.rewrite(c5), None)
    """

  def test_src_permutations(self):
    matcher = PatternMatcher([(UPat(GroupOp.ALU, name="x", src=[UPat(Ops.CONST), UPat(GroupOp.ALU)]), lambda x: x.sink())])
    c1 = UOp.const(1.0)
    c2 = UOp.const(2.0)
    c3 = UOp(Ops.ADD, src=(c1,c2))
    c4 = UOp(Ops.ADD, src=(c3,c2))
    c5 = UOp(Ops.ADD, src=(c2,c3))
    c6 = UOp(Ops.ADD, src=(c3,c4))
    self.assertEqual(matcher.rewrite(c3), None)
    self.assertEqual(matcher.rewrite(c4), c4.sink())
    self.assertEqual(matcher.rewrite(c5), c5.sink())
    self.assertEqual(matcher.rewrite(c6), None)

  def test_src_repeat(self):
    matcher = PatternMatcher([(UPat(GroupOp.ALU, name="x", src=UPat(Ops.CONST)), lambda x: x.sink())])
    c1 = UOp.const(1.0)
    c2 = UOp.const(2.0)
    c3 = UOp(Ops.ADD, src=(c1,c2))
    c4 = UOp(Ops.ADD, src=(c2,c3))
    self.assertEqual(matcher.rewrite(c3), c3.sink())
    self.assertEqual(matcher.rewrite(c4), None)

  def test_src_repeat_named_first_mismatch(self):
    matcher = PatternMatcher([(UPat(Ops.STACK, src=UPat(Ops.CMPLT, src=(UPat.var("y"), UPat()))), lambda y: y)])
    a = UOp.variable("a", 0, 10)
    self.assertIsNone(matcher.rewrite(UOp.stack(UOp.const(False, dtypes.bool), a < 5)))
    self.assertEqual(matcher.rewrite(UOp.stack(a < 5, a < 6)), a)

  def test_any_wildcard(self):
    p = UPat.any(UPat.var("x"), UPat()).sink(name="s")
    for compiled in (False, True):
      pm = PatternMatcher([(p, lambda s,x=None: s.src[0] if x is None else None)], compiled=compiled)
      self.assertIs(pm.rewrite(UOp.const(1).sink()), UOp.const(1))

  def test_named_any(self):
    p = UPat.any(UPat(Ops.CONST, name="inner"), UPat(Ops.CAST)).named("x").sink()
    a = UOp.const(1)
    for compiled in (False, True):
      pm = PatternMatcher([(p, lambda x,inner=None: inner if x.op is Ops.CONST else x)], compiled=compiled)
      self.assertIs(pm.rewrite(a.sink()), a)
      self.assertIs(pm.rewrite((b:=a.cast(dtypes.int)).sink()), b)
      self.assertIsNone(pm.rewrite((a+a).sink()))

  def test_empty_dtype_set(self):
    for compiled in (False, True):
      pm = PatternMatcher([(UPat(Ops.SINK, dtype=set()), lambda: True)], compiled=compiled)
      self.assertIsNone(pm.rewrite(UOp.sink()))

  def test_global_defined_after_matcher(self):
    for pm in late_pms: self.assertEqual(pm.rewrite(UOp.const(1)), UOp.const(1).sink())

  def test_allow_len(self):
    matcher = PatternMatcher([(UPat(Ops.MULACC, name="x", src=(UPat(Ops.CONST),), allow_any_len=True), lambda x: x.sink())])
    c1 = UOp.const(1.0)
    c2 = UOp.const(2.0)
    c3 = UOp.const(3.0)
    c4 = UOp(Ops.EXP2, src=(c1,))
    c5 = UOp(Ops.ADD, src=(c1,c2))
    c6 = UOp(Ops.MULACC, src=(c1,c2,c3))
    self.assertEqual(matcher.rewrite(c4), None)
    self.assertEqual(matcher.rewrite(c5), None)
    self.assertEqual(matcher.rewrite(c6), c6.sink())

  def test_deep_src_permutations(self):
    c1 = UOp.const(1.0)
    c2 = UOp.const(2.0)
    u1 = (c1 + c2) + c1
    u2 = (c2 + c1) + c1
    matcher = PatternMatcher([
      (UPat(GroupOp.ALU, src=[UPat(GroupOp.ALU, src=[UPat(name='a'), UPat(name='b')]), UPat(name='b')]), lambda a,b: b)
    ])
    self.assertIsNotNone(matcher.rewrite(u1))
    self.assertIsNotNone(matcher.rewrite(u2))

  def _assert_eq_upat(self, a:UPat, b:UPat):
    assert (sorted(map(str,a.op)) if a.op else [] == (sorted(map(str,b.op)) if b.op else []))
    assert (sorted(a.match_dtype) if a.match_dtype else [] == (sorted(b.match_dtype) if b.match_dtype else []))
    assert (a.name, type(a.src)) == (b.name, type(b.src))
    def simple_src(u:UPat):
      if u.src is None: return []
      if isinstance(u.src, itertools.repeat): return next(u.src[0])
      return u.src[0]
    for a,b in zip(simple_src(a), simple_src(b)): self._assert_eq_upat(a, b)

@rewrite_group()
def do_compile(up):
  print("\n***** COMPILE", up)
  match_code = _get_code(up, False)
  match = upat_compile(up, lambda **kwargs: None)
  print(match_code[0])
  if DEBUG >= 2: dis.dis(match)
  return match_code[0]

class TestUPatCompile(unittest.TestCase):
  def test_double(self):
    up = UPat.var("x") * UPat.cvar("c0") + UPat.var("x") * UPat.cvar("c1")
    do_compile(up)

  def test_single(self):
    up = UPat.var("x") + UPat.var("y")
    do_compile(up)

  def test_xpx(self):
    up = UPat.var("x") + UPat.var("x")
    do_compile(up)

  def test_xp0(self):
    up = UPat.var("x") + 0
    do_compile(up)

  def test_bool(self):
    up = UPat.var('x', dtype=dtypes.bool) * UPat.var('y', dtype=dtypes.bool)
    do_compile(up)

  def test_single_c(self):
    up = (UPat.var("x") + UPat.var("y")) * UPat.var("c")
    do_compile(up)

  def test_const_folding(self):
    up = UPat(GroupOp.ALU-{Ops.THREEFRY}, name="a", src=UPat((Ops.CONST, Ops.STACK)))
    do_compile(up)

  def test_range_named(self):
    # this should be one src, but this should also still work
    up = UPat(Ops.CAST, dtypes.float, UPat.var("x", dtypes.bfloat16))
    do_compile(up)

if __name__ == '__main__':
  unittest.main(verbosity=2)
