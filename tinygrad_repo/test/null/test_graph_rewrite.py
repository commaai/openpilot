import unittest, math
from tinygrad import dtypes
from tinygrad.helpers import all_same, Context
from tinygrad.uop.ops import GroupOp, UOp, Ops, PatternMatcher, TrackedPatternMatcher, UPat
from test.helpers import full_rewrite
from hypothesis import given, strategies as strat

# Helper function to apply the graph rewrite
@Context(SPEC=0)
def apply_rewrite(expr):
  return full_rewrite(expr.sink()).src[0]

def const_value(uop:UOp):
  if uop.op is Ops.CAST: uop = uop.src[0]
  assert uop.op is Ops.CONST
  return uop.val

class TestModuloAndDivisionFolding(unittest.TestCase):
  def test_graph_rewrite_div_folding_bug(self):
    lhs = UOp.stack(*(UOp.special(32, 'lidx0'),)*4) + UOp.const((0, 256, 512, 768))
    rhs = UOp.const((2,)*4)
    unopt = lhs<rhs
    opt = apply_rewrite(unopt)
    print(unopt)
    print(opt)
    if opt.op is Ops.STACK: self.assertFalse(all_same(opt.src))

class TestEdgeCasesAndSpecialOperations(unittest.TestCase):
  def test_full_graph_rewrite_transcendental_edge_cases(self):
    optimized_sink = full_rewrite(UOp.const(-1.0).log2().sink(UOp.const(0.0).reciprocal()))
    optimized_log2_neg, optimized_recip_zero = optimized_sink.src
    log2_neg, recip_zero = const_value(optimized_log2_neg), const_value(optimized_recip_zero)
    self.assertTrue(math.isnan(log2_neg), f"Expected NaN for log2(-1.0), got {log2_neg}")
    self.assertTrue(math.isinf(recip_zero) and recip_zero > 0, f"Expected +inf for reciprocal(0.0), got {recip_zero}")

class TestGEPAndVectorizeRewrite(unittest.TestCase):
  def test_gep_single_element_extraction(self):
    # GEP on a vector dtype to extract a single element
    base_vector = UOp.const((1.0, 2.0, 3.0, 4.0))
    self.assertIs(apply_rewrite(base_vector.index(2)), apply_rewrite(base_vector.src[2]))

  def test_gep_tuple_extraction(self):
    # GEP on a vector dtype to extract multiple elements as a vector
    base_vector = UOp.const((1.0, 2.0, 3.0, 4.0))
    self.assertIs(apply_rewrite(UOp.stack(*[base_vector.index(i) for i in (2, 3)])),
                  apply_rewrite(UOp.stack(base_vector.src[2], base_vector.src[3])))

  def test_vectorize_multiple_elements(self):
    # Vectorizing multiple elements using GEP
    base_vector = UOp.const((5.0, 10.0, 15.0, 20.0))
    vectorized_uop = UOp.stack(*(base_vector.index(i) for i in range(4)))
    self.assertIs(apply_rewrite(vectorized_uop), apply_rewrite(base_vector))


import inspect
from tinygrad.uop.ops import graph_rewrite, _substitute, rewrite_group
from tinygrad.uop.symbolic import symbolic_simple

class TestBottomUpRewrite(unittest.TestCase):
  def test_const_folding(self):
    a = UOp.const(5)
    ret = (a*3) + (a*7)
    gt = graph_rewrite(ret, symbolic_simple)
    ret = graph_rewrite(ret, symbolic_simple, bottom_up=True)
    self.assertIs(gt, ret)

# normally .substitute would be fine, but it's not tracked
@rewrite_group()
def named_substitute(name:str, uop:UOp, rel:dict[UOp, UOp]): return graph_rewrite(uop, _substitute, rel, bottom_up=True)
def substitute(uop:UOp, rel:dict[UOp, UOp]): return named_substitute(inspect.stack()[1].function, uop, rel)

class TestSubstitute(unittest.TestCase):
  # these work because the substituted things don't have parents
  def test_simple(self):
    a = UOp.variable('a', 0, 10)
    b = UOp.variable('b', 0, 10)
    ret = a + 4
    ret = substitute(ret, {a:b})
    self.assertIs(ret, b+4)

  def test_double(self):
    a = UOp.variable('a', 0, 10)
    b = UOp.variable('b', 0, 10)
    c = UOp.variable('c', 0, 10)
    ret = (a + 4) + b
    ret = substitute(ret, {a:c, b:c})
    self.assertIs(ret, (c + 4) + c)

  def test_diamond(self):
    a = UOp.variable('a', 0, 10)
    b = UOp.variable('b', 0, 10)
    ret = (a + 4) + (a + 5)
    ret = substitute(ret, {a:b})
    self.assertIs(ret, (b + 4) + (b + 5))

  # this works because there's nothing above the substituted node
  def test_sin(self):
    a = UOp.variable('a', 0, 10, dtype=dtypes.float)
    b = UOp.variable('b', 0, 10, dtype=dtypes.float)
    ret = a.sin().sin()
    ret = substitute(ret, {a.sin():b})
    self.assertIs(ret, b.sin())

  def test_sin_to_sqrt(self):
    a = UOp.variable('a', 0, 10, dtype=dtypes.float)
    n1 = a.sin()
    ret = n1.sin()
    ret = substitute(ret, {a.sin():a.sqrt()})
    self.assertIs(ret, a.sqrt().sin())

  def test_double_sin_to_sqrt(self):
    a = UOp.variable('a', 0, 10, dtype=dtypes.float)
    n1 = a.sin()
    ret = n1.sin()
    # NOTE: this would work if it had gone in the opposite order
    ret = substitute(ret, {a.sin():a.sqrt(), n1.sin():n1.sqrt()})
    self.assertIs(ret, a.sqrt().sqrt())

  def test_tagged_replace(self):
    a = UOp.variable('a', 0, 10)
    b = UOp.variable('b', 0, 10)
    ret = (a+4).replace(tag=1)
    ret = substitute(ret, {a:b})
    # the srcs are rewritten but we keep tag
    self.assertIs(ret, (b+4).replace(tag=1))

matchers = strat.sampled_from([PatternMatcher, TrackedPatternMatcher])

class TestRecurse(unittest.TestCase):
  @given(matchers)
  def test_no_inf_loop(self, PatternMatcher):
    a = UOp.variable('a', 0, 10)
    pm = PatternMatcher([(UPat(Ops.PARAM, name="x"), lambda x: x)])
    graph_rewrite(a, pm)

  @given(matchers)
  def test_no_inf_loop_bottom_up(self, PatternMatcher):
    a = UOp.variable('a', 0, 10)
    pm = PatternMatcher([(UPat(Ops.PARAM, name="x"), lambda x: x)])
    graph_rewrite(a, pm, bottom_up=True)

  def test_inf_loop(self):
    a = UOp.const(3)
    pm = PatternMatcher([
      (UPat(Ops.CONST, arg=3, name="x"), lambda x: UOp.const(4, x.dtype)),
      (UPat(Ops.CONST, arg=4, name="x"), lambda x: UOp.const(3, x.dtype)),
    ])
    with self.assertRaises(RuntimeError):
      graph_rewrite(a, pm)

  def test_inf_loop_bottom_up(self):
    a = UOp.const(3)
    pm = PatternMatcher([
      (UPat(Ops.CONST, arg=3, name="x"), lambda x: UOp.const(4, x.dtype)),
      (UPat(Ops.CONST, arg=4, name="x"), lambda x: UOp.const(3, x.dtype)),
    ])
    with self.assertRaises(RuntimeError):
      graph_rewrite(a, pm, bottom_up=True)

def bidir_append(ctx, x, b): ctx.append((x.val if x.op is Ops.CONST else "+", b))
class TestBidirectional(unittest.TestCase):
  def test_simple(self):
    a = UOp.const(1)
    b = UOp.const(2)
    c = a + b
    pm = PatternMatcher([ (UPat(GroupOp.All, name="x"), lambda ctx,x: bidir_append(ctx, x, False)) ])
    bpm = PatternMatcher([ (UPat(GroupOp.All, name="x"), lambda ctx,x: bidir_append(ctx, x, True)) ])
    ctx_list = []
    graph_rewrite(c, pm, ctx=ctx_list, bpm=bpm)
    self.assertListEqual(ctx_list, [('+', True), (1, True), (1, False), (2, True), (2, False), ('+', False)])

class TestStopEarly(unittest.TestCase):
  def test_stop_early(self):
    a = UOp.const(3)
    b = UOp.const(4)
    c = a+b
    cn = UOp.const(7)
    d = UOp.const(2)
    def visit_const(c:UOp):
      print(f"visit {c.val}")
      assert c.val not in (3,4)
    pm_cvisit = PatternMatcher([(UPat(Ops.CONST, name="c"), visit_const),])
    ret = (c+d).substitute({c:cn}, extra_pm=pm_cvisit)
    assert ret == cn+d

class TestWalkRewrite(unittest.TestCase):
  """Tests for graph_rewrite with walk=True (MLIR Walk Pattern Rewrite Driver semantics).
  walk=True gives a single-pass traversal that does NOT revisit or re-traverse into rewritten subtrees.
  Supports both top-down (default) and bottom-up (bottom_up=True) modes."""

  # *** top-down walk (default): process children first, then try pm on rebuilt node ***

  def test_walk_topdown_simple_substitute(self):
    a = UOp.variable('a', 0, 10)
    b = UOp.variable('b', 0, 10)
    ret = graph_rewrite(a + 4, _substitute, {a:b}, walk=True)
    self.assertIs(ret, b+4)

  def test_walk_topdown_does_not_traverse_into_replacement(self):
    """Top-down walk: replacement subtrees are NOT re-entered."""
    a = UOp.variable('a', 0, 10)
    b = UOp.variable('b', 0, 10)
    c = UOp.variable('c', 0, 10)
    d = UOp.variable('d', 0, 10)
    # a is replaced by b+c, but b inside the replacement is NOT further substituted to d
    ret_walk = graph_rewrite(a + 4, _substitute, {a:b+c, b:d}, walk=True)
    self.assertIs(ret_walk, (b+c)+4)
    # contrast: greedy bottom_up WOULD replace b inside the replacement
    ret_greedy = graph_rewrite(a + 4, _substitute, {a:b+c, b:d}, bottom_up=True)
    self.assertIs(ret_greedy, (d+c)+4)

  def test_walk_topdown_no_fixed_point(self):
    """A bouncing pattern applies once and stops instead of looping."""
    a = UOp.const(3)
    pm = PatternMatcher([
      (UPat(Ops.CONST, arg=3, name="x"), lambda x: UOp.const(4, x.dtype)),
      (UPat(Ops.CONST, arg=4, name="x"), lambda x: UOp.const(3, x.dtype)),
    ])
    with self.assertRaises(RuntimeError):
      graph_rewrite(a, pm, bottom_up=True)
    ret = graph_rewrite(a, pm, walk=True)
    self.assertIs(ret, UOp.const(4))

  def test_walk_topdown_rewrites_children(self):
    a = UOp.variable('a', 0, 10)
    b = UOp.variable('b', 0, 10)
    c = UOp.variable('c', 0, 10)
    ret = graph_rewrite((a + 4) + (b + 5), _substitute, {a:c, b:c}, walk=True)
    self.assertIs(ret, (c + 4) + (c + 5))

  def test_walk_topdown_diamond(self):
    a = UOp.variable('a', 0, 10)
    b = UOp.variable('b', 0, 10)
    ret = graph_rewrite((a + 4) + (a + 5), _substitute, {a:b}, walk=True)
    self.assertIs(ret, (b + 4) + (b + 5))

  def test_walk_topdown_children_rewritten_before_parent(self):
    """Top-down walk processes children first: child substitution changes the rebuilt parent."""
    a = UOp.variable('a', 0, 10, dtype=dtypes.float)
    n1 = a.sin()          # sin(a)
    ret = n1.sin()         # sin(sin(a))
    # sin(a)->sqrt(a) fires first (child), parent rebuilds to sin(sqrt(a)), which doesn't match sin(sin(a)) in dvars
    ret_walk = graph_rewrite(ret, _substitute, {a.sin():a.sqrt(), n1.sin():n1.sqrt()}, walk=True)
    self.assertIs(ret_walk, a.sqrt().sin())

  def test_walk_topdown_self_referential_replacement(self):
    """Replacement containing the replaced node works without infinite recursion."""
    a = UOp.variable('a', 0, 10, dtype=dtypes.float)
    ret = graph_rewrite(a.sin() + 4, _substitute, {a.sin(): a.sin().sqrt()}, walk=True)
    self.assertIs(ret, a.sin().sqrt() + 4)

  def test_walk_topdown_visit_order(self):
    """Top-down walk fires pm after children are processed (post-order)."""
    visited = []
    def track_visit(ctx, x):
      ctx.append(x.val if x.op is Ops.CONST else x.op)
      return None
    pm = PatternMatcher([(UPat(GroupOp.All, name="x"), track_visit)])
    a = UOp.const(1)
    b = UOp.const(2)
    graph_rewrite(a + b, pm, ctx=visited, walk=True)
    self.assertEqual(visited, [1, 2, Ops.ADD])

  # *** bottom-up walk: try bpm on node first, skip children if it matches ***

  def test_walk_bottomup_simple_substitute(self):
    a = UOp.variable('a', 0, 10)
    b = UOp.variable('b', 0, 10)
    ret = graph_rewrite(a + 4, _substitute, {a:b}, bottom_up=True, walk=True)
    self.assertIs(ret, b+4)

  def test_walk_bottomup_does_not_traverse_into_replacement(self):
    """Bottom-up walk: replacement subtrees are NOT entered."""
    a = UOp.variable('a', 0, 10)
    b = UOp.variable('b', 0, 10)
    c = UOp.variable('c', 0, 10)
    d = UOp.variable('d', 0, 10)
    ret = graph_rewrite(a + 4, _substitute, {a:b+c, b:d}, bottom_up=True, walk=True)
    self.assertIs(ret, (b+c)+4)

  def test_walk_bottomup_parent_match_skips_children(self):
    """Bottom-up walk matches parent first: if it matches, children are never visited."""
    a = UOp.variable('a', 0, 10, dtype=dtypes.float)
    n1 = a.sin()
    ret = n1.sin()         # sin(sin(a))
    # sin(sin(a)) matches n1.sin()->n1.sqrt() immediately, children never visited, sin(a) inside replacement untouched
    ret_walk = graph_rewrite(ret, _substitute, {a.sin():a.sqrt(), n1.sin():n1.sqrt()}, bottom_up=True, walk=True)
    self.assertIs(ret_walk, a.sin().sqrt())

  def test_walk_bottomup_no_fixed_point(self):
    """Bottom-up walk also applies once per node, no fixed-point iteration."""
    a = UOp.const(3)
    pm = PatternMatcher([
      (UPat(Ops.CONST, arg=3, name="x"), lambda x: UOp.const(4, x.dtype)),
      (UPat(Ops.CONST, arg=4, name="x"), lambda x: UOp.const(3, x.dtype)),
    ])
    ret = graph_rewrite(a, pm, bottom_up=True, walk=True)
    self.assertIs(ret, UOp.const(4))

  def test_walk_bottomup_visit_order(self):
    """Bottom-up walk fires bpm before descending (pre-order)."""
    visited = []
    def track_visit(ctx, x):
      ctx.append(x.val if x.op is Ops.CONST else x.op)
      return None
    pm = PatternMatcher([(UPat(GroupOp.All, name="x"), track_visit)])
    a = UOp.const(1)
    b = UOp.const(2)
    graph_rewrite(a + b, pm, ctx=visited, bottom_up=True, walk=True)
    # bpm fires on each node before children: +, 1, 2
    self.assertEqual(visited, [Ops.ADD, 1, 2])

  def test_walk_bottomup_unmatched_falls_through_to_children(self):
    """Bottom-up walk: if bpm doesn't match a node, its children are still processed."""
    a = UOp.variable('a', 0, 10)
    b = UOp.variable('b', 0, 10)
    c = UOp.variable('c', 0, 10)
    # only a is in dvars, not a+4. bpm won't match a+4, so it descends and finds a.
    ret = graph_rewrite((a + 4) + (b + 5), _substitute, {a:c, b:c}, bottom_up=True, walk=True)
    self.assertIs(ret, (c + 4) + (c + 5))

  # *** bidirectional walk: bpm fires before children, pm fires after rebuild ***

  def test_walk_bidirectional_visit_order(self):
    """Bidirectional walk: bpm fires pre-order, pm fires post-order."""
    visited = []
    def bpm_visit(ctx, x):
      ctx.append((x.val if x.op is Ops.CONST else x.op, "bpm"))
      return None
    def pm_visit(ctx, x):
      ctx.append((x.val if x.op is Ops.CONST else x.op, "pm"))
      return None
    bpm = PatternMatcher([(UPat(GroupOp.All, name="x"), bpm_visit)])
    pm = PatternMatcher([(UPat(GroupOp.All, name="x"), pm_visit)])
    a = UOp.const(1)
    b = UOp.const(2)
    graph_rewrite(a + b, pm, ctx=visited, bpm=bpm, walk=True)
    # bpm fires pre-order, pm fires post-order
    self.assertEqual(visited, [
      (Ops.ADD, "bpm"), (1, "bpm"), (1, "pm"), (2, "bpm"), (2, "pm"), (Ops.ADD, "pm"),
    ])

  def test_walk_bidirectional_bpm_short_circuits(self):
    """If bpm matches, children are skipped and pm never fires on that node."""
    visited = []
    def bpm_match(ctx, x):
      ctx.append((x.val if x.op is Ops.CONST else x.op, "bpm"))
      # rewrite const(1) -> const(10), short-circuiting its subtree
      if x.op is Ops.CONST and x.val == 1: return UOp.const(10, x.dtype)
      return None
    def pm_match(ctx, x):
      ctx.append((x.val if x.op is Ops.CONST else x.op, "pm"))
      return None
    bpm = PatternMatcher([(UPat(GroupOp.All, name="x"), bpm_match)])
    pm = PatternMatcher([(UPat(GroupOp.All, name="x"), pm_match)])
    a = UOp.const(1)
    b = UOp.const(2)
    ret = graph_rewrite(a + b, pm, ctx=visited, bpm=bpm, walk=True)
    # bpm matches const(1) and short-circuits it, so pm never fires on const(1)
    self.assertNotIn((1, "pm"), visited)
    # but pm still fires on const(2) and the rebuilt ADD
    self.assertIn((2, "pm"), visited)
    self.assertIs(ret, UOp.const(10) + b)

if __name__ == '__main__':
  unittest.main()
