import unittest, math
from tinygrad import dtypes
from tinygrad.helpers import all_same, Context
from tinygrad.uop.ops import GroupOp, UOp, Ops, exec_alu, PatternMatcher, TrackedPatternMatcher, UPat
from test.helpers import full_rewrite
from hypothesis import given, strategies as strat

# Helper function to apply the graph rewrite
@Context(SPEC=0)
def apply_rewrite(expr):
  return full_rewrite(expr.sink()).src[0]

@Context(SPEC=0)
def apply_rewrite_values(expr):
  srcs = full_rewrite(expr.sink()).src
  if len(srcs) == 1:
    if srcs[0].op is Ops.CONST: return (srcs[0].arg,)*srcs[0].dtype.count
    if srcs[0].op is Ops.STACK: return tuple(s.arg for s in srcs[0].src)
  return tuple(s.arg for s in srcs)

def evaluate_uop(uop, variables):
  if uop.op == Ops.CONST:
    return uop.arg
  elif uop.op == Ops.DEFINE_VAR:
    var_name = uop.arg[0]
    return variables[var_name]
  elif uop.op in GroupOp.ALU:
    src_values = [evaluate_uop(src, variables) for src in uop.src]
    return exec_alu(uop.op, uop.dtype, src_values)
  else:
    raise NotImplementedError(f"Unsupported UOp {uop.op}")

class TestArithmeticSimplifications(unittest.TestCase):
  def test_full_graph_rewrite_division_by_zero(self):
    optimized_div_uop = apply_rewrite(UOp.const(dtypes.float32, 10.0) / UOp.const(dtypes.float32, 0.0))
    self.assertEqual(optimized_div_uop.op, Ops.CONST)
    self.assertTrue(math.isinf(optimized_div_uop.arg) or math.isnan(optimized_div_uop.arg))

  def test_full_graph_rewrite_redundant_operations(self):
    optimized_uop = apply_rewrite((UOp.const(dtypes.float32, 10.0) + UOp.const(dtypes.float32, 0.0)) * UOp.const(dtypes.float32, 1.0))
    self.assertEqual(optimized_uop.op, Ops.CONST)
    self.assertEqual(optimized_uop.arg, 10.0)

  def test_full_graph_rewrite_large_graph(self):
    prev_uop = UOp.const(dtypes.int32, 0)
    for i in range(1, 101):
      prev_uop += UOp.const(dtypes.int32, i)
    optimized_uop = apply_rewrite(prev_uop)
    self.assertEqual(optimized_uop.op, Ops.CONST)
    self.assertEqual(optimized_uop.arg, sum(range(1, 101)))

  def test_full_graph_rewrite_division_by_one(self):
    optimized_uop = apply_rewrite(UOp.const(dtypes.float32, 42.0) / UOp.const(dtypes.float32, 1.0))
    self.assertEqual(optimized_uop.op, Ops.CONST)
    self.assertEqual(optimized_uop.arg, 42.0)

  def test_full_graph_rewrite_modulo_by_one(self):
    optimized_uop = apply_rewrite(UOp.const(dtypes.int32, 42) % UOp.const(dtypes.int32, 1))
    self.assertEqual(optimized_uop.op, Ops.CONST)
    self.assertEqual(optimized_uop.arg, 0)


class TestFoldingAndReduction(unittest.TestCase):
  @unittest.skip("reduce is removed now")
  def test_full_graph_rewrite_constant_reduction_folding(self):
    const1 = UOp.const(dtypes.int32, 5)
    const2 = UOp.const(dtypes.int32, 10)
    const3 = UOp.const(dtypes.int32, 20)
    optimized_sink = apply_rewrite((const1 + const2 + const3).reduce(Ops.ADD))
    expected_sum = 5 + 10 + 20
    self.assertEqual(optimized_sink.arg, expected_sum)

  @unittest.skip("reduce is removed now")
  def test_full_graph_rewrite_reduction_with_unused_range(self):
    const1 = UOp.const(dtypes.int32, 15)
    const2 = UOp.const(dtypes.int32, 25)
    rng = UOp.range(10, idx=0)
    optimized_sink = apply_rewrite((const1 + const2).reduce(Ops.ADD, rng))
    expected_sum = 10 * (15 + 25)
    self.assertEqual(optimized_sink.arg, expected_sum)

  @unittest.skip("currently failing")
  def test_full_graph_rewrite_range_reduction(self):
    simple_range = UOp.range(5, idx=0)
    optimized_sink = apply_rewrite(simple_range.reduce(Ops.ADD, simple_range))
    expected_sum = sum(range(5))
    self.assertEqual(optimized_sink.arg, expected_sum)

  @unittest.skip("currently failing")
  def test_full_graph_rewrite_simple_reduction_folding(self):
    simple_range = UOp.range(4, idx=0)
    add_uop = simple_range + UOp.const(dtypes.int32, 1)
    optimized_sink = apply_rewrite(add_uop.reduce(Ops.ADD, simple_range))
    expected_sum = sum(i + 1 for i in range(4))
    self.assertEqual(optimized_sink.arg, expected_sum)

  @unittest.skip("currently failing")
  def test_full_graph_rewrite_nested_loop_collapse(self):
    outer_range = UOp.range(8, 0)
    inner_range = UOp.range(4, 1)
    expr = (outer_range * 10) + inner_range
    optimized_reduce_uop = apply_rewrite(expr.reduce(Ops.ADD, outer_range, inner_range))
    self.assertEqual(optimized_reduce_uop.op, Ops.CONST)
    self.assertEqual(optimized_reduce_uop.arg, sum((i * 10) + j for i in range(8) for j in range(4)))


class TestModuloAndDivisionFolding(unittest.TestCase):
  def test_full_graph_rewrite_modulo_folding_with_define_var(self):
    # index dtype because div-mod rules only work on index
    x_var_uop = UOp.variable('x', 0, 100).cast(dtypes.weakint)
    optimized_mod_uop = apply_rewrite(((x_var_uop * 4) + 2) % 4)
    self.assertEqual(optimized_mod_uop.op, Ops.CONST)
    self.assertEqual(optimized_mod_uop.arg, 2)

  def test_full_graph_rewrite_division_folding_with_define_var(self):
    # index dtype because div-mod rules only work on index
    n_var_uop = UOp.variable('n', 1, 1000).cast(dtypes.weakint)
    optimized_div_uop = apply_rewrite((n_var_uop * 6) // 3)
    self.assertEqual(optimized_div_uop.op, Ops.MUL)
    self.assertEqual(optimized_div_uop.src[1].arg, 2)

  def test_full_graph_rewrite_complex_mod_div_folding(self):
    # index dtype because div-mod rules only work on index
    k_var_uop = UOp.variable('k', 0, 50).cast(dtypes.weakint)
    optimized_div_uop = apply_rewrite(((k_var_uop * 12 + 8) % 6) // 2)
    self.assertEqual(optimized_div_uop.op, Ops.CONST)
    self.assertEqual(optimized_div_uop.arg, 1)

  def test_graph_rewrite_div_folding_bug(self):
    lhs = UOp(Ops.ADD, dtypes.int.vec(4), src=(
      UOp(Ops.STACK, dtypes.int.vec(4), arg=None, src=(UOp(Ops.SPECIAL, dtypes.int, arg='lidx0', src=(UOp.const(dtypes.int, 32),)),)*4),
      UOp.const(dtypes.int.vec(4), (0, 256, 512, 768))))
    rhs = UOp.const(dtypes.int.vec(4), 2)
    unopt = lhs<rhs
    opt = apply_rewrite(unopt)
    print(unopt)
    print(opt)
    if opt.op is Ops.STACK: self.assertFalse(all_same(opt.src))

  def test_full_graph_rewrite_modulo_large_divisor(self):
    # index dtype because div-mod rules only work on index
    x_var_uop = UOp.variable('x', 1, 5)
    self.assertIs(apply_rewrite(x_var_uop.cast(dtypes.weakint) % 10).render(simplify=False), x_var_uop.render(simplify=False))

  def test_full_graph_rewrite_division_with_remainder(self):
    x_var_uop = UOp.variable('x', 7, 9)
    optimized_sink = apply_rewrite(x_var_uop // 2)
    for x_value in range(7, 10):
      self.assertEqual(x_value // 2, evaluate_uop(optimized_sink, {'x': x_value}))

  def test_full_graph_rewrite_complex_mod_div_expression(self):
    x_var_uop = UOp.variable('x', 1, 10)
    optimized_sink = apply_rewrite(((x_var_uop * 5) % 3) // 2)
    for x_value in range(1, 11):
      original_result = ((x_value * 5) % 3) // 2
      optimized_result = evaluate_uop(optimized_sink, {'x': x_value})
      self.assertEqual(original_result, optimized_result)


class TestEdgeCasesAndSpecialOperations(unittest.TestCase):
  def test_full_graph_rewrite_transcendental_edge_cases(self):
    optimized_sink = full_rewrite(UOp.const(dtypes.float32, -1.0).log2().sink(UOp.const(dtypes.float32, 0.0).reciprocal()))
    optimized_log2_neg, optimized_recip_zero = optimized_sink.src
    self.assertTrue(math.isnan(optimized_log2_neg.arg), f"Expected NaN for log2(-1.0), got {optimized_log2_neg.arg}")
    self.assertTrue(math.isinf(optimized_recip_zero.arg) and optimized_recip_zero.arg > 0,
                    f"Expected +inf for reciprocal(0.0), got {optimized_recip_zero.arg}")

  @unittest.skip("broken")
  def test_full_graph_rewrite_modulo_negative_dividend(self):
    x_var_uop = UOp.variable('x', -5, -1)
    optimized_sink = full_rewrite((x_var_uop % 3).sink())
    for x_value in range(-5, 0):
      self.assertEqual(x_value % 3, evaluate_uop(optimized_sink.src[0], {'x': x_value}))

  @unittest.skip("broken")
  def test_full_graph_rewrite_division_negative_divisor(self):
    x_var_uop = UOp.variable('x', 1, 5)
    optimized_sink = full_rewrite((x_var_uop // -2).sink())
    for x_value in range(1, 6):
      self.assertEqual(x_value // -2, evaluate_uop(optimized_sink.src[0], {'x': x_value}))

class TestGEPAndVectorizeRewrite(unittest.TestCase):
  def test_gep_single_element_extraction(self):
    # GEP on a vector dtype to extract a single element
    base_vector = UOp.const(dtypes.float32.vec(4), (1.0, 2.0, 3.0, 4.0))
    self.assertEqual(apply_rewrite(base_vector.gep(2)).arg, 3.0)

  def test_gep_tuple_extraction(self):
    # GEP on a vector dtype to extract multiple elements as a vector
    base_vector = UOp.const(dtypes.float32.vec(4), (1.0, 2.0, 3.0, 4.0))
    self.assertEqual(list(apply_rewrite_values(base_vector.gep((2, 3)))), [3.0, 4.0])

  def test_gep_on_const_stack(self):
    # GEP on a const STACK to extract a single element
    const_stack = UOp.const(dtypes.float32.vec(4), (1.0, 2.0, 3.0, 4.0))
    self.assertEqual(apply_rewrite(const_stack.gep(2)).arg, 3.0)

  def test_gep_tuple_on_const_stack(self):
    # GEP on a const STACK using a tuple to extract multiple elements
    const_stack = UOp.const(dtypes.float32.vec(4), (7.0, 8.0, 9.0, 10.0))
    self.assertEqual(list(apply_rewrite_values(const_stack.gep((1, 3)))), [8.0, 10.0])

  def test_gep_gep_simplification(self):
    # Nested GEP simplification on a vector dtype
    base_vector = UOp.const(dtypes.float32.vec(4), (10.0, 20.0, 30.0, 40.0))
    gep_inner = base_vector.gep(1)  # Extract 2nd element (20.0)
    self.assertEqual(apply_rewrite(gep_inner.gep(0)).arg, 20.0)

  def test_vectorize_multiple_elements(self):
    # Vectorizing multiple elements using GEP
    base_vector = UOp.const(dtypes.float32.vec(4), (5.0, 10.0, 15.0, 20.0))
    vectorized_uop = UOp(Ops.STACK, dtypes.float32.vec(4), src=(base_vector.gep(0), base_vector.gep(1), base_vector.gep(2), base_vector.gep(3)))
    self.assertEqual(list(apply_rewrite_values(vectorized_uop)), [5.0, 10.0, 15.0, 20.0])


import inspect
from tinygrad.uop.ops import graph_rewrite, _substitute, track_rewrites
from tinygrad.uop.symbolic import symbolic_simple

class TestBottomUpRewrite(unittest.TestCase):
  def test_const_folding(self):
    a = UOp.const(dtypes.int, 5)
    ret = (a*3) + (a*7)
    gt = graph_rewrite(ret, symbolic_simple)
    ret = graph_rewrite(ret, symbolic_simple, bottom_up=True)
    self.assertIs(gt, ret)

# normally .substitute would be fine, but it's not tracked
@track_rewrites()
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

  # broken due to infinite recursion
  # NOTE: VIZ hangs and doesn't recover if you click this one
  @unittest.skip("recursion error no longer raised")
  def test_assert_inf_recurse(self):
    a = UOp.variable('a', 0, 10)
    n1 = a.sin()
    ret = n1
    with self.assertRaises(RecursionError):
      ret = substitute(ret, {n1:n1.sqrt()})

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
    pm = PatternMatcher([(UPat(Ops.DEFINE_VAR, name="x"), lambda x: x)])
    graph_rewrite(a, pm)

  @given(matchers)
  def test_no_inf_loop_bottom_up(self, PatternMatcher):
    a = UOp.variable('a', 0, 10)
    pm = PatternMatcher([(UPat(Ops.DEFINE_VAR, name="x"), lambda x: x)])
    graph_rewrite(a, pm, bottom_up=True)

  def test_inf_loop(self):
    a = UOp.const(dtypes.int, 3)
    pm = PatternMatcher([
      (UPat(Ops.CONST, arg=3, name="x"), lambda x: x.replace(arg=4)),
      (UPat(Ops.CONST, arg=4, name="x"), lambda x: x.replace(arg=3)),
    ])
    with self.assertRaises(RuntimeError):
      graph_rewrite(a, pm)

  def test_inf_loop_bottom_up(self):
    a = UOp.const(dtypes.int, 3)
    pm = PatternMatcher([
      (UPat(Ops.CONST, arg=3, name="x"), lambda x: x.replace(arg=4)),
      (UPat(Ops.CONST, arg=4, name="x"), lambda x: x.replace(arg=3)),
    ])
    with self.assertRaises(RuntimeError):
      graph_rewrite(a, pm, bottom_up=True)

def bidir_append(ctx, x, b): ctx.append((x.arg if x.op is Ops.CONST else "+", b))
class TestBidirectional(unittest.TestCase):
  def test_simple(self):
    a = UOp.const(dtypes.int, 1)
    b = UOp.const(dtypes.int, 2)
    c = a + b
    pm = PatternMatcher([ (UPat(GroupOp.All, name="x"), lambda ctx,x: bidir_append(ctx, x, False)) ])
    bpm = PatternMatcher([ (UPat(GroupOp.All, name="x"), lambda ctx,x: bidir_append(ctx, x, True)) ])
    ctx_list = []
    graph_rewrite(c, pm, ctx=ctx_list, bpm=bpm)
    self.assertListEqual(ctx_list, [('+', True), (1, True), (1, False), (2, True), (2, False), ('+', False)])

class TestStopEarly(unittest.TestCase):
  def test_stop_early(self):
    a = UOp.const(dtypes.int, 3)
    b = UOp.const(dtypes.int, 4)
    c = a+b
    cn = UOp.const(dtypes.int, 7)
    d = UOp.const(dtypes.int, 2)
    def visit_const(c:UOp):
      print(f"visit {c.arg}")
      assert c.arg not in (3,4)
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
    a = UOp.const(dtypes.int, 3)
    pm = PatternMatcher([
      (UPat(Ops.CONST, arg=3, name="x"), lambda x: x.replace(arg=4)),
      (UPat(Ops.CONST, arg=4, name="x"), lambda x: x.replace(arg=3)),
    ])
    with self.assertRaises(RuntimeError):
      graph_rewrite(a, pm, bottom_up=True)
    ret = graph_rewrite(a, pm, walk=True)
    self.assertIs(ret, UOp.const(dtypes.int, 4))

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
      ctx.append(x.arg if x.op is Ops.CONST else x.op)
      return None
    pm = PatternMatcher([(UPat(GroupOp.All, name="x"), track_visit)])
    a = UOp.const(dtypes.int, 1)
    b = UOp.const(dtypes.int, 2)
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
    a = UOp.const(dtypes.int, 3)
    pm = PatternMatcher([
      (UPat(Ops.CONST, arg=3, name="x"), lambda x: x.replace(arg=4)),
      (UPat(Ops.CONST, arg=4, name="x"), lambda x: x.replace(arg=3)),
    ])
    ret = graph_rewrite(a, pm, bottom_up=True, walk=True)
    self.assertIs(ret, UOp.const(dtypes.int, 4))

  def test_walk_bottomup_visit_order(self):
    """Bottom-up walk fires bpm before descending (pre-order)."""
    visited = []
    def track_visit(ctx, x):
      ctx.append(x.arg if x.op is Ops.CONST else x.op)
      return None
    pm = PatternMatcher([(UPat(GroupOp.All, name="x"), track_visit)])
    a = UOp.const(dtypes.int, 1)
    b = UOp.const(dtypes.int, 2)
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
      ctx.append((x.arg if x.op is Ops.CONST else x.op, "bpm"))
      return None
    def pm_visit(ctx, x):
      ctx.append((x.arg if x.op is Ops.CONST else x.op, "pm"))
      return None
    bpm = PatternMatcher([(UPat(GroupOp.All, name="x"), bpm_visit)])
    pm = PatternMatcher([(UPat(GroupOp.All, name="x"), pm_visit)])
    a = UOp.const(dtypes.int, 1)
    b = UOp.const(dtypes.int, 2)
    graph_rewrite(a + b, pm, ctx=visited, bpm=bpm, walk=True)
    # bpm fires pre-order, pm fires post-order
    self.assertEqual(visited, [
      (Ops.ADD, "bpm"), (1, "bpm"), (1, "pm"), (2, "bpm"), (2, "pm"), (Ops.ADD, "pm"),
    ])

  def test_walk_bidirectional_bpm_short_circuits(self):
    """If bpm matches, children are skipped and pm never fires on that node."""
    visited = []
    def bpm_match(ctx, x):
      ctx.append((x.arg if x.op is Ops.CONST else x.op, "bpm"))
      # rewrite const(1) -> const(10), short-circuiting its subtree
      if x.op is Ops.CONST and x.arg == 1: return x.replace(arg=10)
      return None
    def pm_match(ctx, x):
      ctx.append((x.arg if x.op is Ops.CONST else x.op, "pm"))
      return None
    bpm = PatternMatcher([(UPat(GroupOp.All, name="x"), bpm_match)])
    pm = PatternMatcher([(UPat(GroupOp.All, name="x"), pm_match)])
    a = UOp.const(dtypes.int, 1)
    b = UOp.const(dtypes.int, 2)
    ret = graph_rewrite(a + b, pm, ctx=visited, bpm=bpm, walk=True)
    # bpm matches const(1) and short-circuits it, so pm never fires on const(1)
    self.assertNotIn((1, "pm"), visited)
    # but pm still fires on const(2) and the rebuilt ADD
    self.assertIn((2, "pm"), visited)
    self.assertIs(ret, UOp.const(dtypes.int, 10) + b)

if __name__ == '__main__':
  unittest.main()
