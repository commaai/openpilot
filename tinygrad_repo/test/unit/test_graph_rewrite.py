import unittest, math
from tinygrad import dtypes
from tinygrad.helpers import all_same
from tinygrad.uop.ops import GroupOp, UOp, Ops, exec_alu, PatternMatcher, TrackedPatternMatcher, UPat
from tinygrad.codegen import full_rewrite_to_sink
from hypothesis import given, strategies as strat

# Helper function to apply the graph rewrite
def apply_rewrite(expr):
  return full_rewrite_to_sink(expr.sink()).src[0]

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
    rng = UOp.range(dtypes.int32, 10, idx=0)
    optimized_sink = apply_rewrite((const1 + const2).reduce(Ops.ADD, rng))
    expected_sum = 10 * (15 + 25)
    self.assertEqual(optimized_sink.arg, expected_sum)

  @unittest.skip("currently failing")
  def test_full_graph_rewrite_range_reduction(self):
    simple_range = UOp.range(dtypes.int32, 5, idx=0)
    optimized_sink = apply_rewrite(simple_range.reduce(Ops.ADD, simple_range))
    expected_sum = sum(range(5))
    self.assertEqual(optimized_sink.arg, expected_sum)

  @unittest.skip("currently failing")
  def test_full_graph_rewrite_simple_reduction_folding(self):
    simple_range = UOp.range(dtypes.int32, 4, idx=0)
    add_uop = simple_range + UOp.const(dtypes.int32, 1)
    optimized_sink = apply_rewrite(add_uop.reduce(Ops.ADD, simple_range))
    expected_sum = sum(i + 1 for i in range(4))
    self.assertEqual(optimized_sink.arg, expected_sum)

  @unittest.skip("currently failing")
  def test_full_graph_rewrite_nested_loop_collapse(self):
    outer_range = UOp.range(dtypes.int32, 8, 0)
    inner_range = UOp.range(dtypes.int32, 4, 1)
    expr = (outer_range * 10) + inner_range
    optimized_reduce_uop = apply_rewrite(expr.reduce(Ops.ADD, outer_range, inner_range))
    self.assertEqual(optimized_reduce_uop.op, Ops.CONST)
    self.assertEqual(optimized_reduce_uop.arg, sum((i * 10) + j for i in range(8) for j in range(4)))


class TestModuloAndDivisionFolding(unittest.TestCase):
  def test_full_graph_rewrite_modulo_folding_with_define_var(self):
    x_var_uop = UOp.variable('x', 0, 100)
    optimized_mod_uop = apply_rewrite(((x_var_uop * 4) + 2) % 4)
    self.assertEqual(optimized_mod_uop.op, Ops.CONST)
    self.assertEqual(optimized_mod_uop.arg, 2)

  def test_full_graph_rewrite_division_folding_with_define_var(self):
    n_var_uop = UOp.variable('n', 1, 1000)
    optimized_div_uop = apply_rewrite((n_var_uop * 6) // 3)
    self.assertEqual(optimized_div_uop.op, Ops.MUL)
    self.assertEqual(optimized_div_uop.src[1].arg, 2)

  def test_full_graph_rewrite_complex_mod_div_folding(self):
    k_var_uop = UOp.variable('k', 0, 50)
    optimized_div_uop = apply_rewrite(((k_var_uop * 12 + 8) % 6) // 2)
    self.assertEqual(optimized_div_uop.op, Ops.CONST)
    self.assertEqual(optimized_div_uop.arg, 1)

  def test_graph_rewrite_div_folding_bug(self):
    lhs = UOp(Ops.ADD, dtypes.int.vec(4), src=(
      UOp(Ops.VECTORIZE, dtypes.int.vec(4), arg=None, src=(UOp(Ops.SPECIAL, dtypes.int, arg=('lidx0', 32), src=()),)*4),
      UOp(Ops.VCONST, dtypes.int.vec(4), arg=(0, 256, 512, 768), src=())))
    rhs = UOp.const(dtypes.int.vec(4), 2)
    unopt = lhs<rhs
    opt = apply_rewrite(unopt)
    print(unopt)
    print(opt)
    if opt.op is Ops.VECTORIZE: self.assertFalse(all_same(opt.src))

  def test_full_graph_rewrite_modulo_large_divisor(self):
    x_var_uop = UOp.variable('x', 1, 5)
    self.assertIs(apply_rewrite(x_var_uop % 10), x_var_uop)

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
    optimized_sink = full_rewrite_to_sink(UOp.const(dtypes.float32, -1.0).log2().sink(UOp.const(dtypes.float32, 0.0).reciprocal()))
    optimized_log2_neg, optimized_recip_zero = optimized_sink.src
    self.assertTrue(math.isnan(optimized_log2_neg.arg), f"Expected NaN for log2(-1.0), got {optimized_log2_neg.arg}")
    self.assertTrue(math.isinf(optimized_recip_zero.arg) and optimized_recip_zero.arg > 0,
                    f"Expected +inf for reciprocal(0.0), got {optimized_recip_zero.arg}")

  @unittest.skip("broken")
  def test_full_graph_rewrite_modulo_negative_dividend(self):
    x_var_uop = UOp.variable('x', -5, -1)
    optimized_sink = full_rewrite_to_sink((x_var_uop % 3).sink())
    for x_value in range(-5, 0):
      self.assertEqual(x_value % 3, evaluate_uop(optimized_sink.src[0], {'x': x_value}))

  @unittest.skip("broken")
  def test_full_graph_rewrite_division_negative_divisor(self):
    x_var_uop = UOp.variable('x', 1, 5)
    optimized_sink = full_rewrite_to_sink((x_var_uop // -2).sink())
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
    optimized_uop = apply_rewrite(base_vector.gep((2, 3)))
    self.assertEqual([sub_uop.arg for sub_uop in optimized_uop.src], [3.0, 4.0])

  def test_gep_on_vconst(self):
    # GEP on a VCONST to extract a single element
    vconst = UOp(Ops.VCONST, dtypes.float32.vec(4), arg=(1.0, 2.0, 3.0, 4.0))
    self.assertEqual(apply_rewrite(vconst.gep(2)).arg, 3.0)

  def test_gep_tuple_on_vconst(self):
    # GEP on a VCONST using a tuple to extract multiple elements
    vconst = UOp(Ops.VCONST, dtypes.float32.vec(4), arg=(7.0, 8.0, 9.0, 10.0))
    optimized_uop = apply_rewrite(vconst.gep((1, 3)))
    self.assertEqual([sub_uop.arg for sub_uop in optimized_uop.src], [8.0, 10.0])

  def test_gep_gep_simplification(self):
    # Nested GEP simplification on a vector dtype
    base_vector = UOp.const(dtypes.float32.vec(4), (10.0, 20.0, 30.0, 40.0))
    gep_inner = base_vector.gep(1)  # Extract 2nd element (20.0)
    self.assertEqual(apply_rewrite(gep_inner.gep(0)).arg, 20.0)

  def test_vectorize_multiple_elements(self):
    # Vectorizing multiple elements using GEP
    base_vector = UOp.const(dtypes.float32.vec(4), (5.0, 10.0, 15.0, 20.0))
    vectorized_uop = UOp(Ops.VECTORIZE, dtypes.float32.vec(4), src=(base_vector.gep(0), base_vector.gep(1), base_vector.gep(2), base_vector.gep(3)))
    optimized_uop = apply_rewrite(vectorized_uop)
    self.assertEqual([sub_uop.arg for sub_uop in optimized_uop.src], [5.0, 10.0, 15.0, 20.0])


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
    a = UOp.variable('a', 0, 10)
    b = UOp.variable('b', 0, 10)
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
    a = UOp.variable('a', 0, 10)
    n1 = a.sin()
    ret = n1.sin()
    ret = substitute(ret, {a.sin():a.sqrt()})
    self.assertIs(ret, a.sqrt().sin())

  def test_double_sin_to_sqrt(self):
    a = UOp.variable('a', 0, 10)
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
    a = UOp.variable('a', 0, 10)
    pm = PatternMatcher([
      (UPat(Ops.DEFINE_VAR, name="x"), lambda x: x.replace(op=Ops.DEFINE_REG)),
      (UPat(Ops.DEFINE_REG, name="x"), lambda x: x.replace(op=Ops.DEFINE_VAR)),
    ])
    with self.assertRaises(RuntimeError):
      graph_rewrite(a, pm)

  def test_inf_loop_bottom_up(self):
    a = UOp.variable('a', 0, 10)
    pm = PatternMatcher([
      (UPat(Ops.DEFINE_VAR, name="x"), lambda x: x.replace(op=Ops.DEFINE_REG)),
      (UPat(Ops.DEFINE_REG, name="x"), lambda x: x.replace(op=Ops.DEFINE_VAR)),
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

if __name__ == '__main__':
  unittest.main()
