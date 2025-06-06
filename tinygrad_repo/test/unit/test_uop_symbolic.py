#!/usr/bin/env python
import unittest, pickle, functools
import z3

from tinygrad.dtype import dtypes, ConstType
from tinygrad.codegen import full_rewrite
from tinygrad.codegen.devectorizer import sym
from tinygrad.helpers import Context
from tinygrad.uop.ops import UOp, Ops, graph_rewrite, sym_infer
from tinygrad import Variable
from tinygrad.uop.spec import z3_renderer

def render(self) -> tuple[str, ConstType, ConstType]:
  # NOTE: we need STORE so the ALU op has children
  glbl = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=0)
  uops = full_rewrite(UOp(Ops.STORE, dtypes.void, (glbl.index(UOp.const(dtypes.int, 0)), self)).sink())
  rewritten_uop = [uop for uop in uops if uop.op is Ops.STORE][0].src[-1]
  return rewritten_uop.render(simplify=False), rewritten_uop.vmin, rewritten_uop.vmax

def uconst(val): return UOp.const(dtypes.int, val)
def usum(ops): return functools.reduce(lambda x,y: x+y, ops)
def uand(ops): return functools.reduce(lambda x,y: x*y, ops)

# *** leave tests the same

class TestSymbolicPickle(unittest.TestCase):
  def _test_pickle_unpickle(self, x): self.assertEqual(x, pickle.loads(pickle.dumps(x)))
  def test_pickle_variable(self): self._test_pickle_unpickle(Variable("a", 3, 8))
  def test_pickle_variable_times_2(self): self._test_pickle_unpickle(Variable("a", 3, 8)*2)

class TestSymbolic(unittest.TestCase):
  def helper_test_variable(self, v, n, m, s):
    rendered, nmin, nmax = render(v)
    if isinstance(s, tuple): self.assertIn(rendered, s)
    else: self.assertEqual(rendered, s)
    self.assertEqual(nmin, n)
    self.assertEqual(nmax, m)
    solver = z3.Solver()
    z3_sink = graph_rewrite(v.sink(v.simplify()), z3_renderer, ctx=(solver, {}))
    expr, epxr_simplified = z3_sink.src[0].arg, z3_sink.src[1].arg
    self.assertEqual(solver.check(expr != epxr_simplified), z3.unsat, "simplified expression not equal to original")

  def test_cmp_simple(self):
    self.helper_test_variable(Variable("a", 3, 8) < 4, 0, 1, "(a<4)")
    self.helper_test_variable(Variable("a", 3, 8) >= 8, 0, 1, "((a<8)!=True)")

  def test_ge(self):
    self.helper_test_variable(Variable("a", 3, 8) >= 77, 0, 0, "False")
    self.helper_test_variable(Variable("a", 3, 8) >= 9, 0, 0, "False")
    self.helper_test_variable(Variable("a", 3, 8) >= 8, 0, 1, "((a<8)!=True)")
    self.helper_test_variable(Variable("a", 3, 8) >= 4, 0, 1, "((a<4)!=True)")
    self.helper_test_variable(Variable("a", 3, 8) >= 3, 1, 1, "True")
    self.helper_test_variable(Variable("a", 3, 8) >= 2, 1, 1, "True")

  def test_lt(self):
    self.helper_test_variable(Variable("a", 3, 8) < 77, 1, 1, "True")
    self.helper_test_variable(Variable("a", 3, 8) < 9, 1, 1, "True")
    self.helper_test_variable(Variable("a", 3, 8) < 8, 0, 1, "(a<8)")
    self.helper_test_variable(Variable("a", 3, 8) < 4, 0, 1, "(a<4)")
    self.helper_test_variable(Variable("a", 3, 8) < 3, 0, 0, "False")
    self.helper_test_variable(Variable("a", 3, 8) < 2, 0, 0, "False")
    self.helper_test_variable(Variable("a", 3, 4) < Variable("b", 5, 6), 1, 1, "True")
    self.helper_test_variable(Variable("a", 3, 5) < Variable("b", 5, 6), 0, 1, "(a<b)")
    self.helper_test_variable(Variable("a", 5, 6) < Variable("b", 3, 5), 0, 0, "False")
    self.helper_test_variable(Variable("a", 3, 4) < Variable("a", 3, 4), 0, 0, "False")

  def test_lt_divides(self):
    expr = (Variable("idx", 0, 511)*4 + Variable("FLOAT4_INDEX", 0, 3)) < 512
    self.helper_test_variable(expr, 0, 1, "(idx<128)")

  def test_lt_divides_and(self):
    expr = uand([(Variable("idx1", 0, 511)*4 + Variable("FLOAT4_INDEX", 0, 3)) < 512,
                 (Variable("idx2", 0, 511)*4 + Variable("FLOAT4_INDEX", 0, 3)) < 512])
    self.helper_test_variable(expr, 0, 1, "((idx1<128)&(idx2<128))")

  def test_lt_factors(self):
    expr = (Variable("idx1", 0, 511)*4 + Variable("FLOAT4_INDEX", 0, 256)) < 512
    self.helper_test_variable(expr, 0, 1, ("(((idx1*4)+FLOAT4_INDEX)<512)", "((FLOAT4_INDEX+(idx1*4))<512)"))

  def test_div_reduction(self):
    self.helper_test_variable(Variable("a", 2, 3)//2, 1, 1, "1")

  def test_equality(self):
    idx1 = Variable("idx1", 0, 3)
    idx2 = Variable("idx2", 0, 3)
    assert idx1 is idx1
    assert idx1 is not idx2
    assert idx1*4 is idx1*4
    assert idx1*4 is not idx1*3
    assert idx1*4 is not idx1+4
    assert idx1*4 is not idx2*4
    assert idx1+idx2 is idx1+idx2
    assert idx1+idx2 is not idx2+idx1
    assert idx1+idx2 is not idx2
    assert idx1*idx2 is not idx2*idx1

  def test_factorize(self):
    a = Variable("a", 0, 8)
    b = Variable("b", 0, 8)
    self.helper_test_variable(a*2+a*3, 0, 8*5, "(a*5)")
    self.helper_test_variable(b+a*2+a*3, 0, 8*6, "(b+(a*5))")

  def test_factorize_no_mul(self):
    a = Variable("a", 0, 8)
    b = Variable("b", 0, 8)
    self.helper_test_variable(a+a*3, 0, 8*4, "(a*4)")
    self.helper_test_variable((a+b)+a*3, 0, 8*5, "(b+(a*4))")
    self.helper_test_variable((a*3+b)+b*3, 0, 8*7, "((a*3)+(b*4))")

  def test_neg(self):
    self.helper_test_variable(-Variable("a", 0, 8), -8, 0, "(a*-1)")

  def test_add_1(self):
    self.helper_test_variable(Variable("a", 0, 8)+1, 1, 9, "(a+1)")

  def test_sub_1(self):
    self.helper_test_variable(Variable("a", 0, 8)-1, -1, 7, "(a+-1)")

  def test_const_var(self):
    self.helper_test_variable(Variable("fake", 1, 1), 1, 1, "1")

  def test_add_self(self):
    a = Variable("a", 0, 8)
    b = Variable("b", 0, 8)
    self.helper_test_variable(a+a, 0, 16, "(a*2)")
    self.helper_test_variable((a+b)+b, 0, 24, "(a+(b*2))")

  def test_sub_self(self):
    a = Variable("a", 0, 8)
    self.helper_test_variable(a-a, 0, 0, "0")
    self.helper_test_variable(a*3-a, 0, 16, "(a*2)")

  def test_mul_0(self):
    self.helper_test_variable(Variable("a", 0, 8)*0, 0, 0, "0")

  def test_mul_1(self):
    self.helper_test_variable(Variable("a", 0, 8)*1, 0, 8, "a")

  def test_mul_neg_1(self):
    self.helper_test_variable((Variable("a", 0, 2)*-1)//3, 0, 0, "0")
    self.helper_test_variable((Variable("a", 2, 7)*-1)//3, -2, 0, "((a//3)*-1)")

  def test_mul_2(self):
    self.helper_test_variable(Variable("a", 0, 8)*2, 0, 16, "(a*2)")

  def test_div_1(self):
    self.helper_test_variable(Variable("a", 0, 8)//1, 0, 8, "a")

  def test_mod_1(self):
    self.helper_test_variable(Variable("a", 0, 8)%1, 0, 0, "0")

  def test_max_folds(self):
    self.helper_test_variable(Variable("a", 0, 20).maximum(10).maximum(11), 11, 20, "max(a, 11)")

  def test_add_min_max(self):
    self.helper_test_variable(Variable("a", 0, 8) * 2 + 12, 12, 16+12, "((a*2)+12)")

  def test_div_remove(self):
    self.helper_test_variable(Variable("a", 0, 7) // 20, 0, 0, "0")

  def test_div_min_max(self):
    self.helper_test_variable(Variable("a", 1, 7) // 2, 0, 3, "(a//2)")
    self.helper_test_variable(Variable("a", 0, 6) // 2, 0, 3, "(a//2)")

  def test_div_neg_min_max(self):
    self.helper_test_variable(Variable("a", 1, 7) // -2, -3, 0, "((a//2)*-1)")
    self.helper_test_variable(Variable("a", 0, 6) // -2, -3, 0, "((a//2)*-1)")

  def test_sum_div_remove(self):
    self.helper_test_variable(usum([Variable("a", 0, 7), Variable("b", 0, 3)]) // 20, 0, 0, "0")

  def test_sum_div_min_max(self):
    self.helper_test_variable(usum([Variable("a", 0, 7), Variable("b", 0, 3)]) // 2, 0, 5, "((a+b)//2)")

  def test_sum_div_mod_factor(self):
    self.helper_test_variable(usum([Variable("a", 0, 7)*4, Variable("b", 0, 3)*4]) // 2, 0, 20, "((a*2)+(b*2))")
    self.helper_test_variable(usum([Variable("a", 0, 7)*4, Variable("b", 0, 3)*4]) % 2, 0, 0, "0")

  def test_sum_div_some_factor(self):
    self.helper_test_variable(usum([Variable("a", 0, 7)*5, Variable("b", 0, 3)*4]) // 2, 0, 23, ("(((a*5)//2)+(b*2))", "((b*2)+((a*5)//2))"))

  def test_sum_div_trim_const(self):
    self.helper_test_variable((Variable("a", 0, 7)*4 + Variable("b", 0, 3)*4 + 7) // 16, 0, 2, "(((a+b)+1)//4)")

  def test_sum_div_some_partial_factor(self):
    self.helper_test_variable(usum([Variable("a", 0, 7)*6, Variable("b", 0, 7)*6]) // 16, 0, 5, "(((a*3)+(b*3))//8)")
    self.helper_test_variable(usum([uconst(16), Variable("a", 0, 7)*6, Variable("b", 0, 7)*6]) // 16, 1, 6, "((((a*3)+(b*3))//8)+1)")
    self.helper_test_variable((Variable("a", 0, 7)*30+20)//20, 1, 11, "(((a*3)//2)+1)")

  def test_sum_div_no_factor(self):
    self.helper_test_variable(usum([Variable("a", 0, 7)*5, Variable("b", 0, 3)*5]) // 2, 0, 25, "(((a*5)+(b*5))//2)")

  def test_mod_factor(self):
    self.helper_test_variable(usum([Variable("a", 0, 7)*100, Variable("b", 0, 3)*50]) % 100, 0, 50, "((b%2)*50)")

  def test_mod_to_sub(self):
    self.helper_test_variable((1+Variable("a",1,2))%2, 0, 1, "(a+-1)")

  def test_mod_congruence(self):
    self.helper_test_variable((3+3*Variable("a",0,3))%4, 0, 3, "((a*-1)+3)")
    self.helper_test_variable((17+13*Variable("a",0,3))%18, 2, 17, "((a*-5)+17)")
    self.helper_test_variable((2+9*Variable("a",0,3))%18, 2, 11, "(((a%2)*9)+2)")

  def test_mod_congruence_mul_add(self):
    self.helper_test_variable((6*(Variable("a", 0, 2)+1))%9, 0, 6, "((a*-3)+6)")

  def test_mod_congruence_multiple_vars(self):
    self.helper_test_variable((9+9*Variable("x",0,3)+9*Variable("y",0,3))%10, 3, 9, "(((x*-1)+(y*-1))+9)")
    self.helper_test_variable((7+9*Variable("x",0,2)+9*Variable("y",0,2)+Variable("z",0,2))%10, 3, 9,
                              ("(((z+(x*-1))+(y*-1))+7)", "(((y*-1)+(z+(x*-1)))+7)"))
    self.helper_test_variable((10+12*Variable("x",0,2)+Variable("y", 0, 4)%3)%13, 8, 12, "(((x*-1)+(y%3))+10)")

  def test_div_congruence(self):
    self.helper_test_variable((3+3*Variable("a",0,3))//4, 0, 3, "a")
    self.helper_test_variable((18+17*Variable("a",0,2)+17)//18, 1, 3, "(a+1)")

  def test_div_congruence_multiple_vars(self):
    self.helper_test_variable((9+(9+10)*Variable("x",0,3)+(8+10)*Variable("y",0,2))//10, 0, 10, "((x*2)+(y*2))")

  def test_mod_binary_expression(self):
    self.helper_test_variable((3+Variable("a",0,1))%4, 0, 3, "((a*-3)+3)")
    self.helper_test_variable((3+Variable("a",4,5))%4, 0, 3, "((a*-3)+15)")

  def test_sum_div_const(self):
    self.helper_test_variable(usum([Variable("a", 0, 7)*4, uconst(3)]) // 4, 0, 7, "a")

  def test_sum_div_const_big(self):
    self.helper_test_variable(usum([Variable("a", 0, 7)*4, uconst(3)]) // 16, 0, 1, "(a//4)")

  def test_sum_lt_fold(self):
    self.helper_test_variable(usum([Variable("a", 0, 7) * 4, Variable("b", 0, 3)]) < 16, 0, 1, "(a<4)")
    self.helper_test_variable(usum([Variable("a", 0, 7) * 4, Variable("b", 0, 4)]) < 16, 0, 1,
                              ("(((a*4)+b)<16)", "((b+(a*4))<16)"))
    self.helper_test_variable(usum([Variable("uidx", 0, 3), Variable("a", 0, 1529) * 12]) < (4 * 67), 0, 1, "(a<23)")

  def test_mul_mod_large(self):
    self.helper_test_variable((Variable("a", 0, 20)*10)%9, 0, 8, "(a%9)")

  def test_mul_mod_small(self):
    self.helper_test_variable((Variable("a", 0, 5)*10)%9, 0, 5, "a")

  def test_mod_mod(self):
    self.helper_test_variable((Variable("a", 0, 31)%12)%4, 0, 3, "(a%4)")
    self.helper_test_variable(((4*Variable("a", 0, 31)) % 12) % 4, 0, 0, "0")
    self.helper_test_variable(((5*Variable("a", 0, 31)) % 12) % 5, 0, 4, "(((a*5)%12)%5)")
    self.helper_test_variable((Variable("a", 0, 31) % 4) % 12, 0, 3, "(a%4)")

  def test_mul_mul(self):
    self.helper_test_variable((Variable("a", 0, 5)*10)*9, 0, 5*10*9, "(a*90)")

  def test_mul_lt(self):
    self.helper_test_variable(Variable("a", 0, 5)*4 < 13, 0, 1, "(a<4)")
    self.helper_test_variable(Variable("a", 0, 5)*4 < 16, 0, 1, "(a<4)")
    self.helper_test_variable(Variable("a", 0, 5)*(-2) < 0, 0, 1, "((a*-1)<0)")
    self.helper_test_variable(Variable("a", 0, 5)*4 >= 12, 0, 1, "((a<3)!=True)")
    self.helper_test_variable(Variable("a", 0, 5)*4 >= 13, 0, 1, "((a<4)!=True)")

  def test_div_div(self):
    self.helper_test_variable((Variable("a", 0, 1800)//10)//9, 0, 20, "(a//90)")

  def test_div_const_div(self):
    a = Variable("a", 0, 124)
    self.helper_test_variable((a//2+1)//2, 0, 31, "((a+2)//4)")
    self.helper_test_variable(((-a)//2-1)//2, -31, 0, "(((a+2)//4)*-1)")
    self.helper_test_variable(((-a)//2+10)//2, -26, 5, "((((a//2)*-1)+10)//2)")

  def test_div_const_div_wrong_sign(self):
    a = Variable("a", 0, 124)
    self.helper_test_variable(((a-10)//2+10)//2, 2, 33, "((((a+-10)//2)+10)//2)")

  def test_distribute_mul(self):
    self.helper_test_variable(usum([Variable("a", 0, 3), Variable("b", 0, 5)])*3, 0, 24, "((a*3)+(b*3))")
    self.helper_test_variable((1+Variable("a", 0, 3))*(-2)+12, 4, 10, "((a*-2)+10)")

  def test_mod_mul_sum(self):
    self.helper_test_variable(usum([Variable("b", 0, 2), Variable("a", 0, 5)*10])%9, 0, 7, ("(b+a)", "(a+b)"))

  def test_sum_0(self):
    self.helper_test_variable(usum([Variable("a", 0, 7)]), 0, 7, "a")

  def test_mod_remove(self):
    self.helper_test_variable(Variable("a", 0, 6)%100, 0, 6, "a")

  def test_big_mod(self):
    self.helper_test_variable(Variable("a", -20, 20)%10, -9, 9, "(a%10)")
    self.helper_test_variable(Variable("a", -20, 0)%10, -9, 0, "(((a*-1)%10)*-1)")
    self.helper_test_variable(Variable("a", -20, 1)%10, -9, 9, "(a%10)")  # TODO: tighter max
    self.helper_test_variable(Variable("a", 0, 20)%10, 0, 9, "(a%10)")
    self.helper_test_variable(Variable("a", -1, 20)%10, -9, 9, "(a%10)")  # TODO: tighter min

  def test_ge_remove(self):
    self.helper_test_variable(Variable("a", 0, 6) >= 25, 0, 0, "False")

  def test_lt_remove(self):
    self.helper_test_variable(Variable("a", 0, 6) < -3, 0, 0, "False")
    self.helper_test_variable(Variable("a", 0, 6) < 3, 0, 1, "(a<3)")
    self.helper_test_variable(Variable("a", 0, 6) < 8, 1, 1, "True")

  def test_lt_sum_remove(self):
    self.helper_test_variable(Variable("a", 0, 6) + 2 < 3, 0, 1, "(a<1)")

  def test_lt_simple_factor(self):
    self.helper_test_variable((Variable("a", 0, 6)*6+Variable("b", 0, 6)*6) < 8, 0, 1, "(((a*3)+(b*3))<4)")

  def test_lt_sum_factor_rhs_partial(self):
    self.helper_test_variable((Variable("a", 0, 6)*6 + Variable("b", 0, 6)*4 + Variable("c", 0, 6)*8) < 4, 0, 1,
                              ("((((a*3)+(b*2))+(c*4))<2)", "(((b*2)+((a*3)+(c*4)))<2)"))

  def test_lt_sum_factor_rhs_all(self):
    self.helper_test_variable((Variable("a", 0, 6)*6 + Variable("b", 0, 6)*4 + Variable("c", 0, 6)*8) < 2, 0, 1,
                              ("((((a*3)+(b*2))+(c*4))<1)", "(((b*2)+((a*3)+(c*4)))<1)"))

  def test_and_fold(self):
    self.helper_test_variable(uand([uconst(0), Variable("a", 0, 1)]), 0, 0, "0")

  def test_and_remove(self):
    self.helper_test_variable(uand([uconst(1), Variable("a", 0, 1)]), 0, 1, "a")

  def test_mod_factor_negative(self):
    self.helper_test_variable(usum([uconst(-29), Variable("a", 0, 10), Variable("b", 0, 10)*28]) % 28, -27, 27, "(((a+(b*28))+-29)%28)")
    self.helper_test_variable(usum([uconst(-29), Variable("a", 0, 100), Variable("b", 0, 10)*28]) % 28, -27, 27, "(((a+(b*28))+-29)%28)")

  def test_sum_combine_num(self):
    self.helper_test_variable(usum([uconst(29), Variable("a", 0, 10), uconst(-23)]), 6, 16, "(a+6)")

  def test_sum_num_hoisted_and_factors_cancel_out(self):
    self.helper_test_variable(usum([Variable("a", 0, 1) * -4 + 1, Variable("a", 0, 1) * 4]), 1, 1, "1")

  @unittest.expectedFailure  # only correct for floordiv, not truncdiv
  def test_div_cancel(self):
    self.helper_test_variable(usum([uconst(-40), Variable("a", 0, 10)*2, Variable("b", 0, 10)*40])//40, -1, 9, "(b+-1)")

  def test_div_cancel_correct(self):
    with Context(CORRECT_DIVMOD_FOLDING=1):
      self.helper_test_variable(usum([uconst(-40), Variable("a", 0, 10)*2, Variable("b", 0, 10)*40])//40, -1, 9, "(((a+(b*20))+-20)//20)")

  @unittest.expectedFailure  # only correct for floordiv, not truncdiv
  def test_mod_cancel(self):
    self.helper_test_variable(usum([uconst(-40), Variable("a", 0, 10)*2, Variable("b", 0, 10)*40]) % 40, 0, 20, "(a*2)")

  def test_mod_cancel_correct(self):
    with Context(CORRECT_DIVMOD_FOLDING=1):
      self.helper_test_variable(usum([uconst(-40), Variable("a", 0, 10)*2, Variable("b", 0, 10)*40]) % 40, -38, 38, "((((a+(b*20))+-20)%20)*2)")

  def test_mul_div(self):
    self.helper_test_variable((Variable("a", 0, 10)*4)//4, 0, 10, "a")

  def test_add_div(self):
    # careful about the lower bounds and upper bounds
    self.helper_test_variable((Variable("a", 0, 5)-2)//4, 0, 0, "0")
    self.helper_test_variable((Variable("a", 0, 5)-1)//4, 0, 1, "((a+-1)//4)")
    self.helper_test_variable((Variable("a", 0, 5))//4, 0, 1, "(a//4)")
    self.helper_test_variable((Variable("a", 0, 5)+1)//4, 0, 1, "((a+1)//4)")
    self.helper_test_variable((Variable("a", 0, 5)+2)//4, 0, 1, "((a+2)//4)")
    self.helper_test_variable((Variable("a", 0, 5)+3)//4, 0, 2, "((a+3)//4)")
    self.helper_test_variable((Variable("a", 0, 5)+4)//4, 1, 2, "((a//4)+1)")
    self.helper_test_variable((Variable("a", 0, 5)+5)//4, 1, 2, "(((a+1)//4)+1)")

  def test_div_neg_rem(self):
    self.helper_test_variable((-Variable("a", 0, 255)+256)//2, 0, 128, "((((a+1)//2)*-1)+128)")

  def test_mul_div_factor_mul(self):
    self.helper_test_variable((Variable("a", 0, 10)*8)//4, 0, 20, "(a*2)")

  def test_mul_div_factor_mul_neg(self):
    self.helper_test_variable((Variable("a", 0, 10)*-8+16)//4, -16, 4, "((a*-2)+4)")

  def test_mul_div_factor_div(self):
    self.helper_test_variable((Variable("a", 0, 10)*4)//8, 0, 5, "(a//2)")

  def test_mul_div_factor_div_neg(self):
    self.helper_test_variable((Variable("a", 0, 10)*-4+4)//8, -4, 0, "(((a*-1)+1)//2)")

  def test_mod_gcd_factor_neg(self):
    self.helper_test_variable((Variable("a", 0, 10)*-4+4)%8, -4, 4, "((((a*-1)+1)%2)*4)")

  def test_mod_gcd_fold_neg(self):
    self.helper_test_variable((Variable("a", 0, 10)*-8+20)%4, 0, 0, "0")

  def test_sum_div_partial_remove(self):
    self.helper_test_variable(usum([Variable("idx0", 0, 127)*4, Variable("idx2", 0, 3)])//4, 0, 127, "idx0")

  def test_cdiv_const_evaluation(self):
    self.helper_test_variable((Variable("a", 0, 2)-12)//8, -1, -1, "-1")
    self.helper_test_variable((-Variable("a", 0, 2))//7, 0, 0, "0")

  def test_cmod_const_evaluation(self):
    self.helper_test_variable((Variable("a", 1, 1)*-3)%8, -3, -3, "-3")
    self.helper_test_variable((-Variable("a", 10, 10))%7, -3, -3, "-3")

  def test_div_numerator_negative(self):
    self.helper_test_variable((Variable("idx", 0, 9)*-10)//11, -8, 0, "(((idx*10)//11)*-1)")

  def test_div_into_mod(self):
    self.helper_test_variable((Variable("idx", 0, 16)*4)%8//4, 0, 1, "(idx%2)")

  def test_div_neg_cancel(self):
    self.helper_test_variable((-Variable("idx", 0, 100)+199)//-4 + 50, 1, 26, "((idx//4)+1)")
    self.helper_test_variable((-Variable("idx", 0, 100)+200)//-4 + 50, 0, 25, "((idx+3)//4)")
    self.helper_test_variable((-Variable("idx", 0, 100)+201)//-4 + 50, 0, 25, "((idx+2)//4)")

  def test_sum_div_big_const(self):
    gidx0 = Variable("gidx0", 0, 24)
    self.helper_test_variable((gidx0+19)//20, 0, 2, "((gidx0+19)//20)")
    self.helper_test_variable((gidx0+20)//20, 1, 2, "((gidx0//20)+1)")
    self.helper_test_variable((gidx0+21)//20, 1, 2, "(((gidx0+1)//20)+1)")

  def test_sum_div_complex1(self):
    gidx0 = Variable("gidx0", 0, 24)
    gidx1 = Variable("gidx1", 0, 1)
    gidx2 = Variable("gidx2", 0, 255)
    lidx0 = Variable("lidx0", 0, 1)
    lidx1 = Variable("lidx1", 0, 15)
    lidx2 = Variable("lidx2", 0, 3)
    alu0 = gidx2*640+gidx1*160+(gidx0//5)*2+lidx0*320+lidx1*10
    self.helper_test_variable((alu0+lidx2*2+1)//20, 0, 8192,
                              ("((((((gidx0//5)+lidx2)//5)+lidx1)//2)+(((gidx2*32)+(gidx1*8))+(lidx0*16)))",
                               "(((lidx1+((lidx2+(gidx0//5))//5))//2)+((gidx2*32)+((gidx1*8)+(lidx0*16))))",
                               "((((gidx1*8)+(gidx2*32))+(lidx0*16))+((lidx1+((lidx2+(gidx0//5))//5))//2))"))

  def test_sum_div_complex2(self):
    gidx0 = Variable("gidx0", 0, 7)
    lidx2 = Variable("lidx2", 0, 1)
    lidx3 = Variable("lidx3", 0, 1)
    self.helper_test_variable((gidx0*4+lidx2*2+1)//10, 0, 3, ("(((gidx0*2)+lidx2)//5)", "((lidx2+(gidx0*2))//5)"))
    self.helper_test_variable((gidx0*4+lidx2*2+lidx3)//10, 0, 3, ("(((gidx0*2)+lidx2)//5)", "((lidx2+(gidx0*2))//5)"))
    self.helper_test_variable((gidx0*2+lidx2)//10, 0, 1, "(gidx0//5)")

  def test_sum_div_complex3(self):
    gidx0 = Variable("gidx0", 0, 7)
    lidx2 = Variable("lidx2", 0, 12)
    lidx3 = Variable("lidx3", 0, 1)
    self.helper_test_variable((gidx0*4+lidx2*2+lidx3)//12, 0, 4, ("(((lidx2//2)+gidx0)//3)", "((gidx0+(lidx2//2))//3)"))
    self.helper_test_variable((lidx2*2+gidx0*4+lidx3)//12, 0, 4, ("(((lidx2//2)+gidx0)//3)", "((gidx0+(lidx2//2))//3)"))

  def test_sum_mul_distribute(self):
    gidx0 = Variable("gidx0", 0, 7)
    lidx2 = Variable("lidx2", 0, 12)
    lidx3 = Variable("lidx3", 0, 1)
    self.helper_test_variable((gidx0+lidx2+lidx3)*4, 0, 80,
                              ("(((gidx0*4)+(lidx2*4))+(lidx3*4))","((lidx3*4)+((gidx0*4)+(lidx2*4)))"))

  @unittest.expectedFailure
  def test_variable_divmod(self):
    start_pos = Variable("start_pos", 0, 127)
    v = start_pos + 1
    idx0 = Variable("idx0", 0, 2)
    idx1 = Variable("idx1", 0, start_pos)
    self.helper_test_variable((idx0*v+idx1)//v, 0, 2, "(idx0)")
    self.helper_test_variable((idx0*v+idx1)%v, 0, start_pos, "idx1")

  def test_divmod_variable_denom_fold_to_const(self):
    x = Variable("x", 20, 23)
    y = Variable("y", 8, 10)
    self.helper_test_variable(x//y, 2, 2, "2")
    self.helper_test_variable(x%y, 0, 7, "(x+(y*-2))")
    # ensure all 4 corners are checked
    x = Variable("x", -10, 10)
    y = Variable("y", -8, 9)
    self.helper_test_variable(x//y, -2147483648, 2147483647, "(x//y)")
    self.helper_test_variable(x%y, -2147483648, 2147483647, "(x%y)")

  def test_div_neg_all_range(self):
    gidx = Variable("gidx", 0, 124)
    lidx = Variable("lidx", 0, 7)
    self.helper_test_variable((-gidx*8-lidx+999)//-4 + 250, 1, 250, "(((gidx*2)+(lidx//4))+1)")
    self.helper_test_variable((-gidx*8-lidx+1000)//-4 + 250, 0, 250, "((gidx*2)+((lidx+3)//4))")
    self.helper_test_variable((-gidx*8-lidx+1001)//-4 + 250, 0, 250, "((gidx*2)+((lidx+2)//4))")
    self.helper_test_variable((-gidx*8-lidx+1002)//-4 + 250, 0, 250, "((gidx*2)+((lidx+1)//4))")

  def test_div_neg_then_neg(self):
    # taken from arange opts
    lidx0 = Variable("lidx0", 0, 7)
    lidx1 = Variable("lidx1", 0, 7)
    alu2 = -lidx0-lidx1
    self.helper_test_variable((((alu2+14)//(-32))+4), 4, 4, "4")
    self.helper_test_variable(-(((alu2+14)//(-32))+4), -4, -4, "-4")
    self.helper_test_variable((((alu2+134)//(-32))+4), 0, 1, "(((lidx0+lidx1)+25)//32)")
    self.helper_test_variable((((alu2+142)//(-32))+4), 0, 0, "0")
    self.helper_test_variable((((alu2+150)//(-32))+4), 0, 0, "0")
    self.helper_test_variable((((alu2+158)//(-32))+4), 0, 0, "0")

  def test_div_mod_recombine(self):
    gidx = Variable("gidx", 0, 124)
    self.helper_test_variable(gidx%4+(gidx//4)*4, 0, 124, "gidx")
    self.helper_test_variable((gidx//4)*4+gidx%4, 0, 124, "gidx")

  def test_div_mod_recombine_folded_mod(self):
    a = Variable("a", 0, 2)
    b = Variable("b", 0, 100)
    self.helper_test_variable((31 * a + 1) % 30 + ((31 * a + 1) // 30) * 30, 1, 63, "((a*31)+1)")
    with self.assertRaises(AssertionError):
      self.helper_test_variable((31 * b + 1) % 18 + ((31 * b + 1) // 18) * 18, 1, 3101, "((b*31)+1)")

  def test_div_mod_recombine_with_gcd(self):
    b = Variable("b", 0, 100)
    exp = (16 * b + 2) % 18 + ((16 * b + 2) // 18) * 18
    self.helper_test_variable(exp, 2, 1602, "((b*16)+2)")
    with self.assertRaises(AssertionError):
      self.helper_test_variable((30 * b + 1) % 18 + ((30 * b + 1) // 18) * 18, 1, 3001, "((b*30)+1)")

  def test_arange_unrolled4(self):
    gidx = Variable("gidx", 0, 2559)
    unrolled_div = (gidx+2561)//4+(gidx+2562)//4+(gidx+2560)//4+(gidx+2559)//4
    self.helper_test_variable(unrolled_div, 2559, 5118, "(gidx+2559)")

  def test_arange_unrolled4_mul(self):
    gidx = Variable("gidx", 0, 2559)
    unrolled_div = 2*((gidx+2561)//4)+2*((gidx+2562)//4)+2*((gidx+2560)//4)+2*((gidx+2559)//4)
    self.helper_test_variable(unrolled_div, 5118, 10236, "((gidx*2)+5118)")

  def test_arange_unrolled4_small(self):
    gidx = Variable("gidx", 0, 3)
    unrolled_div = (gidx)//4+(gidx+2)//4+(gidx+3)//4+(gidx+1)//4
    self.helper_test_variable(unrolled_div, 0, 3, "gidx")

    gidx = Variable("gidx", 0, 2)
    unrolled_div = (gidx)//4+(gidx+2)//4+(gidx+3)//4+(gidx+1)//4
    self.helper_test_variable(unrolled_div, 0, 2, "gidx")

    gidx = Variable("gidx", 0, 1)
    unrolled_div = (gidx)//4+(gidx+2)//4+(gidx+3)//4+(gidx+1)//4
    self.helper_test_variable(unrolled_div, 0, 1, "gidx")

  def test_arange_unrolled2(self):
    gidx = Variable("gidx", 0, 2559)
    unrolled_div = (gidx+2559)//2+(gidx+2560)//2+3
    self.helper_test_variable(unrolled_div, 2562, 5121, "(gidx+2562)")

  def test_arange_unrolled2_neg(self):
    ridx = Variable("ridx", 0, 255)
    unrolled_div = -((255-ridx)//2) - ((256-ridx)//2)
    self.helper_test_variable(unrolled_div, -255, 0, "(ridx+-255)")

  def test_gated_load(self):
    idx = Variable("idx", 0, 24)
    self.helper_test_variable(idx//4, 0, 6, "(idx//4)")
    # TODO: simplify the true branch
    self.helper_test_variable((idx<4).where(idx//4, idx.const_like(-1)), -1, 6, "((idx//4) if (idx<4) else -1)")

  def test_idiv_lt(self):
    idx = Variable("idx", 0, 24)
    self.helper_test_variable((idx//4<3), 0, 1, "(idx<12)")
    self.helper_test_variable(((idx-20)//4<-3), 0, 1, "(idx<5)")
    self.helper_test_variable(((idx-10)//4<0), 0, 1, "(idx<7)")
    self.helper_test_variable((idx//-4<-3), 0, 1, "(((idx//4)*-1)<-3)")

  def test_simplex_lt(self):
    a = Variable("a", 0, 3)
    b = Variable("b", 0, 3)
    c = Variable("c", 0, 3)
    d = Variable("d", -3, 3)
    self.helper_test_variable((a<1).ne(True), 0, 1, "((a<1)!=True)")
    self.helper_test_variable((a+b<1).ne(True), 0, 1, "(((a+b)<1)!=True)")
    self.helper_test_variable((a*3+b*4<1).ne(True), 0, 1, "(((a+b)<1)!=True)")
    self.helper_test_variable((a*(-3)+b*4<1).ne(True), 0, 1, "((((a*-3)+(b*4))<1)!=True)")  # negative coeff, should not be simplified
    self.helper_test_variable((a*3+d*4<1).ne(True), 0, 1, "((((a*3)+(d*4))<1)!=True)")  # var can be negative, should not be simplified
    self.helper_test_variable((a+b+c*2<1).ne(True), 0, 1, ("((((a+b)+c)<1)!=True)", "(((c+(a+b))<1)!=True)", '(((b+(a+c))<1)!=True)'))
    self.helper_test_variable((a+b*2+c*4<1).ne(True), 0, 1, ("((((a+b)+c)<1)!=True)", "(((c+(a+b))<1)!=True)", '(((b+(a+c))<1)!=True)'))

  def test_where_removal(self):
    cond = Variable("a", 0, 3) < 2
    u1, u0 = cond.ufix(1), cond.ufix(0)
    self.helper_test_variable(cond, 0, 1, "(a<2)")
    self.helper_test_variable(cond.where(u1, u0), 0, 1, "(a<2)")
    self.helper_test_variable(cond.where(u1, u0).where(u1, u0), 0, 1, "(a<2)")

  def test_where_combine(self):
    cond = Variable("x", 0, 3) < 2
    a = Variable("a", 0, 3)
    b = Variable("b", 0, 3)
    aa = cond.where(a, a.ufix(0))
    bb = cond.where(b, b.ufix(1))
    self.helper_test_variable(aa, 0, 3, "(a if (x<2) else 0)")
    self.helper_test_variable(bb, 0, 3, "(b if (x<2) else 1)")
    self.helper_test_variable(aa+bb, 0, 6, "((a+b) if (x<2) else 1)")
    self.helper_test_variable(aa.maximum(bb), 0, 3, "(max(a, b) if (x<2) else 1)")

    # not combining because it increased total ALU
    c = Variable("c", 0, 3)
    cc = cond.where(c, c+1)
    self.helper_test_variable(bb+cc, 0, 7, "((b if (x<2) else 1)+(c if (x<2) else (c+1)))")

    # not combining  # TODO: can combine if it can further simplify?
    ab = cond.where(a, b)
    ba = cond.where(b, a)
    self.helper_test_variable(ab+ba, 0, 6, "((a if (x<2) else b)+(b if (x<2) else a))")

    # not combining  # TODO: can combine if one is identity element const
    self.helper_test_variable(aa+ab, 0, 6, "((a if (x<2) else b)+(a if (x<2) else 0))")

  def test_negation_in_where(self):
    cond = Variable("x", 0, 3) < 2
    a = Variable("a", 0, 3)
    b = Variable("b", 0, 3)
    w = cond.logical_not().where(a, b)
    self.helper_test_variable(w, 0, 3, "(b if (x<2) else a)")

  def test_neg_in_comp(self):
    a = Variable("a", 0, 3)
    b = Variable("b", 0, 3)
    self.helper_test_variable(-a<-b, False, True, "(b<a)")

  def test_where_cast(self):
    s = Variable("s", 0, 3)
    cond = s < 2
    a = Variable("a", 0, 3)
    b = Variable("b", 0, 3)
    expr = cond.where(a, b).cast(dtypes.half)

    # TODO: copied from render, render does not support cast
    glbl = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=0)
    uops = full_rewrite(UOp(Ops.STORE, dtypes.void, (glbl.index(UOp.const(dtypes.int, 0)), expr)).sink())
    rewritten_uop = [uop for uop in uops if uop.op is Ops.STORE][0].src[-1]

    self.assertEqual(rewritten_uop, cond.where(a.cast(dtypes.half), b.cast(dtypes.half)))

  def test_where_merge_branches(self):
    cond1 = Variable("s", 0, 10) < 6
    cond2 = Variable("s", 0, 10) > 2
    a = Variable("a", 0, 3)
    b = Variable("b", 0, 3)
    expr = cond1.where(cond2.where(a, b), b)
    self.helper_test_variable(expr, 0, 3, "(a if ((s<6)&(2<s)) else b)")

  def test_where_merge_branches2(self):
    cond1 = Variable("s", 0, 10) < 5
    cond2 = Variable("s", 0, 10) < 6
    a = Variable("a", 0, 3)
    b = Variable("b", 0, 3)
    expr = cond1.where(cond2.where(a, b), b)
    # (a if ((s<5)&(s<6)) else b) -> (a if (s<5) else b)
    self.helper_test_variable(expr, 0, 3, "(a if (s<5) else b)")

  def test_symbolic_div(self):
    # from symbolic arange
    a = Variable("a", 1, 10)
    denominator = ((a*-2)+1)
    numerator = (((((a*2)+-1)*2)+1)*a)
    self.helper_test_variable(denominator, -19, -1, "((a*-2)+1)")
    self.helper_test_variable(numerator, 3, 390, "(a*((a*4)+-1))")
    self.helper_test_variable((numerator//denominator)<=0, 1, 1, "True")

class TestSymbolicNumeric(unittest.TestCase):
  def helper_test_numeric(self, f):
    MIN, MAX = 0, 10
    # one number
    for i in range(MIN, MAX):
      v = graph_rewrite(f(uconst(i)), sym)
      self.assertEqual(v.vmin, v.vmax)
      self.assertEqual(v.vmin, f(i))
    for kmin in range(MIN, MAX):
      for kmax in range(MIN, MAX):
        if kmin > kmax: continue
        v = f(Variable("tmp", kmin, kmax))
        values = [f(rv) for rv in range(kmin, kmax+1)]
        # the min and max may not be exact
        self.assertLessEqual(v.vmin, min(values))
        self.assertGreaterEqual(v.vmax, max(values))

  def test_mod_4(self): self.helper_test_numeric(lambda x: (x%4))
  def test_div_4(self): self.helper_test_numeric(lambda x: (x//4))
  def test_plus_1_div_2(self): self.helper_test_numeric(lambda x: (x+1)//2)
  def test_plus_1_mod_2(self): self.helper_test_numeric(lambda x: (x+1)%2)
  def test_times_2(self): self.helper_test_numeric(lambda x: x*2)
  def test_times_2_plus_3(self): self.helper_test_numeric(lambda x: x*2 + 3)
  def test_times_2_plus_3_mod_4(self): self.helper_test_numeric(lambda x: (x*2 + 3)%4)
  def test_times_2_plus_3_div_4(self): self.helper_test_numeric(lambda x: (x*2 + 3)//4)
  def test_times_2_plus_3_div_4_mod_4(self): self.helper_test_numeric(lambda x: ((x*2 + 3)//4)%4)

class TestSymbolicVars(unittest.TestCase):
  def test_simple(self):
    z = uconst(0)
    a = Variable("a", 0, 10)
    b = Variable("b", 0, 10)
    c = Variable("c", 0, 10)
    assert z.vars() == z.vars() == set()
    print(a.vars())
    assert a.vars() == a.vars() == {a}
    m = a * 3
    assert m.vars() == {a}
    s = usum([a, b, c])
    assert s.vars() == {a, b, c}

  def test_compound(self):
    a = Variable("a", 0, 10)
    b = Variable("b", 0, 10)
    c = Variable("c", 0, 10)
    assert (a + b * c).vars() == {a, b, c}
    assert (a % 3 + b // 5).vars() == {a, b}
    # TODO: fix me
    with self.assertRaises(AssertionError):
      assert (a + b + c - a).vars() == {b, c}

  def test_dedup(self):
    a = Variable("a", 0, 10)
    assert (a * a).vars() == {a}
    assert (a//4 + a//6).vars() == {a}

class TestSymInfer(unittest.TestCase):
  def test_sym_infer(self):
    a = Variable("a", 0, 10)
    b = Variable("b", 0, 10)
    c = Variable("c", 0, 10)
    var_vals = {a: 2, b: 3, c: 4}
    assert sym_infer(5, var_vals) == 5
    assert sym_infer(4.2, var_vals) == 4.2
    assert sym_infer(a, var_vals) == 2
    assert sym_infer(b, var_vals) == 3
    assert sym_infer(a+b, var_vals) == 5
    assert sym_infer(a-b, var_vals) == -1
    assert sym_infer(a+b+c, var_vals) == 9
    assert sym_infer(a*b, var_vals) == 6
    assert sym_infer(a*b+c, var_vals) == 10
  def test_sym_infer_cdiv_cmod(self):
    a = Variable("a", -1000, 1)
    b = Variable("b", -1000, 1)
    var_vals = {a: 1, b: -1000}
    assert sym_infer(a%b, var_vals) == 1
    assert sym_infer(a//b, var_vals) == 0

"""
@unittest.skip("not supported on uops yet")
class TestSymbolicSymbolicOps(unittest.TestCase):
  def test_node_divmod_node(self):
    i = Variable("i", 1, 10)
    idx0 = Variable("idx0", 0, i*3-1)
    assert uconst(0) // (Variable("i", 1, 10)*128) == 0
    assert uconst(0) % (Variable("i", 1, 10)*128) == 0
    assert uconst(127) // (Variable("i", 1, 10)*128) == 0
    assert uconst(127) % (Variable("i", 1, 10)*128) == 127
    assert 127 // (Variable("i", 1, 10)*128) == 0
    assert 127 % (Variable("i", 1, 10)*128) == 127
    assert uconst(128) // (Variable("i", 1, 10)*128 + 128) == 0
    assert uconst(128) % (Variable("i", 1, 10)*128 + 128) == 128
    assert 128 // (Variable("i", 1, 10)*128 + 128) == 0
    assert 128 % (Variable("i", 1, 10)*128 + 128) == 128
    assert 0 // (Variable("i", 1, 10)*128) == 0
    assert 0 % (Variable("i", 1, 10)*128) == 0
    assert idx0 // (i*3) == 0
    assert idx0 % (i*3) == idx0
    assert i // i == 1
    assert i % i == 0
    assert 128 // uconst(4) == 32
    assert 128 % uconst(4) == 0
    assert uconst(128) // uconst(4) == 32
    assert uconst(128) % uconst(4) == 0

  def test_mulnode_divmod_node(self):
    i = Variable("i", 1, 10)
    idx0 = Variable("idx0", 0, 31)
    # assert (idx0*(i*4+4)) // (i+1) == (idx0*4)
    # assert (idx0*(i*4+4)) % (i+1) == 0
    assert (idx0*i) % i == 0

  def test_sumnode_divmod_sumnode(self):
    i = Variable("i", 1, 10)
    # idx0 = Variable("idx0", 0, 7)
    # idx1 = Variable("idx1", 0, 3)
    # idx2 = Variable("idx2", 0, i)
    # assert (idx0*(i*4+4)+idx1*(i+1)+idx2) // (i+1) == idx0*4+idx1
    # assert (idx0*(i*4+4)+idx1*(i+1)+idx2) % (i+1) == idx2
    assert (i+1) // (i*128+128) == 0
    assert (i+1) % (i*128+128) == (i+1)
    # assert (i+1+idx2) // (i+1) == 1
    # assert (i+1+idx2) % (i+1) == idx2
    # assert (idx0*(i*4+4)+i+1+idx2) // (i+1) == idx0*4+1
    # assert (idx0*(i*4+4)+i+1+idx2) % (i+1) == idx2
    # assert (i*128+128)*2 // (i*128+128) == 2
    # assert (i*128+128)*2 % (i*128+128) == 0

  def test_sumnode_div_uconst_no_factoring(self):
    gid = Variable("gid", 0, 1023)
    lid = Variable("lid", 0, 3)
    expr_before_div = uconst(-1019)-4*lid-gid
    unfactored_expr = Node.__floordiv__(expr_before_div, uconst(-16), False)
    factored_expr = Node.__floordiv__(expr_before_div, uconst(-16), True)
    self.assertEqual(unfactored_expr.render(), "(((lid*4)+1019+gid)//16)")
    self.assertEqual(factored_expr.render(), "(((((3+gid)//4)+2+lid)//4)+63)")

  def test_mod_node_max(self):
    i = Variable("i", 1, 128)
    gidx0 = Variable("gidx0", 0, i)
    mod = gidx0 % 8
    assert isinstance(mod, ModNode) and mod.a == gidx0 and mod.b == 8
    mod = gidx0 % 2
    assert isinstance(mod, ModNode) and mod.a == gidx0 and mod.b == 2

    gidx0 = Variable("gidx0", 0, i*8+7)
    mod = gidx0 % 8
    assert isinstance(mod, ModNode) and mod.a == gidx0 and mod.b == 8
    mod = gidx0 % 2
    assert isinstance(mod, ModNode) and mod.a == gidx0 and mod.b == 2

  def test_nested_variable_mod(self):
    i = Variable("i", 1, 5)
    idx0 = Variable("idx0", 0, i)
    with self.assertRaises(AssertionError):
      assert idx0 % 2 == idx0

  def test_num_node_mul_node(self):
    a = Variable("a", 1, 5)
    b = uconst(2) * a
    assert b == a * 2
    assert isinstance(b, MulNode)
    b = uconst(1) * a
    assert b == a
    assert isinstance(b, Variable)
    b = uconst(0) * a
    assert b == 0
    assert isinstance(b, uconst)

  def test_substitute(self):
    a = Variable("idx0", 1, 3)
    b = a + 1
    c = b.substitute({a: uconst(1)})
    assert c == uconst(2)
"""

class TestSymbolicRealWorld(unittest.TestCase):
  def test_resnet_half(self):
    gidx0 = Variable("gidx0", 0, 3)
    gidx1 = Variable("gidx1", 0, 127)
    gidx2 = Variable("gidx2", 0, 7)
    lidx3 = Variable("lidx3", 0, 7)
    lidx4 = Variable("lidx4", 0, 1)
    lidx5 = Variable("lidx5", 0, 15)

    idx:UOp = ((((1+lidx5)%16)*49)+(((262145+lidx5)//16)*802816)+(gidx0*3211264)+(gidx1*784)+(gidx2*8)+(lidx4*100352)+-13151129600+lidx3)
    idx = graph_rewrite(idx, sym)
    #print(idx.render())
    # NOTE: this used to have 13,151,129,600 in the output which is out of int32 range.
    self.assertIn(idx.render(),
      ("((((((((((lidx5+1)//16)*802816)+(((lidx5+1)%16)*49))+(gidx0*3211264))+(gidx1*784))+(gidx2*8))+(lidx4*100352))+lidx3)+2207744)",
       '((lidx3+((((((((lidx5+1)//16)*802816)+(((lidx5+1)%16)*49))+(gidx0*3211264))+(gidx1*784))+(gidx2*8))+(lidx4*100352)))+2207744)',
       '((lidx3+((lidx4*100352)+((gidx2*8)+((gidx1*784)+((gidx0*3211264)+((((lidx5+1)//16)*802816)+(((lidx5+1)%16)*49)))))))+2207744)',
      ))

class TestBounds(unittest.TestCase):
  def test_unrolled_arange(self):
    # #include <metal_stdlib>
    # using namespace metal;
    # kernel void r_2560_640_4(device int* data0, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
    #   int gidx0 = gid.x; /* 2560 */
    #   int alu0 = (gidx0*(-1));
    #   int alu1 = max((int)((-640)),((((alu0+2559)/(-4))*(-1))+(-640)));
    #   int alu2 = max((int)((-640)),((((alu0+2560)/(-4))*(-1))+(-640)));
    #   int alu3 = max((int)((-640)),((((alu0+2561)/(-4))*(-1))+(-640)));
    #   int alu4 = max((int)((-640)),((((alu0+2562)/(-4))*(-1))+(-640)));
    #   *(data0+gidx0) = ((alu1*(-1))+(alu2*(-1))+(alu4*(-1))+(alu3*(-1))+(-1));
    # }
    gidx0 = Variable("gidx0", 0, 2559)
    assert gidx0.vmin == 0 and gidx0.vmax == 2559
    alu0 = gidx0 * -1
    assert alu0.vmin == -2559 and alu0.vmax == 0
    assert (alu0+2559).vmin == 0 and (alu0+2559).vmax == 2559
    assert ((alu0+2559)//-4).vmin == -639 and ((alu0+2559)//-4).vmax == 0
    assert (((alu0+2559)//-4)*(-1)).vmin == 0 and (((alu0+2559)//-4)*(-1)).vmax == 639

class TestFuzzFailure(unittest.TestCase):
  def test_fuzz_failure1(self):
    v1=Variable('v1', 0, 8)
    v2=Variable('v2', 0, 2)
    v3=Variable('v3', 0, 1)
    expr = (((((((((((((((((((((((0//4)%2)//8)+-2)+-4)+-3)+v1)+-4)+v2)+-2)+v3)+v2)//3)%7)*1)//2)+v2)*-1)+2)+1)+0)+-3)+v3)
    v1_val, v2_val, v3_val = v1.const_like(8), v2.const_like(0), v3.const_like(0)
    num = expr.simplify().substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    rn = expr.substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    assert num==rn, f"{num} != {rn}"

if __name__ == '__main__':
  unittest.main()
