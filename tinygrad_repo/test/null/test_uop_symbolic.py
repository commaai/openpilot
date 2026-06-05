#!/usr/bin/env python
import unittest, pickle, functools, math
import z3

from tinygrad.dtype import dtypes, ConstType, DType, Invalid
from test.helpers import get_uops
from tinygrad.uop.ops import UOp, Ops, graph_rewrite, sym_infer
from tinygrad.uop.symbolic import sym, commutative, pm_simplify_valid, pm_move_where_on_load
from tinygrad.uop.validate import uops_to_z3

def check_uop_against_string(self, v:UOp, s:str):
  sym_vars = {v.render():v for v in v.toposort() if v.op in (Ops.DEFINE_VAR, Ops.RANGE, Ops.SPECIAL)}
  s_eval = eval(s, sym_vars)
  if isinstance(s_eval, int) and v.dtype==dtypes.weakint: s_eval = UOp.const(dtypes.weakint, s_eval)
  elif isinstance(s_eval, (bool, int, float)): s_eval = UOp.const(dtypes.from_py(s_eval), s_eval)
  s_eval = graph_rewrite(s_eval, commutative, name="cannonicalize eval")
  self.assertIs(s_eval, v, f"eval did not match simplified: {s_eval} != {v.render()} for {s}")

def Variable(name: str, min_val: ConstType, max_val: ConstType, dtype: DType=dtypes.weakint): return UOp.variable(name,min_val,max_val,dtype)
def uconst(val): return UOp.const(dtypes.weakint, val)
def usum(ops): return functools.reduce(lambda x,y: x+y, ops)
def uand(ops): return functools.reduce(lambda x,y: x*y, ops)

# *** leave tests the same

class TestSymbolicPickle(unittest.TestCase):
  def _test_pickle_unpickle(self, x): self.assertEqual(x, pickle.loads(pickle.dumps(x)))
  def test_pickle_variable(self): self._test_pickle_unpickle(Variable("a", 3, 8))
  def test_pickle_variable_times_2(self): self._test_pickle_unpickle(Variable("a", 3, 8)*2)

class TestSymbolic(unittest.TestCase):
  def check_equal_z3(self, expr1, expr2):
    solver = z3.Solver()
    expr1, expr2 = uops_to_z3(solver, expr1, expr2)
    self.assertEqual(solver.check(expr1 != expr2), z3.unsat, "simplified expression not equal to original")

  def helper_test_variable(self, v, n, m, s, test_z3:bool=True):
    v_simplified = graph_rewrite(v, sym, name="simplify symbolic uop")
    if test_z3: self.check_equal_z3(v, v_simplified)
    nmin, nmax = v_simplified.vmin, v_simplified.vmax
    check_uop_against_string(self, v_simplified, s)
    # eval the test string and see if we get the same uop
    self.assertEqual(nmin, n)
    self.assertEqual(nmax, m)

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
    self.helper_test_variable(expr, 0, 1, "(((idx1*4)+FLOAT4_INDEX)<512)")

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

  def test_uop_gcd_method(self):
    a = Variable("a", 0, 8)
    b = Variable("b", 0, 8)
    self.assertEqual(UOp.gcd(a, a*b, a*3).simplify(), a)
    self.assertEqual(UOp.gcd(a*a*a, a*b*a, a*3*a).simplify(), a*a)
    self.assertEqual(UOp.gcd(a*a*10, b*a*5, a*a*5).simplify(), a*5)
    self.assertEqual(UOp.gcd(a*10, b*5, a*5).simplify(), a.const_like(5))
    self.assertEqual(UOp.gcd(a, b*5, a*5).simplify(), a.const_like(1))

  def test_divides_exact(self):
    a = Variable("a", 1, 8)
    b = Variable("b", 1, 8)
    self.assertEqual((a*a*3).divide_exact(a).simplify(), a*3)
    self.assertEqual((a*a*3).divide_exact(a*a*3).simplify(), a.const_like(1))
    self.assertEqual((a*a*6).divide_exact(a*a*3).simplify(), a.const_like(2))
    self.assertEqual((a*b*3).divide_exact(a.const_like(3)).simplify(), a*b)
    self.assertEqual((a*a*3).divide_exact(a*(-3)).simplify(), a*-1)
    self.assertEqual((a*a*b*3).divide_exact(a*b).simplify(), a*3)
    self.assertEqual((a*3+a*b).divide_exact(a).simplify(), b+3)
    self.assertEqual((a*b*3+a*b*b).divide_exact(a*b).simplify(), b+3)
    self.assertEqual((((a*-2)+14)*b).divide_exact(((a*-2)+14)).simplify(), b)

  def test_divide_exact_not(self):
    a = Variable("a", 1, 8)
    b = Variable("b", 1, 8)
    x = Variable("x", -20, 0)
    self.assertEqual((a).divide_exact(b), None)
    self.assertEqual((a+2).divide_exact(a), None)
    self.assertEqual((x*-1).divide_exact(a), None)
    self.assertEqual((a*5).divide_exact(a*10), None)
    self.assertEqual((a*10-1).divide_exact(a*10), None)

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

  def test_xor_0(self):
    self.helper_test_variable(Variable("a", 0, 8, dtypes.int) ^ 0, 0, 8, "a", test_z3=False)

  def test_xor_self_inverse(self):
    self.helper_test_variable((Variable("a", 0, 8, dtypes.int) ^ 5) ^ 5, 0, 8, "a", test_z3=False)

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
    self.helper_test_variable((a*3+b)+a, 0, 40, "(b+(a*4))")
    self.helper_test_variable((a+b)+a*3, 0, 40, "(b+(a*4))")

  def test_sub_self(self):
    a = Variable("a", 0, 8)
    self.helper_test_variable(a-a, 0, 0, "0")
    self.helper_test_variable(a*3-a, 0, 16, "(a*2)")

  def test_mul_0(self):
    self.helper_test_variable(Variable("a", 0, 8)*0, 0, 0, "0")

  def test_mul_1(self):
    self.helper_test_variable(Variable("a", 0, 8)*1, 0, 8, "a")

  def test_mul_neg_1(self):
    self.helper_test_variable((Variable("a", 0, 2)*-1)//3, -1, 0, "((a*-1)//3)")
    self.helper_test_variable((Variable("a", 2, 7)*-1)//3, -3, -1, "((a*-1)//3)")

  def test_mul_2(self):
    self.helper_test_variable(Variable("a", 0, 8)*2, 0, 16, "(a*2)")

  def test_div_1(self):
    self.helper_test_variable(Variable("a", 0, 8)//1, 0, 8, "a")

  def test_mod_1(self):
    self.helper_test_variable(Variable("a", 0, 8)%1, 0, 0, "0")

  def test_max_folds(self):
    self.helper_test_variable(Variable("a", 0, 20).maximum(10).maximum(11), 11, 20, "a.maximum(11)")

  def test_add_min_max(self):
    self.helper_test_variable(Variable("a", 0, 8) * 2 + 12, 12, 16+12, "((a*2)+12)")

  def test_div_remove(self):
    self.helper_test_variable(Variable("a", 0, 7) // 20, 0, 0, "0")

  def test_div_neg_min_max(self):
    self.helper_test_variable(Variable("a", 1, 7) // -2, -4, -1, "(a//-2)")
    self.helper_test_variable(Variable("a", 0, 6) // -2, -3, 0, "(a//-2)")

  def test_div_mod_zero(self):
    with self.assertRaises(ZeroDivisionError):
      (Variable("a", 0, 7) // 0).simplify()
    with self.assertRaises(ZeroDivisionError):
      (Variable("a", 0, 7) % 0).simplify()

  def test_sum_div_remove(self):
    self.helper_test_variable(usum([Variable("a", 0, 7), Variable("b", 0, 3)]) // 20, 0, 0, "0")

  def test_sum_div_min_max(self):
    self.helper_test_variable(usum([Variable("a", 0, 7), Variable("b", 0, 3)]) // 2, 0, 5, "((a+b)//2)")

  def test_sum_div_mod_factor(self):
    self.helper_test_variable(usum([Variable("a", 0, 7)*4, Variable("b", 0, 3)*4]) // 2, 0, 20, "((a*2)+(b*2))")
    self.helper_test_variable(usum([Variable("a", 0, 7)*4, Variable("b", 0, 3)*4]) % 2, 0, 0, "0")

  def test_sum_div_some_factor(self):
    self.helper_test_variable(usum([Variable("a", 0, 7)*5, Variable("b", 0, 3)*4]) // 2, 0, 23, "((a*2)+(b*2)+(a//2))")

  def test_sum_div_trim_const(self):
    self.helper_test_variable((Variable("a", 0, 7)*4 + Variable("b", 0, 3)*4 + 7) // 16, 0, 2, "(((a+b)+1)//4)")

  def test_sum_div_some_partial_factor(self):
    self.helper_test_variable(usum([Variable("a", 0, 7)*6, Variable("b", 0, 7)*6]) // 16, 0, 5, "(((a*3)+(b*3))//8)")
    self.helper_test_variable(usum([uconst(16), Variable("a", 0, 7)*6, Variable("b", 0, 7)*6]) // 16, 1, 6, "((((a*3)+(b*3))//8)+1)")
    self.helper_test_variable((Variable("a", 0, 7)*30+20)//20, 1, 11, "((a+(a//2))+1)")

  def test_sum_div_no_factor(self):
    self.helper_test_variable(usum([Variable("a", 0, 7)*5, Variable("b", 0, 3)*5]) // 2, 0, 25, "((a*2)+(b*2)+((a+b)//2))")

  def test_mod_min_max(self):
    self.helper_test_variable(Variable("x", 0, 10)%Variable("y", 1, 10), 0, 9, "(x%y)")
    self.helper_test_variable(Variable("x", -10, 0)%Variable("y", 1, 10), 0, 9, "(x%y)")
    self.helper_test_variable(Variable("x", 0, 10)%Variable("y", -10, -1), -9, 0, "(x%y)")
    self.helper_test_variable(Variable("x", -10, 0)%Variable("y", -10, -1), -9, 0, "(x%y)")
    self.helper_test_variable(Variable("x", -10, 10)%Variable("y", -10, -1), -9, 0, "(x%y)")

    # test _min_max directly: floor mod with positive divisor is in [0, c-1]; with negative divisor in [c+1, 0]
    self.assertEqual((Variable("x", -10, 0)%Variable("y", -10, -1))._min_max, (-9, 0))
    self.assertEqual((Variable("x", -10, 0)%Variable("y", 1, 10))._min_max, (0, 9))

  def test_range_div_its_symbolic_bound(self):
    a = Variable("a", 1, 10, dtypes.weakint)
    ridx0 = UOp.range(a+2, 0)
    self.helper_test_variable(ridx0//(a+2), 0, 0, "0")

  def test_range_mod_its_symbolic_bound(self):
    a = Variable("a", 1, 10, dtypes.weakint)
    ridx = UOp.range(a+2, 0)
    self.helper_test_variable(ridx%(a+2), 0, 11, "r0")

  def test_div_min_max(self):
    self.helper_test_variable(Variable("a", 2, 7) // 2, 1, 3, "(a//2)")
    self.helper_test_variable(Variable("a", 0, 6) // 2, 0, 3, "(a//2)")

    self.helper_test_variable(Variable("x", 0, 10)//Variable("y", 1, 10), 0, 10, "(x//y)")
    self.helper_test_variable(Variable("x", -10, 0)//Variable("y", 1, 10), -10, 0, "(x//y)")
    self.helper_test_variable(Variable("x", 0, 10)//Variable("y", -10, -1), -10, 0, "(x//y)")
    self.helper_test_variable(Variable("x", -10, 0)//Variable("y", -10, -1), 0, 10, "(x//y)")

    self.helper_test_variable(Variable("x", -10, 10)//Variable("y", 1, 10), -10, 10, "(x//y)")
    self.helper_test_variable(Variable("x", -10, 10)//Variable("y", -10, -1), -10, 10, "(x//y)")

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
                              "(((z+(x*-1))+(y*-1))+7)")
    self.helper_test_variable((10+12*Variable("x",0,2)+Variable("y", 0, 4)%3)%13, 8, 12, "(((x*-1)+(y%3))+10)")

  def test_mod_congruence_tied_remainder(self):
    # when f%c == c/2, both r and r-c have equal abs — try both signs
    self.helper_test_variable((3+2*Variable("x",0,1)+3*Variable("y",0,1))%4, 0, 3, "((x*-2)+(y*-1)+3)")
    self.helper_test_variable((3+6*Variable("x",0,1)+7*Variable("y",0,1))%4, 0, 3, "((x*-2)+(y*-1)+3)")

  def test_div_congruence(self):
    self.helper_test_variable((3+3*Variable("a",0,3))//4, 0, 3, "a")
    self.helper_test_variable((18+17*Variable("a",0,2)+17)//18, 1, 3, "(a+1)")

  def test_div_congruence_multiple_vars(self):
    self.helper_test_variable((9+(9+10)*Variable("x",0,3)+(8+10)*Variable("y",0,2))//10, 0, 10, "((x*2)+(y*2))")

  def test_mod_binary_expression(self):
    self.helper_test_variable((3+Variable("a",0,1))%4, 0, 3, "((a*-3)+3)")
    self.helper_test_variable((3+Variable("a",4,5))%4, 0, 3, "((a*-3)+15)")

  def test_div_binary_expression(self):
    self.helper_test_variable((3+Variable("a",0,1))//4, 0, 1, "a")

  def test_sum_div_const(self):
    self.helper_test_variable(usum([Variable("a", 0, 7)*4, uconst(3)]) // 4, 0, 7, "a")

  def test_sum_div_const_big(self):
    self.helper_test_variable(usum([Variable("a", 0, 7)*4, uconst(3)]) // 16, 0, 1, "(a//4)")

  def test_sum_lt_fold(self):
    self.helper_test_variable(usum([Variable("a", 0, 7) * 4, Variable("b", 0, 3)]) < 16, 0, 1, "(a<4)")
    self.helper_test_variable(usum([Variable("a", 0, 7) * 4, Variable("b", 0, 4)]) < 16, 0, 1, "(((a*4)+b)<16)")
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

  def test_mod_mod_wrong_sign(self):
    v1=Variable("v1", 0, 128)
    v3=Variable("v3", 0, 7)
    self.helper_test_variable((((((v1%2)*2)+((v3+-1)%5))+-2)%5), 0, 4, "((v3+v1%2*2+-3)%5)")

  def test_mod_mod_wrong_sign2(self):
    v2=Variable("v2", 0, 8)
    v3=Variable("v3", 0, 4)
    self.helper_test_variable((((((v3+3)%7)+(v2+-2))%7)%7), 0, 6, "((v2+v3+1)%7)")

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
    self.helper_test_variable(((-a)//2-1)//2, -32, -1, "((a*-1+2)//4+-1)")
    self.helper_test_variable(((-a)//2+10)//2, -26, 5, "(a*-1//4+5)")

  def test_div_const_div_wrong_sign(self):
    a = Variable("a", 0, 124)
    self.helper_test_variable(((a-10)//2+10)//2, 2, 33, "((a+2)//4+2)")

  def test_div_const_div_wrong_sign_divisor(self):
    a = Variable("a", 0, 124)
    self.helper_test_variable(((a+10)//-2+10)//-4, -2, 14, "(((a+10)//-2+10)//-4)")

  def test_nested_div_negative_divisor(self):
    # (x//c1)//c2 -> x//(c1*c2) only when c2>0
    a = Variable("a", 0, 124)
    self.helper_test_variable((a//-2)//-3, 0, 20, "((a//-2)//-3)")
    self.helper_test_variable((a//2)//-3, -21, 0, "((a//2)//-3)")
    self.helper_test_variable((a//-2)//3, -21, 0, "(a//-6)")

  def test_neg_mod(self):
    a = Variable("a", 0, 124)
    self.helper_test_variable((-a)%4, 0, 3, "(a*-1%4)")
    self.helper_test_variable(a%-4, -3, 0, "(a%-4)")

  def test_distribute_mul(self):
    self.helper_test_variable(usum([Variable("a", 0, 3), Variable("b", 0, 5)])*3, 0, 24, "((a*3)+(b*3))")
    self.helper_test_variable((1+Variable("a", 0, 3))*(-2)+12, 4, 10, "((a*-2)+10)")

  def test_mod_mul_sum(self):
    self.helper_test_variable(usum([Variable("b", 0, 2), Variable("a", 0, 5)*10])%9, 0, 7, "(b+a)")

  def test_sum_0(self):
    self.helper_test_variable(usum([Variable("a", 0, 7)]), 0, 7, "a")

  def test_mod_remove(self):
    self.helper_test_variable(Variable("a", 0, 6)%100, 0, 6, "a")

  def test_big_mod(self):
    self.helper_test_variable(Variable("a", -20, 20)%10, 0, 9, "(a%10)")
    self.helper_test_variable(Variable("a", -20, 0)%10, 0, 9, "(a%10)")
    self.helper_test_variable(Variable("a", -20, 1)%10, 0, 9, "(a%10)")
    self.helper_test_variable(Variable("a", 0, 20)%10, 0, 9, "(a%10)")
    self.helper_test_variable(Variable("a", -1, 20)%10, 0, 9, "(a%10)")

  def test_ge_remove(self):
    self.helper_test_variable(Variable("a", 0, 6) >= 25, 0, 0, "False")

  def test_lt_remove(self):
    self.helper_test_variable(Variable("a", 0, 6) < -3, 0, 0, "False")
    self.helper_test_variable(Variable("a", 0, 6) < 3, 0, 1, "(a<3)")
    self.helper_test_variable(Variable("a", 0, 6) < 8, 1, 1, "True")

  def test_cast_bool(self):
    self.helper_test_variable(Variable("a", 0, 10).cast(dtypes.bool), 0, 1, "a!=0")

  def test_lt_sum_remove(self):
    self.helper_test_variable(Variable("a", 0, 6) + 2 < 3, 0, 1, "(a<1)")

  def test_lt_simple_factor(self):
    self.helper_test_variable((Variable("a", 0, 6)*6+Variable("b", 0, 6)*6) < 8, 0, 1, "(((a*3)+(b*3))<4)")

  def test_lt_sum_factor_rhs_partial(self):
    self.helper_test_variable((Variable("a", 0, 6)*6 + Variable("b", 0, 6)*4 + Variable("c", 0, 6)*8) < 4, 0, 1,
                              "((((a*3)+(b*2))+(c*4))<2)")

  def test_lt_sum_factor_rhs_all(self):
    self.helper_test_variable((Variable("a", 0, 6)*6 + Variable("b", 0, 6)*4 + Variable("c", 0, 6)*8) < 2, 0, 1,
                              "((((a*3)+(b*2))+(c*4))<1)")

  def test_and_fold(self):
    self.helper_test_variable(uand([uconst(0), Variable("a", 0, 1)]), 0, 0, "0")

  def test_and_remove(self):
    self.helper_test_variable(uand([uconst(1), Variable("a", 0, 1)]), 0, 1, "a")

  def test_masked_shr_fold(self):
    x = UOp.variable('x', 0, 255, dtype=dtypes.uint32)
    self.helper_test_variable((x & -4) >> 2, 0, 63, "(x>>2)", test_z3=False)

  def test_bool_or_not_tautology(self):
    a = Variable("a", 0, 10)
    c = a<10
    self.helper_test_variable(c | c.logical_not(), True, True, "True")

  def test_bool_and_not_contradiction(self):
    a = Variable("a", 0, 10)
    c = a<10
    self.helper_test_variable(c & c.logical_not(), False, False, "False")

  def test_mod_factor_negative(self):
    self.helper_test_variable(usum([uconst(-29), Variable("a", 0, 10), Variable("b", 0, 10)*28]) % 28, 0, 27, "((a+b*28+-29)%28)")
    self.helper_test_variable(usum([uconst(-29), Variable("a", 0, 100), Variable("b", 0, 10)*28]) % 28, 0, 27, "((a+b*28+-29)%28)")

  def test_sum_combine_num(self):
    self.helper_test_variable(usum([uconst(29), Variable("a", 0, 10), uconst(-23)]), 6, 16, "(a+6)")

  def test_sum_num_hoisted_and_factors_cancel_out(self):
    self.helper_test_variable(usum([Variable("a", 0, 1) * -4 + 1, Variable("a", 0, 1) * 4]), 1, 1, "1")

  def test_div_cancel(self):
    self.helper_test_variable(usum([uconst(-40), Variable("a", 0, 10)*2, Variable("b", 0, 10)*40])//40, -1, 9, "(b+-1)")

  def test_mod_cancel(self):
    self.helper_test_variable(usum([uconst(-40), Variable("a", 0, 10)*2, Variable("b", 0, 10)*40]) % 40, 0, 20, "(a*2)")

  def test_mul_div(self):
    self.helper_test_variable((Variable("a", 0, 10)*4)//4, 0, 10, "a")

  def test_div_drop_small_terms(self):
    # from openpilot, shouldnt simplify
    gidx0 = UOp.variable("gidx0", 0, 10)
    gidx1 = UOp.variable("gidx1", 0, 10)
    lidx0 = UOp.variable("lidx0", 0, 1)
    lidx1 = UOp.variable("lidx1", 0, 1)
    ridx1005 = UOp.variable("ridx1005", 0, 2)
    ridx1006 = UOp.variable("ridx1006", 0, 2)
    self.helper_test_variable((lidx1+((gidx1*18)+(ridx1005*18)+(lidx0*162))+(gidx0*2)+(ridx1006*2)+-40)//18, -3, 20,
      "(gidx1+ridx1005+lidx0*9+(gidx0+ridx1006+7)//9+-3)")

  def test_add_div(self):
    # careful about the lower bounds and upper bounds
    self.helper_test_variable((Variable("a", 0, 5)-2)//4, -1, 0, "((a+2)//4+-1)")
    self.helper_test_variable((Variable("a", 0, 5)-1)//4, -1, 1, "((a+3)//4+-1)")
    self.helper_test_variable((Variable("a", 0, 5))//4, 0, 1, "(a//4)")
    self.helper_test_variable((Variable("a", 0, 5)+1)//4, 0, 1, "((a+1)//4)")
    self.helper_test_variable((Variable("a", 0, 5)+2)//4, 0, 1, "((a+2)//4)")
    self.helper_test_variable((Variable("a", 0, 5)+3)//4, 0, 2, "((a+3)//4)")
    self.helper_test_variable((Variable("a", 0, 5)+4)//4, 1, 2, "(a//4+1)")
    self.helper_test_variable((Variable("a", 0, 5)+5)//4, 1, 2, "((a+1)//4+1)")

  def test_div_neg_rem(self):
    self.helper_test_variable((-Variable("a", 0, 255)+256)//2, 0, 128, "(a*-1//2+128)")

  def test_mul_div_factor_mul(self):
    self.helper_test_variable((Variable("a", 0, 10)*8)//4, 0, 20, "(a*2)")

  def test_mul_div_factor_mul_neg(self):
    self.helper_test_variable((Variable("a", 0, 10)*-8+16)//4, -16, 4, "((a*-2)+4)")

  def test_mul_div_factor_div(self):
    self.helper_test_variable((Variable("a", 0, 10)*4)//8, 0, 5, "(a//2)")

  def test_mul_div_factor_div_neg(self):
    self.helper_test_variable((Variable("a", 0, 10)*-4+4)//8, -5, 0, "((a*-1+1)//2)")

  def test_div_symbolic_const_gcd(self):
    a = Variable("a", -10, 10)
    b = Variable("b", -10, 10)
    d = Variable("d", 1, 10)
    self.helper_test_variable((3*a+9*b)//(3*d), -40, 40, "((a+(b*3))//d)")

  def test_symbolic_gcd_div(self):
    a = Variable("a", -10, 10)
    b = Variable("b", -10, 10)
    c = Variable("c", -10, 10)
    d1 = Variable("d1", 1, 10)
    d2 = Variable("d2", -10, -1)
    self.helper_test_variable((d1*a*b*d1)//(d1), -1000, 1000, "(a*(b*d1))", test_z3=False)
    self.helper_test_variable((d1*a*d2*b*d1)//(d1*d2),  -1000, 1000, "(a*(b*d1))", test_z3=False)
    self.helper_test_variable((d1*a + b*d1)//(d1), -20, 20, "(a+b)", test_z3=False)
    self.helper_test_variable((d1*a + b*d1 + c*d1)//(d1), -30, 30, "(c+(a+b))", test_z3=False)
    self.helper_test_variable((3*a*d1 + 9*b*d1)//(3*d1*d2), -40, 40, "((a+b*3)//d2)", test_z3=False)
    self.helper_test_variable((3*a*d1 + 9*b*d1+3)//(3*d1*d2), -401, 399, "((a*d1+b*d1*3+1)//(d1*d2))", test_z3=False)

  def test_symbolic_factor_remainder_div(self):
    a = Variable("a", 0, 10)
    b = Variable("b", 0, 10)
    d = Variable("d", 1, 10)
    self.helper_test_variable((d*a+b)//d, 0, 20, "(a+(b//d))")
    self.helper_test_variable((d*a*20+b)//(5*d), 0, 42, "((a*4)+(b//(d*5)))")
    self.helper_test_variable((d*a*20+b*d*5+10)//(5*d), 0, 52, "((b+(a*4))+(2//d))")

  def test_mod_gcd_factor_neg(self):
    self.helper_test_variable((Variable("a", 0, 10)*-4+4)%8, 0, 4, "((a*-1+1)%2*4)")

  def test_mod_gcd_fold_neg(self):
    self.helper_test_variable((Variable("a", 0, 10)*-8+20)%4, 0, 0, "0")

  def test_sum_div_partial_remove(self):
    self.helper_test_variable(usum([Variable("idx0", 0, 127)*4, Variable("idx2", 0, 3)])//4, 0, 127, "idx0")

  def test_floordiv_const_evaluation(self):
    self.helper_test_variable((Variable("a", 0, 2)-12)//8, -2, -2, "-2")
    self.helper_test_variable((-Variable("a", 0, 2))//7, -1, 0, "(a*-1//7)")

  def test_floormod_const_evaluation(self):
    self.helper_test_variable((Variable("a", 1, 1)*-3)%8, 5, 5, "5")
    self.helper_test_variable((-Variable("a", 10, 10))%7, 4, 4, "4")

  def test_div_numerator_negative(self):
    self.helper_test_variable((Variable("idx", 0, 9)*-10)//11, -9, 0, "(idx*-1)")

  def test_nest_div_negative_factor(self):
    ridx0=Variable("ridx0", 0, 9)
    ridx1=Variable("ridx1", 0, 6)
    self.helper_test_variable(((((ridx0*-7)+ridx1)+63)//35), 0, 1, "((ridx0*-1+4)//5+1)")

  def test_floordiv_factor_nest_negative_numerator(self):
    # x//c = (x//f)//(c//f) for f|c, any sign of x
    a = Variable("a", -10, 10)
    b = Variable("b", 0, 3)
    self.helper_test_variable((a*4 + b)//12, -4, 3, "(a//3)")

  def test_floordiv_gcd_with_remainder_negative_numerator(self):
    # factor gcd from numerator, even when x crosses zero, as long as the shifted numerator stays nonneg
    a = Variable("a", -1, 5)
    self.helper_test_variable((a*2 + 7)//8, 0, 2, "((a+3)//4)")

  def test_div_into_mod(self):
    self.helper_test_variable((Variable("idx", 0, 16)*4)%8//4, 0, 1, "(idx%2)")

  def test_mod_div_reorder(self):
    # (x % (a*b)) // a -> (x // a) % b, enables div-mod recombine
    x = Variable("x", 0, 23)
    self.helper_test_variable(x % 6 // 3, 0, 1, "(x//3%2)")
    self.helper_test_variable(x % 12 // 4, 0, 2, "(x//4%3)")
    self.helper_test_variable(x%12//4*4 + x%4 + x//12*12, 0, 23, "x")

  def test_div_neg_cancel(self):
    self.helper_test_variable((-Variable("idx", 0, 100)+199)//-4 + 50, 0, 25, "((idx*-1+199)//-4+50)")
    self.helper_test_variable((-Variable("idx", 0, 100)+200)//-4 + 50, 0, 25, "((idx*-1+200)//-4+50)")
    self.helper_test_variable((-Variable("idx", 0, 100)+201)//-4 + 50, -1, 24, "((idx*-1+201)//-4+50)")
    self.helper_test_variable((-Variable("idx", 0, 100))//2, -50, 0, "(idx*-1//2)")
    self.helper_test_variable(Variable("idx", 0, 100)//-2, -50, 0, "(idx//-2)")

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
                              "((((((gidx0//5)+lidx2)//5)+lidx1)//2)+(((gidx2*32)+(gidx1*8))+(lidx0*16)))")

  def test_sum_div_complex2(self):
    gidx0 = Variable("gidx0", 0, 7)
    lidx2 = Variable("lidx2", 0, 1)
    lidx3 = Variable("lidx3", 0, 1)
    self.helper_test_variable((gidx0*4+lidx2*2+1)//10, 0, 3, "(((gidx0*2)+lidx2)//5)")
    self.helper_test_variable((gidx0*4+lidx2*2+lidx3)//10, 0, 3, "(((gidx0*2)+lidx2)//5)")
    self.helper_test_variable((gidx0*2+lidx2)//10, 0, 1, "(gidx0//5)")

  def test_sum_div_complex3(self):
    gidx0 = Variable("gidx0", 0, 7)
    lidx2 = Variable("lidx2", 0, 12)
    lidx3 = Variable("lidx3", 0, 1)
    self.helper_test_variable((gidx0*4+lidx2*2+lidx3)//12, 0, 4, "(((lidx2//2)+gidx0)//3)")
    self.helper_test_variable((lidx2*2+gidx0*4+lidx3)//12, 0, 4, "(((lidx2//2)+gidx0)//3)")

  def test_sum_div_complex4(self):
    gidx0 = Variable("gidx0", 0, 2)
    lidx2 = Variable("lidx2", 0, 12)
    lidx3 = Variable("lidx3", 0, 12)
    self.helper_test_variable((gidx0*3+lidx2*19+lidx3*38)//(3*19), 0, 12, "((lidx2+(lidx3*2))//3)")

  def test_sum_mul_distribute(self):
    gidx0 = Variable("gidx0", 0, 7)
    lidx2 = Variable("lidx2", 0, 12)
    lidx3 = Variable("lidx3", 0, 1)
    self.helper_test_variable((gidx0+lidx2+lidx3)*4, 0, 80, "(((gidx0*4)+(lidx2*4))+(lidx3*4))")

  @unittest.expectedFailure
  def test_variable_divmod(self):
    start_pos = Variable("start_pos", 0, 127)
    v = start_pos + 1
    idx0 = Variable("idx0", 0, 2)
    idx1 = Variable("idx1", 0, start_pos)
    self.helper_test_variable((idx0*v+idx1)//v, 0, 2, "(idx0)")
    self.helper_test_variable((idx0*v+idx1)%v, 0, start_pos, "idx1")

  def test_mod_variable_denom_factor_remainder(self):
    d = Variable("d", 2, 5)
    a = Variable("a", 0, 3)
    b = Variable("b", 0, 1)
    self.helper_test_variable((d*a+b)%d, 0, 1, "b")

  def test_divmod_variable_denom_fold_to_const(self):
    x = Variable("x", 20, 23)
    y = Variable("y", 8, 10)
    self.helper_test_variable(x//y, 2, 2, "2")
    self.helper_test_variable(x%y, 0, 7, "(x+(y*-2))")
    # ensure all 4 corners are checked
    x = Variable("x", -10, 10, dtypes.int)
    y = Variable("y", -8, 9, dtypes.int)
    self.helper_test_variable(x//y, -2147483648, 2147483647, "(x//y)")
    self.helper_test_variable(x%y, -2147483648, 2147483647, "(x%y)")

  def test_div_neg_all_range(self):
    gidx = Variable("gidx", 0, 124)
    lidx = Variable("lidx", 0, 7)
    self.helper_test_variable((-gidx*8-lidx+999)//-4 + 250, 0, 250, "((gidx*-8+lidx*-1+999)//-4+250)")
    self.helper_test_variable((-gidx*8-lidx+1000)//-4 + 250, 0, 249, "((gidx*-8+lidx*-1+1000)//-4+250)")
    self.helper_test_variable((-gidx*8-lidx+1001)//-4 + 250, -1, 249, "((gidx*-8+lidx*-1+1001)//-4+250)")
    self.helper_test_variable((-gidx*8-lidx+1002)//-4 + 250, -1, 249, "((gidx*-8+lidx*-1+1002)//-4+250)")

  def test_div_neg_then_neg(self):
    # taken from arange opts
    lidx0 = Variable("lidx0", 0, 7)
    lidx1 = Variable("lidx1", 0, 7)
    alu2 = -lidx0-lidx1
    self.helper_test_variable((((alu2+14)//(-32))+4), 3, 4, "((lidx0*-1+lidx1*-1+14)//-32+4)")
    self.helper_test_variable(-(((alu2+14)//(-32))+4), -4, -3, "((lidx0*-1+lidx1*-1+14)//-32*-1+-4)")
    self.helper_test_variable((((alu2+134)//(-32))+4), -1, 0, "((lidx0*-1+lidx1*-1+134)//-32+4)")
    self.helper_test_variable((((alu2+142)//(-32))+4), -1, 0, "((lidx0*-1+lidx1*-1+142)//-32+4)")
    self.helper_test_variable((((alu2+150)//(-32))+4), -1, -1, "-1")
    self.helper_test_variable((((alu2+158)//(-32))+4), -1, -1, "-1")

  def test_div_mod_recombine(self):
    gidx = Variable("gidx", 0, 124)
    lidx = Variable("lidx", 0, 124)
    self.helper_test_variable(gidx%4+(gidx//4)*4, 0, 124, "gidx")
    self.helper_test_variable((gidx//4)*4+gidx%4, 0, 124, "gidx")
    self.helper_test_variable(lidx+gidx%4+(gidx//4)*4, 0, 248, "(gidx+lidx)")
    self.helper_test_variable(lidx+(gidx//4)*4+gidx%4, 0, 248, "(gidx+lidx)")
    self.helper_test_variable(lidx+(gidx//4)*8+2*(gidx%4), 0, 372, "(lidx+(gidx*2))")
    self.helper_test_variable(lidx+2*(gidx%4)+(gidx//4)*8, 0, 372, "(lidx+(gidx*2))")

  def test_div_mod_recombine_partial(self):
    gidx = Variable("gidx", 0, 15)
    self.helper_test_variable((gidx//2)%4+(gidx//8)*4, 0, 7, "gidx//2")

  def test_div_mod_recombine_folded_mod(self):
    a = Variable("a", 0, 2)
    b = Variable("b", 0, 100)
    self.helper_test_variable((31 * a + 1) % 30 + ((31 * a + 1) // 30) * 30, 1, 63, "((a*31)+1)")
    self.helper_test_variable((31 * b + 1) % 18 + ((31 * b + 1) // 18) * 18, 1, 3101, "((b*31)+1)")

  def test_div_mod_recombine_3level(self):
    gidx = Variable("gidx", 0, 150527)
    self.helper_test_variable(gidx//3%224*3 + gidx%3 + gidx//672*672, 0, 150527, "gidx")
    # different shapes
    x = Variable("x", 0, 5*7*11-1)
    self.helper_test_variable(x//11%7*11 + x%11 + x//77*77, 0, 5*7*11-1, "x")
    # result is x//a*c2 not just x
    x2 = Variable("x2", 0, 5*6*7-1)
    self.helper_test_variable(x2//7%6*14 + x2//42*84, 0, (5*6*7-1)//7*14, "(x2//7*14)")
    # negative variable range
    xn = Variable("x", -1000, 1000)
    self.helper_test_variable(xn//3%224*3 + xn%3 + xn//672*672, -1000, 1000, "x")
    self.helper_test_variable(xn//3%7*3 + xn//21*21, -1002, 999, "(x//3*3)")
    # should NOT simplify: a*c1 != b (3*224 != 600)
    self.helper_test_variable(gidx//3%224*3 + gidx//600*600, 0, 150669, "(gidx//600*600+gidx//3%224*3)")
    # should NOT simplify: c1*c2 != c3 (224*3 != 700)
    self.helper_test_variable(gidx//3%224*3 + gidx//672*700, 0, 156769, "(gidx//672*700+gidx//3%224*3)")

  def test_div_mod_recombine_with_gcd(self):
    b = Variable("b", 0, 100)
    exp = (16 * b + 2) % 18 + ((16 * b + 2) // 18) * 18
    self.helper_test_variable(exp, 2, 1602, "((b*16)+2)")
    self.helper_test_variable((30 * b + 1) % 18 + ((30 * b + 1) // 18) * 18, 1, 3001, "((b*30)+1)")

  def test_div_partial_quotient(self):
    # FLOORDIV should extract partial quotients when const_factor > divisor, matching what FLOORMOD already does
    # (f*x+c)//d -> (f%d*x+c)//d + (f//d)*x  when f >= d
    b = Variable("b", 0, 100)
    self.helper_test_variable((31*b+1)//18, 0, 172, "(((b*13)+1)//18+b)")
    self.helper_test_variable((19*b+3)//7, 0, 271, "(((b*5)+3)//7+(b*2))")

  def test_gcd_with_remainder(self):
    # gcd_with_remainder: factor GCD out of non-constant terms and denominator
    a = Variable("a", 0, 2)
    self.helper_test_variable((a*4)//6, 0, 1, "(a*2//3)")
    self.helper_test_variable((a*4+1)//6, 0, 1, "(a*2//3)")
    self.helper_test_variable((a*4+2)//6, 0, 1, "((a*2+1)//3)")
    self.helper_test_variable((a*4+3)//6, 0, 1, "((a*2+1)//3)")
    self.helper_test_variable((a*4)%6, 0, 4, "(a*2%3*2)")
    self.helper_test_variable((a*4+1)%6, 1, 5, "(a*2%3*2+1)")
    self.helper_test_variable((a*4+2)%6, 0, 4, "((a*2+1)%3*2)")
    self.helper_test_variable((a*4+3)%6, 1, 5, "((a*2+1)%3*2+1)")

  def test_div_by_factor_tie_break(self):
    a = Variable("a", 0, 1)
    b = Variable("b", 0, 1)
    self.helper_test_variable((a*2+b*3+2)//6, 0, 1, "((a+b+1)//3)")

  def test_div_mod_recombine_large_coeff(self):
    # recombine must work even when coeff > divisor: both mod and div reduce the coeff the same way
    b = Variable("b", 0, 100)
    self.helper_test_variable((19*b+3)%7 + ((19*b+3)//7)*7, 3, 1903, "((b*19)+3)")
    a = Variable("a", 0, 10)
    self.helper_test_variable((25*a+3)%10 + ((25*a+3)//10)*10, 3, 253, "((a*25)+3)")

  def test_div_mod_recombine_negative_div_unsound(self):
    # ((b//d)%div)*mul + (b//(d*div))*(div*mul) only equals (b//d)*mul when div>0
    b = Variable("b", -100, 100)
    self.helper_test_variable(((b//(-3))%(-2)) + (b//6)*(-2), -33, 34, "(b//6*-2+b//-3%-2)")

  def test_mod_recombine_with_outer_mul(self):
    # ((x//c)%d)*(c*mul) + (x%c)*mul -> (x%(c*d))*mul
    x = Variable("x", 0, 100)
    self.helper_test_variable((x%4)*3 + ((x//4)%2)*12, 0, 21, "(x%8*3)")
    y = Variable("y", 0, 5*7*11-1)
    self.helper_test_variable((y%11)*5 + ((y//11)%7)*55, 0, 380, "(y%77*5)")
    # negative mul
    self.helper_test_variable((x%4)*-2 + ((x//4)%2)*-8, -14, 0, "(x%8*-2)")

  def test_mod_nest_by_factor(self):
    # (a*f+b) % (f*k) = (a%k)*f + b when 0<=b<f — mirrors nest_div_by_factor for FLOORMOD
    gidx0 = Variable("gidx0", 0, 15)
    lidx0 = Variable("lidx0", 0, 3)
    # f=4, k=2, c=8: (gidx0*4+lidx0)%8 = (gidx0%2)*4 + lidx0
    self.helper_test_variable((gidx0*4+lidx0)%8, 0, 7, "(lidx0+gidx0%2*4)")
    # f=2, k=4: (gidx0*2+lidx0)%8 where lidx0 in [0,1]
    lidx1 = Variable("lidx1", 0, 1)
    self.helper_test_variable((gidx0*2+lidx1)%8, 0, 7, "(lidx1+gidx0%4*2)")
    # f=3, k=3: (a*3+b)%9 where b in [0,2]
    a = Variable("a", 0, 10)
    b = Variable("b", 0, 2)
    self.helper_test_variable((a*3+b)%9, 0, 8, "(b+a%3*3)")

  def test_mod_nest_by_factor_with_const(self):
    # nest_by_factor FLOORMOD with non-zero constant offset: (a*f+b+const) % (f*k) = (a%k)*f + b + const when 0<=b+const<f
    a = Variable("a", 0, 7)
    b = Variable("b", 0, 1)
    # f=4, k=2, const=2: (a*4+b+2)%8 = (a%2)*4 + b + 2
    self.helper_test_variable((a*4+b+2)%8, 2, 7, "(b+a%2*4+2)")
    # f=6, k=2, const=3: (a*6+b+3)%12 = (a%2)*6 + b + 3
    b2 = Variable("b", 0, 2)
    self.helper_test_variable((a*6+b2+3)%12, 3, 11, "(b+a%2*6+3)")
    # f=3, k=2, const=1: (a*3+b+1)%6 = (a%2)*3 + b + 1
    self.helper_test_variable((a*3+b+1)%6, 1, 5, "(b+a%2*3+1)")

  def test_div_nest_by_factor_with_const(self):
    # nest_by_factor FLOORDIV: (160*a + 5*b + 4*c + K) // 60 should pick div=5 (clean) over div=4 (dirty)
    a = Variable("a", 0, 2)
    b = Variable("b", 0, 31)
    c = Variable("c", 0, 1)
    self.helper_test_variable((160*a + 5*b + 4*c) // 60, 0, 7, "(a*2+(b+a*8)//12)")
    self.helper_test_variable((160*a + 5*b + 4*c + 1) // 60, 0, 8, "(a*2+(b+c+a*8)//12)")
    self.helper_test_variable((160*a + 5*b + 4*c + 2) // 60, 0, 8, "(a*2+(b+c+a*8)//12)")
    self.helper_test_variable((160*a + 5*b + 4*c + 3) // 60, 0, 8, "(a*2+(b+c+a*8)//12)")
    self.helper_test_variable((160*a + 5*b + 4*c + 59) // 60, 0, 8, "(a*2+(b+c+a*8+11)//12)")

  def test_div_mod_recombine_after_nesting(self):
    # when nest_div_by_factor simplifies the div, the mod must also nest so recombine can fire
    gidx0 = Variable("gidx0", 0, 15)
    lidx0 = Variable("lidx0", 0, 3)
    x = gidx0*4+lidx0
    # div nests: x//8 -> gidx0//2, mod nests: x%8 -> (gidx0%2)*4+lidx0, then recombine gives x back
    self.helper_test_variable((x//8)*8 + x%8, 0, 63, "(lidx0+gidx0*4)")
    # with a scaling factor: recombine gives x*2
    self.helper_test_variable((x//8)*16 + (x%8)*2, 0, 126, "(gidx0*8+lidx0*2)")
    # two variables with different factors
    a = Variable("a", 0, 7)
    b = Variable("b", 0, 1)
    y = a*6+b
    # div nests: y//12 -> a//2, mod nests: y%12 -> (a%2)*6+b, recombine
    self.helper_test_variable((y//12)*12 + y%12, 0, 43, "(b+a*6)")

  def test_div_mod_recombine_after_asymmetric_fold(self):
    a = Variable("a", 0, 7)
    b = Variable("b", 0, 14)
    x = a*15+b
    # TODO: expected "(b+a*15)"
    self.helper_test_variable((x//10)*10 + x%10, 0, 119, "(a*10+(a+b//5)//2*10+(b+a*5)%10)")
    self.helper_test_variable((x//10)*2 + (x//5)%2, 0, 23, "(a*3+b//5)")

  def test_div_mod_recombine_in_additive_sum(self):
    x = Variable("x", 0, 31)
    y = Variable("y", 0, 5)
    # recombine should work inside larger additive sums, not just in the two special y+... tree shapes
    self.helper_test_variable((x//8)*4 + y + (x//2)%4, 0, 20, "(y+x//2)")
    self.helper_test_variable(y + (x//8)*4 + (x//2)%4, 0, 20, "(y+x//2)")

  def test_div_mod_recompose_low_order_remainder(self):
    x = Variable("x", 0, 127)
    self.helper_test_variable((x//2)%4*2 + x%2, 0, 7, "(x%8)")

  def test_reshape_index_roundtrip(self):
    # simulate reshape index decompose then recompose — the core pattern this enables
    # (8,8) decomposed for (16,4): combined=r0*8+r1, div and mod by 4
    r0 = Variable("r0", 0, 7)
    r1 = Variable("r1", 0, 7)
    combined = r0*8+r1
    src_idx = (combined//4)*4 + combined%4
    self.helper_test_variable(src_idx, 0, 63, "(r1+r0*8)")

  def test_gated_load(self):
    idx = Variable("idx", 0, 24)
    self.helper_test_variable(idx//4, 0, 6, "(idx//4)")
    # TODO: simplify the true branch
    self.helper_test_variable((idx<4).where(idx//4, idx.const_like(-1)), -1, 6, "(idx<4).where((idx//4), -1)")

  def test_floordiv_lt(self):
    # x//d<c <=> x<c*d for d>0
    idx = Variable("idx", 0, 24)
    self.helper_test_variable((idx//4<3), 0, 1, "(idx<12)")
    self.helper_test_variable(((idx-20)//4<-3), 0, 1, "(idx<8)")
    self.helper_test_variable(((idx-10)//4<0), 0, 1, "(idx<10)")
    self.helper_test_variable((idx//-4<-3), 0, 1, "((idx//-4)<-3)")

  def test_nested_div_mod_negative_inner_divisor(self):
    # (x % (k*c)) // c -> (x // c) % k requires k>0; (x % (k*c)) % c -> x % c is unconditional for c>0
    a = Variable("a", 0, 100)
    self.helper_test_variable((a % -8) // 2, -4, 0, "(a%-8//2)")
    self.helper_test_variable((a % -8) % 2, 0, 1, "(a%2)")

  def test_floordiv_lt_negative_c(self):
    # x//d<c with negative c also reduces to x<c*d for d>0
    idx = Variable("idx", -20, 20)
    self.helper_test_variable((idx//4 < 0), 0, 1, "(idx<0)")
    self.helper_test_variable((idx//4 < -1), 0, 1, "(idx<-4)")
    self.helper_test_variable((idx//4 < -2), 0, 1, "(idx<-8)")

  def test_simplex_lt(self):
    a = Variable("a", 0, 3)
    b = Variable("b", 0, 3)
    c = Variable("c", 0, 3)
    d = Variable("d", -3, 3)
    self.helper_test_variable((a<2), 0, 1, "(a<2)")
    self.helper_test_variable((a<=2), 0, 1, "((2<a)!=True)")
    self.helper_test_variable((a>1), 0, 1, "(1<a)")
    self.helper_test_variable((a>=1), 0, 1, "((a<1)!=True)")
    self.helper_test_variable((a<1).ne(True), 0, 1, "((a<1)!=True)")
    self.helper_test_variable((a+b<1).ne(True), 0, 1, "(((a+b)<1)!=True)")
    self.helper_test_variable((a*3+b*4<1).ne(True), 0, 1, "(((a+b)<1)!=True)")
    self.helper_test_variable((a*(-3)+b*4<1).ne(True), 0, 1, "((((a*-3)+(b*4))<1)!=True)")  # negative coeff, should not be simplified
    self.helper_test_variable((a*3+d*4<1).ne(True), 0, 1, "((((a*3)+(d*4))<1)!=True)")  # var can be negative, should not be simplified
    self.helper_test_variable((a+b+c*2<1).ne(True), 0, 1, "((((a+b)+c)<1)!=True)")
    self.helper_test_variable((a+b*2+c*4<1).ne(True), 0, 1, "((((a+b)+c)<1)!=True)")

  def test_cast_bool_to_int_ne_const(self):
    cond = Variable("a", 0, 3) < 2
    # CAST(bool -> int) != 0  ->  cond
    self.helper_test_variable(cond.cast(dtypes.int).ne(0), 0, 1, "(a<2)")
    # CAST(bool -> int) != 1  ->  !cond
    self.helper_test_variable(cond.cast(dtypes.int).ne(1), 0, 1, "((a<2)!=True)")
    # CAST(bool -> int) != c (c not in {0,1})  ->  always True (CAST is 0 or 1)
    self.helper_test_variable(cond.cast(dtypes.int).ne(2), 1, 1, "True")
    self.helper_test_variable(cond.cast(dtypes.int).ne(-1), 1, 1, "True")
    # CAST(bool -> weakint) folds too
    self.helper_test_variable(cond.cast(dtypes.weakint).ne(0), 0, 1, "(a<2)")
    self.helper_test_variable(cond.cast(dtypes.weakint).ne(1), 0, 1, "((a<2)!=True)")

  def test_where_removal(self):
    cond = Variable("a", 0, 3) < 2
    u1, u0 = cond.const_like(True), cond.const_like(False)
    self.helper_test_variable(cond, 0, 1, "(a<2)")
    self.helper_test_variable(cond.where(u1, u0), 0, 1, "(a<2)")
    self.helper_test_variable(cond.where(u1, u0).where(u1, u0), 0, 1, "(a<2)")
    self.helper_test_variable(cond.where(u0, u1), 0, 1, "((a<2)!=True)")
    self.helper_test_variable(cond.where(u0, u1).where(u0, u1), 0, 1, "(a<2)")

  def test_where_combine(self):
    cond = Variable("x", 0, 3) < 2
    a = Variable("a", 0, 3)
    b = Variable("b", 0, 3)
    c = Variable("c", 0, 3)
    aa = cond.where(a, a.ufix(0))
    bb = cond.where(b, b.ufix(1))
    self.helper_test_variable(aa, 0, 3, "(x<2).where(a, 0)")
    self.helper_test_variable(bb, 0, 3, "(x<2).where(b, 1)")
    self.helper_test_variable(aa+bb, 0, 6, "(x<2).where((a+b), 1)")
    self.helper_test_variable(aa.maximum(bb), 0, 3, "(x<2).where(a.maximum(b), 1)")
    self.helper_test_variable((c+aa)+bb, 0, 9, "(c+(x<2).where((a+b), 1))")

    # not combining because it increased total ALU
    cc = cond.where(c, c+1)
    self.helper_test_variable(bb+cc, 0, 7, "((x<2).where(b, 1)+(x<2).where(c, (c+1)))")

    # not combining  # TODO: can combine if it can further simplify?
    ab = cond.where(a, b)
    ba = cond.where(b, a)
    self.helper_test_variable(ab+ba, 0, 6, "((x<2).where(a, b)+(x<2).where(b, a))")

    # not combining  # TODO: can combine if one is identity element const
    self.helper_test_variable(aa+ab, 0, 6, "((x<2).where(a, b)+(x<2).where(a, 0))")

  def test_negation_in_where(self):
    cond = Variable("x", 0, 3) < 2
    a = Variable("a", 0, 3)
    b = Variable("b", 0, 3)
    w = cond.logical_not().where(a, b)
    self.helper_test_variable(w, 0, 3, "(x<2).where(b, a)")

  def test_neg_in_comp(self):
    a = Variable("a", 0, 3)
    b = Variable("b", 0, 3)
    self.helper_test_variable(-a<-b, False, True, "(b<a)")

  def test_where_cast(self):
    s = Variable("s", 0, 3, dtypes.int)
    cond = s < 2
    a = Variable("a", 0, 3, dtypes.int)
    b = Variable("b", 0, 3, dtypes.int)
    expr = cond.where(a, b).cast(dtypes.half)

    # TODO: copied from render, render does not support cast
    glbl = UOp(Ops.PARAM, dtypes.int.ptr(), arg=0)
    uops = get_uops(UOp(Ops.STORE, dtypes.void, (glbl.index(UOp.const(dtypes.int, 0)), expr)).sink())
    rewritten_uop = [uop for uop in uops if uop.op is Ops.STORE][0].src[1]

    self.assertEqual(rewritten_uop, cond.where(a.cast(dtypes.half), b.cast(dtypes.half)))

  def test_where_merge_branches(self):
    cond1 = Variable("s", 0, 10) < 6
    cond2 = Variable("s", 0, 10) > 2
    a = Variable("a", 0, 3)
    b = Variable("b", 0, 3)
    expr = cond1.where(cond2.where(a, b), b)
    self.helper_test_variable(expr, 0, 3, "((s<6)&(2<s)).where(a, b)")

  def test_where_merge_branches2(self):
    cond1 = Variable("s", 0, 10) < 5
    cond2 = Variable("s", 0, 10) < 6
    a = Variable("a", 0, 3)
    b = Variable("b", 0, 3)
    expr = cond1.where(cond2.where(a, b), b)
    # (a if ((s<5)&(s<6)) else b) -> (a if (s<5) else b)
    self.helper_test_variable(expr, 0, 3, "(s<5).where(a, b)")

  @unittest.expectedFailure
  def test_where_closure_folding(self):
    # cond.where(t, f) where f contains cond.where(a, b) should fold the inner where to b in false branch
    x = Variable("x", 0, 10)
    cond = x < 5
    inner = cond.where(-x, x)  # in false branch (x>=5), this is just x
    outer = cond.where(inner * 2, inner + 1)  # true: -x*2, false: x+1
    # the inner where should be folded: true branch gets -x, false branch gets x
    self.helper_test_variable(outer, -20, 11, "(x<5).where((x*-2), (x+1))")

  def test_symbolic_div(self):
    # from symbolic arange
    a = Variable("a", 1, 10)
    denominator = ((a*-2)+1)
    numerator = (((((a*2)+-1)*2)+1)*a)
    self.helper_test_variable(denominator, -19, -1, "((a*-2)+1)")
    self.helper_test_variable(numerator, 3, 390, "(a*((a*4)+-1))")
    self.helper_test_variable((numerator//denominator)<=0, 1, 1, "True")

  def test_symbolic_range_doesnt_collapse(self):
    r0 = UOp.range((Variable("a", 1, 10)<5).cast(dtypes.weakint), 0)
    self.helper_test_variable(r0, 0, 0, "r0")

  def test_const_reciprocal(self):
    a = Variable("a", 1, 10, dtypes.float)
    # TODO: bounds for reciprocal
    # TODO: should z3 work?
    self.helper_test_variable(2*(2*a).reciprocal(), -math.inf, math.inf, "a.reciprocal()", test_z3=False)

  def test_trunc_noop(self):
    a = Variable("a", 1, 10, dtypes.int)
    self.helper_test_variable(a.trunc(), 1, 10, "a", test_z3=False)

  def test_do_math_in_int32(self):
    a = Variable("a", 1, 10, dtypes.int)
    b = Variable("b", 1, 10, dtypes.int)
    self.assertIn((a.cast(dtypes.long)+b.cast(dtypes.long)).render(), "(long)((a+b))")
    self.assertIn((a.cast(dtypes.long)*b.cast(dtypes.long)).render(), "(long)((a*b))")

  def test_nested_mod_negative_range(self):
    # (x%(k*c))%c = x%c for positive c
    x = Variable("x", 0, 1575)
    self.helper_test_variable(((x + (-1064)) % 512) % 4, 0, 3, "((x+-1064)%4)")
    self.helper_test_variable(((x + (-1064)) % 512) % 128, 0, 127, "((x+-1064)%128)")

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

class TestSymbolicVariables(unittest.TestCase):
  def test_simple(self):
    z = uconst(0)
    a = Variable("a", 0, 10)
    b = Variable("b", 0, 10)
    c = Variable("c", 0, 10)
    assert z.variables() == []
    assert a.variables() == [a]
    m = a * 3
    assert m.variables() == [a]
    s = usum([a, b, c])
    assert s.variables() == [a, b, c]

  def test_compound(self):
    a = Variable("a", 0, 10)
    b = Variable("b", 0, 10)
    c = Variable("c", 0, 10)
    assert (a + b * c).variables() == [a, b, c]
    assert (a % 3 + b // 5).variables() == [a, b]
    # TODO: fix me
    with self.assertRaises(AssertionError):
      assert (a + b + c - a).variables() == [b, c]

  def test_dedup(self):
    a = Variable("a", 0, 10)
    assert (a * a).variables() == [a]
    assert (a//4 + a//6).variables() == [a]

  def test_variable_min_eq_max_bind_folds(self):
    b = Variable("x", 1, 1).bind(1)
    s = b.simplify()
    self.assertEqual(s.op, Ops.CONST)
    self.assertEqual(s.arg, 1)

class TestSymInfer(unittest.TestCase):
  def test_sym_infer(self):
    a = Variable("a", 0, 10)
    b = Variable("b", 0, 10)
    c = Variable("c", 0, 10)
    var_vals = {a.expr: 2, b.expr: 3, c.expr: 4}
    assert sym_infer(5, var_vals) == 5
    assert sym_infer(4.2, var_vals) == 4.2
    assert sym_infer(a, var_vals) == 2
    assert sym_infer(b, var_vals) == 3
    assert sym_infer(a+b, var_vals) == 5
    assert sym_infer(a-b, var_vals) == -1
    assert sym_infer(a+b+c, var_vals) == 9
    assert sym_infer(a*b, var_vals) == 6
    assert sym_infer(a*b+c, var_vals) == 10
  def test_sym_infer_floordiv_floormod(self):
    a = Variable("a", -1000, 1)
    b = Variable("b", -1000, 1)
    var_vals = {a.expr: 1, b.expr: -1000}
    # floor: 1 % -1000 = -999, 1 // -1000 = -1
    assert sym_infer(a%b, var_vals) == -999
    assert sym_infer(a//b, var_vals) == -1
  def test_sym_infer_with_bitcast(self):
    a = Variable("a", 1, 10, dtypes.int)
    expr = ((a.bitcast(dtypes.uint) << UOp.const(dtypes.uint, 1)).bitcast(dtypes.int) + 2)
    ret = sym_infer(expr, {a.expr: 2})
    assert isinstance(ret, int)
    assert ret == 6

    b = Variable("b", -5, 5, dtypes.int)
    assert sym_infer(b.bitcast(dtypes.uint), {b.expr: -1}) == 0xFFFFFFFF

    c = Variable("c", 0, 0xFFFFFFFF, dtypes.uint)
    assert sym_infer(c.bitcast(dtypes.int), {c.expr: 0xFFFFFFFF}) == -1

    assert sym_infer(UOp.const(dtypes.float, 1.5).bitcast(dtypes.uint), {}) == 1069547520

  def test_sym_infer_deeply_nested(self):
    # build an expression that exceeds Python's nested parentheses limit for eval
    # max(x, negative_const) can't be simplified when x can be negative, so nesting compounds
    a = Variable("a", 1, 8192)
    b = Variable("b", 0, 8191)
    expr = a
    for _ in range(200):
      expr = (expr * (b + a)).maximum(uconst(-33554432)) * uconst(-1) + a
    result = sym_infer(expr, {"a": 1, "b": 0})
    assert isinstance(result, int)

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

class TestInvalidIndex(unittest.TestCase):
  def test_invalid_times_0(self):
    ridx = Variable("ridx", 0, 10)
    idx = (ridx<5).where(ridx, UOp.invalid())*0
    self.assertIs(idx.simplify(), (ridx<5).where(0, UOp.invalid()), "multiplying an index by 0 should preserve the invalid")

  def test_invalid_comparison_drops_invalid(self):
    # comparisons return a bool, and bools can't be invalid
    ridx = Variable("ridx", 0, 10)
    idx = (ridx<5).where(ridx, UOp.invalid())<3
    self.assertIs(idx.simplify(), (ridx<3), "comparison of index should drop the invalid")
    self.assertIs(idx.where(UOp.const(dtypes.int, 1), 0).simplify(), (ridx<3).where(UOp.const(dtypes.int, 1), 0),
      "comparison of index should drop the invalid")

  def test_alu_moves_inside_invalid(self):
    ridx = Variable("ridx", 0, 10)
    idx = (ridx<5).where(ridx, UOp.invalid())*10
    self.assertIs(idx.simplify(), (ridx<5).where(ridx*10, UOp.invalid()), "multiplying an index by 0 should preserve the invalid")

  def test_merge_invalid_conditions(self):
    ridx0 = Variable("ridx0", 0, 10)
    ridx1 = Variable("ridx1", 0, 10)
    idx0 = (ridx0<5).where(ridx0, UOp.invalid())
    idx1 = (ridx1<5).where(idx0//2, UOp.invalid())
    self.assertIs(idx1.simplify(), ((ridx1<5)&(ridx0<5)).where(ridx0//2, UOp.invalid()),
      "valid inside a valid should make a single valid and & the conditions")

  def test_alu_invalid(self):
    self.assertIs((UOp.invalid()*2).simplify(), UOp.invalid())
    self.assertIs((UOp.invalid()*0).simplify(), UOp.invalid())
    self.assertIs((UOp.invalid()+8).simplify(), UOp.invalid())
    self.assertIs((UOp.invalid()+Variable("a",0,10)).simplify(), UOp.invalid())
    self.assertIs((UOp.invalid()*Variable("a",0,10)).simplify(), UOp.invalid())
    self.assertIs((UOp.invalid()<Variable("a",0,10)).simplify().dtype, dtypes.bool)

  def test_alu_invalid_vconst(self):
    c1 = UOp.const(dtypes.weakint.vec(4), (1, 1, Invalid, Invalid))
    c2 = UOp.const(dtypes.weakint.vec(4), (1, Invalid, 1, 1))
    self.assertIs((c1+c2).simplify(), UOp.const(dtypes.weakint.vec(4), (2, Invalid, Invalid, Invalid)))

class TestStoreLoadFolding(unittest.TestCase):
  """Tests for store(index, load(index)) -> NOOP rule. This rule matches patterns that EMERGE during simplification."""
  def test_store_load_folding(self):
    # store(idx, load(idx)) -> NOOP, including emergent patterns like store(idx, load(idx) + 0)
    buf = UOp(Ops.PARAM, dtypes.int.ptr(), arg=0)
    index = buf.index(UOp.const(dtypes.weakint, 0))
    # Direct: store(idx, load(idx)) -> NOOP
    self.assertEqual(graph_rewrite(index.store(index.load()), sym).op, Ops.NOOP)
    # Emergent: store(idx, load(idx) + 0) -> store(idx, load(idx)) -> NOOP
    self.assertEqual(graph_rewrite(index.store(index.load() + UOp.const(dtypes.int, 0)), sym).op, Ops.NOOP)
    # Emergent: store(idx, load(idx) * 1) -> store(idx, load(idx)) -> NOOP
    self.assertEqual(graph_rewrite(index.store(index.load() * UOp.const(dtypes.int, 1)), sym).op, Ops.NOOP)
    # Negative: store(idx, load(idx) + 1) should NOT fold
    self.assertEqual(graph_rewrite(index.store(index.load() + UOp.const(dtypes.int, 1)), sym).op, Ops.STORE)

class TestMoveWhereOnLoad(unittest.TestCase):
  def test_bool_index_preserves_dtype(self):
    buf = UOp.param(0, dtypes.bool.ptr(8))
    a = Variable("a", 0, 7)
    r = UOp.range(8, 0)
    # cond has a range that the rewrite can move into the valid: gate (a<4) goes into load valid
    cond = (a < 4) & (r < 2)
    valid = (a < 2)  # pre-existing valid on the load (to pass can_move check for the r-only clause)
    idx = buf.index(a.valid(valid), ptr=True)
    expr = cond.where(idx, 0)
    out = graph_rewrite(expr, pm_move_where_on_load)
    # any WHERE in the rewritten graph must have matched-dtype branches
    for u in out.toposort():
      if u.op is Ops.WHERE:
        self.assertEqual(u.dtype, u.src[1].dtype, f"WHERE branch 1 dtype mismatch: {u}")
        self.assertEqual(u.dtype, u.src[2].dtype, f"WHERE branch 2 dtype mismatch: {u}")

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
      ("(lidx3+((lidx5+1)//16*802816+(lidx5+1)%16*49+gidx0*3211264+gidx1*784+gidx2*8+lidx4*100352)+2207744)",))

class TestGatedUopGivenValid(unittest.TestCase):
  def test_invalid_gate_simplifies_index(self):
    r0 = Variable("r0", 0, 2)

    idx:UOp = (r0 < 3).where((r0 + uconst(-1)) // uconst(3), UOp.invalid())
    idx = graph_rewrite(idx, pm_simplify_valid)
    # (r0-1)//3 = (r0+2)//3 - 1 (constant offset split)
    self.assertEqual(idx, (r0 < 3).where((r0 + uconst(2)) // uconst(3) + uconst(-1), UOp.invalid()))

  def test_invalid_gate_simplifies_vectorize(self):
    r0 = Variable("r0", 0, 2)

    idx0 = (r0 + uconst(-1)) // uconst(3)
    idx1 = r0 % uconst(3)
    idx:UOp = (r0 < 3).where(UOp(Ops.STACK, dtypes.weakint.vec(2), (idx0, idx1)), UOp.invalid())
    idx = graph_rewrite(idx, pm_simplify_valid)
    # independent simplification: (r0-1)//3 -> (r0+2)//3 - 1, and r0%3 -> r0 when r0 in [0,2]
    expected_vec = UOp(Ops.STACK, dtypes.weakint.vec(2), ((r0 + uconst(2)) // uconst(3) + uconst(-1), r0))
    self.assertEqual(idx, (r0 < 3).where(expected_vec, UOp.invalid()))

class TestRangeSplitting(unittest.TestCase):
  def test_range_split_on_mod(self):
    # test that mark_range_mod splits RANGE(8) into RANGE(4)*2 + RANGE(2) when used with %2
    from tinygrad.codegen.simplify import pm_split_ranges, pm_flatten_range
    r0 = UOp.range(uconst(8), 0)
    # create a simple expression using the range with mod: store range%2 to a buffer
    buf = UOp(Ops.PARAM, dtypes.int.ptr(), arg=0)
    val = (r0 % uconst(2)).cast(dtypes.int)
    store = UOp(Ops.STORE, dtypes.void, (buf.index(uconst(0)), val))
    sink = UOp(Ops.SINK, dtypes.void, (UOp(Ops.END, dtypes.void, (store, r0)),))
    # count RANGEs before
    ranges_before = len([u for u in sink.toposort() if u.op is Ops.RANGE])
    # apply the range splitting optimization
    sink_after = graph_rewrite(sink, pm_split_ranges+pm_flatten_range, ctx={}, name="test split ranges")
    # count RANGEs after - should have more due to splitting
    ranges_after = len([u for u in sink_after.toposort() if u.op is Ops.RANGE])
    self.assertGreater(ranges_after, ranges_before, "RANGE should be split when used with mod of divisible constant")

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
    assert ((alu0+2559)//-4).vmin == -640 and ((alu0+2559)//-4).vmax == 0
    assert (((alu0+2559)//-4)*(-1)).vmin == 0 and (((alu0+2559)//-4)*(-1)).vmax == 640

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
