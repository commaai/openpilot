import unittest
from tinygrad import Variable, dtypes
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops, UOp


class TestFuzzFailure(unittest.TestCase):
  def setUp(self):
    self.context = Context(CORRECT_DIVMOD_FOLDING=1)
    self.context.__enter__()

  def tearDown(self):
    self.context.__exit__(None, None, None)

  def test_fuzz_failure1(self):
    v1=Variable('v1', 0, 8)
    v2=Variable('v2', 0, 2)
    v3=Variable('v3', 0, 1)
    expr = (((((((((((((((((((((((0//4)%2)//8)+-2)+-4)+-3)+v1)+-4)+v2)+-2)+v3)+v2)//3)%7)*1)//2)+v2)*-1)+2)+1)+0)+-3)+v3)
    v1_val, v2_val, v3_val = v1.const_like(8), v2.const_like(0), v3.const_like(0)
    num = expr.simplify().substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    rn = expr.substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    self.assertEqual(num, rn)

  def test_fuzz_failure2(self):
    v1=Variable('v1', 0, 16)
    v2=Variable('v2', 0, 5)
    v3=Variable('v3', 0, 3)
    expr = (((((((((((((((((((((((((0*4)//5)*2)*-1)*-2)+-4)*4)*2)*3)*4)+-4)*4)+v2)+v2)+v3)//3)+v2)+v1)//9)+3)+1)//1)+-4)//4)*2)
    expr = (((((v1+(v2+(((v3+(v2*2))+1)//3)))+4)//9)+-57)//(9*4))
    v1_val, v2_val, v3_val = v1.const_like(6), v2.const_like(0), v3.const_like(0)
    num = expr.simplify().substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    rn = expr.substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    self.assertEqual(num, rn)

  def test_fuzz_failure3(self):
    v1=Variable('v1', 0, 2)
    v2=Variable('v2', 0, 1)
    v3=Variable('v3', 0, 2)
    expr = (((((((((((((((((((0//2)//3)+v3)+0)+-4)*-2)*-2)+-1)+2)+3)+v3)+0)//8)*-3)+0)*-2)*-4)*-2)//5)
    v1_val, v2_val, v3_val = v1.const_like(0), v2.const_like(0), v3.const_like(0)
    num = expr.simplify().substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    rn = expr.substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    self.assertEqual(num, rn)

  def test_fuzz_failure4(self):
    v1=Variable('v1', 0, 2)
    v2=Variable('v2', 0, 3)
    v3=Variable('v3', 0, 4)
    expr = (((((((((((((((((((((((((((((0*-2)+0)*-1)//9)//6)//8)+v1)*-4)+v2)//4)//8)+4)*3)+v1)+v3)//8)//7)+4)+v3)*-4)+1)+v1)*3)+4)*2)//5)//2)//3)*-4)
    v1_val, v2_val, v3_val = v1.const_like(2), v2.const_like(0), v3.const_like(2)
    num = expr.simplify().substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    rn = expr.substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    self.assertEqual(num, rn)

  def test_fuzz_failure5(self):
    v1=Variable('v1', 0, 1)
    v2=Variable('v2', 0, 1)
    v3=Variable('v3', 0, 3)
    expr = ((((((((((((((0+v2)+v1)*0)+v2)//1)//7)+-2)+v2)+v1)*4)+-3)//5)+v2)+1)
    v1_val, v2_val, v3_val = v1.const_like(0), v2.const_like(0), v3.const_like(0)
    num = expr.simplify().substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    rn = expr.substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    self.assertEqual(num, rn)

  def test_fuzz_failure6(self):
    v1=Variable('v1', 0, 8)
    v2=Variable('v2', 0, 64)
    v3=Variable('v3', 0, 128)
    expr = (((((((((((((((((((((((((((((0//3)+4)+v1)//2)+-1)//1)*1)*-1)*4)//5)+v1)//6)+v1)*-1)+-4)+v2)+-2)*-3)+v3)+-4)+-2)*-1)//8)//4)*-4)+3)+v3)*
            -2)+v2)
    v1_val, v2_val, v3_val = v1.const_like(8), v2.const_like(3), v3.const_like(2)
    num = expr.simplify().substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    rn = expr.substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    self.assertEqual(num, rn)

  def test_fuzz_failure7(self):
    v1=Variable('v1', 0, 64)
    v2=Variable('v2', 0, 5)
    v3=Variable('v3', 0, 128)
    expr = (((((((((((((((((((((((((((((0+v2)*-4)+0)//9)+-4)*-2)*3)*4)//9)+v3)+v1)//4)+v1)+v3)+-1)*4)//4)+v2)//7)//3)+v1)+v2)+v3)+1)*2)//4)*3)+-1)*1)
    v1_val, v2_val, v3_val = v1.const_like(0), v2.const_like(2), v3.const_like(65)
    num = expr.simplify().substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    rn = expr.substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    self.assertEqual(num, rn)

  def test_fuzz_failure8(self):
    v1=Variable('v1', 0, 2)
    v2=Variable('v2', 0, 8)
    v3=Variable('v3', 0, 9)
    expr = (((((((0+-1)+2)+v1)*-2)//3)+v1)*-4)
    v1_val, v2_val, v3_val = v1.const_like(0), v2.const_like(0), v3.const_like(0)
    num = expr.simplify().substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    rn = expr.substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    self.assertEqual(num, rn)

  def test_fuzz_failure9(self):
    v1=Variable('v1', 0, 256)
    v2=Variable('v2', 0, 1)
    v3=Variable('v3', 0, 8)
    expr = (((((((((((((((((((((((((((((0*-2)//1)+3)*-2)+-3)*-4)*1)+v1)+0)%2)%8)%9)+v2)%9)+-4)//4)+-1)*-2)+0)+v1)+v1)+3)+v1)+4)+-4)+0)*2)+-3)%6)
    v1_val, v2_val, v3_val = v1.const_like(0), v2.const_like(1), v3.const_like(0)
    num = expr.simplify().substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    rn = expr.substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    self.assertEqual(num, rn)

  def test_fuzz_failure10(self):
    v1=Variable("v1", 0, 256)
    v2=Variable("v2", 0, 32)
    v3=Variable("v3", 0, 32)
    expr = UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.MAX, dtypes.int, arg=None, src=(
        UOp(Ops.MUL, dtypes.int, arg=None, src=(
          UOp(Ops.WHERE, dtypes.int, arg=None, src=(
            UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
              UOp(Ops.CMPLT, dtypes.bool, arg=None, src=(
                x5:=UOp(Ops.IDIV, dtypes.int, arg=None, src=(
                  UOp(Ops.WHERE, dtypes.int, arg=None, src=(
                    UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                      UOp(Ops.CMPLT, dtypes.bool, arg=None, src=(
                        x9:=UOp(Ops.CONST, dtypes.int, arg=9, src=()),
                        x10:=UOp(Ops.DEFINE_VAR, dtypes.int, arg=('v1', 0, 256), src=()),)),
                      x11:=UOp(Ops.CONST, dtypes.bool, arg=True, src=()),)),
                    UOp(Ops.ADD, dtypes.int, arg=None, src=(
                      UOp(Ops.MUL, dtypes.int, arg=None, src=(
                         x10,
                        x14:=UOp(Ops.CONST, dtypes.int, arg=-4, src=()),)),
                       x14,)),
                    UOp(Ops.IDIV, dtypes.int, arg=None, src=(
                       x10,
                       x9,)),)),
                   x9,)),
                 x14,)),
               x11,)),
             x5,
            UOp(Ops.IDIV, dtypes.int, arg=None, src=(
              UOp(Ops.ADD, dtypes.int, arg=None, src=(
                UOp(Ops.MOD, dtypes.int, arg=None, src=(
                  x19:=UOp(Ops.DEFINE_VAR, dtypes.int, arg=('v2', 0, 32), src=()),
                  UOp(Ops.CONST, dtypes.int, arg=3, src=()),)),
                 x19,)),
              UOp(Ops.CONST, dtypes.int, arg=5, src=()),)),)),
          x22:=UOp(Ops.CONST, dtypes.int, arg=-1, src=()),)),
        UOp(Ops.MUL, dtypes.int, arg=None, src=(
          UOp(Ops.ADD, dtypes.int, arg=None, src=(
            UOp(Ops.ADD, dtypes.int, arg=None, src=(
              UOp(Ops.MOD, dtypes.int, arg=None, src=(
                UOp(Ops.MUL, dtypes.int, arg=None, src=(
                   x10,
                  UOp(Ops.CONST, dtypes.int, arg=-2, src=()),)),
                UOp(Ops.CONST, dtypes.int, arg=6, src=()),)),
              UOp(Ops.MOD, dtypes.int, arg=None, src=(
                UOp(Ops.DEFINE_VAR, dtypes.int, arg=('v3', 0, 32), src=()),
                UOp(Ops.CONST, dtypes.int, arg=1, src=()),)),)),
            UOp(Ops.CONST, dtypes.int, arg=0, src=()),)),
           x22,)),)),
       x22,))
    v1_val, v2_val, v3_val = UOp.const(dtypes.int, 9), UOp.const(dtypes.int, 0),UOp.const(dtypes.int, 0)
    num = expr.simplify().substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    rn = expr.substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()
    self.assertEqual(num, rn)
