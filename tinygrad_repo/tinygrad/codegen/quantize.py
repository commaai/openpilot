from tinygrad.dtype import dtypes, least_upper_dtype
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.uop.symbolic import symbolic

# **** this is the "quantization preprocessor", it makes ONNX quantized models, and probably also others, actually use ints ****
# this is badly tested and low quality. remove it?

FP = (1 << 15)
pm_quant = symbolic+PatternMatcher([
  # cast after add/mul
  (UPat.var("x").cast(dtypes.float32) + UPat.var("y").cast(dtypes.float32),
   lambda x,y: (x.cast(least_upper_dtype(x.dtype, y.dtype))+y.cast(least_upper_dtype(x.dtype, y.dtype))).cast(dtypes.float32)),
  (UPat.var("x").cast(dtypes.float32) * UPat.var("y").cast(dtypes.float32),
   lambda x,y: (x.cast(least_upper_dtype(x.dtype, y.dtype))*y.cast(least_upper_dtype(x.dtype, y.dtype))).cast(dtypes.float32)),

  # masked MUL after masked ADD
  ((UPat.var("x") + UPat.var("v").where(UPat.var('cadd'), UPat(Ops.CONST, arg=0))) * UPat.var("v").where(UPat.var('cmul'), UPat(Ops.CONST, arg=0)),
   lambda x,v,cadd,cmul: x*v.where(cmul, 0)+v.where(cadd*cmul, 0)),

  # MUL after reduce
  (UPat(Ops.REDUCE_AXIS, src=(UPat.var("x") * UPat.cvar("c"),), name="r"), lambda x,c,r: r.replace(src=(x,))*c.arg),
  # CAST after reduce (doesn't work if it's a size change)
  (UPat(Ops.REDUCE_AXIS, src=(UPat(Ops.CAST, src=(UPat.var("x"),)),), name="r"),
    lambda x,r: r.replace(dtype=x.dtype, src=(x,)).cast(r.dtype) if dtypes.is_float(r.dtype) else None),

  # x*c1 + y*c2 -> (x+y)*c1 (if c1 and c2 are close floats)
  (UPat.var("x")*UPat.cvar("c1", dtype=dtypes.floats) + UPat.var("y")*UPat.cvar("c2", dtype=dtypes.floats),
   lambda x,y,c1,c2: (x+y)*c1 if abs(c1.arg-c2.arg) < 1e-9 else None),
  # mul 0 * c1 is 0
  (UPat(Ops.VALID, src=(UPat(Ops.VIEW, name="v"),)).where(UPat.cvar("c1"), UPat(Ops.CONST, arg=0)) *
   UPat(Ops.LOAD, src=(UPat().view(name="v"),)).cast(dtypes.int).cast(dtypes.float).named("ld"), lambda ld,v,c1: ld*c1),
  # mul (with plus) 0 * c1 is 0
  (UPat(Ops.VALID, src=(UPat(Ops.VIEW, name="v"),)).where(UPat.cvar("c1"), UPat(Ops.CONST, arg=0)) *
   (UPat(Ops.LOAD, src=(UPat().view(name="v"),)).cast(dtypes.int) + \
    UPat(Ops.VALID, src=(UPat(Ops.VIEW, name="v"),)).where(UPat.cvar(), UPat(Ops.CONST, arg=0))).cast(dtypes.float).named("ld"),
      lambda ld,v,c1: ld*c1),

  # const push through add
  ((UPat.var("x")*UPat.cvar("c1") + UPat.var("y")*UPat.cvar("c2")) * UPat.cvar("c3"), lambda x,y,c1,c2,c3: (x*c1*c3) + (y*c2*c3)),

  # fixed point mult, replace (x.float()*c1+c2).int() with an int expression
  ((UPat.var("x").cast(dtypes.float)*UPat.var("c1")+UPat.var("cc")).cast(dtypes.int),
   lambda x,c1,cc: ((x*(c1*FP).cast(x.dtype) + (cc*FP).cast(x.dtype)) // FP).cast(dtypes.int)),
  # fixed point mult, replace (x.float()*c1 + y.float()*c2)*cc.int() with an int expression
  ((UPat.var("x").cast(dtypes.float)*UPat.var("c1")+UPat.var("y").cast(dtypes.float)*UPat.var("c2")+UPat.var("cc")).cast(dtypes.int),
   lambda x,c1,y,c2,cc: ((x*(c1*FP).cast(x.dtype) + y.cast(x.dtype)*(c2*FP).cast(x.dtype) + (cc*FP).cast(x.dtype)) // FP).cast(dtypes.int)),

  # where move
  (UPat.var("valid").where(UPat.var("yes"), UPat(Ops.CONST, arg=0))*UPat.var("mul"), lambda valid, yes, mul:
    (yes*mul*valid.where(UOp.const(mul.dtype, 1), UOp.const(mul.dtype, 0))) if yes.op is not Ops.CONST or yes.arg != 1 else None),
  ((UPat.var("x")*UPat.cvar("c"))*(UPat.var().where(UPat(Ops.CONST, arg=1), UPat(Ops.CONST, arg=0)).named("v")), lambda x,c,v: (x*v)*c),
  (UPat.var("x").cast().named('c') * UPat.var('valid').where(UPat(Ops.CONST, arg=1), UPat(Ops.CONST, arg=0)), lambda x,c,valid:
    (x*valid.where(UOp.const(x.dtype, 1), UOp.const(x.dtype, 0))).cast(c.dtype)),
  ((UPat.var('x') * UPat.var('v1').where(UPat(Ops.CONST, arg=1), UPat(Ops.CONST, arg=0)) *
    UPat.var('v2').where(UPat(Ops.CONST, arg=1), UPat(Ops.CONST, arg=0))).named("mul"), lambda x, mul, v1, v2:
    x * (v1&v2).where(UOp.const(mul.dtype, 1), UOp.const(mul.dtype, 0))),

  # where on two adds
  (UPat.var("x") + UPat.var("v").where(UPat.var("a0"), UPat.var("a1")) + UPat.var("v").where(UPat.var("b0"), UPat.var("b1")),
    lambda x,v,a0,a1,b0,b1: x + v.where(a0+b0, a1+b1)),

  # split REDUCE into multiple reduces (who remembers FOIL?)
  (UPat(Ops.REDUCE_AXIS, src=((UPat(Ops.CAST, name="v1")+UPat.var("c1")) * UPat(Ops.CAST, name="v2"),), name="r"),
    lambda v1,v2,c1,r: r.replace(src=(v1*v2,)) + r.replace(src=(c1*v2,))),
  (UPat(Ops.REDUCE_AXIS, src=((UPat(Ops.CAST, name="v1")+UPat.var("c1")) * (UPat(Ops.CAST, name="v2",)+UPat.var("c2")),), name="r"),
    lambda v1,v2,c1,c2,r: r.replace(src=(v1*v2,)) + r.replace(src=(c2*v1,)) + r.replace(src=(c1*v2,)) + r.replace(src=(c1*c2,))),
])