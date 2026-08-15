# all of symbolic lives here now
import math
from collections import defaultdict
from tinygrad.uop.ops import Ops, PatternMatcher, UPat, UOp, GroupOp, exec_alu
from tinygrad.dtype import PyConst, ConstType, dtypes, can_lossless_cast, Invalid, bitcast
from tinygrad.helpers import partition, all_same, prod, flatten, unwrap, IMAGE, dedup
from tinygrad.uop.divandmod import div_and_mod_symbolic
from tinygrad.uop.movement import mop_cleanup

# TODO: symbolic shouldn't be importing from codegen
from tinygrad.codegen.decomp.transcendental import xpow

# ******** phase 1 of symbolic used to live in ops, it's the most generic folding rules ********

def simplify_pow(x:UOp, c:UOp) -> UOp|None:
  if c.val < 0: return x.reciprocal().pow(-c.val)
  if c.val == 0: return x.const_like(1)
  if int(c.val-0.5)+0.5 == c.val: return x.pow(c.val-0.5) * x.sqrt()
  if int(c.val) == c.val: return (y := x.pow(c.val//2)) * y * (x if c.val%2 == 1 else 1)
  return None

def fold_bitcast(root:UOp, c:UOp) -> UOp|None:
  if c.dtype.fmt is None or root.dtype.fmt is None or c.dtype.itemsize != root.dtype.itemsize: return None
  return root.const_like(bitcast(c.val, c.dtype, root.dtype))

def const_arg(u:UOp) -> ConstType|tuple[ConstType, ...]|None:
  if u.op is Ops.CONST: return u.val
  if u.op is Ops.STACK and all(s.op is Ops.CONST for s in u.src): return tuple(s.val for s in u.src)
  return None

def fold_const_alu(a:UOp) -> UOp|None:
  vals = [const_arg(s) for s in a.src]
  return None if any(v is None for v in vals) else a.const_like(exec_alu(a.op, a.dtype, vals, False))

def _quotient_base(q:UOp, base:UOp, div:int) -> UOp|None:
  # the B with q == B//div and B%div == base%div, or None. only such congruence is needed to recombine, and canonicalization
  # moves consts freely: the quotient may be merged ((x//c + a)//div -> (x + a*c)//(c*div) for div>0) and shifted ((y + k*D)//D == y//D + k)
  (q, s), (num, a) = q.pop_const(), base.pop_const()
  if q.op is not Ops.FLOORDIV or q.src[1].op is not Ops.CONST: return None
  if div > 0 and num.op is Ops.FLOORDIV and num.src[1].op is Ops.CONST and q.src[1].val == (c:=num.src[1].val)*div: num, a, D = num.src[0], a*c, c*div
  elif q.src[1].val == div: D = div
  else: return None
  (x, xa), (p, pa) = num.pop_const(), q.src[0].pop_const()
  if p is not x or (t:=xa + a - pa) % D: return None
  return base - k*div if (k:=t//D - s) else base

def fold_add_divmod_recombine(x:UOp) -> UOp|None:
  # a scaled mod (base%div)*mul recombines with a partner q*(div*mul) carrying the quotient of a b == base (mod div):
  #   q == b//div     -> b*mul              (full recombine)
  #   q == (b//div)%d -> (b%(div*d))*mul    (partial recombine into a wider mod, needs d>0)
  terms = list(x.split_uop(Ops.ADD))
  for i,u in enumerate(terms):
    mod, mul = u.pop_const(Ops.MUL)
    if mod.op is not Ops.FLOORMOD or mod.src[1].op is not Ops.CONST: continue
    base, div = mod.src[0], mod.src[1].val
    for j,v in enumerate(terms):
      q, scale = v.pop_const(Ops.MUL)
      if i == j or scale != div*mul: continue
      rest = [t for k,t in enumerate(terms) if k not in (i,j)]
      if (b:=_quotient_base(q, base, div)) is not None: return (b*mul).usum(*rest)
      if q.op is Ops.FLOORMOD and q.src[1].op is Ops.CONST and (d:=q.src[1].val) > 0 and (b:=_quotient_base(q.src[0], base, div)) is not None:
        return ((b % (div*d))*mul).usum(*rest)
  return None

# Invalid poisons the value: ops move inside the gate so the Invalid reaches the LOAD/STORE and folds there.
# this needs to be before symbolic so that 0*something_that_might_be_invalid doesnt become 0
invalid_pat = UPat(Ops.CONST, arg=Invalid, name="i")
invalid_gate = UPat.var("cond").where(UPat.var("x"), invalid_pat)
pm_data_invalid = PatternMatcher([
  (invalid_pat.broadcast(), lambda i: i),
  (UPat(GroupOp.Unary|{Ops.CAST, Ops.BITCAST}, src=(invalid_pat,)), lambda i: i),
  (UPat(GroupOp.Unary|{Ops.CAST, Ops.BITCAST}, src=(invalid_gate,), name="op"),
   lambda cond,x,op,i: cond.where(op.replace(src=(x,)), i)),
  # binary ops move inside the gate, with Invalid in the false branch
  (UPat(GroupOp.Binary, src=(invalid_gate, UPat.var("y")), name="alu"), lambda cond,x,y,alu,i: cond.where(x.alu(alu.op,y), i)),
  (UPat(GroupOp.Binary, src=(UPat.var("y"), invalid_gate), name="alu"), lambda cond,x,y,alu,i: cond.where(y.alu(alu.op,x), i)),
  (UPat(GroupOp.Binary-GroupOp.Comparison, src=[invalid_pat, UPat()]), lambda i: i),
  # an Invalid condition poisons the whole where; a gated Invalid condition lifts the gate out
  (invalid_pat.where(UPat(), UPat()), lambda i: i),
  (invalid_gate.where(UPat.var("a"), UPat.var("b")), lambda cond,x,i,a,b: cond.where(x.where(a,b), i)),
  # normalize where(cond, Invalid, val) -> where(~cond, val, Invalid)
  (UPat.var("cond").where(invalid_pat, UPat.var("val")), lambda cond, i, val: cond.logical_not().where(val, i) if not val.is_invalid else i),
  # lift Invalid out: a.where(cond.where(x, Invalid), c) -> (~a|cond).where(a.where(x, c), Invalid)
  (UPat.var("a").where(invalid_gate, UPat.var("c")), lambda cond,i,x,a,c:
   (a.logical_not()|cond).where(a.where(x,c), i) if not c.is_invalid else None),
  (UPat.var("a").where(UPat.var("b"), invalid_gate), lambda cond,i,x,a,b: (a|cond).where(a.where(b, x), i) if not b.is_invalid else None),
  # fold gated LOAD/STORE
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(), invalid_pat), allow_any_len=True).or_casted(), UPat())), lambda i: UOp(Ops.NOOP)),
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat(), invalid_pat), allow_any_len=True).or_casted(),), allow_any_len=True, name="x"),
    lambda x,i: x.src[1] if len(x.src) > 1 else x.const_like(0)),
])

pm_remove_invalid = PatternMatcher([
  (invalid_gate.named("w"), lambda cond,x,i,w: w.replace(src=(cond,x,w.const_like(0)))),
  (UPat(Ops.STACK, name="s"), lambda s: s.replace(src=tuple(UOp.const(0, s.dtype) if x.is_invalid else x for x in s.src))
   if any(x.is_invalid for x in s.src) else None),
])

# the one rule that collapses the pair CAST(dt, CONST(v)) into a typed CONST
# TODO: delete this once CONST has no dtype
pm_fold_cast_const = PatternMatcher([(UPat(Ops.CAST, name="root", src=(UPat.cvar("c"),)), lambda root, c: root.const_like(c.val))])

symbolic_simple = pm_data_invalid + PatternMatcher([
  # ** self folding **
  (UPat.var("x") + 0, lambda x: x),    # x+0 -> x
  (UPat.var("x") * 1, lambda x: x),    # x*1 -> x
  (UPat.var("x", dtype=dtypes.ints+(dtypes.bool, dtypes.weakint)) ^ 0, lambda x: x), # x^0 -> x
  (UPat.var("x") // UPat.var("x"), lambda x: x.const_like(1)), # x//x -> 1
  (UPat.var("x") // 1, lambda x: x),   # x//1 -> x
  (UPat.var("x") // -1, lambda x: -x), # x//-1 -> -x
  ((UPat.var("x") ^ UPat.var("y")) ^ UPat.var("y"), lambda x,y: x), # (x^y)^y -> x
  ((UPat.var() % UPat.var("y")).named("base") % UPat.var("y"), lambda base,y: base),  # (x%y)%y = -> x%y (rewritten with base for speed)
  # variations of (x%c)+(x//c)*c = x
  (UPat(Ops.ADD, dtype=dtypes.weakint, name="x"), fold_add_divmod_recombine),
  (UPat.var("x", dtype=dtypes.bool) & UPat.cvar("c"), lambda x,c: x if c.val else c),
  (UPat.var("x", dtype=dtypes.bool) | UPat.cvar("c"), lambda x,c: c if c.val else x),
  (UPat.var("x", dtype=dtypes.bool) != UPat.const(False, dtypes.bool), lambda x: x),  # x != False -> x
  (UPat(GroupOp.Idempotent, src=(UPat.var("x"), UPat.var("x"))), lambda x: x),
  (UPat.var("x", dtype=dtypes.bool).logical_not().logical_not(), lambda x: x),
  (UPat.var("x", dtype=dtypes.bool).where(UPat.const(True, dtypes.bool), UPat.const(False, dtypes.bool)), lambda x: x),
  (UPat.var("x", dtype=dtypes.bool).where(UPat.const(False, dtypes.bool), UPat.const(True, dtypes.bool)), lambda x: x.logical_not()),
  # CAST(bool -> int) != const — CAST(True)=1, CAST(False)=0, so fold based on const value
  (UPat.var("x", dtype=dtypes.bool).cast(dtypes.ints+(dtypes.weakint,)) != UPat.cvar("c"),
   lambda x,c: x if c.val == 0 else x.logical_not() if c.val == 1 else x.const_like(True)),
  (UPat.var("x", dtype=dtypes.ints+(dtypes.bool, dtypes.weakint)).trunc(), lambda x: x),
  # ** zero folding **
  (UPat.var("x") < UPat.var("x"), lambda x: x.const_like(False, dtypes.bool)), # x < x -> False
  (UPat.var("x") % UPat.var("x"), lambda x: x.const_like(0)), # x%x -> 0
  (UPat.var("x") ^ UPat.var("x"), lambda x: x.const_like(0)), # x^x -> 0
  (UPat.var("x") & 0, lambda x: x.const_like(0)), # x&0 -> 0
  # (x&mask)>>k -> x>>k when mask only clears bits below k
  ((UPat.var("x") & UPat.cvar("mask")) >> UPat.cvar("k"),
   lambda x,mask,k: x >> k.val if mask.val | ((1 << k.val) - 1) == -1 else None),
  ((UPat.var("x") & UPat.cvar("mask")) // UPat.cvar("c"),
   lambda x,mask,c: x // c.val if c.val > 0 and c.val & (c.val-1) == 0 and mask.val | (c.val-1) == -1 else None),
  (UPat.var("x", dtype=dtypes.ints+(dtypes.bool, dtypes.weakint)) != UPat.var("x"),
   lambda x: x.const_like(False, dtypes.bool)), # x != x -> False (only ints)
  # ** constant folding **
  (UPat(GroupOp.Unary, src=(UPat((Ops.CONST, Ops.STACK)),), name="a"), fold_const_alu),
  # NOTE: THREEFRY(const,const) folds via its decomposition
  (UPat(GroupOp.Binary-{Ops.THREEFRY}, src=(UPat((Ops.CONST, Ops.STACK)),)*2, name="a"), fold_const_alu),
  (UPat(GroupOp.Ternary, src=(UPat((Ops.CONST, Ops.STACK)),)*3, name="a"), fold_const_alu),
  # bool MUL is AND, ADD/MAX is OR. prevents other rules to rewrite bool ADD/MUL incorrectly
  (UPat.var('x', dtype=dtypes.bool) * UPat.var('y', dtype=dtypes.bool), lambda x,y: x&y),
  (UPat.var('x', dtype=dtypes.bool) + UPat.var('y', dtype=dtypes.bool), lambda x,y: x|y),
  (UPat.var('x', dtype=dtypes.bool).maximum(UPat.var('y', dtype=dtypes.bool)), lambda x,y: x|y),
  # *** div rules ***
  (UPat.cvar('x', arg=0) / 0, lambda x: x.const_like(float('nan'))),   # 0/0 -> nan
  ((UPat.var("x") * 0) / 0, lambda x: x.const_like(float('nan'))),     # (x*0)/0 -> nan
  # can be wrong if x or x2 is 0
  (UPat.var("x") / UPat.var("x"), lambda x: x.const_like(1)),          # x/x -> 1
  ((UPat.var("x") * UPat.var("x2")) / UPat.var("x2"), lambda x,x2: x), # (x*x2)/x2 -> x
  # x*0 -> 0 or 0*x -> 0
  # if x is nan or inf it should render the nan value.
  # NOTE: this can be wrong for loaded NaN
  (UPat.var("x") * 0, lambda x: x.const_like(float("nan") if x.op is Ops.CONST
                                             and isinstance(x.val, float) and (math.isnan(x.val) or math.isinf(x.val)) else 0)),
  # *** cast/bitcast ***
  (UPat((Ops.CAST, Ops.BITCAST), name="root"), lambda root: root.src[0] if root.dtype == root.src[0].dtype else None),
  (UPat(Ops.BITCAST, name="root", src=(UPat.cvar("c"),)), fold_bitcast),
  # b.cast(a).cast(b) -> b if a preserves all values in b
  (UPat.var('x').cast(name="a").cast(name="b"), lambda x,a,b: x if x.dtype == b.dtype and can_lossless_cast(b.dtype, a.dtype) else None),
  # bitcast twice
  (UPat(Ops.BITCAST, name="b", src=(UPat.var('x').bitcast(),)), lambda x,b: x.bitcast(b.dtype)),
  (UPat.var("x").cast(dtypes.bool), lambda x: x != 0),
  # ** pow **
  (UPat.var("x").alu(Ops.POW, UPat.cvar("c")), simplify_pow),
  # positive const ** x
  (UPat.cvar("c").alu(Ops.POW, UPat.var("x")), lambda c,x: c if c.val == 1 else (x*math.log2(c.val)).exp2() if c.val > 0 else None),
  # unpack a uint64 packed from two uint32 (threefry)
  (((UPat.var(None, dtypes.uint64)<<32) | UPat.var('y', dtypes.uint32).cast(dtypes.uint64)).cast(dtypes.uint32), lambda y: y),
  (((UPat.var('x', dtypes.uint32).cast(dtypes.uint64)<<32) | UPat.var(None, dtypes.uint32).cast(dtypes.uint64))>>32,
   lambda x: x.cast(dtypes.uint64)),
  # ** simple where folding **
  # a conditional with the same results either way is a noop, also fold const conditionals
  (UPat.var().where(UPat.var("val"), UPat.var("val")), lambda val: val),
  (UPat.cvar("gate").where(UPat.var("c0"), UPat.var("c1")), lambda gate, c0, c1: c0 if gate.val else c1),
  # a.where(b.where(c, d), d) -> (a & b).where(c, d)
  (UPat.var("a").where(UPat.var("b").where(UPat.var("c"), UPat.var("d")), UPat.var("d")), lambda a,b,c,d: (a&b).where(c,d)),
  # a.where(c, b.where(c, d)) -> (a | b).where(c, d)
  (UPat.var("a").where(UPat.var("c"), UPat.var("b").where(UPat.var("c"), UPat.var("d"))), lambda a,b,c,d: (a|b).where(c,d)),
])+mop_cleanup

# ******** phase 2 builds on phase 1, it includes the old "symbolic", rules that match deeper ********

def lt_folding(x:UOp, c:int) -> UOp|None:
  p, np = partition(x.split_uop(Ops.ADD), lambda u: u.const_factor() == 1)
  if np and (d:=math.gcd(*[u.const_factor() for u in np], c)) > 1 and 0 <= sum(u.vmin for u in p) and sum(u.vmax for u in p) < d:
    return unwrap(UOp.usum(*np).divides(d))<(c//d)
  return None

def canonicalize_simplex(X:UOp) -> UOp|None:
  # (X := a0*x0 + a1*x1 + ...) > 0 is equivalent to x0 + x1 + ... > 0 if xi >= 0 and ai > 0 for ints.
  # returns x0 + x1 + ... in such case, or None if not
  changed, ret = False, []
  for u in X.split_uop(Ops.ADD):
    # assumed the const is the last src of MUL
    if u.op is Ops.MUL and u.src[1].op is Ops.CONST and u.src[1].val > 0:
      changed = True
      u = u.src[0]
    if not (u.op in GroupOp.Irreducible and u.vmin >= 0): return None
    ret.append(u)
  return UOp.usum(*ret) if changed else None

commutative = PatternMatcher([
  # ** COMMUTATIVE flipping (only for index) **
  # NOTE: this can break merging vector math by only flipping some of them
  (UPat(GroupOp.Commutative, dtype=dtypes.weakint, name='x'), lambda x:
    x.replace(src=x.src[::-1]) if x.src[1].tuplize < x.src[0].tuplize and not x.src[0].tuplize < x.src[1].tuplize else None),
])

def fold_where_closure(cond:UOp, t:UOp, f:UOp) -> UOp|None:
  """in cond.where(t, f), cond is True within t and False within f"""
  if cond not in t.bool_slice and cond not in f.bool_slice: return None
  # INDEX gates are owned by the valid/store-coalescing machinery, leave them alone
  if any(u.op_in_backward_slice_with_self(Ops.INDEX) for u in (cond, t, f)): return None
  return cond.where(t.substitute({cond: cond.const_like(True)}), f.substitute({cond: cond.const_like(False)}))

symbolic = symbolic_simple+commutative+PatternMatcher([
  # ** boolean algebra **
  # TODO: make a more general or folder like simplify_valid
  (UPat.var("x", dtype=dtypes.bool) | UPat.var("x", dtype=dtypes.bool).logical_not(), lambda x: x.const_like(True)),  # x|!x -> True
  # ** combine terms **
  (UPat.var("x") * UPat.cvar("c0") + UPat.var("x") * UPat.cvar("c1"), lambda x,c0,c1: x*(c0+c1)), # (x*c0)+(x*c1) -> x*(c0+c1)
  ((UPat.var("y") + UPat.var("x") * UPat.cvar("c0")) + UPat.var("x") * UPat.cvar("c1"), lambda x,y,c0,c1: y+x*(c0+c1)),
  (UPat.var("x") + UPat.var("x") * UPat.cvar("c"), lambda x,c: x*(c+1)), # (x+x*c)-> x*(c+1)
  ((UPat.var("y") + UPat.var("x")) + UPat.var("x") * UPat.cvar("c"), lambda x,y,c: y+x*(c+1)),
  ((UPat.var("y") + UPat.var("x") * UPat.cvar("c")) + UPat.var("x"), lambda x,y,c: y+x*(c+1)),
  (UPat.var("x") + UPat.var("x"), lambda x: x*2), # (x+x)-> x*2
  ((UPat.var("y") + UPat.var("x")) + UPat.var("x"), lambda y,x: y+x*2),
  ((UPat.var("x") / UPat.var("x2")) / UPat.var("x3"), lambda x,x2,x3: x/(x2*x3) if x2 is not x3 else None), # (x/x2)/x3 -> x/(x2*x3)
  (-1 * (UPat.var("x") + UPat.cvar("c")), lambda x,c: (-x)+(-c)),  # -(x+c) -> -x + -c
  (UPat.cvar("y") * (UPat.var("x", dtype=dtypes.weakint) + UPat.cvar("c")), lambda x,y,c: (y*x)+(y*c)),  # y*(x+c) -> y*x + y*c
  # ** where folding **
  (UPat.var("cond", dtype=dtypes.bool).logical_not().where(UPat.var("t"), UPat.var("f")),
   lambda cond, t, f: cond.where(f,t) if not f.is_invalid else None),
  # in cond.where(t, f), uses of cond fold to True within t and False within f
  (UPat.var("cond", dtype=dtypes.bool).where(UPat.var("t"), UPat.var("f")), fold_where_closure),
  # alu of two where with same conds can combine, only do if true branch or false branch is const
  (UPat(GroupOp.Binary, name="alu", src=(UPat.var("c").where(UPat.var("t"), UPat.var("f")), UPat.var("c").where(UPat.var("tt"), UPat.var("ff")))), \
   lambda alu,c,t,tt,f,ff: c.where(t.alu(alu.op, tt), f.alu(alu.op, ff)) if t.op == tt.op == Ops.CONST or f.op == ff.op == Ops.CONST else None),
  # if its a plus we add the associative variation too
  ((UPat.var("y")+UPat.var("c").where(UPat.var("t"), UPat.var("f"))) + UPat.var("c").where(UPat.var("tt"), UPat.var("ff")), \
   lambda y,c,t,tt,f,ff: y+c.where(t+tt, f+ff) if t.op == tt.op == Ops.CONST or f.op == ff.op == Ops.CONST else None),
  # complementary zero branches under the same condition select directly
  (UPat.var("c").where(UPat.var("t"), 0) + UPat.var("c").where(0, UPat.var("f")), lambda c,t,f: c.where(t, f)),
  # ALU/variable min==max -> CONST
  (UPat({Ops.CMPLT, Ops.CMPNE, Ops.FLOORDIV, Ops.FLOORMOD, Ops.PARAM, Ops.BIND, Ops.SPECIAL}, name="x"),
   lambda x: x.const_like(x.vmin) if x.vmin == x.vmax else None),
  (UPat(Ops.RANGE, src=(UPat(Ops.CONST,)), name="x"), lambda x: x.const_like(x.vmin) if x.vmin == x.vmax else None),
  # max folding
  (UPat.maximum(UPat.var("x"), UPat.var("y")), lambda x,y: x if x.vmin >= y.vmax else y if x.vmax <= y.vmin else None),
  # TODO: why does this rule break beautiful_mnist?
  #((UPat.var("x")+UPat.var("z")).maximum(UPat.var("y")+UPat.var("z")), lambda x,y,z: x.maximum(y) + z),
  # ** two stage ALU folding **
  *((UPat.var("x").alu(op, UPat.cvar("c1")).alu(op, UPat.cvar("c2")).named("f"),
     lambda f,x,c1,c2: x.alu(f.op,c1.alu(f.op,c2))) for op in GroupOp.Associative),
  # (x//c1)//c2 -> x//(c1*c2) for c2>0
  ((UPat.var("x") // UPat.cvar("c1")) // UPat.cvar("c2"), lambda x,c1,c2: x//(c1*c2) if c2.vmin>0 else None),
  # ** lt **
  # c0+x<c1 -> x < c1-c0
  ((UPat.cvar("c0") + UPat.var("x", dtype=dtypes.ints+(dtypes.weakint,))) < UPat.cvar("c1"), lambda x,c0,c1: x<(c1-c0)),
  # c0*x<c1 -> sign(c0)*x < ceil(c1/abs(c0))
  ((UPat.cvar("c0")*UPat.var("x", dtype=dtypes.weakint))<UPat.cvar("c1"),
   lambda x,c0,c1: (x if c0.val > 0 else -x)<-(-c1.val//abs(c0.val)) if abs(c0.val) > 1 else None),
  # x//d<c -> x<c*d for d>0, and -> c*d<x for d<0
  ((UPat.var("x", dtype=dtypes.weakint)//UPat.cvar("d"))<UPat.cvar("c"),
   lambda x,d,c: (x<c.val*d.val) if d.val > 0 else (x>c.val*d.val) if d.val < 0 else None),
  # ** move add/mul consts to end (NOTE: this is still happening before constant folding) **
  ((UPat.var("x") + UPat.cvar("c1")) + UPat.var("y"), lambda x,c1,y: (x+y)+c1 if y.op is not Ops.CONST else None),
  ((UPat.var("x") * UPat.cvar("c1")) * UPat.var("y"), lambda x,c1,y: (x*y)*c1 if y.op is not Ops.CONST else None),
  # *** rules from symbolic ***
  # generic lt folding
  (UPat.var("x", dtypes.weakint)<UPat.cvar("c"), lambda x,c: lt_folding(x, c.val) if 0 < c.val else None),
  (UPat.var("x", dtypes.weakint)*-1 < UPat.var("y")*-1, lambda x,y: y<x),
  # canonicalize a simplex with positive coefficients > 0. NOTE: not x < 1 means x > 0
  ((UPat.var("x", dtypes.weakint)<1).ne(True), lambda x: (newx<1).ne(True) if (newx:=canonicalize_simplex(x)) is not None else None),
  # a range mod its own upper bound is just the range
  (UPat(Ops.RANGE, src=UPat.var("end"), name="r")%UPat.var("end"), lambda r,end: r),
  (UPat(Ops.RANGE, src=UPat.var("end"), name="r")//UPat.var("end"), lambda r,end: r.const_like(0)),
  # cast/long folding
  # if the intermediate cast doesnt narrow we can do it in one cast
  (UPat.var('x').cast(name="a").cast(name="b"), lambda x,a,b: x.cast(b.dtype) if can_lossless_cast(x.dtype, a.dtype) else None),
  (UPat.var('x', dtypes.ints+(dtypes.weakint,)).cast(dtypes.ints+(dtypes.weakint,), name="a").cast(name="b"),
    lambda x,a,b: x.cast(b.dtype) if a.dtype.min<=x.vmin and x.vmax<=a.dtype.max else None),
  # try to do math in int instead of long, keep weak const weak
  (UPat(GroupOp.Binary, src=(UPat.var("x", dtypes.long), UPat.var("y", dtypes.long)), name="u"), lambda u,x,y:
    (UOp.const(x.val) if x.op is Ops.CONST else x.cast(dtypes.int)).alu(u.op,
     UOp.const(y.val) if y.op is Ops.CONST else y.cast(dtypes.int)).cast(u.dtype)
    if not any(v.overflows(dtypes.int) for v in (u,x,y)) else None),
  ((UPat.var("x", dtypes.weakint) + UPat.cvar("c")).cast(dtypes.sints, name="cast"), lambda x,c,cast:x.cast(cast.dtype)+cast.const_like(c.val)),
  # only RANGE/IF/STORE/KERNEL have side effects
  (UPat(Ops.AFTER, name="x"), lambda x: x.replace(src=(x.src[0],)+
    tuple(dedup(flatten([(y,) if y.op in {Ops.RANGE, Ops.STORE, Ops.CALL, Ops.FUNCTION, Ops.BARRIER, Ops.END, Ops.LINEAR, Ops.STAGE}
                        else y.src for y in x.src[1:]]))))),
  # after with 1 src is just src[0]
  (UPat(Ops.AFTER, src=(UPat.var("s"),)), lambda s: s),
])+div_and_mod_symbolic

# ******** we take a small aside to "simplify_valid" to rewrite valids ********

def parse_valid(v:UOp) -> tuple[UOp, bool, int]|None:
  # if it's X <= c, returns X, True, c
  # if it's X >= c, returns X, False, c

  if v.op is Ops.CMPNE and v.src[1].op is Ops.CONST and v.src[1].val == 1 and (s0:=v.src[0]).op is Ops.CMPLT and dtypes.is_int(s0.src[0].dtype):
    # (X < c).ne(True) -> X >= c
    return s0.src[0], False, int(s0.src[1].vmin)
  if v.op is Ops.CMPLT and dtypes.is_int(v.src[0].dtype):
    # c < X -> X >= c+1 (a const on the left is a lower bound on the right)
    if v.src[0].op is Ops.CONST: return v.src[1], False, int(v.src[0].val)+1
    # X < c -> X <= c-1
    return v.src[0], True, int((v.src[1]).vmax)-1
  return None

def uop_given_valid(valid:UOp, uop:UOp, try_simplex=True) -> UOp:
  # return simplified uop (might be the same as input)

  # first, parse valid into {expr: (lower_bound, upper_bound)}
  bounds:defaultdict[UOp, list[PyConst|None]] = defaultdict(lambda: [None, None])
  for stmt in valid.split_uop(Ops.AND):
    if (res:=parse_valid(stmt)) is None: continue
    expr, is_upper, c = res
    bounds[expr][int(is_upper)] = c

  # simplify uop given that valid is True
  all_candidates = []
  for i,(expr,v) in enumerate(bounds.items()):
    v0, v1 = (expr.vmin if v[0] is None else v[0], expr.vmax if v[1] is None else v[1])
    # try checking the whole clause
    all_candidates.append((expr, UOp.variable(f"fake{i}", v0, v1, expr.dtype)))

    if try_simplex:
      # every candidate is a set of constrained UOp based on valid, and if every item in a set simplifies the uop into a same output, we rewrite uop
      candidates = [[all_candidates[-1]]]
      if expr.op is Ops.ADD and v0 == 1 and all(u.op in GroupOp.Irreducible for u in expr.split_uop(Ops.ADD)):
        # if the constraint is a simplex: X0 + X1 + ... > 0, we can check if all Xi > 0 simplify into the same output
        candidates.append([(Xi, UOp.variable(f"fake{i}", 1, Xi.vmax, Xi.dtype)) for Xi in expr.split_uop(Ops.ADD)])

      for candidate in candidates:
        # if every branch in candidate gives the same simplified uop, we can rewrite the uop
        if any(X not in uop.backward_slice_with_self for X,_ in candidate): continue  # skip if a branch var isn't in uop
        newuops = [uop.substitute({X:newX}).simplify().substitute({newX:X}).simplify() for X,newX in candidate]
        if all_same(newuops): uop = newuops[0]
        elif uop.op is Ops.STACK and len(uop.src) == 2:
          if all_same([uops.src[0] for uops in newuops]): uop = uop.replace(src=(newuops[0].src[0], uop.src[1]))
          if all_same([uops.src[1] for uops in newuops]): uop = uop.replace(src=(uop.src[0], newuops[0].src[1]))

  # try all the valids together (but only the whole expressions)
  if (s_uop:=uop.substitute(sub_dict:=dict(all_candidates))) is not uop:
    uop = s_uop.simplify().substitute({newX:X for X,newX in sub_dict.items()}).simplify()
  return uop

def _valid_priority(v: UOp, valids:list[UOp]) -> int:
  # we want valid that's in other valids' parents to be first, so it's more likely the other valids get simplified
  return 0 if (res:=parse_valid(v)) is None else sum(-1 for other in valids if res[0] in other.backward_slice_with_self)

def simplify_valid(valid:UOp) -> UOp|None:
  if valid.op_in_backward_slice_with_self(Ops.INDEX): return None  # this should only be for indexing, skip if there's a INDEX
  ret:list[UOp] = []
  valids = list(valid.split_uop(Ops.AND))
  valids = sorted(valids, key=lambda v: _valid_priority(v, valids))
  for stmt in dedup(valids):
    if ret: stmt = uop_given_valid(UOp.uprod(*ret), stmt)
    ret.append(stmt)
  return UOp.uprod(*ret) if ret != valids else None

# ******** phase 3 is the complete symbolic ********

def reduce_mul_chain(r:UOp) -> UOp|None:
  if r.arg[0] not in {Ops.ADD, Ops.MAX}: return None
  if r.dtype != r.src[0].dtype: return None
  inside, outside = [], []
  for m in r.src[0].split_uop(Ops.MUL):
    m_parents = m.backward_slice
    if m not in r.src[1:] and all(r not in m_parents for r in r.src[1:]) and (r.arg[0] != Ops.MAX or m.vmin >= 0): outside.append(m)
    else: inside.append(m)
  if len(outside) == 0: return None
  return r.replace(src=(prod(inside) if len(inside) else r.src[0].const_like(1),)+r.src[1:])*prod(outside)

def drop_and_clauses(cond:UOp, x:UOp, i:UOp) -> UOp|None:
  keep, drop = partition(cond.split_uop(Ops.AND), lambda c: any(r in x.ranges for r in c.ranges))
  return UOp.const(True).uprod(*keep).where(x, i) if drop else None
pm_drop_and_clauses = PatternMatcher([(invalid_gate, drop_and_clauses)])

# move conditions from where to load's valid, drop clauses already in load
def where_on_load(cond:UOp, buf:UOp, idx:UOp, or_cast:UOp) -> UOp|None:
  where_clauses, load_valid = list(cond.split_uop(Ops.AND)), idx.get_valid()
  in_load = set(load_valid.split_uop(Ops.AND))
  idx_index = {u for u in idx.backward_slice_with_self if u.op is Ops.INDEX}
  # can move if: condition's ranges are subset of idx's ranges, and no data dependent INDEX (only idx's INDEX allowed)
  def can_move(c:UOp) -> bool:
    return c.ranges.keys() <= idx.ranges.keys() and all(u in idx_index for u in c.backward_slice_with_self if u.op is Ops.INDEX)
  moved, keep = partition([c for c in where_clauses if c not in in_load], can_move)
  if len(keep) == len(where_clauses): return None
  idx = buf.index(idx.get_idx().valid(load_valid.uprod(*moved)))
  ret_idx = idx.cast(or_cast.dtype) if or_cast.op is Ops.CAST else idx
  return UOp.const(True).uprod(*keep).where(ret_idx, ret_idx.const_like(0))

# where after gated load becomes alt value, TODO: this is sort of duplicated with rules in devectorizer
pm_move_where_on_load = PatternMatcher([
  (UPat.var("cond").where(UPat.var("buf").index(UPat.var("idx")).or_casted("or_cast"), 0), where_on_load),
  (UPat.var("cond").where(0, UPat.var("buf").index(UPat.var("idx")).or_casted("or_cast")),
   lambda cond,buf,idx,or_cast: where_on_load(cond.logical_not(),buf,idx,or_cast)),
])

def gated_given_valid(cond:UOp, x:UOp, i:UOp) -> UOp|None:
  if x.dtype is not dtypes.weakint: return None
  # Skip if x contains DIV/MOD AND IMAGE mode is enabled -> image index e.g. openpilot
  if IMAGE.value > 0 and x.op_in_backward_slice_with_self(Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD): return None
  return cond.where(uop_given_valid(cond, x, try_simplex=False), i)

pm_simplify_valid = PatternMatcher([
  # simplify valid
  (UPat(Ops.AND, name="valid"), simplify_valid),
  (invalid_gate, gated_given_valid),
])

# this is symbolic 2.0
REMOVE_FROM_SINK_LIKE = {Ops.NOOP, Ops.STACK, Ops.SINK, Ops.GROUP}
pm_clean_up_group_sink = PatternMatcher([
  # clean up GROUP/SINK
  (UPat(Ops.GROUP, src=(UPat.var("x"),)), lambda x: x),
  (UPat((Ops.SINK, Ops.GROUP), name="root"),
    lambda root: UOp(root.op, src=tuple(flatten(x.src if x.op in REMOVE_FROM_SINK_LIKE else (x,) for x in root.src)), arg=root.arg)
      if any(x.op in REMOVE_FROM_SINK_LIKE for x in root.src) else None),
])

sym = symbolic+pm_simplify_valid+PatternMatcher([
  # reorder ALU/VECTORIZE
  (UPat(GroupOp.ALU, src=(UPat(Ops.STACK, src=UPat(name='x')), UPat(Ops.STACK, src=UPat(name='y'))), name='alu'),
   lambda x,y,alu: UOp(Ops.STACK, src=(UOp(alu.op, src=(x,y)),))),
  # ** where **
  # push cast to branches
  (UPat.var("s").where(UPat.var("a"), UPat.var("b")).cast().named("cast"), lambda s,a,b,cast: s.where(a.cast(cast.dtype), b.cast(cast.dtype))),
  # ** pow **
  ((UPat(Ops.POW, name="p"), lambda p: xpow(*p.src))),
  # ** load/store folding **
  (UPat.store(UPat(Ops.INDEX, name="index"), UPat.load(UPat(Ops.INDEX, name="index"))), lambda index: UOp(Ops.NOOP)),
  (UPat.store(UPat(Ops.INDEX, name="index"), UPat.var("gate").where(UPat.var("alt"),
                                                                    UPat.load(UPat(Ops.INDEX, name="index")))),
   lambda index, gate, alt: UOp.store(index.src[0].index(index.src[1].valid(gate)), alt)),
  # fold gated LOAD/STORE
  (UPat(Ops.STORE, src=(UPat(), invalid_pat)), lambda i: UOp(Ops.NOOP)),
  # store of where with invalid -> gated store
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, name="index"), UPat.var("cond").where(UPat.var("val"), invalid_pat))),
   lambda index, cond, val, i: UOp.store(index.src[0].index(index.src[1].valid(cond)), val)),
  ((UPat.var("x") * UPat.var("x")).reciprocal(), lambda x: x.reciprocal()*x.reciprocal()),  # 1/(x^c) -> (1/x)^c
  ((UPat.var("x") * UPat.var("x") * UPat.var("x")).reciprocal(), lambda x: x.reciprocal()*x.reciprocal()*x.reciprocal()),
  ((UPat.var("x") * UPat.cvar("c")).reciprocal(), lambda x,c: x.reciprocal()*c.reciprocal()), # 1/(x*c) -> (1/c)*(1/x)
  (UPat.var("x") * ((1+UPat.var("x")).reciprocal().named("d")), lambda x,d: 1-d), # x*/(1+x) -> 1-1/(1+x)
  (UPat.var("x") * ((1+UPat.var("x")).reciprocal().named("d")*UPat.var("y")), lambda x,y,d: y*(1-d)),
  (UPat.var("x") * ((1+UPat.var("x")).reciprocal().named("d")+UPat.var("y")), lambda x,y,d: (1-d)+x*y),
  # move const multiply after REDUCE (NOTE: the mul chain can do this, but only if it's a same dtype reduce)
  ((UPat.var("x")*UPat.cvar("c")).reduce(arg=Ops.ADD, name="r", allow_any_len=True), lambda x,c,r: r.replace(src=(x,)+r.src[1:])*c.val),
  # reduce mul chain, move muls after the reduce
  (UPat(Ops.MUL).reduce(name="r", allow_any_len=True), reduce_mul_chain),
  # ** combine terms (opinionated) **
  (-1 * (UPat.var("x") + UPat.var("y")), lambda x,y: (-x)+(-y)),  # -(x+y) -> -x + -y
  # (x+y)*c -> x*c+y*c. only for int, float has inf*0=nan issue
  ((UPat.var("x", dtypes.weakint) + UPat.var("y")) * UPat.cvar("c"), lambda x,y,c: x*c+y*c),
])+pm_clean_up_group_sink
