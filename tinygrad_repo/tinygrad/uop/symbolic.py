# all of symbolic lives here now
import math, struct
from collections import defaultdict
from tinygrad.uop.ops import Ops, PatternMatcher, UPat, UOp, GroupOp, exec_alu
from tinygrad.dtype import PyConst, ConstType, dtypes, can_lossless_cast, Invalid
from tinygrad.helpers import partition, all_same, prod, flatten, unwrap, IMAGE, dedup
from tinygrad.uop.divandmod import div_and_mod_symbolic
from tinygrad.uop.movement import mop_cleanup

# TODO: symbolic shouldn't be importing from codegen
from tinygrad.codegen.decomp.transcendental import xpow

# ******** phase 1 of symbolic used to live in ops, it's the most generic folding rules ********

def simplify_pow(x:UOp, c:UOp) -> UOp|None:
  if c.arg < 0: return x.reciprocal().pow(-c)
  if c.arg == 0: return x.const_like(1)
  if int(c.arg-0.5)+0.5 == c.arg: return x.pow(c.const_like(c.arg-0.5)) * x.sqrt()
  if int(c.arg) == c.arg: return (y := x.pow(c.const_like(c.arg//2))) * y * (x if c.arg%2 == 1 else 1)
  return None

def fold_bitcast(root:UOp, c:UOp) -> UOp|None:
  if (from_fmt:=c.dtype.fmt) is None or (to_fmt:=root.dtype.fmt) is None: return None
  if c.dtype.itemsize != root.dtype.itemsize: return None
  def convert(v:ConstType) -> ConstType: return struct.unpack(to_fmt, struct.pack(from_fmt, v))[0]
  return root.const_like(convert(c.arg))

def const_arg(u:UOp) -> ConstType|tuple[ConstType, ...]|None:
  if u.op is Ops.CONST: return u.arg
  if u.op is Ops.STACK and all(s.op is Ops.CONST for s in u.src): return tuple(s.arg for s in u.src)
  return None

def fold_const_alu(a:UOp) -> UOp|None:
  vals = [const_arg(s) for s in a.src]
  return None if any(v is None for v in vals) else a.const_like(exec_alu(a.op, a.dtype, vals, False))

def _quotient_base(q:UOp, base:UOp, div:int) -> UOp|None:
  # the B with q == B//div and B%div == base%div, or None. only such congruence is needed to recombine, and canonicalization
  # moves consts freely: the quotient may be merged ((x//c + a)//div -> (x + a*c)//(c*div) for div>0) and shifted ((y + k*D)//D == y//D + k)
  (q, s), (num, a) = q.pop_const(), base.pop_const()
  if q.op is not Ops.FLOORDIV or q.src[1].op is not Ops.CONST: return None
  if div > 0 and num.op is Ops.FLOORDIV and num.src[1].op is Ops.CONST and q.src[1].arg == (c:=num.src[1].arg)*div: num, a, D = num.src[0], a*c, c*div
  elif q.src[1].arg == div: D = div
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
    base, div = mod.src[0], mod.src[1].arg
    for j,v in enumerate(terms):
      q, scale = v.pop_const(Ops.MUL)
      if i == j or scale != div*mul: continue
      rest = [t for k,t in enumerate(terms) if k not in (i,j)]
      if (b:=_quotient_base(q, base, div)) is not None: return (b*mul).usum(*rest)
      if q.op is Ops.FLOORMOD and q.src[1].op is Ops.CONST and (d:=q.src[1].arg) > 0 and (b:=_quotient_base(q.src[0], base, div)) is not None:
        return ((b % (div*d))*mul).usum(*rest)
  return None

# an invalid index is cond.where(idx, Invalid) in index. the consumer reads cond back off the WHERE with UOp.get_valid,
# so casts and comparisons of a gated index can drop the gate: when the index is invalid the result is never used
invalid_idx_gate = UPat().where(UPat.var("x"), UPat(Ops.CONST, dtypes.weakint, arg=Invalid))
pm_index_invalid = PatternMatcher([
  (invalid_idx_gate.cast(name="cast"), lambda x,cast: x.cast(cast.dtype)),
  (UPat(GroupOp.Comparison, src=(invalid_idx_gate, UPat.var("y")), name="alu"), lambda x,y,alu: x.alu(alu.op,y)),
  (UPat(GroupOp.Comparison, src=(UPat.var("y"), invalid_idx_gate), name="alu"), lambda x,y,alu: y.alu(alu.op,x)),
])

# everywhere else Invalid poisons the value: ops move inside the gate so the Invalid reaches the LOAD/STORE and folds there.
# this needs to be before symbolic so that 0*something_that_might_be_invalid doesnt become 0
invalid_pat = UPat(Ops.CONST, arg=Invalid, name="i")
invalid_gate = UPat.var("cond").where(UPat.var("x"), invalid_pat)
pm_data_invalid = PatternMatcher([
  (UPat(GroupOp.Unary|{Ops.BITCAST}, src=(invalid_pat,), name="op"), lambda i,op: i.cast(op.dtype)),
  (UPat(GroupOp.Unary|{Ops.BITCAST}, src=(invalid_gate,), name="op"), lambda cond,x,op,i: cond.where(op.replace(src=(x,)), i.cast(op.dtype))),
  # binary ops move inside the gate, with Invalid cast to the result dtype (bool for comparisons)
  (UPat(GroupOp.Binary, src=(invalid_gate, UPat.var("y")), name="alu"), lambda cond,x,y,alu,i: cond.where(x.alu(alu.op,y), i.cast(alu.dtype))),
  (UPat(GroupOp.Binary, src=(UPat.var("y"), invalid_gate), name="alu"), lambda cond,x,y,alu,i: cond.where(y.alu(alu.op,x), i.cast(alu.dtype))),
  (UPat(GroupOp.Binary-GroupOp.Comparison, src=[invalid_pat, UPat()]), lambda i: i),
  # an Invalid condition poisons the whole where; a gated Invalid condition lifts the gate out
  (invalid_pat.where(UPat.var("a"), UPat()), lambda i,a: i.cast(a.dtype)),
  (invalid_gate.where(UPat.var("a"), UPat.var("b")), lambda cond,x,i,a,b: cond.where(x.where(a,b), i.cast(a.dtype))),
  # normalize where(cond, Invalid, val) -> where(~cond, val, Invalid)
  (UPat.var("cond").where(invalid_pat, UPat.var("val")), lambda cond, i, val: cond.logical_not().where(val, i) if val.arg != Invalid else i),
  # lift Invalid out: a.where(cond.where(x, Invalid), c) -> (~a|cond).where(a.where(x, c), Invalid)
  # when a is cond, ~a|cond is True and would drop the Invalid gate (losing the valid), so keep cond as the gate
  (UPat.var("a").where(invalid_gate, UPat.var("c")), lambda cond,i,x,a,c:
   (cond if a is cond else (a.logical_not()|cond)).where(a.where(x,c), i) if c.arg != Invalid else None),
  (UPat.var("a").where(UPat.var("b"), invalid_gate), lambda cond,i,x,a,b: (a|cond).where(a.where(b, x), i) if b.arg != Invalid else None),
  # fold gated LOAD/STORE
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(), invalid_pat), allow_any_len=True).or_casted(), UPat())), lambda i: UOp(Ops.NOOP)),
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat(), invalid_pat), allow_any_len=True).or_casted(),), allow_any_len=True, name="x"),
    lambda x,i: x.src[1] if len(x.src) > 1 else x.const_like(0)),
])

propagate_invalid = pm_index_invalid + pm_data_invalid

pm_remove_invalid = PatternMatcher([
  (invalid_pat, lambda i: i.const_like(0)),
])

symbolic_simple = propagate_invalid + PatternMatcher([
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
  (UPat.var("x", dtype=dtypes.bool) & UPat.cvar("c"), lambda x,c: x if c.arg else c),
  (UPat.var("x", dtype=dtypes.bool) | UPat.cvar("c"), lambda x,c: c if c.arg else x),
  (UPat(GroupOp.Idempotent, src=(UPat.var("x"), UPat.var("x"))), lambda x: x),
  (UPat.var("x", dtype=dtypes.bool).logical_not().logical_not(), lambda x: x),
  (UPat.var("x", dtype=dtypes.bool).where(UPat.const(dtypes.bool, True), UPat.const(dtypes.bool, False)), lambda x: x),
  (UPat.var("x", dtype=dtypes.bool).where(UPat.const(dtypes.bool, False), UPat.const(dtypes.bool, True)), lambda x: x.logical_not()),
  # CAST(bool -> int) != const — CAST(True)=1, CAST(False)=0, so fold based on const value
  (UPat.var("x", dtype=dtypes.bool).cast(dtypes.ints+(dtypes.weakint,)) != UPat.cvar("c"),
   lambda x,c: x if c.arg == 0 else x.logical_not() if c.arg == 1 else x.const_like(True)),
  (UPat.var("x", dtype=dtypes.ints+(dtypes.bool, dtypes.weakint)).trunc(), lambda x: x),
  # ** zero folding **
  (UPat.var("x") < UPat.var("x"), lambda x: x.const_like(False, dtypes.bool)), # x < x -> False
  (UPat.var("x") % UPat.var("x"), lambda x: x.const_like(0)), # x%x -> 0
  (UPat.var("x") ^ UPat.var("x"), lambda x: x.const_like(0)), # x^x -> 0
  (UPat.var("x") & 0, lambda x: x.const_like(0)), # x&0 -> 0
  # (x&mask)>>k -> x>>k when mask only clears bits below k
  # TODO: combine this with "# rules for threefry" below
  ((UPat.var("x") & UPat.cvar("mask")) >> UPat.cvar("k"),
   lambda x,mask,k: x >> k.arg if mask.arg | ((1 << k.arg) - 1) == -1 else None),
  ((UPat.var("x") & UPat.cvar("mask")) // UPat.cvar("c"),
   lambda x,mask,c: x // c.arg if c.arg > 0 and c.arg & (c.arg-1) == 0 and mask.arg | (c.arg-1) == -1 else None),
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
                                             and isinstance(x.arg, float) and (math.isnan(x.arg) or math.isinf(x.arg)) else 0)),
  # *** cast/bitcast ***
  (UPat(Ops.CAST, name="root", src=(UPat.cvar("c"),)), lambda root, c: root.const_like(c.arg)),
  (UPat((Ops.CAST, Ops.BITCAST), name="root"), lambda root: root.src[0] if root.dtype == root.src[0].dtype else None),
  (UPat(Ops.BITCAST, name="root", src=(UPat.cvar("c"),)), fold_bitcast),
  # b.cast(a).cast(b) -> b if a preserves all values in b
  (UPat.var('x').cast(name="a").cast(name="b"), lambda x,a,b: x if x.dtype == b.dtype and can_lossless_cast(b.dtype, a.dtype) else None),
  (UPat.var("x").cast(dtypes.bool), lambda x: x != 0),
  # ** pow **
  (UPat.var("x").alu(Ops.POW, UPat.cvar("c")), simplify_pow),
  # positive const ** x
  (UPat.cvar("c").alu(Ops.POW, UPat.var("x")), lambda c,x: c if c.arg == 1 else (x*math.log2(c.arg)).exp2() if c.arg > 0 else None),
  # rules for threefry
  ((UPat.var('x', dtypes.uint64)&0xFFFFFFFF).cast(dtypes.uint32), lambda x: x.cast(dtypes.uint32)),
  (((UPat.var(None, dtypes.uint64)*(1<<32)) | UPat.var('y',  dtypes.uint32).cast(dtypes.uint64)).cast(dtypes.uint32), lambda y: y),
  (((UPat.var('x',  dtypes.uint64)*(1<<32)) | UPat.var(None, dtypes.uint32).cast(dtypes.uint64))//(1<<32), lambda x: x),
  (((UPat.var(None, dtypes.uint64)<<32) | UPat.var('y',  dtypes.uint32).cast(dtypes.uint64)).cast(dtypes.uint32), lambda y: y),
  (((UPat.var('x',  dtypes.uint64)<<32) | UPat.var(None, dtypes.uint32).cast(dtypes.uint64))//(1<<32), lambda x: x),
  (((UPat.var('x',  dtypes.uint64)<<32) | UPat.var(None, dtypes.uint32).cast(dtypes.uint64))>>32, lambda x: x),
  # ** simple where folding **
  # a conditional with the same results either way is a noop, also fold const conditionals
  (UPat.var().where(UPat.var("val"), UPat.var("val")), lambda val: val),
  (UPat.cvar("gate").where(UPat.var("c0"), UPat.var("c1")), lambda gate, c0, c1: c0 if gate.arg else c1),
  # a.where(b.where(c, d), d) -> (a & b).where(c, d)
  (UPat.var("a").where(UPat.var("b").where(UPat.var("c"), UPat.var("d")), UPat.var("d")), lambda a,b,c,d: (a&b).where(c,d)),
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
    if u.op is Ops.MUL and u.src[1].op is Ops.CONST and u.src[1].arg > 0:
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
   lambda cond, t, f: cond.where(f,t) if f.arg is not Invalid else None),
  # alu of two where with same conds can combine, only do if true branch or false branch is const
  (UPat(GroupOp.Binary, name="alu", src=(UPat.var("c").where(UPat.var("t"), UPat.var("f")), UPat.var("c").where(UPat.var("tt"), UPat.var("ff")))), \
   lambda alu,c,t,tt,f,ff: c.where(t.alu(alu.op, tt), f.alu(alu.op, ff)) if t.op == tt.op == Ops.CONST or f.op == ff.op == Ops.CONST else None),
  # if its a plus we add the associative variation too
  ((UPat.var("y")+UPat.var("c").where(UPat.var("t"), UPat.var("f"))) + UPat.var("c").where(UPat.var("tt"), UPat.var("ff")), \
   lambda y,c,t,tt,f,ff: y+c.where(t+tt, f+ff) if t.op == tt.op == Ops.CONST or f.op == ff.op == Ops.CONST else None),
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
  ((UPat.cvar("c0") + UPat.var("x")) < UPat.cvar("c1"), lambda x,c0,c1: x<(c1-c0)),  # c0 + x < c1 -> x < c1 - c0
  # (x//c1)//c2 -> x//(c1*c2) for c2>0
  ((UPat.var("x") // UPat.cvar("c1")) // UPat.cvar("c2"), lambda x,c1,c2: x//(c1*c2) if c2.vmin>0 else None),
  # ** lt **
  # c0*x<c1 for positive int c0,c1
  ((UPat.cvar("c0")*UPat.var("x", dtype=dtypes.weakint))<UPat.cvar("c1"),
   lambda x,c0,c1: x<math.ceil(c1.arg/c0.arg) if c0.arg > 0 and c1.arg > 0 else None),
  # c0*x<c1 for negative int c0 and non-positive c1
  ((UPat.cvar("c0")*UPat.var("x", dtype=dtypes.weakint))<UPat.cvar("c1"),
   lambda x,c0,c1: (-x)<(-(math.floor(-c1.arg/-c0.arg))) if c0.arg < 0 and c0.arg != -1 and c1.arg <= 0 else None),
  # x//d<c -> x<c*d for d>0
  ((UPat.var("x", dtype=dtypes.weakint)//UPat.cvar("d"))<UPat.cvar("c"),
   lambda x,d,c: x<(c.arg*d.arg) if d.arg > 0 else None),
  # ** move add/mul consts to end (NOTE: this is still happening before constant folding) **
  ((UPat.var("x") + UPat.cvar("c1")) + UPat.var("y"), lambda x,c1,y: (x+y)+c1),
  ((UPat.var("x") * UPat.cvar("c1")) * UPat.var("y"), lambda x,c1,y: (x*y)*c1),
  # *** rules from symbolic ***
  # generic lt folding
  (UPat.var("x", dtypes.weakint)<UPat.cvar("c"), lambda x,c: lt_folding(x, c.arg) if 0 < c.arg else None),
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
  # try to do math in int instead of long
  (UPat(GroupOp.Binary, src=(UPat.var("x", dtypes.long), UPat.var("y", dtypes.long)), name="u"), lambda u,x,y:
    x.cast(dtypes.int).alu(u.op, y.cast(dtypes.int)).cast(u.dtype) if not any(v.overflows(dtypes.int) for v in (u,x,y)) else None),
  ((UPat.var("x", dtypes.weakint) + UPat.cvar("c")).cast(dtypes.sints, name="cast"), lambda x,c,cast:x.cast(cast.dtype)+c.cast(cast.dtype)),
  # only RANGE/IF/STORE/KERNEL have side effects
  (UPat(Ops.AFTER, name="x"), lambda x: x.replace(src=(x.src[0],)+
    tuple(dedup(flatten([(y,) if y.op in {Ops.RANGE, Ops.LOOP, Ops.STORE, Ops.CALL, Ops.FUNCTION, Ops.BARRIER, Ops.END, Ops.LINEAR, Ops.STAGE}
                        else y.src for y in x.src[1:]]))))),
  # after with 1 src is just src[0]
  (UPat(Ops.AFTER, src=(UPat.var("s"),)), lambda s: s),
])+div_and_mod_symbolic

# ******** we take a small aside to "simplify_valid" to rewrite valids ********

def parse_valid(v:UOp) -> tuple[UOp, bool, int]|None:
  # if it's X <= c, returns X, True, c
  # if it's X >= c, returns X, False, c

  if v.op is Ops.CMPNE and v.src[1].op is Ops.CONST and v.src[1].arg == 1 and (s0:=v.src[0]).op is Ops.CMPLT and dtypes.is_int(s0.src[0].dtype):
    # (X < c).ne(True) -> X >= c
    return s0.src[0], False, int(s0.src[1].vmin)
  if v.op is Ops.CMPLT and dtypes.is_int(v.src[0].dtype):
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
        newuops = [uop.substitute({X:newX}) for X,newX in candidate]
        if any(u is uop for u in newuops): continue  # if any branch doesnt appear in uop, skip
        newuops = [u.simplify().substitute({newX:X}).simplify() for (X,newX),u in zip(candidate,newuops)]
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
  return UOp.const(dtypes.bool, True).uprod(*keep).where(x, i) if drop else None
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
  return UOp.const(dtypes.bool, True).uprod(*keep).where(ret_idx, ret_idx.const_like(0))

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

# TODO: this is O(number of WHERE * number of node)
# def fold_where_closure(cond:UOp, t:UOp, f:UOp) -> UOp|None:
#   """In cond.where(t, f), fold nested cond.where(a, b) -> a in t, -> b in f"""
#   def is_valid_where(u:UOp) -> bool: return u.op is Ops.WHERE and u.src[0] is cond and Invalid not in (u.src[1].arg, u.src[2].arg)
#   t_subs, f_subs = {u: u.src[1] for u in t.toposort() if is_valid_where(u)}, {u: u.src[2] for u in f.toposort() if is_valid_where(u)}
#   if not t_subs and not f_subs: return None
#   new_t, new_f = t.substitute(t_subs).simplify() if t_subs else t, f.substitute(f_subs).simplify() if f_subs else f
#   return None if new_t is t and new_f is f else cond.where(new_t, new_f)

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
  # # fold nested where with same condition: in cond.where(t,f), cond.where(a,b)->a in t, ->b in f
  # (UPat.var("cond").where(UPat.var("t"), UPat.var("f")), fold_where_closure),
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
  ((UPat.var("x")*UPat.cvar("c")).reduce(arg=Ops.ADD, name="r", allow_any_len=True), lambda x,c,r: r.replace(src=(x,)+r.src[1:])*c.arg),
  # reduce mul chain, move muls after the reduce
  (UPat(Ops.MUL).reduce(name="r", allow_any_len=True), reduce_mul_chain),
  # ** combine terms (opinionated) **
  (-1 * (UPat.var("x") + UPat.var("y")), lambda x,y: (-x)+(-y)),  # -(x+y) -> -x + -y
  # (x+y)*c -> x*c+y*c. only for int, float has inf*0=nan issue
  ((UPat.var("x", dtypes.weakint) + UPat.var("y")) * UPat.cvar("c"), lambda x,y,c: x*c+y*c),
])+pm_clean_up_group_sink
