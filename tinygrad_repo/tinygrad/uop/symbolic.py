# all of symbolic lives here now
import math
from collections import defaultdict
from tinygrad.uop.ops import Ops, PatternMatcher, UPat, UOp, GroupOp, exec_alu, promo_dtype, identity_element
from tinygrad.dtype import PyConst, ConstFloat, dtypes, can_lossless_cast, Invalid, bitcast, truncate
from tinygrad.helpers import partition, all_same, prod, flatten, IMAGE, dedup, ceildiv
from tinygrad.uop.divandmod import div_and_mod_symbolic
from tinygrad.uop.movement import mop_cleanup
from tinygrad.uop.weak import pm_uncast_const

# TODO: symbolic shouldn't be importing from codegen
from tinygrad.codegen.decomp.transcendental import xpow, xpow_int

# ******** phase 1 of symbolic used to live in ops, it's the most generic folding rules ********

def simplify_pow(x:UOp, c:UOp) -> UOp|None:
  if c.val < 0 and dtypes.is_int(x.dtype): return None
  if c.val < 0: return x.reciprocal().pow(-c.val)
  if c.val == 0: return x.const_like(1)
  if (h := c.val-0.5) < c.val and int(h)+0.5 == c.val: return x.pow(h) * x.sqrt()
  if int(c.val) == c.val: return (y := x.pow(c.val//2)) * y * (x if c.val%2 == 1 else 1)
  return None

def fold_bitcast(root:UOp, c:UOp) -> UOp|None:
  if c.dtype.itemsize != root.dtype.itemsize: return None
  # the value is mathematical and may not fit: reading it as bits is the emission that pins it to the stated width
  return root.const_like(bitcast(truncate[c.dtype](c.val), c.dtype, root.dtype))

# the mint truncates to a strong dtype, ints wrap and floats re-round
def fold_const_alu(a:UOp) -> UOp: return a.const_like(exec_alu(a.op, a.dtype, [const_arg(s) for s in a.src]))
# a fold mints a signed const only if it fits, the mint would wrap it. unsigned ring ops wrap by definition
def fits(x:UOp, v:PyConst) -> bool: return x.dtype not in dtypes.sints or x.dtype.min <= v <= x.dtype.max
# a rewrite that introduces a signed intermediate must keep its range in the dtype, overflow is UB
def no_overflow(u:UOp) -> bool: return fits(u, u.vmin) and fits(u, u.vmax)

def _quotient_base(q:UOp, base:UOp, div:int) -> UOp|None:
  # the B with q == B//div and B%div == base%div, or None. only such congruence is needed to recombine, and canonicalization
  # moves consts freely: the quotient may be merged ((x//c + a)//div -> (x + a*c)//(c*div) for div>0) and shifted ((y + k*D)//D == y//D + k)
  (q, s), (num, a) = q.pop_const(), base.pop_const()
  if q.op is not Ops.DIV or q.src[1].op is not Ops.CONST: return None
  if div > 0 and num.op is Ops.DIV and num.src[1].op is Ops.CONST and q.src[1].val == (c:=num.src[1].val)*div: num, a, D = num.src[0], a*c, c*div
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
    if mod.op is not Ops.MOD or mod.src[1].op is not Ops.CONST: continue
    base, div = mod.src[0], mod.src[1].val
    for j,v in enumerate(terms):
      q, scale = v.pop_const(Ops.MUL)
      if i == j or scale != div*mul: continue
      rest = [t for k,t in enumerate(terms) if k not in (i,j)]
      if (b:=_quotient_base(q, base, div)) is not None: return (b*mul).usum(*rest)
      if q.op is Ops.MOD and q.src[1].op is Ops.CONST and (d:=q.src[1].val) > 0 and (b:=_quotient_base(q.src[0], base, div)) is not None:
        return ((b % (div*d))*mul).usum(*rest)
  return None

# Invalid poisons the value: ops move inside the gate so the Invalid reaches the LOAD/STORE and folds there.
# this needs to be before symbolic so that 0*something_that_might_be_invalid doesnt become 0
invalid_pat = UPat(Ops.CONST, arg=Invalid, name="i")
invalid_gate = UPat.var("cond").where(UPat.var("x"), invalid_pat)

# the two const spellings: Invalid carries no width, so it rides bare inside either
bare_const = UPat.any(UPat(Ops.CONST), UPat(Ops.STACK, src=UPat(Ops.CONST)))
casted_const = UPat.any(p:=UPat(Ops.CAST, src=(UPat(Ops.CONST),)), UPat(Ops.STACK, src=UPat.any(p, UPat(Ops.CONST, arg=Invalid))))
def const_arg(u:UOp):
  if u.op is Ops.STACK: return tuple(const_arg(s) for s in u.src)
  return float(u.val) if isinstance(u.val, ConstFloat) else u.val

def lift_reduce_gate(red:UOp, cond:UOp, x:UOp, i:UOp) -> UOp|None:
  # a REDUCE moves inside the gate clauses without its ranges: they invalidate every lane at once, so that gate lifts out
  # the clauses with its ranges invalidate single lanes, an Invalid lane is the reduce's identity
  op, horizontal = red.arg
  if horizontal != 0: return None
  keep, lift = partition(cond.split_uop(Ops.AND), lambda c: any(rr in c.ranges for r in red.src[1:] for rr in r.ranges))
  inner = keep[0].uprod(*keep[1:]).where(x, x.const_like(identity_element(op, red.dtype))) if keep else x
  ret = red.replace(src=(inner,)+red.src[1:])
  return lift[0].uprod(*lift[1:]).where(ret, i) if lift else ret

pm_data_invalid = PatternMatcher([
  (invalid_pat.broadcast(name="s"), lambda i,s: i.expand(s.shape)),
  (UPat(GroupOp.Unary|{Ops.CAST, Ops.BITCAST}, src=(invalid_pat,)), lambda i: i),
  (UPat(GroupOp.Unary|{Ops.CAST, Ops.BITCAST}, src=(invalid_gate,), name="op"),
   lambda cond,x,op,i: cond.where(op.replace(src=(x,)), i)),
  # binary ops move inside the gate, with Invalid in the false branch
  (UPat(GroupOp.Binary, src=(invalid_gate, UPat.var("y")), name="alu"), lambda cond,x,y,alu,i: cond.where(x.alu(alu.op,y), i)),
  (UPat(GroupOp.Binary, src=(UPat.var("y"), invalid_gate), name="alu"), lambda cond,x,y,alu,i: cond.where(y.alu(alu.op,x), i)),
  (UPat(GroupOp.Binary-GroupOp.Comparison, src=[invalid_pat, UPat()]), lambda i: i),
  (invalid_gate.reduce(allow_any_len=True, name="red"), lift_reduce_gate),
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
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(), invalid_pat), allow_any_len=True).or_casted(), UPat()), name="x"),
    lambda i,x: UOp(Ops.NOOP).after(*x.src)),
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat(), invalid_pat), allow_any_len=True).or_casted(),), allow_any_len=True, name="x"),
    lambda x,i: x.src[1] if len(x.src) > 1 else x.const_like(0)),
])

pm_remove_invalid = PatternMatcher([
  (invalid_gate.named("w"), lambda cond,x,i,w: w.replace(src=(cond,x,w.const_like(0)))),
  (UPat(Ops.STACK, name="s"), lambda s: s.replace(src=tuple(UOp.const(0, s.dtype) if x.is_invalid else x for x in s.src))),
])

def fold_const_where(gate:UOp, c0:UOp, c1:UOp, w:UOp) -> UOp:
  # folding a strong dtype WHERE to a weak const branch keeps the strong dtype
  ret = c0 if gate.val else c1
  return ret.ccast(w.dtype) if ret.op is Ops.CONST and ret.dtype in dtypes.weaks and w.dtype not in dtypes.weaks else ret

symbolic_simple = pm_data_invalid + PatternMatcher([
  # ** self folding **
  (UPat({Ops.ADD, Ops.XOR, Ops.OR}, src=[UPat.var("x"), UPat.const(0)]), lambda x: x),  # x+0 / x^0 / x|0 -> x
  (UPat({Ops.SHL, Ops.SHR}, src=(UPat.var("x"), UPat.const(0))), lambda x: x),          # x<<0 / x>>0 -> x
  (UPat.var("x") * 1, lambda x: x),    # x*1 -> x
  (UPat.var("x") // UPat.var("x"), lambda x: x.const_like(1)), # x//x -> 1
  (UPat.var("x") // 1, lambda x: x),   # x//1 -> x
  (UPat.var("x", dtypes.sints+(dtypes.weakint,)) // -1, lambda x: -x), # x//-1 -> -x
  ((UPat.var("x") ^ UPat.var("y", dtypes.weakint)) ^ UPat.var("y"), lambda x,y: x), # (x^y)^y -> x
  ((UPat.var() % UPat.var("y")).named("base") % UPat.var("y"), lambda base,y: base),  # (x%y)%y = -> x%y (rewritten with base for speed)
  # variations of (x%c)+(x//c)*c = x
  (UPat(Ops.ADD, dtype=dtypes.weakint, name="x"), fold_add_divmod_recombine),
  (UPat.var("x", dtype=dtypes.bool) & UPat.cvar("c"), lambda x,c: x if c.val else c),
  (UPat.var("x", dtype=dtypes.bool) | UPat.cvar("c"), lambda x,c: c if c.val else x),
  (UPat.var("x", dtype=dtypes.bool).eq(UPat.const(True, dtypes.bool)), lambda x: x),  # x == True -> x
  (UPat(GroupOp.Idempotent, src=(UPat.var("x"), UPat.var("x"))), lambda x: x),
  (UPat.var("x", dtype=dtypes.bool).logical_not().logical_not(), lambda x: x),
  (UPat.var("x", dtype=dtypes.bool).where(UPat.const(True, dtypes.bool), UPat.const(False, dtypes.bool)), lambda x: x),
  (UPat.var("x", dtype=dtypes.bool).where(UPat.const(False, dtypes.bool), UPat.const(True, dtypes.bool)), lambda x: x.logical_not()),
  # CAST(bool -> int) == const — CAST(True)=1, CAST(False)=0, so fold based on const value
  (UPat.var("x", dtype=dtypes.bool).cast(dtypes.wints).eq(UPat.cvar("c")),
   lambda x,c: x.logical_not() if c.val == 0 else x if c.val == 1 else x.const_like(False)),
  (UPat.var("x", dtype=dtypes.ints+(dtypes.bool, dtypes.weakint)).trunc(), lambda x: x),
  # ** zero folding **
  (UPat.var("x") < UPat.var("x"), lambda x: x.const_like(False, dtypes.bool)), # x < x -> False
  (UPat.var("x") % UPat.var("x"), lambda x: x.const_like(0)), # x%x -> 0
  (UPat.var("x") ^ UPat.var("x"), lambda x: x.const_like(0)), # x^x -> 0
  (UPat.var("x") & 0, lambda x: x.const_like(0)), # x&0 -> 0
  # (x&mask)>>k -> x>>k when mask only clears bits below k
  ((UPat.var("x") & UPat.cvar("mask")) >> UPat.cvar("k"),
   lambda x,mask,k: x >> k.val if x.dtype.const(mask.val | ((1 << k.val) - 1)) == x.dtype.const(-1) else None),
  (UPat.var("x", dtype=dtypes.ints+(dtypes.bool, dtypes.weakint)).eq(UPat.var("x")),
   lambda x: x.const_like(True, dtypes.bool)), # x == x -> True (only ints)
  # ** constant folding **
  # canonicalize casted CONST
  (UPat(Ops.CAST, dtypes.all, name="root", src=(UPat.cvar("c"),)), lambda root, c: root.const_like(c.val)),
  # collapse committed const conversions
  (UPat(Ops.CAST, dtypes.all, name="root", src=(UPat(Ops.CAST, dtypes.all, src=(UPat(Ops.CONST, name="c"),)),)),
   lambda root,c: root.const_like(c.val)),
  # one rule per spelling: bare has no width, a pair evaluates at its stated width, mixed commits to the promotion
  # NOTE: THREEFRY(const,const) folds via its decomposition
  (UPat(GroupOp.ALU-{Ops.THREEFRY}, src=bare_const, name="a"), fold_const_alu),
  (UPat(GroupOp.ALU-{Ops.THREEFRY}, src=casted_const, name="a"), fold_const_alu),
  (UPat(GroupOp.Binary-{Ops.THREEFRY}, src=[casted_const, bare_const], name="a"), lambda a:
   a.replace(src=tuple(s.ccast(dt) if s.dtype in dtypes.weaks else s for s in a.src))
   if (dt:=promo_dtype(a.src)) not in dtypes.weaks else None),
  # ** two stage ALU folding **
  *((UPat.var("x", dtypes.wints if op is Ops.ADD else None).alu(op, UPat.cvar("c1")).alu(op, UPat.cvar("c2")).named("f"),
     lambda f,x,c1,c2: x.alu(f.op,c1.alu(f.op,c2)) if fits(x, exec_alu(f.op, dtypes.weakint, [c1.vmin, c2.vmin])) else None)
    for op in GroupOp.Associative),
  # bool MUL is AND, ADD/MAX is OR. prevents other rules to rewrite bool ADD/MUL incorrectly
  (UPat.var('x', dtype=dtypes.bool) * UPat.var('y', dtype=dtypes.bool), lambda x,y: x&y),
  (UPat.var('x', dtype=dtypes.bool) + UPat.var('y', dtype=dtypes.bool), lambda x,y: x|y),
  (UPat.var('x', dtype=dtypes.bool).maximum(UPat.var('y', dtype=dtypes.bool)), lambda x,y: x|y),
  # *** div rules ***
  (UPat.cvar('x', arg=0) / 0, lambda x: x.const_like(float('nan'))),   # 0/0 -> nan
  # can be wrong if x or x2 is 0
  (UPat.var("x") / UPat.var("x"), lambda x: x.const_like(1)),          # x/x -> 1
  # x*0 -> 0 or 0*x -> 0
  # if x is nan or inf it should render the nan value.
  # NOTE: this can be wrong for loaded NaN
  (UPat.var("x") * 0, lambda x: x.const_like(float("nan") if x.op is Ops.CONST
                                             and isinstance(x.val, float) and (math.isnan(x.val) or math.isinf(x.val)) else 0)),
  # *** cast/bitcast ***
  (UPat((Ops.CAST, Ops.BITCAST), name="root"), lambda root: root.src[0] if root.dtype == root.src[0].dtype else None),
  # a BITCAST reads its operand at the width it states, so a weak const is nonsense here: the bare arm is bool only
  (UPat(Ops.BITCAST, name="root", src=(UPat.any(UPat(Ops.CONST, dtypes.bool, name="c"), UPat(Ops.CAST, src=(UPat(Ops.CONST),), name="c")),)),
   fold_bitcast),
  # b.cast(a).cast(b) -> b if a preserves all values in b
  (UPat.var('x').cast(name="a").cast(name="b"), lambda x,a,b: x if x.dtype == b.dtype and can_lossless_cast(b.dtype, a.dtype) else None),
  # bitcast twice
  (UPat(Ops.BITCAST, name="b", src=(UPat.var('x').bitcast(),)), lambda x,b: x.bitcast(b.dtype)),
  (UPat.var("x").cast(dtypes.bool), lambda x: x != 0),
  # ** pow **
  (UPat.var("x").alu(Ops.POW, UPat.cvar("c")), simplify_pow),
  # positive const ** x
  (UPat.cvar("c").alu(Ops.POW, UPat.var("x")),
   lambda c,x: (x*math.log2(c.val)).exp2() if c.val > 0 and dtypes.is_float(x.dtype) else None),
  # unpack a uint64 packed from two uint32 (threefry)
  (((UPat.var(None, dtypes.uint64)<<32) | UPat.var('y', dtypes.uint32).cast(dtypes.uint64)).cast(dtypes.uint32), lambda y: y),
  (((UPat.var('x', dtypes.uint64)<<32) | UPat.var(None, dtypes.uint32).cast(dtypes.uint64))>>32,
   lambda x: x.cast(dtypes.uint32).cast(dtypes.uint64)),
  # ** simple where folding **
  # a conditional with the same results either way is a noop, also fold const conditionals
  (UPat.var().where(UPat.var("val"), UPat.var("val")), lambda val: val),
  (UPat.cvar("gate").where(UPat.var("c0"), UPat.var("c1")).named("w"), fold_const_where),
])+mop_cleanup

# ******** phase 2 builds on phase 1, it includes the old "symbolic", rules that match deeper ********

def lt_folding(x:UOp, c:int) -> UOp|None:
  p, np = partition([u.pop_const(Ops.MUL) for u in x.split_uop(Ops.ADD)], lambda uf: uf[1] == 1)
  # d*y+p < c <=> y < ceildiv(c-p, d), so it folds when that bound is the same for p.vmin and p.vmax
  if np and (d:=math.gcd(*[int(f) for _,f in np])) > 1 and (t:=ceildiv(c-sum(u.vmin for u,_ in p), d)) == ceildiv(c-sum(u.vmax for u,_ in p), d):
    return UOp.usum(*[u*(f//d) for u,f in np])<t
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
  (UPat(GroupOp.Commutative, dtype=dtypes.weakint, name='x'), lambda x: x.replace(src=x.src[::-1]) if x.src[1].tuplize < x.src[0].tuplize else None),
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
  # (x*c0)+(x*c1) -> x*(c0+c1)
  (UPat.var("x", dtypes.wints) * UPat.cvar("c0") + UPat.var("x") * UPat.cvar("c1"), lambda x,c0,c1: x*(c0+c1) if fits(x, c0.vmin+c1.vmin) else None),
  ((UPat.var("y") + UPat.var("x", dtypes.wints) * UPat.cvar("c0")) + UPat.var("x") * UPat.cvar("c1"),
   lambda x,y,c0,c1: y+m if fits(x, c0.vmin+c1.vmin) and no_overflow(m:=x*(c0+c1)) else None),
  (UPat.var("x") + UPat.var("x") * UPat.cvar("c"), # (x+x*c)-> x*(c+1)
   lambda x,c: x*(c+1) if (x.dtype in dtypes.wints or c.arg == -1) and fits(x, c.vmin+1) else None),
  ((UPat.var("y") + UPat.var("x", dtypes.wints)) + UPat.var("x") * UPat.cvar("c"),
   lambda x,y,c: y+m if fits(x, c.vmin+1) and no_overflow(m:=x*(c+1)) else None),
  ((UPat.var("y") + UPat.var("x", dtypes.wints) * UPat.cvar("c")) + UPat.var("x"),
   lambda x,y,c: y+m if fits(x, c.vmin+1) and no_overflow(m:=x*(c+1)) else None),
  (UPat.var("x") + UPat.var("x"), lambda x: x*2), # (x+x)-> x*2
  (((UPat.var("y") + UPat.var("x", dtypes.wints)) + UPat.var("x")).named("a"),
   lambda a,y,x: y+x*2 if fits(a, 2*x.vmin) and fits(a, 2*x.vmax) else None),
  (UPat.cvar("y") * (UPat.var("x", dtype=dtypes.weakint) + UPat.cvar("c")), lambda x,y,c: (y*x)+(y*c)),  # y*(x+c) -> y*x + y*c
  # ** where folding **
  (UPat.var("cond", dtype=dtypes.bool).logical_not().where(UPat.var("t"), UPat.var("f")),
   lambda cond, t, f: cond.where(f,t) if not f.is_invalid else None),
  # in cond.where(t, f), uses of cond fold to True within t and False within f
  (UPat.var("cond").where(UPat.var("t"), UPat.var("f")), fold_where_closure),
  (UPat.var("gate").where(UPat.var("x"), 0) != 0, lambda gate,x: gate & (x != 0)),
  # a.where(b.where(c, d), d) -> (a & b).where(c, d)
  (UPat.var("a").where(UPat.var("b").where(UPat.var("c"), UPat.var("d")), UPat.var("d")), lambda a,b,c,d: (a&b).where(c,d)),
  # a.where(b.where(c, d), c) -> (a & !b).where(d, c)
  (UPat.var("a").where(UPat.var("b").where(UPat.var("c"), UPat.var("d")), UPat.var("c")), lambda a,b,c,d: (a&b.logical_not()).where(d,c)),
  # a.where(c, b.where(c, d)) -> (a | b).where(c, d)
  (UPat.var("a").where(UPat.var("c"), UPat.var("b").where(UPat.var("c"), UPat.var("d"))), lambda a,b,c,d: (a|b).where(c,d)),
  # alu of two where with same conds can combine, only do if true branch or false branch is const
  (UPat(GroupOp.Binary, name="alu", src=(UPat.var("c").where(UPat.var("t"), UPat.var("f")), UPat.var("c").where(UPat.var("tt"), UPat.var("ff")))), \
   lambda alu,c,t,tt,f,ff: c.where(t.alu(alu.op, tt), f.alu(alu.op, ff)) if t.op == tt.op == Ops.CONST or f.op == ff.op == Ops.CONST else None),
  # if its a plus we add the associative variation too
  ((UPat.var("y", dtypes.wints)+UPat.var("c").where(UPat.var("t"), UPat.var("f"))) + UPat.var("c").where(UPat.var("tt"), UPat.var("ff")), \
   lambda y,c,t,tt,f,ff: y+m if (t.op == tt.op == Ops.CONST or f.op == ff.op == Ops.CONST) and no_overflow(m:=c.where(t+tt, f+ff)) else None),
  # complementary zero branches under the same condition select directly
  (UPat.var("c").where(UPat.var("t"), 0) + UPat.var("c").where(0, UPat.var("f")), lambda c,t,f: c.where(t, f)),
  # ALU/variable min==max -> CONST
  (UPat({Ops.CMPLT, Ops.CMPEQ, Ops.DIV, Ops.MOD, Ops.PARAM, Ops.SPECIAL}, name="x"),
   lambda x: x.const_like(x.vmin) if x.dtype is not dtypes.void and x.vmin == x.vmax else None),
  (UPat(Ops.RANGE, src=(UPat.cvar().or_casted(),), name="x"), lambda x: x.const_like(x.vmin) if x.vmin == x.vmax else None),
  # max folding
  ((UPat.cvar("a") < UPat.var("b")).where(UPat.var("b"), UPat.cvar("c")), lambda a,b,c: UOp.maximum(a,b) if a.val == c.val else None),
  # (a<b).where(c, a+(c-b)) -> max(a+(c-b), c), the shift needs a signed int so the unselected arm can't wrap
  ((UPat.var("a") < UPat.cvar("b")).where(UPat.cvar("c"), UPat.var("y")), lambda a,b,c,y: UOp.maximum(y,c) if (y is a and b.val == c.val) or
   (dtypes.is_int(a.dtype) and not dtypes.is_unsigned(a.dtype) and y is a+(c.val-b.val)) else None),
  (UPat.maximum(UPat.var("x"), UPat.var("y")), lambda x,y: x if x.vmin >= y.vmax else None),
  # TODO: why does this rule break beautiful_mnist?
  #((UPat.var("x")+UPat.var("z")).maximum(UPat.var("y")+UPat.var("z")), lambda x,y,z: x.maximum(y) + z),
  # (x//c1)//c2 -> x//(c1*c2) for c2>0 when c1*c2 fits
  ((UPat.var("x") // UPat.cvar("c1")) // UPat.cvar("c2"),
   lambda x,c1,c2: x//(c1*c2) if c2.vmin>0 and x.dtype.min<=c1.vmin*c2.vmin<=x.dtype.max else None),
  # ** lt **
  # c0+x<c1 -> x < c1-c0
  ((UPat.cvar("c0") + UPat.var("x", dtype=dtypes.sints+(dtypes.weakint,))) < UPat.cvar("c1"),
   lambda x,c0,c1: x<(c1-c0) if fits(x, c1.vmin-c0.vmin) else None),
  # x//d<c -> x<c*d for d>0
  ((UPat.var("x", dtype=dtypes.weakint)//UPat.cvar("d"))<UPat.cvar("c"),
   lambda x,d,c: (x<c.val*d.val) if d.val > 0 else None),
  # ** move add/mul consts to end (NOTE: this is still happening before constant folding) **
  ((UPat.var("x", dtypes.wints) + UPat.cvar("c1")) + UPat.var("y"), lambda x,c1,y: m+c1 if y.op is not Ops.CONST and no_overflow(m:=x+y) else None),
  ((UPat.var("x") * UPat.cvar("c1")) * UPat.var("y"), lambda x,c1,y: m*c1 if y.op is not Ops.CONST and
   (x.dtype in dtypes.wints or c1.arg == -1) and no_overflow(m:=x*y) else None),
  # *** rules from symbolic ***
  # generic lt folding
  (UPat.var("x", dtypes.weakint)<UPat.cvar("c"), lambda x,c: lt_folding(x, c.val)),
  (UPat.var("x", dtypes.sints+(dtypes.weakint,))*-1 < UPat.var("y")*-1, lambda x,y: y<x),
  (UPat.var("x", dtypes.sints+(dtypes.weakint,))*-1 < UPat.cvar("c"), lambda x,c: -c<x if fits(x, -c.vmin) else None),
  # canonicalize a simplex with positive coefficients > 0. NOTE: not x < 1 means x > 0
  ((UPat.var("x", dtypes.weakint)<1).logical_not(), lambda x: (newx<1).logical_not() if (newx:=canonicalize_simplex(x)) is not None else None),
  # a range mod its own upper bound is just the range
  (UPat(Ops.RANGE, src=UPat.var("end"), name="r")%UPat.var("end"), lambda r,end: r),
  (UPat(Ops.RANGE, src=UPat.var("end"), name="r")//UPat.var("end"), lambda r,end: r.const_like(0)),
  # cast/long folding
  # if the intermediate cast doesnt narrow we can do it in one cast
  (UPat.var('x').cast(name="a").cast(name="b"), lambda x,a,b: x.cast(b.dtype) if can_lossless_cast(x.dtype, a.dtype) else None),
  (UPat.var('x', dtypes.wints).cast(dtypes.wints, name="a").cast(name="b"), lambda x,a,b: x.ccast(b.dtype) if not x.overflows(a.dtype) else None),
  # try to do math in int instead of long, keep weak const weak=
  (UPat(GroupOp.Binary, src=(UPat(dtype=(dtypes.long, dtypes.weakint)),)*2, name="u"), lambda u:
    UOp(u.op, src=tuple(UOp.const(s.val) if s.op is Ops.CONST else s.cast(dtypes.int) for s in u.src)).cast(u.dtype)
    if dtypes.long in (s.dtype for s in u.src) and not any(v.overflows(dtypes.int) for v in (u,*u.src))
    and not (u.op is Ops.SHR and u.src[1].vmax >= 32)
    and not (u.op in (Ops.MOD, Ops.TRUNCMOD) and u.src[0].vmin <= dtypes.int.min and u.src[1].vmin <= -1 <= u.src[1].vmax) else None),
  ((UPat.var("x", dtypes.weakint) + UPat.cvar("c")).cast(dtypes.sints, name="cast"), lambda x,c,cast:x.cast(cast.dtype)+cast.const_like(c.val)),
  # an AFTER waits only on the effect ops listed here, any other dep is replaced by its srcs
  (UPat(Ops.AFTER, name="x"), lambda x: x.replace(src=(x.src[0],)+
    tuple(dedup(flatten([(y,) if y.op in {Ops.RANGE, Ops.STORE, Ops.CALL, Ops.BARRIER, Ops.END, Ops.BACKEDGE, Ops.LINEAR, Ops.STAGE}
                        else (UOp(Ops.NOOP).after(y.src[0]).end(*y.src[1:]),) if y.op is Ops.REDUCE else y.src for y in x.src[1:]]))))),
  # after/end with 1 src is just src[0]
  (UPat((Ops.AFTER, Ops.END), src=(UPat.var("s"),)), lambda s: s),
  # the rules above key on bare CONSTs, so a redundantly committed const has to be uncast in the same fixpoint
])+div_and_mod_symbolic+pm_uncast_const

# ******** we take a small aside to "simplify_valid" to rewrite valids ********

def parse_valid(v:UOp) -> tuple[UOp, bool, int]|None:
  # if it's X <= c, returns X, True, c
  # if it's X >= c, returns X, False, c

  # a<b gives a<=b.vmax-1 (b>=a+1 for a const a), its negation a>=b.vmin (b<=a for a const a)
  neg = v.op is Ops.CMPEQ and v.src[1].op is Ops.CONST and v.src[1].val == 0
  if (lt:=v.src[0] if neg else v).op is not Ops.CMPLT or not dtypes.is_int(lt.src[0].dtype): return None
  if (a:=lt.src[0]).op is Ops.CONST: return lt.src[1], neg, int(a.val) + (0 if neg else 1)
  return a, not neg, int(lt.src[1].vmin) if neg else int(lt.src[1].vmax)-1

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
    all_candidates.append((expr, UOp.variable(f"fake{i}", v0, v1, expr.dtype)))

    if try_simplex and expr.op is Ops.ADD and v0 == 1 and all(u.op in GroupOp.Irreducible and u.op is not Ops.CONST for u in expr.split_uop(Ops.ADD)):
      # For X0 + X1 + ... > 0, check whether every Xi > 0 gives the same simplified output.
      candidate = [(Xi, UOp.variable(f"fake{i}", 1, Xi.vmax, Xi.dtype)) for Xi in expr.split_uop(Ops.ADD)]
      if any(X not in uop.backward_slice_with_self for X,_ in candidate): continue
      newuops = [uop.substitute({X:newX}).simplify().substitute({newX:X}).simplify() for X,newX in candidate]
      if all_same(newuops): uop = newuops[0]
      elif uop.op is Ops.STACK and len(uop.src) == 2:
        if all_same([uops.src[0] for uops in newuops]): uop = uop.replace(src=(newuops[0].src[0], uop.src[1]))
        if all_same([uops.src[1] for uops in newuops]): uop = uop.replace(src=(uop.src[0], newuops[0].src[1]))

  # try all the valids together (but only the whole expressions)
  if (s_uop:=uop.substitute(sub_dict:=dict(all_candidates))) is not uop:
    uop = s_uop.simplify().substitute({newX:X for X,newX in sub_dict.items()}).simplify()
  return uop

def _valid_priority(v: UOp, valids:list[UOp]) -> tuple[int, int]:
  # prioritize dependencies, then tighter bounds, so weaker clauses don't hide useful simplifications
  return (0, 0) if (res:=parse_valid(v)) is None else (
    sum(-1 for other in valids if res[0] is other or res[0] in other.backward_slice), res[2] if res[1] else -res[2])

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
  # max(x*m) = max(x)*m needs x*m monotone in x, and an unsigned multiply wraps
  if r.arg[0] is Ops.MAX and dtypes.is_unsigned(r.dtype): return None
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
  # can move if: not a const, condition's ranges are subset of idx's ranges, and no data dependent INDEX (only idx's INDEX allowed)
  def can_move(c:UOp) -> bool:
    return c.op is not Ops.CONST and c.ranges.keys() <= idx.ranges.keys() and \
      all(u in idx_index for u in c.backward_slice_with_self if u.op is Ops.INDEX)
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
  # pure index math only: a LOAD in x executes even where cond is false, so its INDEX valid must survive the assumption
  if x.dtype is not dtypes.weakint or x.op_in_backward_slice_with_self(Ops.INDEX): return None
  # Skip if x contains DIV/MOD AND IMAGE mode is enabled -> image index e.g. openpilot
  if IMAGE.value > 0 and x.op_in_backward_slice_with_self(Ops.TRUNCDIV, Ops.TRUNCMOD, Ops.DIV, Ops.MOD): return None
  return cond.where(uop_given_valid(cond, x, try_simplex=False), i)

pm_simplify_valid = PatternMatcher([
  # simplify valid
  (UPat(Ops.AND, dtypes.bool, name="valid"), simplify_valid),
  (invalid_gate, gated_given_valid),
])

# this is symbolic 2.0
REMOVE_FROM_SINK_LIKE = {Ops.NOOP, Ops.STACK, Ops.SINK, Ops.AFTER}
pm_clean_up_group_sink = PatternMatcher([
  # clean up SINK
  (UPat(Ops.SINK, name="root"),
    lambda root: UOp(root.op, src=tuple(flatten(x.src if x.op in REMOVE_FROM_SINK_LIKE else (x,) for x in root.src)), arg=root.arg)
      if any(x.op in REMOVE_FROM_SINK_LIKE for x in root.src) else None),
])

sym = symbolic+pm_simplify_valid+PatternMatcher([
  # ** pow **
  ((UPat(Ops.POW, name="p"), lambda p: xpow_int(p.src[0], p.src[1], p.dtype) if dtypes.is_int(p.dtype) else xpow(*p.src))),
  # ** load/store folding **
  (UPat.store(UPat(Ops.INDEX, name="index"), UPat.load(UPat(Ops.INDEX, name="index")), name="x"), lambda index,x: UOp(Ops.NOOP).after(*x.src)),
  (UPat.store(UPat(Ops.INDEX, name="index"), UPat.var("gate").where(UPat.var("alt"),
                                                                    UPat.load(UPat(Ops.INDEX, name="index")))),
   lambda index, gate, alt: UOp.store(index.src[0].index(index.src[1].valid(gate)), alt)),
  (UPat.store(UPat(Ops.INDEX, name="index"), UPat.var("gate").where(UPat.load(UPat(Ops.INDEX, name="index")), UPat.var("alt"))),
   lambda index, gate, alt: UOp.store(index.src[0].index(index.src[1].valid(gate.logical_not())), alt)),
  # fold gated LOAD/STORE
  (UPat(Ops.STORE, src=(UPat(), invalid_pat), name="x"), lambda i,x: UOp(Ops.NOOP).after(*x.src)),
  # store of where with invalid -> gated store
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, name="index"), UPat.var("cond").where(UPat.var("val"), invalid_pat))),
   lambda index, cond, val, i: UOp.store(index.src[0].index(index.src[1].valid(cond)), val)),
  (UPat.var("x") * ((1+UPat.var("x")).reciprocal().named("d")), lambda x,d: 1-d), # x*/(1+x) -> 1-1/(1+x)
  (UPat.var("x") * ((1+UPat.var("x")).reciprocal().named("d")*UPat.var("y")), lambda x,y,d: y*(1-d)),
  (UPat.var("x") * ((1+UPat.var("x")).reciprocal().named("d")+UPat.var("y")), lambda x,y,d: (1-d)+x*y),
  # reduce mul chain, move muls after the reduce
  (UPat(Ops.MUL).reduce(name="r", allow_any_len=True), reduce_mul_chain),
  # ** combine terms (opinionated) **
  # (x+y)*c -> x*c+y*c. only for int, float has inf*0=nan issue
  ((UPat.var("x", dtypes.weakint) + UPat.var("y")).named("a") * UPat.cvar("c"),
   lambda a,x,y,c: x*c+y*c if all(fits(a, v*c.vmin) for v in (x.vmin, x.vmax, y.vmin, y.vmax)) else None),
])+pm_clean_up_group_sink
