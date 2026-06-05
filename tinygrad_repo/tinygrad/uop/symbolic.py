# all of symbolic lives here now
import math, struct
from collections import defaultdict
from tinygrad.uop.ops import Ops, PatternMatcher, UPat, UOp, GroupOp, exec_alu
from tinygrad.dtype import ConstType, dtypes, PtrDType, can_lossless_cast, Invalid
from tinygrad.helpers import partition, all_same, prod, flatten, get_single_element, unwrap, IMAGE, dedup
from tinygrad.uop.decompositions import threefry2x32, xpow
from tinygrad.uop.divandmod import div_and_mod_symbolic

# ******** phase 1 of symbolic used to live in ops, it's the most generic folding rules ********

def simplify_pow(x:UOp, c:UOp) -> UOp|None:
  if c.arg < 0: return x.reciprocal().pow(-c)
  if c.arg == 0: return x.const_like(1)
  if int(c.arg-0.5)+0.5 == c.arg: return x.pow(c.const_like(c.arg-0.5)) * x.sqrt()
  if int(c.arg) == c.arg: return (y := x.pow(c.const_like(c.arg//2))) * y * (x if c.arg%2 == 1 else 1)
  return None

def fold_bitcast(root:UOp, c:UOp) -> UOp|None:
  if (from_fmt:=c.dtype.scalar().fmt) is None or (to_fmt:=root.dtype.scalar().fmt) is None: return None
  if c.dtype.itemsize != root.dtype.itemsize: return None
  def convert(v:ConstType) -> ConstType: return struct.unpack(to_fmt, struct.pack(from_fmt, v))[0]
  return root.const_like(convert(c.arg) if root.dtype.count == 1 else tuple(map(convert, c.arg)))

def const_arg(u:UOp) -> ConstType|tuple[ConstType, ...]|None:
  if u.op is Ops.CONST: return u.arg
  if u.op is Ops.STACK and all(s.op is Ops.CONST for s in u.src): return tuple(s.arg for s in u.src)
  return None

def fold_const_alu(a:UOp) -> UOp|None:
  vals = [const_arg(s) for s in a.src]
  return None if any(v is None for v in vals) else a.const_like(exec_alu(a.op, a.dtype, vals, False))

invalid_pat = UPat(Ops.CONST, arg=Invalid, name="i")
invalid_gate = UPat.var("cond").where(UPat.var("x"), invalid_pat)

def fold_add_divmod_recombine(x:UOp) -> UOp|None:
  terms = list(x.split_uop(Ops.ADD))
  for i,u in enumerate(terms):
    if u.op is Ops.FLOORMOD and u.src[1].op is Ops.CONST: base, div, mul = u.src[0], u.src[1].arg, 1
    elif u.op is Ops.MUL and u.src[1].op is Ops.CONST and (m:=u.src[0]).op is Ops.FLOORMOD and m.src[1].op is Ops.CONST:
      base, div, mul = m.src[0], m.src[1].arg, u.src[1].arg
    else: continue
    for j,v in enumerate(terms):
      if i == j: continue
      if v.op is not Ops.MUL or v.src[1].op is not Ops.CONST or v.src[1].arg != div*mul: continue
      q, exact = v.src[0], False
      # (base%div)*mul + (base//div)*(div*mul) -> base*mul
      if q.op is Ops.FLOORDIV and q.src[1].op is Ops.CONST and q.src[1].arg == div: exact = q.src[0] is base
      # ((base//d)%div)*mul + (base//(d*div))*(div*mul) -> (base//d)*mul if div>0
      if not exact and div > 0 and base.op is Ops.FLOORDIV and base.src[1].op is Ops.CONST:
        exact = q.op is Ops.FLOORDIV and q.src[1].op is Ops.CONST and q.src[0] is base.src[0] and q.src[1].arg == base.src[1].arg*div
      if exact: return (base*mul).usum(*[t for k,t in enumerate(terms) if k not in (i,j)])
      # ((base//div)%d)*(div*mul) + (base%div)*mul -> (base%(div*d))*mul
      if div > 0 and q.op is Ops.FLOORMOD and q.src[1].op is Ops.CONST and (d:=q.src[1].arg) > 0 and q.src[0].op is Ops.FLOORDIV:
        if q.src[0].src[0] is base and q.src[0].src[1].op is Ops.CONST and q.src[0].src[1].arg == div:
          return ((base % (div*d))*mul).usum(*[t for k,t in enumerate(terms) if k not in (i,j)])
  return None

# this needs to be before symbolic so that 0*something_that_might_be_invalid doesnt become 0
propagate_invalid = PatternMatcher([
  # propagate invalid, push it past children
  (invalid_gate.cast(name="cast"), lambda i,x,cond,cast: x.cast(cast.dtype) if i.dtype is dtypes.weakint else None),
  (UPat(GroupOp.Unary, src=(invalid_gate,), name="alu"), lambda cond,x,alu,i: cond.where(x.alu(alu.op), i)),
  (UPat(GroupOp.Binary-GroupOp.Comparison, src=(invalid_gate, UPat.var("y")), name="alu"), lambda cond,x,y,alu,i: cond.where(x.alu(alu.op,y), i)),
  (UPat(GroupOp.Binary-GroupOp.Comparison, src=(UPat.var("y"), invalid_gate), name="alu"), lambda cond,x,y,alu,i: cond.where(y.alu(alu.op,x), i)),
  # TODO: when can this happen? and is it always safe to just drop invalid?
  (UPat(GroupOp.Comparison, src=(invalid_gate, UPat.var("y")), name="alu"), lambda cond,x,y,alu,i:
     x.alu(alu.op,y) if i.dtype is dtypes.weakint else cond.where(x.alu(alu.op,y), i.cast(dtypes.bool))),
  (UPat(GroupOp.Comparison, src=(UPat.var("y"), invalid_gate), name="alu"), lambda cond,x,y,alu,i:
     y.alu(alu.op,x) if i.dtype is dtypes.weakint else cond.where(y.alu(alu.op,x), i.cast(dtypes.bool))),
  # alu with invalid -> invalid
  (UPat(GroupOp.Unary, src=(invalid_pat,)), lambda i: i),
  (UPat(GroupOp.Binary-GroupOp.Comparison, src=[invalid_pat, UPat()]), lambda i: i),
  # normalize where(cond, Invalid, val) -> where(~cond, val, Invalid)
  (UPat.var("cond").where(invalid_pat, UPat.var("val")), lambda cond, i, val: cond.logical_not().where(val, i) if val.arg != Invalid else i),
  # lift Invalid out  # TODO: this `a is cond` is asymmetric to preserve the pattern
  (UPat.var("a").where(invalid_gate, UPat.var("c")), lambda cond,i,x,a,c:
   (cond if a is cond else (a.logical_not()|cond)).where(a.where(x,c), i) if c.arg != Invalid else None),
  (UPat.var("a").where(UPat.var("b"), invalid_gate), lambda cond,i,x,a,b: (a|cond).where(a.where(b, x), i) if b.arg != Invalid else None),
  (UPat(Ops.BITCAST, src=(invalid_pat,), name="bc"), lambda bc,i: i.cast(bc.dtype)),
  (UPat(Ops.BITCAST, src=(invalid_gate,), name="bc"), lambda bc,cond,x,i: cond.where(x.bitcast(bc.dtype), i.bitcast(bc.dtype))),
  # fold gated LOAD/STORE
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(), invalid_pat), allow_any_len=True).or_casted(), UPat())), lambda i: UOp(Ops.NOOP)),
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat(), invalid_pat), allow_any_len=True).or_casted(),), allow_any_len=True, name="x"),
    lambda x,i: x.src[1] if len(x.src) > 1 else x.const_like(0)),
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
  (UPat.var("x", dtype=dtypes.bool) & UPat.cvar("c", vec=False), lambda x,c: x if c.arg else c),
  (UPat.var("x", dtype=dtypes.bool) | UPat.cvar("c", vec=False), lambda x,c: c if c.arg else x),
  (UPat(GroupOp.Idempotent, src=(UPat.var("x"), UPat.var("x"))), lambda x: x),
  (UPat.var("x", dtype=dtypes.bool).logical_not().logical_not(), lambda x: x),
  (UPat.var("x", dtype=dtypes.bool).where(UPat.const(dtypes.bool, True), UPat.const(dtypes.bool, False)), lambda x: x),
  (UPat.var("x", dtype=dtypes.bool).where(UPat.const(dtypes.bool, False), UPat.const(dtypes.bool, True)), lambda x: x.logical_not()),
  # CAST(bool -> int) != const — CAST(True)=1, CAST(False)=0, so fold based on const value
  (UPat.var("x", dtype=dtypes.bool).cast(dtypes.ints+(dtypes.weakint,)) != UPat.cvar("c", vec=False),
   lambda x,c: x if c.arg == 0 else x.logical_not() if c.arg == 1 else x.const_like(True)),
  (UPat.var("x", dtype=dtypes.ints+(dtypes.bool, dtypes.weakint)).trunc(), lambda x: x),
  # ** zero folding **
  (UPat.var("x") < UPat.var("x"), lambda x: x.const_like(False).cast(dtypes.bool.vec(x.dtype.count))), # x < x -> False
  (UPat.var("x") % UPat.var("x"), lambda x: x.const_like(0)), # x%x -> 0
  (UPat.var("x") ^ UPat.var("x"), lambda x: x.const_like(0)), # x^x -> 0
  (UPat.var("x") & 0, lambda x: x.const_like(0)), # x&0 -> 0
  # (x&mask)>>k -> x>>k when mask only clears bits below k
  # TODO: combine this with "# rules for threefry" below
  ((UPat.var("x") & UPat.cvar("mask", vec=False)) >> UPat.cvar("k", vec=False),
   lambda x,mask,k: x >> k.arg if mask.arg | ((1 << k.arg) - 1) == -1 else None),
  (UPat.var("x", dtype=dtypes.ints+(dtypes.bool, dtypes.weakint)) != UPat.var("x"),
   lambda x: x.const_like(False).cast(dtypes.bool.vec(x.dtype.count))), # x != x -> False (only ints)
  # ** constant folding **
  (UPat(GroupOp.Unary, src=(UPat((Ops.CONST, Ops.STACK)),), name="a"), fold_const_alu),
  (UPat(GroupOp.Binary-{Ops.THREEFRY}, src=(UPat((Ops.CONST, Ops.STACK)),)*2, name="a"), fold_const_alu),
  (UPat(Ops.THREEFRY, src=(UPat.cvar("x"), UPat.cvar("key")), name="a"),
   lambda a, x, key: a.const_like(threefry2x32(x, key).simplify().arg)),
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
  (UPat.var("x").alu(Ops.POW, UPat.cvar("c", vec=False)), simplify_pow),
  # positive const ** x
  (UPat.cvar("c", vec=False).alu(Ops.POW, UPat.var("x")), lambda c,x: c if c.arg == 1 else (x*math.log2(c.arg)).exp2() if c.arg > 0 else None),
  # rules for threefry
  ((UPat.var('x', dtypes.uint64)&0xFFFFFFFF).cast(dtypes.uint32), lambda x: x.cast(dtypes.uint32)),
  (((UPat.var(None, dtypes.uint64)*(1<<32)) | UPat.var('y',  dtypes.uint32).cast(dtypes.uint64)).cast(dtypes.uint32), lambda y: y),
  (((UPat.var('x',  dtypes.uint64)*(1<<32)) | UPat.var(None, dtypes.uint32).cast(dtypes.uint64))//(1<<32), lambda x: x),
  (((UPat.var(None, dtypes.uint64)<<32) | UPat.var('y',  dtypes.uint32).cast(dtypes.uint64)).cast(dtypes.uint32), lambda y: y),
  (((UPat.var('x',  dtypes.uint64)<<32) | UPat.var(None, dtypes.uint32).cast(dtypes.uint64))>>32, lambda x: x),
  # ** simple where folding **
  # a conditional with the same results either way is a noop, also fold const conditionals
  (UPat.var().where(UPat.var("val"), UPat.var("val")), lambda val: val),
  (UPat.cvar("gate", vec=False).where(UPat.var("c0"), UPat.var("c1")), lambda gate, c0, c1: c0 if gate.arg else c1),
  # a.where(b.where(c, d), d) -> (a & b).where(c, d)
  (UPat.var("a").where(UPat.var("b").where(UPat.var("c"), UPat.var("d")), UPat.var("d")), lambda a,b,c,d: (a&b).where(c,d)),
])

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

def gep_through_wmma(gep:UOp, wmma:UOp) -> UOp|None:
  out_sz = prod(x[1] for x in wmma.arg[6][-1])
  wmma_idxs = gep.arg[::out_sz]
  for i in range(out_sz):
    if tuple(x-i for x in gep.arg[i::out_sz]) != wmma_idxs: return None
  tsrcs = []
  for s,sz in zip(wmma.src, wmma.arg[6]):
    src_args = []
    ssz = prod(x[1] for x in sz)
    for w in wmma_idxs: src_args += list(range((w//out_sz)*ssz, (w//out_sz)*ssz + ssz))
    tsrcs.append(s.gep(tuple(src_args)))
  return UOp(Ops.WMMA, gep.dtype, tuple(tsrcs), wmma.arg)

gep_pushing = PatternMatcher([
  # GEP/VECTORIZE, GEP/GEP, GEP/CONST
  (UPat(Ops.GEP, name='g2').f(Ops.GEP, name='g1'),
   lambda g1, g2: g2.src[0].gep(tuple(g2.arg[g1.arg[i]] for i in range(len(g1.arg))))),
  (UPat(Ops.STACK, name='vec').f(Ops.GEP, name='gep'),
   lambda gep, vec: UOp(Ops.STACK, gep.dtype, tuple(vec.src[i] for i in gep.arg)) if len(gep.arg) > 1 else vec.src[gep.arg[0]]),
  (UPat.cvar("c", vec=False).f(Ops.GEP, name="gep"), lambda gep, c: gep.const_like(c.arg)),
  # GEP on void is skipped
  (UPat(Ops.GEP, src=(UPat(dtype=dtypes.void, name="x"),)), lambda x: x),
  # GEP in order is removed
  (UPat(Ops.GEP, name="g"), lambda g: g.src[0] if not isinstance(g.dtype, PtrDType) and g.arg == tuple(range(g.src[0].dtype.count)) else None),
  # push all GEPs through ALUs for index (TODO: remove this)
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST), name='alu').f(Ops.GEP, dtype=dtypes.weakint, name='gep'),
   lambda gep,alu: UOp(alu.op, alu.dtype.scalar().vec(gep.dtype.count), tuple(x.gep(gep.arg) for x in alu.src), alu.arg) \
     if not isinstance(gep.dtype, PtrDType) and not isinstance(alu.dtype, PtrDType) else None),
  # CAT can't be rendered. it's a VECTORIZE on vectors, we expand to a single VECTORIZEs with GEPs (TODO: move this later)
  (UPat(Ops.VCAT, name="x"), lambda x: UOp(Ops.STACK, x.dtype, tuple(y.gep(i) for y in x.src for i in range(y.dtype.count))) \
    if not isinstance(x.dtype, PtrDType) else None),
  # VECTORIZE on same GEP
  (UPat(Ops.STACK, name="v", src=UPat(Ops.GEP, src=(UPat.var("x"),))), lambda v,x: x.gep(tuple(get_single_element(i.arg) for i in v.src))),
  # push some GEPs through WMMAs
  (UPat(Ops.WMMA, name="wmma").f(Ops.GEP, name="gep"), gep_through_wmma),
])

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
  (UPat({Ops.CMPLT, Ops.CMPNE, Ops.FLOORDIV, Ops.FLOORMOD, Ops.DEFINE_VAR, Ops.BIND, Ops.SPECIAL}, name="x"),
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
  ((UPat.cvar("c0", vec=False)*UPat.var("x", dtype=dtypes.weakint))<UPat.cvar("c1", vec=False),
   lambda x,c0,c1: x<math.ceil(c1.arg/c0.arg) if c0.arg > 0 and c1.arg > 0 else None),
  # c0*x<c1 for negative int c0 and non-positive c1
  ((UPat.cvar("c0", vec=False)*UPat.var("x", dtype=dtypes.weakint))<UPat.cvar("c1", vec=False),
   lambda x,c0,c1: (-x)<(-(math.floor(-c1.arg/-c0.arg))) if c0.arg < 0 and c0.arg != -1 and c1.arg <= 0 else None),
  # x//d<c -> x<c*d for d>0
  ((UPat.var("x", dtype=dtypes.weakint)//UPat.cvar("d", vec=False))<UPat.cvar("c", vec=False),
   lambda x,d,c: x<(c.arg*d.arg) if d.arg > 0 else None),
  # ** move add/mul consts to end (NOTE: this is still happening before constant folding) **
  ((UPat.var("x") + UPat.cvar("c1")) + UPat.var("y"), lambda x,c1,y: (x+y)+c1),
  ((UPat.var("x") * UPat.cvar("c1")) * UPat.var("y"), lambda x,c1,y: (x*y)*c1),
  # *** rules from symbolic ***
  # generic lt folding
  (UPat.var("x", dtypes.weakint)<UPat.cvar("c", vec=False), lambda x,c: lt_folding(x, c.arg) if 0 < c.arg else None),
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
    tuple(dedup(flatten([(y,) if y.op in {Ops.RANGE, Ops.STORE, Ops.CALL, Ops.FUNCTION, Ops.BARRIER, Ops.END, Ops.UNROLL, Ops.LINEAR, Ops.STAGE}
                        else y.src for y in x.src[1:]]))))),
  # after with 1 src is just src[0]
  (UPat(Ops.AFTER, src=(UPat.var("s"),)), lambda s: s),
  # VECTORIZE/CONST
  (UPat(Ops.STACK, src=UPat(Ops.CONST), name="vec"),
    lambda vec: UOp.const(vec.dtype, tuple(x.arg for x in vec.src)) if len(vec.src) > 0 else None),
])+div_and_mod_symbolic+gep_pushing

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
  bounds:defaultdict[UOp, list[ConstType|None]] = defaultdict(lambda: [None, None])
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
  return sum(-1 if (res:=parse_valid(v)) is not None and res[0] in other.toposort() else 0 for other in valids)

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
  if x.dtype.scalar() is not dtypes.weakint: return None
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
REMOVE_FROM_SINK_LIKE = {Ops.UNROLL, Ops.NOOP, Ops.STACK, Ops.SINK}
sym = symbolic+pm_simplify_valid+PatternMatcher([
  # reorder ALU/VECTORIZE
  (UPat(GroupOp.ALU, src=(UPat(Ops.STACK, src=UPat(name='x')), UPat(Ops.STACK, src=UPat(name='y'))), name='alu'),
   lambda x,y,alu: UOp(Ops.STACK, alu.dtype, (UOp(alu.op, alu.dtype.scalar(), (x,y)),)*alu.dtype.count)),
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
   lambda index, gate, alt: UOp.store(index.src[0].index(gate.where(index.src[1], UOp.invalid())), alt)),
  # fold gated LOAD/STORE
  (UPat(Ops.STORE, src=(UPat(), invalid_pat)), lambda i: UOp(Ops.NOOP)),
  # store of where with invalid -> gated store
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, name="index"), UPat.var("cond").where(UPat.var("val"), invalid_pat))),
   lambda index, cond, val, i: UOp.store(index.src[0].index(cond.where(index.src[1], UOp.invalid())), val)),
  ((UPat.var("x") * UPat.var("x")).reciprocal(), lambda x: x.reciprocal()*x.reciprocal()),  # 1/(x^c) -> (1/x)^c
  ((UPat.var("x") * UPat.var("x") * UPat.var("x")).reciprocal(), lambda x: x.reciprocal()*x.reciprocal()*x.reciprocal()),
  ((UPat.var("x") * UPat.cvar("c")).reciprocal(), lambda x,c: x.reciprocal()*c.reciprocal()), # 1/(x*c) -> (1/c)*(1/x)
  (UPat.var("x") * ((1+UPat.var("x")).reciprocal().named("d")), lambda x,d: 1-d), # x*/(1+x) -> 1-1/(1+x)
  (UPat.var("x") * ((1+UPat.var("x")).reciprocal().named("d")*UPat.var("y")), lambda x,y,d: y*(1-d)),
  (UPat.var("x") * ((1+UPat.var("x")).reciprocal().named("d")+UPat.var("y")), lambda x,y,d: (1-d)+x*y),
  # move const multiply after REDUCE (NOTE: the mul chain can do this, but only if it's a same dtype reduce)
  ((UPat.var("x")*UPat.cvar("c", vec=False)).reduce(arg=Ops.ADD, name="r", allow_any_len=True), lambda x,c,r: r.replace(src=(x,)+r.src[1:])*c.arg),
  # reduce mul chain, move muls after the reduce
  (UPat(Ops.MUL).reduce(name="r", allow_any_len=True), reduce_mul_chain),
  # clean up GROUP/SINK
  (UPat(Ops.GROUP, src=(UPat.var("x"),)), lambda x: x),
  (UPat((Ops.SINK, Ops.GROUP), name="root"),
    lambda root: UOp(root.op, root.dtype, tuple(flatten(x.src if x.op in REMOVE_FROM_SINK_LIKE else (x,) for x in root.src)), root.arg)
      if any(x.op in REMOVE_FROM_SINK_LIKE for x in root.src) else None),
  # ** combine terms (opinionated) **
  (-1 * (UPat.var("x") + UPat.var("y")), lambda x,y: (-x)+(-y)),  # -(x+y) -> -x + -y
  # (x+y)*c -> x*c+y*c. only for int, float has inf*0=nan issue
  ((UPat.var("x", dtypes.weakint) + UPat.var("y")) * UPat.cvar("c"), lambda x,y,c: x*c+y*c),
])
