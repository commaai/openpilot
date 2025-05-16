# all of symbolic lives here now
from typing import Any, Literal, cast
import math, operator, struct, functools
from collections import defaultdict
from tinygrad.ops import Ops, PatternMatcher, UPat, UOp, GroupOp, exec_alu
from tinygrad.dtype import ConstType, dtypes, PtrDType
from tinygrad.helpers import partition, all_same, prod, flatten, get_single_element, cdiv, cmod
from tinygrad.codegen.transcendental import xpow

# ******** phase 1 of symbolic used to live in ops, it's the most generic folding rules ********

def simplify_pow(x:UOp, c:UOp) -> UOp|None:
  if c.arg < 0: return x.reciprocal().pow(-c)
  if c.arg == 0: return x.const_like(1)
  if int(c.arg-0.5)+0.5 == c.arg: return x.pow(c.const_like(c.arg-0.5)) * x.sqrt()
  if int(c.arg) == c.arg: return (y := x.pow(c.const_like(c.arg//2))) * y * (x if c.arg%2 == 1 else 1)
  return None

def fold_bitcast(root:UOp, c:UOp) -> UOp|None:
  if (from_fmt:=c.dtype.scalar().fmt) is None or (to_fmt:=root.dtype.scalar().fmt) is None: return None
  def convert(v:Any): return struct.unpack(to_fmt, struct.pack(from_fmt, v))[0]
  return root.const_like(convert(c.arg) if root.dtype.count == 1 else tuple(map(convert, c.arg)))

symbolic_simple = PatternMatcher([
  # ** self folding **
  (UPat.var("x") + 0, lambda x: x),    # x+0 -> x
  (UPat.var("x") * 1, lambda x: x),    # x*1 -> x
  (UPat.var("x") // UPat.var("x"), lambda x: x.const_like(1)), # x//x -> 1
  (UPat.var("x") // 1, lambda x: x),   # x//1 -> x
  (UPat.var("x") // -1, lambda x: -x), # x//-1 -> -x
  (UPat.var("x") / UPat.var("x"), lambda x: x.const_like(1)), # x/x -> 1
  ((UPat.var("x") * UPat.var("x2")) / UPat.var("x2"), lambda x,x2: x), # (x*x2)/x2 -> x
  ((UPat.var() % UPat.var("y")).named("base") % UPat.var("y"), lambda base,y: base),  # (x%y)%y = -> x%y (rewritten with base for speed)
  (UPat.var("x")%UPat.cvar("c")+(UPat.var("x")//UPat.cvar("c"))*UPat.cvar("c"), lambda x,c: x), # (x%c)+(x//c)*c = x
  ((UPat.var("x")//UPat.cvar("c1"))*UPat.cvar("c3")+UPat.var("x")%UPat.cvar("c1")*UPat.cvar("c2"),
    lambda x,c1,c2,c3: x*c2 if c1.arg*c2.arg==c3.arg else None), # (x%c1)*c2+(x//c1)*c3 = x*c2 if c1*c2==c3
  (UPat.var("x", dtype=dtypes.bool) & UPat.cvar("c", vec=False), lambda x,c: x if c.arg else c),
  (UPat.var("x", dtype=dtypes.bool) | UPat.cvar("c", vec=False), lambda x,c: c if c.arg else x),
  (UPat(GroupOp.Idempotent, src=(UPat.var("x"), UPat.var("x"))), lambda x: x),
  (UPat.var("x", dtype=dtypes.bool).logical_not().logical_not(), lambda x: x),
  (UPat.var("x", dtype=dtypes.bool).where(UPat.const(dtypes.bool, True), UPat.const(dtypes.bool, False)), lambda x: x),
  # ** zero folding **
  (UPat.var("x") < UPat.var("x"), lambda x: x.const_like(False).cast(dtypes.bool.vec(x.dtype.count))), # x < x -> False
  (UPat.var("x") % UPat.var("x"), lambda x: x.const_like(0)), # x%x -> 0
  (UPat.var("x", dtype=dtypes.ints) != UPat.var("x", dtype=dtypes.ints),
   lambda x: x.const_like(False).cast(dtypes.bool.vec(x.dtype.count))), # x != x -> False (only ints)
  # x*0 -> 0 or 0*x -> 0
  # if x is nan or inf it should render the nan value.
  # NOTE: this can be wrong for loaded NaN
  (UPat.var("x") * 0, lambda x: x.const_like(float("nan") if isinstance(x.arg, float) and (math.isnan(x.arg) or math.isinf(x.arg)) else 0)),
  # ** constant folding **
  # TODO: add const folding for Ops.THREEFRY
  (UPat(GroupOp.Unary, src=(UPat((Ops.VCONST, Ops.CONST)),), name="a"), lambda a: a.const_like(exec_alu(a.op, a.dtype, [a.src[0].arg], False))),
  (UPat(GroupOp.Binary-{Ops.THREEFRY}, src=(UPat((Ops.VCONST, Ops.CONST)),)*2, name="a"),
   lambda a: a.const_like(exec_alu(a.op, a.dtype, [a.src[0].arg, a.src[1].arg], False))),
  (UPat(GroupOp.Ternary, src=(UPat((Ops.VCONST, Ops.CONST)),)*3, name="a"),
   lambda a: a.const_like(exec_alu(a.op, a.dtype, [a.src[0].arg, a.src[1].arg, a.src[2].arg], False))),
  # bool MUL is AND, ADD/MAX is OR. prevents other rules to rewrite bool ADD/MUL incorrectly
  (UPat.var('x', dtype=dtypes.bool) * UPat.var('y', dtype=dtypes.bool), lambda x,y: x&y),
  (UPat.var('x', dtype=dtypes.bool) + UPat.var('y', dtype=dtypes.bool), lambda x,y: x|y),
  (UPat.var('x', dtype=dtypes.bool).maximum(UPat.var('y', dtype=dtypes.bool)), lambda x,y: x|y),
  # *** cast/bitcast ***
  (UPat(Ops.CAST, name="root", src=(UPat.cvar("c"),)), lambda root, c: root.const_like(c.arg)),
  (UPat((Ops.CAST, Ops.BITCAST), name="root"), lambda root: root.src[0] if root.dtype == root.src[0].dtype else None),
  (UPat(Ops.BITCAST, name="root", src=(UPat.cvar("c"),)), fold_bitcast),
  # ** pow **
  (UPat.var("x").alu(Ops.POW, UPat.cvar("c", vec=False)), simplify_pow),
  # positive const ** x
  (UPat.cvar("c", vec=False).alu(Ops.POW, UPat.var("x")), lambda c,x: c if c.arg == 1 else (x*math.log2(c.arg)).exp2() if c.arg > 0 else None),
])

# ******** phase 2 builds on phase 1, it includes the old "symbolic", rules that match deeper ********

def split_uop(x:UOp, sep:Ops):
  if x.op is sep:
    for s in x.src: yield from split_uop(s, sep)
  else: yield x

def fold_unrolled_divs(divs:UOp, denominator: int, fac=1) -> UOp|None:
  # div pattern in unrolled arange
  # example: (x//4+(x+1)//4+(x+2)//4+(x+3)//4 -> x
  seen_const, ans = [], None
  for u in split_uop(divs, Ops.ADD):
    if fac!=1:
      if u.op is not Ops.MUL or u.src[1].op is not Ops.CONST or u.src[1].arg != fac: return None
      u = u.src[0]
    if not (u.op is Ops.IDIV and u.src[1].op is Ops.CONST): return None
    if denominator != u.src[1].arg: return None
    if (s0:=u.src[0]).vmin < 0: return None
    # assumed CONST is the last of an ADD
    if s0.op is Ops.ADD and s0.src[1].op is Ops.CONST and s0.src[1].op is Ops.CONST:
      seen_const.append(s0.src[1].arg)
      s0 = s0.src[0]
    else: seen_const.append(0)
    if ans is None: ans = s0
    if ans is not s0: return None
  if ans is None: return None
  # the first (denominator-len(seen_const)) terms may have been folded to 0 already
  for i in range(denominator-len(seen_const)):
    if ans is not None and 0 <= ans.vmin and ans.vmax + i < denominator: seen_const.append(i)
  if sorted(seen_const)==list(range(denominator)):
    return fac*ans
  return None

def lt_folding(x:UOp, c:int) -> UOp|None:
  p, np = partition(split_uop(x, Ops.ADD), lambda u: u.const_factor() == 1)
  if np and (d:=math.gcd(*[u.const_factor() for u in np], c)) > 1 and 0 <= sum(u.vmin for u in p) and sum(u.vmax for u in p) < d:
    return cast(UOp, functools.reduce(operator.add, np).divides(d))<(c//d)
  return None

def canonicalize_simplex(X:UOp) -> UOp|None:
  # (X := a0*x0 + a1*x1 + ...) > 0 is equivalent to x0 + x1 + ... > 0 if xi >= 0 and ai > 0 for ints.
  # returns x0 + x1 + ... in such case, or None if not
  changed, ret = False, []
  for u in split_uop(X, Ops.ADD):
    # assumed the const is the last src of MUL
    if u.op is Ops.MUL and u.src[1].op is Ops.CONST and u.src[1].arg > 0:
      changed = True
      u = u.src[0]
    if not (u.op in GroupOp.Irreducible and u.vmin >= 0): return None
    ret.append(u)
  return functools.reduce(operator.add, ret) if changed else None

def div_and_mod_folding(x: UOp, y: UOp, which: Literal[Ops.MOD, Ops.IDIV], split_rem: bool=False) -> UOp|None:
  # simplify x // y or x % y, None means no change
  # simple cancel div/mod case
  x_min, x_max, y_min, y_max = x.vmin, x.vmax, y.vmin, y.vmax
  assert isinstance(x_min, int) and isinstance(x_max, int) and isinstance(y_min, int) and isinstance(y_max, int)

  if y_min*y_max > 0 and (q:=cdiv(x_min,y_min)) == cdiv(x_min,y_max) == cdiv(x_max,y_min) == cdiv(x_max,y_max):
    return x - q*y if which is Ops.MOD else x.const_like(q)

  if (y.op is not Ops.CONST) or ((c := y.arg) <= 0) or (x.dtype.count > 1): return None

  svars, factors, quotients, remainders, gcd, div, const, something_changed = [], [], [], [], c, 1, 0, False
  for u in split_uop(x, Ops.ADD):
    if u.op is Ops.MOD and which is Ops.MOD and u.src[1].op is Ops.CONST and u.src[1].arg%c == 0:
      u = u.src[0]
      something_changed = True
    v: UOp = u.divides(f:=u.const_factor())
    q, r = divmod(f, c)
    if r==0 or ((which is Ops.MOD or split_rem or u.op is Ops.CONST) and r!=f): something_changed = True
    if u.op is Ops.CONST: const += f
    else:  # div is the smallest common divisor of all terms
      if f > 1 and c % f == 0 and (div == 1 or div > f): div = f
      gcd = math.gcd(r, gcd)
      factors.append(f); svars.append(v); quotients.append(q); remainders.append(r)  # noqa: E702

  # we can fold if the expression has only one non-constant term and this term can only take on two values
  if len(svars)==1 and (v:=svars[0]).vmax-v.vmin == 1:
    y1 = cmod(factors[0]*v.vmin+const, c) if which is Ops.MOD else cdiv(factors[0]*v.vmin+const, c)
    y2 = cmod(factors[0]*v.vmax+const, c) if which is Ops.MOD else cdiv(factors[0]*v.vmax+const, c)
    return (y2-y1)*(v-v.vmin) + y1

  # a//c = (a-a%c)/c, if we can fold a%c, we can fold a//c
  # within a mod we can freely subtract multiples of c, we use this to see if a is congruent to an expression whose vmin/vmax are between 0 and c
  rems = [min(r, r-c, key=abs) for r in remainders]
  if (rem:=sum(r*v for r,v in zip(rems,svars))+const%c).vmin//c==rem.vmax//c and all(f > 0 for f in factors):
    if which is Ops.MOD: return rem - rem.vmin//c*c
    return sum((f-r)//c * v for f,r,v in zip(factors,rems,svars)) + (const-const%c+rem.vmin//c*c)//c

  if (g:=math.gcd(gcd, const))!=1:
    ret = UOp(which, x.dtype, src=(sum(f//g * v for f,v in zip(factors, svars)) + const//g, x.const_like(c//g)))
    return ret*g if which is Ops.MOD else ret

  if gcd != 1: something_changed = True
  if not something_changed:
    if which is Ops.IDIV and (1 < div < c) and (newx:=div_and_mod_folding(x, x.const_like(div), Ops.IDIV)) is not None: return newx//(c//div)
    return None
  quo, rem = x.const_like(const//c), x.const_like((const%c)//gcd)
  for q,r,f,v in zip(quotients, remainders, factors, svars):
    if which is Ops.IDIV and (not split_rem) and r!=0:
      rem += f//gcd * v
    else:
      rem += r//gcd * v
      quo += q * v

  # if numerator before/after is negative, and it has remainder, don't simplify because C divmod is different from python divmod.
  if (x_min < 0 or rem.vmin < 0) and remainders: return None
  if which is Ops.MOD: return gcd*(rem % (c//gcd)) + const%gcd
  return rem//(c//gcd)+quo

def gep_through_wmma(gep:UOp, wmma:UOp):
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
  # GEP/VECTORIZE, GEP/GEP, GEP/CONST, GEP/VCONST
  (UPat(Ops.GEP, src=(UPat(Ops.GEP, name='g2'),), name='g1'),
   lambda g1, g2: g2.src[0].gep(tuple(g2.arg[g1.arg[i]] for i in range(len(g1.arg))))),
  (UPat(Ops.GEP, src=(UPat(Ops.VECTORIZE, name="vec"),), name="gep"),
   lambda gep, vec: UOp(Ops.VECTORIZE, gep.dtype, tuple(vec.src[i] for i in gep.arg)) if len(gep.arg) > 1 else vec.src[gep.arg[0]]),
  (UPat(Ops.GEP, src=(UPat.cvar("c", vec=False),), name="gep"), lambda gep, c: gep.const_like(c.arg)),
  (UPat(Ops.GEP, src=(UPat(Ops.VCONST, name="c"),), name="gep"), lambda gep, c: gep.const_like(tuple(c.arg[x] for x in gep.arg))),
  # GEP on void is skipped
  (UPat(Ops.GEP, src=(UPat(dtype=dtypes.void, name="x"),)), lambda x: x),
  # GEP in order is removed
  (UPat(Ops.GEP, name="g"), lambda g: g.src[0] if not isinstance(g.dtype, PtrDType) and g.arg == tuple(range(g.src[0].dtype.count)) else None),
  # push all GEPs through ALUs (fix arange stuff)
  (UPat(Ops.GEP, src=(UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST), name='alu'),), name='gep'),
   lambda gep,alu: UOp(alu.op, alu.dtype.scalar().vec(gep.dtype.count), tuple(x.gep(gep.arg) for x in alu.src), alu.arg) \
     if not isinstance(gep.dtype, PtrDType) else None),
  # CAT can't be rendered. it's a VECTORIZE on vectors, we expand to a single VECTORIZEs with GEPs (TODO: move this later)
  (UPat(Ops.CAT, name="x"), lambda x: UOp(Ops.VECTORIZE, x.dtype, tuple(y.gep(i) for y in x.src for i in range(y.dtype.count))) \
    if not isinstance(x.dtype, PtrDType) else None),
  # VECTORIZE on same GEP
  (UPat(Ops.VECTORIZE, name="v", src=UPat(Ops.GEP, src=(UPat.var("x"),))), lambda v,x: x.gep(tuple(get_single_element(i.arg) for i in v.src))),
  # push some GEPs through WMMAs
  (UPat(Ops.GEP, src=(UPat(Ops.WMMA, name="wmma"),), name="gep"), gep_through_wmma),
])

commutative = PatternMatcher([
  # ** COMMUTATIVE flipping (only for ints) **
  # NOTE: this can break merging vector math by only flipping some of them
  (UPat(GroupOp.Commutative, dtype=dtypes.int, name='x'), lambda x: x.replace(src=x.src[::-1]) if x.src[1].tuplize < x.src[0].tuplize else None),
])

symbolic = symbolic_simple+commutative+PatternMatcher([
  # ** boolean algebra **
  (UPat.var("x") | (UPat.var("x") & UPat.var()), lambda x: x), # x|(x&y) -> x
  # ** combine terms **
  (UPat.var("x") * UPat.cvar("c0") + UPat.var("x") * UPat.cvar("c1"), lambda x,c0,c1: x*(c0+c1)), # (x*c0)+(x*c1) -> x*(c0+c1)
  ((UPat.var("y") + UPat.var("x") * UPat.cvar("c0")) + UPat.var("x") * UPat.cvar("c1"), lambda x,y,c0,c1: y+x*(c0+c1)),
  (UPat.var("x") + UPat.var("x") * UPat.cvar("c"), lambda x,c: x*(c+1)), # (x+x*c)-> x*(c+1)
  ((UPat.var("y") + UPat.var("x")) + UPat.var("x") * UPat.cvar("c"), lambda x,y,c: y+x*(c+1)),
  (UPat.var("x") + UPat.var("x"), lambda x: x*2), # (x+x)-> x*2
  ((UPat.var("y") + UPat.var("x")) + UPat.var("x"), lambda y,x: y+x*2),
  ((UPat.var("x") / UPat.var("x2")) / UPat.var("x3"), lambda x,x2,x3: x/(x2*x3) if x2 is not x3 else None), # (x/x2)/x3 -> x/(x2*x3)
  (-1 * (UPat.var("x") + UPat.cvar("c")), lambda x,c: (-x)+(-c)),  # -(x+c) -> -x + -c
  # a conditional with the same results either way is a noop, also fold const conditionals
  (UPat.var().where(UPat.var("val"), UPat.var("val")), lambda val: val),
  (UPat.cvar("gate", vec=False).where(UPat.var("c0"), UPat.var("c1")), lambda gate, c0, c1: c0 if gate.arg else c1),
  (UPat.var("cond", dtype=dtypes.bool).logical_not().where(UPat.var("t"), UPat.var("f")), lambda cond, t, f: cond.where(f,t)),
  # alu of two where with same conds can combine, only do if true branch or false branch is const
  (UPat(GroupOp.Binary, name="alu", src=(UPat.var("c").where(UPat.var("t"), UPat.var("f")), UPat.var("c").where(UPat.var("tt"), UPat.var("ff")))), \
   lambda alu,c,t,tt,f,ff: c.where(t.alu(alu.op, tt), f.alu(alu.op, ff)) if t.op == tt.op == Ops.CONST or f.op == ff.op == Ops.CONST else None),
  # ALU/variable min==max -> CONST (slow!)
  (UPat(GroupOp.ALU|{Ops.DEFINE_VAR, Ops.SPECIAL, Ops.RANGE}, name="x"), lambda x: x.const_like(x.vmin) if x.vmin == x.vmax else None),
  # max folding
  (UPat.maximum(UPat.var("x"), UPat.var("y")), lambda x,y: x if x.vmin >= y.vmax else y if x.vmax <= y.vmin else None),
  # TODO: why does this rule break beautiful_mnist?
  #((UPat.var("x")+UPat.var("z")).maximum(UPat.var("y")+UPat.var("z")), lambda x,y,z: x.maximum(y) + z),
  #((UPat.var("x")*UPat.cvar("c1")).maximum(UPat.var("x")*UPat.cvar("c2")), max_var_const),
  # ** two stage ALU folding **
  *((UPat.var("x").alu(op, UPat.cvar("c1")).alu(op, UPat.cvar("c2")).named("f"),
     lambda f,x,c1,c2: x.alu(f.op,c1.alu(f.op,c2))) for op in GroupOp.Associative),
  ((UPat.cvar("c0") + UPat.var("x")) < UPat.cvar("c1"), lambda x,c0,c1: x<(c1-c0)),  # c0 + x < c1 -> x < c1 - c0
  ((UPat.var("x") // UPat.cvar("c1")) // UPat.cvar("c2"), lambda x,c1,c2: x//(c1*c2)), # (x//c1)//c2 -> x//(c1*c2)
  # ** lt **
  # c0*x<c1 for positive int c0,c1
  ((UPat.cvar("c0", vec=False)*UPat.var("x", dtype=dtypes.ints))<UPat.cvar("c1", vec=False),
   lambda x,c0,c1: x<math.ceil(c1.arg/c0.arg) if c0.arg > 0 and c1.arg > 0 else None),
  # c0*x<c1 for negative int c0 and non-positive c1
  ((UPat.cvar("c0", vec=False)*UPat.var("x", dtype=dtypes.ints))<UPat.cvar("c1", vec=False),
   lambda x,c0,c1: (-x)<(-(math.floor(-c1.arg/-c0.arg))) if c0.arg < 0 and c0.arg != -1 and c1.arg <= 0 else None),
  # x//c0<c1 for positive int c0
  ((UPat.var("x", dtype=dtypes.ints)//UPat.cvar("c0", vec=False))<UPat.cvar("c1", vec=False),
   lambda x,c0,c1: x<(c1.arg*c0.arg) if c0.arg > 0 else None),
  # ** move add/mul consts to end (NOTE: this is still happening before constant folding) **
  ((UPat.var("x") + UPat.cvar("c1")) + UPat.var("y"), lambda x,c1,y: (x+y)+c1),
  ((UPat.var("x") * UPat.cvar("c1")) * UPat.var("y"), lambda x,c1,y: (x*y)*c1),
  # *** rules from symbolic ***
  # unrolled arange div folding
  ((UPat() + UPat()//UPat.cvar("d", vec=False)).named("divs"), lambda divs,d: fold_unrolled_divs(divs, d.arg)),
  ((UPat() + (UPat()//UPat.cvar("d", vec=False))*UPat.cvar("c")).named("divs"), lambda divs,d,c: fold_unrolled_divs(divs, d.arg, c.arg)),
  # generic lt folding
  (UPat.var("x", dtypes.sints)<UPat.cvar("c", vec=False), lambda x,c: lt_folding(x, c.arg) if 0 < c.arg else None),
  (UPat.var("x", dtypes.sints)*-1 < UPat.var("y", dtypes.sints)*-1, lambda x,y: y<x),
  # canonicalize a simplex with positive coefficients > 0
  # not x < 1 -> X > 0
  ((UPat.var("x", dtypes.ints)<1).ne(True), lambda x: (newx<1).ne(True) if (newx:=canonicalize_simplex(x)) is not None else None),
  # ** div **
  # div folding
  ((UPat.var("x")//UPat.cvar("c") + UPat.cvar("a"))//UPat.cvar("d"), lambda x,c,a,d: (x+a*c)//(c*d)
      if x.vmin>=0 or x.vmax<=0 else None),  # (x//c+a)//d -> (x+a*c)//(c*d)
  (UPat.var("x", dtypes.sints) // UPat.var("y"), lambda x,y: div_and_mod_folding(x,y,Ops.IDIV)),
  (UPat.var("x") // UPat.var("d"), lambda x,d: -(x//(-d)) if d.vmax <=0 else None),
  (UPat.var("x") // UPat.var("d"), lambda x,d: -((-x)//d) if x.vmax <=0 else None),
  ((UPat.var("x", dtypes.sints)+UPat.cvar("c", vec=False)).named("n")//UPat.cvar("d", vec=False),
    lambda x,c,n,d: (-(-(c.arg%d.arg + x - (d.arg-1))//d) + c.arg//d.arg) if x.vmax<=0 and n.vmin>=0 and d.arg>0 else None),
  # ** mod **
  # mod folding
  (UPat.var("x") % UPat.var("y"), lambda x,y: div_and_mod_folding(x,y,Ops.MOD)),
  (UPat.var("x") % UPat.var("d"), lambda x,d: -(x%(-d)) if d.vmax <=0 else None),
  (UPat.var("x") % UPat.var("d"), lambda x,d: -((-x)%d) if x.vmax <=0 else None),
])+gep_pushing

symbolic_flat = symbolic+PatternMatcher([
  # ** combine terms (opinionated) **
  (-1 * (UPat.var("x") + UPat.var("y")), lambda x,y: (-x)+(-y)),  # -(x+y) -> -x + -y
  # (x+y)*c -> x*c+y*c. only for int, float has inf*0=nan issue
  ((UPat.var("x", dtypes.ints) + UPat.var("y")) * UPat.cvar("c"), lambda x,y,c: x*c+y*c),
])

# ******** we take a small aside to "simplify_valid" to rewrite valids ********

def parse_valid(valid:UOp) -> tuple[UOp, bool, int]:
  # if it's X <= c, returns X, True, c
  # if it's X >= c, returns X, False, c

  # (X < c).ne(True) -> X >= c
  if valid.op is Ops.CMPNE and valid.src[1].op is Ops.CONST and valid.src[1].arg == 1 and \
    (s0:=valid.src[0]).op is Ops.CMPLT and s0.src[1].op is Ops.CONST: return s0.src[0], False, s0.src[1].arg
  # X < c -> X <= c-1
  if valid.op is Ops.CMPLT and valid.src[1].op is Ops.CONST and dtypes.is_int(valid.src[0].dtype): return valid.src[0], True, valid.src[1].arg-1
  raise ValueError(f"not able to parse {valid=}")

def uop_given_valid(valid:UOp, uop:UOp) -> UOp|None:
  # return None if valid is always False, otherwise the simplified uop (might be the same as input)

  # first, parse valid into {expr: (lower_bound, upper_bound)}
  bounds:defaultdict[UOp, list[ConstType|None]] = defaultdict(lambda: [None, None])
  for stmt in split_uop(valid, Ops.AND):
    try: expr, is_upper, c = parse_valid(stmt)
    except ValueError: return uop  # give up if we cannot parse the valid
    bounds[expr][int(is_upper)] = c

  # simplify uop given that valid is True
  for expr,v in bounds.items():
    v0, v1 = (expr.vmin if v[0] is None else v[0], expr.vmax if v[1] is None else v[1])
    # some expr has lower bound > upper bound -> valid is an empty set and we return None
    if v0 > v1: return None
    # whole node became a const
    if v0 == v1:
      uop = uop.substitute({expr:expr.const_like(v0)}).simplify()
      continue
    # every candidate is a set of contrained UOp based on valid, and if every item in a set simplifies the uop into a same output, we rewrite uop
    candidates = []
    if expr.op is Ops.ADD and v0 == 1 and all(u.op in GroupOp.Irreducible for u in split_uop(expr, Ops.ADD)):
      # if the constraint is a simplex: X0 + X1 + ... > 0, we can check if all Xi > 0 simplify into the same output
      candidates.append([(Xi, UOp.variable("fake", 1, Xi.vmax, Xi.dtype)) for Xi in split_uop(expr, Ops.ADD)])
    # try checking the whole clause
    if expr in uop.toposort(): candidates.append([(expr, UOp.variable("fake", v0, v1, expr.dtype))])

    for candidate in candidates:
      # if every branch in candidate gives the same simplified uop, we can rewrite the uop
      newuops = [uop.substitute({X:newX}).simplify().substitute({newX:X}).simplify() for X,newX in candidate]
      if uop.op is Ops.VECTORIZE and len(uop.src) == 2:
        if all_same([uops.src[0] for uops in newuops]): uop = uop.replace(src=(newuops[0].src[0], uop.src[1]))
        if all_same([uops.src[1] for uops in newuops]): uop = uop.replace(src=(uop.src[0], newuops[0].src[1]))
      elif all_same(newuops): uop = newuops[0]

  return uop

def _valid_priority(v: UOp, valids:list[UOp]):
  # we want valid that's in other valids' parents to be first, so it's more likely the other valids get simplified
  try: return sum(-1 if parse_valid(v)[0] in other.toposort() else 0 for other in valids)
  except ValueError: return 0

def simplify_valid(valid:UOp) -> UOp|None:
  ret:list[UOp] = []
  something_changed = False
  valids = list(split_uop(valid, Ops.AND))
  for stmt in sorted(valids, key=lambda v: _valid_priority(v, valids)):
    ret.append(newstmt if ret and (newstmt:=uop_given_valid(functools.reduce(operator.and_, ret), stmt)) is not None else stmt)
    if ret[-1] is not stmt: something_changed = True
  return functools.reduce(operator.and_, ret) if something_changed else None

# ***** threefry *****

def threefry2x32(x: UOp, key: UOp):
  # split x and key from uint64 to two uint32
  x0, x1 = (x & 0xffffffff).cast(dtypes.uint32), ((x // 2**32) & 0xffffffff).cast(dtypes.uint32)
  key0, key1 = (key & 0xffffffff).cast(dtypes.uint32), ((key // 2**32) & 0xffffffff).cast(dtypes.uint32)

  rotations = [[13, 15, 26, 6], [17, 29, 16, 24]]
  ks = [key1, key0 ^ key1 ^ 0x1BD11BDA, key0]
  xr = [x0 + ks[-1], x1 + ks[0]]
  for i in range(5):
    for r in rotations[i % 2]: xr[0], xr[1] = (x0 := xr[0] + xr[1]), x0 ^ ((xr[1] * 2**r) + (xr[1] // 2**(32 - r)))
    xr = [(xr[0] + ks[i % 3]), (xr[1] + ks[(i + 1) % 3] + i + 1)]

  return xr[1].cast(dtypes.uint64) * 2**32 | xr[0].cast(dtypes.uint64)

# ******** phase 3 is the complete symbolic, and deals with very complex things like loop rewriting and threefry transform ********

def reduce_mul_chain(r:UOp):
  if r.arg not in {Ops.ADD, Ops.MAX}: return None
  if r.dtype != r.src[0].dtype: return None
  inside, outside = [], []
  for m in split_uop(r.src[0], Ops.MUL):
    m_parents = m.toposort()
    if all(r not in m_parents for r in r.src[1:]) and (r.arg != Ops.MAX or m.vmin >= 0): outside.append(m)
    else: inside.append(m)
  if len(outside) == 0: return None
  return r.replace(src=(prod(inside) if len(inside) else r.src[0].const_like(1),)+r.src[1:])*prod(outside)

# this is symbolic 2.0
sym = symbolic_flat+PatternMatcher([
  # self ASSIGN is just self
  (UPat(Ops.ASSIGN, src=(UPat.var('x'), UPat.var('x'))), lambda x: x),
  # VECTORIZE/CONST, VECTORIZE/GEP
  (UPat(Ops.VECTORIZE, src=UPat(Ops.CONST), name="vec"), lambda vec: UOp.const(vec.dtype, tuple(x.arg for x in vec.src))),
  (UPat(Ops.VECTORIZE, src=UPat(Ops.GEP, src=(UPat.var("x"),)), name="vec"), lambda vec,x: x.gep(tuple(y.arg[0] for y in vec.src))),
  # reorder ALU/VECTORIZE
  (UPat(GroupOp.ALU, src=(UPat(Ops.VECTORIZE, src=UPat(name='x')), UPat(Ops.VECTORIZE, src=UPat(name='y'))), name='alu'),
   lambda x,y,alu: UOp(Ops.VECTORIZE, alu.dtype, (UOp(alu.op, alu.dtype.scalar(), (x,y)),)*alu.dtype.count)),
  # VECTORIZE of a single element is just that element
  (UPat(Ops.VECTORIZE, src=(UPat(name='x'),)), lambda x: x),
  # VECTORIZE void is SINK
  (UPat(Ops.VECTORIZE, dtype=dtypes.void, src=UPat(Ops.BARRIER, name='b')), lambda b: b),
  (UPat(Ops.VECTORIZE, dtype=dtypes.void, name='x'), lambda x: UOp(Ops.SINK, dtypes.void, x.src)),
  # tensor core with a 0 input is acc
  (UPat(Ops.WMMA, src=(UPat.const(None, 0.0), UPat.var(), UPat.var("acc"))), lambda acc: acc),
  (UPat(Ops.WMMA, src=(UPat.var(), UPat.const(None, 0.0), UPat.var("acc"))), lambda acc: acc),
  # threefry + remove longs
  (UPat(Ops.THREEFRY, dtype=dtypes.uint64, src=(UPat.var("x"), UPat.var("key"))), threefry2x32),
  (UPat.var('x', dtypes.uint32).cast(dtypes.uint64).cast(dtypes.uint32), lambda x: x),   # cast there and back is noop (TODO: genericize)
  ((UPat.var('x', dtypes.uint64)&0xFFFFFFFF).cast(dtypes.uint32), lambda x: x.cast(dtypes.uint32)),  # cast does truncation
  (((UPat.var(None, dtypes.uint64)*(1<<32)) | UPat.var('y',  dtypes.uint32).cast(dtypes.uint64)).cast(dtypes.uint32), lambda y: y),
  (((UPat.var('x',  dtypes.uint64)*(1<<32)) | UPat.var(None, dtypes.uint32).cast(dtypes.uint64))//(1<<32), lambda x: x),
  # hacks for threefry long removal when padded (TODO: genericize)
  (UPat.var('x', dtypes.uint32).cast(dtypes.uint64) * UPat.var('y').where(UPat.const(dtypes.uint64, 1<<32), UPat.const(dtypes.uint64, 0)),
   lambda x,y: y.where(x, UOp.const(dtypes.uint32, 0)).cast(dtypes.uint64) * (1<<32)),
  ((UPat.var('x', dtypes.uint64)&(UPat.var('y').where(UPat.const(dtypes.uint64, 0xFFFFFFFF), UPat.const(dtypes.uint64, 0)))).cast(dtypes.uint32),
   lambda x,y: y.where(x.cast(dtypes.uint32), UOp.const(dtypes.uint32, 0))),
  # ** self folding **
  (UPat(Ops.DEFINE_ACC, src=(UPat.var("x"),)), lambda x: x),            # a DEFINE_ACC without ranges is a CONST
  (UPat(Ops.ASSIGN, src=(UPat.cvar(),UPat.var("x"))), lambda x: x),     # an ASSIGN to a const is a NOOP
  # x!=0 -> (bool)x
  (UPat.var("x")!=0, lambda x: x.cast(dtypes.bool.vec(x.dtype.count))),
  # ** where **
  # push cast to branches
  (UPat.var("s").where(UPat.var("a"), UPat.var("b")).cast().named("cast"), lambda s,a,b,cast: s.where(a.cast(cast.dtype), b.cast(cast.dtype))),
  # a.where(b.where(c, d), d) -> (a & b).where(c, d)
  (UPat.var("a").where(UPat.var("b").where(UPat.var("c"), UPat.var("d")), UPat.var("d")), lambda a,b,c,d: (a&b).where(c,d)),
  # ** pow **
  ((UPat(Ops.POW, name="p"), lambda p: xpow(*p.src))),
  # ** load/store folding **
  (UPat.store(UPat(Ops.INDEX, name="index"), UPat.load(UPat(Ops.INDEX, name="index"))), lambda index: UOp(Ops.NOOP)),
  (UPat.store(UPat(Ops.INDEX, name="index"), UPat.var("gate").where(UPat.var("alt"), UPat.load(UPat(Ops.INDEX, name="index")))),
   lambda index, gate, alt: UOp.store(index.src[0].index(index.src[1], gate), alt)),
  # fold gated LOAD/STORE
  (UPat().index(UPat(), UPat.const(dtypes.bool, True)).named("idx"), lambda idx: idx.replace(src=idx.src[0:2])), # remove True
  (UPat().index(UPat(), UPat.const(dtypes.bool, False)).named("idx"), lambda idx: idx.const_like(0)),      # False -> NULL pointer
  (UPat(Ops.LOAD, src=(UPat.const(None, 0),), allow_any_len=True, name="x"), lambda x: x.const_like(0)),  # NULL pointer load loads 0
  (UPat(Ops.STORE, src=(UPat.const(None, 0),), allow_any_len=True), lambda: UOp(Ops.NOOP)),  # NULL pointer store does nothing
  # remove NOOPs from SINK
  (UPat(Ops.SINK, name="root"),
    lambda root: UOp(Ops.SINK, root.dtype, a, root.arg) if len(a:=tuple(x for x in root.src if x.op is not Ops.NOOP)) != len(root.src) else None),
  # remove VECTORIZE from SINK/BARRIER
  (UPat(Ops.BARRIER, src=(UPat((Ops.VECTORIZE, Ops.SINK), name='sink'),)), lambda sink: UOp(Ops.BARRIER, dtypes.void, sink.src)),
  (UPat(Ops.SINK, name="root"),
    lambda root: UOp(Ops.SINK, root.dtype, tuple(flatten(x.src if x.op in {Ops.SINK, Ops.UNROLL} else (x,) for x in root.src)), root.arg)
      if any(x.op in {Ops.SINK, Ops.UNROLL} for x in root.src) else None),
  ((UPat.var("x") * UPat.var("x")).reciprocal(), lambda x: x.reciprocal()*x.reciprocal()),  # 1/(x^c) -> (1/x)^c
  ((UPat.var("x") * UPat.var("x") * UPat.var("x")).reciprocal(), lambda x: x.reciprocal()*x.reciprocal()*x.reciprocal()),
  (UPat.var("x") * ((1+UPat.var("x")).reciprocal().named("d")), lambda x,d: 1-d), # x*/(1+x) -> 1-1/(1+x)
  (UPat.var("x") * ((1+UPat.var("x")).reciprocal().named("d")*UPat.var("y")), lambda x,y,d: y*(1-d)),
  (UPat.var("x") * ((1+UPat.var("x")).reciprocal().named("d")+UPat.var("y")), lambda x,y,d: (1-d)+x*y),
  # move const multiply after REDUCE (NOTE: the mul chain can do this, but only if it's a same dtype reduce)
  ((UPat.var("x")*UPat.cvar("c", vec=False)).reduce(arg=Ops.ADD, name="r", allow_any_len=True), lambda x,c,r: r.replace(src=(x,)+r.src[1:])*c.arg),
  # reduce mul chain, move muls after the reduce
  (UPat(Ops.MUL).reduce(name="r", allow_any_len=True), reduce_mul_chain),
])
