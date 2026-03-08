import functools
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp
from tinygrad.dtype import dtypes
from tinygrad.helpers import cdiv, cmod, CORRECT_DIVMOD_FOLDING, unwrap

# NOTE: this cache is only on index UOps
@functools.cache
def fold_divmod_general(d: UOp, correct_divmod_folding: bool) -> UOp|None:
  x, y = d.src

  # cancel_divmod: simple cancel div/mod case when the range of the numerator lies within a single denominator interval
  x_min, x_max, y_min, y_max = x.vmin, x.vmax, y.vmin, y.vmax
  assert isinstance(x_min, int) and isinstance(x_max, int) and isinstance(y_min, int) and isinstance(y_max, int)
  if y_min==y_max==0: raise ZeroDivisionError(f"{'Division' if d.op is Ops.IDIV else 'Mod'} by zero trying to rewrite {x.alu(d.op, y)}")
  if y_min*y_max > 0 and (q:=cdiv(x_min,y_min)) == cdiv(x_min,y_max) == cdiv(x_max,y_min) == cdiv(x_max,y_max):
    return x - q*y if d.op is Ops.MOD else d.const_like(q)

  # split uops for the rest of the processing
  x_peeled, const = x.pop_const()
  uops_no_const = list(x_peeled.split_uop(Ops.ADD))

  # ** Constant Denominator Rules **
  # these rules strictly require y to be a scalar constant > 0
  if y.op is Ops.CONST and (c := y.arg) > 0:
    # remove_nested_mod: remove nested mod in case the inner mod is a multiple of the outer mod, example: (a%4 + b)%2 -> (a+b)%2
    if d.op is Ops.MOD and x.vmin >= 0:
      new_xs, changed = [], False
      for u in uops_no_const:
        if u.op is Ops.MOD and u.src[1].divides(c) is not None:
          u = u.src[0]
          changed = True
        new_xs.append(u)
      if changed and (new_x:=(UOp.sum(*new_xs) + const)).vmin >= 0: return new_x % y

    # Shared decomposition for folding rules
    decomp = [(u.divides(f:=u.const_factor()),f) for u in uops_no_const]
    terms, factors = zip(*decomp)

    # fold_binary_numerator: fold if expression has one non-constant term that takes on two values
    if len(terms)==1 and (v:=terms[0]).vmax-v.vmin == 1:
      y1 = cmod(factors[0]*v.vmin+const, c) if d.op is Ops.MOD else cdiv(factors[0]*v.vmin+const, c)
      y2 = cmod(factors[0]*v.vmax+const, c) if d.op is Ops.MOD else cdiv(factors[0]*v.vmax+const, c)
      return (y2-y1)*(v-v.vmin) + y1

    # fold_divmod_congruence: fold if a is congruent to an expression whose range is between 0 and c
    if not (x.vmin<0 and correct_divmod_folding):
      rems = [min((r:=f%c), r-c, key=abs) for f in factors]
      if (rem:=sum(r*v for r,v in zip(rems,terms))+const%c).vmin//c==rem.vmax//c:
        if d.op is Ops.MOD: return rem - rem.vmin//c*c
        return sum((f-r)//c * v for f,r,v in zip(factors,rems,terms)) + (const-const%c+rem.vmin//c*c)//c

    # gcd_with_remainder: factor out common gcd from numerator
    # Note: this rule uses uops_no_const to exclude the additive constant from the GCD calculation
    if x.vmin >= 0:
      gcd = UOp.gcd(*uops_no_const, y).simplify()
      if gcd.op is Ops.CONST and gcd.arg > 1:
        new_x = unwrap(x_peeled.divide_exact(gcd)).simplify() + (const%c)//gcd.arg
        if new_x.vmin >= 0:
          ret = new_x.alu(d.op, x.ufix(c//gcd.arg))
          return ret*gcd + const%gcd.arg if d.op is Ops.MOD else ret+const//c

    # nest_div_by_smallest_factor: try and nest the div and see if it allows the numerator to be simplified
    if d.op is Ops.IDIV and x.vmin >= 0:
      div = min([c] + [abs(f) for u, f in zip(uops_no_const, factors) if u.op not in (Ops.CONST, Ops.VCONST) and abs(f) > 1 and (c%f)==0])
      # NOTE: this is recursive!
      if div < c and (newxs := fold_divmod_general(x//div, correct_divmod_folding)) is not None and newxs.vmin >= 0:
        return newxs // (c // div)

  # ** Variable Denominator / Fallback Rules **
  # These rules apply to variables OR constants that failed the checks above.
  # Reconstruct all uops including const for these checks.
  all_uops = uops_no_const + ([x.const_like(const)] if const != 0 else [])

  # divide_by_gcd: x//y -> (x//gcd)//(y//gcd)
  gcd = UOp.gcd(*all_uops, y).simplify()
  if not (gcd.op is Ops.CONST and gcd.arg==1):
    ret = unwrap(x.divide_exact(gcd)).alu(d.op, unwrap(y.divide_exact(gcd)))
    return ret*gcd if d.op is Ops.MOD else ret

  # factor_remainder: (d*x+y)//d -> x+y//d
  if y.vmin<0 or x.vmin<0: return None
  quo, rem = [], []
  for u in all_uops:
    if (q:=u.divide_exact(y)) is not None: quo.append(q)
    elif d.op is Ops.MOD and y.op is Ops.CONST and (c:=u.const_factor())%y.arg!=c:
      rem.append(u.divides(c)*(c%y.arg))
      quo.append(u.const_like(0))
    else: rem.append(u)

  if not quo: return None
  new_x = sum(rem)+x.const_like(0)
  if new_x.vmin<0: return None
  return new_x%y if d.op is Ops.MOD else new_x//y+sum(quo)

div_and_mod_symbolic = PatternMatcher([
  # ** 1. Fast Inline Rules **
  ((UPat.var("x")//UPat.cvar("c") + UPat.cvar("a"))//UPat.cvar("d"), lambda x,c,a,d: (x+a*c)//(c*d)
    if c.vmin>0 and d.vmin>0 and ((x.vmin>=0 and a.vmin>=0) or (x.vmax<=0 and a.vmax<=0)) else None),  # (x//c+a)//d -> (x+a*c)//(c*d)
  (UPat.var("x", dtypes.index) // UPat.var("d"), lambda x,d: -(x//(-d)) if d.vmax < 0 else None),
  (UPat.var("x", dtypes.index) // UPat.var("d"), lambda x,d: -((-x)//d) if x.vmax <= 0 else None),
  ((UPat.var("x", dtypes.index)+UPat.cvar("c", vec=False)).named("n")//UPat.cvar("d", vec=False),
    lambda x,c,n,d: ((x+c.arg%d.arg)//d + c.arg//d.arg) if c.arg%d.arg!=c.arg and x.vmin>=0 and n.vmin>=0 and d.arg>0 else None),
  ((UPat.var("x", dtypes.index)+UPat.cvar("c", vec=False)).named("n")//UPat.cvar("d", vec=False),
    lambda x,c,n,d: (-(-(c.arg%d.arg + x - (d.arg-1))//d) + c.arg//d.arg) if x.vmax<=0 and n.vmin>=0 and d.arg>0 else None),

  # ** 2. Slow Rules **
  (UPat((Ops.IDIV, Ops.MOD), dtypes.index, name="d"), lambda d: fold_divmod_general(d, bool(CORRECT_DIVMOD_FOLDING))),

  # NOTE: these have to go at the bottom or TestSymbolicOps.test_var loops
  (UPat.var("x", dtypes.index) % UPat.var("d"), lambda x,d: -((-x)%d) if x.vmax <= 0 else None),
  (UPat.var("x", dtypes.index) % UPat.var("d"), lambda x,d: (x%(-d)) if d.vmax < 0 else None),
])