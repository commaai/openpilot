import functools, itertools, math
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp
from tinygrad.dtype import dtypes
from tinygrad.helpers import floordiv, floormod, unwrap

# NOTE: this cache is only on index UOps
@functools.cache
def fold_divmod_general(d: UOp) -> UOp|None:
  x, y = d.src

  # cancel_divmod: simple cancel div/mod case when the range of the numerator lies within a single denominator interval
  x_min, x_max, y_min, y_max = x.vmin, x.vmax, y.vmin, y.vmax
  assert isinstance(x_min, int) and isinstance(x_max, int) and isinstance(y_min, int) and isinstance(y_max, int)
  if y_min==y_max==0: raise ZeroDivisionError(f"{'Division' if d.op is Ops.FLOORDIV else 'Mod'} by zero trying to rewrite {x.alu(d.op, y)}")
  if y_min*y_max > 0 and (qv:=floordiv(x_min,y_min)) == floordiv(x_min,y_max) == floordiv(x_max,y_min) == floordiv(x_max,y_max):
    return x - qv*y if d.op is Ops.FLOORMOD else d.const_like(qv)

  # split uops for the rest of the processing
  x_peeled, const = x.pop_const()
  uops_no_const = list(x_peeled.split_uop(Ops.ADD))

  # ** Constant Denominator Rules **
  # these rules strictly require y to be a scalar constant > 0
  if y.op is Ops.CONST and (c := y.arg) > 0:
    # nested_div_mod: (x%(k*c))//c -> (x//c)%k (requires k>0), and (x%(k*c))%c -> x%c
    if x.op is Ops.FLOORMOD and (k := x.src[1].divides(c)) is not None:
      if d.op is Ops.FLOORMOD: return x.src[0] % y
      if k > 0: return x.src[0] // y % k

    # remove_nested_mod in sum: (a%4 + b)%2 -> (a+b)%2
    if d.op is Ops.FLOORMOD:
      new_xs, changed = [], False
      for u in uops_no_const:
        if u.op is Ops.FLOORMOD and u.src[1].divides(c) is not None:
          u = u.src[0]
          changed = True
        new_xs.append(u)
      if changed: return (UOp.usum(*new_xs) + const) % y

    # Shared decomposition for folding rules
    decomp = [(u.divides(f:=u.const_factor()),f) for u in uops_no_const]
    terms, factors = zip(*decomp)

    # fold_binary_numerator: fold if expression has one non-constant term that takes on two values
    if len(terms)==1 and (v:=terms[0]).vmax-v.vmin == 1:
      y1 = (floormod if d.op is Ops.FLOORMOD else floordiv)(factors[0]*v.vmin+const, c)
      y2 = (floormod if d.op is Ops.FLOORMOD else floordiv)(factors[0]*v.vmax+const, c)
      return (y2-y1)*(v-v.vmin) + y1

    # fold_divmod_congruence: fold if a is congruent to an expression whose range is between 0 and c
    # when f%c == c//2, abs(r) == abs(r-c) is a tie, try both signs since either may fit in one period
    rem_choices = [(r, r-c) if (r:=f%c)*2 == c else (min(r, r-c, key=abs),) for f in factors]
    for rems in itertools.product(*rem_choices):
      if (rem:=sum(r*v for r,v in zip(rems,terms))+const%c).vmin//c==rem.vmax//c:
        if d.op is Ops.FLOORMOD: return rem - rem.vmin//c*c
        return sum((f-r)//c * v for f,r,v in zip(factors,rems,terms)) + const//c + rem.vmin//c

    # gcd_with_remainder: factor out common gcd from numerator
    if (g:=math.gcd(*factors, c)) > 1:
      new_x = unwrap(x_peeled.divides(g)).simplify() + (const//g)%(c//g)
      if new_x.vmin >= 0:
        if d.op is Ops.FLOORMOD: return new_x % (c//g) * g + const%g
        return new_x // (c//g) + const//c

    # nest_by_factor: x//c -> (x//f)//(c//f), x%c -> (x//f%(c//f))*f + b where b=x%f
    # FLOORDIV identity holds for any sign of x; FLOORMOD reconstruction needs x.vmin>=0
    results = []
    for div in {abs(f) for u, f in zip(uops_no_const, factors) if u.op is not Ops.CONST and 1 < abs(f) < c and (c%f)==0}:
      if (newxs := fold_divmod_general(x//div)) is not None:
        if d.op is Ops.FLOORDIV:
          results.append((len(newxs.backward_slice), newxs // (c // div)))
        elif x.vmin >= 0 and newxs.vmin >= 0:
          b_parts = [f%div*t for f, t in zip(factors, terms) if f%div]
          if const % div: b_parts.append(x.const_like(const % div))
          b = UOp.usum(*b_parts) if b_parts else x.const_like(0)
          if 0 <= b.vmin and b.vmax < div:
            results.append((len((r:=(newxs % x.ufix(c//div))*div + b).backward_slice), r))
    if results: return min(results, key=lambda r: r[0])[1]

  # ** Variable Denominator / Fallback Rules **
  # These rules apply to variables OR constants that failed the checks above.
  # Reconstruct all uops including const for these checks.
  all_uops = list(x.split_uop(Ops.ADD))

  # divide_by_gcd: x//y -> (x//gcd)//(y//gcd)
  gcd = UOp.gcd(*all_uops, y).simplify()
  if not (gcd.op is Ops.CONST and gcd.arg==1):
    ret = unwrap(x.divide_exact(gcd)).alu(d.op, unwrap(y.divide_exact(gcd)))
    return ret*gcd if d.op is Ops.FLOORMOD else ret

  # factor_remainder: (d*x+y)//d -> x+y//d
  if y.vmin<0 or x.vmin<0: return None
  quo, rem = [], []
  for u in all_uops:
    if (q:=u.divide_exact(y)) is not None: quo.append(q)
    elif y.op is Ops.CONST and (c:=u.const_factor())%y.arg!=c:
      rem.append(u.divides(c)*(c%y.arg))
      quo.append(u.divides(c)*(c//y.arg) if d.op is Ops.FLOORDIV else u.const_like(0))
    else: rem.append(u)

  if not quo: return None
  new_x = sum(rem)+x.const_like(0)
  if new_x.vmin<0: return None
  return new_x%y if d.op is Ops.FLOORMOD else new_x//y+sum(quo)

div_and_mod_symbolic = PatternMatcher([
  # ** 1. Fast Inline Rules **
  # (x//c+a)//d -> (x+a*c)//(c*d) for c>0, d>0
  ((UPat.var("x")//UPat.cvar("c") + UPat.cvar("a"))//UPat.cvar("d"), lambda x,c,a,d: (x+a*c)//(c*d) if c.vmin>0 and d.vmin>0 else None),
  # (x+c)//d -> (x+c%d)//d + c//d for d>0 (split out the multiple of d in the constant)
  ((UPat.var("x", dtypes.weakint)+UPat.cvar("c", vec=False))//UPat.cvar("d", vec=False),
    lambda x,c,d: (x+c.arg%d.arg)//d + c.arg//d.arg if c.arg%d.arg!=c.arg and d.arg>0 else None),

  # ** 2. Slow Rules **
  (UPat((Ops.FLOORDIV, Ops.FLOORMOD), dtypes.weakint, name="d"), lambda d: fold_divmod_general(d)),
])
