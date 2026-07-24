from typing import Callable
import functools
from tinygrad.dtype import dtypes, promo_lattice
from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher
from tinygrad.renderer import Renderer

# *** integer division ***

@functools.lru_cache(None)
def magicgu(vmax:int, d:int) -> tuple[int,int]:
  # calculate m,s such that x//d == (x*m) >> s for all 0 <= x <= vmax, d>0; adapted from Hacker's Delight, Chapter 10
  nc = (vmax+1)//(d) * d - 1
  nbits = vmax.bit_length()
  for s in range(0, 2*nbits + 1):
    if 2**s > nc*(d - 1 - (2**s - 1) % d):
      m = (2**s + d - 1 - (2**s - 1) % d)//d
      return m, s
  assert False

def fast_idiv(ren: Renderer, x: UOp, d: int, dont_cast=False) -> UOp|None:
  from tinygrad.renderer.cstyle import MetalRenderer
  # NOTE: disable for METAL due to compiler bug. keccak with -O0 works but not with optimization
  if isinstance(ren, MetalRenderer): return None
  # If d is a power of two this is not valid for signed ints!
  is_unsigned = x.vmin>=0 or x.dtype in dtypes.uints
  assert d>0, "Sign should have been taken out of divisor"
  vmin,vmax = max(x.vmin, x.dtype.min), min(x.vmax, x.dtype.max)
  if vmin > -d and vmax < d: return x.const_like(0)
  m,s = magicgu(max(vmax, abs(vmin)), d)
  if m*vmin >= x.dtype.min and m*vmax <= x.dtype.max:
    return ((x*m) >> s) if is_unsigned else ((x*m) >> s) + (x<0).where(x.ufix(1), 0)
  # before we try casting to a larger dtype (slow), we see if there are powers of two in d we can shift to make x smaller
  # use explicit Ops.CDIV (trunc) since the recursion assumes trunc semantics throughout
  if (largest_factor_of_two_in_d := (d & -d)) > 1:
    if (ret:=fast_idiv(ren, x.alu(Ops.CDIV, x.const_like(largest_factor_of_two_in_d)),
                       d//largest_factor_of_two_in_d, dont_cast=True)) is not None: return ret
  if dont_cast: return None
  # promo_lattice needs to return an unsigned type if the type is unsigned
  if dtypes.is_int(next_dtype := promo_lattice[x.dtype][-1]) and next_dtype in ren.supported_dtypes():
    if m*vmin >= next_dtype.min and m*vmax <= next_dtype.max:
      return ((x.cast(next_dtype)*m) >> s).cast(x.dtype) if is_unsigned else ((x.cast(next_dtype)*m) >> s).cast(x.dtype) + (x<0).where(x.ufix(1), 0)
  return None

# ***** threefry *****

def threefry2x32(x: UOp, key: UOp):
  # split x and key from uint64 to two uint32
  x0, x1 = (x & 0xffffffff).cast(dtypes.uint32), ((x // 2**32) & 0xffffffff).cast(dtypes.uint32)
  key0, key1 = (key & 0xffffffff).cast(dtypes.uint32), ((key // 2**32) & 0xffffffff).cast(dtypes.uint32)

  rotations = [[13, 15, 26, 6], [17, 29, 16, 24]]
  ks = [key1, key0 ^ key1 ^ 0x1BD11BDA, key0]
  xr:list[UOp] = [x0 + ks[-1], x1 + ks[0]]
  for i in range(5):
    for r in rotations[i % 2]: xr[0], xr[1] = (x0 := xr[0] + xr[1]), x0 ^ ((xr[1] * 2**r) + (xr[1] // 2**(32 - r)))
    xr = [(xr[0] + ks[i % 3]), (xr[1] + ks[(i + 1) % 3] + i + 1)]

  return xr[1].cast(dtypes.uint64) * 2**32 | xr[0].cast(dtypes.uint64)

# ***** decomposition patterns *****

def floordiv_to_idiv(a:UOp, b:UOp) -> UOp:
  if (a.vmin >= 0 and b.vmin > 0) or (a.vmax <= 0 and b.vmax < 0): return a.alu(Ops.CDIV, b)
  return a.alu(Ops.CDIV, b) - (a.alu(Ops.CMOD, b).ne(0) & (a<0).ne(b<0)).cast(a.dtype)

def floormod_to_mod(a:UOp, b:UOp) -> UOp:
  if (a.vmin >= 0 and b.vmin > 0) or (a.vmax <= 0 and b.vmax < 0): return a.alu(Ops.CMOD, b)
  r = a.alu(Ops.CMOD, b)
  # use where instead of mul to avoid being fused into MULACC (which int64 long-decomp doesn't handle)
  return r + (r.ne(0) & (a<0).ne(b<0)).where(b, b.const_like(0))

powers_of_two: dict[int, int] = {2**i:i for i in range(64)}
@functools.cache
def get_simplifying_rewrite_patterns(ops:tuple[Ops, ...]) -> PatternMatcher:
  # these are rewrites that make things simpler
  pat: list[tuple[UPat, Callable]] = [(UPat.var("a")//UPat.var("b"), floordiv_to_idiv)]
  # FLOORMOD by 2**y -> x & (2**y-1) (correct floor mod for any sign in two's complement); fires before floormod_to_mod
  if Ops.AND in ops: pat.append((UPat.var("x", dtypes.ints)%UPat.cvar("c"), lambda x,c: x & (c.arg-1) if c.arg in powers_of_two else None))
  pat.append((UPat.var("a")%UPat.var("b"), floormod_to_mod))
  # no real hardware supports THREEFRY, but NullRenderer does
  if Ops.THREEFRY not in ops: pat.append((UPat(Ops.THREEFRY, dtype=dtypes.uint64, src=(UPat.var("x"), UPat.var("key"))), threefry2x32))
  # MAX can be rewritten as CMPLT + WHERE (max function is annoying on many cstyle backends)
  if Ops.MAX not in ops and Ops.CMPLT in ops: pat.append((UPat(Ops.MAX, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])))
  return PatternMatcher(pat)

@functools.cache
def get_late_rewrite_patterns(ops:tuple[Ops, ...], disable_fast_idiv:bool) -> PatternMatcher:
  pat: list[tuple[UPat, Callable]] = []
  if Ops.OR in ops: pat += [(UPat.var("x", dtypes.bool).logical_not()&UPat.var("y", dtypes.bool).logical_not(),
    lambda x,y: (x | y).logical_not())]
  # rewrite MUL/CDIV to SHL+SHR: x*(2**y) -> shl(x,y) and x//(2**y) -> shr(x,y)
  if Ops.SHL in ops: pat += [(UPat.var("x", dtypes.ints)*UPat.cvar("c"), lambda c,x: x << v if (v:=powers_of_two.get(c.arg, 0)) else None)]
  if Ops.SHR in ops:
    # uint CDIV by 2**v -> x >> v (FLOORDIV is lowered to CDIV by the rule above before reaching here)
    pat += [(UPat(Ops.CDIV, src=(UPat.var("x", dtypes.uints), UPat.cvar("c"))),
      lambda x,c: x >> v if (v:=powers_of_two.get(c.arg, 0)) else None)]
    # signed CDIV (trunc) by 2**v -> (x + (x<0 ? c-1 : 0)) >> v
    pat += [(UPat(Ops.CDIV, src=(UPat.var("x", dtypes.ints), UPat.cvar("c"))),
      lambda x,c: (x+(l.const_like(l.vmin) if (l:=(x<0)).vmin==l.vmax else l).where(c-1, 0)) >> v
        if (v:=powers_of_two.get(c.arg, 0)) else None)]
    if not disable_fast_idiv:
      # fast_idiv handles non-pow2: only fire on non-negative inputs (signed magic-mul is unreliable for x<0)
      pat += [(UPat(Ops.CDIV, src=(UPat.var("x", dtypes.ints), UPat.cvar("d"))),
        lambda ctx, x, d: fast_idiv(ctx, x, d.arg) if x.vmin >= 0 or x.dtype in dtypes.uints else None)]
      # rewrite raw CMOD -> x - d*CDIV(x,d) so fast_idiv can pick up the CDIV. only on non-negative inputs;
      # avoids disturbing floormod_to_mod's general-path output (which uses a trunc Ops.CMOD as an implementation detail)
      pat += [(UPat(Ops.CMOD, src=(UPat.var("x", dtypes.ints), UPat.var("d"))),
        lambda x, d: x - d * x.alu(Ops.CDIV, d) if x.vmin >= 0 or x.dtype in dtypes.uints else None)]
  if Ops.NEG in ops:
    pat += [(UPat.var('x')*-1, lambda ctx,x: x.alu(Ops.NEG))]
    if Ops.SUB in ops: pat += [(UPat.var('x')+UPat.var('y').alu(Ops.NEG), lambda ctx,x,y: x.alu(Ops.SUB, y))]
  if Ops.CMPLT in ops:
    # These are late rewrites because simplex expects equalities to be a certain format
    pat += [
      ((UPat.var("x", dtypes.sints) < UPat.cvar("c", dtypes.sints)).logical_not(), lambda x,c: c-1<x),
      ((UPat.cvar("c", dtypes.sints) < UPat.var("x", dtypes.sints)).logical_not(), lambda x,c: x<c+1),
      (UPat.var("x", dtypes.sints)*-1 < UPat.var("y", dtypes.sints)*UPat.cvar("c"), lambda x,y,c: y*(-c)<x),
      (UPat.var("x", dtypes.sints)*-1 < UPat.cvar("c"), lambda x,c:-c<x),
      ((UPat.cvar("c1")<UPat.var("x", dtypes.sints)) & (UPat.var("x", dtypes.sints)<UPat.cvar("c2")),
        lambda x,c1,c2: x.eq(c1+1) if c1.arg+1==c2.arg-1 else None),  # (c-1)<x & x<(c+1) -> x==c
    ]
  if Ops.CMPEQ in ops: pat += [(UPat.var('x').ne(UPat.var('y')).logical_not(), lambda x,y: x.alu(Ops.CMPEQ, y))]
  if Ops.MULACC in ops:
    pat += [(UPat.var('a')*UPat.var('b')+UPat.var('c'), lambda a,b,c: a.alu(Ops.MULACC, b, c))]
    # also fuse (x << n) + c → MULACC(x, 2^n, c) since MUL→SHL may run first
    if Ops.SHL in ops: pat += [(UPat.var('x').alu(Ops.SHL, UPat.cvar('n'))+UPat.var('c'), lambda x,n,c: x.alu(Ops.MULACC, x.const_like(1<<n.arg), c))]
  # some backends emit FDIV for RECIP, in that case: a*(1/b) -> a/b
  if Ops.FDIV in ops:
    pat += [(UPat.var("x").reciprocal(), lambda x: x.const_like(1).alu(Ops.FDIV, x))]
    pat += [(UPat.var("a", dtypes.floats) * UPat.const(dtypes.floats, 1).alu(Ops.FDIV, UPat.var("b")), lambda a,b: a.alu(Ops.FDIV, b))]
  return PatternMatcher(pat)
