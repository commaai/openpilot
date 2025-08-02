# should this merge with transcendental?
from typing import Callable
import functools
from tinygrad.device import is_dtype_supported
from tinygrad.dtype import dtypes, promo_lattice
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher
from tinygrad.helpers import getenv
from tinygrad.uop.transcendental import xexp2, xlog2, xsin, xpow, TRANSCENDENTAL_SUPPORTED_DTYPES
from tinygrad.renderer import Renderer

# ***** optional patterns *****

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

def fast_idiv(ctx: Renderer|None, x: UOp, d: int) -> UOp|None:
  # idiv is truncated division, but arithmetic shift is floored division, so can only do non-negative numbers!
  if x.vmin<0: return None
  sign = 1 if d > 0 else -1
  m,s = magicgu(vmax := min(x.vmax, dtypes.max(x.dtype)), abs(d))
  if m * vmax <= dtypes.max(x.dtype): return sign * ((x*m) >> s)
  # promo_lattice needs to return an unsigned type
  if ctx is not None and dtypes.is_int(next_dtype := promo_lattice[x.dtype][-1]) and is_dtype_supported(next_dtype, ctx.device):
    if m * vmax <= dtypes.max(next_dtype): return sign * ((x.cast(next_dtype)*m) >> s).cast(x.dtype)
  return None

powers_of_two = {2**i:i for i in range(64)}
@functools.cache
def get_late_rewrite_patterns(ops, force_transcendental=False):
  pat: list[tuple[UPat, Callable]] = [(UPat(op, dtype=TRANSCENDENTAL_SUPPORTED_DTYPES, src=(UPat.var("d"),)), f) for op,f in \
           ((Ops.EXP2, xexp2), (Ops.LOG2, xlog2), (Ops.SIN, xsin)) if op not in ops or force_transcendental]
  # rewrite SQRT to xpow 0.5
  if Ops.SQRT not in ops: pat.append((UPat(Ops.SQRT, src=UPat.var("d")), lambda d: xpow(d, d.const_like(0.5))))
  # rewrite MOD to AND (which should always be supported, but not for generic in tests): x % (2**y) -> x & (2**y-1)
  if Ops.AND in ops: pat += [(UPat.var("x", dtypes.ints)%UPat.cvar("c"), lambda x,c: x & (c.arg-1) if c.arg in powers_of_two else None)]
  # rewrite MUL/IDIV to SHL+SHR: x*(2**y) -> shl(x,y) and x//(2**y) -> shr(x,y)
  if Ops.SHL in ops: pat += [(UPat.var("x", dtypes.ints)*UPat.cvar("c"), lambda c,x: x << v if (v:=powers_of_two.get(c.arg, 0)) else None)]
  if Ops.SHR in ops:
    # no reason to check x<0 for uints
    pat += [(UPat.var("x", dtypes.uints)//UPat.cvar("c"), lambda x,c: x >> v if (v:=powers_of_two.get(c.arg, 0)) else None)]
    pat += [(UPat.var("x", dtypes.ints)//UPat.cvar("c"), lambda x,c: (x+(l.const_like(l.vmin) if (l:=(x<0)).vmin==l.vmax else l).where(
      c-1, 0)) >> v if (v:=powers_of_two.get(c.arg, 0)) else None)]  # (x+(x<0).where(c-1, 0)) >> v
    if not getenv("DISABLE_FAST_IDIV"):
      pat += [(UPat.var("x", dtypes.ints)//UPat.cvar("d"), lambda ctx, x, d: fast_idiv(ctx, x, d.arg))]
      pat += [(UPat.var("x", dtypes.ints)%UPat.cvar("d"), lambda ctx, x, d: x - d*f if (f:=fast_idiv(ctx, x, d.arg)) is not None else None)]
  if Ops.NEG in ops:
    pat += [(UPat.var('x')*-1, lambda x: x.alu(Ops.NEG))]
    if Ops.SUB in ops: pat += [(UPat.var('x')+UPat.var('y').alu(Ops.NEG), lambda x,y: x.alu(Ops.SUB, y))]
  if Ops.MULACC in ops: pat += [(UPat.var('a')*UPat.var('b')+UPat.var('c'), lambda a,b,c: a.alu(Ops.MULACC, b, c))]
  return PatternMatcher(pat)
