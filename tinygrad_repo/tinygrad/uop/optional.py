from typing import Callable
import functools
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UPat, PatternMatcher
from tinygrad.helpers import getenv
from tinygrad.uop.transcendental import xexp2, xlog2, xsin, xpow, TRANSCENDENTAL_SUPPORTED_DTYPES, fast_idiv

# ***** optional patterns *****

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
  if Ops.CMPLT in ops:
    # These are late rewrites because simplex expects equalities to be a certain format
    pat += [
      ((UPat.var("x", dtypes.sints) < UPat.cvar("c", dtypes.sints)).logical_not(), lambda x,c: c-1<x),
      ((UPat.cvar("c", dtypes.sints) < UPat.var("x", dtypes.sints)).logical_not(), lambda x,c: x<c+1),
      (UPat.var("x", dtypes.sints)*-1 < UPat.var("y", dtypes.sints)*UPat.cvar("c"), lambda x,y,c: y*(-c)<x),
      (UPat.var("x", dtypes.sints)*-1 < UPat.cvar("c"), lambda x,c:-c<x),
      ((UPat.cvar("c1",vec=False)<UPat.var("x", dtypes.sints)) & (UPat.var("x", dtypes.sints)<UPat.cvar("c2",vec=False)),
        lambda x,c1,c2: x.eq(c1+1) if c1.arg+1==c2.arg-1 else None),  # (c-1)<x & x<(c+1) -> x==c
    ]
  if Ops.CMPEQ in ops: pat += [(UPat.var('x').ne(UPat.var('y')).logical_not(), lambda x,y: x.alu(Ops.CMPEQ, y))]
  if Ops.MULACC in ops: pat += [(UPat.var('a')*UPat.var('b')+UPat.var('c'), lambda a,b,c: a.alu(Ops.MULACC, b, c))]
  return PatternMatcher(pat)
