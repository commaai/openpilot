from dataclasses import replace
from tinygrad.dtype import dtypes, DType, AddrSpace, Invalid, least_upper_dtype, strong_dtype, weak_dtype
from tinygrad.helpers import unwrap
from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher, GroupOp, graph_rewrite, dtype_from_uop

def select_dtype(u:UOp):
  if u.dtype is dtypes.weakfloat: return dtypes.default_float
  return dtypes.long if u.overflows(dtypes.int32) else dtypes.int

def lower_weak_node(u:UOp) -> UOp|None:
  start, src = (1 if u.op is Ops.WHERE else 0), tuple(s.src[0] if s.op is Ops.CAST and s.dtype in dtypes.weaks else s for s in u.src)
  if src == u.src or any(s.dtype in dtypes.weaks for s in src[start:]): return None
  dt = strong_dtype(least_upper_dtype(select_dtype(u), *(s.dtype for s in src)) if u.op in GroupOp.Binary
                    else unwrap(dtype_from_uop(u.op, src, u.arg)))
  return u.replace(dtype=None, src=src[:start]+tuple(s if s.base.is_invalid else s.cast(dt) for s in src[start:])).cast(u.dtype)

pm_lower_weak = PatternMatcher([
  (UPat(Ops.CONST, dtype=dtypes.weaks, name="u"), lambda u: UOp.const(u.val, select_dtype(u)).cast(u.dtype)),
  # two stacked weak casts are a weakint value used as weakfloat (or vice versa): resolve the inner one at the outer kind's default.
  # a SINGLE weak cast is never rewritten here, each consumer absorbs it on its own edge (see lower_weak_srcs)
  (UPat(Ops.CAST, dtype=dtypes.weaks, src=(UPat(Ops.CAST, dtype=dtypes.weaks, src=(UPat.var("x"),)),), name="u"),
   lambda u,x: x.cast(select_dtype(u)).cast(u.dtype) if x.dtype not in dtypes.weaks else None),
  # Binary can widen from the bounds, all other nodes derive from the lowered sources.
  # a weakfloat Unary (sin/exp2/...) must resolve here, before the transcendental decomposition
  (UPat(GroupOp.Binary|GroupOp.Unary|{Ops.WHERE, Ops.RANGE, Ops.STACK, Ops.SPECIAL}, name="u"), lower_weak_node),
  (UPat(Ops.PARAM, dtype=dtypes.weakint, name="u"),
    lambda u: u.replace(dtype=None, arg=replace(u.arg, dtype=select_dtype(u))).cast(dtypes.weakint) if u.addrspace == AddrSpace.ALU else None),
])

def lower_weak_srcs(ctx:dict[UOp, UOp]|None, u:UOp) -> UOp|None:
  if ctx is None: ctx = {}
  def lower(s:UOp) -> UOp:
    if (r:=ctx.get(s)) is None:
      r = graph_rewrite(s, pm_lower_weak)
      # the consumer absorbs the cast on its own edge
      ctx[s] = r = r.src[0] if r.op is Ops.CAST and r.dtype in dtypes.weaks else r
    return r
  # a comparison demands a common operand width: lower it whole so the Binary rule unifies its operands
  ret = lower(u) if u.op in GroupOp.Comparison else u.replace(src=tuple(lower(s) if s.dtype in dtypes.weaks else s for s in u.src))
  return None if ret is u else ret

def commit_weak(s:UOp, dt:DType) -> UOp:
  # a bare weak CONST commits directly (the value stays mathematical, emission truncates), a weak non-const src takes the demand cast
  return UOp.const(s.val, dt) if s.op is Ops.CONST else s.cast(dt)

def commit_weak_srcs(u:UOp) -> UOp|None:
  if not any(s.dtype in dtypes.weaks for s in u.src): return None
  if (dt:=least_upper_dtype(*(s.dtype for s in u.src))) in dtypes.weaks: return None
  # the root re-derives: a shift's dtype is its lhs's, so committing the lhs commits the node too
  return u.replace(dtype=None, src=tuple(commit_weak(s, dt) if s.dtype in dtypes.weaks else s for s in u.src))

# runs in index lowering and in the decomps: a rule that mints a weak const commits it in the same rewrite, so none reaches the renderer
pm_commit_weak = PatternMatcher([
  (UPat(GroupOp.Broadcastable, name="u"), commit_weak_srcs),
  # demand from the destination: a STORE's weak value commits at the destination's dtype
  (UPat(Ops.STORE, src=(UPat(), UPat(dtype=dtypes.weaks)), allow_any_len=True, name="u"),
   lambda u: u.replace(src=(u.src[0], commit_weak(u.src[1], u.src[0].dtype), *u.src[2:]))),
])

# a concrete CAST over a weak node states the width the value will live at. that width is a floor, never a narrowing
def cast_weak_srcs(c:UOp, u:UOp) -> UOp|None:
  if c.dtype in dtypes.weaks or weak_dtype(c.dtype) is not u.dtype: return None
  dt = least_upper_dtype(c.dtype, select_dtype(u))
  return u.replace(dtype=None, src=tuple(commit_weak(s, dt) if s.dtype in dtypes.weaks else s for s in u.src)).cast(c.dtype)

pm_cast_weak = PatternMatcher([
  (UPat(Ops.CAST, name="c", src=(UPat(GroupOp.ALU, dtype=dtypes.weaks, name="u"),)), cast_weak_srcs),
])

pm_lower_index_dtype = pm_commit_weak+pm_cast_weak+PatternMatcher([
  (UPat(GroupOp.All, name="u"),
   lambda ctx,u: lower_weak_srcs(ctx, u) if u.dtype not in dtypes.weaks and any(s.dtype in dtypes.weaks for s in u.src) else None),
  # a valid index into an n-element buffer lives in [0,n): a gated long index narrows when n-1 fits int32 (out-of-gate wraps, discarded)
  # TODO: more generic
  (UPat((Ops.INDEX, Ops.SHRINK), src=(UPat.var("buf"), UPat.var("gate").where(UPat.var("idx", dtypes.long), UPat(Ops.CONST, arg=Invalid))),
        allow_any_len=True, name="u"),
   lambda u,buf,gate,idx: u.replace(src=(buf, idx.cast(dtypes.int).valid(gate))+u.src[2:]) if buf.max_numel()-1 <= dtypes.int32.max else None),
])
