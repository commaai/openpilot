from dataclasses import replace
from tinygrad.dtype import dtypes, DType, AddrSpace, Invalid, least_upper_dtype, strong_dtype, weak_dtype

from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher, GroupOp, dtype_from_uop, promo_dtype

# the decomps and float emulation commit bare consts at a dtype another src already states
def commit_weak_consts(u:UOp, dt:DType|None) -> UOp|None:
  return None if dt is None else u.replace(src=tuple(s.ccast(dt) if s.op is Ops.CONST and s.dtype in dtypes.weaks else s for s in u.src))

# the concrete dtypes u commits its srcs at: the operands' meet and u's own derived dtype, None if either is weak
def derived_dtypes(u:UOp, src:tuple[UOp, ...]) -> tuple[DType, DType]|None:
  if u.op not in GroupOp.Broadcastable or (meet:=promo_dtype(src)) in dtypes.weaks \
     or (result:=dtype_from_uop(u.op, src, u.arg)) in dtypes.weaks: return None
  return meet, result

def commit_srcs_at(u:UOp, dt:DType) -> UOp|None:
  # the root re-derives: a shift's dtype is its lhs's, so committing the lhs commits the node too
  bare = derived_dtypes(u, u.src) is not None
  src = tuple(s if s.dtype not in dtypes.weaks else UOp.const(dt.const(s.val)) if bare and s.op is Ops.CONST else s.ccast(dt) for s in u.src)
  return None if (ret := u.replace(src=src)) is u else ret

def commit_weak_srcs(u:UOp) -> UOp|None:
  if not any(s.dtype in dtypes.weaks for s in u.src) or (dt:=least_upper_dtype(*(s.dtype for s in u.src))) in dtypes.weaks: return None
  return commit_srcs_at(u, dt)

# a concrete CAST over a weak node states the width the value will live at. that width is a floor, never a narrowing
def cast_weak_srcs(c:UOp, u:UOp) -> UOp|None:
  # only within the kind: an int cast of a weakfloat node is a value conversion, not a statement about the node's width
  if c.dtype in dtypes.weaks or weak_dtype(c.dtype) is not u.dtype: return None
  # every weak src commits at the one width: the node's own bounds and each src's, none of them narrowed
  dt = least_upper_dtype(c.dtype, u.commit_dtype(dtypes.int), *(s.commit_dtype(dtypes.int) for s in u.src if s.dtype in dtypes.weaks))
  return None if (ret:=commit_srcs_at(u, dt)) is None else ret.cast(c.dtype)

# rides every round that can mint a weak const, and must reach fixpoint before pm_lower_weak below defaults one
pm_commit_weak = PatternMatcher([
  (UPat(GroupOp.Broadcastable, name="u"), commit_weak_srcs),
  (UPat(Ops.STORE, src=(UPat(), UPat(dtype=dtypes.weaks)), allow_any_len=True, name="u"),
   lambda u: u.replace(src=(u.src[0], u.src[1].ccast(u.src[0].dtype), *u.src[2:]))),
  # no CONST arm: a concrete CAST over a weak CONST is already committed, minted that way by UOp.const
  (UPat(Ops.CAST, name="c", src=(UPat(GroupOp.ALU, dtype=dtypes.weaks, name="u"),)), cast_weak_srcs),
])

# consumers absorb the weak CAST off their srcs and default underivable consts; dtype-producing ops settle here.
# a weakfloat Unary (sin/exp2/...) must resolve before the transcendental decomposition.
_lower_weak_ops = GroupOp.Binary|GroupOp.Unary|{Ops.WHERE, Ops.RANGE, Ops.STACK, Ops.SPECIAL}
def lower_weak_node(u:UOp) -> UOp|None:
  if u.op is Ops.CAST and u.src[0].op is Ops.CONST: return None  # a committed const, not a consumer
  src = tuple(s.src[0] if s.op is Ops.CAST and s.dtype in dtypes.weaks else s for s in u.src)
  if derived_dtypes(u, src) is None:
    src = tuple(s.ccast(s.commit_dtype(dtypes.int)) if s.op is Ops.CONST and s.dtype in dtypes.weaks else s for s in src)
  if src == u.src: return None
  start = 1 if u.op is Ops.WHERE else 0  # WHERE's cond is bool, never part of the width unification
  if u.op not in _lower_weak_ops or any(s.dtype in dtypes.weaks and s.op is not Ops.CONST for s in src[start:]): return u.replace(src=src)
  # resolve whole once every weak expression lowered: a Binary widens from its own bounds too, derivable consts wait
  dt = strong_dtype(least_upper_dtype(u.commit_dtype(dtypes.int), *(s.dtype for s in src)) if u.op in GroupOp.Binary else
                    dtype_from_uop(u.op, src, u.arg))
  src = src[:start]+tuple(s if s.base.is_invalid or s.dtype in dtypes.weaks else s.ccast(dt) for s in src[start:])
  return u.replace(src=src).cast(u.dtype)

pm_lower_weak = PatternMatcher([
  # a gated long index into a small buffer narrows; its out-of-gate value is discarded
  (UPat((Ops.INDEX, Ops.SHRINK), src=(UPat.var("buf"), UPat.var("gate").where(UPat.var("idx", dtypes.long), UPat(Ops.CONST, arg=Invalid))),
        allow_any_len=True, name="u"),
   lambda u,buf,gate,idx: u.replace(src=(buf, idx.cast(dtypes.int).valid(gate))+u.src[2:]) if buf.max_numel()-1 <= dtypes.int32.max else None),
  # two stacked weak casts are two kind conversions: each resolves at its own kind's default
  (UPat(Ops.CAST, dtype=dtypes.weaks, src=(UPat(Ops.CAST, dtype=dtypes.weaks, src=(UPat.var("x"),)),), name="u"),
   lambda u,x: x.cast(u.src[0].commit_dtype(dtypes.int)).cast(u.commit_dtype(dtypes.int)).cast(u.dtype) if x.dtype not in dtypes.weaks else None),
  (UPat((Ops.PARAM, Ops.BUFFER), dtype=dtypes.weakint, name="u"),
    lambda u: u.replace(arg=replace(u.arg, dtype=u.commit_dtype(dtypes.int))).cast(dtypes.weakint) if u.addrspace == AddrSpace.ALU else None),
  (UPat(GroupOp.All, name="u"), lower_weak_node),
])

# drop the CAST off a committed const where the consumer re-derives it anyway, so bare-CONST rules keep matching.
# the drop must change nothing the consumer derives: neither the operands' meet nor the node's own dtype
def uncast_const(u:UOp) -> UOp|None:
  # a weak CAST over a const is not a commit, it is still resolving
  src = tuple(s.src[0] if s.op is Ops.CAST and s.dtype not in dtypes.weaks and s.src[0].op is Ops.CONST
              and s.src[0].dtype in dtypes.weaks else s for s in u.src)
  if src == u.src or (dts:=derived_dtypes(u, src)) is None or dts[0] != promo_dtype(u.src) or dts[1] is not u.dtype: return None
  return u.replace(src=src)

pm_uncast_const = PatternMatcher([(UPat(GroupOp.Broadcastable, name="u"), uncast_const)])

def cast_const(u:UOp, s:UOp) -> UOp:
  if s.op is not Ops.CONST or s.is_invalid: return s  # Invalid never commits
  # bool is the one strong bare dtype: cconst, since .cast(bool) would fold at construction
  if s.dtype is dtypes.bool: return UOp.cconst(s.val, s.dtype)
  # commit at the dtype its consumer derives
  return s.ccast(dts[0]) if (dts:=derived_dtypes(u, u.src)) is not None else s

# commit every remaining bare const, keyed on the consumer: "bare" is a property of the edge
def cast_consts(u:UOp) -> UOp|None:
  if u.op is Ops.CAST and u.src[0].op is Ops.CONST: return None  # a committed const's CONST is its value, not an edge
  return None if (src:=tuple(cast_const(u, s) for s in u.src)) == u.src else u.replace(src=src)

pm_cast_const = PatternMatcher([(UPat(GroupOp.All, name="u"), cast_consts)])
