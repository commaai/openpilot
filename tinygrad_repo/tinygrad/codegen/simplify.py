import itertools
from typing import Callable
from tinygrad.uop.ops import UOp, PatternMatcher, UPat, Ops, GroupOp, graph_rewrite, _substitute, range_start, AxisType, identity_element
from tinygrad.uop.symbolic import symbolic
from tinygrad.helpers import partition, dedup
from tinygrad.dtype import dtypes

def flatten_range(r:UOp) -> UOp|None:
  off = range_start[r.op]
  rngs = r.src[off:]
  return r.replace(src=r.src[:off]+tuple(dedup(x for s in rngs for x in ((s,) if s.op is Ops.RANGE else s.ranges))))

pm_flatten_range = PatternMatcher([
  # real ranges only
  (UPat((Ops.REDUCE, Ops.END), name="r"), flatten_range),
])

# TODO: something in GroupOp
pure_ops = GroupOp.Elementwise|GroupOp.Irreducible|{Ops.INDEX, Ops.LOAD, Ops.STACK}

def mergeable_pairs(x:UOp) -> list[tuple[UOp, UOp]]:
  # an END or a REDUCE with effects runs its ranges as loops nested by axis id, so only neighbors merge
  if x.op is Ops.REDUCE and all(y.op in pure_ops|{Ops.REDUCE} for y in x.backward_slice): return list(itertools.permutations(x.ended_ranges, 2))
  return list(itertools.pairwise(sorted(x.ended_ranges, key=lambda r: r.arg))) if x.op in {Ops.END, Ops.REDUCE} else []

def merge_ranges(u:UOp) -> UOp|None:
  enders = {x:mergeable_pairs(x) for x in u.backward_slice if x.op in range_start}
  impure = [x for x in u.backward_slice if x.op not in pure_ops]
  for r0, r1 in dedup(p for pairs in enders.values() for p in pairs):
    if r0.axis_type is not r1.axis_type or r0 in r1.ranges or r1 in r0.ranges: continue
    # every op that ends r0 or r1 must merge them, and an op that isn't pure must run as often as before
    if any((r0, r1) not in pairs for x, pairs in enders.items() if {r0, r1} & set(x.ended_ranges)): continue
    if any((r0 in x.ranges) != (r1 in x.ranges) for x in impure): continue
    s1 = r1.src[0]
    n = r0.replace(src=((r0.src[0]*s1).simplify(),))
    sub = {r0:n//s1, r1:n%s1}
    ret = graph_rewrite(u, _substitute+symbolic+pm_flatten_range, ctx=sub, name=f"check_merge_{r0.axis_id}_{r1.axis_id}")
    if not any(v in ret.backward_slice for v in sub.values()): return ret
  return None

def mark_gated(ctx, idx, store=None):
  if len(idx.src) > 1 and idx.src[1].op is Ops.WHERE:
    x, cond = idx.src[1].get_idx(), idx.src[1].get_valid()
    # get all ranges r with guards "r < c" for some const c
    guards = {r:c for v in cond.split_uop(Ops.AND) if v.op is Ops.CMPLT and (r:=v.src[0]).op is Ops.RANGE and (c:=v.src[1]).op is Ops.CONST}
  else: x, guards = idx, {}
  # ensure that we choose max(c_i) for all i where r < c_i
  ctx |= {r:c.minimum(r.src[0]) for r,c in guards.items() if (r not in ctx or ctx[r].vmax < c.val)}
  # but if a range is ever ungated, we cannot shrink it, and a STORE runs every iteration of a range its index doesn't gate
  ctx |= {r:r.src[0] for r in (x if store is None else store).ranges if r not in guards}

def do_substitute(ctx:dict, x: UOp, sub_fxn:Callable[[UOp, UOp], UOp]) -> UOp|None:
  # Only the kernel root: rewriting a nested SINK would leave its enclosing END's binders unchanged.
  if x.arg is None: return None
  ret = x.substitute({k:sub_fxn(k,v) for k,v in ctx.items() if v is not None and k in x.backward_slice})
  return None if ret is x else ret.simplify()

pm_simplify_ranges = PatternMatcher([
  (UPat(Ops.SINK, name="u"), merge_ranges),
  (UPat(Ops.INDEX, name="idx"), mark_gated),
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, name="idx"), UPat()), name="store"), mark_gated),
  # reduce ranges can't be shrunk
  (UPat(Ops.REDUCE, name="red"), lambda ctx, red: ctx.update({r:r.src[0] for r in red.src[1:]})),
  (UPat(Ops.SINK, name="x"), lambda ctx, x: do_substitute(ctx, x, lambda r,c: r.replace(src=(c,)))),
])

def mark_range_mod(ctx:dict[UOp, UOp|None], r:UOp, c:UOp) -> None:
  # ranges that aren't looped over can't be split
  if r not in ctx and r.axis_type not in {AxisType.WARP, AxisType.DEVICE} \
    and r.src[0].op is Ops.CONST and r.src[0].divides(c.val) is not None: ctx[r] = c

pm_split_ranges = PatternMatcher([
  (UPat(Ops.RANGE, name="r")%UPat.cvar("c"), mark_range_mod),
  (UPat(Ops.SINK, name="x"), lambda ctx, x: do_substitute(ctx, x,
    lambda k,v: k.replace(src=(k.src[0]//v,), arg=k.arg+(0,))*v + k.replace(src=(v,), arg=k.arg+(1,)))),
])

# **** reduce simplification ****

def no_range(u:UOp) -> bool: return not u.op_in_backward_slice_with_self(Ops.RANGE)

def reduce_unparented(red:UOp) -> UOp|None:
  assert all(x.op is Ops.RANGE for x in red.src[1:]), "some reduce srcs aren't ranges"
  reduce_parented, reduce_unparented = partition(red.src[1:], lambda x: x in red.src[0].ranges)
  if len(reduce_unparented) == 0: return None
  ret = red.replace(src=(red.src[0],)+tuple(reduce_parented)) if len(reduce_parented) else red.src[0]
  # the op applied n times, MAX and bool ADD/MUL (OR/AND) are idempotent. an empty range is the identity
  n, op = UOp.uprod(*[r.src[0] for r in reduce_unparented]), red.arg[0]
  if op is not Ops.MAX and red.dtype is not dtypes.bool: ret = ret*n if op is Ops.ADD else ret**n
  return (n > 0).where(ret, ret.const_like(identity_element(op, red.dtype)))

pm_reduce_unparented = PatternMatcher([
  # remove any ranges from a REDUCE that aren't referenced in the reduce source
  (UPat(Ops.REDUCE, name="red"), reduce_unparented),
])

pm_reduce_collapse = pm_reduce_unparented + PatternMatcher([
  # lift x+y out of reduce on lt
  (UPat(Ops.ADD, dtypes.sints+(dtypes.weakint,), [UPat.var("x"), UPat.var("y")]) < UPat.var("c"),
   lambda x,y,c: (x < (c-y)) if no_range(y) and no_range(c) else None),
  # lift x*y out of reduce
  (UPat(Ops.MUL, dtypes.sints+(dtypes.weakint,), [UPat.var("x"), UPat.var("y")]) < UPat.var("c"),
   lambda x,y,c: (x < ((c+y-1) // y)) if no_range(y) and no_range(c) and y.vmin > 0 else None),
  # sum over r in [0,N) of [lower<=r<upper]*val -> max(min(upper,N) - max(lower,0), 0) * val
  (UPat.any(
    (UPat(Ops.RANGE, name="r") < UPat.var("upper")).where(UPat.var("val"), 0),
    (UPat(Ops.RANGE, name="r") < UPat.var("lower")).where(0, UPat.var("val")),
    ((UPat.var("r")<UPat.var("lower")).logical_not()&(UPat(Ops.RANGE, name="r")<UPat.var("upper"))).where(UPat.var("val"), 0),
  ).reduce(UPat.var("r"), arg=Ops.ADD), lambda r,val,lower=None,upper=None:
    ((upper.minimum(r.src[0]) if upper is not None else r.src[0]) -
     (lower.maximum(0) if lower is not None else r.const_like(0))).maximum(0) * val if no_range(val) else None),
  ((UPat.var("x")+UPat.var("y")).reduce(arg=Ops.ADD, allow_any_len=True, name="r"),
   lambda x,y,r: x.reduce(*r.src[1:], arg=Ops.ADD) + y.reduce(*r.src[1:],arg=Ops.ADD)),
  # AND on WHERE
  ((UPat(Ops.PARAM, name="x") & UPat.var("y")).where(UPat.var("c"), 0).reduce(arg=Ops.ADD, allow_any_len=True, name="r"),
    lambda x,y,c,r: y.where(c, 0).reduce(*r.src[1:], arg=Ops.ADD)*x),
  # MUL casted bool
  ((UPat.var("x") * UPat.var("gate", dtype=dtypes.bool).cast()), lambda x,gate: gate.where(x, 0)),
])+symbolic

pm_reduce_load_collapse = pm_reduce_collapse + PatternMatcher([
  # lift x+y out of reduce on eq
  ((UPat.var("x", dtypes.wints)+UPat.var("y")).or_casted("a").eq(UPat.var("c", dtypes.wints)),
   lambda x,y,a,c: x.eq(c.cast(y.dtype)-y) if no_range(y) and no_range(c) and not no_range(x) and
   (a.op is not Ops.CAST or not a.src[0].overflows(a.dtype)) and not c.overflows(y.dtype) else None),
  # reduce on gated load becomes can substitute the range and remove the reduce
  (UPat.var("idx", dtypes.wints).eq(UPat(Ops.RANGE, name="r").or_casted("rc")).where(UPat.var("expr"), 0)
   .reduce(UPat.var("r"), arg=Ops.ADD),
   lambda r,rc,idx,expr: None if r.overflows(rc.dtype) else
   (v:=(idx.cast(r.dtype) >= 0) & (idx.cast(r.dtype) < r.src[0])).where(expr.substitute({r:idx.cast(r.dtype).valid(v)}),0)),
])

def reduce_collapse(red:UOp, u:UOp, pm:PatternMatcher=pm_reduce_collapse) -> UOp|None:
  for r in red.src[1:]:
    included = u.toposort(gate=lambda x: r in x.ranges)
    if any(x.op in {Ops.STORE, Ops.REDUCE} for x in included): return None
    replaces: dict[UOp, UOp] = {}
    for u in included:
      for s in u.src:
        if s in included or s in replaces or s.op in GroupOp.Defines|{Ops.CONST}: continue
        replaces[s] = UOp.variable(f'in{len(replaces)}', s.vmin, s.vmax, s.dtype)
    collapse_fxn = u.substitute(replaces).reduce(r, arg=Ops.ADD)
    sink = graph_rewrite(collapse_fxn, pm, name="reduce_collapse")
    if not no_range(sink): return None
    u = sink.substitute({v:k for k,v in replaces.items()})
  return u

def reduce_load_collapse(red:UOp, u:UOp) -> UOp|None: return reduce_collapse(red, u, pm=pm_reduce_load_collapse)

# remove REDUCE without loads (generic arange opt / indexing).
pm_reduce_simplify = pm_reduce_unparented + PatternMatcher([
  (UPat(Ops.REDUCE, src=(UPat.var("u"),), allow_any_len=True, arg=(Ops.ADD, 0), name="red"), reduce_collapse),
])
# remove REDUCE on load, comes from indexing a tensor with another tensor
def no_load(u:UOp) -> bool: return not u.op_in_backward_slice_with_self(Ops.INDEX)
pm_load_collapse = PatternMatcher([
  (UPat(Ops.REDUCE, arg=(Ops.ADD, 0), src=(UPat.var("u"), UPat()), name="red"), reduce_load_collapse),
  # we want to make sure we dont do math on a loaded index since that can cause overflow, this undoes the rule in pm_reduce_load_collapse
  ((UPat.var("x", dtypes.weakint)+UPat.var("y"))<UPat.var("c"), lambda x,y,c: x < c-y if no_load(y) and no_load(c) and not no_load(x) else None),
])
