from typing import Iterator
import functools, itertools
from dataclasses import dataclass, field
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp, graph_rewrite, sint, AxisType, profile_matches
from tinygrad.uop.ops import consumer_map_from_toposort, gate_kernel_sink
from tinygrad.uop.symbolic import symbolic, pm_simplify_valid, pm_drop_and_clauses
from tinygrad.helpers import argsort, all_same, cpu_profile, PCONTIG, colored, Context, SPEC

ALWAYS_CONTIGUOUS: set[Ops] = {Ops.CONTIGUOUS, Ops.AFTER, Ops.COPY, Ops.BUFFER, Ops.BUFFER_VIEW,
                     Ops.CONST, Ops.BIND, Ops.DEVICE, Ops.MSELECT, Ops.MSTACK, Ops.PARAM,
                     Ops.DEFINE_LOCAL, Ops.DEFINE_REG, Ops.LOAD, Ops.CALL, Ops.FUNCTION}

def realize(ctx:dict[UOp, None], tr:UOp) -> None: ctx[tr] = None

def realize_srcs(ctx:dict[UOp, None], rb:UOp) -> None:
  for s in rb.src:
    if s.base.op not in ALWAYS_CONTIGUOUS: ctx[s] = None

def realize_store_after_src(ctx:dict[UOp, None], dest:UOp, src:UOp):
  # don't realize COPY/BUFFER_VIEW when they are the direct source of STORE+AFTER — the target buffer is the output
  if src.op in {Ops.COPY, Ops.BUFFER_VIEW} and src in ctx \
     and not dest.op_in_backward_slice_with_self(Ops.SHRINK, Ops.PERMUTE, Ops.FLIP, Ops.PAD):
    del ctx[src]
  # you don't usually have to do this for assign unless there's a WAR hazard like TestAssign.test_assign_double_diamond_reduce
  if dest.base in src.backward_slice_with_self: ctx[src] = None

pm_generate_realize_map = PatternMatcher([
  # always realize
  (UPat({Ops.COPY, Ops.CONTIGUOUS, Ops.STORE}, name="tr"), realize),
  # realize srcs of these
  (UPat((Ops.COPY, Ops.MSELECT, Ops.MSTACK), name="rb"), realize_srcs),
  # sometimes we need to realize the src of STORE if there's a self-access
  (UPat(Ops.STORE, src=(UPat.var("dest"), UPat.var("src"))), realize_store_after_src),
])

@dataclass(frozen=True)
class BufferizeOpts:
  # on AddrSpace.LOCAL, device is the id
  device: str|tuple[str, ...]|int|None
  addrspace: AddrSpace = AddrSpace.GLOBAL
  removable: bool = True

@dataclass
class IndexingContext:
  realize_map: dict[UOp, None|list[int]] = field(default_factory=dict)
  range_map: dict[UOp, tuple[tuple[UOp, ...], tuple[UOp, ...]]] = field(default_factory=dict)

  # create ranges
  range_idx: Iterator[int] = field(default_factory=itertools.count)
  def new_range(self, s:sint, axistype:AxisType=AxisType.LOOP) -> UOp:
    if isinstance(s, UOp) and s.op is Ops.RANGE: return s
    # if a range has a 1 src, it's the same as UOp.const(dtypes.weakint, 0)
    return UOp.range(s, next(self.range_idx), axistype) if resolve(s!=1) else UOp.const(dtypes.weakint, 0)

def create_bufferize_and_index_based_on_ranges(ctx:IndexingContext, x:UOp):
  if x.op in {Ops.STAGE, Ops.INDEX}: return None
  new_srcs = []
  for s in x.src:
    new_src = s
    if s.op in {Ops.PARAM, Ops.BUFFER, Ops.BUFFER_VIEW, Ops.MSTACK, Ops.MSELECT, Ops.AFTER}:
      if x in ctx.range_map: new_src = new_src.index(*ctx.range_map[x][0])
    elif s in ctx.realize_map:
      realized_ranges = ctx.realize_map[s]
      assert isinstance(realized_ranges, list), "realize map must contain range list"
      closed_ranges = tuple([r for i,r in enumerate(ctx.range_map[s][1]) if i in realized_ranges])
      if s.op is Ops.STORE:
        # add the ends if this is a store
        new_src = s.end(*[r for r in closed_ranges if r.op is Ops.RANGE])
        del ctx.realize_map[s]
      else:
        # the Bufferize before a COPY is not removable. there should be a better way to do this
        removable = x.op is not Ops.COPY and s.op not in ALWAYS_CONTIGUOUS
        # None in the device assigns it a number later
        opts = BufferizeOpts(device=s.device, removable=removable) if len(ctx.range_map[s][1]) == len(realized_ranges) else \
               BufferizeOpts(device=s.device, addrspace=AddrSpace.LOCAL, removable=removable)
        new_src = UOp(Ops.STAGE, s.dtype, src=(new_src,)+closed_ranges, arg=opts)
        if x in ctx.range_map: new_src = new_src.index(*[r for i,r in enumerate(ctx.range_map[x][0]) if i in realized_ranges])
    new_srcs.append(new_src)
  # NOTE: do we need this?
  return x.replace(src=tns) if x.src != (tns:=tuple(new_srcs)) else None

def convert_pad_to_where_to_keep_behavior_local(ctx:IndexingContext, x:UOp):
  if x not in ctx.range_map: return None
  valid: UOp = UOp.const(dtypes.bool, True).uprod([r.get_valid() for r in ctx.range_map[x][0]])
  ret = valid.where(x.src[0], UOp.const(x.dtype, 0))
  ctx.range_map[ret] = ctx.range_map[x]
  return ret

def convert_reduce_to_reduce_with_ranges(ctx:IndexingContext, x:UOp):
  if len(x.arg[1]) == 0: return None
  # input ranges
  new_ranges = [r for i,r in enumerate(ctx.range_map[x][0]) if i in x.arg[1]]
  ret = UOp(Ops.REDUCE, x.dtype, src=(x.src[0],)+tuple(new_ranges), arg=(x.arg[0], ()))
  ctx.range_map[ret] = ctx.range_map[x]
  return ret

def remove_movement_op_after_rangeify(ctx:IndexingContext, x:UOp):
  if x in ctx.range_map or x.src[0].op is Ops.INDEX: return x.src[0]

pm_apply_rangeify = PatternMatcher([
  # REDUCE(op, axis) -> REDUCE(op) with ranges
  (UPat(Ops.REDUCE, name="x"), convert_reduce_to_reduce_with_ranges),
  # PAD -> WHERE
  (UPat(Ops.PAD, name="x"), convert_pad_to_where_to_keep_behavior_local),
  # finally, apply_rangeify
  (UPat(GroupOp.All, name="x"), create_bufferize_and_index_based_on_ranges),
  # remove movement op
  (UPat(GroupOp.Movement, name="x"), remove_movement_op_after_rangeify),
])

@functools.cache
def _apply_reshape(in_shape:tuple[sint,...], out_shape:tuple[sint, ...], urngs:UOp) -> UOp:
  acc:sint = 1
  axes_in:list[UOp] = []
  for s,src in list(zip(out_shape, urngs.src))[::-1]:
    axes_in.append(acc*src)
    acc *= s
  combined_axes = UOp.const(dtypes.weakint, 0).usum(axes_in)
  axes_out:list[UOp] = []
  for s in in_shape[::-1]:
    axes_out.append(combined_axes % s)
    combined_axes //= s
  # this simplify is doing a lot of heavy lifting. this is the replacement for the reshape view merging code
  return graph_rewrite(UOp.sink(*axes_out[::-1]), symbolic+pm_simplify_valid+pm_drop_and_clauses, name="reshape")

# this is the definition of the movement ops
@functools.cache
def apply_movement_op(op:Ops, in_shape:tuple[sint,...], arg:tuple, rngs:tuple[UOp, ...]) -> tuple[UOp, ...]:
  match op:
    case Ops.SHRINK:  rngs = tuple(a if ss == 0 else a+ss for a,(ss,_) in zip(rngs, arg))
    case Ops.PERMUTE: rngs = tuple(rngs[p] for p in argsort(arg))
    case Ops.FLIP:    rngs = tuple(((s-1)-a) if f else a for a,s,f in zip(rngs, in_shape, arg))
    case Ops.EXPAND:  rngs = tuple(a if in_sh == out_sh else a.const_like(0) for a,in_sh,out_sh in zip(rngs, in_shape, arg))
    case Ops.PAD:
      # NOTE: the .where(r-s, i) is not inside the graph_rewrite so that `convert_pad_to_where_to_keep_behavior_local`
      #       wraps the pad with only the newly added valid
      rngs = tuple(r if (s == 0 and e == 0) else graph_rewrite((r >= s) & (r < (sh+s)),
        symbolic+pm_simplify_valid, name="pad").where(r-s, UOp.invalid()) for r,sh,(s,e) in zip(rngs, in_shape, arg))
    case Ops.RESHAPE:
      sink = UOp.sink(*rngs).simplify() # NOTE: this applies any commutative flips to the rngs early
      sub_array = {r:UOp.range(r.src[0], i, AxisType.PLACEHOLDER) for i,r in enumerate(sink.ranges)}
      rngs = _apply_reshape(in_shape, arg, sink.substitute(sub_array)).substitute({v:k for k,v in sub_array.items()}).src
    case _: raise RuntimeError(f"{op} is not a MovementOp")
  return rngs

@profile_matches
def run_rangeify(tsink:UOp, debug:bool=False) -> tuple[UOp, IndexingContext]:
  if debug: print("**************************")
  rctx = IndexingContext()

  # get ops to realize
  graph_rewrite(tsink, pm_generate_realize_map, ctx=rctx.realize_map, name="get realize")

  # get the consumer map
  with cpu_profile("consumer map in rangeify", "TINY"):
    consumer_map = consumer_map_from_toposort(tsink_toposort:=tsink.toposort(gate_kernel_sink))

  # explicit rangeify
  ending_ranges: dict[UOp, list[UOp]] = {}
  for x in reversed(tsink_toposort):
    if x.op in {Ops.DEVICE, Ops.UNIQUE}: continue

    # no ranges on kernels, they are internal
    if x.op in {Ops.CALL, Ops.FUNCTION, Ops.LINEAR}: continue

    # AFTER doesn't have range
    if x.op is Ops.AFTER: continue

    # treat MSTACK/MSELECT like SINK
    if x.op in {Ops.MSTACK, Ops.MSELECT}: continue

    if x.dtype.scalar() == dtypes.weakint: continue  # TODO: why do I need this?
    ending_ranges[x] = sum([ending_ranges.get(u, []) for u in consumer_map[x]], [])

    # *** the ranges on the output are
    #  1. new if this op is realized
    #  2. from the single consumer if this op only has one consumer
    #  3. potentially new if this op has 2+ consumers

    consumer_rngs = [rctx.range_map[c][0] for c in consumer_map[x] if c in rctx.range_map]
    if x in rctx.realize_map:
      # if this is in the realize_map, we create new ranges (at the output)
      out_rngs = tuple(rctx.new_range(s) for s in x.shape)
      # all ranges are ended now
      ending_ranges[x] = []
      # mark all ranges as ended
      assert rctx.realize_map[x] is None
      rctx.realize_map[x] = list(range(len(x.shape)))
    elif len(consumer_rngs) == 0:
      # if no consumers have ranges and this isn't realized, this doesn't have ranges either.
      continue
    elif len(consumer_rngs) == 1:
      # if this has one consumer, it inherits the ranges from it
      out_rngs = consumer_rngs[0]
    elif len(consumer_rngs) > 1:
      # if this has two consumers, we have to merge the ranges and might create new ones
      all_rngs: list[tuple[UOp, ...]] = list(zip(*consumer_rngs))
      rngs_valids = []
      for valid_rngs in all_rngs:
        local_rngs, valids = zip(*[(r.get_idx(), r.get_valid()) for r in valid_rngs])
        rngs_valids.append((local_rngs, valids))

      # TODO: in RANGEIFY > 1 all_all_same isn't required
      all_all_same = all(all_same(local_rngs) for local_rngs,_ in rngs_valids)
      _out_rngs = []
      _realize_axis = []
      for i,(local_rngs,valids) in enumerate(rngs_valids):
        # we compare the ranges without their valids
        if all_all_same or (PCONTIG and all_same(local_rngs)):
          # the new valid is the OR of all the children valids
          minimum_valid = UOp.const(dtypes.bool, False).usum(valids)
          _out_rngs.append(graph_rewrite(minimum_valid.where(local_rngs[0], UOp.invalid()), symbolic, name="minimum_valid"))
        else:
          _out_rngs.append(rctx.new_range(x.shape[i]))
          _realize_axis.append(i)
      out_rngs = tuple(_out_rngs)

      # we have to (partially) realize here if there's new ranges
      if len(_realize_axis): rctx.realize_map[x] = _realize_axis

    # if this element is a reduce and there's ended ranges, we might have to end some other ranges
    if len(ending_ranges[x]) and x.op in GroupOp.Elementwise.union({Ops.REDUCE}):
      _realize_axis = rctx.realize_map.get(x) or []
      for i,r in enumerate(out_rngs):
        if i in _realize_axis: continue
        if not (PCONTIG > 1) or any(any(rr.arg > e.arg for e in ending_ranges[x]) for rr in r.ranges):
          _realize_axis.append(i)
      ending_ranges[x] = []
      if len(_realize_axis):
        rctx.realize_map[x] = _realize_axis
        out_rngs = tuple([(rctx.new_range(x.shape[i]) if i in _realize_axis else r) for i,r in enumerate(out_rngs)])

    # TODO: some ops don't have shape, enable this after the `.st` property is removed
    #assert len(out_rngs) == len(x.shape), \
    #  f"shape len mismatch {len(out_rngs)} != {len(x.shape)} on {x.op} with {len(consumer_map[x])} consumers and realize {x in realize_map}"

    # *** the ranges on the inputs are
    #  1. swizzled for MovementOps
    #  2. newly created for REDUCE (tensor graph form with axis)
    #  3. passed through for everything else

    rngs = out_rngs  # rngs is the input ranges  # pylint: disable=possibly-used-before-assignment

    # apply movement ops
    if x.op in GroupOp.Movement: rngs = apply_movement_op(x.op, x.src[0].shape, x.marg, rngs)
    # if the EXPAND is used to inject a range, we don't mark it as ending_ranges. otherwise we do.
    # NOTE: this doesn't actually always end a range, but this is why convs are realized, so for now we need it
    if x.op is Ops.EXPAND and all(isinstance(y, int) or y.op is not Ops.RANGE for y in x.shape):
      ending_ranges[x] += list(UOp.sink(*[ro for ri, ro in zip(rngs, out_rngs) if ri is not ro]).ranges.keys())

    # REDUCE creates ranges for the axes it is reducing
    if x.op is Ops.REDUCE and len(x.arg[1]):
      rngs = tuple(rctx.new_range(s, axistype=AxisType.REDUCE) if i in x.arg[1] else r for i,(r,s) in enumerate(zip(rngs, x.src[0].shape)))

    if debug:
      realized_ranges = rctx.realize_map.get(x, None)
      if x.op is Ops.RESHAPE or len(rngs) != len(out_rngs):
        disp = render_ranges(rngs, realized=realized_ranges) + " -> " + render_ranges(out_rngs, realized=realized_ranges)
      else:
        disp = render_ranges(rngs, out_rngs, realized=realized_ranges)
      print("***" if x in rctx.realize_map else "   ",
            f"{len(consumer_map[x]):2d} {str(x.op):20s} {str(x._shape):35s} {len(ending_ranges[x]):2d}", disp)

    # assign to the range map. rngs are the input ranges, out_rngs are the output ranges, from the x op.
    rctx.range_map[x] = (rngs, out_rngs)

  # NOTE: SPEC=3 is broken here with shape
  with Context(SPEC=min(SPEC.value, 2)):
    tsink = graph_rewrite(tsink, pm_apply_rangeify, ctx=rctx, bottom_up=True, name="apply rangeify")
  return tsink, rctx

def render_ranges(*rngs_list, realized) -> str:
  disp = []
  for i, rs in enumerate(zip(*[[r.render() for r in rngs] for rngs in rngs_list])):
    rng = rs[0] if all_same(rs) else " -> ".join(rs)
    if realized is not None and i in realized: rng = colored(rng, "yellow")
    disp.append("["+rng+"]")
  return ''.join(disp)
