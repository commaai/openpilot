from typing import Iterator
import functools, itertools
from dataclasses import dataclass, field, replace
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp, graph_rewrite, sint, AxisType, profile_matches, broadcast_axes
from tinygrad.uop.ops import consumer_map_from_toposort, gate_kernel_sink
from tinygrad.uop.symbolic import symbolic, pm_simplify_valid, pm_drop_and_clauses
from tinygrad.helpers import argsort, all_same, cpu_profile, PCONTIG, colored, Context, SPEC

ALWAYS_CONTIGUOUS: set[Ops] = {Ops.CONTIGUOUS, Ops.AFTER, Ops.COPY, Ops.BUFFER, Ops.SLICE,
                      Ops.CONST, Ops.BIND, Ops.MSELECT, Ops.MSTACK, Ops.PARAM,
                      Ops.LOAD, Ops.CALL, Ops.FUNCTION}

def realize(ctx:dict[UOp, None], tr:UOp) -> None: ctx[tr] = None

def realize_srcs(ctx:dict[UOp, None], rb:UOp) -> None:
  for s in rb.src:
    if s.base.op not in ALWAYS_CONTIGUOUS: ctx[s] = None

def realize_store_after_src(ctx:dict[UOp, None], dest:UOp, src:UOp):
  # don't realize COPY/SLICE when they are the direct source of STORE+AFTER — the target buffer is the output
  if src.op in {Ops.COPY, Ops.SLICE} and src in ctx \
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
  # loads reachable from each UOp memoized across matches
  buf_cache: dict[UOp, frozenset[UOp]] = field(default_factory=dict)

  # create ranges
  range_idx: Iterator[int] = field(default_factory=itertools.count)
  def new_range(self, s:sint, axistype:AxisType=AxisType.LOOP) -> UOp:
    if isinstance(s, UOp) and s.op is Ops.RANGE: return s
    # if a range has a 1 src, it's the same as UOp.const(dtypes.weakint, 0)
    return UOp.range(s, next(self.range_idx), axistype) if resolve(s!=1) else UOp.const(dtypes.weakint, 0)

def broadcast_rngs(x:UOp, src:UOp, rngs:tuple[UOp, ...]) -> tuple[UOp, ...]:
  if x.op not in GroupOp.Broadcastable: return rngs
  baxes, nleft = broadcast_axes(src.shape, x.shape), len(x.shape)-len(src.shape)
  return tuple(r.const_like(0) if j in baxes else r for j,r in enumerate(rngs) if j >= nleft)

def create_bufferize_and_index_srcs(ctx:IndexingContext, x:UOp) -> list[UOp]:
  new_srcs = []
  for i, s in enumerate(x.src):
    new_src = s
    src_rngs = broadcast_rngs(x, s, ctx.range_map[x][0]) if x in ctx.range_map else ()
    # shape args of movement ops are at src[1:] and should not be indexed
    if s.op in {Ops.PARAM, Ops.BUFFER, Ops.SLICE, Ops.MSTACK, Ops.MSELECT, Ops.AFTER}:
      if x in ctx.range_map and not (x.op in GroupOp.Movement and i > 0): new_src = new_src.index(*src_rngs)
    elif s in ctx.realize_map:
      realized_ranges = ctx.realize_map[s]
      assert isinstance(realized_ranges, list), "realize map must contain range list"
      closed_ranges = tuple([r for i,r in enumerate(ctx.range_map[s][1]) if i in realized_ranges])
      if s.op is Ops.STORE:
        # add the ends if this is a store
        new_src = s.end(*[r for r in closed_ranges if r.op is Ops.RANGE])
        del ctx.realize_map[s]
      else:
        # the Bufferize before a COPY is not removable unless it's a view. there should be a better way to do this
        removable = (x.op is not Ops.COPY or s.has_buffer_identity()) and s.op not in ALWAYS_CONTIGUOUS
        # LOCAL: None in the device assigns it a number later
        opts = BufferizeOpts(device=s.device, removable=removable) if len(ctx.range_map[s][1]) == len(realized_ranges) else \
               BufferizeOpts(device=s.device, addrspace=AddrSpace.LOCAL, removable=removable)
        new_src = UOp(Ops.STAGE, src=(new_src,)+closed_ranges, arg=opts)
        if x in ctx.range_map: new_src = new_src.index(*[r for i,r in enumerate(src_rngs) if i in realized_ranges])
    new_srcs.append(new_src)
  return new_srcs

def create_bufferize_and_index_based_on_ranges(ctx:IndexingContext, x:UOp):
  if x.op in {Ops.STAGE, Ops.INDEX}: return None
  return x.replace(src=tuple(create_bufferize_and_index_srcs(ctx, x)))

def convert_pad_to_where_to_keep_behavior_local(ctx:IndexingContext, x:UOp):
  if x not in ctx.range_map: return None
  bx = create_bufferize_and_index_based_on_ranges(ctx, x)
  valid: UOp = UOp.const(dtypes.bool, True).uprod([r.get_valid() for r in ctx.range_map[x][0]])
  return valid.where(bx.src[0], UOp.const(x.dtype, 0))

def convert_reduce_to_reduce_with_ranges(ctx:IndexingContext, x:UOp):
  if x.arg[1] == 0: return None
  bx = create_bufferize_and_index_based_on_ranges(ctx, x)
  # input ranges
  new_ranges = list(ctx.range_map[x][0][:x.arg[1]])
  return UOp(Ops.REDUCE, src=(bx.src[0],)+tuple(new_ranges), arg=(x.arg[0], 0))

def convert_stack_to_where(ctx:IndexingContext, x:UOp):
  # only data STACKs: shape tuple STACKs aren't in range_map, the empty shape tuple is void
  if x not in ctx.range_map or x.dtype == dtypes.void: return None
  # use the src list directly, a transient STACK of mid-rangeify srcs violates the spec shape rule
  srcs = create_bufferize_and_index_srcs(ctx, x)
  r0 = ctx.range_map[x][1][0]
  ret = srcs[-1]
  for k in range(len(srcs)-2, -1, -1): ret = r0.eq(k).where(srcs[k], ret)
  return ret

def remove_movement_op_after_rangeify(ctx:IndexingContext, x:UOp):
  if x in ctx.range_map or x.src[0].op is Ops.INDEX: return x.src[0]

pm_apply_rangeify = PatternMatcher([
  # REDUCE(op, axis) -> REDUCE(op) with ranges
  (UPat(Ops.REDUCE, name="x"), convert_reduce_to_reduce_with_ranges),
  # PAD -> WHERE
  (UPat(Ops.PAD, name="x"), convert_pad_to_where_to_keep_behavior_local),
  # STACK -> WHERE select on the leading range
  (UPat(Ops.STACK, name="x"), convert_stack_to_where),
  # finally, apply_rangeify
  (UPat(GroupOp.All, name="x"), create_bufferize_and_index_based_on_ranges),
  # remove movement op
  (UPat(GroupOp.Movement, name="x"), remove_movement_op_after_rangeify),
])

pm_fix_deviceless = PatternMatcher([
  (UPat(Ops.STAGE, name="b"),
    lambda ctx,b: b.replace(arg=replace(b.arg, device=ctx)) if b.arg.addrspace is AddrSpace.GLOBAL and b.arg.device is None else None),
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
    case Ops.SHRINK:  rngs = tuple(a if off == 0 else a+off for a,(off,_) in zip(rngs, arg))
    case Ops.PERMUTE: rngs = tuple(rngs[p] for p in argsort(arg))
    case Ops.FLIP:    rngs = tuple(((s-1)-a) if f else a for a,s,f in zip(rngs, in_shape, arg))
    case Ops.EXPAND:  rngs = rngs[len(arg):]
    case Ops.PAD:
      # NOTE: the .where(r-s, i) is not inside the graph_rewrite so that `convert_pad_to_where_to_keep_behavior_local`
      #       wraps the pad with only the newly added valid
      rngs = tuple(r if (sz == sh and off == 0) else (r-off).valid(graph_rewrite((r >= off) & (r < (sh+off)),
        symbolic+pm_simplify_valid, name="pad")) for r,sh,(off,sz) in zip(rngs, in_shape, arg))
    case Ops.RESHAPE:
      sink = UOp.sink(*rngs).simplify() # NOTE: this applies any commutative flips to the rngs early
      sub_array = {r:UOp.range(r.src[0], i, AxisType.PLACEHOLDER, dtype=r.dtype) for i,r in enumerate(sink.ranges)}
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
    # no ranges on kernels, they are internal
    if x.op in {Ops.CALL, Ops.FUNCTION, Ops.LINEAR}: continue

    # AFTER doesn't have range
    if x.op is Ops.AFTER: continue

    # treat MSTACK/MSELECT like SINK
    if x.op in {Ops.MSTACK, Ops.MSELECT}: continue

    if x.dtype == dtypes.weakint: continue  # TODO: why do I need this?
    ending_ranges[x] = sum([ending_ranges.get(u, []) for u in consumer_map[x]], [])

    # *** the ranges on the output are
    #  1. new if this op is realized
    #  2. from the single consumer if this op only has one consumer
    #  3. potentially new if this op has 2+ consumers

    consumer_rngs = [broadcast_rngs(c, x, rctx.range_map[c][0]) for c in consumer_map[x] if c in rctx.range_map]
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
          _out_rngs.append(graph_rewrite(local_rngs[0].valid(minimum_valid), symbolic, name="minimum_valid"))
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
    # STACK: the leading range selects the src, srcs get the trailing ranges
    if x.op is Ops.STACK: rngs = out_rngs[1:]
    # if the EXPAND is used to inject a range, we don't mark it as ending_ranges. otherwise we do.
    # NOTE: this doesn't actually always end a range, but this is why convs are realized, so for now we need it
    if x.op is Ops.EXPAND and all(isinstance(y, int) or y.op is not Ops.RANGE for y in x.shape):
      ending_ranges[x] += list(UOp.sink(*out_rngs[:len(x.marg)]).ranges.keys())

    # REDUCE creates ranges for the axes it is reducing
    if x.op is Ops.REDUCE and x.arg[1]:
      rngs = tuple(rctx.new_range(s, axistype=AxisType.REDUCE) for s in x.src[0].shape[:x.arg[1]]) + out_rngs

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
  # if a deviceless value must materialize, place it on the sink device
  tsink = graph_rewrite(tsink, pm_fix_deviceless, ctx=tsink.device, name="add device to deviceless")
  return tsink, rctx

def render_ranges(*rngs_list, realized) -> str:
  disp = []
  for i, rs in enumerate(zip(*[[r.render() for r in rngs] for rngs in rngs_list])):
    rng = rs[0] if all_same(rs) else " -> ".join(rs)
    if realized is not None and i in realized: rng = colored(rng, "yellow")
    disp.append("["+rng+"]")
  return ''.join(disp)
