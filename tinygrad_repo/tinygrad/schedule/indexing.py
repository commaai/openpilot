from typing import Iterator
import functools, operator, itertools
from dataclasses import dataclass, field
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp, graph_rewrite, sint, AxisType, profile_matches
from tinygrad.uop.ops import consumer_map_from_toposort
from tinygrad.uop.symbolic import symbolic, pm_simplify_valid, pm_drop_and_clauses
from tinygrad.helpers import argsort, all_same, cpu_profile, PCONTIG, colored

ALWAYS_CONTIGUOUS: set[Ops] = {Ops.CONTIGUOUS, Ops.ASSIGN, Ops.COPY, Ops.BUFFER, Ops.BUFFER_VIEW,
                     Ops.CONST, Ops.BIND, Ops.DEVICE, Ops.MSELECT, Ops.MSTACK, Ops.PARAM,
                     Ops.DEFINE_LOCAL, Ops.DEFINE_REG, Ops.LOAD, Ops.KERNEL, Ops.ENCDEC}

def realize(ctx:dict[UOp, None], tr:UOp) -> None: ctx[tr] = None

def realize_srcs(ctx:dict[UOp, None], rb:UOp) -> None:
  for s in rb.src:
    if s.base.op not in ALWAYS_CONTIGUOUS: ctx[s] = None

def realize_assign(ctx:dict[UOp, None], a:UOp) -> None:
  if a.src[1].op not in ALWAYS_CONTIGUOUS: ctx[a.src[1]] = None
  # if it's a kernel, we don't realize it
  if a.src[1].op is not Ops.KERNEL: ctx[a] = None

pm_generate_realize_map = PatternMatcher([
  # always realize SINK src
  (UPat(Ops.SINK, name="s"), lambda ctx,s: ctx.update((x.base, None) for x in s.src if x.base.op not in ALWAYS_CONTIGUOUS)),
  # always realize COPY/BUFFER_VIEW/CONTIGUOUS/STORE/ENCDEC
  (UPat({Ops.COPY, Ops.BUFFER_VIEW, Ops.CONTIGUOUS, Ops.STORE, Ops.ENCDEC}, name="tr"), realize),
  # always realize REDUCE on outer ranges
  (UPat(Ops.REDUCE, name="r"), lambda ctx,r: realize(ctx, r) if any(tr.arg[-1] == AxisType.OUTER for tr in r.src[1:]) else None),
  # realize srcs of COPY, MSELECT, MSTACK, ENCDEC
  (UPat((Ops.COPY, Ops.MSELECT, Ops.MSTACK, Ops.ENCDEC), name="rb"), realize_srcs),
  # realize ASSIGN and input to assign (might be optimized out)
  (UPat(Ops.ASSIGN, name="a"), realize_assign),
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
    # if a range has a 1 src, it's the same as UOp.const(dtypes.index, 0)
    return UOp.range(s, next(self.range_idx), axistype) if resolve(s!=1) else UOp.const(dtypes.index, 0)

def create_bufferize_and_index_based_on_ranges(ctx:IndexingContext, x:UOp):
  if x.op in {Ops.BUFFERIZE, Ops.INDEX, Ops.AFTER}: return None
  new_srcs = []
  for s in x.src:
    new_src = s
    if s.op in {Ops.BUFFER, Ops.BUFFER_VIEW, Ops.MSTACK, Ops.MSELECT, Ops.AFTER}:
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
        new_src = UOp(Ops.BUFFERIZE, s.dtype, src=(new_src,)+closed_ranges, arg=opts, tag=s.tag if opts.addrspace == AddrSpace.GLOBAL else None)
        if x in ctx.range_map: new_src = new_src.index(*[r for i,r in enumerate(ctx.range_map[x][0]) if i in realized_ranges])
    new_srcs.append(new_src)
  # NOTE: do we need this?
  return x.replace(src=tns) if x.src != (tns:=tuple(new_srcs)) else None

def convert_pad_to_where_to_keep_behavior_local(ctx:IndexingContext, x:UOp):
  if x not in ctx.range_map: return None
  valid: UOp = functools.reduce(operator.and_, [r.get_valid() for r in ctx.range_map[x][0]], UOp.const(dtypes.bool, True))
  ret = valid.where(x.src[0], UOp.const(x.dtype, 0))
  ctx.range_map[ret] = ctx.range_map[x]
  return ret

def convert_reduce_axis_to_reduce_with_ranges(ctx:IndexingContext, x:UOp):
  # input ranges
  new_ranges = [r for i,r in enumerate(ctx.range_map[x][0]) if i in x.arg[1]]
  ret = UOp(Ops.REDUCE, x.dtype, src=(x.src[0],)+tuple(new_ranges), arg=x.arg[0], tag=x.tag)
  ctx.range_map[ret] = ctx.range_map[x]
  return ret

def remove_movement_op_after_rangeify(ctx:IndexingContext, x:UOp):
  if x in ctx.range_map or x.src[0].op is Ops.INDEX: return x.src[0]

def add_third_op_to_assign_to_track_shape(ctx:IndexingContext, assign:UOp):
  if assign.src[1].op is Ops.KERNEL: return None
  to_mop = graph_rewrite(assign.src[0], PatternMatcher([(UPat(GroupOp.Movement, name="x"), lambda x: x.replace(tag=()))]))
  ret = assign.replace(src=assign.src+(to_mop,))
  ctx.range_map[ret] = ctx.range_map[assign]
  return ret

pm_apply_rangeify = PatternMatcher([
  # REDUCE_AXIS -> REDUCE
  (UPat(Ops.REDUCE_AXIS, name="x"), convert_reduce_axis_to_reduce_with_ranges),
  # PAD -> WHERE
  (UPat(Ops.PAD, name="x"), convert_pad_to_where_to_keep_behavior_local),
  # add third op to assign
  (UPat(Ops.ASSIGN, src=(UPat(), UPat()), name="assign"), add_third_op_to_assign_to_track_shape),
  # finally, apply_rangeify
  (UPat(GroupOp.All, name="x"), create_bufferize_and_index_based_on_ranges),
  # remove movement op
  (UPat(GroupOp.Movement, name="x"), remove_movement_op_after_rangeify),
  # const/define_var shouldn't have src
  (UPat((Ops.CONST, Ops.DEFINE_VAR), name="c"), lambda ctx,c: c.replace(src=()) if c in ctx.range_map else None),
])

@functools.cache
def _apply_reshape(in_shape:tuple[sint,...], out_shape:tuple[sint, ...], urngs:UOp) -> UOp:
  acc:sint = 1
  axes_in:list[UOp] = []
  for s,src in list(zip(out_shape, urngs.src))[::-1]:
    axes_in.append(acc*src)
    acc *= s
  combined_axes = sum(axes_in, start=UOp.const(dtypes.index, 0))
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
      # TODO: why is multiple graph_rewrites faster than one here?
      # TODO: the .where(r-s, i) is not inside the graph_rewrite so that `convert_pad_to_where_to_keep_behavior_local`
      #       wraps the pad with only the newly added valid
      rngs = tuple(r if (s == 0 and e == 0) else graph_rewrite(((r >= s) & (r < (sh+s))),
        symbolic+pm_simplify_valid, name="pad").where(r-s, UOp.invalid()) for r,sh,(s,e) in zip(rngs, in_shape, arg))
    case Ops.RESHAPE:
      sink = UOp.sink(*rngs)
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
    consumer_map = consumer_map_from_toposort(tsink_toposort:=tsink.toposort())

  # explicit rangeify
  ending_ranges: dict[UOp, list[UOp]] = {}
  for x in reversed(tsink_toposort):
    if x.op in {Ops.DEVICE, Ops.UNIQUE}: continue

    # no ranges on kernels, they are internal
    if x.op is Ops.KERNEL: continue

    if x.dtype.scalar() == dtypes.index: continue  # TODO: why do I need this?
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
    elif x.op in {Ops.MSTACK, Ops.MSELECT}:
      # treat MSTACK/MSELECT like SINK
      continue
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
          minimum_valid = functools.reduce(operator.or_, valids, UOp.const(dtypes.bool, False))
          _out_rngs.append(graph_rewrite(minimum_valid.where(local_rngs[0], UOp.invalid()), symbolic, name="minimum_valid"))
        else:
          _out_rngs.append(rctx.new_range(x.shape[i]))
          _realize_axis.append(i)
      out_rngs = tuple(_out_rngs)

      # we have to (partially) realize here if there's new ranges
      if len(_realize_axis): rctx.realize_map[x] = _realize_axis

    # if this element is a reduce and there's ended ranges, we might have to end some other ranges
    if len(ending_ranges[x]) and x.op in GroupOp.Elementwise.union({Ops.REDUCE_AXIS}):
      _realize_axis = rctx.realize_map.get(x, []) or []
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
    #  2. newly created for REDUCE_AXIS
    #  3. passed through for everything else

    rngs = out_rngs  # rngs is the input ranges  # pylint: disable=possibly-used-before-assignment

    # apply movement ops
    if x.op in GroupOp.Movement: rngs = apply_movement_op(x.op, x.src[0].shape, x.marg, rngs)
    # if the EXPAND is used to inject a range, we don't mark it as ending_ranges. otherwise we do.
    # NOTE: this doesn't actually always end a range, but this is why convs are realized, so for now we need it
    if x.op is Ops.EXPAND and all(isinstance(y, int) or y.op is not Ops.RANGE for y in x.shape):
      ending_ranges[x] += list(UOp.sink(*[ro for ri, ro in zip(rngs, out_rngs) if ri is not ro]).ranges.keys())

    # REDUCE_AXIS creates ranges for the axes it is reducing
    if x.op is Ops.REDUCE_AXIS:
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

  tsink = graph_rewrite(tsink, pm_apply_rangeify, ctx=rctx, bottom_up=True, name="apply rangeify")
  return tsink, rctx

def render_ranges(*rngs_list, realized) -> str:
  disp = []
  for i, rs in enumerate(zip(*[[r.render() for r in rngs] for rngs in rngs_list])):
    rng = rs[0] if all_same(rs) else " -> ".join(rs)
    if realized is not None and i in realized: rng = colored(rng, "yellow")
    disp.append("["+rng+"]")
  return ''.join(disp)
