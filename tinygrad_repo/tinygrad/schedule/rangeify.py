from typing import Any, cast
import functools, operator
from dataclasses import dataclass, field
from tinygrad.dtype import dtypes, PtrDType, ImageDType, AddrSpace
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp, RewriteNotReady, _substitute, ssimplify, graph_rewrite_map
from tinygrad.uop.symbolic import sym, symbolic_simple
from tinygrad.helpers import argsort, prod, all_same, pluralize, getenv, RANGEIFY, Context, flatten, dedup
from tinygrad.schedule.multi import multi_pm

from tinygrad.schedule.kernelize import Kernel
from tinygrad.uop.ops import track_rewrites, graph_rewrite, identity_element, sint, AxisType

# *****************
# 0. do some cleanup rewrites, mostly copied from the old stuff

double_reshape = PatternMatcher([
  # RESHAPE on RESHAPE is the second reshape
  (UPat(Ops.RESHAPE, src=(UPat(Ops.RESHAPE),), name="x"), lambda x: x.replace(src=(x.src[0].src[0],))),
])

earliest_rewrites = double_reshape+PatternMatcher([
  # non shape changing RESHAPE is NOOP
  #(UPat(Ops.RESHAPE, name="x"), lambda x: x.src[0] if x.src[0].shape == x.arg else None),
  # DETACH and CONTIGUOUS_BACKWARD are NOOPs here, so is FUSE
  #(UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD, Ops.FUSE), name="x"), lambda x: x.src[0].f(Ops.NOOP, tag=x.tag)),

  # just removing it works...
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD, Ops.FUSE), name="x"), lambda x: x.src[0]),

  # preserve tags?
  # reduce of size 0 is the identity element
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)),
   lambda reduce,x: reduce.const_like(identity_element(reduce.arg[0], reduce.dtype)) if x.size == 0 and reduce.size != 0 else None),

  # COPY and source size need to match
  # TODO: expand after copy creates issues with tagging
  (UPat(Ops.COPY, src=(UPat(GroupOp.Movement, name="r"), UPat(name="d")), name="c"),
   lambda c,r,d: c.replace(src=(r.contiguous(), d)) if r.size != r.base.size else None),

  # assign only to buffer
  (UPat(Ops.ASSIGN, src=(UPat(GroupOp.All-{Ops.BUFFER}, name="target"), UPat(name="x")), name="assign"),
   lambda x,target,assign: x.f(Ops.NOOP, tag=assign.tag) if target.base.op is not Ops.BUFFER else None),

  # handle disk
  # TODO: this doesn't need to use st.views
  (UPat.var("x").f((Ops.BITCAST, Ops.CONTIGUOUS), name="t"),
   lambda x,t: UOp(Ops.BUFFER_VIEW, t.dtype, (x.base,), (t.size, x.st.views[0].offset), tag=t.tag).reshape(t.shape) if isinstance(x.device, str) \
       and x.device.startswith("DISK") else None),

  # contiguous/buffer/copy/assign is already contiguous
  #(UPat(Ops.CONTIGUOUS, name="root", src=(UPat((Ops.CONTIGUOUS, Ops.BUFFER, Ops.COPY, Ops.ASSIGN)),)), lambda root: root.src[0]),
])

# *****************
# 1. add realize where we have to

ALWAYS_CONTIGUOUS: set[Ops] = {Ops.CONTIGUOUS, Ops.ASSIGN, Ops.COPY, Ops.BUFFER, Ops.BUFFER_VIEW,
                     Ops.CONST, Ops.BIND, Ops.DEVICE, Ops.MSELECT, Ops.MSTACK, Ops.DEFINE_GLOBAL,
                     Ops.DEFINE_LOCAL, Ops.DEFINE_REG, Ops.LOAD}

def realize(ctx:dict[UOp, None], tr:UOp) -> None: ctx[tr] = None

def realize_parents(ctx:dict[UOp, None], rb:UOp) -> None:
  for s in rb.src:
    if s.base.op not in ALWAYS_CONTIGUOUS: ctx[s] = None

def realize_assign(ctx:dict[UOp, None], a:UOp) -> None:
  if a.src[1].op not in ALWAYS_CONTIGUOUS: ctx[a.src[1]] = None

do_realize = PatternMatcher([
  # always realize SINK parents
  (UPat(Ops.SINK, name="s"), lambda ctx,s: ctx.update((x.base, None) for x in s.src if x.base.op not in ALWAYS_CONTIGUOUS)),
  # always realize ASSIGN/COPY/BUFFER_VIEW/CONTIGUOUS
  (UPat({Ops.ASSIGN, Ops.COPY, Ops.BUFFER_VIEW, Ops.CONTIGUOUS}, name="tr"), realize),
  # realize parents of COPY, MSELECT, MSTACK
  (UPat((Ops.COPY, Ops.MSELECT, Ops.MSTACK), name="rb"), realize_parents),
  # realize input to assign (might be optimized out)
  (UPat(Ops.ASSIGN, name="a"), realize_assign),
])


class WrappedContig:
  def __init__(self, x): self.x = x
  def __repr__(self): return f"C({self.x})"
add_contiguous = PatternMatcher([
  (UPat(GroupOp.All, name="x"),
   lambda ctx,x: x.replace(tag=WrappedContig(x.tag)).realize() if x in ctx and not isinstance(x.tag, WrappedContig) else None),
])
remove_contig_tags = PatternMatcher([(UPat(GroupOp.All, name="x"), lambda x: x.replace(tag=x.tag.x) if isinstance(x.tag, WrappedContig) else None)])

# *****************
# 2. mark all children

@dataclass
class ChildrenContext: children: dict[UOp, list[UOp]]|None = None
def extract_children(ctx:ChildrenContext, x:UOp):
  if ctx.children is not None: return
  children_map = x.get_children_map()
  ctx.children = {}
  for k,v in children_map.items():
    non_sink_children = [u for u in v if u.op is not Ops.SINK]
    if len(non_sink_children) <= 1: continue
    # NOTE: this gate shouldn't be here
    if any(x.op is Ops.REDUCE_AXIS for x in k.toposort()) and any(x.op in {Ops.BUFFER, Ops.CONTIGUOUS} for x in k.toposort()):
      ctx.children[k] = non_sink_children

def mark_children(ctx:ChildrenContext, x:UOp):
  assert ctx.children is not None
  new_srcs = [(UOp(Ops.CHILD, s.dtype, src=(UOp(Ops.CHILDREN, s.dtype, (s,), arg=len(ctx.children[s])),),
                   arg=(ctx.children[s].index(x), len(ctx.children[s]))) if s in ctx.children else s) for s in x.src]
  return x.replace(src=tuple(new_srcs))

pm_children = PatternMatcher([
  (UPat(Ops.SINK, name="x"), extract_children),
  (UPat(GroupOp.All-{Ops.CHILD, Ops.CHILDREN, Ops.SINK}, name="x"), mark_children),
])

# *****************
# 3a. rangeify (movement)

@dataclass
class RangeifyContext:
  # block on parent until all children have been seen
  seen_children: dict[UOp, dict[int, UOp]] = field(default_factory=dict)
  seen_child: dict[UOp, Any] = field(default_factory=dict)
  progress: int = 0

  # create ranges
  range_idx: int = 0
  def new_range(self, s:sint, axistype:AxisType=AxisType.LOOP):
    ret = UOp.range(s, self.range_idx, axistype)
    self.range_idx += 1
    return ret

def map_reshape(idx:UOp, r:UOp):
  acc = 1
  to_sum = []
  for s,src in list(zip(idx.shape, idx.src[1:]))[::-1]:
    to_sum.append(acc*src)
    acc *= s
  mish = sum(to_sum, start=UOp.const(dtypes.index, 0))
  ret:list[UOp] = []
  for s in r.src[0].shape[::-1]:
    ret.append(mish % s) # NOTE: simplify will turn this to CONST
    mish //= s
  tret = ret[0].sink(*ret[1:]).simplify().src[::-1] if len(ret) else ()
  return r.src[0].index(*tret, dtype=idx.dtype, arg=idx.arg)

def map_pad(idx:UOp, r:UOp):
  ret = list(idx.src[1:])
  bigwhere = UOp.const(dtypes.bool, True)
  for i,(sh,(s,e)) in enumerate(zip(r.shape, r.arg)):
    if s == 0 and e == 0: continue
    where = UOp.const(dtypes.bool, True)
    if resolve(e > 0): where = where & (ret[i] < (sh-e))
    if resolve(s > 0): where = where & (ret[i] >= s)
    bigwhere = bigwhere & where
    with Context(TRACK_MATCH_STATS=0):
      ret[i] = graph_rewrite(where.where(ret[i]-s, UOp.invalid()), sym)
  # PAD is with 0
  return bigwhere.simplify().where(r.src[0].index(*ret, dtype=idx.dtype, arg=idx.arg), UOp.const(r.dtype, 0))

def map_expand(r:UOp, idx:UOp):
  new_rngs = []
  ending_ranges = []
  non_ending_ranges = []
  for a,x,y in zip(idx.src[1:], r.src[0].shape, r.shape):
    axis_to_range = [u for u in a.toposort() if u.op is Ops.RANGE]
    if resolve(x==y, False):
      non_ending_ranges.extend(axis_to_range)
      new_rngs.append(a)
    else:
      ending_ranges.extend(axis_to_range)
      new_rngs.append(a.const_like(0))
  ending_ranges = [x.arg for x in ending_ranges if x not in non_ending_ranges]
  if idx.arg is not None: ending_ranges.append(idx.arg)
  return r.src[0].index(*new_rngs, arg=min(ending_ranges) if ending_ranges else None)

pm_mops = PatternMatcher([
  # this is like the definitions of these
  (UPat(Ops.SHRINK, name="r").f(Ops.INDEX, allow_any_len=True, name="idx"),
   lambda r,idx: r.src[0].index(*[a+ss if resolve(ss != 0) else a for a,(ss,_) in zip(idx.src[1:], r.arg)], dtype=idx.dtype, arg=idx.arg)),
  (UPat(Ops.PERMUTE, name="r").f(Ops.INDEX, allow_any_len=True, name="idx"),
   lambda r,idx: r.src[0].index(*[idx.src[1+p] for p in argsort(idx.src[0].arg)], dtype=idx.dtype, arg=idx.arg)),
  (UPat(Ops.FLIP, name="r").f(Ops.INDEX, allow_any_len=True, name="idx"),
   lambda r,idx: r.src[0].index(*[((s-1)-a) if f else a for a,s,f in zip(idx.src[1:], r.shape, r.arg)], dtype=idx.dtype, arg=idx.arg)),
  # expand needs to end ranges
  (UPat(Ops.EXPAND, name="r").f(Ops.INDEX, allow_any_len=True, name="idx"), map_expand),
  # reshape does a lot of symbolic stuff
  (UPat(Ops.RESHAPE, name="r").f(Ops.INDEX, allow_any_len=True, name="idx"), map_reshape),
  # pad adds min and max
  (UPat(Ops.PAD, name="r").f(Ops.INDEX, allow_any_len=True, name="idx"), map_pad),
])

# *****************
# 3b. rangeify (ops)

# bufferization can happen in three ways
#   1. there's an explicit REALIZE in the graph
#   2. the ranges from the children don't match and we have to create a buffer (only on children)
#   3. might_end_axis triggers because we should be closing a loop to save compute

@dataclass(frozen=True)
class BufferizeOpts:
  # on AddrSpace.LOCAL, device is the id
  device: str|tuple[str, ...]|int|None
  addrspace: AddrSpace = AddrSpace.GLOBAL

def map_partial_realize(ctx:RangeifyContext, x:UOp, idx:UOp):
  if x.arg is None: return None  # map_contiguous can handle this
  # NOTE: all partial contiguous can safely be replaced by full contiguous. we should be able to match old functionality like this
  if not (RANGEIFY > 1): return idx.replace(src=(x.replace(arg=None),)+idx.src[1:])
  ranges = []
  new_ranges = []
  passthrough_idx = []
  for i,s in enumerate(x.shape):
    if i not in x.arg:
      ranges.append(idx.src[1+i])
      continue
    passthrough_idx.append(idx.src[1+i])
    ranges.append(ctx.new_range(s))
    new_ranges.append(ranges[-1])
  # TODO: this should be able to be global or local
  ret = x.src[0].index(*ranges).bufferize(*[x for x in new_ranges if x.op is not Ops.CONST],
                                          arg=BufferizeOpts(device=None, addrspace=AddrSpace.LOCAL))
  return ret.index(*passthrough_idx)

def map_realize(ctx:RangeifyContext, x:UOp):
  if x.arg is not None: return None
  ranges = [ctx.new_range(s) for s in x.shape]
  return x.src[0].index(*ranges).bufferize(*x.src[1:], *ranges, arg=BufferizeOpts(device=x.device), tag=x.src[0].tag)

def map_reduce(ctx:RangeifyContext, idx:UOp, red:UOp):
  rngs = list(idx.src[1:])
  new_ranges = []
  for i,s in enumerate(red.src[0].shape):
    if i in red.arg[1]:
      rngs[i] = ctx.new_range(s, axistype=AxisType.REDUCE)
      new_ranges.append(rngs[i])
  return UOp(Ops.REDUCE, red.dtype, src=(red.src[0].index(*rngs),)+tuple(new_ranges), arg=red.arg[0], tag=red.tag)

def index_child(ctx:RangeifyContext, c:UOp, x:UOp, idx:UOp):
  if c not in ctx.seen_children: ctx.seen_children[c] = {}
  # wait here until we have seen all the children
  if len(ctx.seen_children[c]) != x.arg[1]:
    ctx.progress += 1
    if ctx.progress > 10000: raise RuntimeError("children not making progress")
    # NOTE: we mark this here
    ctx.seen_children[c][x.arg[0]] = idx
    raise RewriteNotReady
  ctx.progress = 0

  if c not in ctx.seen_child:
    all_rngs = list(zip(*[ch.src[1:] for ch in ctx.seen_children[c].values()]))
    out_rngs = []
    end_ranges = []
    idx_ranges = []
    # NOTE: locals aren't working, so we only fully bufferize here (unless RANGEIFY > 1)
    all_all_same = all(all_same(r) for r in all_rngs)
    for i,valid_rngs in enumerate(all_rngs):
      rngs, valids = zip(*[(r.get_idx(), r.get_valid()) for r in valid_rngs])
      # we compare the ranges without their valids
      if all_same(rngs) and (all_all_same or RANGEIFY > 1):
        # the new valid is the OR of all the children valids
        minimum_valid = functools.reduce(operator.or_, valids, UOp.const(dtypes.bool, False))
        out_rngs.append(minimum_valid.where(rngs[0], UOp.invalid()).simplify())
      else:
        out_rngs.append(ctx.new_range(c.shape[i]))
        end_ranges.append(out_rngs[-1])
        idx_ranges.append(i)
    ctx.seen_child[c] = (out_rngs, idx_ranges, end_ranges)
  else:
    out_rngs, idx_ranges, end_ranges = ctx.seen_child[c]
    for i,nr in zip(idx_ranges, end_ranges): out_rngs[i] = nr
  # index based on the shared ranges
  ret = c.index(*out_rngs)
  # if all ranges aren't the same between children, we have to bufferize
  if len(idx_ranges) > 0:
    if len(idx_ranges) == len(out_rngs):
      # this is a global bufferize
      ret = ret.bufferize(*end_ranges, arg=BufferizeOpts(device=x.device))
    else:
      assert RANGEIFY > 1, "this isn't supported with RANGEIFY=1"
      ret = ret.bufferize(*end_ranges, arg=BufferizeOpts(device=None, addrspace=AddrSpace.LOCAL))
    ret = ret.index(*[idx.src[1+i] for i in idx_ranges])
  return ret

def children_gate(ctx:RangeifyContext, idx:UOp, c:UOp):
  if len(ctx.seen_children[c]) != c.arg: raise RuntimeError("all children should have been seen by now")
  return idx.replace(src=(idx.src[0].src[0],)+idx.src[1:])

def might_end_axis(idx:UOp):
  if idx.arg is None: return None
  # TODO: write a proper cost function here
  if all(x.op not in {Ops.BUFFER, Ops.REALIZE, Ops.BUFFERIZE} for x in idx.toposort()): return None
  if all(x.op not in {Ops.REDUCE_AXIS} for x in idx.toposort()): return None
  to_end_axis = []
  for i,a in enumerate(idx.src[1:]):
    if any(x.arg > idx.arg for x in a.toposort() if x.op is Ops.RANGE):
      to_end_axis.append(i)
  if to_end_axis: return idx.replace(src=(idx.src[0].realize(arg=tuple(to_end_axis)),)+idx.src[1:], arg=None)
  return idx.replace(arg=None)

def unprocessed_index(x:UOp): raise RuntimeError(f"unprocessed index on {x.src[0].op}")

pm_rangeify = pm_mops+PatternMatcher([
  # sink contigs to kick it off
  (UPat(Ops.REALIZE, src=(UPat(),), name="x", allow_any_len=True), map_realize),
  # if there's an INDEX it can support partial contig
  (UPat(Ops.INDEX, src=(UPat(Ops.REALIZE, src=(UPat(),), name="x"),), allow_any_len=True, name="idx"), map_partial_realize),

  # if there are new ended children, tag the SINK
  (UPat(Ops.INDEX, src=(UPat(Ops.CHILD, src=(UPat(name="c"), ), name="x"),), allow_any_len=True, name="idx"), index_child),
  (UPat(Ops.INDEX, src=(UPat(Ops.CHILDREN, name="c"),), allow_any_len=True, name="idx"), children_gate),

  # if we come across this, remove it. it was a CHILD unused in an INDEX
  (UPat(Ops.CHILD, src=(UPat(Ops.CHILDREN, src=(UPat.var("x"),)),)), lambda x: x),

  # CONST (or DEFINE_VAR) can't have axes. remove INDEX when we get here
  (UPat(Ops.INDEX, src=(UPat((Ops.CONST, Ops.DEFINE_VAR), name="c"),)), lambda c: c),

  # handle arg on any op with weight. old endrange stuff
  (UPat(Ops.INDEX, src=(UPat(GroupOp.Elementwise.union({Ops.REDUCE_AXIS})),), allow_any_len=True, name="idx"), might_end_axis),

  # handle size 0
  (UPat(Ops.INDEX, name="x"), lambda x: x.replace(src=(x.const_like(0),)+x.src[1:]) if x.st is not None and x.size == 0 else None),

  # handle assign
  (UPat(Ops.INDEX, src=(UPat(Ops.ASSIGN, name="assign"),), allow_any_len=True, name="x"),
    lambda x,assign: assign.replace(src=tuple([s.index(*x.src[1:]) for s in assign.src])+(assign.src[0],))),

  # move MAP through elementwise ALU / reduce. these are the items with cost
  (UPat(Ops.INDEX, src=(UPat(GroupOp.Elementwise.union(
    {Ops.STORE, Ops.COPY, Ops.BUFFER_VIEW, Ops.DEVICE, Ops.BIND, Ops.CONTIGUOUS, Ops.NOOP})),), allow_any_len=True, name="x"),
   lambda x: x.src[0].replace(src=tuple([s.index(*x.src[1:]) for s in x.src[0].src]))),
  (UPat(Ops.INDEX, src=(UPat(Ops.REDUCE_AXIS, name="red"),), allow_any_len=True, name="idx"), map_reduce),

  # assert if there's any index we didn't process
  (UPat(GroupOp.All-{Ops.REALIZE, Ops.BUFFERIZE}).f(Ops.INDEX, name="x"), unprocessed_index),
])

# *****************
# 3.5 cleanups

# you don't know in the first pass if axes are going to die, this happens if there's an EXPAND to the left
def cleanup_dead_axes(b:UOp):
  new_rng = []
  hit = False
  reshape: list[sint] = []
  for s,rng in zip(b.shape, b.src[1:]):
    if rng not in b.src[0].sparents and rng.op is Ops.RANGE:
      reshape.append(1)
      hit = True
    else:
      reshape.append(s)
      new_rng.append(rng)
  if hit:
    return b.replace(src=b.src[0:1]+tuple(new_rng)).reshape(tuple(reshape)).expand(b.shape)

# if a buffer is being stored just for permutes or something, remove it
# we want to reexpress the indexes of idx2 in terms of the implied b1
def remove_bufferize(src:UOp, buf:UOp, idx:UOp):
  # see if we can't do it, should this ever hit?
  assert len(buf.src) == len(idx.src), "index on wrong bufferize"
  assert all(x.op is Ops.RANGE for x in buf.src[1:])

  # if it's user contiguous, we never remove it
  if src.op is Ops.CONTIGUOUS: return None

  # here is where we compute the cost
  # for now just no REDUCE, COPY, or ASSIGN
  ran = src.toposort(gate=lambda x: x.op not in {Ops.INDEX})
  # we don't want to bufferize threefry, also causes problems because not all platforms support long
  if any(x.op in {Ops.REDUCE, Ops.COPY, Ops.BUFFER_VIEW, Ops.ASSIGN} for x in ran) and src.op is not Ops.THREEFRY: return None

  # simple, matching old behavior
  #if src.op is not Ops.INDEX: return None

  # this is the ranges replaced
  return src.substitute(dict(zip(buf.src[1:], idx.src[1:])))

def pre_bufferize(b:UOp, x:UOp, copy:UOp):
  nb = b.replace(src=(b.src[0].contiguous(),)+b.src[1:])
  return copy.replace(src=(x.replace(src=(nb,)+x.src[1:]), copy.src[1]))

pm_cleanups = double_reshape+pm_mops+PatternMatcher([
  #(UPat(Ops.BUFFERIZE, name="b"), cleanup_dead_axes),
  # remove noop buffers. if we look at the next index we can remove even more of these
  # NOTE: this is mostly the same case as below, but if there's no INDEX this gets more
  (UPat(Ops.INDEX, name="idx").f(Ops.BUFFERIZE, allow_any_len=True, name="b2"),
   lambda idx,b2: idx.src[0].replace(tag=nt if len(nt:=(idx.src[0].tag or ()) + (b2.tag or ())) else None) if idx.src[1:] == b2.src[1:] \
       and idx.src[0].op is not Ops.BUFFER_VIEW else None),
  # remove reindexing with cost function
  (UPat.var("src").f(Ops.BUFFERIZE, allow_any_len=True, name="buf").f(Ops.INDEX, allow_any_len=True, name="idx"), remove_bufferize),
  # no buffers for const
  (UPat(Ops.CONST, name='c').f(Ops.BUFFERIZE, allow_any_len=True, name="b"),
   lambda c,b: c.reshape((1,)*len(b.shape)).expand(b.shape).replace(tag=b.tag)),
  # if any CONST with DEVICE make it here (symbolic/copy issue), remove it
  #(UPat(Ops.DEVICE).f(Ops.CONST, name="c"), lambda c: c.replace(src=())),
  # copy on CONST is CONST
  (UPat(Ops.COPY, src=(UPat.cvar("x"), UPat()), name="copy"), lambda copy,x: copy.const_like(x.arg)),
  (UPat(Ops.COPY, src=(UPat(GroupOp.All-{Ops.CONTIGUOUS, Ops.COPY}).f(Ops.BUFFERIZE, allow_any_len=True, name="b")
                       .f(Ops.INDEX, allow_any_len=True, name="x"), UPat()), name="copy"), pre_bufferize),
])

# *****************
# 4. put in buffers for bufferize
# TODO: should BUFFERIZE look a lot more like STORE
# BUFFERIZE has device in arg
# BUFFERIZE doesn't have indexing, that's implied by the ranges it closes
# BUFFERIZE returns the BUFFER ready for INDEXing (doing this will make splitting a lot easier)
# NOTE: this has been fixed up a bit

def bufferize_to_store(x:UOp):
  rngs = x.src[1:]
  shape = tuple([int(r.vmax+1) for r in rngs])
  sym_shape = tuple([ssimplify(r.src[0]) for r in rngs])
  size = prod(shape)
  assert size > 0, f"no zero sized buffers {shape}"

  sdtype = x.dtype.ptr(size=size, addrspace=x.arg.addrspace)
  if x.src[0].op is Ops.ASSIGN:
    assign_target, assign_src, assign_mops = x.src[0].src
    assert assign_target.op is Ops.INDEX
    # in assign, this is the buffer size, not the bufferize size
    # TODO: assign_mops here
    ret = assign_target.replace(dtype=sdtype).store(assign_src, *rngs, dtype=x.dtype)
    mops = []
    walk = assign_mops
    while walk is not assign_mops.base:
      mops.append((walk.op, walk.arg))
      walk = walk.src[0]
    for m in mops[::-1]: ret = ret._mop(*m)
    return ret.forced_reshape(shape).replace(tag=x.tag)

  # NOTE: the DEFINE_LOCAL needs to be disambiguated here
  if sdtype.addrspace == AddrSpace.GLOBAL:
    buf = UOp.new_buffer(x.arg.device, size, x.dtype)
    ret = buf.reshape(shape).index(*rngs, dtype=sdtype).store(x.src[0], *rngs, dtype=x.dtype)
    ret = ret.forced_reshape(shape)
    # TODO: is this right? what if it's offset
    if shape is not sym_shape: ret = ret.shrink(tuple([(0,x) for x in sym_shape]))
    return ret.replace(tag=x.tag)

  # handle locals
  tag = x.arg.device
  if tag is None: tag = UOp.unique().arg # TODO: hack
  buf = UOp(Ops.DEFINE_LOCAL, sdtype, arg=tag)
  # store has the other dtype here
  # TODO: how is this unified?
  return buf.reshape(shape).index(*rngs, dtype=sdtype).store(x.src[0], *rngs, dtype=sdtype).forced_reshape(shape, dtype=x.dtype)

pm_add_buffers = pm_mops+PatternMatcher([
  (UPat(Ops.BUFFERIZE, name="x"), bufferize_to_store),

  # move RESHAPEs through MSELECT/MSTACK
  (UPat((Ops.MSELECT, Ops.MSTACK), src=UPat(Ops.RESHAPE), name="m"),
   lambda m: m.replace(src=tuple([x.src[0] for x in m.src])).reshape(m.src[0].arg)),
])

# *****************
# 5. split into kernels

@dataclass
class LocalAddBufferContext:
  dg:int = 0
  map:dict = field(default_factory=dict)
  vars:dict = field(default_factory=dict)
  range:int = 0

def debuf(ctx:LocalAddBufferContext, buf:UOp):
  ret = UOp(Ops.DEFINE_GLOBAL, buf.dtype.ptr(buf.arg), arg=ctx.dg)
  if buf not in ctx.map: ctx.map[buf] = buf
  ctx.dg += 1
  return ret

def unbind_kernel(ctx:LocalAddBufferContext, b:UOp):
  ctx.vars[b] = None
  return b.src[0]

def handle_assign(ctx:LocalAddBufferContext, assign:UOp):
  buf = assign.as_buf()
  # HACK to put the buffer in the MAP instead of MSTACK/MSELECT
  if buf.op in {Ops.MSTACK, Ops.MSELECT}: buf = buf.src[0]
  assert buf not in ctx.map
  ctx.map[buf] = assign
  return buf

def renumber_range(ctx:LocalAddBufferContext, r:UOp):
  if r.tag is not None: return None
  ret = r.replace(arg=(ctx.range,)+r.arg[1:], tag=())
  ctx.range += 1
  return ret

to_define_global = PatternMatcher([
  (UPat(Ops.BUFFER, name="buf"), debuf),
  (UPat(Ops.BIND, name="b"), unbind_kernel),
  (UPat((Ops.ASSIGN, Ops.MSTACK, Ops.MSELECT), name="assign"), handle_assign),

  # HACK in case any CONSTs were replaced
  # this is only needed if you are using symbolic
  (UPat((Ops.CONST, Ops.DEFINE_VAR), name="c"), lambda c: c.replace(src=()) if len(c.src) else None),

  # renumber the ranges starting with 0 so that kernel deduping works
  (UPat(Ops.RANGE, name="r"), renumber_range),
])

rangeify_codegen = PatternMatcher([
  # no NOOP in the kernel graph
  # TODO: this can be moved into codegen?
  (UPat((Ops.NOOP, Ops.CONTIGUOUS), name="x"), lambda x: x.src[0]),

  # strip the arg from store
  (UPat(Ops.STORE, name="x"), lambda x: x.replace(arg=None) if x.arg is not None else None),

  # add loads to non ptr indexes
  # TODO: this can be moved into codegen?
  (UPat((Ops.DEFINE_GLOBAL, Ops.STORE), name="dg").f(Ops.INDEX, name="idx", allow_any_len=True),
   lambda dg,idx: None if isinstance(idx.dtype, (PtrDType, ImageDType)) else idx.replace(dtype=dg.dtype, arg=None).load()),

  # TODO: this can be moved into codegen
  (UPat(Ops.STORE, name="store").f(Ops.INDEX, allow_any_len=True, name="idx").f(Ops.LOAD),
    lambda store,idx: idx.replace(src=(store.as_buf(),)+idx.src[1:]).load(store if idx.dtype.addrspace != AddrSpace.LOCAL else store.barrier())),

  # TODO: hack for group for reduce
  (UPat(Ops.IF, src=(UPat.var("gate"), UPat(Ops.LOAD, src=(UPat.var("src"), UPat.var("barrier"))),)),
   lambda src, barrier, gate: src.load(UOp(Ops.IF, src=(gate, barrier)))),
])

def split_store(ctx:list[UOp], x:UOp):
  if len(x.ranges): return None
  if x.src[0].ptrdtype.addrspace is AddrSpace.LOCAL: return None

  # local kernel rewrite
  lctx = LocalAddBufferContext()
  ret = graph_rewrite(x, to_define_global+rangeify_codegen, ctx=lctx, name="kernel split", bottom_up=True)

  # gather the metadata
  metadatas = [ctx[y].metadata for x in ret.sparents if x.tag is not None for y in x.tag]

  # NOTE: the hack for COPY is here
  ret = ret.sink() if ret.src[1].op not in {Ops.COPY, Ops.BUFFER_VIEW} else ret.src[1]
  kernel_arg = Kernel(ret,tuple(dedup(flatten([x for x in metadatas if x is not None]))))
  kernel = UOp(Ops.KERNEL, src=tuple(lctx.map.values())+tuple(lctx.vars.keys()), arg=kernel_arg)
  return x.as_buf().assign(kernel)

split_kernels = PatternMatcher([
  (UPat(Ops.STORE, name="x"), split_store),
])

def tag_uop(ctx:list[UOp], x:UOp):
  if x.tag is not None: return None
  ctx.append(x)
  return x.replace(tag=(len(ctx)-1,))
add_tags = PatternMatcher([
  # don't tag BUFFERs, they are global
  (UPat(GroupOp.All-{Ops.BUFFER, Ops.CONST, Ops.DEVICE, Ops.UNIQUE, Ops.DEFINE_VAR, Ops.BIND}.union(GroupOp.Movement), name="x"), tag_uop),
])

@track_rewrites(lambda _,ret: f"Schedule {pluralize('Kernel', len([u for u in UOp.sink(*ret.values()).toposort() if u.op is Ops.KERNEL]))}", True)
def get_rangeify_map(sink:UOp) -> dict[UOp, UOp]:
  uop_list: list[UOp] = []
  tsink = graph_rewrite(sink, add_tags, ctx=uop_list, bottom_up=True, name="number the uops")

  # HACKS: handle multi with graph_rewrite_map in order to not have to add all the tag logic to multi
  msink = graph_rewrite_map(tsink, multi_pm, name="multi")
  tsink = msink[tsink].substitute({v:v.rtag(k.tag) for k,v in msink.items() if v.tag is None and k.tag is not None})

  tsink = graph_rewrite(tsink, earliest_rewrites, name="earliest rewrites")
  realize_map: dict[UOp, UOp] = {}
  graph_rewrite(tsink, do_realize, ctx=realize_map, name="Input Graph")
  # NOTE: we don't use contiguous here, contiguous is a user op
  tsink = graph_rewrite(tsink, add_contiguous, ctx=realize_map, bottom_up=True, name="add realize")
  tsink = graph_rewrite(tsink, remove_contig_tags, name="remove contiguous tags")
  tsink = graph_rewrite(tsink, pm_children, ctx=ChildrenContext(), bottom_up=True, name="get children")

  # rangeify
  tsink = graph_rewrite(tsink, pm_rangeify, ctx=RangeifyContext(), bottom_up=True, name="rangeify")
  # NOTE: sym (vs symbolic_simple) breaks things here because ranges with len 1 aren't handled right
  tsink = graph_rewrite(tsink, symbolic_simple, name="symbolic")  # this supports const folding
  tsink = graph_rewrite(tsink, pm_cleanups, bottom_up=True, name="remove costly buffers")

  # rebuild the sink with all the BUFFERIZEs with tags, this is what's ending up in the tensor graph
  # if it's not tagged by here, it's out
  tsink = UOp.sink(*[x for x in tsink.parents if (x.op is Ops.BUFFERIZE or x.base.op in {Ops.CONST}) and x.tag is not None])

  if getenv("VIZ"): graph_rewrite(tsink, PatternMatcher([]), name="View Tagged Rangeify")

  # bufferize -> store
  tsink = graph_rewrite(tsink, pm_add_buffers, bottom_up=True, name="bufferize to store")
  tsink = graph_rewrite(tsink, split_kernels, ctx=uop_list, name="split kernels")

  # if a kernel depends on a buffer, and that buffer is later assigned to, make the assign depend on the kernel's assign
  kernel_assign: dict[UOp, UOp] = {}
  assign_rep: dict[UOp, UOp] = {}
  for u in tsink.toposort():
    if u.op is not Ops.ASSIGN: continue
    kernel_assign[u.buf_uop] = u
    for s in u.src[1].src:
      # TODO: this is probably broken for MSELECT/MSTACK
      if s.op is not Ops.BUFFER or s is u.buf_uop or (a:=kernel_assign.get(s)) is None: continue
      if any(x.op is Ops.ASSIGN and x.buf_uop is s for x in u.toposort()):
        raise RuntimeError(f"cycle detected in graph, kernel for {u.buf_uop} must either depend on ASSIGN or BUFFER")
      assign_rep[a] = kernel_assign[s] = a.replace(src=a.src+(u,))
  if assign_rep: tsink = graph_rewrite(tsink, _substitute, ctx=assign_rep, bottom_up=True, name="fix_assign")

  if getenv("VIZ"): graph_rewrite(tsink, PatternMatcher([]), name="View Kernel Graph")

  becomes_map: dict[UOp, UOp] = {}
  for s in tsink.src:
    assert s.tag is not None
    for a in s.tag:
      if a is None: continue
      becomes_map[uop_list[cast(int, a)]] = s.replace(tag=None)
  return becomes_map
