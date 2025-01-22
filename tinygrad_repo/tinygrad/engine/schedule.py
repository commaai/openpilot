import sys, atexit, functools, pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field
from tinygrad.ops import GroupOp, UOp, Ops, PatternMatcher, UPat, Variable, can_pad, graph_rewrite, resolve, track_rewrites, view_left, merge_views
from tinygrad.ops import identity_element, buffers, symbolic_simple, type_verify
from tinygrad.helpers import Context, Metadata, all_int, all_same, colored, diskcache_put, merge_dicts, prod, dedup, getenv, unwrap
from tinygrad.helpers import FUSE_CONV_BW, FUSE_ARANGE, DEBUG, ContextVar
from tinygrad.dtype import DType, ImageDType, dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, strides_for_shape
from tinygrad.device import Buffer

# creation can recurse a lot
sys.setrecursionlimit(10000)

BUF_LIMIT = {"METAL":32}

# **** big graph spec

tensor_uop_spec = PatternMatcher([
  # ** stable and well understood specs

  # DEVICE and BUFFER
  (UPat(Ops.DEVICE, dtypes.void, (), name="device"), lambda device: isinstance(device.arg, str)),
  (UPat(Ops.BUFFER, src=(UPat(Ops.DEVICE),), name="buf"), lambda buf:
   # arg: (number, size)
   isinstance(buf.arg, tuple) and len(buf.arg) == 2 and all_int(buf.arg) and \
   # dtype
   isinstance(buf.dtype, (DType, ImageDType))),

  # movement ops
  (UPat(GroupOp.Movement, name="mv", src=(UPat.var("x"),)), lambda mv,x:
   # naturally correct
   (isinstance(mv.arg, tuple) and mv.dtype == x.dtype) or
   # "make things that can't be images not images" can change the buffer dtype
   # this is fine as long as it's a realized buffer and base dtypes match.
   ((isinstance(mv.dtype, ImageDType) or isinstance(x.dtype, ImageDType)) and x.dtype.base == mv.dtype.base and x.is_realized)),

  # Tensor variable bindings
  (UPat(Ops.BIND, dtypes.int, (UPat(Ops.DEFINE_VAR), UPat.cvar(dtype=dtypes.int)), arg=None), lambda: True),

  # Tensor const has a ShapeTracker of shape=() and a device
  (UPat(Ops.CONST, src=(UPat(Ops.VIEW, src=(UPat(Ops.DEVICE),)),)), lambda: True),

  # DETACH and CONTIGUOUS change how we interpret the source UOp
  # CONTIGUOUS ensures the source UOp realizes
  (UPat((Ops.DETACH, Ops.CONTIGUOUS), name="root", src=(UPat.var("x"),), arg=None), lambda root,x: root.dtype == x.dtype),

  # ** specs with room for refactoring and improving

  # COPY
  (UPat(Ops.COPY, name="copy", src=(UPat.var("copyin"),)), lambda copy,copyin:
   # arg (device, clone?)
   isinstance(copy.arg, tuple) and len(copy.arg) == 2 and isinstance(copy.arg[0], str) and isinstance(copy.arg[1], bool) and \
   # dtype
   copy.dtype == copyin.dtype),

  # VIEW(BUFFER) applies a ShapeTracker on top of the underlying device buffer
  # NOTE: VIEW size exactly matches the underlying BUFFER, tensor doesn't apply movement ops to the VIEW
  (UPat(Ops.VIEW, name="view", src=(UPat(Ops.BUFFER, name="buf"),)),
   lambda view,buf: view.dtype == buf.dtype and view.size == buf.size and view.st.contiguous),

  # ASSIGN changes the value of an existing buffer
  (UPat(Ops.ASSIGN, name="assign", src=(UPat.var("target"), UPat.var("new_val"))), lambda assign,target,new_val:
   # target must be a realized device buffer
   (target.op is Ops.BUFFER or target.is_realized) and
   # dtype
   (assign.dtype == target.dtype == new_val.dtype)),

  # ** TODO: these UOps need new specs, the current representation relies on hacks

  # BUFFER and VIEW specify device and shape for meta ops
  (UPat(Ops.VIEW, name="view", src=(UPat(Ops.BUFFER, name="buf"), UPat(GroupOp.Meta, name="uop"))),
   lambda view,buf,uop: view.dtype == buf.dtype == uop.dtype and view.size == buf.size),

  # DEVICE and VIEW specify device and shape for BIND
  (UPat(Ops.VIEW, src=(UPat(Ops.DEVICE), UPat(Ops.BIND))), lambda: True),

  # NOTE: EMPTY just ensures the source BUFFER is allocated before children run
  # TODO: this should be EMPTY(VIEW(BUFFER))
  (UPat(Ops.EMPTY, src=(), arg=None), lambda: True),

  # TODO: BUFFER_VIEW is overloaded, can we break it into multiple well defined UOps?
  # BUFFER_VIEW shares the device buffer with its source, it uses a subbuffer of the underlying source buffer

  (UPat(Ops.BUFFER_VIEW, name="root", src=(UPat.var("x"),)), lambda root,x:
   # BUFFER_VIEW can replace contiguous, keeping dtype the same
   (root.dtype == x.dtype) or
   # it can also replace bitcast, this changes the dtype, but the itemsize stays the same
   (root.dtype != x.dtype and root.dtype.itemsize == x.dtype.itemsize) or
   # it can also represent shape changing bitcast (only on DISK)
   (root.dtype != x.dtype and root.dtype.itemsize != x.dtype.itemsize and x.device.startswith("DISK"))),
])

# **** ScheduleItem return type

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...]
  assign_preloads: tuple[UOp, ...]
  @property
  def outputs(self) -> tuple[Buffer, ...]:
    """Read/write or write only buffers in the schedule."""
    return tuple(b for i,b in enumerate(self.bufs) if i in self.output_idxs)
  @property
  def inputs(self) -> tuple[Buffer, ...]:
    """Read only buffers in the schedule."""
    return tuple(b for i,b in enumerate(self.bufs) if i not in self.output_idxs)
  @functools.cached_property
  def output_idxs(self) -> tuple[int, ...]: return tuple(x.src[0].arg for x in self.ast.src) if self.ast.op is Ops.SINK else (0,)

# **** Schedule context and big graph

@dataclass(frozen=True)
class ScheduleContext:
  tensor_uops: dict[UOp, list[UOp]] = field(default_factory=dict)    # this maps BUFFER uops of this schedule to the tensor uop
  var_vals: dict[Variable, int] = field(default_factory=dict)        # this maps a BIND's DEFINE_VAR to its value
  assigns: set[UOp] = field(default_factory=set)                     # this holds all the BUFFER uops we ASSIGN to in this schedule
  realizes: dict[UOp, UOp] = field(default_factory=dict)             # this holds all the BUFFER uops we mutate in this schedule
  allbufs: dict[UOp, UOp] = field(default_factory=dict)              # this maps BUFFER uops the actual op
  ops_metadata: dict[UOp, Metadata] = field(default_factory=dict)    # this maps fused ops to Metadata
  contiguous: dict[UOp, UOp] = field(default_factory=dict)           # this maps roots to places they are made contiguous
  children: defaultdict[UOp, dict[UOp, None]] = field(default_factory=lambda: defaultdict(dict))

# TODO: delete this once BIND has a VIEW source
def is_constant(u:UOp): return u.op is Ops.CONST or (u.op is Ops.VIEW and len(u.src) == 2 and u.src[1].op is Ops.BIND)

# wrap tensor uops around a VIEW(BUFFER, <uop>)
# this BUFFER preserves a link back to the uop on the tensor after the scheduler rewrites it.
def add_buffers(buf:UOp, ctx:ScheduleContext, cache:dict[UOp, UOp]) -> UOp:
  if (r:=cache.get(buf)) is not None: return r
  if buf.op is Ops.SINK: return UOp.sink(*[add_buffers(x, ctx, cache) for x in buf.src])
  # shapeless op is passthrough
  # realized is passthrough
  # constants are passthrough
  if buf.st is None or buf.base.is_realized or is_constant(buf.base): return buf
  # view is passthrough
  if buf is not buf.base:
    cache[buf] = ret = add_buffers(buf.base, ctx, cache).view(buf.st)
    return ret
  # make things that can't be images not images
  dtype = buf.dtype
  if isinstance(dtype, ImageDType) and (prod(buf.shape) != prod(dtype.shape) or not any(buf.shape[x]%4 == 0 for x in buf.st.unit_stride_axes())):
    if DEBUG >= 2: print(f"forcing image {dtype} with shape {buf.shape} to {dtype.base}")
    dtype = buf.dtype.base
  # meta ops and assign already have a target buffer, otherwise we create a new one
  buf_uop = buf.buf_uop if buf.op in {Ops.ASSIGN, Ops.VIEW} else UOp.new_buffer(buf.device, buf.size, dtype)
  # TODO: we need to rethink meta ops having buffers at creation time
  if buf.op is Ops.VIEW: op = buf.src[1].replace(src=tuple(add_buffers(x, ctx, cache) for x in buf.src[1].src))
  else: op = buf.replace(dtype=dtype.base, src=tuple(add_buffers(x, ctx, cache) for x in buf.src))
  # track the underlying tensor uop for this op
  ctx.tensor_uops[buf_uop] = [buf]
  # (early) bufferize
  cache[buf] = ret = UOp(Ops.VIEW, dtype.base, (buf_uop, op.alu(Ops.CONTIGUOUS) if buf.forced_realize else op), buf.st)
  return ret

# **** AST graph rewrite

# ** movement ops

def apply_swizzle(u:UOp) -> UOp:
  with Context(TRACK_MATCH_STATS=0): return graph_rewrite(u, view_left)

def swizzle_r(r:UOp, src:UOp, st:ShapeTracker) -> UOp:
  input_st = ShapeTracker.from_shape(unwrap(src.st).shape)
  tmp = input_st.permute(tuple(i for i in range(len(input_st.shape)) if i not in r.axis_arg)+r.axis_arg)
  prshape = prod(rshape:=tmp.shape[-len(r.axis_arg):])
  strides = strides_for_shape(rshape)
  nv = [View.create(v.shape+rshape, tuple(x*prshape for x in v.strides)+strides,
                    v.offset*prshape, v.mask+tuple((0,s) for s in rshape) if v.mask is not None else None) for v in st.views]
  # update input_st and axis
  new_input_st = tmp + ShapeTracker(tuple(nv))
  new_axis = tuple(range(len(st.shape), len(st.shape) + len(r.axis_arg)))
  return apply_swizzle(src.view(new_input_st)).r(r.arg[0], new_axis).view(ShapeTracker.from_shape(st.shape))

def reduceop_view_right(r:UOp, v:UOp, src:UOp) -> UOp:
  if not (swizzle_st:=unwrap(v.st)).contiguous or v.size != src.size: raise AssertionError(f"can't push {v} down through {src}")
  output_shape = swizzle_st.reduce(r.axis_arg)
  return src.r(r.arg[0], tuple(i for i,(s,u) in enumerate(zip(src.shape, output_shape)) if s != u)).view(ShapeTracker.from_shape(output_shape))

def elementwise_view_right(root:UOp) -> UOp|None:
  if len(swizzles:=[x for x in root.src if x.base is not x]) == 0: return None
  assert all(x.base.st is not None for x in swizzles), f"found shapeless VIEW src in {root}"
  assert all_same([x.base.size for x in swizzles]), f"swizzle inputs must have the same size {swizzles}"
  # push the swizzle from src to root
  output_swizzle = swizzles[0]
  new_input_st = ShapeTracker.from_shape(output_swizzle.base.shape)
  ret = root.replace(src=tuple(x if x.st is None else x.base if x in swizzles else apply_swizzle(x.view(new_input_st)) for x in root.src))
  # NOTE: swizzle resolves once we hit STORE
  return ret if ret.op is Ops.STORE else ret.view(ShapeTracker.from_shape(output_swizzle.shape))

def merge_double_reduce(root:UOp, first_reduce:UOp) -> UOp:
  assert root.arg[0] == first_reduce.arg[0], "can't merge reduceops with different alu"
  assert not any(x.op is Ops.REDUCE_AXIS for x in first_reduce.src[0].toposort), "can't merge more than two reduceops at a time"
  return first_reduce.replace(arg=(first_reduce.arg[0], root.axis_arg+first_reduce.axis_arg))

# push VIEW to stores
view_right = merge_views+PatternMatcher([
  # STORE(.., ASSIGN(VIEW(BUFFER), new_val)) -> STORE(.., new_val).view()
  (UPat(Ops.STORE, src=(UPat.var("b"), UPat.var("st"), UPat.assign(UPat.var("target"), UPat.var("val")))),
   lambda b,target,st,val: apply_swizzle(UOp.store(b, st, val).view(target.st))),
  # REDUCE(src.view(contiguous=False)) -> REDUCE(src.view(contiguous=True)).view()
  (UPat(Ops.REDUCE_AXIS, src=(UPat.var("src"),), name="r").view(name="v"), lambda v,r,src: None if v.st.contiguous else swizzle_r(r, src, v.st)),
  # REDUCE(src.view()) -> REDUCE(src).view()
  (UPat(Ops.REDUCE_AXIS, src=(UPat.var("src").view(name="v"),), name="r"), reduceop_view_right),
  # ALU(src.view()) -> ALU(src).view()
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN, Ops.CONTIGUOUS, Ops.STORE), name="root"), elementwise_view_right),
  # double reduce op collapses to a single reduce op
  (UPat(Ops.REDUCE_AXIS, src=(UPat(Ops.REDUCE_AXIS, name="first_reduce"),), name="root"), merge_double_reduce),
])

# ** ScheduleItem context builder

@dataclass(frozen=True)
class ScheduleItemContext:
  tensor_uops: dict[UOp, list[UOp]]
  ops_metadata: dict[UOp, Metadata]
  assigns: set[UOp]
  var_vals: dict[Variable, int]
  sinked: dict[UOp, UOp]
  sts: set[ShapeTracker] = field(default_factory=set)
  bufs: list[UOp] = field(default_factory=list)
  metadata: set[Metadata] = field(default_factory=set)
  assign_adj: dict[UOp, list[UOp]] = field(default_factory=dict)

def _append_st_vars(ctx:ScheduleItemContext, x:UOp) -> UOp|None:
  if (st:=unwrap(x.st)) in ctx.sts: return None
  st, var_vals = st.simplify().unbind()
  ctx.var_vals.update(var_vals)
  ctx.sts.add(st)
  return st.to_uop() if st != x.st else None

def _append_buf(ctx:ScheduleItemContext, x:UOp) -> UOp:
  ctx.bufs.append(x)
  return UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(size=x.size), (), len(ctx.bufs)-1)
append_bufs = PatternMatcher([(UPat(Ops.BUFFER, name="x"), _append_buf)])

def _append_preload(ctx:ScheduleItemContext, x:UOp, b:UOp) -> UOp:
  (adj_loads:=ctx.assign_adj.setdefault(b, [])).append(x)
  if not all_same([x.op for x in adj_loads]): raise RuntimeError(f"Detected cycle when fusing {adj_loads}. Can only fuse PRELOAD or LOAD of {b}")
  return x.replace(op=Ops.LOAD)
check_preload = PatternMatcher([(UPat(Ops.PRELOAD, src=(UPat.var("b"), UPat()), name="x"), _append_preload),])

to_si = PatternMatcher([
  (UPat(Ops.VIEW, name="x"), _append_st_vars),
  (UPat(Ops.SINK, src=(UPat.store(UPat.var("b"), UPat(), UPat(GroupOp.Meta, name="x")),)), lambda b,x: x.replace(src=(b, *x.src))),
  # don't need contiguous or assign anymore
  (UPat(Ops.CONTIGUOUS, src=(UPat.var("x"),)), lambda x: x),
  (UPat(Ops.ASSIGN, src=(UPat(), UPat.var("x"),)), lambda x: x),
])

add_metadata = PatternMatcher([(UPat(tuple(Ops), name="x"), lambda ctx,x: None if (m:=ctx.ops_metadata.get(x)) is None else ctx.metadata.add(m)),])
add_assign_adjacents = PatternMatcher([(UPat.load(UPat.var("b"), UPat(), name="x"), lambda ctx,b,x: ctx.assign_adj.setdefault(b, []).append(x)
                               if b in ctx.assigns else None)])

# late folding for multi output kernels
multioutput = PatternMatcher([(UPat.load(UPat.var("b"), UPat()), lambda ctx,b: ctx.sinked.get(b)),])

def schedule_uop(pre:UOp, ctx:ScheduleContext) -> ScheduleItem:
  # create the ast context
  si_ctx = ScheduleItemContext(ctx.tensor_uops, ctx.ops_metadata, ctx.assigns, ctx.var_vals, {x.buf_uop:x.src[2] for x in pre.src})
  create_ctx = add_metadata if len(si_ctx.assigns) == 0 else add_metadata+add_assign_adjacents
  sink = graph_rewrite(pre, create_ctx if len(si_ctx.sinked) == 1 else multioutput+create_ctx, si_ctx)
  # do movement ops
  sink = graph_rewrite(graph_rewrite(sink, view_left), view_right)
  # convert to AST
  sink = graph_rewrite(graph_rewrite(sink, to_si+check_preload if len(si_ctx.assigns) != 0 else to_si, si_ctx), append_bufs, si_ctx)
  # assert buffer count limit
  if (limit:=BUF_LIMIT.get(device:=si_ctx.bufs[0].device)) is not None and len(si_ctx.bufs) >= limit:
    if DEBUG >= 3: print(sink)
    raise RuntimeError(f"Kernel for {si_ctx.metadata} exceeded the {limit} buffer count limit for {device} with {len(si_ctx.bufs)} buffers.")
  # we also allow masked views. if it has a single view and it's equal when you shrink a contig, it's fine
  for ubuf,ops in si_ctx.assign_adj.items():
    if si_ctx.sinked.get(ubuf) is not None and not all((s:=x.st_arg).contiguous or (len(s.views) == 1 and (m:=s.views[0].mask) is not None \
        and ShapeTracker.from_shape(s.shape).shrink(m) == s.shrink(m)) for x in ops):
      raise RuntimeError("self operand of augmented assign must be contiguous.\nhelp: consider using .contiguous():\n"
                         +colored("   - a += a.T\n", "red")+colored("   + a += a.T.contiguous()", "green"))
  # can only schedule once
  for buf_uop in si_ctx.sinked:
    for luop in si_ctx.tensor_uops[buf_uop]: luop.become(buf_uop.view(unwrap(luop.st)))
  # capture process replay
  if getenv("RUN_PROCESS_REPLAY"):
    PROCESS_REPLAY_CAPTURE[str(pre.key)] = pickle.dumps((pre, si_ctx.assigns, {k:v.value for k,v in ContextVar._cache.items()}, sink))
  return ScheduleItem(sink, tuple(u.buffer for u in si_ctx.bufs if u.size != 0), tuple(si_ctx.metadata),
                      tuple(ubuf for ubuf,ops in si_ctx.assign_adj.items() if any(x.op is Ops.PRELOAD for x in ops)))

PROCESS_REPLAY_CAPTURE: dict[str, bytes] = {}
if getenv("RUN_PROCESS_REPLAY"):
  @atexit.register
  def save_process_replay() -> None:
    for k,v in PROCESS_REPLAY_CAPTURE.items(): diskcache_put("schedule_process_replay", k, v, prepickled=True)

# **** Schedule grouping

def is_scheduled(u:UOp) -> bool: return u.op is Ops.VIEW and len(u.src) == 2 and u.src[0].op is Ops.BUFFER
def uval(u:UOp) -> UOp:
  assert is_scheduled(u), f"must be a scheduled op {u}"
  return r.src[0] if (r:=u.src[1]).op is Ops.CONTIGUOUS and not (r.src[0].base.op is Ops.VIEW and len(r.src[0].base.src) == 2) else r

def recursive_group(tr:UOp, st:ShapeTracker, r:UOp, children:defaultdict[UOp, dict[UOp, None]], allbufs:dict[UOp, UOp], realizes:dict[UOp, UOp],
                     reduce_for_op:dict[UOp, UOp], group:dict[UOp, None], cache:dict[tuple[UOp, ShapeTracker], None]) -> None:
  """recursively search the uop for groupable children, realize the UOp if a child can't group"""
  if (tr, st) in cache: return
  cache.setdefault((tr, st))
  rsize = unwrap(allbufs[r].st).size
  if tr in realizes and tr is not r:
    # can only fuse contiguous
    # max one reduceop per kernel
    if not st.contiguous or st.size != rsize or tr in reduce_for_op: group.setdefault(r)
    return group.setdefault(tr)
  for tr_next in children[tr]:
    # max one reduceop per kernel
    if (tr_next_uop:=uval(allbufs[tr_next]).base).op is Ops.REDUCE_AXIS: return group.setdefault(r)
    # can only fuse contiguous
    if len(st_childs:=dedup(unwrap(x.st) for x in tr_next_uop.src if is_scheduled(x.base) and x.base.buf_uop == tr)) > 1: return group.setdefault(r)
    recursive_group(tr_next, st+st_childs[0], r, children, allbufs, realizes, reduce_for_op, group, cache)

def get_isolated_children(r:UOp, reduce_for_op:dict[UOp, UOp], children:defaultdict[UOp, dict[UOp, None]], allbufs:dict[UOp, UOp],
                           realizes:dict[UOp, UOp], group:dict[UOp, None]) -> dict[UOp, None]:
  rc_parents, cache = deque(group), set()
  while rc_parents:
    if (p:=uval(allbufs[rc_parents.pop()])) in cache: continue
    cache.add(p)
    # max one reduceop per kernel
    if p.op is Ops.REDUCE_AXIS: return {}
    rc_parents.extend(x.base.buf_uop for x in p.src if is_scheduled(x.base) and x.base.buf_uop is not r)
  # search descendants of the reduceop that can cleanly group
  descendants: dict[UOp, None] = {}
  for tr in group: recursive_group(tr, unwrap(allbufs[tr].st), tr, children, allbufs, realizes, reduce_for_op, descendants, cache={})
  return merge_dicts([group, {} if any(tr in group for tr in descendants) else descendants])

def group_realizes(ctx:ScheduleContext) -> list[list[UOp]]:
  """search the big graph for all the reduceops that need to realize, sometimes group/fuse the reduceop"""
  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  reduce_for_op: dict[UOp, UOp] = {}
  reduce_of_const: list[UOp] = []
  double_reduces: list[UOp] = []
  for r, r_uop in ctx.allbufs.items():
    if (r_uop:=uval(r_uop)).op is not Ops.REDUCE_AXIS: continue
    if FUSE_CONV_BW and uval((x:=r_uop.src[0]).base).op is r_uop.op and x.base is not x: double_reduces.append(r)
    if r in ctx.realizes: continue
    group: dict[UOp, None] = {}
    recursive_group(r, unwrap(r_uop.st), r, ctx.children, ctx.allbufs, ctx.realizes, reduce_for_op, group, cache={})
    # max one reduceop per kernel
    can_chase = all(tr not in reduce_for_op for tr in group)
    # TODO: forced_realize exists because the scheduler is incapable of checking for self-contained DAGs
    forced_realize = r in group
    if not forced_realize and len(group) > 1:
      group = get_isolated_children(r, reduce_for_op, ctx.children, ctx.allbufs, ctx.realizes, group)
    # can only fuse assign if no other assign_target is used in the kernel
    if not forced_realize and any(x in ctx.assigns for x in group):
      parents = deque((r, *group))
      while parents and not forced_realize:
        if (p_uop:=ctx.allbufs.get(p:=parents.pop())) is None: continue
        if (p_uop:=uval(p_uop)).op is Ops.ASSIGN and p not in group: forced_realize, can_chase = True, False
        if p in ctx.realizes: continue
        parents.extend([x.base.src[0] for x in p_uop.src if x.base.op is Ops.VIEW and len(x.base.src) != 0])
    if forced_realize or not group:
      tr = r
      if can_chase:
        # can chase this down to contiguous children
        st = unwrap(r_uop.st)
        while len(ctx.children[tr]) == 1:
          tr_next_uop = uval(ctx.allbufs[(tr_next:=next(iter(ctx.children[tr])))])
          st_childs = dedup([unwrap(x.st) for x in tr_next_uop.src if is_scheduled(x.base) and x.base.buf_uop is tr])
          if len(st_childs) > 1: break
          if st.size != st_childs[0].size: break
          st = st + st_childs[0]
          if not st.contiguous or tr_next_uop.op is Ops.REDUCE_AXIS: break
          tr = tr_next
        # don't cast to higher size before store (tr cannot be realized if forced_realize)
        if (tr_uop:=uval(ctx.allbufs[tr])).op is Ops.CAST and tr_uop.dtype.base.itemsize > tr_uop.src[0].dtype.base.itemsize:
          tr = tr_uop.src[0].base.buf_uop
      group = {tr: None}
      ctx.realizes[tr] = tr
    reduce_for_op.update((tr, r) for tr in group)
    if FUSE_ARANGE and r_uop.arg[0] is Ops.ADD and r_uop.src[0].base.op is Ops.CONST: reduce_of_const.append(r)
  # fuse double reduces with no other child
  for reduceop in double_reduces:
    top_reduce = uval(ctx.allbufs[reduceop]).src[0].base.buf_uop
    if len(ctx.children[top_reduce]) == 1: del ctx.realizes[top_reduce]
  # maybe fuse arange with its children
  for rbuf in reduce_of_const:
    group = {tr:None for tr,rop in reduce_for_op.items() if rop is rbuf}
    if any(luop.forced_realize for tr in group for luop in ctx.tensor_uops[tr]): continue
    kernel_children = {c for tr in group for c in ctx.children[tr] if uval(ctx.allbufs[c]).op not in {Ops.COPY, Ops.BUFFER_VIEW}}
    if len(kernel_children) == 0: continue
    for tr in group: del ctx.realizes[tr]
  # group BUFFER uops into kernels
  output_groups: defaultdict[UOp, list[UOp]] = defaultdict(list)
  for ubuf in ctx.realizes: output_groups[reduce_for_op.get(ubuf, ubuf)].append(ubuf)
  return list(output_groups.values())

# **** Schedule creation and BFS toposort

class UPatScheduled(UPat):
  def __init__(self, *args, **kwargs):
    super().__init__(Ops.VIEW, name="base", src=(UPat(Ops.BUFFER, name="b"), UPat(*args, **{"name":"to_store",**kwargs})))

# ** this is schedule level const folding

def simplify_reduceop(reduce:UOp, x:UOp) -> UOp|None:
  if not all_int(x.shape): return None
  # remove reduce on unmasked const
  prshape = prod(unwrap(x.st).shape[i] for i in reduce.arg[1])
  ret = x.const_arg
  match reduce.arg[0]:
    case Ops.ADD: ret *= prshape
    case Ops.MUL: ret **= prshape
    case Ops.MAX: pass # NOTE: Ops.MAX is passthrough
    case _: return None
  return reduce.const_like(ret)

def found_contiguous(ctx:ScheduleContext, contig:UOp, base:UOp, b:UOp):
  if contig.src[0].op is Ops.VIEW and len(contig.src[0].src):
    old_base = contig.src[0].src[0]
    if old_base.op is Ops.VIEW and (sti:=unwrap(contig.src[0].st).invert(old_base.shape)) is not None: ctx.contiguous[old_base] = base.view(sti)
def replace_contiguous(ctx:ScheduleContext, alu:UOp):
  new_src = list(alu.src)
  for i,s in enumerate(alu.src):
    if (replace_src:=ctx.contiguous.get(s, None)) is not None: new_src[i] = replace_src
  if tuple(new_src) != alu.src: return alu.replace(src=tuple(new_src))

ops_folding = symbolic_simple+PatternMatcher([
  # op with size 0 is zero
  (UPatScheduled(), lambda b,to_store,base: base.const_like(0) if base.size == 0 else None),
  # if the uop folded to a CONST we can delete the BUFFER
  (UPatScheduled(Ops.CONST, name="const"), lambda b,base,const: base.const_like(const.const_arg)),
  # DETACH is a NOOP here
  (UPat(Ops.DETACH, name="detach"), lambda detach: detach.src[0]),
  # reduce of size 0 is the identity element
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)),
   lambda reduce,x: reduce.const_like(identity_element(reduce.arg[0], reduce.dtype)) if x.size == 0 and reduce.size != 0 else None),
  # reduce of const is collapsed (TODO: make this a generic rule for stride0)
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.cvar("x"),)), simplify_reduceop),
  # CONST doesn't need COPY
  (UPat(Ops.COPY, src=(UPat.cvar("x"),)), lambda x: x),
  # no double COPY
  (UPat(Ops.COPY, src=(UPat(Ops.VIEW, src=(UPat(), UPat(Ops.COPY, name="base")),))), lambda base: base),
  # no COPY to same device, except clone (arg is True)
  (UPatScheduled(Ops.COPY, src=UPat(Ops.VIEW, name="copyin"), name="copy"),
   lambda base,b,copyin,copy: copyin if base.device == copy.device and copy.arg[1] is not True else None),
  # support for using a contiguous permuted view instead of the parent view if one exists
  (UPatScheduled(Ops.CONTIGUOUS, name="contig"), found_contiguous),
  (UPat(GroupOp.ALU, name="alu"), replace_contiguous),
])

# ** buffer merging

def merge(ctx:ScheduleContext, v1:UOp, b1:UOp, v2:UOp, b2:UOp) -> UOp:
  assert v1.st is not None and v2.st is not None and v1.st == v2.st, f"implicit movementop {v1.st} {v2.st}"
  # if b2 is realized also realize b1
  if b2 in ctx.realizes:
    ctx.realizes[b1] = b1
    del ctx.realizes[b2]
  # ops referring to b2 now ref to b1
  ctx.tensor_uops[b1] += ctx.tensor_uops[b2]
  del ctx.tensor_uops[b2]
  # merge
  return v1

def merge_realized(ctx:ScheduleContext, v1:UOp, b1:UOp, v2:UOp, b2:UOp):
  # early become
  for luop in ctx.tensor_uops.get(b1, [])+ctx.tensor_uops.get(b2, []): luop.become(b1.view(unwrap(luop.st)))
  return v1

merge_bufs = PatternMatcher([
  # merge base
  (UPat(Ops.VIEW, name="v2", src=(UPat(Ops.BUFFER, name="b2"), UPat(Ops.VIEW, name="v1", src=(UPat(Ops.BUFFER, name="b1"), UPat())))), merge),
  (UPat(Ops.VIEW, name="v2", src=(UPat(Ops.BUFFER, name="b2"), UPat(Ops.VIEW, name="v1", src=(UPat(Ops.BUFFER, name="b1"),)))), merge_realized),
  # collapse view
  (UPat(Ops.VIEW, src=(UPat(Ops.BUFFER), UPat(Ops.VIEW, src=(UPat(Ops.BUFFER), UPat())).view(name="mv"))), lambda mv:mv),
  (UPat(Ops.VIEW, src=(UPat(Ops.BUFFER), UPat(Ops.VIEW, src=(UPat(Ops.BUFFER),)).view(name="mv"))), lambda mv:mv),
])

# ** this decides which ops get realized

def realize(ctx:ScheduleContext, b:UOp, to_store:UOp, **kwargs) -> None: ctx.realizes[b] = to_store

def realize_view(ctx:ScheduleContext, view:UOp, src:UOp, b:UOp, **kwargs) -> None:
  if src.st is None: return None
  st = unwrap(view.st)
  # fold simple pads
  if len(st.views) == 1 and (m:=st.views[-1].mask) is not None and all_int(src.shape) and resolve(prod(src.shape) >= prod([y-x for x,y in m])):
    return None if can_pad(src, ctx.realizes, set()) else realize(ctx, b, src)
  # early realize before expand
  if resolve(prod(src.shape) < prod(st.shape)): return realize(ctx, b, src)
  # otherwise safety check pads
  return None if (all(v.mask is None for v in st.views) or can_pad(src, ctx.realizes, set())) else realize(ctx, b, src)

def fold_img_cast(ctx:ScheduleContext, xb:UOp, view:UOp, b:UOp, to_cast:UOp, **kwargs) -> UOp|None:
  if not isinstance(xb.dtype, ImageDType) or b not in ctx.realizes or xb not in ctx.realizes or uval(to_cast).op in GroupOp.Meta: return None
  del ctx.realizes[b]
  return to_cast.view(unwrap(view.st))

def sink_outputs(ctx:ScheduleContext, sink:UOp) -> UOp|None:
  new_src = tuple(x.base for x in sink.src if x.base.realized is None and not is_constant(x.base))
  for x in new_src: realize(ctx, x.buf_uop, x)
  return None if new_src == sink.src else UOp(Ops.NOOP) if len(new_src) == 0 else UOp.sink(*new_src)

do_realize = PatternMatcher([
  # always realize sinked ops
  (UPat(Ops.SINK, name="sink"), sink_outputs),
  # always realize meta ops
  (UPatScheduled({Ops.ASSIGN, Ops.CONTIGUOUS, *GroupOp.Meta}), realize),
  # realize before expand or unsafe pad ops
  (UPatScheduled(name="src").view(name="view"), realize_view),
  # don't realize image to image casts
  (UPatScheduled(Ops.CAST, src=(UPat(Ops.VIEW, src=(UPat.var("xb"), UPat()), name="to_cast"),), dtype=dtypes.float).view(name="view"), fold_img_cast),
  # realize before COPY or BUFFER_VIEW
  (UPat((Ops.COPY, Ops.BUFFER_VIEW), src=(UPat.any(UPatScheduled(), UPatScheduled().view()),)), realize),
])

# **** rewrite VIEW into LOAD/STORE/VALID or fuse the underlying UOp

def unbind_variable(ctx:ScheduleContext, bind:UOp, st:UOp):
  ctx.var_vals.update([bind.unbind()])
  return UOp.const(bind.dtype, bind).valid(unwrap(st.st))

def load_realized(ctx:ScheduleContext, b:UOp, st:UOp):
  # NOTE: if we're assigning to the BUFFER too, PRELOAD tells toposort to place this load before the ASSIGN
  return UOp(Ops.PRELOAD if b in ctx.assigns else Ops.LOAD, b.dtype.base, (b, unwrap(st.st).to_uop()))

def store_or_fuse(ctx:ScheduleContext, b:UOp, x:UOp, st:UOp):
  if (m:=ctx.tensor_uops[b][0].metadata) is not None: ctx.ops_metadata[x] = m
  if b not in ctx.realizes: return x # collapse BUFFER
  ctx.realizes[b] = UOp.store(b, ShapeTracker.from_shape(st.shape).to_uop(), x)
  return UOp(Ops.LOAD, x.dtype, (b, unwrap(st.st).to_uop()))

break_sched = PatternMatcher([
  # CONST is always fused and generated
  (UPat(Ops.CONST, name="x", src=(UPat(Ops.VIEW, name="st"),)), lambda x,st: UOp.const(x.dtype.base, x.const_arg).valid(st.st)),
  (UPat(Ops.VIEW, name="st", src=(UPat(Ops.DEVICE), UPat(Ops.BIND, name="bind"))), unbind_variable),
  # VIEW of BUFFER either becomes a LOAD/STORE or we fuse it
  (UPat(Ops.VIEW, name="st", src=(UPat(Ops.BUFFER, name="b"),)), load_realized),
  (UPat(Ops.VIEW, name="st", src=(UPat(Ops.BUFFER, name="b"), UPat.var("x"))), store_or_fuse),
])

# **** Schedule context builder

def append_uop(ctx:ScheduleContext, view:UOp, buf_uop:UOp) -> None:
  ctx.allbufs[buf_uop] = view
  if (op:=uval(view)).op is Ops.ASSIGN: ctx.assigns.add(buf_uop)
  for x in op.src:
    if is_scheduled(x.base): ctx.children.setdefault(x.base.buf_uop, {})[buf_uop] = None
  # BUFFER_VIEW overrides the underlying buffer
  # TODO: this should be a shrink on the buffer
  if op.op is Ops.BUFFER_VIEW:
    buffers[buf_uop] = (x:=op.src[0]).base.buffer.view(view.size, view.dtype, unwrap(x.st).views[0].offset*x.dtype.itemsize)
  buf_uop.buffer.ref(1)
create_ctx = PatternMatcher([(UPat(Ops.VIEW, name="view", src=(UPat(Ops.BUFFER, name="buf_uop"), UPat())), append_uop)])

# **** movement ops

remove_movement_ops = PatternMatcher([
  # NOTE: movement ops are always applied to base
  (UPat(GroupOp.Movement, name="mov", src=(UPat.any(UPat.var("x").view(), UPat.var("x")))), lambda x,mov: x.view(unwrap(mov.st))),
  # some masked views can collapse to 0, VIEW(x) -> CONST(VIEW)
  (UPat(Ops.VIEW, name="view"),
   lambda view: view.const_like(0) if (vm:=view.st.views[-1].mask) is not None and any((x[1]-x[0]) == 0 for x in vm) else None),
  # merge one src views.
  (UPat(Ops.VIEW, src=(UPat(Ops.VIEW, src=(UPat(),), name="v1")), name="v2"), lambda v1,v2: v1.replace(arg=v1.arg+v2.arg)),
  # merge unmasked const views
  (UPat(Ops.VIEW, name="view", src=(UPat(Ops.CONST, name="const", src=(UPat(Ops.VIEW, name="st"),) ),)),
   lambda st,const,view: const.replace(src=(st.replace(arg=st.st+view.st),)) if all(v.mask is None for v in (st.st+view.st).views) else None),
])

@track_rewrites(named=True)
def create_schedule_with_vars(outs:list[UOp], skip_check:bool=not __debug__) -> tuple[list[ScheduleItem], dict[Variable, int]]:
  if not skip_check: type_verify(list(UOp.sink(*outs).toposort), extra_spec=tensor_uop_spec)
  # to_uop is removing (many) of the movement ops
  sink = add_buffers(UOp.sink(*outs), ctx:=ScheduleContext(), cache={})
  # const folding and fusion
  sink = graph_rewrite(sink, remove_movement_ops+ops_folding+do_realize, ctx)
  sink = graph_rewrite(sink, merge_bufs, ctx)
  # create the scheduler context
  graph_rewrite(sink, create_ctx, ctx)
  # group realizes into kernels
  store_groups = group_realizes(ctx)
  graph_rewrite(sink, break_sched, ctx)
  # preschedule realize groups
  prescheduled: list[ScheduleItem] = []
  for store_uops in store_groups:
    if len(stores:=[ctx.realizes[u] for u in store_uops if ctx.realizes[u].op is Ops.STORE]) != 0:
      prescheduled.append(schedule_uop(UOp.sink(*stores), ctx))
  # do BFS
  schedule_targets = {out:si for si in prescheduled for out in si.outputs}
  graph: defaultdict[ScheduleItem, list[ScheduleItem]] = defaultdict(list)
  in_degree: defaultdict[ScheduleItem, int] = defaultdict(int)
  for si in prescheduled:
    # realize outputs before a parent is assigned to
    parents_assigns = dedup(xsi for x in si.assign_preloads if (xsi:=schedule_targets.get(x.buffer)) and xsi is not si)
    for assign in parents_assigns:
      graph[si].append(assign)
      in_degree[assign] += 1
    # realize outputs after all parents are realized
    scheduled_parents = dedup(xsi for x in si.inputs if (xsi:=schedule_targets.get(x)) is not None and xsi not in parents_assigns)
    for x in scheduled_parents:
      graph[x].append(si)
      in_degree[si] += 1
  queue = deque(si for si in prescheduled if in_degree[si] == 0)
  schedule: list[ScheduleItem] = []
  while queue:
    schedule.append(si:=queue.popleft())
    for x in graph[si]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)
  # confirm everything was scheduled correctly
  if len(schedule) != (groups:=len(prescheduled)): raise RuntimeError(f"cycle detected in graph, grouped {groups} but only scheduled {len(schedule)}")
  if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels")
  return schedule, ctx.var_vals
