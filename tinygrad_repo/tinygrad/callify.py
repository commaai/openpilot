from dataclasses import dataclass, field
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import UOp, UPat, PatternMatcher, Ops, GroupOp, ParamArg, graph_rewrite, track_rewrites
from tinygrad.helpers import VIZ, pluralize, all_int

@dataclass
class AllocCtx:
  uop_list: list[UOp] = field(default_factory=list)
  buffer_map: dict[UOp, UOp] = field(default_factory=dict)
  bases: set[UOp] = field(default_factory=set)
  assigns: list[UOp] = field(default_factory=list)
  replacements: list[UOp] = field(default_factory=list)

def tag_uop(ctx:AllocCtx, x:UOp):
  if x.tag is not None: return None
  ctx.uop_list.append(x)
  return x.replace(tag=(len(ctx.uop_list)-1,))

def disk_like(u:UOp): return isinstance(u.device, str) and u.device.startswith(("DISK", "TINYFS"))

def disk_copy_is_buffer(ctx:AllocCtx, u:UOp):
  # copies to disk are replaced with the disk buffer
  if disk_like(u) and u.tag is None:
    ctx.buffer_map[u] = u.empty_like()
    return u.rtag(())
  # all copies from disk/numpy are realized into a real buffer
  from_creation = isinstance(u.src[0].device, str) and u.src[0].device.startswith(("NPY", "DISK", "PYTHON", "TINYFS"))
  if from_creation: return tag_uop(ctx, u)

# CONTIGUOUS and AFTER + parents are the only nodes that get updated
add_tags = PatternMatcher([
  (UPat(Ops.COPY, name="u"), disk_copy_is_buffer),
  # no tag on copies that are assigned via STORE+AFTER — merge COPY tag into AFTER
  (UPat(Ops.AFTER, src=(UPat(), UPat(Ops.STORE, src=(UPat(name="dest"), UPat(Ops.COPY, name="c")))), name="a"),
   lambda a,c,dest: a.replace(src=(a.src[0], a.src[1].replace(src=(dest, c.rtag(())))), tag=a.tag+c.tag) if a.tag and c.tag else None),
  (UPat((Ops.CONTIGUOUS, Ops.AFTER), name="x"), tag_uop),
  (UPat(GroupOp.All, name="x"), lambda ctx,x: tag_uop(ctx,x) if x in ctx.bases else None),
])

def replace_contig_with_store_after(u:UOp):
  # can't allocate a buffer without a device (e.g., inside a CALL function body with only PARAMs)
  if u.device is None: return None
  # if size is 0, remove the contig
  if 0 in u.shape: return u.src[0]
  # no real contig for DISK/TINYFS tensors, they are left alone
  if disk_like(u): return u.rtag(None)
  buf = u.empty_like()
  return buf.after(buf.store(u.src[0])).rtag(u.tag)

def replace_store_after_with_contig(u:UOp, src:UOp):
  assigned_to = u
  while assigned_to.op in {Ops.BITCAST, Ops.AFTER, Ops.MULTI}: assigned_to = assigned_to.src[0].base
  if assigned_to.op not in {Ops.BUFFER, Ops.SLICE}: return src.contiguous(tag=u.tag)

def _make_buffer_view(src:UOp) -> UOp|None:
  """If movement ops on src collapse to a contiguous range, return SLICE. Otherwise None."""
  if (offset := src.contiguous_view_offset()) is None: return None
  buf = src.base
  if buf.op is Ops.SLICE:
    byte_offset = buf.src[1].arg * buf.src[0].dtype.itemsize + offset * src.dtype.itemsize
    buf = buf.src[0]
    if byte_offset % buf.dtype.itemsize != 0: return None
    offset = byte_offset // buf.dtype.itemsize
  return UOp(Ops.SLICE, src.dtype, (buf, UOp.const(dtypes.weakint, offset)), src.numel())

def contiguous_mops_to_view(c:UOp, src:UOp):
  """MOPS(BUFFER) → SLICE when movement ops collapse to a contiguous range."""
  buf = src.base
  if buf.op not in {Ops.BUFFER, Ops.SLICE, Ops.MULTI}: return None
  if src.op is Ops.RESHAPE and src.src[0].op in {Ops.BUFFER, Ops.SLICE} and c.op is not Ops.BITCAST: return None
  if c.op is not Ops.BITCAST and src.op is Ops.BUFFER: return None

  # no symbolic shape
  if not all_int(c.shape): return None

  if buf.op is not Ops.MULTI and (view := _make_buffer_view(src)) is not None:
    view = (view.replace(dtype=c.dtype, arg=c.numel()) if c.op is Ops.BITCAST else view).reshape(c.shape)
    return c.replace(src=(view,)) if c.op is Ops.COPY else view

  # for MULTI tensors, use multi_pm to resolve per-shard movement ops, then create SLICE on the resolved result
  if not isinstance(c.device, str):
    from tinygrad.schedule.multi import multi_pm
    resolved = graph_rewrite(src, multi_pm, name="multi_buffer_view")
    if resolved.op is not Ops.MULTI: return None
    if (view := _make_buffer_view(resolved.src[0])) is None: return None
    return view.reshape(resolved.src[0].shape).multi(resolved.arg).contiguous(tag=c.tag)

  return None

def _precompiled_output_redirect(s:UOp, t:UOp) -> UOp|None:
  # how output s lands in the caller's buffer t, or None if it must be copied into t
  # materialize straight into t
  if s.op is Ops.CONTIGUOUS: return t.after(t.store(s.src[0]))
  # rebind output storage to t
  if s.op in {Ops.BUFFER, Ops.MULTI} and s.has_buffer_identity(): return t
  return None

def transform_precompiled_call(c:UOp) -> UOp|None:
  if not c.arg.precompile: return None
  assert c.src[0].op is Ops.TUPLE, f"expected TUPLE body for precompiled FUNCTION, got {c.src[0].op}"
  input_buffers = tuple(x.contiguous() if x.op not in {Ops.AFTER, Ops.BIND} else x for x in c.src[1:])

  # add the outputs to the call
  srcs = c.src[0].src
  resolved = [c.gettuple(i) for i in range(len(srcs))]
  outs = tuple(r.empty_like() for r in resolved)
  targets = [o.param_like(len(c.src)-1+i).shrink_to(s.shape) for i,(o,s) in enumerate(zip(outs, srcs))]

  subs:dict[UOp, UOp] = {}
  items:list[UOp] = []
  for s, t in zip(srcs, targets):
    after_deps:list[UOp] = []
    while s.op is Ops.AFTER:
      after_deps.extend(s.src[1:])
      s = s.src[0]
    if (placed := _precompiled_output_redirect(s, t)) is not None and s not in subs:
      subs[s] = placed
      items.append(s.after(*after_deps) if after_deps else s)
    else:
      items.append(t.after(t.store(s.after(*after_deps))))
  fxn = UOp.sink(*(x.substitute(subs) for x in items))

  # body switches from TUPLE to SINK, so the node becomes an opaque CALL (not FUNCTION)
  new_call = UOp(Ops.CALL, src=(fxn, *input_buffers, *outs), arg=c.arg)
  rets = tuple(o.after(new_call) for o in outs)

  # if the CALL has symbolic shapes, shrink the max-sized output to the actual symbolic shape
  # NOTE: must use resolved shapes from the FUNCTION (which substitutes PARAMs with external args), not raw body shapes
  rets = tuple(r.shrink_to(rs.shape) for r,rs in zip(rets, resolved))

  return UOp.maketuple(*rets)

# NOTE: adding rules to here is bad. these all need to run before the schedule cache
pm_early_transform_tensor_graph = PatternMatcher([
  # transform precompiled FUNCTIONs into CALLs (body becomes SINK with stores)
  (UPat(Ops.FUNCTION, name="c"), transform_precompiled_call),

  # resolve TUPLE+GETTUPLE (for precompiled calls)
  (UPat(Ops.GETTUPLE, src=(UPat(Ops.TUPLE, name="t"),), name="g"), lambda g,t: t.src[g.arg]),

  # fold MOPS+BITCAST over BUFFER/SLICE into SLICE when movement ops collapse to contiguous range
  (UPat((Ops.BITCAST, Ops.COPY, Ops.CONTIGUOUS), src=(UPat(GroupOp.Movement|{Ops.BUFFER}, name="src"),), name="c"), contiguous_mops_to_view),

  # remove contiguous on movement ops before a copy on disk
  (UPat(GroupOp.Movement-{Ops.SHRINK, Ops.RESHAPE}, name="x").f(Ops.CONTIGUOUS).f(Ops.COPY, allow_any_len=True, name="copy"), lambda x,copy:
   copy.replace(src=(x,)+copy.src[1:], tag=None) if isinstance(x.device, str) and x.device.startswith("DISK") else None),
  # push copy past movement ops to disk
  (UPat(GroupOp.Movement-{Ops.SHRINK, Ops.RESHAPE}, name="x").f(Ops.COPY, name="copy"), lambda x,copy:
   x.replace(src=(copy.replace(src=(x.src[0],)+copy.src[1:], tag=None),)+x.src[1:]) \
   if isinstance(x.device, str) and x.device.startswith("DISK") else None),

  # add CONTIGUOUS to tagged UOps
  (UPat(GroupOp.All-{Ops.CONTIGUOUS, Ops.AFTER, Ops.STORE}, name="x"),
   lambda x: x.rtag(None).contiguous(tag=x.tag) if x.tag else x.replace(tag=None)),
  # remove extra CONTIGUOUS on AFTER (only when target is contiguous)
  (UPat(Ops.CONTIGUOUS, src=(UPat(Ops.AFTER, name="a"),), name="c"),
   lambda a,c: a.replace(tag=(a.tag or ())+(c.tag or ())) if a.src[0].has_buffer_identity() else None),
  # replace AFTER+STORE with CONTIGUOUS when target is not a buffer
  (UPat(Ops.AFTER, src=(UPat(), UPat(Ops.STORE, src=(UPat(), UPat(name="src")))), name="u"), replace_store_after_with_contig),
  # replace CONTIGUOUS with STORE+AFTER
  (UPat(Ops.CONTIGUOUS, name="u"), replace_contig_with_store_after),
  # remove DETACH/CONTIGUOUS_BACKWARD (allows more contiguous removal)
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD), name="x"), lambda x: x.src[0]),
])

def finalize_after(ctx:AllocCtx, x:UOp):
  # untagged: record as an assign for the call body
  if x.tag is None:
    ctx.assigns.append(x)
    return None
  # tagged: untag and map each original pre-rewrite UOp to the stripped buffer; the untagged result is reprocessed as untagged
  ret = x.replace(tag=None)
  replace_uop = ret
  while replace_uop.op is Ops.AFTER: replace_uop = replace_uop.src[0]
  for t in x.tag:
    original_uop: UOp = ctx.uop_list[t]
    ctx.buffer_map[original_uop] = replace_uop.shrink_to(original_uop.shape)
  return ret

def replace_input_buffer(ctx:AllocCtx, b:UOp):
  ctx.replacements.append(b)
  return UOp.param(len(ctx.replacements)-1, b.dtype, b.shape, b.device,
                   b._min_max if b.op is Ops.BIND else None, name=b.src[0].expr if b.op is Ops.BIND else None,
                   addrspace=b.addrspace if b.addrspace is not None else AddrSpace.GLOBAL,
                   multiple_of=b.src[0].arg.multiple_of if b.op is Ops.BIND else None)

pm_finalize_call = PatternMatcher([
  (UPat(Ops.AFTER, name="x"), finalize_after),
  (UPat(Ops.COPY, name="x"), lambda ctx,x: ctx.assigns.append(x) if isinstance(x.device, str) and x.device.startswith(("DISK", "TINYFS")) else None),
])

pm_replace_buf = PatternMatcher([
  # replace BUFFER with PARAM for cache key normalization
  (UPat(Ops.BUFFER, src=(UPat(),), name="b"), lambda ctx,b:
   replace_input_buffer(ctx, b) if isinstance(b.arg, ParamArg) and b.addrspace is AddrSpace.GLOBAL else None),
  # replace SLICE with PARAM. this rewrite is bottom up so BUFFERs we don't need won't be in the input
  (UPat(Ops.SLICE, src=(UPat(Ops.BUFFER), UPat(Ops.CONST, dtype=dtypes.weakint)), name="b"), replace_input_buffer),
  # strip value from BIND for cache key normalization, so different values hit same cache
  (UPat(Ops.BIND, src=(UPat(Ops.PARAM), UPat(Ops.CONST)), name="b"), replace_input_buffer),
])

@track_rewrites(lambda _,ret: f"Callify {pluralize('Buffer', len(ret[1]))}")
def transform_to_call(big_sink:UOp) -> tuple[UOp, dict[UOp, UOp]]:
  if VIZ: graph_rewrite(big_sink, PatternMatcher([]), name="View Tensor Graph")
  # uop list is a list in the original_sink graph and we can map to the tags later
  # same predicate as Tensor.realize
  ctx = AllocCtx(bases={base for x in big_sink.src if (base:=x.base).device is not None and not base.has_buffer_identity()
                        and base.op is not Ops.AFTER and base.addrspace is not AddrSpace.ALU})

  # this rewrite is "read-only", it adds simple things to buffer_map and may sink things on big_sink, bottom_up
  # this is the only one where we have to be careful to not break the tensor graph
  big_sink = graph_rewrite(big_sink, add_tags, ctx=ctx, bottom_up=True, name="number the uops")

  # here we can break the tensor graph. this is the only place you need to maintain numbered tags
  big_sink = graph_rewrite(big_sink, pm_early_transform_tensor_graph, name="early transform tensor graph")

  # here we construct the final buffer_map: as-built nodes -> their final storage. values are never keys
  graph_rewrite(big_sink, pm_finalize_call, ctx=ctx, name="finalize call")
  ret = graph_rewrite(UOp.sink(*ctx.assigns), pm_replace_buf, ctx=ctx, bottom_up=True, name="replace bufs").call(*ctx.replacements)
  assert not any(x in ctx.buffer_map for x in ctx.buffer_map.values())
  if VIZ: graph_rewrite(ret, PatternMatcher([]), name="View Call")
  return ret, ctx.buffer_map
