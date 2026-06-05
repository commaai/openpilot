from dataclasses import dataclass, field
from tinygrad.uop.ops import UOp, UPat, PatternMatcher, Ops, GroupOp, graph_rewrite, track_rewrites
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

def disk_copy_is_buffer(ctx:AllocCtx, u:UOp):
  # copies to disk are replaced with the disk buffer
  to_disk = isinstance(u.device, str) and u.device.startswith(("DISK", "TINYFS"))
  if to_disk: ctx.buffer_map[u] = u.empty_like()
  # all copies from disk/numpy are realized into a real buffer
  from_creation = isinstance(u.src[0].device, str) and any(u.src[0].device.startswith(x) for x in ["NPY", "DISK", "PYTHON", "TINYFS"])
  if from_creation: return tag_uop(ctx, u)

def apply_after(ctx:AllocCtx, u:UOp):
  base = u.src[0]
  while base.op is Ops.AFTER: base = base.src[0]
  ctx.buffer_map[u] = base

# CONTIGUOUS and AFTER+STORE + parents are the only nodes that get updated
add_tags = PatternMatcher([
  (UPat(Ops.COPY, name="u"), disk_copy_is_buffer),
  # no tag on copies that are assigned via STORE+AFTER — merge COPY tag into AFTER
  (UPat(Ops.AFTER, src=(UPat(), UPat(Ops.STORE, src=(UPat(name="dest"), UPat(Ops.COPY, name="c")))), name="a"),
   lambda a,c,dest: a.replace(src=(a.src[0], a.src[1].replace(src=(dest, c.rtag(())))), tag=a.tag+c.tag) if a.tag and c.tag else None),
  (UPat(Ops.AFTER, src=(UPat(), UPat(Ops.STORE)), name="x"), tag_uop),
  (UPat(Ops.AFTER, name="u"), apply_after),
  (UPat(Ops.CONTIGUOUS, name="x"), tag_uop),
  (UPat(GroupOp.All, name="x"), lambda ctx,x: tag_uop(ctx,x) if x in ctx.bases else None),
])

def replace_contig_with_store_after(u:UOp):
  # can't allocate a buffer without a device (e.g., inside a CALL function body with only PARAMs)
  if u.device is None: return None
  # if size is 0, remove the contig
  if 0 in u.shape: return u.src[0]
  # no real contig for DISK/TINYFS tensors, they are left alone
  if isinstance(u.device, str) and u.device.startswith(("DISK", "TINYFS")): return u.rtag(None)
  buf = u.empty_like()
  return buf.after(buf.store(u.src[0])).rtag(u.tag)

def replace_store_after_with_contig(u:UOp, src:UOp):
  assigned_to = u
  while assigned_to.op in {Ops.BITCAST, Ops.AFTER}: assigned_to = assigned_to.src[0].base
  if assigned_to.op is not Ops.BUFFER: return src.contiguous(tag=u.tag)

def _make_buffer_view(src:UOp) -> UOp|None:
  """If movement ops on src collapse to a contiguous range, return BUFFER_VIEW.reshape(src.shape). Otherwise None."""
  if (offset := src.contiguous_view_offset()) is None: return None
  buf = src.base
  if buf.op is Ops.BUFFER_VIEW: offset, buf = offset + buf.arg[1], buf.src[0]
  return UOp(Ops.BUFFER_VIEW, src.dtype, (buf,), (src.numel(), offset)).reshape(src.shape)

def contiguous_mops_to_view(c:UOp, src:UOp):
  """CONTIGUOUS(MOPS(BUFFER)) → CONTIGUOUS(BUFFER_VIEW) when movement ops collapse to a contiguous range."""
  buf = src.base
  if buf.op not in {Ops.BUFFER, Ops.BUFFER_VIEW}: return None
  if src.op is Ops.RESHAPE and src.src[0].op in {Ops.BUFFER, Ops.BUFFER_VIEW}: return None

  # no symbolic shape
  if not all_int(c.shape): return None

  # check if view is supported
  from tinygrad.device import Device
  if isinstance(c.device, str):
    if not hasattr(Device[c.device].allocator, "_offset"): return None
  elif not all(hasattr(Device[d].allocator, "_offset") for d in c.device): return None

  x = src
  while x.op in GroupOp.Movement: x = x.src[0]
  # NOTE: this contiguous is removed because this BUFFER_VIEW/RESHAPE has_buffer_identity
  if x.op is not Ops.MULTI and (view := _make_buffer_view(src)) is not None:
    return view.contiguous(tag=c.tag)

  # for MULTI tensors, use multi_pm to resolve per-shard movement ops, then create BUFFER_VIEW on the resolved result
  if not isinstance(c.device, str):
    from tinygrad.schedule.multi import multi_pm
    resolved = graph_rewrite(src, multi_pm, name="multi_buffer_view")
    if resolved.op is not Ops.MULTI: return None
    if (view := _make_buffer_view(resolved.src[0])) is None: return None
    return view.multi(resolved.arg).contiguous(tag=c.tag)

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
    base = s.base
    if base.op in {Ops.CONTIGUOUS, Ops.BUFFER} and base.shape == t.shape and base not in subs:
      subs[base] = t.after(t.store(base.src[0])) if base.op is Ops.CONTIGUOUS else t
      items.append(s.after(*after_deps) if after_deps else s)
    else:
      items.append(t.after(t.store(s), *after_deps))
  fxn = UOp.sink(*(x.substitute(subs) for x in items))

  # body switches from TUPLE to SINK, so the node becomes an opaque CALL (not FUNCTION)
  new_call = UOp(Ops.CALL, c.dtype, (fxn, *input_buffers, *outs), c.arg)
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

  # CONTIGUOUS(MOPS(BUFFER/BUFFER_VIEW)) → CONTIGUOUS(BUFFER_VIEW) when movement ops collapse to contiguous range
  (UPat(Ops.CONTIGUOUS, src=(UPat(GroupOp.Movement, name="src"),), name="c"), contiguous_mops_to_view),

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
                   b._min_max if b.op is Ops.BIND else None, b.src[0].arg[0] if b.op is Ops.BIND else None)

pm_finalize_call = PatternMatcher([
  (UPat(Ops.AFTER, name="x"), finalize_after),
  (UPat(Ops.COPY, name="x"), lambda ctx,x: ctx.assigns.append(x) if isinstance(x.device, str) and x.device.startswith(("DISK", "TINYFS")) else None),
  # remove unique from const. TODO: this is copied in function.py
  (UPat(Ops.CONST, src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE, name="d")), name="b"), lambda b,d: b.replace(src=(d,))),
])

pm_replace_buf = PatternMatcher([
  # replace BUFFER with PARAM for cache key normalization
  (UPat(Ops.BUFFER, src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE)), name="b"), replace_input_buffer),
  # replace BUFFER_VIEW with PARAM. this rewrite is bottom up so BUFFERs we don't need won't be in the input
  (UPat(Ops.BUFFER_VIEW, src=(UPat(Ops.BUFFER),), name="b"), replace_input_buffer),
  # strip value from BIND for cache key normalization, so different values hit same cache
  (UPat(Ops.BIND, src=(UPat(Ops.DEFINE_VAR), UPat(Ops.CONST)), name="b"), replace_input_buffer),
])

@track_rewrites(lambda _,ret: f"Callify {pluralize('Buffer', len(ret[1]))}")
def transform_to_call(big_sink:UOp) -> tuple[UOp, dict[UOp, UOp]]:
  if VIZ: graph_rewrite(big_sink, PatternMatcher([]), name="View Tensor Graph")
  # uop list is a list in the original_sink graph and we can map to the tags later
  # here we build buffer map
  dont_realize = {Ops.CONST, Ops.BUFFER, Ops.BIND, Ops.DEFINE_VAR, Ops.AFTER}
  ctx = AllocCtx(bases=set([x.multibase for x in big_sink.src if x.base.op not in dont_realize]))

  # this rewrite is "read-only", it adds simple things to buffer_map and may sink things on big_sink, bottom_up
  # this is the only one where we have to be careful to not break the tensor graph
  big_sink = graph_rewrite(big_sink, add_tags, ctx=ctx, bottom_up=True, name="number the uops")

  # here we can break the tensor graph. this is the only place you need to maintain numbered tags
  big_sink = graph_rewrite(big_sink, pm_early_transform_tensor_graph, name="early transform tensor graph")

  # here we construct the final buffer_map. this is everything that will go into the tensor map
  graph_rewrite(big_sink, pm_finalize_call, ctx=ctx, name="finalize call")
  ret = graph_rewrite(UOp.sink(*ctx.assigns), pm_replace_buf, ctx=ctx, bottom_up=True, name="replace bufs").call(*ctx.replacements)
  if VIZ: graph_rewrite(ret, PatternMatcher([]), name="View Call")
  return ret, ctx.buffer_map
