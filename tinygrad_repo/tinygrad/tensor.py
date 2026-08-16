# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from __future__ import annotations
import time, functools, sys, inspect, pathlib, hashlib, weakref
from dataclasses import dataclass, field
from typing import Any, Callable, cast, get_args, ParamSpec, TypeGuard, TypeVar, Generic, TYPE_CHECKING
if TYPE_CHECKING: import numpy
from tinygrad.dtype import DType, DTypeLike, dtypes, ConstType, least_upper_dtype, to_dtype, strong_dtype, \
  _from_np_dtype, _to_np_dtype, PyConst, AddrSpace
from tinygrad.helpers import all_int, getenv, fetch, Metadata, TRACEMETA, TracingKey
from tinygrad.helpers import cpu_profile, suppress_finalizing, disable_gc, VIZ, pluralize
from tinygrad.uop.ops import UOp, Ops, sint, all_metadata, Variable, ConstLike, UPat, PatternMatcher, GroupOp, ParamArg, graph_rewrite, rewrite_group
from tinygrad.mixin.rand import RandMixin
from tinygrad.schedule import create_linear_with_vars
from tinygrad.device import Buffer, canonicalize_device
from tinygrad.engine.realize import run_linear

# *** callify: transform a tensor graph into a CALL UOp such that all state is properly scoped ***

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
  # can't allocate a buffer for a virtual value
  if u.is_virtual: return None
  # if size is 0, remove the contig
  if 0 in u.shape: return u.src[0]
  # no real contig for DISK/TINYFS tensors, they are left alone
  if disk_like(u): return u.rtag(None)
  buf = u.empty_like()
  return buf.after(buf.store(u.src[0])).rtag(u.tag)

def replace_store_after_with_contig(u:UOp, src:UOp):
  assigned_to = u
  while assigned_to.op in {Ops.BITCAST, Ops.AFTER, Ops.UNSHARD}: assigned_to = assigned_to.src[0].base
  if assigned_to.op not in {Ops.BUFFER, Ops.SLICE}: return src.contiguous(tag=u.tag)

def _make_buffer_view(src:UOp) -> UOp|None:
  """If movement ops on src collapse to a contiguous range, return SLICE. Otherwise None."""
  if (offset := src.contiguous_view_offset()) is None: return None
  buf = src.base
  if buf.op is Ops.SLICE:
    byte_offset = buf.src[1].val * buf.src[0].dtype.itemsize + offset * src.dtype.itemsize
    buf = buf.src[0]
    if byte_offset % buf.dtype.itemsize != 0: return None
    offset = byte_offset // buf.dtype.itemsize
  return UOp(Ops.SLICE, src.dtype, (buf, UOp.const(offset)), src.numel())

def contiguous_mops_to_view(c:UOp, src:UOp):
  """MOPS(BUFFER) → SLICE when movement ops collapse to a contiguous range."""
  buf = src.base
  if buf.op not in {Ops.BUFFER, Ops.SLICE, Ops.UNSHARD}: return None
  if src.op is Ops.RESHAPE and src.src[0].op in {Ops.BUFFER, Ops.SLICE} and c.op is not Ops.BITCAST: return None
  if c.op is not Ops.BITCAST and src.op is Ops.BUFFER: return None

  # no symbolic shape
  if not all_int(c.shape): return None

  if buf.op is not Ops.UNSHARD and (view := _make_buffer_view(src)) is not None:
    view = (view.replace(dtype=c.dtype, arg=c.numel()) if c.op is Ops.BITCAST else view).reshape(c.shape)
    return c.replace(src=(view,)) if c.op is Ops.COPY else view

  # for UNSHARD tensors, use multi_pm to resolve per-shard movement ops, then create SLICE on the resolved result
  if not isinstance(c.device, str):
    from tinygrad.schedule.multi import multi_pm
    resolved = graph_rewrite(src, multi_pm, name="multi_buffer_view")
    if resolved.op is not Ops.UNSHARD: return None
    if (view := _make_buffer_view(resolved.src[0])) is None: return None
    return view.reshape(resolved.src[0].shape).unshard(resolved.arg, resolved.src[1:]).contiguous(tag=c.tag)

  return None

def _precompiled_output_redirect(s:UOp, t:UOp) -> UOp|None:
  # how output s lands in the caller's buffer t, or None if it must be copied into t
  # materialize straight into t
  if s.op is Ops.CONTIGUOUS: return t.after(t.store(s.src[0]))
  # rebind output storage to t
  if s.op in {Ops.BUFFER, Ops.UNSHARD} and s.has_buffer_identity(): return t
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
  (UPat(GroupOp.Movement-{Ops.SHRINK, Ops.RESHAPE}, name="x").f(Ops.CONTIGUOUS).f(Ops.COPY, name="copy"), lambda x,copy:
   copy.replace(src=(x,), tag=None) if isinstance(x.device, str) and x.device.startswith("DISK") else None),
  # push copy past movement ops to disk
  (UPat(GroupOp.Movement-{Ops.SHRINK, Ops.RESHAPE}, name="x").f(Ops.COPY, name="copy"), lambda x,copy:
   x.replace(src=(copy.replace(src=(x.src[0],), tag=None),)+x.src[1:]) \
   if isinstance(x.device, str) and x.device.startswith("DISK") else None),

  # add CONTIGUOUS to tagged UOps
  (UPat(GroupOp.All-{Ops.CONTIGUOUS, Ops.AFTER, Ops.STORE}, name="x"),
   lambda x: None if x.tag is None else x.rtag(None).contiguous(tag=x.tag) if x.tag else x.replace(tag=None)),
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
  # then, add views back
  views:list[UOp] = []
  while replace_uop.op in GroupOp.Movement|{Ops.UNSHARD, Ops.BITCAST, Ops.AFTER}:
    if replace_uop.op is not Ops.AFTER: views.append(replace_uop)
    replace_uop = replace_uop.src[0]
  for v in reversed(views): replace_uop = v.replace(src=(replace_uop,)+v.src[1:])
  for t in x.tag:
    original_uop: UOp = ctx.uop_list[t]
    ctx.buffer_map[original_uop] = replace_uop.shrink_to(original_uop.shape)
  return ret

def replace_input_buffer(ctx:AllocCtx, b:UOp):
  ctx.replacements.append(b)
  if b.op is Ops.BIND: return b.param_like(len(ctx.replacements)-1)
  return UOp.param(len(ctx.replacements)-1, b.dtype, b.shape, b.device,
                   addrspace=b.addrspace if b.addrspace is not None else AddrSpace.GLOBAL)

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

@rewrite_group(lambda _,ret: f"Callify {pluralize('Buffer', len(ret[1]))}")
def transform_to_call(big_sink:UOp) -> tuple[UOp, dict[UOp, UOp]]:
  if VIZ: graph_rewrite(big_sink, PatternMatcher([]), name="View Tensor Graph")
  # uop list is a list in the original_sink graph and we can map to the tags later
  # same predicate as Tensor.realize
  ctx = AllocCtx(bases={base for x in big_sink.src if not (base:=x.base).is_virtual and not base.has_buffer_identity()
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

# *** all in scope Tensors are here. this gets relevant UOps ***

all_tensors: dict[weakref.ref[Tensor], None] = {}
def _apply_map_to_tensors(applied_map:dict[UOp, UOp], name:str) -> None:
  with cpu_profile(TracingKey(name), "TINY"):
    # get tensors in scope
    in_scope: dict[UOp, bool] = {}
    def visitor(node: UOp) -> bool: return True if node in applied_map else any(in_scope.get(s, False) for s in node.src)
    scope_tensors: list[Tensor] = [t for tref in list(all_tensors) if (t:=tref()) is not None and t.uop.topovisit(visitor, in_scope)]

    # get all Tensors and apply the map. always walk: replace exactly the nodes the map names, values are final
    sink = UOp.sink(*[t.uop for t in scope_tensors])
    new_sink = sink.substitute(applied_map, name=f"substitute {name}", walk=True)

    # set the relevant uop to the realized UOps
    for t,s,ns in zip(scope_tensors, sink.src, new_sink.src):
      if s is ns: continue
      t.uop = ns

# **** Tensor helper functions ****

def is_numpy_ndarray(x) -> "TypeGuard[numpy.ndarray]": return str(type(x)) == "<class 'numpy.ndarray'>"

def _fromnp(x: 'numpy.ndarray') -> UOp:
  ret = UOp.new_buffer("NPY", x.size, _from_np_dtype(x.dtype))
  # fake realize
  ret.buffer.allocate(x)
  return ret.reshape(x.shape)

class Tensor(RandMixin):
  """
  A `Tensor` is a multi-dimensional matrix containing elements of a single data type.

  ```python exec="true" session="tensor"
  from tinygrad import Tensor, dtypes, nn, Context
  import numpy as np
  import math
  np.set_printoptions(precision=4)
  ```
  """
  __slots__ = "uop", "is_param", "grad"

  def __init__(self, data:ConstType|bytes|list|tuple|UOp|'numpy.ndarray'|pathlib.Path|None,
               device:str|tuple|list|None=None, dtype:DTypeLike|None=None):
    if device is None:
      if isinstance(data, pathlib.Path): device = f"DISK:{data.resolve()}"  # keep it on the disk if device is None
      elif isinstance(data, UOp): device = data.device
    _dtype:DType|None = to_dtype(dtype) if dtype is not None else None
    _device:str|tuple[str, ...] = canonicalize_device(device)
    del device, dtype

    # tensors can have gradients if you have called .backward
    self.grad:Tensor|None = None

    self.is_param:bool = True

    # create a UOp from the different types of inputs
    if data is None:
      data = UOp.const(0.0, _dtype)
    elif isinstance(data, get_args(ConstType)):
      data = UOp.const(data, _dtype)
    elif is_numpy_ndarray(data) and data.shape == ():
      data = UOp.const(data.item(), _dtype or _from_np_dtype(data.dtype))
    elif not isinstance(data, UOp):
      if _dtype in dtypes.weaks: raise RuntimeError(f"cannot create storage for weak dtype {_dtype}")
      if isinstance(data, bytes):
        data = UOp._frompy(data, _dtype or dtypes.uint8, _device)
      elif isinstance(data, (list, tuple)):
        data = UOp._frompy(data, _dtype or dtypes.from_py(data), _device)
      elif is_numpy_ndarray(data):
        data = _fromnp(data.astype(npdtype) if _dtype is not None and (npdtype:=_to_np_dtype(_dtype)) is not None else data)
      elif isinstance(data, pathlib.Path):
        _dtype = _dtype or dtypes.uint8
        data = UOp.new_buffer(f"DISK:{data.resolve()}", data.stat().st_size // _dtype.itemsize, _dtype)

    # by this point, it has to be a UOp
    if not isinstance(data, UOp): raise RuntimeError(f"can't create Tensor from {data!r} with type {type(data)}")

    # data might be on a different device
    self.uop:UOp = data if data.device is None or data.device == _device else data.copy_to_device(_device)
    # cast on the target device, the source may not hold the dtype (numpy has no fp8/bfloat16) or be able to compute it (DISK)
    if _dtype is not None: self.uop = self.uop.cast(_dtype)

    # add to all_tensors after construction succeeds
    all_tensors[weakref.ref(self)] = None

  @suppress_finalizing
  def __del__(self): all_tensors.pop(weakref.ref(self), None)

  def _apply_uop(self, fxn:Callable[..., UOp], *x:Tensor, **kwargs) -> Tensor:
    srcs = (self,)+x
    new_uop: UOp = fxn(*[t.uop for t in srcs], **kwargs)
    if TRACEMETA >= 1 and (metadata:=_METADATA.get()) is not None: all_metadata[new_uop] = (metadata,)
    # directly create the Tensor
    ret = Tensor.__new__(Tensor)
    ret.uop, ret.grad, ret.is_param = new_uop, None, True
    # add to all_tensors after construction succeeds
    all_tensors[weakref.ref(ret)] = None
    return ret

  # alu, _uop, _wrap_uop and const are used by the mixins
  def alu(self, op: Ops, *src: Tensor) -> Tensor: return self._apply_uop(lambda *u: u[0].alu(op, *u[1:]), *src)
  @property
  def _uop(self) -> UOp: return self.uop
  @classmethod
  def _wrap_uop(cls, u:UOp) -> Tensor: return cls(u)
  @staticmethod
  def const(b:ConstLike, dtype:DType|None=None) -> Tensor: return Tensor(UOp.const(b, dtype))

  def is_param_(self, is_param:bool=True) -> Tensor:
    self.is_param = is_param
    return self

  def __repr__(self):
    ld = self.uop
    ld_repr = f"<UOp {ld.device} {ld.shape} {str(ld.dtype)[7:]}>"
    return f"<Tensor {ld_repr} on {self.device} with grad {(self.grad.uop if self.grad is not None else None)!r}>"

  # Python has a non moving GC, so this should be okay
  def __hash__(self): return id(self)

  def __bool__(self): raise TypeError("__bool__ on Tensor is not defined")

  def __len__(self):
    if not self.shape: raise TypeError("len() of a 0-d tensor")
    return self.shape[0]

  @property
  def device(self) -> str|tuple[str, ...]|None: return self.uop.device

  @property
  def shape(self) -> tuple[sint, ...]: return self.uop.shape

  @property
  def dtype(self) -> DType: return self.uop.dtype

  # ***** data handlers ****

  def as_param(self, slot:int):
    return Tensor(UOp.param(slot, self.dtype, self.uop.shard_shape, self.device, axis=self.uop.axis))

  def call(self, *lst:Tensor, fxn:Tensor|UOp, grad_fxn:Callable|None=None) -> Tensor:
    fret = fxn._uop.call(*[t.uop for t in (self,)+lst], grad_fxn=grad_fxn)
    return Tensor(fret.gettuple(0))

  def custom_kernel(self, *lst:Tensor, fxn:Callable, grad_fxn:Callable|None=None) -> list[Tensor]:
    """
    Call into a custom kernel written in UOps. Returns the Tensors after the Kernel has been applied.

    This API is alpha and may change.
    """
    return [Tensor(u) for u in UOp.custom_kernel(*[t.uop for t in (self,)+lst], fxn=fxn, grad_fxn=grad_fxn)]

  def callify(self, *lst:Tensor) -> Tensor:
    big_sink = UOp.sink(*[x.uop for x in (self,)+lst])
    big_sink, buffer_map = transform_to_call(big_sink)
    _apply_map_to_tensors({x:y.after(big_sink) for x,y in buffer_map.items()}, name="callify")
    return self

  def linear_with_vars(self, *lst:Tensor) -> tuple[UOp, dict[str, int]]:
    """Creates the LINEAR UOp needed to realize these Tensor(s), with Variables."""
    # weakness ends where storage begins
    if any(t.dtype in dtypes.weaks and t.uop.device is not None for t in (self,)+lst):
      raise RuntimeError("cannot realize a weak dtype; cast to a concrete dtype first")
    big_sink, becomes_map = transform_to_call(UOp.sink(*[x.uop for x in (self,)+lst]))
    _apply_map_to_tensors(becomes_map, name="buffers")
    return create_linear_with_vars(big_sink)

  def schedule_linear(self, *lst:Tensor) -> UOp:
    """Creates the schedule needed to realize these Tensor(s)."""
    linear, var_vals = self.linear_with_vars(*lst)
    assert len(var_vals) == 0
    return linear

  @disable_gc()
  def realize(self, *lst:Tensor, do_update_stats=True) -> Tensor:
    """Triggers the computation needed to create these Tensor(s)."""
    to_realize = [x for x in (self,)+lst if not x.uop.is_virtual and not x.uop.has_buffer_identity()]
    if len(to_realize):
      run_linear(*Tensor.linear_with_vars(*to_realize), update_stats=do_update_stats)
    return self

  def replace(self, x:Tensor) -> Tensor:
    """
    Replaces the data of this tensor with the data of another tensor. Only the shape of the tensors must match.
    """
    # used for replacing a Tensor with a new version of it (potentially with a different device and dtype)
    assert self.shape == x.shape, f"replace shape mismatch {self.shape} != {x.shape}"
    self.uop = x.uop
    return self

  def assign(self, x:Tensor|PyConst|list|tuple) -> Tensor:
    if self.dtype in dtypes.weaks: self.uop = self.uop.clone()
    is_disk = isinstance(self.device, str) and self.device.startswith(("DISK", "TINYFS"))
    if not isinstance(x, Tensor): x = Tensor(x, device="CPU" if is_disk else self.device, dtype=self.dtype)
    if self.uop is x.uop: return self  # a self assign is a NOOP
    # broadcast x (shape only, dtype must match)
    x = x._broadcast_to(self.shape)
    if x.dtype in dtypes.weaks: x = x.cast(least_upper_dtype(self.dtype, x.dtype))
    if x.dtype != self.dtype: raise RuntimeError(f"assign dtype mismatch {self.dtype} != {x.dtype}")
    if not is_disk and x.uop.device is not None and self.device is not None and self.device != x.device:
      raise RuntimeError(f"assign device mismatch {self.device} != {x.device}")
    if isinstance(self.device, tuple) and x.uop.device is not None and self.uop.axis != x.uop.axis:
      raise RuntimeError(f"multi axis mismatch {self.uop.axis} != {x.uop.axis}")

    # TODO: this is a hack for writing to DISK. remove with working assign
    if is_disk:
      (b:=self._buffer()).copy_from(Buffer("PYTHON", b.size, b.dtype, opaque=x._data()))
      return self
    # STORE+AFTER: STORE is the write effect (void), AFTER wraps the view for correct shape/ranging
    assign = self.uop.after(self.uop.store(x.uop))
    if (base := self.uop.base).op in {Ops.BUFFER, Ops.AFTER} and self.uop is not base and not self.uop.has_buffer_identity():
      # view assign: replace at the buffer-identity level (e.g. RESHAPE(BUFFER)) so @function's substitution catches it
      ib = self.uop
      while not ib.has_buffer_identity() and ib is not base: ib = ib.src[0]
      assigned_ib = ib.after(assign)
      _apply_map_to_tensors({ib: assigned_ib}, name="Embed View Assign")
    else:
      # simple assign
      self.uop = assign
    return self

  def _buffer(self) -> Buffer:
    from tinygrad.engine.realize import capturing
    if capturing and not getenv("UNSAFE_ALLOW_JIT_BUFFER"):
      from tinygrad.engine.jit import JitError
      raise JitError("cannot access tensor data during JIT capture, the value will be baked in")
    x = self.contiguous()
    if self.uop.device is None or isinstance(self.device, tuple): x = x.clone("CPU")
    return cast(Buffer, x.realize().uop.buffer).ensure_allocated()

  def _data(self) -> memoryview: return self._buffer().as_memoryview()

  def data(self) -> memoryview:
    """
    Returns the data of this tensor as a memoryview.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(np.frombuffer(t.data(), dtype=np.int32))
    ```
    """
    if self.dtype in dtypes.weaks: return self.cast(strong_dtype(self.dtype)).data()
    if 0 in self.shape: return memoryview(bytearray(0)).cast(self.dtype.fmt)  # type: ignore[arg-type,return-value]
    assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
    buf = self._buffer()
    fmt = buf.dtype.fmt
    assert fmt is not None, f"no fmt dtype for {buf.dtype}"
    assert fmt != "e" or sys.version_info >= (3, 12)
    return buf.as_memoryview().cast(fmt, self.shape)  # type: ignore[arg-type,return-value]

  # NOTE: list[Any] because return type is recursive (list[list[...]] for higher dimensions)
  def tolist(self) -> PyConst|list[Any]:
    """
    Returns the value of this tensor as a nested list.
    Returns single value for const tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(t.tolist())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor(5)
    print(t.tolist())
    ```
    """
    # TODO: remove half once minimum python supports it
    if self.dtype in (dtypes.half, dtypes.bfloat16, *dtypes.fp8s): return self.cast(dtypes.float32).tolist()
    if 0 in self.shape:
      assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
      def _tolist(shape:tuple[int, ...]): return [_tolist(shape[1:]) for _ in range(shape[0])]
      return _tolist(self.shape)
    return self.data().tolist()

  def numpy(self) -> 'numpy.ndarray':
    """
    Returns the value of this tensor as a `numpy.ndarray`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(repr(t.numpy()))
    ```
    """
    if self.dtype in dtypes.weaks: return self.cast(strong_dtype(self.dtype)).numpy()
    assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
    import numpy as np
    if self.dtype in { dtypes.bfloat16, *dtypes.fp8s }: return self.float().numpy()
    if 0 in self.shape: return np.empty(self.shape, dtype=_to_np_dtype(self.dtype))
    return self._buffer().numpy().reshape(self.shape)

  def clone(self, device:str|tuple[str, ...]|None=None) -> Tensor:
    """
    Creates a clone of this tensor allocating a separate buffer for the data.
    If `device` is specified, the clone is placed on that device.
    """
    ret = Tensor(self.uop.clone(device=device))
    if self.grad is not None: ret.grad = self.grad.clone(device=device)
    return ret.is_param_(self.is_param)

  def to(self, device:str|tuple[str, ...]|None) -> Tensor:
    """
    Moves the tensor to the given device.
    """
    if self.uop.device is None: return self
    if (device:=canonicalize_device(device)) == self.device: return self
    ret = Tensor(self.uop.copy_to_device(device))
    if self.grad is not None: ret.grad = self.grad.to(device)
    return ret.is_param_(self.is_param)

  def to_(self, device:str|tuple[str, ...]|None) -> Tensor:
    """
    Moves the tensor to the given device in place.
    """
    real = self.to(device)
    if self.grad is not None and real.grad is not None: self.grad.replace(real.grad)
    return self.replace(real)

  def shard(self, devices:tuple[str, ...], axis:int|None=None) -> Tensor:
    """
    Shards the tensor across the given devices. Optionally specify which axis to shard on.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.empty(2, 4)
    print(t.shard((t.device, t.device), axis=1).uop)
    ```
    """
    if self.uop.device is None: return self
    if not isinstance(self.device, str): raise RuntimeError("can't shard a multi-device tensor")
    if len(devices) == 1: return self.to(devices[0])
    devices = cast(tuple[str, ...], canonicalize_device(devices))
    uop = self.uop.shard(devices, None if axis is None else self._resolve_dim(axis))
    return Tensor(uop).is_param_(self.is_param)

  def shard_(self, devices:tuple[str, ...], axis:int|None=None) -> Tensor:
    """
    Shards the tensor across the given devices in place.
    """
    return self.replace(self.shard(devices, axis))

  def shard_like(self, y:Tensor) -> Tensor:
    """
    Shards the tensor the same way as `y` (same devices and axis).
    """
    if y.device is None: return self
    if isinstance(y.device, str): return self.to(y.device)
    return self if isinstance(self.device, tuple) and (y.device, y.uop.axis) == (self.device, self.uop.axis) else self.shard(y.device, y.uop.axis)

  # ***** creation entrypoint *****

  @staticmethod
  def from_blob(ptr:int, shape:tuple[int, ...], **kwargs) -> Tensor:
    """
    Exposes the pointer as a Tensor without taking ownership of the original data.
    The pointer must remain valid for the entire lifetime of the created Tensor.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.
    """
    r = Tensor.empty(*shape, **kwargs)
    assert isinstance(r.device, str)
    cast(Buffer, r.uop.buffer).allocate(external_ptr=ptr)
    return r

  @staticmethod
  def from_url(url:str, gunzip:bool=False, **kwargs) -> Tensor:
    """
    Creates a Tensor from a URL.

    This is the preferred way to access Internet resources.
    It currently returns a DISK Tensor, but in the future it may return an HTTP Tensor.
    This also will soon become lazy (when possible) and not print progress without DEBUG.

    The `gunzip` flag will gzip extract the resource and return an extracted Tensor.
    """
    return Tensor(fetch(url, gunzip=gunzip), **kwargs)

  _seed: int = int(time.time())
  _device_seeds: dict[str, Tensor] = {}
  _device_rng_counters: dict[str, Tensor] = {}
  @staticmethod
  def manual_seed(seed=0) -> None:
    """
    Sets the seed for random operations.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.rand(5).numpy())
    print(Tensor.rand(5).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)  # reset to the same seed
    print(Tensor.rand(5).numpy())
    print(Tensor.rand(5).numpy())
    ```
    """
    Tensor._seed, Tensor._device_seeds, Tensor._device_rng_counters = seed, {}, {}

  @staticmethod
  def _next_counter(device:str, num:int) -> tuple[Tensor, Tensor]:
    if device not in Tensor._device_seeds:
      seed = [int.from_bytes(hashlib.sha256(len(Tensor._device_seeds).to_bytes(4, "big")).digest(), "big"), Tensor._seed]
      Tensor._device_seeds[device] = Tensor(seed, device=device, dtype=dtypes.uint32)
      Tensor._device_rng_counters[device] = Tensor([0, 0], device=device, dtype=dtypes.uint32)
    counter = Tensor._device_rng_counters[device]
    new_low = counter[0:1] + (num & 0xffffffff)
    new_high = counter[1:2] + (num >> 32) + (new_low < counter[0])
    counter.assign(new_low.cat(new_high))
    low = counter[0:1] - (num & 0xffffffff)
    high = counter[1:2] - (num >> 32) - (counter[0] < (num & 0xffffffff))
    return Tensor._device_seeds[device], low.cat(high)

  # ***** toposort and backward pass *****

  def backward(self, gradient:Tensor|None=None) -> Tensor:
    """
    Propagates the gradient of a tensor backwards through the computation graph.
    If the 'gradient' argument is not provided, the tensor must be a scalar, and the gradient is implicitly set to 1.0.
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1.0, 2.0, 3.0, 4.0])
    t.sum().backward()
    print(t.grad.numpy())
    ```
    """
    all_uops = self.uop.toposort()
    # backward fills .grad for every in-scope non-CONST float tensor
    tensors_need_grad: list[Tensor] = [t for tref in all_tensors if (t:=tref()) is not None and \
                                       t.uop in all_uops and t.is_floating_point() and t.uop.op is not Ops.CONST]
    # clear contexts
    for t,g in zip(tensors_need_grad, self.gradient(*tensors_need_grad, gradient=gradient)):
      assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
      if g.device is None and t.device is not None: g = g.clone(device=t.device)
      if t.grad is None: t.grad = g
      else: t.grad.assign(t.grad + g.to(t.grad.device))
    return self

  # ***** movement ops *****

  def _mop(self, op:Ops, arg) -> Tensor: return self._apply_uop(UOp._mop, op=op, arg=arg)
  def _rop(self, op:Ops, axis:tuple[int, ...]) -> Tensor: return self._apply_uop(UOp._rop, op=op, axis=axis)

  def __setitem__(self, indices, v:Tensor|PyConst|list|tuple) -> None:
    if self.dtype in dtypes.weaks: raise RuntimeError("cannot setitem into a weak tensor; it has no storage")
    if isinstance(v, Tensor):
      if v.dtype in dtypes.weaks: v = v.cast(least_upper_dtype(self.dtype, v.dtype))
      if v.dtype != self.dtype: raise RuntimeError(f"setitem dtype mismatch: {self.dtype=} != {v.dtype=}")
    # raise if mutation would diverge from eager (allow only pure views of a realized buffer; exclude +=/-= RHS via v_uop/v_bw)
    v_uop, v_bw = (v.uop, v.uop.backward_slice) if isinstance(v, Tensor) else (None, {})
    if self.uop.op_in_backward_slice_with_self(Ops.BUFFER):
      shared = self.uop.base if self.uop.base.is_realized else None
      if any(self.uop in t.uop.backward_slice_with_self and t.uop.base is not shared for tref in all_tensors
             if (t:=tref()) is not None and t is not self and t.uop is not v_uop and t.uop not in v_bw):
        raise RuntimeError("can't setitem on a tensor with other uses")
    idx = [indices] if (isinstance(indices, list) and all_int(indices)) or not isinstance(indices, (tuple, list)) else list(indices)
    is_disk = isinstance(self.device, str) and self.device.startswith("DISK")
    advanced = any(isinstance(i, (Tensor, list, tuple)) for i in idx)
    realized = is_disk or self.uop.base.op is Ops.BUFFER or self.uop._base_buffer_is_realized()
    if (not self.uop.base.is_realized and self.is_floating_point()) or not (advanced or realized):
      if not isinstance(v, Tensor): v = Tensor(v, device=self.device, dtype=self.dtype)
      # __iadd__/__isub__ creates AFTER(view, STORE(view, computed)); unwrap to get the computed value
      if v.uop.op is Ops.AFTER and any(s.op is Ops.STORE for s in v.uop.src[1:]): v = v._apply_uop(lambda x: x.src[1].src[1])
      self.replace(self._getitem(indices, v))
    elif advanced: # advanced setitem
      if is_disk: raise RuntimeError("advanced setitem is not supported for DISK tensors")
      if not isinstance(v, Tensor): v = Tensor(v, device=self.device, dtype=self.dtype)
      self.assign(self._getitem(indices, v))
    else: # basic setitem
      view = self[indices]
      if isinstance(v, Tensor) and v.uop.op is Ops.AFTER and v.uop in view.uop.base.src: return
      view.assign(v)

  def __delitem__(self, indices) -> None:
    raise TypeError("Tensor does not support deleting items")

  # ***** op wrappers *****

  # unlike Tensors, UOps are immutable, so these don't go in mixin
  def __iadd__(self, x) -> Tensor: return self.assign(self.add(x)) # type: ignore[misc]
  def __isub__(self, x) -> Tensor: return self.assign(self.sub(x)) # type: ignore[misc]
  def __imul__(self, x) -> Tensor: return self.assign(self.mul(x)) # type: ignore[misc]
  def __itruediv__(self, x) -> Tensor: return self.assign(self.div(x)) # type: ignore[misc]
  def __ifloordiv__(self, x) -> Tensor: return self.assign(self.__floordiv__(x)) # type: ignore[misc]
  def __ipow__(self, x) -> Tensor: return self.assign(self.pow(x)) # type: ignore[misc]
  def __iand__(self, x) -> Tensor: return self.assign(self.bitwise_and(x)) # type: ignore[misc]
  def __ior__(self, x) -> Tensor: return self.assign(self.bitwise_or(x)) # type: ignore[misc]
  def __ixor__(self, x) -> Tensor: return self.assign(self.bitwise_xor(x)) # type: ignore[misc]
  def __ilshift__(self, x) -> Tensor: return self.assign(self.lshift(x)) # type: ignore[misc]
  def __irshift__(self, x) -> Tensor: return self.assign(self.rshift(x)) # type: ignore[misc]
  def __imatmul__(self, x) -> Tensor: return self.assign(self.matmul(x)) # type: ignore[misc]

  def __eq__(self, x) -> Tensor: return self.eq(x)                      # type: ignore[override]

  # ***** encoding/decoding ops *****

  def decode_hevc_frame(self, frame_pos:Variable, shape:tuple[int,...], state:Tensor, ref_frames:list[Tensor]|None=None) -> Tensor:
    """
    Creates a Tensor by decoding an HEVC frame chunk.

    You must provide the output shape of the decoded data (`shape`), the HEVC context (`vstate`), and, if required by the chunk,
    the reference frames (`ref_frames`).
    """
    ref_frames = [x.contiguous() for x in ref_frames or []]
    assert frame_pos.op is Ops.BIND, "frame_pos must be a bound Variable"
    srcs = (out:=Tensor.empty(*shape, device=self.device, dtype=self.dtype), self.contiguous(), state.contiguous(), *ref_frames)
    fn = UOp(Ops.CUSTOM_FUNCTION, src=(frame_pos.src[0], *[UOp.const(s, dtypes.int) for s in shape]), arg="encdec")
    return Tensor(out.uop.after(fn.call(*[s.uop for s in srcs], frame_pos)))

P = ParamSpec("P")
T = TypeVar("T")

# this tracks the tensor.py METADATA, contextvars.ContextVar was switched to this due to thread safety issues
class _ContextVar(Generic[T]):
  def __init__(self, default:T): self.state:T = default
  def get(self) -> T: return self.state
  def set(self, x:T) -> T:
    ret, self.state = self.state, x
    return ret
_METADATA: _ContextVar[Metadata|None] = _ContextVar(default=None)

def _metadata_wrapper(fn: Callable[P, T]) -> Callable[P, T]:
  def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
    if TRACEMETA < 1 or _METADATA.get() is not None: return fn(*args, **kwargs)
    token = _METADATA.set(Metadata(name=fn.__name__))
    with cpu_profile(TracingKey(fn.__name__), "USER"):
      ret = fn(*args, **kwargs)
    _METADATA.set(token)
    return ret
  return _wrapper

if TRACEMETA >= 1:
  for name, fn in inspect.getmembers(Tensor, inspect.isfunction):
    if name in ["__class__", "__del__", "__init__", "__new__", "__repr__", "backward", "sequential", "gradient"]: continue
    setattr(Tensor, name, functools.wraps(fn)(_metadata_wrapper(fn)))
