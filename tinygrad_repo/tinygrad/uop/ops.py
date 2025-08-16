from __future__ import annotations
from typing import Any, Callable, cast, TYPE_CHECKING, Type, Sequence
import sys, time, functools, itertools, math, operator, hashlib, os, types, pickle, pathlib, inspect, weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from tinygrad.uop import Ops, GroupOp
from tinygrad.uop.mathtraits import MathTrait
from tinygrad.dtype import ConstType, ImageDType, dtypes, DType, truncate, PtrDType
from tinygrad.helpers import ContextVar, all_int, prod, getenv, all_same, Context, partition, temp, unwrap, T, argfix, Metadata, flatten
from tinygrad.helpers import PICKLE_BUFFERS, PROFILE, dedup, cdiv, cmod, diskcache_put, to_function_name, cpu_profile, TracingKey
if TYPE_CHECKING:
  from tinygrad.shape.shapetracker import ShapeTracker
  from tinygrad.device import Buffer, MultiBuffer

# https://en.wikipedia.org/wiki/Identity_element
def identity_element(op:Ops, dt:DType) -> ConstType: return dtypes.as_const({Ops.ADD:0, Ops.MUL:1, Ops.MAX:dtypes.min(dt)}[op], dt)

def can_pad(root:UOp, edges:dict[UOp, None]) -> bool:
  return all(u.op not in GroupOp.UnsafePad for u in root.toposort(gate=lambda x:x not in edges))

# With True as the default, this matches the old symbolic behavior
def resolve(x:UOp|bool, default:bool=True):
  if isinstance(x, bool): return x
  assert x.dtype == dtypes.bool, "UOp in resolve must be bool"
  # NOTE: generating the text for the exception is expensive, so we do this
  return bool(sx.vmin) if (sx:=x.simplify()).vmin == sx.vmax else default

# smax/smin are replacements for max/min that preserve symbolic
def _suop(lst, uop_fxn, python_fxn):
  uops, nums = partition(lst, lambda x: isinstance(x, UOp))
  return ssimplify(functools.reduce(uop_fxn, uops + ([python_fxn(nums)] if nums else [])))
def smax(*lst): return _suop(argfix(*lst), UOp.maximum, max)
def smin(*lst): return _suop(argfix(*lst), UOp.minimum, min)
def srender(x) -> str: return x.render() if isinstance(x, UOp) else str(x)

def ssimplify(uop): return uop.ssimplify() if isinstance(uop, UOp) else uop
def sym_infer(uop: UOp|int, var_vals: dict[UOp, int]) -> int: return uop.sym_infer(var_vals) if isinstance(uop, UOp) else uop

# used for UOp and UPat
def pretty_print(x:Any, rep:Callable, srcfn=lambda x: x.src, cache=None, d=0)->str:
  def dfs(x:Any, cache:dict):
    for s in srcfn(x) or []:
      cache.setdefault(s, [len(cache), 0, False])[1] += 1
      if cache[s][1] == 1: dfs(s, cache)
  if cache is None: dfs(x, cache:={})
  if (cx:=cache.setdefault(x, [0,0,False]))[2]: return f"{' '*d} x{cx[0]}"
  cx[2], srcs = True, ('None' if srcfn(x) is None else ''.join(f'\n{pretty_print(s, rep, srcfn, cache, d+2)},' for s in srcfn(x)))
  return f"{' '*d}{f'x{cx[0]}:=' * (cx[1]>1)}{rep(x)}" % srcs

class UOpMetaClass(type):
  ucache:dict[tuple, weakref.ReferenceType[UOp]] = {}
  def __call__(cls, op:Ops, dtype:DType=dtypes.void, src:tuple[UOp,...]=tuple(), arg:Any=None, tag:Any=None,
               metadata:tuple[Metadata,...]|None=None, _buffer:Buffer|None=None):
    if (wret:=UOpMetaClass.ucache.get(key:=(op, dtype, src, arg, tag), None)) is not None and (ret:=wret()) is not None: return ret
    UOpMetaClass.ucache[key] = ref = weakref.ref(created:=super().__call__(*key))
    for s in src: s.children.add(ref)
    if metadata is not None: all_metadata[created] = metadata
    # NOTE: this value is set by pickle when pickling a realized tensor
    if _buffer is not None:
      assert op is Ops.BUFFER, f"trying to set Buffer {_buffer} for {op}"
      buffers[created] = _buffer
    return created

# some uops map to other stuff
buffers:weakref.WeakKeyDictionary[UOp, Buffer|MultiBuffer] = weakref.WeakKeyDictionary() # this maps BUFFER uops to their device Buffers
all_metadata:weakref.WeakKeyDictionary[UOp, tuple[Metadata, ...]] = weakref.WeakKeyDictionary() # TODO: should this be here?

# NOTE: this should be frozen, but frozen is slower
@dataclass(eq=False, slots=True)
class UOp(MathTrait, metaclass=UOpMetaClass):
  op:Ops
  dtype:DType = dtypes.void
  src:tuple[UOp, ...] = tuple()
  arg:Any = None
  tag:Any = None
  children:set[weakref.ref[UOp]] = field(default_factory=set)
  def __del__(self):
    if Ops is not None and self.op is Ops.BUFFER and (buffer:=buffers.get(self)) is not None: buffer.ref(-1)
    try:
      if (ref:=UOpMetaClass.ucache.get(k:=(self.op, self.dtype, self.src, self.arg, self.tag))) is not None:
        for s in self.src: s.children.discard(ref)
        del UOpMetaClass.ucache[k]
    except AttributeError: pass
  def __reduce__(self):
    args = [self.op, self.dtype, self.src, self.arg, self.tag, self.metadata]
    if self.op is Ops.BUFFER and self.realized is not None and PICKLE_BUFFERS: args.append(self.realized)
    return UOp, tuple(args)
  def replace(self, **kwargs) -> UOp:
    new_args = (kwargs.pop("op", self.op), kwargs.pop("dtype", self.dtype), kwargs.pop("src", self.src),
                kwargs.pop("arg", self.arg), kwargs.pop("tag", self.tag))
    assert len(kwargs) == 0, f"unused kwargs in replace {list(kwargs)}"
    if (self.op, self.dtype, self.src, self.arg, self.tag) == new_args: return self
    return UOp(*new_args)
  def rtag(self, tag=True): return self.replace(tag=tag)
  @functools.cached_property
  def key(self) -> bytes:
    return hashlib.sha256(str((self.op, self.dtype, self.arg)).encode() + b"".join([s.key for s in self.src])).digest()
  def __repr__(self): return pretty_print(self, lambda x: f"{type(self).__name__}({x.op}, {x.dtype}, arg={x.argstr()}{x.tagstr()}, src=(%s))")
  def argstr(self): return f'({", ".join(map(str, self.arg))})' if self.op is Ops.REDUCE_AXIS else repr(self.arg)
  def tagstr(self): return f", tag={self.tag}" if self.tag is not None else ""

  @functools.cached_property
  def parents(self:UOp) -> dict[UOp, None]:
    ret = {s:None for s in self.src}
    for s in self.src: ret.update(s.parents)
    return ret
  @property
  def sparents(self:UOp) -> dict[UOp, None]: return {self:None, **self.parents}

  def toposort(self, gate:Callable|None=None) -> dict[UOp, None]:
    ret: dict[UOp, None] = {}
    stack: list[tuple[UOp, bool]] = [(self, False)] # each stack entry is (node, visited_flag)
    while stack:
      node, visited = stack.pop()
      if node in ret: continue
      if not visited:
        if gate is None or gate(node):
          stack.append((node, True))  # push node back on stack to process after its parents
          for parent in reversed(node.src): stack.append((parent, False)) # push parents on the stack
      else: ret[node] = None # second time i'm seeing this node, add it to returned toposort
    return ret

  # returns map of UOps to their children in the graph rooted by self
  def get_children_map(self) -> dict[UOp, dict[UOp, None]]:
    ret: dict[UOp, dict[UOp, None]] = {}
    for u in self.toposort():
      ret[u] = {}
      for s in u.src: ret[s][u] = None
    return ret

  @functools.cached_property
  def tuplize(self:UOp) -> tuple:
    return (self.op.value, self.arg, self.dtype,)+tuple([x.tuplize for x in self.src])

  # *** uop shape stuff ***

  @functools.cached_property
  def st(self) -> ShapeTracker|None:
    if self.op in GroupOp.Block or self.op is Ops.INDEX: return None
    from tinygrad.shape.shapetracker import ShapeTracker
    # VIEW and MovementOps define a new ShapeTracker from the arg
    if self.op is Ops.VIEW: return self.arg
    if self.op in GroupOp.Movement: return unwrap(self.src[0].st).mop(self.op, self.arg)
    # CONST with a DEVICE has a shape of ()
    if self.op is Ops.CONST and len(self.src) and self.src[0].op is Ops.DEVICE: return ShapeTracker.from_shape(())
    # BufferOps and ASSIGN flow ShapeTracker from a direct edge
    if self.op in {Ops.STORE, Ops.ASSIGN, Ops.LOAD}: return self.src[0].st
    if self.op in GroupOp.Buffer: return views[0] if (views:=[x.st for x in self.src if x.op is Ops.VIEW]) else None

    # BUFFER/BUFFER_VIEW and KERNEL only have a size
    if self.op in {Ops.BUFFER, Ops.BUFFER_VIEW}: return ShapeTracker.from_shape((self.size,))
    if self.op is Ops.KERNEL: return ShapeTracker.from_shape((self.arg.ast.size,))
    if self.op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_REG}:
      sz = cast(PtrDType, self.dtype).size
      return ShapeTracker.from_shape((sz,)) if sz > 0 else None

    # hack for PTX, CASTing the ptr loses the shape
    if self.op is Ops.CAST and self.src[0].op is Ops.DEFINE_GLOBAL: return None

    # otherwise we get the shape from sources
    if not (src_sts := [x.st for x in self.src if x.st is not None]): return None
    assert all_same([x.shape for x in src_sts]), f"UOp sources must have the same shape {self} {[x.shape for x in src_sts]}"
    match self.op:
      case Ops.MULTI: shape = tuple(self.src[0].shape[a]*len(self.device) if a == self.axis else s for a,s in enumerate(self.src[0].shape))
      case Ops.BITCAST:
        shape = src_sts[0].shape
        if self.dtype.itemsize != (input_sz:=self.src[0].dtype.itemsize): shape = shape[:-1]+((shape[-1]*input_sz) // self.dtype.itemsize,)
      case Ops.REDUCE_AXIS | Ops.WMMA: shape = src_sts[0].reduce(self.axis_arg)
      case _: shape = src_sts[0].shape
    return ShapeTracker.from_shape(shape)

  @functools.cached_property
  def full_shape(self) -> tuple[sint, ...]:
    if self.op is Ops.VIEW: return self.shape
    # NOTE: if a parent doesn't have st its full_shape is empty
    parent_shapes = [x.full_shape for x in self.src]
    return tuple(smax(x) for x in itertools.zip_longest(*parent_shapes, fillvalue=1))
  @property
  def shape(self) -> tuple[sint, ...]:
    assert self.st is not None, f"{self.op} doesn't have a shape"
    return unwrap(self.st).shape
  @property
  def size(self) -> int: return self.arg[0] if self.op is Ops.BUFFER_VIEW else self.arg if self.op is Ops.BUFFER else unwrap(self.st).size

  # *** uop evaluation ***

  def simplify(self):
    # late import!
    from tinygrad.uop.symbolic import symbolic
    with Context(TRACK_MATCH_STATS=0):
      return graph_rewrite(self, symbolic)
  def ssimplify(self) -> UOp|ConstType: return ret.arg if (ret:=self.simplify()).op is Ops.CONST else ret
  def _eval(self, dtype, expected_type:Type[T]) -> T:
    assert self.dtype in dtype, f"eval with wrong dtype {self}"
    vmin, vmax = (simple_self:=self.simplify())._min_max
    if vmin != vmax: raise ValueError(f"eval failed to be a single number, range is {vmin} to {vmax} in {simple_self.render()}")
    assert isinstance(vmin, expected_type), f"vmin is wrong dtype {type(vmin)} != {expected_type}"
    return vmin
  def __bool__(self): return self._eval((dtypes.bool,), bool)
  def __int__(self): return self._eval(dtypes.ints, int)
  def __float__(self): return self._eval(dtypes.floats, float)
  def substitute(self, dvars:dict[UOp, UOp], name:str|None=None):
    dvars = {k:v for k,v in dvars.items() if k is not v}
    if len(dvars) == 0: return self
    with Context(TRACK_MATCH_STATS=(0 if name is None else TRACK_MATCH_STATS.value)):
      return graph_rewrite(self, _substitute, dvars, bottom_up=True, name=name)

  # *** uop syntactic sugar ***

  @property
  def st_arg(self) -> ShapeTracker:
    assert self.op in GroupOp.Buffer, f"st_arg called on {self.op}"
    return unwrap(self.st)
  @property
  def axis_arg(self) -> tuple[int, ...]:
    assert self.op in {Ops.REDUCE_AXIS, Ops.WMMA}, f"axis_arg called on {self.op}"
    ret = self.arg[1] if self.op is Ops.REDUCE_AXIS else self.arg[7]
    assert isinstance(ret, tuple) and all(isinstance(x, int) for x in ret), f"axis_arg trying to return {ret}"
    return ret
  def sink(self, *srcs:UOp|None, **kwargs): return UOp(Ops.SINK, dtypes.void, (self,)+tuple([x for x in srcs if x is not None]), **kwargs)
  def detach(self): return UOp(Ops.DETACH, self.dtype, (self,))
  def index(self, idx:UOp, valid:UOp|None=None): return UOp(Ops.INDEX, self.dtype, (self,idx,valid) if valid is not None else (self,idx))
  def __getitem__(self, idx): return self.index(idx)
  def const_like(self, b:ConstLike):
    # constants can optionally have a DEVICE source
    return UOp.const(self.dtype, b, device=self._device, shape=self.shape if self.st is not None else None)
  def broadcast(self, count:int):
    assert self.dtype.count == 1
    if count == 1: return self
    return UOp(Ops.VECTORIZE, self.dtype.vec(count), (self,)*count)
  def cast(self, dtype:DType):
    if self.dtype == dtype: return self
    return UOp(Ops.CAST, dtype, (self,))
  def cast_vec(self, dtype:DType): return UOp(Ops.CAST, dtype.vec(self.dtype.count), (self,))
  def bitcast(self, dtype:DType): return UOp(Ops.BITCAST, dtype, (self,))
  def gep(self, i:tuple[int, ...]|int):
    if isinstance(i, tuple) and len(i) == 1: return self.gep(i[0])
    if isinstance(i, int):
      # NOTE: these are just shortcuts to not have to create and fold later
      if self.op is Ops.VECTORIZE: return self.src[i]
      if self.op is Ops.VCONST: return UOp.const(self.dtype.scalar(), self.arg[i])
      if self.op is Ops.CONST: return UOp.const(self.dtype.scalar(), self.arg)
      i = (i,)
    return UOp(Ops.GEP, self.dtype.scalar().vec(len(i)) if len(i) > 1 else self.dtype.scalar(), (self,), i)
  def load(self, *src:UOp, **kwargs): return UOp(Ops.LOAD, dtype=kwargs.pop("dtype", self.dtype.base), src=(self,)+src, **kwargs)
  def store(self, *src:UOp, **kwargs): return UOp(Ops.STORE, dtypes.void, (self,)+src, **kwargs)
  def assign(self, x:UOp): return UOp(Ops.ASSIGN, self.dtype, (self, x))
  def barrier(self, *src:UOp): return UOp(Ops.BARRIER, src=(self,)+src)
  def alu(self, op, *src:UOp, **kwargs):
    out_dtype = (self, *src)[-1].dtype
    if op in {Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ}: out_dtype = dtypes.bool.vec(out_dtype.count) if out_dtype.count > 1 else dtypes.bool
    return UOp(op, out_dtype, (self,)+src, **kwargs)
  @staticmethod
  def const(dtype:DType, b:ConstLike, device:str|tuple[str, ...]|None=None, shape:tuple[sint, ...]|None=None):
    if isinstance(b, UOp): return b.unbind()[0] if b.op is Ops.BIND else b
    if isinstance(b, tuple) and all_same(b): b = b[0]  # doesn't have to be a VCONST if they are all the same
    ret = UOp(Ops.VCONST if isinstance(b, tuple) else Ops.CONST, dtype, arg=dtypes.as_const(b, dtype))
    if shape is not None:
      from tinygrad.shape.shapetracker import ShapeTracker
      ret = ret.replace(src=(UOp(Ops.VIEW, dtypes.void, (), ShapeTracker.from_shape(shape, (0,)*len(shape))),))
    if device is not None:
      ret = ret.replace(src=(UOp(Ops.DEVICE, arg=device).view(unwrap(ret.st)),))
    return ret
  @staticmethod
  def range(dtype:DType, end:sint, idx:int): return UOp(Ops.RANGE, dtype=dtype, src=(sint_to_uop(end),), arg=idx)
  def r(self, op:Ops, axis:tuple[int, ...]):
    axis = tuple(sorted([x for x in axis if resolve(self.shape[x] != 1)]))
    if len(axis) == 0: return self
    # move any non reduce axis before the first reduce axis
    move_early, rest = partition(range(axis[0], len(self.shape)), lambda i: i not in axis and resolve(self.shape[i] != 1))
    permaxis = tuple(range(axis[0])) + tuple(move_early) + tuple(rest)
    ret = self.permute(permaxis)
    new_axis = tuple([x for x in range(axis[0]+len(move_early), len(self.shape)) if resolve(ret.shape[x] != 1)])
    assert len(axis) == len(new_axis)
    ret = UOp(Ops.REDUCE_AXIS, self.dtype, (ret,), (op, new_axis))
    return ret.reshape(tuple([x if i not in axis else 1 for i,x in enumerate(self.shape)]))
  def reduce(self, *src:UOp, **kwargs): return UOp(Ops.REDUCE, kwargs.pop('dtype', self.dtype), src=(self,)+src, **kwargs)
  def contiguous(self): return self.alu(Ops.CONTIGUOUS)
  def contiguous_backward(self): return self.alu(Ops.CONTIGUOUS_BACKWARD)
  def fuse(self): return self.alu(Ops.FUSE)
  def allreduce(self, op, device:str|tuple[str, ...]|UOp):
    assert isinstance(self.device, tuple), f"allreduce must be on tuple {self.device} isn't"
    return UOp(Ops.ALLREDUCE, self.dtype, (self, UOp(Ops.DEVICE, arg=device) if not isinstance(device, UOp) else device), op)

  # *** from MultiLazyBuffer ***

  def multi(self, axis:int|None):
    assert isinstance(self.device, tuple), f"multi device must be tuple, {self.device} isn't"
    assert axis is not None, "multi None is no longer supported"
    return UOp(Ops.MULTI, self.dtype, (self,), axis)

  @property
  def bounds(self):
    if self.axis is None: raise RuntimeError("bounds is not defined when axis is None")
    return tuple(itertools.pairwise(itertools.accumulate([self.src[0].shape[self.axis] for _ in self.device], initial=0)))

  @functools.cached_property
  def axis(self) -> int|None:
    if self.op is Ops.MULTI: return self.arg
    # NOTE: they all have to share an axis, we always choose [-1]
    if self.op in GroupOp.ALU: return axes[-1] if (axes := dedup([x.axis for x in self.src if x.axis is not None])) else None
    if len(self.src) == 0: return None
    src_axis = self.src[0].axis
    if self.op is Ops.REDUCE_AXIS: return None if src_axis is not None and src_axis in self.arg[1] else src_axis
    if self.op is Ops.RESHAPE:
      if src_axis is None: return None
      arg_acc:list[sint] = list(itertools.accumulate(self.arg, operator.mul, initial=1))
      # new_axis is the last one that preserves prod(prior to new_axis) and must not move items between shards
      # TODO: what to do about shrinking to self.shape[self.axis]==1 len(self.real_lbs)==1?
      return len(arg_acc) - arg_acc[::-1].index(prod(self.src[0].shape[:src_axis])) - 1
    if self.op is Ops.PERMUTE: return self.arg.index(src_axis) if src_axis is not None else None
    return src_axis

  def _unshard(self, axis:int) -> UOp:
    bsz, dcount = self.shape[axis], len(self.device)
    dnum = UOp.variable("_device_num", 0, dcount-1)
    return self.pad(tuple((0,0) if a != axis else (bsz*dnum, bsz*(dcount-1) - bsz*dnum) for a in range(len(self.shape))))

  def _shard(self, axis:int) -> UOp:
    dcount = len(self.device)
    dnum = UOp.variable("_device_num", 0, dcount-1)
    if self.shape[axis] % dcount != 0: raise RuntimeError(f"multi axis uneven: {self.shape[axis]=} {axis=} {dcount=}")
    sz = self.shape[axis] // dcount
    return self.shrink(tuple((0,s) if i != axis else (dnum*sz,dnum*sz+sz) for i,s in enumerate(self.shape)))
  def shard(self, devices:tuple[str, ...], axis:int) -> UOp: return self.copy_to_device(devices)._shard(axis).multi(axis)

  # *** from LazyBuffer ***

  def copy_to_device(self, device:str|tuple[str, ...]|UOp, arg=None):
    assert arg is None or isinstance(self.device, tuple)
    inp = self if arg is None else UOp(Ops.MSELECT, self.dtype, src=(self,), arg=arg)
    return UOp(Ops.COPY, self.dtype, (inp, UOp(Ops.DEVICE, arg=device) if not isinstance(device, UOp) else device))
  def mselect(self, arg:int) -> UOp: return UOp(Ops.MSELECT, self.dtype, (self,), arg)
  @property
  def metadata(self) -> tuple[Metadata, ...]|None: return all_metadata.get(self, None)

  # *** uop movement ops ***

  @property
  def base(self) -> UOp:
    if (self.op is Ops.VIEW and len(self.src) != 0) or self.op in GroupOp.Movement: return self.src[0].base
    if self.op is Ops.MULTI: return self.src[0].base  # MULTI is really a VIEW
    return self
  def view(self, new_st:ShapeTracker) -> UOp: return UOp(Ops.VIEW, self.dtype, (self,), new_st)

  def _mop(self, op:Ops, arg) -> UOp:
    ret = UOp(op, self.dtype, (self,), arg)
    if self.st == ret.st: return self  # ignore NOOPs, also check ret.st
    return ret

  def reshape(self, arg:tuple[sint, ...]): return self._mop(Ops.RESHAPE, arg)
  def pad(self, arg:tuple[tuple[sint, sint], ...]): return self._mop(Ops.PAD, arg)
  def expand(self, arg:tuple[sint, ...]): return self._mop(Ops.EXPAND, arg)
  def permute(self, arg:tuple[sint, ...]): return self._mop(Ops.PERMUTE, arg)
  def shrink(self, arg:tuple[tuple[sint, sint], ...]): return self._mop(Ops.SHRINK, arg)
  def flip(self, arg:tuple[bool, ...]): return self._mop(Ops.FLIP, arg)

  # *** uop UNIQUE ***

  # TODO: use this in Buffer
  unique_num = itertools.count(0)
  @staticmethod
  def unique(): return UOp(Ops.UNIQUE, arg=next(UOp.unique_num))

  # *** uop Buffer stuff ***

  @staticmethod
  def new_buffer(device:str|tuple[str, ...], size:int, dtype:DType): return UOp(Ops.BUFFER, dtype, (UOp.unique(), UOp(Ops.DEVICE, arg=device)), size)
  @property
  def device(self) -> str|tuple[str, ...]: return cast(str|tuple[str, ...], unwrap(self._device))
  @functools.cached_property
  def _device(self) -> str|tuple[str, ...]|None:
    if self.op is Ops.DEVICE: return self.arg
    if self.op is Ops.MSELECT:
      assert isinstance(self.src[0].device, tuple), "mselect must be on tuple device"
      return self.src[0].device[self.arg]
    if self.op is Ops.MSTACK: return tuple(cast(str, x.device) for x in self.src)
    if self.op in {Ops.COPY, Ops.BUFFER, Ops.ALLREDUCE}: return self.src[1].device
    return next((x._device for x in self.src if x._device is not None), None)
  @property
  def buf_uop(self) -> UOp:
    if self.op is Ops.BUFFER: return self
    if self.op is Ops.MSELECT: return self.src[0].buf_uop.mselect(self.arg)
    if self.op is Ops.MSTACK: return UOp(Ops.MSTACK, self.dtype, src=tuple(x.buf_uop for x in self.src))
    assert self.op is Ops.ASSIGN, f"must be ASSIGN {self.op}"
    return self.src[0].base
  @property
  def buffer(self) -> Buffer|MultiBuffer:
    from tinygrad.device import Buffer, MultiBuffer
    if self is not self.base:
      assert unwrap(self.st).contiguous, "VIEW only works here if it's contiguous"
      return self.src[0].buffer
    if self.op is Ops.MSELECT:
      ret = self.src[0].buffer
      assert isinstance(ret, MultiBuffer)
      return ret.bufs[self.arg]
    if self.op is Ops.MSTACK:
      ret = MultiBuffer.__new__(MultiBuffer)
      ret.bufs = [cast(Buffer, x.buffer) for x in self.src]
      assert all_same([x.size for x in ret.bufs]) and all_same([x.dtype for x in ret.bufs]), "multibuffers mismatch buffers"
      return ret
    assert self.op is Ops.BUFFER, f"must be BUFFER {self.op}"
    if (cret:=buffers.get(self)) is not None: return cret
    rdtype = self.dtype if isinstance(self.dtype, ImageDType) else self.dtype.base
    if isinstance(self.device, tuple): ret = MultiBuffer(self.device, self.size, rdtype).ref(1)
    else: ret = Buffer(self.device, self.size, rdtype).ref(1)
    buffers[self] = ret
    return ret
  @property
  def realized(self) -> Buffer|MultiBuffer|None:
    # NOTE: this is used by the JIT to determine which inputs we capture
    return self.buffer if self.op in {Ops.BUFFER, Ops.MSTACK} and self.buffer.is_allocated() else None
  @property
  def is_realized(self) -> bool:
    return all(x.base.realized is not None for x in self.base.src) if self.base.op is Ops.MULTI else self.base.realized is not None

  # *** uop Variable stuff ***

  @staticmethod
  def variable(name:str, min_val:ConstType, max_val:ConstType, dtype:DType=dtypes.int) -> UOp:
    assert not isinstance(min_val, UOp) and not isinstance(max_val, UOp), f"can't create Variable {name} with {min_val}/{max_val}"
    return UOp(Ops.DEFINE_VAR, dtype, arg=(name, min_val, max_val))
  @property
  def expr(self):
    assert self.op is Ops.DEFINE_VAR, f"op is {self.op}, need DEFINE_VAR"
    return self.arg[0]
  def bind(self, val:int|UOp):
    assert self.op is Ops.DEFINE_VAR, f"op is {self.op}, need DEFINE_VAR"
    uval = self.const_like(val) if isinstance(val, int) else val
    assert self.arg[1] <= uval.vmin and uval.vmax <= self.arg[2], f"bind {val} not in range [{self.arg[1]}, {self.arg[2]}]"
    return UOp(Ops.BIND, self.dtype, (self, uval))
  def unbind(self) -> tuple[Variable, int]:
    assert self.op is Ops.BIND and self.src[0].op is Ops.DEFINE_VAR and self.src[1].op is Ops.CONST, f"can't unbind {self}"
    return self.src[0], self.src[1].arg
  @property
  def val(self) -> int: return self.unbind()[1]
  def vars(self) -> set[UOp]:
    bound_vars = set([x for x in self.toposort() if x.op is Ops.BIND and x.src[0].op is Ops.DEFINE_VAR])
    bound_var_base = set(x.src[0] for x in bound_vars)
    all_vars = set([x for x in self.toposort() if x.op is Ops.DEFINE_VAR])
    return bound_vars.union(set([x for x in all_vars if x not in bound_var_base]))
  def variables(self) -> list[Variable]:
    st_vars: list[set[Variable]] = [x.arg.vars() for x in self.toposort() if x.op is Ops.VIEW]
    return sorted(set.union(*st_vars, set([x.unbind()[0] if x.op is not Ops.DEFINE_VAR else x for x in self.vars()])), key=lambda v: v.arg)

  # *** uop symbolic stuff ***

  def is_increasing(self:UOp) -> bool:
    # is f a monotonically increasing function regards its input
    if self.op in GroupOp.Irreducible: return True
    if self.op is Ops.ADD: return self.src[0].is_increasing() and self.src[1].is_increasing()
    if self.op in (Ops.MUL, Ops.IDIV) and self.src[1].op is Ops.CONST and self.src[1].arg >= 0: return self.src[0].is_increasing()
    return False  # False if not sure
  def const_factor(self) -> int:
    """largest known int that divides self"""
    # TODO: for negatives it's not the largest
    if self.op is Ops.CONST: return self.arg
    if self.op is Ops.VCONST: return math.gcd(*self.arg)
    if self.op is Ops.ADD: return math.gcd(self.src[0].const_factor(), self.src[1].const_factor())
    if self.op is Ops.MUL: return self.src[0].arg if self.src[0].op is Ops.CONST else self.src[1].arg if self.src[1].op is Ops.CONST else 1
    return 1
  def divides(self, v:int) -> UOp|None:
    if v==1: return self
    if self.op is Ops.CONST: return self.const_like(self.arg//v) if self.arg%v == 0 else None
    if self.op is Ops.VCONST: return self.const_like(tuple(x//v for x in self.arg)) if all(x%v == 0 for x in self.arg) else None
    if self.op is Ops.ADD: return d0+d1 if (d0:=self.src[0].divides(v)) is not None and (d1:=self.src[1].divides(v)) is not None else None
    if self.op is Ops.MUL:
      if (d0:=self.src[0].divides(v)) is not None: return d0 * self.src[1]
      if (d1:=self.src[1].divides(v)) is not None: return self.src[0] * d1
    return None # generic None if we aren't sure
  @property
  def vmin(self) -> ConstType: return self._min_max[0]
  @property
  def vmax(self) -> ConstType: return self._min_max[1]
  @functools.cached_property
  def _min_max(self) -> tuple[ConstType, ConstType]:
    if self.op in GroupOp.Binary and not dtypes.is_float(self.dtype):
      (s0_vmin, s0_vmax), (s1_vmin, s1_vmax) = self.src[0]._min_max, self.src[1]._min_max
      if self.op is Ops.ADD: return s0_vmin+s1_vmin, s0_vmax+s1_vmax
      if self.op is Ops.SUB: return s0_vmin-s1_vmax, s0_vmax-s1_vmin
      if self.op is Ops.AND and s1_vmin == s1_vmax and s0_vmin >= 0 and s1_vmin >= 0: return min(0, s0_vmin), min(s0_vmax, s1_vmax)
      if self.op is Ops.MUL: return min(vals:=(s0_vmin*s1_vmin, s0_vmin*s1_vmax, s0_vmax*s1_vmin, s0_vmax*s1_vmax)), max(vals)
      # SHL/SHR on consts only
      if self.op is Ops.SHL and s1_vmin == s1_vmax and all_int(t:=(s0_vmin, s0_vmax, s1_vmin)): return t[0] << t[2], t[1] << t[2]
      if self.op is Ops.SHR and s1_vmin == s1_vmax and all_int(t:=(s0_vmin, s0_vmax, s1_vmin)): return t[0] >> t[2], t[1] >> t[2]
      if self.op is Ops.MOD:
        if s1_vmin > 0: return (0, s1_vmax-1) if s0_vmin >= 0 else (-(s1_vmax-1), 0) if s0_vmax <= 0 else (-(s1_vmax-1), s1_vmax-1)
        if s1_vmax < 0: return (0, -s1_vmin-1) if s0_vmin >= 0 else (-(-s1_vmin-1), 0) if s0_vmax <= 0 else (-(-s1_vmin-1), -s1_vmin-1)
      if self.op is Ops.IDIV:
        assert isinstance(s0_vmin, int) and isinstance(s0_vmax, int) and isinstance(s1_vmin, int) and isinstance(s1_vmax, int)
        if (c:=s1_vmin) == s1_vmax:  # s1 is a const
          if c > 0: return cdiv(s0_vmin, c), cdiv(s0_vmax, c)
          if c < 0: return cdiv(s0_vmax, c), cdiv(s0_vmin, c)
        if (s0_vmax <= 0 and s1_vmax < 0): return cdiv(s0_vmax, s1_vmin), cdiv(s0_vmin, s1_vmax)
        if (s0_vmin >= 0 and s1_vmin > 0): return cdiv(s0_vmin, s1_vmax), cdiv(s0_vmax, s1_vmin)
        if (s0_vmax <= 0 and s1_vmin > 0): return cdiv(s0_vmin, s1_vmin), cdiv(s0_vmax, s1_vmax)
        if (s0_vmin >= 0 and s1_vmax < 0): return cdiv(s0_vmax, s1_vmax), cdiv(s0_vmin, s1_vmin)
      if self.op is Ops.MAX: return max(s0_vmin, s1_vmin), max(s0_vmax, s1_vmax)
      if self.op is Ops.CMPLT: return (s0_vmax<s1_vmin, s0_vmin<s1_vmax)
      if self.op is Ops.CMPNE: return ((s0_vmax < s1_vmin) or (s1_vmax < s0_vmin), not (s0_vmin == s0_vmax == s1_vmin == s1_vmax))
      if self.dtype == dtypes.bool:
        if self.op is Ops.OR: return s0_vmin or s1_vmin, s0_vmax or s1_vmax
        if self.op is Ops.AND: return s0_vmin and s1_vmin, s0_vmax and s1_vmax
    # float has NAN issue and we use explicit NAN in transcendental
    if self.op is Ops.WHERE and dtypes.is_int(self.dtype): return min(self.src[1].vmin, self.src[2].vmin), max(self.src[1].vmax, self.src[2].vmax)
    # NOTE: returned UOp is assumed to be CONST
    if self.op is Ops.DEFINE_VAR and self.arg: return self.arg[1], self.arg[2]
    if self.op is Ops.RANGE: return 0, (self.src[0]-1).vmax
    if self.op is Ops.BIND: return self.src[0]._min_max # ignore the bound value
    if self.op in {Ops.UNROLL, Ops.VECTORIZE}: return min(x.vmin for x in self.src), max(x.vmax for x in self.src)
    # TODO: Ops.SPECIAL is Ops.DEFINE_VAR
    if self.op is Ops.SPECIAL: return 0, self.arg[1]-1 if isinstance(self.arg[1], int) else self.arg[1].vmax
    if self.op is Ops.CONST: return self.arg, self.arg
    if self.op is Ops.VCONST: return (min(self.arg), max(self.arg))
    # TODO: CAST to bool/unsigned is not monotone, still some case can be simplified
    if self.op is Ops.CAST and self.dtype in (dtypes.floats+dtypes.sints):
      return max(dtypes.min(self.dtype), self.src[0].vmin), min(self.src[0].vmax, dtypes.max(self.dtype))
    return dtypes.min(self.dtype), dtypes.max(self.dtype)

  @functools.cached_property
  def _sym_fxn(self):
    sself = self.simplify()
    varnames = tuple(x.arg[0] for x in sself.toposort() if x.op is Ops.DEFINE_VAR)
    # TODO: sanitize varnames, or don't use naked eval while staying fast
    return eval("lambda "+','.join(varnames)+": "+sself.render(pm=renderer_infer)), varnames  # pylint: disable=eval-used

  def sym_infer(self, var_vals:dict[UOp, int]):
    fxn, varnames = self._sym_fxn
    return fxn(**{k.arg[0]:v for k,v in var_vals.items() if k.arg[0] in varnames})

  def render(self, simplify=True, pm:PatternMatcher|None=None) -> str:
    ret = graph_rewrite(self.simplify() if simplify else self, renderer if pm is None else pm)
    return ret.arg if ret.op is Ops.NOOP else str(ret)

class AxisType(Enum):
  GLOBAL = auto(); LOCAL = auto(); LOOP = auto(); GROUP_REDUCE = auto(); REDUCE = auto(); UPCAST = auto(); UNROLL = auto()  # noqa: E702

@dataclass(frozen=True)
class KernelInfo:
  name: str = "test"            # name of the kernel
  axis_types: tuple[AxisType, ...] = tuple()
  dont_use_locals: bool = False # don't use local indexing
  applied_opts: tuple = tuple()
  opts_to_apply: tuple|None = None
  @property
  def function_name(self): return to_function_name(self.name)

# ******** ops in python ********

def safe_exp2(x):
  try: return 2 ** x
  except OverflowError: return math.inf

def safe_pow(x, y):
  try: return math.nan if isinstance(p:=pow(x, y), complex) else p
  except ZeroDivisionError: return math.inf
  except ValueError: return math.inf if x > 0 else -math.inf

python_alu: dict[Ops, Callable]  = {
  Ops.LOG2: lambda x: math.log2(x) if x > 0 else -math.inf if x == 0 else math.nan, Ops.EXP2: safe_exp2,
  Ops.SQRT: lambda x: math.sqrt(x) if x >= 0 else math.nan, Ops.RECIP: lambda x: 1/x if x != 0 else math.copysign(math.inf, x),
  Ops.SIN: lambda x: math.sin(x) if not math.isinf(x) else math.nan, Ops.POW: safe_pow,
  Ops.NEG: operator.neg, Ops.ADD: operator.add, Ops.SUB: operator.sub, Ops.MUL: operator.mul, Ops.CMPNE: operator.ne, Ops.CMPLT: operator.lt,
  Ops.XOR: operator.xor, Ops.OR: operator.or_, Ops.AND: operator.and_, Ops.SHR: operator.rshift, Ops.SHL: operator.lshift, Ops.MAX: max,
  Ops.MOD: cmod, Ops.IDIV: cdiv, Ops.MULACC: lambda x,y,z: (x*y)+z, Ops.WHERE: lambda x,y,z: y if x else z, Ops.CMPEQ: operator.eq}

def exec_alu(op:Ops, dtype:DType, operands, truncate_output=True):
  if dtype.count > 1:
    return tuple([exec_alu(op, dtype.scalar(), [x[i] if isinstance(x, tuple) else x for x in operands]) for i in range(dtype.count)])
  alu = python_alu[op](*operands)
  return truncate.get(dtype, lambda x: x)(alu) if truncate_output else alu

# ***** uop helpers *****

def print_uops(uops:list[UOp]):
  for i,u in enumerate(uops):
    formatted_parents = [(uops.index(x) if x.op is not Ops.CONST else f"{x.arg}") if x in uops else "--" for x in u.src]
    print(f"{i:4d} {str(u.op):20s}: {str(u.dtype):30s} " f"{str(formatted_parents):32s} {u.arg}")

# ***** pattern matcher *****

def get_location() -> tuple[str, int]:
  frm = sys._getframe(1)
  # skip over ops.py/mathtraits.py (unless there's nothing but ops.py/mathtraits.py)
  while pathlib.Path(frm.f_code.co_filename).name in ("ops.py", "mathtraits.py") and frm.f_back is not None and \
      not frm.f_back.f_code.co_filename.startswith("<frozen"):
    frm = frm.f_back
  return frm.f_code.co_filename, frm.f_lineno

@functools.cache
def lines(fn) -> list[str]:
  with open(fn) as f: return f.readlines()

def printable(loc:tuple[str, int]) -> str:
  try: return lines(loc[0])[loc[1]-1].strip()
  except FileNotFoundError: return "<missing>"

class UPat(MathTrait):
  __slots__ = ("op", "dtype", "arg", "name", "src")
  def __init__(self, op:Ops|tuple[Ops, ...]|set[Ops]|None=None, dtype:DType|tuple[DType, ...]|None=None,
               src:tuple[UPat, ...]|list[UPat]|UPat|None=None, arg:Any=None,
               name:str|None=None, allow_any_len:bool=False, custom_early_reject:set[Ops]|None=None, location=None):
    assert op is None or isinstance(op, (Ops, tuple, set)), "op must be Ops or tuple of Ops"
    self.op: tuple[Ops, ...]|None = (op,) if isinstance(op, Ops) else (tuple(op) if isinstance(op, set) else op)
    self.dtype: tuple[DType, ...]|None = (dtype,) if isinstance(dtype, DType) else dtype
    self.arg, self.name, self._in_src, self.custom_early_reject = arg, name, src, custom_early_reject
    self.src: Any = None
    assert self.name != "ctx", "UPat can't be named ctx"
    assert dtype is None or isinstance(dtype, DType) or all(isinstance(x, DType) for x in dtype), f"invalid dtype {dtype}"

    # try all permutations if it's a list
    if isinstance(src, list): self.src = list(itertools.permutations(src)) if not all_same(src) else [tuple(src)]
    # only one if it's a tuple
    elif isinstance(src, tuple): self.src = [src]
    # repeat if it's a UPat
    elif isinstance(src, UPat): self.src = [itertools.repeat(src)]

    self.strict_length = not (allow_any_len or isinstance(src, UPat) or src is None)
    self.required_len: int = 0 if isinstance(src, UPat) or src is None else len(src)
    self.location = location or get_location()

    if custom_early_reject is not None: self.early_reject = custom_early_reject
    else:
      upat_match = [src] if isinstance(src, UPat) else ([] if src is None else self.src[0])
      self.early_reject = {pp.op[0] for pp in upat_match if pp.op is not None and len(pp.op) == 1}

  def __reduce__(self):
    return UPat, (self.op, self.dtype, self._in_src, self.arg, self.name, not self.strict_length, self.custom_early_reject, self.location)
  def named(self, name:str): return UPat(self.op, self.dtype, self._in_src, self.arg, name, not self.strict_length, self.custom_early_reject)

  @staticmethod
  def any(*src): return UPatAny(src=src)
  def or_casted(self, name:str|None=None): return UPat.any(self if name is None else self.named(name), UPat(Ops.CAST, name=name, src=(self,)))

  @staticmethod
  @functools.cache
  def var(name:str|None=None, dtype:DType|tuple[DType, ...]|None=None): return UPat(dtype=dtype, name=name)
  @staticmethod
  @functools.cache
  def cvar(name:str|None=None, dtype:DType|None=None, vec=True): return UPat((Ops.CONST,Ops.VCONST) if vec else Ops.CONST, dtype, name=name)
  @staticmethod
  def const(dtype:DType|tuple[DType, ...]|None, b:ConstType): return UPat(Ops.CONST, dtype=dtype, arg=b)

  # copied from UOp
  def sink(self, *srcs:UPat|None, **kwargs): return UPat(Ops.SINK, dtypes.void, (self,)+tuple([x for x in srcs if x is not None]), **kwargs)
  def index(self, idx:UPat, valid:UPat|None=None): return UPat(Ops.INDEX, self.dtype, (self,idx,valid) if valid is not None else (self,idx))
  def view(self, st=None, **kwargs): return UPat(Ops.VIEW, self.dtype, (self,), st, **kwargs)
  def cast(self, dtype=None, **kwargs): return UPat(Ops.CAST, dtype, (self,), **kwargs)
  def bitcast(self, dtype=None): return UPat(Ops.BITCAST, dtype, (self,))
  def gep(self, i:int|None=None, **kwargs): return UPat(Ops.GEP, None, (self,), (i,) if i is not None else None, **kwargs)
  def load(self, *src:UPat, **kwargs): return UPat(Ops.LOAD, src=(self,)+src, **kwargs)
  def store(self, *src:UPat, **kwargs): return UPat(Ops.STORE, self.dtype, (self,)+src, **kwargs)
  def assign(self, x:UPat, **kwargs): return UPat(Ops.ASSIGN, self.dtype, (self,x), **kwargs)
  def reduce(self, *src:UPat, **kwargs): return UPat(Ops.REDUCE, self.dtype, src=(self,)+src, **kwargs)
  def fuse(self): return self.alu(Ops.FUSE)
  def or_broadcasted(self, **kwargs): return UPat.any(self, UPat(Ops.VECTORIZE, self.dtype, src=self, **kwargs))

  def const_like(self, b:ConstLike): return UPat.const(self.dtype, cast(ConstType, b))
  def alu(self, op:Ops, *src:UPat):
    asrc = (self,)+src
    return UPat(op, dtypes.bool if op in {Ops.CMPLT, Ops.CMPNE} else asrc[-1].dtype, list(asrc) if op in GroupOp.Commutative else asrc)

  def __repr__(self):
    def rep(x):
      form = "UPat(%s, %s, name=%s, dtype=%s, allow_any_len=%s, src=%s)"
      return form % (None if x.op is None else ('(%s)'%', '.join(map(str, x.op))), x.arg, repr(x.name),
        set(x.dtype) if x.dtype else None, not x.strict_length, "[%s]" if x.src and len(x.src)>1 else ("(%s)" if x.src else "%s"))
    return pretty_print(self, rep, srcfn=lambda x:None if x.src is None else [next(x.src[0])] if isinstance(x.src[0], itertools.repeat) else x.src[0])

  def match(self:UPat, uop:UOp, store:dict[str, UOp]) -> list[dict[str, UOp]]:
    if (self.op is not None and uop.op not in self.op) or \
       (self.name is not None and store.setdefault(self.name, uop) is not uop) or \
       (self.dtype is not None and uop.dtype not in self.dtype and uop.dtype.scalar() not in self.dtype) or \
       (self.arg is not None and self.arg != uop.arg) or \
       (len(uop.src) < self.required_len) or \
       (self.strict_length and len(uop.src) != self.required_len): return []
    if self.src is None: return [store]
    res: list[dict[str, UOp]] = []
    for vp in self.src:
      stores, new_stores = [store.copy()], []
      for uu, vv in zip(uop.src, vp):
        for s in stores: new_stores.extend(vv.match(uu, s))
        stores, new_stores = new_stores, []
      res.extend(stores)
    return res

class UPatAny(UPat):
  def match(self:UPat, uop:UOp, store:dict[str, UOp]) -> list[dict[str, UOp]]:
    matches = [x.match(uop, store.copy()) for x in self.src[0]]
    return flatten([x for x in matches if x is not None])

def deconstruct_function(fxn:Callable) -> tuple:
  new_globals = {k:v for k,v in fxn.__globals__.items() if k in fxn.__code__.co_names}
  for co in fxn.__code__.co_consts:
    if isinstance(co, types.CodeType): new_globals.update({k:v for k,v in fxn.__globals__.items() if k in co.co_names})
  # NOTE: optional round trip through pickle!
  assert fxn.__closure__ is None, "closures are not supported in pattern matchers"
  ret = fxn.__code__, new_globals, fxn.__name__, fxn.__defaults__
  return pickle.loads(pickle.dumps(ret)) if getenv("TEST_PICKLE") else ret

@functools.cache
def upat_interpret(p:UPat, fxn:Callable) -> Callable:
  real_fxn = types.FunctionType(*deconstruct_function(fxn))
  if 'ctx' in inspect.signature(real_fxn).parameters:
    def universal_match(uop, ctx):
      for match in p.match(uop, {}):
        if (ret:=real_fxn(ctx=ctx, **match)) is not None: return ret  # pylint: disable=not-callable
      return None
  else:
    def universal_match(uop, _):
      for match in p.match(uop, {}):
        if (ret:=real_fxn(**match)) is not None: return ret  # pylint: disable=not-callable
      return None
  return universal_match

class PatternMatcher:
  def __init__(self, patterns:Sequence[tuple[UPat, Callable|tuple]], compiled=bool(getenv("UPAT_COMPILE", 1))):
    if compiled: from tinygrad.uop.upat import upat_compile
    # if this comes from a pickle, we reconstruct the lambda functions here
    self.patterns:list[tuple[UPat, Callable]] = [(p,types.FunctionType(*fxn) if isinstance(fxn, tuple) else fxn) for p,fxn in patterns]
    # NOTE: use of DefaultDict here is very dangerous! all keys will live for the lifetime of the PatternMatcher!
    self.pdict: dict[Ops, list[tuple[UPat, Callable, set]]] = {}
    # uop is required, arg is optional
    for p,fxn in self.patterns:
      assert p.op is not None
      if compiled and (match:=upat_compile(p, fxn)) is not None: pass # pylint: disable=E0606
      else: match = upat_interpret(p, fxn)
      for uop in p.op: self.pdict.setdefault(uop, []).append((p, match, p.early_reject))

  def __reduce__(self): return PatternMatcher, ([(x,deconstruct_function(fxn) if fxn.__name__ == "<lambda>" else fxn) for x,fxn in self.patterns],)

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def __add__(self, more:PatternMatcher): return PatternMatcher(self.patterns+more.patterns)

  def rewrite(self, uop:UOp, ctx=None) -> UOp|None:
    ler = {u.op for u in uop.src}
    for _,match,early_reject in self.pdict.get(uop.op, []):
      if not early_reject.issubset(ler): continue
      if (ret:=match(uop, ctx)) is not None and ret is not uop: return ret
    return None

  def fixed_point_rewrite(self, uop:UOp, ctx=None) -> UOp:
    # apply rewrite rules until a fixed point is reached. may return `uop` itself if PatternMatcher doesn't match
    new_n: UOp|None = uop
    seen = set()
    while new_n is not None:
      if new_n in seen: raise RuntimeError("infinite loop in fixed_point_rewrite")
      seen.add(new_n)
      last_n, new_n = new_n, self.rewrite(new_n, ctx)
    return last_n

# *** non-blocking UOp tracker ***

ucount = itertools.count()
uop_number:weakref.WeakKeyDictionary[UOp, int] = weakref.WeakKeyDictionary()
uop_fields:dict[int, tuple] = {}
def track_uop(u:UOp):
  if (cret:=uop_number.get(u)) is not None: return cret
  uop_number[u] = num = next(ucount)
  # KERNEL also has a UOp in the arg
  arg = type(u.arg)(track_uop(u.arg.ast), u.arg.metadata) if u.op is Ops.KERNEL else u.arg
  uop_fields[num] = (u.op, u.dtype, tuple(track_uop(s) for s in u.src), arg, u.tag)
  return num

# *** tracking pattern matcher ***

VIZ = ContextVar("VIZ", 0)
TRACK_MATCH_STATS = ContextVar("TRACK_MATCH_STATS", 2 if VIZ else 0)
match_stats:dict[UPat, list[int|float]] = dict()

@dataclass(frozen=True)
class TrackedGraphRewrite:
  loc:tuple[str, int]                    # location that called graph_rewrite
  sink:int                               # the sink input to graph_rewrite
  matches:list[tuple[int, int, tuple]]   # before/after UOp, UPat location
  name:str|None                          # optional name of the rewrite
  depth:int                              # depth if it's a subrewrite
  bottom_up:bool

tracked_keys:list[TracingKey] = []
tracked_ctxs:list[list[TrackedGraphRewrite]] = []
_name_cnt:dict[str, itertools.count] = {}

if getenv("CAPTURE_PROCESS_REPLAY"):
  replay_capture: dict[str, bytes] = {}
  import atexit
  @atexit.register
  def save_to_diskcache():
    for k,v in replay_capture.items(): diskcache_put("process_replay", k, v, prepickled=True)

def track_rewrites(name:Callable[..., str|TracingKey]|bool=True):
  def _decorator(func):
    def __wrapper(*args, **kwargs):
      fn = key = func.__name__
      if TRACK_MATCH_STATS >= 2:
        tracked_keys.append(key:=TracingKey(n:=f"{fn} n{next(_name_cnt.setdefault(fn, itertools.count(1)))}", (n,), cat=fn))
        tracked_ctxs.append([])
      with cpu_profile(key, "TINY") as e:
        ret = func(*args, **kwargs)
      if TRACK_MATCH_STATS >= 2 and callable(name):
        name_ret = name(*args, **kwargs, ret=ret)
        assert isinstance(name_ret, (TracingKey, str)), f"name function returned {type(name_ret)}"
        tracked_keys[-1] = k = TracingKey(n:=tracked_keys[-1].display_name.replace(fn, name_ret), (n,)) if isinstance(name_ret, str) else name_ret
        e.name = TracingKey(k.display_name if isinstance(name_ret, str) else f"{fn} for {k.display_name}", k.keys, cat=fn)
      if getenv("CAPTURE_PROCESS_REPLAY"):
        # find the unittest frame we're capturing in
        frm = sys._getframe(1)
        while (f_back:=frm.f_back) is not None and "unittest" not in f_back.f_code.co_filename: frm = f_back
        loc = f"{frm.f_code.co_filename.split('/')[-1]}:{frm.f_lineno} {frm.f_code.co_name}"
        # capture global context vars and all the args passed in
        with Context(PICKLE_BUFFERS=0):
          inputs = (fn, args, kwargs, ContextVar._cache)
          replay_capture[hashlib.sha256(pickle.dumps(inputs)).hexdigest()] = pickle.dumps(inputs+(loc, ret))
      return ret
    return __wrapper
  return _decorator

active_rewrites:list[TrackedGraphRewrite] = []
def track_matches(func):
  def _track_func(*args, **kwargs):
    if tracking:=(TRACK_MATCH_STATS >= 2 and tracked_ctxs):
      loc = ((frm:=sys._getframe(1)).f_code.co_filename, frm.f_lineno)
      depth = len(active_rewrites)
      tracked_ctxs[-1].append(ctx:=TrackedGraphRewrite(loc, track_uop(args[0]), [], kwargs.get("name", None), depth, kwargs.get("bottom_up", False)))
      active_rewrites.append(ctx)
    with cpu_profile(kwargs.get("name", "<unnamed>"), "TINY", display=tracking):
      ret = func(*args, **kwargs)
    if tracking: active_rewrites.pop()
    return ret
  return _track_func

class TrackedPatternMatcher(PatternMatcher):
  def rewrite(self, uop:UOp, ctx=None) -> UOp|None:
    ret = None
    ler = {u.op for u in uop.src}
    for p,match,early_reject in self.pdict.get(uop.op, []):
      if p not in match_stats: match_stats[p] = [0,0,0.0,0.0]
      st = time.perf_counter()
      if not early_reject.issubset(ler):
        match_stats[p][2] += time.perf_counter()-st
        continue
      match_stats[p][1] += 1
      if (ret:=match(uop, ctx)) is not None and ret is not uop:
        match_stats[p][0] += 1
        match_stats[p][3] += (et:=time.perf_counter()-st)
        if TRACK_MATCH_STATS >= 3: print(f"{et*1e6:7.2f} us -- ", printable(p.location))
        if TRACK_MATCH_STATS >= 2 and isinstance(ret, UOp) and active_rewrites:
          active_rewrites[-1].matches.append((track_uop(uop), track_uop(ret), p.location))
        return ret
      match_stats[p][2] += time.perf_counter()-st
    return None

if TRACK_MATCH_STATS or PROFILE:
  PatternMatcher = TrackedPatternMatcher  # type: ignore
  import atexit
  @atexit.register
  def print_match_stats():
    if TRACK_MATCH_STATS >= 2:
      with open(fn:=temp("rewrites.pkl", append_user=True), "wb") as f:
        print(f"rewrote {len(tracked_ctxs)} graphs and matched {sum(len(r.matches) for x in tracked_ctxs for r in x)} times, saved to {fn}")
        pickle.dump((tracked_keys, tracked_ctxs, uop_fields), f)
    if VIZ: launch_viz(VIZ, temp("rewrites.pkl", append_user=True))
    if getenv("PRINT_MATCH_STATS", TRACK_MATCH_STATS.value):
      ret = [0,0,0.0,0.0]
      for k,v in sorted(list(match_stats.items()), key=lambda x: x[1][2]+x[1][3]):
        loc_str = f"{k.location[0].split('/')[-1]}:{k.location[1]}"
        if v[1] != 0: print(f"{v[0]:6d} / {v[1]:7d} -- {v[3]*1000.:9.2f} / {(v[2]+v[3])*1000.:9.2f} ms -- {loc_str:20s}", printable(k.location))
        ret = [x+y for x,y in zip(ret, v)]
      print(f"{ret[0]:6d} / {ret[1]:7d} -- {ret[3]*1000.:9.2f} / {(ret[2]+ret[3])*1000.:9.2f} ms -- TOTAL")
      print(f"{len(match_stats)} rules, {sum(v[0] > 0 for v in match_stats.values())} matched once")

  def launch_viz(var:ContextVar, data:str):
    os.environ[(env_str:=var.key)] = "0"
    os.environ[f"{env_str}_DATA"] = data
    os.environ[f"{env_str}_VALUE"] = str(var.value)
    if not int(os.getenv("VIZ", "0")) and not int(os.getenv("PROFILE", "0")):
      args = ['--kernels', getenv("VIZ_DATA", "")] if getenv("VIZ_DATA", "") else []
      args += ['--profile', getenv("PROFILE_DATA", "")] if getenv("PROFILE_DATA", "") else []
      os.execv(sys.executable, [sys.executable] + [os.path.join(os.path.dirname(__file__), "../", "viz", "serve.py")] + args)

# *** simple graph rewrite engine ***

class RewriteNotReady(Exception): pass
class RewriteContext:
  def __init__(self, pm, bpm, ctx=None):
    self.pm: PatternMatcher|None = pm
    self.bpm: PatternMatcher|None = bpm
    self.ctx = ctx
    self.replace: dict[UOp, UOp] = {}

  def unified_rewrite(self, root:UOp) -> UOp:
    stack: list[tuple[UOp, int, UOp]] = [(root, 0, root)]
    while stack:
      if len(stack) >= 200000: raise RuntimeError("infinite loop in graph_rewrite")
      n, stage, new_n = stack.pop()
      if n in self.replace: continue  # skip any nodes we have seen
      try:
        if stage == 0:
          # if bottom up, we rewrite this node early. in both cases, we add its parents to the stack
          if self.bpm is not None: new_n = self.bpm.fixed_point_rewrite(new_n, self.ctx)
          stack.append((n, 1, new_n))
          for x in reversed(new_n.src): stack.append((x, 0, x))
        elif stage == 1:
          try: new_src = tuple([self.replace[x] for x in new_n.src])
          except KeyError: raise RewriteNotReady  # pylint: disable=raise-missing-from
          if new_src == new_n.src:
            # if top down, do the rewrite. if no rewrite or bottom up, we are done rewriting this node so we add it to the dict
            if self.pm is None or (new_src_n:=self.pm.rewrite(new_n, self.ctx)) is None:
              self.replace[n] = new_n
              continue
          else:
            # if srcs changed from rewrites, construct a new UOp with the new srcs
            new_src_n = UOp(new_n.op, new_n.dtype, new_src, new_n.arg, new_n.tag)
          # trigger a rewrite of new_src_n, then after that rewrite is done, link it back to n
          stack.append((n, 2, new_src_n))
          stack.append((new_src_n, 0, new_src_n))
        else:
          # in stage 2, we link the result of new_n to the result of n
          try: self.replace[n] = self.replace[new_n]
          except KeyError: raise RewriteNotReady  # pylint: disable=raise-missing-from
      except RewriteNotReady:
        # retry this later
        stack.insert(0, (n, stage, new_n))
    return self.replace[root]

@track_matches
def graph_rewrite(sink:UOp, pm:PatternMatcher, ctx=None, bottom_up=False, name=None, bpm=None) -> UOp:
  rewrite_ctx = RewriteContext(pm if not bottom_up else None, pm if bottom_up else bpm, ctx)
  return rewrite_ctx.unified_rewrite(sink)

@track_matches
def graph_rewrite_map(sink:UOp, pm:PatternMatcher, ctx=None, bottom_up=False, name=None, bpm=None,
                      input_map:dict[UOp, UOp]|None=None, ) -> dict[UOp, UOp]:
  rewrite_ctx = RewriteContext(pm if not bottom_up else None, pm if bottom_up else bpm, ctx)
  new_map: dict[UOp, UOp] = {}
  for k in sink.toposort():
    new_map[k] = v = rewrite_ctx.unified_rewrite(k)
    if k is not v and k.metadata is not None: all_metadata[v] = tuple(dedup(all_metadata.get(v, ())))+k.metadata
  if input_map is not None:
    for k,v in input_map.items(): new_map[k] = new_map.get(v,v)
  return new_map

def sint_to_uop(x:sint, dtype:DType=dtypes.int) -> UOp: return UOp.const(dtype, x) if isinstance(x, int) else x

_substitute = PatternMatcher([(UPat(tuple(Ops), name="x"), lambda ctx,x: ctx.get(x,None))])

# for debug
syms = { Ops.ADD: "+", Ops.SUB: "-", Ops.IDIV: "//", Ops.MOD: "%", Ops.SHL: "<<", Ops.SHR: ">>",
         Ops.MUL: "*", Ops.CMPLT: "<", Ops.CMPNE: "!=", Ops.AND: "&", Ops.OR: "|", Ops.XOR: "^"}
renderer = PatternMatcher([
  (UPat((Ops.DEFINE_VAR, Ops.SPECIAL), name="x"), lambda x: UOp(Ops.NOOP, arg=x.arg[0])),
  (UPat(Ops.RANGE, name="x"), lambda x: UOp(Ops.NOOP, arg=f"ridx{x.arg}")),
  (UPat((Ops.CONST, Ops.VCONST), name="x"), lambda x: UOp(Ops.NOOP, arg=str(x.arg))),
  (UPat(Ops.UNROLL, name="x"), lambda x: UOp(Ops.NOOP, arg=f"UNROLL({x.src[0].arg}, {x.arg})")),
  (UPat(Ops.CAST, name="x"), lambda x: UOp(Ops.NOOP, arg=f"({str(x.dtype)[7:]})({x.src[0].arg})")),
  (UPat(Ops.LOAD), lambda: UOp(Ops.NOOP, arg="load")),
  (UPat(Ops.BIND, src=UPat(Ops.NOOP), name="x"), lambda x: x.src[0]),
  #(UPat(Ops.BIND, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"{x.src[0].arg}[={x.src[1].arg}]")),
  (UPat(Ops.NEG, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"(-{x.src[0].arg})")),
  (UPat(Ops.RECIP, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"(1/{x.src[0].arg})")),
  (UPat(Ops.MAX, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"max({x.src[0].arg}, {x.src[1].arg})")),
  (UPat(Ops.MULACC, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"({x.src[0].arg}*{x.src[1].arg}+{x.src[2].arg})")),
  (UPat(Ops.WHERE, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"({x.src[1].arg} if {x.src[0].arg} else {x.src[2].arg})")),
  (UPat(GroupOp.ALU, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"({x.src[0].arg}{syms[x.op]}{x.src[1].arg})")),
])
renderer_infer = PatternMatcher([
  (UPat(Ops.MOD, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"cmod({x.src[0].arg}, {x.src[1].arg})")),
  (UPat(Ops.IDIV, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"cdiv({x.src[0].arg}, {x.src[1].arg})")),
  *renderer.patterns
])

# *** what was symbolic.py ***

sint = int|UOp
Variable = UOp

ConstLike = ConstType|Variable|tuple[ConstType, ...]
