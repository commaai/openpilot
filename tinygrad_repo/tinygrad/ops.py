from __future__ import annotations
from typing import Any, Optional, Union, Callable, cast, TYPE_CHECKING, Type, get_args
import sys, time, functools, itertools, math, operator, hashlib, os, types, pickle, pathlib, inspect, weakref
from enum import auto, IntEnum, Enum
from dataclasses import dataclass, field
from tinygrad.dtype import ConstType, ImageDType, dtypes, DType, truncate
from tinygrad.helpers import ContextVar, all_int, prod, getenv, all_same, Context, partition, temp, unwrap, T, argfix, Metadata, _METADATA, flatten
from tinygrad.helpers import PICKLE_BUFFERS, dedup
if TYPE_CHECKING:
  from tinygrad.shape.shapetracker import ShapeTracker
  from tinygrad.device import Buffer

# wrapper around IntEnum that preserves Enum.__str__ and makes auto() unique across all FastEnum subclasses
class FastEnum(IntEnum):
  def __str__(self): return Enum.__str__(self)
  @staticmethod
  def _generate_next_value_(_, __, ___, last_values): return 1 + max([0, *last_values, *[max(c) for c in FastEnum.__subclasses__()]])

class SimpleMathTrait:
  # required to implement
  def alu(self:T, arg:Ops, *src) -> T: raise NotImplementedError
  def const_like(self:T, b:ConstLike) -> T: raise NotImplementedError

  # great functions you get!
  def ufix(self, x): return self.const_like(x) if not isinstance(x, MathTrait) else x
  def _binop(self, op, x, reverse): return self.ufix(x).alu(op, self) if reverse else self.alu(op, self.ufix(x))
  def logical_not(self): return self.ne(True)
  def neg(self):
    if (dtype:=getattr(self, 'dtype')) is None: raise TypeError(f"MathTraits __neg__ requires a dtype, {self=}")
    return self.logical_not() if dtype.scalar() == dtypes.bool else self*(-1)
  def add(self, x, reverse=False): return self._binop(Ops.ADD, x, reverse)
  def mul(self, x, reverse=False): return self._binop(Ops.MUL, x, reverse)
  def bitwise_and(self, x, reverse=False): return self._binop(Ops.AND, x, reverse)
  def bitwise_or(self, x, reverse=False): return self._binop(Ops.OR, x, reverse)
  def bitwise_xor(self, x, reverse=False): return self._binop(Ops.XOR, x, reverse)
  def idiv(self, x, reverse=False): return self._binop(Ops.IDIV, x, reverse)
  def mod(self, x, reverse=False): return self._binop(Ops.MOD, x, reverse)
  def sub(self, x, reverse=False): return self.ufix(x).alu(Ops.ADD, -self) if reverse else self.alu(Ops.ADD, self.ufix(-x))
  def div(self, x, reverse=False): return (self.ufix(x)*self.alu(Ops.RECIP)) if reverse else (self*self.ufix(x).alu(Ops.RECIP))

  def __neg__(self): return self.neg()

  def __add__(self, x): return self.add(x)
  def __sub__(self, x): return self.sub(x)
  def __mul__(self, x): return self.mul(x)
  def __truediv__(self, x): return self.div(x)
  def __floordiv__(self, x): return self.idiv(x)  # TODO: idiv is trunc div, not floordiv
  def __mod__(self, x): return self.mod(x)
  def __and__(self, x): return self.bitwise_and(x)
  def __or__(self, x): return self.bitwise_or(x)
  def __xor__(self, x): return self.bitwise_xor(x)

  def __radd__(self, x): return self.add(x, True)
  def __rsub__(self, x): return self.sub(x, True)
  def __rmul__(self, x): return self.mul(x, True)
  def __rtruediv__(self, x): return self.div(x, True)
  def __rfloordiv__(self, x): return self.idiv(x, True)
  def __rand__(self, x): return self.bitwise_and(x, True)
  def __ror__(self, x): return self.bitwise_or(x, True)
  def __rxor__(self, x): return self.bitwise_xor(x, True)
  def __rmod__(self, x): return self.mod(x, True)

  def __lt__(self, x): return self.alu(Ops.CMPLT, self.ufix(x))
  def __gt__(self, x): return self.ufix(x).alu(Ops.CMPLT, self)
  def __ge__(self, x): return (self < x).logical_not()
  def __le__(self, x): return (self > x).logical_not()

  def ne(self, x): return self.alu(Ops.CMPNE, self.ufix(x))
  def eq(self, x): return self.ne(x).logical_not()
  def __ne__(self, x): return self.ne(x)
  # NOTE: __eq__ isn't overridden, and means the same thing as is by default

class MathTrait(SimpleMathTrait):
  # TODO: move to Tensor when new backward is done
  def lshift(self, x, reverse=False): return self._binop(Ops.SHL, x, reverse)
  def rshift(self, x, reverse=False): return self._binop(Ops.SHR, x, reverse)
  def __lshift__(self, x): return self.lshift(x)
  def __rshift__(self, x): return self.rshift(x)
  def __rlshift__(self, x): return self.lshift(x, True)
  def __rrshift__(self, x): return self.rshift(x, True)

  def maximum(self, x): return self.alu(Ops.MAX, self.ufix(x))
  def minimum(self, x): return -(-self).maximum(-x)
  def where(self, x, y): return self.alu(Ops.WHERE, x, x.ufix(y))
  def threefry(self, seed): return self.alu(Ops.THREEFRY, seed)
  def reciprocal(self): return self.alu(Ops.RECIP)
  def sqrt(self): return self.alu(Ops.SQRT)
  def sin(self): return self.alu(Ops.SIN)
  def log2(self): return self.alu(Ops.LOG2)
  def exp2(self): return self.alu(Ops.EXP2)
  def pow(self, x): return self.alu(Ops.POW, self.ufix(x))

# the order of these Ops controls the order of the toposort
class Ops(FastEnum):
  # uops that aren't rendered
  NAME = auto(); SINK = auto(); CONTIGUOUS = auto(); CONTIGUOUS_BACKWARD = auto(); DETACH = auto(); KERNEL = auto(); UNIQUE = auto() # noqa: E702

  # TODO: empty continues to exist because of tensor
  EMPTY = auto()

  # MetaOps
  COPY = auto(); BUFFER_VIEW = auto() # noqa: E702

  # blocks in linearizer
  BLOCK = auto(); BLOCKSTART = auto(); BLOCKFORK = auto(); BLOCKEND = auto() # noqa: E702

  # movement ops!
  RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); FLIP = auto() # noqa: E702

  # misc ops
  UNROLL = auto(); CONTRACT = auto() # noqa: E702
  VIEW = auto(); DEFINE_GLOBAL = auto(); BUFFER = auto() # noqa: E702
  DEFINE_VAR = auto(); DEFINE_LOCAL = auto(); DEFINE_ACC = auto() # noqa: E702
  VALID = auto(); SPECIAL = auto(); NOOP = auto() # noqa: E702

  # reduce
  REDUCE_AXIS = auto()

  # helper ops
  GEP = auto(); VECTORIZE = auto(); CAT = auto() # noqa: E702

  # UnaryOps
  CAST = auto(); BITCAST = auto(); EXP2 = auto(); LOG2 = auto(); SIN = auto(); SQRT = auto(); RECIP = auto(); NEG = auto() # noqa: E702

  # load/store before math
  LOAD = auto(); STORE = auto() # noqa: E702

  # early INDEX
  INDEX = auto()

  # math ops
  WMMA = auto()

  # BinaryOps
  ADD = auto(); MUL = auto(); IDIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto(); CMPNE = auto(); XOR = auto() # noqa: E702
  SHL = auto(); SHR = auto(); OR = auto(); AND = auto(); THREEFRY = auto(); SUB = auto(); FDIV = auto(); POW = auto() # noqa: E702

  # TernaryOps
  WHERE = auto(); MULACC = auto() # noqa: E702

  # assignment ops
  ASSIGN = auto()
  BIND = auto()

  # control flow ops
  BARRIER = auto(); RANGE = auto(); IF = auto(); ENDRANGE = auto(); ENDIF = auto() # noqa: E702

  # consts last!
  VCONST = auto(); CONST = auto() # noqa: E702

  # device
  DEVICE = auto()
  MULTI = auto()
  CUSTOM = auto()

class GroupOp:
  Unary = {Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIP, Ops.NEG}
  Binary = {Ops.ADD, Ops.MUL, Ops.IDIV, Ops.MAX, Ops.MOD, Ops.CMPLT, Ops.CMPNE, Ops.XOR, Ops.SHL, Ops.SHR, Ops.OR, Ops.AND, Ops.THREEFRY,
            Ops.SUB, Ops.FDIV, Ops.POW}
  Ternary = {Ops.WHERE, Ops.MULACC}
  ALU = set.union(Unary, Binary, Ternary)

  Irreducible = {Ops.CONST, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.RANGE}
  Movement = {Ops.RESHAPE, Ops.EXPAND, Ops.PERMUTE, Ops.PAD, Ops.SHRINK, Ops.FLIP}

  Buffer = {Ops.LOAD, Ops.STORE, Ops.VALID, Ops.CONST, Ops.DEFINE_VAR}
  Block = {Ops.BLOCK, Ops.BLOCKEND, Ops.BLOCKFORK, Ops.BLOCKSTART}

  # BinaryOps that can be flipped
  Commutative = {Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPNE, Ops.XOR, Ops.AND, Ops.OR}

  # BinaryOps where f(f(a,b),c) = f(a,f(b,c))
  Associative = {Ops.ADD, Ops.MUL, Ops.AND, Ops.OR, Ops.MAX}

  # BinaryOps that satisfy f(x,x)=x see https://en.wikipedia.org/wiki/Idempotence
  Idempotent = {Ops.OR, Ops.AND, Ops.MAX}

  # do not preserve f(0) = 0
  UnsafePad = {Ops.RECIP, Ops.LOG2, Ops.EXP2, Ops.IDIV, Ops.POW}

  All = set(Ops)

# some BUFFER ops can be processed with only a view
view_supported_devices = {"LLVM", "CPU", "CUDA", "NV", "AMD", "METAL", "QCOM", "DSP", "DISK"}

# https://en.wikipedia.org/wiki/Identity_element
def identity_element(op:Ops, dt:DType) -> ConstType: return dtypes.as_const({Ops.ADD:0, Ops.MUL:1, Ops.MAX:dtypes.min(dt)}[op], dt)

def can_pad(u:UOp, edges:dict[UOp, None], cache:dict[UOp, None]) -> bool:
  if u.op in GroupOp.UnsafePad: return False
  if u in edges or u in cache: return True
  cache[u] = None
  return all(can_pad(x.base, edges, cache) for x in u.src)

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

def ssimplify(uop): return uop.ssimplify() if isinstance(uop, UOp) else uop
def sym_infer(uop: Union[UOp, int], var_vals: dict[UOp, int]) -> int: return uop.sym_infer(var_vals) if isinstance(uop, UOp) else uop

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
  def __call__(cls, op:Ops, dtype:DType=dtypes.void, src:tuple[UOp,...]=tuple(), arg:Any=None, _buffer:Buffer|None=None):
    if (wret:=UOpMetaClass.ucache.get(key:=(op, dtype, src, arg), None)) is not None and (ret:=wret()) is not None: return ret
    UOpMetaClass.ucache[key] = ref = weakref.ref(created:=super().__call__(*key))
    for s in src: s.children.add(ref)
    # NOTE: this will soon be set by Tensor once we remove function.py
    if (metadata:=_METADATA.get()) is not None: all_metadata[created] = metadata
    # NOTE: this value is set by pickle when pickling a realized tensor
    if _buffer is not None:
      assert op is Ops.BUFFER, f"trying to set Buffer {_buffer} for {op}"
      buffers[created] = _buffer
    return created

# some uops map to other stuff
buffers:weakref.WeakKeyDictionary[UOp, Buffer] = weakref.WeakKeyDictionary() # this maps BUFFER uops to their device Buffers
all_metadata:weakref.WeakKeyDictionary[UOp, Metadata] = weakref.WeakKeyDictionary()

# NOTE: this should be frozen, but frozen is slower
@dataclass(eq=False, slots=True)
class UOp(MathTrait, metaclass=UOpMetaClass):
  op:Ops
  dtype:DType = dtypes.void
  src:tuple[UOp, ...] = tuple()
  arg:Any = None
  children:set[weakref.ref[UOp]] = field(default_factory=set)
  def __del__(self):
    if self.op is Ops.BUFFER and (buffer:=buffers.get(self)) is not None: buffer.ref(-1)
    if (ref:=UOpMetaClass.ucache.get(k:=(self.op, self.dtype, self.src, self.arg))) is not None:
      for s in self.src: s.children.discard(ref)
      del UOpMetaClass.ucache[k]
  def __reduce__(self):
    args = [self.op, self.dtype, self.src, self.arg]
    if self.op is Ops.BUFFER and self.realized is not None and PICKLE_BUFFERS: args.append(self.realized)
    return UOp, tuple(args)
  def replace(self, **kwargs) -> UOp:
    new_args = (kwargs.pop("op", self.op), kwargs.pop("dtype", self.dtype), kwargs.pop("src", self.src), kwargs.pop("arg", self.arg))
    assert len(kwargs) == 0, f"unused kwargs in replace {list(kwargs)}"
    if (self.op, self.dtype, self.src, self.arg) == new_args: return self
    return UOp(*new_args)
  @functools.cached_property
  def key(self) -> bytes:
    return hashlib.sha256(str((self.op, self.dtype, self.arg)).encode() + b"".join([s.key for s in self.src])).digest()
  def __repr__(self): return pretty_print(self, lambda x: f"{type(self).__name__}({x.op}, {x.dtype}, arg={x.argstr()}, src=(%s))")
  def argstr(self): return f'({", ".join(map(str, self.arg))})' if self.op is Ops.REDUCE_AXIS else repr(self.arg)

  @property
  def toposort(self) -> dict[UOp, None]:
    def _toposort(u:UOp, cache:set[UOp]):
      if u in cache: return {}
      nodes: dict[UOp, None] = {}
      # NOTE: this is a lot faster than the comprehension in parents
      for parent in u.src: nodes.update(_toposort(parent, cache))
      nodes[u] = None
      cache.add(u)
      return nodes
    return _toposort(self, cache=set())

  @functools.cached_property
  def tuplize(self:UOp) -> tuple[int, Any, Optional[DType], tuple]: return (self.op.value, self.arg, self.dtype, tuple(x.tuplize for x in self.src))

  # *** uop shape stuff ***

  @functools.cached_property
  def st(self) -> ShapeTracker|None:
    from tinygrad.shape.shapetracker import ShapeTracker
    if self.op is Ops.MULTI:
      return ShapeTracker.from_shape(
        tuple(sum(y.shape[a] for y in self.real_lbs) if a == self.axis else s for a,s in enumerate(self.real_lbs[0].shape)))
    if self.op in {Ops.BUFFER, Ops.BUFFER_VIEW}: return ShapeTracker.from_shape((self.size,))
    if self.op is Ops.KERNEL: return ShapeTracker.from_shape(self.arg.ast.shape)
    # these ops define a ShapeTracker from the arg
    if self.op is Ops.VIEW: return self.arg
    if self.op in GroupOp.Movement: return unwrap(self.src[0].st).mop(self.op, self.arg)
    # buffer ops return the ShapeTracker from sources
    if self.op in GroupOp.Buffer: return vsrc[0] if len(vsrc:=[x.st for x in self.src if x.op is Ops.VIEW]) != 0 else None
    if not (src_sts := [x.st for x in self.src if x.st is not None]): return None
    assert all_same([x.shape for x in src_sts]), f"UOp sources must have the same shape {self} {[x.shape for x in src_sts]}"
    if self.op is Ops.BITCAST:
      shape = src_sts[0].shape
      if self.dtype.itemsize != (input_sz:=self.src[0].dtype.itemsize): shape = shape[:-1]+((shape[-1]*input_sz) // self.dtype.itemsize,)
    # only reduce ops are allowed to change shape, everything else derives shape from sources
    elif self.op in {Ops.REDUCE_AXIS, Ops.WMMA}: shape = src_sts[0].reduce(self.axis_arg)
    else: shape = src_sts[0].shape
    return ShapeTracker.from_shape(shape)

  @functools.cached_property
  def full_shape(self) -> tuple[sint, ...]:
    if self.op is Ops.VIEW: return self.shape
    # TODO: this should check if st is None, it cannot because local reduce has implicit movement ops
    return tuple(smax(x) for x in zip(*[x.full_shape for x in self.src if x.op not in {Ops.DEFINE_GLOBAL,Ops.DEFINE_LOCAL} \
        # TODO: this exists because wmma creates consts without ShapeTracker in the AST, there's probably a way to fix this
        and not (x.op is Ops.CONST and x.st is None)]))
  @property
  def shape(self) -> tuple[sint, ...]: return unwrap(self.st).shape
  @property
  def size(self) -> int: return self.arg[0] if self.op is Ops.BUFFER_VIEW else self.arg if self.op is Ops.BUFFER else unwrap(self.st).size

  # *** uop evaluation ***

  def simplify(self):
    # late import!
    from tinygrad.codegen.symbolic import symbolic
    with Context(TRACK_MATCH_STATS=0):
      return graph_rewrite(self, symbolic)
  def ssimplify(self) -> Union[UOp, ConstType]: return ret.arg if (ret:=self.simplify()).op is Ops.CONST else ret
  def _eval(self, dtype, expected_type:Type[T]) -> T:
    assert self.dtype in dtype, f"eval with wrong dtype {self}"
    vmin, vmax = (simple_self:=self.simplify())._min_max
    if vmin != vmax: raise ValueError(f"eval failed to be a single number, range is {vmin} to {vmax} in {simple_self.render()}")
    assert isinstance(vmin, expected_type), f"vmin is wrong dtype {type(vmin)} != {expected_type}"
    return vmin
  def __bool__(self): return self._eval((dtypes.bool,), bool)
  def __int__(self): return self._eval(dtypes.ints, int)
  def __float__(self): return self._eval(dtypes.floats, float)
  def substitute(self, dvars:dict[UOp, UOp]):
    with Context(TRACK_MATCH_STATS=0):
      return graph_rewrite(self, _substitute, dvars, bottom_up=True)

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
  def sink(self, *srcs:UOp): return UOp(Ops.SINK, dtypes.void, (self,)+srcs)
  def detach(self): return UOp(Ops.DETACH, self.dtype, (self,))
  def index(self, idx:UOp, valid:UOp|None=None): return UOp(Ops.INDEX, self.dtype, (self,idx,valid) if valid is not None else (self,idx))
  def const_like(self, b:ConstLike):
    # constants can optionally have a DEVICE source
    if self._device is None: return UOp.const(self.dtype, b)
    if isinstance(self.device, tuple): return UOp.multi(*[UOp.metaop(Ops.CONST, self.shape, self.dtype, d, b) for d in self.device], axis=None)
    return UOp.metaop(Ops.CONST, self.shape, self.dtype, self.device, b)
  def broadcast(self, count:int):
    assert self.dtype.count == 1
    if count == 1: return self
    return UOp(Ops.VECTORIZE, self.dtype.vec(count), (self,)*count)
  def cast(self, dtype:DType): return UOp(Ops.CAST, dtype, (self,))
  def bitcast(self, dtype:DType): return UOp(Ops.BITCAST, dtype, (self,))
  def gep(self, i:Union[tuple[int, ...], int]):
    if isinstance(i, int):
      # NOTE: these are just shortcuts to not have to create and fold later
      if self.op is Ops.VECTORIZE: return self.src[i]
      if self.op is Ops.VCONST: return UOp.const(self.dtype.scalar(), self.arg[i])
      if self.op is Ops.CONST: return UOp.const(self.dtype.scalar(), self.arg)
      i = (i,)
    if (self.dtype.vcount == len(i) and i == tuple(range(len(i)))) or self.dtype == dtypes.void: return self
    return UOp(Ops.GEP, self.dtype.scalar().vec(len(i)) if len(i) > 1 else self.dtype.scalar(), (self,), i)
  def load(self, *src:UOp, **kwargs): return UOp(Ops.LOAD, src=(self,)+src, **kwargs)
  def store(self, *src:UOp, **kwargs): return UOp(Ops.STORE, dtypes.void, (self,)+src, **kwargs)
  def alu(self, arg, *src:UOp):
    out_dtype = (self, *src)[-1].dtype
    if arg in {Ops.CMPLT, Ops.CMPNE}: out_dtype = dtypes.bool.vec(out_dtype.count) if out_dtype.count > 1 else dtypes.bool
    return UOp(arg, out_dtype, (self,)+src)
  @staticmethod
  def const(dtype:DType, b:ConstLike):
    if isinstance(b, UOp): return b.unbind()[0] if b.op is Ops.BIND else b
    if isinstance(b, tuple) and all_same(b): b = b[0]  # doesn't have to be a VCONST if they are all the same
    return UOp(Ops.VCONST if isinstance(b, tuple) else Ops.CONST, dtype, arg=dtypes.as_const(b, dtype))
  def valid(self, st:ShapeTracker):
    assert self.op in {Ops.CONST, Ops.DEFINE_VAR}, f"can only create VALID from a constant, got {self.op}"
    from tinygrad.shape.shapetracker import ShapeTracker
    # NOTE: only VALID has a masked ShapeTracker, the CONST operands are unmasked
    unmasked_st = ShapeTracker.from_shape(()).reshape((1,)*len(st.shape)).expand(st.shape).to_uop()
    return UOp(Ops.VALID, dtypes.bool, (st.to_uop(),)).where(self.replace(src=(unmasked_st,)), UOp.const(self.dtype, 0).replace(src=(unmasked_st,)))
  @staticmethod
  def range(dtype:DType, start:sint, end:sint, idx:int): return UOp(Ops.RANGE, dtype=dtype, src=(sint_to_uop(start), sint_to_uop(end)), arg=idx)
  def r(self, op:Ops, axis:tuple[int, ...]):
    axis = tuple(sorted([x for x in axis if resolve(self.shape[x] != 1)]))
    return self if len(axis) == 0 else UOp(Ops.REDUCE_AXIS, self.dtype, (self,), (op, axis))
  def assign(self, x:UOp): return UOp(Ops.ASSIGN, self.dtype, (self,x))
  def contiguous(self): return self.alu(Ops.CONTIGUOUS)
  def contiguous_backward(self): return self.alu(Ops.CONTIGUOUS_BACKWARD)

  # *** from MultiLazyBuffer ***

  def multi(self, *more:UOp, axis:int|None, real:tuple[bool,...]|None=None):
    parents = (self,)+more
    assert all_same([x.dtype for x in parents]), "multi parents must have the same dtype"
    return UOp(Ops.MULTI, self.dtype, parents, (axis, real if real is not None else (True,)*len(parents)))

  @property
  def bounds(self):
    if self.axis is None: raise RuntimeError("bounds is not defined when axis is None")
    return tuple(itertools.pairwise(itertools.accumulate([lb.shape[self.axis] for lb in self.src], initial=0)))

  @functools.cached_property
  def axis(self) -> Optional[int]:
    if self.op is Ops.MULTI: return self.arg[0]
    # NOTE: they all have to share an axis, we always choose [-1]
    if self.op in GroupOp.ALU: return axes[-1] if (axes := dedup([x.axis for x in self.src if x.axis is not None])) else None
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

  @property
  def real(self):
    assert self.op is Ops.MULTI
    return self.arg[1]

  @property
  def real_lbs(self): return [lb for lb,r in zip(self.src, self.real) if r]

  def shard(self, devices:tuple[str, ...], axis:Optional[int]=None) -> UOp:
    if axis is None: lbs = [self] * len(devices)
    else:
      if self.shape[axis] % len(devices) != 0: raise RuntimeError(f"multi axis uneven: {self.shape[axis]=} {axis=} {len(devices)=}")
      # NOTE: this works for both even shards and uneven shards
      sz = self.shape[axis] // len(devices)
      sizes = [max(0, min(sz, self.shape[axis] - sz*i)) for i in range(len(devices))]
      lbs = []
      for sz,off in zip(sizes, itertools.accumulate(sizes, initial=0)):
        lbs.append(self.shrink(tuple((0,s) if i != axis else (off,off+sz) for i,s in enumerate(self.shape))))
    sharded_lbs = [lb.copy_to_device(d) for lb,d in zip(lbs, devices)]
    return UOp.multi(*[lb.contiguous() for lb in sharded_lbs], axis=axis)

  # *** from LazyBuffer ***

  @staticmethod
  def metaop(op:Ops, shape:tuple[sint, ...], dtype:DType, device:str, arg=None) -> UOp:
    from tinygrad.shape.shapetracker import ShapeTracker
    # Tensor const is CONST(VIEW(DEVICE)) -> RESHAPE -> EXPAND
    if op is Ops.CONST:
      assert isinstance(arg, get_args(ConstType)), f"trying to create CONST with {arg=}"
      return UOp.const(dtype, unwrap(arg)).replace(src=(UOp(Ops.VIEW, dtypes.void, (UOp(Ops.DEVICE, arg=device),),
                 ShapeTracker.from_shape(())),)).reshape((1,)*len(shape)).expand(shape)
    # Tensor variable binding is BIND(VAR(VIEW(DEVICE)), CONST(VIEW(DEVICE)))
    if op is Ops.BIND:
      var, val = arg.unbind()
      return var.replace(src=(UOp(Ops.VIEW, dtypes.void, (UOp(Ops.DEVICE, arg=device),), ShapeTracker.from_shape(shape)),)).bind(val)
    # otherwise it's just a RESHAPE(BUFFER)
    if not isinstance(size:=prod([x.vmax if isinstance(x, UOp) else x for x in shape]), int): raise ValueError(f"size must be int {size}")
    return UOp.new_buffer(device, size, dtype).reshape(shape)
  def copy_to_device(self, device:str|tuple[str, ...], clone:bool=False): return UOp(Ops.COPY, self.dtype, (UOp(Ops.DEVICE, arg=device), self), clone)
  def clone(self) -> UOp: return self.copy_to_device(self.device, clone=True)
  @property
  def metadata(self) -> tuple[Metadata, ...]|Metadata|None: return self.arg.metadata if self.op is Ops.KERNEL else all_metadata.get(self, None)

  # *** uop movement ops ***

  @property
  def base(self) -> UOp:
    if (self.op is Ops.VIEW and len(self.src) != 0) or self.op in GroupOp.Movement: return self.src[0].base
    return self
  def view(self, new_st:ShapeTracker) -> UOp: return UOp(Ops.VIEW, self.dtype, (self.base,), new_st)

  def _mop(self, op:Ops, arg):
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
  def new_buffer(device:str, size:int, dtype:DType): return UOp(Ops.BUFFER, dtype, (UOp(Ops.DEVICE, arg=device), UOp.unique()), size)
  @property
  def device(self) -> str|tuple[str, ...]: return cast(str|tuple[str, ...], unwrap(self._device))
  @functools.cached_property
  def _device(self) -> Optional[str|tuple[str, ...]]:
    if self.op is Ops.DEVICE: return self.arg
    if self.op is Ops.MULTI: return tuple(cast(str, x.device) for x in self.src)
    return dsrcs[0]._device if len(dsrcs:=[x for x in self.src if x._device is not None]) != 0 else None
  @property
  def buf_uop(self) -> UOp:
    assert self.op is Ops.ASSIGN, f"must be ASSIGN {self.op}"
    return self.src[0].base
  @property
  def buffer(self) -> Buffer:
    if self is not self.base:
      assert unwrap(self.st).contiguous, "VIEW only works here if it's contiguous"
      return self.src[0].buffer
    assert self.op is Ops.BUFFER, f"must be BUFFER {self.op}"
    if (cret:=buffers.get(self)) is not None: return cret
    from tinygrad.device import Buffer
    assert isinstance(self.device, str), f"buffer not supported on multi {self.device}"
    buffers[self] = ret = Buffer(self.device, self.size, self.dtype if isinstance(self.dtype, ImageDType) else self.dtype.base)
    return ret
  @property
  def realized(self) -> Optional[Buffer]: return self.buffer if self.op is Ops.BUFFER and self.buffer.is_allocated() else None
  @property
  def is_realized(self) -> bool:
    return all(x.base.realized is not None for x in self.base.real_lbs) if self.base.op is Ops.MULTI else self.base.realized is not None

  # *** uop Variable stuff ***

  @staticmethod
  def variable(name:str, min_val:ConstType, max_val:ConstType, dtype:DType=dtypes.int):
    assert not isinstance(min_val, UOp) and not isinstance(max_val, UOp), f"can't create Variable {name} with {min_val}/{max_val}"
    return UOp(Ops.DEFINE_VAR, dtype, arg=(name, min_val, max_val))
  @property
  def expr(self):
    assert self.op is Ops.DEFINE_VAR, f"op is {self.op}, need DEFINE_VAR"
    return self.arg[0]
  def bind(self, val:int):
    assert self.op is Ops.DEFINE_VAR, f"op is {self.op}, need DEFINE_VAR"
    assert self.arg[1] <= val and val <= self.arg[2], f"bind {val} not in range [{self.arg[1]}, {self.arg[2]}]"
    return UOp(Ops.BIND, self.dtype, (self, self.const_like(val)))
  def unbind(self) -> tuple[Variable, int]:
    assert self.op is Ops.BIND and self.src[0].op is Ops.DEFINE_VAR and self.src[1].op is Ops.CONST, f"can't unbind {self}"
    return self.src[0], self.src[1].arg
  @property
  def val(self) -> int: return self.unbind()[1]
  def vars(self) -> set[UOp]:
    bound_vars = set([x for x in self.toposort if x.op is Ops.BIND and x.src[0].op is Ops.DEFINE_VAR])
    bound_var_base = set(x.src[0] for x in bound_vars)
    all_vars = set([x for x in self.toposort if x.op is Ops.DEFINE_VAR])
    return bound_vars.union(set([x for x in all_vars if x not in bound_var_base]))
  def variables(self) -> list[Variable]:
    st_vars: list[set[Variable]] = [x.st_arg.vars() for x in self.toposort if x.op in GroupOp.Buffer]
    return sorted(set.union(*st_vars, [x.unbind()[0] if x.op is not Ops.DEFINE_VAR else x for x in self.vars()]), key=lambda v: v.arg)

  # *** uop symbolic stuff ***

  def is_increasing(self:UOp) -> bool:
    # is f a monotonically increasing function regards its input
    if self.op in GroupOp.Irreducible: return True
    if self.op is Ops.ADD: return self.src[0].is_increasing() and self.src[1].is_increasing()
    if self.op in (Ops.MUL, Ops.IDIV) and self.src[1].op is Ops.CONST and self.src[1].arg >= 0: return self.src[0].is_increasing()
    return False  # False if not sure
  def const_factor(self) -> int:
    """largest known int that divides self"""
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
      if self.op is Ops.MUL: return min(vals:=(s0_vmin*s1_vmin, s0_vmin*s1_vmax, s0_vmax*s1_vmin, s0_vmax*s1_vmax)), max(vals)
      # SHL/SHR on consts only
      if self.op is Ops.SHL and s1_vmin == s1_vmax and all_int(t:=(s0_vmin, s0_vmax, s1_vmin)): return t[0] << t[2], t[1] << t[2]
      if self.op is Ops.SHR and s1_vmin == s1_vmax and all_int(t:=(s0_vmin, s0_vmax, s1_vmin)): return t[0] >> t[2], t[1] >> t[2]
      if self.op is Ops.MOD and s1_vmin > 0:
        return (0, s1_vmax-1) if s0_vmin >= 0 else (-(s1_vmax-1), s1_vmax-1)
      if self.op is Ops.IDIV:
        if s1_vmin == s1_vmax:  # min/max are equal in a CONST
          if s1_vmin > 0: return s0_vmin//s1_vmin, s0_vmax//s1_vmin
          if s1_vmin < 0 and s0_vmin >= 0: return -(s0_vmax//-s1_vmin), -(s0_vmin//-s1_vmin)
        # don't know exact bounds, but know the sign
        if (s0_vmax <= 0 and s1_vmin < 0) or (s0_vmin >= 0 and s1_vmin > 0): return 0, dtypes.max(self.dtype)
        if (s0_vmax <= 0 and s1_vmin > 0) or (s0_vmin >= 0 and s1_vmin < 0): return dtypes.min(self.dtype), 0
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
    if self.op is Ops.RANGE: return self.src[0].vmin, (self.src[1]-1).vmax
    if self.op is Ops.BIND: return self.src[0]._min_max # ignore the bound value
    if self.op in {Ops.UNROLL, Ops.VECTORIZE}: return min(x.vmin for x in self.src), max(x.vmax for x in self.src)
    # TODO: Ops.SPECIAL is Ops.DEFINE_VAR
    if self.op is Ops.SPECIAL: return 0, self.arg[1]-1 if isinstance(self.arg[1], int) else self.arg[1].vmax
    if self.op is Ops.CONST: return self.arg, self.arg
    if self.op is Ops.VCONST: return (min(self.arg), max(self.arg))
    return dtypes.min(self.dtype), dtypes.max(self.dtype)

  @functools.cached_property
  def _sym_fxn(self):
    sself = self.simplify()
    varnames = tuple(x.arg[0] for x in sself.toposort if x.op is Ops.DEFINE_VAR)
    # TODO: sanitize varnames, or don't use naked eval while staying fast
    return eval("lambda "+','.join(varnames)+": "+sself.render()), varnames  # pylint: disable=eval-used

  def sym_infer(self, var_vals:dict[UOp, int]):
    fxn, varnames = self._sym_fxn
    return fxn(**{k.arg[0]:v for k,v in var_vals.items() if k.arg[0] in varnames})

  def render(self, simplify=True) -> str:
    ret = graph_rewrite(self.simplify() if simplify else self, renderer)
    return ret.arg if ret.op is Ops.NOOP else str(ret)

@dataclass(frozen=True)
class KernelInfo:
  name: str = "test"            # name of the kernel
  local_dims: int = 0           # number of local dimensions  (this is remapping RANGE to SPECIAL)
  upcasted: int = 0             # count that are upcasted     (this is remapping RANGE to UNROLL)
  dont_use_locals: bool = False # don't use local indexing

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
  Ops.MOD: lambda x,y: abs(int(x))%abs(int(y))*(1,-1)[x<0], Ops.IDIV: lambda x,y: abs(x)//abs(y)*(1,-1)[x*y<0] if y != 0 else 0,
  Ops.MULACC: lambda x,y,z: (x*y)+z, Ops.WHERE: lambda x,y,z: y if x else z}

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
  # find the real frame in the file that has the UPat, TODO: is there a better way to do this?
  while frm.f_back is not None and pathlib.Path(frm.f_back.f_code.co_filename).name in {"ops.py", "rewriter.py", "schedule.py", "multi.py",
                                                                                        "symbolic.py", "expander.py", "lowerer.py", "cstyle.py",
                                                                                        "linearize.py"}:
    frm = frm.f_back
  return frm.f_code.co_filename, frm.f_lineno
@functools.lru_cache(None)
def lines(fn) -> list[str]:
  with open(fn) as f: return f.readlines()

class UPat(MathTrait):
  __slots__ = ("op", "dtype", "arg", "name", "src")
  def __init__(self, op:Optional[Union[Ops, tuple[Ops, ...], set[Ops]]]=None, dtype:Optional[Union[DType, tuple[DType, ...]]]=None,
               src:Optional[Union[tuple[UPat, ...], list[UPat], UPat]]=None, arg:Any=None,
               name:Optional[str]=None, allow_any_len:bool=False, location=None, custom_early_reject:Optional[set[Ops]]=None):
    assert op is None or isinstance(op, Ops) or isinstance(op, tuple) or isinstance(op, set), "op must be Ops or tuple of Ops"
    self.op: Optional[tuple[Ops, ...]] = (op,) if isinstance(op, Ops) else (tuple(op) if isinstance(op, set) else op)
    self.dtype: Optional[tuple[DType, ...]] = (dtype,) if isinstance(dtype, DType) else dtype
    self.arg, self.name, self._in_src, self.custom_early_reject = arg, name, src, custom_early_reject
    self.src: Any = None
    assert self.name != "ctx", "UPat can't be named ctx"

    # try all permutations if it's a list
    if isinstance(src, list): self.src = list(itertools.permutations(src)) if not all_same(src) else [src]
    # only one if it's a tuple
    elif isinstance(src, tuple): self.src = [src]
    # repeat if it's a UPat
    elif isinstance(src, UPat): self.src = [itertools.repeat(src)]

    self.allowed_len: int = -1 if allow_any_len or isinstance(src, UPat) or src is None else len(src)
    self.location = location or get_location()

    if custom_early_reject is not None: self.early_reject = custom_early_reject
    else:
      upat_match = [src] if isinstance(src, UPat) else ([] if src is None else self.src[0])
      self.early_reject = {pp.op[0] for pp in upat_match if pp.op is not None and len(pp.op) == 1}

  def named(self, name:str): return UPat(self.op, self.dtype, self._in_src, self.arg, name, self.allowed_len == -1, self.custom_early_reject)

  @staticmethod
  def any(*src): return UPatAny(src=src)

  @staticmethod
  @functools.lru_cache(None)
  def var(name:Optional[str]=None, dtype:Optional[Union[DType, tuple[DType, ...]]]=None): return UPat(dtype=dtype, name=name)
  @staticmethod
  @functools.lru_cache(None)
  def cvar(name:Optional[str]=None, dtype:Optional[DType]=None, vec=True): return UPat((Ops.CONST,Ops.VCONST) if vec else Ops.CONST, dtype, name=name)
  @staticmethod
  def const(dtype:Optional[Union[DType, tuple[DType, ...]]], b:ConstType): return UPat(Ops.CONST, dtype=dtype, arg=b)

  # copied from UOp
  def index(self, idx:UPat, valid:Optional[UPat]=None): return UPat(Ops.INDEX, self.dtype, (self,idx,valid) if valid is not None else (self,idx))
  def view(self, st=None, **kwargs): return UPat(Ops.VIEW, self.dtype, (self,), st, **kwargs)
  def cast(self, dtype=None): return UPat(Ops.CAST, dtype, (self,))
  def bitcast(self, dtype=None): return UPat(Ops.BITCAST, dtype, (self,))
  def gep(self, i:int): return UPat(Ops.GEP, None, (self,), (i,))
  def load(self, *src:UPat, **kwargs): return UPat(Ops.LOAD, src=(self,)+src, **kwargs)
  def store(self, *src:UPat, **kwargs): return UPat(Ops.STORE, dtypes.void, (self,)+src, **kwargs)
  def assign(self, x:UPat, **kwargs): return UPat(Ops.ASSIGN, self.dtype, (self,x), **kwargs)

  def const_like(self, b:ConstLike): return UPat.const(self.dtype, cast(ConstType, b))
  def alu(self, op:Ops, *src:UPat):
    asrc = (self,)+src
    return UPat(op, dtypes.bool if op in {Ops.CMPLT, Ops.CMPNE} else asrc[-1].dtype, list(asrc) if op in GroupOp.Commutative else asrc)

  def printable(self:UPat) -> str:
    try: return lines(self.location[0])[self.location[1]-1].strip()
    except FileNotFoundError: return "<missing>"

  def __repr__(self):
    def rep(x):
      form = "UPat(%s, %s, name=%s, dtype=%s, allow_any_len=%s, src=%s)"
      return form % (None if x.op is None else ('(%s)'%', '.join(map(str, x.op))), x.arg, repr(x.name),
        set(x.dtype) if x.dtype else None, x.allowed_len == 0, "[%s]" if x.src and len(x.src)>1 else "(%s)")
    return pretty_print(self, rep, srcfn=lambda x:None if x.src is None else [next(x.src[0])] if isinstance(x.src[0], itertools.repeat) else x.src[0])

  def match(self:UPat, uop:UOp, store:dict[str, UOp]) -> list[dict[str, UOp]]:
    if (self.op is not None and uop.op not in self.op) or \
       (self.name is not None and store.setdefault(self.name, uop) is not uop) or \
       (self.dtype is not None and uop.dtype not in self.dtype and uop.dtype.scalar() not in self.dtype) or \
       (self.arg is not None and self.arg != uop.arg) or \
       (self.allowed_len != -1 and len(uop.src) != self.allowed_len): return []
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

class PatternMatcher:
  def __init__(self, patterns:list[tuple[UPat, Callable]]):
    self.patterns = patterns
    # NOTE: use of DefaultDict here is very dangerous! all keys will live for the lifetime of the PatternMatcher!
    self.pdict: dict[Ops, list[tuple[UPat, Callable, set, bool]]] = {}
    # uop is required, arg is optional
    for p,fxn in self.patterns:
      assert p.op is not None
      tuple_fxn = fxn if isinstance(fxn, tuple) else deconstruct_function(fxn)
      real_fxn = types.FunctionType(*tuple_fxn)
      for uop in p.op: self.pdict.setdefault(uop, []).append((p, real_fxn, p.early_reject, 'ctx' in inspect.signature(real_fxn).parameters))

  def __reduce__(self): return PatternMatcher, ([(x,deconstruct_function(fxn) if fxn.__name__ == "<lambda>" else fxn) for x,fxn in self.patterns],)

  @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none
  def __add__(self, more:PatternMatcher): return PatternMatcher(self.patterns+more.patterns)

  def rewrite(self, uop:UOp, ctx=None) -> UOp|None:
    ler = {u.op for u in uop.src}
    for p,fxn,early_reject,has_ctx in self.pdict.get(uop.op, []):
      if not early_reject.issubset(ler): continue
      for match in p.match(uop, {}):
        if (ret:=(fxn(ctx=ctx, **match) if has_ctx else fxn(**match))) is not None: return ret
    return None

# *** tracking pattern matcher ***

TRACK_MATCH_STATS = ContextVar("TRACK_MATCH_STATS", 2 if getenv("VIZ") else 0)
match_stats:dict[UPat, list[Union[int, float]]] = dict()
@dataclass(frozen=True)
class TrackedGraphRewrite:
  loc: tuple[str, int]                                                                       # location that called graph_rewrite
  sink: UOp                                                                                  # the sink input to graph_rewrite
  bottom_up: bool
  matches: list[tuple[UOp, UOp, UPat]] = field(default_factory=list)                         # before+after of all the matches
  name: Optional[str] = None
tracked_keys:list[Any] = []
tracked_ctxs:list[list[TrackedGraphRewrite]] = []
_name_cnt:dict[str, int] = {}
def track_rewrites(named=False, name_fxn:Callable|None=None):
  def _decorator(func):
    def __wrapper(self, *args, **kwargs):
      if TRACK_MATCH_STATS >= 2:
        if (count_names:=(named or name_fxn)): _name_cnt[func.__name__] = _name_cnt.get(func.__name__, 0)+1
        tracked_keys.append(f"{func.__name__}_{_name_cnt[func.__name__]}" if count_names else self)
        tracked_ctxs.append([])
      ret = func(self, *args, **kwargs)
      if TRACK_MATCH_STATS >= 2 and name_fxn is not None: tracked_keys[-1] = f"{name_fxn(ret)} n{_name_cnt[func.__name__]}"
      return ret
    return __wrapper
  return _decorator

class TrackedPatternMatcher(PatternMatcher):
  def rewrite(self, uop:UOp, ctx=None) -> UOp|None:
    ret = None
    ler = {u.op for u in uop.src}
    for p,fxn,early_reject,has_ctx in self.pdict.get(uop.op, []):
      if p not in match_stats: match_stats[p] = [0,0,0.0,0.0]
      st = time.perf_counter()
      if not early_reject.issubset(ler):
        match_stats[p][2] += time.perf_counter()-st
        continue
      match_stats[p][1] += 1
      for match in p.match(uop, {}):
        if (ret:=(fxn(ctx=ctx, **match) if has_ctx else fxn(**match))) is not None:
          match_stats[p][0] += 1
          match_stats[p][3] += (et:=time.perf_counter()-st)
          if TRACK_MATCH_STATS >= 3: print(f"{et*1e6:7.2f} us -- ", p.printable())
          if TRACK_MATCH_STATS >= 2 and isinstance(ret, UOp) and len(tracked_ctxs) != 0: tracked_ctxs[-1][-1].matches.append((uop, ret, p))
          return ret # NOTE: if it returns None, we keep trying to match
      match_stats[p][2] += time.perf_counter()-st
    return None

if TRACK_MATCH_STATS:
  PatternMatcher = TrackedPatternMatcher  # type: ignore
  import atexit
  @atexit.register
  def print_match_stats():
    if TRACK_MATCH_STATS >= 2:
      with open(fn:=temp("rewrites.pkl", append_user=True), "wb") as f:
        print(f"rewrote {len(tracked_ctxs)} graphs and matched {sum(len(r.matches) for x in tracked_ctxs for r in x)} times, saved to {fn}")
        with Context(PICKLE_BUFFERS=0): pickle.dump((tracked_keys, tracked_ctxs), f)
    if getenv("VIZ"): launch_viz("VIZ", temp("rewrites.pkl", append_user=True))
    if getenv("PRINT_MATCH_STATS", 1):
      ret = [0,0,0.0,0.0]
      for k,v in sorted(list(match_stats.items()), key=lambda x: x[1][2]+x[1][3]):
        loc_str = f"{k.location[0].split('/')[-1]}:{k.location[1]}"
        if v[1] != 0: print(f"{v[0]:6d} / {v[1]:7d} -- {v[3]*1000.:9.2f} / {(v[2]+v[3])*1000.:9.2f} ms -- {loc_str:15s}", k.printable())
        ret = [x+y for x,y in zip(ret, v)]
      print(f"{ret[0]:6d} / {ret[1]:7d} -- {ret[3]*1000.:9.2f} / {(ret[2]+ret[3])*1000.:9.2f} ms -- TOTAL")

def launch_viz(env_str:str, data:str):
  os.environ[env_str] = "0"
  os.environ[f"{env_str}_DATA"] = data
  if not int(os.getenv("VIZ", "0")) and not int(os.getenv("PROFILE", "0")):
    args = ['--kernels', getenv("VIZ_DATA", "")] if getenv("VIZ_DATA", "") else []
    args += ['--profile', getenv("PROFILE_DATA", "")] if getenv("PROFILE_DATA", "") else []
    os.execv(sys.executable, [sys.executable] + [os.path.join(os.path.dirname(__file__), ".", "viz", "serve.py")] + args)

# *** simple graph rewrite engine ***

class RewriteContext:
  def __init__(self, pm, ctx=None):
    self.pm: PatternMatcher = pm
    self.ctx = ctx
    self.replace: dict[UOp, UOp] = {}
  def top_down_rewrite(self, n:UOp) -> UOp:
    if (rn := self.replace.get(n)) is not None: return rn
    new_src = tuple([self.top_down_rewrite(x) for x in n.src])
    new_n = self.pm.rewrite(n, self.ctx) if new_src == n.src else UOp(n.op, n.dtype, new_src, n.arg)
    self.replace[n] = ret = n if new_n is None else self.top_down_rewrite(new_n)
    return ret
  def bottom_up_rewrite(self, n:UOp) -> UOp:
    if (rn := self.replace.get(n)) is not None: return rn
    new_n: UOp|None = n
    while new_n is not None: last_n, new_n = new_n, self.pm.rewrite(new_n, self.ctx)
    new_src = tuple([self.bottom_up_rewrite(x) for x in last_n.src])
    self.replace[n] = ret = last_n if new_src == last_n.src else self.bottom_up_rewrite(UOp(last_n.op, last_n.dtype, new_src, last_n.arg))
    return ret

def graph_rewrite(sink:UOp, pm:PatternMatcher, ctx=None, bottom_up=False, name=None) -> UOp:
  if TRACK_MATCH_STATS >= 2 and len(tracked_ctxs) != 0:
    tracked_ctxs[-1].append(TrackedGraphRewrite(((frm:=sys._getframe(1)).f_code.co_filename, frm.f_lineno), sink, bottom_up, name=name))
  return RewriteContext(pm, ctx).bottom_up_rewrite(sink) if bottom_up else RewriteContext(pm, ctx).top_down_rewrite(sink)

def graph_rewrite_map(sink:UOp, pm:PatternMatcher, ctx=None, bottom_up=False, name=None) -> dict[UOp, UOp]:
  if TRACK_MATCH_STATS >= 2 and len(tracked_ctxs) != 0:
    tracked_ctxs[-1].append(TrackedGraphRewrite(((frm:=sys._getframe(1)).f_code.co_filename, frm.f_lineno), sink, bottom_up, name=name))
  rewrite_ctx = RewriteContext(pm, ctx)
  return {k:(rewrite_ctx.bottom_up_rewrite(k) if bottom_up else rewrite_ctx.top_down_rewrite(k)) for k in list(sink.toposort)[::-1]}

def sint_to_uop(x:sint, dtype:DType=dtypes.int) -> UOp: return UOp.const(dtype, x) if isinstance(x, int) else x

_substitute = PatternMatcher([(UPat(tuple(Ops), name="x"), lambda ctx,x: ctx.get(x,None))])

# for debug
syms = { Ops.ADD: "+", Ops.SUB: "-", Ops.IDIV: "//", Ops.MOD: "%", Ops.SHL: "<<", Ops.SHR: ">>",
         Ops.MUL: "*", Ops.CMPLT: "<", Ops.CMPNE: "!=", Ops.AND: "&", Ops.OR: "|", Ops.XOR: "^"}
renderer = PatternMatcher([
  (UPat((Ops.DEFINE_VAR, Ops.SPECIAL), name="x"), lambda x: UOp(Ops.NOOP, arg=x.arg[0])),
  (UPat(Ops.RANGE, name="x"), lambda x: UOp(Ops.NOOP, arg=f"ridx{x.arg}")),
  (UPat(Ops.CONST, name="x"), lambda x: UOp(Ops.NOOP, arg=str(x.arg))),
  (UPat(Ops.BIND, src=UPat(Ops.NOOP), name="x"), lambda x: x.src[0]),
  (UPat(Ops.NEG, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"(-{x.src[0].arg})")),
  (UPat(Ops.MAX, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"max({x.src[0].arg}, {x.src[1].arg})")),
  (UPat(Ops.MULACC, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"({x.src[0].arg}*{x.src[1].arg}+{x.src[2].arg})")),
  (UPat(Ops.WHERE, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"({x.src[1].arg} if {x.src[0].arg} else {x.src[2].arg})")),
  (UPat(GroupOp.ALU, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"({x.src[0].arg}{syms[x.op]}{x.src[1].arg})")),
])

# *** what was symbolic.py ***

sint = Union[int, UOp]
Variable = UOp

ConstLike = Union[ConstType, Variable, tuple[ConstType, ...]]

# *** UOp merge views and swizzling ***

merge_views = PatternMatcher([
  # VIEW(VIEW) merges to a single VIEW
  (UPat(Ops.VIEW, name="vm1", src=(UPat(Ops.VIEW, name="vm2"),)), lambda vm1,vm2: vm2.replace(arg=vm2.st+vm1.st)),
  # remove VIEW if it's contiguous and same as the base shape
  (UPat(Ops.VIEW, name="vm", src=(UPat(GroupOp.All-{Ops.DEVICE}, name="x"),)), lambda vm,x: x if vm.st.contiguous and x.shape == vm.shape else None),
  # merge unmasked const views
  (UPat(Ops.VIEW, name="view", src=(UPat((Ops.CONST, Ops.DEFINE_VAR), name="const", src=(UPat(Ops.VIEW, name="st"),) ),)),
   lambda st,const,view: const.replace(src=(st.replace(arg=st.st+view.st),)) if all(v.mask is None for v in (st.st+view.st).views) else None),
])

# push VIEW to parents
view_left = merge_views+PatternMatcher([
  # VIEW(CONST) becomes VALID
  (UPat(Ops.VIEW, name="vm", src=(UPat((Ops.CONST, Ops.DEFINE_VAR), name="x"),)), lambda vm,x: x.valid(vm.st)),
  # VIEW before elementwise/buffer ops
  (UPat(Ops.VIEW, name="vm", src=(UPat({*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN}, name="e"),)),
   lambda e,vm: e.replace(src=tuple(s if s.st is None else s.view(vm.st) if s is s.base else s.base.view(s.st+vm.st) for s in e.src))),
  (UPat(Ops.VIEW, name="vm", src=(UPat(GroupOp.Buffer, name="b"),)),
   lambda b,vm: b.replace(src=tuple((s.st+vm.st).to_uop() if s.op is Ops.VIEW else s for s in b.src))),
])
