from enum import Enum, auto
import functools
from typing import Callable
from dataclasses import dataclass
from tinygrad.dtype import AddrSpace, DType
from tinygrad.mixin import MathMixin
from tinygrad.uop.ops import UOp, Ops

from extra.thunder.tiny.tk import WARP_THREADS

def unwrap(x):
  if hasattr(x, "_uop"): return x._uop
  if isinstance(x, (list, tuple)): return type(x)(unwrap(y) for y in x)
  if isinstance(x, dict): return {k: unwrap(v) for k,v in x.items()}
  return x

def wrap(x, s):
  if isinstance(x, UOp): return s.ruop(x)
  if isinstance(x, (list, tuple)): return type(x)(wrap(y, s) for y in x)
  return x

def autowrap(source_cls, blacklist=None):
  if blacklist is None:
    blacklist = {
      "__init__", "__new__", "__str__", "__del__", "__repr__", "__dict__", "__getattribute__",
      "__setattr__", "__delattr__", "__weakref__", "__slots__", "__class__",
      "__reduce__", "__reduce_ex__", "__getstate__", "__setstate__", "__hash__"
    }

  def decorator(cls):
    def __getattr__(self, name):
      uop = object.__getattribute__(self, "_uop")
      val = getattr(uop, name)
      if callable(val):
        @functools.wraps(val)
        def proxy(*args, **kwargs):
          return wrap(val(*unwrap(args), **unwrap(kwargs)), self)
        return proxy
      if name in UOp.__slots__: return val # type: ignore
      return wrap(val, self)
    cls.__getattr__ = __getattr__

    for name in dir(source_cls):
      if name in blacklist or not name.startswith("__"): continue

      for base in cls.mro():
        if base is source_cls: break
        if name in base.__dict__: break
      else:
        original = getattr(source_cls, name)
        if callable(original):
          def make_proxy(_, func):
            def proxy(self, *args, **kwargs):
              return wrap(func(self._uop, *unwrap(args), **unwrap(kwargs)), self)
            return proxy
          setattr(cls, name, make_proxy(name, original))

    return cls
  return decorator

class TileMathMixin(MathMixin):
  def alu(self, op, *src, inner_op=lambda x:x):
    assert isinstance(self, (RT, RV))
    if len(src) == 0:
      if self._uop._shape is None: uop = UOp.alu(self._uop, op)
      else: uop = self.ker.warp.map(self._uop, lambda x: UOp.alu(x, op))
    elif len(src) == 1:
      if self._uop._shape is None: uop = UOp.alu(self._uop, op, inner_op(self._uop.ufix(src[0])))
      elif isinstance(src[0], (int,float,bool)): uop = self.ker.warp.map(self._uop, lambda x: UOp.alu(x, op, inner_op(x.ufix(src[0]))))
      elif src[0]._shape is None: uop = UOp.alu(self._uop, op, inner_op(self._uop.ufix(src[0])))
      else:
        if isinstance(self, RT) and isinstance(src[0], RV):
          match self.layout:
            case TileLayout.ROW: uop = self.ker.warp.map(self._uop, lambda x, idx: UOp.alu(x, op, inner_op(src[0]._uop[idx[0], 0])))
            case TileLayout.COL: uop = self.ker.warp.map(self._uop, lambda x, idx: UOp.alu(x, op, inner_op(src[0]._uop[idx[1], 0])))
        else: uop = self.ker.warp.map(self._uop, lambda x, idx: UOp.alu(x, op, inner_op(src[0]._uop[*idx])))
    else: raise NotImplementedError
    return self.ruop(uop)
  def const_like(self, b): return b

  # override ops that do compute on the src uop
  def sub(self, x, reverse=False):
    return self.ufix(x).alu(Ops.ADD, self, inner_op=lambda y: -y) if reverse else self.alu(Ops.ADD, self.ufix(x), inner_op=lambda y: -y)
  def div(self, x, reverse=False):
    return self.ufix(x).alu(Ops.MUL, self, inner_op=lambda y: 1/y) if reverse else self.alu(Ops.MUL, self.ufix(x), inner_op=lambda y: 1/y)

@autowrap(UOp)
class GL:
  def __init__(self, uop:UOp, ker):
    self._uop, self.ker = uop, ker

  def ruop(self, uop:UOp):
    return GL(uop, self.ker)

  @classmethod
  def create(cls, shape, dtype:DType, ker):
    uop = ker.alloc(shape, dtype, AddrSpace.GLOBAL)
    return cls(uop, ker)

class TileLayout(Enum):
  ROW = auto()
  COL = auto()

class VecLayout(Enum):
  ORTHO = auto()

@dataclass(frozen=True)
class BaseShape:
  rows: int
  cols: int

  @property
  def num_elements(self): return self.rows * self.cols
  @property
  def elements_per_thread(self): return self.num_elements // WARP_THREADS

@dataclass(frozen=True)
class STBaseShape(BaseShape):
  _swizzle: Callable[[UOp, DType], UOp]
  bytes_per_thread: Callable[[DType], int]

  def swizzle(self, row, col, dtype:DType):
    offset = row * self.cols + col
    offset *= dtype.itemsize
    offset = self._swizzle(offset, dtype)
    offset //= dtype.itemsize
    return offset

def st_16x16_swizzle(offset:UOp, _): return offset
def st_16x16_bpt(dtype:DType):
  if dtype.itemsize == 2 or dtype.itemsize == 4: return 16
  else: raise NotImplementedError
ST_16X16 = STBaseShape(16, 16, st_16x16_swizzle, st_16x16_bpt)

def st_16x16_swizzled_swizzle(offset:UOp, dtype:DType):
  if dtype.itemsize == 2:
    swizzle = ((offset % 512) >> 7) << 3
    return offset ^ swizzle
  elif dtype.itemsize == 4:
    return offset
  else: raise NotImplementedError
def st_16x16_swizzled_bpt(dtype:DType):
  if dtype.itemsize == 2: return 4
  elif dtype.itemsize == 4: return 16
  else: raise NotImplementedError
ST_16X16_SWIZZLED = STBaseShape(16, 16, st_16x16_swizzled_swizzle, st_16x16_swizzled_bpt)

def st_32x32_swizzle(offset:UOp, dtype:DType):
  if dtype.itemsize == 2:
    first_swizzle = ((offset % 1024) >> 9) << 5
    second_swizzle = ((offset % 2048) >> 10) << 4
    return offset ^ first_swizzle ^ second_swizzle
  elif dtype.itemsize == 4:
    return offset
  else: raise NotImplementedError
def st_32x32_bpt(dtype:DType):
  if dtype.itemsize == 2 or dtype.itemsize == 4: return 16
  else: raise NotImplementedError
ST_32X32 = STBaseShape(32, 32, st_32x32_swizzle, st_32x32_bpt)

def st_16x32_swizzle(offset:UOp, dtype:DType):
  if dtype.itemsize == 2:
    swizzle = ((offset % 1024) >> 9) << 5
    return offset ^ swizzle
  elif dtype.itemsize == 4:
    return offset
  else: raise NotImplementedError
def st_16x32_bpt(dtype:DType):
  if dtype.itemsize == 2 or dtype.itemsize == 4: return 16
  else: raise NotImplementedError
ST_16X32 = STBaseShape(16, 32, st_16x32_swizzle, st_16x32_bpt)

def st_32x16_swizzle(offset:UOp, dtype:DType):
  if dtype.itemsize == 2:
    swizzle = ((offset % 1024) >> 9) << 4
    return offset ^ swizzle
  elif dtype.itemsize == 4:
    return offset
  else: raise NotImplementedError
def st_32x16_bpt(dtype:DType):
  if dtype.itemsize == 2 or dtype.itemsize == 4: return 16
  else: raise NotImplementedError
ST_32X16 = STBaseShape(32, 16, st_32x16_swizzle, st_32x16_bpt)

@autowrap(UOp)
class ST:
  def __init__(self, uop:UOp, rows:int, cols:int, layout:TileLayout, base_shape:STBaseShape, ker):
    self._uop, self.rows, self.cols, self.layout, self.base_shape, self.ker = uop, rows, cols, layout, base_shape, ker

  def ruop(self, uop:UOp):
    return ST(uop, self.rows, self.cols, self.layout, self.base_shape, self.ker)

  @classmethod
  def create(cls, shape, dtype:DType, layout:TileLayout, base_shape:STBaseShape, ker):
    rows = shape[-2]
    cols = shape[-1]
    assert rows % base_shape.rows == 0
    assert cols % base_shape.cols == 0
    assert cols % base_shape.elements_per_thread == 0

    height = rows // base_shape.rows
    width = cols // base_shape.cols

    uop = ker.alloc(shape[:-2] + (height, width, base_shape.rows, base_shape.cols), dtype, AddrSpace.LOCAL)
    return cls(uop, rows, cols, layout, base_shape, ker)

  def swizzle(self, row, col):
    swizzled_offset = self.base_shape.swizzle(row, col, self._uop.dtype.base.scalar())

    row = swizzled_offset // self.base_shape.cols
    col = swizzled_offset % self.base_shape.cols

    return row, col

@dataclass(frozen=True)
class RTBaseShape(BaseShape):
  stride: int

  @property
  def num_strides(self):
    return self.elements_per_thread // self.stride

RT_16X16 = RTBaseShape(rows=16, cols=16, stride=4)
RT_32X32 = RTBaseShape(rows=32, cols=32, stride=4)
RT_32X32_8 = RTBaseShape(rows=32, cols=32, stride=8)
RT_16X32 = RTBaseShape(rows=16, cols=32, stride=8)
RT_32X16 = RTBaseShape(rows=32, cols=16, stride=8)
RT_32X16_4 = RTBaseShape(rows=32, cols=16, stride=4)
RT_16X32_4 = RTBaseShape(rows=16, cols=32, stride=4)

@autowrap(UOp)
class RT(TileMathMixin):
  def __init__(self, uop:UOp, layout:TileLayout, base_shape:RTBaseShape, ker):
    self._uop, self.layout, self.base_shape, self.ker = uop, layout, base_shape, ker

  def ruop(self, uop:UOp):
    return RT(uop, self.layout, self.base_shape, self.ker)

  @classmethod
  def create(cls, shape, dtype:DType, layout:TileLayout, base_shape:RTBaseShape, ker):
    assert len(shape) == 2
    assert shape[0] % base_shape.rows == 0
    assert shape[1] % base_shape.cols == 0

    height = shape[0] // base_shape.rows
    width = shape[1] // base_shape.cols

    uop = ker.alloc((height, width, base_shape.elements_per_thread), dtype, AddrSpace.REG)
    return cls(uop, layout, base_shape, ker)

@autowrap(UOp)
class RV(TileMathMixin):
  def __init__(self, uop:UOp, length:int, layout:VecLayout, base_shape:RTBaseShape, ker):
    self._uop, self.ker = uop, ker
    self.length, self.layout, self.base_shape = length, layout, base_shape

  def ruop(self, uop:UOp):
    return RV(uop, self.length, self.layout, self.base_shape, self.ker)

  @classmethod
  def create(cls, length, dtype:DType, layout:VecLayout, base_shape:RTBaseShape, ker):
    tiles = length // base_shape.rows

    match layout:
      case VecLayout.ORTHO:
        inner_dim = 1
        outer_dim = tiles

    uop = ker.alloc((outer_dim, inner_dim), dtype, AddrSpace.REG)
    return RV(uop, length, layout, base_shape, ker)

ALL_TILES = UOp | GL | ST | RT | RV
