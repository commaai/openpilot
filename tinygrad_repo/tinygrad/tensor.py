# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from __future__ import annotations
import time, math, itertools, functools, struct, sys, inspect, pathlib, string, dataclasses, hashlib
from contextlib import ContextDecorator
from typing import List, Tuple, Callable, Optional, ClassVar, Type, Union, Sequence, Dict, cast, get_args, Literal, TYPE_CHECKING, SupportsIndex
from tinygrad.dtype import DType, DTypeLike, dtypes, ImageDType, ConstType, least_upper_float, least_upper_dtype, sum_acc_dtype, to_dtype, truncate
from tinygrad.helpers import argfix, make_tuple, flatten, prod, all_int, round_up, merge_dicts, argsort, getenv, all_same, fully_flatten, dedup
from tinygrad.helpers import IMAGE, DEBUG, WINO, _METADATA, Metadata, TRACEMETA, ceildiv, fetch, polyN
from tinygrad.multi import MultiLazyBuffer
from tinygrad.ops import smax, smin, resolve, UOp, Ops, sint, Variable, SimpleMathTrait, identity_element
from tinygrad.device import Device, Buffer, BufferSpec
from tinygrad.engine.lazy import LazyBuffer
from tinygrad.engine.realize import run_schedule
from tinygrad.engine.memory import memory_planner
from tinygrad.engine.schedule import ScheduleItem, create_schedule_with_vars

# **** start with two base classes, Tensor and Function ****

class Function:
  def __init__(self, device:Union[str, Tuple[str, ...]], *tensors:Tensor, metadata:Optional[Metadata]=None):
    self.device = device
    self.needs_input_grad = [t.requires_grad for t in tensors]
    self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False
    if self.requires_grad: self.parents = tensors
    self.metadata = metadata

  def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
  def backward(self, *args, **kwargs): raise RuntimeError(f"backward not implemented for {type(self)}")

  @classmethod
  def apply(fxn:Type[Function], *x:Tensor, **kwargs) -> Tensor:
    ctx = fxn(x[0].device, *x, metadata=_METADATA.get())
    ret = Tensor.__new__(Tensor)
    ret.lazydata, ret.requires_grad, ret.grad = ctx.forward(*[t.lazydata for t in x], **kwargs), ctx.requires_grad, None
    ret._ctx = ctx if ctx.requires_grad and not Tensor.no_grad else None  # used by autograd engine
    return ret

import tinygrad.function as F

def _metaop(op, shape:Tuple[sint,...], dtype:DType, device:Union[str, Tuple[str, ...]], arg=None, src:Tuple[LazyBuffer, ...]=()):
  if isinstance(device, str): return LazyBuffer.metaop(op, shape, dtype, device, arg, src)
  return MultiLazyBuffer([LazyBuffer.metaop(op, shape, dtype, d, arg, src) for d in device], None)

def _from_np_dtype(npdtype:'np.dtype') -> DType: # type: ignore [name-defined] # noqa: F821
  import numpy as np
  return dtypes.fields()[np.dtype(npdtype).name]
def _to_np_dtype(dtype:DType) -> Optional[type]:
  import numpy as np
  return np.dtype(dtype.fmt).type if dtype.fmt is not None else None

def _fromnp(x: 'np.ndarray') -> LazyBuffer:  # type: ignore [name-defined] # noqa: F821
  ret = LazyBuffer.metaop(Ops.EMPTY, x.shape, _from_np_dtype(x.dtype), "NPY")
  # fake realize
  ret.buffer.allocate(x)
  del ret.srcs
  return ret

def get_shape(x) -> Tuple[int, ...]:
  # NOTE: str is special because __getitem__ on a str is still a str
  if not hasattr(x, "__len__") or not hasattr(x, "__getitem__") or isinstance(x, str) or (hasattr(x, "shape") and x.shape == ()): return ()
  if not all_same(subs:=[get_shape(xi) for xi in x]): raise ValueError(f"inhomogeneous shape from {x}")
  return (len(subs),) + (subs[0] if subs else ())

def _frompy(x:Union[List, Tuple, bytes], dtype:DType) -> LazyBuffer:
  if isinstance(x, bytes): ret, data = LazyBuffer.metaop(Ops.EMPTY, (len(x)//dtype.itemsize,), dtype, "PYTHON"), x
  else:
    ret = LazyBuffer.metaop(Ops.EMPTY, get_shape(x), dtype, "PYTHON")
    assert dtype.fmt is not None, f"{dtype=} has None fmt"
    truncate_function = truncate[dtype]
    data = struct.pack(f"@{ret.size}{dtype.fmt}", *[truncate_function(xi) for xi in fully_flatten(x)])
  # fake realize
  ret.buffer.allocate(memoryview(data if Device.DEFAULT != "PYTHON" else bytearray(data)))
  del ret.srcs
  return ret

def _get_winograd_matcols(mat, dims:int, shp:Tuple[sint, ...], device:Union[str, Tuple[str, ...]], dtype:DType) -> List[List[Tensor]]:
  return [[Tensor.cat(*[Tensor.full(shp[:dim] + (1,) + shp[dim+1:], float(m[k]), device=device, dtype=dtype) for m in mat], dim=dim)
           for k in range(len(mat[0]))] for dim in range(dims)]

# winograd conv 3 kernel f(4x4,3x3) see: http://arxiv.org/abs/1509.09308
def _apply_winograd_matrix(mat, t:Tensor, dims:int) -> Tensor:
  # multiply mat_1 @ mat_2 @ t with foldable constants, where mat_i acts on vector t along dimension i; roughly kron(mat, mat) @ t
  # due to realize-before-expand rule in lazy.py, we must operate in this order: reshape -> expand -> arithmetic
  t_ = t.reshape(t.shape[:dims] + (1,) * dims + t.shape[dims:]).expand(t.shape[:dims] + (len(mat),) * dims + t.shape[dims:])  # add output dims
  # precalculate mat columns for each dim; prod(itertools.product(matcols)) gives the columns of kron(mat, mat, ...)
  matcols = _get_winograd_matcols(mat, dims, t_.shape[dims:], t_.device, t_.dtype)
  # multiply each element of t_ by the corresponding stacked column of kron(mat, mat), producing only one view for each element of t
  ret = sum(prod(col[idx] for col, idx in zip(matcols, mat_is)) * t_[mat_is] for mat_is in itertools.product(range(len(mat[0])), repeat=dims))
  assert isinstance(ret, Tensor), "sum didn't return a Tensor"
  return ret

def _align_left(*shapes:Tuple[sint, ...]) -> Tuple[Tuple[sint, ...], ...]:
  # unsqueeze left to make every shape same length
  max_dim = max(len(shape) for shape in shapes)
  return tuple((1,) * (max_dim - len(shape)) + shape for shape in shapes)
def _broadcast_shape(*shapes:Tuple[sint, ...]) -> Tuple[sint, ...]:
  return tuple(0 if 0 in nth_dim_sizes else smax(nth_dim_sizes) for nth_dim_sizes in zip(*_align_left(*shapes)))

def _masked_setitem(target:Tensor, values:Tensor, mask:Tensor, axes:Tuple[int, ...]):
  # apply mask to values (already broadcasted) and reduce such that if mask contains repeated indices the last one remains
  values = values * mask
  for dim in axes: mask, values = functools.reduce(lambda x,y: (x[0]|y[0], y[0].where(y[1], x[1])), zip(mask.split(1, dim), values.split(1, dim)))
  # remove extra dims from reduce
  for dim in reversed(axes): mask, values = mask.squeeze(dim), values.squeeze(dim)
  # select from values for each True element in mask else select from self
  return mask.where(values, target)

ReductionStr = Literal["mean", "sum", "none"]

class Tensor(SimpleMathTrait):
  """
  A `Tensor` is a multi-dimensional matrix containing elements of a single data type.

  ```python exec="true" session="tensor"
  from tinygrad import Tensor, dtypes, nn
  import numpy as np
  import math
  np.set_printoptions(precision=4)
  ```
  """
  __slots__ = "lazydata", "requires_grad", "grad", "_ctx"
  __deletable__ = ('_ctx',)
  training: ClassVar[bool] = False
  no_grad: ClassVar[bool] = False

  def __init__(self, data:Union[None, ConstType, UOp, bytes, List, Tuple, LazyBuffer, MultiLazyBuffer, 'np.ndarray', pathlib.Path],  # type: ignore [name-defined] # noqa: F821
               device:Optional[Union[str, tuple, list]]=None, dtype:Optional[DTypeLike]=None, requires_grad:Optional[bool]=None):
    if dtype is not None: dtype = to_dtype(dtype)
    assert dtype is None or isinstance(dtype, DType), f"invalid dtype {dtype}"
    if device is None and isinstance(data, pathlib.Path): device = f"DISK:{data.resolve()}"  # keep it on the disk if device is None
    device = tuple(Device.canonicalize(x) for x in device) if isinstance(device, (tuple, list)) else Device.canonicalize(device)

    # tensors can have gradients if you have called .backward
    self.grad: Optional[Tensor] = None

    # NOTE: this can be in three states. False and None: no gradient, True: gradient
    # None (the default) will be updated to True if it's put in an optimizer
    self.requires_grad: Optional[bool] = requires_grad

    # internal variable used for autograd graph construction
    self._ctx: Optional[Function] = None

    # create a LazyBuffer from the different types of inputs
    if isinstance(data, (LazyBuffer, MultiLazyBuffer)): assert dtype is None or dtype==data.dtype, "dtype doesn't match, and casting isn't supported"
    elif data is None: data = _metaop(Ops.EMPTY, (0,), dtype or dtypes.default_float, device)
    elif isinstance(data, get_args(ConstType)): data = _metaop(Ops.CONST, tuple(), dtype or dtypes.from_py(data), device, data)
    elif isinstance(data, UOp):
      assert data.op is Ops.BIND and data.src[0].op is Ops.DEFINE_VAR and data.src[1].op is Ops.CONST, f"can't create tensor from UOp {data}"
      data = _metaop(Ops.CONST, tuple(), dtype or data.dtype, device, data)
    elif isinstance(data, bytes): data = _frompy(data, dtypes.uint8 if dtype is None else dtype)
    elif isinstance(data, (list, tuple)):
      if dtype is None:
        if (d := fully_flatten(data)) and all(isinstance(s, bool) for s in d): dtype = dtypes.bool
        else: dtype = dtypes.default_int if d and all_int(d) else dtypes.default_float
      if dtype == dtypes.bfloat16: data = Tensor(_frompy(data, dtypes.float32), device=device).cast(dtypes.bfloat16).lazydata
      else: data = _frompy(data, dtype)
    elif str(type(data)) == "<class 'numpy.ndarray'>":
      import numpy as np
      assert isinstance(data, np.ndarray), f"expected np.ndarray, got {data}"
      if data.shape == (): data = _metaop(Ops.CONST, tuple(), dtype or _from_np_dtype(data.dtype), device, data.item())
      else: data = _fromnp(data.astype(npdtype) if dtype is not None and (npdtype:=_to_np_dtype(dtype)) is not None else data)  # type: ignore [name-defined]
    elif isinstance(data, pathlib.Path):
      dtype = dtype or dtypes.uint8
      data = _metaop(Ops.EMPTY, (data.stat().st_size // dtype.itemsize,), dtype, f"DISK:{data.resolve()}")

    # by this point, it has to be a LazyBuffer
    if not isinstance(data, (LazyBuffer, MultiLazyBuffer)): raise RuntimeError(f"can't create Tensor from {data!r} with type {type(data)}")

    # data might be on a different device
    if isinstance(device, str): self.lazydata:Union[LazyBuffer, MultiLazyBuffer] = data if data.device == device else data.copy_to_device(device)
    # if device is a tuple, we should have/construct a MultiLazyBuffer
    elif isinstance(data, LazyBuffer): self.lazydata = MultiLazyBuffer.from_sharded(data, device, None, None)
    else:
      assert data.device == device, f"MultiLazyBuffer device mismatch, {data.device} != {device}"
      self.lazydata = data

  class train(ContextDecorator):
    def __init__(self, mode:bool = True): self.mode = mode
    def __enter__(self): self.prev, Tensor.training = Tensor.training, self.mode
    def __exit__(self, exc_type, exc_value, traceback): Tensor.training = self.prev

  class test(ContextDecorator):
    def __init__(self, mode:bool = True): self.mode = mode
    def __enter__(self): self.prev, Tensor.no_grad = Tensor.no_grad, self.mode
    def __exit__(self, exc_type, exc_value, traceback): Tensor.no_grad = self.prev

  def __repr__(self):
    return f"<Tensor {self.lazydata!r} on {self.device} with grad {(self.grad.lazydata if self.grad is not None else None)!r}>"

  # Python has a non moving GC, so this should be okay
  def __hash__(self): return id(self)

  def __bool__(self): raise TypeError("__bool__ on Tensor is not defined")

  def __len__(self):
    if not self.shape: raise TypeError("len() of a 0-d tensor")
    return self.shape[0]

  @property
  def device(self) -> Union[str, Tuple[str, ...]]: return self.lazydata.device

  @property
  def shape(self) -> Tuple[sint, ...]: return self.lazydata.shape

  @property
  def dtype(self) -> DType: return self.lazydata.dtype

  # ***** data handlers ****

  def schedule_with_vars(self, *lst:Tensor) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
    """
    Creates the schedule needed to realize these Tensor(s), with Variables.

    NOTE: A Tensor can only be scheduled once.
    """
    schedule, var_vals = create_schedule_with_vars(flatten([x.lazydata.lbs for x in (self,)+lst]))
    return memory_planner(schedule), var_vals

  def schedule(self, *lst:Tensor) -> List[ScheduleItem]:
    """Creates the schedule needed to realize these Tensor(s)."""
    schedule, var_vals = self.schedule_with_vars(*lst)
    assert len(var_vals) == 0
    return schedule

  def realize(self, *lst:Tensor, do_update_stats=True) -> Tensor:
    """Triggers the computation needed to create these Tensor(s)."""
    run_schedule(*self.schedule_with_vars(*lst), do_update_stats=do_update_stats)
    return self

  def replace(self, x:Tensor) -> Tensor:
    """
    Replaces the data of this tensor with the data of another tensor. Only the shape of the tensors must match.
    """
    # used for replacing a Tensor with a new version of it (potentially with a different device and dtype)
    assert not x.requires_grad and getattr(self, '_ctx', None) is None
    assert self.shape == x.shape, f"replace shape mismatch {self.shape} != {x.shape}"
    self.lazydata = x.lazydata
    return self

  def assign(self, x) -> Tensor:
    # TODO: this is a hack for writing to DISK. remove with working assign
    if isinstance(self.device, str) and self.device.startswith("DISK"):
      if x.__class__ is not Tensor: x = Tensor(x, device="CLANG", dtype=self.dtype)
      self.contiguous().realize().lazydata.base.realized.copyin(x._data())
      return self
    if x.__class__ is not Tensor: x = Tensor(x, device=self.device, dtype=self.dtype)
    if DEBUG >= 4: print(f"assign {self.lazydata} <- {x.lazydata}")
    if self.lazydata is x.lazydata: return self  # a self assign is a NOOP
    # NOTE: we allow cross device assign
    assert self.shape == x.shape, f"assign shape mismatch {self.shape} != {x.shape}"
    assert self.device == x.device, f"assign device mismatch {self.device} != {x.device}"
    assert self.dtype == x.dtype, f"assign dtype mismatch {self.dtype} != {x.dtype}"
    assert not isinstance(self.lazydata, MultiLazyBuffer) or self.lazydata.axis == x.lazydata.axis, "axis must match on MultiLazyBuffer"
    assert not x.requires_grad  # self requires_grad is okay?
    if not self.lazydata.is_realized: return self.replace(x)
    self.lazydata = self.lazydata.assign(x.lazydata)
    return self

  def detach(self) -> Tensor:
    """
    Returns a new tensor with the same data as this tensor, but detached from the autograd graph.
    """
    return Tensor(self.lazydata, device=self.device, requires_grad=False)

  def _data(self) -> memoryview:
    if 0 in self.shape: return memoryview(bytearray(0))
    # NOTE: this realizes on the object from as_buffer being a Python object
    cpu = self.cast(self.dtype.base).contiguous().to("CLANG").realize()
    buf = cast(Buffer, cast(LazyBuffer, cpu.lazydata).base.realized)
    if self.device != "CLANG": buf.options = BufferSpec(nolru=True)
    return buf.as_buffer(allow_zero_copy=True if self.device != "CLANG" else False)

  def data(self) -> memoryview:
    """
    Returns the data of this tensor as a memoryview.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(np.frombuffer(t.data(), dtype=np.int32))
    ```
    """
    assert self.dtype.base.fmt is not None, f"no fmt dtype for {self.dtype.base}"
    assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
    if TYPE_CHECKING or sys.version_info < (3, 12): assert self.dtype.base.fmt != "e"
    return cast(memoryview, self._data().cast(self.dtype.base.fmt) if 0 in self.shape else self._data().cast(self.dtype.base.fmt, self.shape))

  def item(self) -> ConstType:
    """
    Returns the value of this tensor as a standard Python number.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor(42)
    print(t.item())
    ```
    """
    assert self.numel() == 1, "must have one element for item"
    return self.data()[(0,) * len(self.shape)]

  # TODO: should be Tensor.tolist() -> Union[List[ConstType], ConstType]. The List is Sequence because mypy expects memoryview.tolist() -> list[int]
  # src: https://github.com/python/mypy/blob/release-1.6/mypy/typeshed/stdlib/builtins.pyi#L803
  def tolist(self) -> Union[Sequence[ConstType], ConstType]:
    """
    Returns the value of this tensor as a nested list.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(t.tolist())
    ```
    """
    return self.data().tolist()

  def numpy(self) -> 'np.ndarray':  # type: ignore [name-defined] # noqa: F821
    """
    Returns the value of this tensor as a `numpy.ndarray`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(repr(t.numpy()))
    ```
    """
    import numpy as np
    if self.dtype.base == dtypes.bfloat16: return self.float().numpy()
    assert _to_np_dtype(self.dtype.base) is not None, f"no np dtype for {self.dtype.base}"
    assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
    return np.frombuffer(self._data(), dtype=_to_np_dtype(self.dtype.base)).reshape(self.shape)

  def clone(self) -> Tensor:
    """
    Creates a clone of this tensor allocating a seperate buffer for the data.
    """
    ret = Tensor(self.lazydata.clone(), self.device, requires_grad=self.requires_grad)
    if self.grad is not None: ret.grad = self.grad.clone()
    if hasattr(self, '_ctx'): ret._ctx = self._ctx
    return ret

  def to(self, device:Optional[Union[str, Tuple[str, ...]]]) -> Tensor:
    """
    Moves the tensor to the given device.
    """
    device = tuple(Device.canonicalize(x) for x in device) if isinstance(device, (tuple, list)) else Device.canonicalize(device)
    if device == self.device: return self
    if not isinstance(device, str): return self.shard(device)
    ret = Tensor(self.lazydata, device, requires_grad=self.requires_grad)
    if self.grad is not None: ret.grad = self.grad.to(device)
    if hasattr(self, '_ctx'): ret._ctx = self._ctx
    return ret

  def to_(self, device:Optional[Union[str, Tuple[str, ...]]]):
    """
    Moves the tensor to the given device in place.
    """
    real = self.to(device)
    # TODO: is this assign?
    if self.grad is not None and real.grad is not None: self.grad.lazydata = real.grad.lazydata
    self.lazydata = real.lazydata

  def shard(self, devices:Tuple[str, ...], axis:Optional[int]=None, splits:Optional[Tuple[int, ...]]=None) -> Tensor:
    """
    Shards the tensor across the given devices. Optionally specify which axis to shard on, and how to split it across devices.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.empty(2, 3)
    print(t.shard((t.device, t.device), axis=1, splits=(2, 1)).lazydata)
    ```

    """
    assert isinstance(self.lazydata, LazyBuffer), "can't shard a MultiLazyBuffer"
    devices, bounds = tuple(Device.canonicalize(x) for x in devices), None
    if axis is not None:
      if axis < 0: axis += len(self.shape)
      if splits is None:
        if not isinstance(total:=self.shape[axis], int): raise RuntimeError(f"cannot shard symbolic shape {self.shape=}, {axis=}")
        sz = ceildiv(total, len(devices))
        splits = tuple([max(0, min(sz, total - sz*i)) for i in range(len(devices))])
      assert sum(splits) == self.shape[axis], "specified splits do not sum up to axis shape"
      boundaries = tuple(itertools.accumulate(splits))
      bounds = tuple(zip((0,) + boundaries, boundaries))
    return Tensor(MultiLazyBuffer.from_sharded(self.lazydata, devices, axis, bounds), device=devices, requires_grad=self.requires_grad)

  def shard_(self, devices:Tuple[str, ...], axis:Optional[int]=None, splits:Optional[Tuple[int, ...]]=None):
    """
    Shards the tensor across the given devices in place.
    """
    self.lazydata = self.shard(devices, axis, splits).lazydata
    return self

  @staticmethod
  def from_uop(y:UOp, **kwargs) -> Tensor:
    if y.op is Ops.BIND: return Tensor(y, **kwargs, requires_grad=False)   # this is the only UOp allowed in Tensor
    if y.op is Ops.CONST: return Tensor(y.arg, **kwargs, requires_grad=False)
    if y.op is Ops.MUL: return Tensor.from_uop(y.src[0]) * Tensor.from_uop(y.src[1])
    if y.op is Ops.ADD: return Tensor.from_uop(y.src[0]) + Tensor.from_uop(y.src[1])
    if y.op is Ops.MAX: return Tensor.from_uop(y.src[0]).maximum(Tensor.from_uop(y.src[1]))
    raise RuntimeError(f"unhandled UOp {y}")

  # ***** creation entrypoint *****

  @staticmethod
  def _metaop(op, shape, device:Optional[Union[Tuple[str, ...], str]]=None, dtype:Optional[DTypeLike]=None, arg=None, **kwargs):
    dtype = to_dtype(dtype) if dtype is not None else dtypes.default_float
    if isinstance(device, tuple):
      return Tensor(MultiLazyBuffer([LazyBuffer.metaop(op, shape, dtype, Device.canonicalize(d), arg) for d in device], None),
                    device, dtype, **kwargs)
    return Tensor(LazyBuffer.metaop(op, shape, dtype, Device.canonicalize(device), arg), device, dtype, **kwargs)

  @staticmethod
  def empty(*shape, **kwargs):
    """
    Creates an empty tensor with the given shape.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.empty(2, 3)
    print(t.shape)
    ```
    """
    return Tensor._metaop(Ops.EMPTY, argfix(*shape), **kwargs)

  @staticmethod
  def from_blob(ptr:int, shape:Tuple[int, ...], **kwargs) -> Tensor:
    """
    Exposes the pointer as a Tensor without taking ownership of the original data.
    The pointer must remain valid for the entire lifetime of the created Tensor.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.
    """

    r = Tensor._metaop(Ops.EMPTY, shape, **kwargs)
    r.lazydata.buffer.allocate(external_ptr=ptr)
    del r.lazydata.srcs # fake realize
    return r

  @staticmethod
  def from_url(url:str, gunzip:bool=False, **kwargs) -> Tensor:
    """
    Create a Tensor from a URL.

    This is the preferred way to access Internet resources.
    It currently returns a DISK Tensor, but in the future it may return an HTTP Tensor.
    This also will soon become lazy (when possible) and not print progress without DEBUG.

    THe `gunzip` flag will gzip extract the resource and return an extracted Tensor.
    """
    return Tensor(fetch(url, gunzip=gunzip), **kwargs)

  _seed: int = int(time.time())
  _device_seeds: Dict[str, Tensor] = {}
  _device_rng_counters: Dict[str, Tensor] = {}
  @staticmethod
  def manual_seed(seed=0):
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
  def _threefry_random_bits(key:Tensor, counts0:Tensor, counts1:Tensor):
    x = (counts1.cast(dtypes.uint64) << 32) | counts0.cast(dtypes.uint64)
    x = F.Threefry.apply(x, (key[1]._broadcast_to(x.shape).cast(dtypes.uint64) << 32) | key[0]._broadcast_to(x.shape).cast(dtypes.uint64))
    counts0, counts1 = (x & 0xffffffff).cast(dtypes.uint32), ((x >> 32) & 0xffffffff).cast(dtypes.uint32)
    return counts0.cat(counts1)

  @staticmethod
  def rand(*shape, device:Optional[str]=None, dtype:Optional[DTypeLike]=None, contiguous:bool=True, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a uniform distribution over the interval `[0, 1)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.rand(2, 3)
    print(t.numpy())
    ```
    """
    if not dtypes.is_float(dtype := to_dtype(dtype or dtypes.default_float)): raise ValueError(f"rand only supports float dtypes, got {dtype}")
    if not all_int(shape:=argfix(*shape)) or not all(s >= 0 for s in shape): raise ValueError(f"invalid input {shape=}")
    if device is not None and not isinstance(device, str): raise ValueError(f"rand only supports single device, got {device=}")
    _device = device = Device.canonicalize(device)

    # when using MOCKGPU and NV generate rand on CLANG
    if getenv("MOCKGPU") and device.startswith("NV"): device = "CLANG"

    # generate per device seeds and rng counter if we haven't seen this device yet
    if device not in Tensor._device_seeds:
      Tensor._device_seeds[device] = Tensor(
        [int.from_bytes(hashlib.sha256(len(Tensor._device_seeds).to_bytes(4, "big")).digest(), "big"), Tensor._seed],
        device=device, dtype=dtypes.uint32, requires_grad=False)
      Tensor._device_rng_counters[device] = Tensor([0], device=device, dtype=dtypes.uint32, requires_grad=False)
      had_counter = False
    else: had_counter = True

    # if shape has 0, return zero tensor
    if (numel := prod(shape)) == 0: return Tensor.zeros(shape, device=_device, dtype=dtype, **kwargs)
    num = ceildiv(numel * dtype.itemsize, 4)

    # increment rng counter for devices
    if had_counter: Tensor._device_rng_counters[device].assign(Tensor._device_rng_counters[device] + num).contiguous()

    # threefry random bits
    counts0 = (Tensor.arange(ceildiv(num, 2), device=device, dtype=dtypes.uint32, requires_grad=False)+Tensor._device_rng_counters[device])
    counts1 = counts0 + ceildiv(num, 2)
    bits = Tensor._threefry_random_bits(Tensor._device_seeds[device], counts0, counts1)[:num]

    # bitcast to uint with same number of bits
    _, nmant = dtypes.finfo(dtype)
    uint_dtype = {1: dtypes.uint8, 2: dtypes.uint16, 4: dtypes.uint32, 8: dtypes.uint64}[dtype.itemsize]
    bits = bits.bitcast(uint_dtype)
    # only randomize the mantissa bits and set the exponent to 1
    one = Tensor.ones_like(bits, device=bits.device, dtype=dtype).bitcast(uint_dtype)
    bits = bits.rshift((dtype.itemsize * 8) - nmant).bitwise_or(one)
    # bitcast back to the original dtype and reshape
    out = bits.bitcast(dtype)[:numel].sub(1).reshape(shape)

    # move back to the original device if we were using MOCKGPU
    if getenv("MOCKGPU") and _device: out = out.to(_device)

    out.requires_grad = kwargs.get("requires_grad")
    return out.contiguous() if contiguous else out

  # ***** creation helper functions *****

  @staticmethod
  def full(shape:Tuple[sint, ...], fill_value:ConstType, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with the given value.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 3), 42).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 3), False).numpy())
    ```
    """
    return Tensor(fill_value, **kwargs).reshape((1, )*len(new_shape := argfix(shape))).expand(new_shape)

  @staticmethod
  def zeros(*shape, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with zeros.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.zeros(2, 3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.zeros(2, 3, dtype=dtypes.int32).numpy())
    ```
    """
    return Tensor.full(argfix(*shape), 0.0, **kwargs)

  @staticmethod
  def ones(*shape, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with ones.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(2, 3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(2, 3, dtype=dtypes.int32).numpy())
    ```
    """
    return Tensor.full(argfix(*shape), 1.0, **kwargs)

  @staticmethod
  def arange(start, stop=None, step=1, **kwargs) -> Tensor:
    """
    Returns a 1-D tensor of size `ceil((stop - start) / step)` with values from `[start, stop)`, with spacing between values given by `step`.

    If `stop` is not specified, values are generated from `[0, start)` with the given `step`.

    If `stop` is specified, values are generated from `[start, stop)` with the given `step`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5, 10).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5, 10, 2).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5.5, 10, 2).numpy())
    ```
    """
    if stop is None: stop, start = start, 0
    dtype = kwargs.pop("dtype", dtypes.default_float if any(isinstance(x, float) for x in (start, stop, step)) else dtypes.default_int)
    # NOTE: this matches numpy, torch raises RuntimeError if stop-start and step have different signs
    if (output_len:=ceildiv(stop-start, step)) <= 0: return Tensor([], dtype=dtype, **kwargs)
    return (Tensor.full((output_len,), step, dtype=dtype, **kwargs)._cumalu(0, Ops.ADD) + (start - step)).cast(dtype)

  @staticmethod
  def linspace(start:Union[int, float], stop:Union[int, float], steps:int, **kwargs) -> Tensor:
    """
    Returns a 1-D tensor of `steps` evenly spaced values from `start` to `stop`, inclusive.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.linspace(0, 10, 5).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.linspace(-1, 1, 5).numpy())
    ```
    """
    if steps < 0: raise ValueError("number of steps must be non-negative")
    if (dtype := to_dtype(kwargs.pop("dtype", dtypes.default_float))) == dtypes.bool: raise ValueError("linspace with bool dtype is not supported")
    if steps == 1: return Tensor([start], dtype=dtype, **kwargs)
    return (start + Tensor.arange(steps, **kwargs) * ((stop - start) / (steps - 1))).cast(dtype)

  @staticmethod
  def eye(n:int, m:Optional[int]=None, **kwargs) -> Tensor:
    """
    Returns a 2-D tensor with `n` rows and `m` columns, with ones on the diagonal and zeros elsewhere.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.eye(3).numpy())
    ```

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.eye(2, 4).numpy())
    ```
    """
    if n < 0 or (m is not None and m < 0): raise ValueError(f"cannot have negative {n=}, {m=}")
    x = Tensor.ones((n,1),**kwargs).pad((None,(0,n))).flatten().shrink(((0,n*n),)).reshape(n,n)
    return x if m is None else x.pad((None, (0, m-n))) if m > n else x.shrink((None, (0, m)))

  def full_like(self, fill_value:ConstType, **kwargs) -> Tensor:
    """
    Creates a tensor with the same shape as `self`, filled with the given value.
    If `dtype` is not specified, the dtype of `self` is used.

    You can pass in the `device` keyword argument to control device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(Tensor.full_like(t, 42).numpy())
    ```
    """
    return Tensor.full(self.shape, fill_value, dtype=kwargs.pop("dtype", self.dtype), device=kwargs.pop("device", self.device), **kwargs)

  def zeros_like(self, **kwargs) -> Tensor:
    """
    Creates a tensor with the same shape as `self`, filled with zeros.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(Tensor.zeros_like(t).numpy())
    ```
    """
    return self.full_like(0, **kwargs)

  def ones_like(self, **kwargs) -> Tensor:
    """
    Creates a tensor with the same shape as `self`, filled with ones.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.zeros(2, 3)
    print(Tensor.ones_like(t).numpy())
    ```
    """
    return self.full_like(1, **kwargs)

  def rand_like(self, **kwargs) -> Tensor:
    """
    Creates a tensor with the same shape and sharding as `self`, filled with random values from a uniform distribution over the interval `[0, 1)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(Tensor.rand_like(t).numpy())
    ```
    """
    dtype = kwargs.pop("dtype", self.dtype)
    if isinstance(self.device, tuple) and isinstance(self.lazydata, MultiLazyBuffer):
      if kwargs.get("device") is not None: raise RuntimeError("cannot specify `device` on `rand_like` of a multi device tensor")
      if self.lazydata.axis is None: return Tensor.rand(*self.shape, dtype=dtype, **kwargs).shard(self.device)
      contiguous = kwargs.pop("contiguous", True)
      rands = [Tensor.rand(*lb.shape, device=lb.device, dtype=dtype, contiguous=contiguous, **kwargs).lazydata for lb in self.lazydata.lbs]
      return Tensor(MultiLazyBuffer(cast(List[LazyBuffer], rands), self.lazydata.axis), device=self.device, dtype=dtype, **kwargs)
    return Tensor.rand(*self.shape, device=kwargs.pop("device", self.device), dtype=dtype, **kwargs)

  # ***** rng hlops *****

  @staticmethod
  def randn(*shape, dtype:Optional[DTypeLike]=None, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a normal distribution with mean `0` and standard deviation `1`.
    If `dtype` is not specified, the default type is used.

    You can pass in the `device` keyword argument to control device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.randn(2, 3).numpy())
    ```
    """
    # https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    src = Tensor.rand((2, *argfix(*shape)), **{**kwargs, "dtype": dtypes.float32})
    return src[0].mul(2*math.pi).cos().mul((1 - src[1]).log().mul(-2).sqrt()).cast(dtype or dtypes.default_float)

  @staticmethod
  def randint(*shape, low=0, high=10, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random integer values generated uniformly from the interval `[low, high)`.
    If `dtype` is not specified, the default type is used.

    You can pass in the `device` keyword argument to control device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.randint(2, 3, low=5, high=10).numpy())
    ```
    """
    if not isinstance(low, int) or not isinstance(high, int): raise TypeError(f"{low=} and {high=} must be integers")
    dtype = to_dtype(kwargs.pop("dtype", dtypes.int32))
    if not dtypes.is_int(dtype): raise TypeError(f"{dtype=} must be int")
    return Tensor.uniform(*shape, low=low, high=high, dtype=dtype, **kwargs)

  @staticmethod
  def normal(*shape, mean=0.0, std=1.0, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a normal distribution with the given `mean` and standard deviation `std`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.normal(2, 3, mean=10, std=2).numpy())
    ```
    """
    return (std * Tensor.randn(*shape, **kwargs)) + mean

  @staticmethod
  def uniform(*shape, low=0.0, high=1.0, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a uniform distribution over the interval `[low, high)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.uniform(2, 3, low=2, high=10).numpy())
    ```
    """
    dtype = kwargs.pop("dtype", dtypes.default_float)
    return ((high-low) * Tensor.rand(*shape, **kwargs)).cast(dtype) + low

  @staticmethod
  def scaled_uniform(*shape, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a uniform distribution
    over the interval `[-prod(shape)**-0.5, prod(shape)**-0.5)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.scaled_uniform(2, 3).numpy())
    ```
    """
    return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul(prod(argfix(*shape))**-0.5)

  # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
  @staticmethod
  def glorot_uniform(*shape, **kwargs) -> Tensor:
    """
    <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform>

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.glorot_uniform(2, 3).numpy())
    ```
    """
    return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul((6/(argfix(*shape)[0]+prod(argfix(*shape)[1:])))**0.5)

  # https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
  @staticmethod
  def kaiming_uniform(*shape, a:float = 0.01, **kwargs) -> Tensor:
    """
    <https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_>

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.kaiming_uniform(2, 3).numpy())
    ```
    """
    bound = math.sqrt(3.0) * math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(argfix(*shape)[1:]))
    return Tensor.uniform(*shape, low=-bound, high=bound, **kwargs)

  # https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_
  @staticmethod
  def kaiming_normal(*shape, a:float = 0.01, **kwargs) -> Tensor:
    """
    <https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_>

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.kaiming_normal(2, 3).numpy())
    ```
    """
    std = math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(argfix(*shape)[1:]))
    return Tensor.normal(*shape, mean=0.0, std=std, **kwargs)

  def multinomial(self:Tensor, num_samples:int = 1, replacement:bool = False) -> Tensor:
    assert 1 <= self.ndim <= 2 and num_samples > 0, f"{self.ndim=} must be 1 or 2 dim, {num_samples=} must be positive"
    assert replacement or num_samples == 1, "no replacement only supports num_samples = 1"
    weight = self.unsqueeze(0) if self.ndim == 1 else self
    cdf = (cw := weight.cumsum(1).float()) / cw[:, -1].unsqueeze(1)
    unif_samples = Tensor.rand(num_samples, cdf.shape[0], 1).to(self.device)
    indices = (unif_samples.expand((-1, -1, cdf.shape[1])) >= cdf).sum(2).permute((1, 0))
    return (indices.squeeze(0) if self.ndim == 1 else indices).cast(dtypes.int32)

  # ***** toposort and backward pass *****

  def _deepwalk(self):
    def _walk(node, visited):
      visited.add(node)
      # if tensor is not leaf, reset grad
      if (ctx := getattr(node, "_ctx", None)) is not None and len(ctx.parents) != 0: node.grad = None
      if ctx:
        for i in node._ctx.parents:
          if i not in visited: yield from _walk(i, visited)
        yield node
    return list(_walk(self, set()))

  def backward(self, gradient:Optional[Tensor]=None, retain_graph:bool=False) -> Tensor:
    """
    Propagates the gradient of a tensor backwards through the computation graph.
    If the 'gradient' argument is not provided, the tensor must be a scalar, and the gradient is implicitly set to 1.0.
    If 'retain_graph' is false, the graph used to compute the grads will be freed. Otherwise, it will be kept. Keeping it can increase memory usage.
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    t.sum().backward()
    print(t.grad.numpy())
    ```
    """
    toposorted = self._deepwalk()
    if gradient is None:
      assert self.shape == tuple(), "when no gradient is provided, backward must be called on a scalar tensor"
      # fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
      # this is "implicit gradient creation"
      gradient = Tensor(1.0, dtype=self.dtype, device=self.device, requires_grad=False)

    assert self.shape == gradient.shape, f"grad shape must match tensor shape, {gradient.shape!r} != {self.shape!r}"
    self.grad = gradient
    for t0 in reversed(toposorted):
      if t0.grad is None: raise RuntimeError(f"tensor {t0} has no grad")
      token = _METADATA.set(dataclasses.replace(md, backward=True) if (md := t0._ctx.metadata) is not None else None)
      grads = t0._ctx.backward(t0.grad.lazydata)
      _METADATA.reset(token)
      grads = [Tensor(g, device=self.device, requires_grad=False) if g is not None else None
        for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
      for t, g in zip(t0._ctx.parents, grads):
        if g is not None and t.requires_grad:
          assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
          t.grad = g if t.grad is None else (t.grad + g)
      if not retain_graph: del t0._ctx
    return self

  # ***** movement low level ops *****

  def view(self, *shape) -> Tensor:
    """`.view` is an alias for `.reshape`."""
    return self.reshape(shape)

  def reshape(self, shape, *args) -> Tensor:
    """
    Returns a tensor with the same data as the original tensor but with a different shape.
    `shape` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6)
    print(t.reshape(2, 3).numpy())
    ```
    """
    # resolve None and args
    new_shape = tuple([s if s is not None else self.shape[i] for i,s in enumerate(argfix(shape, *args))])
    # resolve -1
    if (c := new_shape.count(-1)) > 1: raise RuntimeError(f"only one dimension can be inferred using -1, getting {new_shape}")
    if c: new_shape = tuple([-prod(self.shape) // prod(new_shape) if s == -1 else s for s in new_shape])
    return F.Reshape.apply(self, shape=new_shape) if new_shape != self.shape else self

  def expand(self, shape, *args) -> Tensor:
    """
    Returns a tensor that is expanded to the shape that is specified.
    Expand can also increase the number of dimensions that a tensor has.

    Passing a `-1` or `None` to a dimension means that its size will not be changed.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t.expand(4, -1).numpy())
    ```
    """
    new_shape = tuple(from_ if to == -1 or to is None else to for from_, to in zip(*(_align_left(self.shape, argfix(shape, *args)))))
    return self._broadcast_to(new_shape)

  def permute(self, order, *args) -> Tensor:
    """
    Returns a tensor that is a permutation of the original tensor.
    The new tensor has the same data as the original tensor but with the dimensions permuted according to the order specified.
    `order` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.permute(1, 0).numpy())
    ```
    """
    order_arg = tuple(self._resolve_dim(x) for x in argfix(order, *args))
    if sorted(order_arg) != list(range(self.ndim)): raise RuntimeError(f"order is not a valid permutation, getting {order_arg}")
    return F.Permute.apply(self, order=order_arg)

  def flip(self, axis, *args) -> Tensor:
    """
    Returns a tensor that reverses the order of the original tensor along given `axis`.
    `axis` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.flip(0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.flip((0, 1)).numpy())
    ```
    """
    axis_arg = tuple(self._resolve_dim(x) for x in argfix(axis, *args))
    if len(axis_arg) != len(dedup(axis_arg)): raise RuntimeError(f"dim can appear at most once, getting {axis_arg}")
    return F.Flip.apply(self, axis=axis_arg)

  def shrink(self, arg:Tuple[Optional[Tuple[sint, sint]], ...]) -> Tensor:
    """
    Returns a tensor that shrinks the each axis based on input arg.
    `arg` must have the same length as `self.ndim`.
    For each axis, it can be `None`, which means no shrink, or a tuple `(start, end)` that works the same as Python slice.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(3, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.shrink(((None, (1, 3)))).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.shrink((((0, 2), (0, 2)))).numpy())
    ```
    """
    if (shrink_arg:=[x if x is not None else (0,s) for x,s in zip(arg, self.shape)]) == [(0,s) for s in self.shape]: return self
    return F.Shrink.apply(self, arg=tuple(shrink_arg))

  def pad(self, padding:Union[Sequence[sint], Sequence[Optional[Tuple[sint, sint]]]], mode:str="constant", value:float=0.0) -> Tensor:
    """
    Returns a tensor with padding applied based on the input `padding`.
    `padding` supports two padding structures:

    1. Flat padding: (padding_left, padding_right, padding_top, padding_bottom, ...)
       - This structure matches PyTorch's pad.
       - `padding` length must be even.

    2. Group padding: (..., (padding_top, padding_bottom), (padding_left, padding_right))
       - This structure matches pad for jax, numpy, tensorflow and others.
       - For each axis, padding can be `None`, meaning no padding, or a tuple `(start, end)`.
       - `padding` must have the same length as `self.ndim`.

    Padding values can be negative, resulting in dimension shrinks that work similarly to Python negative slices.
    Padding modes is selected with `mode` which supports `constant`, `reflect` and `replicate`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(1, 1, 3, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.pad((1, 2, 0, -1)).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.pad(((None, None, (0, -1), (1, 2)))).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.pad((1, 2, 0, -1), value=-float('inf')).numpy())
    ```
    """
    if mode not in {"constant", "reflect", "replicate", "circular"}: raise NotImplementedError(f"{mode=} is not supported")
    if (flat:=all(isinstance(p, (int,UOp)) for p in padding)) and len(padding)%2 != 0: raise ValueError("Flat padding must have even number of pads")
    # turn flat padding into group padding
    pX = ((0,0),)*(self.ndim - len(padding)//2) + tuple(zip(padding[-2::-2], padding[::-2])) if flat else padding
    if len(pX) != self.ndim: raise ValueError(f"padding length is improper, {padding=} {self.ndim=}")
    X, pX = self, cast(Tuple[Tuple[sint, sint]], tuple((0,0) if p is None else p for p in pX))
    pads = tuple((smax(pB,0), smax(pA,0)) for pB,pA in pX)
    if mode == "constant":
      def _constant(x,px,v): return F.Pad.apply(x, arg=px) if v == 0 else F.Pad.apply(x, arg=px) + F.Pad.apply(Tensor.ones_like(x), arg=px).where(0,v)
      return _constant(X, pX, value) if all(resolve(p >= 0) for p in flatten(pX)) else \
             _constant(X.shrink(tuple((-smin(pB,0),smin(pA+s,s)) for (pB,pA),s in zip(pX, X.shape))), pads, value)
    assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
    if mode == "circular":
      if any(pB>sh or pA>sh for (pB,pA),sh in zip(pX, X.shape)): raise ValueError('Padding value causes wrapping around more than once.')
      if any(pB<0 or pA<0 for pB,pA in pX): raise NotImplementedError("Negative pads with circular pads is not supported")
      orig_shape, X = X.shape, X.repeat(tuple(1 + bool(pB) + bool(pA) for pB,pA in pads))
      return X.shrink(tuple((0 if pB == 0 else osh-pB, xsh if pA == 0 else xsh-osh+pA) for (pB,pA),osh,xsh in zip(pads, orig_shape, X.shape)))
    for d,(pB,pA) in enumerate(pads):
      if mode == "reflect":
        if pB >= (s:=X.shape[d]) or pA>=s: raise ValueError(f"Padding ({pB}, {pA}) should be less than the input size={s} for dim={d}.")
        slcB, slcA, = slice(pB,0,-1), slice(s-2 if s-2>=0 else None, s-2-pA if s-2-pA>=0 else None, -1)
        xB, xA = (X[[slc if i == d else slice(None) for i in range(X.ndim)]] if p > 0 else None for slc, p in ((slcB, pB), (slcA, pA)))
      if mode == "replicate":
        shrB, shrA, = tuple((0,1) if i==d else None for i in range(X.ndim)), tuple((X.shape[i]-1,X.shape[i]) if i==d else None for i in range(X.ndim))
        xB, xA = (X.shrink(shr).expand(tuple(p if i==d else None for i in range(X.ndim))) if p > 0 else None for shr, p in ((shrB, pB), (shrA, pA)))
      X = Tensor.cat(*(X_ for X_ in (xB, X, xA) if X_ is not None), dim=d)
    return X.shrink(tuple((-min(pB,0), min(pA+s,s)) for (pB,pA),s in zip(pX, X.shape)))

  # ***** movement high level ops *****

  # Supported Indexing Implementations:
  #   1. Int indexing (no copy)
  #     - for all dims where there's int, shrink -> reshape
  #     - negative indices are taken relative to the end of the sequence, so X[-2] returns the 2nd-to-last element
  #     - X = Tensor.rand(4,5,9); X[2,-2] shrinks the Tensor to X.shrink(((2, 3), (3, 4), (0, 9))) -> X.shape=(1,1,9)
  #     - Then we reshape (collapse) the int dim away such that for X: (1,1,9) -> (9,)
  #   2. Slice indexing (no copy)
  #     - for all dims where slice is start:end:stride, shrink -> Optional[flip] -> pad -> reshape -> shrink
  #     - first shrink the Tensor to X.shrink(((start, end),))
  #     - then we apply stride through Optional[flip] -> pad -> reshape -> shrink
  #       - flip where dim value is negative
  #       - pad on dims to be multiple of strides, such that reshaping [dim_size_padded] -> [dim_size_padded // stride, stride] is possible
  #       - shrink [dim_size_padded // stride, stride] -> [dim_size_padded // stride, 1]
  #       - reshape [dim_size_padded // stride, 1] -> [dim_size_padded // stride] and now you have your stride
  #   3. None indexing (no copy)
  #     - reshape (inject) a dim at the dim where there's None
  #   4. Tensor indexing (copy)
  #     - use Tensor.arange == tensor_index to create masks for dims with Tensors (adds a dim for each mask)
  #     - combine masks together with mul
  #     - apply mask to self by mask * self
  #     - sum reduce away the extra dims added from creating masks
  # Tiny Things:
  #   1. Supported indices: Union[int, slice, Tensor, None, List, Tuple, Ellipsis]
  #     - for any list, List[Union[List, Tuple, int]], must have homogeneous shape
  #     - for any tuple, Tuple[Union[List, Tuple, int]], must have homogeneous shape
  #   2. Bool indexing is not supported
  #   3. Out of bounds Tensor indexing results in 0
  #     - e.g: Tensor([1, 2, 3])[Tensor([4, 3, 2])] -> [0, 0, 3] index 4 and 3 are out of bounds
  def _getitem(self, indices, v: Optional[Tensor] = None) -> Tensor:
    # wrap single index into a list
    if (isinstance(indices, list) and all_int(indices)) or not isinstance(indices, (tuple, list)): indices = [indices]
    # turn scalar Tensors into const val for int indexing if possible
    x, indices = self, [self._to_const_val(i) if isinstance(i, Tensor) and i.shape == () else i for i in indices]

    # filter ellipsis and fill with slice(None) or fill rest of indices with slice(None)
    if len(ellipsis_idx := [dim for dim, i in enumerate(indices) if i is Ellipsis]) > 1: raise IndexError("indices can only have a single ellipsis")
    fill_idx = ellipsis_idx[0] if ellipsis_idx else len(indices)
    num_indices = len(indices) - len(ellipsis_idx) - sum(1 for i in indices if i is None)
    if num_indices > self.ndim: raise IndexError(f"too many {num_indices=} for {self.ndim=}")
    indices[fill_idx:fill_idx+1] = [slice(None)] * (self.ndim - num_indices)

    indices_parsed, dim = [], 0
    for index in indices:
      size = 1 if index is None else self.shape[dim]
      boundary, stride = [0, size], 1  # defaults
      match index:
        case list() | tuple() | Tensor():
          if not isinstance(index, Tensor): index = Tensor(index, self.device, requires_grad=False)
          if not dtypes.is_int(index.dtype): raise IndexError(f"index dtype {index.dtype} is not supported")
          index = (index.to(self.device) < 0).where(size, 0) + index # treat negative index values
        case int() | UOp(): # sint
          if index >= size or index < -size: raise IndexError(f"{index=} is out of bounds with {size=}")
          boundary = [index, index+1] if index >= 0 else [index+size, index+size+1]
        case slice():
          if index.step == 0: raise ValueError(f"{index=} cannot have 0 as step")
          if not all(isinstance(s,int) or s is None for s in (index.start,index.stop,index.step)): raise TypeError("only int slicing is supported")
          # handle int slicing
          *boundary, stride = index.indices(cast(SupportsIndex, size))
          if stride * (boundary[1] - boundary[0]) < 0: boundary = [0, 0]
          elif stride < 0: boundary = [boundary[1] + 1, boundary[0] + 1]
          # update size for slice
          size = ceildiv((boundary[1] - boundary[0]), abs(stride))
        case None: pass # do nothing
        case _: raise IndexError(f"{type(index).__name__} indexing is not supported")
      indices_parsed.append({"index":index, "size":size, "boundary":tuple(boundary), "stride":stride})
      if index is not None: dim += 1

    # movement op indexing
    if mops := [i for i in indices_parsed if i['index'] is not None]:
      # flip negative strides
      shrinks, strides = zip(*((i['boundary'], i['stride']) for i in mops))
      x = x.shrink(shrinks).flip(tuple(i for i,st in enumerate(strides) if st < 0))
      # handle stride != 1 or -1
      if any(abs(st) != 1 for st in strides):
        strides = tuple(abs(s) for s in strides)
        # pad shape to multiple of stride
        if not all_int(x.shape): raise RuntimeError("symbolic shape not supprted")
        x = x.pad(tuple((0, round_up(s, st) - s) for s, st in zip(x.shape, strides)))
        x = x.reshape(tuple(flatten((s // st, st) for s, st in zip(x.shape, strides))))
        x = x.shrink(tuple(flatten(((0, s), (0, 1)) for s in x.shape[::2]))).reshape(x.shape[::2])

    # dim injection from None by including None dim size (which is 1) and dim collapse by skipping int dim size
    x = x.reshape(tuple(index['size'] for index in indices_parsed if not isinstance(index['index'], int)))

    # tensor indexing
    if tops := [(d,i) for d,i in enumerate(i_ for i_ in indices_parsed if not isinstance(i_['index'], int)) if isinstance(i['index'], Tensor)]:
      # unload the tensor object into actual tensors
      dims, tensors, masks = [d for d,_ in tops], cast(list[Tensor], [i['index'] for _,i in tops]), []
      pre_reduce_shape = x.shape[:dims[0]] + (big_shape := _broadcast_shape(*(t.shape for t in tensors))) + x.shape[dims[0]:]

      # create index masks
      for dim, tensor in zip(dims, tensors):
        try: i = tensor.reshape(tensor.shape + (1,)*(x.ndim - dims[0])).expand(pre_reduce_shape)
        except ValueError as e: raise IndexError(f"cannot broadcast indices: {e}") from e
        masks.append(i._one_hot_along_dim(num_classes=x.shape[dim], dim=(dim - x.ndim)))

      # reduce masks to 1 mask
      mask: Tensor = functools.reduce(lambda x,y: x.mul(y), masks)

      # inject 1's for the extra dims added in create masks
      reshape_arg = x.shape[:dims[0]] + (1,) * len(big_shape) + x.shape[dims[0]:]
      # sum reduce the extra dims introduced in create masks
      x = (x.reshape(reshape_arg) * mask).sum(sum_axis:=tuple(d + len(big_shape) for d in dims), acc_dtype=x.dtype)

      # special permute case
      if dims[0] != 0 and len(dims) != 1 and tuple(dims) != tuple(range(dims[0], dims[-1]+1)):
        x = x.permute(*range(dims[0], dims[0]+len(big_shape)), *range(0, dims[0]), *range(dims[0]+len(big_shape), x.ndim))

      # for advanced setitem, returns whole tensor with indices replaced
      if v is not None:
        vb = v.cast(self.dtype)._broadcast_to(_broadcast_shape(x.shape, v.shape))
        # add back reduced dims from sum
        for dim in sum_axis: vb = vb.unsqueeze(dim)
        # run _masked_setitem on tuple of axis that is to be reduced to match self.shape
        x = _masked_setitem(self, vb, mask, tuple(range(dims[0], dims[0] + len(big_shape))))

    return x

  def __getitem__(self, indices) -> Tensor:
    return self._getitem(indices)

  def __setitem__(self, indices, v:Union[Tensor, ConstType]) -> None:
    if isinstance(self.device, str) and self.device.startswith("DISK"):
      self._getitem(indices).assign(v)
      return
    # NOTE: check that setitem target is valid first
    if not all(lb.st.contiguous for lb in self.lazydata.lbs): raise RuntimeError("setitem target needs to be contiguous")
    if not isinstance(v, (Tensor, float, int, bool)): raise TypeError(f"can't set a {type(v).__name__} to a Tensor")
    if not isinstance(v, Tensor): v = Tensor(v, device=self.device, dtype=self.dtype)
    if self.requires_grad or v.requires_grad: raise NotImplementedError("setitem with requires_grad is not supported")

    res = self.realize()._getitem(indices, v)
    # if shapes match and data is not shared it's a copy and we assign to self
    if res.shape == self.shape and res.lazydata is not self.lazydata:
      self.assign(res).realize()
    else: # no copy, basic setitem
      v = v.cast(res.dtype)._broadcast_to(_broadcast_shape(res.shape, v.shape)).contiguous()
      res.assign(v).realize()

  def gather(self:Tensor, dim:int, index:Tensor) -> Tensor:
    """
    Gathers values along an axis specified by `dim`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2], [3, 4]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.gather(1, Tensor([[0, 0], [1, 0]])).numpy())
    ```
    """
    assert index.ndim == self.ndim, f"self.ndim must equal index.ndim, {self.ndim=}, {index.ndim=}"
    dim = self._resolve_dim(dim)
    assert all(s >= i for d,(s,i) in enumerate(zip(self.shape, index.shape)) if d != dim), "requires self.shape[d] >= index.shape[d] for all d != dim"
    index = index.to(self.device)
    x = self.shrink(tuple((0, i) if d != dim else None for d,i in enumerate(index.shape))).unsqueeze(-1).transpose(-1, dim)
    return (x * index.unsqueeze(-1)._one_hot_along_dim(self.shape[dim])).sum(-1, acc_dtype=self.dtype)

  def cat(self:Tensor, *args:Tensor, dim:int=0) -> Tensor:
    """
    Concatenates self with other `Tensor` in `args` along an axis specified by `dim`.
    All tensors must have the same shape except in the concatenating dimension.

    ```python exec="true" source="above" session="tensor" result="python"
    t0, t1, t2 = Tensor([[1, 2]]), Tensor([[3, 4]]), Tensor([[5, 6]])
    print(t0.cat(t1, t2, dim=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t0.cat(t1, t2, dim=1).numpy())
    ```
    """
    dim = self._resolve_dim(dim)
    for arg in args: assert arg.ndim==self.ndim and all(ti==ai for i,(ti,ai) in enumerate(zip(self.shape, arg.shape)) if i!=dim)
    tensors = [self, *args]
    dim_cumsum = list(itertools.accumulate([t.shape[dim] for t in tensors], initial=0))
    for i,t in enumerate(tensors): tensors[i] = t.pad([(dim_cumsum[i], dim_cumsum[-1]-dim_cumsum[i+1]) if j==dim else None for j in range(t.ndim)])
    return functools.reduce(Tensor.add, tensors)

  def stack(self:Tensor, *args:Tensor, dim:int=0) -> Tensor:
    """
    Concatenates self with other `Tensor` in `args` along a new dimension specified by `dim`.

    ```python exec="true" source="above" session="tensor" result="python"
    t0, t1, t2 = Tensor([1, 2]), Tensor([3, 4]), Tensor([5, 6])
    print(t0.stack(t1, t2, dim=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t0.stack(t1, t2, dim=1).numpy())
    ```
    """
    # checks for shapes and number of dimensions delegated to cat
    return Tensor.cat(*[t.unsqueeze(dim) for t in [self, *args]], dim=dim)

  def repeat_interleave(self, repeats:int, dim:Optional[int]=None) -> Tensor:
    """
    Repeat elements of a tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t.repeat_interleave(2).numpy())
    ```
    """
    x, dim = (self.flatten(), 0) if dim is None else (self, self._resolve_dim(dim))
    shp = x.shape
    return x.reshape(*shp[:dim+1], 1, *shp[dim+1:]).expand(*shp[:dim+1], repeats, *shp[dim+1:]).reshape(*shp[:dim], shp[dim]*repeats, *shp[dim+1:])

  def repeat(self, repeats, *args) -> Tensor:
    """
    Repeats tensor number of times along each dimension specified by `repeats`.
    `repeats` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t.repeat(4, 2).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.repeat(4, 2, 1).shape)
    ```
    """
    repeats = argfix(repeats, *args)
    base_shape = _align_left(self.shape, repeats)[0]
    unsqueezed_shape = flatten([[1, s] for s in base_shape])
    expanded_shape = flatten([[r, s] for r,s in zip(repeats, base_shape)])
    final_shape = [r*s for r,s in zip(repeats, base_shape)]
    return self.reshape(unsqueezed_shape).expand(expanded_shape).reshape(final_shape)

  def _resolve_dim(self, dim:int, *, extra:bool=False) -> int:
    total = self.ndim + int(extra)
    if not -max(1, total) <= dim <= max(1, total)-1: raise IndexError(f"{dim=} out of range {[-max(1, total), max(1, total)-1]}")
    return dim + total if dim < 0 else dim

  def split(self, sizes:Union[int, List[int]], dim:int=0) -> Tuple[Tensor, ...]:
    """
    Splits the tensor into chunks along the dimension specified by `dim`.
    If `sizes` is an integer, it splits into equally sized chunks if possible, otherwise the last chunk will be smaller.
    If `sizes` is a list, it splits into `len(sizes)` chunks with size in `dim` according to `size`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(10).reshape(5, 2)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    split = t.split(2)
    print("\\n".join([repr(x.numpy()) for x in split]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    split = t.split([1, 4])
    print("\\n".join([repr(x.numpy()) for x in split]))
    ```
    """
    assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
    dim = self._resolve_dim(dim)
    if isinstance(sizes, int): sizes = [min(sizes, self.shape[dim]-i) for i in range(0, max(1, self.shape[dim]), max(1, sizes))]
    assert sum(sizes) == self.shape[dim], f"expect sizes to sum exactly to {self.shape[dim]}, but got {sum(sizes)}"
    return tuple(self[sl] for sl in [tuple([slice(None)]*dim + [slice(sum(sizes[:i]), sum(sizes[:i + 1]))]) for i in range(len(sizes))])

  def chunk(self, chunks:int, dim:int=0) -> List[Tensor]:
    """
    Splits the tensor into `chunks` number of chunks along the dimension `dim`.
    If the tensor size along `dim` is not divisible by `chunks`, all returned chunks will be the same size except the last one.
    The function may return fewer than the specified number of chunks.

    ```python exec="true" source="above" session="tensor" result="python"
    chunked = Tensor.arange(11).chunk(6)
    print("\\n".join([repr(x.numpy()) for x in chunked]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    chunked = Tensor.arange(12).chunk(6)
    print("\\n".join([repr(x.numpy()) for x in chunked]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    chunked = Tensor.arange(13).chunk(6)
    print("\\n".join([repr(x.numpy()) for x in chunked]))
    ```
    """
    assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
    assert chunks > 0, f"expect chunks to be greater than 0, got: {chunks}"
    dim = self._resolve_dim(dim)
    return list(self.split(ceildiv(self.shape[dim], chunks) if self.shape[dim] else [0]*chunks, dim=dim))

  def meshgrid(self:Tensor, *args:Tensor, indexing:Union[Literal["ij"], Literal["xy"]]="ij") -> Tuple[Tensor, ...]:
    """
    Generates coordinate matrices from coordinate vectors.
    Input tensors can be scalars or 1D tensors.

    `indexing` determines how the output grids are aligned.
    `ij` indexing follows matrix-style indexing and `xy` indexing follows Cartesian-style indexing.

    ```python exec="true" source="above" session="tensor" result="python"
    x, y = Tensor([1, 2, 3]), Tensor([4, 5, 6])
    grid_x, grid_y = x.meshgrid(y)
    print(grid_x.numpy())
    print(grid_y.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    grid_x, grid_y = x.meshgrid(y, indexing="xy")
    print(grid_x.numpy())
    print(grid_y.numpy())
    ```
    """
    if indexing not in ("ij", "xy"): raise RuntimeError(f'indexing must be in ("ij", "xy"), got {indexing}')
    if len(tensors:=(self, *args)) == 1: return tensors
    basis = tuple(range(len(tensors))) if indexing == "ij" else (1, 0) + tuple(range(2, len(tensors)))
    tensors = tuple(t.reshape((-1,) + (1,)*(len(args) - i)) for i,t in zip(basis, tensors))
    output_shape = _broadcast_shape(*(t.shape for t in tensors))
    return tuple(t._broadcast_to(output_shape) for t in tensors)

  def squeeze(self, dim:Optional[int]=None) -> Tensor:
    """
    Returns a tensor with specified dimensions of input of size 1 removed.
    If `dim` is not specified, all dimensions with size 1 are removed.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.zeros(2, 1, 2, 1, 2)
    print(t.squeeze().shape)
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.squeeze(0).shape)
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.squeeze(1).shape)
    ```
    """
    if dim is None: return self.reshape(tuple(dim for dim in self.shape if dim != 1))
    dim = self._resolve_dim(dim)
    return self if not self.ndim or self.shape[dim] != 1 else self.reshape(self.shape[:dim] + self.shape[dim+1:])

  def unsqueeze(self, dim:int) -> Tensor:
    """
    Returns a tensor with a new dimension of size 1 inserted at the specified `dim`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(t.unsqueeze(0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.unsqueeze(1).numpy())
    ```
    """
    dim = self._resolve_dim(dim, extra=True)
    return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])

  @property
  def T(self) -> Tensor:
    """`.T` is an alias for `.transpose()`."""
    return self.transpose()

  def transpose(self, dim0=1, dim1=0) -> Tensor:
    """
    Returns a tensor that is a transposed version of the original tensor.
    The given dimensions `dim0` and `dim1` are swapped.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.transpose(0, 1).numpy())
    ```
    """
    order = list(range(self.ndim))
    order[dim0], order[dim1] = order[dim1], order[dim0]
    return self.permute(order)

  def flatten(self, start_dim=0, end_dim=-1):
    """
    Flattens the tensor by reshaping it into a one-dimensional tensor.
    If `start_dim` or `end_dim` are passed, only dimensions starting with `start_dim` and ending with `end_dim` are flattened.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(8).reshape(2, 2, 2)
    print(t.flatten().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.flatten(start_dim=1).numpy())
    ```
    """
    start_dim, end_dim = self._resolve_dim(start_dim), self._resolve_dim(end_dim)
    return self.reshape(self.shape[:start_dim] + (prod(self.shape[start_dim:end_dim+1]), ) + self.shape[end_dim+1:])

  def unflatten(self, dim:int, sizes:Tuple[int,...]):
    """
    Unflattens dimension `dim` of the tensor into multiple dimensions specified by `sizes`. `Tensor.flatten()` is the inverse of this function.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(3, 4, 1).unflatten(1, (2, 2)).shape)
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(3, 4, 1).unflatten(1, (-1, 2)).shape)
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(5, 12, 3).unflatten(-2, (2, 2, 3, 1, 1)).shape)
    ```
    """
    dim = self._resolve_dim(dim)
    return self.reshape(self.shape[:dim] + sizes + self.shape[dim+1:])

  def roll(self, shifts:Union[int, Tuple[int, ...]], dims:Union[int, Tuple[int, ...]]) -> Tensor:
    """
    Rolls the tensor along specified dimension(s).
    The rolling operation is circular, meaning that elements that go beyond the edge are wrapped around to the beginning of the dimension.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(4)
    print(t.roll(shifts=1, dims=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.roll(shifts=-1, dims=0).numpy())
    ```
    """
    dims, rolled = tuple(self._resolve_dim(d) for d in make_tuple(dims, 1)), self
    for dim, shift in zip(dims, make_tuple(shifts, 1)):
      shift = shift % self.shape[dim]
      rolled = Tensor.cat(rolled[tuple(slice(None) if i != dim else slice(-shift, None) for i in range(rolled.ndim))],
                          rolled[tuple(slice(None) if i != dim else slice(None, -shift) for i in range(rolled.ndim))], dim=dim)
    return rolled

  # ***** reduce ops *****

  def _reduce(self, fxn:Type[Function], axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False) -> Tensor:
    axis = tuple(self._resolve_dim(x) for x in (range(self.ndim) if axis is None else make_tuple(axis, 1)))
    if self.ndim == 0: axis = ()
    ret = fxn.apply(self, axis=axis)
    return ret if keepdim else ret.reshape(tuple(s for i,s in enumerate(self.shape) if i not in axis))

  def sum(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False, acc_dtype:Optional[DTypeLike]=None):
    """
    Returns the sum of the elements of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the maximum is computed and whether the reduced dimensions are retained.

    You can pass in `acc_dtype` keyword argument to control the data type of the accumulation.
    If not specified, the accumulation data type is chosen based on the input tensor's data type.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.sum().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.sum(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.sum(axis=1).numpy())
    ```
    """
    ret = self.cast(sum_acc_dtype(self.dtype) if acc_dtype is None else acc_dtype)._reduce(F.Sum, axis, keepdim)
    return ret.cast(self.dtype) if acc_dtype is None and self.dtype in (dtypes.float16, dtypes.bfloat16) else ret

  def prod(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False, acc_dtype:Optional[DTypeLike]=None):
    """
    Returns the product of the elements of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the maximum is computed and whether the reduced dimensions are retained.

    You can pass in `acc_dtype` keyword argument to control the data type of the accumulation.
    If not specified, the accumulation data type is chosen based on the input tensor's data type.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, -2, -3, 1, 2, 3]).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.prod().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.prod(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.prod(axis=1).numpy())
    ```
    """
    return self.cast(acc_dtype if acc_dtype is not None else self.dtype)._reduce(F.Prod, axis, keepdim)

  def max(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False):
    """
    Returns the maximum value of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the maximum is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 0, 2], [5, 4, 3]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.max().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.max(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.max(axis=1, keepdim=True).numpy())
    ```
    """
    return self._reduce(F.Max, axis, keepdim)

  def min(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False):
    """
    Returns the minimum value of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the minimum is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 0, 2], [5, 4, 3]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.min().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.min(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.min(axis=1, keepdim=True).numpy())
    ```
    """
    if dtypes.is_int(self.dtype) or self.dtype == dtypes.bool: return ~((~self).max(axis=axis, keepdim=keepdim))
    return -((-self).max(axis=axis, keepdim=keepdim))

  def any(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False):
    """
    Tests if any element evaluates to `True` along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the reduce axis and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[True, True], [True, False], [False, False]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.any().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.any(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.any(axis=1, keepdim=True).numpy())
    ```
    """
    return self.bool().max(axis, keepdim)

  def all(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False):
    """
    Tests if all element evaluates to `True` along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the reduce axis and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[True, True], [True, False], [False, False]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.all().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.all(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.all(axis=1, keepdim=True).numpy())
    ```
    """
    return self.logical_not().any(axis, keepdim).logical_not()

  def mean(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False):
    """
    Returns the mean value of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the mean is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mean().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mean(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mean(axis=1).numpy())
    ```
    """
    output_dtype = self.dtype if dtypes.is_float(self.dtype) else dtypes.float32
    numerator = self.cast(sum_acc_dtype(self.dtype)).sum(axis=axis, keepdim=keepdim)
    return numerator.div(prod([si for si, so in zip(self.shape, self.sum(axis=axis, keepdim=True).shape) if resolve(si != so)])).cast(output_dtype)

  def var(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False, correction=1):
    """
    Returns the variance of the tensor along the specified axis or axes.

    You can pass in `axis`, `keepdim`, and `correction` keyword arguments to control the axis along
    which the variance is computed, whether the reduced dimensions are retained, and the Bessel's correction applied.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.var().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.var(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.var(axis=1).numpy())
    ```
    """
    squares = (self - self.mean(axis=axis, keepdim=True)).square()
    n = prod([si for si, so in zip(self.shape, squares.sum(axis=axis, keepdim=True).shape) if resolve(si != so)])
    return squares.sum(axis=axis, keepdim=keepdim).div(smax([0, n-correction]))

  def std(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False, correction=1):
    """
    Returns the standard deviation of the tensor along the specified axis or axes.

    You can pass in `axis`, `keepdim`, and `correction` keyword arguments to control the axis along
    which the standard deviation is computed, whether the reduced dimensions are retained, and the Bessel's correction applied.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.std().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.std(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.std(axis=1).numpy())
    ```
    """
    return self.var(axis, keepdim, correction).sqrt()

  def std_mean(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False, correction=1):
    """
    Calculates the standard deviation and mean over the dimensions specified by dim.
    Syntactic sugar around `Tensor.std` and `Tensor.mean` to match `torch.std_mean`.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    std, mean = t.std_mean()
    print(std.numpy(), mean.numpy())
    ```
    """
    return self.std(axis, keepdim, correction), self.mean(axis, keepdim)

  def _softmax(self, axis, dtype:Optional[DTypeLike]=None):
    x = self.cast(dtype) if dtype is not None else self
    m = x - x.max(axis=axis, keepdim=True).detach()
    e = m.exp()
    return m, e, e.sum(axis=axis, keepdim=True)

  def softmax(self, axis=-1, dtype:Optional[DTypeLike]=None):
    """
    Applies the softmax function to the tensor along the specified axis.

    Rescales the elements of the tensor such that they lie in the range [0, 1] and sum to 1.

    You can pass in the `axis` keyword argument to control the axis along which the softmax is computed.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.softmax().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.softmax(axis=0).numpy())
    ```
    """
    _, e, ss = self._softmax(axis, dtype)
    return e.div(ss)

  def log_softmax(self, axis=-1, dtype:Optional[DTypeLike]=None):
    """
    Applies the log-softmax function to the tensor along the specified axis.

    The log-softmax function is a numerically stable alternative to the softmax function in log space.

    You can pass in the `axis` keyword argument to control the axis along which the log-softmax is computed.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.log_softmax().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.log_softmax(axis=0).numpy())
    ```
    """
    m, _, ss = self._softmax(axis, dtype)
    return m - ss.log()

  def logsumexp(self, axis=None, keepdim=False):
    """
    Computes the log-sum-exp of the tensor along the specified axis or axes.

    The log-sum-exp function is a numerically stable way to compute the logarithm of the sum of exponentials.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the log-sum-exp is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logsumexp().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logsumexp(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logsumexp(axis=1).numpy())
    ```
    """
    m = self.max(axis=axis, keepdim=True)
    return (self - m).exp().sum(axis=axis, keepdim=keepdim).log() + m.squeeze(axis)

  def logcumsumexp(self, axis=0):
    """
    Computes the log-cumsum-exp of the tensor along the specified axis or axes.

    The log-cumsum-exp function is a numerically stable way to compute the logarithm of the cumulative sum of exponentials.

    You can pass in the `axis` keyword argument to control the axis along which
    the log-cum-sum-exp is computed.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logcumsumexp().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logcumsumexp(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logcumsumexp(axis=1).numpy())
    ```
    """
    m = self.max(axis=axis, keepdim=True)
    return (self - m).exp().cumsum(axis=axis).log() + m

  def argmax(self, axis=None, keepdim=False):
    """
    Returns the indices of the maximum value of the tensor along the specified axis.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the maximum is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 0, 2], [5, 4, 3]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmax().numpy()) # Returns the index of the maximum value in the flattened tensor.
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmax(axis=0).numpy()) # Returns the indices of the maximum values along axis 0.
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmax(axis=1).numpy()) # Returns the indices of the maximum values along axis 1.
    ```
    """
    if axis is None: return self.flatten().argmax(0)
    axis = self._resolve_dim(axis)
    m = self == self.max(axis=axis, keepdim=True)
    idx = m * Tensor.arange(self.shape[axis],0,-1, requires_grad=False, device=self.device).reshape(self.shape[axis], *[1]*(self.ndim-axis-1))
    return (self.shape[axis]-idx.max(axis=axis, keepdim=keepdim)).cast(dtypes.int32)

  def argmin(self, axis=None, keepdim=False):
    """
    Returns the indices of the minimum value of the tensor along the specified axis.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the minimum is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 0, 2], [5, 4, 3]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmin().numpy()) # Returns the index of the minimum value in the flattened tensor.
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmin(axis=0).numpy()) # Returns the indices of the minimum values along axis 0.
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmin(axis=1).numpy()) # Returns the indices of the minimum values along axis 1.
    ```
    """
    return (-self).argmax(axis=axis, keepdim=keepdim)

  def rearrange(self, formula: str, **sizes) -> Tensor:
    """
    Rearranges input according to formula

    See: https://einops.rocks/api/rearrange/

    ```python exec="true" source="above" session="tensor" result="python"
    x = Tensor([[1, 2], [3, 4]])
    print(Tensor.rearrange(x, "batch channel -> (batch channel)).numpy())
    ```
    """
    def parse_formula(formula: str):
      tokens = f" {formula} ".replace("…", "...").replace("(", " ( ").replace(")", " ) ").replace(" ", "  ").replace(" 1 ", " ( ) ").split()
      lparens, rparens = map(lambda x: [i for i, ch in enumerate(tokens) if ch == x], ("(", ")"))
      pairs = list(zip(lparens, rparens))
      assert len(lparens) == len(rparens) and sorted(flatten(pairs)) == flatten(pairs), "bracket mismatch"
      return [name for name in tokens if name not in ("(", ")")], [(s - 2*i, e - 1 - 2*i) for i, (s, e) in enumerate(pairs)]

    assert formula.count("->") == 1, 'need exactly one "->" in formula'

    (lhs, unflatten_dims), (rhs, flatten_dims) = map(parse_formula, formula.split("->"))

    for name in sizes: assert name in lhs, f"axis {name} is not used in transform"
    assert sorted(lhs) == sorted(rhs) and len(lhs) == len(set(lhs)), f"name mismatch in {formula}"
    for name in flatten((lhs, rhs)): assert name == "..." or (name.isidentifier() and "_" not in (name[0], name[-1])), f"invalid axis name {name}"
    assert "..." not in flatten([lhs[s:e] for s, e in unflatten_dims]), f"cannot have collapsed ellipsis (...) in lhs of {formula}"
    assert lhs.count("...") <= 1, f"too many ellipses in {formula}"

    # resolve ellipsis
    if "..." in lhs: ell_len = len(self.shape) - len(lhs) + 1 + sum(e - s - 1 for s, e in unflatten_dims)
    lhs, rhs = map(lambda l: l[:(i:=l.index("..."))] + [f"...{j}" for j in range(ell_len)] + l[i + 1:] if "..." in l else l, (lhs, rhs))
    unflatten_dims = [(s + (ell_len - 1 if "...0" in lhs[:s] else 0), e + (ell_len - 1 if "...0" in lhs[:e] else 0)) for s, e in unflatten_dims]
    flatten_dims = [(s + (ell_len - 1 if "...0" in rhs[:s] else 0), e + (ell_len - 1 if "...0" in rhs[:e] else 0)) for s, e in flatten_dims]

    # apply movement ops in order unflatten -> permute -> flatten/unsqueeze
    t = functools.reduce(lambda x, dims: x.unflatten(dims[0], tuple(sizes.get(lhs[d], -1) for d in range(*dims))), unflatten_dims, self)
    for i, name in enumerate(lhs): assert (name not in sizes) or sizes[name] == t.shape[i], f"size provided for dimension {name} incorrect"
    t = t.permute([lhs.index(name) for name in rhs])
    return functools.reduce(lambda x, dims: x.flatten(dims[0], dims[1] - 1) if dims[0]<dims[1] else x.unsqueeze(dims[0]), reversed(flatten_dims), t)

  @staticmethod
  def einsum(formula:str, *operands:Tensor|Sequence[Tensor], acc_dtype:Optional[DTypeLike]=None) -> Tensor:
    """
    Sums the product of the elements of the input tensors according to a formula based on the Einstein summation convention.

    See: https://pytorch.org/docs/stable/generated/torch.einsum.html

    ```python exec="true" source="above" session="tensor" result="python"
    x = Tensor([[1, 2], [3, 4]])
    y = Tensor([[5, 6], [7, 8]])
    print(Tensor.einsum("ij,ij->", x, y).numpy())
    ```
    """
    def parse_formula(formula:str, *operands:Tensor):
      if "..." in (formula := formula.replace(" ", "")):
        ell_chars, ell_longest = "".join(set(string.ascii_letters) - set(formula)), 0
        for i, inp in enumerate(filter(lambda x: "..." in x, inputs := formula.split("->")[0].split(","))):
          if (ell_count := max(operands[i].ndim, 1) - (len(inp) - len("..."))) > ell_longest: ell_longest = ell_count
          inputs[i] = inp.replace("...", ell_chars[-ell_count:])
        inputs_str, out_ellipse = ",".join(inputs), ell_chars[-ell_longest:]
        return (inputs_str, formula.split("->")[1].replace("...", out_ellipse)) if "->" in formula else \
          (inputs_str, out_ellipse + ''.join(sorted(c for c in inputs_str if inputs_str.count(c) == 1 and c.isalpha() and c not in out_ellipse)))
      return formula.split("->") if "->" in formula else (formula, ''.join(c for c in sorted(formula) if formula.count(c) == 1 and c.isalpha()))

    xs:Tuple[Tensor, ...] = argfix(*operands)
    inputs_str, output = parse_formula(formula, *xs)
    inputs = inputs_str.split(",")
    assert len(xs) == len(inputs), f"number of inputs doesn't match number of operands in formula, expected {len(inputs)}, got {len(xs)}"

    # map the value of each letter in the formula
    letter_val = sorted(merge_dicts([dict(zip(letters, tensor.shape)) for letters, tensor in zip(inputs, xs)]).items())

    xs_:List[Tensor] = []
    lhs = [sorted(enumerate(s), key=lambda e:e[1]) for s in inputs]
    for x,(order,letters) in zip(xs, [list(zip(*l)) for l in lhs]):
      # permute to the sorted letter order, then reshape/expand to create dimensions for the missing letters
      xs_.append(x.permute(order).reshape([val if letter in letters else 1 for letter,val in letter_val]).expand([val for _,val in letter_val]))

    # ordinal encode the output alphabet
    rhs_order = argsort(argsort(list(output)))

    # sum over all axes that's not in the output, then permute to the output order
    return functools.reduce(lambda a,b:a*b, xs_) \
      .sum(axis=[axis for axis,(letter,_) in enumerate(letter_val) if letter not in output], acc_dtype=acc_dtype).permute(rhs_order)

  # ***** processing ops *****

  def _pool(self, k_:Tuple[sint, ...], stride:Union[Tuple[int, ...], int]=1, dilation:Union[Tuple[int, ...], int]=1) -> Tensor:
    assert len(self.shape) >= len(k_), f"can't pool {self.shape} with {k_}"
    s_, d_ = make_tuple(stride, len(k_)), make_tuple(dilation, len(k_))
    assert len(k_) == len(s_) == len(d_), f"stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}"
    noop, i_ = [None] * (self.ndim-len(k_)), self.shape[-len(k_):]
    assert all(resolve(d*(k-1)+1 <= i) for k,d,i in zip(k_,d_,i_)), "kernel size cannot be greater than actual input size"
    o_ = [ceildiv(i-d*(k-1), s) for i,d,k,s in zip(i_,d_,k_,s_)]
    if any(resolve(k > s) for k,s in zip(k_,s_)) or any(d != 1 for d in d_):
      # input size scaling factor to make sure shrink for stride is possible
      f_ = [1 + int(resolve(o*s > i+d)) for o,s,i,d in zip(o_,s_,i_,d_)]
      # # repeats such that we don't need padding
      x = self.repeat([1]*len(noop) + [ceildiv(k*(i*f+d),i) for k,i,d,f in zip(k_,i_,d_,f_)])
      # handle dilation
      x = x.shrink(tuple(noop + [(0,k*(i*f+d)) for k,i,d,f in zip(k_,i_,d_,f_)])).reshape(noop + flatten((k,(i*f+d)) for k,i,d,f in zip(k_,i_,d_,f_)))
      # handle stride
      x = x.shrink(tuple(noop + flatten(((0,k), (0,o*s)) for k,o,s in zip(k_,o_,s_)))).reshape(noop + flatten((k,o,s) for k,o,s in zip(k_,o_,s_)))
      x = x.shrink(tuple(noop + flatten(((0,k), (0,o), (0,1)) for k,o in zip(k_,o_)))).reshape(noop + flatten((k,o) for k,o in zip(k_,o_)))
      # permute to move reduce to the end
      return x.permute(*range(len(noop)), *[len(noop)+i*2+1 for i in range(len(i_))], *[len(noop)+i*2 for i in range(len(i_))])
    # TODO: once the shapetracker can optimize well, remove this alternative implementation
    x = self.pad(tuple(noop + [(0, max(0,o*s-i)) for i,o,s in zip(i_,o_,s_)])).shrink(tuple(noop + [(0,o*s) for o,s in zip(o_,s_)]))
    x = x.reshape(noop + flatten(((o,s) for o,s in zip(o_,s_))))
    x = x.shrink(tuple(noop + flatten(((0,o), (0,k)) for o,k in zip(o_,k_))))
    return x.permute(*range(len(noop)), *[len(noop)+i*2 for i in range(len(i_))], *[len(noop)+i*2+1 for i in range(len(i_))])

  def _padding2d(self, padding:Union[int, Sequence[int]], dims:int) -> Sequence[int]:
    return [padding]*2*dims if isinstance(padding, int) else (padding if len(padding) == 2*dims else [p for p in padding for _ in range(2)][::-1])

  def _ceil_mode_padding2d(self,k_:Tuple[sint, ...], s_:Union[Tuple[int, ...], int], d_:Union[Tuple[int, ...], int],
                           p_:Union[Tuple[int, ...], int]) -> Sequence[int]:
    (d_,s_,p_), i_ = (make_tuple(x, len(k_)) for x in (d_,s_,p_)), self.shape[-len(k_):]
    # https://arxiv.org/pdf/1603.07285 section 5.1, relationship 15.
    o_ = [ceildiv(i+2*p - (d*(k-1)+1), s) + 1 for i,d,k,s,p in zip(i_,d_,k_,s_,p_)]
    pads = list(self._padding2d(p_, len(k_)))
    # we have to do additional padding before `_pool` so that `o_` in `_pool` is calculated correctly
    # `s*(o-1) + (d*(k-1)+1) - (i+2*p)` -> last_sliding_window_start + full_kernel_size - padded_input_shape
    # we decrease padding in the case that a sliding window starts in the end padded region, thereby decreasing `o_` in `_pool`
    # `smax(s*(o-1) - (p+i-1), 0)` -> last_sliding_window_start - (left_pad + input_size - zero_offset)
    for dim,(o,i,s,p,k,d) in enumerate(zip(o_,i_,s_,p_,k_,d_)): pads[-1-dim*2] += s*(o-1) + (d*(k-1)+1) - (i+2*p) - smax(s*(o-1) - (p+i-1), 0)
    return pads

  # NOTE: these work for more than 2D
  def avg_pool2d(self, kernel_size=(2,2), stride=None, dilation=1, padding=0, ceil_mode=False, count_include_pad=True):
    """
    Applies average pooling over a tensor.

    When `ceil_mode` is set to True, output shape will be determined using ceil division.
    When `count_include_pad` is set to False, zero padding will not be included in the averaging calculation.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d pooling and instead works for any number of dimensions.

    See: https://paperswithcode.com/method/average-pooling

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(25).reshape(1, 1, 5, 5)
    print(t.avg_pool2d().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.avg_pool2d(ceil_mode=True).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.avg_pool2d(padding=1).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.avg_pool2d(padding=1, count_include_pad=False).numpy())
    ```
    """
    axis = tuple(range(-len(k_ := make_tuple(kernel_size, 2)), 0))
    reg_pads, ceil_pads = self._padding2d(padding,len(k_)), self._ceil_mode_padding2d(k_, stride if stride is not None else k_, dilation, padding)
    def pool(x:Tensor, padding_:Sequence[int]) -> Tensor: return x.pad(padding_)._pool(k_, stride if stride is not None else k_, dilation)
    if not count_include_pad:
      pads = ceil_pads if ceil_mode else reg_pads
      return pool(self, pads).sum(axis) / pool(self.ones_like(), pads).sum(axis)
    if not ceil_mode: return pool(self, reg_pads).mean(axis)
    return pool(self, ceil_pads).sum(axis) / pool(self.pad(reg_pads).ones_like(), tuple(cp-rp for cp,rp in zip(ceil_pads, reg_pads))).sum(axis)

  def max_pool2d(self, kernel_size=(2,2), stride=None, dilation=1, padding=0, ceil_mode=False):
    """
    Applies max pooling over a tensor.

    When `ceil_mode` is set to True, output shape will be determined using ceil division.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d pooling and instead works for any number of dimensions.

    See: https://paperswithcode.com/method/max-pooling

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(25).reshape(1, 1, 5, 5)
    print(t.max_pool2d().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.max_pool2d(ceil_mode=True).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.max_pool2d(padding=1).numpy())
    ```
    """
    k_ = make_tuple(kernel_size, 2)
    pads = self._ceil_mode_padding2d(k_, stride if stride is not None else k_, dilation, padding) if ceil_mode else self._padding2d(padding, len(k_))
    return self.pad(pads, value=dtypes.min(self.dtype))._pool(k_, stride if stride is not None else k_, dilation).max(tuple(range(-len(k_), 0)))

  def conv2d(self, weight:Tensor, bias:Optional[Tensor]=None, groups=1, stride=1, dilation=1, padding:int|Tuple[int, ...]=0,
             acc_dtype:Optional[DTypeLike]=None) -> Tensor:
    """
    Applies a convolution over a tensor with a given `weight` and optional `bias`.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d convolutions and instead works for any number of dimensions.

    See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(1, 1, 3, 3)
    w = Tensor.ones(1, 1, 2, 2)
    print(t.conv2d(w).numpy())
    ```
    """
    if IMAGE: return self.image_conv2d(weight, bias, groups, stride, dilation, padding, acc_dtype)
    (bs,cin_), (cout,cin), HW = self.shape[:2], weight.shape[:2], weight.shape[2:]
    assert groups*cin == cin_ and len(self.shape) == len(weight.shape), f"Input Tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"  # noqa: E501
    if isinstance(padding, (tuple,list)): assert len(padding) == 2*len(HW) or len(padding) == len(HW), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {self.shape}"  # noqa: E501
    padding_ = self._padding2d(padding, len(HW))

    # conv2d is a pooling op (with padding)
    x = self.pad(padding_)._pool(HW, stride, dilation)   # (bs, groups*cin, oy, ox, H, W)
    rcout, oyx = cout//groups, x.shape[2:-len(HW)]
    if not all(x == 3 for x in HW) or stride != 1 or dilation != 1 or not WINO:
      # normal conv
      x = x.reshape(bs, groups, cin, 1, *oyx, *HW).expand(bs, groups, cin, rcout, *oyx, *HW).permute(0,1,3,*[4+i for i in range(len(oyx))],2,*[4+len(oyx)+i for i in range(len(HW))])  # noqa: E501

      # conv! broadcasted to (bs, groups, rcout, *oyx, cin, *HW)
      ret = (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW)).sum([-1-i for i in range(1+len(oyx))], keepdim=True, acc_dtype=acc_dtype).reshape(bs, cout, *oyx)  # noqa: E501
      return ret if bias is None else ret.add(bias.reshape(1, -1, *[1] * len(HW)))

    HWI, HWO = (6,) * len(HW), (4,) * len(HW)  # F(4x4,3x3) winograd tiles
    winograd_G = [[1/4, 0, 0], [-1/6, -1/6, -1/6], [-1/6, 1/6, -1/6], [1/24, 1/12, 1/6], [1/24, -1/12, 1/6], [0, 0, 1]]
    winograd_Bt = [[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0], [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]]
    winograd_At = [[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]] # applying At in pre-order doubles compile time

    # todo: stride == dilation
    # use padding to round up to 4x4 output tiles
    # (bs, cin_, tyx, HWI)
    d = self.pad(sum([[padding_[i*2], padding_[i*2+1] + (-(dim + sum(padding_[i * 2:(i + 1) * 2]) - 2) % 4)] for i, dim in enumerate(self.shape[-len(HW):])], []))._pool(HWI, HWO)  # noqa: E501
    # move HW to the front: # (HWI, bs, cin_, tyx)
    d = d.permute(*range(len(d.shape)-len(HW),len(d.shape)), *range(len(d.shape)-len(HW)))
    tyx = d.shape[-len(HWI):]  # dim of tiling

    g = weight.permute(*range(len(weight.shape)-len(HW),len(weight.shape)), *range(len(weight.shape)-len(HW)))  # move HW to the front

    # compute 6x6 winograd tiles: GgGt, BtdB
    # (HWI, groups * rcout, cin) -> (HWI, bs=1, groups, rcout, cin, tyx=(1,1))
    gfactors = _apply_winograd_matrix(winograd_G, g, len(HW)).reshape(*HWI, 1, groups, rcout, cin, *([1]*len(tyx)))
    # (HWI, bs, cin_, tyx) -> (HWI, bs, groups, 1 ,cin, *tyx)
    dfactors = _apply_winograd_matrix(winograd_Bt, d, len(HW)).reshape(*HWI, bs, groups, 1, cin, *tyx)

    # matmul; sum across cin: (HWI, bs, groups, rcout, *tyx); then HWI -> HWO: (HWO, bs, groups, rcout, *tyx)
    ret = _apply_winograd_matrix(winograd_At, (gfactors * dfactors).sum(axis=-1-len(HW), acc_dtype=acc_dtype), len(HW))

    # interleave tyx and HWO: (bs, groups, rcout, oy, HO, ox, WO)
    ret = ret.permute([*range(len(HW), len(ret.shape)-len(HW)), *[i+o for i in range(len(HW)) for o in [len(ret.shape)-len(HW),0]]])
    # merge groups and rcout, tyx and HWO: (bs, groups, cout, *yx), shrink to final
    ret = ret.reshape(bs, cout, *[c * HWO[i] for i, c in enumerate(tyx)]).shrink(tuple((0, s) for s in [bs, cout, *oyx]))

    return (ret if bias is None else ret.add(bias.reshape(1, -1, *[1 for _ in range(len(HW))]))).contiguous().contiguous_backward()

  def conv_transpose2d(self, weight:Tensor, bias:Optional[Tensor]=None, groups=1, stride=1, dilation=1, padding=0, output_padding=0) -> Tensor:
    """
    Applies a transposed convolution over a tensor with a given `weight` and optional `bias`.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d transposed convolutions and instead works for any number of dimensions.

    See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(1, 1, 3, 3)
    w = Tensor.ones(1, 1, 2, 2)
    print(t.conv_transpose2d(w).numpy())
    ```
    """
    x, w = self, weight.unflatten(0, (groups, -1)).transpose(1, 2).flip(*range(3, len(weight.shape)+1))
    HW = weight.shape[2:]
    stride, dilation, padding, output_padding = [make_tuple(x, len(HW)) for x in (stride, dilation, padding, output_padding)]
    if any(s>1 for s in stride):
      # handle strides: (k) -> reshape -> (k,1) -> pad -> (k,s) -> reshape -> (k*s) -> shrink (k-(s-1))
      x = x.reshape(None, None, *flatten((k,1) for k in x.shape[2:]))
      x = x.pad((None, None, *flatten((None,(0,s-1)) for s in stride)))
      x = x.reshape(None, None, *[k*s for k,s in zip(x.shape[2::2], stride)])
      x = x.shrink((None, None, *[(0,k-(s-1)) for k,s in zip(x.shape[2:], stride)]))
    padding = flatten((((k-1)*d-p,(k-1)*d-p+op) for k,d,p,op in reversed(list(zip(HW, dilation, padding, output_padding)))))
    return x.conv2d(w.flatten(end_dim=1), groups=groups, bias=bias, dilation=dilation, padding=padding)

  def dot(self, w:Tensor, acc_dtype:Optional[DTypeLike]=None) -> Tensor:

    """
    Performs dot product between two tensors.
    If `w` is 1-D, it's a sum product over the last axis of `self` and `w`.
    If `w` is N-D with N>=2, it's a sum product over the last axis of `self` and the second-to-last axis of `w`.

    You can pass in the optional `acc_dtype` keyword argument to control the data type of the accumulation.

    ```python exec="true" source="above" session="tensor" result="python"
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 0])
    print(a.dot(b).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    print(a.dot(b).numpy())
    ```
    """
    if IMAGE: return self.image_dot(w, acc_dtype)
    x, dx, dw = self, self.ndim, w.ndim
    if not (dx > 0 and dw > 0): raise RuntimeError(f"both tensors need to be at least 1D, got {dx}D and {dw}D")
    if x.shape[-1] != w.shape[axis_w:=-min(w.ndim,2)]: raise RuntimeError(f"cannot dot {x.shape} and {w.shape}")
    x = x.reshape(*x.shape[0:-1], *[1]*min(dx-1, dw-1, 1), x.shape[-1])
    w = w.reshape(*w.shape[0:-2], *[1]*min(dx-1, dw-1, 1), *w.shape[axis_w:]).transpose(-1, axis_w)
    return (x*w).sum(-1, acc_dtype=acc_dtype).cast(least_upper_dtype(x.dtype, w.dtype) if acc_dtype is None else acc_dtype)

  def matmul(self, x:Tensor, reverse=False, acc_dtype:Optional[DTypeLike]=None) -> Tensor:
    """
    Performs matrix multiplication between two tensors.

    You can pass in the `reverse` keyword argument to control the order of the matrix multiplication.
    You can pass in the optional `acc_dtype` keyword argument to control the data type of the accumulation.

    ```python exec="true" source="above" session="tensor" result="python"
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    print(a.matmul(b).numpy())
    ```
    """
    return x.dot(self, acc_dtype=acc_dtype) if reverse else self.dot(x, acc_dtype=acc_dtype)

  def _cumalu(self, axis:int, op:Ops, _include_initial=False) -> Tensor:
    assert self.shape[axis] != 0 and op in (Ops.ADD, Ops.MAX)
    pl_sz = self.shape[axis] - int(not _include_initial)
    pooled = self.transpose(axis,-1).pad((pl_sz, -int(_include_initial)), value=identity_element(op, self.dtype))._pool((self.shape[axis],))
    return (pooled.sum(-1) if op is Ops.ADD else pooled.max(-1)).transpose(axis,-1)

  def _split_cumalu(self, axis:int, op:Ops) -> Tensor:
    axis = self._resolve_dim(axis)
    if self.ndim == 0 or 0 in self.shape: return self
    # TODO: someday the optimizer will find this on it's own
    # for now this is a two stage cumsum
    SPLIT = 256
    if not isinstance(s:=self.shape[axis], int) or s <= SPLIT*2: return self._cumalu(axis, op)
    ret = self.transpose(axis,-1).pad((round_up(s, SPLIT)-s, 0), value=identity_element(op, self.dtype)).unflatten(-1, (-1, SPLIT))._cumalu(-1, op)
    base = ret[..., -1]._cumalu(-1, op, _include_initial=True)
    base = base.unsqueeze(-1).expand(*base.shape, ret.shape[-1])
    def fix(x:Tensor): return x.flatten(start_dim=-2)[..., -s:].transpose(axis,-1)
    return fix(ret) + fix(base) if op is Ops.ADD else fix(ret).maximum(fix(base))

  def cumsum(self, axis:int=0) -> Tensor:
    """
    Computes the cumulative sum of the tensor along the specified `axis`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.cumsum(1).numpy())
    ```
    """
    return self._split_cumalu(axis, Ops.ADD)

  def cummax(self, axis:int=0) -> Tensor:
    """
    Computes the cumulative max of the tensor along the specified `axis`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([0, 1, -1, 2, -2, 3, -3])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.cummax(0).numpy())
    ```
    """
    return self._split_cumalu(axis, Ops.MAX)

  @staticmethod
  def _tri(r:sint, c:sint, diagonal:int=0, **kwargs) -> Tensor:
    assert isinstance(r, int) and isinstance(c, int), f"does not support symbolic, getting {r=}, {c=}"
    if r == 0 or c == 0 or diagonal >= c: return Tensor.zeros(r,c,**kwargs)
    if r+diagonal <= 0: return Tensor.ones(r,c,**kwargs)
    s = r+c-1
    # build a (s, s) upper triangle
    t = Tensor.ones(s,s,**kwargs).pad((None,(0,s))).flatten().shrink(((0,s*(2*s-1)),)).reshape(s,-1).shrink((None,(0,s)))
    return t[:r,-diagonal:c-diagonal] if diagonal <= 0 else t[diagonal:r+diagonal,:c]

  def triu(self, diagonal:int=0) -> Tensor:
    """
    Returns the upper triangular part of the tensor, the other elements are set to 0.

    The argument `diagonal` determines which diagonal is on the boundary. `diagonal = 0` means the main diagonal.
    Positive `diagonal` means above the main diagonal, and negative `diagonal` means below the main diagonal.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.triu(diagonal=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.triu(diagonal=1).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.triu(diagonal=-1).numpy())
    ```
    """
    return Tensor._tri(self.shape[-2], self.shape[-1], diagonal=diagonal, device=self.device, dtype=dtypes.bool).where(self, 0).cast(self.dtype)

  def tril(self, diagonal:int=0) -> Tensor:
    """
    Returns the lower triangular part of the tensor, the other elements are set to 0.

    The argument `diagonal` determines which diagonal is on the boundary. `diagonal = 0` means the main diagonal.
    Positive `diagonal` means above the main diagonal, and negative `diagonal` means below the main diagonal.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.tril(diagonal=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.tril(diagonal=1).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.tril(diagonal=-1).numpy())
    ```
    """
    return Tensor._tri(self.shape[-2], self.shape[-1], diagonal=diagonal+1, device=self.device, dtype=dtypes.bool).where(0, self).cast(self.dtype)

  def interpolate(self, size:Tuple[int, ...], mode:str="linear", align_corners:bool=False) -> Tensor:
    """
    Downsamples or Upsamples to the input `size`, accepts 0 to N batch dimensions.

    The interpolation algorithm is selected with `mode` which currently only supports `linear`, `nearest` and `nearest-exact`.
    To run `bilinear` or `trilinear`, pass in a 2D or 3D size.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2, 3, 4], [21, 22, 23, 24], [41, 42, 43, 44]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.interpolate(size=(2,3), mode="linear").numpy())
    ```
    """
    assert isinstance(size, (tuple,list)) and all_int(size) and 0 < len(size) <= self.ndim, f"invalid {size=}"
    assert mode in ("linear", "nearest", "nearest-exact"), "only supports linear, nearest or nearest-exact interpolate"
    assert not (align_corners and mode != "linear"), "align_corners option can only be set with the interpolating mode linear"
    x, expand = self, list(self.shape)
    for i in range(-1,-len(size)-1,-1):
      scale = (self.shape[i] - int(align_corners)) / (size[i] - int(align_corners))
      arr, reshape = Tensor.arange(size[i], dtype=dtypes.float32, device=self.device), [1] * self.ndim
      reshape[i] = expand[i] = size[i]
      if mode == "linear":
        index = (scale*arr if align_corners else (scale*(arr+0.5))-0.5).clip(0, self.shape[i]-1)
        low, high, perc = [y.reshape(reshape).expand(expand) for y in (index.floor(), index.ceil(), index - index.floor())]
        x = x.gather(i, low).lerp(x.gather(i, high), perc)
      else:
        index = (scale*(arr+0.5) if mode=="nearest-exact" else scale*arr).cast(dtypes.int32).reshape(reshape).expand(expand)
        x = x.gather(i, index)
    return x.cast(self.dtype)

  def scatter(self, dim:int, index:Tensor, src:Union[Tensor, ConstType], reduce:Union[None, Literal['multiply'], Literal['add']]=None) -> Tensor:
    """
    Scatters `src` values along an axis specified by `dim`.
    Apply `add` or `multiply` reduction operation with `reduce`.

    ```python exec="true" source="above" session="tensor" result="python"
    src = Tensor.arange(1, 11).reshape(2, 5)
    print(src.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    index = Tensor([[0, 1, 2, 0]])
    print(Tensor.zeros(3, 5, dtype=src.dtype).scatter(0, index, src).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    index = Tensor([[0, 1, 2], [0, 1, 4]])
    print(Tensor.zeros(3, 5, dtype=src.dtype).scatter(1, index, src).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 4), 2.0).scatter(1, Tensor([[2], [3]]), 1.23, reduce='multiply').numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 4), 2.0).scatter(1, Tensor([[2], [3]]), 1.23, reduce='add').numpy())
    ```
    """
    index, dim = index.to(self.device), self._resolve_dim(dim)
    src = src.cast(self.dtype) if isinstance(src, Tensor) else Tensor(src, device=self.device, dtype=self.dtype)._broadcast_to(index.shape)
    assert index.ndim == self.ndim == src.ndim, f"self.ndim, index.ndim and src.dim must all equal, {self.ndim=} {index.ndim=} {src.ndim=}"
    assert all((d == dim or self_ >= index_) and src_ >= index_ for d,(self_,index_,src_) in enumerate(zip(self.shape, index.shape, src.shape))), \
      f"All dimensions of {index.shape=} should be <= to all dimensions of {src.shape=} and all dimensions except dimension {dim} of {self.shape=}"
    # shrink src to index shape to shrink away the unused values
    src = src.shrink(tuple((0,s) for s in index.shape))
    # prepare src and mask for reduce with respect to dim
    src = src.unsqueeze(-1).expand(*src.shape, self.shape[dim]).transpose(-1, dim)
    mask = index.unsqueeze(-1)._one_hot_along_dim(self.shape[dim]).transpose(-1, dim)
    # pad src and mask to self.shape so that reduce can be done with padded values as no-ops
    src, mask = (x.pad(tuple((0, self.shape[i] - x.shape[i]) if i != dim else None for i in range(self.ndim)) + (None,)) for x in (src, mask))
    if reduce == "add": return mask.where(src, 0).sum(-1, acc_dtype=self.dtype) + self
    if reduce == "multiply": return mask.where(src, 1).prod(-1, acc_dtype=self.dtype) * self
    return _masked_setitem(self, src, mask, (-1,))

  # ***** unary ops *****

  def logical_not(self):
    """
    Computes the logical NOT of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([False, True]).logical_not().numpy())
    ```
    """
    return F.Neq.apply(*self.cast(dtypes.bool)._broadcasted(True))
  def neg(self):
    """
    Negates the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).neg().numpy())
    ```
    """
    return self*-1 if self.dtype != dtypes.bool else self.logical_not()
  def contiguous(self):
    """
    Returns a contiguous tensor.
    """
    return F.Contiguous.apply(self)
  def contiguous_backward(self):
    """
    Inserts a contiguous operation in the backward pass.
    """
    return F.ContiguousBackward.apply(self)
  def log(self):
    """
    Computes the natural logarithm element-wise.

    See: https://en.wikipedia.org/wiki/Logarithm

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 4., 8.]).log().numpy())
    ```
    """
    return F.Log.apply(self.cast(least_upper_float(self.dtype)))
  def log2(self):
    """
    Computes the base-2 logarithm element-wise.

    See: https://en.wikipedia.org/wiki/Logarithm

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 4., 8.]).log2().numpy())
    ```
    """
    return self.log()/math.log(2)
  def exp(self):
    """
    Computes the exponential function element-wise.

    See: https://en.wikipedia.org/wiki/Exponential_function

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., 1., 2., 3.]).exp().numpy())
    ```
    """
    return F.Exp.apply(self.cast(least_upper_float(self.dtype)))
  def exp2(self):
    """
    Computes the base-2 exponential function element-wise.

    See: https://en.wikipedia.org/wiki/Exponential_function

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., 1., 2., 3.]).exp2().numpy())
    ```
    """
    return F.Exp.apply(self*math.log(2))
  def relu(self):
    """
    Applies the Rectified Linear Unit (ReLU) function element-wise.

    - Described: https://paperswithcode.com/method/relu

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).relu().numpy())
    ```
    """
    return F.Relu.apply(self)
  def sigmoid(self):
    """
    Applies the Sigmoid function element-wise.

    - Described: https://en.wikipedia.org/wiki/Sigmoid_function

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sigmoid().numpy())
    ```
    """
    return F.Sigmoid.apply(self.cast(least_upper_float(self.dtype)))
  def hardsigmoid(self, alpha:float=1/6, beta:float=0.5):
    """
    Applies the Hardsigmoid function element-wise.
    NOTE: default `alpha` and `beta` values is taken from torch

    - Described: https://paperswithcode.com/method/hard-sigmoid
    - See: https://pytorch.org/docs/stable/generated/torch.nn.functional.hardsigmoid.html

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).hardsigmoid().numpy())
    ```
    """
    return (alpha * self + beta).relu() - (alpha * self + beta - 1).relu()

  def sqrt(self):
    """
    Computes the square root of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 3., 4.]).sqrt().numpy())
    ```
    """
    return F.Sqrt.apply(self.cast(least_upper_float(self.dtype)))
  def rsqrt(self):
    """
    Computes the reciprocal of the square root of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 3., 4.]).rsqrt().numpy())
    ```
    """
    return self.reciprocal().sqrt()
  def sin(self):
    """
    Computes the sine of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).sin().numpy())
    ```
    """
    return F.Sin.apply(self.cast(least_upper_float(self.dtype)))
  def cos(self):
    """
    Computes the cosine of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).cos().numpy())
    ```
    """
    return ((math.pi/2)-self).sin()
  def tan(self):
    """
    Computes the tangent of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., math.pi/4, math.pi/2, 3*math.pi/4, math.pi]).tan().numpy())
    ```
    """
    return self.sin() / self.cos()

  def asin(self):
    """
    Computes the inverse sine (arcsine) of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).asin().numpy())
    ```
    """
    # https://personal.math.ubc.ca/~cbm/aands/page_81.htm 4.4.46
    coefficients = [-0.0012624911, 0.0066700901, -0.0170881256, 0.0308918810, -0.0501743046, 0.0889789874, -0.2145988016, 1.5707963050]
    x = math.pi / 2 - (1.0 - self.abs()).sqrt() * polyN(self.abs(), coefficients)
    return self.sign() * x

  def acos(self):
    """
    Computes the inverse cosine (arccosine) of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).acos().numpy())
    ```
    """
    return math.pi / 2 - self.asin()

  def atan(self):
    """
    Computes the inverse tangent (arctan) of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).atan().numpy())
    ```
    """
    return (self / (1 + self * self).sqrt()).asin()

  # ***** math functions *****

  def trunc(self: Tensor) -> Tensor:
    """
    Truncates the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).trunc().numpy())
    ```
    """
    return self.cast(dtypes.int32).cast(self.dtype)
  def ceil(self: Tensor) -> Tensor:
    """
    Rounds the tensor element-wise towards positive infinity.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).ceil().numpy())
    ```
    """
    return (self > (b := self.trunc())).where(b+1, b)
  def floor(self: Tensor) -> Tensor:
    """
    Rounds the tensor element-wise towards negative infinity.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).floor().numpy())
    ```
    """
    return (self < (b := self.trunc())).where(b-1, b)
  def round(self: Tensor) -> Tensor:
    """
    Rounds the tensor element-wise with rounding half to even.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).round().numpy())
    ```
    """
    return ((self > 0) == ((b := self.cast(dtypes.int32) / 2.0).cast(dtypes.int32) == b)).where((self - 0.5).ceil(), (self + 0.5).floor())

  def isinf(self:Tensor, detect_positive:bool=True, detect_negative:bool=True):
    """
    Checks the tensor element-wise to return True where the element is infinity, otherwise returns False

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isinf().numpy())
    ```
    """
    return (self == float("inf")) * detect_positive + (self == float("-inf")) * detect_negative
  def isnan(self:Tensor):
    """
    Checks the tensor element-wise to return True where the element is NaN, otherwise returns False

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isnan().numpy())
    ```
    """
    return self != self

  def lerp(self, end: Tensor, weight: Union[Tensor, float]) -> Tensor:
    """
    Linearly interpolates between `self` and `end` by `weight`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 3.]).lerp(Tensor([4., 5., 6.]), 0.5).numpy())
    ```
    """
    if self.dtype == dtypes.uint8 and isinstance(weight, Tensor):
      w_i = (weight * (1<<(W_PREC:=7)) + 0.5).cast(dtypes.int16)
      return (self+(((end - self).cast(dtypes.int8) * w_i + (1<<W_PREC-1)).cast(dtypes.uint16) >> W_PREC)).cast(dtypes.uint8)
    return self + (end - self) * weight

  def square(self):
    """
    Squares the tensor element-wise.
    Equivalent to `self*self`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).square().numpy())
    ```
    """
    return self*self
  def clamp(self, min_=None, max_=None):
    """
    Clips (clamps) the values in the tensor between `min_` and `max_` element-wise.
    If `min_` is `None`, there is no lower bound. If `max_` is None, there is no upper bound.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).clip(-1, 1).numpy())
    ```
    """
    if min_ is None and max_ is None: raise RuntimeError("at least one of 'min_' or 'max_' must not be None")
    ret = self.maximum(min_) if min_ is not None else self
    return ret.minimum(max_) if max_ is not None else ret
  def clip(self, min_=None, max_=None):
    """
    Alias for `Tensor.clamp`.
    """
    return self.clamp(min_, max_)
  def sign(self):
    """
    Returns the sign of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sign().numpy())
    ```
    """
    return F.Sign.apply(self)
  def abs(self):
    """
    Computes the absolute value of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).abs().numpy())
    ```
    """
    return self * self.sign()
  def reciprocal(self):
    """
    Compute `1/x` element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 3., 4.]).reciprocal().numpy())
    ```
    """
    return F.Reciprocal.apply(self.cast(least_upper_float(self.dtype)))

  # ***** activation functions *****

  def elu(self, alpha=1.0):
    """
    Applies the Exponential Linear Unit (ELU) function element-wise.

    - Described: https://paperswithcode.com/method/elu
    - Paper: https://arxiv.org/abs/1511.07289v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).elu().numpy())
    ```
    """
    return self.relu() - alpha*(1-self.exp()).relu()

  def celu(self, alpha=1.0):
    """
    Applies the Continuously differentiable Exponential Linear Unit (CELU) function element-wise.

    - Described: https://paperswithcode.com/method/celu
    - Paper: https://arxiv.org/abs/1704.07483

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).celu().numpy())
    ```
    """
    return self.maximum(0) + (alpha * ((self / alpha).exp() - 1)).minimum(0)

  def selu(self, alpha=1.67326, gamma=1.0507):
    """
    Applies the Scaled Exponential Linear Unit (SELU) function element-wise.

    - Described: https://paperswithcode.com/method/selu
    - Paper: https://arxiv.org/abs/1706.02515v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).selu().numpy())
    ```
    """
    return gamma * (self >= 0).detach().where(self, alpha * (self.exp() - 1))

  def swish(self):
    """
    See `.silu()`

    - Paper: https://arxiv.org/abs/1710.05941v1

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).swish().numpy())
    ```
    """
    return self * self.sigmoid()

  def silu(self):
    """
    Applies the Sigmoid Linear Unit (SiLU) function element-wise.

    - Described: https://paperswithcode.com/method/silu
    - Paper: https://arxiv.org/abs/1606.08415

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).silu().numpy())
    ```
    """
    return self.swish()   # The SiLU function is also known as the swish function.

  def relu6(self):
    """
    Applies the ReLU6 function element-wise.

    - Described: https://paperswithcode.com/method/relu6
    - Paper: https://arxiv.org/abs/1704.04861v1

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-9., -6., -3., 0., 3., 6., 9.]).relu6().numpy())
    ```
    """
    return self.relu() - (self-6).relu()

  def hardswish(self):
    """
    Applies the Hardswish function element-wise.

    - Described: https://paperswithcode.com/method/hard-swish
    - Paper: https://arxiv.org/abs/1905.02244v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).hardswish().numpy())
    ```
    """
    return self * (self+3).relu6() * (1/6)

  def tanh(self):
    """
    Applies the Hyperbolic Tangent (tanh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Tanh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).tanh().numpy())
    ```
    """
    return 2.0 * ((2.0 * self).sigmoid()) - 1.0

  def sinh(self):
    """
    Applies the Hyperbolic Sine (sinh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Sinh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sinh().numpy())
    ```
    """
    return (self.exp() - self.neg().exp()) / 2

  def cosh(self):
    """
    Applies the Hyperbolic Cosine (cosh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Cosh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).cosh().numpy())
    ```
    """
    return (self.exp() + self.neg().exp()) / 2

  def atanh(self):
    """
    Applies the Inverse Hyperbolic Tangent (atanh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#atanh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).atanh().numpy())
    ```
    """
    return ((1 + self)/(1 - self)).log() / 2

  def asinh(self):
    """
    Applies the Inverse Hyperbolic Sine (asinh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#asinh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).asinh().numpy())
    ```
    """
    return (self + (self.square() + 1).sqrt()).log()

  def acosh(self):
    """
    Applies the Inverse Hyperbolic Cosine (acosh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#acosh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).acosh().numpy())
    ```
    """
    return (self + (self.square() - 1).sqrt()).log()

  def hardtanh(self, min_val=-1, max_val=1):
    """
    Applies the Hardtanh function element-wise.

    - Described: https://paperswithcode.com/method/hardtanh-activation

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5]).hardtanh().numpy())
    ```
    """
    return self.clip(min_val, max_val)

  def erf(self):
    """
    Applies error function element-wise.

    - Described: https://en.wikipedia.org/wiki/Error_function

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5]).erf().numpy())
    ```
    """
    # https://personal.math.ubc.ca/~cbm/aands/page_299.htm 7.1.26
    t = 1.0 / (1.0 + 0.3275911 * self.abs())
    return self.sign() * (1.0 - t * polyN(t, [1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592]) * (-self.square()).exp())

  def gelu(self):
    """
    Applies the Gaussian Error Linear Unit (GELU) function element-wise.

    - Described: https://paperswithcode.com/method/gelu
    - Paper: https://arxiv.org/abs/1606.08415v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).gelu().numpy())
    ```
    """
    return 0.5 * self * (1 + (math.sqrt(2 / math.pi) * (self + 0.044715 * self ** 3)).tanh())

  def quick_gelu(self):
    """
    Applies the Sigmoid GELU approximation element-wise.

    - Described: https://paperswithcode.com/method/gelu

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).quick_gelu().numpy())
    ```
    """
    return self * (self * 1.702).sigmoid()

  def leakyrelu(self, neg_slope=0.01):
    """
    Applies the Leaky ReLU function element-wise.

    - Described: https://paperswithcode.com/method/leaky-relu

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).leakyrelu().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).leakyrelu(neg_slope=0.42).numpy())
    ```
    """
    return self.relu() - (-neg_slope*self).relu()

  def mish(self):
    """
    Applies the Mish function element-wise.

    - Described: https://paperswithcode.com/method/mish
    - Paper: https://arxiv.org/abs/1908.08681v3

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).mish().numpy())
    ```
    """
    return self * self.softplus().tanh()

  def softplus(self, beta=1):
    """
    Applies the Softplus function element-wise.

    - Described: https://paperswithcode.com/method/softplus

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).softplus().numpy())
    ```
    """
    return (1/beta) * (1 + (self*beta).exp()).log()

  def softsign(self):
    """
    Applies the Softsign function element-wise.

    - Described: https://paperswithcode.com/method/softsign

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).softsign().numpy())
    ```
    """
    return self / (1 + self.abs())

  # ***** broadcasted elementwise ops *****
  def _broadcast_to(self, new_shape:Tuple[sint, ...]) -> Tensor:
    if self.shape == new_shape: return self
    if self.ndim > len(new_shape): raise ValueError(f"cannot broadcast tensor to fewer dimensions. shape={self.shape} to {new_shape=}")
    # first unsqueeze left with 1s https://data-apis.org/array-api/latest/API_specification/broadcasting.html
    shape, _ = _align_left(self.shape, new_shape)
    # for each dimension, check either dim is 1, or it does not change
    if not all(resolve(s == ns) or resolve(s == 1) for s,ns in zip(shape, new_shape)):
      raise ValueError(f"cannot broadcast {self.shape} to {new_shape=}")
    return F.Expand.apply(self.reshape(shape), shape=new_shape)

  def _broadcasted(self, y:Union[Tensor, UOp, ConstType], reverse:bool=False, match_dtype:bool=True) -> Tuple[Tensor, Tensor]:
    x: Tensor = self
    if not isinstance(y, Tensor):
      # make y a Tensor
      assert isinstance(y, (*get_args(ConstType), UOp)), f"{type(y)=}, {y=}"
      if isinstance(x.dtype, ImageDType) or dtypes.is_float(x.dtype) or (dtypes.is_int(x.dtype) and isinstance(y, int)): y_dtype = x.dtype
      elif not isinstance(y, UOp): y_dtype = dtypes.from_py(y)
      if isinstance(y, UOp): y = Tensor.from_uop(y, device=x.device)
      else: y = Tensor(dtypes.as_const(y, y_dtype), x.device, y_dtype, requires_grad=False)

    if match_dtype and x.dtype != y.dtype:
      output_dtype = least_upper_dtype(x.dtype, y.dtype)
      x, y = x.cast(output_dtype), y.cast(output_dtype)

    if reverse: x, y = y, x

    # broadcast
    return x._broadcast_to(out_shape:=_broadcast_shape(x.shape, y.shape)), y._broadcast_to(out_shape)

  def _to_const_val(self, x:Union[Tensor, ConstType]) -> Union[Tensor, ConstType]:
    return x.lazydata.base.arg if isinstance(x, Tensor) and isinstance(x.lazydata, LazyBuffer) and x.lazydata.is_unrealized_unmasked_const() \
      and not x.requires_grad and self._broadcasted(x)[0].shape == self.shape else x

  def add(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Adds `self` and `x`.
    Equivalent to `self + x`.
    Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(4)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.add(20).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.add(Tensor([[2.0], [3.5]])).numpy())
    ```
    """
    return F.Add.apply(*self._broadcasted(x, reverse))

  def sub(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Subtracts `x` from `self`.
    Equivalent to `self - x`.
    Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(4)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.sub(20).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.sub(Tensor([[2.0], [3.5]])).numpy())
    ```
    """
    a, b = self._broadcasted(x, reverse)
    return a + (-b)

  def mul(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Multiplies `self` and `x`.
    Equivalent to `self * x`.
    Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(4)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mul(3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mul(Tensor([[-1.0], [2.0]])).numpy())
    ```
    """
    return F.Mul.apply(*self._broadcasted(x, reverse))

  def idiv(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Divides `self` by `x`.
    Equivalent to `self // x`.
    Supports broadcasting to a common shape, type promotion, and integer inputs.
    `idiv` performs integer division.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, 4, 10]).idiv(Tensor([2, 3, 4])).numpy())
    ```
    """
    return F.IDiv.apply(*self._broadcasted(x, reverse))

  def div(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Divides `self` by `x`.
    Equivalent to `self / x`.
    Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.
    `div` performs true division.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(4)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.div(3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, 4, 10]).div(Tensor([2, 3, 4])).numpy())
    ```
    """
    numerator, denominator = self._broadcasted(x, reverse)
    return numerator.cast(least_upper_float(numerator.dtype)) * denominator.cast(least_upper_float(denominator.dtype)).reciprocal()

  def xor(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Computes bitwise xor of `self` and `x`.
    Equivalent to `self ^ x`.
    Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, -2, 3]).xor(Tensor([1, 0, 3])).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([True, True, False, False]).xor(Tensor([True, False, True, False])).numpy())
    ```
    """
    if self.dtype != dtypes.bool and not dtypes.is_int(self.dtype): raise RuntimeError(f"{self.dtype} is not supported")
    return F.Xor.apply(*self._broadcasted(x, reverse))

  def bitwise_and(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Compute the bit-wise AND of `self` and `x`.
    Equivalent to `self & x`.
    Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([2, 5, 255]).bitwise_and(Tensor([3, 14, 16])).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([True, True, False, False]).bitwise_and(Tensor([True, False, True, False])).numpy())
    ```
    """
    if self.dtype != dtypes.bool and not dtypes.is_int(self.dtype): raise RuntimeError(f"{self.dtype} is not supported")
    return F.BitwiseAnd.apply(*self._broadcasted(x, reverse))

  def bitwise_or(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Compute the bit-wise OR of `self` and `x`.
    Equivalent to `self | x`.
    Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([2, 5, 255]).bitwise_or(Tensor([4, 4, 4])).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([True, True, False, False]).bitwise_or(Tensor([True, False, True, False])).numpy())
    ```
    """
    if self.dtype != dtypes.bool and not dtypes.is_int(self.dtype): raise RuntimeError(f"{self.dtype} is not supported")
    return F.BitwiseOr.apply(*self._broadcasted(x, reverse))

  def bitwise_not(self) -> Tensor:
    """
    Compute the bit-wise NOT of `self`.
    Equivalent to `~self`.
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0, 2, 5, 255], dtype="int8").bitwise_not().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([True, False]).bitwise_not().numpy())
    ```
    """
    if self.dtype != dtypes.bool and not dtypes.is_int(self.dtype): raise RuntimeError(f"{self.dtype} is not supported")
    return self.logical_not() if self.dtype == dtypes.bool else self ^ ((1<<8*self.dtype.itemsize)-1)

  def lshift(self, x:int):
    """
    Computes left arithmetic shift of `self` by `x` bits. `self` must have unsigned dtype.
    Equivalent to `self << x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, 3, 31], dtype=dtypes.uint8).lshift(2).numpy())
    ```
    """
    assert dtypes.is_unsigned(self.dtype) and isinstance(x, int) and x >= 0, f"not supported {self.dtype=} {x=}"
    return self.mul(2 ** x)

  def rshift(self, x:int):
    """
    Computes right arithmetic shift of `self` by `x` bits. `self` must have unsigned dtype.
    Equivalent to `self >> x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([4, 13, 125], dtype=dtypes.uint8).rshift(2).numpy())
    ```
    """
    assert dtypes.is_unsigned(self.dtype) and isinstance(x, int) and x >= 0, f"not supported {self.dtype=} {x=}"
    return self.idiv(2 ** x)

  def pow(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Computes power of `self` with `x`.
    Equivalent to `self ** x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).pow(2).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).pow(Tensor([-1.5, 0.5, 1.5])).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print((2 ** Tensor([-1, 2, 3])).numpy())
    ```
    """
    x = self._to_const_val(x)
    if not isinstance(x, Tensor) and not reverse:
      # simple pow identities
      if x < 0: return self.reciprocal().pow(-x)
      if x == 0: return 1 + self * 0
      if int(x - 0.5) + 0.5 == x: return self.pow(int(x - 0.5)) * self.sqrt()
      if int(x) == x: return self.pow(x // 2).square() * (1 if x % 2 == 0 else self)

    # positive const ** self
    if not isinstance(x, Tensor) and reverse and x > 0: return self.mul(math.log(x)).exp()

    base, exponent = self._broadcasted(x, reverse=reverse)
    # start with b ** e = exp(e * log(b))
    ret = base.abs().log().mul(exponent).exp()
    # correct sign of negative base with odd exponent (cos has a period of 2pi so we use it here to get the oddness of the exponent)
    negative_base = (base < 0).detach().where(1, 0)
    # 1 for non-negative base or negative even exponent, -1 for negative odd exponent, don't care about non-integer exponent
    correct_sign = 1 + negative_base * ((exponent * math.pi).cos() - 1)
    # inject nan for negative base and non-integer exponent
    inject_nan = (negative_base * (exponent != exponent.trunc())).detach().where(math.nan, 1)
    # apply correct_sign inject_nan, and fix 0 ** 0 = 1
    ret = ((base == 0) * (exponent == 0)).detach().where(1, ret * correct_sign * inject_nan)
    return ret.round().cast(self.dtype) if not dtypes.is_float(self.dtype) else ret

  def maximum(self, x:Union[Tensor, ConstType]) -> Tensor:
    """
    Computes element-wise maximum of `self` and `x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).maximum(1).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).maximum(Tensor([-4, -2, 9])).numpy())
    ```
    """
    return (self<x).detach().where(x, (self==x).detach().where(((self * 0.5 + x * 0.5).cast(self.dtype)), self))

  def minimum(self, x:Union[Tensor, ConstType]) -> Tensor:
    """
    Computes element-wise minimum of `self` and `x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).minimum(1).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).minimum(Tensor([-4, -2, 9])).numpy())
    ```
    """
    return -((-self).maximum(-x))

  def where(self:Tensor, x:Union[Tensor, ConstType, sint], y:Union[Tensor, ConstType, sint]):
    """
    Return a tensor of elements selected from either `x` or `y`, depending on `self`.
    `output_i = x_i if self_i else y_i`.

    ```python exec="true" source="above" session="tensor" result="python"
    cond = Tensor([[True, True, False], [True, False, False]])
    print(cond.where(1, 3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    cond = Tensor.randn(2, 3)
    print(cond.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print((cond > 0).where(cond, -float("inf")).numpy())
    ```
    """
    if isinstance(x, Tensor): x, y = x._broadcasted(y)
    elif isinstance(y, Tensor): y, x = y._broadcasted(x)
    cond, x = self._broadcasted(x, match_dtype=False)
    cond, y = cond._broadcasted(y, match_dtype=False)
    return F.Where.apply(cond.cast(dtypes.bool), *x._broadcasted(y))

  def masked_fill(self:Tensor, mask:Tensor, value:Union[Tensor, ConstType]): return mask.where(value, self)

  # ***** op wrappers *****

  def __invert__(self) -> Tensor: return self.bitwise_not()

  def __lshift__(self, x) -> Tensor: return self.lshift(x)
  def __rshift__(self, x) -> Tensor: return self.rshift(x)

  def __pow__(self, x) -> Tensor: return self.pow(x)
  def __matmul__(self, x) -> Tensor: return self.matmul(x)

  def __rpow__(self, x) -> Tensor: return self.pow(x, True)
  def __rmatmul__(self, x) -> Tensor: return self.matmul(x, True)

  def __iadd__(self, x) -> Tensor: return self.assign(self.add(x))
  def __isub__(self, x) -> Tensor: return self.assign(self.sub(x))
  def __imul__(self, x) -> Tensor: return self.assign(self.mul(x))
  def __ipow__(self, x) -> Tensor: return self.assign(self.pow(x))
  def __itruediv__(self, x) -> Tensor: return self.assign(self.div(x))
  def __ifloordiv__(self, x) -> Tensor: return self.assign(self.idiv(x))
  def __imatmul__(self, x) -> Tensor: return self.assign(self.matmul(x))
  def __iand__(self, x) -> Tensor: return self.assign(self.bitwise_and(x))
  def __ior__(self, x) -> Tensor: return self.assign(self.bitwise_or(x))
  def __ixor__(self, x) -> Tensor: return self.assign(self.xor(x))
  def __ilshift__(self, x) -> Tensor: return self.assign(self.lshift(x))
  def __irshift__(self, x) -> Tensor: return self.assign(self.rshift(x))

  def __lt__(self, x) -> Tensor: return F.Less.apply(*self._broadcasted(x, False))
  def __gt__(self, x) -> Tensor: return F.Less.apply(*self._broadcasted(x, True))
  def ne(self, x) -> Tensor: return F.Neq.apply(*self._broadcasted(x))

  def __eq__(self, x) -> Tensor: return self.eq(x)                      # type: ignore[override]

  # ***** functional nn ops *****

  def linear(self, weight:Tensor, bias:Optional[Tensor]=None):
    """
    Applies a linear transformation to `self` using `weight` and `bias`.

    See: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2], [3, 4]])
    weight = Tensor([[1, 2], [3, 4]])
    bias = Tensor([1, 2])
    print(t.linear(weight, bias).numpy())
    ```
    """
    x = self.mul(weight) if len(weight.shape) == 1 else self.dot(weight)
    return x.add(bias) if bias is not None else x

  def sequential(self, ll:List[Callable[[Tensor], Tensor]]):
    """
    Applies a sequence of functions to `self` chaining the output of each function to the input of the next.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t.sequential([lambda x: x * 2, lambda x: x + 1]).numpy())
    ```
    """
    return functools.reduce(lambda x,f: f(x), ll, self)

  def layernorm(self, axis:Union[int,Tuple[int,...]]=-1, eps:float=1e-5) -> Tensor:
    """
    Applies Layer Normalization over a mini-batch of inputs.

    - Described: https://paperswithcode.com/method/layer-normalization
    - Paper: https://arxiv.org/abs/1607.06450v1

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.randn(8, 10, 16) * 2 + 8
    print(t.mean().item(), t.std().item())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.layernorm()
    print(t.mean().item(), t.std().item())
    ```
    """
    y = (self - self.mean(axis, keepdim=True))
    return y.mul((y*y).mean(axis, keepdim=True).add(eps).rsqrt())

  def batchnorm(self, weight:Optional[Tensor], bias:Optional[Tensor], mean:Tensor, invstd:Tensor, axis:Union[int,Tuple[int,...]]=1) -> Tensor:
    """
    Applies Batch Normalization over a mini-batch of inputs.

    - Described: https://paperswithcode.com/method/batch-normalization
    - Paper: https://arxiv.org/abs/1502.03167

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.randn(8, 4, 16, 16) * 2 + 8
    print(t.mean().item(), t.std().item())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.batchnorm(None, None, t.mean(axis=(0,2,3)), t.var(axis=(0,2,3)).add(1e-5).rsqrt())
    print(t.mean().item(), t.std().item())
    ```
    """
    axis_ = argfix(axis)
    shape = tuple(s if ax in axis_ else 1 for ax, s in enumerate(self.shape))
    x = self - mean.reshape(shape)
    if weight is not None: x = x * weight.reshape(shape)
    ret = x.mul(invstd.reshape(shape) if len(invstd.shape) == len(axis_) else invstd)
    return (ret + bias.reshape(shape)) if bias is not None else ret

  def dropout(self, p=0.5) -> Tensor:
    """
    Applies dropout to `self`.

    NOTE: dropout is only applied when `Tensor.training` is `True`.

    - Described: https://paperswithcode.com/method/dropout
    - Paper: https://jmlr.org/papers/v15/srivastava14a.html

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 2)
    with Tensor.train():
      print(t.dropout().numpy())
    ```
    """
    if not Tensor.training or p == 0: return self
    return (Tensor.rand_like(self, requires_grad=False, dtype=dtypes.default_float, contiguous=False) >= p).contiguous().where(self, 0) / (1.0 - p)

  # helper function commonly used for indexing
  def _one_hot_along_dim(self:Tensor, num_classes:sint, dim:int=-1):
    offset = self.ndim - self._resolve_dim(dim) - 1
    return self == Tensor.arange(num_classes, device=self.device, requires_grad=False).reshape((num_classes,) + (1,) * offset)

  def one_hot(self, num_classes:int=-1) -> Tensor:
    """
    Converts `self` to a one-hot tensor.

    `num_classes` defaults to -1, which means num_classes will be inferred as max(self) + 1.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([0, 1, 3, 3, 4])
    print(t.one_hot(5).numpy())
    ```
    """
    if num_classes == -1: num_classes = (self.max()+1).item()
    return self[..., None]._one_hot_along_dim(num_classes).where(1, 0)

  def scaled_dot_product_attention(self, key:Tensor, value:Tensor, attn_mask:Optional[Tensor]=None,
                                   dropout_p:float=0.0, is_causal:bool=False) -> Tensor:
    """
    Computes scaled dot-product attention.
    `self` is the query tensor, `key` is the key tensor, and `value` is the value tensor.

    - Described: https://paperswithcode.com/method/scaled
    - Paper: https://arxiv.org/abs/1706.03762v7

    ```python exec="true" source="above" session="tensor" result="python"
    q = Tensor.randn(2, 4, 8)
    k = Tensor.randn(2, 4, 8)
    v = Tensor.randn(2, 4, 8)
    print(q.scaled_dot_product_attention(k, v).numpy())
    ```
    """
    # NOTE: it also works when `key` and `value` have symbolic shape.
    assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
    if is_causal: attn_mask = Tensor.ones(self.shape[-2], key.shape[-2], requires_grad=False, device=self.device).tril(0).cast(dtypes.bool)
    if attn_mask is not None and attn_mask.dtype == dtypes.bool: attn_mask = (attn_mask == 0).where(-float("inf"), 0)
    qk = self.matmul(key.transpose(-2,-1), acc_dtype=least_upper_dtype(self.dtype, key.dtype, dtypes.float32)) / math.sqrt(self.shape[-1])
    return ((qk+attn_mask) if attn_mask is not None else qk).softmax(-1).cast(self.dtype).dropout(dropout_p) @ value

  def _do_reduction(self, reduction:ReductionStr="mean") -> Tensor:
    if reduction not in get_args(ReductionStr): raise ValueError(f"{reduction=} must be one of {get_args(ReductionStr)}")
    reductions: Dict[str, Callable[[Tensor], Tensor]] = {"mean": Tensor.mean, "sum": Tensor.sum, "none": lambda x: x}
    return reductions[reduction](self)

  def binary_crossentropy(self, Y:Tensor, reduction:ReductionStr="mean") -> Tensor:
    """
    Computes the binary cross-entropy loss between `self` and `Y`.

    See: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([0.1, 0.9, 0.2])
    Y = Tensor([0, 1, 0])
    print(t.binary_crossentropy(Y).item())
    ```
    """
    return (-Y*self.log() - (1-Y)*(1-self).log())._do_reduction(reduction)

  def binary_crossentropy_logits(self, Y:Tensor, reduction:ReductionStr="mean") -> Tensor:
    """
    Computes the binary cross-entropy loss between `self` and `Y` where `self` is logits.

    See: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2, -3])
    Y = Tensor([0, 1, 0])
    print(t.binary_crossentropy_logits(Y).item())
    ```
    """
    return (self.maximum(0) - Y * self + (1 + self.abs().neg().exp()).log())._do_reduction(reduction)

  def sparse_categorical_crossentropy(self, Y:Tensor, ignore_index:int=-1, label_smoothing=0.0, reduction:ReductionStr="mean") -> Tensor:
    """
    Computes the sparse categorical cross-entropy loss between `self` and `Y`.

    NOTE: `self` is logits and `Y` is the target labels.
    NOTE: unlike PyTorch, this function expects the class axis to be -1

    See: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[-1, 2, -3], [1, -2, 3]])
    Y = Tensor([1, 2])
    print(t.sparse_categorical_crossentropy(Y).item())
    ```
    """
    assert 0.0 <= label_smoothing <= 1.0, "label_smoothing must be in [0.0, 1.0]"
    assert reduction in ("mean", "sum", "none"), "reduction must be one of ['mean', 'sum', 'none']"
    log_probs, loss_mask = self.log_softmax(), (Y != ignore_index) if ignore_index != -1 else Y.ones_like(dtype=dtypes.bool)
    y_counted = Y.to(self.device).flatten().reshape(-1, 1)._one_hot_along_dim(self.shape[-1])
    y = (y_counted * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    smoothing = label_smoothing * (log_probs.mean(-1) * loss_mask)
    unreduced = ((1 - label_smoothing) * (log_probs * y).sum(-1) + smoothing)
    # NOTE: because of ignore_index, we can't use Tensor.mean (so can't use `_do_reduction` here)
    return -(unreduced.sum() / loss_mask.sum() if reduction == "mean" else (unreduced.sum() if reduction == "sum" else unreduced))

  def cross_entropy(self, Y:Tensor, reduction:ReductionStr="mean", label_smoothing:float=0.0) -> Tensor:
    """
    Compute the cross entropy loss between input logits and target.

    NOTE: `self` are logits and `Y` are the target labels or class probabilities.

    See: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[-1, 2, -3], [1, -2, 3]])
    Y = Tensor([1, 2])
    print(t.cross_entropy(Y).item())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[-1, 2, -3], [1, -2, 3]])
    Y = Tensor([1, 2])
    print(t.cross_entropy(Y, reduction='none').numpy())
    ```
    """
    assert 0.0 <= label_smoothing <= 1.0, "label_smoothing must be in [0.0, 1.0]"
    Y = Y.one_hot(num_classes=cast(int, self.shape[1])) if Y.ndim < 2 else Y
    Y = (1 - label_smoothing)*Y + label_smoothing / cast(int, Y.shape[1])
    ret = -self.log_softmax(axis=1).mul(Y).sum(axis=1)
    return ret._do_reduction(reduction)

  def nll_loss(self, Y:Tensor, weight:Optional[Tensor]=None, ignore_index:Optional[int]=None, reduction:ReductionStr="mean") -> Tensor:
    """
    Compute the negative log likelihood loss between log-probabilities and target labels.

    NOTE: `self` is log-probabilities and `Y` is the Y labels or class probabilities.

    See: https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[-1, 2, -3], [1, -2, 3]])
    Y = Tensor([1, 2])
    print(t.log_softmax().nll_loss(Y).item())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[-1, 2, -3], [1, -2, 3]])
    Y = Tensor([1, 2])
    print(t.log_softmax().nll_loss(Y, reduction='none').numpy())
    ```
    """
    weight = Tensor.ones_like(Y, requires_grad=False) if weight is None else weight[Y]
    masked_weight = weight if ignore_index is None else weight * (Y != ignore_index)
    nll = -self.gather(1, Y.unsqueeze(1)).squeeze(1) * masked_weight
    return nll.sum() / masked_weight.sum() if reduction == "mean" else nll._do_reduction(reduction)

  # ***** Tensor Properties *****

  @property
  def ndim(self) -> int:
    """
    Returns the number of dimensions in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2], [3, 4]])
    print(t.ndim)
    ```
    """
    return len(self.shape)

  def numel(self) -> sint:
    """
    Returns the total number of elements in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(t.numel())
    ```
    """
    return prod(self.shape)

  def element_size(self) -> int:
    """
    Returns the size in bytes of an individual element in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([5], dtype=dtypes.int16)
    print(t.element_size())
    ```
    """
    return self.dtype.itemsize

  def nbytes(self) -> int:
    """
    Returns the total number of bytes of all elements in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([8, 9], dtype=dtypes.float)
    print(t.nbytes())
    ```
    """
    return self.numel() * self.element_size()

  def is_floating_point(self) -> bool:
    """
    Returns `True` if the tensor contains floating point types, i.e. is one of `dtype.float64`, `dtype.float32`,
    `dtype.float16`, `dtype.bfloat16`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([8, 9], dtype=dtypes.float32)
    print(t.is_floating_point())
    ```
    """
    return dtypes.is_float(self.dtype)

  def size(self, dim:Optional[int]=None) -> Union[sint, Tuple[sint, ...]]:
    """
    Return the size of the tensor. If `dim` is specified, return the length along dimension `dim`. Otherwise return the shape of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[4, 5, 6], [7, 8, 9]])
    print(t.size())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.size(dim=1))
    ```
    """
    return self.shape if dim is None else self.shape[dim]

  # ***** cast ops *****

  def llvm_bf16_cast(self, dtype:DTypeLike):
    # hack for devices that don't support bfloat16
    assert self.dtype == dtypes.bfloat16
    return self.to("LLVM").bitcast(dtypes.uint16).cast(dtypes.uint32).mul(1<<16).bitcast(dtypes.float32).cast(dtype)

  def cast(self, dtype:DTypeLike) -> Tensor:
    """
    Casts `self` to the given `dtype`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2.5, 3], dtype=dtypes.float)
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.cast(dtypes.int32)
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.cast(dtypes.uint8)
    print(t.dtype, t.numpy())
    ```
    """
    if (dt:=to_dtype(dtype)) in {dtypes.uint8, dtypes.uint16} and dtypes.is_float(self.dtype):
      # NOTE: values within the int32 range and outside the unsigned dtype range will cause values to wrap around
      return F.Cast.apply(F.Cast.apply(self, dtype=dtypes.int32), dtype=dt)
    return self if self.dtype == dt else F.Cast.apply(self, dtype=dt)

  def bitcast(self, dtype:DTypeLike) -> Tensor:
    """
    Bitcasts `self` to the given `dtype` of the same itemsize.

    `self` must not require a gradient.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2, 3], dtype=dtypes.int32)
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.bitcast(dtypes.uint32)
    print(t.dtype, t.numpy())
    ```
    """
    if self.requires_grad: raise RuntimeError("can't backprop through bitcast")
    dt = to_dtype(dtype)
    if (not isinstance(self.device, str) or not self.device.startswith("DISK")) and (ns:=dt.itemsize) != (os:=self.dtype.itemsize):
      if (self.shape[-1]*os) % ns != 0: raise RuntimeError("unsupported size in bitcast")
      new_uint, old_uint = to_dtype(f"uint{8*ns}"), to_dtype(f"uint{8*os}")
      tmp = self.bitcast(old_uint)
      if ns > os: return functools.reduce(Tensor.add, (tmp[..., i::ns//os].cast(new_uint) << 8*i*os for i in range(ns//os))).bitcast(dtype)
      return Tensor.stack(*(tmp>>8*i*ns for i in range(os//ns)), dim=-1).flatten(-2).cast(new_uint).bitcast(dtype)
    return F.Cast.apply(self, dtype=dt, bitcast=True) if self.dtype != dt else self

  def float(self) -> Tensor:
    """
    Convenience method to cast `self` to a `float32` Tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2, 3], dtype=dtypes.int32)
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.float()
    print(t.dtype, t.numpy())
    ```
    """
    return self.cast(dtypes.float32)

  def half(self) -> Tensor:
    """
    Convenience method to cast `self` to a `float16` Tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2, 3], dtype=dtypes.int32)
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.half()
    print(t.dtype, t.numpy())
    ```
    """
    return self.cast(dtypes.float16)

  def int(self) -> Tensor:
    """
    Convenience method to cast `self` to a `int32` Tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1.5, -0.5, 0.0, 0.5, 1.5])
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.int()
    print(t.dtype, t.numpy())
    ```
    """
    return self.cast(dtypes.int32)

  def bool(self) -> Tensor:
    """
    Convenience method to cast `self` to a `bool` Tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 0, 1])
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.bool()
    print(t.dtype, t.numpy())
    ```
    """
    return self.cast(dtypes.bool)

  # *** image Tensor function replacements ***

  def image_dot(self, w:Tensor, acc_dtype:Optional[DTypeLike]=None) -> Tensor:
    # NOTE: we use a 1x1 conv2d to do the matmul. mxk @ kxn = (1,k,m,1).conv2d(n,k,1,1)
    x, dx, dw = self, self.ndim, w.ndim
    if not (dx > 0 and dw > 0): raise RuntimeError(f"both tensors need to be at least 1D, got {dx}D and {dw}D")
    if x.shape[-1] != w.shape[-min(w.ndim, 2)]: raise RuntimeError(f"cannot image_dot {x.shape} and {w.shape}")

    bs, groups, cin, cout = prod(self.shape[0:-2]), prod(w.shape[0:-2]), w.shape[-2], w.shape[-1]
    out_shape_t = self.shape[0:-2] + (cout,-1) if len(self.shape) > 1 else (cout, )

    # NOTE: with NHWC we can remove the transposes
    # bs x groups*cin x H x W
    cx = self.transpose(self.ndim-1, self.ndim-2).reshape((bs//groups, groups*cin, -1, 1))
    # groups*cout x cin x H, W
    cw = w.transpose(w.ndim-1, w.ndim-2).reshape((groups*cout, cin, 1, 1))
    return cx.image_conv2d(cw, groups=groups, acc_dtype=acc_dtype).reshape(out_shape_t).transpose(self.ndim-1, self.ndim-2)

  def image_conv2d(self, weight:Tensor, bias:Optional[Tensor]=None, groups=1, stride=1, dilation=1, padding=0, acc_dtype=None) -> Tensor:
    base_image_type = dtypes.imageh if getenv("FLOAT16", 0) else dtypes.imagef

    (bs,_,iy,ix), (cout,cin,H,W) = self.shape, weight.shape
    x, w = self, weight.reshape(groups, (rcout := cout//groups), cin, H, W)

    # hack for non multiples of 4 on cin
    if cin % 4 != 0 and not (cin == 1 and groups%4 == 0):
      x = x.reshape(bs, groups, cin, iy, ix)   # do this always?
      added_input_channels = 4 - (cin % 4)
      w = w.pad(tuple((0, added_input_channels) if i == 2 else None for i in range(w.ndim)))
      x = x.pad(tuple((0, added_input_channels) if i == 2 else None for i in range(x.ndim)))
      cin = cin + added_input_channels
      x = x.reshape(bs, groups*cin, iy, ix)

    # hack for non multiples of 4 on rcout
    added_output_channels = 0
    if rcout % 4 != 0 and not (rcout == 1 and groups%4 == 0):
      added_output_channels = 4 - (rcout % 4)
      rcout += added_output_channels
      cout = groups * rcout
      w = w.pad(tuple((0, added_output_channels) if i == 1 else None for i in range(w.ndim)))

    # packed (note: flipping bs and iy would make the auto-padding work)
    x = x.permute(0,2,3,1)
    cin_last = iy == 1 and ix == 1
    if cin == 1: w = w.reshape(cout//4,4,H,W).permute(0,2,3,1)
    elif cin_last: w = w.reshape(cout//4,4,cin//4,4,H,W).permute(0,4,2,5,1,3)
    else: w = w.reshape(cout//4,4,cin//4,4,H,W).permute(0,4,2,5,3,1)

    # contiguous creates the image, and early realize static weights (TODO: test for the static weight)
    if IMAGE >= 2: x,w = x.cast(base_image_type((bs*iy, ix*groups*cin//4, 4))), w.cast(base_image_type((cout//4, H*W*cin, 4)))
    x, w = x.contiguous(), w.contiguous()

    # expand out
    rcin_hi, rcin_lo = cin//4 if cin >= 4 else 1, 4 if cin >= 4 else 1
    cout_expand = [groups//4 if cin == 1 else groups, 4 if cin == 1 else 1, rcout//4 if rcout >= 4 else 1, 4 if rcout >= 4 else 1]
    x = x.reshape(bs, iy, ix, groups, rcin_hi, rcin_lo)
    if cin_last: w = w.reshape(cout//4, H, rcin_hi, W, 4, rcin_lo)
    else: w = w.reshape(cout//4, H, rcin_hi, W, rcin_lo, 4).permute(0,1,2,3,5,4)

    # prepare input
    x = x.permute(0,3,4,5,1,2).pad(self._padding2d(padding, 2))._pool((H, W), stride, dilation) # -> (bs, groups, rcin_hi, rcin_lo, oy, ox, H, W)
    x = x.permute(0,4,5,1,2,3,6,7).reshape(bs, (oy := x.shape[4]), (ox := x.shape[5]), *cout_expand[0:2], 1, 1, rcin_hi, rcin_lo, H, W)

    # prepare weights
    w = w.permute(0,4,2,5,1,3).reshape((1, 1, 1, *cout_expand, rcin_hi, rcin_lo, H, W))

    # the conv!
    ret = (x*w).cast(base_image_type((bs*oy, ox*cout//4, 4)) if IMAGE >= 2 else dtypes.float32).sum((-4, -3, -2, -1), acc_dtype=acc_dtype)

    # undo hack for non multiples of 4 on C.rcout
    if added_output_channels != 0:
      ret = ret.reshape(bs, oy, ox, groups, rcout)[:, :, :, :, :-added_output_channels]
      cout = groups * (rcout - added_output_channels)

    # NCHW output
    ret = ret.reshape(bs, oy, ox, cout).permute(0,3,1,2)
    return ret if bias is None else ret.add(bias.reshape(1, -1, 1, 1))

def _metadata_wrapper(fn):
  def _wrapper(*args, **kwargs):
    if _METADATA.get() is not None: return fn(*args, **kwargs)

    if TRACEMETA >= 2:
      caller_frame = sys._getframe(frame := 1)
      caller_module = caller_frame.f_globals.get("__name__", None)
      caller_func = caller_frame.f_code.co_name
      if caller_module is None: return fn(*args, **kwargs)

      # if its called from nn we want to step up frames until we are out of nn
      while caller_module.startswith("tinygrad.nn") and "optim" not in caller_module:
        caller_frame = sys._getframe(frame := frame + 1)
        caller_module = caller_frame.f_globals.get("__name__", None)
        if caller_module is None: return fn(*args, **kwargs)

      # if its called from a lambda in tinygrad we want to look two more frames up
      if caller_module.startswith("tinygrad") and caller_func == "<lambda>": caller_frame = sys._getframe(frame := frame + 2)
      caller_module = caller_frame.f_globals.get("__name__", None)
      if caller_module is None: return fn(*args, **kwargs)
      caller_func = caller_frame.f_code.co_name
      caller_lineno = caller_frame.f_lineno

      caller = f"{caller_module}:{caller_lineno}::{caller_func}"
    else: caller = ""

    token = _METADATA.set(Metadata(name=fn.__name__, caller=caller))
    ret = fn(*args, **kwargs)
    _METADATA.reset(token)
    return ret
  return _wrapper

if TRACEMETA >= 1:
  for name, fn in inspect.getmembers(Tensor, inspect.isfunction):
    if name in ["__class__", "__init__", "__new__", "__repr__", "backward", "sequential"]: continue
    setattr(Tensor, name, functools.wraps(fn)(_metadata_wrapper(fn)))
