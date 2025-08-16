# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from __future__ import annotations
import time, math, itertools, functools, struct, sys, inspect, pathlib, string, hashlib, weakref, contextvars
from contextlib import ContextDecorator
from typing import Callable, ClassVar, Sequence, cast, get_args, Literal, SupportsIndex, ParamSpec, TypeVar
from tinygrad.dtype import DType, DTypeLike, dtypes, ImageDType, ConstType, least_upper_float, least_upper_dtype, sum_acc_dtype, to_dtype, truncate
from tinygrad.dtype import _from_np_dtype, _to_np_dtype
from tinygrad.helpers import argfix, make_tuple, flatten, prod, all_int, round_up, merge_dicts, argsort, getenv, all_same, fully_flatten, dedup
from tinygrad.helpers import IMAGE, WINO, Metadata, TRACEMETA, ceildiv, fetch, polyN, unwrap, DEBUG, is_numpy_ndarray
from tinygrad.gradient import compute_gradient
from tinygrad.uop.ops import smax, smin, resolve, UOp, Ops, sint, Variable, MathTrait, identity_element, all_metadata
from tinygrad.uop.spec import tensor_uop_spec, type_verify
from tinygrad.device import Device, Buffer
from tinygrad.engine.realize import run_schedule
from tinygrad.engine.memory import memory_planner
from tinygrad.engine.schedule import ScheduleItem, create_schedule_with_vars
from tinygrad.schedule.kernelize import get_kernelize_map

# *** all in scope Tensors are here. this gets relevant UOps ***

all_tensors: dict[weakref.ref[Tensor], None] = {}
def _find_all_tensors_for_uops(all_uops: set[UOp]) -> list[Tensor]:
  return [t for tref in all_tensors if (t:=tref()) is not None and t.uop in all_uops]

def _apply_map_to_tensors(applied_map:dict[UOp, UOp], name:str|None=None) -> None:
  # get all children of keys in applied_map
  all_uops: set[UOp] = set()
  search_uops = list(applied_map)
  while len(search_uops):
    x = search_uops.pop()
    if x in all_uops: continue
    all_uops.add(x)
    search_uops.extend([u for c in x.children if (u:=c()) is not None])

  # link the found UOps back to Tensors. exit early if there's no Tensors to realize
  # NOTE: this uses all_tensors, but it's fast
  if len(fixed_tensors := _find_all_tensors_for_uops(all_uops)):
    # potentially rewrite all the discovered Tensors
    sink = UOp.sink(*[t.uop for t in fixed_tensors])
    new_sink = sink.substitute(applied_map, name=name)

    # set the relevant uop to the realized UOps
    for t,s,ns in zip(fixed_tensors, sink.src, new_sink.src):
      if s is ns: continue
      t.uop = ns

# **** Tensor helper functions ****

# this tracks the tensor.py METADATA
_METADATA: contextvars.ContextVar[Metadata|None] = contextvars.ContextVar("_METADATA", default=None)

def _fromnp(x: 'np.ndarray') -> UOp:  # type: ignore [name-defined] # noqa: F821
  ret = UOp.new_buffer("NPY", x.size, _from_np_dtype(x.dtype))
  # fake realize
  ret.buffer.allocate(x)
  return ret.reshape(x.shape)

def get_shape(x) -> tuple[int, ...]:
  # NOTE: str is special because __getitem__ on a str is still a str
  if not hasattr(x, "__len__") or not hasattr(x, "__getitem__") or isinstance(x, str) or (hasattr(x, "shape") and x.shape == ()): return ()
  if not all_same(subs:=[get_shape(xi) for xi in x]): raise ValueError(f"inhomogeneous shape from {x}")
  return (len(subs),) + (subs[0] if subs else ())

def _frompy(x:list|tuple|bytes, dtype:DType) -> UOp:
  if isinstance(x, bytes): ret, data = UOp.new_buffer("PYTHON", len(x)//dtype.itemsize, dtype), x
  else:
    ret = UOp.new_buffer("PYTHON", prod(shape:=get_shape(x)), dtype).reshape(shape)
    assert dtype.fmt is not None, f"{dtype=} has None fmt"
    truncate_function = truncate[dtype]
    data = struct.pack(f"@{ret.size}{dtype.fmt}", *[truncate_function(xi) for xi in fully_flatten(x)])
  # fake realize
  ret.buffer.allocate(memoryview(data if Device.DEFAULT != "PYTHON" else bytearray(data)))
  return ret

def _get_winograd_matcols(mat, dims:int, shp:tuple[sint, ...], device:str|tuple[str, ...], dtype:DType) -> list[list[Tensor]]:
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

def _align_left(*shapes:tuple[sint, ...]) -> tuple[tuple[sint, ...], ...]:
  # unsqueeze left to make every shape same length
  max_dim = max(len(shape) for shape in shapes)
  return tuple((1,) * (max_dim - len(shape)) + shape for shape in shapes)
def _broadcast_shape(*shapes:tuple[sint, ...]) -> tuple[sint, ...]:
  return tuple(0 if 0 in nth_dim_sizes else smax(nth_dim_sizes) for nth_dim_sizes in zip(*_align_left(*shapes)))

def _masked_setitem(target:Tensor, values:Tensor, mask:Tensor, axes:tuple[int, ...]) -> Tensor:
  # reduce such that if mask contains repeated indices the last one remains
  for dim in axes: mask, values = functools.reduce(lambda x,y: (x[0]|y[0], y[0].where(y[1], x[1])), zip(mask.split(1, dim), values.split(1, dim)))
  # remove extra dims from reduce
  for dim in reversed(axes): mask, values = mask.squeeze(dim), values.squeeze(dim)
  # select from values for each True element in mask else select from target
  return mask.where(values, target)

#  `(padding_left, padding_right, padding_top, padding_bottom, ...)` ->  `(..., (padding_top, padding_bottom), (padding_left, padding_right))`
def _flat_to_grouped(padding:Sequence[sint]) -> tuple[tuple[sint, sint], ...]: return tuple(zip(padding[-2::-2], padding[::-2]))

ReductionStr = Literal["mean", "sum", "none"]

class Tensor(MathTrait):
  """
  A `Tensor` is a multi-dimensional matrix containing elements of a single data type.

  ```python exec="true" session="tensor"
  from tinygrad import Tensor, dtypes, nn
  import numpy as np
  import math
  np.set_printoptions(precision=4)
  ```
  """
  __slots__ = "uop", "requires_grad", "grad"
  training: ClassVar[bool] = False

  def __init__(self, data:ConstType|bytes|list|tuple|UOp|'np.ndarray'|pathlib.Path|None,  # type: ignore [name-defined] # noqa: F821
               device:str|tuple|list|None=None, dtype:DTypeLike|None=None, requires_grad:bool|None=None):
    if dtype is not None: dtype = to_dtype(dtype)
    if device is None and isinstance(data, pathlib.Path): device = f"DISK:{data.resolve()}"  # keep it on the disk if device is None
    device = tuple(Device.canonicalize(x) for x in device) if isinstance(device, (tuple, list)) else Device.canonicalize(device)

    # tensors can have gradients if you have called .backward
    self.grad:Tensor|None = None

    # NOTE: this can be in three states. False and None: no gradient, True: gradient
    # None (the default) will be updated to True if it's put in an optimizer
    self.requires_grad:bool|None = requires_grad

    # create a UOp from the different types of inputs
    if isinstance(data, UOp):
      assert dtype is None or dtype==data.dtype, "dtype doesn't match, and casting isn't supported"
      if data.op is Ops.BIND:
        var, val = data.unbind()
        # give the bound constant a device
        const = UOp.const(var.dtype, val, device, ())
        data = data.replace(src=(var.replace(src=const.src), const))
    elif data is None: data = UOp.const(dtype or dtypes.default_float, 0, device, ())
    elif isinstance(data, get_args(ConstType)): data = UOp.const(dtype or dtypes.from_py(data), data, device, ())
    elif isinstance(data, bytes): data = _frompy(data, dtypes.uint8 if dtype is None else dtype)
    elif isinstance(data, (list, tuple)):
      if dtype is None:
        if (d := fully_flatten(data)) and all(isinstance(s, bool) for s in d): dtype = dtypes.bool
        else: dtype = dtypes.default_int if d and all_int(d) else dtypes.default_float  # NOTE: this works because all_int([True, False]) is True
      if dtype in [dtypes.bfloat16, *dtypes.fp8s]: data = Tensor(_frompy(data, dtypes.float32), device=device).cast(dtype).uop
      else: data = _frompy(data, dtype)
    elif is_numpy_ndarray(data):
      import numpy as np
      assert isinstance(data, np.ndarray), f"expected np.ndarray, got {data}"
      if data.shape == (): data = UOp.const(dtype or _from_np_dtype(data.dtype), data.item(), device, ())
      else: data = _fromnp(data.astype(npdtype) if dtype is not None and (npdtype:=_to_np_dtype(dtype)) is not None else data)  # type: ignore [name-defined]
    elif isinstance(data, pathlib.Path):
      dtype = dtype or dtypes.uint8
      data = UOp.new_buffer(f"DISK:{data.resolve()}", data.stat().st_size // dtype.itemsize, dtype)

    # by this point, it has to be a UOp
    if not isinstance(data, UOp): raise RuntimeError(f"can't create Tensor from {data!r} with type {type(data)}")

    # data might be on a different device
    if isinstance(device, str): self.uop:UOp = data if data.device == device else data.copy_to_device(device)
    # if device is a tuple, we should have/construct a MultiLazyBuffer
    elif isinstance(data.device, str): self.uop = Tensor(data).shard(device).uop
    else:
      assert data.device == device, f"MultiLazyBuffer device mismatch, {data.device} != {device}"
      self.uop = data

    # add to all_tensors after construction succeeds
    all_tensors[weakref.ref(self)] = None
  def __del__(self): all_tensors.pop(weakref.ref(self), None)

  def _apply_uop(self, fxn:Callable, *x:Tensor, **kwargs) -> Tensor:
    new_uop: UOp = fxn(*[t.uop for t in (self,)+x], **kwargs)
    if (metadata:=_METADATA.get()) is not None: all_metadata[new_uop] = (metadata,)
    needs_input_grad = [t.requires_grad for t in (self,)+x]
    return Tensor(new_uop, device=new_uop.device, requires_grad=True if any(needs_input_grad) else None if None in needs_input_grad else False)

  def _apply_broadcasted_uop(self, fxn:Callable, x:Tensor|ConstType, reverse=False) -> Tensor:
    lhs,rhs = self._broadcasted(x, reverse)
    return lhs._apply_uop(fxn, rhs)

  # _binop is used by MathTrait
  def _binop(self, op, x, reverse): return self._apply_broadcasted_uop(lambda *u: UOp.alu(u[0], op, *u[1:]), x, reverse)

  def requires_grad_(self, requires_grad=True) -> Tensor:
    self.requires_grad = requires_grad
    return self

  class train(ContextDecorator):
    def __init__(self, mode:bool = True): self.mode = mode
    def __enter__(self): self.prev, Tensor.training = Tensor.training, self.mode
    def __exit__(self, exc_type, exc_value, traceback): Tensor.training = self.prev

  def __repr__(self):
    ld = self.uop
    ld_repr = f"<UOp {ld.device} {ld.shape} {str(ld.dtype)[7:]} {ld.st if ld.base is not ld else (ld.op, ld.realized)}>"
    return f"<Tensor {ld_repr} on {self.device} with grad {(self.grad.uop if self.grad is not None else None)!r}>"

  # Python has a non moving GC, so this should be okay
  def __hash__(self): return id(self)

  def __bool__(self): raise TypeError("__bool__ on Tensor is not defined")

  def __len__(self):
    if not self.shape: raise TypeError("len() of a 0-d tensor")
    return self.shape[0]

  @property
  def device(self) -> str|tuple[str, ...]: return self.uop.device

  @property
  def shape(self) -> tuple[sint, ...]: return self.uop.shape

  @property
  def dtype(self) -> DType: return self.uop.dtype

  # ***** data handlers ****

  def kernelize(self, *lst:Tensor) -> Tensor:
    """
    Creates the kernels and buffers needed to realize these Tensor(s).

    NOTE: Kernelize can be called multiple times on a Tensor
    """
    big_sink = UOp.sink(*[x.uop for x in (self,)+lst])

    # verify Tensors match the spec
    if __debug__: type_verify(list(big_sink.toposort()), tensor_uop_spec)

    becomes_map = get_kernelize_map(big_sink)
    _apply_map_to_tensors(becomes_map, name="Apply Kernelize Map")
    return self

  def schedule_with_vars(self, *lst:Tensor) -> tuple[list[ScheduleItem], dict[Variable, int]]:
    """
    Creates the schedule needed to realize these Tensor(s), with Variables.

    NOTE: A Tensor can only be scheduled once.
    """
    st = time.perf_counter()
    self.kernelize(*lst)
    sink = UOp.sink(*[x.uop for x in (self,)+lst])

    # remove all ASSIGNs, after scheduling, the tensors are just buffers
    remove_assign_map = {u:u.buf_uop for u in sink.toposort() if u.op is Ops.ASSIGN}
    _apply_map_to_tensors(remove_assign_map, name="Remove Assigns")

    # create the schedule
    schedule, var_vals = create_schedule_with_vars(sink)
    schedule = memory_planner(schedule)
    if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels in {(time.perf_counter()-st)*1000:.2f} ms")
    return schedule, var_vals

  def schedule(self, *lst:Tensor) -> list[ScheduleItem]:
    """Creates the schedule needed to realize these Tensor(s)."""
    schedule, var_vals = self.schedule_with_vars(*lst)
    assert len(var_vals) == 0
    return schedule

  def realize(self, *lst:Tensor, do_update_stats=True) -> Tensor:
    """Triggers the computation needed to create these Tensor(s)."""
    run_schedule(*self.schedule_with_vars(*lst), do_update_stats=do_update_stats)
    return self

  def replace(self, x:Tensor, allow_shape_mismatch=False) -> Tensor:
    """
    Replaces the data of this tensor with the data of another tensor. Only the shape of the tensors must match.
    """
    # used for replacing a Tensor with a new version of it (potentially with a different device and dtype)
    assert self.shape == x.shape or allow_shape_mismatch, f"replace shape mismatch {self.shape} != {x.shape}"
    self.uop = x.uop
    return self

  def assign(self, x) -> Tensor:
    # TODO: this is a hack for writing to DISK. remove with working assign
    if isinstance(self.device, str) and self.device.startswith("DISK"):
      if x.__class__ is not Tensor: x = Tensor(x, device="CPU", dtype=self.dtype)
      self._buffer().copyin(x._data())
      return self
    if x.__class__ is not Tensor: x = Tensor(x, device=self.device, dtype=self.dtype)
    if self.uop is x.uop: return self  # a self assign is a NOOP
    # NOTE: we allow cross device assign
    assert self.shape == x.shape, f"assign shape mismatch {self.shape} != {x.shape}"
    assert self.device == x.device, f"assign device mismatch {self.device} != {x.device}"
    assert self.dtype == x.dtype, f"assign dtype mismatch {self.dtype} != {x.dtype}"
    self.uop = self.uop.assign(x.uop)
    return self

  def detach(self) -> Tensor:
    """
    Returns a new tensor with the same data as this tensor, but detached from the autograd graph.
    """
    return Tensor(self.uop.detach(), device=self.device, requires_grad=False)

  def _buffer(self) -> Buffer:
    x = self.cast(self.dtype.base).contiguous()
    if isinstance(self.device, tuple): x = x.to("CPU")
    return cast(Buffer, x.realize().uop.base.buffer).ensure_allocated()
  def _data(self) -> memoryview: return self._buffer().as_buffer()

  def data(self) -> memoryview:
    """
    Returns the data of this tensor as a memoryview.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(np.frombuffer(t.data(), dtype=np.int32))
    ```
    """
    if 0 in self.shape: return memoryview(bytearray(0)).cast(self.dtype.base.fmt)
    assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
    return self._buffer().as_typed_buffer(self.shape)

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

  # TODO: should be Tensor.tolist() -> Union[list[ConstType], ConstType]. The list is Sequence because mypy expects memoryview.tolist() -> list[int]
  # src: https://github.com/python/mypy/blob/release-1.6/mypy/typeshed/stdlib/builtins.pyi#L803
  def tolist(self) -> Sequence[ConstType]|ConstType:
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
    return self.data().tolist()

  def numpy(self) -> 'np.ndarray':  # type: ignore [name-defined] # noqa: F821
    """
    Returns the value of this tensor as a `numpy.ndarray`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(repr(t.numpy()))
    ```
    """
    assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
    import numpy as np
    if self.dtype.base == dtypes.bfloat16: return self.float().numpy()
    if 0 in self.shape: return np.empty(self.shape, dtype=_to_np_dtype(self.dtype.base))
    return self._buffer().numpy().reshape(self.shape)

  def clone(self) -> Tensor:
    """
    Creates a clone of this tensor allocating a separate buffer for the data.
    """
    ret = Tensor.empty(self.shape, device=self.device, dtype=self.dtype)
    if self.grad is not None: ret.grad = self.grad.clone()
    return ret.assign(self)

  def to(self, device:str|tuple[str, ...]|None) -> Tensor:
    """
    Moves the tensor to the given device.
    """
    device = tuple(Device.canonicalize(x) for x in device) if isinstance(device, (tuple, list)) else Device.canonicalize(device)
    if device == self.device: return self
    if not isinstance(device, str): return self.shard(device)
    ret = Tensor(self.uop, device, requires_grad=self.requires_grad)
    if self.grad is not None: ret.grad = self.grad.to(device)
    return ret

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
    assert isinstance(self.device, str), "can't shard a MultiLazyBuffer"
    devices = tuple(Device.canonicalize(x) for x in devices)
    mlb = self.uop.shard(devices, self._resolve_dim(axis)) if axis is not None else self.uop.copy_to_device(devices)
    return Tensor(mlb, device=devices, requires_grad=self.requires_grad)

  def shard_(self, devices:tuple[str, ...], axis:int|None=None) -> Tensor:
    """
    Shards the tensor across the given devices in place.
    """
    return self.replace(self.shard(devices, axis))

  @staticmethod
  def from_uop(y:UOp, **kwargs) -> Tensor:
    if y.op is Ops.BIND: return Tensor(y, **kwargs, requires_grad=False)
    if y.op is Ops.CONST: return Tensor(y.arg, **kwargs, requires_grad=False)
    if y.op is Ops.MUL: return Tensor.from_uop(y.src[0]) * Tensor.from_uop(y.src[1])
    if y.op is Ops.ADD: return Tensor.from_uop(y.src[0]) + Tensor.from_uop(y.src[1])
    raise RuntimeError(f"unhandled UOp {y}")

  # ***** creation entrypoint *****

  @staticmethod
  def empty(*shape, device:str|tuple[str, ...]|None=None, dtype:DTypeLike|None=None, **kwargs) -> Tensor:
    """
    Creates an empty tensor with the given shape.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.empty(2, 3)
    print(t.shape)
    ```
    """
    dtype, shape = to_dtype(dtype) if dtype is not None else dtypes.default_float, argfix(*shape)
    if not isinstance(size:=prod([x.vmax if isinstance(x, UOp) else x for x in shape]), int): raise ValueError(f"size must be int {size}")
    # TODO: add test for multidevice tensor
    device = tuple(Device.canonicalize(d) for d in device) if isinstance(device, tuple) else Device.canonicalize(device)
    return Tensor(UOp.new_buffer(device, size, dtype), device, dtype, **kwargs).reshape(shape)

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
  def _threefry_random_bits(key:Tensor, counts0:Tensor, counts1:Tensor) -> Tensor:
    x = (counts1.cast(dtypes.uint64) << 32) | counts0.cast(dtypes.uint64)
    x = x._apply_uop(UOp.threefry, (key[1]._broadcast_to(x.shape).cast(dtypes.uint64) << 32) | key[0]._broadcast_to(x.shape).cast(dtypes.uint64))
    counts0, counts1 = (x & 0xffffffff).cast(dtypes.uint32), ((x >> 32) & 0xffffffff).cast(dtypes.uint32)
    return counts0.cat(counts1)

  @staticmethod
  def rand(*shape, device:str|None=None, dtype:DTypeLike|None=None, contiguous:bool=True, **kwargs) -> Tensor:
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
    device = Device.canonicalize(device)

    # if shape has 0, return zero tensor
    if (numel := prod(shape)) == 0: return Tensor.zeros(shape, device=device, dtype=dtype, **kwargs)
    num = ceildiv(numel * dtype.itemsize, 4)

    # generate per device seeds and rng counter if we haven't seen this device yet
    if device not in Tensor._device_seeds:
      Tensor._device_seeds[device] = Tensor(
        [int.from_bytes(hashlib.sha256(len(Tensor._device_seeds).to_bytes(4, "big")).digest(), "big"), Tensor._seed],
        device=device, dtype=dtypes.uint32, requires_grad=False)
      Tensor._device_rng_counters[device] = Tensor([num], device=device, dtype=dtypes.uint32, requires_grad=False)
    # increment rng counter for devices
    else: Tensor._device_rng_counters[device].assign(Tensor._device_rng_counters[device] + num).contiguous()

    # threefry random bits
    bits_count = Tensor._device_rng_counters[device] - num
    counts0 = (Tensor.arange(ceildiv(num, 2), device=device, dtype=dtypes.uint32, requires_grad=False)+bits_count)
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
    out = bits.bitcast(dtype)[:numel].sub(1).reshape(shape).requires_grad_(kwargs.get("requires_grad"))
    return out.contiguous() if contiguous else out

  # ***** creation helper functions *****

  @staticmethod
  def full(shape:tuple[sint, ...], fill_value:ConstType, **kwargs) -> Tensor:
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
  def linspace(start:int|float, stop:int|float, steps:int, **kwargs) -> Tensor:
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
  def eye(n:int, m:int|None=None, **kwargs) -> Tensor:
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
    x = Tensor.ones(n, **kwargs).diag()
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
    if isinstance(self.device, tuple):
      if kwargs.get("device") is not None: raise RuntimeError("cannot specify `device` on `rand_like` of a multi device tensor")
      if self.uop.axis is None: return Tensor.rand(*self.shape, dtype=dtype, **kwargs).shard(self.device)
      contiguous = kwargs.pop("contiguous", True)
      sharded_shape = tuple(s//len(self.device) if a==self.uop.axis else s for a,s in enumerate(self.shape))
      rands = UOp(Ops.MSTACK, dtype=dtype,
                  src=tuple([Tensor.rand(sharded_shape, device=d, dtype=dtype, contiguous=contiguous, **kwargs).uop for d in self.device]))
      return Tensor(UOp.multi(rands, axis=self.uop.axis), device=self.device, dtype=dtype, **kwargs)
    return Tensor.rand(*self.shape, device=kwargs.pop("device", self.device), dtype=dtype, **kwargs)

  # ***** rng hlops *****

  def randn_like(self, dtype:DTypeLike|None=None, requires_grad:bool|None=None, **kwargs) -> Tensor:
    """
    Creates a tensor with the same shape and sharding as `self`, filled with random values from a normal distribution with mean 0 and variance 1.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(Tensor.randn_like(t).numpy())
    ```
    """
    src = self.stack(self).rand_like(**{**kwargs, "dtype": dtypes.float32})
    # https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    return (src[0].mul(2*math.pi).cos().mul((1 - src[1]).log().mul(-2).sqrt()).cast(dtype or self.dtype)).requires_grad_(requires_grad)

  @staticmethod
  def randn(*shape, dtype:DTypeLike|None=None, requires_grad:bool|None=None, **kwargs) -> Tensor:
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
    return Tensor.empty(*shape, **kwargs).randn_like(dtype=dtype, requires_grad=requires_grad)

  @staticmethod
  def randint(*shape, low=0, high=10, dtype=dtypes.int32, **kwargs) -> Tensor:
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
    dtype = to_dtype(dtype)
    if not dtypes.is_int(dtype): raise TypeError(f"{dtype=} must be int")
    return Tensor.uniform(*shape, low=low, high=high, dtype=dtype, **kwargs)

  @staticmethod
  def normal(*shape, mean=0.0, std=1.0, requires_grad:bool|None=None, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a normal distribution with the given `mean` and standard deviation `std`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.normal(2, 3, mean=10, std=2).numpy())
    ```
    """
    return ((std * Tensor.randn(*shape, **kwargs)) + mean).requires_grad_(requires_grad)

  @staticmethod
  def uniform(*shape, low=0.0, high=1.0, dtype:DTypeLike|None=None, requires_grad:bool|None=None, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a uniform distribution over the interval `[low, high)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.uniform(2, 3, low=2, high=10).numpy())
    ```
    """
    return (((high-low) * Tensor.rand(*shape, **kwargs)).cast(dtype or dtypes.default_float) + low).requires_grad_(requires_grad)

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

  @staticmethod
  def randperm(n:int, device=None, dtype=dtypes.int32, **kwargs) -> Tensor:
    """
    Returns a tensor with a random permutation of integers from `0` to `n-1`.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.randperm(6).numpy())
    ```
    """
    return Tensor.rand(n, device=device, **kwargs).argsort().cast(dtype)

  def multinomial(self:Tensor, num_samples:int = 1, replacement:bool = False) -> Tensor:
    """
    Returns a tensor with `num_samples` indices sampled from a multinomial distribution weighted by `self`.

    NOTE: `replacement=False` for `num_samples > 1` is not supported yet.
    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor([1, 2, 3, 4])
    print(t.multinomial(20, replacement=True).numpy())
    ```
    """
    assert 1 <= self.ndim <= 2 and num_samples > 0, f"{self.ndim=} must be 1 or 2 dim, {num_samples=} must be positive"
    assert replacement or num_samples == 1, "no replacement only supports num_samples = 1"
    weight = self.unsqueeze(0) if self.ndim == 1 else self
    cdf = (cw := weight.cumsum(1).float()) / cw[:, -1].unsqueeze(1)
    unif_samples = Tensor.rand(num_samples, cdf.shape[0], 1).to(self.device)
    indices = (unif_samples.expand((-1, -1, cdf.shape[1])) >= cdf).sum(2).permute((1, 0))
    return (indices.squeeze(0) if self.ndim == 1 else indices).cast(dtypes.int32)

  # ***** toposort and backward pass *****

  def gradient(self, *targets:Tensor, gradient:Tensor|None=None, materialize_grads=False) -> list[Tensor]:
    """
    Computes the gradient of the targets with respect to self.

    ```python exec="true" source="above" session="tensor" result="python"
    x = Tensor.eye(3)
    y = Tensor([[2.0,0,-2.0]])
    z = y.matmul(x).sum()
    dx, dy = z.gradient(x, y)

    print(dx.tolist())  # dz/dx
    print(dy.tolist())  # dz/dy
    ```
    """
    assert gradient is not None or self.shape == tuple(), "when no gradient is provided, backward must be called on a scalar tensor"
    if not (self.is_floating_point() and all(t.is_floating_point() for t in targets)): raise RuntimeError("only float Tensors have gradient")
    if gradient is None: gradient = Tensor(1.0, dtype=self.dtype, device=self.device, requires_grad=False)
    target_uops = [x.uop for x in targets]
    grads = compute_gradient(self.uop, gradient.uop, set(target_uops))
    ret = []
    for x in target_uops:
      if (y:=grads.get(x)) is None:
        if materialize_grads: y = x.const_like(0)
        else: raise RuntimeError(f"{x}\n\nnot found in\n\n{self.uop}")
      ret.append(y)
    # create returned Tensors
    return [Tensor(u, device=t.device) for t,u in zip(targets, ret)]

  def backward(self, gradient:Tensor|None=None) -> Tensor:
    """
    Propagates the gradient of a tensor backwards through the computation graph.
    If the 'gradient' argument is not provided, the tensor must be a scalar, and the gradient is implicitly set to 1.0.
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    t.sum().backward()
    print(t.grad.numpy())
    ```
    """
    all_uops = self.uop.toposort()
    tensors_need_grad: list[Tensor] = [t for tref in all_tensors if (t:=tref()) is not None and \
                                       t.uop in all_uops and t.requires_grad]
    # clear contexts
    for t,g in zip(tensors_need_grad, self.gradient(*tensors_need_grad, gradient=gradient, materialize_grads=True)):
      assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
      t.grad = g if t.grad is None else (t.grad + g)
    return self

  # ***** movement low level ops *****

  def view(self, shape:tuple[sint, ...], *args) -> Tensor:
    """`.view` is an alias for `.reshape`."""
    return self.reshape(shape, *args)

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
    return self._apply_uop(UOp.reshape, arg=new_shape) if new_shape != self.shape else self

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
    t = Tensor.empty(2, 3, 5)
    print(t.shape)
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.permute(2, 0, 1).shape)
    ```
    """
    order_arg = tuple(self._resolve_dim(x) for x in argfix(order, *args))
    if sorted(order_arg) != list(range(self.ndim)): raise RuntimeError(f"order is not a valid permutation, getting {order_arg}")
    return self._apply_uop(UOp.permute, arg=order_arg)

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
    return self._apply_uop(UOp.flip, arg=tuple([i in axis_arg for i in range(len(self.shape))]))

  def shrink(self, arg:tuple[tuple[sint, sint]|None, ...]) -> Tensor:
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
    return self._apply_uop(UOp.shrink, arg=tuple(shrink_arg))

  def pad(self, padding:Sequence[sint]|Sequence[tuple[sint, sint]|None], mode:str="constant", value:float=0.0) -> Tensor:
    """
    Returns a tensor with padding applied based on the input `padding`.

    `padding` supports two padding structures:

    1. Flat padding: `(padding_left, padding_right, padding_top, padding_bottom, ...)`
        - This structure matches PyTorch's pad.
        - `padding` length must be even.

    2. Group padding: `(..., (padding_top, padding_bottom), (padding_left, padding_right))`
        - This structure matches pad for JAX, NumPy, TensorFlow, and others.
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
    # flat padding
    if all(isinstance(p, (int,UOp)) for p in padding):
      if len(padding)%2 != 0: raise ValueError("Flat padding must have even number of pads")
      pX = _flat_to_grouped(tuple(cast(Sequence[sint], padding)) + (0,0)*(self.ndim - len(padding)//2))
    # group padding
    else: pX = tuple((0,0) if p is None else p for p in cast(Sequence[tuple[sint, sint]|None], padding))
    if len(pX) != self.ndim: raise ValueError(f"padding length is improper, {padding=} {self.ndim=}")
    X, pads = self, tuple((smax(pB,0), smax(pA,0)) for pB,pA in pX)
    if mode == "constant":
      def _constant(x:Tensor,px,v) -> Tensor:
        return x._apply_uop(UOp.pad, arg=px) if v == 0 else (x._apply_uop(UOp.pad, arg=px)+Tensor.ones_like(x)._apply_uop(UOp.pad, arg=px).where(0,v))
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

  def _getitem(self, indices, v: Tensor|None = None) -> Tensor:
    # wrap single index into a list
    if (isinstance(indices, list) and all_int(indices)) or not isinstance(indices, (tuple, list)): indices = [indices]
    x, indices = self, list(indices)

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
        case Tensor():
          if not dtypes.is_int(index.dtype): raise IndexError(f"index dtype {index.dtype} is not supported")
          index = (index < 0).where(index+size, index).to(self.device)  # treat negative index values
        case list() | tuple():
          if not dtypes.is_int((ti:=Tensor(index)).dtype): raise IndexError(f"{index=} contains non-int element")
          index = Tensor([i+size if i<0 else i for i in fully_flatten(index)], self.device, requires_grad=False).reshape(ti.shape)
        case int() | UOp(): # sint
          if index >= size or index < -size: raise IndexError(f"{index=} is out of bounds with {size=}")
          boundary = [index, index+1] if index >= 0 else [index+size, index+size+1]
        case slice():
          if index.step == 0: raise ValueError(f"{index=} cannot have 0 as step")
          start, stop = 0 if index.start is None else index.start, size if index.stop is None else index.stop
          step = 1 if index.step is None else index.step
          boundary, stride = [start, stop], step
          if all(isinstance(s, int) for s in (start,stop,step)):
            # handle int slicing
            *boundary, stride = index.indices(cast(SupportsIndex, size))
            if stride * (boundary[1] - boundary[0]) < 0: boundary = [0, 0]
            elif stride < 0: boundary = [boundary[1] + 1, boundary[0] + 1]
            # update size for slice
            size = ceildiv((boundary[1] - boundary[0]), abs(stride))
          elif (step == 1) and isinstance(step, int) and all(isinstance(s,(int,UOp)) for s in (start, stop)) and resolve((stop-start) > 0, False):
            # simple symbolic slice
            size = cast(UOp|int, cast(UOp, (stop - start)).ssimplify())
          else: raise TypeError(f"slice {index=} is not supported")
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
        if not all_int(x.shape): raise RuntimeError("symbolic shape not supported")
        x = x.pad(tuple((0, round_up(s, st) - s) for s, st in zip(x.shape, strides)))
        x = x.reshape(tuple(flatten((s // st, st) for s, st in zip(x.shape, strides))))
        x = x.shrink(tuple(flatten(((0, s), (0, 1)) for s in x.shape[::2]))).reshape(x.shape[::2])

    # dim injection from None by including None dim size (which is 1) and dim collapse by skipping int dim size
    x = x.reshape(tuple(index['size'] for index in indices_parsed if not isinstance(index['index'], (int, UOp))))

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
      x = (x.reshape(reshape_arg) * mask).sum(sum_axis:=tuple(d + len(big_shape) for d in dims), dtype=x.dtype)

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
    """
    Retrieves a sub-tensor using indexing.

    Supported Index Types: `int | slice | Tensor | None | list | tuple | Ellipsis`

    Examples:
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(12).reshape(3, 4)
    print(t.numpy())
    ```

    - Int Indexing: Select an element or sub-tensor using integers for each dimension.
      ```python exec="true" source="above" session="tensor" result="python"
      print(t[1, 2].numpy())
      ```

    - Slice Indexing: Select a range of elements using slice notation (`start:end:stride`).
      ```python exec="true" source="above" session="tensor" result="python"
      print(t[0:2, ::2].numpy())
      ```

    - Tensor Indexing: Use another tensor as indices for advanced indexing. Using `tuple` or `list` here also works.
      ```python exec="true" source="above" session="tensor" result="python"
      print(t[Tensor([2, 0, 1]), Tensor([1, 2, 3])].numpy())
      ```

    - `None` Indexing: Add a new dimension to the tensor.
      ```python exec="true" source="above" session="tensor" result="python"
      print(t[:, None].shape)
      ```

    NOTE: Out-of-bounds indexing results in a value of `0`.
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t[Tensor([4, 3, 2])].numpy())
    ```
    """
    return self._getitem(indices)

  def __setitem__(self, indices, v:Tensor|ConstType) -> None:
    if isinstance(self.device, str) and self.device.startswith("DISK"):
      self.realize()._getitem(indices).assign(v)
      return
    # NOTE: check that setitem target is valid first
    if not unwrap(self.uop.st).contiguous: raise RuntimeError("setitem target needs to be contiguous")
    if isinstance(v, get_args(ConstType)): v = Tensor(v, device=self.device, dtype=self.dtype)
    if not isinstance(v, Tensor): raise TypeError(f"can't set a {type(v).__name__} to a Tensor")
    if self.requires_grad or v.requires_grad: raise NotImplementedError("setitem with requires_grad is not supported")

    res = self.realize()._getitem(indices, v)
    # if shapes match and data is not shared it's a copy and we assign to self
    if res.shape == self.shape and res.uop is not self.uop:
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
    return (x * index.unsqueeze(-1)._one_hot_along_dim(self.shape[dim])).sum(-1, dtype=self.dtype)

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
    return Tensor.cat(*[t.unsqueeze(dim) for t in argfix(self, *args)], dim=dim)

  def repeat_interleave(self, repeats:int, dim:int|None=None) -> Tensor:
    """
    Repeats elements of a tensor.

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

  def split(self, sizes:int|Sequence[int], dim:int=0) -> tuple[Tensor, ...]:
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

  def chunk(self, chunks:int, dim:int=0) -> list[Tensor]:
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

  def unfold(self, dim:int, size:sint, step:int) -> Tensor:
    """
    Unfolds the tensor along dimension `dim` into overlapping windows.

    Each window has length `size` and begins every `step` elements of `self`.
    Returns the input tensor with dimension `dim` replaced by dims `(n_windows, size)`
    where `n_windows = (self.shape[dim] - size) // step + 1`.

    ```python exec="true" source="above" session="tensor" result="python"
    unfolded = Tensor.arange(8).unfold(0,2,2)
    print("\\n".join([repr(x.numpy()) for x in unfolded]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    unfolded = Tensor.arange(27).reshape(3,3,3).unfold(-1,2,3)
    print("\\n".join([repr(x.numpy()) for x in unfolded]))
    ```
    """
    if size < 0: raise RuntimeError(f'size must be >= 0 but got {size=}')
    if step <= 0: raise RuntimeError(f'step must be > 0 but got {step=}')
    if size > self.shape[dim]: raise RuntimeError(f'maximum size for tensor at dimension {dim} is {self.shape[dim]} but size is {size}')
    dim = self._resolve_dim(dim)
    perm_to_last = tuple(i for i in range(self.ndim) if i != dim) + (dim,)
    return self.permute(perm_to_last)._pool((size,), step).permute(argsort(perm_to_last) + (self.ndim,))

  def meshgrid(self:Tensor, *args:Tensor, indexing:Literal["ij", "xy"]="ij") -> tuple[Tensor, ...]:
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

  def squeeze(self, dim:int|None=None) -> Tensor:
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

  def flatten(self, start_dim=0, end_dim=-1) -> Tensor:
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

  def unflatten(self, dim:int, sizes:tuple[int,...]) -> Tensor:
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

  def diag(self) -> Tensor:
    """
    Returns a 2-D square tensor with the elements of input as the main diagonal.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, 2, 3]).diag().numpy())
    ```
    """
    if self.ndim != 1: raise ValueError(f"expect input to be 1-D, getting {self.ndim}-D")
    return self.unsqueeze(-1).pad((None,(0,n:=self.shape[0]))).flatten().shrink(((0,n*n),)).reshape(n,n)

  def diagonal(self) -> Tensor:
    """
    Returns a view of input tensor with its main diagonal elements.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(3, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.diagonal().numpy())
    ```
    """
    if self.ndim != 2 or (n:=self.shape[0]) != self.shape[1]: raise ValueError(f"only 2-D square tensor is supported, getting {self.shape=}")
    return self.flatten().pad(((0, n))).reshape(n, n+1)[:, 0]

  def roll(self, shifts:int|tuple[int, ...], dims:int|tuple[int, ...]|None=None) -> Tensor:
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
    if dims is None: return self.flatten().roll(shifts, 0).reshape(self.shape)
    dims, shifts, slices = tuple(self._resolve_dim(d) for d in make_tuple(dims, 1)), make_tuple(shifts, 1), [slice(None)] * self.ndim
    if len(dims) != len(shifts): raise RuntimeError(f"{len(dims)=} != {len(shifts)=}")
    for dim, shift in zip(dims, shifts): slices[dim] = slice(delta:=self.shape[dim]-shift%self.shape[dim], delta+self.shape[dim])
    return self.repeat(*tuple(2 if i in dims else 1 for i in range(self.ndim)))[slices]

  def rearrange(self, formula:str, **sizes) -> Tensor:
    """
    Rearranges input according to formula

    See: https://einops.rocks/api/rearrange/

    ```python exec="true" source="above" session="tensor" result="python"
    x = Tensor([[1, 2], [3, 4]])
    print(Tensor.rearrange(x, "batch channel -> (batch channel)").numpy())
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

  def masked_select(self, mask):
    """
    Selects elements from `self` based on the boolean `mask`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    mask = Tensor([[True, False, True], [False, True, False], [False, False, True]])
    print(t.numpy())
    print(mask.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.masked_select(mask).numpy())
    ```
    """
    if not dtypes.is_bool(mask.dtype): raise RuntimeError(f"masked_select expects bool mask tensor, got {mask.dtype}")
    x, mask = self.flatten(), mask._broadcast_to(self.shape).flatten()
    mask_cumsum = mask.cumsum()
    counts = Tensor.zeros(mask_cumsum[-1].item(), dtype=dtypes.int32)
    idxs = counts.scatter(0, mask_cumsum, 1, reduce='add').cumsum()
    return x[idxs]

  def masked_fill(self:Tensor, mask:Tensor, value:Tensor|ConstType) -> Tensor:
    """
    Replaces `self` with `value` wherever the elements of `mask` are True.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4, 5])
    mask = Tensor([True, False, True, False, False])
    print(t.masked_fill(mask, -12).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4, 5])
    mask = Tensor([True, False, True, False, False])
    value = Tensor([-1, -2, -3, -4, -5])
    print(t.masked_fill(mask, value).numpy())
    ```
    """
    return mask.where(value, self)

  # ***** reduce ops *****

  def _reduce(self, op:Ops, axis:int|Sequence[int]|None=None, keepdim=False) -> Tensor:
    axis = tuple(self._resolve_dim(x) for x in (range(self.ndim) if axis is None else make_tuple(axis, 1)))
    if self.ndim == 0: axis = ()
    ret = self._apply_uop(UOp.r, op=op, axis=axis)
    return ret if keepdim else ret.reshape(tuple(s for i,s in enumerate(self.shape) if i not in axis))

  def sum(self, axis:int|Sequence[int]|None=None, keepdim=False, dtype:DTypeLike|None=None) -> Tensor:
    """
    Returns the sum of the elements of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the maximum is computed and whether the reduced dimensions are retained.

    You can pass in `dtype` keyword argument to control the data type of the accumulation.
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
    ret = self.cast(sum_acc_dtype(self.dtype) if dtype is None else dtype)._reduce(Ops.ADD, axis, keepdim)
    return ret.cast(self.dtype) if dtype is None and self.dtype in (dtypes.float16, dtypes.bfloat16) else ret

  def prod(self, axis:int|Sequence[int]|None=None, keepdim=False, dtype:DTypeLike|None=None) -> Tensor:
    """
    Returns the product of the elements of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the maximum is computed and whether the reduced dimensions are retained.

    You can pass in `dtype` keyword argument to control the data type of the accumulation.
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
    return self.cast(dtype if dtype is not None else self.dtype)._reduce(Ops.MUL, axis, keepdim)

  def max(self, axis:int|Sequence[int]|None=None, keepdim=False) -> Tensor:
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
    return self._reduce(Ops.MAX, axis, keepdim)

  def _inverse(self) -> Tensor: return -self if self.is_floating_point() else ~self if dtypes.is_int(self.dtype) else self.logical_not()

  def min(self, axis:int|Sequence[int]|None=None, keepdim=False) -> Tensor:
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
    return self._inverse().max(axis=axis, keepdim=keepdim)._inverse()

  def any(self, axis:int|Sequence[int]|None=None, keepdim=False) -> Tensor:
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

  def all(self, axis:int|Sequence[int]|None=None, keepdim=False) -> Tensor:
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

  def isclose(self, other:Tensor, rtol:float=1e-05, atol:float=1e-08, equal_nan=False) -> Tensor:
    """
    Returns a new tensor with element-wise comparison of closeness to `other` within a tolerance.

    The `rtol` and `atol` keyword arguments control the relative and absolute tolerance of the comparison.

    By default, two `NaN` values are not close to each other. If `equal_nan` is `True`, two `NaN` values are considered close.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1e-7, 1e-8, 1e-9, float('nan')]).isclose(Tensor([0.0, 0.0, 0.0, float('nan')])).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([float('nan')]).isclose(Tensor([float('nan')]), equal_nan=True).numpy())
    ```
    """
    is_finite_close = self.isfinite() & other.isfinite() & ((self - other).abs() <= atol + rtol * other.abs())
    is_infinite_close = (self.isinf() | other.isinf()) & (self == other)
    is_nan_close = (self.isnan() & other.isnan()) & equal_nan
    return is_finite_close | is_infinite_close | is_nan_close

  def mean(self, axis:int|Sequence[int]|None=None, keepdim=False) -> Tensor:
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
    return numerator.div(prod([cast(int, si) for si, so in zip(self.shape, self.sum(axis=axis, keepdim=True).shape) if resolve(si != so)])) \
      .cast(output_dtype)

  def var(self, axis:int|Sequence[int]|None=None, keepdim=False, correction=1) -> Tensor:
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

  def var_mean(self, axis:int|Sequence[int]|None=None, keepdim=False, correction=1) -> tuple[Tensor, Tensor]:
    """
    Calculates the variance and mean over the dimensions specified by dim.
    Syntactic sugar around `Tensor.var` and `Tensor.mean` to match `torch.var_mean`.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    var, mean = t.var_mean()
    print(var.numpy(), mean.numpy())
    ```
    """
    return self.var(axis, keepdim, correction), self.mean(axis, keepdim)

  def std(self, axis:int|Sequence[int]|None=None, keepdim=False, correction=1) -> Tensor:
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

  def std_mean(self, axis:int|Sequence[int]|None=None, keepdim=False, correction=1) -> tuple[Tensor, Tensor]:
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

  def keccak(self, cfg:str|tuple[int, int]="sha3_256"):
    """
    Calculates a Keccak hash over the last dimension. Uses "sha3_256" by default.

    ```python exec="false" source="above" session="tensor" result="python"
    t = Tensor(b"Hello World!").keccak()
    print(t.data().hex())
    ```
    """

    # https://keccak.team/keccak_specs_summary.html

    def ctensor(l: Sequence[ConstType], dtype: DType = dtypes.uint64):
      # TODO: contiguous is here for compile speed
      return Tensor.stack(*(Tensor(v, dtype=dtype, device=self.device) for v in l)).contiguous()
    rot_offsets = [44, 43, 21, 14, 28, 20, 3, 45, 61, 1, 6, 25, 8, 18, 27, 36, 10, 15, 56, 62, 55, 39, 41, 2]
    rot_offsets_v0, rot_offsets_v1 =  ctensor([0] + [1 << v for v in rot_offsets]), ctensor([1] + [1 << (64 - v) for v in rot_offsets])

    # calculated from π step
    reorder_indexes = ctensor([0,6,12,18,24,3,9,10,16,22,1,7,13,19,20,4,5,11,17,23,2,8,14,15,21], dtype=dtypes.int32)
    rnd_const_masks = [ctensor([v]).pad((0, 24)) for v in (1, 0x8082, 0x800000000000808a, 0x8000000080008000, 0x808b, 0x80000001, 0x8000000080008081,
    0x8000000000008009, 0x8a, 0x88, 0x80008009, 0x8000000a, 0x8000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x800a, 0x800000008000000a, 0x8000000080008081, 0x8000000000008080, 0x80000001, 0x8000000080008008)]

    rate, dsbyte = {"sha3_224": (144, 6), "sha3_256": (136, 6), "shake_128": (168, 31)}[cfg] if isinstance(cfg, str) else cfg
    data, data_pad = self.bitcast(dtypes.uint8).reshape(prod(self.shape[:-1]), self.shape[-1]), rate - (self.shape[-1] * self.dtype.itemsize % rate)
    # pad batches then pad blocks
    data = data.pad((None, (0, data_pad))).reshape(bs := data.shape[0], -1, rate).pad((None, None, (0, 200 - rate)))

    # create pad mask
    lbe = prod(data.shape[1:]) + rate - data_pad - 200
    if data_pad == 1: mb = [(lbe, 0), (1, dsbyte ^ 0x80), (200 - rate, 0)]
    else: mb = [(lbe, 0), (1, dsbyte), (data_pad - 2, 0), (1, 0x80), (200 - rate, 0)]
    pad_mask = Tensor.cat(*(Tensor(v, dtype=dtypes.uint8, device=data.device).expand(l) for l, v in mb if l > 0)).unsqueeze(0)

    data = (data.flatten(1) ^ pad_mask).reshape(*data.shape[:2], 200).bitcast(dtypes.uint64)

    state = Tensor.zeros(bs, 25, device=self.device, dtype=dtypes.uint64)
    for k in range(int(data.shape[1])):
      state = state.bitwise_xor(data[:,k].reshape(bs, 25))
      for i in range(24): # f1600
        # θ step
        p = state.reshape(bs, 5, 5).transpose(2, 1)
        t1 = (p[:,:,0] ^ p[:,:,1] ^ p[:,:,2] ^ p[:,:,3] ^ p[:,:,4]).roll(-1, 1) # xor reduce
        state = state ^ (t1.roll(2, 1).bitwise_xor((t1 << 1) ^ (t1 >> 63)).unsqueeze(2).expand(bs, 5, 5).transpose(2, 1).flatten(1))
        # ρ and π steps
        state = state[:, reorder_indexes]
        state = (state * rot_offsets_v0).bitwise_or(state // rot_offsets_v1).reshape(bs, 5, 5)
        # χ and ι step
        state = state.bitwise_xor(~state.roll(shifts=-1, dims=2) & state.roll(shifts=-2, dims=2))
        state = state.flatten(1) ^ rnd_const_masks[i]
      # NOTE: kernelize here to prevent internal stack from growing propotional to data size
      state = state.kernelize()
    return state.bitcast(dtypes.uint8)[:,:(obytes:=(200 - rate) // 2)].reshape(*self.shape[:-1], obytes)

  def _hash_1mb(self) -> Tensor:
    assert self.dtype == dtypes.uint8, "only support uint8 tensors for hashing"
    assert self.ndim == 2, "only support batched 1d tensors"
    assert self.shape[1] == 1024 * 1024, "only support messages of 1mb"

    blocks = self.shape[0] * self.shape[1] // 4096
    data = self.reshape(blocks, 4096)
    block_hashes = data.keccak("shake_128").reshape(self.shape[0], 4096)
    return block_hashes.keccak("shake_128").reshape(self.shape[0], 16)

  def hash(self) -> Tensor:
    """
    Calculates a 16-byte hash of the tensor.
    ```python exec="false source="above" session="tensor" result="python"
    t = Tensor(b"Hello World!").hash()
    print(t.data().hex())
    ```
    """

    data = self.flatten().bitcast(dtypes.uint8)
    if (tsize := data.shape[0]) % 2**20 != 0: data = data.pad((0, 2**20 - tsize % 2**20))
    base_chunks = ceildiv(data.shape[0], 2**20)
    tree_depth = math.ceil(math.log(base_chunks, 65536)) if base_chunks > 1 else 0

    level_chunks = base_chunks
    for _ in range(tree_depth + 1):
      data = data.reshape(level_chunks, 2**20)._hash_1mb().flatten()
      if (tsize := data.shape[0]) % 2**20 != 0: data = data.pad((0, 2**20 - tsize % 2**20))
      level_chunks = ceildiv(data.shape[0], 2**20)

    return data[:16]

  def _softmax(self, axis, dtype:DTypeLike|None=None) -> tuple[Tensor, Tensor, Tensor]:
    m = self - self.max(axis=axis, keepdim=True).detach()
    if dtype is not None: m = m.cast(dtype)
    e = m.exp()
    return m, e, e.sum(axis=axis, keepdim=True)

  def softmax(self, axis=-1, dtype:DTypeLike|None=None, _single_kernel=getenv("SINGLE_KERNEL_SOFTMAX")) -> Tensor:
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
    if _single_kernel:
      _, e, ss = self.contiguous()._softmax(axis, dtype)
      return e.div(ss).fuse()
    _, e, ss = self._softmax(axis, dtype)
    return e.div(ss)

  def log_softmax(self, axis=-1, dtype:DTypeLike|None=None) -> Tensor:
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

  def logsumexp(self, axis=None, keepdim=False) -> Tensor:
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

  def logcumsumexp(self, axis=0) -> Tensor:
    """
    Computes the log-cumsum-exp of the tensor along the specified axis or axes.

    The log-cumsum-exp function is a numerically stable way to compute the logarithm of the cumulative sum of exponentials.

    You can pass in the `axis` keyword argument to control the axis along which
    the log-cumsum-exp is computed.

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
    if self.ndim == 0: return self
    x = self.transpose(axis, -1)
    last_dim_size = x.shape[-1]
    x_unsqueezed = x.unsqueeze(-2).expand((None,)*(self.ndim-1)+(last_dim_size, None))
    x_cummax = x.cummax(-1)
    mask = Tensor.ones(last_dim_size, last_dim_size, requires_grad=False, device=self.device).tril()
    ret = mask.where(x_unsqueezed - x_cummax.unsqueeze(-1), dtypes.min(self.dtype)).exp().sum(-1).log() + x_cummax
    return ret.transpose(-1, axis)

  def argmax(self, axis=None, keepdim=False) -> Tensor:
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

  def argmin(self, axis=None, keepdim=False) -> Tensor:
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
    return self._inverse().argmax(axis=axis, keepdim=keepdim)

  @staticmethod
  def einsum(formula:str, *operands:Tensor|Sequence[Tensor], dtype:DTypeLike|None=None) -> Tensor:
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
        ell_chars, ell_longest = "".join(c for c in string.ascii_letters if c not in formula), 0
        for i, inp in enumerate(filter(lambda x: "..." in x, inputs := formula.split("->")[0].split(","))):
          if (ell_count := max(operands[i].ndim, 1) - (len(inp) - len("..."))) > ell_longest: ell_longest = ell_count
          inputs[i] = inp.replace("...", ell_chars[-ell_count:])
        inputs_str, out_ellipse = ",".join(inputs), ell_chars[-ell_longest:]
        return (inputs_str, formula.split("->")[1].replace("...", out_ellipse)) if "->" in formula else \
          (inputs_str, out_ellipse + ''.join(sorted(c for c in inputs_str if inputs_str.count(c) == 1 and c.isalpha() and c not in out_ellipse)))
      return formula.split("->") if "->" in formula else (formula, ''.join(c for c in sorted(formula) if formula.count(c) == 1 and c.isalpha()))

    xs:tuple[Tensor, ...] = argfix(*operands)
    inputs_str, output = parse_formula(formula, *xs)
    inputs = inputs_str.split(",")
    assert len(xs) == len(inputs), f"number of inputs doesn't match number of operands in formula, expected {len(inputs)}, got {len(xs)}"

    # map the value of each letter in the formula
    letter_val = sorted(merge_dicts([dict(zip(letters, tensor.shape)) for letters, tensor in zip(inputs, xs)]).items())

    xs_:list[Tensor] = []
    lhs = [sorted(enumerate(s), key=lambda e:e[1]) for s in inputs]
    for x,(order,letters) in zip(xs, [list(zip(*l)) for l in lhs]):
      # permute to the sorted letter order, then reshape/expand to create dimensions for the missing letters
      xs_.append(x.permute(order).reshape([val if letter in letters else 1 for letter,val in letter_val]).expand([val for _,val in letter_val]))

    # ordinal encode the output alphabet
    rhs_order = argsort(argsort(list(output)))

    # sum over all axes that's not in the output, then permute to the output order
    return functools.reduce(lambda a,b:a*b, xs_) \
      .sum(axis=[axis for axis,(letter,_) in enumerate(letter_val) if letter not in output], dtype=dtype).permute(rhs_order)

  # ***** processing ops *****

  def _pool(self, k_:tuple[sint, ...], stride:int|tuple[int, ...]=1, dilation:int|tuple[int, ...]=1) -> Tensor:
    assert len(self.shape) >= len(k_), f"can't pool {self.shape} with {k_}"
    s_, d_ = make_tuple(stride, len(k_)), make_tuple(dilation, len(k_))
    assert len(k_) == len(s_) == len(d_), f"stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}"
    noop, i_ = [None] * (self.ndim-len(k_)), self.shape[-len(k_):]
    assert all(resolve(d*(k-1)+1 <= i) for k,d,i in zip(k_,d_,i_)), "kernel size cannot be greater than actual input size"
    o_ = [ceildiv(i-d*(k-1), s) for i,d,k,s in zip(i_,d_,k_,s_)]
    if any(resolve(k > s) for k,s in zip(k_,s_)) or any(d != 1 for d in d_):
      # input size scaling factor to make sure shrink for stride is possible
      f_ = [1 + int(resolve(o*s > (i - d*(k-1)))) for o,s,i,d,k in zip(o_,s_,i_,d_,k_)]
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

  def _resolve_pool_pads(self, padding:int|Sequence[int], dims:int) -> Sequence[int]:
    if not isinstance(padding, int) and not (len(padding) == 2*dims or len(padding) == dims):
      raise ValueError(f"Padding must be an int or a sequence of length {dims} or {2*dims}, but got {padding=} for {self.shape=} with {dims=}.")
    return [padding]*2*dims if isinstance(padding, int) else (padding if len(padding) == 2*dims else [p for p in padding for _ in range(2)][::-1])

  def _apply_ceil_mode(self, pads:Sequence[int], k_:tuple[sint, ...], s_:int|tuple[int, ...], d_:int|tuple[int, ...]) -> list[int]:
    (d_,s_), i_ = (make_tuple(x, len(k_)) for x in (d_,s_)), self.shape[-len(k_):]
    pads, grouped_pads = list(pads), _flat_to_grouped(pads)
    # https://arxiv.org/pdf/1603.07285 section 5.1, relationship 15.
    o_ = [ceildiv(i+pB+pA - (d*(k-1)+1), s) + 1 for i,d,k,s,(pB,pA) in zip(i_,d_,k_,s_,grouped_pads)]
    for dim,(o,i,s,k,d,(pB,pA)) in enumerate(zip(o_,i_,s_,k_,d_,grouped_pads)):
      # we have to do additional padding before `_pool` so that `o_` in `_pool` is calculated correctly
      # `s*(o-1) + (d*(k-1)+1) - (i+pB+pA)` -> last_sliding_window_start + full_kernel_size - padded_input_shape
      # we decrease padding in the case that a sliding window starts in the end padded region, thereby decreasing `o_` in `_pool`
      # `smax(s*(o-1) - (pB+i-1), 0)` -> last_sliding_window_start - (pad_before + input_size - zero_offset)
      pads[-1-dim*2] += s*(o-1) + (d*(k-1)+1) - (i+pB+pA) - smax(s*(o-1) - (pB+i-1), 0)
    return pads

  # NOTE: these work for more than 2D
  def avg_pool2d(self, kernel_size:tuple[int, ...]=(2,2), stride=None, dilation=1, padding:int|tuple[int, ...]=0,
                 ceil_mode=False, count_include_pad=True) -> Tensor:
    """
    Applies average pooling over a tensor.

    This function supports three different types of `padding`

    1. `int` (single value):
      Applies the same padding value uniformly to all spatial dimensions.

    2. `tuple[int, ...]` (length = number of spatial dimensions):
      Specifies a distinct padding value for each spatial dimension in the form `(padding_height, padding_width, ...)`.

    3. `tuple[int, ...]` (length = 2 * number of spatial dimensions):
      Specifies explicit padding for each side of each spatial dimension in the form
      `(padding_left, padding_right, padding_top, padding_bottom, ...)`.

    When `ceil_mode` is set to `True`, output shape will be determined using ceil division.
    When `count_include_pad` is set to `False`, zero padding will not be included in the averaging calculation.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d pooling and instead works for any number of dimensions.

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
    def pool(x:Tensor, padding_:Sequence[int]) -> Tensor: return x.pad(padding_)._pool(k_, stride if stride is not None else k_, dilation)
    reg_pads = self._resolve_pool_pads(padding, len(k_))
    ceil_pads = self._apply_ceil_mode(reg_pads, k_, stride if stride is not None else k_, dilation)
    if not count_include_pad:
      pads = ceil_pads if ceil_mode else reg_pads
      return pool(self, pads).sum(axis) / pool(self.ones_like(), pads).sum(axis)
    if not ceil_mode: return pool(self, reg_pads).mean(axis)
    return pool(self, ceil_pads).sum(axis) / pool(self.pad(reg_pads).ones_like(), tuple(cp-rp for cp,rp in zip(ceil_pads, reg_pads))).sum(axis)

  def max_pool2d(self, kernel_size:tuple[int, ...]=(2,2), stride=None, dilation=1, padding:int|tuple[int, ...]=0,
                 ceil_mode=False, return_indices=False) -> Tensor | tuple[Tensor, Tensor]:
    """
    Applies max pooling over a tensor.

    This function supports three different types of `padding`

    1. `int` (single value):
      Applies the same padding value uniformly to all spatial dimensions.

    2. `tuple[int, ...]` (length = number of spatial dimensions):
      Specifies a distinct padding value for each spatial dimension in the form `(padding_height, padding_width, ...)`.

    3. `tuple[int, ...]` (length = 2 * number of spatial dimensions):
      Specifies explicit padding for each side of each spatial dimension in the form
      `(padding_left, padding_right, padding_top, padding_bottom, ...)`.

    When `ceil_mode` is set to `True`, output shape will be determined using ceil division.
    When `return_indices` is set to `True`, the argmax will be returned along with the max values.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d pooling and instead works for any number of dimensions.

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
    axis = tuple(range(-len(k_ := make_tuple(kernel_size, 2)), 0))
    pads = self._resolve_pool_pads(padding, len(k_))
    if ceil_mode: pads = self._apply_ceil_mode(pads, k_, stride if stride is not None else k_, dilation)
    pooled = self.pad(pads, value=dtypes.min(self.dtype))._pool(k_, stride if stride is not None else k_, dilation)
    if not return_indices: return pooled.max(axis)
    spatial_sz = math.prod(spatial_shape := self.shape[-len(k_):])
    idx = Tensor.arange(spatial_sz,0,-1, requires_grad=False, device=self.device).reshape(spatial_shape)
    m = pooled == pooled.max(axis, keepdim=True)
    idx = m * idx.pad(pads, value=dtypes.min(idx.dtype))._pool(k_, stride if stride is not None else k_, dilation)
    return pooled.max(axis), spatial_sz - idx.max(axis)

  def max_unpool2d(self, indices:Tensor, kernel_size:tuple[int, ...]=(2,2), stride=None, dilation=1, padding:int|tuple[int, ...]=0, output_size=None):
    """
    Performs a partial inverse of `max_pool2d` using the indices from the argmax.

    When `output_size` is provided, the output shape disambiguates to the provided shape.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d pooling and instead works for any number of dimensions.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(1, 17).reshape(1, 1, 4, 4)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    output, indices = Tensor.max_pool2d(t, return_indices=True)
    print(output.numpy())
    print(indices.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.max_unpool2d(output, indices).numpy())
    ```
    """
    bs,c,*spatial_shape = self.shape
    if output_size is None:
      k_,d_,s_ = (make_tuple(x, len(spatial_shape)) for x in (kernel_size, dilation, stride if stride is not None else kernel_size))
      p_ = _flat_to_grouped(self._resolve_pool_pads(padding, len(spatial_shape)))
      # https://arxiv.org/pdf/1603.07285 inverse of relationship 15 in section 5.1.
      output_size = tuple((i-1)*s - (pB+pA) + (d*(k-1)+1) for i,k,d,s,(pA,pB) in zip(spatial_shape,k_,d_,s_,p_))
    else: output_size = output_size[-len(spatial_shape):]
    ret = (indices.reshape(bs,c,1,-1)._one_hot_along_dim(prod(output_size), 2) * self.reshape(bs,c,1,-1)).sum(3)
    return ret.reshape(bs,c,*output_size)

  def conv2d(self, weight:Tensor, bias:Tensor|None=None, groups=1, stride=1, dilation=1, padding:int|tuple[int, ...]=0,
             dtype:DTypeLike|None=None) -> Tensor:
    """
    Applies a convolution over a tensor with a given `weight` and optional `bias`.

    This function supports three different types of `padding`

    1. `int` (single value):
      Applies the same padding value uniformly to all spatial dimensions.

    2. `tuple[int, ...]` (length = number of spatial dimensions):
      Specifies a distinct padding value for each spatial dimension in the form `(padding_height, padding_width, ...)`.

    3. `tuple[int, ...]` (length = 2 * number of spatial dimensions):
      Specifies explicit padding for each side of each spatial dimension in the form
      `(padding_left, padding_right, padding_top, padding_bottom, ...)`.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d convolutions and instead works for any number of dimensions.

    See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(1, 1, 3, 3)
    w = Tensor.ones(1, 1, 2, 2)
    print(t.conv2d(w).numpy())
    ```
    """
    if IMAGE: return self.image_conv2d(weight, bias, groups, stride, dilation, padding, dtype)
    (bs,cin_), (cout,cin), HW = self.shape[:2], weight.shape[:2], weight.shape[2:]
    padding_ = self._resolve_pool_pads(padding, len(HW))
    assert groups*cin == cin_ and len(self.shape) == len(weight.shape), f"Input Tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"  # noqa: E501

    # conv2d is a pooling op (with padding)
    x = self.pad(padding_)._pool(HW, stride, dilation)   # (bs, groups*cin, oy, ox, H, W)
    rcout, oyx = cout//groups, x.shape[2:-len(HW)]
    if not all(x == 3 for x in HW) or stride != 1 or dilation != 1 or not WINO:
      # normal conv
      x = x.reshape(bs, groups, cin, 1, *oyx, *HW).expand(bs, groups, cin, rcout, *oyx, *HW).permute(0,1,3,*[4+i for i in range(len(oyx))],2,*[4+len(oyx)+i for i in range(len(HW))])  # noqa: E501

      # conv! broadcasted to (bs, groups, rcout, *oyx, cin, *HW)
      ret = (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW)).sum([-1-i for i in range(1+len(oyx))], keepdim=True, dtype=dtype).reshape(bs, cout, *oyx)  # noqa: E501
      return ret if bias is None else ret.add(bias.reshape(1, -1, *[1] * len(HW)))

    HWI, HWO = (6,) * len(HW), (4,) * len(HW)  # F(4x4,3x3) winograd tiles
    winograd_G = [[1/4, 0, 0], [-1/6, -1/6, -1/6], [-1/6, 1/6, -1/6], [1/24, 1/12, 1/6], [1/24, -1/12, 1/6], [0, 0, 1]]
    winograd_Bt = [[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0], [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]]
    winograd_At = [[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]] # applying At in pre-order doubles compile time

    # TODO: stride == dilation
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
    ret = _apply_winograd_matrix(winograd_At, (gfactors * dfactors).sum(axis=-1-len(HW), dtype=dtype), len(HW))

    # interleave tyx and HWO: (bs, groups, rcout, oy, HO, ox, WO)
    ret = ret.permute([*range(len(HW), len(ret.shape)-len(HW)), *[i+o for i in range(len(HW)) for o in [len(ret.shape)-len(HW),0]]])
    # merge groups and rcout, tyx and HWO: (bs, groups, cout, *yx), shrink to final
    ret = ret.reshape(bs, cout, *[c * HWO[i] for i, c in enumerate(tyx)]).shrink(tuple((0, s) for s in [bs, cout, *oyx]))

    return (ret if bias is None else ret.add(bias.reshape(1, -1, *[1 for _ in range(len(HW))]))).contiguous().contiguous_backward()

  def conv_transpose2d(self, weight:Tensor, bias:Tensor|None=None, groups=1, stride=1, dilation=1, padding=0, output_padding=0) -> Tensor:
    """
    Applies a transposed convolution over a tensor with a given `weight` and optional `bias`.

    This function supports three different types of `padding`

    1. `int` (single value):
      Applies the same padding value uniformly to all spatial dimensions.

    2. `tuple[int, ...]` (length = number of spatial dimensions):
      Specifies a distinct padding value for each spatial dimension in the form `(padding_height, padding_width, ...)`.

    3. `tuple[int, ...]` (length = 2 * number of spatial dimensions):
      Specifies explicit padding for each side of each spatial dimension in the form
      `(padding_left, padding_right, padding_top, padding_bottom, ...)`.

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
    padding = _flat_to_grouped(self._resolve_pool_pads(padding, len(HW)))
    stride, dilation, output_padding = [make_tuple(x, len(HW)) for x in (stride, dilation, output_padding)]
    if any(s>1 for s in stride):
      # handle strides: (k) -> reshape -> (k,1) -> pad -> (k,s) -> reshape -> (k*s) -> shrink (k-(s-1))
      x = x.reshape(None, None, *flatten((k,1) for k in x.shape[2:]))
      x = x.pad((None, None, *flatten((None,(0,s-1)) for s in stride)))
      x = x.reshape(None, None, *[k*s for k,s in zip(x.shape[2::2], stride)])
      x = x.shrink((None, None, *[(0,k-(s-1)) for k,s in zip(x.shape[2:], stride)]))
    padding = flatten((((k-1)*d-pB,(k-1)*d-pA+op) for k,d,(pB,pA),op in reversed(list(zip(HW, dilation, padding, output_padding)))))
    return x.conv2d(w.flatten(end_dim=1), groups=groups, bias=bias, dilation=dilation, padding=padding)

  def dot(self, w:Tensor, dtype:DTypeLike|None=None) -> Tensor:

    """
    Performs dot product between two tensors.
    If `w` is 1-D, it's a sum product over the last axis of `self` and `w`.
    If `w` is N-D with N>=2, it's a sum product over the last axis of `self` and the second-to-last axis of `w`.

    You can pass in the optional `dtype` keyword argument to control the data type of the accumulation.

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
    if IMAGE: return self.image_dot(w, dtype)
    x, dx, dw = self, self.ndim, w.ndim
    if not (dx > 0 and dw > 0): raise RuntimeError(f"both tensors need to be at least 1D, got {dx}D and {dw}D")
    if x.shape[-1] != w.shape[axis_w:=-min(w.ndim,2)]: raise RuntimeError(f"cannot dot {x.shape} and {w.shape}")
    x = x.reshape(*x.shape[0:-1], *[1]*min(dx-1, dw-1, 1), x.shape[-1])
    w = w.reshape(*w.shape[0:-2], *[1]*min(dx-1, dw-1, 1), *w.shape[axis_w:]).transpose(-1, axis_w)
    return (x*w).sum(-1, dtype=dtype).cast(least_upper_dtype(x.dtype, w.dtype) if dtype is None else dtype)

  def matmul(self, x:Tensor, reverse=False, dtype:DTypeLike|None=None) -> Tensor:
    """
    Performs matrix multiplication between two tensors.

    You can pass in the `reverse` keyword argument to control the order of the matrix multiplication.
    You can pass in the optional `dtype` keyword argument to control the data type of the accumulation.

    ```python exec="true" source="above" session="tensor" result="python"
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    print(a.matmul(b).numpy())
    ```
    """
    return x.dot(self, dtype=dtype) if reverse else self.dot(x, dtype=dtype)

  def _cumalu(self, axis:int, op:Ops, _include_initial=False) -> Tensor:
    assert self.shape[axis] != 0 and op in (Ops.ADD, Ops.MAX, Ops.MUL)
    pl_sz = self.shape[axis] - int(not _include_initial)
    pooled = self.transpose(axis,-1).pad((pl_sz, -int(_include_initial)), value=identity_element(op, self.dtype))._pool((self.shape[axis],))
    return {Ops.ADD: pooled.sum(-1), Ops.MAX: pooled.max(-1), Ops.MUL: pooled.prod(-1)}[op].transpose(axis, -1)

  def _split_cumalu(self, axis:int, op:Ops) -> Tensor:
    axis = self._resolve_dim(axis)
    if self.ndim == 0 or 0 in self.shape: return self
    # TODO: someday the optimizer will find this on its own
    # for now this is a two stage cumsum
    SPLIT = 256
    if not isinstance(s:=self.shape[axis], int) or s <= SPLIT*2: return self._cumalu(axis, op)
    ret = self.transpose(axis,-1).pad((round_up(s, SPLIT)-s, 0), value=identity_element(op, self.dtype)).unflatten(-1, (-1, SPLIT))._cumalu(-1, op)
    base = ret[..., -1]._cumalu(-1, op, _include_initial=True)
    base = base.unsqueeze(-1).expand(*base.shape, ret.shape[-1])
    def fix(x: Tensor) -> Tensor: return x.flatten(start_dim=-2)[..., -s:].transpose(axis,-1)
    return {Ops.ADD: Tensor.__add__, Ops.MAX: Tensor.maximum, Ops.MUL: Tensor.__mul__}[op](fix(ret), fix(base))

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

  def cumprod(self, axis:int) -> Tensor:
    """
    Computes the cumulative product of the elements of the tensor along the specified `axis`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(1, 7).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.cumprod(axis=0).numpy())
    ```
    """
    return self._split_cumalu(axis, Ops.MUL)

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

  def interpolate(self, size:tuple[int, ...], mode:str="linear", align_corners:bool=False) -> Tensor:
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
        low, high, perc = [y.reshape(reshape).expand(expand) for y in (index.floor().int(), index.ceil().int(), index - index.floor())]
        x = x.gather(i, low).lerp(x.gather(i, high), perc)
      else:
        index = (scale*(arr+0.5) if mode=="nearest-exact" else scale*arr).cast(dtypes.int32).reshape(reshape).expand(expand)
        x = x.gather(i, index)
    return x.cast(self.dtype)

  def _pre_scatter(self, dim:int, index:Tensor, src:Tensor) -> tuple[Tensor, Tensor]:
    index, dim = index.to(self.device), self._resolve_dim(dim)
    assert index.ndim == self.ndim == src.ndim, f"self.ndim, index.ndim and src.ndim must all equal, {self.ndim=} {index.ndim=} {src.ndim=}"
    assert all((d == dim or self_ >= index_) and src_ >= index_ for d,(self_,index_,src_) in enumerate(zip(self.shape, index.shape, src.shape))), \
      f"All dimensions of {index.shape=} should be <= to all dimensions of {src.shape=} and all dimensions except dimension {dim} of {self.shape=}"
    if self.dtype != src.dtype: raise RuntimeError(f"expect {self.dtype=} to be equal to {src.dtype=}")
    # shrink src to index shape to shrink away the unused values
    src = src.shrink(tuple((0,s) for s in index.shape))
    # prepare src and mask for reduce with respect to dim
    src = src.unsqueeze(-1).expand(*src.shape, self.shape[dim]).transpose(-1, dim)
    mask = index.unsqueeze(-1)._one_hot_along_dim(self.shape[dim]).transpose(-1, dim)
    # pad src and mask to self.shape so that reduce can be done with padded values as no-ops
    src, mask = (x.pad(tuple((0, self.shape[i] - x.shape[i]) if i != dim else None for i in range(self.ndim)) + (None,)) for x in (src, mask))
    return src, mask

  def scatter(self, dim:int, index:Tensor, src:Tensor|ConstType, reduce:Literal['multiply', 'add']|None=None) -> Tensor:
    """
    Scatters `src` values along an axis specified by `dim`.
    Apply `add` or `multiply` reduction operation with `reduce`.

    NOTE: To use the `reduce` argument with a Tensor `src`, see `Tensor.scatter_reduce`.

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
    if reduce not in {None, "add", "multiply"}: raise TypeError(f"{reduce=} must be one of None, 'multiply', or 'add'")
    if reduce and isinstance(src, Tensor): raise TypeError("Tensor src is not supported with reduce arg. see scatter_reduce")
    if not isinstance(src, Tensor): src = index.full_like(src, device=self.device, dtype=self.dtype)
    if reduce == "add": return self.scatter_reduce(dim, index, src, "sum", include_self=True)
    if reduce == "multiply": return self.scatter_reduce(dim, index, src, "prod", include_self=True)
    src, mask = self._pre_scatter(dim, index, src)
    return _masked_setitem(self, src, mask, (-1,))

  def scatter_reduce(self, dim:int, index:Tensor, src:Tensor, reduce:Literal["sum", "prod", "mean", "amax", "amin"],
                     include_self:bool=True) -> Tensor:
    """
    Scatters `src` values along an axis specified by `dim`.
    Apply `"sum"`, `"prod"`, `"mean"`, `"amax"`, or `"amin"` reduction operations with `reduce`.

    Set `include_self=False` to exclude values in the `self` Tensor from the reduction.

    ```python exec="true" source="above" session="tensor" result="python"
    src = Tensor.arange(1, 11).cast(dtypes.float).reshape(2, 5)
    print(src.numpy())
    index = Tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    print(index.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(1, 5, dtype=src.dtype).scatter_reduce(0, index, src, reduce='sum').numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(1, 5, dtype=src.dtype).scatter_reduce(0, index, src, reduce='prod').numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(1, 5, dtype=src.dtype).scatter_reduce(0, index, src, reduce='mean', include_self=False).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([[-10, 20, 0, 5, 10]], dtype=src.dtype).scatter_reduce(0, index, src, reduce='amax').numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([[-10, 20, 0, 5, 10]], dtype=src.dtype).scatter_reduce(0, index, src, reduce='amin').numpy())
    ```
    """
    src, mask = self._pre_scatter(dim, index, src)
    def _inv_mask(a:Tensor|ConstType, b:Tensor|ConstType) -> Tensor: return mask.any(-1).logical_not().where(a, b)
    if reduce == "sum": return mask.where(src, 0).sum(-1).add(self if include_self else _inv_mask(self, 0))
    if reduce == "prod": return mask.where(src, 1).prod(-1).mul(self if include_self else _inv_mask(self, 1))
    if reduce == "amax": return mask.where(src, m := dtypes.min(src.dtype)).max(-1).maximum(self if include_self else _inv_mask(self, m))
    if reduce == "amin": return mask.where(src, m := dtypes.max(src.dtype)).min(-1).minimum(self if include_self else _inv_mask(self, m))
    if reduce == "mean":
      count = mask.where(1, 0).sum(-1).add(1 if include_self else _inv_mask(1, 0))
      return mask.where(src, 0).sum(-1).add(self if include_self else _inv_mask(self, 0)).div(count)
    raise RuntimeError(f"{reduce=} must be one of 'sum', 'prod', 'mean', 'amax', 'amin'")

  def sort(self, dim:int=-1, descending:bool=False) -> tuple[Tensor, Tensor]:
    """
    Performs a bitonic sort on the tensor along the specified dimension.

    Order of indices for equivalent elements is always preserved.

    See: https://en.wikipedia.org/wiki/Bitonic_sorter

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[0.1, 0.5, 1.2, 3.4, 2.1], [2.2, 1.9, 0.3, 4.5, 0.8]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    sorted_values, indices = t.sort(dim=1, descending=True)
    print(sorted_values.numpy())
    print(indices.numpy())
    ```
    """
    x, dim = self, self._resolve_dim(dim)
    if (orig_len:= x.shape[dim]) <= 1: return x, x.zeros_like(dtype=dtypes.default_int)
    # pad to power of 2
    n_stages = (orig_len-1).bit_length()
    pads = tuple((0, 2**n_stages - orig_len) if i == dim else None for i in range(x.ndim))
    x = x.pad(pads, value=dtypes.min(x.dtype) if descending else dtypes.max(x.dtype)).unflatten(dim, (2,)*n_stages)
    # https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort1.svg
    for stage in range(1, n_stages+1):
      if stage != n_stages:
        # flip so arrows of green boxes point the same way as blue boxes
        crossover_dim = dim + n_stages - stage - 1
        blue_box, green_box = x.split(1, crossover_dim)
        flip_dims = tuple(-i for i in range(1, stage+1+(self.ndim-dim)))
        x = (blue_box.cat(green_box.flip(flip_dims), dim=crossover_dim)).contiguous()
      for substage in range(stage-1, -1, -1):
        partner_dim = dim + n_stages - substage - 1
        x_top, x_bottom = x.split(1, partner_dim)
        x_larger, x_smaller = x_top.maximum(x_bottom), x_top.minimum(x_bottom)
        x = (x_larger.cat(x_smaller, dim=partner_dim) if descending else x_smaller.cat(x_larger, dim=partner_dim)).contiguous()
      if stage != n_stages:
        # flip wires back to undo the crossover
        blue_box, flipped_green_box = x.split(1, crossover_dim)
        x = blue_box.cat(flipped_green_box.flip(flip_dims), dim=crossover_dim)
    x = x.flatten(dim, dim+n_stages-1).shrink(tuple((0, s) for s in self.shape))
    # compute indices for sorted values
    idx = Tensor.arange(orig_len, requires_grad=False, device=self.device).reshape(tuple(orig_len if i == dim else 1 for i in range(x.ndim)))
    idx = idx.expand(x.shape)
    def compute_counts(t:Tensor): return ((idx.unsqueeze(dim) <= idx.unsqueeze(dim+1)) & (t.unsqueeze(dim) == t.unsqueeze(dim+1))).sum(dim+1)
    count_orig, count_sorted = compute_counts(self), compute_counts(x)
    cond = (self.unsqueeze(dim+1) == x.unsqueeze(dim)) & (count_orig.unsqueeze(dim+1) == count_sorted.unsqueeze(dim))
    idx = (cond * idx.unsqueeze(dim+1)).sum(dim)
    return x, idx

  def argsort(self, dim:int=-1, descending:bool=False) -> Tensor:
    """
    Returns the indices that sort input tensor along given `dimension` in given `descending` order by value.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[2, 3, 4, 1], [1, 4, 3, 2]])
    print(t.argsort().numpy())
    ```
    """
    return self.sort(dim, descending)[1]

  def topk(self, k:int, dim:int=-1, largest:bool=True, sorted_:bool=True) -> tuple[Tensor, Tensor]:
    """
    Computes the top-k elements of the tensor along the specified `dim`.

    Order of indices for equivalent elements is always preserved.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[0.1, 0.5, 1.2, 3.4, 2.1], [2.2, 1.9, 0.3, 4.5, 0.8]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    topk_values, topk_indices = t.topk(2, dim=1)
    print(topk_values.numpy())
    print(topk_indices.numpy())
    ```
    """
    if not sorted_: raise NotImplementedError("topk with sorted_=False is not supported")
    if k > self.shape[dim:=self._resolve_dim(dim)]: raise ValueError(f"selected index {k=} is out of range")
    x, idx = self.sort(dim, descending=largest)
    shrink_to_k = tuple((0, k) if i == dim else None for i in range(self.ndim))
    return x.shrink(shrink_to_k), idx.shrink(shrink_to_k)

  # ***** unary ops *****

  def logical_not(self) -> Tensor:
    """
    Computes the logical NOT of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([False, True]).logical_not().numpy())
    ```
    """
    return self.cast(dtypes.bool)._apply_broadcasted_uop(UOp.ne, True)

  def neg(self) -> Tensor:
    """
    Negates the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).neg().numpy())
    ```
    """
    return self*-1 if self.dtype != dtypes.bool else self.logical_not()

  def contiguous(self) -> Tensor:
    """
    Returns a contiguous tensor.
    """
    return self._apply_uop(UOp.contiguous)

  def fuse(self) -> Tensor:
    """
    Makes this a single kernel back to Ops.CONTIGUOUS on the inputs.

    Useful for single kernel softmax and flash attention.
    Careful, this can break codegen or make kernels really slow.
    """
    return self._apply_uop(UOp.fuse)

  def contiguous_backward(self) -> Tensor:
    """
    Inserts a contiguous operation in the backward pass.
    """
    return self._apply_uop(UOp.contiguous_backward)

  def log(self) -> Tensor:
    """
    Computes the natural logarithm element-wise.

    See: https://en.wikipedia.org/wiki/Logarithm

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 4., 8.]).log().numpy())
    ```
    """
    return self.log2()*math.log(2)

  def log2(self) -> Tensor:
    """
    Computes the base-2 logarithm element-wise.

    See: https://en.wikipedia.org/wiki/Logarithm

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 4., 8.]).log2().numpy())
    ```
    """
    return self.cast(least_upper_float(self.dtype))._apply_uop(UOp.log2)

  def exp(self) -> Tensor:
    """
    Computes the exponential function element-wise.

    See: https://en.wikipedia.org/wiki/Exponential_function

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., 1., 2., 3.]).exp().numpy())
    ```
    """
    return self.mul(1/math.log(2)).exp2()

  def exp2(self) -> Tensor:
    """
    Computes the base-2 exponential function element-wise.

    See: https://en.wikipedia.org/wiki/Exponential_function

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., 1., 2., 3.]).exp2().numpy())
    ```
    """
    return self.cast(least_upper_float(self.dtype))._apply_uop(UOp.exp2)

  def relu(self) -> Tensor:
    """
    Applies the Rectified Linear Unit (ReLU) function element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).relu().numpy())
    ```
    """
    # NOTE: if you write this as self.maximum(0) the gradient is wrong, passing through half when self is 0
    return (self>0).where(self, 0)

  def sigmoid(self) -> Tensor:
    """
    Applies the Sigmoid function element-wise.

    - Described: https://en.wikipedia.org/wiki/Sigmoid_function

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sigmoid().numpy())
    ```
    """
    return (1 + (self * (-1/math.log(2))).exp2()).reciprocal()

  def logsigmoid(self) -> Tensor:
    """
    Applies the LogSigmoid function element-wise.

    - See: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.logsigmoid.html

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).logsigmoid().numpy())
    ```
    """
    return -(-self).softplus()

  def hardsigmoid(self, alpha:float=1/6, beta:float=0.5) -> Tensor:
    """
    Applies the Hardsigmoid function element-wise.
    NOTE: default `alpha` and `beta` values are taken from torch

    - See: https://pytorch.org/docs/stable/generated/torch.nn.functional.hardsigmoid.html

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).hardsigmoid().numpy())
    ```
    """
    return (alpha * self + beta).relu() - (alpha * self + beta - 1).relu()

  def sqrt(self) -> Tensor:
    """
    Computes the square root of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 3., 4.]).sqrt().numpy())
    ```
    """
    return self.cast(least_upper_float(self.dtype))._apply_uop(UOp.sqrt)

  def rsqrt(self) -> Tensor:
    """
    Computes the reciprocal of the square root of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 3., 4.]).rsqrt().numpy())
    ```
    """
    return self.sqrt().reciprocal()

  def sin(self) -> Tensor:
    """
    Computes the sine of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).sin().numpy())
    ```
    """
    return self.cast(least_upper_float(self.dtype))._apply_uop(UOp.sin)

  def cos(self) -> Tensor:
    """
    Computes the cosine of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).cos().numpy())
    ```
    """
    return ((math.pi/2)-self).sin()

  def tan(self) -> Tensor:
    """
    Computes the tangent of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., math.pi/4, math.pi/2, 3*math.pi/4, math.pi]).tan().numpy())
    ```
    """
    return self.sin() / self.cos()

  def asin(self) -> Tensor:
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

  def acos(self) -> Tensor:
    """
    Computes the inverse cosine (arccosine) of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).acos().numpy())
    ```
    """
    return math.pi / 2 - self.asin()

  def atan(self) -> Tensor:
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

  def isinf(self:Tensor, detect_positive:bool=True, detect_negative:bool=True) -> Tensor:
    """
    Checks the tensor element-wise to return True where the element is infinity, otherwise returns False

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isinf().numpy())
    ```
    """
    return (self == float("inf")) * detect_positive + (self == float("-inf")) * detect_negative

  def isnan(self:Tensor) -> Tensor:
    """
    Checks the tensor element-wise to return True where the element is NaN, otherwise returns False

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isnan().numpy())
    ```
    """
    return self != self

  def isfinite(self:Tensor) -> Tensor:
    """
    Checks the tensor element-wise to return True where the element is finite, otherwise returns False

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isfinite().numpy())
    ```
    """
    return (self.isinf()|self.isnan()).logical_not()

  def lerp(self, end:Tensor, weight:Tensor|float) -> Tensor:
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

  def square(self) -> Tensor:
    """
    Squares the tensor element-wise.
    Equivalent to `self*self`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).square().numpy())
    ```
    """
    return self*self

  def clamp(self, min_=None, max_=None) -> Tensor:
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

  def clip(self, min_=None, max_=None) -> Tensor:
    """
    Alias for `Tensor.clamp`.
    """
    return self.clamp(min_, max_)

  def sign(self) -> Tensor:
    """
    Returns the sign of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sign().numpy())
    ```
    """
    return self.ne(0).where((self<0).where(self.full_like(-1), self.full_like(1)), self.full_like(0)) + self*0

  def abs(self) -> Tensor:
    """
    Computes the absolute value of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).abs().numpy())
    ```
    """
    return self * self.sign()

  def reciprocal(self) -> Tensor:
    """
    Computes `1/x` element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 3., 4.]).reciprocal().numpy())
    ```
    """
    return self.cast(least_upper_float(self.dtype))._apply_uop(UOp.reciprocal)

  # ***** activation functions *****

  def elu(self, alpha=1.0) -> Tensor:
    """
    Applies the Exponential Linear Unit (ELU) function element-wise.

    - Paper: https://arxiv.org/abs/1511.07289v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).elu().numpy())
    ```
    """
    return self.relu() - alpha*(1-self.exp()).relu()

  def celu(self, alpha=1.0) -> Tensor:
    """
    Applies the Continuously differentiable Exponential Linear Unit (CELU) function element-wise.

    - Paper: https://arxiv.org/abs/1704.07483

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).celu().numpy())
    ```
    """
    return self.maximum(0) + (alpha * ((self / alpha).exp() - 1)).minimum(0)

  def selu(self, alpha=1.67326, gamma=1.0507) -> Tensor:
    """
    Applies the Scaled Exponential Linear Unit (SELU) function element-wise.

    - Paper: https://arxiv.org/abs/1706.02515v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).selu().numpy())
    ```
    """
    return gamma * (self >= 0).detach().where(self, alpha * (self.exp() - 1))

  def swish(self) -> Tensor:
    """
    See `.silu()`

    - Paper: https://arxiv.org/abs/1710.05941v1

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).swish().numpy())
    ```
    """
    return self * self.sigmoid()

  def silu(self) -> Tensor:
    """
    Applies the Sigmoid Linear Unit (SiLU) function element-wise.

    - Paper: https://arxiv.org/abs/1606.08415

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).silu().numpy())
    ```
    """
    return self.swish()   # The SiLU function is also known as the swish function.

  def relu6(self) -> Tensor:
    """
    Applies the ReLU6 function element-wise.

    - Paper: https://arxiv.org/abs/1704.04861v1

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-9., -6., -3., 0., 3., 6., 9.]).relu6().numpy())
    ```
    """
    return self.relu() - (self-6).relu()

  def hardswish(self) -> Tensor:
    """
    Applies the Hardswish function element-wise.

    - Paper: https://arxiv.org/abs/1905.02244v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).hardswish().numpy())
    ```
    """
    return self * (self+3).relu6() * (1/6)

  def tanh(self) -> Tensor:
    """
    Applies the Hyperbolic Tangent (tanh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Tanh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).tanh().numpy())
    ```
    """
    return 2.0 * ((2.0 * self).sigmoid()) - 1.0

  def sinh(self) -> Tensor:
    """
    Applies the Hyperbolic Sine (sinh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Sinh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sinh().numpy())
    ```
    """
    return (self.exp() - self.neg().exp()) / 2

  def cosh(self) -> Tensor:
    """
    Applies the Hyperbolic Cosine (cosh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Cosh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).cosh().numpy())
    ```
    """
    return (self.exp() + self.neg().exp()) / 2

  def atanh(self) -> Tensor:
    """
    Applies the Inverse Hyperbolic Tangent (atanh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#atanh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).atanh().numpy())
    ```
    """
    return ((1 + self)/(1 - self)).log() / 2

  def asinh(self) -> Tensor:
    """
    Applies the Inverse Hyperbolic Sine (asinh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#asinh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).asinh().numpy())
    ```
    """
    return (self + (self.square() + 1).sqrt()).log()

  def acosh(self) -> Tensor:
    """
    Applies the Inverse Hyperbolic Cosine (acosh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#acosh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).acosh().numpy())
    ```
    """
    return (self + (self.square() - 1).sqrt()).log()

  def hardtanh(self, min_val=-1, max_val=1) -> Tensor:
    """
    Applies the Hardtanh function element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5]).hardtanh().numpy())
    ```
    """
    return self.clip(min_val, max_val)

  def erf(self) -> Tensor:
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

  def gelu(self) -> Tensor:
    """
    Applies the Gaussian Error Linear Unit (GELU) function element-wise.

    - Paper: https://arxiv.org/abs/1606.08415v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).gelu().numpy())
    ```
    """
    return 0.5 * self * (1 + (math.sqrt(2 / math.pi) * (self + 0.044715 * self ** 3)).tanh())

  def quick_gelu(self) -> Tensor:
    """
    Applies the Sigmoid GELU approximation element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).quick_gelu().numpy())
    ```
    """
    return self * (self * 1.702).sigmoid()

  def leaky_relu(self, neg_slope=0.01) -> Tensor:
    """
    Applies the Leaky ReLU function element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).leaky_relu().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).leaky_relu(neg_slope=0.42).numpy())
    ```
    """
    return (self<0).where(neg_slope*self, self)

  def mish(self) -> Tensor:
    """
    Applies the Mish function element-wise.

    - Paper: https://arxiv.org/abs/1908.08681v3

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).mish().numpy())
    ```
    """
    return self * self.softplus().tanh()

  def softplus(self, beta=1.0, threshold=20.0) -> Tensor:
    """
    Applies the Softplus function element-wise.
    For numerical stability, the implementation folds into identity function when `self * beta > threshold`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).softplus().numpy())
    ```
    """
    return (self * beta > threshold).where(self, (1/beta) * (1 + (self*beta).exp()).log())

  def softsign(self) -> Tensor:
    """
    Applies the Softsign function element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).softsign().numpy())
    ```
    """
    return self / (1 + self.abs())

  # ***** broadcasted elementwise ops *****
  def _broadcast_to(self, new_shape:tuple[sint, ...]) -> Tensor:
    if self.shape == new_shape: return self
    if self.ndim > len(new_shape): raise ValueError(f"cannot broadcast tensor to fewer dimensions. shape={self.shape} to {new_shape=}")
    # first unsqueeze left with 1s https://data-apis.org/array-api/latest/API_specification/broadcasting.html
    shape, _ = _align_left(self.shape, new_shape)
    # for each dimension, check either dim is 1, or it does not change
    if not all(resolve(s == ns) or resolve(s == 1) for s,ns in zip(shape, new_shape)):
      raise ValueError(f"cannot broadcast {self.shape} to {new_shape=}")
    # NOTE: this cast is no-op in forward and uses sum_acc_dtype in the backward sum
    return self.reshape(shape).cast(sum_acc_dtype(self.dtype))._apply_uop(UOp.expand, arg=new_shape).cast(self.dtype)

  def _broadcasted(self, y:Tensor|ConstType|UOp, reverse:bool=False, match_dtype:bool=True) -> tuple[Tensor, Tensor]:
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

  def sub(self, x:Tensor|ConstType, reverse=False) -> Tensor:
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

  def div(self, x:Tensor|ConstType, reverse=False, rounding_mode:Literal["trunc", "floor"]|None=None) -> Tensor:
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
    d = numerator.cast(least_upper_float(numerator.dtype)) * denominator.cast(least_upper_float(denominator.dtype)).reciprocal()
    output_dtype = numerator.dtype if dtypes.is_int(numerator.dtype) else d.dtype
    if dtypes.is_int(dt:=least_upper_dtype(numerator.dtype, denominator.dtype)) and rounding_mode is not None:
      numerator, denominator = numerator.cast(dt), denominator.cast(dt)
      if rounding_mode == "trunc": return numerator.idiv(denominator)
      if rounding_mode == "floor":
        truncate_div, truncate_mod = numerator.idiv(denominator), numerator._apply_broadcasted_uop(UOp.mod, denominator)
        opposite_sign = ((numerator>0)&(denominator<0)) | ((numerator<0)&(denominator>0))
        return (opposite_sign&(truncate_mod!=0)).where(truncate_div-1, truncate_div)
    if rounding_mode == "trunc": return d.trunc().cast(output_dtype)
    if rounding_mode == "floor": return d.floor().cast(output_dtype)
    if rounding_mode is not None: raise RuntimeError(f"{rounding_mode=} is not supported")
    return d

  def mod(self, x:Tensor|ConstType, reverse=False) -> Tensor:
    """
    Mod `self` by `x`.
    Equivalent to `self % x`.
    Supports broadcasting to a common shape, type promotion, and integer inputs.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-4, 7, 5, 4, -7, 8]).mod(Tensor([2, -3, 8, -2, 3, 5])).numpy())
    ```
    """
    a, b = self._broadcasted(x, reverse)
    return a - a.div(b, rounding_mode="floor") * b

  def bitwise_not(self) -> Tensor:
    """
    Computes the bitwise NOT of `self`.
    Equivalent to `~self`.
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0, 2, 5, 255], dtype="int8").bitwise_not().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([True, False]).bitwise_not().numpy())
    ```
    """
    if self.dtype != dtypes.bool and not dtypes.is_int(self.dtype): raise RuntimeError(f"{self.dtype} is not supported")
    return self.logical_not() if self.dtype == dtypes.bool else self ^ -1

  def lshift(self, x:int, reverse=False) -> Tensor:
    """
    Computes left arithmetic shift of `self` by `x` bits. `self` must have unsigned dtype.
    Equivalent to `self << x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, 3, 31], dtype=dtypes.uint8).lshift(2).numpy())
    ```
    """
    assert dtypes.is_unsigned(self.dtype) and isinstance(x, int) and x >= 0 and not reverse, f"not supported {self.dtype=} {x=}"
    return self.mul(2 ** x, reverse)

  def rshift(self, x:int, reverse=False) -> Tensor:
    """
    Computes right arithmetic shift of `self` by `x` bits. `self` must have unsigned dtype.
    Equivalent to `self >> x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([4, 13, 125], dtype=dtypes.uint8).rshift(2).numpy())
    ```
    """
    assert dtypes.is_unsigned(self.dtype) and isinstance(x, int) and x >= 0 and not reverse, f"not supported {self.dtype=} {x=}"
    return self.idiv(2 ** x, reverse)

  def pow(self, x:Tensor|ConstType, reverse=False) -> Tensor:
    """
    Computes power of `self` with `x`.
    Equivalent to `self ** x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).pow(2.0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).pow(Tensor([-1.5, 0.5, 1.5])).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print((2.0 ** Tensor([-1, 2, 3])).numpy())
    ```
    """
    base, exponent = self._broadcasted(x, reverse=reverse)
    # TODO: int pow
    if not base.is_floating_point(): raise RuntimeError("base needs to be float")

    ret = base._apply_uop(UOp.pow, exponent)
    # NOTE: pow(int, float) -> int
    return ret.round().cast(self.dtype) if not reverse and not dtypes.is_float(self.dtype) else ret

  def maximum(self, x:Tensor|ConstType) -> Tensor:
    """
    Computes element-wise maximum of `self` and `x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).maximum(1).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).maximum(Tensor([-4, -2, 9])).numpy())
    ```
    """
    return self._apply_broadcasted_uop(UOp.maximum, x)

  def minimum(self, x:Tensor|ConstType) -> Tensor:
    """
    Computes element-wise minimum of `self` and `x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).minimum(1).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).minimum(Tensor([-4, -2, 9])).numpy())
    ```
    """
    t, x = self._broadcasted(x)
    return t._inverse().maximum(x._inverse())._inverse()

  def where(self:Tensor, x:Tensor|ConstType|sint, y:Tensor|ConstType|sint) -> Tensor:
    """
    Returns a tensor of elements selected from either `x` or `y`, depending on `self`.
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
    return cond.cast(dtypes.bool)._apply_uop(UOp.where, *x._broadcasted(y))

  def copysign(self, other) -> Tensor:
    """
    Returns a tensor of with the magnitude of `self` and the sign of `other`, elementwise.
    """
    # NOTE: torch always return in float, we return based on the broadcasting rule.
    other = self._broadcasted(other)[1]
    # TODO: remove other*0?
    return (other < 0).where(-self.abs(), self.abs()) + other*0

  # ***** op wrappers *****

  def __invert__(self) -> Tensor: return self.bitwise_not()

  # TODO: combine with UOps __floordiv__
  def __floordiv__(self, x): return self.div(x, rounding_mode="floor")
  def __rfloordiv__(self, x): return self.div(x, rounding_mode="floor", reverse=True)

  def __pow__(self, x) -> Tensor: return self.pow(x)
  def __matmul__(self, x) -> Tensor: return self.matmul(x)

  def __rpow__(self, x) -> Tensor: return self.pow(x, True)
  def __rmatmul__(self, x) -> Tensor: return self.matmul(x, True)

  def __iadd__(self, x) -> Tensor: return self.assign(self.add(x))
  def __isub__(self, x) -> Tensor: return self.assign(self.sub(x))
  def __imul__(self, x) -> Tensor: return self.assign(self.mul(x))
  def __ipow__(self, x) -> Tensor: return self.assign(self.pow(x))
  def __itruediv__(self, x) -> Tensor: return self.assign(self.div(x))
  def __ifloordiv__(self, x) -> Tensor: return self.assign(self.__floordiv__(x))
  def __imatmul__(self, x) -> Tensor: return self.assign(self.matmul(x))
  def __iand__(self, x) -> Tensor: return self.assign(self.bitwise_and(x))
  def __ior__(self, x) -> Tensor: return self.assign(self.bitwise_or(x))
  def __ixor__(self, x) -> Tensor: return self.assign(self.bitwise_xor(x))
  def __ilshift__(self, x) -> Tensor: return self.assign(self.lshift(x))
  def __irshift__(self, x) -> Tensor: return self.assign(self.rshift(x))

  def __lt__(self, x) -> Tensor: return self._apply_broadcasted_uop(UOp.__lt__, x, False)
  def __gt__(self, x) -> Tensor: return self._apply_broadcasted_uop(UOp.__lt__, x, True)
  def ne(self, x) -> Tensor: return self._apply_broadcasted_uop(UOp.ne, x, False)

  def __eq__(self, x) -> Tensor: return self.eq(x)                      # type: ignore[override]

  # ***** functional nn ops *****

  def linear(self, weight:Tensor, bias:Tensor|None=None, dtype:DTypeLike|None=None) -> Tensor:
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
    if dtype is not None: return self.cast(dtype).linear(weight.cast(dtype), bias.cast(dtype) if bias is not None else bias)
    x = self.mul(weight) if len(weight.shape) == 1 else self.dot(weight)
    return x.add(bias) if bias is not None else x

  def sequential(self, ll:list[Callable[[Tensor], Tensor]]) -> Tensor:
    """
    Applies a sequence of functions to `self` chaining the output of each function to the input of the next.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t.sequential([lambda x: x * 2, lambda x: x + 1]).numpy())
    ```
    """
    return functools.reduce(lambda x,f: f(x), ll, self)

  def layernorm(self, axis:int|tuple[int,...]=-1, eps:float=1e-5) -> Tensor:
    """
    Applies Layer Normalization over a mini-batch of inputs.

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

  def batchnorm(self, weight:Tensor|None, bias:Tensor|None, mean:Tensor, invstd:Tensor, axis:int|tuple[int, ...]=1) -> Tensor:
    """
    Applies Batch Normalization over a mini-batch of inputs.

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

    - Paper: https://jmlr.org/papers/v15/srivastava14a.html

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 2)
    with Tensor.train():
      print(t.dropout().numpy())
    ```
    """
    if not 0 <= p <= 1: raise ValueError(f"{p=} is out of range [0, 1]")
    if not Tensor.training or p == 0: return self
    if p == 1: return self.zeros_like()
    return (Tensor.rand_like(self, requires_grad=False, dtype=dtypes.default_float, contiguous=False) >= p).contiguous().where(self, 0) / (1.0 - p)

  # helper function commonly used for indexing
  def _one_hot_along_dim(self:Tensor, num_classes:sint, dim:int=-1) -> Tensor:
    if not dtypes.is_int(self.dtype): raise RuntimeError(f"_one_hot_along_dim expects int index tensor, getting {self.dtype}")
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
    if not dtypes.is_int(self.dtype): raise RuntimeError(f"expect integer dtype, getting {self.dtype=}")
    if num_classes == -1: num_classes = (self.max()+1).item()
    return self[..., None]._one_hot_along_dim(num_classes).where(1, 0)

  def scaled_dot_product_attention(self, key:Tensor, value:Tensor, attn_mask:Tensor|None=None, dropout_p:float=0.0,
                                   is_causal:bool=False, enable_gqa:bool=False) -> Tensor:
    """
    Computes scaled dot-product attention.
    `self` is the query tensor, `key` is the key tensor, and `value` is the value tensor.

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
    # GQA: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    if enable_gqa:
      key = key.repeat_interleave(self.shape[-3] // key.shape[-3], dim=-3)
      value = value.repeat_interleave(self.shape[-3] // value.shape[-3], dim=-3)
    qk = self.matmul(key.transpose(-2,-1), dtype=least_upper_dtype(self.dtype, key.dtype, dtypes.float32)) / math.sqrt(self.shape[-1])
    # handle attention mask
    if is_causal:
      if attn_mask is not None: raise RuntimeError("cannot set attn_mask when is_causal=True")
      attn_mask = qk.ones_like(requires_grad=False, device=self.device, dtype=dtypes.bool).tril()
    if attn_mask is not None:
      if attn_mask.dtype == dtypes.bool: attn_mask = attn_mask.where(0, -float("inf"))
      qk = qk + attn_mask
    return qk.cast(self.dtype).softmax(-1).dropout(dropout_p) @ value

  def _do_reduction(self, reduction:ReductionStr="mean") -> Tensor:
    if reduction not in get_args(ReductionStr): raise ValueError(f"{reduction=} must be one of {get_args(ReductionStr)}")
    reductions: dict[str, Callable[[Tensor], Tensor]] = {"mean": Tensor.mean, "sum": Tensor.sum, "none": lambda x: x}
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

  def binary_crossentropy_logits(self, Y:Tensor, reduction:ReductionStr="mean", pos_weight:Tensor|None=None) -> Tensor:
    """
    Computes the binary cross-entropy loss between `self` and `Y` where `self` is logits.

    See: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2, -3])
    Y = Tensor([0, 1, 0])
    print(t.binary_crossentropy_logits(Y).item())
    ```
    """
    log_p, log_1_minus_p = self.logsigmoid(), (-self).logsigmoid()
    return (-((1 if pos_weight is None else pos_weight) * Y * log_p + (1-Y) * log_1_minus_p))._do_reduction(reduction)

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
    assert reduction in get_args(ReductionStr), f"reduction must be one of {get_args(ReductionStr)}"
    log_probs = self.log_softmax()
    loss_mask = (Y != ignore_index) if ignore_index != -1 else Y.ones_like(dtype=dtypes.bool)
    y = Y.to(self.device).unsqueeze(-1)._one_hot_along_dim(self.shape[-1], dim=-1) * loss_mask.unsqueeze(-1)
    smoothing = label_smoothing * (log_probs.mean(-1) * loss_mask)
    unreduced = ((1 - label_smoothing) * (log_probs * y).sum(-1) + smoothing)
    # NOTE: because of ignore_index, we can't use Tensor.mean (so can't use `_do_reduction` here)
    return -(unreduced.sum() / loss_mask.sum() if reduction == "mean" else (unreduced.sum() if reduction == "sum" else unreduced))

  def cross_entropy(self, Y:Tensor, reduction:ReductionStr="mean", label_smoothing:float=0.0) -> Tensor:
    """
    Computes the cross entropy loss between input logits and target.

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
    classes_dim = 0 if self.ndim == 1 else 1
    if self.shape != Y.shape:
      if self.max(classes_dim).shape != Y.shape: raise RuntimeError(f"shape mismatch: {self.shape=}, {Y.shape=}")
      Y = Y.unsqueeze(classes_dim)._one_hot_along_dim(num_classes=self.shape[classes_dim], dim=classes_dim)
    Y = (1 - label_smoothing)*Y + label_smoothing / int(Y.shape[classes_dim])
    return -self.log_softmax(classes_dim).mul(Y).sum(classes_dim)._do_reduction(reduction)

  def nll_loss(self, Y:Tensor, weight:Tensor|None=None, ignore_index:int|None=None, reduction:ReductionStr="mean") -> Tensor:
    """
    Computes the negative log likelihood loss between log-probabilities and target labels.

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

  def qr(self) -> tuple[Tensor, Tensor]:
    assert self.ndim > 1, f"expected two or more dimensions, got {self.ndim}"
    R = self.clone()
    b_shape, m, n = self.shape[0:self.ndim - 2], int(R.shape[-2]), int(R.shape[-1])
    Q = Tensor.eye(m, dtype = self.dtype).reshape((1,) * (len(self.shape) - 2) + 2 * (m,)).expand(b_shape + 2 * (m,)).contiguous()
    for i in range(int(min(m, n))):
      x = R[..., i:m, i]
      s = -x[..., 0].sign()
      u1 = x[..., 0] - s * x.square().sum(-1).sqrt()
      w = x.unsqueeze(-1) / u1.reshape(b_shape + 2 * (1,))
      w[..., 0, 0] = 1
      tau = (-s * u1 / x.square().sum(-1).sqrt()).reshape(b_shape + 2 * (1,)).expand(w.shape)
      R[..., i:m, :] = R[..., i:m, :] - (w * tau) @ (w.transpose(-2, -1) @ R[..., i:m, :])
      Q[..., :, i:m] = Q[..., :, i:m] - (Q[..., :, i:m] @ w) @ (tau.transpose(-2, -1) * w.transpose(-2, -1))
    return Q,R

  def svd(self, full_matrices = True) -> tuple[Tensor, Tensor, Tensor]:
    #partial implementation of https://www.netlib.org/lapack/lawnspdf/lawn169.pdf , pg 26
    assert self.ndim > 1, f"expected two or more dimensions, got {self.ndim}"
    b_shape, m, n = self.shape[:-2], int(self.shape[-2]), int(self.shape[-1])
    #preprocess the matrix
    Q, R = (Tensor.qr(self) if m >= n else Tensor.qr(self.transpose(-2, -1)))
    num, q_num = int(min(m, n)), int(max(m, n))
    U = R.shrink(tuple([(0, self.shape[i]) for i in range(self.ndim - 2)] + [(0, num), (0, num)])).contiguous()
    V = Tensor.eye(num, dtype = self.dtype).reshape((1,) * (self.ndim - 2) + (num, num)).expand(b_shape + 2 * (num,)).contiguous()
    #prepare round robin pairing
    permute, inverse_permute = Tensor.arange(0, num, dtype = dtypes.int), Tensor.zeros(num, dtype = dtypes.int).contiguous()
    permute[num//2:num] = permute[num//2:num].flip(0)
    inverse_permute[permute] = Tensor.arange(num, dtype = dtypes.int)
    def one_round_jacobi(U, V,permute,inverse_permute):
      #pair all the columns
      V_permuted, runoff_V = (V[..., permute].split(num - 1, -1)) if num % 2 == 1 else (V[..., permute], None)
      V_left, V_right = V_permuted.split(num//2, -1)
      U_permuted, runoff_U = (U[..., permute].split(num - 1, -1)) if num % 2 == 1 else (U[..., permute], None)
      U_left, U_right = U_permuted.split(num//2, -1)
      #compute the jacobi rotations for each pairing
      gamma = (U_left * U_right).sum(-2).reshape(b_shape + (1, num//2))
      alpha, beta = U_permuted.square().sum(-2).unsqueeze(-2).split(num//2, -1)
      tau = (beta - alpha) / (2 * gamma)
      t = tau.sign() / (tau.abs() + (1 + tau.square()).sqrt())
      c = 1 / (1 + t.square()).sqrt()
      s = c * t
      #apply the rotations
      U_left, U_right = c * U_left - s * U_right, s * U_left + c * U_right
      U = U_left.cat(U_right.cat(runoff_U, dim = -1) if num % 2 == 1 else U_right, dim = -1)[..., inverse_permute]
      V_left, V_right = c * V_left - s * V_right, s * V_left + c * V_right
      V = V_left.cat(V_right.cat(runoff_V, dim = -1) if num % 2 == 1 else V_right, dim = -1)[..., inverse_permute]
      #prepare the next round robin pairings
      if num % 2 == 1: permute = ((permute - 1) % num)
      else: permute = permute[0].reshape(1).cat(((permute[1:num] - 2) % (num - 1)) + 1)
      inverse_permute = inverse_permute.scatter(0,permute,Tensor.arange(num,dtype=dtypes.int32))
      return U, V, permute, inverse_permute
    max_iterations, iterations_per_round = 1, int((num) * math.log2(num) * 2 + 2)#sorta heuristic, most use num*log2(num)
    for _ in range(max_iterations * iterations_per_round): U, V, permute, inverse_permute = one_round_jacobi(U, V, permute, inverse_permute)
    #extract singular values and sort. construct U from Q
    S, indices = U.square().sum(-2).sqrt().sort(dim = -1, descending=True)
    new_indices = Tensor.arange(num).reshape((1,) * (self.ndim - 1) + (num,)).expand(b_shape + 2 * (num,)).contiguous()
    new_indices[..., :num] = indices.reshape(b_shape + (1,) + (num,)).expand(b_shape + 2 * (num,))
    U,V = U.gather(-1, new_indices[...,0:num,0:num]) / S.unsqueeze(-2), V.gather(-1, new_indices[..., 0:num, 0:num]).realize()

    padded_u = Tensor.eye(q_num, dtype = U.dtype).reshape((1,) * (self.ndim - 2) + 2 * (q_num,)).expand(b_shape + 2 * (q_num,)).contiguous()
    padded_u[..., 0:num, 0:num] = U
    U = Q @ padded_u
    if not full_matrices: U, V = U[..., 0:num], V[..., 0:num]
    return (U, S, V.transpose(-2,-1)) if m >= n else (V, S, U.transpose(-2, -1))

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
    Returns `True` if the tensor contains floating point types, i.e. is one of `dtypes.float64`, `dtypes.float32`,
    `dtypes.float16`, `dtypes.bfloat16`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([8, 9], dtype=dtypes.float32)
    print(t.is_floating_point())
    ```
    """
    return dtypes.is_float(self.dtype)

  def size(self, dim:int|None=None) -> sint|tuple[sint, ...]:
    """
    Returns the size of the tensor. If `dim` is specified, return the length along dimension `dim`. Otherwise return the shape of the tensor.

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

  def llvm_bf16_cast(self, dtype:DTypeLike) -> Tensor:
    # hack for devices that don't support bfloat16
    assert self.dtype == dtypes.bfloat16
    return self.to("LLVM").cast(dtype)

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
      return self._apply_uop(UOp.cast, dtype=dtypes.int32)._apply_uop(UOp.cast, dtype=dt)
    return self if self.dtype == dt else self._apply_uop(UOp.cast, dtype=dt)

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
    if (ns:=dt.itemsize) != (os:=self.dtype.itemsize) and (self.shape[-1]*os) % ns != 0: raise RuntimeError("unsupported size in bitcast")
    if (not isinstance(self.device, str) or not self.device.startswith("DISK")) and ns != os:
      new_uint, old_uint = to_dtype(f"uint{8*ns}"), to_dtype(f"uint{8*os}")
      tmp = self.bitcast(old_uint)
      if ns > os:
        tmp = tmp.reshape(self.shape[:-1] + (self.shape[-1]//(rate := ns//os), rate))
        nones = (None,) * (tmp.ndim - 1)
        return functools.reduce(Tensor.add, (tmp.shrink(nones + ((i, i+1),)).cast(new_uint)<<8*i*os for i in range(rate))).squeeze(-1).bitcast(dtype)
      return Tensor.stack(*(tmp>>8*i*ns for i in range(os//ns)), dim=-1).flatten(-2).cast(new_uint).bitcast(dtype)
    return self._apply_uop(UOp.bitcast, dtype=dt) if self.dtype != dt else self

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

  def bfloat16(self) -> Tensor: return self.cast(dtypes.bfloat16)
  def double(self) -> Tensor: return self.cast(dtypes.double)
  def long(self) -> Tensor: return self.cast(dtypes.long)
  def short(self) -> Tensor: return self.cast(dtypes.short)

  # *** image Tensor function replacements ***

  def image_dot(self, w:Tensor, dtype:DTypeLike|None=None) -> Tensor:
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
    return cx.image_conv2d(cw, groups=groups, dtype=dtype).reshape(out_shape_t).transpose(self.ndim-1, self.ndim-2)

  def image_conv2d(self, weight:Tensor, bias:Tensor|None=None, groups=1, stride=1, dilation=1, padding=0, dtype=None) -> Tensor:
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
    x = x.permute(0,3,4,5,1,2).pad(self._resolve_pool_pads(padding,2))._pool((H,W), stride, dilation)# -> (bs, groups, rcin_hi, rcin_lo, oy, ox, H, W)
    x = x.permute(0,4,5,1,2,3,6,7).reshape(bs, (oy := x.shape[4]), (ox := x.shape[5]), *cout_expand[0:2], 1, 1, rcin_hi, rcin_lo, H, W)

    # prepare weights
    w = w.permute(0,4,2,5,1,3).reshape((1, 1, 1, *cout_expand, rcin_hi, rcin_lo, H, W))

    # the conv!
    ret = (x*w).cast(base_image_type((bs*oy, ox*cout//4, 4)) if IMAGE >= 2 else dtypes.float32).sum((-4, -3, -2, -1), dtype=dtype)

    # undo hack for non multiples of 4 on C.rcout
    if added_output_channels != 0:
      ret = ret.reshape(bs, oy, ox, groups, rcout)[:, :, :, :, :-added_output_channels]
      cout = groups * (rcout - added_output_channels)

    # NCHW output
    ret = ret.reshape(bs, oy, ox, cout).permute(0,3,1,2)
    return ret if bias is None else ret.add(bias.reshape(1, -1, 1, 1))

P = ParamSpec("P")
T = TypeVar("T")
def _metadata_wrapper(fn: Callable[P, T]) -> Callable[P, T]:
  def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
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
    if name in ["__class__", "__init__", "__new__", "__repr__", "backward", "sequential", "gradient"]: continue
    setattr(Tensor, name, functools.wraps(fn)(_metadata_wrapper(fn)))
