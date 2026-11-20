from typing import TYPE_CHECKING, Callable, Self
from tinygrad.dtype import ConstType, DType, DTypeLike, Invalid, dtypes, to_dtype
from tinygrad.helpers import argfix
from tinygrad.mixin.dtype import DTypeMixin
from tinygrad.mixin.movement import MovementMixin

if TYPE_CHECKING:
  from tinygrad.uop.ops import sint, UOp

class CreationMixin(DTypeMixin, MovementMixin):
  @staticmethod
  def const(b, dtype=None): raise NotImplementedError

  def const_like(self, b: ConstType, dtype:DType|None=None) -> Self: return self._wrap_uop(self._uop.const_like(b, dtype))

  def _multi_like(self, fxn:'Callable[[tuple[sint, ...], str|None], Self]') -> Self:
    from tinygrad.uop.ops import UOp
    assert isinstance(self.device, tuple), f"_multi_like needs a multi device tensor, got {self.device}"
    if self._uop.axis is None: return self._wrap_uop(fxn(self.shape, None)._uop.shard(self.device, None))
    return self._wrap_uop(UOp.mstack(*[fxn(self._uop.shard_shape, d)._uop for d in self.device]).unshard(self._uop.axis))

  @classmethod
  def empty(cls, *shape, device:str|tuple[str, ...]|None=None, dtype:DTypeLike|None=None, axis:int|None=None) -> Self:
    """
    Creates an empty tensor with the given shape.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    With multiple devices, optionally specify which `axis` to shard on.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.empty(2, 3)
    print(t.shape)
    ```
    """
    from tinygrad.uop.ops import UOp
    from tinygrad.device import canonicalize_device
    dt = to_dtype(dtype) if dtype is not None else dtypes.default_float
    if dt in dtypes.weaks: raise RuntimeError(f"cannot create storage for weak dtype {dt}")
    new_shape = argfix(*shape)
    dev = canonicalize_device(device)
    if axis is not None and isinstance(dev, tuple):
      axis = range(max(1, len(new_shape)))[axis]
      if new_shape:
        if new_shape[axis] % len(dev) != 0: raise RuntimeError(f"can't split {new_shape} on axis {axis} over {dev}")
        local = tuple(s//len(dev) if i == axis else s for i,s in enumerate(new_shape))
        return cls._wrap_uop(cls.empty(*local, device=dev, dtype=dt)._uop.unshard(axis))
    return cls._wrap_uop(UOp.alloc(new_shape, dt, device=dev, bind_on_realize=True))

  def empty_like(self, dtype: DTypeLike|None=None, device: str|tuple[str, ...]|None=None) -> Self:
    """
    Creates an empty tensor with the same shape as `self`.
    If `dtype` is not specified, the dtype of `self` is used.
    """
    return self._wrap_uop(self._uop.empty_like(dtype, device))

  @classmethod
  def invalids(cls, *shape, device:str|tuple[str, ...]|None=None, dtype:DTypeLike|None=None) -> Self:
    """
    Creates a tensor with the given shape, filled with Invalid.

    This is an alternative to Tensor.empty when you want an "anonymous" buffer.

    Eventually Tensor.empty will be replaced by this.
    """
    return cls.full(argfix(*shape), Invalid, dtype=dtype, device=device)

  @classmethod
  def full(cls, shape:'tuple[sint, ...]', fill_value:'ConstType|UOp', dtype:DTypeLike|None=None,
           device:str|tuple[str, ...]|None=None, buffer=True, axis:int|None=None) -> Self:
    """
    Creates a tensor with the given shape, filled with the given value.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    With multiple devices, optionally specify which `axis` to shard on.
    Pass `buffer=False` to get a broadcast const value instead of a materialized buffer.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 3), 42).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 3), False).numpy())
    ```
    """
    # TODO: enable this check
    # if not buffer: assert device is None, "buffer=False does not support device specification"
    from tinygrad.uop.ops import UOp
    if axis is not None and (isinstance(fill_value, UOp) or not buffer): raise RuntimeError("axis requires a scalar fill value and buffer=True")
    new_shape = argfix(shape)
    dt = to_dtype(dtype) if dtype is not None else fill_value.dtype if isinstance(fill_value, UOp) else dtypes.from_py(fill_value)
    val = cls.const(fill_value, dt).expand(new_shape)
    if not buffer: return val
    if axis is not None: ret = cls.empty(new_shape, device=device, dtype=val.commit_dtype() if dt in dtypes.weaks else dt, axis=axis)
    else: ret = val.empty_like(None if dt in dtypes.weaks else dt, device)
    return cls._wrap_uop(ret._uop.after(ret._uop.store(val._uop)))

  def full_like(self, fill_value:ConstType, dtype:DTypeLike|None=None, device:str|tuple[str, ...]|None=None, buffer=True) -> Self:
    """
    Creates a tensor with the same shape as `self`, filled with the given value.
    If `dtype` is not specified, the dtype of `self` is used.

    You can pass in the `device` keyword argument to control device of the tensor.
    Pass `buffer=False` to get a broadcast const value instead of a materialized buffer.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(Tensor.full_like(t, 42).numpy())
    ```
    """
    if isinstance(self.device, tuple):
      if device is not None: raise RuntimeError("cannot specify `device` on `*_like` of a multi device tensor")
      return self._multi_like(lambda shape, dev: type(self).full(shape, fill_value, dtype=dtype or self.dtype, device=dev, buffer=buffer))
    return type(self).full(self.shape, fill_value, dtype=dtype or self.dtype, device=self.device if device is None else device, buffer=buffer)

  @classmethod
  def zeros(cls, *shape, **kwargs) -> Self:
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
    return cls.full(argfix(*shape), 0.0, **kwargs)

  def zeros_like(self, **kwargs) -> Self:
    """
    Creates a tensor with the same shape as `self`, filled with zeros.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(Tensor.zeros_like(t).numpy())
    ```
    """
    return self.full_like(0, **kwargs)

  @classmethod
  def ones(cls, *shape, **kwargs) -> Self:
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
    return cls.full(argfix(*shape), 1.0, **kwargs)

  def ones_like(self, **kwargs) -> Self:
    """
    Creates a tensor with the same shape as `self`, filled with ones.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.zeros(2, 3)
    print(Tensor.ones_like(t).numpy())
    ```
    """
    return self.full_like(1, **kwargs)
