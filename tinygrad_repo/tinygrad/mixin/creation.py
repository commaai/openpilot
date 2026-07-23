from typing import TYPE_CHECKING, Callable, Self
from tinygrad.dtype import ConstType, DTypeLike, Invalid, dtypes, to_dtype
from tinygrad.helpers import argfix, prod
from tinygrad.mixin.dtype import DTypeMixin
from tinygrad.mixin.movement import MovementMixin

if TYPE_CHECKING:
  from tinygrad.uop.ops import sint, UOp

class CreationMixin(DTypeMixin, MovementMixin):
  @staticmethod
  def const(dtype, b): raise NotImplementedError

  def const_like(self, b: ConstType) -> Self: return self._wrap_uop(self._uop.const_like(b))

  def _multi_like(self, fxn:'Callable[[tuple[sint, ...], str|None], Self]') -> Self:
    from tinygrad.uop.ops import UOp
    assert isinstance(self.device, tuple), f"_multi_like needs a multi device tensor, got {self.device}"
    if self._uop.axis is None: return self._wrap_uop(fxn(self.shape, None)._uop.shard(self.device, None))
    return self._wrap_uop(UOp.mstack(*[fxn(self._uop.shard_shape, d)._uop for d in self.device]).multi(self._uop.axis))

  @classmethod
  def empty(cls, *shape, device:str|tuple[str, ...]|None=None, dtype:DTypeLike|None=None) -> Self:
    """
    Creates an empty tensor with the given shape.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.empty(2, 3)
    print(t.shape)
    ```
    """
    from tinygrad.uop.ops import UOp, to_max_shape
    from tinygrad.device import canonicalize_device
    dt = to_dtype(dtype) if dtype is not None else dtypes.default_float
    new_shape = argfix(*shape)
    max_shape = to_max_shape(new_shape)
    u = UOp.new_buffer(canonicalize_device(device), prod(max_shape), dt).reshape(max_shape).shrink_to(new_shape)
    return cls._wrap_uop(u)

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
           device:str|tuple[str, ...]|None=None, buffer=True) -> Self:
    """
    Creates a tensor with the given shape, filled with the given value.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
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
    new_shape = argfix(shape)
    dt = to_dtype(dtype) if dtype is not None else fill_value.dtype if isinstance(fill_value, UOp) else dtypes.from_py(fill_value)
    val = cls.const(dt, fill_value)
    val = val.reshape((1,)*len(new_shape)).expand(new_shape)
    return val.clone(device=device) if buffer else val

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
