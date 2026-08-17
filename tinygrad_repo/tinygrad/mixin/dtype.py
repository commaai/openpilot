from typing import TYPE_CHECKING, Self
from tinygrad.dtype import DType, DTypeLike, dtypes, to_dtype
from tinygrad.uop import Ops

if TYPE_CHECKING:
  from tinygrad.uop.ops import UOp

class DTypeMixin:
  @property
  def dtype(self) -> DType: raise NotImplementedError
  @property
  def _uop(self) -> 'UOp': raise NotImplementedError
  @classmethod
  def _wrap_uop(cls, u:'UOp') -> Self: raise NotImplementedError

  def cast(self, dtype:DTypeLike) -> Self:
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
    return self if self.dtype == (dt:=to_dtype(dtype)) else self._wrap_uop(self._uop.alu(Ops.CAST, arg=dt))

  def bitcast(self, dtype:DTypeLike) -> Self:
    """
    Bitcasts `self` to the given `dtype`. If the itemsize differs, the last axis is rescaled.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2, 3], dtype=dtypes.int32)
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.bitcast(dtypes.uint32)
    print(t.dtype, t.numpy())
    ```
    """
    dt = to_dtype(dtype)
    if self.dtype in dtypes.weaks or dt in dtypes.weaks: raise RuntimeError(f"bitcast requires concrete dtypes, got {self.dtype} -> {dt}")
    return self if self.dtype == dt else self._wrap_uop(self._uop.alu(Ops.BITCAST, arg=dt))

  def element_size(self) -> int:
    """
    Returns the size in bytes of an individual element in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([5], dtype=dtypes.int16)
    print(t.element_size())
    ```
    """
    if self.dtype in dtypes.weaks: raise RuntimeError(f"element_size requires a concrete dtype, got {self.dtype}")
    return self.dtype.itemsize

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

  def float(self) -> Self:
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

  def half(self) -> Self:
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

  def int(self) -> Self:
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

  def bool(self) -> Self:
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

  def bfloat16(self) -> Self: return self.cast(dtypes.bfloat16)
  def double(self) -> Self: return self.cast(dtypes.double)
  def long(self) -> Self: return self.cast(dtypes.long)
  def short(self) -> Self: return self.cast(dtypes.short)
