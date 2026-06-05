from typing import Self
from tinygrad.dtype import DType, dtypes

class DTypeMixin:
  @property
  def dtype(self) -> DType: raise NotImplementedError

  def cast(self, dtype:DType) -> Self: raise NotImplementedError

  def bitcast(self, dtype:DType) -> Self: raise NotImplementedError

  def element_size(self) -> int:
    """
    Returns the size in bytes of an individual element in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([5], dtype=dtypes.int16)
    print(t.element_size())
    ```
    """
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
    return dtypes.is_float(self.dtype.base)

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
