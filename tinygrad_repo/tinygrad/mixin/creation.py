from typing import Self
from tinygrad.dtype import ConstType, DType

class CreationMixin:
  def const_like(self, b: ConstType) -> Self: raise NotImplementedError
  def cast(self, dtype: DType) -> Self: raise NotImplementedError

  def full_like(self, fill_value: ConstType, dtype: DType|None=None) -> Self:
    """Creates a tensor with the same shape as `self`, filled with the given value."""
    return self.const_like(fill_value) if dtype is None else self.const_like(fill_value).cast(dtype)

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
