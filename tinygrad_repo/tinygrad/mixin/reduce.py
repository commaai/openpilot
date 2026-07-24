from typing import Self, Sequence
from tinygrad.uop import Ops
from tinygrad.dtype import DTypeLike, dtypes, sum_acc_dtype, to_dtype
from tinygrad.helpers import make_tuple
from tinygrad.mixin.dtype import DTypeMixin
from tinygrad.mixin.movement import MovementMixin


class ReduceMixin(DTypeMixin, MovementMixin):
  def _rop(self, op: Ops, axis: tuple[int, ...]) -> Self:
    raise NotImplementedError

  def _reduce(self, op:Ops, axis:int|Sequence[int]|None=None, keepdim=False) -> Self:
    axis = tuple(self._resolve_dim(x) for x in (range(self.ndim) if axis is None else make_tuple(axis, 1)))
    if self.ndim == 0: axis = ()
    ret = self._rop(op, axis)
    return ret.reshape(tuple(1 if i in axis else s for i,s in enumerate(self.shape))) if keepdim else ret

  def sum(self, axis:int|Sequence[int]|None=None, keepdim=False, dtype:DTypeLike|None=None) -> Self:
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
    ret = self.cast(sum_acc_dtype(self.dtype) if dtype is None else to_dtype(dtype))._reduce(Ops.ADD, axis, keepdim)
    return ret.cast(self.dtype) if dtype is None and self.dtype in (dtypes.float16, dtypes.bfloat16, *dtypes.fp8s) else ret

  def prod(self, axis:int|Sequence[int]|None=None, keepdim=False, dtype:DTypeLike|None=None) -> Self:
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
    return self.cast(to_dtype(dtype) if dtype is not None else self.dtype)._reduce(Ops.MUL, axis, keepdim)

  def max(self, axis:int|Sequence[int]|None=None, keepdim=False) -> Self:
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

  def any(self, axis:int|Sequence[int]|None=None, keepdim=False) -> Self:
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

  def all(self, axis:int|Sequence[int]|None=None, keepdim=False) -> Self:
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
    return self.bool().prod(axis, keepdim)
