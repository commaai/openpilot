import string
from typing import Self, Sequence, cast
from tinygrad.uop import Ops
from tinygrad.dtype import DTypeLike, dtypes, sum_acc_dtype, to_dtype
from tinygrad.helpers import argfix, argsort, make_tuple, merge_dicts
from tinygrad.mixin.dtype import DTypeMixin
from tinygrad.mixin.movement import MovementMixin


class ReduceMixin(DTypeMixin, MovementMixin):
  def _rop(self, op: Ops, axis: tuple[int, ...]) -> Self:
    raise NotImplementedError

  def _reduce(self, op:Ops, axis:int|Sequence[int]|None=None, keepdim=False) -> Self:
    axis = tuple(self._resolve_dim(x) for x in (range(self.ndim) if axis is None else make_tuple(axis, 1)))
    if self.ndim == 0: axis = ()
    ret = self._rop(op, axis)
    return ret if keepdim else ret.reshape(tuple(s for i,s in enumerate(self.shape) if i not in axis))

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

  @classmethod
  def einsum(cls, formula:str, *operands:Self|Sequence[Self], dtype:DTypeLike|None=None) -> Self:
    """
    Sums the product of the elements of the input tensors according to a formula based on the Einstein summation convention.

    See: https://pytorch.org/docs/stable/generated/torch.einsum.html

    ```python exec="true" source="above" session="tensor" result="python"
    x = Tensor([[1, 2], [3, 4]])
    y = Tensor([[5, 6], [7, 8]])
    print(Tensor.einsum("ij,ij->", x, y).numpy())
    ```
    """
    xs, formula = list(argfix(*operands)), formula.replace(" ", "")
    # expand ellipsis to letters, determine output
    if "..." in formula:
      ell, lhs = "".join(c for c in string.ascii_letters if c not in formula), (formula.split("->") + [""])[0]
      ell_n = [max(0, x.ndim - len(s) + 3) if "..." in s else 0 for s, x in zip(lhs.split(","), xs)]
      for i, (s, x) in enumerate(zip(inputs := lhs.split(","), xs)): inputs[i] = s.replace("...", ell[max(ell_n)-ell_n[i]:max(ell_n)])
      lhs, auto = ",".join(inputs), "".join(sorted(c for c in lhs if lhs.count(c) == 1 and c.isalpha() and c not in ell))
      formula = f"{lhs}->{formula.split('->')[1].replace('...', ell[:max(ell_n)]) if '->' in formula else ell[:max(ell_n)] + auto}"
    lhs, rhs = formula.split("->") if "->" in formula else (formula, "".join(sorted(c for c in formula if formula.count(c)==1 and c.isalpha())))
    inputs = lhs.split(",")
    if len(xs) != len(inputs): raise ValueError(f"number of operands doesn't match, expected {len(inputs)}, got {len(xs)}")
    # trace: take diagonal when letter repeats in single input
    for i, (s, x) in enumerate(zip(inputs, xs)):
      for c in set(s):
        while s.count(c) > 1:
          j, k, n = s.index(c), s.index(c, s.index(c)+1), cast(int, x.shape[s.index(c)])
          perm = [d for d in range(x.ndim) if d not in (j,k)]+[j,k]
          x = x.permute(perm).flatten(-2).pad(((0,0),)*(x.ndim-2)+((0,n),)).unflatten(-1,(n,n+1))[...,0] if x.ndim > 2 else x.diagonal()
          s = s[:k] + s[k+1:]
      inputs[i], xs[i] = s, x
    # check sizes and build sorted alphabet
    sz = merge_dicts([dict(zip(s, x.shape)) for s, x in zip(inputs, xs)])
    alpha = sorted(sz)
    # align all tensors to alphabet, multiply, sum non-output, permute to output order
    xs = [x.permute(*[s.index(c) for c in sorted(s)]).reshape([sz[c] if c in s else 1 for c in alpha]).expand([sz[c] for c in alpha]) if s else x
          for s, x in zip(inputs, xs)]
    return xs[0].uprod(*xs[1:]).sum([i for i,c in enumerate(alpha) if c not in rhs], dtype=dtype).permute(argsort(argsort(list(rhs))))
