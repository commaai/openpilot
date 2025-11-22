from typing import Self
from tinygrad.uop import Ops
from tinygrad.dtype import dtypes, ConstType


class MathMixin:
  # required to implement
  def alu(self, op: Ops, *src: Self) -> Self:
    raise NotImplementedError

  def const_like(self, b: ConstType) -> Self:
    raise NotImplementedError

  # great functions you get!
  def ufix(self, x: Self | ConstType) -> Self:
    return self.const_like(x) if not isinstance(x, MathMixin) else x

  def _binop(self, op: Ops, x: Self | ConstType, reverse: bool) -> Self:
    return self.ufix(x).alu(op, self) if reverse else self.alu(op, self.ufix(x))

  def logical_not(self):
    return self.ne(True)

  def neg(self):
    if (dtype := getattr(self, "dtype")) is None:
      raise TypeError(f"MathTraits __neg__ requires a dtype, {self=}")
    return self.logical_not() if dtype.scalar() == dtypes.bool else self * (-1)

  def _check_dtype(self):
    if (dtype := getattr(self, "dtype")) is not None:
      if isinstance(dtype, tuple):
        dtype = dtype[0]
      if not (dtypes.is_bool(dtype) or dtypes.is_int(dtype)):
        raise RuntimeError(f"{dtype} is not supported")

  def add(self, x: Self | ConstType, reverse: bool = False):
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
    return self._binop(Ops.ADD, x, reverse)

  def mul(self, x: Self | ConstType, reverse: bool = False):
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
    return self._binop(Ops.MUL, x, reverse)

  def bitwise_and(self, x: Self | ConstType, reverse: bool = False):
    """
    Computes the bitwise AND of `self` and `x`.
    Equivalent to `self & x`.
    Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([2, 5, 255]).bitwise_and(Tensor([3, 14, 16])).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([True, True, False, False]).bitwise_and(Tensor([True, False, True, False])).numpy())
    ```
    """
    self._check_dtype()
    return self._binop(Ops.AND, x, reverse)

  def bitwise_or(self, x: Self | ConstType, reverse: bool = False):
    """
    Computes the bitwise OR of `self` and `x`.
    Equivalent to `self | x`.
    Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([2, 5, 255]).bitwise_or(Tensor([4, 4, 4])).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([True, True, False, False]).bitwise_or(Tensor([True, False, True, False])).numpy())
    ```
    """
    self._check_dtype()
    return self._binop(Ops.OR, x, reverse)

  def bitwise_xor(self, x: Self | ConstType, reverse: bool = False):
    """
    Computes bitwise xor of `self` and `x`.
    Equivalent to `self ^ x`.
    Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, -2, 3]).bitwise_xor(Tensor([1, 0, 3])).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([True, True, False, False]).bitwise_xor(Tensor([True, False, True, False])).numpy())
    ```
    """
    self._check_dtype()
    return self._binop(Ops.XOR, x, reverse)

  def idiv(self, x: Self | ConstType, reverse: bool = False):
    """
    Divides `self` by `x`.
    Equivalent to `self // x`.
    Supports broadcasting to a common shape, type promotion, and integer inputs.
    `idiv` performs integer division (truncate towards zero).

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-4, 7, 5, 4, -7, 8]).idiv(Tensor([2, -3, 8, -2, 3, 5])).numpy())
    ```
    """
    return self._binop(Ops.IDIV, x, reverse)

  def mod(self, x: Self | ConstType, reverse: bool = False):
    return self._binop(Ops.MOD, x, reverse)

  def sub(self, x: Self | ConstType, reverse: bool = False):
    return self.ufix(x).alu(Ops.ADD, -self) if reverse else self.alu(Ops.ADD, self.ufix(-x))

  def div(self, x: Self | ConstType, reverse: bool = False):
    return (self.ufix(x) * self.alu(Ops.RECIPROCAL)) if reverse else (self * self.ufix(x).alu(Ops.RECIPROCAL))

  def __neg__(self):
    return self.neg()

  def __add__(self, x: Self | ConstType):
    return self.add(x)

  def __sub__(self, x: Self | ConstType):
    return self.sub(x)

  def __mul__(self, x: Self | ConstType):
    return self.mul(x)

  def __truediv__(self, x: Self | ConstType):
    return self.div(x)

  def __floordiv__(self, x: Self | ConstType):
    return self.idiv(x)  # TODO: idiv is trunc div, not floordiv

  def __mod__(self, x: Self | ConstType):
    return self.mod(x)

  def __and__(self, x: Self | ConstType):
    return self.bitwise_and(x)

  def __or__(self, x: Self | ConstType):
    return self.bitwise_or(x)

  def __xor__(self, x: Self | ConstType):
    return self.bitwise_xor(x)

  def __radd__(self, x: Self | ConstType):
    return self.add(x, True)

  def __rsub__(self, x: Self | ConstType):
    return self.sub(x, True)

  def __rmul__(self, x: Self | ConstType):
    return self.mul(x, True)

  def __rtruediv__(self, x: Self | ConstType):
    return self.div(x, True)

  def __rfloordiv__(self, x: Self | ConstType):
    return self.idiv(x, True)

  def __rand__(self, x: Self | ConstType):
    return self.bitwise_and(x, True)

  def __ror__(self, x: Self | ConstType):
    return self.bitwise_or(x, True)

  def __rxor__(self, x: Self | ConstType):
    return self.bitwise_xor(x, True)

  def __rmod__(self, x: Self | ConstType):
    return self.mod(x, True)

  def __lt__(self, x: Self | ConstType):
    return self.alu(Ops.CMPLT, self.ufix(x))

  def __gt__(self, x: Self | ConstType):
    return self.ufix(x).alu(Ops.CMPLT, self)

  def __ge__(self, x: Self | ConstType):
    return (self < x).logical_not()

  def __le__(self, x: Self | ConstType):
    return (self > x).logical_not()

  def ne(self, x: Self | ConstType):
    return self.alu(Ops.CMPNE, self.ufix(x))

  def eq(self, x: Self | ConstType):
    return self.ne(x).logical_not()

  def __ne__(self, x: Self | ConstType):  # type: ignore[override]
    return self.ne(x)

  # NOTE: __eq__ isn't overridden, and means the same thing as is by default

  def lshift(self, x: Self | int, reverse: bool = False):
    return self._binop(Ops.SHL, x, reverse)

  def rshift(self, x: Self | int, reverse: bool = False):
    return self._binop(Ops.SHR, x, reverse)

  def __lshift__(self, x: Self | int):
    return self.lshift(x)

  def __rshift__(self, x: Self | int):
    return self.rshift(x)

  def __rlshift__(self, x: Self | int):
    return self.lshift(x, True)

  def __rrshift__(self, x: Self | int):
    return self.rshift(x, True)

  def maximum(self, x: Self | ConstType):
    return self.alu(Ops.MAX, self.ufix(x))

  def minimum(self, x: Self | ConstType):
    return -(-self).maximum(-x)

  def where(self, x: Self | ConstType, y: Self | ConstType):
    if isinstance(x, type(self)):
      return self.alu(Ops.WHERE, x, x.ufix(y))
    if isinstance(y, type(self)):
      return self.alu(Ops.WHERE, y.ufix(x), y)
    raise RuntimeError("where needs at least one UOp arg")

  def threefry(self, seed: Self):
    return self.alu(Ops.THREEFRY, seed)

  def reciprocal(self):
    return self.alu(Ops.RECIPROCAL)

  def trunc(self):
    return self.alu(Ops.TRUNC)

  def sqrt(self):
    return self.alu(Ops.SQRT)

  def sin(self):
    return self.alu(Ops.SIN)

  def log2(self):
    return self.alu(Ops.LOG2)

  def exp2(self):
    return self.alu(Ops.EXP2)

  def pow(self, x: Self | ConstType):
    return self.alu(Ops.POW, self.ufix(x))

  def __pow__(self, x: Self | ConstType):
    return self.pow(x)
