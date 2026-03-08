import math
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

  def logical_not(self) -> Self:
    return self.ne(True)

  def neg(self) -> Self:
    if (dtype := getattr(self, "dtype")) is None:
      raise TypeError(f"MathTraits __neg__ requires a dtype, {self=}")
    return self.logical_not() if dtype.scalar() == dtypes.bool else self * (-1)

  def _check_dtype(self) -> None:
    if (dtype := getattr(self, "dtype")) is not None:
      if isinstance(dtype, tuple):
        dtype = dtype[0]
      if not (dtypes.is_bool(dtype) or dtypes.is_int(dtype)):
        raise RuntimeError(f"{dtype} is not supported")

  def add(self, x: Self | ConstType, reverse: bool = False) -> Self:
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

  def mul(self, x: Self | ConstType, reverse: bool = False) -> Self:
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

  def bitwise_and(self, x: Self | ConstType, reverse: bool = False) -> Self:
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

  def bitwise_or(self, x: Self | ConstType, reverse: bool = False) -> Self:
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

  def bitwise_xor(self, x: Self | ConstType, reverse: bool = False) -> Self:
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

  def idiv(self, x: Self | ConstType, reverse: bool = False) -> Self:
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

  def mod(self, x: Self | ConstType, reverse: bool = False) -> Self:
    return self._binop(Ops.MOD, x, reverse)

  def sub(self, x: Self | ConstType, reverse: bool = False) -> Self:
    return self.ufix(x).alu(Ops.ADD, -self) if reverse else self.alu(Ops.ADD, -self.ufix(x))

  def div(self, x: Self | ConstType, reverse: bool = False) -> Self:
    return (self.ufix(x) * self.alu(Ops.RECIPROCAL)) if reverse else (self * self.ufix(x).alu(Ops.RECIPROCAL))

  def __neg__(self) -> Self:
    return self.neg()

  def __add__(self, x: Self | ConstType) -> Self:
    return self.add(x)

  def __sub__(self, x: Self | ConstType) -> Self:
    return self.sub(x)

  def __mul__(self, x: Self | ConstType) -> Self:
    return self.mul(x)

  def __truediv__(self, x: Self | ConstType) -> Self:
    return self.div(x)

  def __floordiv__(self, x: Self | ConstType) -> Self:
    return self.idiv(x)  # TODO: idiv is trunc div, not floordiv

  def __mod__(self, x: Self | ConstType) -> Self:
    return self.mod(x)

  def __and__(self, x: Self | ConstType) -> Self:
    return self.bitwise_and(x)

  def __or__(self, x: Self | ConstType) -> Self:
    return self.bitwise_or(x)

  def __xor__(self, x: Self | ConstType) -> Self:
    return self.bitwise_xor(x)

  def __radd__(self, x: Self | ConstType) -> Self:
    return self.add(x, True)

  def __rsub__(self, x: Self | ConstType) -> Self:
    return self.sub(x, True)

  def __rmul__(self, x: Self | ConstType) -> Self:
    return self.mul(x, True)

  def __rtruediv__(self, x: Self | ConstType) -> Self:
    return self.div(x, True)

  def __rfloordiv__(self, x: Self | ConstType) -> Self:
    return self.idiv(x, True)

  def __rand__(self, x: Self | ConstType) -> Self:
    return self.bitwise_and(x, True)

  def __ror__(self, x: Self | ConstType) -> Self:
    return self.bitwise_or(x, True)

  def __rxor__(self, x: Self | ConstType) -> Self:
    return self.bitwise_xor(x, True)

  def __rmod__(self, x: Self | ConstType) -> Self:
    return self.mod(x, True)

  def __lt__(self, x: Self | ConstType) -> Self:
    return self.alu(Ops.CMPLT, self.ufix(x))

  def __gt__(self, x: Self | ConstType) -> Self:
    return self.ufix(x).alu(Ops.CMPLT, self)

  def __ge__(self, x: Self | ConstType) -> Self:
    return (self < x).logical_not()

  def __le__(self, x: Self | ConstType) -> Self:
    return (self > x).logical_not()

  def ne(self, x: Self | ConstType) -> Self:
    return self.alu(Ops.CMPNE, self.ufix(x))

  def eq(self, x: Self | ConstType) -> Self:
    return self.ne(x).logical_not()

  def __ne__(self, x: Self | ConstType) -> Self:  # type: ignore[override]
    return self.ne(x)

  # NOTE: __eq__ isn't overridden, and means the same thing as is by default

  def lshift(self, x: Self | int, reverse: bool = False) -> Self:
    return self._binop(Ops.SHL, x, reverse)

  def rshift(self, x: Self | int, reverse: bool = False) -> Self:
    return self._binop(Ops.SHR, x, reverse)

  def __lshift__(self, x: Self | int) -> Self:
    return self.lshift(x)

  def __rshift__(self, x: Self | int) -> Self:
    return self.rshift(x)

  def __rlshift__(self, x: Self | int) -> Self:
    return self.lshift(x, True)

  def __rrshift__(self, x: Self | int) -> Self:
    return self.rshift(x, True)

  def maximum(self, x: Self | ConstType) -> Self:
    return self.alu(Ops.MAX, self.ufix(x))

  def minimum(self, x: Self | ConstType) -> Self:
    return -(-self).maximum(-self.ufix(x))

  def where(self, x: Self | ConstType, y: Self | ConstType) -> Self:
    if isinstance(x, type(self)):
      return self.alu(Ops.WHERE, x, x.ufix(y))
    if isinstance(y, type(self)):
      return self.alu(Ops.WHERE, y.ufix(x), y)
    raise RuntimeError("where needs at least one UOp arg")

  def threefry(self, seed: Self) -> Self:
    return self.alu(Ops.THREEFRY, seed)

  def reciprocal(self) -> Self:
    return self.alu(Ops.RECIPROCAL)

  def trunc(self) -> Self:
    """
    Truncates the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).trunc().numpy())
    ```
    """
    return self.alu(Ops.TRUNC)

  def sqrt(self) -> Self:
    return self.alu(Ops.SQRT)

  def sin(self) -> Self:
    return self.alu(Ops.SIN)

  def log2(self) -> Self:
    return self.alu(Ops.LOG2)

  def exp2(self) -> Self:
    return self.alu(Ops.EXP2)

  def pow(self, x: Self | ConstType) -> Self:
    return self.alu(Ops.POW, self.ufix(x))

  def __pow__(self, x: Self | ConstType) -> Self:
    return self.pow(x)

  def square(self) -> Self:
    """
    Squares the tensor element-wise.
    Equivalent to `self*self`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).square().numpy())
    ```
    """
    return self * self

  def clamp(self, min_=None, max_=None) -> Self:
    """
    Clips (clamps) the values in the tensor between `min_` and `max_` element-wise.
    If `min_` is `None`, there is no lower bound. If `max_` is None, there is no upper bound.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).clip(-1, 1).numpy())
    ```
    """
    if min_ is None and max_ is None: raise RuntimeError("at least one of 'min_' or 'max_' must not be None")
    ret = (self < min_).where(min_, self) if min_ is not None else self
    return (ret > max_).where(max_, ret) if max_ is not None else ret

  def clip(self, min_=None, max_=None) -> Self:
    """Alias for `Tensor.clamp`."""
    return self.clamp(min_, max_)

  def isnan(self) -> Self:
    """
    Checks the tensor element-wise to return True where the element is NaN, otherwise returns False

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isnan().numpy())
    ```
    """
    return self != self

  def isinf(self, detect_positive: bool = True, detect_negative: bool = True) -> Self:
    """
    Checks the tensor element-wise to return True where the element is infinity, otherwise returns False

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isinf().numpy())
    ```
    """
    return self.eq(float("inf")) * detect_positive + self.eq(float("-inf")) * detect_negative

  def isfinite(self) -> Self:
    """
    Checks the tensor element-wise to return True where the element is finite, otherwise returns False

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isfinite().numpy())
    ```
    """
    return (self.isinf() | self.isnan()).logical_not()

  def ceil(self) -> Self:
    """
    Rounds the tensor element-wise towards positive infinity.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).ceil().numpy())
    ```
    """
    return (self > (b := self.trunc())).where(b+1, b)

  def floor(self) -> Self:
    """
    Rounds the tensor element-wise towards negative infinity.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).floor().numpy())
    ```
    """
    return (self < (b := self.trunc())).where(b-1, b)

  def relu(self) -> Self:
    """
    Applies the Rectified Linear Unit (ReLU) function element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).relu().numpy())
    ```
    """
    # NOTE: if you write this as self.maximum(0) the gradient is wrong, passing through half when self is 0
    return (self > 0).where(self, 0)

  def sigmoid(self) -> Self:
    """
    Applies the Sigmoid function element-wise.

    - Described: https://en.wikipedia.org/wiki/Sigmoid_function

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sigmoid().numpy())
    ```
    """
    return (1 + (self * (-1/math.log(2))).exp2()).reciprocal()

  def relu6(self) -> Self:
    """
    Applies the ReLU6 function element-wise.

    - Paper: https://arxiv.org/abs/1704.04861v1

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-9., -6., -3., 0., 3., 6., 9.]).relu6().numpy())
    ```
    """
    return self.relu() - (self-6).relu()

  def hardswish(self) -> Self:
    """
    Applies the Hardswish function element-wise.

    - Paper: https://arxiv.org/abs/1905.02244v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).hardswish().numpy())
    ```
    """
    return self * (self+3).relu6() * (1/6)

  def hardsigmoid(self, alpha: float = 1/6, beta: float = 0.5) -> Self:
    """
    Applies the Hardsigmoid function element-wise.
    NOTE: default `alpha` and `beta` values are taken from torch

    - See: https://pytorch.org/docs/stable/generated/torch.nn.functional.hardsigmoid.html

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).hardsigmoid().numpy())
    ```
    """
    return (alpha * self + beta).relu() - (alpha * self + beta - 1).relu()

  def hardtanh(self, min_val=-1, max_val=1) -> Self:
    """
    Applies the Hardtanh function element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5]).hardtanh().numpy())
    ```
    """
    return self.clip(min_val, max_val)

  def leaky_relu(self, neg_slope=0.01) -> Self:
    """
    Applies the Leaky ReLU function element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).leaky_relu().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).leaky_relu(neg_slope=0.42).numpy())
    ```
    """
    return (self < 0).where(neg_slope*self, self)

  def tanh(self) -> Self:
    """
    Applies the Hyperbolic Tangent (tanh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Tanh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).tanh().numpy())
    ```
    """
    return 2.0 * ((2.0 * self).sigmoid()) - 1.0

  def quick_gelu(self) -> Self:
    """
    Applies the Sigmoid GELU approximation element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).quick_gelu().numpy())
    ```
    """
    return self * (self * 1.702).sigmoid()

  def gelu(self) -> Self:
    """
    Applies the Gaussian Error Linear Unit (GELU) function element-wise.

    - Paper: https://arxiv.org/abs/1606.08415v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).gelu().numpy())
    ```
    """
    return 0.5 * self * (1 + (math.sqrt(2 / math.pi) * (self + 0.044715 * self ** 3)).tanh())

  def swish(self) -> Self:
    """
    See `.silu()`

    - Paper: https://arxiv.org/abs/1710.05941v1

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).swish().numpy())
    ```
    """
    return self * self.sigmoid()

  def silu(self) -> Self:
    """
    Applies the Sigmoid Linear Unit (SiLU) function element-wise.

    - Paper: https://arxiv.org/abs/1606.08415

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).silu().numpy())
    ```
    """
    return self.swish()  # The SiLU function is also known as the swish function.

  def rsqrt(self) -> Self:
    """
    Computes the reciprocal of the square root of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 3., 4.]).rsqrt().numpy())
    ```
    """
    return self.sqrt().reciprocal()
