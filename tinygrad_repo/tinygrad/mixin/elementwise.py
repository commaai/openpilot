import math, functools, operator
from typing import Literal, Self
from tinygrad.uop import Ops
from tinygrad.dtype import dtypes, ConstType, PyConst, least_upper_dtype, least_upper_float
from tinygrad.helpers import argfix, polyN
from tinygrad.mixin.dtype import DTypeMixin
from tinygrad.mixin.creation import CreationMixin


class ElementwiseMixin(DTypeMixin, CreationMixin):
  # required to implement
  def alu(self, op: Ops, *src: Self) -> Self:
    raise NotImplementedError

  def _broadcasted(self, y: Self | ConstType, reverse: bool = False) -> tuple[Self, Self]:
    raise NotImplementedError

  # great functions you get!
  def ufix(self, x: Self | ConstType) -> Self:
    return self.const_like(x) if not isinstance(x, ElementwiseMixin) else x

  def _binop(self, op: Ops, x: Self | ConstType, reverse: bool) -> Self:
    return self.ufix(x).alu(op, self) if reverse else self.alu(op, self.ufix(x))

  def usum(self, *uops) -> Self: return functools.reduce(operator.or_ if self.dtype is dtypes.bool else operator.add, argfix(*uops), self)
  def uprod(self, *uops) -> Self: return functools.reduce(operator.and_ if self.dtype is dtypes.bool else operator.mul, argfix(*uops), self)

  def detach(self) -> Self:
    """
    Returns a new tensor with the same data as this tensor, but detached from the autograd graph.
    """
    return self.alu(Ops.DETACH)

  def logical_not(self) -> Self:
    """
    Computes the logical NOT of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([False, True]).logical_not().numpy())
    ```
    """
    return self.cast(dtypes.bool).ne(True)

  def contiguous(self, *args, **kwargs) -> Self: raise NotImplementedError

  def contiguous_backward(self) -> Self:
    """
    Inserts a contiguous operation in the backward pass.
    """
    return self.alu(Ops.CONTIGUOUS_BACKWARD)

  def neg(self) -> Self:
    """
    Negates the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).neg().numpy())
    ```
    """
    return self.logical_not() if self.dtype.scalar() == dtypes.bool else self * (-1)

  def _check_dtype(self) -> None:
    if not (dtypes.is_bool(self.dtype) or dtypes.is_int(self.dtype)):
      raise RuntimeError(f"{self.dtype} is not supported")

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

  def sub(self, x: Self | ConstType, reverse: bool = False) -> Self:
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

  def bitwise_not(self) -> Self:
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
    self._check_dtype()
    if self.dtype == dtypes.bool: return self.logical_not()
    return (self ^ self.dtype.max) if dtypes.is_unsigned(self.dtype) else (self ^ -1)

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

  def mod(self, x: Self | ConstType, reverse: bool = False) -> Self:
    """
    Mod `self` by `x`.
    Equivalent to `self % x`.
    Supports broadcasting to a common shape, type promotion, and integer inputs.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-4, 7, 5, 4, -7, 8]).mod(Tensor([2, -3, 8, -2, 3, 5])).numpy())
    ```
    """
    a, b = self._broadcasted(x, reverse)
    if dtypes.is_int(a.dtype): return a.alu(Ops.FLOORMOD, b)
    return a - a.div(b, rounding_mode="floor") * b

  def fmod(self, x: Self | ConstType) -> Self:
    """
    C-style remainder of `self` divided by `x` (sign follows the dividend), using truncating division.
    Differs from `mod`/`%`, which uses Python floor remainder.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-4, 7, 5, 4, -7, 8]).fmod(Tensor([2, -3, 8, -2, 3, 5])).numpy())
    ```
    """
    a, b = self._broadcasted(x)
    if dtypes.is_int(a.dtype): return a.alu(Ops.CMOD, b)
    return a - a.div(b, rounding_mode="trunc") * b

  def div(self, x: Self | ConstType, reverse: bool = False, rounding_mode: Literal["trunc", "floor"] | None = None) -> Self:
    """
    Divides `self` by `x`.
    Equivalent to `self / x`.
    Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.
    `div` performs true division by default; pass `rounding_mode="trunc"` for truncating toward zero
    or `rounding_mode="floor"` for floor division.

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
    a, b = self._broadcasted(x, reverse)
    if dtypes.is_int(a.dtype):
      if rounding_mode == "trunc": return a.alu(Ops.CDIV, b)
      if rounding_mode == "floor": return a.alu(Ops.FLOORDIV, b)
    d = a * b.reciprocal()
    if rounding_mode is None: return d
    if rounding_mode == "trunc": return d.trunc()
    if rounding_mode == "floor": return d.floor()
    raise RuntimeError(f"{rounding_mode=} is not supported")

  def __neg__(self) -> Self:
    return self.neg()

  def __invert__(self) -> Self:
    return self.bitwise_not()

  def __add__(self, x: Self | ConstType) -> Self:
    return self.add(x)

  def __sub__(self, x: Self | ConstType) -> Self:
    return self.sub(x)

  def __mul__(self, x: Self | ConstType) -> Self:
    return self.mul(x)

  def __truediv__(self, x: Self | ConstType) -> Self:
    return self.div(x)

  def __floordiv__(self, x: Self | ConstType) -> Self:
    return self.div(x, rounding_mode="floor")

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
    return self.div(x, reverse=True, rounding_mode="floor")

  def __rand__(self, x: Self | ConstType) -> Self:
    return self.bitwise_and(x, True)

  def __ror__(self, x: Self | ConstType) -> Self:
    return self.bitwise_or(x, True)

  def __rxor__(self, x: Self | ConstType) -> Self:
    return self.bitwise_xor(x, True)

  def __rmod__(self, x: Self | ConstType) -> Self:
    return self.mod(x, True)

  def __lt__(self, x: Self | ConstType) -> Self:
    return self._binop(Ops.CMPLT, x, False)

  def __gt__(self, x: Self | ConstType) -> Self:
    return self._binop(Ops.CMPLT, x, True)

  def __ge__(self, x: Self | ConstType) -> Self:
    return (self < x).logical_not()

  def __le__(self, x: Self | ConstType) -> Self:
    return (self > x).logical_not()

  def ne(self, x: Self | ConstType) -> Self:
    return self._binop(Ops.CMPNE, x, False)

  def eq(self, x: Self | ConstType) -> Self:
    return self.ne(x).logical_not()

  def __ne__(self, x: Self | ConstType) -> Self:  # type: ignore[override]
    return self.ne(x)

  # NOTE: __eq__ isn't overridden, and means the same thing as is by default

  def lshift(self, x: Self | int, reverse: bool = False) -> Self:
    """
    Computes left arithmetic shift of `self` by `x` bits. `self` must have integer dtype.
    Equivalent to `self << x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, 3, 31], dtype=dtypes.uint8).lshift(2).numpy())
    ```
    """
    return self._binop(Ops.SHL, x, reverse)

  def rshift(self, x: Self | int, reverse: bool = False) -> Self:
    """
    Computes right arithmetic shift of `self` by `x` bits. `self` must have integer dtype.
    Equivalent to `self >> x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([4, 13, 125], dtype=dtypes.uint8).rshift(2).numpy())
    ```
    """
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
    """
    Computes element-wise maximum of `self` and `x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).maximum(1).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).maximum(Tensor([-4, -2, 9])).numpy())
    ```
    """
    return self._binop(Ops.MAX, x, False)

  def _inverse(self) -> Self: return -self if self.is_floating_point() else ~self

  def minimum(self, x: Self | ConstType) -> Self:
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

  def copysign(self, other: Self | ConstType) -> Self:
    """
    Returns a tensor of with the magnitude of `self` and the sign of `other`, elementwise.
    """
    # NOTE: torch always return in float, we return based on the broadcasting rule.
    a, b = self._broadcasted(other)
    return a.abs() * ((b < 0) | (b.reciprocal() < 0)).where(-1, 1)

  def logaddexp(self, other: Self | ConstType) -> Self:
    """
    Calculates (self.exp()+other.exp()).log(), elementwise.
    """
    a, b = self._broadcasted(other)
    m = a.maximum(b)
    return ((a-m).exp() + (b-m).exp()).log() + m

  def where(self, x: Self | ConstType, y: Self | ConstType) -> Self:
    ref: Self = x if isinstance(x, type(self)) else y if isinstance(y, type(self)) else \
      self.cast(least_upper_dtype(dtypes.from_py(x), dtypes.from_py(y)))
    return self.alu(Ops.WHERE, ref.ufix(x), ref.ufix(y))

  def masked_fill(self, mask:Self, value:Self|PyConst) -> Self:
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

  def threefry(self, seed: Self) -> Self:
    return self.alu(Ops.THREEFRY, seed)

  def _ensure_float(self) -> Self:
    return self if self.is_floating_point() else self.cast(least_upper_float(self.dtype))

  def reciprocal(self) -> Self:
    """
    Computes `1/x` element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 3., 4.]).reciprocal().numpy())
    ```
    """
    return self._ensure_float().alu(Ops.RECIPROCAL)

  def trunc(self) -> Self:
    """
    Truncates the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).trunc().numpy())
    ```
    """
    return self.alu(Ops.TRUNC)

  def sqrt(self) -> Self:
    """
    Computes the square root of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 3., 4.]).sqrt().numpy())
    ```
    """
    return self._ensure_float().alu(Ops.SQRT)

  def sin(self) -> Self:
    """
    Computes the sine of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).sin().numpy())
    ```
    """
    return self._ensure_float().alu(Ops.SIN)

  def cos(self) -> Self:
    """
    Computes the cosine of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).cos().numpy())
    ```
    """
    if self.is_floating_point(): return ((math.pi/2)-self.cast(least_upper_dtype(self.dtype, dtypes.float32))).sin().cast(self.dtype)
    return ((math.pi/2)-self).sin()

  def exp(self) -> Self:
    """
    Computes the exponential function element-wise.

    See: https://en.wikipedia.org/wiki/Exponential_function

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., 1., 2., 3.]).exp().numpy())
    ```
    """
    if self.is_floating_point():
      return self.cast(least_upper_dtype(self.dtype, dtypes.float32)).mul(1/math.log(2)).exp2().cast(self.dtype)
    return self.mul(1/math.log(2)).exp2()

  def log2(self) -> Self:
    """
    Computes the base-2 logarithm element-wise.

    See: https://en.wikipedia.org/wiki/Logarithm

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 4., 8.]).log2().numpy())
    ```
    """
    return self._ensure_float().alu(Ops.LOG2)

  def exp2(self) -> Self:
    """
    Computes the base-2 exponential function element-wise.

    See: https://en.wikipedia.org/wiki/Exponential_function

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., 1., 2., 3.]).exp2().numpy())
    ```
    """
    return self._ensure_float().alu(Ops.EXP2)

  def pow(self, x: Self | ConstType, reverse: bool = False) -> Self:
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
    if not base.is_floating_point() and not isinstance(x, ElementwiseMixin) and not (isinstance(x, int) and x >= 0):
      raise RuntimeError("base needs to be float")
    ret = base.alu(Ops.POW, exponent)
    # NOTE: pow(int, float) -> int
    return ret.round().cast(self.dtype) if not reverse and not dtypes.is_float(self.dtype) and dtypes.is_float(exponent.dtype) else ret

  def __pow__(self, x: Self | ConstType) -> Self:
    return self.pow(x)

  def __rpow__(self, x: Self | ConstType) -> Self:
    return self.pow(x, True)

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

  def isclose(self, other, rtol:float=1e-05, atol:float=1e-08, equal_nan=False) -> Self:
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
    is_infinite_close = (self.isinf() | other.isinf()) & self.eq(other)
    is_nan_close = (self.isnan() & other.isnan()) & equal_nan
    return is_finite_close | is_infinite_close | is_nan_close

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

  def gelu(self, approximate:str="tanh") -> Self:
    """
    Applies the Gaussian Error Linear Unit (GELU) function element-wise.

    - Paper: https://arxiv.org/abs/1606.08415v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).gelu().numpy())
    ```
    """
    if approximate == "tanh":
      return 0.5 * self * (1 + (math.sqrt(2 / math.pi) * (self + 0.044715 * self ** 3)).tanh())
    elif approximate == "none":
      return self * 0.5 * (1.0 + (self / math.sqrt(2)).erf())
    else:
      raise RuntimeError(f"{approximate=} is not supported")

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

  def log(self) -> Self:
    """
    Computes the natural logarithm element-wise.

    See: https://en.wikipedia.org/wiki/Logarithm

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 4., 8.]).log().numpy())
    ```
    """
    return self.log2()*math.log(2)

  def log10(self) -> Self:
    """
    Computes the base-10 logarithm element-wise.

    See: https://en.wikipedia.org/wiki/Logarithm

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 4., 8.]).log10().numpy())
    ```
    """
    return self.log2()*math.log10(2)

  def atanh(self) -> Self:
    """
    Applies the Inverse Hyperbolic Tangent (atanh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#atanh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).atanh().numpy())
    ```
    """
    return ((1 + self)/(1 - self)).log() / 2

  def asinh(self) -> Self:
    """
    Applies the Inverse Hyperbolic Sine (asinh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#asinh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).asinh().numpy())
    ```
    """
    return (self + (self.square() + 1).sqrt()).log()

  def acosh(self) -> Self:
    """
    Applies the Inverse Hyperbolic Cosine (acosh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#acosh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).acosh().numpy())
    ```
    """
    return (self + (self.square() - 1).sqrt()).log()

  def round(self) -> Self:
    """
    Rounds the tensor element-wise with rounding half to even.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).round().numpy())
    ```
    """
    return ((self > 0).eq((b := self.trunc() / 2.0).trunc().eq(b))).where((self - 0.5).ceil(), (self + 0.5).floor())

  def sign(self) -> Self:
    """
    Returns the sign of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sign().numpy())
    ```
    """
    return self.ne(0).where((self < 0).where(self.const_like(-1), self.const_like(1)), self.const_like(0))

  def abs(self) -> Self:
    """
    Computes the absolute value of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).abs().numpy())
    ```
    """
    return self * self.sign()

  def tan(self) -> Self:
    """
    Computes the tangent of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., math.pi/4, math.pi/2, 3*math.pi/4, math.pi]).tan().numpy())
    ```
    """
    return self.sin() / self.cos()

  def asin(self) -> Self:
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

  def acos(self) -> Self:
    """
    Computes the inverse cosine (arccosine) of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).acos().numpy())
    ```
    """
    return math.pi / 2 - self.asin()

  def atan(self) -> Self:
    """
    Computes the inverse tangent (arctan) of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).atan().numpy())
    ```
    """
    return (self / (1 + self * self).sqrt()).asin()

  def elu(self, alpha=1.0) -> Self:
    """
    Applies the Exponential Linear Unit (ELU) function element-wise.

    - Paper: https://arxiv.org/abs/1511.07289v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).elu().numpy())
    ```
    """
    return self.relu() - alpha*(1-self.exp()).relu()

  def celu(self, alpha=1.0) -> Self:
    """
    Applies the Continuously differentiable Exponential Linear Unit (CELU) function element-wise.

    - Paper: https://arxiv.org/abs/1704.07483

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).celu().numpy())
    ```
    """
    return self.maximum(0) + (alpha * ((self / alpha).exp() - 1)).minimum(0)

  def selu(self, alpha=1.67326, gamma=1.0507) -> Self:
    """
    Applies the Scaled Exponential Linear Unit (SELU) function element-wise.

    - Paper: https://arxiv.org/abs/1706.02515v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).selu().numpy())
    ```
    """
    return gamma * (self >= 0).where(self, alpha * (self.exp() - 1))

  def softplus(self, beta=1.0) -> Self:
    """
    Applies the Softplus function element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).softplus().numpy())
    ```
    """
    return (1/beta) * (self*beta).logaddexp(0.0)

  def mish(self) -> Self:
    """
    Applies the Mish function element-wise.

    - Paper: https://arxiv.org/abs/1908.08681v3

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).mish().numpy())
    ```
    """
    return self * self.softplus().tanh()

  def logsigmoid(self) -> Self:
    """
    Applies the LogSigmoid function element-wise.

    - See: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.logsigmoid.html

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).logsigmoid().numpy())
    ```
    """
    return -(-self).softplus()

  def sinh(self) -> Self:
    """
    Applies the Hyperbolic Sine (sinh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Sinh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sinh().numpy())
    ```
    """
    return (self.exp() - self.neg().exp()) / 2

  def cosh(self) -> Self:
    """
    Applies the Hyperbolic Cosine (cosh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Cosh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).cosh().numpy())
    ```
    """
    return (self.exp() + self.neg().exp()) / 2

  def erf(self) -> Self:
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

  def softsign(self) -> Self:
    """
    Applies the Softsign function element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).softsign().numpy())
    ```
    """
    return self / (1 + self.abs())

  def lerp(self, end: Self, weight: Self | ConstType) -> Self:
    """
    Linearly interpolates between `self` and `end` by `weight`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 3.]).lerp(Tensor([4., 5., 6.]), 0.5).numpy())
    ```
    """
    if self.dtype == dtypes.uint8 and isinstance(weight, ElementwiseMixin):
      w_i = (weight * (1<<(W_PREC:=7)) + 0.5).cast(dtypes.int16)
      return (self+(((end - self).cast(dtypes.int8) * w_i + (1<<W_PREC-1)).cast(dtypes.uint16) >> W_PREC)).cast(dtypes.uint8)
    return self + (end - self) * weight
