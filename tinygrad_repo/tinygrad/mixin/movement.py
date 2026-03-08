# mixins add syntactic sugar to Tensor and UOp
from __future__ import annotations
from typing import TYPE_CHECKING, Self
from tinygrad.uop import Ops
from tinygrad.helpers import prod, argfix, flatten, dedup, make_tuple, ceildiv
from tinygrad.uop.ops import resolve, smax

if TYPE_CHECKING:
  from tinygrad.uop.ops import sint


def _align_left(*shapes: tuple[sint, ...]) -> tuple[tuple[sint, ...], ...]:
  # unsqueeze left to make every shape same length
  max_dim = max(len(shape) for shape in shapes)
  return tuple((1,) * (max_dim - len(shape)) + shape for shape in shapes)


class MovementMixin:
  # required to implement
  def _mop(self, op: Ops, arg) -> Self:
    raise NotImplementedError

  @property
  def shape(self) -> tuple[sint, ...]:
    raise NotImplementedError

  # great functions you get!
  @property
  def ndim(self) -> int:
    """
    Returns the number of dimensions in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2], [3, 4]])
    print(t.ndim)
    ```
    """
    return len(self.shape)

  def numel(self) -> sint:
    """
    Returns the total number of elements in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(t.numel())
    ```
    """
    return prod(self.shape)

  def _resolve_dim(self, dim: int, *, extra: bool = False) -> int:
    total = self.ndim + int(extra)
    if not -max(1, total) <= dim <= max(1, total) - 1:
      raise IndexError(f"{dim=} out of range {[-max(1, total), max(1, total) - 1]}")
    return dim + total if dim < 0 else dim

  def _broadcast_to(self, new_shape: tuple[sint, ...]) -> Self:
    if self.shape == new_shape:
      return self
    if self.ndim > len(new_shape):
      raise ValueError(f"cannot broadcast tensor to fewer dimensions. shape={self.shape} to {new_shape=}")
    # first unsqueeze left with 1s https://data-apis.org/array-api/latest/API_specification/broadcasting.html
    shape, _ = _align_left(self.shape, new_shape)
    # for each dimension, check either dim is 1, or it does not change
    if not all(s == ns or s == 1 for s, ns in zip(shape, new_shape)):
      raise ValueError(f"cannot broadcast {self.shape} to {new_shape=}")
    reshaped = self.reshape(shape)
    ret = reshaped._mop(Ops.EXPAND, arg=new_shape)
    return reshaped if ret.shape == reshaped.shape else ret

  def expand(self, shape, *args) -> Self:
    """
    Returns a tensor that is expanded to the shape that is specified.
    Expand can also increase the number of dimensions that a tensor has.

    Passing a `-1` or `None` to a dimension means that its size will not be changed.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t.expand(4, -1).numpy())
    ```
    """
    new_shape = tuple(from_ if to == -1 or to is None else to for from_, to in zip(*(_align_left(self.shape, argfix(shape, *args)))))
    return self._broadcast_to(new_shape)

  def reshape(self, shape, *args) -> Self:
    """
    Returns a tensor with the same data as the original tensor but with a different shape.
    `shape` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6)
    print(t.reshape(2, 3).numpy())
    ```
    """
    # resolve None and args
    new_shape = tuple([s if s is not None else self.shape[i] for i, s in enumerate(argfix(shape, *args))])
    # resolve -1
    if (c := new_shape.count(-1)) > 1:
      raise RuntimeError(f"only one dimension can be inferred using -1, getting {new_shape}")
    if c:
      new_shape = tuple([-prod(self.shape) // prod(new_shape) if s == -1 else s for s in new_shape])
    if prod(self.shape) != prod(new_shape):
      raise ValueError(f"size mismatch, can't reshape ({self.shape}) -> ({new_shape})")
    ret = self._mop(Ops.RESHAPE, arg=new_shape)
    return self if ret.shape == self.shape else ret

  def shrink(self, arg: tuple[tuple[sint, sint] | None, ...]) -> Self:
    """
    Returns a tensor that shrinks the each axis based on input arg.
    `arg` must have the same length as `self.ndim`.
    For each axis, it can be `None`, which means no shrink, or a tuple `(start, end)` that works the same as Python slice.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(3, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.shrink(((None, (1, 3)))).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.shrink((((0, 2), (0, 2)))).numpy())
    ```
    """
    if self.ndim != len(arg):
      raise ValueError(f"{self.ndim=} != {len(arg)=}")
    ret = self._mop(Ops.SHRINK, arg=[x if x is not None else (0, s) for x, s in zip(arg, self.shape)])
    return self if ret.shape == self.shape else ret

  def permute(self, order, *args) -> Self:
    """
    Returns a tensor that is a permutation of the original tensor.
    The new tensor has the same data as the original tensor but with the dimensions permuted according to the order specified.
    `order` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.empty(2, 3, 5)
    print(t.shape)
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.permute(2, 0, 1).shape)
    ```
    """
    order_arg = tuple(self._resolve_dim(x) for x in argfix(order, *args))
    if sorted(order_arg) != list(range(self.ndim)):
      raise RuntimeError(f"order is not a valid permutation, getting {order_arg}")
    return self._mop(Ops.PERMUTE, arg=order_arg) if order_arg != tuple(range(self.ndim)) else self

  def flip(self, axis, *args) -> Self:
    """
    Returns a tensor that reverses the order of the original tensor along given `axis`.
    `axis` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.flip(0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.flip((0, 1)).numpy())
    ```
    """
    axis_arg = tuple(self._resolve_dim(x) for x in argfix(axis, *args))
    assert all(not isinstance(x, bool) and x >= 0 and x < self.ndim for x in axis_arg), f"flip args must be axis ints {axis_arg}"
    if len(axis_arg) != len(dedup(axis_arg)):
      raise RuntimeError(f"dim can appear at most once, getting {axis_arg}")
    flip_arg = tuple([i in axis_arg for i in range(len(self.shape))])
    return self._mop(Ops.FLIP, arg=flip_arg) if any(flip_arg) else self

  # **** high level ****

  def shrink_to(self, shape, *args) -> Self:
    return self.shrink(tuple([None if ns is None else (0, ns) for ns in argfix(shape, *args)]))

  def view(self, shape, *args) -> Self:
    """`.view` is an alias for `.reshape`."""
    return self.reshape(shape, *args)

  def squeeze(self, dim: int | None = None) -> Self:
    """
    Returns a tensor with specified dimensions of input of size 1 removed.
    If `dim` is not specified, all dimensions with size 1 are removed.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.zeros(2, 1, 2, 1, 2)
    print(t.squeeze().shape)
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.squeeze(0).shape)
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.squeeze(1).shape)
    ```
    """
    if dim is None:
      return self.reshape(tuple(dim for dim in self.shape if dim != 1))
    dim = self._resolve_dim(dim)
    return self if not self.ndim or self.shape[dim] != 1 else self.reshape(self.shape[:dim] + self.shape[dim + 1 :])

  def unsqueeze(self, dim: int) -> Self:
    """
    Returns a tensor with a new dimension of size 1 inserted at the specified `dim`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(t.unsqueeze(0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.unsqueeze(1).numpy())
    ```
    """
    dim = self._resolve_dim(dim, extra=True)
    return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])

  @property
  def T(self) -> Self:
    """`.T` is an alias for `.transpose()`."""
    return self.transpose()

  def transpose(self, dim0=1, dim1=0) -> Self:
    """
    Returns a tensor that is a transposed version of the original tensor.
    The given dimensions `dim0` and `dim1` are swapped.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.transpose(0, 1).numpy())
    ```
    """
    order = list(range(self.ndim))
    order[dim0], order[dim1] = order[dim1], order[dim0]
    return self.permute(order)

  def flatten(self, start_dim=0, end_dim=-1) -> Self:
    """
    Flattens the tensor by reshaping it into a one-dimensional tensor.
    If `start_dim` or `end_dim` are passed, only dimensions starting with `start_dim` and ending with `end_dim` are flattened.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(8).reshape(2, 2, 2)
    print(t.flatten().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.flatten(start_dim=1).numpy())
    ```
    """
    start_dim, end_dim = self._resolve_dim(start_dim), self._resolve_dim(end_dim)
    return self.reshape(self.shape[:start_dim] + (prod(self.shape[start_dim : end_dim + 1]),) + self.shape[end_dim + 1 :])

  def unflatten(self, dim: int, sizes: tuple[int, ...]) -> Self:
    """
    Unflattens dimension `dim` of the tensor into multiple dimensions specified by `sizes`. `Tensor.flatten()` is the inverse of this function.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(3, 4, 1).unflatten(1, (2, 2)).shape)
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(3, 4, 1).unflatten(1, (-1, 2)).shape)
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(5, 12, 3).unflatten(-2, (2, 2, 3, 1, 1)).shape)
    ```
    """
    dim = self._resolve_dim(dim)
    return self.reshape(self.shape[:dim] + sizes + self.shape[dim + 1 :])

  def rearrange(self, formula: str, **sizes) -> Self:
    """
    Rearranges input according to formula

    See: https://einops.rocks/api/rearrange/

    ```python exec="true" source="above" session="tensor" result="python"
    x = Tensor([[1, 2], [3, 4]])
    print(Tensor.rearrange(x, "batch channel -> (batch channel)").numpy())
    ```
    """

    def parse_side(s: str) -> tuple[list[str], list[tuple[int, int]]]:
      """Parse one side of formula into (axis_names, dims) where dims are (start, end) index pairs for parens."""
      tokens = f" {s} ".replace("…", "...").replace("(", " ( ").replace(")", " ) ").replace(" ", "  ").replace(" 1 ", " ( ) ").split()
      lparens, rparens = [i for i, tok in enumerate(tokens) if tok == "("], [i for i, tok in enumerate(tokens) if tok == ")"]
      pairs = list(zip(lparens, rparens))
      assert len(lparens) == len(rparens) and sorted(flatten(pairs)) == flatten(pairs), "bracket mismatch"
      return [tok for tok in tokens if tok not in ("(", ")")], [(lp - 2*i, rp - 1 - 2*i) for i, (lp, rp) in enumerate(pairs)]

    assert formula.count("->") == 1, 'need exactly one "->" in formula'
    (lhs, unflatten_dims), (rhs, flatten_dims) = map(parse_side, formula.split("->"))

    for name in sizes: assert name in lhs, f"axis {name} is not used in transform"
    assert sorted(lhs) == sorted(rhs) and len(lhs) == len(set(lhs)), f"name mismatch in {formula}"
    for name in lhs+rhs: assert name == "..." or (name.isidentifier() and "_" not in (name[0], name[-1])), f"invalid axis name {name}"
    assert "..." not in flatten([lhs[s:e] for s, e in unflatten_dims]), f"cannot have collapsed ellipsis (...) in lhs of {formula}"
    assert lhs.count("...") <= 1, f"too many ellipses in {formula}"

    # resolve ellipsis
    if "..." in lhs:
      ell_len = len(self.shape) - len(lhs) + 1 + sum(e - s - 1 for s, e in unflatten_dims)
      lhs, rhs = map(lambda l: l[:(i := l.index("..."))] + [f"...{j}" for j in range(ell_len)] + l[i + 1:] if "..." in l else l, (lhs, rhs))
      def newdims(side, s, e): return (s + (ell_len - 1 if "...0" in side[:s] else 0), e + (ell_len - 1 if "...0" in side[:e] else 0))
      unflatten_dims, flatten_dims = [newdims(lhs, s, e) for s, e in unflatten_dims], [newdims(rhs, s, e) for s, e in flatten_dims]

    # unflatten -> permute -> flatten
    t = self
    for start, end in unflatten_dims: t = t.unflatten(start, tuple(sizes.get(lhs[i], -1) for i in range(start, end)))
    for i, name in enumerate(lhs):
      if name in sizes: assert sizes[name] == t.shape[i], f"size provided for dimension {name} incorrect"
    t = t.permute([lhs.index(name) for name in rhs])
    for start, end in reversed(flatten_dims): t = t.flatten(start, end - 1) if start < end else t.unsqueeze(start)
    return t

  # *** movement ops with expand ***

  def repeat_interleave(self, repeats: int, dim: int | None = None) -> Self:
    """
    Repeats elements of a tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t.repeat_interleave(2).numpy())
    ```
    """
    x, dim = (self.flatten(), 0) if dim is None else (self, self._resolve_dim(dim))
    shp = x.shape
    x = x.reshape(*shp[: dim + 1], 1, *shp[dim + 1 :])
    x = x.expand(*shp[: dim + 1], repeats, *shp[dim + 1 :])
    x = x.reshape(*shp[:dim], shp[dim] * repeats, *shp[dim + 1 :])
    return x

  def repeat(self, repeats, *args) -> Self:
    """
    Repeats tensor number of times along each dimension specified by `repeats`.
    `repeats` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t.repeat(4, 2).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.repeat(4, 2, 1).shape)
    ```
    """
    repeats = argfix(repeats, *args)
    base_shape = _align_left(self.shape, repeats)[0]
    unsqueezed_shape = flatten([[s] if r == 1 else [1, s] for r, s in zip(repeats, base_shape)])
    expanded_shape = flatten([[s] if r == 1 else [r, s] for r, s in zip(repeats, base_shape)])
    final_shape = [r * s for r, s in zip(repeats, base_shape)]
    return self.reshape(unsqueezed_shape).expand(expanded_shape).reshape(final_shape)

  # **** pool level ****

  def _pool(self, k_: tuple[sint, ...], stride: int | tuple[int, ...] = 1, dilation: int | tuple[int, ...] = 1) -> Self:
    assert len(self.shape) >= len(k_), f"can't pool {self.shape} with {k_}"
    s_, d_ = make_tuple(stride, len(k_)), make_tuple(dilation, len(k_))
    assert len(k_) == len(s_) == len(d_), f"stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}"
    noop, i_ = [None] * (self.ndim - len(k_)), self.shape[-len(k_) :]
    assert all(resolve(d * (k - 1) + 1 <= i) for k, d, i in zip(k_, d_, i_)), "kernel size cannot be greater than actual input size"
    o_ = [ceildiv(i - d * (k - 1), s) for i, d, k, s in zip(i_, d_, k_, s_)]
    # input size scaling factor to make sure shrink for stride is possible
    f_ = [smax(1, ceildiv(o * s - d, i)) for o, s, i, d in zip(o_, s_, i_, d_)]
    # repeats such that we don't need padding
    x = self.repeat([1] * len(noop) + [ceildiv(k * (i * f + d), i) for k, i, d, f in zip(k_, i_, d_, f_)])
    # handle dilation
    x = x.shrink_to(noop + [k * (i * f + d) for k, i, d, f in zip(k_, i_, d_, f_)])
    x = x.reshape(noop + flatten((k, (i * f + d)) for k, i, d, f in zip(k_, i_, d_, f_)))
    # handle stride
    x = x.shrink_to(noop + flatten((k, o * s) for k, o, s in zip(k_, o_, s_))).reshape(noop + flatten((k, o, s) for k, o, s in zip(k_, o_, s_)))
    x = x.shrink_to(noop + flatten((k, o, 1) for k, o in zip(k_, o_))).reshape(noop + flatten((k, o) for k, o in zip(k_, o_)))
    # permute to move reduce to the end
    return x.permute(*range(len(noop)), *[len(noop) + i * 2 + 1 for i in range(len(i_))], *[len(noop) + i * 2 for i in range(len(i_))])
