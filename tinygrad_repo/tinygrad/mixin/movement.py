# mixins add syntactic sugar to Tensor and UOp
from __future__ import annotations
from typing import TYPE_CHECKING, Self, Sequence
from tinygrad.uop import Ops
from tinygrad.helpers import prod, argfix, argsort, flatten, dedup, make_tuple, ceildiv, round_up, all_int
from tinygrad.uop.ops import resolve, smax, _align_left, _broadcast_shape

if TYPE_CHECKING:
  from tinygrad.uop.ops import sint


class MovementMixin:
  # required to implement
  def _mop(self, op: Ops, arg) -> Self:
    raise NotImplementedError

  @property
  def shape(self) -> tuple[sint, ...]:
    raise NotImplementedError

  @property
  def device(self) -> str|tuple[str, ...]|None:
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

  def size(self, dim:int|None=None) -> sint|tuple[sint, ...]:
    """
    Returns the size of the tensor. If `dim` is specified, return the length along dimension `dim`. Otherwise return the shape of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[4, 5, 6], [7, 8, 9]])
    print(t.size())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.size(dim=1))
    ```
    """
    return self.shape if dim is None else self.shape[dim]

  def _normalize_indices(self, indices:list) -> list:
    if len(ell := [i for i,x in enumerate(indices) if x is Ellipsis]) > 1: raise IndexError("indices can only have a single ellipsis")
    num_real = len(indices) - len(ell) - sum(1 for i in indices if i is None)
    if num_real > self.ndim: raise IndexError(f"too many indices ({num_real}) for {self.ndim}D")
    fill_idx = ell[0] if ell else len(indices)
    indices[fill_idx:fill_idx+1] = [slice(None)] * (self.ndim - num_real)
    return indices

  def _resolve_dim(self, dim: int, *, extra: bool = False) -> int:
    total = self.ndim + int(extra)
    if not -max(1, total) <= dim <= max(1, total) - 1:
      raise IndexError(f"{dim=} out of range {[-max(1, total), max(1, total) - 1]}")
    return dim + total if dim < 0 else dim

  def _parse_view_index(self, index, size: sint) -> dict:
    # parses a single slice/int/None/sint index into {boundary, stride, size, collapse_dim}
    from tinygrad.uop.ops import UOp, sint
    match index:
      case None: return {"size":1, "boundary":(0,1), "stride":1, "collapse_dim":False}
      case int() | UOp(): # sint
        if resolve(index >= size, False) or resolve(index < -size, False): raise IndexError(f"{index=} is out of bounds with {size=}")
        # TODO: is this right for (negative) symbolic?
        b = index if resolve(index >= 0, False) else index + size
        return {"size":size, "boundary":(b, b+1), "stride":1, "collapse_dim":True}
      case slice():
        if not all(s is None or isinstance(s, sint) for s in (index.start, index.stop, index.step)):
          raise TypeError(f"slice {index=} is not supported")
        if resolve(index.step == 0, False): raise ValueError(f"{index=} cannot have 0 as step")
        start, stop = 0 if index.start is None else index.start, size if index.stop is None else index.stop
        step = 1 if index.step is None else index.step
        if all_int((start, stop, step)):
          # handle int slicing (resolve negative bounds, clamp, stride)
          *bound, stride = index.indices(int(size.vmax) if isinstance(size, UOp) else size)
          bound = [0, 0] if stride * (bound[1] - bound[0]) < 0 else ([bound[1]+1, bound[0]+1] if stride < 0 else bound)
          return {"size":ceildiv(bound[1]-bound[0], abs(stride)), "boundary":tuple(bound), "stride":stride, "collapse_dim":False}
        if resolve(step == 1, False) and resolve((stop-start) >= 0, False):
          return {"size":stop-start, "boundary":(start, stop), "stride":step, "collapse_dim":False}
        raise TypeError(f"slice {index=} is not supported")
      case _: raise IndexError(f"{type(index).__name__} indexing is not supported")

  def _apply_view_ops(self, mops:list) -> Self:
    # applies shrink + flip + stride from a list of parsed view indices
    # flip negative strides
    x = self.shrink(tuple(m["boundary"] for m in mops)).flip(tuple(i for i, m in enumerate(mops) if m["stride"] < 0))
    strides = tuple(abs(m["stride"]) for m in mops)
    # apply stride
    if any(st != 1 for st in strides):
      if not all_int(x.shape): raise RuntimeError("symbolic shape not supported")
      x = x.pad_to(tuple(round_up(s, st) for s, st in zip(x.shape, strides)))
      x = x.reshape(tuple(flatten((s // st, st) for s, st in zip(x.shape, strides))))
      x = x.shrink_to(tuple(flatten((s, 1) for s in x.shape[::2]))).reshape(x.shape[::2])
    return x

  def __getitem__(self, indices) -> Self:
    # wrap single index into a list
    if (isinstance(indices, list) and all_int(indices)) or not isinstance(indices, (tuple, list)): indices = [indices]
    indices_parsed, dim = [], 0
    for index in self._normalize_indices(list(indices)):
      indices_parsed.append({**self._parse_view_index(index, 1 if index is None else self.shape[dim]), "index":index})
      if index is not None: dim += 1
    x = self._apply_view_ops(mops) if (mops := [p for p in indices_parsed if p["index"] is not None]) else self
    # dim injection from None (size 1) and dim collapse from int indices
    return x.reshape(tuple(p["size"] for p in indices_parsed if not p["collapse_dim"]))

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

  def pad(self, arg:tuple[tuple[sint, sint] | None, ...]) -> Self:
    if self.ndim != len(arg):
      raise ValueError(f"{self.ndim=} != {len(arg)=}")
    ret = self._mop(Ops.PAD, tuple(x if x is not None else (0, 0) for x in arg))
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

  def pad_to(self, shape, *args) -> Self:
    return self._mop(Ops.PAD, tuple([(0, 0 if ns is None else ns-s) for s,ns in zip(self.shape, argfix(shape, *args), strict=True)]))

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

  def split(self, sizes:int|Sequence[int], dim:int=0) -> tuple[Self, ...]:
    """
    Splits the tensor into chunks along the dimension specified by `dim`.
    If `sizes` is an integer, it splits into equally sized chunks if possible, otherwise the last chunk will be smaller.
    If `sizes` is a list, it splits into `len(sizes)` chunks with size in `dim` according to `size`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(10).reshape(5, 2)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    split = t.split(2)
    print("\\n".join([repr(x.numpy()) for x in split]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    split = t.split([1, 4])
    print("\\n".join([repr(x.numpy()) for x in split]))
    ```
    """
    dim = self._resolve_dim(dim)
    dim_sz = self.shape[dim]
    assert isinstance(dim_sz, int), f"does not support symbolic shape in split dimension {dim}: {self.shape}"
    if isinstance(sizes, int): sizes = [min(sizes, dim_sz-i) for i in range(0, max(1, dim_sz), max(1, sizes))]
    assert sum(sizes) == dim_sz, f"expect sizes to sum exactly to {dim_sz}, but got {sum(sizes)}"
    return tuple(self.shrink(tuple((sum(sizes[:i]), sum(sizes[:i+1])) if j == dim else None for j in range(self.ndim))) for i in range(len(sizes)))

  def chunk(self, chunks:int, dim:int=0) -> list[Self]:
    """
    Splits the tensor into `chunks` number of chunks along the dimension `dim`.
    If the tensor size along `dim` is not divisible by `chunks`, all returned chunks will be the same size except the last one.
    The function may return fewer than the specified number of chunks.

    ```python exec="true" source="above" session="tensor" result="python"
    chunked = Tensor.arange(11).chunk(6)
    print("\\n".join([repr(x.numpy()) for x in chunked]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    chunked = Tensor.arange(12).chunk(6)
    print("\\n".join([repr(x.numpy()) for x in chunked]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    chunked = Tensor.arange(13).chunk(6)
    print("\\n".join([repr(x.numpy()) for x in chunked]))
    ```
    """
    dim = self._resolve_dim(dim)
    dim_sz = self.shape[dim]
    assert isinstance(dim_sz, int), f"does not support symbolic shape in split dimension {dim}: {self.shape}"
    assert chunks > 0, f"expect chunks to be greater than 0, got: {chunks}"
    return list(self.split(ceildiv(dim_sz, chunks) if dim_sz else [0]*chunks, dim=dim))

  def meshgrid(self, *args, indexing:str="ij") -> tuple[Self, ...]:
    """
    Generates coordinate matrices from coordinate vectors.
    Input tensors can be scalars or 1D tensors.

    `indexing` determines how the output grids are aligned.
    `ij` indexing follows matrix-style indexing and `xy` indexing follows Cartesian-style indexing.

    ```python exec="true" source="above" session="tensor" result="python"
    x, y = Tensor([1, 2, 3]), Tensor([4, 5, 6])
    grid_x, grid_y = x.meshgrid(y)
    print(grid_x.numpy())
    print(grid_y.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    grid_x, grid_y = x.meshgrid(y, indexing="xy")
    print(grid_x.numpy())
    print(grid_y.numpy())
    ```
    """
    if indexing not in ("ij", "xy"): raise RuntimeError(f'indexing must be in ("ij", "xy"), got {indexing}')
    if len(tensors:=(self, *args)) == 1: return tensors
    basis = tuple(range(len(tensors))) if indexing == "ij" else (1, 0) + tuple(range(2, len(tensors)))
    tensors = tuple(t.reshape((-1,) + (1,)*(len(args) - i)) for i,t in zip(basis, tensors))
    output_shape = _broadcast_shape(*(t.shape for t in tensors))
    return tuple(t._broadcast_to(output_shape) for t in tensors)

  def diag(self) -> Self:
    """
    Returns a 2-D square tensor with the elements of input as the main diagonal.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, 2, 3]).diag().numpy())
    ```
    """
    if self.ndim != 1: raise ValueError(f"expect input to be 1-D, getting {self.ndim}-D")
    return self.unsqueeze(-1).pad_to((None, 1+(n:=self.shape[0]))).flatten().shrink_to((n*n,)).reshape(n,n)

  def diagonal(self, offset:int=0, dim1:int=0, dim2:int=1) -> Self:
    """
    Returns a view of the diagonal elements with respect to `dim1` and `dim2`.
    `offset` controls which diagonal: 0 is main, positive is above, negative is below.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(3, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.diagonal().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.diagonal(offset=1).numpy())
    ```
    """
    if (dim1:=self._resolve_dim(dim1)) == (dim2:=self._resolve_dim(dim2)): raise RuntimeError("dim1 and dim2 cannot be the same dimension")
    x = self.permute(*[i for i in range(self.ndim) if i != dim1 and i != dim2], dim1, dim2)
    if offset >= 0: x = x.shrink(tuple(None for _ in x.shape[:-1]) + ((offset, x.shape[-1]),))
    else: x = x.shrink(tuple(None for _ in x.shape[:-2]) + ((-offset, x.shape[-2]), None))
    if (d := min(int(x.shape[-2]), int(x.shape[-1]))) <= 0: return x.reshape(*x.shape[:-2], 0)
    nones, x = tuple(None for _ in x.shape[:-2]), x.shrink_to(tuple(None for _ in x.shape[:-2]) + (d, d))
    return x.flatten(-2).pad_to(nones+(d*(d+1),)).unflatten(-1, (d, d+1)).shrink_to(nones+(None, 1)).squeeze(-1)

  def roll(self, shifts:int|tuple[int, ...], dims:int|tuple[int, ...]|None=None) -> Self:
    """
    Rolls the tensor along specified dimension(s).
    The rolling operation is circular, meaning that elements that go beyond the edge are wrapped around to the beginning of the dimension.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(4)
    print(t.roll(shifts=1, dims=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.roll(shifts=-1, dims=0).numpy())
    ```
    """
    if dims is None: return self.flatten().roll(shifts, 0).reshape(self.shape)
    dims, shifts = tuple(self._resolve_dim(d) for d in make_tuple(dims, 1)), make_tuple(shifts, 1)
    if len(dims) != len(shifts): raise RuntimeError(f"{len(dims)=} != {len(shifts)=}")
    shrink_arg: list[tuple[sint, sint]|None] = [None] * self.ndim
    for d, s in zip(dims, shifts): shrink_arg[d] = (delta:=self.shape[d]-s%self.shape[d], delta+self.shape[d])
    return self.repeat(*tuple(2 if i in dims else 1 for i in range(self.ndim))).shrink(tuple(shrink_arg))

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

  def unfold(self, dim:int, size, step:int) -> Self:
    """
    Unfolds the tensor along dimension `dim` into overlapping windows.

    Each window has length `size` and begins every `step` elements of `self`.
    Returns the input tensor with dimension `dim` replaced by dims `(n_windows, size)`
    where `n_windows = (self.shape[dim] - size) // step + 1`.

    ```python exec="true" source="above" session="tensor" result="python"
    unfolded = Tensor.arange(8).unfold(0,2,2)
    print("\\n".join([repr(x.numpy()) for x in unfolded]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    unfolded = Tensor.arange(27).reshape(3,3,3).unfold(-1,2,3)
    print("\\n".join([repr(x.numpy()) for x in unfolded]))
    ```
    """
    if size < 0: raise RuntimeError(f'size must be >= 0 but got {size=}')
    if step <= 0: raise RuntimeError(f'step must be > 0 but got {step=}')
    if size > self.shape[dim]: raise RuntimeError(f'maximum size for tensor at dimension {dim} is {self.shape[dim]} but size is {size}')
    dim = self._resolve_dim(dim)
    perm_to_last = tuple(i for i in range(self.ndim) if i != dim) + (dim,)
    return self.permute(perm_to_last)._pool((size,), step).permute(argsort(perm_to_last) + (self.ndim,))
