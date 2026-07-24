from __future__ import annotations
import functools, itertools, math, string
from typing import TYPE_CHECKING, Callable, Self, Sequence, Literal, get_args
from tinygrad.mixin.elementwise import ElementwiseMixin
from tinygrad.mixin.movement import MovementMixin
from tinygrad.mixin.reduce import ReduceMixin
from tinygrad.uop import Ops
from tinygrad.uop.ops import _broadcast_shape, resolve, smax, smin, identity_element
from tinygrad.dtype import ConstType, DType, DTypeLike, Invalid, PyConst, dtypes, least_upper_dtype, sum_acc_dtype, to_dtype
from tinygrad.helpers import all_int, argfix, argsort, ceildiv, flatten, flat_to_grouped, fully_flatten, get_shape, make_tuple, merge_dicts, prod
from tinygrad.helpers import resolve_pool_pads, round_up, IMAGE, FLOAT16, WINO

if TYPE_CHECKING:
  from tinygrad.uop.ops import sint, UOp

ReductionStr = Literal["mean", "sum", "none"]


class OpMixin(ElementwiseMixin, ReduceMixin):
  def data(self) -> memoryview: raise NotImplementedError("data requires Tensor realization to host memory")

  def item(self) -> PyConst:
    """
    Returns the value of this tensor as a standard Python number.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor(42)
    print(t.item())
    ```
    """
    assert self.numel() == 1, "must have one element for item"
    return self.data()[(0,) * len(self.shape)]

  def __getitem__(self, indices) -> Self:
    """
    Retrieves a sub-tensor using indexing.

    Supported Index Types: `int | slice | Tensor | None | list | tuple | Ellipsis`

    Examples:
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(12).reshape(3, 4)
    print(t.numpy())
    ```

    - Int Indexing: Select an element or sub-tensor using integers for each dimension.
      ```python exec="true" source="above" session="tensor" result="python"
      print(t[1, 2].numpy())
      ```

    - Slice Indexing: Select a range of elements using slice notation (`start:end:stride`).
      ```python exec="true" source="above" session="tensor" result="python"
      print(t[0:2, ::2].numpy())
      ```

    - Tensor Indexing: Use another tensor as indices for advanced indexing. Using `tuple` or `list` here also works.
      ```python exec="true" source="above" session="tensor" result="python"
      print(t[Tensor([2, 0, 1]), Tensor([1, 2, 3])].numpy())
      ```

    - `None` Indexing: Add a new dimension to the tensor.
      ```python exec="true" source="above" session="tensor" result="python"
      print(t[:, None].shape)
      ```

    NOTE: Out-of-bounds indexing results in a value of `0`.
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t[Tensor([4, 3, 2])].numpy())
    ```
    """
    return self._getitem(indices)

  def _getitem(self, indices, v=None) -> Self:
    from tinygrad.uop.ops import UOp
    def is_adv(i): return isinstance(i,(list,tuple)) or (isinstance(i,type(self)) and (not isinstance(i,UOp) or i.shape != ()))
    # wrap single index into a list
    if (isinstance(indices, list) and all_int(indices)) or not isinstance(indices, (tuple, list)): indices = [indices]
    indices_parsed, dim = [], 0
    for index in self._normalize_indices(list(indices)):
      size = 1 if index is None else self.shape[dim]
      parsed = {"size":size, "boundary":(0, size), "stride":1, "collapse_dim":False}
      if isinstance(index,(list,tuple)):
        flat = fully_flatten(index)
        inferred = dtypes.bool if (flat and all(isinstance(s,bool) for s in flat)) else \
          (dtypes.default_int if flat and all_int(flat) else dtypes.default_float)
        if not dtypes.is_int(inferred): raise IndexError(f"{index=} contains non-int element")
        index = self._wrap_uop(UOp._frompy([i+size if i<0 else i for i in flat], inferred, self.device)).reshape(get_shape(index))
      elif is_adv(index):
        if not dtypes.is_int(index.dtype): raise IndexError(f"index dtype {index.dtype} is not supported")
        if index.device is not None and self.device is not None and index.device != self.device:
          raise RuntimeError(f"expected index and self on the same device, {index.device=}, {self.device=}")
        assert isinstance(size, int), "size must be an int"
        index = (index < 0).where(index+size, index)  # treat negative index values
      else: parsed = self._parse_view_index(index, size)
      indices_parsed.append({**parsed, "index":index})
      if index is not None: dim += 1

    # apply view ops then dim injection (None) and collapse (int)
    x = self._apply_view_ops(mops := [p for p in indices_parsed if p["index"] is not None])
    x_dims = [p for p in indices_parsed if not p["collapse_dim"]]
    x = x.reshape(tuple(p["size"] for p in x_dims))

    # tensor indexing
    if tops := [(d, p) for d, p in enumerate(x_dims) if is_adv(p['index'])]:
      dims, tensors, masks = [d for d, _ in tops], [p['index'] for _, p in tops], []
      big_shape = _broadcast_shape(*(t.shape for t in tensors))

      # consecutive tensor indices with int shapes: use linear indexing instead of one-hot masks
      consecutive = dims == list(range(dims[0], dims[0] + len(dims)))
      if v is None and len(dims) > 1 and consecutive and all_int(ishp := tuple(x.shape[d] for d in dims)):
        strides = tuple(prod(ishp[i+1:]) for i in range(len(dims)))
        try: linear_idx = type(self).usum(*[t._broadcast_to(big_shape) * s for t, s in zip(tensors, strides)])
        except ValueError as err: raise IndexError(f"cannot broadcast indices: {err}") from err
        valid = type(self).uprod(*[(t >= 0) & (t < s) for t, s in zip(tensors, ishp)])
        pre, post = x.shape[:dims[0]], x.shape[dims[-1]+1:]
        x = x.reshape(pre + (prod(ishp),) + post)[tuple([slice(None)] * len(pre)) + (valid.where(linear_idx, 0),)]
        return valid.reshape((1,) * len(pre) + big_shape + (1,) * len(post)).where(x, 0)

      pre_reduce_shape = x.shape[:dims[0]] + big_shape + x.shape[dims[0]:]

      # create index masks
      for dim, tensor in zip(dims, tensors):
        try: i = tensor.reshape(tensor.shape + (1,)*(x.ndim - dims[0])).expand(pre_reduce_shape)
        except ValueError as err: raise IndexError(f"cannot broadcast indices: {err}") from err
        masks.append(i._one_hot_along_dim(num_classes=x.shape[dim], dim=(dim - x.ndim)))

      # reduce masks to 1 mask
      mask = type(self).uprod(*masks)

      # inject 1's for the extra dims added in create masks
      reshape_arg = x.shape[:dims[0]] + (1,) * len(big_shape) + x.shape[dims[0]:]
      # sum reduce the extra dims introduced in create masks
      x_pre = x  # save collapsed shape for advanced setitem
      x = (mask.where(x.reshape(reshape_arg), 0)).sum(sum_axis:=tuple(d + len(big_shape) for d in dims), dtype=x.dtype)

      # special permute case
      if (permuted := dims[0] != 0 and len(dims) != 1 and tuple(dims) != tuple(range(dims[0], dims[-1]+1))):
        mask, x = (y.permute(*range(dims[0], dims[0]+len(big_shape)), *range(0, dims[0]), *range(dims[0]+len(big_shape), y.ndim)) for y in (mask, x))

      if v is None: return x  # advanced getitem
      # advanced setitem: resolve tensor dims in collapsed space, then fall through to basic setitem path
      vb = v.cast(self.dtype)._broadcast_to(_broadcast_shape(x.shape, v.shape))
      for dim in sum_axis: vb = vb.unsqueeze(dim)  # add back reduced dims from sum
      start = dims[0] if not permuted else 0
      vb = x_pre._masked_merge(vb, mask, tuple(range(start, start + len(big_shape))))
    elif v is None: return x  # basic getitem
    # basic setitem: broadcast v, reshape to self.ndim (unsqueeze int dims, squeeze None dims)
    else: vb = v.cast(self.dtype)._broadcast_to(x.shape)
    vb = vb.reshape(tuple(1 if p['collapse_dim'] else p['size'] for p in indices_parsed if p['index'] is not None))
    per_dim = []
    for d, m in enumerate(mops):
      (s, e), st = m['boundary'], abs(m['stride'])
      if st != 1 and vb.shape[d] > 1:  # un-stride: interleave with zeros
        vb = vb.unsqueeze(d+1)
        vb = vb.pad_to(tuple(st if j == d+1 else None for j in range(vb.ndim)))
        vb = vb.reshape(vb.shape[:d] + (vb.shape[d]*vb.shape[d+1],) + vb.shape[d+2:])
        vb = vb.shrink_to(tuple(e-s if j == d else None for j in range(self.ndim)))
      idx = type(self).arange(self.shape[d]).reshape([1]*d + [self.shape[d]] + [1]*(self.ndim - d - 1))
      per_dim.append((idx >= s) & (idx < e) & (((e-1-idx) if m['stride'] < 0 else (idx-s)) % st == 0))
    vb = vb.flip(tuple(d for d, m in enumerate(mops) if m['stride'] < 0))
    vb = vb.pad(tuple((m['boundary'][0], self.shape[d] - m['boundary'][1]) for d, m in enumerate(mops)))
    return (type(self).uprod(*per_dim) if per_dim else type(self).const(dtypes.bool, True)).where(vb, self)

  @classmethod
  def arange(cls, start, stop=None, step=1, dtype:DTypeLike|None=None) -> Self:
    """
    Returns a 1-D tensor of size `ceil((stop - start) / step)` with values from `[start, stop)`, with spacing between values given by `step`.

    If `stop` is not specified, values are generated from `[0, start)` with the given `step`.

    If `stop` is specified, values are generated from `[start, stop)` with the given `step`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5, 10).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5, 10, 2).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5.5, 10, 2).numpy())
    ```
    """
    if stop is None: stop, start = start, 0
    if dtype is None: dtype = dtypes.default_float if any(isinstance(x, float) for x in (start, stop, step)) else dtypes.default_int
    lo, hi = (start, stop-step) if step > 0 else (stop-step, start)
    if lo < (dt:=to_dtype(dtype)).min or dt.max < hi: raise OverflowError(f"arange [{start}, {stop}) is not representable in dtype {dtype}")
    # NOTE: this matches numpy, torch raises RuntimeError if stop-start and step have different signs
    if (output_len:=ceildiv(stop-start, step)) <= 0: return cls.full((0,), 0, dtype=dtype, buffer=False)
    return (cls.full((output_len,), step, dtype=dtype, buffer=False)._cumalu(0, Ops.ADD) + (start - step)).cast(dtype)

  @classmethod
  def linspace(cls, start:int|float, stop:int|float, steps:int, dtype:DTypeLike|None=None) -> Self:
    """
    Returns a 1-D tensor of `steps` evenly spaced values from `start` to `stop`, inclusive.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.linspace(0, 10, 5).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.linspace(-1, 1, 5).numpy())
    ```
    """
    if steps < 0: raise ValueError("number of steps must be non-negative")
    if (dtype := to_dtype(dtype or dtypes.default_float)) == dtypes.bool: raise ValueError("linspace with bool dtype is not supported")
    if steps == 1: return cls.full((1,), start, dtype=dtype, buffer=False)
    return (start + cls.arange(steps, dtype=dtypes.default_float) * ((stop - start) / (steps - 1))).cast(dtype)

  @classmethod
  def eye(cls, n:int, m:int|None=None, dtype:DTypeLike|None=None) -> Self:
    """
    Returns a 2-D tensor with `n` rows and `m` columns, with ones on the diagonal and zeros elsewhere.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.eye(3).numpy())
    ```

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.eye(2, 4).numpy())
    ```
    """
    m_ = n if m is None else m
    if n < 0 or m_ < 0: raise ValueError(f"cannot have negative {n=}, {m_=}")
    out_dtype = to_dtype(dtype) if dtype is not None else dtypes.default_float
    return cls.arange(n).unsqueeze(-1).eq(cls.arange(m_)).cast(out_dtype)

  @classmethod
  def _tri(cls, r:sint, c:sint, diagonal=0) -> Self:
    return cls.arange(r).unsqueeze(-1) + diagonal <= cls.arange(c)

  def triu(self, diagonal:sint=0) -> Self:
    """
    Returns the upper triangular part of the tensor, the other elements are set to 0.

    The argument `diagonal` determines which diagonal is on the boundary. `diagonal = 0` means the main diagonal.
    Positive `diagonal` means above the main diagonal, and negative `diagonal` means below the main diagonal.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.triu(diagonal=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.triu(diagonal=1).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.triu(diagonal=-1).numpy())
    ```
    """
    return self._tri(self.shape[-2], self.shape[-1], diagonal).where(self, self.const_like(0))

  def tril(self, diagonal:sint=0) -> Self:
    """
    Returns the lower triangular part of the tensor, the other elements are set to 0.

    The argument `diagonal` determines which diagonal is on the boundary. `diagonal = 0` means the main diagonal.
    Positive `diagonal` means above the main diagonal, and negative `diagonal` means below the main diagonal.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.tril(diagonal=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.tril(diagonal=1).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.tril(diagonal=-1).numpy())
    ```
    """
    return self._tri(self.shape[-2], self.shape[-1], diagonal+1).where(self.const_like(0), self)

  def _pad_constant(self, pX, value:ConstType) -> Self:
    # shrink first for negative pads, then pad with only non-negative values
    pX = tuple((0, 0) if p is None else p for p in pX)
    has_neg = not all(resolve(p >= 0) for p in flatten(pX))
    X = self.shrink(tuple((-smin(pB,0),smin(pA+s,s)) for (pB,pA),s in zip(pX, self.shape))) if has_neg else self
    pads = tuple((smax(pB,0), smax(pA,0)) for pB,pA in pX) if has_neg else pX
    base = MovementMixin.pad(X, pads)
    if value == 0: return base
    if value is not Invalid: base = base.cast(least_upper_dtype(base.dtype, dtypes.from_py(value)))
    return MovementMixin.pad(X.const_like(1).cast(dtypes.bool), pads).where(base, base.const_like(value))

  def _pad_circular(self, pX:tuple[tuple[sint, sint], ...]) -> Self:
    # shrink first for negative pads, then wrap the non-negative remainder
    X = self.shrink(tuple((-smin(pB,0), smin(pA+sh,sh)) for (pB,pA),sh in zip(pX, self.shape)))
    pX = tuple((smax(pB,0), smax(pA,0)) for pB,pA in pX)
    if any(pB>sh or pA>sh for (pB,pA),sh in zip(pX, X.shape)): raise ValueError('Padding value causes wrapping around more than once.')
    orig_shape, X = X.shape, X.repeat(tuple(1 + bool(pB) + bool(pA) for pB,pA in pX))
    return X.shrink(tuple((0 if pB == 0 else osh-pB, xsh if pA == 0 else xsh-osh+pA) for (pB,pA),osh,xsh in zip(pX, orig_shape, X.shape)))

  def _pad_reflect_replicate(self, pX:tuple[tuple[sint, sint], ...], mode:str) -> Self:
    X, pads = self, tuple((smax(pB,0), smax(pA,0)) for pB,pA in pX)
    for d,(pB,pA) in enumerate(pads):
      if mode == "reflect":
        if pB >= (s:=X.shape[d]) or pA>=s: raise ValueError(f"Padding ({pB}, {pA}) should be less than the input size={s} for dim={d}.")
        slcB, slcA = slice(pB,0,-1), slice(s-2, s-2-pA if s-2-pA>=0 else None, -1)
        xB, xA = (X[[slc if i == d else slice(None) for i in range(X.ndim)]] if p > 0 else None for slc, p in ((slcB, pB), (slcA, pA)))
      else:
        shrB, shrA = tuple((0,1) if i==d else None for i in range(X.ndim)), tuple((X.shape[i]-1,X.shape[i]) if i==d else None for i in range(X.ndim))
        xB, xA = (X.shrink(shr).expand(tuple(p if i==d else None for i in range(X.ndim))) if p > 0 else None for shr, p in ((shrB, pB), (shrA, pA)))
      pieces = [X_ for X_ in (xB, X, xA) if X_ is not None]
      X = pieces[0].cat(*pieces[1:], dim=d)
    # shrink after for negative pads (reflection/replication must see full data first)
    return X.shrink(tuple((-min(pB,0), min(pA+s,s)) for (pB,pA),s in zip(pX, X.shape)))

  def pad(self, padding:Sequence[sint]|Sequence[tuple[sint, sint]|None], mode:str="constant", value:ConstType=0.0) -> Self:
    """
    Returns a tensor with padding applied based on the input `padding`.

    `padding` supports two padding structures:

    1. Flat padding: `(padding_left, padding_right, padding_top, padding_bottom, ...)`
        - This structure matches PyTorch's pad.
        - `padding` length must be even.

    2. Group padding: `(..., (padding_top, padding_bottom), (padding_left, padding_right))`
        - This structure matches pad for JAX, NumPy, TensorFlow, and others.
        - For each axis, padding can be `None`, meaning no padding, or a tuple `(start, end)`.
        - `padding` must have the same length as `self.ndim`.

    Padding values can be negative, resulting in dimension shrinks that work similarly to Python negative slices.
    Padding modes is selected with `mode` which supports `constant`, `reflect` and `replicate`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(1, 1, 3, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.pad((1, 2, 0, -1)).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.pad(((None, None, (0, -1), (1, 2)))).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.pad((1, 2, 0, -1), value=-float('inf')).numpy())
    ```
    """
    # normalize to grouped format
    pX: tuple[tuple[sint, sint], ...]
    if not any(isinstance(p, (tuple, type(None))) for p in padding):
      if len(padding)%2 != 0: raise ValueError("Flat padding must have even number of pads")
      pX = ((0,0),)*(self.ndim - len(padding)//2) + flat_to_grouped(padding)  # type: ignore[arg-type]
    else: pX = tuple((0,0) if p is None else p for p in padding)  # type: ignore[misc]
    if len(pX) != self.ndim: raise ValueError(f"padding length is improper, {padding=} {self.ndim=}")
    # dispatch
    if mode == "constant": return self._pad_constant(pX, value)
    assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
    if mode == "circular": return self._pad_circular(pX)
    if mode in {"reflect", "replicate"}: return self._pad_reflect_replicate(pX, mode)
    raise NotImplementedError(f"{mode=} is not supported")

  def _broadcasted(self, y:Self|ConstType|UOp, reverse:bool=False) -> tuple[Self, Self]:
    if not isinstance(y, type(self)): y = self.ufix(y)
    x, y = (self, y) if not reverse else (y, self)
    out_shape = _broadcast_shape(x.shape, y.shape)
    x, y = x._broadcast_to(out_shape), y._broadcast_to(out_shape)
    if x.dtype == y.dtype: return x, y
    return x.cast(out_dtype := least_upper_dtype(x.dtype, y.dtype)), y.cast(out_dtype)

  def dot(self, w:Self, dtype:DTypeLike|None=None) -> Self:
    """
    Performs dot product between two tensors.
    If `w` is 1-D, it's a sum product over the last axis of `self` and `w`.
    If `w` is N-D with N>=2, it's a sum product over the last axis of `self` and the second-to-last axis of `w`.

    You can pass in the optional `dtype` keyword argument to control the data type of the accumulation.

    ```python exec="true" source="above" session="tensor" result="python"
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 0])
    print(a.dot(b).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    print(a.dot(b).numpy())
    ```
    """
    if IMAGE: return self.image_dot(w, dtype)
    x, dx, dw = self, self.ndim, w.ndim
    if not (dx > 0 and dw > 0): raise RuntimeError(f"both tensors need to be at least 1D, got {dx}D and {dw}D")
    if x.shape[-1] != w.shape[axis_w:=-min(w.ndim,2)]: raise RuntimeError(f"cannot dot {x.shape} and {w.shape}")
    x = x.reshape(*x.shape[0:-1], *[1]*min(dx-1, dw-1, 1), x.shape[-1])
    w = w.reshape(*w.shape[0:-2], *[1]*min(dx-1, dw-1, 1), *w.shape[axis_w:]).transpose(-1, axis_w)
    return (x*w).sum(-1, dtype=dtype).cast(least_upper_dtype(x.dtype, w.dtype) if dtype is None else to_dtype(dtype))

  def matmul(self, x:Self, reverse=False, dtype:DTypeLike|None=None) -> Self:
    """
    Performs matrix multiplication between two tensors.

    You can pass in the `reverse` keyword argument to control the order of the matrix multiplication.
    You can pass in the optional `dtype` keyword argument to control the data type of the accumulation.

    ```python exec="true" source="above" session="tensor" result="python"
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    print(a.matmul(b).numpy())
    ```
    """
    return x.dot(self, dtype=dtype) if reverse else self.dot(x, dtype=dtype)

  def __matmul__(self, x:Self) -> Self: return self.matmul(x)
  def __rmatmul__(self, x:Self) -> Self: return self.matmul(x, True)

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
          j, k, n = s.index(c), s.index(c, s.index(c)+1), x.shape[s.index(c)]
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

  def gradient(self, *targets:Self, gradient:Self|None=None) -> list[Self]:
    """
    Computes the gradient of the targets with respect to self.

    ```python exec="true" source="above" session="tensor" result="python"
    x = Tensor.eye(3)
    y = Tensor([[2.0,0,-2.0]])
    z = y.matmul(x).sum()
    dx, dy = z.gradient(x, y)

    print(dx.tolist())  # dz/dx
    print(dy.tolist())  # dz/dy
    ```
    """
    assert gradient is not None or self.shape == tuple(), "when no gradient is provided, backward must be called on a scalar tensor"
    if not (self.is_floating_point() and all(t.is_floating_point() for t in targets)): raise RuntimeError("only float Tensors have gradient")
    from tinygrad.mixin.gradient import compute_gradient
    if gradient is None: gradient = self.const_like(1.0)
    target_uops = [t._uop for t in targets]
    grads = compute_gradient(self._uop, gradient._uop, set(target_uops))
    return [self._wrap_uop(grads[x] if x in grads else x.const_like(0)) for x in target_uops]

  def min(self, axis:int|Sequence[int]|None=None, keepdim=False) -> Self:
    """
    Returns the minimum value of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the minimum is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 0, 2], [5, 4, 3]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.min().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.min(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.min(axis=1, keepdim=True).numpy())
    ```
    """
    return self._inverse().max(axis=axis, keepdim=keepdim)._inverse()

  def mean(self, axis:int|Sequence[int]|None=None, keepdim=False) -> Self:
    """
    Returns the mean value of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the mean is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mean().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mean(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mean(axis=1).numpy())
    ```
    """
    output_dtype = self.dtype if dtypes.is_float(self.dtype) else dtypes.float32
    numerator = self.cast(sum_acc_dtype(self.dtype)).sum(axis=axis, keepdim=keepdim)
    denominator = prod([si for si, so in zip(self.shape, self.sum(axis=axis, keepdim=True).shape) if resolve(si != so)])
    return numerator.div(denominator).cast(output_dtype)  # type: ignore[arg-type]

  def var(self, axis:int|Sequence[int]|None=None, keepdim=False, correction=1) -> Self:
    """
    Returns the variance of the tensor along the specified axis or axes.

    You can pass in `axis`, `keepdim`, and `correction` keyword arguments to control the axis along
    which the variance is computed, whether the reduced dimensions are retained, and the Bessel's correction applied.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.var().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.var(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.var(axis=1).numpy())
    ```
    """
    squares = (self - self.mean(axis=axis, keepdim=True)).square()
    n = prod([si for si, so in zip(self.shape, squares.sum(axis=axis, keepdim=True).shape) if resolve(si != so)])
    reduced = squares.sum(axis=axis, keepdim=keepdim)
    denominator = reduced.const_like(n) - correction  # type: ignore[arg-type]
    # TODO: remove relu?
    return reduced.div(denominator.relu())

  def var_mean(self, axis:int|Sequence[int]|None=None, keepdim=False, correction=1) -> tuple[Self, Self]:
    """
    Calculates the variance and mean over the dimensions specified by dim.
    Syntactic sugar around `Tensor.var` and `Tensor.mean` to match `torch.var_mean`.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    var, mean = t.var_mean()
    print(var.numpy(), mean.numpy())
    ```
    """
    return self.var(axis, keepdim, correction), self.mean(axis, keepdim)

  def std(self, axis:int|Sequence[int]|None=None, keepdim=False, correction=1) -> Self:
    """
    Returns the standard deviation of the tensor along the specified axis or axes.

    You can pass in `axis`, `keepdim`, and `correction` keyword arguments to control the axis along
    which the standard deviation is computed, whether the reduced dimensions are retained, and the Bessel's correction applied.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.std().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.std(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.std(axis=1).numpy())
    ```
    """
    return self.var(axis, keepdim, correction).sqrt()

  def std_mean(self, axis:int|Sequence[int]|None=None, keepdim=False, correction=1) -> tuple[Self, Self]:
    """
    Calculates the standard deviation and mean over the dimensions specified by dim.
    Syntactic sugar around `Tensor.std` and `Tensor.mean` to match `torch.std_mean`.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    std, mean = t.std_mean()
    print(std.numpy(), mean.numpy())
    ```
    """
    return self.std(axis, keepdim, correction), self.mean(axis, keepdim)

  def normalize(self, p:float=2.0, dim:int=1, eps:float=1e-12) -> Self:
    """
    Performs Lp normalization of the tensor along the specified dimension.

    See: https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.normalize().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.normalize(p=1, dim=0).numpy())
    ```
    """
    if p == 0: return self / self.ne(0).sum(dim, keepdim=True).maximum(eps)
    return self / self.abs().pow(p).sum(dim, keepdim=True).pow(1/p).maximum(eps)

  def logsumexp(self, axis=None, keepdim=False) -> Self:
    """
    Computes the log-sum-exp of the tensor along the specified axis or axes.

    The log-sum-exp function is a numerically stable way to compute the logarithm of the sum of exponentials.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the log-sum-exp is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logsumexp().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logsumexp(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logsumexp(axis=1).numpy())
    ```
    """
    m = self.max(axis=axis, keepdim=True).detach()
    return (self - m).exp().sum(axis=axis, keepdim=keepdim).log() + (m if keepdim else m.squeeze(axis))

  def _softmax(self, axis, dtype:DTypeLike|None=None) -> tuple[Self, Self, Self]:
    m = self - self.max(axis=axis, keepdim=True).detach()
    if dtype is not None: m = m.cast(to_dtype(dtype))
    e = m.exp()
    return m, e, e.sum(axis=axis, keepdim=True)

  def softmax(self, axis=-1, dtype:DTypeLike|None=None) -> Self:
    """
    Applies the softmax function to the tensor along the specified axis.

    Rescales the elements of the tensor such that they lie in the range [0, 1] and sum to 1.

    You can pass in the `axis` keyword argument to control the axis along which the softmax is computed.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.softmax().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.softmax(axis=0).numpy())
    ```
    """
    _, e, ss = self._softmax(axis, dtype)
    return e * ss.reciprocal()

  def log_softmax(self, axis=-1, dtype:DTypeLike|None=None) -> Self:
    """
    Applies the log-softmax function to the tensor along the specified axis.

    The log-softmax function is a numerically stable alternative to the softmax function in log space.

    You can pass in the `axis` keyword argument to control the axis along which the log-softmax is computed.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.log_softmax().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.log_softmax(axis=0).numpy())
    ```
    """
    m, _, ss = self._softmax(axis, dtype)
    return m - ss.log()

  def cat(self, *args:Self, dim:int=0) -> Self:
    """
    Concatenates self with other tensors in `args` along an axis specified by `dim`.
    All tensors must have the same shape except in the concatenating dimension.

    ```python exec="true" source="above" session="tensor" result="python"
    t0, t1, t2 = Tensor([[1, 2]]), Tensor([[3, 4]]), Tensor([[5, 6]])
    print(t0.cat(t1, t2, dim=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t0.cat(t1, t2, dim=1).numpy())
    ```
    """
    dim = self._resolve_dim(dim)
    for arg in args: assert arg.ndim==self.ndim and all(ti==ai for i,(ti,ai) in enumerate(zip(self.shape, arg.shape)) if i!=dim)
    tensors = [self, *args]
    dim_cumsum = list(itertools.accumulate([t.shape[dim] for t in tensors], initial=0))
    padded = [t.pad(tuple((dim_cumsum[i], dim_cumsum[-1]-dim_cumsum[i+1]) if j==dim else None for j in range(t.ndim))) for i,t in enumerate(tensors)]
    return padded[0].usum(*padded[1:])

  def _cumalu(self, axis:int, op:Ops) -> Self:
    assert self.shape[axis] != 0 and op in (Ops.ADD, Ops.MAX, Ops.MUL)
    pads = (None,)*(self.ndim-1) + ((self.shape[axis]-1, 0),)
    pooled = self.transpose(axis,-1)._pad_constant(pads, identity_element(op, self.dtype))._pool((self.shape[axis],))
    return getattr(pooled, {Ops.ADD: "sum", Ops.MAX: "max", Ops.MUL: "prod"}[op])(-1).transpose(axis, -1)

  def _split_cumalu(self, axis:int, op:Ops) -> Self:
    axis = self._resolve_dim(axis)
    if self.ndim == 0 or 0 in self.shape: return self.cast(self.sum().dtype) if op is Ops.ADD else self
    # TODO: someday the optimizer will find this on its own
    # for now this is a two stage cumsum
    SPLIT = 256
    value = identity_element(op, self.dtype)
    if not isinstance(s:=self.shape[axis], int) or s <= SPLIT*2: return self._cumalu(axis, op)
    ret = self.transpose(axis,-1)._pad_constant((None,)*(self.ndim-1)+((round_up(s,SPLIT)-s,0),), value).unflatten(-1,(-1,SPLIT))._cumalu(-1, op)
    base = ret[..., -1]._cumalu(-1, op)._pad_constant((None,)*(ret.ndim-2) + ((1, -1),), value)
    base = base.unsqueeze(-1).expand(*base.shape, ret.shape[-1])
    def fix(x: Self) -> Self: return x.flatten(start_dim=-2)[..., -s:].transpose(axis,-1)
    return fix(ret).alu(op, fix(base))

  def cumsum(self, axis:int=0) -> Self:
    """
    Computes the cumulative sum of the tensor along the specified `axis`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.cumsum(1).numpy())
    ```
    """
    return self._split_cumalu(axis, Ops.ADD)

  def cumprod(self, axis:int) -> Self:
    """
    Computes the cumulative product of the elements of the tensor along the specified `axis`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(1, 7).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.cumprod(axis=0).numpy())
    ```
    """
    return self._split_cumalu(axis, Ops.MUL)

  def cummax(self, axis:int=0) -> tuple[Self, Self]:
    """
    Computes the cumulative max of the tensor along `axis`, returning (values, indices).

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([0, 1, -1, 2, -2, 3, -3])
    values, indices = t.cummax(0)
    print(values.numpy())
    print(indices.numpy())
    ```
    """
    if self.ndim == 0: return self._split_cumalu(axis, Ops.MAX), type(self).zeros(self.shape, dtype=dtypes.int32, buffer=False)
    values, n = self._split_cumalu(axis, Ops.MAX), int(self.shape[axis])
    x, values_t = self.transpose(axis, -1), values.transpose(axis, -1)
    match = x.unsqueeze(-1).eq(values_t.unsqueeze(-2)) * type(self).ones(n, n, buffer=False).triu()
    idx = (-(match * type(self).arange(n, 0, -1).reshape(n, 1)).max(-2) + n).cast(dtypes.int32)
    return values, idx.transpose(-1, axis)

  def cummin(self, axis:int=0) -> tuple[Self, Self]:
    """
    Computes the cumulative min of the tensor along `axis`, returning (values, indices).

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([0, 1, -1, 2, -2, 3, -3])
    values, indices = t.cummin(0)
    print(values.numpy())
    print(indices.numpy())
    ```
    """
    values, indices = self._inverse().cummax(axis)
    return values._inverse(), indices

  def logcumsumexp(self, axis=0) -> Self:
    """
    Computes the log-cumsum-exp of the tensor along the specified axis or axes.

    The log-cumsum-exp function is a numerically stable way to compute the logarithm of the cumulative sum of exponentials.

    You can pass in the `axis` keyword argument to control the axis along which
    the log-cumsum-exp is computed.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logcumsumexp().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logcumsumexp(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logcumsumexp(axis=1).numpy())
    ```
    """
    if self.ndim == 0: return self
    x = self.transpose(axis, -1)
    last_dim_size = x.shape[-1]
    x_unsqueezed = x.unsqueeze(-2).expand((None,)*(self.ndim-1)+(last_dim_size, None))
    x_cummax = x.cummax(-1)[0].detach()
    mask = type(self).ones(last_dim_size, last_dim_size, buffer=False, dtype=dtypes.bool).tril()
    ret = mask.where(x_unsqueezed - x_cummax.unsqueeze(-1), self.dtype.min).exp().sum(-1).log() + x_cummax
    return ret.transpose(-1, axis)

  def argmax(self, axis=None, keepdim=False) -> Self:
    """
    Returns the indices of the maximum value of the tensor along the specified axis.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the maximum is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 0, 2], [5, 4, 3]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmax().numpy()) # Returns the index of the maximum value in the flattened tensor.
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmax(axis=0).numpy()) # Returns the indices of the maximum values along axis 0.
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmax(axis=1).numpy()) # Returns the indices of the maximum values along axis 1.
    ```
    """
    if axis is None: return self.flatten().argmax(0)
    axis = self._resolve_dim(axis)
    m = self.eq(self.max(axis=axis, keepdim=True))
    idx = m * type(self).arange(self.shape[axis], 0, -1).reshape(self.shape[axis], *[1]*(self.ndim-axis-1))
    return (self.shape[axis] - idx.max(axis=axis, keepdim=keepdim)).cast(dtypes.int32)

  def argmin(self, axis=None, keepdim=False) -> Self:
    """
    Returns the indices of the minimum value of the tensor along the specified axis.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the minimum is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 0, 2], [5, 4, 3]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmin().numpy()) # Returns the index of the minimum value in the flattened tensor.
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmin(axis=0).numpy()) # Returns the indices of the minimum values along axis 0.
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmin(axis=1).numpy()) # Returns the indices of the minimum values along axis 1.
    ```
    """
    return self._inverse().argmax(axis=axis, keepdim=keepdim)

  def sort(self, dim:int=-1, descending:bool=False) -> tuple[Self, Self]:
    """
    Performs a bitonic sort on the tensor along the specified dimension.

    Order of indices for equivalent elements is always preserved.

    See: https://en.wikipedia.org/wiki/Bitonic_sorter

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[0.1, 0.5, 1.2, 3.4, 2.1], [2.2, 1.9, 0.3, 4.5, 0.8]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    sorted_values, indices = t.sort(dim=1, descending=True)
    print(sorted_values.numpy())
    print(indices.numpy())
    ```
    """
    x, dim = self, self._resolve_dim(dim)
    if (orig_len := int(x.shape[dim])) <= 1: return x, x.const_like(0).cast(dtypes.default_int)
    # pad to power of 2
    n_stages = (orig_len-1).bit_length()
    pads = tuple((0, 2**n_stages - orig_len) if i == dim else None for i in range(x.ndim))
    x = x._pad_constant(pads, x.dtype.min if descending else x.dtype.max).unflatten(dim, (2,)*n_stages)
    # https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort1.svg
    for stage in range(1, n_stages+1):
      if stage != n_stages:
        # flip so arrows of green boxes point the same way as blue boxes
        crossover_dim = dim + n_stages - stage - 1
        blue_box, green_box = x.split(1, crossover_dim)
        flip_dims = tuple(-i for i in range(1, stage+1+(self.ndim-dim)))
        x = (blue_box.cat(green_box.flip(flip_dims), dim=crossover_dim)).contiguous()
      for substage in range(stage-1, -1, -1):
        partner_dim = dim + n_stages - substage - 1
        x_top, x_bottom = x.split(1, partner_dim)
        x_larger, x_smaller = x_top.maximum(x_bottom), x_top.minimum(x_bottom)
        x = (x_larger.cat(x_smaller, dim=partner_dim) if descending else x_smaller.cat(x_larger, dim=partner_dim)).contiguous()
      if stage != n_stages:
        # flip wires back to undo the crossover
        blue_box, flipped_green_box = x.split(1, crossover_dim)
        x = blue_box.cat(flipped_green_box.flip(flip_dims), dim=crossover_dim)
    x = x.flatten(dim, dim+n_stages-1).shrink_to(self.shape)
    # compute indices for sorted values
    mask = type(self).ones(orig_len, orig_len, dtype=dtypes.bool, buffer=False).tril()
    mask = mask.reshape((None, None) + (1,)*(self.ndim-dim-1))
    def compute_counts(t:Self): return (mask & t.unsqueeze(dim).eq(t.unsqueeze(dim+1))).sum(dim+1)
    count_orig, count_sorted = compute_counts(self), compute_counts(x)
    cond = self.unsqueeze(dim+1).eq(x.unsqueeze(dim)) & count_orig.unsqueeze(dim+1).eq(count_sorted.unsqueeze(dim))
    idx = type(self).arange(orig_len).reshape(tuple(orig_len if i == dim else 1 for i in range(x.ndim)))
    idx = (cond * idx.unsqueeze(dim+1)).sum(dim)
    return x, idx

  def argsort(self, dim:int=-1, descending:bool=False) -> Self:
    """
    Returns the indices that sort input tensor along given `dimension` in given `descending` order by value.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[2, 3, 4, 1], [1, 4, 3, 2]])
    print(t.argsort().numpy())
    ```
    """
    return self.sort(dim, descending)[1]

  def topk(self, k:int, dim:int=-1, largest:bool=True, sorted_:bool=True) -> tuple[Self, Self]:
    """
    Computes the top-k elements of the tensor along the specified `dim`.

    Order of indices for equivalent elements is always preserved.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[0.1, 0.5, 1.2, 3.4, 2.1], [2.2, 1.9, 0.3, 4.5, 0.8]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    topk_values, topk_indices = t.topk(2, dim=1)
    print(topk_values.numpy())
    print(topk_indices.numpy())
    ```
    """
    if not sorted_: raise NotImplementedError("topk with sorted_=False is not supported")
    if k > self.shape[dim:=self._resolve_dim(dim)]: raise ValueError(f"selected index {k=} is out of range")
    x, idx = self.sort(dim, descending=largest)
    topk_shape = tuple(k if i == dim else None for i in range(self.ndim))
    return x.shrink_to(topk_shape), idx.shrink_to(topk_shape)

  def allclose(self, other:Self, rtol:float=1e-05, atol:float=1e-08, equal_nan=False) -> Self:
    """
    Check if all self and other are close.
    """
    return self.isclose(other, rtol=rtol, atol=atol, equal_nan=equal_nan).all()

  # helper function commonly used for indexing
  def _one_hot_along_dim(self, num_classes:sint, dim:int=-1) -> Self:
    from tinygrad.uop.ops import sint_to_uop
    if not dtypes.is_int(self.dtype): raise RuntimeError(f"_one_hot_along_dim expects int index tensor, getting {self.dtype}")
    offset = self.ndim - self._resolve_dim(dim) - 1
    dt = dtypes.int64 if sint_to_uop(num_classes).overflows(dtypes.int32) else dtypes.int32
    return self.eq(type(self).arange(num_classes, dtype=dt).reshape((num_classes,) + (1,) * offset))

  def one_hot(self, num_classes:int) -> Self:
    """
    Converts `self` to a one-hot tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([0, 1, 3, 3, 4])
    print(t.one_hot(5).numpy())
    ```
    """
    if not dtypes.is_int(self.dtype): raise RuntimeError(f"expect integer dtype, getting {self.dtype=}")
    if num_classes < 0: raise ValueError(f"num_classes must be non-negative, got {num_classes}")
    return self[..., None]._one_hot_along_dim(num_classes).where(1, 0)

  def gather(self, dim:int, index:Self) -> Self:
    """
    Gathers values along an axis specified by `dim`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2], [3, 4]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.gather(1, Tensor([[0, 0], [1, 0]])).numpy())
    ```
    """
    if index.device is not None and self.device is not None and index.device != self.device:
      raise RuntimeError(f"expected index and self on the same device, {index.device=}, {self.device=}")
    if index.ndim != self.ndim: raise RuntimeError(f"self.ndim must equal index.ndim, {self.ndim=}, {index.ndim=}")
    dim = self._resolve_dim(dim)
    assert all(s >= i for d,(s,i) in enumerate(zip(self.shape, index.shape)) if d != dim), "requires self.shape[d] >= index.shape[d] for all d != dim"
    x = self.shrink_to(tuple(i if d != dim else None for d,i in enumerate(index.shape))).unsqueeze(-1).transpose(-1, dim)
    return (index.unsqueeze(-1)._one_hot_along_dim(self.shape[dim]).where(x, 0)).sum(-1, dtype=self.dtype)

  def interpolate(self, size:tuple[int, ...], mode:str="linear", align_corners:bool=False) -> Self:
    """
    Downsamples or Upsamples to the input `size`, accepts 0 to N batch dimensions.

    The interpolation algorithm is selected with `mode` which currently only supports `linear`, `nearest` and `nearest-exact`.
    To run `bilinear` or `trilinear`, pass in a 2D or 3D size.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2, 3, 4], [21, 22, 23, 24], [41, 42, 43, 44]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.interpolate(size=(2,3), mode="linear").numpy())
    ```
    """
    assert isinstance(size, (tuple,list)) and all_int(size) and 0 < len(size) <= self.ndim, f"invalid {size=}"
    assert mode in ("linear", "nearest", "nearest-exact"), "only supports linear, nearest or nearest-exact interpolate"
    assert not (align_corners and mode != "linear"), "align_corners option can only be set with the interpolating mode linear"
    x, expand = self, list(self.shape)
    for i in range(-1,-len(size)-1,-1):
      scale = (int(self.shape[i]) - int(align_corners)) / (size[i] - int(align_corners))
      arr, reshape = type(self).arange(size[i], dtype=dtypes.float32), [1] * self.ndim
      reshape[i] = expand[i] = size[i]
      if mode == "linear":
        index = (scale*arr if align_corners else (scale*(arr+0.5))-0.5).clip(0, self.shape[i]-1)
        low, high, perc = [y.reshape(reshape).expand(expand) for y in (index.floor().int(), index.ceil().int(), index - index.floor())]
        x = x.gather(i, low).lerp(x.gather(i, high), perc)
      else:
        index = (scale*(arr+0.5) if mode=="nearest-exact" else scale*arr).cast(dtypes.int32).reshape(reshape).expand(expand)
        x = x.gather(i, index)
    return x.cast(self.dtype)

  def _pre_scatter(self, dim:int, index:Self, src:Self) -> tuple[Self, Self]:
    if index.device is not None and self.device is not None and index.device != self.device:
      raise RuntimeError(f"expected index and self on the same device, {index.device=}, {self.device=}")
    if src.device is not None and self.device is not None and src.device != self.device:
      raise RuntimeError(f"expected src and self on the same device, {src.device=}, {self.device=}")
    dim = self._resolve_dim(dim)
    assert index.ndim == self.ndim == src.ndim, f"self.ndim, index.ndim and src.ndim must all equal, {self.ndim=} {index.ndim=} {src.ndim=}"
    assert all((d == dim or self_ >= index_) and src_ >= index_ for d,(self_,index_,src_) in enumerate(zip(self.shape, index.shape, src.shape))), \
      f"All dimensions of {index.shape=} should be <= to all dimensions of {src.shape=} and all dimensions except dimension {dim} of {self.shape=}"
    if self.dtype != src.dtype: raise RuntimeError(f"expect {self.dtype=} to be equal to {src.dtype=}")
    # shrink src to index shape to shrink away the unused values
    src = src.shrink_to(index.shape)
    # prepare src and mask for reduce with respect to dim
    src = src.unsqueeze(-1).expand(*src.shape, self.shape[dim]).transpose(-1, dim)
    mask = index.unsqueeze(-1)._one_hot_along_dim(self.shape[dim]).transpose(-1, dim)
    # pad src and mask to self.shape so that reduce can be done with padded values as no-ops
    return src.pad_to(*self.shape, None), mask.pad_to(*self.shape, None)

  def scatter_reduce(self, dim:int, index:Self, src:Self, reduce:Literal["sum", "prod", "mean", "amax", "amin"],
                     include_self:bool=True) -> Self:
    """
    Scatters `src` values along an axis specified by `dim`.
    Apply `"sum"`, `"prod"`, `"mean"`, `"amax"`, or `"amin"` reduction operations with `reduce`.

    Set `include_self=False` to exclude values in the `self` Tensor from the reduction.

    ```python exec="true" source="above" session="tensor" result="python"
    src = Tensor.arange(1, 11).cast(dtypes.float).reshape(2, 5)
    print(src.numpy())
    index = Tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    print(index.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(1, 5, dtype=src.dtype).scatter_reduce(0, index, src, reduce='sum').numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(1, 5, dtype=src.dtype).scatter_reduce(0, index, src, reduce='prod').numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(1, 5, dtype=src.dtype).scatter_reduce(0, index, src, reduce='mean', include_self=False).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([[-10, 20, 0, 5, 10]], dtype=src.dtype).scatter_reduce(0, index, src, reduce='amax').numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([[-10, 20, 0, 5, 10]], dtype=src.dtype).scatter_reduce(0, index, src, reduce='amin').numpy())
    ```
    """
    src, mask = self._pre_scatter(dim, index, src)
    def _inv_mask(a:Self|PyConst, b:Self|PyConst) -> Self: return mask.any(-1).logical_not().where(a, b)
    if reduce == "sum": return mask.where(src, 0).sum(-1).add(self if include_self else _inv_mask(self, 0))
    if reduce == "prod": return mask.where(src, 1).prod(-1).mul(self if include_self else _inv_mask(self, 1))
    if reduce == "amax": return mask.where(src, m := src.dtype.min).max(-1).maximum(self if include_self else _inv_mask(self, m))
    if reduce == "amin": return mask.where(src, m := src.dtype.max).min(-1).minimum(self if include_self else _inv_mask(self, m))
    if reduce == "mean":
      count = mask.where(1, 0).sum(-1).add(1 if include_self else _inv_mask(1, 0))
      return mask.where(src, 0).sum(-1).add(self if include_self else _inv_mask(self, 0)).div(count)
    raise RuntimeError(f"{reduce=} must be one of 'sum', 'prod', 'mean', 'amax', 'amin'")

  def scatter(self, dim:int, index:Self, src:Self|PyConst, reduce:Literal['multiply', 'add']|None=None) -> Self:
    """
    Scatters `src` values along an axis specified by `dim`.
    Apply `add` or `multiply` reduction operation with `reduce`.

    NOTE: To use the `reduce` argument with a Tensor `src`, see `Tensor.scatter_reduce`.

    ```python exec="true" source="above" session="tensor" result="python"
    src = Tensor.arange(1, 11).reshape(2, 5)
    print(src.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    index = Tensor([[0, 1, 2, 0]])
    print(Tensor.zeros(3, 5, dtype=src.dtype).scatter(0, index, src).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    index = Tensor([[0, 1, 2], [0, 1, 4]])
    print(Tensor.zeros(3, 5, dtype=src.dtype).scatter(1, index, src).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 4), 2.0).scatter(1, Tensor([[2], [3]]), 1.23, reduce='multiply').numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 4), 2.0).scatter(1, Tensor([[2], [3]]), 1.23, reduce='add').numpy())
    ```
    """
    if reduce not in {None, "add", "multiply"}: raise TypeError(f"{reduce=} must be one of None, 'multiply', or 'add'")
    if isinstance(src, (int, float, bool)): src = type(self).full(index.shape, src, dtype=self.dtype, buffer=False)
    elif reduce: raise TypeError("non-scalar src is not supported with reduce arg. use scatter_reduce")
    if reduce == "add": return self.scatter_reduce(dim, index, src, "sum", include_self=True)
    if reduce == "multiply": return self.scatter_reduce(dim, index, src, "prod", include_self=True)
    src, mask = self._pre_scatter(dim, index, src)
    return self._masked_merge(src, mask, (-1,))

  def _masked_merge(self, values:Self, mask:Self, axes:tuple[int, ...]) -> Self:
    # reduce such that if mask contains repeated indices the last one remains
    for dim in reversed(axes):
      mask, values = functools.reduce(lambda x,y: (x[0]|y[0], y[0].where(y[1], x[1])), zip(mask.split(1, dim), values.split(1, dim)))
    # remove extra dims from reduce
    for dim in reversed(axes): mask, values = mask.squeeze(dim), values.squeeze(dim)
    # select from values for each True element in mask else select from self
    return mask.where(values, self)

  def masked_select(self, mask, size:int|None=None, fill_value:ConstType=0):
    """
    Selects elements from `self` based on the boolean `mask`.

    With `size=None` (default), output length equals the number of `True` values (not jittable).
    With `size=N`, output length is `N`, padded with `fill_value` or truncated (jittable).

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    mask = Tensor([[True, False, True], [False, True, False], [False, False, True]])
    print(t.numpy())
    print(mask.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.masked_select(mask).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.masked_select(mask, size=6, fill_value=-1).numpy())
    ```
    """
    if not dtypes.is_bool(mask.dtype): raise RuntimeError(f"masked_select expects bool mask tensor, got {mask.dtype}")
    x, mask = self.flatten(), mask._broadcast_to(self.shape).flatten()
    mask_cumsum = mask.cumsum()
    if size is None:
      counts = type(self).zeros(mask_cumsum[-1].item() if mask.numel() else 0, dtype=dtypes.int32, buffer=False)
      return x[counts.scatter(0, mask_cumsum, 1, reduce='add').cumsum()]
    counts = type(self).zeros(size, dtype=dtypes.int32, buffer=False).scatter(0, mask_cumsum, 1, reduce='add')
    return (type(self).arange(size) < mask.sum()).where(x[counts.cumsum()], fill_value).cast(self.dtype)

  def nonzero(self, size:int|None=None, fill_value:ConstType=0) -> Self:
    """
    Returns the indices of the elements that are non-zero.

    With `size=None` (default), output shape is `(n_nonzero, ndim)` (not jittable).
    With `size=N`, output shape is `(N, ndim)`, padded with `fill_value` or truncated (jittable).

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 0, 2, 0, 3])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.nonzero().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 0], [0, 2]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.nonzero().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.nonzero(size=3, fill_value=-1).numpy())
    ```
    """
    if self.ndim == 0:
      return type(self).zeros(size if size is not None else int(self.ne(0).item()), 0, dtype=dtypes.int32, device=self.device)
    mask = self.ne(0).flatten()
    indices = type(self).stack(*[type(self).arange(s).reshape(*[1]*i, s, *[1]*(self.ndim-i-1)).expand(self.shape).flatten()
                             for i, s in enumerate(self.shape)], dim=-1)
    return indices.masked_select(mask.unsqueeze(-1).expand(*mask.shape, self.ndim),
                                 size=size*self.ndim if size is not None else None, fill_value=fill_value).reshape(-1, self.ndim)

  # ***** functional nn ops *****

  def sequential(self, ll:list[Callable[[Self], Self]]) -> Self:
    """
    Applies a sequence of functions to `self` chaining the output of each function to the input of the next.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t.sequential([lambda x: x * 2, lambda x: x + 1]).numpy())
    ```
    """
    return functools.reduce(lambda x,f: f(x), ll, self)

  def linear(self, weight:Self, bias:Self|None=None, dtype:DTypeLike|None=None) -> Self:
    """
    Applies a linear transformation to `self` using `weight` and `bias`.

    See: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2], [3, 4]])
    weight = Tensor([[1, 2], [3, 4]])
    bias = Tensor([1, 2])
    print(t.linear(weight, bias).numpy())
    ```
    """
    if dtype is not None:
      dt = to_dtype(dtype)
      return self.cast(dt).linear(weight.cast(dt), bias.cast(dt) if bias is not None else bias)
    x = self.mul(weight) if len(weight.shape) == 1 else self.dot(weight)
    return x.add(bias) if bias is not None else x

  def _apply_ceil_mode(self, pads:Sequence[int], k_:tuple[sint, ...], s_:int|tuple[int, ...], d_:int|tuple[int, ...]) -> list[int]:
    (d_,s_), i_ = (make_tuple(x, len(k_)) for x in (d_,s_)), self.shape[-len(k_):]
    grouped_pads = list(flat_to_grouped(pads))
    # https://arxiv.org/pdf/1603.07285 section 5.1, relationship 15.
    o_ = [ceildiv(i+pB+pA - (d*(k-1)+1), s) + 1 for i,d,k,s,(pB,pA) in zip(i_,d_,k_,s_,grouped_pads)]
    for dim,(o,i,s,k,d,(pB,pA)) in enumerate(zip(o_,i_,s_,k_,d_,grouped_pads)):
      # we have to do additional padding before `_pool` so that `o_` in `_pool` is calculated correctly
      # `s*(o-1) + (d*(k-1)+1) - (i+pB+pA)` -> last_sliding_window_start + full_kernel_size - padded_input_shape
      # we decrease padding in the case that a sliding window starts in the end padded region, thereby decreasing `o_` in `_pool`
      # `smax(s*(o-1) - (pB+i-1), 0)` -> last_sliding_window_start - (pad_before + input_size - zero_offset)
      grouped_pads[dim] = (pB, pA + s*(o-1) + (d*(k-1)+1) - (i+pB+pA) - smax(s*(o-1) - (pB+i-1), 0))
    return flatten(reversed(grouped_pads))

  # NOTE: these work for more than 2D
  def avg_pool2d(self, kernel_size:tuple[int, ...]=(2,2), stride=None, dilation=1, padding:int|tuple[int, ...]=0,
                 ceil_mode=False, count_include_pad=True) -> Self:
    """
    Applies average pooling over a tensor.

    This function supports three different types of `padding`

    1. `int` (single value):
      Applies the same padding value uniformly to all spatial dimensions.

    2. `tuple[int, ...]` (length = number of spatial dimensions):
      Specifies a distinct padding value for each spatial dimension in the form `(padding_height, padding_width, ...)`.

    3. `tuple[int, ...]` (length = 2 * number of spatial dimensions):
      Specifies explicit padding for each side of each spatial dimension in the form
      `(padding_left, padding_right, padding_top, padding_bottom, ...)`.

    When `ceil_mode` is set to `True`, output shape will be determined using ceil division.
    When `count_include_pad` is set to `False`, zero padding will not be included in the averaging calculation.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d pooling and instead works for any number of dimensions.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(25).reshape(1, 1, 5, 5)
    print(t.avg_pool2d().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.avg_pool2d(ceil_mode=True).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.avg_pool2d(padding=1).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.avg_pool2d(padding=1, count_include_pad=False).numpy())
    ```
    """
    axis = tuple(range(-len(k_ := make_tuple(kernel_size, 2)), 0))
    s_ = stride if stride is not None else k_
    def pool(x:Self, padding_:Sequence[int]) -> Self:
      return x._pad_constant(((0,0),)*(x.ndim-len(k_)) + flat_to_grouped(padding_), 0.0)._pool(k_, s_, dilation)
    reg_pads = resolve_pool_pads(padding, len(k_))
    pads = self._apply_ceil_mode(reg_pads, k_, s_, dilation) if ceil_mode else reg_pads
    if not count_include_pad:
      return pool(self, pads).sum(axis) / pool(self.const_like(1), pads).sum(axis)
    if not ceil_mode: return pool(self, pads).mean(axis)
    return pool(self, pads).sum(axis) / pool(self._pad_constant(((0,0),)*(self.ndim-len(k_)) + flat_to_grouped(reg_pads), 0.0).const_like(1),
                                              tuple(cp-rp for cp,rp in zip(pads, reg_pads))).sum(axis)

  def max_pool2d(self, kernel_size:tuple[int, ...]=(2,2), stride=None, dilation=1, padding:int|tuple[int, ...]=0,
                 ceil_mode=False, return_indices=False) -> Self | tuple[Self, Self]:
    """
    Applies max pooling over a tensor.

    This function supports three different types of `padding`

    1. `int` (single value):
      Applies the same padding value uniformly to all spatial dimensions.

    2. `tuple[int, ...]` (length = number of spatial dimensions):
      Specifies a distinct padding value for each spatial dimension in the form `(padding_height, padding_width, ...)`.

    3. `tuple[int, ...]` (length = 2 * number of spatial dimensions):
      Specifies explicit padding for each side of each spatial dimension in the form
      `(padding_left, padding_right, padding_top, padding_bottom, ...)`.

    When `ceil_mode` is set to `True`, output shape will be determined using ceil division.
    When `return_indices` is set to `True`, the argmax will be returned along with the max values.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d pooling and instead works for any number of dimensions.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(25).reshape(1, 1, 5, 5)
    print(t.max_pool2d().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.max_pool2d(ceil_mode=True).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.max_pool2d(padding=1).numpy())
    ```
    """
    axis = tuple(range(-len(k_ := make_tuple(kernel_size, 2)), 0))
    s_ = stride if stride is not None else k_
    pads = resolve_pool_pads(padding, len(k_))
    if ceil_mode: pads = self._apply_ceil_mode(pads, k_, s_, dilation)
    pooled = self._pad_constant(((0,0),)*(self.ndim-len(k_)) + flat_to_grouped(pads), self.dtype.min)._pool(k_, s_, dilation)
    if not return_indices: return pooled.max(axis)
    spatial_sz = int(prod(spatial_shape := self.shape[-len(k_):]))
    idx = type(self).arange(spatial_sz, 0, -1).reshape(spatial_shape)
    m = pooled.eq(pooled.max(axis, keepdim=True))
    idx = m * idx._pad_constant(((0,0),)*(idx.ndim-len(k_)) + flat_to_grouped(pads), idx.dtype.min)._pool(k_, s_, dilation)
    return pooled.max(axis), spatial_sz - idx.max(axis)

  def max_unpool2d(self, indices:Self, kernel_size:tuple[int, ...]=(2,2), stride=None, dilation=1, padding:int|tuple[int, ...]=0,
                   output_size=None) -> Self:
    """
    Performs a partial inverse of `max_pool2d` using the indices from the argmax.

    When `output_size` is provided, the output shape disambiguates to the provided shape.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d pooling and instead works for any number of dimensions.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(1, 17).reshape(1, 1, 4, 4)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    output, indices = Tensor.max_pool2d(t, return_indices=True)
    print(output.numpy())
    print(indices.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.max_unpool2d(output, indices).numpy())
    ```
    """
    bs,c,*spatial_shape = self.shape
    if output_size is None:
      k_,d_,s_ = (make_tuple(x, len(spatial_shape)) for x in (kernel_size, dilation, stride if stride is not None else kernel_size))
      p_ = flat_to_grouped(resolve_pool_pads(padding, len(spatial_shape)))
      # https://arxiv.org/pdf/1603.07285 inverse of relationship 15 in section 5.1.
      output_size = tuple((i-1)*s - (pB+pA) + (d*(k-1)+1) for i,k,d,s,(pA,pB) in zip(spatial_shape,k_,d_,s_,p_))
    else: output_size = output_size[-len(spatial_shape):]
    ret = (indices.reshape(bs,c,1,-1)._one_hot_along_dim(prod(output_size), 2).where(self.reshape(bs,c,1,-1), 0)).sum(3)
    return ret.reshape(bs,c,*output_size)

  @classmethod
  def _get_winograd_matcols(cls, mat, dims:int, shp:tuple[sint, ...], dtype:DType) -> list[list[Self]]:
    return [[cls.cat(*[cls.full(shp[:dim] + (1,) + shp[dim+1:], float(m[k]), dtype=dtype, buffer=False) for m in mat], dim=dim)
             for k in range(len(mat[0]))] for dim in range(dims)]

  # winograd conv 3 kernel f(4x4,3x3) see: http://arxiv.org/abs/1509.09308
  def _apply_winograd_matrix(self, mat, dims:int) -> Self:
    # multiply mat_1 @ mat_2 @ t with foldable constants, where mat_i acts on vector t along dimension i; roughly kron(mat, mat) @ t
    # due to realize-before-expand rule in lazy.py, we must operate in this order: reshape -> expand -> arithmetic
    t_ = self.reshape(self.shape[:dims] + (1,) * dims + self.shape[dims:]).expand(
      self.shape[:dims] + (len(mat),) * dims + self.shape[dims:])  # add output dims
    # precalculate mat columns for each dim; prod(itertools.product(matcols)) gives the columns of kron(mat, mat, ...)
    matcols = type(self)._get_winograd_matcols(mat, dims, t_.shape[dims:], t_.dtype)
    # multiply each element of t_ by the corresponding stacked column of kron(mat, mat), producing only one view for each element of t
    ret = sum(prod(col[idx] for col, idx in zip(matcols, mat_is)) * t_[mat_is] for mat_is in itertools.product(range(len(mat[0])), repeat=dims))
    assert not isinstance(ret, int), "sum over empty winograd matrix"
    return ret

  # TODO: winograd can be a rewrite rule like split_reduceop
  def _conv2d_winograd(self, weight:Self, bias:Self|None, groups:int, padding:int|Sequence[int], dtype:DTypeLike|None) -> Self:
    (bs,cin_), (cout,cin), HW = self.shape[:2], weight.shape[:2], weight.shape[2:]
    padding_ = resolve_pool_pads(padding, len(HW))
    assert groups*cin == cin_ and len(self.shape) == len(weight.shape),\
        f"Input Tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"
    rcout, oyx = cout//groups, self.pad(padding_)._pool(HW, 1, 1).shape[2:-len(HW)]
    HWI, HWO = (6,) * len(HW), (4,) * len(HW)  # F(4x4,3x3) winograd tiles
    winograd_G = [[1/4, 0, 0], [-1/6, -1/6, -1/6], [-1/6, 1/6, -1/6], [1/24, 1/12, 1/6], [1/24, -1/12, 1/6], [0, 0, 1]]
    winograd_Bt = [[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0], [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]]
    winograd_At = [[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]] # applying At in pre-order doubles compile time

    # TODO: stride == dilation
    # use padding to round up to 4x4 output tiles
    # (bs, cin_, tyx, HWI)
    pads = [(pB, pA + (-(s + pB + pA - 2) % 4)) for (pB, pA), s in zip(flat_to_grouped(padding_), self.shape[-len(HW):])]
    d = self.pad(flatten(reversed(pads)))._pool(HWI, HWO)
    # move HW to the front: # (HWI, bs, cin_, tyx)
    d = d.permute(*range(len(d.shape)-len(HW),len(d.shape)), *range(len(d.shape)-len(HW)))
    tyx = d.shape[-len(HWI):]  # dim of tiling

    g = weight.permute(*range(len(weight.shape)-len(HW),len(weight.shape)), *range(len(weight.shape)-len(HW)))  # move HW to the front

    # compute 6x6 winograd tiles: GgGt, BtdB. contiguous so the transforms are materialized once
    # (HWI, groups * rcout, cin) -> (HWI, bs=1, groups, rcout, cin, tyx=(1,1))
    gfactors = g._apply_winograd_matrix(winograd_G, len(HW)).contiguous().reshape(*HWI, 1, groups, rcout, cin, *([1]*len(tyx)))
    # (HWI, bs, cin_, tyx) -> (HWI, bs, groups, 1 ,cin, *tyx)
    dfactors = d._apply_winograd_matrix(winograd_Bt, len(HW)).contiguous().reshape(*HWI, bs, groups, 1, cin, *tyx)

    # matmul; sum across cin: (HWI, bs, groups, rcout, *tyx); then HWI -> HWO: (HWO, bs, groups, rcout, *tyx)
    ret = (gfactors * dfactors).sum(axis=-1-len(HW), dtype=dtype)._apply_winograd_matrix(winograd_At, len(HW))

    # interleave tyx and HWO: (bs, groups, rcout, oy, HO, ox, WO)
    ret = ret.permute([*range(len(HW), len(ret.shape)-len(HW)), *[i+o for i in range(len(HW)) for o in [len(ret.shape)-len(HW),0]]])
    # merge groups and rcout, tyx and HWO: (bs, groups, cout, *yx), shrink to final
    ret = ret.reshape(bs, cout, *[c * HWO[i] for i, c in enumerate(tyx)]).shrink_to(bs, cout, *oyx)

    return (ret if bias is None else ret.add(bias.reshape(1, -1, *[1 for _ in range(len(HW))]))).contiguous().contiguous_backward()

  # *** image function replacements (used when IMAGE is set) ***

  def image_dot(self, w:Self, dtype:DTypeLike|None=None) -> Self:
    # NOTE: we use a 1x1 conv2d to do the matmul. mxk @ kxn = (1,k,m,1).conv2d(n,k,1,1)
    if not (self.ndim > 0 and w.ndim > 0): raise RuntimeError(f"both tensors need to be at least 1D, got {self.ndim=}, {w.ndim=}")
    if self.shape[-1] != w.shape[-min(w.ndim, 2)]: raise RuntimeError(f"cannot image_dot {self.shape} and {w.shape}")

    bs, groups, cin, cout = prod(self.shape[0:-2]), prod(w.shape[0:-2]), w.shape[-2], w.shape[-1]
    out_shape_t = self.shape[0:-2] + (cout,-1) if len(self.shape) > 1 else (cout,)

    # NOTE: with NHWC we can remove the transposes
    # bs x groups*cin x H x W
    cx = self.transpose(self.ndim-1, self.ndim-2).reshape(bs//groups, groups*cin, -1, 1)
    # groups*cout x cin x H, W
    cw = w.transpose(w.ndim-1, w.ndim-2).reshape(groups*cout, cin, 1, 1)
    return cx.image_conv2d(cw, groups=groups, dtype=dtype).reshape(out_shape_t).transpose(self.ndim-1, self.ndim-2)

  def image_conv2d(self, weight:Self, bias:Self|None=None, groups=1, stride=1, dilation=1, padding=0, dtype=None) -> Self:
    dtsz = 2 if FLOAT16 else 4

    (bs,_,_,_), (cout,cin,H,W) = self.shape, weight.shape
    assert isinstance(cin, int) and isinstance(cout, int)
    x, w = self, weight.reshape(groups, (rcout := cout//groups), cin, H, W)

    padding_neg, padding_pos = [min(0, p) for p in resolve_pool_pads(padding, 2)], [max(0, p) for p in resolve_pool_pads(padding, 2)]
    x = x.pad(padding_neg)
    iy, ix = x.shape[2:]

    # hack for non multiples of 4 on cin
    if cin % 4 != 0 and not (cin == 1 and groups%4 == 0):
      new_cin = round_up(cin, 4)
      w = w.pad_to(None, None, new_cin, None, None)
      x = x.reshape(bs, groups, cin, iy, ix)
      x = x.pad_to(None, None, new_cin, None, None).reshape(bs, groups*new_cin, iy, ix)
      cin = new_cin

    # hack for non multiples of 4 on rcout
    added_output_channels = 0
    if rcout % 4 != 0 and not (rcout == 1 and groups%4 == 0):
      added_output_channels = 4 - (rcout % 4)
      rcout += added_output_channels
      cout = groups * rcout
      w = w.pad_to(None, rcout, None, None, None)

    # packed (note: flipping bs and iy would make the auto-padding work)
    x = x.permute(0,2,3,1)
    cin_last = iy == 1 and ix == 1
    if cin == 1: w = w.reshape(cout//4,4,H,W).permute(0,2,3,1)
    elif cin_last: w = w.reshape(cout//4,4,cin//4,4,H,W).permute(0,4,2,5,1,3)
    else: w = w.reshape(cout//4,4,cin//4,4,H,W).permute(0,4,2,5,3,1)

    def is_pow2(v): return v > 0 and v & (v - 1) == 0
    # pad dimension i to amt with invalids
    def ipad(t, i, amt):
      return t.pad(tuple(None if d != i else (0, amt-s) for d,s in enumerate(t.shape)), value=Invalid) if amt != t.shape[i] else t
    # align a dimension, use at to specify the dimension to pad in, defaults to first
    def pad_align(t, dim, at=None, force=False):
      # align to 64 pixels when height is real, otherwise 64 bytes is sufficient
      align = (64 // dtsz) if prod(t.shape[:dim]) == 1 or prod(t.shape) < 16384 * 4 else 256
      return ipad(t, at:=at or dim, round_up(t.shape[at] + int(force), align // math.gcd(prod(t.shape[dim:]) // t.shape[at], align)))

    # bank conflicts
    bank_conflict = cin >= 8 and is_pow2(cin // 4)
    if bank_conflict:
      x, w = pad_align(x.reshape(bs, iy, ix, groups, cin // 4, 4), 2, at=4, force=True), pad_align(w, 1, at=2, force=True)
    else: x, w = pad_align(x, 2), pad_align(w, 1)

    # contiguous creates the image, and early realize static weights (TODO: test for the static weight)
    if FLOAT16: x, w = x.cast(dtypes.half).contiguous().cast(dtypes.float), w.cast(dtypes.half).contiguous().cast(dtypes.float)
    else: x, w = x.contiguous(), w.contiguous()

    # undo alignment hacks
    if bank_conflict: x, w = x[:, :, :, :, :cin // 4, :], w[:, :, :cin // 4, ...]
    else: x, w = x[:, :, :ix, :], w[:, :H, ...]

    # expand out
    rcin_hi, rcin_lo = (cin//4, 4) if cin >= 4 else (1, 1)
    group_shape, rcout_expand = (groups//4, 4) if cin == 1 else (groups, 1), (rcout//4, 4) if rcout >= 4 else (1, 1)
    x = x.reshape(bs, iy, -1, groups, rcin_hi, rcin_lo)
    if cin_last: w = w.reshape(cout//4, H, rcin_hi, W, 4, rcin_lo)
    else: w = w.reshape(cout//4, H, rcin_hi, W, rcin_lo, 4).permute(0,1,2,3,5,4)

    # prepare input
    x = x.permute(0,3,4,5,1,2).pad(padding_pos)._pool((H,W), stride, dilation)# -> (bs, groups, rcin_hi, rcin_lo, oy, ox, H, W)
    x = x.permute(0,4,5,1,2,3,6,7).reshape(bs, (oy := x.shape[4]), (ox := x.shape[5]), *group_shape, 1, 1, rcin_hi, rcin_lo, H, W)

    # prepare weights
    w = w.permute(0,4,2,5,1,3).reshape((1, 1, 1, *group_shape, *rcout_expand, rcin_hi, rcin_lo, H, W))

    # the conv!
    ret = (x*w).cast(dtypes.float32).sum((-4, -3, -2, -1), dtype=dtype)

    ret = ret.reshape(bs, oy, ox, groups, rcout)
    # undo hack for non multiples of 4 on C.rcout
    if added_output_channels: ret = ret[:, :, :, :, :-added_output_channels]
    # NCHW output
    ret = ret.reshape(bs, oy, ox, groups * (rcout - added_output_channels)).permute(0,3,1,2)
    return ret if bias is None else ret.add(bias.reshape(1, -1, 1, 1))

  def conv2d(self, weight:Self, bias:Self|None=None, groups=1, stride=1, dilation=1, padding:int|Sequence[int]=0,
             dtype:DTypeLike|None=None) -> Self:
    """
    Applies a convolution over a tensor with a given `weight` and optional `bias`.

    This function supports three different types of `padding`

    1. `int` (single value):
      Applies the same padding value uniformly to all spatial dimensions.

    2. `tuple[int, ...]` (length = number of spatial dimensions):
      Specifies a distinct padding value for each spatial dimension in the form `(padding_height, padding_width, ...)`.

    3. `tuple[int, ...]` (length = 2 * number of spatial dimensions):
      Specifies explicit padding for each side of each spatial dimension in the form
      `(padding_left, padding_right, padding_top, padding_bottom, ...)`.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d convolutions and instead works for any number of dimensions.

    See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(1, 1, 3, 3)
    w = Tensor.ones(1, 1, 2, 2)
    print(t.conv2d(w).numpy())
    ```
    """
    if IMAGE: return self.image_conv2d(weight, bias, groups, stride, dilation, padding, dtype)
    if WINO and all(x == 3 for x in weight.shape[2:]) and stride == dilation == 1: return self._conv2d_winograd(weight, bias, groups, padding, dtype)
    (bs,cin_), (cout,cin), HW = self.shape[:2], weight.shape[:2], weight.shape[2:]
    padding_ = resolve_pool_pads(padding, len(HW))
    assert groups*cin == cin_ and len(self.shape) == len(weight.shape),\
        f"Input Tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"
    # conv2d is a pooling op (with padding, possibly negative — _pad_constant handles the shrink)
    x = self._pad_constant(((0,0),)*(self.ndim-len(HW)) + flat_to_grouped(padding_), 0.0)._pool(HW, stride, dilation)
    rcout, oyx = cout//groups, x.shape[2:-len(HW)]
    x = x.reshape(bs, groups, cin, 1, *oyx, *HW).expand(bs, groups, cin, rcout, *oyx, *HW)\
      .permute(0,1,3,*[4+i for i in range(len(oyx))],2,*[4+len(oyx)+i for i in range(len(HW))])
    # conv! broadcasted to (bs, groups, rcout, *oyx, cin, *HW)
    ret = (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW))\
      .sum([-1-i for i in range(1+len(oyx))], keepdim=True, dtype=dtype).reshape(bs, cout, *oyx)
    return ret if bias is None else ret.add(bias.reshape(1, -1, *[1] * len(HW)))

  def conv_transpose2d(self, weight:Self, bias:Self|None=None, groups=1, stride=1, dilation=1, padding=0, output_padding=0) -> Self:
    """
    Applies a transposed convolution over a tensor with a given `weight` and optional `bias`.

    This function supports three different types of `padding`

    1. `int` (single value):
      Applies the same padding value uniformly to all spatial dimensions.

    2. `tuple[int, ...]` (length = number of spatial dimensions):
      Specifies a distinct padding value for each spatial dimension in the form `(padding_height, padding_width, ...)`.

    3. `tuple[int, ...]` (length = 2 * number of spatial dimensions):
      Specifies explicit padding for each side of each spatial dimension in the form
      `(padding_left, padding_right, padding_top, padding_bottom, ...)`.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d transposed convolutions and instead works for any number of dimensions.

    See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(1, 1, 3, 3)
    w = Tensor.ones(1, 1, 2, 2)
    print(t.conv_transpose2d(w).numpy())
    ```
    """
    x, w = self, weight.unflatten(0, (groups, -1)).transpose(1, 2).flip(*range(3, len(weight.shape)+1))
    HW = weight.shape[2:]
    padding = flat_to_grouped(resolve_pool_pads(padding, len(HW)))
    stride, dilation, output_padding = [make_tuple(x, len(HW)) for x in (stride, dilation, output_padding)]
    if any(s>1 for s in stride):
      # handle strides: (k) -> reshape -> (k,1) -> pad -> (k,s) -> reshape -> (k*s) -> shrink (k-(s-1))
      x = x.reshape(None, None, *flatten((k,1) for k in x.shape[2:]))
      x = x.pad((None, None, *flatten((None,(0,s-1)) for s in stride)))
      x = x.reshape(None, None, *[k*s for k,s in zip(x.shape[2::2], stride)])
      x = x.shrink_to(None, None, *[k-(s-1) for k,s in zip(x.shape[2:], stride)])
    padding = flatten((((k-1)*d-pB,(k-1)*d-pA+op) for k,d,(pB,pA),op in reversed(list(zip(HW, dilation, padding, output_padding)))))
    return x.conv2d(w.flatten(end_dim=1), groups=groups, bias=bias, dilation=dilation, padding=padding)

  def layernorm(self, axis:int|tuple[int,...]=-1, eps:float=1e-5) -> Self:
    """
    Applies Layer Normalization over a mini-batch of inputs.

    - Paper: https://arxiv.org/abs/1607.06450v1

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.randn(8, 10, 16) * 2 + 8
    print(t.mean().item(), t.std().item())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.layernorm()
    print(t.mean().item(), t.std().item())
    ```
    """
    y = (self - self.mean(axis, keepdim=True))
    return y.mul((y*y).mean(axis, keepdim=True).add(eps).rsqrt())

  def batchnorm(self, weight:Self|None, bias:Self|None, mean:Self, invstd:Self, axis:int|tuple[int, ...]=1) -> Self:
    """
    Applies Batch Normalization over a mini-batch of inputs.

    - Paper: https://arxiv.org/abs/1502.03167

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.randn(8, 4, 16, 16) * 2 + 8
    print(t.mean().item(), t.std().item())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.batchnorm(None, None, t.mean(axis=(0,2,3)), t.var(axis=(0,2,3)).add(1e-5).rsqrt())
    print(t.mean().item(), t.std().item())
    ```
    """
    axis_ = argfix(axis)
    shape = tuple(s if ax in axis_ else 1 for ax, s in enumerate(self.shape))
    x = self - mean.reshape(shape)
    if weight is not None: x = x * weight.reshape(shape)
    ret = x.mul(invstd.reshape(shape) if len(invstd.shape) == len(axis_) else invstd)
    return (ret + bias.reshape(shape)) if bias is not None else ret

  # ***** loss ops *****

  def _do_reduction(self, reduction:ReductionStr="mean") -> Self:
    if reduction == "none": return self
    if reduction == "sum": return self.sum()
    if reduction == "mean": return self.mean()
    raise ValueError(f"{reduction=} must be one of {get_args(ReductionStr)}")

  def binary_crossentropy(self, Y:Self, reduction:ReductionStr="mean") -> Self:
    """
    Computes the binary cross-entropy loss between `self` and `Y`.

    See: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([0.1, 0.9, 0.2])
    Y = Tensor([0, 1, 0])
    print(t.binary_crossentropy(Y).item())
    ```
    """
    return (-Y*self.log() - (1-Y)*(1-self).log())._do_reduction(reduction)

  def binary_crossentropy_logits(self, Y:Self, reduction:ReductionStr="mean", pos_weight:Self|None=None) -> Self:
    """
    Computes the binary cross-entropy loss between `self` and `Y` where `self` is logits.

    See: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2, -3])
    Y = Tensor([0, 1, 0])
    print(t.binary_crossentropy_logits(Y).item())
    ```
    """
    log_p, log_1_minus_p = self.logsigmoid(), (-self).logsigmoid()
    return (-((1 if pos_weight is None else pos_weight) * Y * log_p + (1-Y) * log_1_minus_p))._do_reduction(reduction)

  def sparse_categorical_crossentropy(self, Y:Self, ignore_index:int=-1, label_smoothing=0.0, reduction:ReductionStr="mean") -> Self:
    """
    Computes the sparse categorical cross-entropy loss between `self` and `Y`.

    NOTE: `self` is logits and `Y` is the target labels.
    NOTE: unlike PyTorch, this function expects the class axis to be -1

    See: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[-1, 2, -3], [1, -2, 3]])
    Y = Tensor([1, 2])
    print(t.sparse_categorical_crossentropy(Y).item())
    ```
    """
    assert 0.0 <= label_smoothing <= 1.0, "label_smoothing must be in [0.0, 1.0]"
    if Y.device is not None and self.device is not None and Y.device != self.device:
      raise RuntimeError(f"expected Y and self on the same device, {Y.device=}, {self.device=}")
    log_probs = self.log_softmax()
    loss_mask = Y.ne(ignore_index) if ignore_index != -1 else Y.const_like(1).cast(dtypes.bool)
    y = Y.unsqueeze(-1)._one_hot_along_dim(self.shape[-1], dim=-1) * loss_mask.unsqueeze(-1)
    smoothing = label_smoothing * (log_probs.mean(-1) * loss_mask)
    unreduced = ((1 - label_smoothing) * (log_probs * y).sum(-1) + smoothing)
    return -unreduced.sum() / loss_mask.sum() if reduction == "mean" else -unreduced._do_reduction(reduction)

  def cross_entropy(self, Y:Self, reduction:ReductionStr="mean", label_smoothing:float=0.0) -> Self:
    """
    Computes the cross entropy loss between input logits and target.

    NOTE: `self` are logits and `Y` are the target labels or class probabilities.

    See: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[-1, 2, -3], [1, -2, 3]])
    Y = Tensor([1, 2])
    print(t.cross_entropy(Y).item())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[-1, 2, -3], [1, -2, 3]])
    Y = Tensor([1, 2])
    print(t.cross_entropy(Y, reduction='none').numpy())
    ```
    """
    assert 0.0 <= label_smoothing <= 1.0, "label_smoothing must be in [0.0, 1.0]"
    classes_dim = 0 if self.ndim == 1 else 1
    if self.shape != Y.shape:
      if self.max(classes_dim).shape != Y.shape: raise RuntimeError(f"shape mismatch: {self.shape=}, {Y.shape=}")
      Y = Y.unsqueeze(classes_dim)._one_hot_along_dim(num_classes=self.shape[classes_dim], dim=classes_dim)
    Y = (1 - label_smoothing)*Y + label_smoothing / int(Y.shape[classes_dim])
    return -self.log_softmax(classes_dim).mul(Y).sum(classes_dim)._do_reduction(reduction)

  def nll_loss(self, Y:Self, weight:Self|None=None, ignore_index:int|None=None, reduction:ReductionStr="mean") -> Self:
    """
    Computes the negative log likelihood loss between log-probabilities and target labels.

    NOTE: `self` is log-probabilities and `Y` is the Y labels or class probabilities.

    See: https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[-1, 2, -3], [1, -2, 3]])
    Y = Tensor([1, 2])
    print(t.log_softmax().nll_loss(Y).item())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[-1, 2, -3], [1, -2, 3]])
    Y = Tensor([1, 2])
    print(t.log_softmax().nll_loss(Y, reduction='none').numpy())
    ```
    """
    weight = Y.const_like(1) if weight is None else weight.gather(0, Y.flatten()).reshape(Y.shape)
    masked_weight = weight if ignore_index is None else weight * Y.ne(ignore_index)
    nll = -self.gather(1, Y.unsqueeze(1)).squeeze(1) * masked_weight
    return nll.sum() / masked_weight.sum() if reduction == "mean" else nll._do_reduction(reduction)

  # ***** matrix ops *****

  def qr(self) -> tuple[Self, Self]:
    assert self.ndim > 1, f"expected two or more dimensions, got {self.ndim}"
    b_shape, m, n = self.shape[:-2], int(self.shape[-2]), int(self.shape[-1])
    R, Q = self, type(self).eye(m, dtype=self.dtype).expand(b_shape + (m, m))
    idx = type(self).arange(m)
    for i in range(min(m, n)):
      # full-length Householder reflector v with zeros above row i; w = tau*v is the rank-1 update factor
      at_i, x = idx.eq(i), (idx >= i).where(R[..., :, i], 0)
      norm = x.square().sum(-1, keepdim=True).sqrt()
      x0 = at_i.where(x, 0).sum(-1, keepdim=True)
      sgn, active = x0.ne(0).where(x0.sign(), 1), norm.ne(0)
      u0 = x0 + sgn * norm
      v = (at_i.where(u0, x) / active.where(u0, 1)).unsqueeze(-1)
      w = active.where(sgn * u0 / active.where(norm, 1), 0).unsqueeze(-1) * v
      R = R - w @ (v.transpose(-2, -1) @ R)
      Q = Q - (Q @ v) @ w.transpose(-2, -1)
    return Q, R

  def svd(self, full_matrices = True) -> tuple[Self, Self, Self]:
    #partial implementation of https://www.netlib.org/lapack/lawnspdf/lawn169.pdf , pg 26
    assert self.ndim > 1, f"expected two or more dimensions, got {self.ndim}"
    b_shape, m, n = self.shape[:-2], int(self.shape[-2]), int(self.shape[-1])
    #preprocess the matrix
    Q, R = (self if m >= n else self.transpose(-2, -1)).qr()
    num, q_num = min(m, n), max(m, n)
    # TODO: codegen infinite loop without contiguous
    U = R[..., :num, :num].contiguous()
    V = type(self).eye(num, dtype=self.dtype).expand(b_shape + (num, num)).contiguous()
    #prepare round robin pairing: identity on first half, reversed on second half
    permute = type(self).arange(num//2, dtype=dtypes.int).cat(
                type(self).arange(num//2, num, dtype=dtypes.int).flip(0))
    cols, h = type(self).arange(num, dtype=dtypes.int), num // 2
    eye_num = type(self).eye(num, dtype=self.dtype).expand(b_shape + (num, num))
    def one_round_jacobi(U, V, permute):
      # permutation matrix P with P[a,b] = (a == permute[b]); first 2h columns are paired-column selectors
      P = cols.unsqueeze(1).eq(permute.unsqueeze(0)).cast(U.dtype)
      P_pair = P[..., :2*h]  # drops the runoff column for odd num
      # extract paired columns to compute Jacobi rotation params
      U_pair = U @ P_pair
      U_left, U_right = U_pair.split(h, -1)
      gamma = (U_left * U_right).sum(-2).reshape(b_shape + (1, h))
      alpha, beta = U_pair.square().sum(-2).unsqueeze(-2).split(h, -1)
      rot = gamma.ne(0)
      tau = (beta - alpha) / (2 * rot.where(gamma, 1))
      t = tau.ne(0).where(tau.sign(), 1) / (tau.abs() + (1 + tau.square()).sqrt())
      t = rot.where(t, 0)
      c = 1 / (1 + t.square()).sqrt()
      s = c * t
      # build rotation matrix R: identity + sum over pairs of 2x2 rotation deltas at (i_k, j_k) positions
      Mi, Mj = P_pair.transpose(-2, -1).split(h, -2)  # paired-column selectors, each shape (h, num)
      Mi_a, Mi_b = Mi.unsqueeze(-1), Mi.unsqueeze(-2)
      Mj_a, Mj_b = Mj.unsqueeze(-1), Mj.unsqueeze(-2)
      cc, ss = (c - 1).reshape(b_shape + (h, 1, 1)), s.reshape(b_shape + (h, 1, 1))
      R = eye_num + (cc * (Mi_a * Mi_b + Mj_a * Mj_b) + ss * (Mi_a * Mj_b - Mj_a * Mi_b)).sum(-3)
      U, V = U @ R, V @ R
      #prepare the next round robin pairings
      if num % 2 == 1: permute = (permute - 1) % num
      else: permute = permute[0].reshape(1).cat(((permute[1:num] - 2) % (num - 1)) + 1)
      return U, V, permute
    # classical Jacobi converges in ~4 sweeps; one full sweep is (num-1) rounds for even num
    for _ in range(4 * num): U, V, permute = one_round_jacobi(U, V, permute)
    #extract singular values and sort. construct U from Q
    S, indices = U.square().sum(-2).sqrt().sort(dim=-1, descending=True)
    new_indices = indices.unsqueeze(-2).expand(b_shape + (num, num))
    U = U.gather(-1, new_indices) / S.ne(0).where(S, 1).unsqueeze(-2)
    V = V.gather(-1, new_indices)
    # place U into the top-left num×num block of a q_num×q_num identity matrix
    pad_arg = (None,) * len(b_shape) + ((0, q_num - num), (0, q_num - num))
    eye_q = type(self).eye(q_num, dtype=U.dtype).expand(b_shape + (q_num, q_num))
    eye_n = type(self).eye(num, dtype=U.dtype).expand(b_shape + (num, num)).pad(pad_arg)
    U = Q @ (U.pad(pad_arg) + eye_q - eye_n)
    if not full_matrices: U = U[..., 0:num]
    return (U, S, V.transpose(-2, -1)) if m >= n else (V, S, U.transpose(-2, -1))

  def newton_schulz(self, steps:int, params:tuple[int, ...], eps:float=1.0e-7) -> Self:
    """
    Performs the newton-schulz algorithm for odd polynomials. The degree of the odd polynomial depends on the number of params.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.randn(4, 4)
    print(t.newton_schulz(steps=5, params=(2,-1.5,0.5)).numpy())
    ```
    """
    assert self.ndim > 1, "NS only works for two or more dims"
    if self.shape[-2] > self.shape[-1]: return self.transpose(-2, -1).newton_schulz(steps, params, eps).transpose(-2, -1)
    G = self / (self.square().sum(axis=(-2, -1), keepdim=True).sqrt() + eps)
    for _ in range(steps):
      G = functools.reduce(lambda a, b: a + b, (p * functools.reduce(lambda x, y: (y @ y.transpose(-2, -1)) @ x, [G]*i, G)
                                                 for i,p in enumerate(params)))
    return G

  # ***** hashing ops *****

  def keccak(self, cfg:str|tuple[int, int]="sha3_256"):
    """
    Calculates a Keccak hash over the last dimension. Uses "sha3_256" by default.

    ```python exec="false" source="above" session="tensor" result="python"
    t = Tensor(b"Hello World!").keccak()
    print(t.data().hex())
    ```
    """

    # https://keccak.team/keccak_specs_summary.html

    def ctensor(l: Sequence[PyConst], dtype: DType = dtypes.uint64):
      # TODO: contiguous is here for compile speed
      return type(self).stack(*(type(self).const(dtype, v) for v in l)).contiguous()
    rot_offsets = [44, 43, 21, 14, 28, 20, 3, 45, 61, 1, 6, 25, 8, 18, 27, 36, 10, 15, 56, 62, 55, 39, 41, 2]
    rot_offsets_v0, rot_offsets_v1 =  ctensor([0] + [1 << v for v in rot_offsets]), ctensor([1] + [1 << (64 - v) for v in rot_offsets])

    # calculated from π step
    reorder_indexes = ctensor([0,6,12,18,24,3,9,10,16,22,1,7,13,19,20,4,5,11,17,23,2,8,14,15,21], dtype=dtypes.int32)
    rnd_const_masks = [ctensor([v]).pad((0, 24)) for v in (1, 0x8082, 0x800000000000808a, 0x8000000080008000, 0x808b, 0x80000001, 0x8000000080008081,
    0x8000000000008009, 0x8a, 0x88, 0x80008009, 0x8000000a, 0x8000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x800a, 0x800000008000000a, 0x8000000080008081, 0x8000000000008080, 0x80000001, 0x8000000080008008)]

    rate, dsbyte = {"sha3_224": (144, 6), "sha3_256": (136, 6), "shake_128": (168, 31)}[cfg] if isinstance(cfg, str) else cfg
    data = self.bitcast(dtypes.uint8).reshape(prod(self.shape[:-1]), self.shape[-1])
    data_pad = rate - data.shape[-1] % rate
    # pad batches then pad blocks
    data = data.pad((None, (0, data_pad))).reshape(bs := data.shape[0], -1, rate).pad_to(None, None, 200)

    # create pad mask
    lbe = (data.shape[1] - 1) * 200 + rate - data_pad
    if data_pad == 1: mb = [(lbe, 0), (1, dsbyte ^ 0x80), (200 - rate, 0)]
    else: mb = [(lbe, 0), (1, dsbyte), (data_pad - 2, 0), (1, 0x80), (200 - rate, 0)]
    pad_mask = type(self).cat(*(type(self).const(dtypes.uint8, v).expand(l) for l, v in mb if l > 0)).unsqueeze(0)

    data = (data.flatten(1) ^ pad_mask).reshape(*data.shape[:2], 200).bitcast(dtypes.uint64)

    state = type(self).zeros(bs, 25, dtype=dtypes.uint64, buffer=False)
    for k in range(int(data.shape[1])):
      state = state ^ data[:, k]
      for i in range(24): # f1600
        # θ step
        p = state.reshape(bs, 5, 5).transpose(2, 1)
        t1 = (p[:,:,0] ^ p[:,:,1] ^ p[:,:,2] ^ p[:,:,3] ^ p[:,:,4]).roll(-1, 1) # xor reduce
        state = state ^ (t1.roll(2, 1).bitwise_xor((t1 << 1) ^ (t1 >> 63)).unsqueeze(2).expand(bs, 5, 5).transpose(2, 1).flatten(1))
        # ρ and π steps
        state = state[:, reorder_indexes]
        state = (state * rot_offsets_v0).bitwise_or(state // rot_offsets_v1).reshape(bs, 5, 5)
        # χ and ι step
        state = state.bitwise_xor(~state.roll(shifts=-1, dims=2) & state.roll(shifts=-2, dims=2))
        state = state.flatten(1) ^ rnd_const_masks[i]
      # NOTE: there was a kernelize here to prevent internal stack from growing propotional to data size, do we need something else?
    return state.bitcast(dtypes.uint8)[:,:(obytes:=(200 - rate) // 2)].reshape(*self.shape[:-1], obytes)

  def _hash_1mb(self) -> Self:
    assert self.dtype == dtypes.uint8, "only support uint8 tensors for hashing"
    assert self.ndim == 2, "only support batched 1d tensors"
    assert self.shape[1] == 1024 * 1024, "only support messages of 1mb"
    return self.reshape(-1, 4096).keccak("shake_128").reshape(self.shape[0], -1).keccak("shake_128")

  def hash(self) -> Self:
    """
    Calculates a 16-byte hash of the tensor.
    ```python exec="false source="above" session="tensor" result="python"
    t = Tensor(b"Hello World!").hash()
    print(t.data().hex())
    ```
    """
    data = self.flatten().bitcast(dtypes.uint8)
    n = data.shape[0]
    assert isinstance(n, int), "hash requires concrete shape"
    chunks = ceildiv(n, 2**20)
    while chunks > 1:
      data = data.pad_to(chunks * 2**20).reshape(chunks, 2**20)._hash_1mb().flatten()
      chunks = ceildiv(chunks, 65536)
    return data.pad_to(2**20).unsqueeze(0)._hash_1mb().flatten()[:16]

  # ***** tensor properties *****

  def nbytes(self) -> int:
    """
    Returns the total number of bytes of all elements in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([8, 9], dtype=dtypes.float)
    print(t.nbytes())
    ```
    """
    return int(self.numel()) * self.element_size()
