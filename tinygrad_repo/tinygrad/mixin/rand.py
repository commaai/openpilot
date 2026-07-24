from __future__ import annotations
import math
from typing import Self, cast
from tinygrad.dtype import DType, DTypeLike, dtypes, least_upper_dtype, to_dtype
from tinygrad.helpers import all_int, argfix, ceildiv, prod, TRAINING
from tinygrad.mixin.op import OpMixin
from tinygrad.device import canonicalize_device


class RandMixin(OpMixin):
  @staticmethod
  def _threefry_random_bits(key, counts0, counts1):
    x = (counts1.cast(dtypes.uint64) << 32) | counts0.cast(dtypes.uint64)
    x = x.threefry((key[1]._broadcast_to(x.shape).cast(dtypes.uint64) << 32) | key[0]._broadcast_to(x.shape).cast(dtypes.uint64))
    return (x & 0xffffffff).cast(dtypes.uint32).cat(((x >> 32) & 0xffffffff).cast(dtypes.uint32))

  @classmethod
  def random_bits(cls, key:Self, counter:Self, num:int) -> Self:
    low, high = counter[0:1], counter[1:2]
    bits = []
    for i in range(0, num, dtypes.uint32.max):
      chunk_num = min(num - i, dtypes.uint32.max)
      c_low = low + (i & 0xffffffff)
      c_high = high + (i >> 32) + (c_low < low).cast(dtypes.uint32)
      new_key = cls._threefry_random_bits(key, c_low, c_high)
      counts0 = cls.arange(ceildiv(chunk_num, 2), dtype=dtypes.uint32)
      counts1 = counts0 + ceildiv(chunk_num, 2)
      bits.append(cls._threefry_random_bits(new_key, counts0, counts1)[:chunk_num])
    return bits[0].cat(*bits[1:]) if bits else counter[0:0]

  @staticmethod
  def _bits_to_rand(bits, shape:tuple[int, ...], dtype:DType):
    _, nmant = dtypes.finfo(dtype)
    uint_dtype = {1: dtypes.uint8, 2: dtypes.uint16, 4: dtypes.uint32, 8: dtypes.uint64}[dtype.itemsize]
    uint_bits = bits.bitcast(uint_dtype)
    float_one_bits = uint_bits.const_like(1).cast(dtype).bitcast(uint_dtype)
    return uint_bits.rshift(dtype.bitsize - nmant).bitwise_or(float_one_bits).bitcast(dtype)[:prod(shape)].sub(1).reshape(shape)

  @classmethod
  def _rand(cls, key:Self, counter:Self, shape:tuple[int, ...], dtype:DType, contiguous:bool=True) -> Self:
    bits = cls.random_bits(key, counter, ceildiv(prod(shape) * dtype.itemsize, 4))
    out = cls._bits_to_rand(bits, shape, dtype)
    return out.contiguous() if contiguous else out

  @staticmethod
  def _next_counter(device:str, num:int):
    raise NotImplementedError("_next_counter requires the stateful per-device RNG counter, only implemented on Tensor")

  @classmethod
  def rand(cls, *shape, device:str|None=None, dtype:DTypeLike|None=None, contiguous:bool=True) -> Self:
    """
    Creates a tensor with the given shape, filled with random values from a uniform distribution over the interval `[0, 1)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.rand(2, 3)
    print(t.numpy())
    ```
    """
    dt = to_dtype(dtype or dtypes.default_float)
    if not dtypes.is_float(dt) or dt in dtypes.weaks: raise ValueError(f"rand only supports concrete float dtypes, got {dt}")
    if not all_int(shape:=argfix(*shape)) or not all(s >= 0 for s in shape): raise ValueError(f"invalid input {shape=}")
    if device is not None and not isinstance(device, str): raise ValueError(f"rand only supports single device, got {device=}")
    device = cast(str, canonicalize_device(device))
    key, counter = cls._next_counter(device, ceildiv(prod(shape) * dt.itemsize, 4))
    return cls._rand(key, counter, shape, dt, contiguous=contiguous)

  def rand_like(self, **kwargs) -> Self:
    """
    Creates a tensor with the same shape and sharding as `self`, filled with random values from a uniform distribution over the interval `[0, 1)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(Tensor.rand_like(t).numpy())
    ```
    """
    if isinstance(self.device, tuple):
      if kwargs.pop("device", None) is not None: raise RuntimeError("cannot specify `device` on `*_like` of a multi device tensor")
      dtype = kwargs.pop("dtype", self.dtype)
      return self._multi_like(lambda shape, dev: type(self).rand(*shape, dtype=dtype, device=dev, **kwargs))
    return type(self).rand(*self.shape, device=kwargs.pop("device", self.device), dtype=kwargs.pop("dtype", self.dtype), **kwargs)

  def randn_like(self, dtype:DTypeLike|None=None, **kwargs) -> Self:
    """
    Creates a tensor with the same shape and sharding as `self`, filled with random values from a normal distribution with mean 0 and variance 1.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(Tensor.randn_like(t).numpy())
    ```
    """
    if (dt:=to_dtype(dtype or self.dtype)) in dtypes.weaks and dtype is None: raise ValueError(f"randn_like requires an explicit dtype for {dt}")
    src = self.stack(self).rand_like(**{**kwargs, "dtype": dtypes.float32})
    # https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    return src[0].mul(2*math.pi).cos().mul((1 - src[1]).log().mul(-2).sqrt()).cast(dt)

  @classmethod
  def randn(cls, *shape, dtype:DTypeLike|None=None, **kwargs) -> Self:
    """
    Creates a tensor with the given shape, filled with random values from a normal distribution with mean `0` and standard deviation `1`.
    If `dtype` is not specified, the default type is used.

    You can pass in the `device` keyword argument to control device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.randn(2, 3).numpy())
    ```
    """
    return cls.empty(*shape, **kwargs).randn_like(dtype=dtype)  # type: ignore[attr-defined]

  @classmethod
  def randint(cls, *shape, low=0, high=10, dtype=dtypes.int32, **kwargs) -> Self:
    """
    Creates a tensor with the given shape, filled with random integer values generated uniformly from the interval `[low, high)`.
    Requires `low < high`. If `dtype` is not specified, the default type is used.

    You can pass in the `device` keyword argument to control device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.randint(2, 3, low=5, high=10).numpy())
    ```
    """
    if not all_int([low, high]): raise TypeError(f"{low=} and {high=} must be integers")
    if not dtypes.is_int(dtype := to_dtype(dtype)): raise TypeError(f"{dtype=} must be int")
    if low >= high: raise ValueError(f"Tensor.randint requires low < high, got {low=}, {high=}")
    return cls.uniform(*shape, low=low, high=high, dtype=dtype, **kwargs)

  @classmethod
  def normal(cls, *shape, mean=0.0, std=1.0, **kwargs) -> Self:
    """
    Creates a tensor with the given shape, filled with random values from a normal distribution with the given `mean` and standard deviation `std`.
    Requires `std >= 0`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.normal(2, 3, mean=10, std=2).numpy())
    ```
    """
    if std < 0: raise ValueError(f"Tensor.normal requires std >= 0, got {std=}")
    return std * cls.randn(*shape, **kwargs) + mean

  @classmethod
  def uniform(cls, *shape, low=0.0, high=1.0, dtype:DTypeLike|None=None, **kwargs) -> Self:
    """
    Creates a tensor with the given shape, filled with random values from a uniform distribution over the interval `[low, high)`.
    Requires `low < high`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.uniform(2, 3, low=2, high=10).numpy())
    ```
    """
    if not all_int(shape:=argfix(*shape)) or not all(s >= 0 for s in shape): raise ValueError(f"invalid input {shape=}")
    if low >= high: raise ValueError(f"Tensor.uniform requires low < high, got {low=}, {high=}")
    return ((high-low) * cls.rand(*shape, **kwargs)).cast(dtype or dtypes.default_float) + low

  @classmethod
  def scaled_uniform(cls, *shape, **kwargs) -> Self:
    """
    Creates a tensor with the given shape, filled with random values from a uniform distribution
    over the interval `[-prod(shape)**-0.5, prod(shape)**-0.5)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.scaled_uniform(2, 3).numpy())
    ```
    """
    return cls.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul(prod(argfix(*shape))**-0.5)

  @classmethod
  def glorot_uniform(cls, *shape, **kwargs) -> Self:
    """
    <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform>

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.glorot_uniform(2, 3).numpy())
    ```
    """
    bound = (6 / (argfix(*shape)[0]+prod(argfix(*shape)[1:]))) ** 0.5
    return cls.uniform(*shape, low=-bound, high=bound, **kwargs)

  @classmethod
  def kaiming_uniform(cls, *shape, a:float = 0.01, **kwargs) -> Self:
    """
    <https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_>

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.kaiming_uniform(2, 3).numpy())
    ```
    """
    bound = (6 / (1 + a ** 2) / prod(argfix(*shape)[1:])) ** 0.5
    return cls.uniform(*shape, low=-bound, high=bound, **kwargs)

  @classmethod
  def kaiming_normal(cls, *shape, a:float = 0.01, **kwargs) -> Self:
    """
    <https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_>

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.kaiming_normal(2, 3).numpy())
    ```
    """
    std = (2 / (1 + a ** 2) / prod(argfix(*shape)[1:])) ** 0.5
    return cls.normal(*shape, mean=0.0, std=std, **kwargs)

  @classmethod
  def randperm(cls, n:int, device=None, dtype=dtypes.int32, **kwargs) -> Self:
    """
    Returns a tensor with a random permutation of integers from `0` to `n-1`.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.randperm(6).numpy())
    ```
    """
    return cls.rand(n, device=device, **kwargs).argsort().cast(dtype)

  def multinomial(self, num_samples:int = 1, replacement:bool = False) -> Self:
    """
    Returns a tensor with `num_samples` indices sampled from a multinomial distribution weighted by `self`.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor([1, 2, 3, 4])
    print(t.multinomial(20, replacement=True).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor([1, 2, 3, 4])
    print(t.multinomial(3, replacement=False).numpy())
    ```
    """
    assert 1 <= self.ndim <= 2 and num_samples > 0, f"{self.ndim=} must be 1 or 2 dim, {num_samples=} must be positive"
    weight = self.unsqueeze(0) if self.ndim == 1 else self
    assert replacement or num_samples <= weight.shape[1], "no replacement samples must not exceed population size"
    if replacement or num_samples == 1:
      cdf = (cw := weight.cumsum(1).float()) / cw[:, -1].unsqueeze(1)
      unif_samples = type(self).rand(num_samples, cdf.shape[0], 1).to(self.device)  # type: ignore[attr-defined]
      indices = (unif_samples.expand((-1, -1, cdf.shape[1])) >= cdf).sum(2).permute((1, 0))
    else:
      # Efraimidis-Spirakis
      indices = (weight.rand_like(dtype=dtypes.float32).log2() / weight).topk(num_samples, dim=1)[1]
    return (indices.squeeze(0) if self.ndim == 1 else indices).cast(dtypes.int32)

  def dropout(self, p=0.5) -> Self:
    """
    Applies dropout to `self`.

    NOTE: dropout is only applied when `TRAINING` is set (e.g. inside `Context(TRAINING=1)`).

    - Paper: https://jmlr.org/papers/v15/srivastava14a.html

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 2)
    with Context(TRAINING=1):
      print(t.dropout().numpy())
    ```
    """
    if not 0 <= p <= 1: raise ValueError(f"{p=} is out of range [0, 1]")
    if not TRAINING or p == 0: return self
    if p == 1: return self.const_like(0)
    return (self.rand_like(dtype=dtypes.default_float, contiguous=False) >= p).contiguous().where(self, 0) / (1.0 - p)

  def scaled_dot_product_attention(self, key:Self, value:Self, attn_mask:Self|None=None, dropout_p:float=0.0,
                                   is_causal:bool=False, enable_gqa:bool=False) -> Self:
    """
    Computes scaled dot-product attention.
    `self` is the query tensor, `key` is the key tensor, and `value` is the value tensor.

    - Paper: https://arxiv.org/abs/1706.03762v7

    ```python exec="true" source="above" session="tensor" result="python"
    q = Tensor.randn(2, 4, 8)
    k = Tensor.randn(2, 4, 8)
    v = Tensor.randn(2, 4, 8)
    print(q.scaled_dot_product_attention(k, v).numpy())
    ```
    """
    # GQA: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    if enable_gqa:
      key = key.repeat_interleave(int(self.shape[-3] // key.shape[-3]), dim=-3)
      value = value.repeat_interleave(int(self.shape[-3] // value.shape[-3]), dim=-3)

    q = self
    qk = q.matmul(key.transpose(-2,-1), dtype=least_upper_dtype(q.dtype, key.dtype, dtypes.float32)) / math.sqrt(q.shape[-1])
    # handle attention mask
    if is_causal:
      if attn_mask is not None: raise RuntimeError("cannot set attn_mask when is_causal=True")
      attn_mask = qk.const_like(1).cast(dtypes.bool).tril()
    if attn_mask is not None:
      if attn_mask.dtype == dtypes.bool: attn_mask = attn_mask.where(0, -float("inf"))
      qk = qk + attn_mask
    return qk.cast(self.dtype).softmax(-1).dropout(dropout_p) @ value
