# ruff: noqa: E501, A001, A002, A006
# A001 Variable `input` is shadowing a Python builtin
# A002 Function argument `input` is shadowing a Python builtin
# A006 Lambda argument `input` is shadowing a Python builtin
from tinygrad import Tensor, dtypes, Device
from tinygrad.uop.ops import Ops
from tinygrad.helpers import getenv, prod
import torch.lib
TORCH_DEBUG = getenv("TORCH_DEBUG")
import torch, pathlib, math, operator, functools, inspect
torch.autograd.grad_mode.set_multithreading_enabled(False)
from tinygrad.dtype import _from_torch_dtype, _to_torch_dtype

# https://pytorch.org/docs/stable/torch.compiler_ir.html

def _from_torch_device(device: torch.device): return f"{Device.DEFAULT}:{device.index or 0}"
def _to_torch_device(device: str): return torch.device("tiny", int(device.partition(":")[2] or 0))

import torch.utils.cpp_extension
mod = torch.utils.cpp_extension.load(name="custom_device_extension", sources=[str(pathlib.Path(__file__).parent / "wrapped_tensor.cpp")])
def wrap(x:Tensor) -> torch.Tensor: return mod.wrap(x, _to_torch_dtype(x.dtype), _to_torch_device(x.device).index)
def unwrap(x:torch.Tensor) -> Tensor:
  assert isinstance(x, torch.Tensor), f"x isn't {type(x)}"
  return mod.unwrap(x)
class TinyBackend:
  def is_initialized(self): return True
  def is_available(self): return True
  def current_device(self): return 0
  def _is_in_bad_fork(self): return False
  def manual_seed_all(self, seed: int): Tensor.manual_seed(seed)
  def device_count(self): return getenv("GPUS", 1) # TODO: device count in tiny?
torch.utils.rename_privateuse1_backend("tiny")
torch._register_device_module("tiny", TinyBackend())
torch.utils.generate_methods_for_privateuse1_backend()
aten = torch.ops.aten

# track view relationships for in place operations
def is_view(tensor: Tensor): return hasattr(tensor, "_view_base")
def canonical_base(view: Tensor): return getattr(view, "_view_base", view)
def derived_views(base: Tensor): return [t for tref in getattr(base, "_views", set()) if (t:=tref()) is not None]
def wrap_view_op(fn):
  def _wrap(*args,**kwargs):
    args = [unwrap(x) if isinstance(x, torch.Tensor) else x for x in args]
    kwargs = {k:unwrap(v) if isinstance(v, torch.Tensor) else v for k,v in kwargs.items()}
    ret = fn(*args,**kwargs)
    ret._view_base = base = canonical_base(args[0])
    if not hasattr(base, "_views"): base._views = set()
    base._views.add(weakref.ref(ret))
    return wrap(ret)
  return _wrap

view_ops = {
  "aten.view": Tensor.reshape,
  "aten._unsafe_view": Tensor.reshape,  # when are views unsafe, and do we care?
  "aten.view.dtype": lambda self,dtype: self.bitcast(_from_torch_dtype(dtype)),
  "aten.expand": Tensor.expand,
  "aten.t": Tensor.transpose,
  "aten.transpose.int": Tensor.transpose,
  "aten.squeeze.dim": Tensor.squeeze,
  "aten.unsqueeze": Tensor.unsqueeze,
  "aten.detach": Tensor.detach,
  "aten.select.int": lambda self, dim, idx: self[(slice(None),) * (dim%self.ndim) + (idx,)],
}

for k,v in view_ops.items(): torch.library.impl(k.replace("aten.", "aten::"), "privateuseone")(wrap_view_op(v))

# in place operations with views
def realize_with_views(self: Tensor, views: Tensor):
  if not self.uop.st.contiguous: self.replace(self.contiguous())
  self.replace(self.clone().realize())
  for v in views:
    if v.uop.base.op is Ops.BUFFER_VIEW: continue # skip subbuffer, we just use the real buffer view
    ret = self
    st = ShapeTracker(self.uop.st.views + v.uop.st.views) # TODO: is this right?
    for mo in cached_to_movement_ops(self.shape, st): ret = apply_mop(ret, mo)
    v.replace(ret)
def maybe_realize_storage(self: Tensor) -> bool:
  if realize:=is_view(self): realize_with_views((base:=canonical_base(self)), derived_views(base))
  return realize
def inplace_fn(outvars: str|list[str]):
  if type(outvars) is str: outvars = [outvars]
  def decorator(fn):
    sig = inspect.signature(fn)
    def wrapper(*args, **kwargs):
      bound = sig.bind(*args, **kwargs)
      outs = [kwargs.get(v, bound.arguments.get(v)) for v in outvars]
      outs = [unwrap(o) if isinstance(o, torch.Tensor) else o for o in outs]
      realize = any(maybe_realize_storage(o) for o in outs)
      ret = fn(*args, **kwargs)
      if realize: Tensor.realize(*(o for o in outs))
      return ret
    return wrapper
  return decorator

# *** bad functions on CPU ***

@torch.library.impl("aten::_index_put_impl_", "privateuseone")
@inplace_fn("self")
def _index_put_impl_(self, indices, values, accumulate=False, unsafe=False):
  # TODO: move to tinygrad
  ret = aten._index_put_impl_(self.cpu(), [x.cpu() if isinstance(x, torch.Tensor) else None for x in indices], values.cpu(), accumulate, unsafe).to(self.device)
  return wrap(unwrap(self).assign(unwrap(ret)))

@torch.library.impl("aten::index_put", "privateuseone")
def index_put(self, indices, values, accumulate=False):
  return aten.index_put(self.cpu(), [z.cpu() if isinstance(z, torch.Tensor) else None for z in indices], values.clone().cpu(), accumulate).tiny()

@torch.library.impl("aten::isin.Tensor_Tensor_out", "privateuseone")
def isin_tensor_tensor_out(x, y, *, assume_unique=False, invert=False, out=None): return out.copy_(aten.isin(x.cpu(), y.cpu(), assume_unique=assume_unique, invert=invert).tiny())

@torch.library.impl("aten::randperm.generator_out", "privateuseone")
def randperm_generator(n, generator=None, out=None):
  return out.copy_(wrap(Tensor.randperm(n, generator=generator, device=unwrap(out).device)))

@torch.library.impl("aten::cummax", "privateuseone")
def cummax(self, dim):
  # TODO: support cummax with indices to match torch
  cummax, indices = aten.cummax(self.cpu(), dim)
  return (cummax.tiny(), indices.tiny())

@torch.library.impl("aten::nonzero", "privateuseone")
# TODO: move to tinygrad
def nonzero(self): return aten.nonzero(self.cpu()).tiny()

@torch.library.impl("aten::_linalg_eigh", "privateuseone")
# TODO: move to tinygrad
def _linalg_eigh(self, UPLO: str = 'U'):
  w, v = torch.linalg.eigh(self.cpu(), UPLO=UPLO)
  return w.tiny(), v.tiny()

@torch.library.impl("aten::_linalg_det", "privateuseone")
# TODO: move to tinygrad
def _linalg_det(self: torch.Tensor):
  result = aten._linalg_det(self.cpu())
  return result[0].tiny(), result[1].tiny(), result[2].tiny()

def upsample_backward(grad_out, output_size, input_size, *args, f=None): return f(grad_out.cpu(), output_size, input_size, *args).tiny()

for i in [
  "upsample_linear1d_backward", "upsample_nearest1d_backward", "_upsample_nearest_exact1d_backward",
  "upsample_nearest2d_backward", "_upsample_nearest_exact2d_backward",
  "upsample_nearest3d_backward", "_upsample_nearest_exact3d_backward",
  "upsample_trilinear3d_backward", "upsample_bilinear2d_backward"
]:
  torch.library.impl(f"aten::{i}", "privateuseone")(functools.partial(upsample_backward, f=getattr(aten, i)))

# *** end bad functions on CPU ***

@torch.library.impl("aten::index.Tensor", "privateuseone")
def index_tensor(x, y):
  return wrap(unwrap(x)[[unwrap(_y.to(x.device)) if _y is not None else slice(None) for _y in y]])

@torch.library.impl("aten::zero_", "privateuseone")
@inplace_fn("x")
def zero_(x):
  if TORCH_DEBUG: print(f"zero_ {x.shape}")
  tt = unwrap(x)
  # NOTE: unconditional contiguous covers if x is contiguous (match it) or if x is view (realize for inplace)
  # TODO: consolidate
  tt.assign(tt.zeros_like().contiguous())

@torch.library.impl("aten::fill_.Scalar", "privateuseone")
@inplace_fn("x")
def fill_scalar(x, y):
  if TORCH_DEBUG: print(f"fill_.Scalar {x.shape} {y}")
  tt = unwrap(x)
  tt.assign(tt.full_like(y).contiguous())

@torch.library.impl("aten::_local_scalar_dense", "privateuseone")
def _local_scalar_dense(tensor): return unwrap(tensor).item()

@functools.cache
def cached_to_movement_ops(shape, st) -> list:
  mops = to_movement_ops(st)
  if mops[0] == (MovementOps.RESHAPE, shape): mops = mops[1:]
  return mops

from tinygrad.shape.shapetracker import ShapeTracker, View
from extra.to_movement_ops import to_movement_ops, apply_mop, MovementOps
@torch.library.impl("aten::as_strided", "privateuseone")
def as_strided(tensor:torch.Tensor, size, stride, storage_offset=None):
  storage_offset = storage_offset or tensor.storage_offset()
  @wrap_view_op
  def _as_strided(tensor:Tensor, size, stride, storage_offset=None):
    # multiple as_strided do not compound
    base = canonical_base(tensor)
    # TODO: this is heavyweight
    st = ShapeTracker(base.uop.st.views + (View.create(tuple(size), tuple(stride), storage_offset),))
    ret = base
    if TORCH_DEBUG >= 1: print("**** as_strided", tensor.shape, size, stride, st)
    if prod(size) == 1: return ret.flatten()[storage_offset].reshape(size)
    for mo in cached_to_movement_ops(tuple(base.shape), st): ret = apply_mop(ret, mo)
    return ret
  return _as_strided(tensor, size, stride, storage_offset)

@torch.library.impl("aten::empty_strided", "privateuseone")
def empty_strided(size, stride, dtype, layout=None, device=None, pin_memory=False):
  if TORCH_DEBUG: print(f"empty_strided {size=} {stride=} {dtype=} {layout=} {device=} {pin_memory=}")
  ret = Tensor.empty(*size, dtype=_from_torch_dtype(dtype), device=_from_torch_device(device)).contiguous()
  # TODO: should return with requested strides
  return wrap(ret)

@torch.library.impl("aten::empty.memory_format", "privateuseone")
def empty_memory_format(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
  if TORCH_DEBUG: print(f"empty.memory_format {size=} {dtype=} {layout=} {device=} {pin_memory=} {memory_format=}")
  ret = Tensor.empty(*size, dtype=_from_torch_dtype(dtype or torch.get_default_dtype()), device=_from_torch_device(device)).contiguous()
  return wrap(ret)

@torch.library.impl("aten::max_pool2d_with_indices", "privateuseone")
def max_pool2d_with_indices(self:torch.Tensor, kernel_size:tuple[int, ...], stride=None, padding=0, dilation=1, ceil_mode=False):
  # TODO: supprt stride [] in tinygrad?
  if stride is not None and len(stride) == 0: stride = None
  ret, idx = unwrap(self).max_pool2d(kernel_size, stride, dilation, padding, ceil_mode, return_indices=True)
  return (wrap(ret), wrap(idx.cast(dtypes.int64)))

@torch.library.impl("aten::max_pool2d_with_indices_backward", "privateuseone")
def max_pool2d_with_indices_backward(grad_out:torch.Tensor, self:torch.Tensor, kernel_size:tuple[int, ...], stride=None, padding=0, dilation=1, ceil_mode=False, indices=None):
  return wrap(Tensor.max_unpool2d(unwrap(grad_out), unwrap(indices), output_size=unwrap(self).shape))

@torch.library.impl("aten::max_unpool2d", "privateuseone")
def max_unpool2d(self:torch.Tensor, indices:torch.Tensor, output_size):
  return wrap(unwrap(self).max_unpool2d(unwrap(indices), output_size=output_size))

@torch.library.impl("aten::arange", "privateuseone")
def arange(end, dtype=None, device=None, pin_memory=None):
  return wrap(Tensor.arange(0, end, dtype=_from_torch_dtype(dtype or torch.get_default_dtype())))

@torch.library.impl("aten::arange.start", "privateuseone")
def arange_start(start, end, dtype=None, device=None, pin_memory=None):
  return wrap(Tensor.arange(start, end, dtype=_from_torch_dtype(dtype or torch.get_default_dtype())))

@torch.library.impl("aten::arange.start_step", "privateuseone")
def arange_start_step(start, end, step, dtype=None, device=None, pin_memory=None):
  return wrap(Tensor.arange(start, end, step, dtype=_from_torch_dtype(dtype or torch.get_default_dtype())))

@torch.library.impl("aten::convolution_overrideable", "privateuseone")
def convolution_overrideable(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
  if TORCH_DEBUG >= 1:
    print(f"convolution {input.shape=} {weight.shape=} {stride=} {padding=} {dilation=} {transposed=} {output_padding=} {groups=}")
  input, weight, bias = unwrap(input), unwrap(weight), unwrap(bias) if bias is not None else None
  # TODO: fix test_biased_conv2d fails without realize()
  if not transposed: return wrap(input.conv2d(weight, bias, groups=groups, stride=stride, dilation=dilation, padding=padding).realize())
  return wrap(input.conv_transpose2d(weight, bias, groups=groups, stride=stride, dilation=dilation, padding=padding, output_padding=output_padding).realize())

@torch.library.impl("aten::convolution_backward_overrideable", "privateuseone")
def convolution_backward_overrideable(grad_out, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask):
  if TORCH_DEBUG >= 1:
    print(f"convolution_backward {input.shape=} {weight.shape=} {stride=} {padding=} {dilation=} {transposed=} {output_padding=} {groups=}")
  grad_out, input, weight, bias = unwrap(grad_out), unwrap(input), unwrap(weight), Tensor.zeros(weight.shape[0], device=_from_torch_device(weight.device))
  if not transposed: out = Tensor.conv2d(input, weight, bias, groups=groups, stride=stride, dilation=dilation, padding=padding)
  else:
    bias = Tensor.zeros(weight.shape[1] * groups)
    out = Tensor.conv_transpose2d(input, weight, bias, groups=groups, stride=stride, dilation=dilation, padding=padding, output_padding=output_padding)
  grads = out.gradient(*[t for t,m in zip([input, weight, bias], output_mask) if m], gradient=grad_out)
  return tuple([wrap(grads.pop(0)) if m else None for m in output_mask])

@torch.library.impl("aten::slice.Tensor", "privateuseone")
@wrap_view_op
def slice_tensor(self, dim=0, start=None, end=None, step=1):
  slices = [slice(None)] * self.ndim
  slices[dim] = slice(start, end, step)
  return self[slices]

@torch.library.impl("aten::slice_backward", "privateuseone")
def slice_backward(grad_out, input_sizes, dim, start, end, step):
  grad_input = Tensor.zeros(input_sizes).contiguous()
  slices = [slice(None)] * len(input_sizes)
  slices[dim] = slice(start, end, step)
  grad_input[slices] = unwrap(grad_out)
  return wrap(grad_input)

@torch.library.impl("aten::select_backward", "privateuseone")
def select_backward(grad_out, input_sizes, dim, index):
  grad_input = Tensor.zeros(input_sizes).contiguous()
  slices = [slice(None)] * len(input_sizes)
  slices[dim] = index
  grad_input[slices] = unwrap(grad_out)
  return wrap(grad_input)

def avg_pool(self, kernel_size, stride=[], padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
  return wrap(unwrap(self).avg_pool2d(kernel_size, stride if stride != [] else None, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad))

def avg_pool_backward(grad_out, self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
  self, grad_out = unwrap(self), unwrap(grad_out)
  out = Tensor.avg_pool2d(self, kernel_size, stride if stride != [] else None, dilation=1, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)
  return wrap(out.gradient(self, gradient=grad_out)[0])

for dim in [2, 3]:
  torch.library.impl(f"aten::avg_pool{dim}d", "privateuseone")(avg_pool)
  torch.library.impl(f"aten::avg_pool{dim}d_backward", "privateuseone")(avg_pool_backward)

def pad_forward(self, padding, mode=None): return wrap(Tensor.pad(unwrap(self), padding, mode=mode))

def pad_backward(grad_out, self, padding, mode):
  self, grad_out = unwrap(self), unwrap(grad_out)
  out = Tensor.pad(self, padding, mode=mode)
  return wrap(out.gradient(self, gradient=grad_out)[0])

for dim in [1, 2, 3]:
  for pad_type, mode in [("replication", "replicate"), ("reflection", "reflect")]:
    torch.library.impl(f"aten::{pad_type}_pad{dim}d", "privateuseone")(functools.partial(pad_forward, mode=mode))
    torch.library.impl(f"aten::{pad_type}_pad{dim}d_backward", "privateuseone")(functools.partial(pad_backward, mode=mode))

def upsample(self, size, align_corners=False, mode=None): return wrap(Tensor.interpolate(unwrap(self), size, mode=mode, align_corners=align_corners))
for i,pre in enumerate(["", "bi", "tri"]):
  torch.library.impl(f"aten::upsample_{pre}linear{i+1}d", "privateuseone")(functools.partial(upsample, mode="linear"))
  torch.library.impl(f"aten::upsample_nearest{i+1}d", "privateuseone")(functools.partial(upsample, mode="nearest"))
  torch.library.impl(f"aten::_upsample_nearest_exact{i+1}d", "privateuseone")(functools.partial(upsample, mode="nearest-exact"))

@torch.library.impl("aten::scatter_add.out", "privateuseone")
@inplace_fn("out")
def scatter_add(self, dim, index, src, out):
  self, index, src, out = unwrap(self), unwrap(index), unwrap(src), unwrap(out)
  if self.shape == (): return wrap(out.assign(src))
  return wrap(out.assign(Tensor.scatter_reduce(self, dim, index, src, reduce='sum')))

@torch.library.impl("aten::_copy_from", "privateuseone")
def _copy_from(src: torch.Tensor, dest, non_blocking=False):
  realize = dest.is_tiny and maybe_realize_storage(unwrap(dest))
  cast_dtype = _from_torch_dtype(dest.dtype)
  if src.is_tiny and dest.is_tiny:
    to_device = _from_torch_device(dest.device)
    src,dest = unwrap(src),unwrap(dest)
    # TODO we need to properly match dest shape and strides, not blindly assign
    if dest.uop.st.contiguous or dest.uop.is_realized: src = src.contiguous() # this only solves some cases
    dest.assign(src.cast(cast_dtype).to(to_device))
    if realize: Tensor.realize(dest)
  elif src.is_tiny and dest.is_cpu:
    # TODO: is there a better way?
    dest.resize_(src.numel()).resize_(src.shape)
    dest.copy_(torch.from_numpy(unwrap(src).cast(cast_dtype).numpy()))
  elif src.is_cpu and dest.is_tiny:
    to_device = _from_torch_device(dest.device)
    # TODO we need to properly match dest shape and strides, not blindly assign
    unwrap(dest).assign(Tensor(src.numpy()).cast(cast_dtype).to(to_device))
    if realize: Tensor.realize(unwrap(dest))
  else:
    raise NotImplementedError(f"can't copy from {src.device} -> {dest.device}")

@torch.library.impl("aten::cat.out", "privateuseone")
@inplace_fn("out")
def cat_out(tensors, dim=0, out=None):
  unwrap(out).assign(Tensor.cat(*[unwrap(x) for x in tensors], dim=dim))

@torch.library.impl("aten::topk.values", "privateuseone")
@inplace_fn(["values", "indices"])
def topk_values(input, k, dim=None, largest=True, sorted=True, values=None, indices=None):
  out_values, out_indices = unwrap(input).topk(k, dim if dim is not None else -1, largest, sorted)
  unwrap(values).assign(out_values)
  unwrap(indices).assign(out_indices.cast(dtypes.int64))
  return wrap(out_values), wrap(out_indices)

@torch.library.impl("aten::sort.values_stable", "privateuseone")
@inplace_fn(["values", "indices"])
def sort_values(input, dim=-1, descending=False, stable=True, values=None, indices=None):
  out_values, out_indices = unwrap(input).sort(dim, descending)
  unwrap(values).assign(out_values)
  unwrap(indices).assign(out_indices.cast(dtypes.int64))
  return wrap(out_values), wrap(out_indices)

@torch.library.impl("aten::_linalg_svd", "privateuseone")
def _linalg_svd(self, full_matrices=False):
  U, S, Vh = unwrap(self).svd(full_matrices)
  return wrap(U), wrap(S), wrap(Vh)

# register some decompositions
from torch._decomp import get_decompositions
decomps = [
  aten.native_batch_norm, aten.native_batch_norm_backward,
  aten.native_layer_norm_backward,
  aten.addmm,
  aten.addcmul,
  aten.addcdiv,
  aten._log_softmax_backward_data,
  aten.threshold_backward,
  aten.softplus_backward,
  aten.elu,  # elu has a scale + input_scale param
  aten.elu_backward,
  aten.softplus,
  aten.threshold,
  aten.nll_loss_forward,
  aten.nll_loss_backward,
  aten.nll_loss2d_backward,
  # AttributeError: 'int' object has no attribute '_broadcasted'
  aten.sigmoid_backward,
  aten.tanh_backward,
  aten.sinc,
  aten._prelu_kernel,
  aten.softshrink,
  aten.hardshrink,
  aten.log_sigmoid_forward,
  aten.log_sigmoid_backward,
  aten.isneginf,
  aten.isposinf,
  aten.nan_to_num,
  aten.logit,
  aten.rsub,
  aten.index_select,
  aten.native_dropout, aten.native_dropout_backward,
  aten._softmax_backward_data, aten.embedding_dense_backward,
  aten.linalg_vector_norm,
  aten.binary_cross_entropy, aten.binary_cross_entropy_backward,
  aten.upsample_nearest2d.out,
  # activations
  aten.hardswish, aten.hardswish_backward,
  aten.hardtanh, aten.hardtanh_backward,
  aten.gelu, aten.gelu_backward,
  aten.logical_and,
  aten.randint,
  aten.eye,
  aten.hardsigmoid_backward,
  aten.leaky_relu_backward,
  aten.nll_loss2d_forward,
  aten.unfold_backward,
  # NOTE: many of these don't work or cause infinite loops
  #aten.var_mean,
  #aten.var,
  #aten.rsqrt,
  #aten.max_pool2d_with_indices,
  # NOTE: these are prims
  #aten.digamma,
  #aten.erfinv,
  #aten.lgamma,
  # this needs copy_strided
  #aten.lerp,
  aten.norm,
]
for k,v in get_decompositions(decomps).items():
  key = str(k._schema).split("(")[0]
  if TORCH_DEBUG >= 2: print("register decomp for", k)
  torch.library.impl(key, "privateuseone")(v)

# NOTE: we should only implement the "out" form, it should be 0 overhead
# TODO: due to issue with empty / is_realized, it is slow to use assign so we use replace
# the goal is to make as much as we can this
simple_tensor_methods = [
  # unary (ish)
  "log", "log2", "sqrt", "rsqrt", "sign", "silu", "hardsigmoid", "exp", "exp2", "neg", "reciprocal", "bitwise_not",
  "sigmoid", "clamp", "mish", "erf", "leaky_relu",
  # trig
  "acos", "acosh", "cos", "cosh", "asin", "asinh", "sin", "sinh", "atan", "atanh", "tan", "tanh",
  # rounding
  "ceil", "round", "floor", "trunc",
  # binary
  "mul", "div", "maximum", "minimum", "copysign",
  # modify
  "tril", "triu",
  # reduce
  "all", "any", "argmax", "argmin", "cumsum", "cumprod",
  # complex
  "avg_pool2d", "linspace"]

tiny_backend_out = {**{f"aten.{x}.out":getattr(Tensor,x) for x in simple_tensor_methods}, **{
  "aten.add.out": lambda input,other,alpha=1: input+alpha*other,
  "aten.sub.out": lambda input,other,alpha=1: input-alpha*other, # NOTE: this is also needed to handle reverse
  "aten.div.out_mode": Tensor.div,
  "aten.mul.out": operator.mul,
  "aten.bmm.out": operator.matmul,
  # NOTE: because these methods have a name with "Tensor" in them, they can't go in simple tensor methods
  "aten.remainder.Tensor_out": Tensor.mod,
  "aten.pow.Tensor_Tensor_out": Tensor.pow,
  "aten.pow.Tensor_Scalar_out": Tensor.pow,
  "aten.pow.Scalar_out": lambda input,exponent: input**exponent,
  "aten.bitwise_and.Tensor_out": Tensor.bitwise_and,
  "aten.bitwise_or.Tensor_out": Tensor.bitwise_or,
  "aten.bitwise_xor.Tensor_out": Tensor.bitwise_xor,
  "aten.eq.Tensor_out": Tensor.eq, "aten.eq.Scalar_out": Tensor.eq,
  "aten.ne.Tensor_out": Tensor.ne, "aten.ne.Scalar_out": Tensor.ne,
  "aten.ge.Tensor_out": Tensor.__ge__, "aten.ge.Scalar_out": Tensor.__ge__,
  "aten.gt.Tensor_out": Tensor.__gt__, "aten.gt.Scalar_out": Tensor.__gt__,
  "aten.lt.Tensor_out": Tensor.__lt__, "aten.lt.Scalar_out": Tensor.__lt__,
  "aten.le.Tensor_out": Tensor.__le__, "aten.le.Scalar_out": Tensor.__le__,
  "aten.clamp_max.Tensor_out": lambda input,max_: input.clamp(max_=max_),
  "aten.clamp_min.Tensor_out": lambda input,min_: input.clamp(min_=min_),
  "aten.fmod.Tensor_out": lambda input,other: input-input.div(other, rounding_mode="trunc")*other,
  # TODO: this might result in overflow issues
  "aten.round.decimals_out": lambda self,decimals: (self*10**decimals).round()/10**decimals,
  # TODO: support this in tinygrad
  "aten.bitwise_left_shift.Tensor_out": lambda x,y: x*(2**y),
  "aten.bitwise_right_shift.Tensor_out": lambda x,y: x//(2**y),
  # not in tinygrad. are there decomps for these?
  "aten.log10.out": lambda self: self.log2() * (math.log(2) / math.log(10)),
  "aten.log1p.out": lambda self: (self+1).log(),
  "aten.expm1.out": lambda self: self.exp() - 1,
  "aten.fmax.out": lambda input,other: Tensor.where(input.isnan() & ~other.isnan(), other, Tensor.where(~input.isnan() & other.isnan(), input, Tensor.maximum(input, other))),
  "aten.fmin.out": lambda input,other: Tensor.where(input.isnan() & ~other.isnan(), other, Tensor.where(~input.isnan() & other.isnan(), input, Tensor.minimum(input, other))),
  "aten.amax.out": lambda self,dim=None: self.max(axis=dim),
  "aten.amin.out": lambda self,dim=None: self.min(axis=dim),
  # TODO: this gets the shape wrong
  #"aten.arange.start_out": Tensor.arange,
  "aten.lerp.Scalar_out": Tensor.lerp,
  "aten.scatter.value_out": Tensor.scatter,
  "aten.where.self_out": Tensor.where,
  "aten.prod.int_out": Tensor.prod,
  "aten.scatter.src_out": Tensor.scatter,
  # NOTE: axis=[] in torch means all, change tinygrad?
  "aten.sum.IntList_out": lambda self,axis,keepdim=False,dtype=None:
    self.sum(axis if axis is None or len(axis) else None, keepdim,
                         dtype = _from_torch_dtype(dtype) if dtype is not None else None),
}}

# we add the "out" here
def wrap_out(f):
  @inplace_fn("out")
  def _wrap_out(*args, **kwargs):
    out = kwargs.pop('out')
    assigned = f(*args, **kwargs)
    if getenv("ALLOW_DTYPE_MISMATCH", 1): assigned = assigned.cast(out.dtype)
    assert out.shape == assigned.shape, f"shape mismatch: {assigned.shape} -> {out.shape}"
    assert out.device == assigned.device, f"device mismatch: {assigned.device} -> {out.device}"
    assert out.dtype == assigned.dtype, f"dtype mismatch: {assigned.dtype} -> {out.dtype}"
    if out.uop.is_realized: assigned = assigned.contiguous() # TODO: how does this map to torch's semantics
    return out.assign(assigned)
  return _wrap_out

tiny_backend = {**{k:wrap_out(v) for k,v in tiny_backend_out.items()}, **{
  "aten.remainder.Scalar_Tensor": lambda x,y: x%y,
  "aten.floor_divide": lambda x,y: x//y,
  "aten.floor_divide_.Tensor": inplace_fn("x")(lambda x,y: x.assign(x//y)),
  # TODO: use tinygrad methods, but they require x to be unsigned
  "aten.__lshift__.Scalar": lambda x,y: x*(2**y),
  "aten.__ilshift__.Scalar": inplace_fn("x")(lambda x,y: x.assign(x*(2**y))),
  "aten.__rshift__.Scalar": lambda x,y: x//(2**y),
  "aten.__irshift__.Scalar": inplace_fn("x")(lambda x,y: x.assign(x//(2**y))),
  # relu doesn't have an out form?
  "aten.relu": Tensor.relu,
  "aten.relu_": inplace_fn("x")(lambda x: x.assign(x.relu())),
  "aten.mean": Tensor.mean,
  "aten.mean.dim": Tensor.mean,
  "aten.min": Tensor.min,
  "aten.max": Tensor.max,
  "aten.mm": Tensor.matmul,
  "aten.mv": Tensor.matmul,
  "aten.dot": Tensor.dot,
  "aten.prod": Tensor.prod,
  "aten.isnan": Tensor.isnan,
  "aten.std.correction": Tensor.std,
  "aten.std_mean.correction": Tensor.std_mean,
  "aten.var.correction": Tensor.var,
  "aten.var_mean.correction": Tensor.var_mean,
  "aten.scatter.value": Tensor.scatter,
  "aten.scatter.value_reduce": Tensor.scatter,
  "aten.gather": lambda self, dim, index: self.gather(dim, index.cast(dtypes.int)),
  "aten.where.self": Tensor.where, # NOTE: this is needed as well as the out type
  "aten.repeat": lambda x,*repeats: Tensor.repeat(x,*repeats).contiguous(), # not a view
  "aten._softmax": lambda self,dim,half_to_float: self.softmax(dim),
  "aten._log_softmax": lambda self,dim,half_to_float: self.log_softmax(dim),
  "aten.random_": inplace_fn("self")(lambda self:
    self.assign(Tensor.randint(*self.shape, low=dtypes.min(self.dtype), high=dtypes.max(self.dtype), device=self.device, dtype=self.dtype))),
  "aten.random_.from": inplace_fn("self")(lambda self, from_, to:
    self.assign(Tensor.randint(*self.shape, low=from_, high=to, device=self.device, dtype=self.dtype))),
  "aten.uniform_": inplace_fn("self")(lambda self, low=0, high=1: self.assign(Tensor.uniform(*self.shape, low=low, high=high, dtype=self.dtype))),
  "aten.normal_": inplace_fn("self")(lambda self, mean=0, std=1: self.assign(Tensor.normal(*self.shape, mean=mean, std=std, dtype=self.dtype))),
  # these don't work in out form, they have size 0
  "aten.abs": Tensor.abs,
  "aten.logical_not": Tensor.logical_not,
  "aten.logical_or_": inplace_fn("x")(lambda x, y: x.assign(x | y)),
  "aten.multinomial": Tensor.multinomial,
  "aten.masked_fill_.Scalar": inplace_fn("self")(lambda self, mask, value: self.assign(self.masked_fill(mask, value))),
  "aten.masked_fill_.Tensor": inplace_fn("self")(lambda self, mask, value: self.assign(self.masked_fill(mask, value))),
  "aten.masked_fill.Scalar": Tensor.masked_fill,
  "aten.masked_fill.Tensor": Tensor.masked_fill,
  "aten.masked_select": Tensor.masked_select,
  "aten.all": Tensor.all,
  "aten.sgn": Tensor.sign,
  "aten.acos": Tensor.acos,
  "aten.any": Tensor.any,
  "aten.bitwise_not": Tensor.bitwise_not,
  "aten.argmax": Tensor.argmax,
  "aten.argmin": Tensor.argmin,
  "aten.asinh": Tensor.asinh,
  "aten.mul": Tensor.mul,
  "aten.atanh": Tensor.atanh,
  "aten.fill_.Tensor": Tensor.full, # TODO: looks wrong
  "aten.flip": Tensor.flip,
  "aten.scatter_reduce.two": Tensor.scatter_reduce,
  "aten.squeeze_.dim": lambda self, dim: self.replace(self.squeeze(dim), allow_shape_mismatch=True), # TODO: inplace view op, here?
  "aten.add.Tensor": lambda input,other,alpha=1: input+alpha*other,
  "aten.linspace": lambda start, stop, steps, dtype=None, **kwargs:
    Tensor.linspace(start, stop, steps, **({"dtype": _from_torch_dtype(dtype)} if dtype is not None else {})),
  "aten.topk": Tensor.topk,
  "aten.constant_pad_nd": lambda self, padding, value=0.0: self.pad(padding, mode="constant", value=value).contiguous(),
  "aten.cumsum": lambda self, dim: self.cumsum(dim).contiguous(), # TODO: fix test_simple_cumsum, fails without contiguous for shapes >512
  "aten.logsumexp": lambda self, axis, keepdim=False: self.logsumexp(axis[0], keepdim=keepdim),
  "aten.roll": Tensor.roll,
  "aten.logcumsumexp": Tensor.logcumsumexp,
  "aten.lerp.Tensor": Tensor.lerp,
  "aten.ones_like": lambda self, dtype=None, device=None, **kwargs:
    self.ones_like(**{k: v for k, v in {"dtype": _from_torch_dtype(dtype) if dtype else None,
                                        "device": _from_torch_device(device) if device else None}.items() if v is not None}),
  "aten.max.dim": lambda self, dim, keepdim=False: (self.max(dim, keepdim), self.argmax(dim, keepdim).cast(dtype=dtypes.int64)),
  "aten.unfold": Tensor.unfold,
}}

def wrap_fxn(k,f):
  def nf(*args, **kwargs):
    if TORCH_DEBUG:
      print(k, len(args), [x.shape if isinstance(x, torch.Tensor) else x for x in args],
                          {k:v.shape if isinstance(v, torch.Tensor) else v for k,v in kwargs.items()})
    args = [unwrap(x) if isinstance(x, torch.Tensor) else x for x in args]
    kwargs = {k:unwrap(v) if isinstance(v, torch.Tensor) else v for k,v in kwargs.items()}
    out = f(*args, **kwargs)
    if isinstance(out, Tensor): return wrap(out)
    elif isinstance(out, tuple): return tuple(wrap(x) for x in out)
    else: raise RuntimeError(f"unknown output type {type(out)}")
  return nf

for k,v in tiny_backend.items(): torch.library.impl(k.replace("aten.", "aten::"), "privateuseone")(wrap_fxn(k,v))

@torch.library.impl("aten::equal", "privateuseone")
def equal(x: torch.Tensor, y: torch.Tensor): return (x==y).all().item()

if TORCH_DEBUG:
  from torch.utils._python_dispatch import TorchDispatchMode
  class DispatchLog(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
      #print(f"Dispatch Log: {func}(*{args}, **{kwargs})")
      print(f"Dispatch Log: {func}")
      return func(*args, **(kwargs or {}))
  (_dispatch_log:=DispatchLog()).__enter__() # NOTE: must be kept alive

# NOTE: patch torch optimizer step to avoid continously growing the computation graph
import weakref
_torch_modules_with_buffers: weakref.WeakSet[torch.nn.Module] = weakref.WeakSet()
def register_torch_buffer(mod, _name, _buffer): _torch_modules_with_buffers.add(mod)
def get_real_tinygrad_buffers():
  res = set()
  for mod in _torch_modules_with_buffers:
    for _,b in mod.named_buffers(recurse=False):
      if b is not None and b.is_tiny:
        res.add(unwrap(b))
  return res
torch.nn.modules.module.register_module_buffer_registration_hook(register_torch_buffer)

from torch.nn.modules import Module
def backward_hook(model:Module, _grad_input, _grad_out):
  grads_to_realize = [unwrap(p.grad) for p in model.parameters() if p.grad is not None]
  if len(grads_to_realize): Tensor.realize(*grads_to_realize)
def module_hook(module:Module, _name, _submodule): module.register_backward_hook(backward_hook)
torch.nn.modules.module.register_module_module_registration_hook(module_hook)

def realize_optimizer_step(optimizer: torch.optim.Optimizer, *args, **kwargs):
  tinygrad_tensors = []
  for param_group in optimizer.param_groups:
    for param in param_group["params"]:
      if param is None: continue
      tinygrad_tensors.append(param.data)
  for state_dict in optimizer.state.values():
    for _, value in state_dict.items():
      if torch.is_tensor(value): tinygrad_tensors.append(value)
  real_tinygrad_tensors = [unwrap(x) for x in tinygrad_tensors if x.is_tiny]
  real_tinygrad_tensors += get_real_tinygrad_buffers()
  if len(real_tinygrad_tensors): Tensor.realize(*real_tinygrad_tensors)

_optimizer_init = torch.optim.Optimizer.__init__
def _optimizer_patched_init(self, *args, **kwargs):
  _optimizer_init(self, *args, **kwargs)
  self.register_step_post_hook(realize_optimizer_step)
torch.optim.Optimizer.__init__ = _optimizer_patched_init
