from typing import Callable, Any
from tinygrad import Tensor, dtypes, nn, UOp
from tinygrad.uop.ops import KernelInfo, AxisType, Ops

def quantize_to_fp8(x: Tensor, dtype=dtypes.fp8e4m3):
  fp8_min = -448.0 if dtype == dtypes.fp8e4m3 else -57344.0
  fp8_max = 448.0 if dtype == dtypes.fp8e4m3 else 57344.0
  x_abs_max = x.abs().max().detach()
  scale = fp8_max / (x_abs_max + 1e-8)
  x_scaled = x * scale
  x_det = x_scaled.detach()
  x_clamped = x_det.clamp(fp8_min, fp8_max)
  x_clamped_ste = x_scaled + (x_clamped - x_det)
  res = x_clamped_ste.cast(dtype)
  return res, scale.float().reciprocal()

def custom_matmul(output: UOp, inp: UOp, weight: UOp) -> UOp:
  SEQ = inp.shape[1]
  OUT = weight.shape[0]
  IN = weight.shape[-1]
  seq_idx = UOp.range(SEQ, 2, AxisType.LOOP)
  out_idx = UOp.range(OUT, 3, AxisType.LOOP)
  batch_idx = UOp.range(output.size//SEQ//OUT, 1, AxisType.LOOP)
  reduce_idx = UOp.range(IN, 0, AxisType.REDUCE)
  product = (inp.index((seq_idx*IN+reduce_idx+batch_idx*IN*SEQ)) * weight.index((out_idx*IN+reduce_idx))).cast(dtypes.float)
  reduced = product.reduce(reduce_idx, arg=Ops.ADD)
  store_op = output.index((seq_idx*OUT+out_idx+batch_idx*OUT*SEQ), ptr=True).store(reduced).end(batch_idx, seq_idx, out_idx)
  return store_op.sink(arg=KernelInfo(name=f"fp8_matmul_{inp.shape}x{weight.shape}"))

def custom_matmul_backward(gradient: UOp, kernel: UOp) -> tuple[UOp, UOp]:
  _, input_uop, weight_uop = kernel.src
  input_tensor = Tensor(input_uop, device=input_uop.device)
  grad_tensor = Tensor(gradient, device=gradient.device)
  weight_tensor = Tensor(weight_uop, device=weight_uop.device)
  grad_quantized, scale = quantize_to_fp8(grad_tensor)
  scale_scalar = scale.reshape(())
  grad_weight = Tensor.einsum("bso,bsi->oi", grad_quantized, input_tensor, dtype=dtypes.float)
  grad_weight = grad_weight * scale_scalar
  grad_2d = grad_quantized.reshape(grad_tensor.shape[0] * grad_tensor.shape[1], grad_tensor.shape[-1])
  grad_input = (grad_2d.dot(weight_tensor, dtype=dtypes.float)).contiguous().reshape(input_tensor.shape) * scale
  return (None, grad_input.uop, grad_weight.uop)

class FP8Linear:
  def __init__(self, in_features:int, out_features:int, bias:bool=True):
    self.weight = Tensor.empty(out_features, in_features, dtype=dtypes.float32)
    self.bias = Tensor.empty(out_features, dtype=dtypes.float32) if bias else None

  def __call__(self, x: Tensor) -> Tensor:
    original_ndim = len(x.shape)
    if original_ndim == 2: x = x.reshape(x.shape[0], 1, x.shape[1])
    batch, seq, _ = x.shape
    w_fp8, w_scale = quantize_to_fp8(self.weight)
    x_fp8, x_scale = quantize_to_fp8(x)
    GPUS = self.weight.device
    if isinstance(GPUS, tuple) and len(GPUS) > 1:
      y = Tensor(Tensor.empty((batch//len(GPUS), seq, self.weight.shape[0]), dtype=dtypes.float, device=GPUS).uop.multi(0), device=GPUS)
    else:
      y = Tensor.empty((batch, seq, self.weight.shape[0]), dtype=dtypes.float)
    y = Tensor.custom_kernel(y, x_fp8, w_fp8, fxn=custom_matmul, grad_fxn=custom_matmul_backward)[0]
    y = y * w_scale * x_scale
    if self.bias is not None: y = y + self.bias
    if original_ndim == 2: y = y.reshape(batch, self.weight.shape[0])
    return y.cast(x.dtype)

def _replace_linear(layer: nn.Linear):
  fp8_linear = FP8Linear(layer.weight.shape[1], layer.weight.shape[0], layer.bias is not None)
  fp8_linear.weight = layer.weight
  if layer.bias is not None: fp8_linear.bias = layer.bias
  return fp8_linear

def _swap_linear_with_fp8(model, module_filter_fn:Callable[[Any, str],bool]|None=None, fqn:str="", parent:Any|None=None,
                          attr_name:str="", visited:set|None=None):
  if visited is None: visited = set()
  if id(model) in visited: return
  visited.add(id(model))
  if isinstance(model, (str, int, float, bool, type(None), Tensor, UOp)): return
  elif isinstance(model, nn.Linear):
    if module_filter_fn is not None and not module_filter_fn(model, fqn): return
    fp8_linear = _replace_linear(model)
    if parent is not None and attr_name:
      setattr(parent, attr_name, fp8_linear)
  elif isinstance(model, list):
    for i, item in enumerate(model):
      child_fqn = f"{fqn}.{i}" if fqn else str(i)
      if isinstance(item, nn.Linear) and (module_filter_fn is None or module_filter_fn(item, child_fqn)): model[i] = _replace_linear(item)
      else: _swap_linear_with_fp8(item, module_filter_fn, child_fqn, None, "", visited)
  elif isinstance(model, dict):
    for key, item in list(model.items()):
      child_fqn = f"{fqn}.{key}" if fqn else str(key)
      if isinstance(item, nn.Linear) and (module_filter_fn is None or module_filter_fn(item, child_fqn)): model[key] = _replace_linear(item)
      else: _swap_linear_with_fp8(item, module_filter_fn, child_fqn, None, "", visited)
  elif hasattr(model, "__dict__"):
    for attr_key in list(vars(model).keys()):
      try: attr = getattr(model, attr_key)
      except Exception: continue
      child_fqn = f"{fqn}.{attr_key}" if fqn else attr_key
      _swap_linear_with_fp8(attr, module_filter_fn, child_fqn, model, attr_key, visited)

def convert_to_float8_training(model, module_filter_fn:Callable[[Any,str],bool]|None=None):
  _swap_linear_with_fp8(model, module_filter_fn, "", None, "")
  return model
