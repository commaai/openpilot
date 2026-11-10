import torch


def _autocast_input(module, args):
  x, = args
  if torch.is_autocast_enabled(x.device.type):
    x = x.to(torch.get_autocast_dtype(x.device.type))
  return (x,)


def quantize_fp8(model):
  from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Float8Tensor, PerTensor, quantize_

  linears = {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
  selected = {name: module for name, module in linears.items() if all(size % 16 == 0 for size in module.weight.shape)}
  quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
            filter_fn=lambda module, name: name in selected)
  assert selected and all(isinstance(module.weight, Float8Tensor) for module in selected.values())
  assert all(module.weight.qdata.dtype == torch.float8_e4m3fn for module in selected.values())
  # TorchAO's linear override otherwise bypasses BF16 autocast for FP32 activations.
  for module in selected.values():
    module.register_forward_pre_hook(_autocast_input)
  return {'implementation': 'torchao W8A8', 'weight_dtype': 'float8_e4m3fn', 'activation_dtype': 'float8_e4m3fn',
          'scaling': 'dynamic per-tensor activations, per-tensor weights', 'attention_and_kv_dtype': 'bfloat16',
          'linear_output_dtype': 'bfloat16 under autocast',
          'linear_layers': len(selected), 'weight_parameters': sum(module.weight.numel() for module in selected.values()),
          'excluded_linears': sorted(set(linears) - set(selected)), 'modules': sorted(selected)}
