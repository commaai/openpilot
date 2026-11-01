import json, pathlib
from tinygrad import Tensor, dtypes, nn, Device

def e4m3fn_to_fnuz(weight:Tensor, scale:Tensor) -> tuple[Tensor, Tensor]:
  bits = weight.bitcast(dtypes.uint8)
  return bits.eq(0x80).where(0, bits).contiguous().bitcast(dtypes.fp8e4m3fnuz), scale * 2

def hf_load(path:pathlib.Path, devices:tuple[str, ...], names:dict[str, str|list[str]], shard_map:dict[str, int]) -> dict[str, Tensor]:
  index = path / "model.safetensors.index.json"
  weights:dict[str, Tensor] = {}
  if index.exists(): weight_map = json.loads(index.read_text())["weight_map"]
  else:
    weights = nn.state.safe_load(path / "model.safetensors")
    weight_map = dict.fromkeys(weights, "model.safetensors")

  def pop_tensor(source:str|list[str], axis:int|None) -> Tensor:
    sources = source if isinstance(source, list) else [source]
    stack = isinstance(source, list)
    parts = []
    for s in sources:
      if s not in weights: weights.update(nn.state.safe_load(path / weight_map[s]))
      parts.append(weights.pop(s))
    dtype, shape = parts[0].dtype, parts[0].shape
    if stack: shape = (len(parts), *shape)
    assert axis is None or shape[axis] % len(devices) == 0, f"cannot split {sources[0]} on axis {axis}"
    if stack: w = Tensor.stack(*[t.bitcast(dtypes.uint8).to(devices[0]) for t in parts]).contiguous().realize()
    else: w = parts[0].bitcast(dtypes.uint8)
    return w.shard(devices, axis).contiguous().realize().bitcast(dtype)

  use_fnuz = dtypes.fp8e4m3fnuz in Device[devices[0]].renderer.supported_dtypes()
  state_dict:dict[str, Tensor] = {}
  for name, source in names.items():
    first = source[0] if isinstance(source, list) else source
    if first not in weight_map: continue
    axis = shard_map.get(name)
    w = pop_tensor(source, axis)
    scale_source = [s + "_scale_inv" for s in source] if isinstance(source, list) else source + "_scale_inv"
    scale = pop_tensor(scale_source, axis) if first + "_scale_inv" in weight_map else None
    if scale is not None and use_fnuz:
      w, scale = e4m3fn_to_fnuz(w, scale)
      Tensor.realize(w, scale)
    state_dict[name] = w
    if scale is not None: state_dict[name + "_scale_inv"] = scale
  return state_dict
