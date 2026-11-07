import json, pathlib
from tinygrad import Tensor, dtypes, nn, Device
from tinygrad.helpers import prod

def e4m3fn_to_fnuz(weight:Tensor, scale:Tensor) -> tuple[Tensor, Tensor]:
  bits = weight.bitcast(dtypes.uint8)
  return bits.eq(0x80).where(0, bits).contiguous().bitcast(dtypes.fp8e4m3fnuz), scale * 2

def hf_load(path:pathlib.Path, devices:tuple[str, ...], names:dict[str, str|list[str]], shard_map:dict[str, int]) -> dict[str, Tensor]:
  index = path / "model.safetensors.index.json"
  if index.exists(): weight_map = json.loads(index.read_text())["weight_map"]
  else:
    _, _, metadata = nn.state.safe_load_metadata(path / "model.safetensors")
    weight_map = dict.fromkeys(metadata, "model.safetensors")
  needed = {s for source in names.values() for s in (source if isinstance(source, list) else [source])}
  needed |= {s + "_scale_inv" for s in needed}
  weights:dict[str, tuple[Tensor, dict]] = {}

  def pop_tensor(source:str|list[str], axis:int|None) -> Tensor:
    sources = source if isinstance(source, list) else [source]
    stack = isinstance(source, list)
    parts = []
    for s in sources:
      if s not in weights:
        data, data_start, metadata = nn.state.safe_load_metadata(path / weight_map[s])
        data = data[data_start:].to(devices[0]).realize()
        weights.update({k: (data, m) for k, m in metadata.items() if k in needed})
      parts.append(weights.pop(s))

    m0 = parts[0][1]
    shape, dtype = m0['shape'], nn.state.safe_dtypes[m0['dtype']]
    nbytes = prod(shape) * dtype.itemsize
    assert all(m['shape'] == shape and m['dtype'] == m0['dtype'] and
               0 <= m['data_offsets'][0] <= m['data_offsets'][1] <= d.numel() and
               m['data_offsets'][1] - m['data_offsets'][0] == nbytes for d, m in parts), f"mismatched parts for {sources[0]}"
    slices = [(d, m['data_offsets'][0]) for d, m in parts]
    byte_shape = (*shape[:-1], shape[-1] * dtype.itemsize)
    if stack: shape, byte_shape = (len(parts), *shape), (len(parts), *byte_shape)
    assert axis is None or shape[axis] % len(devices) == 0, f"cannot split {sources[0]} on axis {axis}"

    if stack:
      if nbytes % 4 == 0 and all(d.numel() % 4 == 0 and o % 4 == 0 for d, o in slices):
        words = Tensor.arange(nbytes // 4, dtype=dtypes.int64).to(devices[0])
        chunks = []
        for data in {id(d): d for d, _ in slices}.values():
          offsets = Tensor([-1 if d is not data else o // 4 for d, o in slices], device=devices[0], dtype=dtypes.int64)
          gathered = data.bitcast(dtypes.uint32)[offsets.clamp(min_=0)[:, None] + words[None, :]]
          chunks.append((offsets >= 0)[:, None].where(gathered, 0))
        w = Tensor.usum(*chunks).bitcast(dtypes.uint8)
      else:
        w = Tensor.stack(*[d[o:o+nbytes] for d, o in slices])
      w = w.reshape(byte_shape).clone().realize().shard(devices, axis).contiguous()
    else:
      data, o = slices[0]
      w = data[o:o+nbytes].reshape(byte_shape).shard(devices, axis).clone()
    return w.realize().bitcast(dtype)

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
