"""Compile ONNX with optional input warps, temporal sampling and packed host transfers."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from tinygrad import Device, dtypes
from tinygrad.dtype import DType, _to_np_dtype
from tinygrad.helpers import fetch
from openpilot.selfdrive.modeld.helpers import allocate_inputs, compile_jit, dump_pickle
from openpilot.selfdrive.modeld.compile_warp import NV12Frame, make_warp
from tinygrad.nn.onnx import OnnxPBParser, OnnxRunner


def onnx_metadata(path):
  parser = OnnxPBParser(path, load_external_data=False)
  metadata, output_shapes = {}, {}
  for field, wire_type in parser._parse_message(parser.reader.len):
    if field == 14:
      entry = parser._parse_StringStringEntryProto()
      metadata[entry['key']] = entry['value']
    elif field == 7:
      # Read output declarations without parsing graph nodes or weight tensors.
      for field, wire_type in parser._parse_message(parser._decode_end_pos()):
        if field == 12:
          value = parser._parse_ValueInfoProto()
          output_shapes[value['name']] = value['parsed_type'].shape if value['parsed_type'] is not None else ()
        else:
          parser.reader.skip_field(wire_type)
    else:
      parser.reader.skip_field(wire_type)
  return metadata, output_shapes


def sample_history(buffer, value, shape, *, axis, size=1, stride=1, reduce='sample', delay=0):
  """Append a frame, then sample or pool history ending delay frames before it."""
  buffer.assign(buffer[1:].cat(value.unsqueeze(0), dim=0).contiguous())
  if delay:
    buffer = buffer[:-delay]
  samples = shape[axis] // size
  sampled = buffer.reshape(samples, stride, *value.shape).max(1) if reduce == 'max' else buffer[:samples*stride:stride]
  return sampled.permute(*range(1, axis+1), 0, *range(axis+1, sampled.ndim)).contiguous().reshape(shape)


def prepare_inputs(runner, config, device_inputs, float32):
  sources:dict[str, tuple[tuple[int, ...], DType, str]] = {}
  histories, warps = {}, {}
  shapes = {name: tuple(s if isinstance(s, int) else 1 for s in spec.shape) for name, spec in runner.graph_inputs.items()}
  for name, spec in runner.graph_inputs.items():
    cfg = config.get('inputs', {}).get(name, {})
    shape = shapes[name]
    dtype = dtypes.float32 if float32 and spec.dtype == dtypes.float16 else spec.dtype
    if history := cfg.get('history'):
      axis, size, stride = history['axis'], history.get('size', 1), history.get('stride', 1)
      samples = shape[axis] // size
      shape = shape[:axis] + (size,) + shape[axis+1:]
      length = (samples*stride if history.get('reduce') == 'max' else (samples-1)*stride + 1) + history.get('delay', 0)
      histories[name] = ((length, *shape), dtypes.uint8 if 'warp' in cfg else dtype)
    if warp := cfg.get('warp'):
      frame = NV12Frame(*warp['frame'])
      warps[name] = make_warp(frame, warp['output_size'], warp['layout'], warp.get('border_fill'))
      shape, dtype = (frame.copy_size,), dtypes.uint8
      sources[warp['transform']] = ((3, 3), dtypes.float32, 'NPY')
    sources[cfg.get('source', name)] = (shape, dtype, Device.DEFAULT if name in device_inputs else 'NPY')
  return sources, histories, warps, shapes


def compile_onnx(path, *, device_inputs=(), float32=False, output_name=None, benchmark_runs=20, out_of_band=False, configs=None):
  runner = OnnxRunner(path)
  properties, output_shapes = onnx_metadata(path)
  metadata = {'metadata': properties} | {
    f'{kind}_shapes': {name: tuple(d if isinstance(d, int) else 0 for d in shape) for name, shape in shapes.items()}
    for kind, shapes in [('input', {name: spec.shape for name, spec in runner.graph_inputs.items()}), ('output', output_shapes)]}
  if unknown := set(device_inputs) - runner.graph_inputs.keys():
    raise ValueError(f"Unknown inputs: {unknown}")
  if output_name is not None and output_name not in runner.graph_outputs:
    raise ValueError(f"Unknown output: {output_name}")

  def compile_config(config):
    sources, histories, warps, shapes = prepare_inputs(runner, config, device_inputs, float32)
    packed_specs, offset = {}, 0
    for name in config.get('pack', []):
      shape, dtype, _ = sources[name]
      offset = (offset + dtype.itemsize - 1) // dtype.itemsize * dtype.itemsize
      packed_specs[name] = (offset, shape, np.dtype(_to_np_dtype(dtype)).str)
      offset += math.prod(shape)*dtype.itemsize

    specs = {name: (shape, np.dtype(_to_np_dtype(dtype)).str, device)
             for name, (shape, dtype, device) in sources.items() if name not in packed_specs}
    if packed_specs:
      specs['packed_inputs'] = ((offset,), np.dtype(np.uint8).str, 'NPY')
    specs.update({name+'_history': (shape, np.dtype(_to_np_dtype(dtype)).str, Device.DEFAULT) for name, (shape, dtype) in histories.items()})

    def make_inputs(seed):
      rng = np.random.default_rng(seed)
      def initialize(views):
        for name in list(packed_specs) + [k for k in sources if k not in packed_specs]:
          shape, dtype, _ = sources[name]
          views[name][...] = (rng.standard_normal(shape) if dtypes.is_float(dtype) else
                             rng.integers(0, 256, shape, dtype=np.uint8) if dtype == dtypes.uint8 else
                             rng.integers(0, 2 if dtype == dtypes.bool else 16, shape))
      return (), allocate_inputs(specs, packed_specs, initialize)[0]

    def run(**inputs):
      values = {name: inputs[name].to(Device.DEFAULT) for name in sources if name not in packed_specs}
      if packed_specs:
        packed = inputs['packed_inputs'].to(Device.DEFAULT).realize()
        for name, (start, shape, _) in packed_specs.items():
          dtype = sources[name][1]
          values[name] = packed[start:start+math.prod(shape)*dtype.itemsize].bitcast(dtype).reshape(shape)
      model_inputs = {}
      for name, spec in runner.graph_inputs.items():
        cfg = config.get('inputs', {}).get(name, {})
        value = values[cfg.get('source', name)]
        if name in warps:
          value = warps[name](value, values[cfg['warp']['transform']]).realize()
        if name in histories:
          value = sample_history(inputs[name+'_history'], value.reshape(histories[name][0][1:]), shapes[name], **cfg['history'])
        model_inputs[name] = value.cast(spec.dtype)
      outputs = runner(model_inputs)
      if float32:
        outputs = {k: v.cast(dtypes.float32) for k, v in outputs.items()}
      if output_name is not None:
        return outputs[output_name]
      return next(iter(outputs.values())) if len(outputs) == 1 else outputs

    jit = compile_jit(run, make_inputs, benchmark_runs, out_of_band=out_of_band)
    return {'run': jit, 'input_specs': specs, 'packed_specs': packed_specs}

  return {'metadata': metadata, 'variants': {name: compile_config(config) for name, config in (configs or {'default': {}}).items()}}


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('onnx')
  parser.add_argument('output')
  parser.add_argument('--device-input', action='append', default=[], help='input placed on DEV instead of host NPY (repeatable)')
  parser.add_argument('--float32', action='store_true', help='expose float16 inputs and model outputs as float32')
  parser.add_argument('--output-name', help='select one model output')
  parser.add_argument('--benchmark-runs', type=int, default=20)
  parser.add_argument('--out-of-band', action='store_true', help='stream protocol-5 buffers for large models')
  parser.add_argument('--config', action='append', help='NAME=JSON: per-input source, warp and history settings, plus host input packing order')
  args = parser.parse_args()
  path = fetch(args.onnx) if '://' in args.onnx else Path(args.onnx)
  configs = {name: json.loads(config) for name, config in (value.split('=', 1) for value in args.config)} if args.config else None
  artifact = compile_onnx(path, device_inputs=args.device_input, float32=args.float32, output_name=args.output_name,
                          benchmark_runs=args.benchmark_runs, out_of_band=args.out_of_band, configs=configs)
  with open(args.output, 'wb') as f:
    dump_pickle(artifact, f, out_of_band=args.out_of_band)
