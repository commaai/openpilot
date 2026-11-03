#!/usr/bin/env python3
"""Apply weight patches to the master small driving model, one by one.

The base model must have the known master sha256. Each patch is a .npz made by make_patch.py: a list of
(initializer, flat indices, new values). Patches may not touch the same weights (checked), so they can be
applied in any order and stacked.

usage: patch_model.py --out OUT.onnx PATCH.npz [PATCH.npz ...] [--base MODEL.onnx]
"""
import argparse
import base64
import hashlib
import json
import pickle
import numpy as np
import onnx
from onnx import numpy_helper

MASTER_SHA256 = '65a08adc31d5c456219687d99b7bf5e44d61dae2d49ea67850e76105c7248cce'  # driving_supercombo.onnx, master 64f9b47b6e
DEFAULT_BASE = '/home/batman/openpilot/openpilot/selfdrive/modeld/models/driving_supercombo.onnx'
TRUNK_TENSOR = 'model/p_relu_7'        # 256-d plan trunk (input of final_layer.plan)
TRAFFIC_UNITS = [69, 248, 201, 239]    # stop-control units: hold, hold, braking to red, intersection ahead


def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
      h.update(chunk)
  return h.hexdigest()


def add_traffic_output(m):
  """Append the 4 stop-control trunk units to the 'outputs' vector, and add their slice to the metadata."""
  g = m.graph
  out = next(o for o in g.output if o.name == 'outputs')
  n_out = out.type.tensor_type.shape.dim[1].dim_value
  prod = next(n for n in g.node if 'outputs' in n.output)
  prod.output[list(prod.output).index('outputs')] = 'outputs_base'
  # nodes that read 'outputs' must read the original vector; tinygrad runs nodes in file order, not sorted
  for n in g.node:
    for i, name in enumerate(n.input):
      if name == 'outputs':
        n.input[i] = 'outputs_base'
  idx = numpy_helper.from_array(np.array(TRAFFIC_UNITS, dtype=np.int64), 'traffic_unit_idx')
  g.initializer.append(idx)
  g.node.extend([
    onnx.helper.make_node('Gather', [TRUNK_TENSOR, 'traffic_unit_idx'], ['traffic_units_raw'], axis=1, name='traffic_gather'),
    onnx.helper.make_node('Cast', ['traffic_units_raw'], ['traffic_units_out'], to=out.type.tensor_type.elem_type, name='traffic_cast'),
    onnx.helper.make_node('Concat', ['outputs_base', 'traffic_units_out'], ['outputs'], axis=1, name='traffic_concat'),
  ])
  out.type.tensor_type.shape.dim[1].dim_value = n_out + len(TRAFFIC_UNITS)
  meta = next(p for p in m.metadata_props if p.key == 'output_slices')
  slices = pickle.loads(base64.b64decode(meta.value))
  pad = slices['pad']
  slices['pad'] = slice(n_out + pad.start, n_out + (pad.stop or 0)) if pad.start < 0 else pad
  slices['traffic_units'] = slice(n_out, n_out + len(TRAFFIC_UNITS))
  meta.value = base64.b64encode(pickle.dumps(slices)).decode()
  print(f'added traffic_units output {slices["traffic_units"]} (units {TRAFFIC_UNITS}), pad now {slices["pad"]}')


def load_patch(path):
  z = np.load(path)
  meta = json.loads(str(z['meta']))
  return meta, [(e['name'], z[f'idx{i}'], z[f'val{i}']) for i, e in enumerate(meta['edits'])]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('patches', nargs='+')
  ap.add_argument('--base', default=DEFAULT_BASE)
  ap.add_argument('--out', required=True)
  ap.add_argument('--traffic-output', action='store_true', help='append the stop-control units to the outputs as traffic_units')
  args = ap.parse_args()

  digest = sha256(args.base)
  if digest != MASTER_SHA256:
    raise SystemExit(f'base model sha256 {digest} is not the known master model {MASTER_SHA256}')
  m = onnx.load(args.base)
  inits = {t.name: t for t in m.graph.initializer}
  by_suffix = lambda s: next(n for n in inits if n.endswith(s))

  touched = {}
  total = 0
  for p in args.patches:
    meta, edits = load_patch(p)
    n_patch = 0
    for name, idx, val in edits:
      full = by_suffix(name)
      prev = touched.setdefault(full, {})
      clash = set(idx.tolist()) & set(prev)
      if clash:
        raise SystemExit(f'{p} changes {len(clash)} weights of {name} that {prev[next(iter(clash))]} already changed')
      prev.update(dict.fromkeys(idx.tolist(), p))
      t = inits[full]
      arr = numpy_helper.to_array(t).copy()
      flat = arr.reshape(-1)
      flat[idx] = val.astype(arr.dtype)
      t.CopyFrom(numpy_helper.from_array(arr, full))
      n_patch += len(idx)
    total += n_patch
    print(f"applied {meta['name']}: {n_patch} weights ({meta['desc']})")
  if args.traffic_output:
    add_traffic_output(m)
  onnx.save(m, args.out)
  print(f'{total} weights changed ({100 * total / 30_003_340:.4f}% of the model) -> {args.out}')


if __name__ == '__main__':
  main()
