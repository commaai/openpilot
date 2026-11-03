#!/usr/bin/env python3
"""Make an edited copy of an ONNX model: scale chosen plan-trunk units' readout into the long plan.

Only the mean rows of the long channels (x, vx, ax over all 33 times) are scaled, so the lateral plan
and the plan stds keep their weights.

usage: edit.py SRC DST --units 69,248,201,239 --gain 2.0
"""
import argparse
import numpy as np
import onnx
from onnx import numpy_helper

W_NAME = 'policy_model.temporal_hydra.final_layer.plan.weight'
LONG_CHANNELS = (0, 3, 6)  # x, vx, ax in the 15-wide plan row
N_T, WIDTH = 33, 15


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('src')
  ap.add_argument('dst')
  ap.add_argument('--units', required=True)
  ap.add_argument('--gain', type=float, required=True)
  args = ap.parse_args()
  units = [int(u) for u in args.units.split(',')]

  m = onnx.load(args.src)
  init = next(t for t in m.graph.initializer if t.name.endswith(W_NAME))
  W = numpy_helper.to_array(init).copy()  # (990, 256): first 495 rows are means, (33, 15) row-major
  rows = [t * WIDTH + c for t in range(N_T) for c in LONG_CHANNELS]
  before = W[np.ix_(rows, units)].copy()
  W[np.ix_(rows, units)] *= args.gain
  init.CopyFrom(numpy_helper.from_array(W.astype(numpy_helper.to_array(init).dtype), init.name))
  onnx.save(m, args.dst)
  changed = len(rows) * len(units)
  print(f'scaled {changed} weights ({100 * changed / W.size:.3f}% of the layer, {changed / 30e6 * 100:.5f}% of the model) by {args.gain}; '
        f'mean |w| {np.abs(before).mean():.3f} -> {np.abs(W[np.ix_(rows, units)]).mean():.3f}')


if __name__ == '__main__':
  main()
