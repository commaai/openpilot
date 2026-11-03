#!/usr/bin/env python3
"""Make weight patches for patch_model.py from the fitted edits.

  make_patch.py stop_gain OUT.npz --units 69,248,201,239 --gain 2.5      readout gain on the stop circuit (long rows)
  make_patch.py cut_gate OUT.npz --circuit cut2_circuit.npz             gated corner-cut circuit in 2 dead trunk units
"""
import argparse
import json
import numpy as np

from openpilot.tools.weight_lab.lab_paths import DATA
from openpilot.tools.weight_lab.slots import DEAD_R6, DEAD_TRUNK

H = 'policy_model.temporal_hydra.'
N_T, WIDTH = 33, 15
N_PLAN = 990


def weights():
  # master weights: the model file in the working tree is already patched
  z = np.load(f'{DATA}/head_weights.npz')
  return {k: z[k] for k in z.files}


def save(out, name, desc, edits):
  arrays = {}
  for i, (_, idx, val) in enumerate(edits):
    arrays[f'idx{i}'] = np.asarray(idx, dtype=np.int64)
    arrays[f'val{i}'] = np.asarray(val, dtype=np.float32)
  meta = {'name': name, 'desc': desc, 'edits': [{'name': n} for n, _, _ in edits]}
  np.savez(out, meta=json.dumps(meta), **arrays)
  print(f'{out}: {sum(len(i) for _, i, _ in edits)} weights')


def stop_gain(args, P):
  units = [int(u) for u in args.units.split(',')]
  W = P[H + 'final_layer.plan.weight']
  rows = [t * WIDTH + c for t in range(N_T) for c in (0, 3, 6)]
  idx = np.array([r * W.shape[1] + u for r in rows for u in units])
  save(args.out, 'stop_gain', f'x{args.gain} long-plan readout of stop-control units {units}',
       [(H + 'final_layer.plan.weight', idx, W.reshape(-1)[idx] * args.gain)])


def cut_gate(args, P):
  c = np.load(args.circuit)
  rows = np.array([t * WIDTH + ch for ch in (1, 11, 14) for t in range(N_T)])
  w_in, b_in, C = c['w_in'], c['b_in'], c['C']
  units = c['units'].tolist()
  if 'cap_units' in c.files:
    # out = relu(z) - relu(z - cap): each cap unit reads the same input with the bias lowered by cap, and the negated readout
    units += c['cap_units'].tolist()
    w_in = np.concatenate([w_in, w_in])
    b_in = np.concatenate([b_in, b_in - float(c['cap'])])
    C = np.concatenate([C, -C])
  W_in = P[H + 'in_layer.plan.weight']        # (256, 512)
  W_r0 = P[H + 'res_layer.plan.0.weight']     # (256, 256), input side of the residual block
  W_r2 = P[H + 'res_layer.plan.2.weight']     # (256, 256), output side
  W_f = P[H + 'final_layer.plan.weight']      # (990, 256)
  edits = []
  # 1) unit input: h_j = relu(w . x + b)
  edits.append((H + 'in_layer.plan.weight', np.concatenate([j * 512 + np.arange(512) for j in units]), np.concatenate(list(w_in))))
  edits.append((H + 'in_layer.plan.bias', np.array(units), b_in))
  # 2) cut the unit out of the residual block, both ways
  edits.append((H + 'res_layer.plan.0.weight', np.concatenate([np.arange(256) * 256 + j for j in units]), np.zeros(256 * len(units))))
  edits.append((H + 'res_layer.plan.2.weight', np.concatenate([j * 256 + np.arange(256) for j in units]), np.zeros(256 * len(units))))
  edits.append((H + 'res_layer.plan.2.bias', np.array(units), np.zeros(len(units))))
  # 3) readout: lateral mean rows get the fitted correction, every other row reads 0 from these units
  col = np.zeros((N_PLAN, len(units)), dtype=np.float32)
  for k, j in enumerate(units):
    col[rows, k] = C[k]
  edits.append((H + 'final_layer.plan.weight', np.concatenate([np.arange(N_PLAN) * W_f.shape[1] + j for j in units]),
                np.concatenate([col[:, k] for k in range(len(units))])))
  assert all(np.all(P[H + 'final_layer.plan.weight'][:, j] == W_f[:, j]) for j in units)
  save(args.out, 'cut_gate', f'gated corner-cut circuit in dead trunk units {units}', edits)


DEAD_UNITS = DEAD_TRUNK  # never active in 129k replayed frames, kept for circuits


def lat_readout(args, P):
  """Corner-cut fix: add the fitted delta to the lateral mean rows (y, yaw, yaw rate) of the plan readout."""
  z = np.load(args.delta_file)
  D, rows = z['D'], z['rows']               # D: (257, 99) in weight units, last row is the bias
  W_f = P[H + 'final_layer.plan.weight']
  b_f = P[H + 'final_layer.plan.bias']
  cols = np.array([j for j in range(W_f.shape[1]) if j not in DEAD_UNITS])
  w_idx = (rows[:, None] * W_f.shape[1] + cols[None, :]).reshape(-1)
  w_val = (W_f[np.ix_(rows, cols)] + D[cols].T).reshape(-1)
  print(f'largest delta on a dead unit column (dropped): {np.abs(D[list(DEAD_UNITS)]).max():.2e}')
  save(args.out, 'lat_readout', f'corner-cut readout fit {args.delta_file.rsplit("/", 1)[-1]} on y/yaw/yaw-rate rows',
       [(H + 'final_layer.plan.weight', w_idx, w_val), (H + 'final_layer.plan.bias', rows, b_f[rows] + D[-1])])


def soft_brake(args, P):
  """Early soft brake: dead units copy the pre-activation of the stop units, shifted by DELTA and capped,
  and read out BETA x the long-plan direction of u201."""
  src = [int(u) for u in args.soft_src.split(',')]
  dst = [int(u) for u in args.soft_units.split(',')]  # first len(src) are main units, the rest cap units
  delta, cap, beta = args.delta, args.cap, args.beta
  W_in, b_in = P[H + 'in_layer.plan.weight'], P[H + 'in_layer.plan.bias']
  W_r2, b_r2 = P[H + 'res_layer.plan.2.weight'], P[H + 'res_layer.plan.2.bias']
  W_f = P[H + 'final_layer.plan.weight']
  long_rows = np.array([t * WIDTH + c for t in range(N_T) for c in (0, 3, 6)])
  col = np.zeros(N_PLAN, dtype=np.float32)
  col[long_rows] = beta * W_f[long_rows, 201]
  edits = {k: ([], []) for k in ('in_w', 'in_b', 'r0', 'r2_w', 'r2_b', 'f')}
  def add(k, idx, val):
    edits[k][0].append(np.asarray(idx)); edits[k][1].append(np.asarray(val, dtype=np.float32))
  for i, s in enumerate(src):
    for j, shift, sign in ((dst[i], delta, 1.0), (dst[len(src) + i], delta - cap, -1.0)):
      add('in_w', j * 512 + np.arange(512), W_in[s])            # same first layer as the source unit
      add('in_b', [j], [b_in[s]])
      add('r0', np.arange(256) * 256 + j, np.zeros(256))        # its first layer does not feed the residual block
      r2_row = W_r2[s].copy()
      r2_row[list(DEAD_R6)] = 0.0                                # dead r6 units are free for other circuits
      add('r2_w', j * 256 + np.arange(256), r2_row)             # gets the same residual input as the source unit
      add('r2_b', [j], [b_r2[s] + shift])
      add('f', np.arange(N_PLAN) * W_f.shape[1] + j, sign * col)
  names = {'in_w': 'in_layer.plan.weight', 'in_b': 'in_layer.plan.bias', 'r0': 'res_layer.plan.0.weight',
           'r2_w': 'res_layer.plan.2.weight', 'r2_b': 'res_layer.plan.2.bias', 'f': 'final_layer.plan.weight'}
  save(args.out, 'soft_brake', f'early soft brake from {src} into {dst}, delta {delta} cap {cap} beta {beta}',
       [(H + names[k], np.concatenate(i), np.concatenate(v)) for k, (i, v) in edits.items()])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('kind', choices=['stop_gain', 'cut_gate', 'soft_brake', 'lat_readout'])
  ap.add_argument('--delta-file')
  ap.add_argument('--soft-src', default='201,239')
  ap.add_argument('--soft-units', default='51,78,108,166')
  ap.add_argument('--delta', type=float, default=0.6)
  ap.add_argument('--cap', type=float, default=0.4)
  ap.add_argument('--beta', type=float, default=2.5)
  ap.add_argument('out')
  ap.add_argument('--units', default='69,248,201,239')
  ap.add_argument('--gain', type=float, default=2.5)
  ap.add_argument('--circuit')
  args = ap.parse_args()
  P = weights()
  {'stop_gain': stop_gain, 'cut_gate': cut_gate, 'soft_brake': soft_brake, 'lat_readout': lat_readout}[args.kind](args, P)


if __name__ == '__main__':
  main()
