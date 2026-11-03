#!/usr/bin/env python3
"""Try edits of the stop-control circuit and score them on every controlled stop and on open road, no replay.

The 4 circuit units only feed the final layer, so changing their bias or their readout changes the plan by
  delta = sum_j (h'_j * g_j - h_j) * W[:, j] * scale     with h'_j = relu(pre_j + bias_j)
which is exact given the saved pre-ReLU trunk activations.

Edits (name = BIAS69,BIAS248,BIAS201,BIAS239/GAIN69,GAIN248,GAIN201,GAIN239), applied to the long plan rows.
Writes seg/<name>_<seg>.npz with the edited plan and accel for the viewer.

usage: lights_edit.py NAME=b69,b248,b201,b239/g69,g248,g201,g239 [...]
"""
import glob
import os
import sys
import numpy as np

from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, smooth_value
from openpilot.tools.weight_lab.lab_paths import DATA
from openpilot.tools.weight_lab.lights import events, SKIP, CIRCUIT

H = 'policy_model.temporal_hydra.'
T = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
LONG_T = 0.525
ROWS = np.arange(495)  # all mean rows: a unit's bias change moves everything it reads into


def accels(plans):
  out, prev = [], 0.0
  for p in plans:
    prev = smooth_value(get_accel_from_plan(p[:, 3], p[:, 6], T, action_t=LONG_T), prev, 0.3)
    out.append(prev)
  return np.array(out)


BASE_PREFIXES = ('fe', 'fd', 'f7', 'id4', 'n76', 'n77', 'b2d', 'be', 's39', 's606', 's608')
SOFT_SRC = (2, 3)  # u201, u239 (positions in CIRCUIT): the soft brake copies their pre-activation


def need_at_30(v, dist, a):
  """Drive the desired accel from 120 m out: decel needed to stop in the last 30 m (open loop, kinematic)."""
  idx = np.flatnonzero(np.isfinite(dist) & (dist <= 120) & (dist > 30))
  if len(idx) < 5:
    return np.nan
  vs = v[idx[0]]
  for i in idx:
    vs = max(vs + min(a[i], 0.0) * 0.05, 0.0)  # only the braking part, the video sets the rest
  return vs ** 2 / 60.0


def parse(spec):
  """NAME=b69,b248,b201,b239/g69,g248,g201,g239[/DELTA,CAP,BETA] (last part: soft early brake)."""
  name, vals = spec.split('=')
  parts = vals.split('/')
  soft = np.array([float(x) for x in parts[2].split(',')]) if len(parts) > 2 else None
  return name, np.array([float(x) for x in parts[0].split(',')]), np.array([float(x) for x in parts[1].split(',')]), soft


def edit(d, bias, gain, Wf, soft=None, w_soft=None):
  pre = d['x_plan_trunk_pre'][:, CIRCUIT]
  h0 = d['x_plan_trunk'][:, CIRCUIT]
  h1 = np.maximum(pre + bias, 0) * gain
  dmu = (h1 - h0) @ Wf.T
  c = h1.sum(1)
  if soft is not None:
    # two dead units copy the pre-activation of u201 and u239, shifted by DELTA and saturating at CAP,
    # and read out BETA x the long-plan direction of u201
    delta, cap, beta = soft
    z = pre[:, SOFT_SRC] + delta
    s = (np.maximum(z, 0) - np.maximum(z - cap, 0)).sum(1)
    dmu = dmu + beta * s[:, None] * w_soft[None, :]
    c = c + s
  return d['plan'] + dmu.reshape(-1, 33, 15), c


def main():
  P = np.load(f'{DATA}/head_weights.npz')
  Wf = (P[H + 'final_layer.plan.weight'] * P[H + 'scale_layer.plan.scale'][:, None])[:495][:, CIRCUIT]  # (495, 4)
  long_mask = np.zeros(495, bool)
  long_mask[[t * 15 + c for t in range(33) for c in (0, 3, 6)]] = True
  w_soft = np.where(long_mask, Wf[:, SOFT_SRC[0]], 0.0)
  specs = [parse(s) for s in sys.argv[1:]]
  files = [p for p in sorted(glob.glob(f'{DATA}/seg/*.npz')) if os.path.basename(p).split('_')[0] in BASE_PREFIXES]
  res = {n: {'stops': [], 'open': [], 'green': [], 'stop_line': []} for n, _, _, _ in specs}
  for p in files:
    try:
      d = np.load(p)
      if 'x_plan_trunk_pre' not in d.files:
        continue
    except Exception:
      continue
    base_a = accels(d['plan'])
    v = d['v_ego']
    evs = events(d)
    pos = np.concatenate([[0], np.cumsum(v * 0.05)])[:-1]
    dist = np.full(len(v), np.inf)
    for e in evs:
      s = int(np.flatnonzero(d['frame'] == e['stop'])[0])
      dist = np.where(np.arange(len(v)) <= s, np.minimum(dist, pos[s] - pos), dist)
    h = d['x_plan_trunk'][:, CIRCUIT]
    green = (h[:, 3] > 0.2) & (h[:, :3].sum(1) < 0.05) & ~np.isfinite(dist)
    open_road = (dist > 250) & (v > 8)
    for name, bias, gain, soft in specs:
      W = np.where(long_mask[:, None], Wf, 0.0) if np.all(bias == 0) else Wf  # pure readout gains only touch long rows
      plan, c1 = edit(d, bias, gain, W, soft, w_soft)
      a1 = accels(plan)
      r = res[name]
      for e in evs:
        i1 = int(np.flatnonzero(d['frame'] == e['stop'])[0])
        i0 = i1
        while i0 > 0 and (c1[i0 - 1] > 0.3 or i1 - i0 < 5):
          i0 -= 1
        win = slice(max(0, i1 - 300), i1)
        r['stops'].append((e['dist'], float(np.sum(v[i0:i1]) * 0.05), base_a[win].min(), a1[win].min(), e['v_on'],
                           need_at_30(v, dist, base_a), need_at_30(v, dist, a1)))
      r['open'].append(base_a[open_road] - a1[open_road])
      r['green'].append(base_a[green] - a1[green])
      np.savez_compressed(f'{DATA}/seg/{name}_{os.path.basename(p)}', frame=d['frame'], seg=d['seg'], plan=plan.astype(np.float32),
                          accel=a1.astype(np.float32), circuit=c1.astype(np.float32))
  for name, bias, gain, soft in specs:
    r = res[name]
    s = np.array(r['stops'])
    fast = s[:, 4] > 11
    op, gr = np.concatenate(r['open']), np.concatenate(r['green'])
    print(f'{name}: bias {bias.tolist()} gain {gain.tolist()} soft {None if soft is None else soft.tolist()} | {len(s)} stops')
    print(f'  onset distance median {np.median(s[:, 0]):.0f} -> {np.median(s[:, 1]):.0f} m | stops from > 11 m/s (n={fast.sum()}): '
          f'hardest decel median {np.median(s[fast, 2]):+.2f} -> {np.median(s[fast, 3]):+.2f} | '
          f'kinematic: decel needed in the last 30 m {np.nanmedian(s[fast, 5]):.2f} -> {np.nanmedian(s[fast, 6]):.2f} m/s²')
    print(f'  open road: extra braking > 0.3 m/s² on {100 * np.mean(op > 0.3):.2f}% of frames (max {op.max():.2f}) | '
          f'green passes: extra braking > 0.3 on {100 * np.mean(gr > 0.3):.1f}% (max {gr.max() if len(gr) else 0:.2f})')


if __name__ == '__main__':
  main()
