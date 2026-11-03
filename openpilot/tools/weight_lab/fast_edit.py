#!/usr/bin/env python3
"""Score a readout edit (gain on chosen trunk units' long-plan weights) from saved activations, no replay.

The plan head output does not feed back into the model state, so edited plan = plan + delta exactly.

usage: fast_edit.py GAIN [UNITS]
"""
import sys
import numpy as np

from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, smooth_value
from openpilot.selfdrive.modeld.constants import ModelConstants

from openpilot.tools.weight_lab.lab_paths import CODE, DATA as HERE  # noqa: F401
H = 'policy_model.temporal_hydra.'
T = np.array(ModelConstants.T_IDXS)
LONG_CHANNELS = (0, 3, 6)
LONG_T = 0.15 + 0.3 + 0.075  # Tesla long delay + smoothing + frame/action delay, as in replay.py


def edited_plans(d, units, gain):
  P = np.load(f'{HERE}/head_weights.npz')
  W, s = P[H + 'final_layer.plan.weight'], P[H + 'scale_layer.plan.scale']
  rows = [t * 15 + c for t in range(33) for c in LONG_CHANNELS]
  h = d['x_plan_trunk'][:, units]
  delta = np.zeros((len(h), 495), dtype=np.float32)
  delta[:, rows] = (gain - 1) * (h @ (W[np.ix_(rows, units)] * s[rows][:, None]).T)
  return d['plan'] + delta.reshape(-1, 33, 15)


def accels(plans):
  out, prev = [], 0.0
  for p in plans:
    prev = smooth_value(get_accel_from_plan(p[:, 3], p[:, 6], T, action_t=LONG_T), prev, 0.3)
    out.append(prev)
  return np.array(out)


def window(d, a, b):
  return np.flatnonzero((d['frame'] >= a) & (d['frame'] <= b))


def main():
  gain = float(sys.argv[1])
  units = [int(u) for u in sys.argv[2].split(',')] if len(sys.argv) > 2 else [69, 248, 201, 239]
  segs = {k: np.load(f'{HERE}/seg/fd_{k}.npz') for k in (0, 1, 2)}
  res = {}
  for k, d in segs.items():
    p0, p1 = d['plan'], edited_plans(d, units, gain)
    res[k] = (p0, p1, accels(p0), accels(p1))
  d = segs[0]
  w = window(d, 860, 1020)
  p0, p1, a0, a1 = res[0]
  print(f'gain {gain} units {units}')
  print(f"  STOP line (s0 f860-1020): plan min speed {p0[w, :, 3].min(1).min():.2f} -> {p1[w, :, 3].min(1).min():.2f} m/s, "
        f"min accel {a0[w].min():+.2f} -> {a1[w].min():+.2f}")
  d = segs[2]
  p0, p1, a0, a1 = res[2]
  w = window(d, 400, 510)
  print(f"  red light stop (s2 f400-510): min accel {a0[w].min():+.2f} -> {a1[w].min():+.2f}, "
        f"plan v@2s at onset f420 {np.interp(2, T, p0[w[1], :, 3]):.1f} -> {np.interp(2, T, p1[w[1], :, 3]):.1f}")
  w = window(d, 790, 830)
  print(f"  green launch (s2 f790-830): mean accel {a0[w].mean():+.2f} -> {a1[w].mean():+.2f}")
  d = segs[1]
  p0, p1, a0, a1 = res[1]
  w = window(d, 375, 425)
  print(f"  green pass (s1 f375-425): min accel {a0[w].min():+.2f} -> {a1[w].min():+.2f}")
  # everywhere else: how much does the edit move accel when the circuit is quiet
  for k, d in segs.items():
    quiet = d['x_plan_trunk'][:, units].sum(1) < 0.3
    da = np.abs(res[k][3] - res[k][2])
    print(f"  s{k}: |d accel| quiet frames max {da[quiet].max():.3f}, firing frames max {da[~quiet].max() if (~quiet).any() else 0:.2f}")


if __name__ == '__main__':
  main()
