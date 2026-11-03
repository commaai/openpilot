#!/usr/bin/env python3
"""Does the plan trunk encode the corner-cut offset separately from curvature?

Target: signed offset of the planned path from the model's own lane center at 20 m, on frames with
confident lanes and v > 8 m/s. Ridge probes on the 256-d trunk, scored leave-one-route-out.
"""
import glob
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cut import inside_offset  # noqa: E402

from openpilot.tools.weight_lab.lab_paths import CODE, DATA as HERE  # noqa: F401


def load():
  rows = {}
  for p in sorted(glob.glob(f'{HERE}/seg/*.npz')):
    name = os.path.basename(p).split('_')[0]
    if name in ('g2', 'fdp'):  # edited / duplicate replays
      continue
    d = np.load(p)
    ins, raw, prob, width = inside_offset(d)
    m = (prob > 0.5) & (d['v_ego'] > 8) & (np.abs(raw[:, 1]) < 2.5)
    if m.sum() < 50:
      continue
    route = str(d['seg'][0]).rsplit('/', 1)[0]
    r = rows.setdefault(route, {'h': [], 'off': [], 'curv': [], 'la': []})
    r['h'].append(d['x_plan_trunk'][m]); r['off'].append(raw[m, 1]); r['curv'].append(d['curv'][m]); r['la'].append((d['curv'] * d['v_ego'] ** 2)[m])
  return {k: {kk: np.concatenate(v) for kk, v in r.items()} for k, r in rows.items()}


def ridge(X, y, lam=1.0):
  m, sd = X.mean(0), X.std(0) + 1e-6
  Z = (X - m) / sd
  w = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
  return lambda A: ((A - m) / sd) @ w + y.mean(), w / sd


def main():
  R = load()
  print({k.split('/')[-1]: len(v['off']) for k, v in R.items()})
  for test in R:
    tr = [k for k in R if k != test]
    X = np.concatenate([R[k]['h'] for k in tr]); y = np.concatenate([R[k]['off'] for k in tr])
    f, _ = ridge(X, y, lam=0.01)
    t = R[test]
    pred = f(t['h'])
    r2 = 1 - np.mean((pred - t['off']) ** 2) / np.var(t['off'])
    curve = np.abs(t['la']) > 0.7
    print(f"held out {test.split('/')[-1]:>22}: R² {r2:+.2f}, corr {np.corrcoef(pred, t['off'])[0, 1]:+.2f} | in curves (|lat acc|>0.7, n={curve.sum()}): "
          f"corr {np.corrcoef(pred[curve], t['off'][curve])[0, 1] if curve.sum() > 20 else float('nan'):+.2f}, true inside median {np.median(np.sign(t['curv'][curve]) * t['off'][curve]) if curve.sum() else float('nan'):+.2f}")
  # is the offset direction separate from the curvature direction?
  X = np.concatenate([R[k]['h'] for k in R]); y = np.concatenate([R[k]['off'] for k in R]); c = np.concatenate([R[k]['curv'] for k in R])
  _, w_off = ridge(X, y, 0.01)
  _, w_curv = ridge(X, c, 0.01)
  print(f'cos(offset direction, curvature direction) in trunk: {w_off @ w_curv / np.linalg.norm(w_off) / np.linalg.norm(w_curv):+.2f}')


if __name__ == '__main__':
  main()
