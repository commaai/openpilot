#!/usr/bin/env python3
"""Can a probe on the 512-d temporal feature see a controlled stop 50-120 m out, without false alarms?

Stops come from lights.events (stop circuit active, stop from > 4 m/s). Each frame gets its distance to the next
such stop. Ridge probe: target 1 for frames 30-120 m before a stop, 0 for frames > 250 m from any stop (v > 5).
Leave-one-route-out. Threshold set for a false-alarm rate on open-road frames, then detection is read by distance.
Also reports how often it fires at green-light passes (u239-only episodes, no stop follows).

usage: probe_stop2.py [FP_RATE]
"""
import glob
import os
import sys
import numpy as np

from openpilot.tools.weight_lab.lab_paths import DATA
from openpilot.tools.weight_lab.lights import events, SKIP, CIRCUIT

BANDS = [(0, 30), (30, 50), (50, 80), (80, 120), (120, 200)]


def label(d):
  v = d['v_ego']
  pos = np.concatenate([[0], np.cumsum(v * 0.05)])[:-1]
  dist = np.full(len(v), np.inf)
  for e in events(d):
    s = int(np.flatnonzero(d['frame'] == e['stop'])[0])
    dist = np.where(np.arange(len(v)) <= s, np.minimum(dist, pos[s] - pos), dist)
  h = d['x_plan_trunk'][:, CIRCUIT]
  green = (h[:, 3] > 0.2) & (h[:, :3].sum(1) < 0.05) & ~np.isfinite(dist)  # u239 alone, and no stop follows in this segment
  return dist, green


def ridge(X, y, lam=1.0):
  m, sd = X.mean(0), X.std(0) + 1e-6
  Z = (X - m) / sd
  w = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
  return lambda A: ((A - m) / sd) @ w + y.mean()


def main():
  fp = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
  segs = []
  for p in sorted(glob.glob(f'{DATA}/seg/*.npz')):
    if os.path.basename(p).startswith(SKIP):
      continue
    d = np.load(p)
    dist, green = label(d)
    segs.append(dict(route=str(d['seg'][0]).rsplit('/', 1)[0], x=d[os.environ.get('FEAT', 'x_plan_feat')], dist=dist, green=green, v=d['v_ego']))
  routes = sorted({s['route'] for s in segs})
  print(f'{len(segs)} segments, {len(routes)} routes, stops: {sum(np.isfinite(s["dist"]).any() for s in segs)} segments with a stop')
  score, dist, v, green = [], [], [], []
  for r in routes:
    tr = [s for s in segs if s['route'] != r]
    X = np.concatenate([s['x'] for s in tr])
    dd = np.concatenate([s['dist'] for s in tr]); vv = np.concatenate([s['v'] for s in tr])
    pos = (dd >= 30) & (dd <= 120)
    neg = (dd > 250) & (vv > 5)
    sel = pos | neg
    f = ridge(X[sel], pos[sel].astype(np.float32))
    for s in [s for s in segs if s['route'] == r]:
      score.append(f(s['x'])); dist.append(s['dist']); v.append(s['v']); green.append(s['green'])
  score, dist, v, green = map(np.concatenate, (score, dist, v, green))
  open_road = (dist > 250) & (v > 8)
  thr = np.percentile(score[open_road], 100 * (1 - fp))
  print(f'threshold for {100 * fp:.1f}% false alarms on held-out open road (v > 8, > 250 m from a stop): {thr:.2f}')
  for lo, hi in BANDS:
    m = (dist >= lo) & (dist < hi) & (v > 3)
    print(f'  {lo:3d}-{hi:3d} m before a stop: detected {100 * np.mean(score[m] > thr):4.0f}%  (n={m.sum()})')
  print(f'  green-light passes: fires on {100 * np.mean(score[green] > thr):.0f}% of frames (n={green.sum()})')


if __name__ == '__main__':
  main()
